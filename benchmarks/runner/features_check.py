#!/usr/bin/env python3
"""The feature-controls check's predeclared readings (D99, with D97's continuity rule in session face; stdlib only).

    features_check.py face RUN_DIR --f1 F1_DIR --run F1_RUN --identity-dir DIR [--recheck-root ROOT]
    features_check.py scene RUN_DIR

Session "face" (f-L face): monitoring, face recognition (a validation run, production detection, the continuity rule),
the live view and the feature controls. At cued times the operator turns away while seated, leaves the camera's view
and comes back, then turns face recognition and the live view off and on again in the dashboard's Features panel.
Session "scene" (f-L scene): monitoring with a whole-view restricted zone, scene analysis, chat (a validation run), the
live view, incident snapshots and the feature controls; the operator asks questions and turns scene analysis, chat and
snapshots off and on again. Face recognition and chat never run together here.

Each item prints PASS, FAIL, MISSING or NOT_EXERCISED; the last line says ``features-<session>: execution complete
(...)`` or ``execution incomplete``, and the exit status is 0 only when every judged item passed. Missing evidence never
passes. NOT_EXERCISED (the run did not test what the item is about: the change was not made, or no status capture fell
where the item needs one) never passes either. DESCRIPTIVE lines are reported and never judged.

Only the runtime's own evidence is read: its control lines (one per change its loop applied, with the loop's monotonic
time), its identity records, its status captures and the guard's result. A capture is placed among the changes by its
own features log (the changes the runtime had applied when it published that status), never by the capture's clock
time. A feature is off from its applied "off" line to its next applied "on" line (or the run's end). Whether the page
showed each state is the operator's observation sheet, never read as a pass. Nothing is written, and no secret, name,
key, question or similarity is read or shown.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import alpha_check as ac
import face_check as fc
import step5_check as s5

MISSING = s5.MISSING
NS = 1_000_000_000
LOG_KEPT = 20  # features.LOG_KEPT: a capture whose log is full cannot be placed among the changes
LOOP_RESULTS = ("applied", "unchanged", "refused")  # features: what the loop records, once per change (control lines)
CONTROL_KEYS = {"control", "change", "at_utc", "feature", "requested", "result", "mono_ns"}
BREAK_KEYS = {"identity", "schema", "stream_epoch", "track_id", "cause", "mono_ns"}  # runtime.break_record (D97)
FEATURES = ("face", "scene", "chat", "live_view", "snapshots")  # features.FEATURES
MIN_OFF_S = 20.0  # an off interval shorter than this does not exercise the records' items
FRAME_MARGIN_S = fc.STATE_TOLERANCE_S  # a frame from just before a change may still be offered after it
CONTINUITY = {"max_gap_s": 0.5, "overlap_iou": 0.3, "min_step_iou": 0.4, "retention_s": 120.0}  # D97's defaults
FACE_ENTRY = {**fc.F2_FACE, "continuity": CONTINUITY}  # the startup's face entry of a --face-continuity run
SWITCHABLE = {"face": {"face": True, "scene": False, "chat": False, "live_view": True, "snapshots": False},
              "scene": {"face": False, "scene": True, "chat": True, "live_view": True, "snapshots": True}}
FACE_OFF_REASON = "face recognition disabled"  # runtime._transitions while face recognition is off
SCENE_OFF_REASON = "scene analysis turned off"  # runtime: the scene reason while it is off
CHAT_OFF_REASON = "chat is turned off in Features"  # scene.chat.OPERATOR_OFF
FACE_SCOPE = {"before_launch": {"runtime": 1}, "workload_verified": {"runtime": 0}}  # no scene server
FACE_RELEASES = {"detector": {"engine": "returned_0"}, "face": fc.FACE_RELEASES}
DURATION_S = 540.0
FACE_CUES = {"t0": 0.0, "away": 90.0, "leave": 180.0, "back": 240.0, "faceoff": 300.0, "faceon": 345.0,
             "liveoff": 390.0, "liveon": 435.0, "pause": 465.0, "resume": 495.0, "stop": 525.0}
FACE_CAPTURES = ((75.0, "seated"), (135.0, "turned"), (210.0, "out"), (285.0, "returned"), (315.0, "faceoff1"),
                 (335.0, "faceoff2"), (380.0, "faceon"), (405.0, "liveoff1"), (425.0, "liveoff2"),
                 (450.0, "liveon"), (520.0, "end"))
FACE_SEGMENTS = (("seated work", "t0", "away"), ("turned away, seated", "away", "leave"),
                 ("away from the seat", "leave", "back"), ("back at the seat", "back", "faceoff"),
                 ("face recognition turned off", "faceoff", "faceon"), ("face recognition on again", "faceon", "stop"))
SCENE_CUES = {"t0": 0.0, "ask": 60.0, "sceneoff": 120.0, "askoff": 150.0, "sceneon": 210.0, "chatoff": 255.0,
              "chaton": 315.0, "snapoff": 345.0, "snapon": 390.0, "pause": 435.0, "resume": 465.0, "stop": 525.0}
SCENE_CAPTURES = ((105.0, "asked"), (145.0, "sceneoff1"), (195.0, "sceneoff2"), (240.0, "sceneon"),
                  (280.0, "chatoff1"), (305.0, "chatoff2"), (335.0, "chaton"), (370.0, "snapoff"), (520.0, "end"))
PLAN = {"face": (FACE_CUES, FACE_CAPTURES), "scene": (SCENE_CUES, SCENE_CAPTURES)}


# ---------------------------------------------------------------- the evidence


def control_lines(path: Path) -> tuple[list[dict], list[str]]:
    """The runtime's control lines (features.FeatureControls.done), in the loop's order, and the malformed ones."""
    lines, problems = [], []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return [], ["run.jsonl unreadable"]
    for number, raw in enumerate(text.splitlines(), 1):
        try:
            item = json.loads(raw)
        except ValueError:
            continue  # D48 is judged separately
        if not isinstance(item, dict) or "control" not in item:
            continue
        if (set(item) != CONTROL_KEYS or item["control"] != "feature" or item["feature"] not in FEATURES
                or item["requested"] not in ("on", "off") or item["result"] not in LOOP_RESULTS
                or type(item["change"]) is not int or type(item["mono_ns"]) is not int):
            problems.append(f"line {number}: not a control line")
            continue
        lines.append(item)
    return lines, problems


def identity_records(path: Path) -> tuple[dict[str, list[dict]], list[str], Any]:
    """The identity records as face_check reads them, with D97's break records checked here (face_check's readings
    refuse them), and the run's own starting line."""
    found, problems, starting, _ = fc.read_records(path)
    kept = [problem for problem in problems if "unknown identity record 'break'" not in problem]
    found["break"] = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        text = ""
    for number, raw in enumerate(text.splitlines(), 1):
        try:
            item = json.loads(raw)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("identity") == "break":
            if set(item) != BREAK_KEYS or item["cause"] not in ("gap", "jump", "overlap") \
                    or item["schema"] not in fc.SCHEMAS or type(item["mono_ns"]) is not int \
                    or type(item["track_id"]) is not int:
                kept.append(f"line {number}: a malformed break record")
            else:
                found["break"].append(item)
    return found, kept, starting


@dataclass
class Capture:
    name: str
    at_s: float  # planned, s after T0
    t: float | None  # when the guard took it, s after T0
    doc: Any
    applied: frozenset[int] | None  # the changes its log shows the loop had recorded; None: its log was full
    waiting: frozenset[str]  # the features with a change accepted but not yet applied (or replaced) there


def captures(directory: Path, result: Any, planned: tuple[tuple[float, str], ...]) -> list[Capture]:
    """Each planned status capture the guard saved, in the planned order."""
    ready_after = s5._get(result, "child", "ready_after_s")
    taken = {c.get("name"): c for c in s5._list(s5._get(result, "captures")) if isinstance(c, dict)}
    out = []
    for at, name in planned:
        path = directory / f"status-{name}.json"
        if s5._get(taken.get(name), "status") != "saved" or not path.is_file():
            continue
        doc = s5._load(path)
        when = s5._get(taken.get(name), "taken_after_launch_s")
        log = s5._get(doc, "features", "log")
        entries = [e for e in log if isinstance(e, dict) and type(e.get("change")) is int] if isinstance(log, list) \
            else []
        applied = None
        if isinstance(log, list) and len(log) < LOG_KEPT:
            applied = frozenset(e["change"] for e in entries if e.get("result") in LOOP_RESULTS)
        settled = {e["change"] for e in entries if e.get("result") != "accepted"}
        waiting = frozenset(e.get("feature") for e in entries if e.get("result") == "accepted"
                            and e["change"] not in settled)
        out.append(Capture(name, at, round(when - ready_after, 1) if isinstance(when, float)
                           and isinstance(ready_after, float) else None, doc, applied, waiting))
    return out


class Changes:
    """The loop's control lines in order, the off intervals they make, and where each capture falls among them."""

    def __init__(self, lines: list[dict], t0_ns: int | None, end_s: float) -> None:
        self.lines, self.t0_ns, self.end_s = lines, t0_ns, end_s

    def position(self, capture: Capture) -> int | None:
        """How many control lines the runtime had recorded when it published the capture; None if its log cannot
        tell (full) or does not match the lines (not a prefix of them)."""
        if capture.applied is None:
            return None
        n = len(capture.applied)
        return n if {line["change"] for line in self.lines[:n]} == capture.applied else None

    def inconsistent(self, shots: list[Capture]) -> list[str]:
        return [c.name for c in shots if c.applied is not None and self.position(c) is None]

    def state(self, feature: str, position: int) -> str:
        """The feature's state after the first ``position`` lines (every switchable feature starts on)."""
        state = "on"
        for line in self.lines[:position]:
            if line["feature"] == feature and line["result"] == "applied":
                state = line["requested"]
        return state

    def off(self, feature: str) -> list[tuple[int, int | None]]:
        """(index of the off line, index of the next on line or None), per off interval of the feature."""
        out, since = [], None
        for i, line in enumerate(self.lines):
            if line["feature"] != feature or line["result"] != "applied":
                continue
            if line["requested"] == "off" and since is None:
                since = i
            elif line["requested"] == "on" and since is not None:
                out.append((since, i))
                since = None
        if since is not None:
            out.append((since, None))
        return out

    def seconds(self, index: int | None) -> float:
        """A line's time in s after T0 (the run's end for None)."""
        if index is None or self.t0_ns is None:
            return self.end_s
        return (self.lines[index]["mono_ns"] - self.t0_ns) / NS

    def spans(self, feature: str) -> list[tuple[float, float]]:
        return [(self.seconds(a), self.seconds(b)) for a, b in self.off(feature)]

    def inside(self, shots: list[Capture], feature: str) -> list[tuple[int, list[Capture]]]:
        """Per off interval of the feature (by its number), the captures placed inside it."""
        out = []
        for k, (a, b) in enumerate(self.off(feature)):
            within = [c for c in shots if (p := self.position(c)) is not None and a < p and (b is None or p <= b)]
            out.append((k, within))
        return out

    def before_off(self, shots: list[Capture], feature: str) -> list[Capture]:
        """The captures placed before the feature was first turned off, with no change of it waiting there."""
        intervals = self.off(feature)
        first = intervals[0][0] if intervals else None
        return [c for c in shots if (p := self.position(c)) is not None and (first is None or p <= first)
                and feature not in c.waiting]

    def after_on(self, shots: list[Capture], feature: str) -> list[Capture]:
        """The captures placed after one of the feature's on lines and before its next off line, with no change of
        it waiting there."""
        intervals = self.off(feature)
        out = []
        for k, (_, b) in enumerate(intervals):
            if b is None:
                continue
            following = intervals[k + 1][0] if k + 1 < len(intervals) else None
            out += [c for c in shots if (p := self.position(c)) is not None and b < p
                    and (following is None or p <= following) and feature not in c.waiting]
        return out


def _counter(doc: Any, *path: str) -> Any:
    """A counter from a capture: 0 when its counters omit it (they list only what happened), MISSING without them."""
    counters = s5._get(doc, *path[:-1])
    return counters.get(path[-1], 0) if isinstance(counters, dict) else MISSING


def _span_text(spans: list[tuple[float, float]]) -> list[list[float]]:
    return [[round(a, 1), round(b, 1)] for a, b in spans]


# ---------------------------------------------------------------- common items


def _execution(r: fc.Reading, directory: Path, result: Any, starting: Any, session: str, lines: list[dict],
               line_problems: list[str]) -> tuple[list[Capture], Changes, dict[str, float]]:
    cues, planned = PLAN[session]
    switchable = s5._get(starting, "feature_controls")
    r.item("feature controls on, with exactly this session's features switchable", switchable,
           switchable == SWITCHABLE[session])
    stopped = s5._get(result, "run_output", "stopped")
    parts = (stopped.get("stopped") or {}) if isinstance(stopped, dict) else {}
    expected = {"face": ("face_worker", "live_view", "status_page", "control_key_removed"),
                "scene": ("scene_server", "scene_worker", "live_view", "snapshots", "status_page",
                          "control_key_removed")}[session]
    r.item("clean shutdown: all stopped (" + ", ".join(expected) + "), database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True
           and all(parts.get(part) is True for part in expected))
    output = s5._get(result, "run_output")
    counts = {k: output.get(k) for k in ("lines", "non_json_lines")} if isinstance(output, dict) else MISSING
    r.item("D48: every stdout line is JSON", counts,
           isinstance(counts, dict) and counts["non_json_lines"] == 0 and (counts["lines"] or 0) >= 2)
    offsets = fc._cue_offsets(result)
    duration = s5._get(result, "parameters", "duration_s")
    r.item(f"the run's {DURATION_S:g} s and every cue printed within {fc.CUE_TOLERANCE_S:g} s of its time",
           {"duration_s": duration, "cues_s_after_t0": {k: round(v, 2) for k, v in offsets.items()}},
           duration == DURATION_S
           and all(name in offsets and 0 <= offsets[name] - at <= fc.CUE_TOLERANCE_S for name, at in cues.items()))
    shots = captures(directory, result, planned)
    r.item("every status capture saved, each with the features block", [c.name for c in shots] or MISSING,
           len(shots) == len(planned) and all(isinstance(s5._get(c.doc, "features"), dict) for c in shots))
    ready = s5._get(result, "child", "ready_mono_s")
    t0_ns = round((offsets["t0"] + ready) * NS) if isinstance(ready, float) and "t0" in offsets else None
    changes = Changes(lines, t0_ns, DURATION_S)
    unplaced = changes.inconsistent(shots)
    r.item("control lines: well formed, each applied or found unchanged by the loop (none refused), only switchable "
           "features, and each capture's log a prefix of them",
           {"lines": len(lines), "problems": line_problems[:5], "results": sorted({x["result"] for x in lines}),
            "captures_not_matching": unplaced},
           not line_problems and not unplaced and all(x["result"] in ("applied", "unchanged") for x in lines)
           and all((switchable if isinstance(switchable, dict) else {}).get(x["feature"]) is True for x in lines))
    return shots, changes, offsets


def _describe_changes(r: fc.Reading, result: Any, changes: Changes, shots: list[Capture]) -> None:
    r.describe("guard peak pressure, min MemFree, min MemAvailable (B)",
               [s5._get(result, "guard", k) for k in s5.GUARD_FIGURES])
    r.describe("the changes the loop recorded: [change, feature, requested, result, s after T0]",
               [[x["change"], x["feature"], x["requested"], x["result"], round(changes.seconds(i), 1)]
                for i, x in enumerate(changes.lines)])
    placed = {}
    for c in shots:
        position = changes.position(c)
        placed[c.name] = {"taken_s": c.t, "after_lines": position,
                          "off": None if position is None else
                          [f for f in FEATURES if changes.state(f, position) == "off"]}
    r.describe("each capture: when the guard took it (s after T0), how many control lines came before it (by its own "
               "features log; null: its log was full) and which features were off there", placed)
    last = shots[-1].doc if shots else MISSING
    r.describe("the feature controls' counters at the last capture (refused attempts are counted, never logged)",
               s5._get(last, "features", "counters"))


def _resumed(r: fc.Reading, changes: Changes, shots: list[Capture], checks: dict[str, Any]) -> None:
    """Each switchable feature runs at every capture before its first off and after each on (a capture with a change
    of it still waiting is left out)."""
    name = "before each feature was first turned off: every capture shows it running"
    seen = {feature: [[c.name, check(c.doc)] for c in changes.before_off(shots, feature)]
            for feature, check in checks.items()}
    r.item(name, seen if all(seen.values()) else MISSING, all(seen.values())
           and all(ok for rows in seen.values() for _, ok in rows))
    name = ("after each feature was turned on again: every later capture (until it was next turned off) shows it "
            "running again")
    seen = {feature: [[c.name, check(c.doc)] for c in changes.after_on(shots, feature)] for feature, check in
            checks.items()}
    seen = {feature: rows for feature, rows in seen.items() if rows}
    if not seen:
        r.not_exercised(name, "no capture after a feature was turned on again")
        return
    r.item(name, seen, all(ok for rows in seen.values() for _, ok in rows))


# ---------------------------------------------------------------- session "face"


def read_face(directory: Path, f1: Path, gate: tuple[Path, Path, Path | None] | None) -> fc.Reading:
    r = fc.Reading()
    if gate is not None:
        ok, gate_lines = fc.f1_gate(*gate)
        r.item("F1 basis for this run: the run's own F1 validated, or the maintainer's accepted corrected F1 reading "
               "of it", dict(line.split("=", 1) for line in gate_lines),
               ok and f1.resolve() == (gate[0] / "f1").resolve())
    result = s5._load(directory / "result.json")
    found, problems, starting = identity_records(directory / "run.jsonl")
    lines, line_problems = control_lines(directory / "run.jsonl")
    s5._supervision(r, result, "duration_stop", 0)
    start = s5._startup(r, result, scope=FACE_SCOPE, releases=FACE_RELEASES)
    r.item("face admitted for this validation run with production detection and the continuity rule (D97): one "
           "enrolled identity, no problem; no scene server",
           {"face": s5._get(start, "face"), "face_problem": s5._get(start, "face_problem"),
            "scene_server": s5._get(start, "scene_server")},
           s5._get(start, "face") == FACE_ENTRY and s5._get(start, "face_problem") is None
           and s5._get(start, "scene_server") in (None, MISSING))
    r.item("no notification channel (Telegram disabled)", s5._get(start, "notifier_problems"),
           s5._get(start, "notifier_problems") == {})
    shots, changes, offsets = _execution(r, directory, result, starting, "face", lines, line_problems)
    s5._listeners(r, result, {18090})
    total = sum(len(v) for v in found.values())
    schemas = sorted({item["schema"] for kind in found.values() for item in kind})
    last = shots[-1].doc if shots else MISSING
    dropped = s5._get(last, "components", "face", "records", "dropped")
    r.item(f"identity records: schema {fc.FULL_SCALE_SCHEMA}, complete, none dropped, no comparison or capture "
           "record, breaks well formed",
           {"records": {k: len(v) for k, v in found.items()}, "schemas": schemas, "problems": problems[:5],
            "dropped": dropped} if total else MISSING,
           total > 0 and not problems and schemas == [fc.FULL_SCALE_SCHEMA] and dropped == 0
           and not found.get("comparison") and not found.get("capture") and total < fc.MAX_IDENTITY_RECORDS)
    fc._counts_items(r, directory)
    if problems or not total or changes.t0_ns is None:
        r.item("face recognition and the live view turned off and on: judged from the records and captures", MISSING,
               False)
        return r
    line = fc.Timeline(found, changes.t0_ns, DURATION_S)
    _face_off_records(r, line, changes)
    _face_off_captures(r, changes, shots)
    _face_on(r, line, changes)
    _live_off(r, changes, shots)
    _resumed(r, changes, shots, {"face": _face_running, "live_view": _live_running})
    _describe_changes(r, result, changes, shots)
    _describe_face(r, line, found, offsets, shots)
    return r


def _face_off_records(r: fc.Reading, line: fc.Timeline, changes: Changes) -> None:
    name = (f"while face recognition was off ({MIN_OFF_S:g} s or more): no frame offered, no face result applied and "
            "no identity reached; each identity KNOWN when it was turned off ended within "
            f"{fc.STATE_TOLERANCE_S:g} s, saying {FACE_OFF_REASON}")
    spans = changes.spans("face")
    long = [(a, b) for a, b in spans if b - a >= MIN_OFF_S]
    if not long:
        r.not_exercised(name, {"intervals_s": _span_text(spans),
                               "why": f"face recognition was not off for {MIN_OFF_S:g} s or more"})
        return
    offered = [round(t["t"], 2) for a, b in long for t in line.ticks
               if t["tick"] == "offered" and a < t["t"] < b - FRAME_MARGIN_S]
    applied = [round(x["applied"], 2) for a, b in long for x in line.results
               if x["outcome"] == "applied" and a < x["applied"] < b]
    reached = [[t["to"], round(t["t"], 2)] for a, b in long for t in line.transitions
               if t["to"] in ("known", "unknown", "uncertain") and a < t["t"] < b]
    held = []
    for a, _ in long:
        for key, intervals in line.known.items():
            for interval in intervals:
                if interval["start"] <= a and (interval["end"] is None or interval["end"] > a - fc.EXACT_S):
                    end = interval["end"]
                    held.append({"track": list(key), "ended_s": None if end is None else round(end, 2),
                                 "to": interval["end_to"], "reason": interval["end_reason"],
                                 "ok": end is not None and a - fc.EXACT_S <= end <= a + fc.STATE_TOLERANCE_S
                                 and interval["end_to"] == "unresolved" and interval["end_reason"] == FACE_OFF_REASON})
    r.item(name, {"intervals_s": _span_text(long), "offered_ticks": offered[:5], "applied_results": applied[:5],
                  "identities_reached": reached[:5], "known_when_turned_off": held},
           not offered and not applied and not reached and all(h["ok"] for h in held))


def _face_off_captures(r: fc.Reading, changes: Changes, shots: list[Capture]) -> None:
    name = ("every capture while face recognition was off shows it off and paused (Features: off, or stopping while a "
            "face finished), with no recognition rows and no identity counted")
    seen = [{"capture": c.name, "actual": s5._get(c.doc, "features", "items", "face", "actual"),
             "paused": s5._get(c.doc, "components", "face", "paused"),
             "rows": "faces" in (s5._get(c.doc, "live") or {}) if isinstance(s5._get(c.doc, "live"), dict) else MISSING,
             "identities": {k: s5._get(c.doc, "live", "identity", k) for k in ("known", "unknown", "uncertain")},
             "people": s5._get(c.doc, "live", "people")}
            for _, within in changes.inside(shots, "face") for c in within]
    if not seen:
        r.not_exercised(name, "no capture while face recognition was off")
        return
    r.item(name, seen, all(s["actual"] in ("off", "stopping") and s["paused"] is True and s["rows"] is False
                           and all(v == 0 for v in s["identities"].values()) for s in seen))


def _face_on(r: fc.Reading, line: fc.Timeline, changes: Changes) -> None:
    name = (f"after face recognition was turned on again: each identity KNOWN again only on {fc.CONFIRMATIONS} match "
            "votes the runtime applied once it was on, none from a frame before it was turned off")
    turned = [(a, b) for a, b in changes.spans("face") if b < changes.end_s]
    if not turned:
        r.not_exercised(name, "face recognition was not turned on again")
        return
    firsts = []
    for k, (a, b) in enumerate(turned):
        following = turned[k + 1][0] if k + 1 < len(turned) else changes.end_s
        for key, intervals in sorted(line.known.items()):
            interval = next((i for i in intervals if b <= i["start"] < following), None)
            if interval is None:
                continue
            votes = sorted((v for v in line.votes.get(key, ()) if v["applied"] <= interval["start"] + 1e-6),
                           key=lambda v: v["t"])[-fc.CONFIRMATIONS:]
            # The loop applies a change at the start of its step, before any result: a vote applied in that step
            # (at the same instant on a coarse clock) was applied once face recognition was on.
            fresh = all(v["vote"] == "match" and v["identity_id"] == interval["identity_id"] and v["applied"] >= b
                        and v["t"] > a for v in votes)
            firsts.append({"track": list(key), "known_s": round(interval["start"], 2), "turned_off_s": round(a, 2),
                           "turned_on_s": round(b, 2),
                           "votes": [[v["vote"], round(v["t"], 2), round(v["applied"], 2)] for v in votes],
                           "ok": len(votes) == fc.CONFIRMATIONS and fresh})
    if not firsts:
        r.not_exercised(name, {"turned_on_s": [round(b, 1) for _, b in turned],
                               "why": "nobody was KNOWN after it was turned on again"})
        return
    r.item(name, firsts, all(f["ok"] for f in firsts))


def _live_off(r: fc.Reading, changes: Changes, shots: list[Capture]) -> None:
    name = ("every capture while the live view was off shows it off: paused, no picture kept, the page's live view "
            "paused")
    groups = changes.inside(shots, "live_view")
    seen = [{"capture": c.name, "actual": s5._get(c.doc, "features", "items", "live_view", "actual"),
             "component": s5._get(c.doc, "components", "preview", "state"),
             "pictures_kept": s5._get(c.doc, "components", "preview", "pictures_kept"),
             "view": s5._get(c.doc, "preview", "state")} for _, within in groups for c in within]
    if not seen:
        r.not_exercised(name, "no capture while the live view was off")
    else:
        r.item(name, seen, all(s["actual"] == "off" and s["component"] == "paused" and s["pictures_kept"] == 0
                               and s["view"] == "paused" for s in seen))
    name = "no picture encoded while the live view was off: the encode count unchanged between the captures inside"
    pairs = [(within[0], within[-1]) for _, within in groups if len(within) >= 2]
    if not pairs:
        r.not_exercised(name, "fewer than two captures inside one off interval")
        return
    value = [{"from": a.name, "to": b.name, "encodes": [_counter(a.doc, "components", "preview", "counters", "encodes"),
                                                       _counter(b.doc, "components", "preview", "counters", "encodes")]}
             for a, b in pairs]
    r.item(name, value, all(v["encodes"][0] == v["encodes"][1] for v in value))


def _face_running(doc: Any) -> bool:
    face = s5._get(doc, "components", "face")
    ok = isinstance(face, dict) and face.get("state") == "available" and "paused" not in face \
        and isinstance(s5._get(doc, "live", "faces"), list) \
        and s5._get(doc, "features", "items", "face", "actual") == "running"
    return ok


def _live_running(doc: Any) -> bool:
    ok = s5._get(doc, "components", "preview", "state") == "enabled" \
        and s5._get(doc, "preview", "state") not in ("paused", MISSING) \
        and s5._get(doc, "features", "items", "live_view", "actual") == "running"
    return ok


def _describe_face(r: fc.Reading, line: fc.Timeline, found: dict, offsets: dict[str, float],
                   shots: list[Capture]) -> None:
    if all(name in offsets for name in FACE_CUES):
        r.describe("per cue interval (s after T0; when the operator moved within it is the sheet's): KNOWN "
                   "person-seconds, KNOWN intervals, entries into KNOWN and how intervals ended", {
                       label: {"window": [round(offsets[a], 1), round(offsets[b], 1)],
                               **fc._identity_time(line, (offsets[a], offsets[b]))} for label, a, b in FACE_SEGMENTS})
    breaks: dict[str, int] = {}
    for item in found.get("break", ()):
        breaks[item["cause"]] = breaks.get(item["cause"], 0) + 1
    r.describe("breaks in track continuity by cause (D97; geometry only: a break is not a person changing)",
               dict(sorted(breaks.items())))
    longest = 0.0
    for intervals in line.known.values():
        for interval in intervals:
            steps = interval["basis_changes"] + [(line.end_s if interval["end"] is None else interval["end"], None)]
            for (t, basis), (t_next, _) in zip(steps, steps[1:]):
                if basis == "retained":
                    longest = max(longest, t_next - t)
    r.describe("the longest stretch KNOWN with basis retained (s)", round(longest, 1))
    reached: dict[str, int] = {}
    for transition in line.transitions:
        label = f"{transition['to']}/{transition['basis'] or 'none'}"
        reached[label] = reached.get(label, 0) + 1
    r.describe("identity transitions over the run, by state reached and basis", dict(sorted(reached.items())))
    r.describe("each capture's recognition rows as the page showed them: per listed track its state, basis, newest "
               "vote's age (ms) and time followed without a break (ms)", {
                   c.name: [[x.get("track_id"), x.get("state"), x.get("basis"), x.get("vote_age_ms"),
                             x.get("followed_ms")] for x in s5._get(c.doc, "live", "faces") or []
                            if isinstance(x, dict)]
                   if isinstance(s5._get(c.doc, "live", "faces"), list) else None for c in shots})
    last = shots[-1].doc if shots else MISSING
    r.describe("the live view's counters at the last capture", s5._get(last, "components", "preview", "counters"))


# ---------------------------------------------------------------- session "scene"


def read_scene(directory: Path) -> fc.Reading:
    r = fc.Reading()
    result = s5._load(directory / "result.json")
    found, problems, starting = identity_records(directory / "run.jsonl")
    lines, line_problems = control_lines(directory / "run.jsonl")
    s5._supervision(r, result, "duration_stop", 0)
    start = s5._startup(r, result, scope=ac.SCOPE, releases=ac.RELEASES)
    server = s5._get(start, "scene_server")
    r.item("scene server ready, 17/17 layers, vision encoder on GPU; scene admitted; no face recognition",
           {"scene_server": server, "scene_problem": s5._get(start, "scene_problem"), "face": s5._get(start, "face")},
           isinstance(server, dict) and server.get("state") == "ready" and server.get("problem") is None
           and server.get("layers") == s5.LAYERS and server.get("vision_on_gpu") is True
           and s5._get(start, "scene_problem") is None and s5._get(start, "face") in (None, MISSING))
    r.item("chat offered as a validation run (not admitted); the live view and incident snapshots on",
           {"chat": s5._get(starting, "chat"), "live_preview": isinstance(s5._get(starting, "live_preview"), dict),
            "snapshots": isinstance(s5._get(starting, "snapshots"), dict)},
           s5._get(starting, "chat") == {"state": "available", "reason": ac.VALIDATION_REASON}
           and isinstance(s5._get(starting, "live_preview"), dict) and isinstance(s5._get(starting, "snapshots"), dict))
    r.item("no notification channel (Telegram disabled)", s5._get(start, "notifier_problems"),
           s5._get(start, "notifier_problems") == {})
    shots, changes, _ = _execution(r, directory, result, starting, "scene", lines, line_problems)
    s5._listeners(r, result, {18081, 18090})
    r.item("no identity record (face recognition is not part of this session)",
           {"records": sum(len(v) for v in found.values()), "problems": problems[:5]},
           not problems and not any(found.values()))
    _camera_counts(r, directory)
    if changes.t0_ns is None:
        r.item("scene analysis, chat and snapshots turned off and on: judged from the captures", MISSING, False)
        return r
    _scene_off(r, changes, shots)
    _chat_while_scene_off(r, changes, shots)
    _chat_off(r, changes, shots)
    _snapshots(r, changes, shots)
    _resumed(r, changes, shots, {"scene": _scene_running, "chat": _chat_running, "snapshots": _snapshots_running})
    _describe_changes(r, result, changes, shots)
    _describe_scene(r, shots)
    return r


def _camera_counts(r: fc.Reading, directory: Path) -> None:
    expected = {p.name for p in directory.iterdir() if p.is_file() and p.name not in fc.NOT_COUNTED}
    value, passed = fc._count_check(fc._counts(directory, "secret-counts-camera.txt"), expected, ("userinfo",))
    r.item("camera userinfo counts: every evidence file counted, all 0", value, passed)


def _periodic(doc: Any) -> Any:
    return _counter(doc, "components", "scene", "lane", "submitted_periodic")


def _scene_off(r: fc.Reading, changes: Changes, shots: list[Capture]) -> None:
    name = ("every capture while scene analysis was off shows it off: no current scene result (turned off), the lane "
            "paused, Features off or stopping, and the scene model still loaded (its server ready)")
    groups = changes.inside(shots, "scene")
    seen = [{"capture": c.name, "scene": s5._get(c.doc, "live", "scene"),
             "reason": s5._get(c.doc, "live", "scene_reason"),
             "paused": s5._get(c.doc, "components", "scene", "paused"),
             "actual": s5._get(c.doc, "features", "items", "scene", "actual"),
             "server": s5._get(c.doc, "components", "scene", "server", "state")}
            for _, within in groups for c in within]
    if not seen:
        r.not_exercised(name, "no capture while scene analysis was off")
    else:
        r.item(name, seen, all(s["scene"] == "no_current_result" and s["reason"] == SCENE_OFF_REASON
                               and s["paused"] is True and s["actual"] in ("off", "stopping") and s["server"] == "ready"
                               for s in seen))
    name = "no periodic scene check submitted while scene analysis was off, between the captures inside"
    pairs = [(within[0], within[-1]) for _, within in groups if len(within) >= 2]
    if not pairs:
        r.not_exercised(name, "fewer than two captures inside one off interval")
        return
    value = [{"from": a.name, "to": b.name, "submitted_periodic": [_periodic(a.doc), _periodic(b.doc)]}
             for a, b in pairs]
    r.item(name, value, all(v["submitted_periodic"][0] == v["submitted_periodic"][1] for v in value))


def _chat_counts(doc: Any) -> dict[str, Any]:
    return {k: _counter(doc, "components", "chat", "counters", k) for k in ("accepted", "answered", "timeouts",
                                                                             "errors")}


def _chat_while_scene_off(r: fc.Reading, changes: Changes, shots: list[Capture]) -> None:
    name = ("while scene analysis was off (chat on): chat offered at every capture, and a question accepted between "
            "the captures inside was answered (scene analysis off never takes the model chat uses)")
    pairs = []
    for _, within in changes.inside(shots, "scene"):
        on = [c for c in within if changes.state("chat", changes.position(c)) == "on"]  # type: ignore[arg-type]
        if on:
            pairs.append(on)
    offered = [[c.name, s5._get(c.doc, "components", "chat", "state")] for group in pairs for c in group]
    if not offered:
        r.not_exercised(name, "no capture while scene analysis was off and chat on")
        return
    spans = [{"from": g[0].name, "to": g[-1].name, "before": _chat_counts(g[0].doc), "after": _chat_counts(g[-1].doc)}
             for g in pairs if len(g) >= 2]
    asked = [s for s in spans if not s5._has_missing(s) and s["after"]["accepted"] > s["before"]["accepted"]]
    value = {"chat_state": offered, "between_captures": spans}
    if not asked:
        r.not_exercised(name, {**value,
                               "why": "no question accepted between two captures while scene analysis was off"})
        return
    r.item(name, value, all(state == "available" for _, state in offered) and all(
        s["after"]["answered"] - s["before"]["answered"] >= s["after"]["accepted"] - s["before"]["accepted"]
        and s["after"]["timeouts"] == s["before"]["timeouts"] and s["after"]["errors"] == s["before"]["errors"]
        for s in asked))


def _chat_off(r: fc.Reading, changes: Changes, shots: list[Capture]) -> None:
    name = "every capture while chat was off shows it off, saying it is turned off in Features"
    groups = changes.inside(shots, "chat")
    seen = [{"capture": c.name, "state": s5._get(c.doc, "components", "chat", "state"),
             "reason": s5._get(c.doc, "components", "chat", "reason"),
             "actual": s5._get(c.doc, "features", "items", "chat", "actual")} for _, within in groups for c in within]
    if not seen:
        r.not_exercised(name, "no capture while chat was off")
    else:
        r.item(name, seen, all(s["state"] == "off" and s["reason"] == CHAT_OFF_REASON and s["actual"] == "off"
                               for s in seen))
    name = ("periodic scene checks still submitted while chat was off (scene analysis on), between the captures "
            "inside (chat off never stops scene analysis)")
    pairs = []
    for _, within in groups:
        on = [c for c in within if changes.state("scene", changes.position(c)) == "on"]  # type: ignore[arg-type]
        if len(on) >= 2:
            pairs.append((on[0], on[-1]))
    if not pairs:
        r.not_exercised(name, "fewer than two captures inside one chat-off interval with scene analysis on")
        return
    value = [{"from": a.name, "to": b.name, "submitted_periodic": [_periodic(a.doc), _periodic(b.doc)]}
             for a, b in pairs]
    r.item(name, value, all(not s5._has_missing(v) and v["submitted_periodic"][1] > v["submitted_periodic"][0]
                            for v in value))


def _snapshots(r: fc.Reading, changes: Changes, shots: list[Capture]) -> None:
    name = "every capture while incident snapshots were off shows them off (or finishing a queued one)"
    seen = [{"capture": c.name, "actual": s5._get(c.doc, "features", "items", "snapshots", "actual")}
            for _, within in changes.inside(shots, "snapshots") for c in within]
    if not seen:
        r.not_exercised(name, "no capture while snapshots were off")
    else:
        r.item(name, seen, all(s["actual"] in ("off", "stopping") for s in seen))
    name = "no incident snapshot deleted over the run: no pruning at any capture (the saved ones stay)"
    stores = [(c.name, s5._get(c.doc, "components", "snapshots", "store")) for c in shots]
    if not stores or not all(isinstance(store, dict) for _, store in stores):
        r.item(name, MISSING, False)
        return
    pruned = [[n, sum(v for k, v in store.items() if k.startswith("pruned:"))] for n, store in stores]
    saved = [[n, store.get("saved", 0)] for n, store in stores]
    if saved[-1][1] < 1:
        r.not_exercised(name, {"saved": saved, "why": "no snapshot was saved in this run"})
        return
    r.item(name, {"pruned": pruned, "saved": saved}, all(p == 0 for _, p in pruned))


def _scene_running(doc: Any) -> bool:
    scene = s5._get(doc, "components", "scene")
    ok = isinstance(scene, dict) and scene.get("state") == "available" and "paused" not in scene \
        and s5._get(doc, "live", "scene_reason") != SCENE_OFF_REASON \
        and s5._get(doc, "features", "items", "scene", "actual") == "running"
    return ok


def _chat_running(doc: Any) -> bool:
    ok = s5._get(doc, "components", "chat", "state") in ("available", "paused") \
        and s5._get(doc, "features", "items", "chat", "actual") in ("running", "busy")  # busy: chat's own gate
    return ok


def _snapshots_running(doc: Any) -> bool:
    return s5._get(doc, "features", "items", "snapshots", "actual") == "running"


def _describe_scene(r: fc.Reading, shots: list[Capture]) -> None:
    r.describe("chat's counts at each capture: accepted, answered, timeouts, errors, and questions refused while chat "
               "was off or unavailable (a question's text is never recorded)",
               {c.name: {**_chat_counts(c.doc), "refused": _counter(c.doc, "components", "chat", "counters",
                                                                    "refused_unavailable")} for c in shots})
    r.describe("periodic scene checks submitted, at each capture", {c.name: _periodic(c.doc) for c in shots})
    last = shots[-1].doc if shots else MISSING
    r.describe("the scene lane's counters at the last capture", s5._get(last, "components", "scene", "lane"))
    r.describe("the snapshot store's counters at the last capture (missing:snapshots_off counts an incident opened "
               "while snapshots were off)", s5._get(last, "components", "snapshots", "store"))
    r.describe("the live view's counters at the last capture", s5._get(last, "components", "preview", "counters"))


# ---------------------------------------------------------------- the command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="session", required=True)
    face = commands.add_parser("face")
    face.add_argument("run_dir", type=Path)
    face.add_argument("--f1", type=Path, required=True)
    face.add_argument("--run", type=Path, required=True)
    face.add_argument("--identity-dir", type=Path, required=True)
    face.add_argument("--recheck-root", type=Path)
    scene = commands.add_parser("scene")
    scene.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    if args.session == "face":
        reading = read_face(args.run_dir, args.f1, (args.run, args.identity_dir, args.recheck_root))
    else:
        reading = read_scene(args.run_dir)
    for line in reading.lines:
        print(line)
    print(f"features-{args.session}: " + ("execution complete (descriptive where it says so; it validates nothing)"
                                          if reading.ok else "execution incomplete"))
    return 0 if reading.ok else 1


if __name__ == "__main__":
    sys.exit(main())
