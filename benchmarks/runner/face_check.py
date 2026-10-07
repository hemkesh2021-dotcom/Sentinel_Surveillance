#!/usr/bin/env python3
"""The V2-25 device check's predeclared readings (F1 consented enrollment, F2 guarded live validation).

    face_check.py f1 F1_DIR --identity-dir DIR
    face_check.py f2 F2_DIR --f1 F1_DIR

Each item prints PASS, FAIL, MISSING or NOT_EXERCISED, then ``face-<part>: validated`` or ``not validated``; exit
status 0 only when every item passed. Missing evidence never passes, and NOT_EXERCISED (the run did not test what the
item is about, e.g. retention while new matching evidence kept arriving) never passes either. DESCRIPTIVE lines are
reported and never judged. Nothing is written, and no secret, name or embedding is read or shown; the account's notes
are never printed.

**F2's time base.** T0 is the moment the guard printed its ``t0`` cue, which it does when the runtime's ``starting``
line appears, i.e. once every model is loaded and released (``result.json`` ``cues``, CLOCK_MONOTONIC like the
runtime's records). The operator started the stopwatch at that cue, and the account gives the observed times on it.
Judged windows come from the account, never from the schedule, with a washout of WASHOUT_S after each observed
movement and before the next (an assumption, not a measurement; it also absorbs the stopwatch's start delay):

    E1 empty     out_of_view_1 + W .. seated_facing_1 - W
    K1 facing    seated_facing_1 + W .. turned_away - W
    A  away      turned_away + W .. stood_up - W          (the same track, face turned away)
    E2 empty     out_of_view_2 + W .. seated_facing_2 - W
    K2 return    seated_facing_2 + W .. duration - W

Ticks and votes are placed by their frame's time; a result applied after its frame (in flight across a boundary)
counts where its frame was. Whether a state was justified at a moment uses only results applied by that moment.
Identity rules follow ``sentinel.identity.state`` with the runtime's configured values (a test ties the constants
below to the runtime's defaults).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

import face_account
import step5_check as s5
from operator_account import parse as parse_account
from operator_account import seconds as stopwatch_seconds

SCHEMA = 1
ID = re.compile(r"^idn-[0-9a-f]{12}$")
NS = 1_000_000_000
# The runtime's identity values (config.IdentityConfig defaults; test-tied).
VOTE_TTL_S = 30.0
FRESH_S = 3.0
CONFIRMATIONS = 2
# This check's proposed values (maintainer review, 2026-10-08).
WASHOUT_S = 15.0
MIN_WINDOW_S = 45.0
MIN_TICKS_EMPTY = 30
MIN_OFFERED = 30
MIN_APPLIED = 30
MIN_MATCHES_FACING = 10
MIN_CONTINUITY = 0.9  # share of the away window's offered ticks that list the retained track
MIN_COMPLETION = 0.95  # approved 2026-10-08 for this bounded check: completed / offered, with 0 processing errors
# F1's package minimum (maintainer, 2026-10-08): at least 3 usable photos. The enrollment API's own minimum
# (sentinel.identity.enroll.MIN_PHOTOS, 2) is separate and unchanged; this check is stricter.
MIN_USABLE_PHOTOS = 3
MAX_USABLE_PHOTOS = 8  # MAX_PROTOTYPES
STATE_TOLERANCE_S = 2.0  # a state change may lag its cause by the loop's step and a capture
EXACT_S = 0.01  # a state change never precedes its cause (rounding only)
CUE_TOLERANCE_S = 0.5
MAX_IDENTITY_RECORDS = 10_000  # the runtime's cap
FACE_RELEASES = {"facenet512_weights.h5": "returned_0", "face_detection_yunet_2023mar.onnx": "returned_0"}
SCENE_SCOPE = {"before_launch": {"runtime": 1}, "llama_ready": {"runtime": 1, "scene_server": 1},
               "workload_verified": {"runtime": 0, "scene_server": 1}}
RELEASES = {"scene": {"llm": "returned_0", "mmproj": "returned_0"}, "detector": {"engine": "returned_0"},
            "face": FACE_RELEASES}
TICK_KEYS = {"identity", "schema", "stream_epoch", "frame_seq", "frame_mono_ns", "tick", "tracks"}
RESULT_KEYS = {"identity", "schema", "stream_epoch", "frame_seq", "frame_mono_ns", "applied_mono_ns", "outcome",
               "faces", "processing_ms", "error", "persons"}
PERSON_KEYS = {"track_id", "ownership", "vote", "identity_id", "label"}
TRANSITION_KEYS = {"identity", "schema", "stream_epoch", "track_id", "from", "to", "identity_id", "basis",
                   "last_vote_age_ms", "reason", "mono_ns"}
TICKS = {"offered", "skipped_no_person", "stopping"}
OUTCOMES = {"applied", "failed", "rejected_not_live", "rejected_boot", "rejected_age", "ignored_face_unavailable"}
STATES = {"unresolved", "unknown", "known", "uncertain", "cleared"}
MISSING = s5.MISSING
WINDOWS = ("E1", "K1", "A", "E2", "K2")


class Reading(s5.Reading):
    def not_exercised(self, name: str, value: Any) -> None:
        self.ok = False
        self.lines.append(f"NOT_EXERCISED {name}: {s5._text(value)}")


def _int(value: Any) -> bool:
    return type(value) is int


# ---------------------------------------------------------------- F2 records


def read_records(path: Path) -> tuple[dict[str, list[dict]], list[str], dict | None, dict | None]:
    """(records by kind, schema problems, starting line, stopped line) from the runtime's stdout file."""
    records: dict[str, list[dict]] = {"tick": [], "result": [], "transition": []}
    problems: list[str] = []
    starting = stopped = None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return records, ["run.jsonl unreadable"], None, None
    for number, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
        except ValueError:
            continue  # D48 is judged separately
        if not isinstance(item, dict):
            continue
        if item.get("run") == "starting" and starting is None:
            starting = item.get("startup")
        elif item.get("run") == "stopped":
            stopped = item
        kind = item.get("identity")
        if not isinstance(kind, str):
            continue  # not an identity record (e.g. a status line)
        problem = _record_problem(kind, item)
        if problem is not None:
            problems.append(f"line {number}: {problem}")
        else:
            records[kind].append(item)
    return records, problems, starting, stopped


def _record_problem(kind: Any, item: dict) -> str | None:
    keys = {"tick": TICK_KEYS, "result": RESULT_KEYS, "transition": TRANSITION_KEYS}.get(kind)
    if keys is None:
        return f"unknown identity record {kind!r}"
    if item.get("schema") != SCHEMA:
        return f"{kind}: schema {item.get('schema')!r}, not {SCHEMA}"
    if set(item) != keys:
        return f"{kind}: fields {sorted(set(item) ^ keys)} differ from schema {SCHEMA}"
    if kind == "tick":
        if not (_int(item["frame_mono_ns"]) and item["tick"] in TICKS and isinstance(item["tracks"], list)
                and all(_int(t) for t in item["tracks"])):
            return "tick: a field has the wrong type or value"
    elif kind == "result":
        if not (_int(item["frame_mono_ns"]) and _int(item["applied_mono_ns"]) and item["outcome"] in OUTCOMES
                and isinstance(item["persons"], list)):
            return "result: a field has the wrong type or value"
        for person in item["persons"]:
            if not isinstance(person, dict) or set(person) != PERSON_KEYS or not _int(person["track_id"]) \
                    or person["vote"] not in ("match", "unknown", "none") \
                    or (person["vote"] == "match") != (person["identity_id"] is not None) \
                    or (person["identity_id"] is not None and not ID.match(str(person["identity_id"]))):
                return "result: a person entry differs from schema 1"
    else:
        if not (_int(item["mono_ns"]) and _int(item["track_id"]) and item["from"] in STATES and item["to"] in STATES
                and (item["identity_id"] is None or ID.match(str(item["identity_id"])))
                and ((item["to"] == "known") == (item["identity_id"] is not None))):
            return "transition: a field has the wrong type or value"
    return None


class Timeline:
    """Seconds after T0, per (epoch, track): applied votes and KNOWN intervals."""

    def __init__(self, records: dict[str, list[dict]], t0_ns: int, end_s: float) -> None:
        self.t0_ns = t0_ns
        self.end_s = end_s
        self.ticks = [dict(r, t=self.rel(r["frame_mono_ns"])) for r in records["tick"]]
        self.results = [dict(r, t=self.rel(r["frame_mono_ns"]), applied=self.rel(r["applied_mono_ns"]))
                        for r in records["result"]]
        self.transitions = sorted((dict(r, t=self.rel(r["mono_ns"])) for r in records["transition"]),
                                  key=lambda r: r["t"])
        self.votes: dict[tuple[int, int], list[dict]] = {}
        for result in self.results:
            if result["outcome"] != "applied":
                continue
            for person in result["persons"]:
                if person["vote"] in ("match", "unknown"):
                    key = (result["stream_epoch"], person["track_id"])
                    self.votes.setdefault(key, []).append(
                        {"t": result["t"], "applied": result["applied"], "vote": person["vote"],
                         "identity_id": person["identity_id"]})
        for votes in self.votes.values():
            votes.sort(key=lambda v: v["t"])
        self.known: dict[tuple[int, int], list[dict]] = {}
        open_: dict[tuple[int, int], dict] = {}
        for transition in self.transitions:
            key = (transition["stream_epoch"], transition["track_id"])
            if transition["to"] == "known":
                if key not in open_ or open_[key]["identity_id"] != transition["identity_id"]:
                    if key in open_:
                        open_[key]["end"] = transition["t"]
                    open_[key] = {"start": transition["t"], "end": None, "identity_id": transition["identity_id"],
                                  "basis_changes": [(transition["t"], transition["basis"])],
                                  "end_to": None, "end_reason": None}
                    self.known.setdefault(key, []).append(open_[key])
                else:
                    open_[key]["basis_changes"].append((transition["t"], transition["basis"]))
            elif key in open_:
                interval = open_.pop(key)
                interval.update(end=transition["t"], end_to=transition["to"], end_reason=transition["reason"])
        for interval in open_.values():
            interval["end"] = None  # still KNOWN when the records end

    def rel(self, ns: int) -> float:
        return (ns - self.t0_ns) / NS

    def in_window(self, items: list[dict], window: tuple[float, float]) -> list[dict]:
        return [item for item in items if window[0] <= item["t"] < window[1]]

    def known_overlapping(self, window: tuple[float, float]) -> list[tuple[tuple[int, int], dict]]:
        out = []
        for key, intervals in self.known.items():
            for interval in intervals:
                end = self.end_s if interval["end"] is None else interval["end"]
                if interval["start"] < window[1] and end > window[0]:
                    out.append((key, interval))
        return out

    def known_at(self, key: tuple[int, int], t: float) -> dict | None:
        for interval in self.known.get(key, ()):
            end = self.end_s if interval["end"] is None else interval["end"]
            if interval["start"] <= t < end:
                return interval
        return None

    def votes_known_by(self, key: tuple[int, int], t: float) -> list[dict]:
        """Applied votes on the track that the runtime had at time t, in frame order, still within the TTL."""
        return [v for v in self.votes.get(key, ()) if v["applied"] <= t + 1e-6 and t - v["t"] < VOTE_TTL_S]

    def justified(self, key: tuple[int, int], t: float, identity_id: str) -> bool:
        votes = self.votes_known_by(key, t)
        return len(votes) >= CONFIRMATIONS and all(
            v["vote"] == "match" and v["identity_id"] == identity_id for v in votes[-CONFIRMATIONS:])


# ---------------------------------------------------------------- F2 reading


def _account(directory: Path) -> tuple[dict[str, str] | None, str | None]:
    path = directory / face_account.ACCOUNT
    if not path.is_file():
        return None, "no account.txt"
    data = path.read_bytes()
    locked = face_account.lock_state(directory)
    if locked is None:
        return None, "the account is not locked (no account_sha256 in provenance.txt)"
    if hashlib.sha256(data).hexdigest() != locked:
        return None, "account.txt differs from its recorded SHA-256"
    return parse_account(data.decode("utf-8", "replace")), None


def _windows(account: dict[str, str], duration_s: float) -> dict[str, tuple[float, float]] | None:
    s = {key: stopwatch_seconds(account.get(key)) for key in face_account.TIMES}
    if any(value is None for value in s.values()):
        return None
    w = WASHOUT_S
    return {"E1": (s["out_of_view_1"] + w, s["seated_facing_1"] - w),
            "K1": (s["seated_facing_1"] + w, s["turned_away"] - w),
            "A": (s["turned_away"] + w, s["stood_up"] - w),
            "E2": (s["out_of_view_2"] + w, s["seated_facing_2"] - w),
            "K2": (s["seated_facing_2"] + w, duration_s - w)}


def _enrolled_id(f1: Path) -> str | None:
    listing = s5._load(f1 / "list.json")
    identities = s5._get(listing, "identities")
    if not isinstance(identities, list) or len(identities) != 1:
        return None
    identity_id = identities[0].get("identity_id") if isinstance(identities[0], dict) else None
    return identity_id if isinstance(identity_id, str) and ID.match(identity_id) else None


def _counts(directory: Path, name: str) -> Any:
    """``<path> <key>=<n>`` lines of a counts file, or MISSING."""
    path = directory / name
    if not path.is_file():
        return MISSING
    out = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9._/-]+) ([a-z_]+)=(\d+)", line)
        if match is None:
            return MISSING
        out[f"{match[1]} {match[2]}"] = int(match[3])
    return out or MISSING


def read_f2(directory: Path, f1: Path) -> Reading:
    r = Reading()
    result = s5._load(directory / "result.json")
    s5._supervision(r, result, "duration_stop", 0)
    start = s5._get(result, "run_output", "starting")
    r.item("memory policy selected: THP workload_disabled, release post_load", s5._get(start, "memory_policy"),
           s5._get(start, "memory_policy") == s5.CANDIDATE_POLICY)
    r.item("THP scope at each startup checkpoint", s5._get(start, "thp_scope"), s5._get(start, "thp_scope") == SCENE_SCOPE)
    disable = s5._get(start, "thp_disable")
    r.item("THP disable verified in the runtime process", disable,
           isinstance(disable, dict) and disable.get("verified") is True and disable.get("reason") is None)
    r.item("post-load releases, the face weights' included: every call returned 0", s5._get(start, "releases"),
           s5._get(start, "releases") == RELEASES)
    server = s5._get(start, "scene_server")
    r.item("scene server ready, 17/17 layers, vision encoder on GPU; scene and detector admitted",
           {"scene_server": server, "scene_problem": s5._get(start, "scene_problem"),
            "detector_problem": s5._get(start, "detector_problem")},
           isinstance(server, dict) and server.get("state") == "ready" and server.get("problem") is None
           and server.get("layers") == s5.LAYERS and server.get("vision_on_gpu") is True
           and s5._get(start, "scene_problem") is None and s5._get(start, "detector_problem") is None)
    r.item("face admitted for this validation run: one enrolled identity, no problem",
           {"face": s5._get(start, "face"), "face_problem": s5._get(start, "face_problem")},
           s5._get(start, "face") == {"validation_run": True, "identities_enrolled": 1}
           and s5._get(start, "face_problem") is None)
    r.item("no notification channel (Telegram disabled)", s5._get(start, "notifier_problems"),
           s5._get(start, "notifier_problems") == {})

    names = [c.get("name") for c in s5._list(s5._get(result, "captures")) if isinstance(c, dict)]
    captures = [s5._load(directory / f"status-{name}.json") for name in names]  # in the guard's time order
    seen = [{"face": s5._get(c, "components", "face", "state"),
             "validation_run": s5._get(c, "components", "face", "validation_run"),
             "enrolled": s5._get(c, "components", "face", "identities_enrolled"),
             "scene": s5._get(c, "components", "scene", "state"), "video": s5._get(c, "live", "video"),
             "channels": s5._get(c, "components", "notifications", "channels"),
             "delivery": s5._get(c, "delivery", "totals")} for c in captures]
    r.item("each status capture: face available (validation run, 1 enrolled), scene available, video fresh, "
           "no channel, no delivery row", seen if seen else MISSING,
           bool(seen) and not s5._has_missing(seen) and all(s["face"] == "available" and s["validation_run"] is True and s["enrolled"] == 1
                              and s["scene"] == "available" and s["video"] == "fresh" and s["channels"] == {}
                              and isinstance(s["delivery"], dict) and not any(s["delivery"].values()) for s in seen))
    s5._listeners(r, result, {18081, 18090})
    stopped = s5._get(result, "run_output", "stopped")
    r.item("clean shutdown: all stopped (the face worker included), database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True
           and (stopped.get("stopped") or {}).get("face_worker") is True)
    output = s5._get(result, "run_output")
    counts = {k: output.get(k) for k in ("lines", "non_json_lines")} if isinstance(output, dict) else MISSING
    r.item("D48: every stdout line is JSON", counts,
           isinstance(counts, dict) and counts["non_json_lines"] == 0 and (counts["lines"] or 0) >= 2)

    records, problems, _, _ = read_records(directory / "run.jsonl")
    total = sum(len(v) for v in records.values())
    last = captures[-1] if captures else MISSING
    dropped = s5._get(last, "components", "face", "records", "dropped")
    r.item("identity records: schema 1, complete, none dropped",
           {"records": {k: len(v) for k, v in records.items()}, "problems": problems[:5], "dropped": dropped}
           if total else MISSING,
           total > 0 and not problems and dropped == 0 and total < MAX_IDENTITY_RECORDS)

    cue = next((c for c in s5._list(s5._get(result, "cues")) if isinstance(c, dict) and c.get("name") == "t0"),
               MISSING)
    ready = s5._get(result, "child", "ready_mono_s")
    printed = s5._get(cue, "printed_mono_s")
    r.item("T0: the t0 cue printed when the runtime was ready", {"printed_mono_s": printed, "ready_mono_s": ready},
           isinstance(printed, float) and isinstance(ready, float) and 0 <= printed - ready <= CUE_TOLERANCE_S)
    account, problem = _account(directory)
    r.item("account saved and locked before any result was shown", problem or "locked",
           account is not None)
    if account is None:
        return r
    review = face_account.review(account)
    r.item("account complete and in order", review, review[-1] == "Complete and in order: yes")
    declared = {k: account.get(k) for k in ("stopwatch_started_at_t0", "stayed_until_stop", "other_faces_in_view")}
    r.item("stopwatch started at the T0 cue; stayed until the stop; no other face in view", declared,
           declared == {"stopwatch_started_at_t0": "yes", "stayed_until_stop": "yes", "other_faces_in_view": "no"})
    duration = s5._get(result, "parameters", "duration_s")
    windows = _windows(account, duration) if isinstance(duration, (int, float)) else None
    r.item(f"judged windows from the account (washout {WASHOUT_S:g} s), each at least {MIN_WINDOW_S:g} s",
           {k: [round(a, 1), round(b, 1)] for k, (a, b) in windows.items()} if windows else MISSING,
           windows is not None and all(b - a >= MIN_WINDOW_S for a, b in windows.values()))
    enrolled = _enrolled_id(f1)
    r.item("the enrolled identity, from F1's list", enrolled if enrolled else MISSING, enrolled is not None)
    if windows is None or enrolled is None or not isinstance(printed, float) or problems:
        return r
    t0_ns = round(printed * NS)
    line = Timeline(records, t0_ns, float(duration))
    _completion(r, line)
    _sampling(r, line, windows, enrolled)
    _identity(r, line, windows, enrolled, account)
    _counts_items(r, directory)
    _describe(r, result, line, windows, last)
    return r


def _completion(r: Reading, line: Timeline) -> None:
    offered = sum(1 for t in line.ticks if t["tick"] == "offered")
    completed = sum(1 for x in line.results if x["outcome"] != "failed")
    errors = sum(1 for x in line.results if x["outcome"] == "failed")
    value = {"offered": offered, "completed": completed, "processing_errors": errors,
             "completed_per_offered": round(completed / offered, 4) if offered else None}
    r.item(f"completion: completed / offered >= {MIN_COMPLETION} with 0 processing errors "
           "(the approved bounded-check criterion; not the replay profile's 0.95 Hz)",
           value, offered > 0 and completed <= offered and completed / offered >= MIN_COMPLETION and errors == 0)


def _sampling(r: Reading, line: Timeline, windows: dict, enrolled: str) -> None:
    for name in ("E1", "E2"):
        ticks = line.in_window(line.ticks, windows[name])
        r.item(f"{name} sampling: at least {MIN_TICKS_EMPTY} ticks", len(ticks), len(ticks) >= MIN_TICKS_EMPTY)
    for name in ("K1", "A", "K2"):
        ticks = line.in_window(line.ticks, windows[name])
        offered = sum(1 for t in ticks if t["tick"] == "offered")
        applied = sum(1 for x in line.in_window(line.results, windows[name]) if x["outcome"] == "applied")
        r.item(f"{name} sampling: at least {MIN_OFFERED} offered ticks and {MIN_APPLIED} applied results",
               {"offered": offered, "applied": applied}, offered >= MIN_OFFERED and applied >= MIN_APPLIED)
    for name in ("K1", "K2"):
        matches = sum(1 for x in line.in_window(line.results, windows[name]) if x["outcome"] == "applied"
                      for p in x["persons"] if p["vote"] == "match" and p["identity_id"] == enrolled)
        r.item(f"{name}: at least {MIN_MATCHES_FACING} match votes for the enrolled identity", matches,
               matches >= MIN_MATCHES_FACING)


def _identity(r: Reading, line: Timeline, windows: dict, enrolled: str, account: dict[str, str]) -> None:
    others = sorted({p["identity_id"] for x in line.results for p in x["persons"]
                     if p["vote"] == "match" and p["identity_id"] != enrolled}
                    | {t["identity_id"] for t in line.transitions if t["to"] == "known" and t["identity_id"] != enrolled})
    r.item("only the enrolled identity is ever matched or known", others, not others)
    spans = [(key, i["start"], line.end_s if i["end"] is None else i["end"], i["identity_id"])
             for key, intervals in line.known.items() for i in intervals]
    overlaps = [(a[0], b[0]) for n, a in enumerate(spans) for b in spans[n + 1:]
                if a[0] != b[0] and a[3] == b[3] and a[1] < b[2] and b[1] < a[2]]
    r.item("one identity is never known on two tracks at once", [list(map(list, o)) for o in overlaps], not overlaps)
    unjustified = [(t["stream_epoch"], t["track_id"], round(t["t"], 1)) for t in line.transitions
                   if t["to"] == "known" and t["from"] != "known"
                   and not line.justified((t["stream_epoch"], t["track_id"]), t["t"], t["identity_id"])]
    r.item(f"every change to known follows {CONFIRMATIONS} applied matches on that track within {VOTE_TTL_S:g} s",
           unjustified, not unjustified)
    for name in ("E1", "E2"):
        known = [(k, round(i["start"], 1)) for k, i in line.known_overlapping(windows[name])]
        r.item(f"{name}: no track is known while nobody is in view", known, not known)
    for name in ("K1", "K2"):
        reached = [(k, i) for k, i in line.known_overlapping(windows[name])
                   if i["identity_id"] == enrolled and i["basis_changes"][0][1] == "fresh"]
        r.item(f"{name}: the enrolled identity becomes known (fresh) while facing the camera",
               [(list(k), round(i["start"], 1)) for k, i in reached], bool(reached))
    _return(r, line, windows, account)
    _retention(r, line, windows, enrolled, account)


def _return(r: Reading, line: Timeline, windows: dict, account: dict[str, str]) -> None:
    left = stopwatch_seconds(account["out_of_view_2"])
    before = [(k, i) for k, intervals in line.known.items() for i in intervals if i["start"] < left]
    carried = [(list(k), round(i["start"], 1)) for k, i in before
               if i["end"] is None or i["end"] >= windows["E2"][0]]
    r.item("every identity known before leaving is cleared before the empty window E2", carried, not carried)
    late = []
    for key, interval in line.known_overlapping(windows["K2"]):
        votes = line.votes_known_by(key, interval["start"])[-CONFIRMATIONS:]
        if interval["start"] < left or any(v["t"] < left for v in votes):
            late.append((list(key), round(interval["start"], 1)))
    r.item("K2: nothing carried over: known again only from votes after the return", late, not late)


def _retention(r: Reading, line: Timeline, windows: dict, enrolled: str, account: dict[str, str]) -> None:
    turned = stopwatch_seconds(account["turned_away"])
    away = windows["A"]
    held = [(key, interval) for key, intervals in line.known.items() for interval in intervals
            if interval["identity_id"] == enrolled and interval["start"] <= turned
            and (interval["end"] is None or interval["end"] > turned)]
    name = "A: a known identity is retained on the same track, then cleared when its older confirming vote is " \
           f"{VOTE_TTL_S:g} s old"
    if account.get("face_hidden_while_turned") != "yes":
        r.not_exercised(name, "the account does not say the face was hidden while turned away")
        return
    if len(held) != 1:
        r.not_exercised(name, f"{len(held)} tracks known at the turn")
        return
    key, interval = held[0]
    offered = [t for t in line.in_window(line.ticks, away) if t["tick"] == "offered"]
    continuity = sum(1 for t in offered if key[1] in t["tracks"]) / len(offered) if offered else 0.0
    matches = [v for v in line.votes.get(key, ()) if v["vote"] == "match" and turned <= v["t"] < away[1]]
    r.describe("A: new matching evidence on the retained track after the turn (match votes)", len(matches))
    if continuity < MIN_CONTINUITY or interval["end_to"] == "cleared":
        r.not_exercised(name, {"track": list(key), "continuity": round(continuity, 3), "ended": interval["end_to"]})
        return
    if interval["end"] is None or interval["end"] > away[1]:
        recent = [v for v in matches if v["t"] >= away[1] - VOTE_TTL_S]
        value = {"still_known_at_window_end": True, "recent_match_votes": len(recent)}
        if recent:  # new matching evidence kept it known: retention itself was not tested
            r.not_exercised(name, value)
        else:  # no new evidence for 30 s, yet still known
            r.item(name, value, False)
        return
    support = line.votes_known_by(key, interval["end"] - 1e-3)[-CONFIRMATIONS:]
    older = support[0]["t"] if len(support) == CONFIRMATIONS else None
    expected = None if older is None else older + VOTE_TTL_S
    newest = support[-1]["t"] if support else None
    retained_at = next((t for t, basis in interval["basis_changes"] if basis == "retained"), None)
    value = {"track": list(key), "continuity": round(continuity, 3), "older_vote_s": None if older is None
             else round(older, 1), "cleared_s": round(interval["end"], 1), "cleared_to": interval["end_to"],
             "expected_clear_s": None if expected is None else round(expected, 1),
             "retained_from_s": None if retained_at is None else round(retained_at, 1)}
    if interval["end_to"] == "uncertain":
        r.not_exercised(name, {**value, "why": "a contradicting vote ended it"})
        return
    r.item(name, value, expected is not None and interval["end_to"] == "unresolved"
           and expected - EXACT_S <= interval["end"] <= expected + STATE_TOLERANCE_S
           and retained_at is not None and newest is not None
           and newest + FRESH_S - EXACT_S <= retained_at <= newest + FRESH_S + STATE_TOLERANCE_S)


COUNT_FILES = ("secret-counts-camera.txt", "secret-counts-identity.txt")
NOT_COUNTED = {"SHA256SUMS", "check.txt", *COUNT_FILES}  # written after the counts, or the counts themselves


def _counts_items(r: Reading, directory: Path) -> None:
    expected = {p.name for p in directory.iterdir() if p.is_file() and p.name not in NOT_COUNTED}
    for name, label in zip(COUNT_FILES, ("camera userinfo", "passphrase and name")):
        counts = _counts(directory, name)
        files = {line.split(" ")[0] for line in counts} if isinstance(counts, dict) else set()
        r.item(f"{label} counts: every evidence file counted, all 0",
               {"lines": len(counts), "nonzero": [k for k, v in counts.items() if v]} if isinstance(counts, dict)
               else MISSING, isinstance(counts, dict) and not any(counts.values()) and expected <= files)


def _describe(r: Reading, result: Any, line: Timeline, windows: dict, last: Any) -> None:
    r.describe("guard peak pressure, min MemFree, min MemAvailable (B)",
               [s5._get(result, "guard", k) for k in s5.GUARD_FIGURES])
    r.describe("face processing ms at the last capture", s5._get(last, "components", "face", "worker", "processing_ms"))
    span = max(line.end_s, 1e-9)
    r.describe("scheduling rate: ticks per second over the run, and per window (s^-1)",
               {"run": round(len([t for t in line.ticks if 0 <= t["t"] < span]) / span, 3),
                **{k: round(len(line.in_window(line.ticks, w)) / (w[1] - w[0]), 3) for k, w in windows.items()}})
    r.describe("results by outcome", {o: sum(1 for x in line.results if x["outcome"] == o) for o in sorted(OUTCOMES)})
    r.describe("offered ticks in the empty windows (person detections while nobody was in view)",
               {k: sum(1 for t in line.in_window(line.ticks, windows[k]) if t["tick"] == "offered")
                for k in ("E1", "E2")})
    usable = {}
    for k in ("K1", "A", "K2"):
        applied = [x for x in line.in_window(line.results, windows[k]) if x["outcome"] == "applied"]
        assigned = sum(1 for x in applied if any(p["ownership"] == "assigned" for p in x["persons"]))
        usable[k] = None if not applied else round(assigned / len(applied), 3)
    r.describe("usable-face rate (applied results with an assigned face) per window", usable)
    firsts = {}
    for k, start_key in (("K1", "seated_facing_1"), ("K2", "seated_facing_2")):
        known = sorted(i["start"] for _, i in line.known_overlapping((windows[k][0] - WASHOUT_S, windows[k][1])))
        firsts[k] = None if not known else round(known[0] - (windows[k][0] - WASHOUT_S), 1)
    r.describe("time to known after sitting down facing the camera (s)", firsts)


# ---------------------------------------------------------------- F1 reading


def _provenance(directory: Path) -> dict[str, str]:
    path = directory / "provenance.txt"
    if not path.is_file():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def _mode(path: Path) -> Any:
    try:
        info = os.lstat(path)
    except OSError:
        return MISSING
    return {"mode": oct(stat.S_IMODE(info.st_mode)), "owned": info.st_uid == os.getuid(),
            "regular": stat.S_ISREG(info.st_mode), "directory": stat.S_ISDIR(info.st_mode)}


def read_f1(directory: Path, identity_dir: Path) -> Reading:
    r = Reading()
    prov = _provenance(directory)
    exits = {k: prov.get(k) for k in ("dryrun_exit", "enroll_exit", "list_exit")}
    r.item("each identity command exited 0", exits if all(exits.values()) else MISSING,
           exits == {"dryrun_exit": "0", "enroll_exit": "0", "list_exit": "0"})
    supplied = prov.get("photos_supplied")
    supplied_n = int(supplied) if supplied and supplied.isdigit() else MISSING
    dry = s5._load(directory / "dryrun.json")
    r.item(f"dry run: photos reported by index, at least {MIN_USABLE_PHOTOS} usable, nothing written",
           {"photos": s5._get(dry, "photos"), "accepted": s5._get(dry, "accepted"),
            "dry_run": s5._get(dry, "dry_run"), "supplied": supplied_n,
            "gallery_before": prov.get("gallery_before_dryrun", MISSING),
            "gallery_after": prov.get("gallery_after_dryrun", MISSING)},
           s5._get(dry, "dry_run") is True and "identity_id" not in (dry if isinstance(dry, dict) else {})
           and s5._get(dry, "photos") == supplied_n and isinstance(s5._get(dry, "accepted"), int)
           and s5._get(dry, "accepted") >= MIN_USABLE_PHOTOS
           and [x.get("photo") for x in s5._list(s5._get(dry, "results"))] == list(range(1, (supplied_n or 0) + 1))
           and prov.get("gallery_before_dryrun") == "absent" and prov.get("gallery_after_dryrun") == "absent")
    enroll = s5._load(directory / "enroll.json")
    identity_id = s5._get(enroll, "identity_id")
    r.item(f"enrollment verified from {MIN_USABLE_PHOTOS}-{MAX_USABLE_PHOTOS} usable photos; exactly the supplied "
           "copies deleted; the folder removed",
           {k: s5._get(enroll, k) for k in ("verified", "identity_id", "photos", "accepted", "photos_deleted",
                                             "photos_changed_not_deleted", "folder_entries_left")},
           s5._get(enroll, "verified") is True and isinstance(identity_id, str) and bool(ID.match(identity_id))
           and s5._get(enroll, "photos") == supplied_n and s5._get(enroll, "photos_deleted") == supplied_n
           and s5._get(enroll, "photos_changed_not_deleted") == 0 and s5._get(enroll, "folder_entries_left") == 0
           and isinstance(s5._get(enroll, "accepted"), int)
           and MIN_USABLE_PHOTOS <= s5._get(enroll, "accepted") <= MAX_USABLE_PHOTOS
           and "refused" not in (enroll if isinstance(enroll, dict) else {}))
    listing = s5._get(s5._load(directory / "list.json"), "identities")
    r.item("the gallery holds exactly the enrolled identity, its prototypes and consent date", listing,
           isinstance(listing, list) and len(listing) == 1 and isinstance(listing[0], dict)
           and listing[0].get("identity_id") == identity_id
           and listing[0].get("prototypes") == s5._get(enroll, "accepted")
           and listing[0].get("consent_date") == prov.get("consent_date"))
    folder = prov.get("inbox_folder_name")
    states = {"identity_dir": _mode(identity_dir), "gallery": _mode(identity_dir / "gallery.sealed"),
              "inbox": _mode(identity_dir / "inbox"),
              "inbox_entries": len(list((identity_dir / "inbox").iterdir())) if (identity_dir / "inbox").is_dir()
              else MISSING,
              "dedicated_folder_gone": None if not folder else not (identity_dir / "inbox" / folder).exists()}
    r.item("private storage: directory and inbox 0700, gallery 0600, owned; the inbox empty", states,
           states["identity_dir"] != MISSING and states["gallery"] != MISSING and states["inbox"] != MISSING
           and states["identity_dir"]["mode"] == "0o700" and states["identity_dir"]["owned"]
           and states["inbox"]["mode"] == "0o700" and states["gallery"]["mode"] == "0o600"
           and states["gallery"]["regular"] and states["inbox_entries"] == 0 and states["dedicated_folder_gone"] is True)
    audit = []
    if (identity_dir / "audit.jsonl").is_file():
        audit = [json.loads(x) for x in (identity_dir / "audit.jsonl").read_text().splitlines() if x.strip()]
    enrolls = [a for a in audit if a.get("action") == "enroll"]
    r.item("audit: one enroll line for that identity, with its counts", enrolls if enrolls else MISSING,
           len(enrolls) == 1 and enrolls[0].get("identity_id") == identity_id
           and enrolls[0].get("photos_used") == s5._get(enroll, "accepted")
           and enrolls[0].get("photos_rejected") == (supplied_n if isinstance(supplied_n, int) else -1)
           - (s5._get(enroll, "accepted") or 0))
    counts = _counts(directory, "secret-counts-identity.txt")
    files = {line.split(" ")[0] for line in counts} if isinstance(counts, dict) else set()
    expected = {p.name for p in directory.iterdir() if p.is_file() and p.name not in NOT_COUNTED} | {"audit.jsonl"}
    r.item("passphrase, name and photo-file-name counts: every evidence file counted, all 0",
           {"lines": len(counts), "nonzero": [k for k, v in counts.items() if v]} if isinstance(counts, dict)
           else MISSING, isinstance(counts, dict) and not any(counts.values()) and expected <= files)
    return r


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="part", required=True)
    f1 = commands.add_parser("f1", help="read the consented enrollment's evidence")
    f1.add_argument("directory", type=Path)
    f1.add_argument("--identity-dir", type=Path, required=True)
    f2 = commands.add_parser("f2", help="read the guarded live validation's evidence")
    f2.add_argument("directory", type=Path)
    f2.add_argument("--f1", type=Path, required=True, help="F1's evidence directory (the enrolled ID)")
    args = parser.parse_args(argv)
    reading = read_f1(args.directory, args.identity_dir) if args.part == "f1" else read_f2(args.directory, args.f1)
    for line in reading.lines:
        print(line)
    print(f"face-{args.part}: {'validated' if reading.ok else 'not validated'}")
    return 0 if reading.ok else 1


if __name__ == "__main__":
    sys.exit(main())
