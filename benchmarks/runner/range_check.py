#!/usr/bin/env python3
"""Route B's range session (V3): the predeclared single-person reading, and the schedule the package's guard prints.

    range_check.py read RUN_DIR --f1 F1_DIR [--run RUN --identity-dir DIR --recheck-root ROOT]
    range_check.py cues        # the guard's --cue-at arguments, one per line (face_account.RANGE_CUES)
    range_check.py duration    # the run's duration in s (face_account.RANGE_DURATION_S)
    range_check.py config      # the session's face.yaml, exactly as the reading requires it

**The session.** One enrolled person, alone, on ``capture.stream_profile: main_single`` with face recognition and no
scene or chat (``sentinel run --face --face-validation``; the guard's ``--require-ready face_windows``). After an empty
room, five holds at each block: the seat (the control), 4.0 m facing the lens, 4.0 m acting naturally, 5.0 m facing,
5.0 m acting naturally; then three walk-ins. Each hold has three bells: walk to the mark with the face turned away and
wait there turned away (on the mark within 10 s of the 15 s); face the lens and hold still (30 s) or act naturally on
the mark (60 s); turn away and
leave the camera's view until the next bell, so that the next hold starts on a new track with nothing KNOWN. The marks'
distances, lens to face, are typed before launch (``marks.txt`` in the session directory, the run directory's
parent, its SHA-256 in the session's provenance before the launch line).

**Time base.** T0 is the guard's ``t0`` cue (printed at readiness). Every hold is placed by the guard's own printed
cue times. A vote counts as the mark's when its frame was ingested from the hold bell on and before the end bell. A
frame is ingested after it was captured, so the end needs no margin; at the start, a hold done as cued (the account)
had the operator on the mark, turned away, from 10 s after the walk bell, 5 s before the hold bell, so a frame ingested
from the hold bell on was captured on the mark unless the camera-to-ingest delay exceeded 5 s. That delay is an
assumption, not a measurement: it is not measured (the largest main-stream delivery backlog seen in
st-20261106T080730Z was 1.25 s). Cues are not movement times: the account says which holds were done as cued.

**Criteria, fixed before launch (session 116, part 14; unchanged):**

- **At the mark:** a hold's recognition counts only if both confirming votes (the last CONFIRMATIONS votes the runtime
  had on that track when it became KNOWN) come from frames inside the hold. A KNOWN already present at the hold bell,
  or a first KNOWN in the hold built on a vote from a frame before the hold bell, is ``not_at_mark``: counted apart,
  never a success.
- **Seat control:** KNOWN at the mark in 5 of 5 holds (at any time in the hold: the 10 s limit is the facing holds').
- **Facing the lens, at 4.0 m and at 5.0 m:** KNOWN at the mark within ACQUIRE_S of the hold bell in at least 4 of 5
  holds. Also reported: whether KNOWN then lasted to the hold's end.
- **Natural activity, at 4.0 m and at 5.0 m:** five 60 s holds, reported per hold (acquisition time and whether at the
  mark, KNOWN duration and share of the hold, interruptions and their durations, longest gap); judged only on wrong
  associations, which must be zero. A natural hold in which no face was assigned to a person exercised no association,
  so its block reads NOT_EXERCISED (a condition of the criterion, never a bar on recognition). Bars on the others are set
  from this session's data and predeclared before any acceptance run.
- **A block is NOT_EXERCISED** when its holds cannot settle it: its mark was not measured within MARK_TOLERANCE_M, the
  account's answer for it is unknown, someone else or a face picture may have been in view, or the holds not done as
  cued (left out by number) and the ``not_at_mark`` holds leave the criterion unsettled. A block whose misses already
  make the criterion unreachable FAILS: the shortfall is reported unmet. A natural block needs all five holds as cued.
- **Walk-ins:** EXPLORATORY only, labelled so: the time and face width at the first KNOWN; never a substitute for the
  holds and never judged.

**Wrong associations in a one-person session** are what such a session can show: the enrolled identity KNOWN on two
tracks at the same time (one person is at most one of them), or a KNOWN of any other identity. Side-by-side and
crossing ownership needs a second person: the two-person check (V4) is PENDING, as is the non-enrolled check.

**Four layers, kept apart** in each block's DESCRIPTIVE lines, from each hold's vote window: person detection (the face
ticks' people), face detection (one-person results: faces assigned, their width in the main stream's pixels and score,
the best candidate under 0.9 where none was kept; results with more people apart), matching (the vote labels), and
retained identity (KNOWN from the hold bell to its end bell). Each block's face width is shown beside the seat's,
scaled by the typed distances, as a cross-check of where the face was; it is never judged.

**Unchanged:** YuNet 0.9, match 0.70, margin 0.05, quality 0.6, two confirmations, 30 s votes, the association rule and
route B's same-frame checks (face_check: schema 8, every frame reference, the startup's markers). The session's
face.yaml must be RANGE_CONFIG exactly, so every identity and tracker setting is the runtime's default.

Each item prints PASS, FAIL, MISSING or NOT_EXERCISED; exit 0 only when every item passed. Nothing is written, no
similarity, identity ID, picture or note is printed. A pass is one person's session: never a rate, never range
acceptance (V4 is pending) and never beta readiness. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import face_account as fa
import face_check as fc
import step5_check as s5

MISSING = s5.MISSING
NS = fc.NS
READINESS = "face_windows"  # step5_guard --require-ready: route B's face on head windows, no scene or chat
FACE_ONLY_SCOPE = {"before_launch": {"runtime": 1}, "workload_verified": {"runtime": 0}}  # no scene server
FACE_ONLY_RELEASES = {"detector": {"engine": "returned_0"}, "face": fc.FACE_RELEASES}
STATUS_PORTS = {18090}  # the status page only: no scene server
ARRIVAL_MARGIN_S = fa.RANGE_APPROACH_S - fa.RANGE_ARRIVAL_S  # the camera delay a hold as cued absorbs (an assumption)
ACQUIRE_S = 10.0  # part 14: KNOWN within 10 s of the hold's start
REQUIRED = {"seat": 5, "facing": 4}  # of RANGE_HOLDS_PER_BLOCK: the seat 5 of 5, each mark facing 4 of 5
MARK_TOLERANCE_M = 0.1
MARKS_M = {"mark4": 4.0, "mark5": 5.0}
SEAT_RANGE_M = (0.3, 2.0)  # a plausible seat distance; the seat is the control, not a range
EMPTY_WINDOW_S = (25.0, float(fa.RANGE_FIRST_GO_S) - 5.0)  # after leaving at T0, before the first walk bell
MARKS_FILE = "marks.txt"
MARKS_KEYS = ("seat_m", "mark4_m", "mark5_m", "measured_with", "out_of_view_spot")
MEASURED_WITH = ("tape", "laser")
LAUNCH_LINE = "RV_launch_utc="  # the session provenance's line written just before the guard starts
RANGE_CONFIG = (
    "config_version: 1\n"
    "camera: {id: cam-1}\n"
    "capture: {stream_profile: main_single}\n"
    "adapters:\n"
    "  - {adapter_id: legacy-deepface-facenet512-yunet, contract_version: 1, implementation_revision: \"1\", "
    "enabled: true, input_kinds: [frame], output_kinds: [face.observation], model_revision: "
    "facenet512-yunet-deepface-0.0.99, resource_profile_id: step4cand-demo-20261007T090339Z, timeout_ms: 3000}\n")
PENDING = (
    "PENDING the non-enrolled check (V4; needs the consenting participant): zero KNOWN anywhere and every match vote "
    "listed, at the seat and at both marks. Nothing in this session stands in for it",
    "PENDING the two-person check (V4; needs the participant): side by side at the marks and crossing, recognition and "
    "ownership per person, zero wrong associations, AMBIGUOUS left unresolved. Nothing in this session stands in for it",
    "LIMIT one enrolled person alone, one session, one boot, one order of blocks: counts are k of n holds, never a rate; "
    "holds seconds apart are not independent. A pass is not range acceptance (V4 is pending) and not beta readiness")


def _median(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 1) if values else None


def _provenance_lines(session: Path) -> list[str]:
    path = session / "provenance.txt"
    return path.read_bytes().decode("utf-8", "replace").splitlines() if path.is_file() else []


def _before_launch(session: Path, wanted: str) -> bool:
    """``wanted`` is a line of the session's provenance, before its first launch line."""
    lines = _provenance_lines(session)
    found = [i for i, line in enumerate(lines) if line == wanted]
    launched = [i for i, line in enumerate(lines) if line.startswith(LAUNCH_LINE)]
    return bool(found and launched and found[0] < launched[0])


def _marks(session: Path) -> tuple[dict[str, Any] | None, str]:
    """The typed marks (numbers as typed) and whether they were recorded before the launch; (None, why) otherwise."""
    path, provenance = session / MARKS_FILE, session / "provenance.txt"
    if not path.is_file() or not provenance.is_file():
        return None, f"no {MARKS_FILE} or provenance.txt in the session directory"
    data = path.read_bytes()
    values = fc.parse_account(data.decode("utf-8", "replace"))
    if set(values) != set(MARKS_KEYS):
        return None, f"{MARKS_FILE} keys differ from {list(MARKS_KEYS)}"
    if not _before_launch(session, f"marks_sha256={hashlib.sha256(data).hexdigest()}"):
        return None, f"{MARKS_FILE} is not the file recorded in the session's provenance before the launch"
    out: dict[str, Any] = {  # fixed labels only: a typed value outside them is never printed
        "measured_with": values["measured_with"] if values["measured_with"] in MEASURED_WITH else "other",
        "out_of_view_spot": values["out_of_view_spot"] if values["out_of_view_spot"] in ("yes", "no") else "other"}
    for key in ("seat_m", "mark4_m", "mark5_m"):
        try:
            out[key] = float(values[key])
        except ValueError:
            return None, f"{key} is not a number"
    return out, "recorded before the launch"


def _left_out(value: str | None, count: int) -> frozenset[int] | None:
    """An account's hold numbers within 1..count; None (unknown, settling nothing) for anything else."""
    numbers = fa.hold_numbers(value)
    return numbers if numbers is not None and numbers <= set(range(1, count + 1)) else None


def _mark_ok(marks: dict[str, Any] | None, mark: str) -> bool:
    if marks is None or marks.get("measured_with") not in MEASURED_WITH or marks.get("out_of_view_spot") != "yes":
        return False
    if mark == "seat":
        return SEAT_RANGE_M[0] <= marks["seat_m"] <= SEAT_RANGE_M[1]
    return abs(marks[f"{mark}_m"] - MARKS_M[mark]) <= MARK_TOLERANCE_M + 1e-9


class Votes:
    """Each track's applied votes with the assigned face's width and score (Timeline's votes plus the face numbers)."""

    def __init__(self, line: fc.Timeline) -> None:
        self.by_key: dict[tuple[int, int], list[dict]] = {}
        for result in line.results:
            if result["outcome"] != "applied":
                continue
            for person in result["persons"]:
                if person["vote"] in ("match", "unknown"):
                    face = person.get("face") or {}
                    self.by_key.setdefault((result["stream_epoch"], person["track_id"]), []).append(
                        {"t": result["t"], "applied": result["applied"], "vote": person["vote"],
                         "identity_id": person["identity_id"], "w_px": face.get("w_px")})
        for votes in self.by_key.values():
            votes.sort(key=lambda v: v["t"])

    def confirming(self, key: tuple[int, int], t: float) -> list[dict]:
        held = [v for v in self.by_key.get(key, ()) if v["applied"] <= t + 1e-6 and t - v["t"] < fc.VOTE_TTL_S]
        return held[-fc.CONFIRMATIONS:]


def _known(line: fc.Timeline, start: float, end: float) -> list[tuple[tuple[int, int], float, float, dict]]:
    """Every KNOWN interval overlapping [start, end), clipped to it: (track key, from, to, interval)."""
    out = []
    for key, interval in line.known_overlapping((start, end)):
        stop = line.end_s if interval["end"] is None else interval["end"]
        out.append((key, max(start, interval["start"]), min(end, stop), interval))
    return sorted(out, key=lambda item: item[1])


def _cover(spans: list[tuple[float, float]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def _wrong(known: list[tuple[tuple[int, int], float, float, dict]], enrolled: str) -> int:
    """One person's session: the enrolled identity KNOWN on two tracks at once (each overlapping pair of intervals on
    different tracks), and any KNOWN of another identity."""
    wrong = sum(1 for _, _, _, interval in known if interval["identity_id"] != enrolled)
    own = [(key, a, b) for key, a, b, interval in known if interval["identity_id"] == enrolled]
    for i, (key, a, b) in enumerate(own):
        wrong += sum(1 for other, c, d in own[i + 1:] if other != key and min(b, d) - max(a, c) > 0)
    return wrong


def _layers(line: fc.Timeline, start: float, end: float) -> dict[str, Any]:
    """Person detection, face detection and matching over [start, end) by frame time (D88's one denominator: the person
    on each applied result whose frame had exactly one person; results with more people apart)."""
    ticks = line.in_window(line.ticks, (start, end))
    tracks = {track for tick in ticks for track in tick["tracks"]}
    results = line.in_window(line.results, (start, end))
    applied = [x for x in results if x["outcome"] == "applied"]
    single = [x for x in applied if len(x["persons"]) == 1]
    people = [x["persons"][0] for x in single]
    faces = [p["face"] for p in people if p["ownership"] == "assigned" and p.get("face")]
    without = [x["detection"]["best_below_score"] for x, p in zip(single, people)
               if p["ownership"] != "assigned" and isinstance(x.get("detection"), dict)
               and x["detection"].get("best_below_score") is not None]
    more = [p["ownership"] for x in applied if len(x["persons"]) > 1 for p in x["persons"]]
    return {
        "person": {"ticks": len(ticks), **dict(sorted(Counter(t["tick"] for t in ticks).items())),
                   "tracks": len(tracks), "most_tracks_in_a_tick": max((len(t["tracks"]) for t in ticks), default=0)},
        "face": {"results": dict(sorted(Counter(x["outcome"] for x in results).items())),
                 "one_person": len(single), "more_people_apart": len(applied) - len(single),
                 **dict(sorted(Counter(p["ownership"] for p in people).items())),
                 "w_px_median": _median([f["w_px"] for f in faces]),
                 "w_px_range": [min(f["w_px"] for f in faces), max(f["w_px"] for f in faces)] if faces else None,
                 "score_median": round(statistics.median(f["score"] for f in faces), 3) if faces else None,
                 "no_face_best_candidate_under_0.9": round(max(without), 3) if without else None,
                 **({"more_people_ownership": dict(sorted(Counter(more).items()))} if more else {})},
        "matching": dict(sorted(Counter(p["label"] for p in people).items())),
    }


def _hold(line: fc.Timeline, votes: Votes, enrolled: str, hold: float, end: float) -> dict[str, Any]:
    """One hold, from its hold bell to its end bell (s after T0)."""
    known = _known(line, hold, end)
    at_start = any(interval["identity_id"] == enrolled and interval["start"] < hold for _, _, _, interval in known)
    entries = [(interval["start"], key, interval) for key, _, _, interval in known
               if interval["identity_id"] == enrolled and hold <= interval["start"] < end]
    out: dict[str, Any] = {"outcome": "none", "acquired_s": None, "vote_frames_s": None, "vote_w_px": None,
                           "known_at_start": at_start}
    if entries:
        t_k, key, _ = min(entries, key=lambda e: e[0])
        confirming = votes.confirming(key, t_k)
        justified = line.justified(key, t_k, enrolled)
        in_hold = len(confirming) == fc.CONFIRMATIONS and all(v["t"] >= hold for v in confirming)
        out.update(acquired_s=round(t_k - hold, 2), vote_frames_s=[round(v["t"] - hold, 2) for v in confirming],
                   vote_w_px=[v["w_px"] for v in confirming])
        out["outcome"] = ("not_at_mark" if at_start or (justified and not in_hold) else "not_justified" if not justified
                          else "acquired" if t_k - hold <= ACQUIRE_S + 1e-9 else "late")
    elif at_start:
        out["outcome"] = "not_at_mark"
    cover = _cover([(a, b) for _, a, b, interval in known if interval["identity_id"] == enrolled])
    first = next((a for a, _ in cover), None)
    gaps = [round(c - b, 2) for (_, b), (c, _) in zip(cover, cover[1:])] if cover else []
    if cover and cover[-1][1] < end - 1e-6:
        gaps.append(round(end - cover[-1][1], 2))  # lost before the end bell, not regained in the hold
    known_s = sum(b - a for a, b in cover)
    out.update(known_s=round(known_s, 1), known_share=round(known_s / (end - hold), 3), interruptions=len(gaps),
               gaps_s=gaps, longest_gap_s=max(gaps, default=0.0),
               lasted_to_end=bool(cover) and cover[-1][1] >= end - 1e-6,
               known_from_s=None if first is None else round(first - hold, 2), wrong=_wrong(known, enrolled))
    return out


def _judge(r: fc.Reading, block: str, kind: str, name: str, holds: dict[int, dict], left_out: frozenset[int] | None,
           mark_ok: bool, alone: bool, mark_m: Any, assigned: dict[int, int] | None = None) -> None:
    total = fa.RANGE_HOLDS_PER_BLOCK
    summary = {str(n): h["outcome"] + ("" if h["acquired_s"] is None else f" {h['acquired_s']} s")
               + (" (not as cued)" if left_out is not None and n in left_out else "") for n, h in holds.items()}
    eligible = [n for n in holds if left_out is not None and n not in left_out]
    value: dict[str, Any] = {"mark_m": mark_m, "holds": summary,
                             "not_as_cued": sorted(left_out) if left_out is not None else "unknown"}
    blocked = [why for why, bad in (("mark not measured within 0.1 m, or no out-of-view spot", not mark_ok),
                                    ("account: holds not as cued unknown", left_out is None),
                                    ("account: another face may have been in view", not alone)) if bad]
    if kind == "natural":
        wrong = sum(h["wrong"] for h in holds.values())
        unexercised = sorted(n for n in eligible if not (assigned or {}).get(n))
        value.update(wrong_associations=wrong, holds_as_cued=len(eligible), holds_with_no_face_assigned=unexercised)
        text = (f"{name}: zero wrong associations over {total} natural 60 s holds at the mark, each with a face assigned "
                "(one person: the enrolled identity KNOWN on two tracks at once, or another identity KNOWN)")
        if wrong:
            r.item(text, value, False)
        elif blocked or len(eligible) < total or unexercised:
            r.not_exercised(text, {**value, "why": blocked or ([f"{total - len(eligible)} hold(s) not as cued"]
                                                               if len(eligible) < total else
                                                               ["no face assigned in a hold: no association exercised"])})
        else:
            r.item(text, value, True)
        return
    seat = block == "seat"  # part 14: the seat KNOWN at the mark in 5 of 5; the facing holds within 10 s in 4 of 5
    required = REQUIRED["seat" if seat else "facing"]
    good = ("acquired", "late") if seat else ("acquired",)
    successes = sum(1 for n in eligible if holds[n]["outcome"] in good)
    misses = sum(1 for n in eligible if holds[n]["outcome"] in ("none", "not_justified", *(() if seat else ("late",))))
    value.update(successes=successes, misses=misses,
                 not_at_mark=sum(1 for n in eligible if holds[n]["outcome"] == "not_at_mark"))
    text = (f"{name}: KNOWN at the mark (both confirming votes from frames in the hold) "
            + ("" if seat else f"within {ACQUIRE_S:g} s of the hold bell ") + f"in at least {required} of {total} holds")
    if blocked:
        r.not_exercised(text, {**value, "why": blocked})
    elif successes >= required:
        r.item(text, value, True)
    elif misses > total - required:
        r.item(text + " (unmet)", value, False)
    else:
        r.not_exercised(text, {**value, "why": ["holds not as cued or not_at_mark leave it unsettled"]})


def _describe_block(r: fc.Reading, name: str, kind: str, holds: dict[int, dict], layers: dict[int, dict],
                    estimate: float | None) -> None:
    for layer, text in (("person", "person detection (face ticks by frame time in each hold's vote window: ticks by "
                                   "kind, distinct tracks, the most tracks in one tick)"),
                        ("face", "face detection (results by outcome; one-person results: ownership, the assigned "
                                 "faces' width in main-stream pixels and score, the best candidate under 0.9 where no "
                                 "face was kept; results with more people apart)"),
                        ("matching", "matching (the vote labels on one-person results)")):
        r.describe(f"{name} {text}", {str(n): layers[n][layer] for n in holds})
    keys = ("acquired_s", "known_from_s", "vote_frames_s", "known_s", "known_share", "interruptions", "gaps_s",
            "longest_gap_s", "lasted_to_end", "wrong")
    r.describe(f"{name} retained identity (s from the hold bell: the first KNOWN entry, its confirming votes' frames, "
               "KNOWN time and share of the hold, interruptions and their durations, the longest gap, KNOWN at the end "
               "bell, wrong associations)", {str(n): {k: h[k] for k in keys} for n, h in holds.items()})
    widths = [w for h in holds.values() for w in (h["vote_w_px"] or []) if w is not None]
    r.describe(f"{name} face width cross-check (never judged): the confirming votes' widths beside the seat's median "
               "scaled by the typed distances", {"confirming_w_px_median": _median(widths),
                                                 "estimate_from_seat_w_px": estimate})


def read_range(directory: Path, f1: Path, gate: tuple[Path, Path, Path | None] | None = None) -> fc.Reading:
    r = fc.Reading()
    _execution_and_holds(r, directory, f1, gate)
    for text in PENDING:
        r.lines.append(text)
    return r


def _execution_and_holds(r: fc.Reading, directory: Path, f1: Path, gate: Any) -> None:
    if gate is not None:
        ok, lines = fc.f1_gate(*gate)
        shown = {k: ("recorded" if k == "f1_identity_id" else v)  # never the identity ID itself
                 for k, v in (line.split("=", 1) for line in lines if "=" in line)}
        r.item("F1 basis for this run: the run's own F1 validated, or the maintainer's accepted corrected F1 reading "
               "of it", shown, ok and f1.resolve() == (gate[0] / "f1").resolve())
    result = s5._load(directory / "result.json")
    s5._supervision(r, result, "duration_stop", 0)
    start = s5._get(result, "run_output", "starting")
    r.item("memory policy selected: THP workload_disabled, release post_load", s5._get(start, "memory_policy"),
           s5._get(start, "memory_policy") == s5.CANDIDATE_POLICY)
    r.item("THP scope at each startup checkpoint (no scene server)", s5._get(start, "thp_scope"),
           s5._get(start, "thp_scope") == FACE_ONLY_SCOPE)
    disable = s5._get(start, "thp_disable")
    r.item("THP disable verified in the runtime process", disable,
           isinstance(disable, dict) and disable.get("verified") is True and disable.get("reason") is None)
    r.item("post-load releases, the detector's and the face weights': every call returned 0",
           s5._get(start, "releases"), s5._get(start, "releases") == FACE_ONLY_RELEASES)
    r.item("no scene server or scene analysis; detector loaded",
           {k: s5._get(start, k) for k in ("scene_server", "scene_problem", "detector_problem")},
           isinstance(start, dict) and start.get("scene_server") is None and start.get("scene_problem") is None
           and start.get("detector_problem") is None)
    r.item("face admitted for this validation run on route B (main_single, head windows as pinned): one enrolled "
           "identity, no problem", {"face": s5._get(start, "face"), "face_problem": s5._get(start, "face_problem")},
           s5._get(start, "face") == fc.WINDOWS_FACE and s5._get(start, "face_problem") is None)
    r.item("no notification channel (Telegram disabled)", s5._get(start, "notifier_problems"),
           s5._get(start, "notifier_problems") == {})
    readiness = s5._get(result, "readiness")
    seen_ready = {k: s5._get(readiness, k) for k in ("required", "met", "reason")}
    r.item(f"the guard's readiness: {READINESS} met before T0", seen_ready,
           seen_ready == {"required": READINESS, "met": True, "reason": None})
    names = [c.get("name") for c in s5._list(s5._get(result, "captures")) if isinstance(c, dict)]
    captures = [s5._load(directory / f"status-{name}.json") for name in names]
    seen = [{"face": s5._get(c, "components", "face", "state"),
             "validation_run": s5._get(c, "components", "face", "validation_run"),
             "enrolled": s5._get(c, "components", "face", "identities_enrolled"),
             "stream": s5._get(c, "components", "capture", "stream", "profile"),
             "scene": s5._get(c, "components", "scene", "state"), "video": s5._get(c, "live", "video"),
             "channels": s5._get(c, "components", "notifications", "channels"),
             "delivery": s5._get(c, "delivery", "totals")} for c in captures]
    r.item("each status capture: face available (validation run, 1 enrolled), stream main_single, scene disabled, "
           "video fresh, no channel, no delivery row", seen if seen else MISSING,
           bool(seen) and not s5._has_missing(seen)
           and all(s["face"] == "available" and s["validation_run"] is True and s["enrolled"] == 1
                   and s["stream"] == fc.STREAM_MAIN_SINGLE and s["scene"] == "disabled" and s["video"] == "fresh"
                   and s["channels"] == {} and isinstance(s["delivery"], dict) and not any(s["delivery"].values())
                   for s in seen))
    s5._listeners(r, result, STATUS_PORTS)
    stopped = s5._get(result, "run_output", "stopped")
    r.item("clean shutdown: all stopped (the face worker included), database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True
           and isinstance(stopped.get("stopped"), dict) and stopped["stopped"].get("face_worker") is True)
    output = s5._get(result, "run_output")
    counts = {k: output.get(k) for k in ("lines", "non_json_lines")} if isinstance(output, dict) else MISSING
    r.item("D48: every stdout line is JSON", counts,
           isinstance(counts, dict) and counts["non_json_lines"] == 0 and isinstance(counts["lines"], int)
           and counts["lines"] >= 2)
    config = directory.parent / "face.yaml"
    wanted = hashlib.sha256(RANGE_CONFIG.encode()).hexdigest()
    same = config.is_file() and config.read_bytes() == RANGE_CONFIG.encode()
    launched = _before_launch(directory.parent, f"RV_config_sha256={wanted}")
    r.item("the session's face.yaml, as the launch block ran it (its SHA-256 recorded just before the launch line): route "
           "B's stream profile and the face adapter only, every identity and tracker setting the runtime's default (the "
           "runtime records no configuration path; the tie is the package's fixed command)",
           {"file": "as required" if same else "differs" if config.is_file() else MISSING,
            "recorded_before_launch": launched}, same and launched)

    records, problems, starting, _ = fc.read_records(directory / "run.jsonl")
    total = sum(len(v) for v in records.values())
    schemas = sorted({item["schema"] for kind in records.values() for item in kind})
    if schemas and schemas != [fc.WINDOWS_SCHEMA]:
        problems.append(f"schema {schemas}, not route B's {fc.WINDOWS_SCHEMA}")
    problems.extend(fc._route_b_startup_problems(start, starting, True))
    if schemas == [fc.WINDOWS_SCHEMA]:
        problems.extend(fc.frame_problems(records))
    last = captures[-1] if captures else MISSING
    dropped = s5._get(last, "components", "face", "records", "dropped")
    r.item(f"identity records: schema {fc.WINDOWS_SCHEMA}, complete, none dropped; route B's startup markers and every "
           "frame reference consistent (same-frame association)",
           {"records": {k: len(v) for k, v in records.items()}, "problems": problems[:5], "dropped": dropped}
           if total else MISSING,
           total > 0 and not problems and dropped == 0 and total < fc.MAX_IDENTITY_RECORDS
           and not records.get("comparison") and not records.get("capture"))
    fc._route_b_items(r, records)

    ready = s5._get(result, "child", "ready_mono_s")
    cues = fc._cue_offsets(result)
    duration = s5._get(result, "parameters", "duration_s")
    r.item("T0: the t0 cue printed when the runtime was ready", {"t0_s_after_ready": cues.get("t0", MISSING)},
           "t0" in cues and 0 <= cues["t0"] <= fc.CUE_TOLERANCE_S)
    late = {name: round(cues[name] - at, 3) for name, at in fa.RANGE_CUE_AT_S.items()
            if name in cues and not 0 <= cues[name] - at <= fc.CUE_TOLERANCE_S}
    absent = sorted(set(fa.RANGE_CUE_AT_S) - set(cues))
    extra = sorted(set(cues) - set(fa.RANGE_CUE_AT_S))
    r.item(f"the run's {fa.RANGE_DURATION_S} s and all {len(fa.RANGE_CUE_AT_S)} cues of the schedule, each printed "
           f"within {fc.CUE_TOLERANCE_S:g} s of its time", {"duration_s": duration, "late": late, "absent": absent,
                                                            "extra": extra},
           duration == fa.RANGE_DURATION_S and not (late or absent or extra))
    account, problem, fmt = fc._account(directory)
    r.item("account saved and locked before any result was shown", problem or "locked", account is not None)
    review = (fa.review_separate(account, fa.FIELDS_V12) if account is not None and fmt == 12
              else [f"format {fmt} is not this session's account (format 12)"])
    if account is not None:
        r.item("account in format 12, every answer accepted (unknown is accepted)", review,
               fmt == 12 and review[-1].endswith(": yes"))
    marks, why = _marks(directory.parent)
    r.item("the marks, lens to face, typed and recorded before the launch (4.0 and 5.0 m within 0.1 m; the seat; "
           "tape or laser; an out-of-view spot)", {**marks, "recorded": why} if marks else MISSING if why.startswith("no ")
           else why, marks is not None and all(_mark_ok(marks, m) for m in ("seat", "mark4", "mark5")))
    fc._counts_items(r, directory)
    enrolled = fc._enrolled_id(f1)
    r.item("the enrolled identity, from F1's list", "one identity" if enrolled else MISSING, enrolled is not None)
    if account is None or fmt != 12 or problems or enrolled is None or not isinstance(ready, float) \
            or "t0" not in cues or absent or not isinstance(duration, (int, float)):
        for _, _, kind, _, name in fa.RANGE_BLOCKS:  # holds are read only from usable records, cues and a locked account
            r.not_exercised(f"{name}: not read", "the records, the schedule's printed cues, the account or the enrolled "
                            "identity above are not usable, so no hold of this block was read")
        return
    rel = {name: cues[name] - cues["t0"] for name in cues}  # s after T0, as the Timeline's times
    line = fc.Timeline(records, round((cues["t0"] + ready) * NS), float(duration))
    votes = Votes(line)
    alone = account.get("other_faces_in_view") == "no"
    seat_widths: list[float] = []
    plans: dict[str, list[dict]] = {}
    for plan in fa.RANGE_HOLD_PLAN:
        plans.setdefault(plan["block"], []).append(plan)
    read: dict[str, tuple[dict[int, dict], dict[int, dict]]] = {}
    for block, block_plans in plans.items():
        holds, layers = {}, {}
        for plan in block_plans:
            hold, end = rel[plan["cues"][1]], rel[plan["cues"][2]]
            holds[plan["number"]] = _hold(line, votes, enrolled, hold, end)
            layers[plan["number"]] = _layers(line, hold, end)
        read[block] = (holds, layers)
    left = {block: _left_out(account.get(f"{block}_holds_not_as_cued"), fa.RANGE_HOLDS_PER_BLOCK) for block in plans}
    for n, h in read["seat"][0].items():
        if left["seat"] is not None and n not in left["seat"] and h["outcome"] == "acquired":
            seat_widths += [w for w in h["vote_w_px"] if w is not None]
    seat_w = statistics.median(seat_widths) if seat_widths else None
    empty = _layers(line, *EMPTY_WINDOW_S)
    empty_known = _known(line, *EMPTY_WINDOW_S)
    r.describe(f"the empty room ({EMPTY_WINDOW_S[0]:g}-{EMPTY_WINDOW_S[1]:g} s after T0; the account: nobody in view "
               f"{account.get('empty_at_start')}): person, face and matching layers, and KNOWN intervals (each one there "
               "is a false detection or a wrong association)", {**empty, "known_intervals": len(empty_known)})
    for block, _, kind, mark, name in fa.RANGE_BLOCKS:
        holds, layers = read[block]
        mark_m = marks.get(f"{mark}_m") if marks else None
        _judge(r, block, kind, name, holds, left[block], _mark_ok(marks, mark), alone, mark_m,
               {n: layers[n]["face"].get("assigned", 0) + layers[n]["face"].get("more_people_ownership", {}).get("assigned", 0)
                for n in holds})
        estimate = (round(seat_w * marks["seat_m"] / mark_m, 1)
                    if seat_w is not None and marks and mark != "seat" and mark_m else None)
        _describe_block(r, name, kind, holds, layers, estimate)
    walk_left = _left_out(account.get("walk_ins_not_as_cued"), fa.RANGE_WALK_INS)
    for plan in fa.RANGE_WALK_PLAN:
        go, end = rel[plan["cues"][0]], rel[plan["cues"][1]]
        walk = _hold(line, votes, enrolled, go, end)
        r.lines.append("EXPLORATORY walk-in {} (not judged; never a substitute for the holds): {}".format(
            plan["number"], s5._text({"as_cued": None if walk_left is None else plan["number"] not in walk_left,
                                      "first_known_s_after_walk_bell": walk["acquired_s"],
                                      "known_at_walk_bell": walk["known_at_start"],
                                      "confirming_w_px": walk["vote_w_px"], "known_s": walk["known_s"],
                                      **_layers(line, go, end)["face"]})))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("read", help="read a range session's run directory")
    read.add_argument("directory", type=Path)
    read.add_argument("--f1", type=Path, required=True, help="F1's evidence directory (the enrolled ID)")
    read.add_argument("--run", type=Path, help="with --identity-dir: also require this run's F1 basis")
    read.add_argument("--identity-dir", type=Path)
    read.add_argument("--recheck-root", type=Path)
    commands.add_parser("cues", help="the guard's --cue-at arguments, one per line")
    commands.add_parser("duration", help="the run's duration in s")
    commands.add_parser("config", help="the session's face.yaml, exactly")
    args = parser.parse_args(argv)
    if args.command == "cues":
        print("\n".join(f"{at}:{name}:{text}" for name, at, text in fa.RANGE_CUES))
        return 0
    if args.command == "duration":
        print(fa.RANGE_DURATION_S)
        return 0
    if args.command == "config":
        sys.stdout.write(RANGE_CONFIG)
        return 0
    if (args.run is None) != (args.identity_dir is None):
        parser.error("read: --run and --identity-dir go together")
    gate = None if args.run is None else (args.run, args.identity_dir, args.recheck_root)
    reading = read_range(args.directory, args.f1, gate)
    for text in reading.lines:
        print(text)
    print("face-range: " + ("every predeclared single-person item passed; range is not accepted (V4 pending) and this "
                            "is not beta readiness" if reading.ok else "not passed (see the FAIL, MISSING and "
                            "NOT_EXERCISED items)"))
    return 0 if reading.ok else 1


if __name__ == "__main__":
    sys.exit(main())
