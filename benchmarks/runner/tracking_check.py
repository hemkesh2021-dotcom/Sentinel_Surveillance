"""The tracking diagnostic's reading (session 79; ``face_check.py tracking <session>/tv --f1 ...``).

A diagnostic session runs F2's script (the cues, the format-9 account, the confirmed still posture) on the normal
recognition path with ``--tracking-diagnostic``, which records per processed frame what the detector handed the
tracker, what the tracker published, its own association costs, its lost tracks, a separate shadow NMS below the
production confidence, and the tracks the runtime held current (tracking.diagnostic). This reading judges execution
and privacy only. Everything about detection, tracking and association is DESCRIPTIVE, per window and per gap, and it
is not an F2 reading: the startup marker that names the diagnostic makes ``face_check f2`` refuse such a run, and
this reading refuses a run without it. It validates nothing and changes nothing. Since session 80 the guard's
``--require-ready tracking_diagnostic`` makes the diagnostic's readiness a prerequisite for T0 (the detector loaded
with the diagnostic's hooks and the tracker's settings in the starting line); this reading judges the guard's own
record of it, so a run whose guard did not require it, or found it unmet, never passes.

Evidence rules it keeps:

- an absence ("no detection at the seat", "nothing below the confidence") is claimed only for frames whose record is
  present, whose relevant list was not shortened and whose shadow pass hit no limit; a gap touched by missing frames,
  shortened lists or shadow limits is marked INCOMPLETE and its counts cover the complete frames only;
- the association cost read for a failed re-attachment is the tracker's own matrix at the pre-assignment stage, with
  its limit and the competing tracks and detections on that frame; a cost within the limit beside a new track is
  reported as a conflict to read, never as an association that "should have happened";
- the seat is followed box by box from the frame before each gap, with the intersection over the smaller box and the
  centre tests, so a torso-sized box inside the last full box is at the seat and a sideways box is inconclusive;
- departures use the account's earliest and latest possible times: nothing inside the departure interval is judged,
  and persistence is counted only after the latest possible time;
- every gap is described on its own; repeatability is a count of gaps, never a rate; no gap in A means the target
  failure was not reproduced in this session; the summaries' fractions are exploratory, not decision thresholds;
- diagnostic health (session 81): a run in which no frame's shadow pass completed is a failed diagnostic, never an
  empty result. A completed pass that found nothing counts as completed; a pass that raised (``shadow_failed``) or had
  no clone to read (``raw_missing``, ``clone_failed``) does not. The device run trk-20261016T100149Z recorded
  ``shadow_failed`` on all 242 frames; such frames stay unreadable for absences whatever the run's verdict.

The short integration check (session 81; ``face_check.py tracking-integration``) reads a run of the same workload (the
scene model and face recognition loaded, production settings, the diagnostic on, the guard's memory limits unchanged)
for at most 120 s after readiness, with no movement cue and no account. It judges execution, readiness, the
diagnostic's health and privacy, and describes the shadow passes, the records' coverage, timing and memory. A pass
establishes only that the parts ran together for that short run: not ten-minute stability, not tracking behaviour, not
recognition, and nothing about F2.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import face_account
import operator_check as oc
import step5_check as s5
from face_check import (
    CUE_TOLERANCE_S,
    F2_FACE,
    FULL_SCALE_SCHEMA,
    MAX_IDENTITY_RECORDS,
    MIN_WINDOW_S,
    MISSING,
    RELEASES,
    SCENE_SCOPE,
    WASHOUT_S,
    Reading,
    Timeline,
    _account,
    _bounds,
    _counts_items,
    _cue_offsets,
    _declared_cued,
    _key_values,
    _startup_keys,
    _windows,
    f1_gate,
    read_records,
)

NS = 1_000_000_000
# The tracker settings the pinned code runs with (legacy_ultralytics.TRACK_ARGS and the installed bytetrack.yaml, read
# from the loaded objects at startup); the reading checks the run recorded exactly these.
PINNED_TRACKER = {"conf": 0.4, "match_thresh": 0.8, "track_high_thresh": 0.25, "track_low_thresh": 0.1,
                  "new_track_thresh": 0.25, "track_buffer": 30, "max_time_lost": 30, "fuse_score": True,
                  "unconfirmed_thresh": 0.7, "shadow_conf": 0.1}
NOT_IN_THIS_RUN = ("live_preview", "snapshots", "chat", "feature_controls")  # the scene server runs, as in f-V
# A frame's shadow pass did not complete when one of these is on its record (session 81): it raised, or there was no
# clone of the raw output to read. Anything else (an empty result included) is a completed pass.
SHADOW_NOT_COMPLETED = ("raw_missing", "clone_failed", "shadow_failed")
INTEGRATION_MAX_S = 120.0  # the short integration check: at most this long after readiness (session 81)
INTEGRATION_CUES = ("t0", "stop")  # its only cues: the start and the stop; no movement is asked
SCOPE_INTEGRATION = ("scope: a short integration check of the tracking diagnostic on the normal recognition path, the "
                     "scene model and face recognition loaded as in the diagnostic session, production settings and the "
                     "guard's memory limits unchanged; at most 120 s after readiness, no movement asked, no account; a "
                     "pass establishes only that these parts ran together for this short run, not ten-minute "
                     "stability, tracking behaviour, recognition or anything about F2")
MIN_GAP_FRAMES = 2
AT_SEAT = 0.5  # exploratory: intersection over the smaller box at or above this, with a centre inside, is "at the seat"
ELSEWHERE = 0.1  # exploratory: below this, with neither centre inside, is "elsewhere"; between: inconclusive
SEATED = ("K1", "A", "K2")
EMPTY = ("E1", "E2")
DET_BANDS = ((0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))
SHADOW_BANDS = ((0.1, 0.2), (0.2, 0.3), (0.3, 0.4))
SCOPE = ("scope: a tracking diagnostic on the normal recognition path, not an F2 attempt; F2's criteria, the identity "
         "rules, the thresholds and the tracker's settings are unchanged and nothing here validates recognition or "
         "retention; one session, one light, one posture: event-level evidence, with repeatability a count of gaps")
CONDITIONS = ("framing_since_capture", "lighting")


# ---------------------------------------------------------------- geometry (fractions of the frame)


def _area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: list[float], b: list[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _iou(a: list[float], b: list[float]) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def _of_smaller(a: list[float], b: list[float]) -> float:
    s = min(_area(a), _area(b))
    return _inter(a, b) / s if s > 0 else 0.0


def _centre_in(a: list[float], b: list[float]) -> bool:
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def relation(candidate: list[float], reference: list[float]) -> dict[str, Any]:
    """How a box relates to the seat's reference box, with its label: at_seat, elsewhere or inconclusive."""
    of_smaller = _of_smaller(candidate, reference)
    centres = _centre_in(candidate, reference) or _centre_in(reference, candidate)
    label = ("at_seat" if of_smaller >= AT_SEAT and centres else
             "elsewhere" if of_smaller < ELSEWHERE and not centres else "inconclusive")
    return {"iou": round(_iou(candidate, reference), 3), "of_smaller": round(of_smaller, 3), "centre": centres,
            "label": label}


def _best(entries: list[dict], reference: list[float]) -> tuple[dict | None, dict | None]:
    """The entry whose box relates best to the reference (by intersection over the smaller box), and its relation."""
    best, rel = None, None
    for entry in entries:
        r = relation(entry["b"], reference)
        if rel is None or r["of_smaller"] > rel["of_smaller"]:
            best, rel = entry, r
    return best, rel


# ---------------------------------------------------------------- the records


def tracking_records(path: Path) -> tuple[list[dict], dict | None, list[str]]:
    """The frame lines in order of ``n``, the summary line, and the problems found in their shape."""
    frames: list[dict] = []
    summary = None
    problems: list[str] = []
    try:
        with path.open() as handle:
            for number, raw in enumerate(handle, 1):
                try:
                    item = json.loads(raw)
                except ValueError:
                    problems.append(f"line {number}: not JSON")
                    continue
                if not isinstance(item, dict) or "tracking" not in item:
                    continue
                if item.get("tracking") == "summary":
                    if summary is not None:
                        problems.append(f"line {number}: a second summary")
                    summary = item
                elif item.get("tracking") == "frame":
                    if not all(isinstance(item.get(k), int) for k in ("n", "frame_seq", "frame_mono_ns")) or not all(
                            isinstance(item.get(k), list) for k in ("det", "pub", "unpub", "lost", "removed",
                                                                    "assign", "shadow", "cur", "problems")) \
                            or not isinstance(item.get("trunc"), dict) or not isinstance(item.get("ns"), dict):
                        problems.append(f"line {number}: a malformed frame record")
                        continue
                    frames.append(item)
                else:
                    problems.append(f"line {number}: unknown tracking record {item.get('tracking')!r}")
    except OSError:
        problems.append("run.jsonl unreadable")
    frames.sort(key=lambda f: f["n"])
    for a, b in zip(frames, frames[1:]):
        if b["n"] <= a["n"]:
            problems.append(f"frame records out of order at n={b['n']}")
            break
    return frames, summary, problems[:5]


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]

    return {"n": len(values), "p50": round(at(0.5), 1), "p95": round(at(0.95), 1), "max": round(ordered[-1], 1)}


def _bands(scores: list[float], bands: tuple[tuple[float, float], ...]) -> dict[str, int]:
    return {f"{lo:g}-{hi if hi < 1.01 else 1:g}": sum(1 for s in scores if lo <= s < hi) for lo, hi in bands}


def _complete(frame: dict, *lists: str) -> bool:
    """Whether an absence may be read from this frame for the named lists: nothing shortened, nothing missing."""
    return not any(frame["trunc"].get(name) for name in lists) and not set(frame["problems"]) & {
        "capture_missing", "capture_failed", "tracker_replaced", "snapshot_failed", "no_backend_diagnostics"}


def _shadow_readable(frame: dict) -> bool:
    return (_complete(frame, "shadow") and not set(frame["problems"]) & {"raw_missing", "clone_failed",
                                                                        "shadow_failed", "layout_unknown"}
            and frame.get("sh_lim") is False)


def shadow_health(frames: list[dict]) -> dict[str, Any]:
    """Per frame, whether its shadow pass completed (session 81): no SHADOW_NOT_COMPLETED problem on its record and its
    limit flag recorded. A completed pass counts whether or not it found a box below the confidence."""
    completed = [f for f in frames if not set(f["problems"]) & set(SHADOW_NOT_COMPLETED)
                 and isinstance(f.get("sh_lim"), bool)]
    by_problem = Counter(p for f in frames for p in f["problems"] if p in SHADOW_NOT_COMPLETED)
    unflagged = sum(1 for f in frames if not set(f["problems"]) & set(SHADOW_NOT_COMPLETED)
                    and not isinstance(f.get("sh_lim"), bool))
    return {"frames": len(frames), "completed": len(completed),
            "completed_with_shadow_boxes": sum(1 for f in completed if f["shadow"]),
            "completed_without_shadow_boxes": sum(1 for f in completed if not f["shadow"]),
            "completed_limited": sum(1 for f in completed if f["sh_lim"] is True),
            "not_completed": len(frames) - len(completed), "not_completed_by_problem": dict(sorted(by_problem.items())),
            "not_completed_without_a_problem_named": unflagged}


def _health_item(r: Reading, frames: list[dict]) -> dict[str, Any]:
    health = shadow_health(frames)
    r.item("diagnostic health: at least one frame's shadow pass completed (a pass that found nothing below the "
           "confidence counts; one that raised or had no clone to read does not); none completed is a failed "
           "diagnostic, never an empty result (the frames without one stay unreadable for absences)",
           health if frames else MISSING, health["completed"] > 0)
    return health


def _window_frames(frames: list[dict], window: tuple[float, float], t0_ns: int) -> list[dict]:
    lo, hi = window
    return [f for f in frames if lo <= (f["frame_mono_ns"] - t0_ns) / NS < hi]


def _coverage(frames: list[dict]) -> dict[str, Any]:
    missing = sum(b["n"] - a["n"] - 1 for a, b in zip(frames, frames[1:]) if b["n"] > a["n"] + 1)
    truncated = Counter()
    problems = Counter()
    for f in frames:
        for name, count in f["trunc"].items():
            truncated[name] += count
        for p in f["problems"]:
            problems[p] += 1
    return {"frames": len(frames), "missing_frames_between": missing, "shortened": dict(sorted(truncated.items())),
            "problems": dict(sorted(problems.items())),
            "shadow_limited_frames": sum(1 for f in frames if f.get("sh_lim") is True),
            "complete": missing == 0 and not truncated and not problems}


def _window_summary(frames: list[dict]) -> dict[str, Any]:
    unpub = Counter(u["o"] for f in frames for u in f["unpub"])
    pub = Counter(p["o"] for f in frames for p in f["pub"])
    det_scores = [d["s"] for f in frames for d in f["det"]]
    shadow_scores = [s["s"] for f in frames for s in f["shadow"]]
    span = (frames[-1]["frame_mono_ns"] - frames[0]["frame_mono_ns"]) / NS if len(frames) > 1 else 0.0
    return {"frames": len(frames), "with_det": sum(1 for f in frames if f["det"]),
            "with_pub": sum(1 for f in frames if f["pub"]),
            "shadow_only": sum(1 for f in frames if not f["det"] and f["shadow"] and _shadow_readable(f)),
            "nothing_readable": sum(1 for f in frames if not f["det"] and not f["shadow"] and _complete(f, "det")
                                    and _shadow_readable(f)),
            "pub_by_outcome": dict(sorted(pub.items())), "unpub_by_outcome": dict(sorted(unpub.items())),
            "det_score_bands": _bands(det_scores, DET_BANDS), "shadow_score_bands": _bands(shadow_scores, SHADOW_BANDS),
            "second_stage_dets": sum(f.get("ss", 0) for f in frames),
            "shadow_disagreement_frames": sum(1 for f in frames if f["sh_hi"] != len(f["det"])
                                              or f["sh_agree"] != len(f["det"])),
            "processed_fps": round((len(frames) - 1) / span, 2) if span > 0 else None}


# ---------------------------------------------------------------- the seat, followed box by box, and its gaps


def _seat_start(frames: list[dict]) -> tuple[int, dict] | None:
    for index, f in enumerate(frames):
        if f["pub"]:
            return index, max(f["pub"], key=lambda p: _area(p["b"]))
    return None


def follow_seat(frames: list[dict]) -> dict[str, Any]:
    """Follow the seated person's published box forward through a window's frames: a frame whose best published box
    is at the seat continues it (under any number; number changes are counted); otherwise the frame is a gap frame.
    Returns the gaps of MIN_GAP_FRAMES or more, each described, and the single-frame gaps counted."""
    start = _seat_start(frames)
    if start is None:
        return {"seat_found": False, "gaps": [], "single_frame_gaps": 0, "id_changes": 0}
    index, first = start
    seat_box, seat_id = first["b"], first["id"]
    gaps: list[dict] = []
    run: list[int] = []
    singles = id_changes = 0
    before: dict[str, Any] = {"index": index, "box": seat_box, "id": seat_id}
    for i in range(index + 1, len(frames)):
        f = frames[i]
        best, rel = _best(f["pub"], seat_box)
        if best is not None and rel["label"] == "at_seat":
            if run:
                if len(run) >= MIN_GAP_FRAMES:
                    gaps.append(_gap(frames, run, before, i, best, rel))
                else:
                    singles += 1
                run = []
            if best["id"] != seat_id:
                id_changes += 1
            seat_box, seat_id = best["b"], best["id"]
            before = {"index": i, "box": seat_box, "id": seat_id}
        else:
            run.append(i)
    if len(run) >= MIN_GAP_FRAMES:
        gaps.append(_gap(frames, run, before, None, None, None))
    elif run:
        singles += 1
    return {"seat_found": True, "gaps": gaps, "single_frame_gaps": singles, "id_changes": id_changes}


def _gap(frames: list[dict], run: list[int], before: dict, end_index: int | None, end_pub: dict | None,
         end_rel: dict | None) -> dict[str, Any]:
    """One gap described from its own frames: what was at the seat (relative to the box before the gap), the lost
    track, the association costs at the frames with a detection at the seat, how it ended, and what the runtime did."""
    reference, seat_id = before["box"], before["id"]
    first, last = frames[run[0]], frames[run[-1]]
    gap_frames = [frames[i] for i in run]
    t_before = frames[before["index"]]["frame_mono_ns"]
    coverage = _coverage(gap_frames)
    counts = Counter()
    pub_elsewhere = Counter()
    unpub_at_seat = Counter()
    lost_seen: list[int] = []
    removed_at: int | None = None
    assignment: dict[str, Any] | None = None
    assignment_frames = 0
    for f in gap_frames:
        det_readable = _complete(f, "det")
        best_det, det_rel = _best(f["det"], reference)
        if best_det is not None:
            counts[f"det_{det_rel['label']}"] += 1
        elif det_readable:
            counts["det_none"] += 1
        else:
            counts["det_unreadable"] += 1
        if f["pub"]:
            _, rel = _best(f["pub"], reference)
            pub_elsewhere[rel["label"]] += 1
        best_shadow, shadow_rel = _best(f["shadow"], reference)
        if best_shadow is not None and _shadow_readable(f):
            counts[f"shadow_{shadow_rel['label']}"] += 1
        elif _shadow_readable(f):
            counts["shadow_none"] += 1
        else:
            counts["shadow_unreadable"] += 1
        for u in f["unpub"]:
            det = next((d for d in f["det"] if d["i"] == u["i"]), None)
            if det is not None and relation(det["b"], reference)["label"] == "at_seat":
                unpub_at_seat[u["o"]] += 1
        for item in f["lost"]:
            if item["id"] == seat_id:
                lost_seen.append(item["lost"])
        if seat_id in f["removed"] and removed_at is None:
            removed_at = f["n"]
        if best_det is not None and det_rel["label"] == "at_seat":
            for a in f["assign"]:
                if a["stage"] != "pool" or a["cut"]:
                    continue
                row = next((r for r in a["rows"] if r["id"] == seat_id), None)
                if row is None or best_det["i"] not in a["dets"]:
                    continue
                column = a["dets"].index(best_det["i"])
                if column >= len(row["c"]):
                    continue
                assignment_frames += 1
                if assignment is None:
                    assignment = {"n": f["n"], "cost_seat_track_to_seat_det": row["c"][column], "thresh": a["thresh"],
                                  "within_limit": row["c"][column] <= a["thresh"],
                                  "seat_track": next((t for t in a["tracks"] if t["id"] == seat_id), None),
                                  "competing_tracks": [{"id": r["id"], "cost": r["c"][column]} for r in a["rows"]
                                                       if r["id"] != seat_id and column < len(r["c"])],
                                  "seat_track_to_other_dets": [{"det": d, "cost": row["c"][k]}
                                                               for k, d in enumerate(a["dets"])
                                                               if k != column and k < len(row["c"])],
                                  "det_outcome": next((u["o"] for u in f["unpub"] if u["i"] == best_det["i"]),
                                                      next((p["o"] for p in f["pub"] if p["i"] == best_det["i"]),
                                                           None))}
    end: dict[str, Any] = {"open_at_window_end": end_index is None}
    if end_index is not None and end_pub is not None and end_rel is not None:
        nxt = frames[end_index]
        end.update({"n": nxt["n"], "outcome": end_pub["o"], "id": end_pub["id"], "same_id": end_pub["id"] == seat_id,
                    "relation_to_box_before": end_rel, "seat_id_removed_before": removed_at is not None,
                    "span_s": round((nxt["frame_mono_ns"] - t_before) / NS, 2)})
    cur_last = max((f["n"] for f in [frames[before["index"]], *gap_frames] if seat_id in f["cur"]), default=None)
    incomplete = not coverage["complete"]
    consistent = []
    if not incomplete or counts["det_at_seat"] or counts["shadow_at_seat"]:
        if counts["det_at_seat"] == 0 and counts["shadow_at_seat"] > 0:
            consistent.append("filtered_by_the_confidence_cut")
        if counts["det_none"] and counts["shadow_none"] and not counts["det_at_seat"] and not counts["shadow_at_seat"]:
            consistent.append("no_box_at_the_seat_at_or_above_the_shadow_confidence")
        if assignment is not None and not assignment["within_limit"]:
            consistent.append("association_cost_above_the_limit_while_the_seat_track_was_lost")
        if assignment is not None and assignment["within_limit"]:
            consistent.append("cost_within_the_limit_beside_a_new_or_unpublished_track_(a_conflict_to_read)")
        if end.get("seat_id_removed_before") and not end.get("same_id", True):
            consistent.append("first_published_box_after_the_tracker_removed_the_seat_track")
        if end.get("same_id") and cur_last is not None and end.get("n") is not None and cur_last < end["n"]:
            consistent.append("same_number_refound_after_the_runtime_expired_it")
    return {"n_first": first["n"], "n_last": last["n"], "frames": len(run),
            "span_s": round((last["frame_mono_ns"] - t_before) / NS, 2),
            "box_before": reference, "seat_id_before": seat_id, "coverage": coverage,
            "evidence": "INCOMPLETE" if incomplete else "complete",
            "at_seat_counts": dict(sorted(counts.items())), "pub_elsewhere": dict(sorted(pub_elsewhere.items())),
            "unpub_at_seat": dict(sorted(unpub_at_seat.items())),
            "seat_track_lost": {"frames_seen_lost": len(lost_seen), "max_updates_lost": max(lost_seen, default=None),
                                "removed_at_n": removed_at},
            "association_at_seat_dets": {"frames": assignment_frames, "first": assignment},
            "runtime_current_until_n": cur_last, "end": end,
            "consistent_with": consistent or (["nothing_read_(incomplete)"] if incomplete else ["unclassified"])}


# ---------------------------------------------------------------- the reading


def _execution(r: Reading, directory: Path, f1: Path, gate: tuple[Path, Path, Path | None] | None) -> dict[str, Any]:
    """The items a diagnostic session and the short integration check judge alike, in this order: the F1 basis, the
    supervision, readiness, the run's configuration, the clean shutdown, the identity records, the tracking records,
    the diagnostic's health and T0. Returns what the rest of each reading uses."""
    if gate is not None:
        ok, lines = f1_gate(*gate)
        r.item("F1 basis for this run: the run's own F1 validated, or the maintainer's accepted corrected F1 reading "
               "of it", dict(line.split("=", 1) for line in lines), ok and f1.resolve() == (gate[0] / "f1").resolve())
    result = s5._load(directory / "result.json")
    s5._supervision(r, result, "duration_stop", 0)
    readiness = s5._get(result, "readiness")  # session 80: the guard's own record, never inferred from the exit
    r.item("readiness required before T0: the guard required the diagnostic's readiness on the runtime's starting "
           "line (no detector problem; the marker with the tracker's settings complete) and found it met, so the cues "
           "ran against a loaded detector with the diagnostic's hooks",
           readiness,
           isinstance(readiness, dict) and readiness.get("required") == "tracking_diagnostic"
           and readiness.get("met") is True and readiness.get("reason") is None)
    start = s5._get(result, "run_output", "starting")
    face = s5._get(start, "face")
    marker = face.get("tracking_diagnostic") if isinstance(face, dict) else None
    tracker = marker.get("tracker") if isinstance(marker, dict) else None
    settings_ok = isinstance(tracker, dict) and all(
        isinstance(tracker.get(k), (int, float, bool)) and abs(float(tracker.get(k)) - float(v)) < 1e-9
        for k, v in PINNED_TRACKER.items())
    plain = isinstance(face, dict) and {k: v for k, v in face.items() if k != "tracking_diagnostic"} == F2_FACE
    present = sorted(_startup_keys(directory / "run.jsonl") & set(NOT_IN_THIS_RUN))
    r.item("a tracking diagnostic run on the normal recognition path: face admitted for this validation run, one "
           "enrolled identity, the diagnostic's marker present with the tracker's settings as pinned (conf 0.4, "
           "ByteTrack 0.25/0.1/0.25, buffer 30, match 0.8, score fusion on, shadow 0.1), production detection, no "
           "candidate, comparison, experiment or continuity rule; detector and face admitted; no live view, snapshots, "
           "chat or feature controls",
           {"face": face, "face_problem": s5._get(start, "face_problem"),
            "detector_problem": s5._get(start, "detector_problem"), "present": present},
           plain and isinstance(marker, dict) and settings_ok and s5._get(start, "face_problem") is None
           and s5._get(start, "detector_problem") is None and not present)
    stopped = s5._get(result, "run_output", "stopped")
    r.item("clean shutdown: all stopped (the face worker included), database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True
           and (stopped.get("stopped") or {}).get("face_worker") is True)
    records, problems, starting_line, _ = read_records(directory / "run.jsonl")
    total = sum(len(v) for v in records.values())
    schemas = sorted({item["schema"] for kind in records.values() for item in kind})
    names = [c.get("name") for c in s5._list(s5._get(result, "captures")) if isinstance(c, dict)]
    captures = [s5._load(directory / f"status-{name}.json") for name in names]
    last = captures[-1] if captures else MISSING
    dropped = s5._get(last, "components", "face", "records", "dropped")
    r.item(f"identity records: schema {FULL_SCALE_SCHEMA}, complete, none dropped",
           {"records": {k: len(v) for k, v in records.items()}, "schemas": schemas, "problems": problems[:5],
            "dropped": dropped} if total else MISSING,
           total > 0 and not problems and schemas == [FULL_SCALE_SCHEMA] and dropped == 0
           and total < MAX_IDENTITY_RECORDS)
    frames, summary, shape = tracking_records(directory / "run.jsonl")
    status_tracking = s5._get(last, "components", "detector", "tracking")
    kept = summary.get("kept") if isinstance(summary, dict) else None
    r.item("tracking records: one well-formed frame record per processed frame, the summary present, none dropped "
           "(shortened lines and per-frame problems are reported, not failed)",
           {"frames": len(frames), "summary": summary, "shape_problems": shape, "status": status_tracking}
           if frames or summary else MISSING,
           bool(frames) and isinstance(summary, dict) and not shape and summary.get("dropped") == 0
           and kept == len(frames) and summary.get("processed") == kept)
    health = _health_item(r, frames)
    cue = next((c for c in s5._list(s5._get(result, "cues")) if isinstance(c, dict) and c.get("name") == "t0"),
               MISSING)
    ready = s5._get(result, "child", "ready_mono_s")
    printed = s5._get(cue, "printed_mono_s")
    r.item("T0: the t0 cue printed when the runtime was ready", {"printed_mono_s": printed, "ready_mono_s": ready},
           isinstance(printed, float) and isinstance(ready, float) and 0 <= printed - ready <= CUE_TOLERANCE_S)
    return {"result": result, "start": start, "marker": marker, "records": records, "frames": frames,
            "summary": summary, "captures": captures, "last": last, "printed": printed, "health": health,
            "starting_line": starting_line}


def read_tracking(directory: Path, f1: Path, gate: tuple[Path, Path, Path | None] | None = None) -> Reading:
    r = Reading()
    run = _execution(r, directory, f1, gate)
    result, marker, records, frames = run["result"], run["marker"], run["records"], run["frames"]
    captures, printed = run["captures"], run["printed"]
    account, problem, fmt = _account(directory)
    r.item("account saved and locked before any result was shown", problem or "locked", account is not None)
    if account is None:
        _counts_items(r, directory)
        return r
    review = (face_account.review(account, face_account.FIELDS_V9) if fmt == 9
              else [f"format {fmt} is not this run's account (format 9)"])
    r.item("account in format 9, complete and in order", review, fmt == 9 and review[-1] == "Complete and in order: yes")
    cues = _cue_offsets(result)
    _declared_cued(r, account, cues)
    duration = s5._get(result, "parameters", "duration_s")
    bounds = _bounds(account, 9, cues) if fmt == 9 else None
    windows = _windows(bounds, duration) if bounds is not None and isinstance(duration, (int, float)) else None
    r.item(f"judged windows from the account's intervals (washout {WASHOUT_S:g} s after the latest and before the "
           f"earliest possible time of each movement), each at least {MIN_WINDOW_S:g} s",
           {k: [round(a, 1), round(b, 1)] for k, (a, b) in windows.items()} if windows else MISSING,
           windows is not None and all(b - a >= MIN_WINDOW_S for a, b in windows.values()))
    _counts_items(r, directory)
    r.describe(SCOPE, {"account": {k: v for k, v in account.items() if k != "notes"}})
    r.describe("the diagnostic's bounds and the tracker's settings as the run recorded them at startup", marker)
    r.describe("the run's conditions as its launch recorded them (the framing since the capture and the room light)",
               {k: _key_values(directory / "provenance.txt").get(k, MISSING) for k in CONDITIONS})
    if not frames or windows is None or bounds is None or not isinstance(printed, float):
        r.describe("nothing further can be described: frame records, windows or T0 are missing", MISSING)
        return r
    t0_ns = round(printed * NS)
    line = Timeline(records, t0_ns, float(duration))
    r.describe("overhead (UNMEASURED until a device run; these are that run's own numbers): the production backend's "
               "time per frame and the diagnostic's parts (ns, p50/p95/max), the processed rate over the run and the "
               "status captures' rates", {
                   "backend_ns": _percentiles([f["ns"]["backend"] for f in frames if "backend" in f["ns"]]),
                   "diagnostic_ns": _percentiles([sum(v for k, v in f["ns"].items() if k != "backend")
                                                  for f in frames]),
                   **{f"{k}_ns": _percentiles([f["ns"][k] for f in frames if k in f["ns"]])
                      for k in ("capture", "clone", "shadow", "snapshot", "record")},
                   "processed_fps_over_the_records": _window_summary(frames)["processed_fps"],
                   "captures_processed_fps": [s5._get(c, "rates", "processed_fps") for c in captures]})
    r.describe("record coverage over the run (a frame absent from the records, a shortened list or a per-frame "
               "problem makes an absence unreadable there)", _coverage(frames))
    by_window = {name: _window_frames(frames, windows[name], t0_ns) for name in windows}
    r.describe("per window: coverage (INCOMPLETE where frames are missing, lists shortened, problems recorded or the "
               "shadow pass limited)", {k: _coverage(v) for k, v in by_window.items()})
    r.describe("per window: frames with a detection handed to the tracker, with a published track, with a shadow box "
               "only (readable frames), with nothing readable at or above the shadow confidence; published and "
               "unpublished outcomes; score bands; second-stage detections; frames where the shadow pass disagreed "
               "with production (exploratory summaries)", {k: _window_summary(v) if v else None
                                                           for k, v in by_window.items()})
    seated: dict[str, Any] = {}
    for name in SEATED:
        followed = follow_seat(by_window[name])
        seated[name] = followed
    r.describe("per seated window: the seat followed box by box; gaps of at least two frames, each described on its "
               "own (what was at the seat relative to the box before the gap: detections, shadow boxes, unpublished "
               "detections; the seat track lost and removed; the association cost the tracker computed at the first "
               "frame with a detection at the seat, with its limit and competitors; how the gap ended; until when the "
               "runtime held the track current); single-frame gaps and number changes counted",
               {k: {"seat_found": v["seat_found"], "gaps": len(v["gaps"]), "single_frame_gaps": v["single_frame_gaps"],
                    "id_changes": v["id_changes"]} for k, v in seated.items()})
    for name in SEATED:
        for index, gap in enumerate(seated[name]["gaps"], 1):
            r.describe(f"{name} gap {index}", gap)
    a_gaps = seated["A"]["gaps"]
    if not by_window["A"]:
        r.describe("A: no frame records in the window: the target failure is unobserved (INCOMPLETE)", MISSING)
    elif not seated["A"]["seat_found"]:
        r.describe("A: no published track found in the window to follow: unobserved", MISSING)
    elif not a_gaps:
        r.describe("A: no gap of two or more frames in the seat's published track: the target failure was not "
                   "reproduced in this session (not evidence that it cannot occur)",
                   {"single_frame_gaps": seated["A"]["single_frame_gaps"], "id_changes": seated["A"]["id_changes"]})
    else:
        r.describe("A: gaps observed; each gap's own evidence is above; repeatability as counts, never a rate",
                   {"gaps": len(a_gaps), "complete": sum(1 for g in a_gaps if g["evidence"] == "complete"),
                    "consistent_with": dict(Counter(c for g in a_gaps for c in g["consistent_with"]))})
    cleared = [(t["track_id"], round(t["t"], 2)) for t in line.transitions if t["to"] == "cleared"]
    r.describe("identity transitions to cleared (track, s after T0): where a track with an identity state stopped "
               "being current", cleared)
    departure = bounds["out_of_view_2"]
    r.describe("control: the empty windows (frames with any detection, any published track, any readable shadow "
               "box, unpublished detections by outcome; tracks published for the first time; lost tracks refound)",
               {k: {**_window_summary(v), "confirmed": sum(1 for f in v for p in f["pub"] if p["o"] == "confirmed"),
                    "refound": sum(1 for f in v for p in f["pub"] if p["o"] == "refound")} if v else None
                for k, v in by_window.items() if k in EMPTY})
    inside = _window_frames(frames, departure, t0_ns)
    after = _window_frames(frames, (departure[1], windows["E2"][0]), t0_ns)
    r.describe("the departure at 6:00 by the account's interval (s after T0: earliest, latest possible): frames "
               "inside it are not judged; after the latest possible time and before E2's washout ends, published "
               "tracks and detections are counted as persistence to read, not as a person; the tracker's removals "
               "and the runtime's current tracks over that stretch",
               {"interval_s": [round(departure[0], 2), round(departure[1], 2)],
                "inside_frames_with_det": sum(1 for f in inside if f["det"]),
                "after_latest": {"frames": len(after), "with_pub": sum(1 for f in after if f["pub"]),
                                 "with_det": sum(1 for f in after if f["det"]),
                                 "removed": sorted({n for f in after for n in f["removed"]}),
                                 "current_ids": sorted({n for f in after for n in f["cur"]})}})
    r.describe("guard peak pressure, min MemFree, min MemAvailable (B)",
               [s5._get(result, "guard", k) for k in s5.GUARD_FIGURES])
    return r


# ---------------------------------------------------------------- the short integration check (session 81)


def _margin(value: Any, limit: Any, *, below: bool) -> Any:
    """How far a figure stayed from its guard limit (bytes): limit - value for a ceiling, value - limit for a floor."""
    if not isinstance(value, int) or not isinstance(limit, int):
        return MISSING
    return limit - value if below else value - limit


def _workload_items(r: Reading, start: Any, result: Any) -> None:
    """The production settings and the diagnostic session's workload at startup, judged as F2's reading judges them
    (the memory policy, THP, the post-load releases, the scene server), and the guard's limits as pinned."""
    r.item("memory policy selected: THP workload_disabled, release post_load", s5._get(start, "memory_policy"),
           s5._get(start, "memory_policy") == s5.CANDIDATE_POLICY)
    r.item("THP scope at each startup checkpoint", s5._get(start, "thp_scope"), s5._get(start, "thp_scope") == SCENE_SCOPE)
    disable = s5._get(start, "thp_disable")
    r.item("THP disable verified in the runtime process", disable,
           isinstance(disable, dict) and disable.get("verified") is True and disable.get("reason") is None)
    r.item("post-load releases, the face weights' included: every call returned 0", s5._get(start, "releases"),
           s5._get(start, "releases") == RELEASES)
    server = s5._get(start, "scene_server")
    r.item("the scene model loaded as in the diagnostic session: scene server ready, 17/17 layers, vision encoder on "
           "GPU; scene admitted", {"scene_server": server, "scene_problem": s5._get(start, "scene_problem")},
           isinstance(server, dict) and server.get("state") == "ready" and server.get("problem") is None
           and server.get("layers") == s5.LAYERS and server.get("vision_on_gpu") is True
           and s5._get(start, "scene_problem") is None)
    limits, headroom = s5._get(result, "limits"), s5._get(result, "parameters", "step4_headroom")
    r.item("the guard's memory limits as pinned (pressure 4.8 GB, MemFree 1 GiB, MemAvailable 2 GiB, swap counters "
           "unchanged, sampling gaps of 0.5 s at most) and step 4's headroom required before launch",
           {"limits": limits, "step4_headroom": headroom},
           limits == {"pressure_stop_bytes": oc.PRESSURE_STOP, "mem_free_floor_bytes": oc.FREE_FLOOR,
                      "mem_available_floor_bytes": oc.AVAILABLE_FLOOR, "swap_counters": "unchanged",
                      "max_sample_gap_s": 0.5} and headroom is True)


def read_integration(directory: Path, f1: Path, gate: tuple[Path, Path, Path | None] | None = None) -> Reading:
    """The short integration check: the diagnostic session's execution items, its duration and cues (at most
    INTEGRATION_MAX_S after readiness, only INTEGRATION_CUES, each printed), the production settings and workload at
    startup and the guard's limits as pinned, the privacy counts; then the shadow passes, the records' coverage, timing
    and memory described. No account, window or gap: nothing about tracking behaviour."""
    r = Reading()
    run = _execution(r, directory, f1, gate)
    result, frames, summary, captures = run["result"], run["frames"], run["summary"], run["captures"]
    parameters = s5._get(result, "parameters")
    duration, from_launch = s5._get(parameters, "duration_s"), s5._get(parameters, "from_launch")
    declared = [c.get("name") for c in s5._list(s5._get(parameters, "cues")) if isinstance(c, dict)]
    printed = {c.get("name"): c.get("printed_mono_s") for c in s5._list(s5._get(result, "cues")) if isinstance(c, dict)}
    r.item(f"a short run: at most {INTEGRATION_MAX_S:g} s counted from readiness, with only the start and stop cues "
           "(no movement asked), each printed",
           {"duration_s": duration, "from_launch": from_launch, "cues": declared, "printed": printed}
           if isinstance(parameters, dict) else MISSING,
           isinstance(duration, (int, float)) and 0 < duration <= INTEGRATION_MAX_S and from_launch is False
           and declared == list(INTEGRATION_CUES) and sorted(printed) == sorted(INTEGRATION_CUES)
           and all(isinstance(v, float) for v in printed.values()))
    _workload_items(r, run["start"], result)
    _counts_items(r, directory)
    r.describe(SCOPE_INTEGRATION, {"duration_s": duration})
    r.describe("the diagnostic's bounds and the tracker's settings as the run recorded them at startup", run["marker"])
    r.describe("the run's conditions as its launch recorded them (the framing since the capture and the room light)",
               {k: _key_values(directory / "provenance.txt").get(k, MISSING) for k in CONDITIONS})
    r.describe("shadow passes per frame (completed, with and without a box below the confidence, limited; not "
               "completed, by problem) and the runtime's own problem counts in the summary line",
               {"frames": run["health"],
                "summary": {k: summary.get(k) for k in ("processed", "kept", "dropped", "problem_counts")}
                if isinstance(summary, dict) else MISSING})
    r.describe("record coverage over the run (missing frames between records, shortened lists, per-frame problems, "
               "shadow passes at a limit)", _coverage(frames) if frames else MISSING)
    started = s5._get(result, "child", "ready_after_s")
    r.describe("timing (this run's own numbers, at most two minutes of them): startup to readiness (s), the production "
               "backend's time per frame and the diagnostic's parts (ns, p50/p95/max), the processed rate over the "
               "records and the status captures' rates", {
                   "ready_after_launch_s": started,
                   "backend_ns": _percentiles([f["ns"]["backend"] for f in frames if "backend" in f["ns"]]),
                   "diagnostic_ns": _percentiles([sum(v for k, v in f["ns"].items() if k != "backend")
                                                  for f in frames]),
                   **{f"{k}_ns": _percentiles([f["ns"][k] for f in frames if k in f["ns"]])
                      for k in ("capture", "clone", "shadow", "snapshot", "record")},
                   "processed_fps_over_the_records": _window_summary(frames)["processed_fps"] if frames else None,
                   "captures": [s5._get(c, "rates") for c in captures]})
    limits, guard, baseline = s5._get(result, "limits"), s5._get(result, "guard"), s5._get(result, "baseline")
    loads = run["starting_line"] if isinstance(run["starting_line"], dict) else MISSING  # the runtime's own line
    r.describe("memory (bytes; the guard's limits unchanged): its baseline before launch, the peak pressure and the "
               "lowest MemFree and MemAvailable over the run with each one's margin to its limit, and the runtime's "
               "readings before each model load", {
                   "limits": limits,
                   "baseline": {k: s5._get(baseline, k) for k in ("MemFree", "MemAvailable", "pressure_bytes")},
                   **{k: s5._get(guard, k) for k in s5.GUARD_FIGURES},
                   "margin_to_pressure_stop": _margin(s5._get(guard, "peak_pressure_bytes"),
                                                      s5._get(limits, "pressure_stop_bytes"), below=True),
                   "margin_to_mem_free_floor": _margin(s5._get(guard, "min_mem_free_bytes"),
                                                       s5._get(limits, "mem_free_floor_bytes"), below=False),
                   "margin_to_mem_available_floor": _margin(s5._get(guard, "min_mem_available_bytes"),
                                                            s5._get(limits, "mem_available_floor_bytes"), below=False),
                   "before_loads": {k: s5._get(loads, k) for k in ("scene_memory_before", "detector_memory_before",
                                                                   "face_memory_before")}})
    last = run["last"]
    results = run["records"].get("result", [])
    r.describe("the workload at the last status capture (scene jobs submitted and delivered, the face worker's "
               "counters and processing time) and the face results by outcome in the records",
               {"scene_worker": s5._get(last, "components", "scene", "worker"),
                "face_worker": s5._get(last, "components", "face", "worker"),
                "face_results_by_outcome": dict(sorted(Counter(x.get("outcome") for x in results).items()))})
    return r
