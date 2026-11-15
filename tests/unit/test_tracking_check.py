"""The tracking diagnostic's records in a real-runtime run and its reading (session 79): synthetic runs through the
tests' simulate() with a scripted reading per frame, the records within their bounds, and the reading's per-gap
evidence rules (absences only from complete evidence, the tracker's own costs, the seat followed box by box, the
account's departure interval, no gap = not reproduced); F2's reading refuses such a run and this reading refuses an
F2 run; the privacy profile accepts every line and refuses an undeclared key. Session 81: zero completed shadow passes
fail the diagnostic's health (an empty pass is a completed one), and the short integration check's reading."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from face_runs import BOX, real_face_startup, startup_line
from sentinel.tracking.diagnostic import (
    Assignment,
    Detection,
    DiagnosticBounds,
    Lost,
    Published,
    ShadowBox,
    TrackerDiagnostics,
    Unpublished,
)
from sentinel.tracking.tracker import RawTrack

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib mode: the tests' own module is not importable by name
from test_face_check import (  # noqa: E402  (the shared scripted run, its finish, and the fixtures it defines)
    FORMAT3_CUES,
    account_v3,
    f1_list,  # noqa: F401
    finish,
    line_for,
    mods,  # noqa: F401
    simulate,
    startup,  # noqa: F401
)

FPS = 4
WIDE = tuple(float(v) for v in BOX)  # the seated person's box, in pixels
TORSO = (WIDE[0] + 5.0, WIDE[1] + (WIDE[3] - WIDE[1]) / 2, WIDE[2] - 5.0, WIDE[3])  # inside it: the lower half
SIDEWAYS = (WIDE[0] + 60.0, WIDE[1], WIDE[2] + 60.0, WIDE[3])  # beside it: partly overlapping
FAR = (500.0, 300.0, 560.0, 470.0)


def plain(tf: int, **fields: Any) -> TrackerDiagnostics:
    return TrackerDiagnostics(tf, shadow_candidates=fields.pop("cand", 1), shadow_limited=False, **fields)


def make_script(kind: str | None = None, *, start: float = 262.0, end: float = 267.0, after_id: int | None = None):
    """What the diagnostic backend reads on each frame of the F2 script, with one gap in A from ``start`` to ``end``
    of the given kind (the published track is withheld there), and how the track comes back after it."""
    state = {"in_gap": False}

    def script(t: float, phase: str, track: int | None, tracker: Any) -> TrackerDiagnostics:
        tf = int(round(t * FPS))
        if kind is not None and kind != "stray" and start <= t < end:
            state["in_gap"] = True
            tracker.next = []
            k = int(round((t - start) * FPS)) + 1  # updates since the seat track's last detection
            lost = (Lost(track, k, WIDE),) if k <= 30 else ()
            removed = (track,) if k == 31 else ()
            if kind == "filtered":
                return plain(tf, lost=lost, removed=removed, shadow=(ShadowBox(0.3, TORSO),))
            if kind == "missing":
                return plain(tf, lost=lost, removed=removed, cand=0)
            if kind in ("association", "conflict"):
                cost = 0.95 if kind == "association" else 0.5
                return plain(tf, detections=(Detection(0, 0.45, TORSO),),
                             unpublished=(Unpublished(0, "new_unconfirmed", 90 + k),), lost=lost, removed=removed,
                             assignments=(Assignment("pool", 0.8, ((track, "lost", k),), (0,), ((cost,),)),))
            if kind == "buffer":
                return plain(tf, lost=lost, removed=removed, cand=0)
            if kind == "sideways":
                return plain(tf, detections=(Detection(0, 0.8, SIDEWAYS),),
                             unpublished=(Unpublished(0, "unmatched", None),), lost=lost, removed=removed, cand=0)
            if kind == "unreadable":
                return TrackerDiagnostics(tf, lost=lost, removed=removed, problems=("raw_missing",))
            raise ValueError(kind)
        if phase == "absent":
            if kind == "stray" and 50.0 <= t < 51.0:  # a box in the empty window, starting an unpublished track
                return plain(tf, detections=(Detection(0, 0.5, FAR),), unpublished=(Unpublished(0, "new_unconfirmed", 50),))
            return plain(tf, cand=0)
        published_id, outcome = track, "continued"
        if state["in_gap"]:  # the first frame after the gap
            state["in_gap"] = False
            outcome = "confirmed" if after_id is not None else "refound"
        if after_id is not None and kind is not None and t >= end:
            published_id = after_id
            tracker.next = [RawTrack(after_id, *BOX, 0.9)]
        return plain(tf, detections=(Detection(0, 0.9, WIDE),), published=(Published(published_id, 0, outcome, WIDE),),
                     assignments=(Assignment("pool", 0.8, ((published_id, "tracked", 0),), (0,), ((0.1,),)),))

    return script


@pytest.fixture
def tracking_startup(tmp_path, accepted_scene) -> dict:
    root = tmp_path / "startup-tracking"
    root.mkdir()
    return startup_line(real_face_startup(root, accepted_scene, tracking=True))


def tracking_lines(out: Path) -> tuple[list[dict], dict | None]:
    frames, summary = [], None
    for raw in (out / "run.jsonl").read_text().splitlines():
        item = json.loads(raw)
        if item.get("tracking") == "frame":
            frames.append(item)
        elif item.get("tracking") == "summary":
            summary = item
    return frames, summary


READINESS = {"required": "tracking_diagnostic", "met": True, "reason": None, "starting_after_s": 70.0}  # session 80


def session(mods, tmp_path, tracking_startup, kind=None, **kwargs) -> Path:
    out = simulate(tmp_path, tracking_startup, tracking=make_script(kind, **kwargs))
    finish(mods, out, account=account_v3(mods), cues=FORMAT3_CUES, cued_format=9, readiness=READINESS)
    return out


def described(text: str, start: str) -> Any:
    """The JSON value of the plain DESCRIPTIVE line whose name starts with ``start`` (the name may itself hold ": ")."""
    line = next(line for line in text.splitlines() if line.startswith(f"DESCRIPTIVE {start}"))
    at = line.find(": ")
    while at != -1:
        try:
            return json.loads(line[at + 2:])
        except ValueError:
            at = line.find(": ", at + 1)
    raise AssertionError(line)


def gap(text: str, window: str = "A", index: int = 1) -> dict:
    return described(text, f"{window} gap {index}")


# ---------------------------------------------------------------- the records in a real-runtime run


def test_each_processed_frame_gets_one_bounded_record_and_the_summary_closes_the_run(tmp_path, tracking_startup) -> None:
    out = simulate(tmp_path, tracking_startup, tracking=make_script(), duration=10.0, captures=())
    frames, summary = tracking_lines(out)
    assert [f["n"] for f in frames] == list(range(1, 41)) and all(f["schema"] == 1 for f in frames)
    facing = [f for f in frames if f["pub"]]
    assert facing and all(f["cur"] == [f["pub"][0]["id"]] for f in facing)  # the runtime's current tracks
    assert all(set(f["ns"]) == {"backend", "record"} for f in frames)  # the scripted backend times no part of its own
    assert all(f["problems"] == [] and f["trunc"] == {} for f in frames)
    assert frames[0]["det"] == [{"i": 0, "s": 0.9, "b": [round(WIDE[0] / 640, 3), round(WIDE[1] / 480, 3),
                                                        round(WIDE[2] / 640, 3), round(WIDE[3] / 480, 3)]}]
    assert summary == {"tracking": "summary", "schema": 1, "kept": 40, "dropped": 0, "bytes": summary["bytes"],
                       "truncated_lines": 0, "max_line_bytes_seen": summary["max_line_bytes_seen"], "processed": 40,
                       "problem_counts": {}, "limits": {"max_records": 20000, "max_bytes": 32000000,
                                                        "max_line_bytes": 4096}}
    assert summary["bytes"] == sum(len(json.dumps(f, separators=(",", ":")).encode()) for f in frames)
    first = json.loads((out / "run.jsonl").read_text().splitlines()[0])
    assert first["startup"]["face"]["tracking_diagnostic"]["tracker"]["conf"] == 0.4


def test_the_budgets_drop_and_shorten_visibly_and_never_the_authoritative_detections_first(tmp_path,
                                                                                            tracking_startup) -> None:
    out = simulate(tmp_path, tracking_startup, tracking=make_script("filtered", start=2.0, end=4.0), duration=10.0,
                   captures=(), tracking_bounds=DiagnosticBounds(max_records=12, max_line_bytes=330))
    frames, summary = tracking_lines(out)
    assert summary["kept"] == 12 and summary["dropped"] == 28 and summary["processed"] == 40 and len(frames) == 12
    shortened = [f for f in frames if f["trunc"]]
    assert shortened and summary["truncated_lines"] == len(shortened)
    assert any(f["trunc"].get("shadow") == 1 for f in shortened)  # the gap frames lost their shadow box
    order = ("shadow", "assign", "unpub", "lost", "pub", "det")
    for f in shortened:  # a list is cut only once every list before it in that order is already empty
        for name in f["trunc"]:
            assert all(f[earlier] == [] for earlier in order[:order.index(name)]), (name, f)
    floor = ("shadow", "assign", "unpub", "lost", "pub", "det")
    assert all(len(json.dumps(f, separators=(",", ":")).encode()) <= 330 or all(f[k] == [] for k in floor)
               for f in frames)  # within the bound, or every list already empty (the fixed fields are the floor)
    (tmp_path / "b").mkdir()
    by_bytes = simulate(tmp_path / "b", tracking_startup, tracking=make_script(), duration=5.0, captures=(),
                        tracking_bounds=DiagnosticBounds(max_bytes=1500))
    frames, summary = tracking_lines(by_bytes)
    assert summary["dropped"] > 0 and summary["bytes"] <= 1500 and len(frames) == summary["kept"]


def test_without_the_flag_no_tracking_record_marker_or_status_block_exists(tmp_path, startup) -> None:
    out = simulate(tmp_path, startup, duration=70.0, captures=((60.0, "e1"),))
    frames, summary = tracking_lines(out)
    assert frames == [] and summary is None
    status = json.loads((out / "status-e1.json").read_text())
    assert "tracking" not in status["components"]["detector"]
    first = json.loads((out / "run.jsonl").read_text().splitlines()[0])
    assert first["startup"]["face"] == {"validation_run": True, "identities_enrolled": 1}


def test_the_status_capture_carries_the_budget(tmp_path, tracking_startup) -> None:
    out = simulate(tmp_path, tracking_startup, tracking=make_script(), duration=70.0, captures=((60.0, "e1"),))
    status = json.loads((out / "status-e1.json").read_text())
    tracking = status["components"]["detector"]["tracking"]
    assert tracking["kept"] == 240 and tracking["dropped"] == 0 and tracking["problems"] == {}


# ---------------------------------------------------------------- the reading


def test_a_filtered_gap_is_read_from_shadow_boxes_at_the_seat_and_the_runtimes_expiry(mods, tmp_path,
                                                                                       tracking_startup, f1_list) -> None:
    out = session(mods, tmp_path, tracking_startup, "filtered")
    reading = mods.check.read_tracking(out, f1_list)
    text = "\n".join(reading.lines)
    assert reading.ok, text
    assert line_for(text, "a tracking diagnostic run").startswith("PASS")
    assert line_for(text, "tracking records").startswith("PASS")
    assert line_for(text, "passphrase, name and nonconforming counts").startswith("PASS")  # the real counts
    g = gap(text)
    assert g["frames"] == 20 and g["evidence"] == "complete" and g["seat_id_before"] == 1
    assert g["at_seat_counts"] == {"det_none": 20, "shadow_at_seat": 20}
    assert g["seat_track_lost"] == {"frames_seen_lost": 20, "max_updates_lost": 20, "removed_at_n": None}
    assert g["end"]["outcome"] == "refound" and g["end"]["same_id"] is True and g["end"]["span_s"] == 5.25
    assert g["runtime_current_until_n"] < g["end"]["n"]
    assert set(g["consistent_with"]) == {"filtered_by_the_confidence_cut",
                                         "same_number_refound_after_the_runtime_expired_it"}
    assert described(text, "A: gaps observed")["gaps"] == 1
    assert 1 in [t for t, _ in described(text, "identity transitions to cleared")]  # the seat track, still retained


def test_a_missing_gap_claims_nothing_beyond_readable_frames(mods, tmp_path, tracking_startup, f1_list) -> None:
    out = session(mods, tmp_path, tracking_startup, "missing")
    text = "\n".join(mods.check.read_tracking(out, f1_list).lines)
    g = gap(text)
    assert g["at_seat_counts"] == {"det_none": 20, "shadow_none": 20}
    assert "no_box_at_the_seat_at_or_above_the_shadow_confidence" in g["consistent_with"]


def test_an_association_failure_is_read_from_the_trackers_own_cost_and_its_limit(mods, tmp_path, tracking_startup,
                                                                                  f1_list) -> None:
    out = session(mods, tmp_path, tracking_startup, "association")
    g = gap("\n".join(mods.check.read_tracking(out, f1_list).lines))
    assert g["at_seat_counts"]["det_at_seat"] == 20 and g["unpub_at_seat"] == {"new_unconfirmed": 20}
    first = g["association_at_seat_dets"]["first"]
    assert first["cost_seat_track_to_seat_det"] == 0.95 and first["thresh"] == 0.8 and first["within_limit"] is False
    assert first["seat_track"] == {"id": 1, "st": "lost", "lost": 1} and first["det_outcome"] == "new_unconfirmed"
    assert first["competing_tracks"] == [] and first["seat_track_to_other_dets"] == []
    assert "association_cost_above_the_limit_while_the_seat_track_was_lost" in g["consistent_with"]
    (tmp_path / "c").mkdir()
    conflict = gap("\n".join(mods.check.read_tracking(
        session(mods, tmp_path / "c", tracking_startup, "conflict"), f1_list).lines))
    assert conflict["association_at_seat_dets"]["first"]["within_limit"] is True
    assert "cost_within_the_limit_beside_a_new_or_unpublished_track_(a_conflict_to_read)" in conflict["consistent_with"]
    assert "association_cost_above_the_limit_while_the_seat_track_was_lost" not in conflict["consistent_with"]


def test_a_gap_past_the_buffer_ends_in_a_new_number_after_the_removal(mods, tmp_path, tracking_startup, f1_list) -> None:
    out = session(mods, tmp_path, tracking_startup, "buffer", start=262.0, end=272.0, after_id=7)
    g = gap("\n".join(mods.check.read_tracking(out, f1_list).lines))
    assert g["frames"] == 40 and g["seat_track_lost"]["max_updates_lost"] == 30
    assert g["seat_track_lost"]["removed_at_n"] is not None
    assert g["end"]["outcome"] == "confirmed" and g["end"]["id"] == 7 and g["end"]["same_id"] is False
    assert g["end"]["seat_id_removed_before"] is True
    assert "first_published_box_after_the_tracker_removed_the_seat_track" in g["consistent_with"]


def test_a_sideways_box_is_inconclusive_never_missing_and_an_unreadable_gap_is_incomplete(mods, tmp_path,
                                                                                          tracking_startup, f1_list) -> None:
    sideways = gap("\n".join(mods.check.read_tracking(session(mods, tmp_path, tracking_startup, "sideways"),
                                                       f1_list).lines))
    assert sideways["at_seat_counts"] == {"det_inconclusive": 20, "shadow_none": 20}
    assert not {"filtered_by_the_confidence_cut", "no_box_at_the_seat_at_or_above_the_shadow_confidence"} & set(
        sideways["consistent_with"])  # a sideways box is neither a detection at the seat nor an absence
    (tmp_path / "u").mkdir()
    unreadable = gap("\n".join(mods.check.read_tracking(session(mods, tmp_path / "u", tracking_startup, "unreadable"),
                                                         f1_list).lines))
    assert unreadable["evidence"] == "INCOMPLETE" and unreadable["coverage"]["problems"] == {"raw_missing": 20}
    assert unreadable["at_seat_counts"] == {"det_none": 20, "shadow_unreadable": 20}
    assert unreadable["consistent_with"] == ["nothing_read_(incomplete)"]


def test_no_gap_in_a_is_not_reproduced_and_the_control_reads_the_empty_windows(mods, tmp_path, tracking_startup,
                                                                                f1_list) -> None:
    out = session(mods, tmp_path, tracking_startup, "stray")
    reading = mods.check.read_tracking(out, f1_list)
    text = "\n".join(reading.lines)
    assert reading.ok, text
    assert "A: no gap of two or more frames" in text and "not reproduced" in text
    assert not any(line.startswith("DESCRIPTIVE A gap") for line in reading.lines)
    control = described(text, "control: the empty windows")
    assert control["E1"]["with_det"] == 4 and control["E1"]["unpub_by_outcome"] == {"new_unconfirmed": 4}
    assert control["E1"]["with_pub"] == 0 and control["E2"]["with_det"] == 0
    departure = described(text, "the departure at 6:00")
    assert departure["interval_s"] == [360.1, 380.1] and departure["after_latest"]["with_pub"] == 0
    assert "UNMEASURED" in line_for(text, "overhead")
    assert described(text, "overhead")["backend_ns"]["n"] == 2400


def test_dropped_records_fail_the_item_and_mark_coverage_incomplete(mods, tmp_path, tracking_startup, f1_list) -> None:
    out = simulate(tmp_path, tracking_startup, tracking=make_script("filtered"),
                   tracking_bounds=DiagnosticBounds(max_records=1400))
    finish(mods, out, account=account_v3(mods), cues=FORMAT3_CUES, cued_format=9)
    reading = mods.check.read_tracking(out, f1_list)
    text = "\n".join(reading.lines)
    assert not reading.ok and line_for(text, "tracking records").startswith("FAIL")
    assert described(text, "per window: coverage")["K2"]["frames"] == 0  # the budget ran out before K2
    assert "A: no frame records" in text or "A gap" in text  # whatever A holds is read from what exists


def test_f2s_reading_refuses_a_diagnostic_run_and_this_reading_refuses_an_f2_run(mods, tmp_path, tracking_startup,
                                                                                 startup, f1_list) -> None:
    diagnostic = session(mods, tmp_path, tracking_startup)
    f2 = mods.check.read_f2(diagnostic, f1_list)
    assert not f2.ok and line_for("\n".join(f2.lines), "face admitted for this validation run").startswith("FAIL")
    assert "face-f2" not in "\n".join(f2.lines)  # F2's verdict line is printed by its command, never by a reading
    (tmp_path / "plain").mkdir()
    plain = simulate(tmp_path / "plain", startup)
    finish(mods, plain, account=account_v3(mods), cues=FORMAT3_CUES, cued_format=9)
    reading = mods.check.read_tracking(plain, f1_list)
    text = "\n".join(reading.lines)
    assert not reading.ok and line_for(text, "a tracking diagnostic run").startswith("FAIL")
    assert line_for(text, "tracking records").startswith("MISSING")


def test_the_tracking_lines_are_numbers_and_fixed_labels_to_the_privacy_profile(mods, tmp_path, tracking_startup) -> None:
    out = simulate(tmp_path, tracking_startup, tracking=make_script("association"), duration=300.0, captures=())
    assert mods.evidence.read(out / "run.jsonl", "f2").nonconforming == []
    lines = (out / "run.jsonl").read_text().splitlines()
    for index, raw in enumerate(lines):
        item = json.loads(raw)
        if item.get("tracking") == "frame" and item["unpub"]:
            item["unpub"][0]["note"] = "Jane"  # an undeclared key, a name: nonconforming
            lines[index] = json.dumps(item)
            break
    (out / "run.jsonl").write_text("\n".join(lines) + "\n")
    assert len(mods.evidence.read(out / "run.jsonl", "f2").nonconforming) == 1


# ---------------------------------------------------------------- session 80: readiness before T0


def test_the_readiness_item_reads_the_guards_own_record_and_never_passes_without_it(mods, tmp_path, tracking_startup,
                                                                                  f1_list) -> None:
    out = session(mods, tmp_path, tracking_startup, "filtered")
    text = "\n".join(mods.check.read_tracking(out, f1_list).lines)
    assert line_for(text, "readiness required before T0").startswith("PASS")
    result = json.loads((out / "result.json").read_text())
    for readiness, label in (({**READINESS, "met": False, "reason": "tracker_settings_missing"}, "FAIL"),
                             ({**READINESS, "required": "detector"}, "FAIL"), (None, "MISSING")):
        edited = dict(result)
        if readiness is None:
            edited.pop("readiness")  # a guard that required nothing (the session-79 package): never a pass
        else:
            edited["readiness"] = readiness
        (out / "result.json").write_text(json.dumps(edited))
        reading = mods.check.read_tracking(out, f1_list)
        line = line_for("\n".join(reading.lines), "readiness required before T0")
        assert line.startswith(label) and not reading.ok, line
        if readiness is not None and readiness.get("reason"):
            assert "tracker_settings_missing" in line


def test_a_run_refused_before_t0_reads_as_incomplete_with_the_reason_and_claims_nothing(mods, tmp_path,
                                                                                         tracking_startup,
                                                                                         f1_list) -> None:
    """The device run's shape after the fix: the guard stopped the runtime on its starting line (no cue, no capture,
    no record); the reading fails the supervision and readiness items with the reason, finds no records and no T0,
    describes no window or gap, and the refusal's keys conform to the evidence profile."""
    out = session(mods, tmp_path, tracking_startup, "filtered")
    starting = json.loads((out / "run.jsonl").read_text().splitlines()[0])
    starting["startup"]["detector_problem"] = "diagnostic_unsupported (callback_order)"
    starting["startup"]["face"]["tracking_diagnostic"]["tracker"] = None
    stopped = {"run": "stopped", "shutdown": {"stopped": {"capture": True, "face_worker": True}, "all_stopped": True,
                                              "signals_not_recorded": 0}, "database_closed": True}
    (out / "run.jsonl").write_text(json.dumps(starting) + "\n" + json.dumps(stopped) + "\n")
    result = json.loads((out / "result.json").read_text())
    result.update(status="readiness_refused", stop={"requested_by": "readiness_refused", "requested_at_s": 55.0,
                                                    "exited_after_stop_s": 14.0, "forced": False},
                  child={"returncode": 0, "ran_s": 69.0, "launched_utc": "2026-10-16T08:15:21Z"},
                  captures=[{"name": c["name"], "at_s": c["at_s"], "status": "not_reached", "bytes": 0,
                             "listeners": None} for c in result["captures"]],
                  cues=[{"name": c["name"], "at_s": c["at_s"], "printed_after_launch_s": None, "printed_mono_s": None}
                        for c in result["cues"]],
                  readiness={"required": "tracking_diagnostic", "met": False,
                             "reason": "detector_problem:diagnostic_unsupported (callback_order)",
                             "starting_after_s": 55.0},
                  run_output=mods.guard.read_run_output(out / "run.jsonl"))
    (out / "result.json").write_text(json.dumps(result))
    reading = mods.check.read_tracking(out, f1_list)
    text = "\n".join(reading.lines)
    assert not reading.ok
    assert line_for(text, "supervision status").startswith("FAIL")
    readiness = line_for(text, "readiness required before T0")
    assert readiness.startswith("FAIL") and "detector_problem:diagnostic_unsupported (callback_order)" in readiness
    assert line_for(text, "a tracking diagnostic run").startswith("FAIL")
    assert line_for(text, "tracking records").startswith("MISSING")
    assert line_for(text, "T0:").startswith("MISSING")
    assert "A gap" not in text and "per window: coverage" not in text  # nothing described without T0 and windows
    assert mods.evidence.read(out / "result.json", "f2").nonconforming == []  # the refusal's keys are declared
    result["readiness"]["note"] = "Jane"  # an undeclared key, a name: nonconforming
    (out / "result.json").write_text(json.dumps(result))
    assert len(mods.evidence.read(out / "result.json", "f2").nonconforming) == 1


def test_the_cli_reads_a_tracking_session_and_never_prints_an_f2_verdict(mods, tmp_path, tracking_startup, f1_list,
                                                                          capsys) -> None:
    out = session(mods, tmp_path, tracking_startup, "filtered")
    assert mods.check.main(["tracking", str(out), "--f1", str(f1_list)]) == 0
    printed = capsys.readouterr().out
    assert printed.rstrip().endswith("face-tracking: execution complete (descriptive only; it validates nothing)")
    assert "face-f2" not in printed


# ---------------------------------------------------------------- session 81: the diagnostic's health


def failing_shadow(base, *, start: float = 0.0, end: float = 1e9):
    """``base``'s readings with the shadow pass raising between ``start`` and ``end`` (s), as on the device run
    trk-20261016T100149Z (shadow_failed on every frame; the candidates still counted, no limit flag)."""
    def script(t: float, phase: str, track: int | None, tracker: Any) -> TrackerDiagnostics:
        d = base(t, phase, track, tracker)
        if not start <= t < end:
            return d
        return replace(d, shadow=(), shadow_high=0, shadow_agree=0, shadow_limited=None,
                       problems=(*d.problems, "shadow_failed"))
    return script


def test_zero_completed_shadow_passes_fail_the_diagnostic_and_an_empty_pass_is_completed(mods, tmp_path,
                                                                                          tracking_startup,
                                                                                          f1_list) -> None:
    out = simulate(tmp_path, tracking_startup, tracking=failing_shadow(make_script("filtered")))
    finish(mods, out, account=account_v3(mods), cues=FORMAT3_CUES, cued_format=9, readiness=READINESS)
    reading = mods.check.read_tracking(out, f1_list)
    text = "\n".join(reading.lines)
    health = line_for(text, "diagnostic health")
    assert health.startswith("FAIL") and not reading.ok, health
    value = json.loads(health[health.index(": {") + 2:])
    assert value["completed"] == 0 and value["not_completed"] == 2400
    assert value["not_completed_by_problem"] == {"shadow_failed": 2400}
    g = gap(text)  # what was read in the hold stays unreadable for absences, never "nothing at the seat"
    assert g["evidence"] == "INCOMPLETE" and g["at_seat_counts"]["shadow_unreadable"] == 20
    assert g["consistent_with"] == ["nothing_read_(incomplete)"]
    (tmp_path / "empty").mkdir()
    empty = session(mods, tmp_path / "empty", tracking_startup, "missing")  # no shadow box anywhere in the run
    reading = mods.check.read_tracking(empty, f1_list)
    text = "\n".join(reading.lines)
    assert reading.ok, text
    health = line_for(text, "diagnostic health")
    assert health.startswith("PASS")
    counts = json.loads(health[health.index(": {") + 2:])
    assert counts["completed"] == 2400 and counts["completed_without_shadow_boxes"] == 2400


def test_some_failed_passes_keep_the_diagnostic_healthy_and_their_gap_incomplete(mods, tmp_path, tracking_startup,
                                                                                 f1_list) -> None:
    out = simulate(tmp_path, tracking_startup, tracking=failing_shadow(make_script("filtered"), start=262.0, end=267.0))
    finish(mods, out, account=account_v3(mods), cues=FORMAT3_CUES, cued_format=9, readiness=READINESS)
    reading = mods.check.read_tracking(out, f1_list)
    text = "\n".join(reading.lines)
    health = line_for(text, "diagnostic health")
    assert health.startswith("PASS") and reading.ok, text
    counts = json.loads(health[health.index(": {") + 2:])
    assert counts["not_completed_by_problem"] == {"shadow_failed": 20} and counts["completed"] == 2380
    g = gap(text)
    assert g["evidence"] == "INCOMPLETE" and g["coverage"]["problems"] == {"shadow_failed": 20}
    assert g["at_seat_counts"] == {"det_none": 20, "shadow_unreadable": 20}
    assert described(text, "record coverage over the run")["complete"] is False


def test_the_shadow_health_counts_by_frame(mods) -> None:
    import tracking_check  # beside face_check, on the path the mods fixture prepends

    def frame(problems=(), sh_lim=False, shadow=()):
        return {"problems": list(problems), "sh_lim": sh_lim, "shadow": list(shadow)}

    frames = [frame(), frame(shadow=[{"s": 0.2}]), frame(sh_lim=True), frame(["shadow_failed"], None),
              frame(["raw_missing"], None), frame(["clone_failed"], None), frame(["layout_unknown"]),
              frame(sh_lim=None)]
    assert tracking_check.shadow_health(frames) == {
        "frames": 8, "completed": 4, "completed_with_shadow_boxes": 1, "completed_without_shadow_boxes": 3,
        "completed_limited": 1, "not_completed": 4,
        "not_completed_by_problem": {"clone_failed": 1, "raw_missing": 1, "shadow_failed": 1},
        "not_completed_without_a_problem_named": 1}
    assert tracking_check.shadow_health([])["completed"] == 0


# ---------------------------------------------------------------- session 81: the short integration check

SEATED_RUN = ((0, 121, "facing", 1),)  # seated facing the camera throughout: no movement is asked
INTEGRATION_CUES = {"t0": 0.0, "stop": 115.0}
INTEGRATION_CAPTURES = ((60.0, "i1"), (110.0, "end"))
LIMITS = {"pressure_stop_bytes": 4_800_000_000, "mem_free_floor_bytes": 1_073_741_824,
          "mem_available_floor_bytes": 2_147_483_648, "swap_counters": "unchanged", "max_sample_gap_s": 0.5}
GUARD = {"samples": 1100, "largest_gap_s": 0.21, "peak_pressure_bytes": 4_200_000_000,
         "min_mem_free_bytes": 1_300_000_000, "min_mem_available_bytes": 3_500_000_000, "trigger": None,
         "trigger_after_stop_request": False}


def integration(mods, root: Path, tracking_startup, *, tracking=None, duration: float = 120.0,
                cues=INTEGRATION_CUES, result: dict | None = None) -> Path:
    """The short integration check's run as its launch block makes it: 120 s from readiness, the start and stop cues,
    two status captures, the diagnostic on; no account; the counts by the real workflow."""
    out = simulate(root, tracking_startup, script=SEATED_RUN, duration=duration, captures=INTEGRATION_CAPTURES,
                   tracking=tracking or make_script())
    parameters = {"duration_s": duration, "from_launch": False, "ready_timeout_s": 300.0, "stop_grace_s": 90.0,
                  "step4_headroom": True,
                  "status_at": [{"at_s": at, "name": name} for at, name in INTEGRATION_CAPTURES],
                  "cues": [{"at_s": at, "name": name} for name, at in cues.items()],
                  "require_ready": "tracking_diagnostic"}
    finish(mods, out, account=None, cues=cues, duration_s=duration, captures=INTEGRATION_CAPTURES, readiness=READINESS,
           result_update={"parameters": parameters, "limits": LIMITS, "guard": GUARD,
                          "baseline": {"MemFree": 5_400_000_000, "MemAvailable": 6_600_000_000,
                                       "pressure_bytes": 1_390_000_000, "t_mono": 0.5}, **(result or {})})
    return out


def test_the_integration_check_passes_and_describes_shadow_passes_coverage_timing_and_memory(
        mods, tmp_path, tracking_startup, f1_list, capsys) -> None:
    out = integration(mods, tmp_path, tracking_startup)
    reading = mods.check.read_tracking_integration(out, f1_list)
    text = "\n".join(reading.lines)
    assert reading.ok, text
    for item in ("supervision status", "readiness required before T0", "a tracking diagnostic run", "clean shutdown",
                 "identity records", "tracking records", "diagnostic health", "T0:", "a short run",
                 "memory policy selected", "THP scope", "THP disable verified", "post-load releases",
                 "the scene model loaded", "the guard's memory limits as pinned",
                 "camera userinfo counts", "passphrase, name and nonconforming counts"):
        assert line_for(text, item).startswith("PASS"), item
    passes = described(text, "shadow passes per frame")
    assert passes["frames"]["frames"] == 480 and passes["frames"]["completed"] == 480
    assert passes["frames"]["completed_without_shadow_boxes"] == 480  # an empty result is a completed pass
    assert passes["summary"] == {"processed": 480, "kept": 480, "dropped": 0, "problem_counts": {}}
    assert described(text, "record coverage over the run")["complete"] is True
    timing = described(text, "timing")
    assert timing["ready_after_launch_s"] == 70.0 and timing["backend_ns"]["n"] == 480
    assert timing["processed_fps_over_the_records"] == 4.0 and len(timing["captures"]) == 2
    memory = described(text, "memory")
    assert memory["limits"] == LIMITS and memory["peak_pressure_bytes"] == 4_200_000_000
    assert memory["margin_to_pressure_stop"] == 600_000_000
    assert memory["margin_to_mem_free_floor"] == 1_300_000_000 - 1_073_741_824
    assert memory["margin_to_mem_available_floor"] == 3_500_000_000 - 2_147_483_648
    starting = json.loads((out / "run.jsonl").read_text().splitlines()[0])["startup"]
    assert memory["before_loads"] == {k: starting.get(k, "missing") for k in (
        "scene_memory_before", "detector_memory_before", "face_memory_before")}  # the runtime's own starting line
    workload = described(text, "the workload at the last status capture")
    assert sum(workload["face_results_by_outcome"].values()) > 0 and workload["scene_worker"] is not None
    assert "scope: a short integration check" in text and "not ten-minute stability" in text
    assert not any(line.startswith(("DESCRIPTIVE A gap", "DESCRIPTIVE per window")) for line in reading.lines)
    assert mods.evidence.read(out / "result.json", "f2").nonconforming == []
    assert mods.check.main(["tracking-integration", str(out), "--f1", str(f1_list)]) == 0
    printed = capsys.readouterr().out
    assert printed.rstrip().endswith(mods.check.INTEGRATION_VERDICT[True])
    assert "face-f2" not in printed and "face-tracking:" not in printed


def test_the_sealed_guard_stop_fails_the_integration_check_and_its_memory_is_described(mods, tmp_path,
                                                                                       tracking_startup, f1_list,
                                                                                       capsys) -> None:
    """The device run's outcome on this check's shape: the guard's own trigger, status, stop and memory figures as
    sealed in trk-20261016T100149Z's result.json; the stop cue never printed. Every count still 0 (the trigger's
    numbers and labels are declared), the memory described, the verdict incomplete."""
    sealed = json.loads((Path(__file__).resolve().parents[1] / "support/fixtures/"
                         "trk-20261016T100149Z-tv-result.json").read_text())
    cues = [{"name": "t0", "at_s": 0.0, "printed_after_launch_s": 70.1, "printed_mono_s": 1.1},
            {"name": "stop", "at_s": 115.0, "printed_after_launch_s": None, "printed_mono_s": None}]
    out = integration(mods, tmp_path, tracking_startup, duration=18.75, result={
        "status": sealed["status"], "stop": sealed["stop"], "guard": sealed["guard"], "limits": sealed["limits"],
        "cues": cues})
    reading = mods.check.read_tracking_integration(out, f1_list)
    text = "\n".join(reading.lines)
    assert not reading.ok
    assert line_for(text, "supervision status").startswith("FAIL")
    guard = line_for(text, "no memory-guard stop")
    assert guard.startswith("FAIL") and "free_or_available_stop" in guard
    assert line_for(text, "a short run").startswith("FAIL")  # the stop cue never printed
    assert line_for(text, "passphrase, name and nonconforming counts").startswith("PASS")
    assert mods.evidence.read(out / "result.json", "f2").nonconforming == []
    memory = described(text, "memory")
    assert memory["min_mem_free_bytes"] == 1_071_239_168 and memory["margin_to_mem_free_floor"] == -2_502_656
    assert mods.check.main(["tracking-integration", str(out), "--f1", str(f1_list)]) == 1
    assert capsys.readouterr().out.rstrip().endswith(mods.check.INTEGRATION_VERDICT[False])


def test_zero_completed_shadow_passes_fail_the_integration_check(mods, tmp_path, tracking_startup, f1_list) -> None:
    out = integration(mods, tmp_path, tracking_startup, tracking=failing_shadow(make_script()))
    reading = mods.check.read_tracking_integration(out, f1_list)
    text = "\n".join(reading.lines)
    assert not reading.ok and line_for(text, "diagnostic health").startswith("FAIL")
    assert described(text, "shadow passes per frame")["frames"]["not_completed_by_problem"] == {"shadow_failed": 480}


@pytest.mark.parametrize("change, item", [
    (lambda start, result: start.update(scene_server={**start["scene_server"], "state": "failed"}),
     "the scene model loaded"),
    (lambda start, result: start.update(scene_problem="scene_server_failed"), "the scene model loaded"),
    (lambda start, result: start.update(memory_policy={"thp": "system", "model_file_release": "none"}),
     "memory policy selected"),
    (lambda start, result: result["limits"].update(mem_free_floor_bytes=512 * 1024 * 1024),
     "the guard's memory limits as pinned"),
    (lambda start, result: result["parameters"].update(step4_headroom=False), "the guard's memory limits as pinned"),
])
def test_another_workload_or_other_limits_are_not_this_check(mods, tmp_path, tracking_startup, f1_list, change,
                                                              item) -> None:
    out = integration(mods, tmp_path, tracking_startup)
    result = json.loads((out / "result.json").read_text())
    change(result["run_output"]["starting"], result)
    (out / "result.json").write_text(json.dumps(result))
    reading = mods.check.read_tracking_integration(out, f1_list)
    assert not reading.ok and line_for("\n".join(reading.lines), item).startswith("FAIL")


@pytest.mark.parametrize("kind", ["longer", "movement_cue", "from_launch"])
def test_a_longer_run_or_a_movement_cue_is_not_this_check(mods, tmp_path, tracking_startup, f1_list, kind) -> None:
    if kind == "longer":
        out = integration(mods, tmp_path, tracking_startup, duration=180.0)
    elif kind == "movement_cue":
        out = integration(mods, tmp_path, tracking_startup, cues={"t0": 0.0, "a": 60.0, "stop": 115.0})
    else:
        out = integration(mods, tmp_path, tracking_startup)
        result = json.loads((out / "result.json").read_text())
        result["parameters"]["from_launch"] = True
        (out / "result.json").write_text(json.dumps(result))
    reading = mods.check.read_tracking_integration(out, f1_list)
    assert not reading.ok and line_for("\n".join(reading.lines), "a short run").startswith("FAIL")
