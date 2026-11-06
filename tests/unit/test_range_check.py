"""Route B's range session reading (V3, session 116 part 18): range_check.py and face_account's format 12.

The sessions are the real runtime's records: test_face_check's scripted run on route B (main_single, schema 8) with no
scene analysis, paced by the range schedule's own cues (face_account.RANGE_CUES), each script simulated once and copied
per test. Synthetic faces and embeddings only."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from face_runs import ENROLLED, real_face_startup, startup_line

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib mode: the tests' own module is not importable by name
from test_face_check import F2_PROVENANCE, RUNNERS, _mutate, finish, mods, save_account, simulate  # noqa: E402,F401

sys.path.insert(0, str(RUNNERS))
fa = importlib.import_module("face_account")
rc = importlib.import_module("range_check")
fc = importlib.import_module("face_check")

FPS = 2.0  # frames per simulated second: the face ticks stay at 1 Hz
CAPTURES = ((300.0, "c1"), (900.0, "c2"), (1500.0, "c3"), (2075.0, "end"))
READY = {"required": "face_windows", "met": True, "reason": None, "starting_after_s": 70.0}
ACCOUNT = {"empty_at_start": "yes", **{f"{b[0]}_holds_not_as_cued": "none" for b in fa.RANGE_BLOCKS},
           "walk_ins_not_as_cued": "none", "other_faces_in_view": "no", "notes": "a note that is never printed"}
MARKS = {"seat_m": "0.90", "mark4_m": "4.02", "mark5_m": "4.97", "measured_with": "tape", "out_of_view_spot": "yes"}
HOLDS = {(p["block"], p["number"]): p for p in fa.RANGE_HOLD_PLAN}


def script(changes: dict | None = None) -> tuple:
    """The schedule acted out (simulated seconds; a frame at t is 0.1 s before the cue printed at t): at the Mac at T0,
    out of view until the first walk bell; each hold walked into turned away from 3 s after its walk bell, the face
    turned to the lens 1 s after its hold bell (natural holds: 5 s turned away from 20 s in, and 10 s from 40 s), turned
    away at the end bell and gone 3 s later, on a new track each hold; then the walk-ins, facing all the way.
    ``changes`` maps (block, number) to "absent" (nobody at that hold), "early" (facing from the walk bell on),
    ("late", s) (facing only from s after the hold bell) or "stay" (the previous hold's track continues, facing)."""
    changes = changes or {}
    segments, track = [(0, 10, "facing", 900), (10, 60, "absent", None)], 1
    for plan in fa.RANGE_HOLD_PLAN:
        go, hold, end = plan["go"], plan["hold"], plan["end"]
        change = changes.get((plan["block"], plan["number"]))
        if change == "absent":
            track += 1
            continue
        if change == "stay":
            segments.append((go - fa.RANGE_OUT_S, end + 3, "facing", track - 1))
            continue
        if change == "early":
            segments.append((go + 3, end, "facing", track))
        elif isinstance(change, tuple):
            segments += [(go + 3, hold + change[1], "away", track), (hold + change[1], end, "facing", track)]
        elif plan["kind"] == "natural":
            segments += [(go + 3, hold + 1, "away", track), (hold + 1, hold + 20, "facing", track),
                         (hold + 20, hold + 25, "away", track), (hold + 25, hold + 40, "facing", track),
                         (hold + 40, hold + 50, "away", track), (hold + 50, end, "facing", track)]
        else:
            segments += [(go + 3, hold + 1, "away", track), (hold + 1, end, "facing", track)]
        segments.append((end, end + 3, "away", track))
        track += 1
    for plan in fa.RANGE_WALK_PLAN:
        segments += [(plan["go"] + 1, plan["end"], "facing", track), (plan["end"], plan["end"] + 3, "away", track)]
        track += 1
    return tuple(sorted(segments))


_BUILT: dict[str, Path] = {}


def session(mods, tmp_path, tmp_path_factory, accepted_scene, *, key: str = "pass", changes=None,
            account=ACCOUNT, marks=MARKS, config: str | None = None, launch_after_marks: bool = True,
            scene: bool = False, duration: float | None = None) -> Path:
    """A range session's directory (the run in its rv/), simulated once per ``key`` and copied here; ``account``,
    ``marks`` and ``config`` are written into the copy, as the package's blocks would."""
    if key not in _BUILT:
        root = tmp_path_factory.mktemp(f"range-{key}")
        startup_root = root / "startup"
        startup_root.mkdir()
        startup = startup_line(real_face_startup(startup_root, accepted_scene, windows=True), scene=scene)
        out = simulate(root, startup, script=script(changes), fps=FPS, route_b=True, scene=scene,
                       duration=duration or fa.RANGE_DURATION_S, captures=CAPTURES)
        listeners = [{"family": "tcp", "port": 18090, "scope": "loopback"}]  # the status page alone: no scene server
        finish(mods, out, account=None, cues=dict(fa.RANGE_CUE_AT_S), duration_s=fa.RANGE_DURATION_S,
               captures=CAPTURES, readiness=READY, result_update={"captures": [
                   {"name": name, "at_s": at, "status": "saved", "listeners": listeners} for at, name in CAPTURES
                   if at <= (duration or fa.RANGE_DURATION_S)]})
        _BUILT[key] = out
    target = tmp_path / "session"
    target.mkdir()
    run = target / "rv"
    shutil.copytree(_BUILT[key], run)
    (target / "face.yaml").write_text(rc.RANGE_CONFIG if config is None else config)
    text = "".join(f"{k}={v}\n" for k, v in marks.items())
    (target / "marks.txt").write_text(text)
    lines = [f"marks_sha256={hashlib.sha256(text.encode()).hexdigest()}",
             f"RV_config_sha256={hashlib.sha256(rc.RANGE_CONFIG.encode()).hexdigest()}", "RV_launch_utc=2026-11-07T09:00:00Z"]
    (target / "provenance.txt").write_text("\n".join(lines if launch_after_marks else lines[::-1]) + "\n")
    if account is not None:
        save_account(mods, run, account, 12)
        _counts(run)  # after the account, as the package's reading block
    return run


def _counts(run: Path) -> None:
    from test_face_check import bash_counts

    for function, name in (("face_counts_camera", "secret-counts-camera.txt"),
                           ("face_counts_f2", "secret-counts-identity.txt")):
        done = bash_counts(function, str(run))
        assert done.returncode == 0, done.stderr
        (run / name).write_text(done.stdout)


@pytest.fixture
def f1_list(tmp_path) -> Path:
    f1 = tmp_path / "f1"
    f1.mkdir()
    (f1 / "list.json").write_text(json.dumps({"identity": "list", "identities": [
        {"identity_id": ENROLLED, "prototypes": 3, "consent_date": "2026-10-08", "enrolled_utc": "2026-10-08T09:00:00+00:00"}]}))
    return f1


def read(run: Path, f1: Path) -> tuple[bool, str]:
    reading = rc.read_range(run, f1)
    return reading.ok, "\n".join(reading.lines)


def line_for(text: str, start: str) -> str:
    return next(line for line in text.splitlines() if line.split(" ", 1)[1].startswith(start))


# ---------------------------------------------------------------- the schedule and the account


def test_the_schedule_is_what_the_guard_accepts_and_the_operator_can_follow(mods) -> None:
    """Every cue through the guard's own parser (1-160 printable ASCII, a short name), in time order, inside the guard's
    longest duration; each hold's bells as part 14 fixed them (30 s facing, 60 s natural, five per block), a new track
    possible between holds (out of view at least RANGE_OUT_S), and the run's duration after the stop bell."""
    parsed = [mods.guard._cue_at(text) for text in
              (f"{at}:{name}:{words}" for name, at, words in fa.RANGE_CUES)]
    assert [p[1] for p in parsed] == [name for name, _, _ in fa.RANGE_CUES] and len(set(fa.RANGE_CUE_AT_S)) == 83
    assert importlib.import_module("guard_verdict")._cues(",".join(fa.RANGE_CUE_AT_S)) == tuple(fa.RANGE_CUE_AT_S)  # rb-V's --cues
    times = [p[0] for p in parsed]
    assert times == sorted(times) and len(set(times)) == len(times)
    assert fa.RANGE_DURATION_S <= mods.guard.MAX_DURATION_S and fa.RANGE_DURATION_S == times[-1] + fa.RANGE_STOP_S
    for plan in fa.RANGE_HOLD_PLAN:
        assert plan["hold"] - plan["go"] == fa.RANGE_APPROACH_S > fa.RANGE_ARRIVAL_S
        assert plan["end"] - plan["hold"] == {"facing": 30, "natural": 60}[plan["kind"]]
    for a, b in zip(fa.RANGE_HOLD_PLAN, fa.RANGE_HOLD_PLAN[1:]):
        assert b["go"] - a["end"] == (fa.RANGE_TRANSITION_S if a["block"] != b["block"] else fa.RANGE_OUT_S)
    blocks = [p["block"] for p in fa.RANGE_HOLD_PLAN]
    assert {b: blocks.count(b) for b in blocks} == {b[0]: 5 for b in fa.RANGE_BLOCKS}
    assert fa.RANGE_CUE_AT_S["t0"] == 0 and rc.EMPTY_WINDOW_S[1] < fa.RANGE_FIRST_GO_S
    for name, _, words in fa.RANGE_CUES:  # the hold bells say what to do; the walk bells say where and within 10 s
        if name.endswith("go") and not name.startswith("walk"):
            assert "face turned away" in words and "within 10 s" in words
    out = __import__("subprocess").run([sys.executable, str(RUNNERS / "range_check.py"), "cues"], capture_output=True,
                                       text=True, check=True).stdout.splitlines()
    assert out == [f"{at}:{name}:{words}" for name, at, words in fa.RANGE_CUES]


def test_format_12_accepts_hold_numbers_and_unknown_and_locks(mods, tmp_path) -> None:
    run = tmp_path / "rv"
    run.mkdir()
    (run / "provenance.txt").write_text("x=1\n")
    (run / "result.json").write_text("{}")
    answers = {**ACCOUNT, "m4_facing_holds_not_as_cued": "4 2", "m5_natural_holds_not_as_cued": "unknown"}
    typed = iter(["7", "1 1", answers["empty_at_start"]] + [answers[f.key] for f in fa.FIELDS_V12[1:]] + ["SAVE"])
    said: list[str] = []
    assert fa.enter(run, ask=lambda prompt: next(typed), say=said.append, account_format_number=12) == 0
    text = (run / "account.txt").read_text()
    assert text.splitlines()[0] == fa.HEADER_V12 and "m4_facing_holds_not_as_cued=2,4" in text
    values = fc.parse_account(text)
    assert fa.hold_numbers(values["m4_facing_holds_not_as_cued"]) == {2, 4}
    assert fa.hold_numbers(values["seat_holds_not_as_cued"]) == frozenset()
    assert fa.hold_numbers(values["m5_natural_holds_not_as_cued"]) is None
    assert fa.review_separate(values, fa.FIELDS_V12)[-1].endswith(": yes")
    assert fa.account_format(text)[0] == 12 and fa.lock_state(run) == hashlib.sha256(text.encode()).hexdigest()
    assert any("not accepted" in line for line in said)  # "7" and "1 1" were refused, never saved
    walk = next(f for f in fa.FIELDS_V12 if f.key == "walk_ins_not_as_cued")
    assert fa.accepted(walk, "4")[0] is None and fa.accepted(walk, "3,1") == ("1,3", None)


# ---------------------------------------------------------------- the reading


def test_a_session_as_scripted_passes_every_item_and_still_claims_no_range_or_beta(
        mods, tmp_path, tmp_path_factory, accepted_scene, f1_list) -> None:
    run = session(mods, tmp_path, tmp_path_factory, accepted_scene)
    ok, text = read(run, f1_list)
    assert ok, text
    assert not [line for line in text.splitlines() if line.startswith(("FAIL", "MISSING", "NOT_EXERCISED"))]
    for name, required in (("Seat", 5), ("4.0 m facing", 4), ("5.0 m facing", 4)):
        item = line_for(text, f"{name}: KNOWN at the mark")
        assert item.startswith("PASS") and f"at least {required} of 5" in item and '"successes": 5' in item
    for name in ("4.0 m natural", "5.0 m natural"):
        assert line_for(text, f"{name}: zero wrong associations").startswith("PASS")
    for layer in ("person detection", "face detection", "matching", "retained identity"):
        assert line_for(text, f"5.0 m facing {layer}").startswith("DESCRIPTIVE")
    assert text.count("EXPLORATORY walk-in") == 3 and "not judged" in line_for(text, "walk-in 1")
    pending = [line for line in text.splitlines() if line.startswith("PENDING")]
    assert len(pending) == 2 and "non-enrolled" in pending[0] and "two-person" in pending[1]
    assert "not beta readiness" in text.splitlines()[-1] and "never a rate" in text.splitlines()[-1]
    # nothing private: no note, no identity ID, no similarity value
    assert "a note that is never printed" not in text and ENROLLED not in text and "similarity" not in text
    natural = line_for(text, "4.0 m natural retained identity")  # the scripted turns away keep the same track, so
    assert '"interruptions": 0' in natural and '"lasted_to_end": true' in natural  # KNOWN is retained through them


def test_votes_from_the_approach_are_never_the_marks_and_a_late_known_misses(
        mods, tmp_path, tmp_path_factory, accepted_scene, f1_list) -> None:
    """5.0 m facing: the face shown from the walk bell on in every hold, so KNOWN comes before the hold bell or from
    approach votes: not_at_mark, never a success, and the block is not exercised (no miss). 4.0 m facing: holds 1 and 2
    face the lens only 12 s after the bell: late, two misses, so 4 of 5 is unreachable and the block fails, unmet. The
    seat's hold 3 has nobody: a miss, so the seat control fails."""
    changes = {**{("m5_facing", n): "early" for n in range(1, 6)}, ("m4_facing", 1): ("late", 12),
               ("m4_facing", 2): ("late", 12), ("seat", 3): "absent"}
    run = session(mods, tmp_path, tmp_path_factory, accepted_scene, key="faults", changes=changes)
    ok, text = read(run, f1_list)
    assert not ok
    far = line_for(text, "5.0 m facing: KNOWN at the mark")
    assert far.startswith("NOT_EXERCISED") and '"not_at_mark": 5' in far and '"successes": 0' in far
    near = line_for(text, "4.0 m facing: KNOWN at the mark")
    assert near.startswith("FAIL") and "(unmet)" in near and '"misses": 2' in near and '"1": "late' in near
    seat = line_for(text, "Seat: KNOWN at the mark")
    assert seat.startswith("FAIL") and '"3": "none"' in seat
    for name in ("4.0 m natural", "5.0 m natural"):  # untouched blocks are read as before
        assert line_for(text, f"{name}: zero wrong associations").startswith("PASS")


def test_a_known_carried_into_the_next_hold_is_not_at_mark(mods, tmp_path, tmp_path_factory, accepted_scene,
                                                           f1_list) -> None:
    """The operator stays in view on the same track from hold 1 into hold 2: KNOWN at hold 2's bell is not the mark's
    (it is retained from before), whatever happens in the hold."""
    run = session(mods, tmp_path, tmp_path_factory, accepted_scene, key="stay", changes={("m4_facing", 2): "stay"})
    ok, text = read(run, f1_list)
    near = line_for(text, "4.0 m facing: KNOWN at the mark")
    assert near.startswith("PASS") and '"2": "not_at_mark' in near and '"successes": 4' in near


@pytest.mark.parametrize(("answers", "block", "label", "why"), [
    ({"m4_facing_holds_not_as_cued": "3"}, "4.0 m facing: KNOWN", "PASS", '"not_as_cued": [3]'),
    ({"m4_facing_holds_not_as_cued": "2,3"}, "4.0 m facing: KNOWN", "NOT_EXERCISED", "leave it unsettled"),
    ({"m5_natural_holds_not_as_cued": "unknown"}, "5.0 m natural: zero", "NOT_EXERCISED", "holds not as cued unknown"),
    ({"m4_natural_holds_not_as_cued": "5"}, "4.0 m natural: zero", "NOT_EXERCISED", "1 hold(s) not as cued"),
    ({"other_faces_in_view": "unknown"}, "Seat: KNOWN", "NOT_EXERCISED", "another face may have been in view"),
])
def test_the_account_leaves_holds_out_by_number_and_never_fills_them_in(
        mods, tmp_path, tmp_path_factory, accepted_scene, f1_list, answers, block, label, why) -> None:
    """A hold left out by number is never a success: with one left out the facing block still has 4 of 5; with two
    left out it has 3 successes and no miss, so it is unsettled (not exercised), never failed and never passed. A
    natural block needs all five holds as cued; an unknown answer, or another face possibly in view, settles nothing."""
    run = session(mods, tmp_path, tmp_path_factory, accepted_scene, account={**ACCOUNT, **answers})
    ok, text = read(run, f1_list)
    item = line_for(text, block)
    assert item.startswith(label), item
    assert not ok or label == "PASS"
    assert why is None or why in item


@pytest.mark.parametrize(("marks", "launch_after", "block"), [
    ({**MARKS, "mark5_m": "5.3"}, True, "5.0 m facing: KNOWN"),
    ({**MARKS, "out_of_view_spot": "no"}, True, "4.0 m natural: zero"),
    (MARKS, False, "Seat: KNOWN"),
])
def test_marks_must_be_measured_within_tolerance_and_recorded_before_the_launch(
        mods, tmp_path, tmp_path_factory, accepted_scene, f1_list, marks, launch_after, block) -> None:
    run = session(mods, tmp_path, tmp_path_factory, accepted_scene, marks=marks, launch_after_marks=launch_after)
    ok, text = read(run, f1_list)
    assert not ok
    assert line_for(text, "the marks, lens to face").startswith("FAIL"), text
    if launch_after:
        assert line_for(text, block).startswith("NOT_EXERCISED") and "not measured within 0.1 m" in line_for(text, block)
    else:
        assert "before the launch" in line_for(text, "the marks, lens to face")
        assert line_for(text, block).startswith("NOT_EXERCISED")


def test_a_changed_configuration_or_frame_reference_fails_before_any_hold_is_read(
        mods, tmp_path, tmp_path_factory, accepted_scene, f1_list) -> None:
    run = session(mods, tmp_path, tmp_path_factory, accepted_scene,
                  config=rc.RANGE_CONFIG + "identity: {match_threshold: 0.6}\n")
    ok, text = read(run, f1_list)
    assert not ok and line_for(text, "the session's face.yaml").startswith("FAIL")
    other = tmp_path / "other"
    other.mkdir()
    run = session(mods, other, tmp_path_factory, accepted_scene)

    def edit(lines):  # one applied result now names the next frame: no longer its own tick's
        for item in lines:
            if item.get("identity") == "result" and item.get("outcome") == "applied" and item["persons"]:
                item["frame_seq"] += 1
                return lines
        raise AssertionError("no applied result")

    _mutate(run, edit)
    _counts(run)
    ok, text = read(run, f1_list)
    assert not ok and line_for(text, "identity records").startswith("FAIL") and "no offered tick" in text
    assert "Seat: KNOWN" not in text and line_for(text, "Seat: not read").startswith("NOT_EXERCISED")  # nothing judged


def test_a_run_with_the_scene_server_is_not_this_sessions_run(mods, tmp_path, tmp_path_factory, accepted_scene,
                                                             f1_list) -> None:
    run = session(mods, tmp_path, tmp_path_factory, accepted_scene, key="scene", scene=True, duration=120.0)
    ok, text = read(run, f1_list)
    assert not ok
    for item in ("THP scope", "post-load releases", "no scene server", "each status capture"):
        assert line_for(text, item).startswith(("FAIL", "MISSING")), item


# ---------------------------------------------------------------- the hold logic on hand-built records


def _line(transitions, results=(), end=100.0):
    ns = fc.NS
    records = {"tick": [], "result": [dict(r, frame_mono_ns=round(r["t"] * ns), applied_mono_ns=round(r["applied"] * ns))
                                      for r in results],
               "transition": [{"stream_epoch": 1, "track_id": track, "to": to, "from": "unresolved", "identity_id":
                               ENROLLED if to == "known" else None, "basis": None, "reason": None,
                               "mono_ns": round(t * ns)} for track, t, to in transitions]}
    return fc.Timeline(records, 0, end)


def _vote(track, t, applied, w=30):
    return {"t": t, "applied": applied, "outcome": "applied", "stream_epoch": 1,
            "persons": [{"track_id": track, "vote": "match", "identity_id": ENROLLED, "ownership": "assigned",
                         "label": "match", "face": {"w_px": w, "h_px": w, "score": 0.95}}]}


def test_two_tracks_known_at_once_is_a_wrong_association_and_fails_a_natural_block() -> None:
    line = _line([(1, 12.0, "known"), (2, 20.0, "known"), (1, 30.0, "cleared"), (2, 40.0, "cleared")],
                 [_vote(1, 11.0, 11.5), _vote(1, 11.8, 12.0), _vote(2, 19.0, 19.5), _vote(2, 19.8, 20.0)])
    hold = rc._hold(line, rc.Votes(line), ENROLLED, 10.0, 70.0)
    assert hold["wrong"] == 1 and hold["outcome"] == "acquired" and hold["acquired_s"] == 2.0
    assert hold["known_s"] == 28.0 and hold["lasted_to_end"] is False and hold["gaps_s"] == [30.0]
    r = fc.Reading()
    holds = {n: dict(hold, wrong=0) for n in range(1, 6)}
    holds[3] = hold
    rc._judge(r, "m4_natural", "natural", "4.0 m natural", holds, frozenset(), True, True, 4.0)
    assert r.lines[0].startswith("FAIL") and '"wrong_associations": 1' in r.lines[0]


def test_an_approach_vote_makes_the_first_known_not_at_mark_even_if_a_later_one_is_in_the_hold() -> None:
    line = _line([(1, 11.0, "known"), (1, 13.0, "cleared"), (2, 16.0, "known")],
                 [_vote(1, 9.5, 10.2), _vote(1, 10.5, 11.0), _vote(2, 14.0, 14.5), _vote(2, 15.0, 16.0)])
    hold = rc._hold(line, rc.Votes(line), ENROLLED, 10.0, 40.0)
    assert hold["outcome"] == "not_at_mark" and hold["vote_frames_s"] == [-0.5, 0.5]
    line = _line([(1, 15.0, "known")], [_vote(1, 11.0, 11.5), _vote(1, 12.0, 15.0)])
    assert rc._hold(line, rc.Votes(line), ENROLLED, 10.0, 40.0)["outcome"] == "acquired"
    line = _line([(1, 21.0, "known")], [_vote(1, 19.0, 19.5), _vote(1, 20.0, 21.0)])  # 11 s after the bell: late
    assert rc._hold(line, rc.Votes(line), ENROLLED, 10.0, 40.0)["outcome"] == "late"
    line = _line([(1, 15.0, "known")], [_vote(1, 14.0, 15.0)])  # one vote cannot make KNOWN: the records disagree
    assert rc._hold(line, rc.Votes(line), ENROLLED, 10.0, 40.0)["outcome"] == "not_justified"


@pytest.mark.parametrize(("outcomes", "left", "seat", "label"), [
    (["acquired"] * 4 + ["none"], frozenset(), False, "PASS"),
    (["acquired"] * 3 + ["none", "late"], frozenset(), False, "FAIL"),
    (["acquired"] * 3 + ["not_at_mark", "none"], frozenset(), False, "NOT_EXERCISED"),
    (["acquired"] * 4 + ["none"], frozenset(), True, "FAIL"),
    (["acquired"] * 4 + ["late"], frozenset(), True, "PASS"),  # the seat: KNOWN at the mark, no 10 s limit (part 14)
    (["acquired"] * 3 + ["late", "late"], frozenset(), False, "FAIL"),  # facing: late is a miss
    (["acquired"] * 4 + ["not_at_mark"], frozenset(), True, "NOT_EXERCISED"),
    (["acquired"] * 5, frozenset({5}), True, "NOT_EXERCISED"),
    (["acquired"] * 5, None, False, "NOT_EXERCISED"),
])
def test_the_facing_and_seat_rules(outcomes, left, seat, label) -> None:
    holds = {n: {"outcome": o, "acquired_s": 3.0 if o == "acquired" else None, "wrong": 0}
             for n, o in enumerate(outcomes, 1)}
    r = fc.Reading()
    rc._judge(r, "seat" if seat else "m4_facing", "facing", "x", holds, left, True, True, 4.0)
    assert r.lines[0].startswith(label), r.lines[0]


def test_the_readers_constants_are_the_runtimes_and_the_guards(mods) -> None:
    from sentinel.config import IdentityConfig, parse_config
    import yaml

    config = parse_config(yaml.safe_load(rc.RANGE_CONFIG))
    assert config.capture.stream_profile == "main_single" and config.identity == IdentityConfig()
    assert rc.READINESS in mods.guard.READINESS and fc.CONFIRMATIONS == IdentityConfig().confirmations
    assert rc.FACE_ONLY_RELEASES["face"] == fc.FACE_RELEASES and rc.STATUS_PORTS == {18090}
    assert rc.ARRIVAL_MARGIN_S == 5 and rc.ACQUIRE_S == 10.0 and rc.REQUIRED == {"seat": 5, "facing": 4}


def test_a_hold_that_begins_known_is_not_at_mark_even_with_a_new_known_from_hold_votes() -> None:
    """KNOWN on track 1 at the hold bell (carried from before it), then track 1 lost and track 2 KNOWN from votes on
    frames inside the hold: the hold began KNOWN, so nothing in it is the mark's."""
    line = _line([(1, 5.0, "known"), (1, 12.0, "cleared"), (2, 16.0, "known")],
                 [_vote(1, 3.0, 3.5), _vote(1, 4.0, 5.0), _vote(2, 14.0, 14.5), _vote(2, 15.0, 16.0)])
    hold = rc._hold(line, rc.Votes(line), ENROLLED, 10.0, 40.0)
    assert hold["outcome"] == "not_at_mark" and hold["acquired_s"] == 6.0 and hold["vote_frames_s"] == [4.0, 5.0]


@pytest.mark.parametrize("change", ["absent", "late", "extra"])
def test_the_schedule_must_be_printed_as_fixed(mods, tmp_path, tmp_path_factory, accepted_scene, f1_list,
                                               change) -> None:
    """A cue missing, printed late or not in the schedule: the session was not paced as predeclared; with a cue missing
    no hold is read at all."""
    run = session(mods, tmp_path, tmp_path_factory, accepted_scene)
    result = json.loads((run / "result.json").read_text())
    cues = result["cues"]
    target = next(c for c in cues if c["name"] == "m4f3hold")
    if change == "absent":
        cues.remove(target)
    elif change == "late":
        target["printed_mono_s"] += 1.0
    else:
        cues.append(dict(target, name="look"))
    (run / "result.json").write_text(json.dumps(result))
    _counts(run)
    ok, text = read(run, f1_list)
    assert not ok and line_for(text, f"the run's {fa.RANGE_DURATION_S} s and all 83 cues").startswith("FAIL")
    assert ("4.0 m facing: KNOWN" in text) == (change != "absent")


def test_a_natural_block_with_no_face_assigned_exercised_no_association() -> None:
    holds = {n: {"outcome": "none", "acquired_s": None, "wrong": 0} for n in range(1, 6)}
    r = fc.Reading()
    rc._judge(r, "m5_natural", "natural", "5.0 m natural", holds, frozenset(), True, True, 5.0, {n: 0 for n in holds})
    assert r.lines[0].startswith("NOT_EXERCISED") and "no association exercised" in r.lines[0]
    r = fc.Reading()
    rc._judge(r, "m5_natural", "natural", "5.0 m natural", holds, frozenset(), True, True, 5.0, {n: 3 for n in holds})
    assert r.lines[0].startswith("PASS")


def test_hold_numbers_stay_inside_their_own_field() -> None:
    assert rc._left_out("2,4", 5) == {2, 4} and rc._left_out("none", 3) == frozenset()
    assert rc._left_out("6", 5) is None and rc._left_out("4", 3) is None and rc._left_out("unknown", 5) is None


def test_the_f1_gate_never_prints_the_identity_id_and_the_config_must_be_launched(
        mods, tmp_path, tmp_path_factory, accepted_scene, f1_list, monkeypatch) -> None:
    run = session(mods, tmp_path, tmp_path_factory, accepted_scene)
    monkeypatch.setattr(fc, "f1_gate", lambda *a: (True, ["f1_gate=passed", f"f1_identity_id={ENROLLED}"]))
    gate = (f1_list.parent, f1_list.parent, None)
    reading = rc.read_range(run, f1_list, gate)
    text = "\n".join(reading.lines)
    assert ENROLLED not in text and '"f1_identity_id": "recorded"' in text
    provenance = run.parent / "provenance.txt"
    provenance.write_bytes(b"\xff\xfe not text\n" + b"".join(
        line + b"\n" for line in provenance.read_bytes().splitlines() if not line.startswith(b"RV_config_sha256=")))
    ok, text = read(run, f1_list)  # bytes that are not UTF-8 read as text, never a crash
    assert not ok and line_for(text, "the session's face.yaml, as the launch block ran it").startswith("FAIL")
