"""R2: zero-person scenes, freshness and scene hazards with nobody present.

Guide ch. 2 rows 1 and 3, ch. 21 ("No people for several minutes; empty-scene
observation still runs and UI becomes empty"; "camera unplug ... no stale
frame reprocessed"), ch. 22 freshness gate; audit finding 4. Timelines are
synthetic; each test also shows the B0 v1 loop on the same frames.
"""

from __future__ import annotations

from core_harness import EMPTY_ROOM, SMOKE, always, run_core
from timeline_builder import FRAME_MS, TimelineBuilder, frame_at

from sentinel.contracts import EvidenceStatus
from sentinel.live_state import Capability, Occupancy, SceneStatus
from sentinel.media.health import VideoState
from sentinel.rules.scene_hazard import Severity

MINUTES_EMPTY_MS = 3 * 60_000


def test_minutes_without_people_publish_empty_and_keep_scene_checks_running() -> None:
    person_leaves = 3000
    stall = frame_at(person_leaves + MINUTES_EMPTY_MS) - FRAME_MS  # last frame before the stall
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(person_leaves, persons=1)
        .frames_until(person_leaves + MINUTES_EMPTY_MS, persons=0)
        .build()
    )
    run, origin = run_core(timeline, answer=always(EMPTY_ROOM))
    last_detection = frame_at(person_leaves) - FRAME_MS

    # Every step publishes, in order, whether or not anyone is present.
    sequences = [state.sequence for _, state in run.states]
    assert sequences == sorted(set(sequences)) and len(sequences) >= stall // FRAME_MS

    # Before the first frame the state is explicit, not empty.
    first = run.states[0][1]
    assert (first.video, first.occupancy) == (VideoState.STARTING, Occupancy.UNKNOWN)
    # The person is present until one track expiry (1 s) after their last detection.
    present = run.during(0, last_detection + 1000)[1:]
    assert present and all(s.occupancy is Occupancy.OCCUPIED for s in present)
    empty = run.during(last_detection + 1000, stall + 1)
    assert empty and all(
        s.occupancy is Occupancy.EMPTY and s.people == () and s.video is VideoState.FRESH
        for s in empty
    )
    assert empty[-1].occupancy_reason == "nobody detected on fresh video"

    # Scene checks keep their interval for the whole empty period.
    starts = run.job_starts_ms(origin)
    in_empty = [ms for ms in starts if ms >= last_detection + 1000]
    assert len(in_empty) >= MINUTES_EMPTY_MS // 4000 - 2
    assert all(b - a >= 4000 for a, b in zip(starts, starts[1:]))
    assert all(b - a < 4000 + 2 * FRAME_MS for a, b in zip(starts, starts[1:]))
    reported = [s for s in empty if s.scene is SceneStatus.REPORTED]
    assert reported and reported[-1].scene_report.persons_visible == 0

    # B0: v1 stops requesting scene checks and never publishes the empty room.
    assert run.v1.dashboard_persons == 1
    assert run.v1.dashboard_updates == len(range(0, person_leaves, FRAME_MS))
    assert run.v1.scene_requests == run.v1.dashboard_updates


def test_stall_shows_stale_at_2s_and_offline_at_10s_and_nothing_revives() -> None:
    last = frame_at(6000) - FRAME_MS  # last frame before the camera stalls
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(6000, persons=1)
        .tick(last + 999)
        .tick(last + 1000)
        .tick(last + 1999)
        .tick(last + 2000)
        .tick(last + 9999)
        .tick(last + 10_000)
        .disconnect(last + 12_000)
        .tick(last + 15_000)
        .connect(last + 20_000)
        .tick(last + 20_010)
        .frames_until(last + 24_000, persons=0)
        .build()
    )
    run, _ = run_core(timeline, answer=always(EMPTY_ROOM, delay_ms=500), frozen_reader=True)

    def at(offset: int):
        return run.at(last + offset)

    assert at(0).occupancy is Occupancy.OCCUPIED and at(0).scene is SceneStatus.REPORTED
    # No new frame for 1 s: the detector has not looked, so occupancy is unknown, not empty.
    assert at(999).occupancy is Occupancy.OCCUPIED
    assert (at(1000).video, at(1000).occupancy) == (VideoState.FRESH, Occupancy.UNKNOWN)
    assert at(1000).occupancy_reason == "detector has not processed a recent frame"
    assert at(1999).video is VideoState.FRESH
    stale = at(2000)
    assert (stale.video, stale.occupancy, stale.people) == (VideoState.STALE, Occupancy.UNKNOWN, ())
    assert stale.occupancy_reason == "no fresh video (stale)"
    assert stale.scene is SceneStatus.NO_CURRENT_RESULT and stale.scene_report is None
    assert stale.scene_reason == "last scene result not current (no_live_stream)"
    assert at(9999).video is VideoState.STALE and at(10_000).video is VideoState.OFFLINE
    assert at(15_000).video is VideoState.OFFLINE and at(15_000).last_frame_age_ms == 15_000
    assert at(20_010).video is VideoState.OFFLINE  # connected, but no frame yet

    resumed = [s for ms, s in run.states if ms >= last + 20_000 and s.video is VideoState.FRESH]
    assert resumed and resumed[0].occupancy is Occupancy.EMPTY
    # Pre-stall scene evidence never becomes current again; only new-epoch reports do.
    pre_stall_ids = {r.evidence.evidence_id for ms, r in run.evidence if ms <= last}
    assert pre_stall_ids and not any(s.scene_evidence_id in pre_stall_ids for s in resumed)
    # The frozen frame re-delivered on each of the 8 ticks was never processed again:
    # a duplicate while the stream was still fresh, not live afterwards.
    counters = run.core._tracks.counters
    assert (counters["frame_duplicate"], counters["frame_not_live"]) == (3, 5)
    # B0: v1 keeps "seeing" the person in its frozen frame through stall and outage.
    assert run.v1.dashboard_persons == 1 and run.v1.dashboard_updates > len(range(0, 6000, FRAME_MS))


def test_unavailable_detector_is_unknown_occupancy_while_scene_checks_continue() -> None:
    timeline = TimelineBuilder().connect(0).frames_until(6000, persons=0).build()
    run, _ = run_core(timeline, answer=always(EMPTY_ROOM), detector=Capability.UNAVAILABLE)
    final = run.states[-1][1]
    assert (final.video, final.detector, final.occupancy) == (
        VideoState.FRESH,
        Capability.UNAVAILABLE,
        Occupancy.UNKNOWN,
    )
    assert final.occupancy_reason == "person detector unavailable"
    assert final.scene is SceneStatus.REPORTED and len(run.analyzer.jobs) == 2


def smoke_on(*jobs: int, default: dict = EMPTY_ROOM, failures: dict[int, str] | None = None):
    failures = failures or {}

    def answer(number: int, job):
        if number in failures:
            return (1500, failures[number])
        return (1500, SMOKE if number in jobs else default)

    return answer


def test_scene_hazard_with_nobody_present_is_a_capped_candidate_once_per_episode() -> None:
    timeline = TimelineBuilder().connect(0).frames_until(40_000, persons=0).build()
    # Jobs start every ~4 s: 1-2 calm, 3-5 smoke, 6 times out, 7-8 smoke.
    run, origin = run_core(timeline, answer=smoke_on(3, 4, 5, 7, 8, failures={6: "timeout"}))
    starts = run.job_starts_ms(origin)
    assert len(starts) >= 8

    candidates = [c for _, c in run.candidates]
    assert len(candidates) == 2  # one per episode: jobs 3-5, then 7-8 after the timeout reset
    for candidate in candidates:
        assert candidate.severity is Severity.WARNING and not candidate.primary_signal
        assert candidate.people_count == 0
        assert "not a fire alarm" in candidate.reason
    assert [c.evidence_ids for c in candidates] == [("job-3.observed", "job-4.observed"), ("job-7.observed", "job-8.observed")]
    assert all(c.severity is not Severity.CRITICAL for c in candidates)


def test_scene_hazard_confirmation_expires_across_stall_and_reconnect() -> None:
    # Audit finding 4 repro: positive -> stale gap -> positive must not confirm.
    # Job 1 is calm; every later check reports smoke, but a stall and then a
    # reconnect separate them, so no two positives are ever consecutive.
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(6000, persons=0)  # job 2 (4026 ms) reports smoke at 5544 ms
        .tick(9000)  # frames stopped at 5940 ms: stale, so that positive is gone
        .frames_until(13_000, persons=0)  # job 3 (9042 ms) reports smoke at 10560 ms
        .disconnect(13_000)
        .connect(13_500)
        .frames_until(18_000, persons=0)  # job 4 (13530 ms, new epoch) reports smoke
        .build()
    )
    run, origin = run_core(timeline, answer=smoke_on(2, 3, 4))
    assert run.job_starts_ms(origin) == [0, 4026, 9042, 13530, 17556]
    assert run.at(9000).video is VideoState.STALE
    observed = [r.evidence.evidence_id for _, r in run.evidence if r.evidence.status is EvidenceStatus.OBSERVED]
    assert observed == ["job-1.observed", "job-2.observed", "job-3.observed", "job-4.observed"]
    assert run.candidates == []
    # B0-style counterfactual: the same three positives within 12 s, without the
    # stall and reconnect, do confirm (see the episode test above).
