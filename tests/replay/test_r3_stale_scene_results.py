"""R3: delayed, stale and failed scene (VLM) results.

Audit finding 3; guide ch. 2 row 4, ch. 21 ("VLM returns malformed JSON, wrong
event ID, timeout or a very late answer; core events continue") and ch. 27.
Each test replays one synthetic timeline through v2 and shows the B0 v1
behaviour on the same outcomes.
"""

from __future__ import annotations

import json

from scene_harness import ERROR_DETAIL, SETTINGS, StubAnalyzer, run_scene
from timeline_builder import TimelineBuilder, frame_at

from sentinel.contracts import Applicability, EvidenceStatus
from sentinel.jobs import JobPurpose, OutcomeKind, WorkerOutcome
from sentinel.media.clock import FakeClock
from sentinel.replay import ReplayDriver
from sentinel.scene.lane import SceneLane
from sentinel.scene.state import CurrentScene

CALM = {
    "persons_visible": 1,
    "fire_or_smoke": False,
    "threat": "none",
    "observations": ["person standing"],
    "uncertainty": "low",
    "summary": "one person standing",
}
FIRE = {**CALM, "fire_or_smoke": True, "threat": "high", "summary": "possible smoke"}


def test_late_answer_for_incident_a_never_becomes_current_and_only_annotates_a(
    load_timeline,
) -> None:
    run = run_scene(load_timeline("r3_late_scene_result.jsonl"))

    jobs = run.analyzer.jobs
    assert [job.purpose for job in jobs] == [
        JobPurpose.PERIODIC,
        JobPurpose.ENRICHMENT,
        JobPurpose.PERIODIC,
        JobPurpose.ENRICHMENT,
    ]
    assert [job.incident_id for job in jobs] == [None, "inc-A", None, "inc-B"]
    assert run.analyzer.cancelled == ["job-2"]  # the lane gave up on it at its deadline

    by_id = {evidence.evidence_id: (ms, evidence, routing) for ms, evidence, routing in run.evidence}
    assert list(by_id) == ["job-1.observed", "job-2.timeout", "job-3.observed", "job-2.observed.late"]
    ms, timeout, routing = by_id["job-2.timeout"]
    assert ms == 9000 and timeout.status is EvidenceStatus.TIMEOUT
    assert routing.applicability is Applicability.CURRENT  # "scene unknown" is current news
    ms, late, routing = by_id["job-2.observed.late"]
    assert ms == 11500 and late.value is not None and late.value["fire_or_smoke"] is True
    assert routing.applicability is Applicability.EXPIRED and not routing.updates_current

    # The late fire report enriches incident A only; B is untouched by job 2.
    assert run.annotations == {"inc-A": ["job-2.timeout", "job-2.observed.late"]}

    # Current scene: never fire; the timeout replaced job 1's verdict, then job 3 took over.
    assert all(view.report is None or not view.report.fire_or_smoke for _, view in run.views)
    assert run.view_at(8999).evidence.evidence_id == "job-1.observed"
    at_timeout = run.view_at(9000)
    assert at_timeout.report is None and at_timeout.evidence.status is EvidenceStatus.TIMEOUT
    assert run.view_at(12012).evidence.evidence_id == "job-3.observed"

    # B0: v1 makes whatever arrives last the current verdict, however old its frame.
    assert min(run.v1_fire_alert_ms) == 11500
    assert run.v1_threat[-1] == (12012, "high")


def test_answer_after_its_deadline_is_late_even_while_its_frame_is_within_ttl() -> None:
    # Deadline 8 s, TTL 10 s: an answer at 9 s is about a frame young enough to be
    # current, but the job was already given up. It may only annotate history.
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(400)
        .incident(400, "inc-A")
        .result(500, 1, CALM)  # periodic job 1; enrichment job 2 (for A) starts next frame
        .frames_until(8700)
        .result(9000, 2, FIRE)
        .frames_until(9300)
        .build()
    )
    run = run_scene(timeline)
    ids = [e.evidence_id for _, e, _ in run.evidence]
    assert ids == ["job-1.observed", "job-2.timeout", "job-2.observed.late"]
    _, late, routing = run.evidence[-1]
    source_age_ms = 9000 - 396
    assert source_age_ms * 1_000_000 < SETTINGS.ttl_ns  # within TTL, yet:
    assert routing.applicability is Applicability.EXPIRED and not routing.updates_current
    assert routing.annotates == "inc-A"
    assert all(view.report is None or not view.report.fire_or_smoke for _, view in run.views)
    assert min(run.v1_fire_alert_ms) == 9000  # B0: alerts


def test_answer_after_reconnect_is_superseded_not_current() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(700)
        .disconnect(700)
        .connect(1500)
        .frames_until(2000)
        .result(2000, 1, FIRE)
        .frames_until(2400)
        .build()
    )
    run = run_scene(timeline)
    ((_, evidence, routing),) = run.evidence
    assert evidence.status is EvidenceStatus.OBSERVED  # on time, but about the old epoch
    assert routing.applicability is Applicability.SUPERSEDED_EPOCH and not routing.updates_current
    assert all(view.evidence is None for _, view in run.views)
    assert min(run.v1_fire_alert_ms) == 2000  # B0: the old epoch's fire flag alerts


def test_malformed_or_mistyped_output_is_an_error_not_a_verdict() -> None:
    mistyped = {**FIRE, "fire_or_smoke": "true"}  # a string, which v1 coerced to True
    truncated = '```json\n{"threat": "none"\n```'
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(1400)
        .result(1400, 1, CALM)
        .frames_until(4800)  # periodic job 2 starts on the first frame 4 s after job 1
        .result(4800, 2, text=truncated)
        .frames_until(9000)  # periodic job 3
        .result(9000, 3, mistyped)
        .frames_until(9700)
        .build()
    )
    run = run_scene(timeline)
    starts = [job.frame.ingest_mono_ns - run.analyzer.jobs[0].frame.ingest_mono_ns for job in run.analyzer.jobs]
    assert starts == [0, frame_at(4000) * 1_000_000, frame_at(frame_at(4000) + 4000) * 1_000_000]
    statuses = [(e.evidence_id, e.status) for _, e, _ in run.evidence]
    assert statuses == [
        ("job-1.observed", EvidenceStatus.OBSERVED),
        ("job-2.error", EvidenceStatus.ERROR),
        ("job-3.error", EvidenceStatus.ERROR),
    ]
    assert [e.reason for _, e, _ in run.evidence[1:]] == [
        "invalid scene report: not a single JSON object",
        "invalid scene report: fire_or_smoke: bool_type",
    ]
    # The failure replaced the calm verdict instead of letting it stand in.
    after_error = run.view_at(4800)
    assert after_error.report is None and after_error.evidence.status is EvidenceStatus.ERROR
    # B0: truncated text became a 'none' verdict, and the string "true" a fire alert.
    assert run.v1_threat[-1][1] == "high"
    assert [t for ms, t in run.v1_threat if 4800 <= ms < 9000] and all(
        t == "none" for ms, t in run.v1_threat if 4800 <= ms < 9000
    )
    assert min(run.v1_fire_alert_ms) == 9000


def test_response_naming_another_job_is_rejected() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(900)
        .result(900, 1, FIRE, claimed_job="job-99")
        .frames_until(1200)
        .build()
    )
    run = run_scene(timeline)
    ((_, evidence, routing),) = run.evidence
    assert evidence.status is EvidenceStatus.ERROR and evidence.reason == "response names another job"
    assert evidence.value is None and routing.updates_current  # current: "scene unknown"
    assert run.view_at(900).report is None


def test_worker_timeout_and_error_replace_the_previous_verdict_and_are_redacted() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(1300)
        .result(1300, 1, FIRE)
        .frames_until(4500)
        .result(4500, 2, failure="timeout")
        .frames_until(8500)
        .result(8500, 3, failure="error")
        .frames_until(9200)
        .build()
    )
    run = run_scene(timeline)
    assert [e.evidence_id for _, e, _ in run.evidence] == ["job-1.observed", "job-2.timeout", "job-3.error"]
    assert run.view_at(1300).report is not None and run.view_at(1300).report.fire_or_smoke
    for at_ms in (4500, 8500):
        view = run.view_at(at_ms)
        assert view.report is None and view.evidence.value is None  # no old verdict stands in
    error = run.evidence[2][1]
    assert "hunter2" in ERROR_DETAIL and "hunter2" not in error.reason and "admin" not in error.reason
    assert "<redacted>@127.0.0.1" in error.reason
    # B0: after worker failures v1 keeps alerting on the earlier fire verdict.
    assert run.v1_fire_alert_ms[-1] == run.views[-1][0]


def test_scene_checks_run_on_schedule_with_nobody_present() -> None:
    empty = {**CALM, "persons_visible": 0, "observations": [], "summary": "empty room"}
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(2000, persons=0)
        .result(2000, 1, empty)
        .frames_until(8700, persons=0)
        .build()
    )
    run = run_scene(timeline)
    starts = [job.frame.ingest_mono_ns for job in run.analyzer.jobs]
    assert len(starts) == 2 and starts[1] - starts[0] >= SETTINGS.interval_ns
    assert all(job.purpose is JobPurpose.PERIODIC for job in run.analyzer.jobs)
    assert run.view_at(2000).report is not None and run.view_at(2000).report.persons_visible == 0


def test_waiting_enrichment_is_replaced_or_skipped_with_evidence_for_its_incident() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(400)
        .incident(400, "inc-A")
        .incident(500, "inc-B")  # replaces A's waiting request
        .frames_until(600)
        .tick(9000)  # job 1 times out; B's frame is now older than the 8 s job timeout
        .frames_until(9300)
        .build()
    )
    run = run_scene(timeline)
    skipped = [(e.incident_id, r.updates_current) for _, e, r in run.evidence if e.kind == "scene.job_skipped"]
    assert skipped == [("inc-A", False), ("inc-B", False)]
    assert [e.status for _, e, _ in run.evidence if e.kind == "scene.job_skipped"] == [
        EvidenceStatus.UNAVAILABLE
    ] * 2
    assert run.lane.counters["enrichment_replaced"] == 1
    assert run.lane.counters["enrichment_skipped"] == 1
    assert set(run.annotations) == {"inc-A", "inc-B"}
    assert [job.purpose for job in run.analyzer.jobs] == [JobPurpose.PERIODIC, JobPurpose.PERIODIC]
    assert all(view.evidence is None or view.evidence.kind == "scene.report" for _, view in run.views)


def _one_frame(clock: FakeClock):
    driver = ReplayDriver(TimelineBuilder().connect(0).frames_until(1).build(), clock=clock)
    (_, step) = list(driver.steps())
    return step


def test_unknown_or_repeated_results_are_ignored_and_counted() -> None:
    clock = FakeClock()
    lane = SceneLane(StubAnalyzer(), clock, SETTINGS, id_prefix="job")
    calm = json.dumps(CALM)
    assert lane.complete("job-1", WorkerOutcome.completed("job-1", calm)) is None

    step = _one_frame(clock)
    assert lane.on_frame(step.frame, step.stream) == []
    first = lane.complete("job-1", WorkerOutcome.completed("job-1", calm))
    again = lane.complete("job-1", WorkerOutcome.failed("job-1", OutcomeKind.ERROR, "duplicate"))
    assert first is not None and first.status is EvidenceStatus.OBSERVED and again is None
    assert lane.counters["result_for_unknown_job"] == 2


def test_adapter_submit_failure_is_error_evidence_and_frees_the_slot() -> None:
    class Broken(StubAnalyzer):
        def submit(self, job) -> None:
            raise ConnectionError("connect to http://u:p4ss@127.0.0.1:8080 refused")

    clock = FakeClock()
    lane = SceneLane(Broken(), clock, SETTINGS, id_prefix="job")
    step = _one_frame(clock)
    (evidence,) = lane.on_frame(step.frame, step.stream)
    assert evidence.status is EvidenceStatus.ERROR and "p4ss" not in evidence.reason
    assert lane.in_flight is None and lane.counters["submit_failed"] == 1


def test_enrichment_of_an_older_frame_cannot_overwrite_newer_scene_state() -> None:
    # A dwell rule may ask about the entry frame long after it: the answer is on
    # time and within TTL, but it is older than the scene state already shown.
    clock = FakeClock()
    analyzer = StubAnalyzer()
    lane = SceneLane(analyzer, clock, SETTINGS, id_prefix="job")
    scene = CurrentScene()
    timeline = TimelineBuilder().connect(0).frames_until(4500).build()
    for step in ReplayDriver(timeline, clock=clock).steps():  # lazily: the clock moves per step
        if step.frame is None:
            continue
        if step.frame.frame_seq == 15:
            entry = step  # the 990 ms frame
        lane.on_frame(step.frame, step.stream)
        if lane.in_flight is not None:  # periodic jobs answer at once
            job = lane.in_flight
            evidence = lane.complete(job.job_id, WorkerOutcome.completed(job.job_id, json.dumps(CALM)))
            scene.offer(evidence, step.stream, step.now)
    live, now = step.stream, step.now
    assert [job.frame.frame_seq for job in analyzer.jobs] == [0, frame_at(4000) // 66]

    lane.request_enrichment(entry.frame, "inc-dwell")
    assert lane.poll(live) == [] and lane.in_flight.frame == entry.frame
    job = lane.in_flight
    late_fire = lane.complete(job.job_id, WorkerOutcome.completed(job.job_id, json.dumps(FIRE)))
    routing = scene.offer(late_fire, live, now)
    assert routing.updates_current and routing.annotates == "inc-dwell"  # current, yet older:
    view = scene.view(live, now)
    assert view.evidence.source.frame_seq == analyzer.jobs[1].frame.frame_seq
    assert not view.report.fire_or_smoke


def test_an_outage_clears_the_current_scene_and_it_never_revives() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(500)
        .result(500, 1, CALM)
        .frames_until(1000)
        .disconnect(1000)
        .tick(1500)
        .connect(2000)
        .frames_until(3000)
        .build()
    )
    run = run_scene(timeline)
    assert run.view_at(999).report is not None
    outage = run.view_at(1500)
    assert outage.evidence is None and outage.reason is Applicability.NO_LIVE_STREAM
    after = [view for ms, view in run.views if ms >= 2000]
    assert after and all(view.evidence is None for view in after)
