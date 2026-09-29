"""Replays a timeline through the v2 scene lane and, side by side, the B0 v1 snapshot."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from b0_v1_snapshot import V1SceneResult

from sentinel.contracts import Evidence, FrameRef
from sentinel.jobs import AnalysisJob, OutcomeKind, WorkerOutcome
from sentinel.replay import IncidentEvent, ReplayDriver, ResultEvent, Timeline
from sentinel.scene.lane import SceneLane, SceneLaneSettings
from sentinel.scene.state import CurrentScene, Routing, SceneView

SETTINGS = SceneLaneSettings(
    interval_ns=4_000_000_000, timeout_ns=8_000_000_000, ttl_ns=10_000_000_000
)
# What a failing worker reports; it embeds a credential that must never surface.
ERROR_DETAIL = "HTTP 500 from http://admin:hunter2@127.0.0.1:8080/v1/chat/completions"


class StubAnalyzer:
    revision = "stub-vlm-1"

    def __init__(self) -> None:
        self.jobs: list[AnalysisJob] = []
        self.cancelled: list[str] = []

    def submit(self, job: AnalysisJob) -> None:
        self.jobs.append(job)

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)


@dataclass
class SceneRun:
    analyzer: StubAnalyzer
    lane: SceneLane
    evidence: list[tuple[int, Evidence, Routing]] = field(default_factory=list)
    views: list[tuple[int, SceneView]] = field(default_factory=list)
    annotations: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    v1_fire_alert_ms: list[int] = field(default_factory=list)
    v1_threat: list[tuple[int, str]] = field(default_factory=list)

    def evidence_ids(self) -> list[str]:
        return [evidence.evidence_id for _, evidence, _ in self.evidence]

    def view_at(self, at_ms: int) -> SceneView:
        return [view for ms, view in self.views if ms <= at_ms][-1]


def run_scene(timeline: Timeline, settings: SceneLaneSettings = SETTINGS) -> SceneRun:
    driver = ReplayDriver(timeline)
    origin_ns = driver.clock.monotonic_ns()
    analyzer = StubAnalyzer()
    lane = SceneLane(analyzer, driver.clock, settings, id_prefix="job")
    scene = CurrentScene()
    v1 = V1SceneResult()
    run = SceneRun(analyzer=analyzer, lane=lane)
    last_frame: FrameRef | None = None

    for step in driver.steps():
        at_ms = (step.now.ns - origin_ns) // 1_000_000
        live = step.stream  # connected stream; freshness gating joins in R2
        event = step.event
        new: list[Evidence] = []
        if step.frame is not None:
            last_frame = step.frame
            new += lane.on_frame(step.frame, live)
        elif isinstance(event, IncidentEvent):
            assert last_frame is not None, "an incident needs a frame to enrich"
            new += lane.request_enrichment(last_frame, event.incident_id)
            new += lane.poll(live)
        elif isinstance(event, ResultEvent):
            job = analyzer.jobs[event.job - 1]
            outcome = _outcome(job, event)
            completed = lane.complete(job.job_id, outcome)
            new += [completed] if completed is not None else []
            if event.text is not None:
                v1.worker_returned(as_v1_answer(event.text))
            else:
                v1.worker_raised()
        else:
            new += lane.poll(live)

        for evidence in new:
            routing = scene.offer(evidence, live, step.now)
            run.evidence.append((at_ms, evidence, routing))
            if routing.annotates is not None:
                run.annotations[routing.annotates].append(evidence.evidence_id)
        run.views.append((at_ms, scene.view(live, step.now)))
        if v1.fire_alert():
            run.v1_fire_alert_ms.append(at_ms)
        run.v1_threat.append((at_ms, v1.status_threat()))
    return run


def _outcome(job: AnalysisJob, event: ResultEvent) -> WorkerOutcome:
    claimed = event.claimed_job or job.job_id
    if event.failure == "timeout":
        return WorkerOutcome.failed(claimed, OutcomeKind.TIMEOUT, "worker gave up after 30 s")
    if event.failure == "error":
        return WorkerOutcome.failed(claimed, OutcomeKind.ERROR, ERROR_DETAIL)
    assert event.text is not None, "a result event needs text or a failure"
    return WorkerOutcome.completed(claimed, event.text)


_V1_FIELDS = {
    "persons_visible": "persons",
    "fire_or_smoke": "fire_smoke",
    "observations": "activities",
    "summary": "description",
}


def as_v1_answer(text: str) -> str:
    """The same answer in the field names of v1's prompt (surveillance4_1.py L204-214).

    Values, including wrongly typed ones, are passed through unchanged; text that
    is not a JSON object is passed through as is.
    """
    try:
        data = json.loads(text)
    except ValueError:
        return text
    if not isinstance(data, dict):
        return text
    answer = {_V1_FIELDS.get(key, key): value for key, value in data.items()}
    if "threat" in answer:
        answer["harmful"] = answer["threat"] in ("medium", "high")
    return json.dumps(answer)
