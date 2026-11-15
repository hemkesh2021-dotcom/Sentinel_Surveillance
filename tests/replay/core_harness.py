"""Replays a timeline through EdgeCore, with scripted scene-worker answers and the B0 v1 loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from b0_v1_snapshot import V1MainLoop

from sentinel.config import SentinelConfig, parse_config
from sentinel.jobs import AnalysisJob, OutcomeKind, WorkerOutcome
from sentinel.live_state import Capability, LiveState
from sentinel.media.clock import Clock
from sentinel.replay import IncidentEvent, ReplayDriver, ResultEvent, TickEvent, Timeline
from sentinel.rules.scene_hazard import HazardCandidate
from sentinel.runtime import CoreOutput, EdgeCore, RoutedEvidence

CONFIG = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
EMPTY_ROOM = {
    "persons_visible": 0,
    "fire_or_smoke": False,
    "threat": "none",
    "observations": [],
    "uncertainty": "low",
    "summary": "empty room",
}
SMOKE = {**EMPTY_ROOM, "fire_or_smoke": True, "threat": "high", "summary": "smoke, nobody present"}

# answer(job_number_from_1, job) -> (delay_ms, report dict | "timeout" | "error") or None (never answers)
Answer = tuple[int, dict | str]
AnswerFn = Callable[[int, AnalysisJob], Answer | None]


def always(report: dict, delay_ms: int = 1500) -> AnswerFn:
    return lambda number, job: (delay_ms, report)


class TimedAnalyzer:
    revision = "stub-vlm-1"

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self.jobs: list[tuple[AnalysisJob, int]] = []  # (job, submitted monotonic ns)
        self.cancelled: list[str] = []

    def submit(self, job: AnalysisJob) -> None:
        self.jobs.append((job, self._clock.monotonic_ns()))

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)


@dataclass
class CoreRun:
    analyzer: TimedAnalyzer
    core: EdgeCore
    states: list[tuple[int, LiveState]] = field(default_factory=list)
    evidence: list[tuple[int, RoutedEvidence]] = field(default_factory=list)
    candidates: list[tuple[int, HazardCandidate]] = field(default_factory=list)
    v1: V1MainLoop = field(default_factory=V1MainLoop)

    def at(self, at_ms: int) -> LiveState:
        return [state for ms, state in self.states if ms <= at_ms][-1]

    def during(self, start_ms: int, end_ms: int) -> list[LiveState]:
        """States published in [start_ms, end_ms)."""
        return [state for ms, state in self.states if start_ms <= ms < end_ms]

    def job_starts_ms(self, origin_ns: int) -> list[int]:
        return [(submitted - origin_ns) // 1_000_000 for _, submitted in self.analyzer.jobs]


def run_core(
    timeline: Timeline,
    *,
    answer: AnswerFn | None = None,
    config: SentinelConfig = CONFIG,
    detector: Capability = Capability.AVAILABLE,
    frozen_reader: bool = False,
) -> tuple[CoreRun, int]:
    """Returns the run and the monotonic origin (ns) that timeline offsets count from.

    ``frozen_reader`` re-delivers the last frame on every tick, as v1's FrameReader
    does after a stall; v2 must treat those as duplicates.
    """
    driver = ReplayDriver(timeline)
    origin_ns = driver.clock.monotonic_ns()
    analyzer = TimedAnalyzer(driver.clock)
    core = EdgeCore(config, driver.clock, analyzer, detector=detector, scene_id_prefix="job")
    run = CoreRun(analyzer=analyzer, core=core)
    scheduled: dict[int, tuple[int, AnalysisJob, dict | str]] = {}  # job index -> due
    last_step = None

    def record(at_ms: int, output: CoreOutput) -> None:
        run.states.append((at_ms, output.state))
        run.evidence += [(at_ms, routed) for routed in output.evidence]
        run.candidates += [(at_ms, candidate) for candidate in output.candidates]

    for step in driver.steps():
        at_ms = (step.now.ns - origin_ns) // 1_000_000
        connected = step.stream
        # Worker answers that fell due since the previous step arrive first.
        if answer is not None:
            for index, (job, submitted) in enumerate(analyzer.jobs):
                if index not in scheduled:
                    planned = answer(index + 1, job)
                    scheduled[index] = (
                        (submitted + planned[0] * 1_000_000, job, planned[1]) if planned else None
                    )
            for index, due in sorted(scheduled.items()):
                if due is not None and due[0] <= step.now.ns:
                    _, job, reply = due
                    scheduled[index] = None
                    record(at_ms, core.on_scene_outcome(job.job_id, _outcome(job, reply), connected))

        event = step.event
        if step.frame is not None:
            last_step = step
            record(at_ms, core.on_frame(step.frame, step.persons, connected))
        elif isinstance(event, TickEvent) and frozen_reader and last_step is not None:
            record(at_ms, core.on_frame(last_step.frame, last_step.persons, connected))
        elif isinstance(event, IncidentEvent):
            assert last_step is not None
            record(at_ms, core.request_enrichment(last_step.frame, event.incident_id, connected))
        elif isinstance(event, ResultEvent):
            job, _ = analyzer.jobs[event.job - 1]
            reply = event.failure or json.loads(event.text or "null")
            record(at_ms, core.on_scene_outcome(job.job_id, _outcome(job, reply), connected))
        else:
            record(at_ms, core.tick(connected))
        # v1 processes whatever its reader holds: the last frame, even after a stall.
        run.v1.iteration(len(last_step.persons) if last_step is not None else 0)
    return run, origin_ns


def _outcome(job: AnalysisJob, reply: dict | str) -> WorkerOutcome:
    if reply == "timeout":
        return WorkerOutcome.failed(job.job_id, OutcomeKind.TIMEOUT, "worker timeout")
    if reply == "error":
        return WorkerOutcome.failed(job.job_id, OutcomeKind.ERROR, "worker error")
    return WorkerOutcome.completed(job.job_id, json.dumps(reply))
