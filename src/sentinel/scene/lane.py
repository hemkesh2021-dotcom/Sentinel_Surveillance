"""The scene lane: scheduled scene analysis independent of people (guide chapters 9, 12, 27).

Periodic scene checks run on fresh frames whether or not anyone is present.
Incidents may request enrichment of their own frame; that request takes the
next free slot and replaces any enrichment request still waiting. One job is
in flight at a time.

Every job ends in exactly one piece of evidence about its source frame:
observed (a valid report), timeout, error, or unavailable (not started because
its deadline passed while it waited, or was replaced by a newer enrichment
request; kind ``scene.job_skipped``, which is never scene state). A result that arrives after its job's
deadline is late: its evidence is valid only up to the deadline, so it can
annotate the job's incident but never change current scene state. The lane
never reuses a previous verdict.

Not thread-safe: the runtime delivers worker outcomes on its own loop thread.
"""

from __future__ import annotations

import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Protocol

from ..config import SceneConfig
from pydantic import JsonValue

from ..contracts import (
    Applicability,
    Evidence,
    EvidenceStatus,
    FrameRef,
    StreamIdentity,
    stream_relation,
)
from ..jobs import AnalysisJob, JobPurpose, OutcomeKind, WorkerOutcome
from ..media.clock import Clock
from ..redaction import redact_line
from .report import SceneReportError, parse_scene_report
from .state import SCENE_STATE_KIND

EVIDENCE_KIND = SCENE_STATE_KIND  # scene state: a report, or why there is none for that frame
SKIPPED_KIND = "scene.job_skipped"  # a job that never ran: history for its incident only
PRODUCER = "scene-vlm"
TIMED_OUT_JOBS_KEPT = 8  # late results for older timed-out jobs are dropped and counted


class SceneAnalyzer(Protocol):
    """Worker adapter (guide chapter 27). ``submit`` must not block on inference."""

    @property
    def revision(self) -> str: ...

    def submit(self, job: AnalysisJob) -> None: ...

    def cancel(self, job_id: str) -> None: ...


@dataclass(frozen=True)
class SceneLaneSettings:
    interval_ns: int  # between periodic submissions
    timeout_ns: int  # from the source frame's ingest to the job deadline
    ttl_ns: int  # how long an on-time report stays current, from its frame's ingest

    @classmethod
    def from_config(cls, config: SceneConfig) -> SceneLaneSettings:
        return cls(
            interval_ns=config.interval_ns,
            timeout_ns=config.job_timeout_ns,
            ttl_ns=config.evidence_ttl_ns,
        )


class SceneLane:
    def __init__(
        self,
        analyzer: SceneAnalyzer,
        clock: Clock,
        settings: SceneLaneSettings,
        *,
        id_prefix: str | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._clock = clock
        self._settings = settings
        self._prefix = id_prefix or f"scene-{uuid.uuid4().hex[:12]}"
        self._submitted = 0
        self._in_flight: AnalysisJob | None = None
        self._enrichment: tuple[FrameRef, str] | None = None
        self._last_periodic_ns: int | None = None
        self._timed_out: OrderedDict[str, AnalysisJob] = OrderedDict()
        self.counters: Counter[str] = Counter()

    @property
    def in_flight(self) -> AnalysisJob | None:
        return self._in_flight

    def on_frame(self, frame: FrameRef, live: StreamIdentity | None) -> list[Evidence]:
        """Call for every new frame of the live stream, whether or not anyone is in it."""
        return self._step(frame, live)

    def poll(self, live: StreamIdentity | None) -> list[Evidence]:
        """Call regularly, also while no frames arrive, so deadlines are enforced."""
        return self._step(None, live)

    def request_enrichment(self, frame: FrameRef, incident_id: str) -> list[Evidence]:
        """Queue enrichment of ``frame`` for ``incident_id``, replacing a waiting request.

        A replaced request is skipped, and says so in evidence for its own incident.
        """
        out: list[Evidence] = []
        if self._enrichment is not None:
            self.counters["enrichment_replaced"] += 1
            evicted = self._job(JobPurpose.ENRICHMENT, *self._enrichment)
            reason = "skipped: replaced by a newer enrichment request"
            out.append(self._skipped(evicted, reason))
        self._enrichment = (frame, incident_id)
        return out

    def complete(self, job_id: str, outcome: WorkerOutcome) -> Evidence | None:
        """Deliver ``outcome`` for the job the worker was running (``job_id``)."""
        now_ns = self._clock.monotonic_ns()
        if self._in_flight is not None and job_id == self._in_flight.job_id:
            job, on_time = self._in_flight, now_ns < self._in_flight.deadline_mono_ns
            self._in_flight = None
        elif job_id in self._timed_out:
            job, on_time = self._timed_out.pop(job_id), False
        else:
            self.counters["result_for_unknown_job"] += 1
            return None
        if not on_time:
            self.counters["late_result"] += 1
        if outcome.job_id != job.job_id:
            return self._evidence(
                job, EvidenceStatus.ERROR, "response names another job", on_time=on_time
            )
        if outcome.kind is OutcomeKind.TIMEOUT:
            reason = outcome.detail or "worker timeout"
            return self._evidence(job, EvidenceStatus.TIMEOUT, reason, on_time=on_time)
        if outcome.kind is OutcomeKind.ERROR:
            reason = outcome.detail or "worker error"
            return self._evidence(job, EvidenceStatus.ERROR, reason, on_time=on_time)
        assert outcome.text is not None
        try:
            report = parse_scene_report(outcome.text)
        except SceneReportError as exc:
            return self._evidence(job, EvidenceStatus.ERROR, str(exc), on_time=on_time)
        reason = "scene report" if on_time else "scene report arrived after the job deadline"
        return self._evidence(
            job,
            EvidenceStatus.OBSERVED,
            reason,
            on_time=on_time,
            value=report.model_dump(mode="json"),
        )

    def _step(self, frame: FrameRef | None, live: StreamIdentity | None) -> list[Evidence]:
        out: list[Evidence] = []
        now_ns = self._clock.monotonic_ns()
        job = self._in_flight
        if job is not None and now_ns >= job.deadline_mono_ns:
            self._in_flight = None
            self._remember_timed_out(job)
            self._cancel(job)
            reason = "no result by the deadline"
            out.append(self._evidence(job, EvidenceStatus.TIMEOUT, reason, on_time=True))
        if self._in_flight is None and self._enrichment is not None:
            source, incident_id = self._enrichment
            self._enrichment = None
            if now_ns >= source.ingest_mono_ns + self._settings.timeout_ns:
                self.counters["enrichment_skipped"] += 1
                job = self._job(JobPurpose.ENRICHMENT, source, incident_id)
                out.append(self._skipped(job, "skipped: deadline passed while waiting"))
            else:
                out.extend(self._submit(self._job(JobPurpose.ENRICHMENT, source, incident_id)))
        if (
            self._in_flight is None
            and frame is not None
            and stream_relation(frame.stream, live) is Applicability.CURRENT
            and (
                self._last_periodic_ns is None
                or now_ns - self._last_periodic_ns >= self._settings.interval_ns
            )
        ):
            self._last_periodic_ns = now_ns
            out.extend(self._submit(self._job(JobPurpose.PERIODIC, frame, None)))
        return out

    def _job(self, purpose: JobPurpose, frame: FrameRef, incident_id: str | None) -> AnalysisJob:
        self._submitted += 1
        return AnalysisJob(
            job_id=f"{self._prefix}-{self._submitted}",
            purpose=purpose,
            frame=frame,
            incident_id=incident_id,
            deadline_mono_ns=frame.ingest_mono_ns + self._settings.timeout_ns,
        )

    def _submit(self, job: AnalysisJob) -> list[Evidence]:
        try:
            self._analyzer.submit(job)
        except Exception as exc:  # adapter failure is an observation, not a crash
            self.counters["submit_failed"] += 1
            detail = redact_line(f"submit failed: {type(exc).__name__}: {exc}")
            return [self._evidence(job, EvidenceStatus.ERROR, detail, on_time=True)]
        self.counters[f"submitted_{job.purpose.value}"] += 1
        self._in_flight = job
        return []

    def _cancel(self, job: AnalysisJob) -> None:
        try:
            self._analyzer.cancel(job.job_id)
        except Exception:  # cancellation is best effort; the timeout evidence stands
            self.counters["cancel_failed"] += 1

    def _remember_timed_out(self, job: AnalysisJob) -> None:
        self._timed_out[job.job_id] = job
        while len(self._timed_out) > TIMED_OUT_JOBS_KEPT:
            self._timed_out.popitem(last=False)
            self.counters["timed_out_job_forgotten"] += 1

    def _skipped(self, job: AnalysisJob, reason: str) -> Evidence:
        return self._evidence(
            job, EvidenceStatus.UNAVAILABLE, reason, on_time=False, kind=SKIPPED_KIND
        )

    def _evidence(
        self,
        job: AnalysisJob,
        status: EvidenceStatus,
        reason: str,
        *,
        on_time: bool,
        value: JsonValue = None,
        kind: str = EVIDENCE_KIND,
    ) -> Evidence:
        # A late outcome is only valid up to its deadline, which has passed: it can
        # annotate its incident but is already expired for current decisions.
        ttl_ns = self._settings.ttl_ns if on_time else self._settings.timeout_ns
        frame = job.frame
        return Evidence.observed_on(
            frame,
            evidence_id=f"{job.job_id}.{status.value}" + ("" if on_time else ".late"),
            kind=kind,
            status=status,
            producer=PRODUCER,
            producer_revision=self._analyzer.revision,
            ttl_ns=ttl_ns,
            correlation_group=f"{PRODUCER}:{frame.run_id}:{frame.stream_epoch}",
            reason=redact_line(reason, 500),
            value=value,
            incident_id=job.incident_id,
        )
