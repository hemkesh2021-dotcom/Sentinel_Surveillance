"""Optional-worker job and outcome contracts (guide chapter 27).

A job names the frame it is about and, for enrichment, the incident it was
requested for. Its deadline counts from the source frame's ingest time, so it
includes queue wait and source age: a job cannot stay useful by waiting.
Outcomes carry the job ID the worker's response claims; the receiver compares
it with the job it was waiting for rather than trusting it.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import StringConstraints, model_validator

from .contracts import Contract, FrameRef, Identifier
from .media.clock import MonoInstant
from .redaction import redact_line

MAX_OUTPUT_CHARS = 8_192  # raw worker text accepted for validation; larger is an error
MAX_DETAIL_CHARS = 200


class JobPurpose(str, Enum):
    PERIODIC = "periodic"  # scene coverage, independent of people
    ENRICHMENT = "enrichment"  # requested for one incident


class AnalysisJob(Contract):
    job_id: Identifier
    purpose: JobPurpose
    frame: FrameRef
    incident_id: Identifier | None = None
    deadline_mono_ns: int

    @model_validator(mode="after")
    def _consistent(self) -> AnalysisJob:
        if (self.purpose is JobPurpose.ENRICHMENT) != (self.incident_id is not None):
            raise ValueError("an enrichment job, and only an enrichment job, names an incident")
        if self.deadline_mono_ns <= self.frame.ingest_mono_ns:
            raise ValueError("the deadline must be after the source frame's ingest time")
        return self

    @property
    def deadline(self) -> MonoInstant:
        return MonoInstant(self.frame.boot_id, self.deadline_mono_ns)


class OutcomeKind(str, Enum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"


class WorkerOutcome(Contract):
    """What a worker delivered. ``text`` is untrusted and validated by the receiver."""

    job_id: str  # as claimed by the response; not trusted
    kind: OutcomeKind
    text: Annotated[str, StringConstraints(max_length=MAX_OUTPUT_CHARS)] | None = None
    detail: Annotated[str, StringConstraints(max_length=MAX_DETAIL_CHARS)] | None = None

    @model_validator(mode="after")
    def _text_iff_completed(self) -> WorkerOutcome:
        if (self.kind is OutcomeKind.COMPLETED) != (self.text is not None):
            raise ValueError("a completed outcome, and only a completed outcome, carries text")
        return self

    @classmethod
    def completed(cls, job_id: str, text: str) -> WorkerOutcome:
        if len(text) > MAX_OUTPUT_CHARS:
            return cls.failed(job_id, OutcomeKind.ERROR, f"output over {MAX_OUTPUT_CHARS} chars")
        return cls(job_id=job_id, kind=OutcomeKind.COMPLETED, text=text)

    @classmethod
    def failed(cls, job_id: str, kind: OutcomeKind, detail: str) -> WorkerOutcome:
        """A failure with a redacted, single-line, bounded detail."""
        if kind is OutcomeKind.COMPLETED:
            raise ValueError("use completed() for a completed outcome")
        return cls(job_id=job_id, kind=kind, detail=redact_line(detail, MAX_DETAIL_CHARS))
