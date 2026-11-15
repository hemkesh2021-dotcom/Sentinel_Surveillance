from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel.contracts import FrameRef
from sentinel.jobs import (
    MAX_DETAIL_CHARS,
    MAX_OUTPUT_CHARS,
    AnalysisJob,
    JobPurpose,
    OutcomeKind,
    WorkerOutcome,
)


def test_failure_details_are_redacted_single_line_and_bounded() -> None:
    detail = "POST http://admin:hunter2@10.0.0.2/x failed\n  bot123:ABC-def token=s3cret " + "x" * 500
    outcome = WorkerOutcome.failed("job-1", OutcomeKind.ERROR, detail)
    assert outcome.detail is not None and len(outcome.detail) <= MAX_DETAIL_CHARS
    for secret in ("hunter2", "ABC-def", "s3cret", "\n"):
        assert secret not in outcome.detail
    assert outcome.detail.startswith("POST http://<redacted>@10.0.0.2/x failed bot<redacted> token=<redacted>")


def test_oversized_output_becomes_an_error_and_text_goes_with_completion_only() -> None:
    assert WorkerOutcome.completed("j", "x" * (MAX_OUTPUT_CHARS + 1)).kind is OutcomeKind.ERROR
    with pytest.raises(ValidationError):
        WorkerOutcome(job_id="j", kind=OutcomeKind.TIMEOUT, text="late text")
    with pytest.raises(ValidationError):
        WorkerOutcome(job_id="j", kind=OutcomeKind.COMPLETED)
    with pytest.raises(ValueError):
        WorkerOutcome.failed("j", OutcomeKind.COMPLETED, "no")


def test_job_invariants(next_frame) -> None:
    frame: FrameRef = next_frame()
    deadline = frame.ingest_mono_ns + 1
    AnalysisJob(job_id="j", purpose=JobPurpose.PERIODIC, frame=frame, deadline_mono_ns=deadline)
    with pytest.raises(ValidationError, match="names an incident"):
        AnalysisJob(job_id="j", purpose=JobPurpose.ENRICHMENT, frame=frame, deadline_mono_ns=deadline)
    with pytest.raises(ValidationError, match="names an incident"):
        AnalysisJob(
            job_id="j", purpose=JobPurpose.PERIODIC, frame=frame, incident_id="i", deadline_mono_ns=deadline
        )
    with pytest.raises(ValidationError, match="after the source frame"):
        AnalysisJob(
            job_id="j", purpose=JobPurpose.PERIODIC, frame=frame, deadline_mono_ns=frame.ingest_mono_ns
        )
