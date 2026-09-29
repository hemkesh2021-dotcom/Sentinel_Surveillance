from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import timedelta

import pytest
from pydantic import ValidationError

from sentinel.contracts import Applicability, ConfidenceKind, Evidence, EvidenceStatus, FrameRef
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.media.frames import FrameStamper

NextFrame = Callable[..., FrameRef]
_evidence_ids = itertools.count(1)


def vlm_evidence(frame: FrameRef, **overrides: object) -> Evidence:
    """A scene-description result about ``frame``, as a VLM worker would report it."""
    fields: dict[str, object] = {
        "evidence_id": f"ev-{next(_evidence_ids)}",
        "kind": "scene.description",
        "status": EvidenceStatus.OBSERVED,
        "value": {"summary": "one person near the door"},
        "producer": "vlm.stub",
        "producer_revision": "test",
        "ttl_ns": 10 * NS_PER_SECOND,
        "correlation_group": f"scene:{frame.camera_id}",
        "reason": "stub result",
    }
    fields.update(overrides)
    return Evidence.observed_on(frame, **fields)


def test_result_completing_after_its_ttl_is_already_expired(
    next_frame: NextFrame, stamper: FrameStamper, clock: FakeClock
) -> None:
    # Audit H3: v1 stamped AI results with time.time() on completion, so an
    # answer about an old frame looked fresh for another interval on arrival.
    frame = next_frame()
    clock.advance(12.0)  # queueing plus inference took 12 s
    result = vlm_evidence(frame, ttl_ns=10 * NS_PER_SECOND)
    assert result.applicability(stamper.current_stream, clock.mono()) is Applicability.EXPIRED


def test_evidence_applies_for_exactly_its_ttl_from_the_source_frame(
    next_frame: NextFrame, stamper: FrameStamper, clock: FakeClock
) -> None:
    result = vlm_evidence(next_frame(), ttl_ns=10 * NS_PER_SECOND)
    live = stamper.current_stream
    clock.advance(ns=10 * NS_PER_SECOND - 1)
    assert result.applicability(live, clock.mono()) is Applicability.CURRENT
    clock.advance(ns=1)
    assert result.applicability(live, clock.mono()) is Applicability.EXPIRED


def test_wall_clock_steps_neither_extend_nor_shorten_validity(
    next_frame: NextFrame, stamper: FrameStamper, clock: FakeClock
) -> None:
    result = vlm_evidence(next_frame(), ttl_ns=10 * NS_PER_SECOND)
    live = stamper.current_stream

    clock.step_utc(timedelta(days=1))  # a forward jump must not expire it early
    assert result.applicability(live, clock.mono()) is Applicability.CURRENT

    clock.advance(10.0)
    clock.step_utc(timedelta(days=-2))  # a backward jump must not revive it
    assert result.applicability(live, clock.mono()) is Applicability.EXPIRED


def test_outage_and_reconnect_end_evidence_before_its_ttl(
    next_frame: NextFrame, stamper: FrameStamper, clock: FakeClock
) -> None:
    result = vlm_evidence(next_frame(), ttl_ns=60 * NS_PER_SECOND)
    clock.advance(1.0)
    stamper.disconnect()
    assert (
        result.applicability(stamper.current_stream, clock.mono())
        is Applicability.NO_LIVE_STREAM
    )
    clock.advance(1.0)
    live = stamper.connect()
    assert result.applicability(live, clock.mono()) is Applicability.SUPERSEDED_EPOCH


def test_evidence_from_a_previous_boot_is_never_current(
    next_frame: NextFrame, clock: FakeClock
) -> None:
    frame = next_frame()
    result = vlm_evidence(frame, ttl_ns=60 * NS_PER_SECOND)
    # The new boot's counter happens to equal the old reading: subtracting would give age 0.
    clock.reboot("fake-boot-2", mono_ns=frame.ingest_mono_ns)
    live = FrameStamper("cam-1", clock).connect()
    assert result.applicability(live, clock.mono()) is Applicability.OTHER_BOOT


def test_late_result_may_annotate_only_the_incident_it_was_requested_for(
    next_frame: NextFrame, stamper: FrameStamper, clock: FakeClock
) -> None:
    # Guide chapter 2: a delayed result from an old event cannot modify a new incident.
    frame = next_frame()
    clock.advance(30.0)
    late = vlm_evidence(frame, incident_id="inc-a")
    assert late.applicability(stamper.current_stream, clock.mono()) is Applicability.EXPIRED
    assert late.may_annotate("inc-a")
    assert not late.may_annotate("inc-b")
    assert not vlm_evidence(frame).may_annotate("inc-a")


def test_evidence_newer_than_the_reference_point_is_not_current(
    next_frame: NextFrame, stamper: FrameStamper, clock: FakeClock
) -> None:
    # e.g. a replay that evaluates before advancing its clock to the frame time
    reference = clock.mono()
    result = vlm_evidence(next_frame())
    assert result.applicability(stamper.current_stream, reference) is Applicability.FUTURE


@pytest.mark.parametrize(
    "status",
    [
        EvidenceStatus.UNKNOWN,
        EvidenceStatus.UNAVAILABLE,
        EvidenceStatus.TIMEOUT,
        EvidenceStatus.ERROR,
    ],
)
def test_failures_are_explicit_and_carry_no_verdict(
    next_frame: NextFrame, status: EvidenceStatus
) -> None:
    # A failure for a new frame must not be answered with an earlier successful verdict.
    frame = next_frame()
    failure = vlm_evidence(frame, status=status, value=None, reason="llama-server timed out")
    assert failure.value is None and failure.confidence is None
    with pytest.raises(ValidationError, match="must not carry a value"):
        vlm_evidence(frame, status=status, value={"threat": "none"})


def test_observed_evidence_requires_a_value(next_frame: NextFrame) -> None:
    with pytest.raises(ValidationError, match="requires a value"):
        vlm_evidence(next_frame(), value=None)


@pytest.mark.parametrize(
    ("confidence", "kind", "valid"),
    [
        (-0.2, ConfidenceKind.SIMILARITY, True),  # cosine similarity may be negative
        (1.2, ConfidenceKind.SIMILARITY, False),
        (1.3, ConfidenceKind.DETECTOR_SCORE, False),
        (0.4, ConfidenceKind.NONE, False),  # a number without a stated meaning
    ],
)
def test_confidence_states_what_it_means(
    next_frame: NextFrame, confidence: float, kind: ConfidenceKind, valid: bool
) -> None:
    frame = next_frame()

    def build() -> Evidence:
        return vlm_evidence(
            frame, kind="face.identity", value="known", confidence=confidence, confidence_kind=kind
        )

    if valid:
        assert build().confidence == confidence
    else:
        with pytest.raises(ValidationError, match="confidence"):
            build()
