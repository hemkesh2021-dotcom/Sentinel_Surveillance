"""Observation contracts (guide chapter 6).

Identity
    A frame is identified by (camera_id, boot_id, run_id, stream_epoch, frame_seq).
    run_id is random per ingest run, stream_epoch counts connections within the
    run and frame_seq counts frames within an epoch. No part of identity comes
    from a clock or from source PTS, which can step or reset. Tracker IDs are
    only unique within a stream epoch.

Time
    Ages are measured on the monotonic clock from the source frame's ingest
    time, never from when a result was computed or delivered, and never across
    boots. UTC fields are for display and audit.

Applicability
    Evidence and track observations may drive current decisions only while their
    stream epoch is the live one and their source age is below their TTL. Late or
    expired evidence may still annotate the incident it was requested for.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    model_validator,
)

from .media.clock import MonoInstant, require_utc

Identifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
]
EvidenceKind = Annotated[
    str,
    StringConstraints(max_length=64, pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$"),
]
UtcDatetime = Annotated[AwareDatetime, AfterValidator(require_utc)]
PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

# Preprocessing may round the resized content size; allow that much overhang.
_PIXEL_TOLERANCE = 1.0


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class SourceTimeQuality(str, Enum):
    """What is known about when the camera captured a frame."""

    NONE = "none"  # no source timestamp: report ingest-to-event latency
    STREAM_RELATIVE = "stream_relative"  # PTS known but not mapped to any clock
    CAPTURE_SYNCED = "capture_synced"  # PTS mapped to capture UTC by a validated sync


class PixelFormat(str, Enum):
    BGR = "BGR"
    RGB = "RGB"
    BGRX = "BGRx"
    RGBA = "RGBA"
    NV12 = "NV12"
    GRAY8 = "GRAY8"


class Applicability(str, Enum):
    """Whether an observation may drive a current decision, and if not, why."""

    CURRENT = "current"
    EXPIRED = "expired"  # source age reached its TTL
    SUPERSEDED_EPOCH = "superseded_epoch"  # the stream reconnected after the source frame
    NO_LIVE_STREAM = "no_live_stream"  # stream down, stale or offline: nothing is current
    OTHER_RUN = "other_run"  # another ingest run in this boot: its epochs are not comparable
    OTHER_BOOT = "other_boot"  # monotonic time from another boot is not comparable
    OTHER_CAMERA = "other_camera"
    FUTURE = "future"  # newer than the reference point: clock or replay misuse


class StreamIdentity(Contract):
    """One continuous connection to one camera, within one ingest run and boot."""

    camera_id: Identifier
    boot_id: Identifier
    run_id: Identifier
    stream_epoch: NonNegativeInt


class FrameKey(Contract):
    """Unique identity of one ingested frame."""

    camera_id: Identifier
    boot_id: Identifier
    run_id: Identifier
    stream_epoch: NonNegativeInt
    frame_seq: NonNegativeInt

    @property
    def stream(self) -> StreamIdentity:
        return StreamIdentity(
            camera_id=self.camera_id,
            boot_id=self.boot_id,
            run_id=self.run_id,
            stream_epoch=self.stream_epoch,
        )


class ResizeTransform(Contract):
    """Maps native-image pixels into processed-image pixels: processed = native * scale + pad.

    Record the transform that preprocessing actually applied, including any rounding.
    """

    output_width: PositiveInt
    output_height: PositiveInt
    scale_x: PositiveFinite
    scale_y: PositiveFinite
    pad_x: NonNegativeFinite = 0.0
    pad_y: NonNegativeFinite = 0.0

    @classmethod
    def identity(cls, width: int, height: int) -> ResizeTransform:
        return cls(output_width=width, output_height=height, scale_x=1.0, scale_y=1.0)

    @classmethod
    def letterbox(
        cls, native_width: int, native_height: int, output_width: int, output_height: int
    ) -> ResizeTransform:
        """Aspect-preserving resize centred in the output with padding (unrounded)."""
        scale = min(output_width / native_width, output_height / native_height)
        return cls(
            output_width=output_width,
            output_height=output_height,
            scale_x=scale,
            scale_y=scale,
            pad_x=(output_width - native_width * scale) / 2,
            pad_y=(output_height - native_height * scale) / 2,
        )

    def to_processed(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale_x + self.pad_x, y * self.scale_y + self.pad_y

    def to_native(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.pad_x) / self.scale_x, (y - self.pad_y) / self.scale_y


class FrameRef(Contract):
    """One ingested frame: identity, timing and geometry.

    Fields follow guide chapter 6 plus run_id. The optional buffer reference is
    deferred to the buffer slice (V2-05/V2-09); adding it is backward compatible.
    """

    camera_id: Identifier
    boot_id: Identifier
    run_id: Identifier
    stream_epoch: NonNegativeInt
    frame_seq: NonNegativeInt
    source_pts: int | None = None
    source_time_quality: SourceTimeQuality = SourceTimeQuality.NONE
    ingest_mono_ns: NonNegativeInt
    ingest_utc: UtcDatetime
    native_width: PositiveInt
    native_height: PositiveInt
    resize: ResizeTransform
    pixel_format: PixelFormat

    @model_validator(mode="after")
    def _consistent(self) -> FrameRef:
        if self.source_pts is None and self.source_time_quality is not SourceTimeQuality.NONE:
            raise ValueError("source_time_quality other than 'none' requires source_pts")
        right, bottom = self.resize.to_processed(self.native_width, self.native_height)
        if (
            right > self.resize.output_width + _PIXEL_TOLERANCE
            or bottom > self.resize.output_height + _PIXEL_TOLERANCE
        ):
            raise ValueError("resize maps the native image outside the processed image")
        return self

    @property
    def key(self) -> FrameKey:
        return FrameKey(
            camera_id=self.camera_id,
            boot_id=self.boot_id,
            run_id=self.run_id,
            stream_epoch=self.stream_epoch,
            frame_seq=self.frame_seq,
        )

    @property
    def stream(self) -> StreamIdentity:
        return self.key.stream

    @property
    def ingest_instant(self) -> MonoInstant:
        return MonoInstant(self.boot_id, self.ingest_mono_ns)


def stream_relation(source: StreamIdentity, live: StreamIdentity | None) -> Applicability:
    """CURRENT if ``source`` is the live stream, otherwise the reason it is not.

    Pass ``live=None`` while the stream is not live (disconnected, stale or offline).
    """
    if live is None:
        return Applicability.NO_LIVE_STREAM
    if source.camera_id != live.camera_id:
        return Applicability.OTHER_CAMERA
    if source.boot_id != live.boot_id:
        return Applicability.OTHER_BOOT
    if source.run_id != live.run_id:
        return Applicability.OTHER_RUN
    if source.stream_epoch < live.stream_epoch:
        return Applicability.SUPERSEDED_EPOCH
    if source.stream_epoch > live.stream_epoch:
        return Applicability.FUTURE
    return Applicability.CURRENT


def _assess(
    source: StreamIdentity,
    measured_at: MonoInstant,
    ttl_ns: int,
    live: StreamIdentity | None,
    now: MonoInstant,
) -> Applicability:
    if ttl_ns <= 0:
        raise ValueError("ttl_ns must be positive")
    if live is not None and now.boot_id != live.boot_id:
        raise ValueError("'now' must be read in the live stream's boot")
    relation = stream_relation(source, live)
    if relation is not Applicability.CURRENT:
        return relation
    age_ns = now.ns_since(measured_at)
    if age_ns < 0:
        return Applicability.FUTURE
    if age_ns >= ttl_ns:
        return Applicability.EXPIRED
    return Applicability.CURRENT


class NormalizedBox(Contract):
    """Axis-aligned box in native-image coordinates, normalized to [0, 1]."""

    x1: UnitInterval
    y1: UnitInterval
    x2: UnitInterval
    y2: UnitInterval

    @model_validator(mode="after")
    def _ordered(self) -> NormalizedBox:
        if not (self.x1 < self.x2 and self.y1 < self.y2):
            raise ValueError("box must have x1 < x2 and y1 < y2")
        return self


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"


class TrackKey(Contract):
    """A tracker ID is only unique within one stream epoch."""

    camera_id: Identifier
    boot_id: Identifier
    run_id: Identifier
    stream_epoch: NonNegativeInt
    track_id: NonNegativeInt


class TrackObservation(Contract):
    """A tracked person on one frame: a new detection, or a labelled prediction."""

    frame: FrameKey
    observed_mono_ns: NonNegativeInt
    observed_utc: UtcDatetime
    track_id: NonNegativeInt
    box: NormalizedBox
    status: TrackStatus
    predicted: bool
    detector_confidence: UnitInterval | None
    last_measured_mono_ns: NonNegativeInt

    @model_validator(mode="after")
    def _measurement_consistent(self) -> TrackObservation:
        if self.predicted:
            if self.detector_confidence is not None:
                raise ValueError("a predicted observation has no detector confidence")
            if self.last_measured_mono_ns > self.observed_mono_ns:
                raise ValueError("last measurement cannot be later than the observation")
        else:
            if self.detector_confidence is None:
                raise ValueError("a detection requires detector_confidence")
            if self.last_measured_mono_ns != self.observed_mono_ns:
                raise ValueError("a detection is itself the last measurement")
        return self

    @classmethod
    def detected(
        cls,
        frame: FrameRef,
        *,
        track_id: int,
        box: NormalizedBox,
        confidence: float,
        status: TrackStatus,
    ) -> TrackObservation:
        return cls(
            frame=frame.key,
            observed_mono_ns=frame.ingest_mono_ns,
            observed_utc=frame.ingest_utc,
            track_id=track_id,
            box=box,
            status=status,
            predicted=False,
            detector_confidence=confidence,
            last_measured_mono_ns=frame.ingest_mono_ns,
        )

    def predicted_on(self, frame: FrameRef, *, box: NormalizedBox) -> TrackObservation:
        """A tracker prediction for a later frame; it does not refresh the measurement age."""
        if frame.stream != self.frame.stream:
            raise ValueError(
                f"cannot continue track {self.track_id} across stream epochs, boots or cameras"
            )
        if frame.frame_seq <= self.frame.frame_seq:
            raise ValueError("a prediction must be for a later frame")
        return TrackObservation(
            frame=frame.key,
            observed_mono_ns=frame.ingest_mono_ns,
            observed_utc=frame.ingest_utc,
            track_id=self.track_id,
            box=box,
            status=self.status,
            predicted=True,
            detector_confidence=None,
            last_measured_mono_ns=self.last_measured_mono_ns,
        )

    @property
    def key(self) -> TrackKey:
        return TrackKey(
            camera_id=self.frame.camera_id,
            boot_id=self.frame.boot_id,
            run_id=self.frame.run_id,
            stream_epoch=self.frame.stream_epoch,
            track_id=self.track_id,
        )

    def applicability(
        self, live_stream: StreamIdentity | None, now: MonoInstant, *, expiry_ns: int
    ) -> Applicability:
        """Tracks expire ``expiry_ns`` after their last real measurement."""
        last_measured = MonoInstant(self.frame.boot_id, self.last_measured_mono_ns)
        return _assess(self.frame.stream, last_measured, expiry_ns, live_stream, now)


class EvidenceStatus(str, Enum):
    """Outcome of producing evidence. Failures are explicit observations, never old verdicts."""

    OBSERVED = "observed"  # the producer returned a valid value
    UNKNOWN = "unknown"  # it ran but could not determine a value, e.g. no usable face
    UNAVAILABLE = "unavailable"  # disabled, not admitted or not loaded
    TIMEOUT = "timeout"
    ERROR = "error"  # failed, or returned output that failed validation


class ConfidenceKind(str, Enum):
    """What a confidence number means. A score is not a probability unless calibrated."""

    NONE = "none"
    DETECTOR_SCORE = "detector_score"
    SIMILARITY = "similarity"
    CALIBRATED_PROBABILITY = "calibrated_probability"


_CONFIDENCE_RANGES = {
    ConfidenceKind.DETECTOR_SCORE: (0.0, 1.0),
    ConfidenceKind.SIMILARITY: (-1.0, 1.0),
    ConfidenceKind.CALIBRATED_PROBABILITY: (0.0, 1.0),
}


class Evidence(Contract):
    """One versioned piece of evidence about a source frame (and optionally a track)."""

    contract_version: Literal[1] = 1
    evidence_id: Identifier
    source: FrameKey
    source_track_id: NonNegativeInt | None = None
    incident_id: Identifier | None = None
    kind: EvidenceKind
    status: EvidenceStatus
    value: JsonValue = None
    confidence: FiniteFloat | None = None
    confidence_kind: ConfidenceKind = ConfidenceKind.NONE
    producer: Identifier
    producer_revision: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    observed_mono_ns: NonNegativeInt
    observed_utc: UtcDatetime
    ttl_ns: PositiveInt
    correlation_group: Identifier
    reason: Annotated[str, StringConstraints(min_length=1, max_length=500)]

    @model_validator(mode="after")
    def _consistent(self) -> Evidence:
        if self.status is EvidenceStatus.OBSERVED:
            if self.value is None:
                raise ValueError("observed evidence requires a value")
        elif self.value is not None or self.confidence is not None:
            raise ValueError(f"{self.status.value} evidence must not carry a value or confidence")
        if (self.confidence is None) != (self.confidence_kind is ConfidenceKind.NONE):
            raise ValueError("confidence and a confidence_kind other than 'none' go together")
        if self.confidence is not None:
            low, high = _CONFIDENCE_RANGES[self.confidence_kind]
            if not low <= self.confidence <= high:
                raise ValueError(
                    f"{self.confidence_kind.value} confidence must be within [{low}, {high}]"
                )
        return self

    @classmethod
    def observed_on(
        cls,
        frame: FrameRef,
        *,
        evidence_id: str,
        kind: str,
        status: EvidenceStatus,
        producer: str,
        producer_revision: str,
        ttl_ns: int,
        correlation_group: str,
        reason: str,
        value: JsonValue = None,
        confidence: float | None = None,
        confidence_kind: ConfidenceKind = ConfidenceKind.NONE,
        track_id: int | None = None,
        incident_id: str | None = None,
    ) -> Evidence:
        """Evidence about ``frame``. Its age counts from the frame's ingest time, so a
        result that completes late is already old; completion time is not an input."""
        return cls(
            evidence_id=evidence_id,
            source=frame.key,
            source_track_id=track_id,
            incident_id=incident_id,
            kind=kind,
            status=status,
            value=value,
            confidence=confidence,
            confidence_kind=confidence_kind,
            producer=producer,
            producer_revision=producer_revision,
            observed_mono_ns=frame.ingest_mono_ns,
            observed_utc=frame.ingest_utc,
            ttl_ns=ttl_ns,
            correlation_group=correlation_group,
            reason=reason,
        )

    @property
    def observed_at(self) -> MonoInstant:
        return MonoInstant(self.source.boot_id, self.observed_mono_ns)

    def applicability(self, live_stream: StreamIdentity | None, now: MonoInstant) -> Applicability:
        """May this evidence drive a current decision? Pass None while the stream is not live."""
        return _assess(self.source.stream, self.observed_at, self.ttl_ns, live_stream, now)

    def may_annotate(self, incident_id: str) -> bool:
        """Late or expired evidence may enrich the incident it was requested for, and only that."""
        return self.incident_id is not None and self.incident_id == incident_id
