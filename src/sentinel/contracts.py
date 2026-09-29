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
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
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
UtcDatetime = Annotated[AwareDatetime, AfterValidator(require_utc)]
PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]

# Preprocessing may round the resized content size; allow that much overhang.
_PIXEL_TOLERANCE = 1.0


class _Contract(BaseModel):
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


class StreamIdentity(_Contract):
    """One continuous connection to one camera, within one ingest run and boot."""

    camera_id: Identifier
    boot_id: Identifier
    run_id: Identifier
    stream_epoch: NonNegativeInt


class FrameKey(_Contract):
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


class ResizeTransform(_Contract):
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


class FrameRef(_Contract):
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
