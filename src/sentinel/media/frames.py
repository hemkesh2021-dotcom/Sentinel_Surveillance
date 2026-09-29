"""Frame identity at ingest, and frame novelty for consumers (guide chapters 6 and 32).

Portable: hardware capture adapters call FrameStamper but live in their own
modules, so importing this one never loads a decoder or camera library.
"""

from __future__ import annotations

import uuid
from enum import Enum

from ..contracts import (
    Applicability,
    FrameKey,
    FrameRef,
    PixelFormat,
    ResizeTransform,
    SourceTimeQuality,
    StreamIdentity,
    stream_relation,
)
from .clock import Clock


class FrameStamper:
    """Assigns FrameRef identity for one camera.

    Each stamper is one ingest run with a random run_id; the runtime keeps one
    stamper per camera, so in practice there is one run per process. connect()
    increments the stream epoch within the run and disconnect() ends it;
    frame_seq counts frames within an epoch. Identity never depends on clock
    readings or source PTS, so clock steps, PTS resets and runtime restarts
    cannot recycle it.

    Source PTS is advisory. The camera's first packets can lack timestamps or
    repeat/regress them, so a PTS that does not increase over the previous
    frame's is kept for diagnostics but stamped with quality NONE: consumers
    then fall back to the ingest (receive) time, which every frame has.
    """

    def __init__(self, camera_id: str, clock: Clock) -> None:
        self._camera_id = camera_id
        self._clock = clock
        self._run_id = f"run-{uuid.uuid4().hex}"
        self._epoch = 0
        self._stream: StreamIdentity | None = None
        self._next_seq = 0
        self._last_pts: int | None = None

    @property
    def current_stream(self) -> StreamIdentity | None:
        """The connected stream epoch, or None while disconnected."""
        return self._stream

    def connect(self) -> StreamIdentity:
        """Start the next stream epoch, e.g. after (re)opening the camera stream."""
        self._epoch += 1
        self._stream = StreamIdentity(
            camera_id=self._camera_id,
            boot_id=self._clock.boot_id,
            run_id=self._run_id,
            stream_epoch=self._epoch,
        )
        self._next_seq = 0
        self._last_pts = None
        return self._stream

    def disconnect(self) -> None:
        """End the current epoch; frames stamped before this are no longer live."""
        self._stream = None

    def stamp(
        self,
        *,
        native_width: int,
        native_height: int,
        pixel_format: PixelFormat,
        source_pts: int | None = None,
        source_time_quality: SourceTimeQuality | None = None,
        resize: ResizeTransform | None = None,
    ) -> FrameRef:
        stream = self._stream
        if stream is None:
            raise RuntimeError("cannot stamp a frame while the stream is disconnected")
        now = self._clock.mono()
        if now.boot_id != stream.boot_id:
            raise RuntimeError("the clock's boot changed; connect() a new stream")
        pts_increased = source_pts is not None and (
            self._last_pts is None or source_pts > self._last_pts
        )
        if not pts_increased:
            source_time_quality = SourceTimeQuality.NONE
        elif source_time_quality is None:
            source_time_quality = SourceTimeQuality.STREAM_RELATIVE
        frame = FrameRef(
            camera_id=stream.camera_id,
            boot_id=stream.boot_id,
            run_id=stream.run_id,
            stream_epoch=stream.stream_epoch,
            frame_seq=self._next_seq,
            source_pts=source_pts,
            source_time_quality=source_time_quality,
            ingest_mono_ns=now.ns,
            ingest_utc=self._clock.utc_now(),
            native_width=native_width,
            native_height=native_height,
            resize=resize or ResizeTransform.identity(native_width, native_height),
            pixel_format=pixel_format,
        )
        self._next_seq += 1
        if source_pts is not None:
            self._last_pts = source_pts
        return frame


class FrameNovelty(str, Enum):
    NEW = "new"  # a later frame of the epoch already being processed
    NEW_EPOCH = "new_epoch"  # first frame processed from the live epoch: reset per-epoch state
    DUPLICATE = "duplicate"  # already processed, e.g. a frozen latest-frame buffer
    OUT_OF_ORDER = "out_of_order"  # older than a frame already processed from this epoch
    NOT_LIVE = "not_live"  # not the live stream (epoch, run, boot or camera); see stream_relation()


def frame_novelty(
    candidate: FrameKey, last_processed: FrameKey | None, live_stream: StreamIdentity | None
) -> FrameNovelty:
    """Should a consumer that last processed ``last_processed`` process ``candidate``?"""
    if stream_relation(candidate.stream, live_stream) is not Applicability.CURRENT:
        return FrameNovelty.NOT_LIVE
    if last_processed is None or last_processed.stream != candidate.stream:
        return FrameNovelty.NEW_EPOCH
    if candidate.frame_seq > last_processed.frame_seq:
        return FrameNovelty.NEW
    if candidate.frame_seq == last_processed.frame_seq:
        return FrameNovelty.DUPLICATE
    return FrameNovelty.OUT_OF_ORDER
