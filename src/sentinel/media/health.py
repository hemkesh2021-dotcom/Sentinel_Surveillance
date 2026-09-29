"""Video freshness: fresh, stale, offline (guide chapters 6 and 22).

Freshness is the age of the newest frame of the connected stream on the
monotonic clock. Only a FRESH stream is live for decisions (unresolved U1,
settled here): while video is stale or offline, or the stream is
disconnected or has not delivered a frame since reconnecting, no evidence is
current, so a stall withdraws current scene and track state instead of
freezing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config import FreshnessConfig
from ..contracts import FrameRef, StreamIdentity
from .clock import Clock, MonoInstant


class VideoState(str, Enum):
    STARTING = "starting"  # no frame yet since monitoring started, within the offline threshold
    FRESH = "fresh"
    STALE = "stale"  # no fresh frame for stale_after_s
    OFFLINE = "offline"  # no fresh frame for offline_after_s


@dataclass(frozen=True)
class VideoFreshness:
    state: VideoState
    last_frame_age_ns: int | None  # None if no frame was ever received in this boot
    live: StreamIdentity | None  # the stream decisions may use: set only when FRESH


class FreshnessMonitor:
    def __init__(self, config: FreshnessConfig, clock: Clock) -> None:
        self._config = config
        self._clock = clock
        self._started = clock.mono()
        self._last_frame: FrameRef | None = None

    def on_frame(self, frame: FrameRef) -> None:
        last = self._last_frame
        if last is None or last.boot_id != frame.boot_id or frame.ingest_mono_ns >= last.ingest_mono_ns:
            self._last_frame = frame

    def assess(self, connected: StreamIdentity | None) -> VideoFreshness:
        now = self._clock.mono()
        last = self._last_frame
        if last is None or last.boot_id != now.boot_id:
            since_start = now.ns_since(self._started) if self._started.boot_id == now.boot_id else None
            if since_start is not None and since_start < self._config.offline_after_ns:
                return VideoFreshness(VideoState.STARTING, None, None)
            return VideoFreshness(VideoState.OFFLINE, None, None)
        age = now.ns_since(MonoInstant(last.boot_id, last.ingest_mono_ns))
        if age >= self._config.offline_after_ns:
            state = VideoState.OFFLINE
        elif age >= self._config.stale_after_ns or connected is None or last.stream != connected:
            state = VideoState.STALE
        else:
            return VideoFreshness(VideoState.FRESH, age, connected)
        return VideoFreshness(state, age, None)
