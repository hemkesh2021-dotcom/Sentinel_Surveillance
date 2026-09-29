"""Current tracks (guide chapters 6 and 8).

The table keeps the latest observation of each track and reports only tracks
that are current: from the live stream epoch and measured within the expiry
window. A person who leaves disappears one expiry after their last real
detection, even if frames stop or the tracker keeps predicting. Frames that
are not new for this consumer (a frozen latest-frame buffer, out-of-order or
from a previous epoch) are ignored, so a stalled camera cannot keep anyone
"present".
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ..contracts import Applicability, FrameKey, FrameRef, StreamIdentity, TrackKey, TrackObservation
from ..media.clock import MonoInstant
from ..media.frames import FrameNovelty, frame_novelty

MAX_TRACKS = 64  # guide ch. 28 starting bound; overflow drops the oldest measurements


class TrackTable:
    def __init__(self, expiry_ns: int) -> None:
        if expiry_ns <= 0:
            raise ValueError("expiry_ns must be positive")
        self._expiry_ns = expiry_ns
        self._tracks: dict[TrackKey, TrackObservation] = {}
        self._last_frame: FrameKey | None = None
        self.counters: Counter[str] = Counter()

    @property
    def size(self) -> int:
        return len(self._tracks)

    def update(
        self,
        frame: FrameRef,
        observations: Sequence[TrackObservation],
        live: StreamIdentity | None,
    ) -> FrameNovelty:
        """Record the tracker's output for ``frame``; returns how the frame was classified."""
        novelty = frame_novelty(frame.key, self._last_frame, live)
        if novelty not in (FrameNovelty.NEW, FrameNovelty.NEW_EPOCH):
            self.counters[f"frame_{novelty.value}"] += 1
            return novelty
        if novelty is FrameNovelty.NEW_EPOCH:
            self._tracks.clear()  # track IDs mean nothing across epochs
        self._last_frame = frame.key
        for observation in observations:
            if observation.frame != frame.key:
                raise ValueError("observation belongs to another frame")
            self._tracks[observation.key] = observation
        if len(self._tracks) > MAX_TRACKS:
            self.counters["track_overflow"] += len(self._tracks) - MAX_TRACKS
            by_age = sorted(self._tracks.items(), key=lambda item: item[1].last_measured_mono_ns)
            for key, _ in by_age[: len(self._tracks) - MAX_TRACKS]:
                del self._tracks[key]
        return novelty

    def current(self, live: StreamIdentity | None, now: MonoInstant) -> tuple[TrackObservation, ...]:
        """Current tracks, oldest ID first. Non-current tracks are forgotten."""
        keep: dict[TrackKey, TrackObservation] = {}
        for key, observation in self._tracks.items():
            status = observation.applicability(live, now, expiry_ns=self._expiry_ns)
            if status is Applicability.CURRENT:
                keep[key] = observation
            else:
                self.counters[f"track_dropped_{status.value}"] += 1
        self._tracks = keep
        return tuple(sorted(keep.values(), key=lambda observation: observation.track_id))
