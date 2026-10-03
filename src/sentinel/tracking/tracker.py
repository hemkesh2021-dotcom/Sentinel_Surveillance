"""Person tracker boundary (guide ch. 8; V2-09/V2-10 demo form, full acceptance pending).

A TrackerBackend detects and tracks people on one decoded image at a time and
reports tracked boxes in that image's pixels. PersonTracker turns its output
into TrackObservations for EdgeCore and owns the rules a backend cannot know:

- **One tracker state per stream epoch.** A frame from a new epoch, run, boot
  or camera resets the backend first, so a track never continues across a
  reconnect or a geometry change.
- **Each frame at most once, in order.** A frame that is not newer than the
  last processed frame of its epoch, or comes from an older epoch, is skipped
  without inference.
- **Detections only.** Every observation is a real detection on this frame
  (``predicted=False``); a backend's lost or predicted tracks are never
  published. TrackTable expires a track one ``track_expiry`` after its last
  detection.
- **Track IDs are unique within an epoch.** After a mid-epoch backend reset
  (after a failure) the backend restarts its IDs, so published IDs are offset
  past every ID already used in the epoch. Identity votes (keyed by track) can
  therefore never pass from one person to another.
- **Confirmation as v1 does it** (``surveillance4_1.py@2b2d639`` lines
  418-437): a score rises by 1 per detection (capped at ``confirm_detections``
  + 2) and falls by 1 per processed frame without the track. A track is
  CONFIRMED from the first time its score reaches ``confirm_detections`` until
  the score falls to 0. v1's display hold of unseen tracks is not kept. Zone
  rules count only CONFIRMED tracks (D30).
- **Backend output is untrusted.** Non-finite values, confidence outside
  [0, 1], unordered boxes, negative, non-integer or duplicate track IDs fail
  the frame. Boxes are clipped to the image; a box under one pixel wide or high
  after clipping is dropped. At most MAX_TRACKS boxes are kept, highest
  confidence first.
- **Failures are fixed labels** (an exception's class name, never its text).
  The backend is reset. The caller reports the detector unavailable for the
  frame (``EdgeCore.set_detector``), so occupancy reads UNKNOWN, not EMPTY (D18).

Not thread-safe: one consumer thread calls process().
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from ..contracts import FrameKey, FrameRef, NormalizedBox, TrackObservation, TrackStatus
from .tracks import MAX_TRACKS

MIN_BOX_PX = 1.0


@dataclass(frozen=True)
class RawTrack:
    """One tracked person as a backend reports it, in pixels of the image it was given."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class TrackerBackend(Protocol):
    def track(self, image: Any) -> Sequence[RawTrack]:
        """Detect and track people on the next image of the current stream."""

    def reset(self) -> None:
        """Forget every track; the next image starts a new sequence."""


class TrackerError(Exception):
    """A backend problem described by a fixed label."""

    def __init__(self, label: str, error_type: str | None = None) -> None:
        super().__init__(label)
        self.label = label
        self.error_type = error_type


class FrameOutcome(str, Enum):
    PROCESSED = "processed"
    SKIPPED = "skipped"  # not newer than a processed frame; no inference
    FAILED = "failed"  # the detector is unavailable for this frame


@dataclass(frozen=True)
class TrackingResult:
    outcome: FrameOutcome
    persons: tuple[TrackObservation, ...] = ()
    problem: str | None = None  # fixed label when SKIPPED or FAILED
    error_type: str | None = None  # exception class name, when a backend call raised
    backend_ns: int | None = None  # time spent in backend.track() on this frame


class PersonTracker:
    def __init__(
        self,
        backend: TrackerBackend,
        *,
        confirm_detections: int = 2,
        timer: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if confirm_detections < 1:
            raise ValueError("confirm_detections must be at least 1")
        self._backend = backend
        self._confirm = confirm_detections
        self._timer = timer
        self._last: FrameKey | None = None
        self._needs_reset = True  # the backend may hold state from before this tracker
        self._id_offset = 0
        self._max_id = -1  # highest published ID in this epoch
        self._scores: dict[int, int] = {}
        self._confirmed: set[int] = set()
        self.counters: Counter[str] = Counter()

    def process(self, frame: FrameRef, image: Any) -> TrackingResult:
        """Track people on ``image``, the native-resolution picture of ``frame``."""
        skip = self._skip_reason(frame.key)
        if skip is not None:
            self.counters[f"skipped_{skip}"] += 1
            return TrackingResult(FrameOutcome.SKIPPED, problem=skip)
        if self._last is None or self._last.stream != frame.stream:
            self._new_epoch()
        self._last = frame.key
        if self._needs_reset:
            try:
                self._backend.reset()
            except Exception as exc:  # retried on the next frame; never track on stale state
                return self._failed("reset_failed", type(exc).__name__)
            self._needs_reset = False
        shape = getattr(image, "shape", None)
        if shape is not None and tuple(shape[:2]) != (frame.native_height, frame.native_width):
            return self._failed("image_size_mismatch")
        started = self._timer()
        try:
            raw = list(self._backend.track(image))
        except TrackerError as exc:
            return self._failed(exc.label, exc.error_type, self._timer() - started)
        except Exception as exc:  # a broken detector must not stop the runtime
            return self._failed("backend_error", type(exc).__name__, self._timer() - started)
        elapsed = self._timer() - started
        try:
            persons = self._observations(frame, raw)
        except ValueError:
            return self._failed("invalid_output", None, elapsed)
        self.counters["processed"] += 1
        return TrackingResult(FrameOutcome.PROCESSED, persons, backend_ns=elapsed)

    def _skip_reason(self, key: FrameKey) -> str | None:
        last = self._last
        if last is None:
            return None
        if key.stream == last.stream:
            return None if key.frame_seq > last.frame_seq else "not_newer"
        same_run = (key.camera_id, key.boot_id, key.run_id) == (last.camera_id, last.boot_id, last.run_id)
        return "older_epoch" if same_run and key.stream_epoch < last.stream_epoch else None

    def _new_epoch(self) -> None:
        if self._last is not None:
            self.counters["epoch_resets"] += 1
        self._needs_reset = True
        self._id_offset = 0
        self._max_id = -1
        self._scores.clear()
        self._confirmed.clear()

    def _failed(self, label: str, error_type: str | None = None, backend_ns: int | None = None) -> TrackingResult:
        self.counters["failed"] += 1
        self.counters[f"failed_{label}"] += 1
        if not self._needs_reset:
            # Reset after a failure, and publish later IDs past every one used in this epoch.
            self._needs_reset = True
            self._id_offset = self._max_id + 1
            self.counters["failure_resets"] += 1
        return TrackingResult(FrameOutcome.FAILED, problem=label, error_type=error_type, backend_ns=backend_ns)

    def _observations(self, frame: FrameRef, raw: list[RawTrack]) -> tuple[TrackObservation, ...]:
        width, height = frame.native_width, frame.native_height
        ids: set[int] = set()
        kept: list[tuple[float, int, NormalizedBox]] = []
        for track in raw:
            backend_id, confidence = _checked(track)
            if backend_id in ids:
                raise ValueError("duplicate track ID")
            ids.add(backend_id)
            x1, x2 = min(max(track.x1, 0.0), width), min(max(track.x2, 0.0), width)
            y1, y2 = min(max(track.y1, 0.0), height), min(max(track.y2, 0.0), height)
            if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                self.counters["boxes_dropped_small"] += 1
                continue
            box = NormalizedBox(x1=x1 / width, y1=y1 / height, x2=x2 / width, y2=y2 / height)
            kept.append((confidence, self._id_offset + backend_id, box))
        if len(kept) > MAX_TRACKS:
            self.counters["boxes_dropped_overflow"] += len(kept) - MAX_TRACKS
            kept = sorted(kept, key=lambda entry: (-entry[0], entry[1]))[:MAX_TRACKS]
        statuses = self._score({track_id for _, track_id, _ in kept})
        persons = []
        for confidence, track_id, box in sorted(kept, key=lambda entry: entry[1]):
            self._max_id = max(self._max_id, track_id)
            persons.append(
                TrackObservation.detected(
                    frame, track_id=track_id, box=box, confidence=confidence, status=statuses[track_id]
                )
            )
        return tuple(persons)

    def _score(self, seen: set[int]) -> dict[int, TrackStatus]:
        cap = self._confirm + 2
        for track_id in list(self._scores):
            if track_id not in seen:
                self._scores[track_id] -= 1
                if self._scores[track_id] <= 0:
                    del self._scores[track_id]
                    self._confirmed.discard(track_id)
        statuses = {}
        for track_id in seen:
            score = min(self._scores.get(track_id, 0) + 1, cap)
            self._scores[track_id] = score
            if score >= self._confirm:
                self._confirmed.add(track_id)
            statuses[track_id] = TrackStatus.CONFIRMED if track_id in self._confirmed else TrackStatus.TENTATIVE
        return statuses


def _checked(track: RawTrack) -> tuple[int, float]:
    """The track's ID and confidence if every field is well formed; ValueError otherwise."""
    track_id = track.track_id
    if isinstance(track_id, bool) or not isinstance(track_id, int) or track_id < 0:
        raise ValueError("track ID must be a non-negative integer")
    values = (track.x1, track.y1, track.x2, track.y2, track.confidence)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise ValueError("non-finite box or confidence")
    if not 0.0 <= track.confidence <= 1.0:
        raise ValueError("confidence outside [0, 1]")
    if not (track.x1 < track.x2 and track.y1 < track.y2):
        raise ValueError("unordered box")
    return track_id, float(track.confidence)
