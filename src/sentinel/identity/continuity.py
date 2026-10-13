"""Track continuity for identity retention (D97; opt-in, face validation runs only: ``sentinel run --face-continuity``).

A face identity lives on a tracker's track number, and the tracker can keep a number across a detection gap, a jump or
two boxes meeting, exactly where numbers get swapped between people. A track number alone is therefore no evidence
that the same person is still under the box. ContinuityMonitor watches every processed frame's tracks and reports a
**break** for a track whenever that evidence lapses:

- **gap**: the track was missing from at least one processed frame and more than ``max_gap_ns`` passed between two of
  its detections (an occlusion, a lost detection, someone leaving while someone else arrives);
- **jump**: two consecutive detections of the track overlap less than ``min_step_iou`` (intersection over union): the
  box moved further than a person moves between two frames, so it may now cover someone else;
- **overlap**: on one frame its box overlaps another track's by at least ``overlap_iou``: where boxes meet, the tracker
  may swap their numbers, so neither keeps an identity.

The two thresholds cover each other (``overlap_iou`` < ``min_step_iou``): a swap onto a box that overlapped the track
by at least ``overlap_iou`` on the frame before is an overlap break there, and a swap onto any other box is a jump.
Every value here is a PROPOSED starting value, unvalidated on the device. Box overlap is geometry only: a break says
that the evidence of continuity lapsed, never that the box now covers someone else, or whom.

What a break does is IdentityResolver's (``break_track``): the track's votes are dropped, a face result for a frame
from before the break is not counted for it (results arrive about a second late), and the identity has to be earned
again from new matches. Between breaks, votes on a continuously followed track live ``retention_ns``, a bound, instead
of the plain vote expiry: a confirmed person who looks away stays known, marked retained, while the evidence of
continuity holds and never longer than that bound. Contradicting face evidence still ends it at once; ownership,
ambiguity, thresholds and confirmations are unchanged.

Not thread-safe: EdgeCore calls it on its loop thread. Bounded by the current tracks.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from ..contracts import FrameRef, NormalizedBox, TrackKey, TrackObservation

NS_PER_SECOND = 1_000_000_000
MAX_RETENTION_NS = 600 * NS_PER_SECOND  # no configuration may keep votes longer than this


def iou(a: NormalizedBox, b: NormalizedBox) -> float:
    """Intersection over union of two boxes (geometry only)."""
    width = min(a.x2, b.x2) - max(a.x1, b.x1)
    height = min(a.y2, b.y2) - max(a.y1, b.y1)
    inter = width * height if width > 0 and height > 0 else 0.0
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union


class Cause(str, Enum):
    GAP = "gap"
    JUMP = "jump"
    OVERLAP = "overlap"


# The identity reason a broken track shows until new matches earn an identity again (fixed text, never free text).
NOT_CARRIED = {
    Cause.GAP: "not carried over a break in tracking (a detection gap)",
    Cause.JUMP: "not carried over a break in tracking (the box jumped)",
    Cause.OVERLAP: "not carried over a break in tracking (another detection overlapped)",
}


@dataclass(frozen=True)
class ContinuityPolicy:
    """PROPOSED starting values (D97), to be validated on the device; see the module docstring."""

    max_gap_ns: int = NS_PER_SECOND // 2
    overlap_iou: float = 0.3
    min_step_iou: float = 0.4
    retention_ns: int = 120 * NS_PER_SECOND  # how long votes live on a continuously followed track

    def __post_init__(self) -> None:
        if self.max_gap_ns <= 0:
            raise ValueError("max_gap_ns must be positive")
        if not 0 < self.overlap_iou < self.min_step_iou <= 1:
            raise ValueError("thresholds must satisfy 0 < overlap_iou < min_step_iou <= 1")
        if not 0 < self.retention_ns <= MAX_RETENTION_NS:
            raise ValueError(f"retention_ns must be positive and at most {MAX_RETENTION_NS}")

    def numbers(self) -> dict[str, float]:
        """For the status and the startup record: the rule as configured (numbers only)."""
        return {"max_gap_s": self.max_gap_ns / NS_PER_SECOND, "overlap_iou": self.overlap_iou,
                "min_step_iou": self.min_step_iou, "retention_s": self.retention_ns / NS_PER_SECOND}


@dataclass(frozen=True)
class Break:
    key: TrackKey
    at_ns: int  # the ingest time of the frame on which the evidence lapsed
    cause: Cause


@dataclass
class _Followed:
    box: NormalizedBox
    seen_ns: int  # its latest detection's frame time
    missed: int  # processed frames without it since then
    since_ns: int  # when its current unbroken stretch began (its first detection, or its latest break)


class ContinuityMonitor:
    def __init__(self, policy: ContinuityPolicy = ContinuityPolicy()) -> None:
        self._policy = policy
        self._tracks: dict[TrackKey, _Followed] = {}
        self.counters: Counter[str] = Counter()

    @property
    def policy(self) -> ContinuityPolicy:
        return self._policy

    def observe(self, frame: FrameRef, persons: Sequence[TrackObservation]) -> list[Break]:
        """One processed frame and every detection on it (none when nobody was detected or the detector failed: each
        known track then counts the frame as missed). Returns the breaks found on this frame, at most one per track."""
        now = frame.ingest_mono_ns
        policy = self._policy
        causes: dict[TrackKey, Cause] = {}
        detected = [p for p in persons if not p.predicted]  # PersonTracker publishes detections only; never trust more
        for person in detected:
            followed = self._tracks.get(person.key)
            if followed is None:
                self._tracks[person.key] = _Followed(person.box, now, 0, now)
                continue
            if followed.missed and now - followed.seen_ns > policy.max_gap_ns:
                causes[person.key] = Cause.GAP
            elif iou(followed.box, person.box) < policy.min_step_iou:
                causes[person.key] = Cause.JUMP
            followed.box, followed.seen_ns, followed.missed = person.box, now, 0
        for i, a in enumerate(detected):
            for b in detected[i + 1:]:
                if iou(a.box, b.box) >= policy.overlap_iou:
                    causes.setdefault(a.key, Cause.OVERLAP)
                    causes.setdefault(b.key, Cause.OVERLAP)
        seen = {p.key for p in detected}
        for key, followed in self._tracks.items():
            if key not in seen:
                followed.missed += 1
        for key, cause in causes.items():
            self._tracks[key].since_ns = now
            self.counters[f"breaks_{cause.value}"] += 1
        return [Break(key, now, cause) for key, cause in causes.items()]

    def followed_since(self, key: TrackKey) -> int | None:
        """When the track's current unbroken stretch began (frame time), or None if it is not followed."""
        followed = self._tracks.get(key)
        return None if followed is None else followed.since_ns

    def retain_only(self, keys: Iterable[TrackKey]) -> None:
        """Forget tracks that are no longer current."""
        keep = set(keys)
        for key in [k for k in self._tracks if k not in keep]:
            del self._tracks[key]

    @property
    def followed(self) -> int:
        return len(self._tracks)
