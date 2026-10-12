"""The full-scale embedding comparison (D87): validation runs only, diagnostic, never a vote.

The question it answers: do faces that only the full-scale detector finds (D86) give embeddings that match the
existing enrollment? On a frame the face worker has just analysed, a person is **eligible** when the full-scale pass
assigned them a face scoring at least FULL_SCALE_COUNTED (the same one-to-one association and geometry as production),
and either the production pass left them without a face (``full_scale_only``, the question itself) or the production
pass assigned them a face too (``both``, a control: the same face through both paths, so the comparison path's
agreement with production can be seen). A person whose production face was ambiguous is not eligible.

**The budget** (predeclared; it bounds the extra work on a worker already about 94 % busy):
- at most one comparison per analysed frame, a ``full_scale_only`` person first, then the lowest track ID;
- a comparison starts at least ``interval_ns`` after the previous one started, and at most ``cap`` start per run;
- a control is attempted only while controls stay at most one in ``control_share`` of the attempts (counting this
  one), so the first comparisons are always ``full_scale_only`` ones;
- none starts while a production job is already waiting (it would run late), while the worker is stopping, when the
  frame is older than ``max_frame_age_ns``, or when the candidate has no landmarks.

Every eligible person gets exactly one disposition, so sampling bias stays visible: ``completed`` (an embedding was
made), ``failed`` (with a fixed label; never a nonmatch), or ``skipped_<reason>``; the runtime adds
``production_not_applied`` when that frame's production result was not applied. The comparison runs in the face
worker after that frame's production result has been published, and the worker yields the next production tick that
arrives meanwhile (face.worker), so production results never run late. Embeddings stay in memory: only the
similarity the runtime computes from them is ever recorded.
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..contracts import FrameRef, TrackKey, TrackObservation
from ..media.clock import NS_PER_SECOND
from .association import AssociationGeometry, FaceObservation, Ownership, associate

FULL_SCALE_COUNTED = 0.9  # runtime.FULL_SCALE_COUNTED (test-tied): the production detector's own threshold
COMPARISON_INTERVAL_S = 4.0
COMPARISON_CAP = 75  # a 300 s run at one every 4 s
CONTROL_SHARE = 4  # at most one control in four attempts
LATENCIES_KEPT = 600


class Kind(str, Enum):
    FULL_SCALE_ONLY = "full_scale_only"
    BOTH = "both"


class Disposition(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    PRODUCTION_NOT_APPLIED = "production_not_applied"  # set by the runtime, never by the worker
    SKIPPED_ONE_PER_RESULT = "skipped_one_per_result"
    SKIPPED_STOPPING = "skipped_stopping"
    SKIPPED_PRODUCTION_WAITING = "skipped_production_waiting"  # a production job was already waiting: it goes first
    SKIPPED_NO_LANDMARKS = "skipped_no_landmarks"
    SKIPPED_STALE = "skipped_stale"
    SKIPPED_CAP = "skipped_cap"
    SKIPPED_INTERVAL = "skipped_interval"
    SKIPPED_QUOTA = "skipped_quota"


@dataclass(frozen=True)
class ComparisonPolicy:
    interval_ns: int = round(COMPARISON_INTERVAL_S * NS_PER_SECOND)
    cap: int = COMPARISON_CAP
    control_share: int = CONTROL_SHARE
    max_frame_age_ns: int = 3 * NS_PER_SECOND  # the runtime's face_result_max_age (assemble passes the configured one)

    def __post_init__(self) -> None:
        if self.interval_ns <= 0 or self.cap < 0 or self.control_share < 1 or self.max_frame_age_ns <= 0:
            raise ValueError("invalid comparison policy")


@dataclass(frozen=True)
class Target:
    key: TrackKey
    kind: Kind
    candidate: int  # index into the full-scale pass's candidates (and rows)
    score: float
    width_px: int


@dataclass(frozen=True)
class ComparisonOutcome:
    """One eligible person on one analysed frame. ``embedding`` (memory only) exactly when completed; ``ms`` exactly when
    attempted; ``error`` exactly when failed."""

    frame: FrameRef
    track_id: int
    kind: Kind
    disposition: Disposition
    score: float  # the full-scale candidate's score
    width_px: int  # its width in frame pixels
    embedding: tuple[float, ...] | None = None
    ms: float | None = None
    error: str | None = None


def comparison_targets(frame: FrameRef, persons: Sequence[TrackObservation], faces: Sequence[FaceObservation],
                       full_scale: Any, geometry: AssociationGeometry) -> list[Target]:
    """The eligible people on one frame, by track ID: each with the full-scale candidate the association gives them."""
    if full_scale is None or not persons:
        return []
    strong = [(index, score, box) for index, (score, box) in enumerate(full_scale.candidates)
              if score >= FULL_SCALE_COUNTED]
    if not strong:
        return []
    production = associate(persons, faces, geometry)
    full = associate(persons, [FaceObservation(frame=frame.key, box=box, quality=1.0) for _, _, box in strong],
                     geometry)
    targets = []
    for person in sorted(persons, key=lambda p: p.track_id):
        assigned = full.face_of.get(person.key)
        ownership = production.person_ownership.get(person.key, Ownership.NO_FACE)
        if assigned is None or ownership not in (Ownership.NO_FACE, Ownership.ASSIGNED):
            continue
        index, score, box = strong[assigned]
        kind = Kind.FULL_SCALE_ONLY if ownership is Ownership.NO_FACE else Kind.BOTH
        targets.append(Target(person.key, kind, index, score, round((box.x2 - box.x1) * frame.native_width)))
    return targets


def error_label(exc: BaseException) -> str:
    """A fixed label (the adapter's FaceBackendError label, else ``comparison_failed``) and the exception's class."""
    label = getattr(exc, "label", None)
    detail = getattr(exc, "error_type", None) if label else type(exc).__name__
    return f"{label or 'comparison_failed'}" + (f":{detail}" if detail else "")


class FullScaleComparator:
    """The budget and the comparison itself. FaceWorker calls ``plan``, ``record`` and ``status`` under its own lock
    and ``compare`` (the embedding) outside it, on its thread; nothing here touches production's result."""

    def __init__(self, embed: Callable[[FrameRef, Any, Any], Sequence[float]], geometry: AssociationGeometry,
                 policy: ComparisonPolicy = ComparisonPolicy(), *,
                 perf_counter: Callable[[], float] = time.perf_counter) -> None:
        self._embed = embed  # (frame, image, the candidate's row) -> unit embedding; raises on failure
        self.geometry = geometry
        self.policy = policy
        self._perf_counter = perf_counter
        self._last_start_ns: int | None = None
        self._attempts = 0
        self._controls = 0
        self.counters: Counter[str] = Counter()
        self._latencies: deque[float] = deque(maxlen=LATENCIES_KEPT)

    def plan(self, frame: FrameRef, targets: Sequence[Target], rows: Sequence[Any], now_ns: int,
             blocked: Disposition | None = None) -> tuple[Target | None, list[tuple[Target, Disposition]]]:
        """Which target (if any) is compared now, and why each other one is not; ``blocked`` (stopping, or a
        production job waiting) refuses them all. Updates the budget."""
        chosen: Target | None = None
        skipped: list[tuple[Target, Disposition]] = []
        for target in sorted(targets, key=lambda t: (t.kind is not Kind.FULL_SCALE_ONLY, t.key.track_id)):
            self.counters[f"eligible_{target.kind.value}"] += 1
            reason = blocked or self._refusal(frame, target, rows, now_ns, chosen is not None)
            if reason is None:
                chosen = target
            else:
                skipped.append((target, reason))
                self.counters[reason.value] += 1
        if chosen is not None:
            self._last_start_ns = now_ns
            self._attempts += 1
            self._controls += chosen.kind is Kind.BOTH
            self.counters[f"attempted_{chosen.kind.value}"] += 1
        return chosen, skipped

    def _refusal(self, frame: FrameRef, target: Target, rows: Sequence[Any], now_ns: int,
                 taken: bool) -> Disposition | None:
        policy = self.policy
        if taken:
            return Disposition.SKIPPED_ONE_PER_RESULT
        if target.candidate >= len(rows) or rows[target.candidate] is None:
            return Disposition.SKIPPED_NO_LANDMARKS
        if now_ns - frame.ingest_mono_ns > policy.max_frame_age_ns:
            return Disposition.SKIPPED_STALE
        if self._attempts >= policy.cap:
            return Disposition.SKIPPED_CAP
        if self._last_start_ns is not None and now_ns - self._last_start_ns < policy.interval_ns:
            return Disposition.SKIPPED_INTERVAL
        if target.kind is Kind.BOTH and (self._controls + 1) * policy.control_share > self._attempts + 1:
            return Disposition.SKIPPED_QUOTA
        return None

    def compare(self, frame: FrameRef, image: Any, target: Target, rows: Sequence[Any]) -> ComparisonOutcome:
        """The embedding of the chosen candidate (no shared state; called outside the worker's lock)."""
        started = self._perf_counter()
        try:
            embedding = tuple(float(v) for v in self._embed(frame, image, rows[target.candidate]))
            if not embedding or not all(math.isfinite(v) for v in embedding):
                raise ValueError("embedding")
        except Exception as exc:  # noqa: BLE001 - recorded as failed, never as a nonmatch; production is untouched
            return ComparisonOutcome(frame, target.key.track_id, target.kind, Disposition.FAILED, target.score,
                                     target.width_px, ms=(self._perf_counter() - started) * 1000,
                                     error=error_label(exc))
        return ComparisonOutcome(frame, target.key.track_id, target.kind, Disposition.COMPLETED, target.score,
                                 target.width_px, embedding=embedding, ms=(self._perf_counter() - started) * 1000)

    def record(self, outcome: ComparisonOutcome) -> None:
        """Counts an attempted comparison's outcome and its time."""
        if outcome.disposition is Disposition.FAILED:
            self.counters[f"failed_{outcome.kind.value}"] += 1
            self.counters[f"failed:{outcome.error}"] += 1
        else:
            self.counters[f"completed_{outcome.kind.value}"] += 1
        if outcome.ms is not None:
            self._latencies.append(outcome.ms)

    def status(self) -> dict[str, Any]:
        return {"counters": dict(sorted(self.counters.items())), "ms": _spread(sorted(self._latencies)),
                "policy": {"interval_s": self.policy.interval_ns / NS_PER_SECOND, "cap": self.policy.cap,
                           "control_share": self.policy.control_share,
                           "max_frame_age_s": self.policy.max_frame_age_ns / NS_PER_SECOND}}


def _spread(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "p50": None, "max": None}
    return {"n": len(values), "min": round(values[0], 1), "p50": round(values[(len(values) - 1) // 2], 1),
            "max": round(values[-1], 1)}
