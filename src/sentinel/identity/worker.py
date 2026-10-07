"""The 1 Hz face worker (V2-25 demo form; D34).

The loop thread calls ``consider()`` for every frame the tracker processed on the live stream. Scheduling is by the
frame's own ingest time on the monotonic clock: a **tick** is the first such frame at least ``interval_ns`` after the
previous tick. A tick with no person track is **skipped** (``skipped_no_person``): the model does not run on an empty
room. Otherwise the frame is **offered**.

**Ownership.** An offer makes one private, read-only copy of the frame (``private_copy``) and hands it to the worker;
the worker never shares the array the loop, the tracker or the scene path use, so nothing that happens to those can
change the pixels being analysed. The copy is released when the analysis returns, also on error.

**Bounds.** At most one job runs and one waits: a newer offer replaces a waiting one (``replaced``), so at most two
frame copies exist (2 x 921,600 B at 640x480). A waiting job whose frame is older than ``pending_max_age_ns`` when the
worker is free is discarded (``expired_before_start``). Results go to a one-slot outbox: a result the loop has not
taken yet is replaced by a newer one (``results_replaced``). Results carry boxes, quality and embeddings, never an
image. The loop applies a result through ``EdgeCore.on_face_outcome``, which rejects it beyond its own age bound.

**Counters**, never a rate by themselves: ``ticks``, ``skipped_no_person``, ``offered``, ``replaced``,
``expired_before_start``, ``started``, ``completed``, ``errors`` (and ``error:<class>``), ``results_replaced``,
``abandoned_at_stop``, ``faces_detected``, ``fallback_dropped`` and ``faces_returned``; with processing latencies.

**Stop.** A DeepFace call cannot be interrupted: ``stop()`` drops a waiting job, waits up to its bound for a running
one, and reports False if the thread is still busy.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..contracts import FrameRef, TrackObservation
from ..media.clock import NS_PER_SECOND, Clock

LATENCIES_KEPT = 600
TICK_OFFERED = "offered"
TICK_SKIPPED_NO_PERSON = "skipped_no_person"
TICK_STOPPING = "stopping"  # a tick after stop(): nothing is offered


def private_copy(image: Any) -> Any:
    """A copy only the worker holds, made read-only where the array type allows it (NumPy)."""
    copied = image.copy()
    setflags = getattr(copied, "setflags", None)
    if callable(setflags):
        setflags(write=False)
    return copied


@dataclass(frozen=True)
class FaceJob:
    frame: FrameRef
    persons: tuple[TrackObservation, ...]
    image: Any  # the worker's private copy


@dataclass(frozen=True)
class FaceResult:
    frame: FrameRef
    persons: tuple[TrackObservation, ...]  # the tracker's output for that same frame
    faces: tuple[Any, ...] | None  # FaceObservations; None after an error
    error: str | None
    processing_ms: float


class FaceWorker:
    def __init__(
        self,
        analyze: Callable[[FrameRef, Any], Any],
        clock: Clock,
        *,
        interval_ns: int = NS_PER_SECOND,
        pending_max_age_ns: int = 2 * NS_PER_SECOND,
        copy: Callable[[Any], Any] = private_copy,
        perf_counter: Callable[[], float] = time.perf_counter,
    ) -> None:
        if interval_ns <= 0 or pending_max_age_ns <= 0:
            raise ValueError("interval and pending age must be positive")
        self._analyze = analyze  # returns a FaceRun (faces, detected, fallback_dropped)
        self._clock = clock
        self._interval_ns = interval_ns
        self._pending_max_age_ns = pending_max_age_ns
        self._copy = copy
        self._perf_counter = perf_counter
        self._next_tick_ns: int | None = None
        self._condition = threading.Condition()
        self._pending: FaceJob | None = None
        self._result: FaceResult | None = None
        self._running = False
        self._stopping = False
        self._thread: threading.Thread | None = None
        self.counters: Counter[str] = Counter()
        self._latencies: deque[float] = deque(maxlen=LATENCIES_KEPT)

    # ------------------------------------------------------------ loop thread

    def consider(self, frame: FrameRef, persons: Sequence[TrackObservation], image: Any) -> str | None:
        """Called for each processed live frame. None if it is not a tick; otherwise TICK_OFFERED,
        TICK_SKIPPED_NO_PERSON or TICK_STOPPING."""
        if self._next_tick_ns is not None and frame.ingest_mono_ns < self._next_tick_ns:
            return None
        self._next_tick_ns = frame.ingest_mono_ns + self._interval_ns
        with self._condition:
            self.counters["ticks"] += 1
            if not persons:
                self.counters["skipped_no_person"] += 1
                return TICK_SKIPPED_NO_PERSON
            if self._stopping:
                return TICK_STOPPING
        job = FaceJob(frame, tuple(persons), self._copy(image))
        with self._condition:
            if self._pending is not None:
                self.counters["replaced"] += 1
            self._pending = job
            self.counters["offered"] += 1
            self._condition.notify()
        return TICK_OFFERED

    def drain(self) -> FaceResult | None:
        with self._condition:
            result, self._result = self._result, None
            return result

    def status(self) -> dict[str, Any]:
        with self._condition:
            latencies = sorted(self._latencies)
            counters = dict(sorted(self.counters.items()))
            busy, waiting = self._running, self._pending is not None
        thread = self._thread
        state = "not_started" if thread is None else "running" if thread.is_alive() else "stopped"
        return {"state": state, "busy": busy, "waiting": waiting, "counters": counters,
                "processing_ms": _percentiles(latencies)}

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the face worker can be started once")
        self._thread = threading.Thread(target=self._run, name="sentinel-face", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float) -> bool:
        with self._condition:
            self._stopping = True
            if self._pending is not None:
                self._pending = None
                self.counters["abandoned_at_stop"] += 1
            self._condition.notify_all()
        if self._thread is None:
            return True
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    # ------------------------------------------------------------ worker thread

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                job, self._pending = self._pending, None
                if self._clock.monotonic_ns() - job.frame.ingest_mono_ns > self._pending_max_age_ns:
                    self.counters["expired_before_start"] += 1
                    continue
                self.counters["started"] += 1
                self._running = True
            result = self._process(job)
            del job  # the private copy ends here
            with self._condition:
                self._running = False
                if self._result is not None:
                    self.counters["results_replaced"] += 1
                self._result = result

    def _process(self, job: FaceJob) -> FaceResult:
        started = self._perf_counter()
        try:
            run = self._analyze(job.frame, job.image)
        except Exception as exc:  # noqa: BLE001 - counted by class; core monitoring continues
            elapsed = (self._perf_counter() - started) * 1000
            name = type(exc).__name__
            label = getattr(exc, "label", None)
            with self._condition:
                self.counters["errors"] += 1
                self.counters[f"error:{name}"] += 1
            return FaceResult(job.frame, job.persons, None, f"{label or 'analysis_failed'}:{name}", elapsed)
        elapsed = (self._perf_counter() - started) * 1000
        with self._condition:
            self.counters["completed"] += 1
            self.counters["faces_detected"] += run.detected
            self.counters["fallback_dropped"] += run.fallback_dropped
            self.counters["faces_returned"] += len(run.faces)
            self._latencies.append(elapsed)
        return FaceResult(job.frame, job.persons, tuple(run.faces), None, elapsed)


def _percentiles(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}

    def nearest_rank(q: float) -> float:
        return round(values[max(0, math.ceil(q * len(values)) - 1)], 1)

    return {"n": len(values), "p50": nearest_rank(0.50), "p95": nearest_rank(0.95), "p99": nearest_rank(0.99),
            "max": round(values[-1], 1)}
