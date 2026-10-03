"""Bounded scene worker (guide chapters 12 and 27; V2-26 demo form, full acceptance pending).

SceneLane calls submit() and cancel() on the runtime's loop thread and must not
block on inference. ThreadedSceneAnalyzer owns one worker thread that runs each
job with a blocking ``run_job(job, image)`` and puts the outcome in a bounded
inbox. The runtime drains the inbox on its loop thread and passes each outcome
to ``EdgeCore.on_scene_outcome()``, so the core stays single-threaded.

- **Images.** An AnalysisJob names a frame, not pixels. RecentImages keeps
  references to the images of the last few frames; the runtime puts every frame
  it passes to EdgeCore. submit() takes the job's own image from there. A job
  whose image is gone gets an error outcome at once, never another frame's image.
- **Bounds.** At most one job runs and one waits, and at most IMAGES_KEPT
  images are retained. The inbox keeps at most INBOX_LIMIT outcomes; older ones
  are dropped and counted. A new job replaces a waiting one, which then gets an
  error outcome. The lane does not do this: it cancels a job before it submits
  the next.
- **Cancellation.** The lane cancels a job at its deadline. A waiting job is
  dropped without an outcome, because the lane has already recorded its
  timeout. A running request is not interrupted: it ends within its own request
  timeout, and its outcome is still delivered, late. The lane then lets it
  annotate the job's incident without changing current scene state (D15, D35).
  The next job waits behind it, so one heavy job runs at a time.
- **Failures.** An exception from ``run_job`` becomes that job's error outcome
  (exception class name only), and the worker keeps going. submit() raises
  after stop() or if the worker thread died; the lane records that as error
  evidence.
"""

from __future__ import annotations

import threading
from collections import Counter, OrderedDict, deque
from collections.abc import Callable
from typing import Any

from ..contracts import FrameKey
from ..jobs import AnalysisJob, OutcomeKind, WorkerOutcome

IMAGES_KEPT = 4  # 640x480 BGR: 921,600 B each, plus at most one waiting and one running job's image
INBOX_LIMIT = 8


class RecentImages:
    """The images of the last ``capacity`` frames, by frame identity. Thread-safe."""

    def __init__(self, capacity: int = IMAGES_KEPT) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._images: OrderedDict[FrameKey, Any] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, frame: FrameKey, image: Any) -> None:
        with self._lock:
            self._images[frame] = image
            self._images.move_to_end(frame)
            while len(self._images) > self._capacity:
                self._images.popitem(last=False)

    def get(self, frame: FrameKey) -> Any | None:
        with self._lock:
            return self._images.get(frame)

    def __len__(self) -> int:
        with self._lock:
            return len(self._images)


class ThreadedSceneAnalyzer:
    """A SceneAnalyzer that runs one job at a time on its own thread; drain() collects outcomes."""

    def __init__(
        self,
        run_job: Callable[[AnalysisJob, Any], WorkerOutcome],
        images: RecentImages,
        *,
        revision: str,
    ) -> None:
        self._run_job = run_job
        self._images = images
        self._revision = revision
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._waiting: tuple[AnalysisJob, Any] | None = None
        self._running: str | None = None
        self._inbox: deque[tuple[str, WorkerOutcome]] = deque()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self.counters: Counter[str] = Counter()

    @property
    def revision(self) -> str:
        return self._revision

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("the scene worker can be started once")
            self._thread = threading.Thread(target=self._work, name="sentinel-scene", daemon=True)
            self._thread.start()

    def submit(self, job: AnalysisJob) -> None:
        image = self._images.get(job.frame.key)
        with self._lock:
            if self._stopping or self._thread is None or not self._thread.is_alive():
                raise RuntimeError("scene worker not running")
            if image is None:
                self.counters["image_not_retained"] += 1
                self._deliver(job.job_id, WorkerOutcome.failed(
                    job.job_id, OutcomeKind.ERROR, "source frame image no longer retained"))
                return
            if self._waiting is not None:
                replaced = self._waiting[0].job_id
                self.counters["replaced_waiting"] += 1
                self._deliver(replaced, WorkerOutcome.failed(replaced, OutcomeKind.ERROR, "replaced before it started"))
            self._waiting = (job, image)
            self.counters["submitted"] += 1
            self._wake.notify()

    def cancel(self, job_id: str) -> None:
        with self._lock:
            if self._waiting is not None and self._waiting[0].job_id == job_id:
                self._waiting = None
                self.counters["cancelled_waiting"] += 1
            elif self._running == job_id:
                self.counters["cancelled_running"] += 1  # its outcome still arrives, late

    def drain(self) -> list[tuple[str, WorkerOutcome]]:
        """Outcomes delivered since the last call, oldest first: (job ID run, outcome)."""
        with self._lock:
            outcomes = list(self._inbox)
            self._inbox.clear()
            return outcomes

    def busy(self) -> bool:
        with self._lock:
            return self._running is not None or self._waiting is not None

    def stop(self, timeout_s: float) -> bool:
        """Drop any waiting job and stop the worker; False if a request is still running at the timeout."""
        with self._lock:
            self._stopping = True
            if self._waiting is not None:
                self._waiting = None
                self.counters["dropped_at_stop"] += 1
            self._wake.notify()
            thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            return not thread.is_alive()
        return True

    def _work(self) -> None:
        while True:
            with self._lock:
                while self._waiting is None and not self._stopping:
                    self._wake.wait()
                if self._stopping:
                    return
                job, image = self._waiting  # type: ignore[misc]
                self._waiting = None
                self._running = job.job_id
            try:
                outcome = self._run_job(job, image)
            except Exception as exc:  # one bad job must not stop scene analysis
                outcome = WorkerOutcome.failed(job.job_id, OutcomeKind.ERROR, f"worker error: {type(exc).__name__}")
            del image
            with self._lock:
                self._running = None
                self._deliver(job.job_id, outcome)

    def _deliver(self, job_id: str, outcome: WorkerOutcome) -> None:
        # Called with the lock held.
        self._inbox.append((job_id, outcome))
        self.counters["delivered"] += 1
        while len(self._inbox) > INBOX_LIMIT:
            self._inbox.popleft()
            self.counters["inbox_dropped"] += 1
