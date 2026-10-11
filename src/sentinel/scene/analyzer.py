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
- **A side request (chat, D74).** submit_side() offers one lower-priority
  request to the same worker thread, so the model still answers one request
  at a time. It is accepted only while the worker is idle (nothing running or
  waiting) and no other side request is pending, and a scene job submitted
  before it starts runs first. It is never queued behind another and never
  replaces a scene job; cancel_side() drops it while it has not started.
- **A running side request yields (D77).** A scene job submitted while a side
  request runs aborts it at once (its ``abort`` callback: the chat closes its
  request's connection, and llama-server cancels the task), so the scene job
  waits only for the side request to end, never for its answer. abort_side()
  ends it for other reasons (its deadline, memory pressure, stopping). Each
  scene job's wait from submission to start is measured; the waits of jobs
  submitted while a side request ran are kept (SIDE_WAITS_KEPT) and summarised
  by side_delays().
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable
from typing import Any

from ..contracts import FrameKey
from ..jobs import AnalysisJob, OutcomeKind, WorkerOutcome

SIDE = "side"  # what _running holds while a side request runs; never a job ID
IMAGES_KEPT = 4  # 640x480 BGR: 921,600 B each, plus at most one waiting and one running job's image
INBOX_LIMIT = 8
SIDE_WAITS_KEPT = 64
PREEMPTED = "a scene job is waiting: monitoring comes first"


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
        now: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._run_job = run_job
        self._images = images
        self._revision = revision
        self._now = now  # waits are measured on this monotonic clock (ns)
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._waiting: tuple[AnalysisJob, Any] | None = None
        self._waiting_since: tuple[int, bool] | None = None  # when the waiting job came, and whether a side request ran
        self._side: tuple[Callable[[], Any], Callable[[], None], Callable[[Any], None],
                          Callable[[str], bool] | None] | None = None
        self._running_abort: Callable[[str], bool] | None = None  # the running side request's abort
        self._side_waits_ms: deque[float] = deque(maxlen=SIDE_WAITS_KEPT)
        self._side_wait_max_ms: float | None = None  # over the whole run, not only the waits kept
        self._running: str | None = None  # a job ID, or SIDE
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
            behind_side = self._running == SIDE
            self._waiting = (job, image)
            self._waiting_since = (self._now(), behind_side)
            self.counters["submitted"] += 1
            self._wake.notify()
            abort = self._running_abort if behind_side else None
        if abort is not None and abort(PREEMPTED):  # outside the lock: it closes a connection
            with self._lock:
                self.counters["side_preempted"] += 1

    def cancel(self, job_id: str) -> None:
        with self._lock:
            if self._waiting is not None and self._waiting[0].job_id == job_id:
                self._waiting = None
                self._waiting_since = None
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
            return self._running is not None or self._waiting is not None or self._side is not None

    def submit_side(self, work: Callable[[], Any], *, started: Callable[[], None],
                    finished: Callable[[Any], None], abort: Callable[[str], bool] | None = None) -> bool:
        """Offer one lower-priority request (D74); whether it was accepted.

        ``started()`` and ``finished(result)`` are called on the worker thread; ``result`` is what ``work()``
        returned or the exception it raised. Accepted only while nothing runs or waits here. ``abort(reason)``, if
        given, ends the request while it runs (D77): a scene job's arrival calls it, so does abort_side().
        """
        with self._lock:
            if self._stopping or self._thread is None or not self._thread.is_alive():
                return False
            if self._running is not None or self._waiting is not None or self._side is not None:
                self.counters["side_refused"] += 1
                return False
            self._side = (work, started, finished, abort)
            self.counters["side_accepted"] += 1
            self._wake.notify()
            return True

    def abort_side(self, reason: str) -> bool:
        """End the running side request (its deadline, memory pressure, stopping); True if one was running."""
        with self._lock:
            abort = self._running_abort if self._running == SIDE else None
        if abort is None or not abort(reason):
            return False
        with self._lock:
            self.counters["side_aborted"] += 1
        return True

    def side_delays(self) -> dict[str, Any]:
        """How long scene jobs submitted while a side request ran waited to start (ms): the latest SIDE_WAITS_KEPT,
        and the largest over the whole run."""
        with self._lock:
            waits, ever = list(self._side_waits_ms), self._side_wait_max_ms
        if not waits:
            return {"n": 0, "median_ms": None, "max_ms": None, "last_ms": None, "max_ever_ms": None}
        return {"n": len(waits), "median_ms": round(statistics.median(waits), 1), "max_ms": round(max(waits), 1),
                "last_ms": round(waits[-1], 1), "max_ever_ms": round(ever or 0.0, 1)}

    def cancel_side(self) -> bool:
        """Drop a side request that has not started; True if one was dropped."""
        with self._lock:
            if self._side is None:
                return False
            self._side = None
            self.counters["side_cancelled"] += 1
            return True

    def stop(self, timeout_s: float) -> bool:
        """Drop any waiting job and stop the worker; False if a request is still running at the timeout."""
        with self._lock:
            self._stopping = True
            if self._waiting is not None:
                self._waiting = None
                self._waiting_since = None
                self.counters["dropped_at_stop"] += 1
            side, self._side = self._side, None
            self._wake.notify()
            thread = self._thread
            running_abort = self._running_abort if self._running == SIDE else None
        if side is not None:
            self.counters["side_dropped_at_stop"] += 1
            side[2](RuntimeError("stopped"))
        if running_abort is not None:
            running_abort("the runtime is stopping")
        if thread is not None:
            thread.join(timeout_s)
            return not thread.is_alive()
        return True

    def _work(self) -> None:
        while True:
            with self._lock:
                while self._waiting is None and self._side is None and not self._stopping:
                    self._wake.wait()
                if self._stopping:
                    return
                if self._waiting is None:  # scene jobs first; a side request only when none waits
                    side, self._side = self._side, None
                    self._running = SIDE
                    self._running_abort = side[3]  # type: ignore[index]
                else:
                    side = None
                    job, image = self._waiting
                    self._waiting = None
                    self._running = job.job_id
                    if self._waiting_since is not None:
                        since, behind_side = self._waiting_since
                        if behind_side:
                            waited = (self._now() - since) / 1e6
                            self._side_waits_ms.append(waited)
                            self._side_wait_max_ms = max(waited, self._side_wait_max_ms or 0.0)
                            self.counters["jobs_waited_for_side"] += 1
                    self._waiting_since = None
            if side is not None:
                self._run_side(*side[:3])  # type: ignore[arg-type]
                del side
                continue
            try:
                outcome = self._run_job(job, image)
            except Exception as exc:  # one bad job must not stop scene analysis
                outcome = WorkerOutcome.failed(job.job_id, OutcomeKind.ERROR, f"worker error: {type(exc).__name__}")
            del image
            with self._lock:
                self._running = None
                self._deliver(job.job_id, outcome)

    def _run_side(self, work: Callable[[], Any], started: Callable[[], None], finished: Callable[[Any], None]) -> None:
        try:
            started()
            result: Any = work()
        except Exception as exc:  # the side request's error, never the worker's
            result = exc
        try:
            finished(result)
        except Exception:  # noqa: BLE001 - a broken callback must not stop scene analysis
            self.counters["side_callback_failed"] += 1
        with self._lock:
            self._running = None
            self._running_abort = None
            self.counters["side_completed"] += 1

    def _deliver(self, job_id: str, outcome: WorkerOutcome) -> None:
        # Called with the lock held.
        self._inbox.append((job_id, outcome))
        self.counters["delivered"] += 1
        while len(self._inbox) > INBOX_LIMIT:
            self._inbox.popleft()
            self.counters["inbox_dropped"] += 1
