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
- **A side request (chat, D74, D81).** submit_side() offers one lower-priority
  request to the same worker thread, so the model still answers one request
  at a time. It is accepted while no other side request is pending or running,
  also while a scene job runs or waits (D81: it then starts when no scene job
  runs or waits, so scene jobs always go first). It is never queued behind
  another and never replaces a scene job; cancel_side() drops it while it has
  not started.
- **A running side request yields (D77, D81).** An enrichment job (an
  incident's) submitted while a side request runs aborts it at once (its
  ``abort`` callback: the chat closes its request's connection, and
  llama-server cancels the task). A periodic job lets it run for at most
  ``side_grace_s`` (SIDE_GRACE_S, PROPOSED) from its arrival, then aborts it
  the same way; a side request that ends within the grace is not aborted. So a
  scene job waits at most the grace plus the time an abort takes, never for an
  answer. abort_side() ends a side request for other reasons (its deadline,
  memory pressure, stopping). Each scene job's wait from submission to start is
  measured; the waits of jobs submitted while a side request ran are kept
  (SIDE_WAITS_KEPT) and summarised by side_delays(), with the run times of
  those jobs and of the others, since llama-server's own cancellation (it
  notices a closed request about once a second) shows in the next job's run
  time, not in its wait.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable
from typing import Any

from ..contracts import FrameKey
from ..jobs import AnalysisJob, JobPurpose, OutcomeKind, WorkerOutcome

SIDE = "side"  # what _running holds while a side request runs; never a job ID
IMAGES_KEPT = 4  # 640x480 BGR: 921,600 B each, plus at most one waiting and one running job's image
INBOX_LIMIT = 8
SIDE_WAITS_KEPT = 64
PREEMPTED = "a scene job is waiting: monitoring comes first"
# PROPOSED (D81), for the maintainer's decision before any device run: how long a periodic scene job lets a running
# side request finish. With the profiled 4 s scene interval and scene jobs of about 2.6 s (p50 under the detector,
# face and scene load), a question started when the worker became idle has the rest of the interval plus this grace.
# The wait it causes is judged against alpha_check.MAX_CHAT_DELAY_MS (2000 ms, also PROPOSED): the grace plus the
# time an abort takes to end the request.
SIDE_GRACE_S = 1.5


def _timer(delay_s: float, call: Callable[[], None]) -> Callable[[], None]:
    """Run ``call`` once after ``delay_s`` on a short-lived daemon thread; returns its cancel."""
    timer = threading.Timer(delay_s, call)
    timer.name = "sentinel-scene-grace"
    timer.daemon = True
    timer.start()
    return timer.cancel


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
        side_grace_s: float = SIDE_GRACE_S,
        schedule: Callable[[float, Callable[[], None]], Callable[[], None]] = _timer,
    ) -> None:
        if side_grace_s < 0:
            raise ValueError("side_grace_s must not be negative")
        self._run_job = run_job
        self._images = images
        self._revision = revision
        self._now = now  # waits are measured on this monotonic clock (ns)
        self._side_grace_s = side_grace_s
        self._schedule = schedule  # starts the grace's timer; must not call back before it returns
        self._side_run = 0  # numbers side requests as they start, so a late timer cannot end a later one
        self._grace: tuple[int, Callable[[], None] | None] | None = None  # (side run, cancel) while a grace runs
        self._side_ended_by: str | None = None  # how the running side request was ended, if it was
        self._run_ms: deque[tuple[float, bool]] = deque(maxlen=SIDE_WAITS_KEPT * 4)  # (run ms, behind a side)
        self._behind_side: bool = False  # whether the running scene job waited behind a side request
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
            abort, grace, run = None, None, self._side_run
            if behind_side and self._running_abort is not None:
                if job.purpose is JobPurpose.PERIODIC and self._side_grace_s > 0:
                    if self._grace is None:  # one grace per side request, from the first job's arrival
                        grace = run
                        self._grace = (grace, None)
                        self.counters["side_grace_started"] += 1
                else:
                    abort = self._running_abort
        if abort is not None:
            self._end_side(abort, "preempted", run)  # outside the lock: it closes a connection
        if grace is not None:
            cancel = self._schedule(self._side_grace_s, lambda: self._grace_over(grace))
            with self._lock:
                if self._grace is not None and self._grace[0] == grace:
                    self._grace = (grace, cancel)
                    cancel = None
            if cancel is not None:
                cancel()  # the side request ended before its timer was kept

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
        given, ends the request while it runs (D77): a scene job's arrival calls it (an enrichment job's at once, a
        periodic job's after the grace, D81), so does abort_side(). Accepted while a scene job runs or waits too; it
        starts when none does.
        """
        with self._lock:
            if self._stopping or self._thread is None or not self._thread.is_alive():
                return False
            if self._running == SIDE or self._side is not None:
                self.counters["side_refused"] += 1
                return False
            if self._running is not None or self._waiting is not None:
                self.counters["side_accepted_behind_scene"] += 1
            self._side = (work, started, finished, abort)
            self.counters["side_accepted"] += 1
            self._wake.notify()
            return True

    def abort_side(self, reason: str) -> bool:
        """End the running side request (its deadline, memory pressure, stopping); True if one was running."""
        with self._lock:
            abort = self._running_abort if self._running == SIDE else None
            if abort is not None:
                self._side_ended_by = "aborted"
        if abort is None or not abort(reason):
            return False
        with self._lock:
            self.counters["side_aborted"] += 1
        return True

    def side_delays(self) -> dict[str, Any]:
        """How long scene jobs submitted while a side request ran waited to start (ms): the latest SIDE_WAITS_KEPT,
        and the largest over the whole run; the grace; and the run times (ms) of the latest scene jobs that waited
        behind a side request and of the others, since llama-server's own cancellation shows in the next run."""
        with self._lock:
            waits, ever, runs = list(self._side_waits_ms), self._side_wait_max_ms, list(self._run_ms)

        def spread(values: list[float]) -> dict[str, Any]:
            return {"n": len(values), "median_ms": round(statistics.median(values), 1) if values else None,
                    "max_ms": round(max(values), 1) if values else None}

        runs_of = {"behind_side": spread([ms for ms, behind in runs if behind]),
                   "others": spread([ms for ms, behind in runs if not behind])}
        if not waits:
            return {"n": 0, "median_ms": None, "max_ms": None, "last_ms": None, "max_ever_ms": None,
                    "grace_ms": round(self._side_grace_s * 1000), "run_ms": runs_of}
        return {"n": len(waits), "median_ms": round(statistics.median(waits), 1), "max_ms": round(max(waits), 1),
                "last_ms": round(waits[-1], 1), "max_ever_ms": round(ever or 0.0, 1),
                "grace_ms": round(self._side_grace_s * 1000), "run_ms": runs_of}

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
            grace, self._grace = self._grace, None
        if grace is not None and grace[1] is not None:
            grace[1]()
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
                    self._side_run += 1
                    self._side_ended_by = None
                else:
                    side = None
                    job, image = self._waiting
                    self._waiting = None
                    self._running = job.job_id
                    self._behind_side = False
                    if self._waiting_since is not None:
                        since, behind_side = self._waiting_since
                        if behind_side:
                            waited = (self._now() - since) / 1e6
                            self._side_waits_ms.append(waited)
                            self._side_wait_max_ms = max(waited, self._side_wait_max_ms or 0.0)
                            self.counters["jobs_waited_for_side"] += 1
                            self._behind_side = True
                    self._waiting_since = None
                    began = self._now()
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
                self._run_ms.append(((self._now() - began) / 1e6, self._behind_side))
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
            grace, self._grace = self._grace, None
            if grace is not None and self._side_ended_by is None:
                self.counters["side_finished_in_grace"] += 1  # it ended by itself before a waiting job ended it
        if grace is not None and grace[1] is not None:
            grace[1]()

    def _grace_over(self, run: int) -> None:
        """The grace's timer: end the side request it was started for, if it still runs and a scene job waits."""
        with self._lock:
            if self._running != SIDE or self._side_run != run:
                return
            if self._waiting is None:  # the job was given up meanwhile: a later one starts its own grace
                if self._grace is not None and self._grace[0] == run:
                    self._grace = None
                return
            abort = self._running_abort
        if abort is not None:
            self._end_side(abort, "grace_over", run)

    def _end_side(self, abort: Callable[[str], bool], how: str, run: int) -> None:
        """Abort side request ``run`` for a waiting scene job, if it still runs (the abort outside the lock: it
        closes a connection)."""
        with self._lock:
            if self._running != SIDE or self._side_run != run:
                return
            self._side_ended_by = how  # first, so that its end is never counted as finishing within the grace
        if abort(PREEMPTED):
            with self._lock:
                self.counters["side_preempted"] += 1
                if how == "grace_over":
                    self.counters["side_preempted_after_grace"] += 1

    def _deliver(self, job_id: str, outcome: WorkerOutcome) -> None:
        # Called with the lock held.
        self._inbox.append((job_id, outcome))
        self.counters["delivered"] += 1
        while len(self._inbox) > INBOX_LIMIT:
            self._inbox.popleft()
            self.counters["inbox_dropped"] += 1
