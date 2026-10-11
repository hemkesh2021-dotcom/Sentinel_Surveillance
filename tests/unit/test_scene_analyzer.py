"""V2-26 demo form: the bounded scene worker, alone and behind EdgeCore (fake transport; no server, no model)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable

import pytest

from sentinel.config import parse_config
from sentinel.contracts import FrameRef, PixelFormat
from sentinel.jobs import AnalysisJob, JobPurpose, OutcomeKind, WorkerOutcome
from sentinel.live_state import SceneStatus
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.runtime import EdgeCore
from sentinel.scene.analyzer import INBOX_LIMIT, RecentImages, ThreadedSceneAnalyzer
from sentinel.scene.llama_server import LlamaSceneRequest, TransportError

REPORT = {
    "persons_visible": 0,
    "fire_or_smoke": False,
    "threat": "none",
    "observations": [],
    "uncertainty": "low",
    "summary": "An empty room.",
}
REPLY = json.dumps(
    {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(REPORT)}}]}
).encode()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def stamper(clock: FakeClock) -> FrameStamper:
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    return stamper


def frame_at(stamper: FrameStamper, clock: FakeClock, seconds: float = 1 / 15) -> FrameRef:
    clock.advance(seconds)
    return stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)


def job_for(frame: FrameRef, job_id: str = "job-1", incident: str | None = None) -> AnalysisJob:
    purpose = JobPurpose.ENRICHMENT if incident else JobPurpose.PERIODIC
    return AnalysisJob(job_id=job_id, purpose=purpose, frame=frame, incident_id=incident,
                       deadline_mono_ns=frame.ingest_mono_ns + 8_000_000_000)


class Gate:
    """run_job that records its calls and blocks until released."""

    def __init__(self, *, open_: bool = True) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls: list[tuple[str, object]] = []
        if open_:
            self.release.set()

    def __call__(self, job: AnalysisJob, image: object) -> WorkerOutcome:
        self.calls.append((job.job_id, image))
        self.started.set()
        assert self.release.wait(5), "test never released the job"
        return WorkerOutcome.completed(job.job_id, json.dumps(REPORT))


class GatedTransport:
    def __init__(self) -> None:
        self.release = threading.Event()
        self.release.set()
        self.posts = 0

    def post(self, body: bytes) -> bytes:
        self.posts += 1
        assert self.release.wait(5), "test never released the request"
        return REPLY


def wait_for(predicate: Callable[[], bool], timeout_s: float = 3.0) -> None:
    end = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < end, "condition not reached"
        time.sleep(0.005)


def drained(analyzer: ThreadedSceneAnalyzer, count: int = 1) -> list[tuple[str, WorkerOutcome]]:
    got: list[tuple[str, WorkerOutcome]] = []
    wait_for(lambda: bool(got.extend(analyzer.drain()) or len(got) >= count))
    return got


def make(run_job: Callable[[AnalysisJob, object], WorkerOutcome], images: RecentImages | None = None) -> ThreadedSceneAnalyzer:
    analyzer = ThreadedSceneAnalyzer(run_job, RecentImages() if images is None else images, revision="1+lfm2-vl")
    analyzer.start()
    return analyzer


# --- the worker alone ---------------------------------------------------------------------------

def test_recent_images_keep_only_the_latest_frames(stamper: FrameStamper, clock: FakeClock) -> None:
    images = RecentImages(capacity=2)
    frames = [frame_at(stamper, clock) for _ in range(3)]
    for index, frame in enumerate(frames):
        images.put(frame.key, f"image-{index}")
    assert (images.get(frames[0].key), images.get(frames[1].key), images.get(frames[2].key)) == (None, "image-1", "image-2")
    assert len(images) == 2


def test_submit_does_not_wait_for_inference_and_the_outcome_comes_through_drain(
    stamper: FrameStamper, clock: FakeClock
) -> None:
    gate = Gate(open_=False)
    images = RecentImages()
    analyzer = make(gate, images)
    frame = frame_at(stamper, clock)
    images.put(frame.key, "pixels")

    started = time.monotonic()
    analyzer.submit(job_for(frame))
    assert time.monotonic() - started < 0.5 and analyzer.busy()
    assert gate.started.wait(2) and analyzer.drain() == []
    gate.release.set()

    ((job_id, outcome),) = drained(analyzer)
    assert job_id == "job-1" and outcome.kind is OutcomeKind.COMPLETED
    assert gate.calls == [("job-1", "pixels")]  # the job's own frame image
    assert analyzer.stop(2.0)


def test_a_job_whose_image_is_gone_fails_at_once_and_never_runs(stamper: FrameStamper, clock: FakeClock) -> None:
    gate = Gate()
    analyzer = make(gate)
    analyzer.submit(job_for(frame_at(stamper, clock)))
    ((job_id, outcome),) = analyzer.drain()
    assert (job_id, outcome.kind, outcome.detail) == ("job-1", OutcomeKind.ERROR, "source frame image no longer retained")
    assert gate.calls == [] and analyzer.counters["image_not_retained"] == 1
    analyzer.stop(2.0)


def test_cancelling_a_waiting_job_drops_it_and_a_running_one_still_reports_late(
    stamper: FrameStamper, clock: FakeClock
) -> None:
    gate = Gate(open_=False)
    images = RecentImages()
    analyzer = make(gate, images)
    first, second = frame_at(stamper, clock), frame_at(stamper, clock)
    images.put(first.key, "a")
    images.put(second.key, "b")
    analyzer.submit(job_for(first, "job-1"))
    assert gate.started.wait(2)
    analyzer.submit(job_for(second, "job-2"))  # waits behind the running request

    analyzer.cancel("job-2")
    analyzer.cancel("job-1")
    gate.release.set()

    ((job_id, outcome),) = drained(analyzer)
    assert job_id == "job-1" and outcome.kind is OutcomeKind.COMPLETED
    assert analyzer.stop(2.0)
    assert [call[0] for call in gate.calls] == ["job-1"]
    assert (analyzer.counters["cancelled_waiting"], analyzer.counters["cancelled_running"]) == (1, 1)


def test_one_job_runs_and_one_waits_a_replaced_waiting_job_says_so(stamper: FrameStamper, clock: FakeClock) -> None:
    gate = Gate(open_=False)
    images = RecentImages()
    analyzer = make(gate, images)
    frames = [frame_at(stamper, clock) for _ in range(3)]
    for frame in frames:
        images.put(frame.key, frame.frame_seq)
    analyzer.submit(job_for(frames[0], "job-1"))
    assert gate.started.wait(2)
    analyzer.submit(job_for(frames[1], "job-2"))
    analyzer.submit(job_for(frames[2], "job-3"))
    assert analyzer.drain()[0][1].detail == "replaced before it started"
    gate.release.set()
    outcomes = drained(analyzer, 2)
    assert [job_id for job_id, _ in outcomes] == ["job-1", "job-3"]
    assert [call[0] for call in gate.calls] == ["job-1", "job-3"]
    analyzer.stop(2.0)


def test_a_worker_exception_is_that_jobs_error_and_the_worker_keeps_going(stamper: FrameStamper, clock: FakeClock) -> None:
    calls = []

    def flaky(job: AnalysisJob, image: object) -> WorkerOutcome:
        calls.append(job.job_id)
        if job.job_id == "job-1":
            raise RuntimeError("CUDA error at /home/user/secret")
        return WorkerOutcome.completed(job.job_id, json.dumps(REPORT))

    images = RecentImages()
    analyzer = make(flaky, images)
    for job_id in ("job-1", "job-2"):
        frame = frame_at(stamper, clock)
        images.put(frame.key, "x")
        analyzer.submit(job_for(frame, job_id))
        wait_for(lambda: not analyzer.busy())
    outcomes = analyzer.drain()
    assert [(j, o.kind, o.detail) for j, o in outcomes] == [
        ("job-1", OutcomeKind.ERROR, "worker error: RuntimeError"),
        ("job-2", OutcomeKind.COMPLETED, None),
    ]
    assert "secret" not in repr(outcomes)
    analyzer.stop(2.0)


def test_the_inbox_is_bounded(stamper: FrameStamper, clock: FakeClock) -> None:
    analyzer = make(Gate())
    for index in range(INBOX_LIMIT + 3):  # image missing: each fails at once into the inbox
        analyzer.submit(job_for(frame_at(stamper, clock), f"job-{index}"))
    outcomes = analyzer.drain()
    assert len(outcomes) == INBOX_LIMIT and outcomes[0][0] == "job-3"
    assert analyzer.counters["inbox_dropped"] == 3
    analyzer.stop(2.0)


def test_stop_is_bounded_and_submit_after_stop_is_refused(stamper: FrameStamper, clock: FakeClock) -> None:
    gate = Gate(open_=False)
    images = RecentImages()
    analyzer = make(gate, images)
    frame = frame_at(stamper, clock)
    images.put(frame.key, "x")
    analyzer.submit(job_for(frame))
    assert gate.started.wait(2)

    started = time.monotonic()
    assert analyzer.stop(0.2) is False  # the running request is not interrupted
    assert time.monotonic() - started < 1.0
    with pytest.raises(RuntimeError, match="not running"):
        analyzer.submit(job_for(frame, "job-2"))
    gate.release.set()
    assert analyzer.stop(2.0) is True
    with pytest.raises(RuntimeError):
        analyzer.start()


class Side:
    """A side request (D74): records its callbacks in ``log`` and blocks until released."""

    def __init__(self, log: list[object] | None = None, *, open_: bool = True, result: object = "answer") -> None:
        self.log = [] if log is None else log
        self.release = threading.Event()
        self.started = threading.Event()
        self.finished = threading.Event()
        self.result = result
        if open_:
            self.release.set()

    def work(self) -> object:
        self.log.append("side")
        assert self.release.wait(5), "test never released the side request"
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def on_started(self) -> None:
        self.started.set()

    def on_finished(self, result: object) -> None:
        self.log.append(result)
        self.finished.set()

    def submit(self, analyzer: ThreadedSceneAnalyzer) -> bool:
        return analyzer.submit_side(self.work, started=self.on_started, finished=self.on_finished)


class Asleep:
    """Keeps the worker asleep through one submit_side(), so that the request stays pending, not started."""

    def __init__(self, analyzer: ThreadedSceneAnalyzer) -> None:
        self.condition = analyzer._wake  # noqa: SLF001

    def __enter__(self) -> None:
        self.condition.notify = lambda n=1: None  # type: ignore[method-assign]

    def __exit__(self, *exc: object) -> None:
        del self.condition.notify  # the class's own notify again


def test_a_side_request_offered_while_a_scene_job_runs_starts_after_it_and_never_queues(
    stamper: FrameStamper, clock: FakeClock
) -> None:
    gate = Gate(open_=False)
    images = RecentImages()
    analyzer = make(gate, images)
    frame = frame_at(stamper, clock)
    images.put(frame.key, "x")
    analyzer.submit(job_for(frame))
    assert gate.started.wait(2)
    side = Side(open_=False)
    assert side.submit(analyzer) is True  # D81: accepted while a scene job runs ...
    assert Side().submit(analyzer) is False  # ... one at a time, never queued
    assert not side.started.is_set()  # ... and it waits: the worker is still in the scene job
    gate.release.set()
    assert side.started.wait(2) and analyzer.busy()
    assert Side().submit(analyzer) is False  # one at a time while it runs too
    side.release.set()
    assert side.finished.wait(2) and side.log == ["side", "answer"]
    wait_for(lambda: not analyzer.busy())
    assert [job_id for job_id, _ in analyzer.drain()] == ["job-1"]  # never in the scene inbox
    counters = analyzer.counters
    assert (counters["side_refused"], counters["side_completed"], counters["side_accepted_behind_scene"]) == (2, 1, 1)
    assert analyzer.stop(2.0)


def test_scene_jobs_go_first_a_running_side_request_delays_one_and_drops_none(
    stamper: FrameStamper, clock: FakeClock
) -> None:
    order: list[object] = []

    def scene(job: AnalysisJob, image: object) -> WorkerOutcome:
        order.append(job.job_id)
        return WorkerOutcome.completed(job.job_id, json.dumps(REPORT))

    images = RecentImages()
    analyzer = make(scene, images)
    frames = [frame_at(stamper, clock) for _ in range(2)]
    for frame in frames:
        images.put(frame.key, "x")
    first = Side(order)
    with Asleep(analyzer):
        assert first.submit(analyzer)  # accepted while idle, not started yet
    analyzer.submit(job_for(frames[0], "job-1"))  # arrives before it starts: runs first
    assert first.finished.wait(2)
    wait_for(lambda: not analyzer.busy())
    second = Side(order, open_=False)
    assert second.submit(analyzer) and second.started.wait(2)
    analyzer.submit(job_for(frames[1], "job-2"))  # waits for the running side request, then runs
    second.release.set()
    wait_for(lambda: len(order) == 6)
    assert order == ["job-1", "side", "answer", "side", "answer", "job-2"]
    assert [job_id for job_id, _ in drained(analyzer, 2)] == ["job-1", "job-2"]
    assert analyzer.stop(2.0)


class Timers:
    """The grace's timers, by hand (D81): each started timer is kept, and the test fires it."""

    def __init__(self) -> None:
        self.started: list[list[object]] = []  # [delay_s, call, cancelled]

    def __call__(self, delay_s: float, call: Callable[[], None]) -> Callable[[], None]:
        entry: list[object] = [delay_s, call, False]
        self.started.append(entry)

        def cancel() -> None:
            entry[2] = True

        return cancel

    def fire(self, index: int = -1, *, even_if_cancelled: bool = False) -> None:
        _, call, cancelled = self.started[index]
        if even_if_cancelled or not cancelled:
            call()  # type: ignore[operator]


class Abortable(Side):
    """A side request whose ``abort`` ends it as closing its connection would: at once, with the reason."""

    def __init__(self, log: list[object] | None = None) -> None:
        super().__init__(log, open_=False)
        self.aborted: list[str] = []

    def abort(self, reason: str) -> bool:
        if self.aborted:
            return False
        self.aborted.append(reason)
        self.result = RuntimeError(reason)
        self.release.set()
        return True

    def submit(self, analyzer: ThreadedSceneAnalyzer) -> bool:
        return analyzer.submit_side(self.work, started=self.on_started, finished=self.on_finished, abort=self.abort)


def graced(order: list[object], timers: Timers, grace_s: float = 1.5) -> tuple[ThreadedSceneAnalyzer, RecentImages]:
    def scene(job: AnalysisJob, image: object) -> WorkerOutcome:
        order.append(job.job_id)
        return WorkerOutcome.completed(job.job_id, json.dumps(REPORT))

    images = RecentImages()
    analyzer = ThreadedSceneAnalyzer(scene, images, revision="1+lfm2-vl", side_grace_s=grace_s, schedule=timers)
    analyzer.start()
    return analyzer, images


def held_frame(stamper: FrameStamper, clock: FakeClock, images: RecentImages) -> FrameRef:
    frame = frame_at(stamper, clock)
    images.put(frame.key, "x")
    return frame


def test_a_periodic_job_lets_a_running_side_request_finish_within_the_grace(
    stamper: FrameStamper, clock: FakeClock
) -> None:
    order: list[object] = []
    timers = Timers()
    analyzer, images = graced(order, timers)
    side = Abortable(order)
    assert side.submit(analyzer) and side.started.wait(2)
    analyzer.submit(job_for(held_frame(stamper, clock, images), "job-1"))
    assert [entry[0] for entry in timers.started] == [1.5] and side.aborted == []  # a grace, not an abort
    side.release.set()  # the answer comes within the grace
    wait_for(lambda: order == ["side", "answer", "job-1"])
    assert timers.started[0][2] is True  # its end cancelled the grace's timer
    timers.fire(even_if_cancelled=True)  # a timer that fires anyway ends nothing
    assert side.aborted == []
    counters = analyzer.counters
    assert (counters["side_grace_started"], counters["side_finished_in_grace"], counters["side_preempted"]) == (1, 1, 0)
    assert analyzer.side_delays()["n"] == 1 and analyzer.side_delays()["grace_ms"] == 1500
    assert analyzer.stop(2.0)


def test_when_the_grace_is_over_the_side_request_is_ended_for_the_waiting_periodic_job(
    stamper: FrameStamper, clock: FakeClock
) -> None:
    order: list[object] = []
    timers = Timers()
    analyzer, images = graced(order, timers)
    side = Abortable(order)
    assert side.submit(analyzer) and side.started.wait(2)
    analyzer.submit(job_for(held_frame(stamper, clock, images), "job-1"))
    timers.fire()
    wait_for(lambda: order[-1:] == ["job-1"])
    assert side.aborted == ["a scene job is waiting: monitoring comes first"]
    assert isinstance(order[1], RuntimeError)  # the side request was told why it ended
    counters = analyzer.counters
    assert (counters["side_preempted"], counters["side_preempted_after_grace"], counters["side_finished_in_grace"]) == (
        1, 1, 0)
    delays = analyzer.side_delays()
    assert delays["n"] == 1 and delays["run_ms"]["behind_side"]["n"] == 1 and delays["run_ms"]["others"]["n"] == 0
    assert analyzer.stop(2.0)


def test_an_incidents_job_ends_a_running_side_request_at_once(stamper: FrameStamper, clock: FakeClock) -> None:
    order: list[object] = []
    timers = Timers()
    analyzer, images = graced(order, timers)
    side = Abortable(order)
    assert side.submit(analyzer) and side.started.wait(2)
    analyzer.submit(job_for(held_frame(stamper, clock, images), "job-1", incident="inc-1"))
    wait_for(lambda: order[-1:] == ["job-1"])
    assert timers.started == [] and side.aborted == ["a scene job is waiting: monitoring comes first"]
    assert (analyzer.counters["side_preempted"], analyzer.counters["side_preempted_after_grace"]) == (1, 0)
    assert analyzer.stop(2.0)


def test_without_a_grace_a_periodic_job_ends_it_at_once(stamper: FrameStamper, clock: FakeClock) -> None:
    order: list[object] = []
    timers = Timers()
    analyzer, images = graced(order, timers, grace_s=0.0)
    side = Abortable(order)
    assert side.submit(analyzer) and side.started.wait(2)
    analyzer.submit(job_for(held_frame(stamper, clock, images), "job-1"))
    wait_for(lambda: order[-1:] == ["job-1"])
    assert timers.started == [] and len(side.aborted) == 1
    with pytest.raises(ValueError, match="negative"):
        ThreadedSceneAnalyzer(lambda job, image: None, RecentImages(), revision="1", side_grace_s=-0.1)
    assert analyzer.stop(2.0)


def test_a_late_grace_timer_never_ends_a_later_side_request(stamper: FrameStamper, clock: FakeClock) -> None:
    order: list[object] = []
    timers = Timers()
    analyzer, images = graced(order, timers)
    first = Abortable(order)
    assert first.submit(analyzer) and first.started.wait(2)
    analyzer.submit(job_for(held_frame(stamper, clock, images), "job-1"))
    first.release.set()
    wait_for(lambda: order[-1:] == ["job-1"] and not analyzer.busy())
    second = Abortable(order)
    assert second.submit(analyzer) and second.started.wait(2)
    analyzer.submit(job_for(held_frame(stamper, clock, images), "job-2"))  # job-2 waits for the second request
    timers.fire(0, even_if_cancelled=True)  # the first request's timer, late
    assert second.aborted == []
    timers.fire(1)  # the second request's own grace
    wait_for(lambda: order[-1:] == ["job-2"])
    assert second.aborted == ["a scene job is waiting: monitoring comes first"] and first.aborted == []
    assert analyzer.stop(2.0)


def test_the_grace_ends_nothing_once_no_scene_job_waits_a_later_job_has_its_own_and_stop_cancels_it(
    stamper: FrameStamper, clock: FakeClock
) -> None:
    order: list[object] = []
    timers = Timers()
    analyzer, images = graced(order, timers)
    side = Abortable(order)
    assert side.submit(analyzer) and side.started.wait(2)
    analyzer.submit(job_for(held_frame(stamper, clock, images), "job-1"))
    analyzer.cancel("job-1")  # the lane gave the waiting job up at its deadline
    timers.fire()
    assert side.aborted == []  # nothing waits: the question goes on
    analyzer.submit(job_for(held_frame(stamper, clock, images), "job-2"))
    assert len(timers.started) == 2  # a later job's own grace, from its arrival
    analyzer.submit(job_for(held_frame(stamper, clock, images), "job-3"))  # replaces job-2 within that grace
    assert len(timers.started) == 2  # the earlier deadline stands
    assert analyzer.stop(2.0)
    assert timers.started[1][2] is True and side.aborted == ["the runtime is stopping"]


def test_a_pending_side_request_can_be_withdrawn_and_stop_tells_it(stamper: FrameStamper, clock: FakeClock) -> None:
    analyzer = make(Gate())
    withdrawn = Side()
    with Asleep(analyzer):
        assert withdrawn.submit(analyzer)
    assert analyzer.cancel_side() is True and analyzer.cancel_side() is False and not analyzer.busy()
    dropped = Side()
    with Asleep(analyzer):
        assert dropped.submit(analyzer)
    assert analyzer.stop(2.0)
    assert withdrawn.log == [] and len(dropped.log) == 1 and str(dropped.log[0]) == "stopped"
    assert isinstance(dropped.log[0], RuntimeError) and analyzer.counters["side_dropped_at_stop"] == 1
    assert Side().submit(analyzer) is False  # after stop


def test_a_failing_side_request_or_callback_never_stops_the_worker(stamper: FrameStamper, clock: FakeClock) -> None:
    images = RecentImages()
    analyzer = make(Gate(), images)
    failing = Side(result=RuntimeError("model gone at /home/user/secret"))
    assert failing.submit(analyzer) and failing.finished.wait(2)
    assert isinstance(failing.log[1], RuntimeError)  # the request's own error, handed to its caller
    wait_for(lambda: not analyzer.busy())

    def broken(result: object) -> None:
        raise ValueError("callback bug")

    assert analyzer.submit_side(lambda: "x", started=lambda: None, finished=broken)
    wait_for(lambda: analyzer.counters["side_callback_failed"] == 1 and not analyzer.busy())
    frame = frame_at(stamper, clock)
    images.put(frame.key, "x")
    analyzer.submit(job_for(frame))
    ((job_id, outcome),) = drained(analyzer)
    assert (job_id, outcome.kind) == ("job-1", OutcomeKind.COMPLETED)  # scene analysis goes on
    assert analyzer.stop(2.0)


def test_submit_before_start_is_refused(stamper: FrameStamper, clock: FakeClock) -> None:
    analyzer = ThreadedSceneAnalyzer(Gate(), RecentImages(), revision="1")
    with pytest.raises(RuntimeError, match="not running"):
        analyzer.submit(job_for(frame_at(stamper, clock)))
    assert analyzer.stop(0.1)


# --- behind EdgeCore: on time, late, stale -------------------------------------------------------

class Demo:
    """EdgeCore with the real llama request code over a gated fake transport, driven like D-1's loop."""

    def __init__(self, clock: FakeClock, stamper: FrameStamper) -> None:
        self.clock, self.stamper = clock, stamper
        self.images = RecentImages()
        self.transport = GatedTransport()
        self.analyzer = make(LlamaSceneRequest(self.transport, encode=lambda image: b"jpeg"), self.images)
        config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
        self.core = EdgeCore(config, clock, self.analyzer, scene_id_prefix="job")

    def frame(self) -> tuple[FrameRef, object]:
        frame = frame_at(self.stamper, self.clock)
        self.images.put(frame.key, "pixels")
        return frame, self.core.on_frame(frame, [], self.stamper.current_stream)

    def deliver(self) -> object:
        ((job_id, outcome),) = drained(self.analyzer)
        return self.core.on_scene_outcome(job_id, outcome, self.stamper.current_stream)


def test_an_on_time_report_becomes_current_scene_state(clock: FakeClock, stamper: FrameStamper) -> None:
    demo = Demo(clock, stamper)
    demo.frame()  # the first fresh frame starts a periodic scene check

    out = demo.deliver()

    ((routed),) = out.evidence
    assert routed.evidence.evidence_id == "job-1.observed" and routed.routing.updates_current
    assert out.state.scene is SceneStatus.REPORTED and out.state.scene_report.summary == "An empty room."
    assert demo.transport.posts == 1
    demo.analyzer.stop(2.0)


def test_a_late_result_annotates_its_incident_and_never_becomes_current(clock: FakeClock, stamper: FrameStamper) -> None:
    demo = Demo(clock, stamper)
    frame, _ = demo.frame()
    demo.deliver()  # periodic job-1, on time
    demo.transport.release.clear()
    demo.core.request_enrichment(frame, "inc-7", stamper.current_stream)  # job-2 starts at once
    wait_for(lambda: demo.transport.posts == 2)

    clock.advance(8.0)  # past job-2's deadline (8 s from its frame) with the request still running
    timed_out = demo.core.tick(stamper.current_stream)
    demo.transport.release.set()
    late = demo.deliver()

    assert [r.evidence.evidence_id for r in timed_out.evidence] == ["job-2.timeout"]
    assert demo.analyzer.counters["cancelled_running"] == 1
    ((routed),) = late.evidence
    assert routed.evidence.evidence_id == "job-2.observed.late"
    assert (routed.routing.annotates, routed.routing.updates_current) == ("inc-7", False)
    assert late.state.scene is not SceneStatus.REPORTED
    demo.analyzer.stop(2.0)


def test_a_result_for_a_superseded_stream_epoch_is_not_current(clock: FakeClock, stamper: FrameStamper) -> None:
    demo = Demo(clock, stamper)
    demo.transport.release.clear()
    demo.frame()  # job-1 on epoch 1
    wait_for(lambda: demo.transport.posts == 1)
    stamper.disconnect()
    stamper.connect()  # the camera reconnected: epoch 2
    demo.frame()
    demo.transport.release.set()

    out = demo.deliver()  # before its deadline, but about the old epoch's frame

    ((routed),) = out.evidence
    assert routed.evidence.evidence_id == "job-1.observed"
    assert routed.routing.applicability.value == "superseded_epoch" and not routed.routing.updates_current
    assert out.state.scene is not SceneStatus.REPORTED
    demo.analyzer.stop(2.0)


def test_a_failed_request_replaces_the_previous_verdict(clock: FakeClock, stamper: FrameStamper) -> None:
    demo = Demo(clock, stamper)
    demo.frame()
    assert demo.deliver().state.scene is SceneStatus.REPORTED

    def loading(body: bytes) -> bytes:
        raise TransportError("http_503")  # e.g. the server restarted and is loading again

    demo.transport.post = loading  # type: ignore[method-assign]
    for _ in range(61):  # 4 s at 15 fps: the next periodic check
        demo.frame()
    out = demo.deliver()
    assert out.evidence[0].evidence.evidence_id == "job-2.error"
    assert out.state.scene is SceneStatus.UNKNOWN  # never the previous report standing in
    demo.analyzer.stop(2.0)
