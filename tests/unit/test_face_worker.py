"""The 1 Hz face worker (V2-25 demo form): scheduling by frame time, private frame copies, explicit bounds, counters."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from sentinel.contracts import NormalizedBox, TrackObservation, TrackStatus
from sentinel.identity.worker import FaceWorker, private_copy

FPS = 1 / 15


class Image:
    """An array stand-in: copy() and setflags() as NumPy has them."""

    def __init__(self, pixels: list[int]) -> None:
        self.pixels = pixels
        self.writeable = True

    def copy(self) -> "Image":
        return Image(list(self.pixels))

    def setflags(self, *, write: bool) -> None:
        self.writeable = write


def person(frame, track: int = 1) -> TrackObservation:
    return TrackObservation.detected(frame, track_id=track, box=NormalizedBox(x1=0.1, y1=0.1, x2=0.4, y2=0.9),
                                     confidence=0.9, status=TrackStatus.CONFIRMED)


def run(faces=(), detected=0, dropped=0):
    return SimpleNamespace(faces=tuple(faces), detected=detected, fallback_dropped=dropped)


def wait_for(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.002)


class Gate:
    """An analysis that blocks until released, recording what it saw."""

    def __init__(self) -> None:
        self.entered = threading.Semaphore(0)
        self.release = threading.Semaphore(0)
        self.seen: list[tuple[int, list[int], bool]] = []

    def __call__(self, frame, image):
        self.seen.append((frame.frame_seq, list(image.pixels), image.writeable))
        self.entered.release()
        assert self.release.acquire(timeout=5)
        return run(detected=1)


@pytest.fixture
def gate():
    return Gate()


def test_ticks_follow_frame_time_and_empty_ticks_are_skipped(clock, next_frame) -> None:
    worker = FaceWorker(lambda frame, image: run(), clock)
    offered = []
    for index in range(45):  # 3 s of 15 fps frames; a person from frame 10 (0.67 s) on
        frame = next_frame(dt=FPS)
        offered.append(worker.consider(frame, [person(frame)] if index >= 10 else [], Image([index])))
    assert worker.counters["ticks"] == 3  # at frames 0, 15 and 30: one per second of frame time
    assert worker.counters["skipped_no_person"] == 1 and worker.counters["offered"] == 2
    assert [o for o in offered if o is not None] == ["skipped_no_person", "offered", "offered"]
    assert offered.count(None) == 42  # not a tick


def test_the_worker_analyses_a_private_read_only_copy(clock, next_frame, gate) -> None:
    worker = FaceWorker(gate, clock)
    worker.start()
    frame, image = next_frame(), Image([1, 2, 3])
    worker.consider(frame, [person(frame)], image)
    assert gate.entered.acquire(timeout=5)
    image.pixels[0] = 99  # the loop's array changes (e.g. reused) while the face model runs
    gate.release.release()
    wait_for(lambda: worker.counters["completed"] == 1)
    assert gate.seen == [(frame.frame_seq, [1, 2, 3], False)] and image.writeable is True
    result = worker.drain()
    assert result is not None and result.frame == frame and result.persons == (person(frame),)
    assert result.error is None and not hasattr(result, "image")
    assert worker.stop(5.0)


def test_one_job_runs_one_waits_and_a_newer_offer_replaces_the_waiting_one(clock, next_frame, gate) -> None:
    worker = FaceWorker(gate, clock)
    worker.start()
    frames = []
    for _ in range(3):
        frame = next_frame(dt=1.0)
        frames.append(frame)
        worker.consider(frame, [person(frame)], Image([frame.frame_seq]))
        if len(frames) == 1:
            assert gate.entered.acquire(timeout=5)  # the first runs; the next two wait in turn
    assert worker.counters["replaced"] == 1 and worker.status()["waiting"] is True
    clock.advance(0.5)
    gate.release.release()
    assert gate.entered.acquire(timeout=5)
    gate.release.release()
    wait_for(lambda: worker.counters["completed"] == 2)
    assert [seq for seq, _, _ in gate.seen] == [frames[0].frame_seq, frames[2].frame_seq]  # the second never ran
    assert worker.counters["results_replaced"] == 1  # the first result was not taken before the second came
    assert worker.drain().frame == frames[2] and worker.drain() is None
    assert worker.stop(5.0)


def test_a_waiting_job_older_than_its_bound_is_discarded(clock, next_frame, gate) -> None:
    worker = FaceWorker(gate, clock, pending_max_age_ns=2_000_000_000)
    worker.start()
    first = next_frame(dt=1.0)
    worker.consider(first, [person(first)], Image([1]))
    assert gate.entered.acquire(timeout=5)
    second = next_frame(dt=1.0)
    worker.consider(second, [person(second)], Image([2]))
    clock.advance(2.5)  # the running analysis takes long: the waiting frame is now 2.5 s old
    gate.release.release()
    wait_for(lambda: worker.counters["expired_before_start"] == 1)
    assert worker.counters["started"] == 1 and worker.counters["completed"] == 1
    assert worker.stop(5.0)


def test_errors_are_counted_by_class_and_the_worker_continues(clock, next_frame) -> None:
    class Broken(Exception):
        label = "inconsistent_output"

    calls = iter([Broken(), run(faces=["f1", "f2"], detected=2, dropped=1)])

    def analyze(frame, image):
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    worker = FaceWorker(analyze, clock)
    worker.start()
    results = []
    for _ in range(2):
        frame = next_frame(dt=1.0)
        worker.consider(frame, [person(frame)], Image([0]))
        wait_for(lambda: (worker.counters["completed"] + worker.counters["errors"]) == len(results) + 1)
        wait_for(lambda: worker.status()["busy"] is False)
        results.append(worker.drain())
    assert results[0].error == "inconsistent_output:Broken" and results[0].faces is None
    assert results[1].error is None and results[1].faces == ("f1", "f2")
    status = worker.status()
    assert status["counters"]["errors"] == 1 and status["counters"]["error:Broken"] == 1
    assert (status["counters"]["completed"], status["counters"]["faces_detected"], status["counters"]["fallback_dropped"],
            status["counters"]["faces_returned"]) == (1, 2, 1, 2)
    assert status["processing_ms"]["n"] == 1
    assert worker.stop(5.0)


def test_frame_copies_are_released_after_analysis(clock, next_frame) -> None:
    kept = []

    def copy(image):
        copied = private_copy(image)
        kept.append(weakref.ref(copied))
        return copied

    worker = FaceWorker(lambda frame, image: run(), clock, copy=copy)
    worker.start()
    for _ in range(3):
        frame = next_frame(dt=1.0)
        worker.consider(frame, [person(frame)], Image([1]))
        wait_for(lambda: worker.status()["busy"] is False and worker.status()["waiting"] is False)
    assert worker.stop(5.0)
    gc.collect()
    assert len(kept) == 3 and all(ref() is None for ref in kept)


def test_stop_drops_the_waiting_job_and_reports_a_busy_thread(clock, next_frame, gate) -> None:
    worker = FaceWorker(gate, clock)
    worker.start()
    first = next_frame(dt=1.0)
    worker.consider(first, [person(first)], Image([1]))
    assert gate.entered.acquire(timeout=5)
    second = next_frame(dt=1.0)
    worker.consider(second, [person(second)], Image([2]))
    assert worker.stop(0.05) is False  # the running analysis cannot be interrupted
    assert worker.counters["abandoned_at_stop"] == 1
    third = next_frame(dt=1.0)
    assert worker.consider(third, [person(third)], Image([3])) == "stopping"  # nothing new after stop
    gate.release.release()
    assert worker.stop(5.0) is True and worker.status()["state"] == "stopped"
    assert [seq for seq, _, _ in gate.seen] == [first.frame_seq]


def test_settings_must_be_positive(clock) -> None:
    with pytest.raises(ValueError):
        FaceWorker(lambda f, i: run(), clock, interval_ns=0)
