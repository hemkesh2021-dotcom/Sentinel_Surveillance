"""V2-05 demo form: bounded single-owner capture, stream epochs and reconnects (no camera, no decoder)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import pytest

from sentinel.cli import main
from sentinel.config import CaptureConfig
from sentinel.contracts import FrameRef, PixelFormat, SourceTimeQuality
from sentinel.media.capture import (
    CapturedFrame,
    CaptureState,
    CaptureWorker,
    DecodedFrame,
    LatestFrame,
    SourceError,
)
from sentinel.media.clock import NS_PER_SECOND, FakeClock, SystemClock
from sentinel.media.frames import FrameNovelty, FrameStamper, frame_novelty
from sentinel.media.opencv_source import RtspEndpoint
from sentinel.media.probe import ProbeStats, established_connections, percentiles_ms, run_probe

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
FRAME_S = 1 / 15


def picture(pts: int | None = None, width: int = 640, height: int = 480) -> DecodedFrame:
    return DecodedFrame(object(), width, height, PixelFormat.BGR, pts)


class ScriptedSource:
    """Plays one script entry per open(): an exception (open fails) or the reads of one connection.

    A read entry is a DecodedFrame (the fake clock advances one 15 fps interval
    first), None (stream ended or read timed out) or an exception to raise.
    After the entries, read() returns None.
    """

    def __init__(self, connections: Iterable[object], clock: FakeClock | None = None) -> None:
        self._connections = list(connections)
        self._clock = clock
        self._reads: list[object] = []
        self.is_open = False
        self.opens = self.closes = 0

    @property
    def exhausted(self) -> bool:
        return not self._connections

    def open(self) -> None:
        assert not self.is_open, "opened twice without close"
        self.opens += 1
        entry = self._connections.pop(0)
        if isinstance(entry, Exception):
            raise entry
        self._reads = list(entry)  # type: ignore[call-overload]
        self.is_open = True

    def read(self) -> DecodedFrame | None:
        assert self.is_open
        if not self._reads:
            return None
        entry = self._reads.pop(0)
        if isinstance(entry, Exception):
            raise entry
        if entry is not None and self._clock is not None:
            self._clock.advance(FRAME_S)
        return entry  # type: ignore[return-value]

    def close(self) -> None:
        self.closes += int(self.is_open)
        self.is_open = False


class RecordingSlot(LatestFrame):
    def __init__(self) -> None:
        super().__init__()
        self.frames: list[FrameRef] = []

    def publish(self, captured: CapturedFrame) -> None:
        self.frames.append(captured.frame)
        super().publish(captured)


def make_worker(
    source: ScriptedSource, clock: FakeClock, config: CaptureConfig | None = None
) -> tuple[CaptureWorker, RecordingSlot, list[float]]:
    """A worker whose waits are recorded and advance the fake clock; it stops once the script is done."""
    slot = RecordingSlot()
    waits: list[float] = []

    def wait(seconds: float) -> bool:
        waits.append(seconds)
        clock.advance(seconds)
        return source.exhausted

    worker = CaptureWorker(source, FrameStamper("cam-1", clock), slot, config or CaptureConfig(), wait=wait)
    return worker, slot, waits


def test_each_connection_is_a_new_epoch_with_its_own_sequence_and_ingest_times(clock: FakeClock) -> None:
    source = ScriptedSource([[picture(0), picture(6000), picture(12000)], [picture(0), picture(6000)]], clock)
    worker, slot, waits = make_worker(source, clock)
    start = clock.monotonic_ns()

    worker.run()

    frames = slot.frames
    assert [(f.stream_epoch, f.frame_seq) for f in frames] == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    assert len({f.run_id for f in frames}) == 1
    assert all(f.boot_id == clock.boot_id for f in frames)
    # Ingest time is the injected clock at stamping: 1/15 s apart, plus the 1 s reconnect wait.
    step = round(FRAME_S * NS_PER_SECOND)
    offsets = [f.ingest_mono_ns - start for f in frames]
    assert offsets[:3] == [step, 2 * step, 3 * step]
    assert offsets[3] - offsets[2] == NS_PER_SECOND + step
    assert {(f.native_width, f.native_height, f.pixel_format) for f in frames} == {(640, 480, PixelFormat.BGR)}
    # A PTS restart in the new epoch is not compared with the old epoch's PTS.
    assert [f.source_time_quality for f in frames] == [SourceTimeQuality.STREAM_RELATIVE] * 5
    status = worker.status()
    assert (status.state, status.connects, status.stream_ends, status.frames) == (CaptureState.STOPPED, 2, 2, 5)
    assert status.problem == "no_frame"
    assert waits == [1.0, 1.0]
    assert source.opens == source.closes == 2


def test_a_reconnect_withdraws_the_old_epoch_and_drops_its_unconsumed_frame(clock: FakeClock) -> None:
    observed: list[tuple[int | None, int]] = []

    class Watching(ScriptedSource):
        def read(self) -> DecodedFrame | None:
            connected = worker.connected
            observed.append((connected.stream_epoch if connected else None, slot.counts()["pending"]))
            return super().read()

    source = Watching([[picture(1)], [picture(1)]], clock)
    worker, slot, _ = make_worker(source, clock)
    seen_during_wait: list[object] = []
    inner_wait = worker._wait

    def wait(seconds: float) -> bool:
        seen_during_wait.append((worker.connected, worker.status().state, slot.counts()["pending"]))
        return inner_wait(seconds)

    worker._wait = wait
    worker.run()

    # While streaming, connected is the epoch being read; between connections it is None
    # and the old epoch's undelivered frame is gone, so a consumer cannot process it.
    assert observed == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert seen_during_wait == [(None, CaptureState.WAITING, 0)] * 2
    assert slot.counts() == {"published": 2, "delivered": 0, "replaced": 0, "discarded": 2, "pending": 0}


def test_a_frame_taken_before_a_reconnect_is_not_live_afterwards(clock: FakeClock) -> None:
    taken: list[CapturedFrame] = []

    class Consuming(ScriptedSource):
        def read(self) -> DecodedFrame | None:
            if slot.counts()["pending"] and not taken:
                taken.append(slot.take(timeout_s=0))
            return super().read()

    source = Consuming([[picture(1), picture(2)], [picture(1)]], clock)
    worker, slot, _ = make_worker(source, clock)
    worker.run()

    stale = taken[0].frame
    final = slot.frames[-1]
    assert (stale.stream_epoch, final.stream_epoch) == (1, 2)
    assert frame_novelty(stale.key, None, final.stream) is FrameNovelty.NOT_LIVE


def test_backoff_doubles_while_connections_deliver_nothing_and_resets_after_frames(clock: FakeClock) -> None:
    refused = SourceError("open_failed")
    config = CaptureConfig(reconnect_initial_s=1.0, reconnect_max_s=5.0)
    source = ScriptedSource(
        [refused, refused, [], refused, refused, [picture(1)], refused, refused], clock
    )
    worker, _, waits = make_worker(source, clock, config)

    worker.run()

    # Bounded: 1, 2, 4, then capped at 5; one delivered frame restores the initial wait.
    assert waits == [1.0, 2.0, 4.0, 5.0, 5.0, 1.0, 1.0, 2.0]
    status = worker.status()
    assert (status.open_failures, status.connects, status.stream_ends) == (6, 2, 2)
    assert status.problem == "open_failed"
    assert source.closes == 2


def test_a_source_that_fails_half_way_through_open_is_still_closed(clock: FakeClock) -> None:
    class HalfOpen(ScriptedSource):
        def open(self) -> None:
            self.opens += 1
            self.is_open = True  # e.g. a decoder handle exists before negotiation fails
            raise SourceError("open_failed")

    source = HalfOpen([], clock)
    worker = CaptureWorker(source, FrameStamper("cam-1", clock), LatestFrame(), CaptureConfig(), wait=lambda s: True)
    worker.run()
    assert source.opens == source.closes == 1
    assert worker.status().open_failures == 1


def test_problems_are_labels_or_class_names_never_exception_text(clock: FakeClock) -> None:
    secret = "rtsp://admin:hunter2@192.0.2.10:554/live"
    source = ScriptedSource(
        [RuntimeError(secret), [picture(1), ValueError(secret)], SourceError(f"failed {secret}")], clock
    )
    worker, _, _ = make_worker(source, clock)
    problems: list[str | None] = []
    inner_wait = worker._wait
    worker._wait = lambda s: problems.append(worker.status().problem) or inner_wait(s)  # type: ignore[method-assign]

    worker.run()

    assert problems == ["open_error:RuntimeError", "read_error:ValueError", "source_error"]
    assert "hunter2" not in repr(worker.status())


@pytest.mark.parametrize(
    ("reads", "problem"),
    [
        ([picture(1), picture(2, width=1280, height=720)], "frame_size_changed"),
        ([picture(1), picture(2, width=0)], "invalid_frame"),
        ([picture(1), DecodedFrame(object(), 640.0, 480, PixelFormat.BGR)], "invalid_frame"),  # type: ignore[arg-type]
    ],
)
def test_a_geometry_change_or_invalid_frame_ends_the_connection(
    clock: FakeClock, reads: list[object], problem: str
) -> None:
    source = ScriptedSource([reads, [picture(1, width=1280, height=720)]], clock)
    worker, slot, _ = make_worker(source, clock)
    problems: list[str | None] = []
    inner_wait = worker._wait
    worker._wait = lambda s: problems.append(worker.status().problem) or inner_wait(s)  # type: ignore[method-assign]

    worker.run()

    # Boxes and tracks assume one geometry per epoch, so the new size arrives in a new epoch.
    assert [(f.stream_epoch, f.native_width) for f in slot.frames] == [(1, 640), (2, 1280)]
    assert problems == [problem, "no_frame"]
    assert worker.status().stream_ends == 2
    assert source.opens == source.closes == 2


def test_stop_during_a_connection_closes_the_source_and_is_not_a_stream_end(clock: FakeClock) -> None:
    class Stopping(ScriptedSource):
        def read(self) -> DecodedFrame | None:
            if len(slot.frames) == 2:
                worker.request_stop()
            return super().read()

    source = Stopping([[picture(k) for k in range(10)]], clock)
    worker, slot, waits = make_worker(source, clock)

    worker.run()

    assert len(slot.frames) == 3  # the read in progress when stop arrived still completes
    status = worker.status()
    assert (status.state, status.stream, status.stream_ends) == (CaptureState.STOPPED, None, 0)
    assert waits == []
    assert source.opens == source.closes == 1
    started = time.monotonic()
    assert slot.take(timeout_s=5) is None  # the epoch's frame was dropped
    assert time.monotonic() - started < 1  # and the slot is closed: no consumer waits


def test_a_slow_consumer_gets_the_newest_frame_and_at_most_one_waits() -> None:
    slot = LatestFrame()
    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    frames = []
    for _ in range(5):
        clock.advance(FRAME_S)
        frames.append(stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR))
        slot.publish(CapturedFrame(frames[-1], object()))
        assert slot.counts()["pending"] == 1

    taken = slot.take(timeout_s=0)
    assert taken is not None and taken.frame == frames[-1]
    assert slot.take(timeout_s=0) is None  # each frame at most once
    counts = slot.counts()
    assert counts == {"published": 5, "delivered": 1, "replaced": 4, "discarded": 0, "pending": 0}
    assert counts["published"] == counts["delivered"] + counts["replaced"] + counts["discarded"]


def test_take_waits_for_a_frame_and_close_wakes_a_waiting_consumer() -> None:
    slot = LatestFrame()
    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    frame = stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
    results: list[object] = []

    consumer = threading.Thread(target=lambda: results.append(slot.take(timeout_s=10)))
    consumer.start()
    slot.publish(CapturedFrame(frame, object()))
    consumer.join(5)
    assert results and isinstance(results[0], CapturedFrame)

    consumer = threading.Thread(target=lambda: results.append(slot.take(timeout_s=10)))
    started = time.monotonic()
    consumer.start()
    slot.close()
    consumer.join(5)
    assert results[1] is None and time.monotonic() - started < 5
    assert slot.take(timeout_s=0.01) is None


class PacedSource:
    """Real-time fake for thread tests: ~200 fps of tiny frames until closed."""

    endpoint = None

    def __init__(self, *, fail_open: bool = False) -> None:
        self.fail_open = fail_open
        self.opens = self.closes = 0
        self.thread_names: set[str] = set()
        self._pts = 0

    def open(self) -> None:
        self.thread_names.add(threading.current_thread().name)
        self.opens += 1
        if self.fail_open:
            raise SourceError("open_failed")

    def read(self) -> DecodedFrame | None:
        self.thread_names.add(threading.current_thread().name)
        time.sleep(0.005)
        self._pts += 6000
        return picture(self._pts)

    def close(self) -> None:
        self.thread_names.add(threading.current_thread().name)
        self.closes += 1


def test_the_worker_thread_owns_the_source_and_stops_within_its_bound() -> None:
    clock = SystemClock()
    source = PacedSource()
    slot = LatestFrame()
    worker = CaptureWorker(source, FrameStamper("cam-1", clock), slot, CaptureConfig(), name="cap-test")
    worker.start()
    with pytest.raises(RuntimeError):
        worker.start()
    first = slot.take(timeout_s=5)
    assert first is not None and worker.connected == first.frame.stream

    assert worker.stop(timeout_s=5)
    assert source.thread_names == {"cap-test"}  # never touched from the consumer's thread
    assert source.opens == source.closes == 1
    assert worker.status().state is CaptureState.STOPPED
    assert worker.connected is None


def test_stop_interrupts_the_reconnect_wait() -> None:
    clock = SystemClock()
    source = PacedSource(fail_open=True)
    worker = CaptureWorker(
        source, FrameStamper("cam-1", clock), LatestFrame(), CaptureConfig(reconnect_initial_s=30, reconnect_max_s=30)
    )
    worker.start()
    deadline = time.monotonic() + 5
    while worker.status().state is not CaptureState.WAITING and time.monotonic() < deadline:
        time.sleep(0.01)
    started = time.monotonic()
    assert worker.stop(timeout_s=5)
    assert time.monotonic() - started < 2
    assert worker.status().retry_delay_s is None and worker.status().open_failures == 1


def test_a_worker_bug_is_reported_as_failed_with_its_class_name(clock: FakeClock) -> None:
    class Rebooting(FrameStamper):
        """The clock's boot changes right after connect, so stamping refuses (a programming error)."""

        def connect(self):  # type: ignore[no-untyped-def]
            stream = super().connect()
            clock.reboot("fake-boot-2")
            return stream

    source = ScriptedSource([[picture(1)]], clock)
    worker = CaptureWorker(source, Rebooting("cam-1", clock), LatestFrame(), CaptureConfig(), wait=lambda s: True)
    with pytest.raises(RuntimeError):
        worker.run()
    status = worker.status()
    assert (status.state, status.problem) == (CaptureState.FAILED, "worker_error:RuntimeError")
    assert source.opens == source.closes == 1


# ------------------------------------------------------------------ probe


def proc_line(remote: str, state: str = "01") -> str:
    return f"   0: 0F02000A:D2F0 {remote} {state} 00000000:00000000 00:00000000 00000000  1000 0 1 1"


def write_proc(tmp_path: Path, tcp: list[str], tcp6: list[str] = ()) -> Path:  # type: ignore[assignment]
    header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode"
    (tmp_path / "tcp").write_text("\n".join([header, *tcp]) + "\n")
    (tmp_path / "tcp6").write_text("\n".join([header, *tcp6]) + "\n")
    return tmp_path


def test_upstream_connections_count_established_sockets_to_the_camera_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.byteorder", "little")
    camera = "0A0200C0:022A"  # 192.0.2.10:554 as the kernel prints it on a little-endian host
    proc = write_proc(
        tmp_path,
        [
            proc_line(camera),
            proc_line(camera, state="06"),  # TIME_WAIT: a closed earlier session
            proc_line("0A0200C0:0050"),  # same host, port 80
            proc_line("0B0200C0:022A"),  # 192.0.2.11
        ],
        # IPv4-mapped IPv6 (::ffff:192.0.2.10) from a dual-stack socket
        [proc_line("0000000000000000FFFF00000A0200C0:022A")],
    )
    assert established_connections(RtspEndpoint("192.0.2.10", 554), proc) == 2
    assert established_connections(RtspEndpoint("192.0.2.11", 554), proc) == 1
    assert established_connections(RtspEndpoint("10.2.0.192", 554), proc) == 0  # byte order matters
    assert established_connections(RtspEndpoint("camera.local", 554), proc) is None
    assert established_connections(RtspEndpoint("192.0.2.10", 554), tmp_path / "missing") == 0


def test_percentiles_use_nearest_rank_in_milliseconds() -> None:
    assert percentiles_ms([]) == {"count": 0, "p50": None, "p95": None, "max": None}
    values = [k * 1_000_000 for k in range(1, 101)]
    assert percentiles_ms(values) == {"count": 100, "p50": 50.0, "p95": 95.0, "max": 100.0}
    assert percentiles_ms([66_666_667]) == {"count": 1, "p50": 66.667, "p95": 66.667, "max": 66.667}


def test_probe_cadence_uses_consecutive_frames_only(clock: FakeClock) -> None:
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    stats = ProbeStats()
    frames = []
    for pts in (None, 3000, 6000, 9000):
        clock.advance(FRAME_S)
        frames.append(stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR, source_pts=pts))
    for frame in (frames[0], frames[1], frames[3]):  # frame 2 was replaced before the consumer took it
        stats.observe(CapturedFrame(frame, object()), frame.ingest_mono_ns + 2_000_000)
    summary = stats.summary()
    assert summary["ingest_interval_ms"]["count"] == 1  # 0->1 only; 1->3 skipped a frame
    assert summary["ingest_interval_ms"]["p50"] == pytest.approx(66.667, abs=0.001)
    assert summary["handoff_age_ms"]["max"] == 2.0
    assert summary["pts_quality"] == {"none": 1, "stream_relative": 2, "capture_synced": 0}
    assert (summary["consumed"], summary["epochs"], summary["native_sizes"]) == (3, 1, [[640, 480]])


def test_probe_reports_numbers_and_labels_and_stops_the_worker(tmp_path: Path) -> None:
    clock = SystemClock()
    source = PacedSource()
    slot = LatestFrame()
    worker = CaptureWorker(source, FrameStamper("cam-1", clock), slot, CaptureConfig())
    proc = write_proc(tmp_path, [])

    summary = run_probe(worker, slot, clock, 0.3, endpoint=RtspEndpoint("192.0.2.10", 554), proc_net=proc)

    assert summary["status"] == "frames_received"
    assert summary["worker"]["stopped"] and summary["worker"]["state"] == "stopped"
    frames = summary["frames"]
    assert frames["captured"] == frames["delivered"] + frames["replaced"] + frames["discarded"]
    assert summary["upstream_connections"]["status"] == "observed"
    assert summary["upstream_connections"]["before"] == 0
    assert summary["cpu"]["process_s"] >= 0
    assert "192.0.2.10" not in json.dumps(summary)
    assert source.opens == source.closes == 1


def test_cli_probe_prints_a_summary_and_refuses_without_a_url(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["capture", "probe", str(DEFAULT_CONFIG), "--seconds", "1"], capture_source=lambda c: PacedSource()) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "frames_received"
    assert summary["upstream_connections"] == {"status": "not_applicable"}
    assert summary["settings"]["decode_threads"] == 1

    assert main(["capture", "probe", str(DEFAULT_CONFIG), "--seconds", "1"], capture_source=lambda c: PacedSource(fail_open=True)) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "no_frames"

    monkeypatch.delenv("SENTINEL_RTSP_URL", raising=False)
    assert main(["capture", "probe", str(DEFAULT_CONFIG)]) == 1
    assert capsys.readouterr().err.strip() == "capture probe: rtsp_url_missing"

    with pytest.raises(SystemExit):
        main(["capture", "probe", str(DEFAULT_CONFIG), "--seconds", "301"])
