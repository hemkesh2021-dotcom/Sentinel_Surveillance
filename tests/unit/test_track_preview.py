"""P1 (session 26): the operator preview and the opt-in box summary of `sentinel track probe`.

Synthetic frames and fake tracker results only: no camera, GPU, model, cv2 or
saved image. The renderer is a fake that returns the update's numbers as JSON,
so a test can check which frame and which boxes a viewer received.
"""

from __future__ import annotations

import contextlib
import gc
import json
import signal
import socket
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from sentinel.cli import main
from sentinel.contracts import PixelFormat, TrackStatus
from sentinel.media.capture import CapturedFrame, DecodedFrame
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.tracking.box_summary import BoxSummary
from sentinel.tracking.preview import (
    MAX_FPS,
    TOKEN_ENV,
    PreviewBox,
    PreviewFeed,
    PreviewServer,
    PreviewUpdate,
    box_label,
    header_text,
    pixel_box,
    preview_update,
)
from sentinel.tracking.probe import TrackProbe
from sentinel.tracking.tracker import PersonTracker, RawTrack, TrackerError

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
PLENTY = {"MemFree": 4_000_000_000, "MemAvailable": 5_000_000_000}
TOKEN = "Pv7_q-Zr2LkW9xYtB4nH0s"  # 22 characters, synthetic


class Image:
    """A stand-in for a decoded frame that can be weakly referenced."""

    shape = (480, 640, 3)

    def __init__(self, ident: int) -> None:
        self.ident = ident


class ScriptedBackend:
    def __init__(self, script: list[object]) -> None:
        self.script = list(script)

    def load(self) -> None:
        pass

    def track(self, image: object) -> Sequence[RawTrack]:
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]

    def reset(self) -> None:
        pass


def json_render(update: PreviewUpdate) -> bytes:
    return json.dumps({
        "epoch": update.stream_epoch, "seq": update.frame_seq, "image": getattr(update.image, "ident", None),
        "failed": update.failed,
        "boxes": [[b.track_id, round(b.confidence, 3), b.confirmed, list(pixel_box(b, update.width, update.height))]
                  for b in update.boxes],
    }).encode()


def update_for(seq: int, *, boxes: tuple[PreviewBox, ...] = (), image: Any = None) -> PreviewUpdate:
    from datetime import datetime, timezone

    return PreviewUpdate(image=image if image is not None else Image(seq), width=640, height=480, stream_epoch=1,
                         frame_seq=seq, ingest_utc=datetime(2026, 1, 1, tzinfo=timezone.utc), failed=False,
                         boxes=boxes)


def box(track_id: int, confidence: float = 0.8, confirmed: bool = True) -> PreviewBox:
    return PreviewBox(track_id, 0.1, 0.1, 0.5, 1.0, confidence, confirmed)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextlib.contextmanager
def serving(feed: PreviewFeed, **options: Any) -> Iterator[PreviewServer]:
    server = PreviewServer(0, feed, TOKEN, render=options.pop("render", json_render), **options)
    server.start()
    try:
        yield server
    finally:
        server.stop(3.0)


@contextlib.contextmanager
def connection(port: int, path: str, *, host: str | None = None, method: str = "GET",
               rcvbuf: int | None = None) -> Iterator[tuple[socket.socket, Any]]:
    sock = socket.socket()
    if rcvbuf is not None:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    sock.settimeout(5.0)
    try:
        sock.connect(("127.0.0.1", port))
        sock.sendall(f"{method} {path} HTTP/1.1\r\nHost: {host or f'127.0.0.1:{port}'}\r\n\r\n".encode())
        with sock.makefile("rb") as stream:
            yield sock, stream
    finally:
        sock.close()


def response_head(stream: Any) -> tuple[int, dict[str, str]]:
    code = int(stream.readline().split()[1])
    headers = {}
    while (line := stream.readline()) not in (b"\r\n", b""):
        name, _, value = line.decode().partition(":")
        headers[name.strip().lower()] = value.strip()
    return code, headers


def request(port: int, path: str, **options: Any) -> tuple[int, dict[str, str], bytes]:
    with connection(port, path, **options) as (_, stream):
        code, headers = response_head(stream)
        return code, headers, stream.read()


def read_part(stream: Any) -> bytes:
    assert stream.readline() == b"--frame\r\n"
    headers = {}
    while (line := stream.readline()) not in (b"\r\n", b""):
        name, _, value = line.decode().partition(":")
        headers[name.strip().lower()] = value.strip()
    assert headers["content-type"] == "image/jpeg"
    body = stream.read(int(headers["content-length"]))
    assert stream.read(2) == b"\r\n"
    return body


def wait_until(condition: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@contextlib.contextmanager
def publishing(feed: PreviewFeed, make: Callable[[int], PreviewUpdate], interval_s: float = 0.02,
               durations: list[float] | None = None) -> Iterator[list[PreviewUpdate]]:
    """Publish updates from a thread, as the probe would, until the block ends."""
    published: list[PreviewUpdate] = []
    done = threading.Event()

    def run() -> None:
        seq = 0
        while not done.is_set():
            seq += 1
            update = make(seq)
            started = time.perf_counter()
            feed.publish(update)
            if durations is not None:
                durations.append(time.perf_counter() - started)
            published.append(update)
            done.wait(interval_s)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield published
    finally:
        done.set()
        thread.join(5.0)


# ------------------------------------------------------------ frame/box correspondence


def probe_with(observers: Sequence[Callable[[CapturedFrame, Any], None]], script: list[object]) -> tuple[
        TrackProbe, list[CapturedFrame]]:
    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    probe = TrackProbe(PersonTracker(ScriptedBackend(script)), clock, observers)
    captured = []
    for index in range(len(script)):
        clock.advance(0.066)
        frame = stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        captured.append(CapturedFrame(frame, Image(index)))
    for item in captured:
        probe.observe(item)
    probe.observe(captured[2])  # a repeat: skipped, so neither counted nor shown
    return probe, captured


def test_each_update_is_the_counted_result_on_the_frame_it_was_computed_from() -> None:
    a, b = RawTrack(7, 64, 48, 320, 480, 0.81), RawTrack(9, 400, 0, 600, 240, 0.45)
    script: list[object] = [[], [a], [a, b], RuntimeError("private detail"), [a], [b]]
    seen: list[tuple[CapturedFrame, Any, PreviewUpdate | None]] = []
    feed = PreviewFeed()
    probe, captured = probe_with([lambda c, r: seen.append((c, r, preview_update(c, r))), feed.observe], script)

    assert len(seen) == 6 and feed.counts()["published"] == 6  # the skipped repeat was not offered
    for (frame, result, update), original in zip(seen, captured):
        assert frame is original and update is not None
        assert update.image is original.image  # the very object the tracker processed
        assert (update.stream_epoch, update.frame_seq) == (original.frame.stream_epoch, original.frame.frame_seq)
        assert update.ingest_utc == original.frame.ingest_utc and update.failed == (result.problem is not None)
        assert [(x.track_id, x.confidence, x.confirmed, (x.x1, x.y1, x.x2, x.y2)) for x in update.boxes] == [
            (p.track_id, p.detector_confidence, p.status is TrackStatus.CONFIRMED, (p.box.x1, p.box.y1, p.box.x2,
                                                                                    p.box.y2))
            for p in result.persons]
    shown = [update for _, _, update in seen if update is not None]
    persons = probe.summary()["persons"]
    assert persons["frames_with_persons"] == sum(bool(u.boxes) for u in shown) == 4
    assert persons["max_per_frame"] == max(len(u.boxes) for u in shown) == 2
    assert shown[3].failed and shown[3].boxes == ()
    # Pixel positions come back to the tracker's clipped pixels.
    assert pixel_box(shown[2].boxes[0], 640, 480) == (64, 48, 320, 479)
    assert pixel_box(shown[2].boxes[1], 640, 480) == (400, 0, 600, 240)


def test_observers_do_not_change_the_counts() -> None:
    a = RawTrack(7, 64, 48, 320, 480, 0.81)
    script: list[object] = [[], [a], [a, RawTrack(9, 400, 0, 600, 240, 0.45)], RuntimeError("x"), [a]]
    plain, _ = probe_with([], list(script))
    observed, _ = probe_with([PreviewFeed().observe, BoxSummary().observe], list(script))
    without = {k: v for k, v in plain.summary().items() if not k.endswith("_ms")}
    with_observers = {k: v for k, v in observed.summary().items() if not k.endswith("_ms")}
    assert without == with_observers


def test_labels_and_header_show_id_confidence_state_epoch_and_sequence() -> None:
    assert box_label(box(12, 0.456, True)) == "#12 0.46 C"
    assert box_label(box(3, 0.4, False)) == "#3 0.40 T"
    update = update_for(42, boxes=(box(1), box(2)))
    assert header_text(update) == "epoch 1 seq 42 00:00:00.000Z boxes 2"
    failed = PreviewUpdate(**{**update.__dict__, "failed": True, "boxes": ()})
    assert header_text(failed).endswith("boxes 0 FAILED")
    assert pixel_box(PreviewBox(1, 0.0, 0.0, 1.0, 1.0, 0.5, True), 640, 480) == (0, 0, 639, 479)


# ------------------------------------------------------------ bounded retention


def test_the_feed_keeps_one_update_and_releases_older_images() -> None:
    feed = PreviewFeed()
    images = [Image(i) for i in range(200)]
    refs = [weakref.ref(image) for image in images]
    for seq, image in enumerate(images):
        feed.publish(update_for(seq, image=image))
    del images, image
    gc.collect()
    assert [ref() is not None for ref in refs].count(True) == 1 and refs[-1]() is not None
    assert feed.counts() == {"published": 200, "shown": 0, "dropped": 199, "pending": 1}
    taken = feed.take(0.0)
    assert taken is not None and taken.frame_seq == 199 and feed.take(0.0) is None
    feed.publish(update_for(500))
    feed.close()  # releases the pending image
    feed.publish(update_for(501))  # ignored after close
    assert feed.counts() == {"published": 201, "shown": 1, "dropped": 200, "pending": 0}
    del taken
    gc.collect()
    assert all(ref() is None for ref in refs)


# ------------------------------------------------------------ viewers


def test_a_viewer_receives_counted_boxes_on_their_frames_at_most_five_per_second() -> None:
    feed = PreviewFeed()
    boxes = {seq: (box(seq % 7, 0.5 + (seq % 5) / 10, seq % 2 == 0),) for seq in range(1, 1000)}
    with serving(feed) as server, publishing(feed, lambda s: update_for(s, boxes=boxes[s])) as published:
        port = server.address[1]
        with connection(port, f"/{TOKEN}/stream.mjpg") as (_, stream):
            code, headers = response_head(stream)
            assert code == 200 and headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"
            assert headers["cache-control"] == "no-store"
            parts, times = [], []
            for _ in range(4):
                parts.append(json.loads(read_part(stream)))
                times.append(time.monotonic())
        assert wait_until(lambda: server.summary()["viewers"]["disconnected"] == 1)
    by_seq = {u.frame_seq: u for u in published}
    seqs = [part["seq"] for part in parts]
    assert seqs == sorted(set(seqs))
    for part in parts:
        update = by_seq[part["seq"]]
        assert part["image"] == update.image.ident == part["seq"]
        assert part["boxes"] == [[b.track_id, round(b.confidence, 3), b.confirmed, list(pixel_box(b, 640, 480))]
                                 for b in update.boxes]
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    assert min(gaps) >= 1 / MAX_FPS - 0.05  # the cap, with scheduling slack
    summary = server.summary()
    assert summary["frames_sent"] >= 4 and summary["updates"]["dropped"] > 0  # newer updates replace, never queue
    updates = summary["updates"]
    assert updates["published"] == updates["shown"] + updates["dropped"] + updates["pending"]


def test_a_stalled_viewer_never_blocks_publishing_and_is_disconnected() -> None:
    feed = PreviewFeed()
    big = b"x" * 1_000_000
    durations: list[float] = []
    with serving(feed, render=lambda update: big, write_timeout_s=0.5, send_buffer_bytes=16_384) as server, \
            publishing(feed, update_for, durations=durations):
        port = server.address[1]
        with connection(port, f"/{TOKEN}/stream.mjpg", rcvbuf=4096):  # sends the request, then never reads
            assert wait_until(lambda: server.summary()["viewers"]["write_timeouts"] == 1)
            # The slot is free again while the stalled socket is still open.
            with connection(port, f"/{TOKEN}/stream.mjpg") as (_, stream):
                assert response_head(stream)[0] == 200 and read_part(stream) == big
        assert wait_until(lambda: server.summary()["handlers_open"] == 0)
    assert len(durations) > 20 and max(durations) < 0.05  # publishing never waited on a viewer
    assert feed.counts()["pending"] == 0 and server.summary()["closed"]


def test_a_disconnected_viewer_frees_the_slot_for_the_next() -> None:
    feed = PreviewFeed()
    with serving(feed) as server, publishing(feed, update_for):
        port = server.address[1]
        with connection(port, f"/{TOKEN}/stream.mjpg") as (_, stream):
            assert response_head(stream)[0] == 200
            read_part(stream)
        assert wait_until(lambda: server.summary()["viewers"]["disconnected"] == 1)
        with connection(port, f"/{TOKEN}/stream.mjpg") as (_, stream):
            assert response_head(stream)[0] == 200
            assert json.loads(read_part(stream))["seq"] > 0
    viewers = server.summary()["viewers"]
    assert viewers["served"] == 2 and viewers["refused_busy"] == 0


# ------------------------------------------------------------ access


def test_the_preview_refuses_other_hosts_bad_tokens_and_faster_rates() -> None:
    feed = PreviewFeed()
    for host in ("0.0.0.0", "::", "192.0.2.1", "localhost"):
        with pytest.raises(ValueError, match="binds 127.0.0.1 only"):
            PreviewServer(0, feed, TOKEN, host=host)
    for token in ("", "short", "a" * 21, "a" * 129, "has space in it xxxxxxxx", "slash/in/the/token/xxxxx"):
        with pytest.raises(ValueError, match="token"):
            PreviewServer(0, feed, token)
    with pytest.raises(ValueError, match="max_fps"):
        PreviewServer(0, feed, TOKEN, max_fps=MAX_FPS + 1)


def test_requests_need_the_loopback_host_header_and_the_token(capsys: pytest.CaptureFixture[str]) -> None:
    feed = PreviewFeed()
    with serving(feed) as server, publishing(feed, update_for):
        port = server.address[1]
        assert server.address[0] == "127.0.0.1"
        assert request(port, f"/{TOKEN}/", host=f"evil.example:{port}")[0] == 421
        assert request(port, f"/{TOKEN}/", host="127.0.0.1")[0] == 421  # the port must match too
        for path in ("/", f"/{TOKEN}", f"/{TOKEN}/other", f"/{TOKEN[:-1]}x/", f"/{TOKEN[:-1]}x/stream.mjpg",
                     f"/{TOKEN}x/", "/../" + TOKEN + "/"):
            assert request(port, path)[0] == 404, path
        assert request(port, f"/{TOKEN}/", method="POST")[0] == 405
        assert request(port, f"/{TOKEN}/", method="HEAD")[0] == 405
        code, headers, body = request(port, f"/{TOKEN}/?q=1", host=f"localhost:{port}")
        assert code == 200 and b'src="stream.mjpg"' in body
        assert headers["cache-control"] == "no-store" and headers["x-frame-options"] == "DENY"
        assert headers["referrer-policy"] == "no-referrer" and "img-src 'self'" in headers["content-security-policy"]
        with connection(port, f"/{TOKEN}/stream.mjpg") as (_, first):
            assert response_head(first)[0] == 200
            assert request(port, f"/{TOKEN}/stream.mjpg")[0] == 503  # one viewer at a time
    summary = server.summary()
    assert summary["requests"]["refused_host"] == 2 and summary["requests"]["not_found"] == 7
    assert summary["requests"]["refused_method"] == 2 and summary["viewers"]["refused_busy"] == 1
    output = capsys.readouterr()
    assert output.out == "" and output.err == ""  # no request logging
    assert TOKEN not in json.dumps(summary)


# ------------------------------------------------------------ shutdown


def test_stop_ends_an_open_stream_and_closes_the_port() -> None:
    feed = PreviewFeed()
    server = PreviewServer(0, feed, TOKEN, render=json_render)
    server.start()
    port = server.address[1]
    with publishing(feed, update_for), connection(port, f"/{TOKEN}/stream.mjpg") as (_, stream):
        assert response_head(stream)[0] == 200
        read_part(stream)
        assert server.stop(3.0) is True
        with contextlib.suppress(ConnectionResetError):
            while stream.readline():  # the stream ends
                pass
    with pytest.raises(ConnectionRefusedError), socket.create_connection(("127.0.0.1", port), timeout=1):
        pass
    summary = server.summary()
    assert summary["closed"] and summary["handlers_open"] == 0 and summary["viewers"]["ended_by_stop"] == 1
    assert server.stop(1.0) is True  # idempotent

    never_started = PreviewServer(0, PreviewFeed(), TOKEN)
    unused = never_started.address[1]
    assert never_started.stop(1.0) is True
    with pytest.raises(ConnectionRefusedError), socket.create_connection(("127.0.0.1", unused), timeout=1):
        pass


def test_stop_closes_a_connection_that_never_sent_a_request() -> None:
    feed = PreviewFeed()
    server = PreviewServer(0, feed, TOKEN, render=json_render, write_timeout_s=30)
    server.start()
    with socket.create_connection(server.address, timeout=5) as idle:
        assert wait_until(lambda: server.summary()["handlers_open"] == 1)
        started = time.monotonic()
        assert server.stop(3.0) is True
        assert time.monotonic() - started < 2.0  # not the 30 s request timeout
        assert idle.recv(1) == b""
    assert server.summary()["handlers_open"] == 0


def test_a_failing_renderer_is_counted_and_shows_nothing() -> None:
    def broken(update: PreviewUpdate) -> bytes:
        raise RuntimeError("private detail")

    feed = PreviewFeed()
    with serving(feed, render=broken) as server, publishing(feed, update_for):
        with connection(server.address[1], f"/{TOKEN}/stream.mjpg") as (_, stream):
            assert response_head(stream)[0] == 200
            assert wait_until(lambda: server.summary()["render_failures"] >= 2)
    summary = server.summary()
    assert summary["frames_sent"] == 0 and "private" not in json.dumps(summary)


# ------------------------------------------------------------ box summary


def test_the_box_summary_is_bounded_and_summarizes_the_counted_observations() -> None:
    a, b, c = RawTrack(1, 64, 48, 320, 480, 0.5), RawTrack(2, 0, 0, 64, 48, 0.9), RawTrack(3, 10, 10, 20, 20, 0.6)
    a2 = RawTrack(1, 128, 96, 384, 432, 0.7)
    boxes = BoxSummary(max_tracks=2)
    probe, captured = probe_with([boxes.observe], [[a], [a2, b], [], [a, c], [c]])
    result = boxes.summary()["track_boxes"]
    assert result["max_tracks"] == 2 and result["observations_not_summarized"] == 2  # track 3's two
    first, second = result["tracks"]
    epoch = captured[0].frame.stream_epoch
    assert first["track"] == [epoch, 1] and second["track"] == [epoch, 2]
    assert (first["frames"], first["confirmed_frames"]) == (3, 2)  # confirmed from its second detection
    assert first["confidence"] == {"min": 0.5, "mean": 0.567, "max": 0.7}
    assert first["box_first"] == first["box_last"] == [0.1, 0.1, 0.5, 1.0]
    assert first["box_union"] == [0.1, 0.1, 0.6, 1.0] and first["box_mean"] == [0.133, 0.133, 0.533, 0.967]
    assert first["first_utc"].endswith("Z") and first["first_utc"] < first["last_utc"]
    assert second["frames"] == 1 and second["box_first"] == [0.0, 0.0, 0.1, 0.1]
    assert probe.summary()["persons"]["frames_with_persons"] == 4  # the summary changes no count

    def numbers_only(value: Any) -> bool:
        if isinstance(value, dict):
            return all(numbers_only(v) for v in value.values())
        if isinstance(value, list):
            return all(numbers_only(v) for v in value)
        return isinstance(value, (int, float, str))

    assert numbers_only(result)
    with pytest.raises(ValueError):
        BoxSummary(max_tracks=0)


# ------------------------------------------------------------ the command


class PacedSource:
    endpoint = None

    def __init__(self) -> None:
        self.opens = 0
        self.reads = 0

    def open(self) -> None:
        self.opens += 1

    def read(self) -> DecodedFrame:
        time.sleep(0.01)
        self.reads += 1
        return DecodedFrame(Image(self.reads), 640, 480, PixelFormat.BGR, self.reads * 66_667)

    def close(self) -> None:
        pass


class Backend:
    def __init__(self, *, on_load: Callable[[], None] | None = None, on_track: Callable[[], None] | None = None,
                 load_error: TrackerError | None = None) -> None:
        self.on_load, self.on_track, self.load_error = on_load, on_track, load_error
        self.loaded = False

    def load(self) -> None:
        if self.load_error is not None:
            raise self.load_error
        if self.on_load is not None:
            self.on_load()
        self.loaded = True

    def track(self, image: object) -> Sequence[RawTrack]:
        if self.on_track is not None:
            self.on_track()
        return [RawTrack(1, 64, 48, 320, 480, 0.8)]

    def reset(self) -> None:
        pass


def run_command(backend: Backend, port: int, *flags: str, seconds: str = "1") -> tuple[int, PacedSource]:
    source = PacedSource()
    code = main(
        ["track", "probe", str(DEFAULT_CONFIG), "--engine", "/models/yolov8n.engine", "--seconds", seconds, *flags],
        capture_source=lambda config: source,
        tracker_backend=lambda engine: backend,
        meminfo=lambda: PLENTY,
        preview_port=port,
        preview_render=json_render,
    )
    return code, source


def assert_closed(port: int) -> None:
    with pytest.raises(ConnectionRefusedError), socket.create_connection(("127.0.0.1", port), timeout=1):
        pass


def test_the_command_serves_the_preview_and_adds_only_its_blocks(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    port = free_port()
    received: list[dict[str, Any]] = []

    def ready() -> bool:
        try:
            return request(port, f"/{TOKEN}/")[0] == 200
        except OSError:
            return False

    def view() -> None:
        assert wait_until(ready)
        with connection(port, f"/{TOKEN}/stream.mjpg") as (_, stream):
            response_head(stream)
            received.extend(json.loads(read_part(stream)) for _ in range(2))

    viewer = threading.Thread(target=view, daemon=True)
    viewer.start()
    code, source = run_command(Backend(), port, "--preview", "--box-summary", seconds="2")
    viewer.join(5.0)
    output = capsys.readouterr()
    summary = json.loads(output.out)
    assert code == 0 and source.opens == 1
    preview = summary["preview"]
    assert preview["ended_by"] == "duration" and preview["closed"] and preview["handlers_open"] == 0
    assert preview["bind"] == "127.0.0.1" and preview["max_fps"] == MAX_FPS and "not a performance baseline" in \
        preview["note"]
    assert preview["frames_sent"] >= 2 and len(received) == 2
    assert all([b[:2] for b in part["boxes"]] == [[1, 0.8]] for part in received)
    assert summary["track_boxes"]["tracks"][0]["frames"] == summary["persons"]["frames_with_persons"]
    assert TOKEN not in output.out and TOKEN not in output.err
    assert_closed(port)

    code, _ = run_command(Backend(), port)
    plain = json.loads(capsys.readouterr().out)
    assert code == 0 and "preview" not in plain and "track_boxes" not in plain
    assert set(plain) == set(summary) - {"preview", "track_boxes"}


def test_preview_refusals_come_before_the_model_loads(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    backend = Backend()
    code, source = run_command(backend, free_port(), "--preview")
    assert (code, backend.loaded, source.opens) == (1, False, 0)
    assert capsys.readouterr().err.strip() == "track probe: preview_token_missing"

    monkeypatch.setenv(TOKEN_ENV, "not a valid token, but private")
    code, _ = run_command(backend, free_port(), "--preview")
    err = capsys.readouterr().err
    assert code == 1 and not backend.loaded and err.strip() == "track probe: preview_token_invalid"
    assert "private" not in err

    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        code, source = run_command(backend, taken.getsockname()[1], "--preview")
    assert (code, backend.loaded, source.opens) == (1, False, 0)
    assert capsys.readouterr().err.strip() == "track probe: preview_port_unavailable:OSError"


class Boom(BaseException):
    """An exception the tracker boundary does not catch, as an unexpected abort would be."""


def test_the_preview_closes_on_a_load_failure_an_exception_and_a_signal(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    handler = signal.getsignal(signal.SIGINT)

    port = free_port()
    code, source = run_command(Backend(load_error=TrackerError("load_failed", "RuntimeError")), port, "--preview")
    assert code == 1 and source.opens == 0 and capsys.readouterr().err.strip() == "track probe: load_failed (RuntimeError)"
    assert_closed(port) and signal.getsignal(signal.SIGINT) is handler

    def boom() -> None:
        raise Boom

    port = free_port()
    with pytest.raises(Boom):
        run_command(Backend(on_track=boom), port, "--preview")
    assert_closed(port)
    assert signal.getsignal(signal.SIGINT) is handler

    # Ctrl-C during the model load: the camera is never opened; stdout is still one JSON document.
    port = free_port()
    code, source = run_command(Backend(on_load=lambda: signal.raise_signal(signal.SIGINT)), port, "--preview")
    stopped = json.loads(capsys.readouterr().out)
    assert code == 130 and source.opens == 0
    assert stopped["status"] == "interrupted" and stopped["reason"] == "signal_before_capture"
    assert stopped["preview"]["closed"] and stopped["preview"]["ended_by"] == "signal"
    assert_closed(port) and signal.getsignal(signal.SIGINT) is handler

    # Ctrl-C while tracking: the run ends early with its summary, marked as ended by the signal.
    calls = []

    def interrupt_once() -> None:
        calls.append(1)
        if len(calls) == 10:
            signal.raise_signal(signal.SIGINT)

    port = free_port()
    code, source = run_command(Backend(on_track=interrupt_once), port, "--preview", seconds="30")
    summary = json.loads(capsys.readouterr().out)
    assert code == 130 and summary["preview"]["ended_by"] == "signal" and summary["preview"]["closed"]
    assert summary["seconds"] < 5 and summary["tracking"]["processed"] >= 10 and summary["worker"]["stopped"]
    assert_closed(port) and signal.getsignal(signal.SIGINT) is handler
