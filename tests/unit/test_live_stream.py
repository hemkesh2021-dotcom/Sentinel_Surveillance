"""D75: continuous live pictures for the status page's player (LivePreview.next_frame, the encoder, GET /live/frame).

No camera, model or cv2: frames come from a real FrameStamper on a FakeClock, images are labelled stand-ins, and the
encoder is a fake that names the image it was given. Covered: one picture per request, the newest live one newer than
the viewer's, never a backlog; a picture whose frame is no longer live is never sent; the encoder runs only while
someone watches, each frame once, at a lower OS priority (checked against this kernel's per-thread nice values), and
yields while monitoring work waits; closing answers waiting requests; the route's token, origin, host, parameters,
caps and headers; a slow viewer holds only its own request.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import socket
import sys
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from sentinel import status_page
from sentinel.config import FreshnessConfig
from sentinel.contracts import PixelFormat
from sentinel.live_preview import (
    ENCODER_NICE,
    PREVIEW_MAX_JPEG_BYTES,
    PREVIEW_YIELD_ENCODES_PER_S,
    LivePreview,
    lower_thread_priority,
)
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.media.jpeg import SOI, EncodedJpeg
from sentinel.media.recent import RecentFrames
from sentinel.status_page import LIVE_FRAME_PATH, StatusServer
from sentinel.storage.database import Database

T0 = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
FRAME_S = 1 / 15
EOI = b"\xff\xd9"
SNAPSHOT: dict[str, Any] = {"runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": T0.isoformat()}}


class Image:
    def __init__(self, label: str) -> None:
        self.label = label
        self.shape = (480, 640, 3)


class Rig:
    """A LivePreview over a real FrameStamper and RecentFrames, on a FakeClock; ``wall`` is the idleness clock."""

    def __init__(self, *, picture_bytes: int = 0, **options: Any) -> None:
        self.clock = FakeClock(utc=T0)
        self.stamper = FrameStamper("cam-1", self.clock)
        self.frames = RecentFrames()
        self.encoded: list[str] = []
        self.picture_bytes = picture_bytes
        self.wall = [0.0]
        options.setdefault("encode", self.encode)
        options.setdefault("wall", lambda: self.wall[0])
        self.preview = LivePreview(self.frames, self.clock, FreshnessConfig(), lambda: self.stamper.current_stream,
                                   **options)
        self.stamper.connect()

    def encode(self, image: Image, pixel_format: PixelFormat, *, max_width: int, max_height: int,
               quality: int) -> EncodedJpeg:
        self.encoded.append(image.label)
        body = image.label.encode() + b"\0" * max(0, self.picture_bytes - len(image.label) - 4)
        return EncodedJpeg(SOI + body + EOI, 640, 480)

    def frame(self, dt: float = FRAME_S):
        self.clock.advance(dt)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        self.frames.put(frame, Image(f"e{frame.stream_epoch}-f{frame.frame_seq}"))
        return frame


def sent(answer) -> str | None:
    return None if answer.picture is None else answer.picture.token


# ---------------------------------------------------------------- what a request gets


def test_a_request_gets_the_newest_live_picture_newer_than_its_own_and_never_a_backlog() -> None:
    rig = Rig(threaded=False)
    rig.frame()
    assert rig.preview.encode_newest() == "encoded"
    first = rig.preview.next_frame(None, 0)
    assert sent(first) == "1-0" and first.age_ms == 0 and first.view is None
    assert rig.preview.next_frame((1, 0), 0).picture is None  # nothing newer yet: no repeat of the one it has
    for _ in range(3):
        rig.frame()
    assert rig.preview.encode_newest() == "encoded"  # the newest only: frames 1 and 2 were never encoded
    answer = rig.preview.next_frame((1, 0), 0)
    assert sent(answer) == "1-3" and answer.age_ms == 0 and rig.encoded == ["e1-f0", "e1-f3"]
    rig.clock.advance(0.5)
    assert rig.preview.next_frame((1, 0), 0).age_ms == 500  # its age when sent
    assert rig.preview.next_frame((1, 3), 0).picture is None
    counters = rig.preview.summary()["counters"]
    assert counters["stream_pictures_sent"] == 3 and counters["stream_no_newer_picture"] == 2


def test_without_a_newer_live_picture_the_answer_is_the_view_with_its_reason() -> None:
    rig = Rig(threaded=False)
    rig.frame()
    rig.preview.encode_newest()
    rig.clock.advance(2.5)  # the camera stalls: the kept picture's frame is no longer live
    answer = rig.preview.next_frame(None, 0)
    assert answer.picture is None and answer.view["state"] == "stale"
    assert answer.view["reason"] == "no new frame for 2.5 s (stale after 2 s)" and answer.view["picture"] is None
    assert rig.preview.encode_newest() == "not_live"  # and the encoder never encodes it again
    rig.stamper.disconnect()
    rig.stamper.connect()
    rig.frame()  # a new epoch: its frames order after every frame of the old one
    rig.preview.encode_newest()
    assert sent(rig.preview.next_frame((1, 99), 0)) == "2-0"


def test_a_frame_from_an_ended_epoch_is_never_sent() -> None:
    rig = Rig(threaded=False)
    rig.frame()
    rig.preview.encode_newest()
    rig.stamper.disconnect()  # a moment later: the picture's epoch has ended
    answer = rig.preview.next_frame(None, 0)
    assert answer.picture is None and answer.view["state"] == "stale"
    assert answer.view["reason"] == "the stream connection this frame came from has ended; reconnecting"


def test_a_waiting_request_gets_a_new_picture_as_soon_as_it_is_made() -> None:
    rig = Rig(threaded=False)
    rig.frame()
    rig.preview.encode_newest()
    got: list[Any] = []
    waiter = threading.Thread(target=lambda: got.append((rig.preview.next_frame((1, 0), 5.0), time.monotonic())))
    waiter.start()
    time.sleep(0.2)
    assert not got  # it waits, without returning the picture it already has
    rig.frame()
    made = time.monotonic()
    rig.preview.encode_newest()
    waiter.join(5.0)
    (answer, arrived), = got
    assert sent(answer) == "1-1" and arrived - made < 0.2


def test_a_waiting_request_gives_up_after_its_wait_and_says_why() -> None:
    rig = Rig(threaded=False)
    rig.frame()
    rig.preview.encode_newest()
    started = time.monotonic()
    answer = rig.preview.next_frame((1, 0), 0.4)
    assert 0.35 < time.monotonic() - started < 1.5
    assert answer.picture is None and answer.view["state"] == "live"  # live, but nothing newer within the wait


# ---------------------------------------------------------------- the encoder thread


def wait_until(predicate: Any, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_nothing_is_encoded_while_nobody_watches() -> None:
    rig = Rig()
    for _ in range(30):
        rig.frame()
    assert rig.encoded == [] and rig.preview.summary()["encoder"] == {"running": False, "nice": None}


def test_a_viewer_starts_the_encoder_which_encodes_frames_as_they_arrive_and_ends_when_nobody_watches() -> None:
    rig = Rig()
    rig.frame()
    try:
        assert sent(rig.preview.next_frame(None, 2.0)) == "1-0"  # the request started the encoder; it made this one
        assert rig.preview.summary()["encoder"]["running"]
        last, received = (1, 0), []
        for _ in range(10):
            rig.frame()
            answer = rig.preview.next_frame(last, 2.0)
            assert answer.picture is not None
            last = answer.picture.order
            received.append(last[1])
        assert received == sorted(set(received)) and received[-1] == 10  # newer each time, ending at the newest
        assert len(rig.encoded) == len(set(rig.encoded))  # each frame once
        rig.wall[0] += 3.5  # nobody has asked for longer than VIEWER_IDLE_S
        assert wait_until(lambda: not rig.preview.summary()["encoder"]["running"], 3.0)
        rig.frame()
        time.sleep(0.4)
        assert "e1-f11" not in rig.encoded  # an idle encoder made nothing
        assert sent(rig.preview.next_frame((1, 10), 2.0)) == "1-11"  # the next viewer starts it again
        assert rig.preview.summary()["counters"]["encoder_starts"] == 2
    finally:
        assert rig.preview.close(2.0)


def test_close_answers_a_waiting_request_and_stops_the_encoder() -> None:
    rig = Rig()
    rig.frame()
    assert rig.preview.next_frame(None, 2.0).picture is not None
    got: list[Any] = []
    waiter = threading.Thread(target=lambda: got.append(rig.preview.next_frame((1, 0), 30.0)))
    waiter.start()
    time.sleep(0.2)
    started = time.monotonic()
    assert rig.preview.close(2.0)
    waiter.join(2.0)
    assert got and got[0].picture is None and time.monotonic() - started < 1.0
    assert not rig.preview.summary()["encoder"]["running"]
    assert rig.preview.next_frame(None, 1.0).view["state"] == "closed"


def nice_of(tid: int) -> int:
    with open(f"/proc/self/task/{tid}/stat") as handle:
        return int(handle.read().rsplit(")", 1)[1].split()[16])  # field 19, counted after the command name


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="per-thread nice values are Linux's")
def test_the_encoder_runs_at_a_lower_priority_and_only_its_own_thread_changes() -> None:
    main_before = nice_of(threading.get_native_id())
    rig = Rig()
    rig.frame()
    try:
        assert rig.preview.next_frame(None, 2.0).picture is not None
        (encoder,) = [t for t in threading.enumerate() if t.name == "sentinel-live-encoder"]
        expected = min(19, main_before + ENCODER_NICE)
        assert nice_of(encoder.native_id) == expected == rig.preview.summary()["encoder"]["nice"]
        assert nice_of(threading.get_native_id()) == main_before  # this kernel keeps nice values per thread
        others = [t for t in threading.enumerate() if t is not encoder and t.native_id is not None]
        assert all(nice_of(t.native_id) == main_before for t in others)
    finally:
        assert rig.preview.close(2.0)


def test_the_main_thread_is_never_reniced() -> None:
    if sys.platform.startswith("linux"):
        before = os.getpriority(os.PRIO_PROCESS, os.getpid())
        assert lower_thread_priority() is None and os.getpriority(os.PRIO_PROCESS, os.getpid()) == before


# ---------------------------------------------------------------- monitoring first


def encode_every_frame(rig: Rig, frames: int) -> list[float]:
    times = []
    for _ in range(frames):
        rig.frame()
        if rig.preview.encode_newest() == "encoded":
            times.append(rig.clock.monotonic_ns() / 1e9)
    return times


def test_while_monitoring_work_waits_the_encoder_yields() -> None:
    waiting = [True]
    rig = Rig(threaded=False, pressure=lambda: waiting[0])
    times = encode_every_frame(rig, 45)  # 3 s at 15 fps
    assert all(sum(1 for t in times if start <= t < start + 1.0) <= PREVIEW_YIELD_ENCODES_PER_S for start in times)
    assert 4 <= len(times) <= 7 and rig.preview.summary()["counters"]["yielded"] == 45
    waiting[0] = False
    rig.wall[0] += 1
    assert len(encode_every_frame(rig, 15)) == 15  # back to every frame once the work is done


def test_pressure_that_cannot_be_read_counts_as_pressure() -> None:
    def broken() -> bool:
        raise RuntimeError("status unavailable")

    rig = Rig(threaded=False, pressure=broken)
    assert len(encode_every_frame(rig, 30)) <= PREVIEW_YIELD_ENCODES_PER_S * 2 + 1


def test_pictures_and_frames_stay_bounded() -> None:
    rig = Rig(threaded=False)
    for _ in range(200):
        rig.frame()
        rig.preview.encode_newest()
    assert rig.preview.summary()["pictures_kept"] == 2 and len(rig.frames) == 2
    assert rig.preview.summary()["encode_ms"]["last"] == 64  # durations kept for the summary are bounded too


# ---------------------------------------------------------------- the route


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    Database.open(tmp_path / "sentinel.db").close()
    return tmp_path / "sentinel.db"


@contextlib.contextmanager
def serving(db_path: Path, preview: Any, **options: Any) -> Iterator[StatusServer]:
    server = StatusServer(0, lambda: SNAPSHOT, db_path, preview=preview, **options)
    server.start()
    try:
        yield server
    finally:
        assert server.stop(3.0)


def get(server: StatusServer, path: str, *, headers: dict[str, str] | None = None, method: str = "GET",
        host: str | None = None) -> tuple[int, dict[str, str], bytes]:
    address, port = server.address
    conn = http.client.HTTPConnection(address, port, timeout=10)
    try:
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", host if host is not None else f"127.0.0.1:{port}")
        for name, value in (headers or {}).items():
            conn.putheader(name, value)
        conn.endheaders()
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def token(server: StatusServer) -> dict[str, str]:
    return {"X-Sentinel-Token": server.chat_token}


def test_a_live_picture_comes_with_its_frame_ingest_time_and_age(db_path: Path) -> None:
    rig = Rig()
    rig.frame()
    with serving(db_path, lambda: rig.preview) as server:
        code, headers, body = get(server, LIVE_FRAME_PATH, headers=token(server))
        assert (code, headers["Content-Type"], body) == (200, "image/jpeg", SOI + b"e1-f0" + EOI)
        assert headers["X-Frame"] == "1-0" and headers["X-Frame-Ingest"] == "2026-10-11T12:00:00.066667Z"
        assert headers["X-Frame-Age-Ms"] == "0" and headers["X-Frame-Size"] == "640x480"
        assert headers["X-Stale-After-Ms"] == "2000" and headers["X-Time-Quality"] == "none"
        assert headers["Cache-Control"] == "no-store" and headers["X-Content-Type-Options"] == "nosniff"
        assert "default-src 'none'" in headers["Content-Security-Policy"]
        rig.frame()
        code, headers, _ = get(server, f"{LIVE_FRAME_PATH}?after=1-0", headers=token(server))
        assert (code, headers["X-Frame"]) == (200, "1-1")
    rig.preview.close(2.0)


@pytest.mark.parametrize(("headers", "host", "code"), [
    ({}, None, 403),
    ({"X-Sentinel-Token": "0" * 32}, None, 403),
    ({"Sec-Fetch-Site": "cross-site"}, None, 403),
    ({"Sec-Fetch-Site": "same-site"}, None, 403),
    ({}, "evil.example:80", 421),
])
def test_only_this_servers_pages_may_ask(db_path: Path, headers: dict[str, str], host: str | None, code: int) -> None:
    rig = Rig()
    rig.frame()
    with serving(db_path, lambda: rig.preview) as server:
        if "Sec-Fetch-Site" in headers:
            headers = {**token(server), **headers}
        status, _, body = get(server, LIVE_FRAME_PATH, headers=headers, host=host)
        assert status == code and server.chat_token.encode() not in body
        if code == 403:
            assert json.loads(body) == {"state": "refused",
                                        "reason": "this page was served by an earlier runtime: reload it"}
            assert server.connection_counts()["stream_refused"] == 1
    assert rig.encoded == []  # a refused request never starts the encoder


@pytest.mark.parametrize("query", ["after=abc", "after=1-", "after=-1-2", "after=1-2-3", "after=%3Cscript%3E"])
def test_after_is_validated_and_never_echoed(db_path: Path, query: str) -> None:
    rig = Rig()
    rig.frame()
    with serving(db_path, lambda: rig.preview) as server:
        status, _, body = get(server, f"{LIVE_FRAME_PATH}?{query}", headers=token(server))
        assert status == 400 and json.loads(body)["state"] == "invalid"
        assert query.split("=", 1)[1].encode() not in body
    rig.preview.close(2.0)


def test_without_a_newer_live_picture_the_route_says_why(db_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(status_page, "STREAM_WAIT_S", 0.3)
    rig = Rig()
    rig.frame()
    with serving(db_path, lambda: rig.preview) as server:
        assert get(server, LIVE_FRAME_PATH, headers=token(server))[0] == 200
        rig.clock.advance(3.0)
        code, headers, body = get(server, f"{LIVE_FRAME_PATH}?after=1-0", headers=token(server))
        document = json.loads(body)
        assert (code, headers["Content-Type"]) == (200, "application/json")
        assert document == {"state": "stale", "reason": "no new frame for 3.0 s (stale after 2 s)", "age_ms": 3000,
                            "frame": document["frame"], "stale_after_ms": 2000}
        assert document["frame"]["frame_seq"] == 0
        code, _, body = get(server, LIVE_FRAME_PATH, headers=token(server))  # not even the kept one: it is stale
        assert code == 200 and json.loads(body)["state"] == "stale"
    rig.preview.close(2.0)


def test_the_route_is_off_without_the_live_view_and_takes_get_only(db_path: Path) -> None:
    with serving(db_path, None) as server:
        code, _, body = get(server, LIVE_FRAME_PATH, headers=token(server))
        assert code == 404 and json.loads(body)["state"] == "off"
    rig = Rig()
    rig.frame()
    with serving(db_path, lambda: rig.preview) as server:
        for method in ("POST", "PUT", "DELETE", "OPTIONS", "HEAD"):
            assert get(server, LIVE_FRAME_PATH, headers=token(server), method=method)[0] == 405, method
    assert rig.encoded == []


def stalled(port: int, path: str, headers: dict[str, str]) -> socket.socket:
    """A request sent and its answer never read."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    sock.settimeout(5.0)
    sock.connect(("127.0.0.1", port))
    extra = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    sock.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{extra}\r\n".encode())
    return sock


def test_live_requests_are_capped_and_the_rest_of_the_page_still_answers(db_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(status_page, "MAX_STREAM_REQUESTS", 2)
    monkeypatch.setattr(status_page, "STREAM_WAIT_S", 1.0)
    rig = Rig()
    rig.frame()
    with serving(db_path, lambda: rig.preview) as server:
        port = server.address[1]
        assert get(server, LIVE_FRAME_PATH, headers=token(server))[0] == 200
        waiting = [stalled(port, f"{LIVE_FRAME_PATH}?after=1-0", token(server)) for _ in range(2)]  # long-polling
        try:
            time.sleep(0.2)
            code, _, body = get(server, f"{LIVE_FRAME_PATH}?after=1-0", headers=token(server))
            assert code == 503 and json.loads(body)["state"] == "busy"
            assert get(server, "/")[0] == 200 and get(server, "/status.json")[0] == 200
            assert server.connection_counts()["stream_busy"] == 1
            time.sleep(1.2)  # their waits end; their slots are free again
            assert get(server, LIVE_FRAME_PATH, headers=token(server))[0] == 200
        finally:
            for sock in waiting:
                sock.close()
    rig.preview.close(2.0)


def test_a_slow_viewer_holds_only_its_own_request_and_never_the_encoder(db_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(status_page, "SEND_BUFFER_BYTES", 4096)
    rig = Rig(picture_bytes=PREVIEW_MAX_JPEG_BYTES)
    rig.frame()
    with serving(db_path, lambda: rig.preview, handler_timeout_s=0.5) as server:
        port = server.address[1]
        slow = stalled(port, LIVE_FRAME_PATH, token(server))  # its 256 KB picture is never read
        try:
            assert wait_until(lambda: server.connection_counts()["write_timeouts"] == 1, 5.0)
            last = (1, 0)
            for _ in range(5):  # meanwhile another viewer gets every new picture
                rig.frame()
                code, headers, body = get(server, f"{LIVE_FRAME_PATH}?after={last[0]}-{last[1]}", headers=token(server))
                assert code == 200 and len(body) == PREVIEW_MAX_JPEG_BYTES
                last = tuple(int(n) for n in headers["X-Frame"].split("-"))
            assert last == (1, 5)
        finally:
            slow.close()
    rig.preview.close(2.0)
