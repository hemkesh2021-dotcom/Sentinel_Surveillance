"""D69: the live view on the D-2 status page (RecentFrames, LivePreview, the page's picture route and its bounds).

No camera, model or cv2: frames come from a real FrameStamper on a FakeClock, images are labelled stand-ins, and
the encoder is a fake that names the image it was given. The device encoder is tested only where cv2 exists.
"""

from __future__ import annotations

import contextlib
import http.client
import socket
import threading
import time
from collections.abc import Iterator
from html.parser import HTMLParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from sentinel import status_page
from sentinel.config import FreshnessConfig
from sentinel.contracts import PixelFormat, SourceTimeQuality
from sentinel.live_preview import PREVIEW_MAX_ENCODES_PER_S, PREVIEW_MAX_JPEG_BYTES, LivePreview, time_label
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.media.jpeg import SOI, EncodedJpeg, EncodeError, fit_within, jpeg_comments, with_comment
from sentinel.media.recent import RecentFrames
from sentinel.status_page import PAGE_CURRENT_S, StatusServer, render_html
from sentinel.storage.database import Database

T0 = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
FRAME_S = 1 / 15
EOI = b"\xff\xd9"
SNAPSHOT: dict[str, Any] = {"runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": T0.isoformat()}}


class Image:
    """A stand-in for a decoded frame: the fake encoder reports which one it was given."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.shape = (480, 640, 3)


class Rig:
    """A LivePreview over a real FrameStamper and RecentFrames, on a FakeClock."""

    def __init__(self, *, picture_bytes: int = 0, **options: Any) -> None:
        self.clock = FakeClock(utc=T0)
        self.stamper = FrameStamper("cam-1", self.clock)
        self.frames = RecentFrames()
        self.encoded: list[str] = []
        self.picture_bytes = picture_bytes
        options.setdefault("encode", self.encode)
        self.preview = LivePreview(self.frames, self.clock, FreshnessConfig(), lambda: self.stamper.current_stream,
                                   **options)

    def encode(self, image: Image, pixel_format: PixelFormat, *, max_width: int, max_height: int,
               quality: int) -> EncodedJpeg:
        self.encoded.append(image.label)
        body = image.label.encode() + b"\0" * max(0, self.picture_bytes - len(image.label) - 4)
        return EncodedJpeg(SOI + body + EOI, min(640, max_width), min(480, max_height))

    def frame(self, dt: float = FRAME_S):
        self.clock.advance(dt)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        self.frames.put(frame, Image(f"e{frame.stream_epoch}-f{frame.frame_seq}"))
        return frame


def live_rig(**options: Any) -> Rig:
    rig = Rig(**options)
    rig.stamper.connect()
    rig.frame()
    return rig


# ---------------------------------------------------------------- frames and pictures


def test_recent_frames_keep_the_last_two_and_find_only_the_exact_frame() -> None:
    rig = Rig()
    rig.stamper.connect()
    first, second, third = rig.frame(), rig.frame(), rig.frame()
    assert len(rig.frames) == 2
    assert rig.frames.get(first.key) is None  # released, never replaced by another frame
    assert rig.frames.get(second.key)[1].label == "e1-f1"  # type: ignore[index]
    assert rig.frames.latest()[0] == third  # type: ignore[index]
    rig.stamper.disconnect()
    rig.stamper.connect()  # epoch 2 starts at frame 0 again: identity, not the sequence number, finds a frame
    again = rig.frame()
    assert again.frame_seq == 0 and rig.frames.get(again.key)[1].label == "e2-f0"  # type: ignore[index]
    with pytest.raises(ValueError):
        RecentFrames(0)


def test_pictures_are_scaled_down_only_keeping_their_shape() -> None:
    assert fit_within(640, 480, 640, 480) == (640, 480)
    assert fit_within(320, 240, 640, 480) == (320, 240)  # never up
    assert fit_within(1920, 1080, 640, 480) == (640, 360)
    assert fit_within(1080, 1920, 640, 480) == (270, 480)
    assert fit_within(1921, 1081, 1920, 1080) == (1918, 1080) or fit_within(1921, 1081, 1920, 1080)[0] <= 1920
    with pytest.raises(ValueError):
        fit_within(0, 480, 640, 480)


def test_a_comment_names_the_picture_without_touching_its_data() -> None:
    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    jpeg = SOI + app0 + b"\xff\xdb\x00\x03\x07" + b"\xff\xda\x00\x02scan" + EOI
    tagged = with_comment(jpeg, "Sentinel incident inc-1; stream epoch 2 frame 9")
    assert tagged.startswith(SOI + app0)  # the JFIF header stays first
    assert tagged.replace(b"\xff\xfe\x00\x31Sentinel incident inc-1; stream epoch 2 frame 9", b"") == jpeg
    assert jpeg_comments(tagged) == ["Sentinel incident inc-1; stream epoch 2 frame 9"]
    assert jpeg_comments(with_comment(SOI + EOI, "x")) == ["x"]
    assert jpeg_comments(b"not a jpeg") == []
    for bad in ("", "tab\there", "é", "x" * 513):
        with pytest.raises(ValueError):
            with_comment(jpeg, bad)
    with pytest.raises(EncodeError):
        with_comment(b"GIF89a", "x")


def test_the_time_label_says_ingest_not_capture() -> None:
    for quality in SourceTimeQuality:
        label = time_label(quality)
        assert label.startswith("ingest time: when this device received and decoded the frame, not when the camera "
                                "captured it")
        assert f"(source time quality: {quality.value})" in label
    assert "capture time is not known" in time_label(SourceTimeQuality.NONE)


# ---------------------------------------------------------------- states


def test_a_live_frame_gives_its_own_picture_with_age_and_ingest_time() -> None:
    rig = live_rig()
    rig.clock.advance(0.25)
    view = rig.preview.view(with_picture=True)
    assert view["state"] == "live" and view["reason"] == "the newest frame of the connected stream"
    picture = view["picture"]
    assert picture["url"] == "/preview/1-0.jpg" and (picture["stream_epoch"], picture["frame_seq"]) == (1, 0)
    assert picture["age_ms"] == 250 and picture["ingest_utc"] == "2026-10-09T12:00:00.066667Z"
    assert picture["source_time_quality"] == "none" and "not when the camera captured it" in view["time_label"]
    assert rig.preview.picture("1-0").jpeg == SOI + b"e1-f0" + EOI  # type: ignore[union-attr]
    assert rig.encoded == ["e1-f0"]


def test_a_frame_older_than_stale_after_shows_no_picture_and_its_picture_is_refused() -> None:
    rig = live_rig()
    assert rig.preview.view(with_picture=True)["picture"] is not None
    rig.clock.advance(2.1)  # no new frame: the stream may still be connected
    view = rig.preview.view(with_picture=True)
    assert view["state"] == "stale" and view["picture"] is None
    assert view["reason"] == "no new frame for 2.1 s (stale after 2 s)"
    assert view["age_ms"] == 2100 and view["frame"]["frame_seq"] == 0  # what the last frame was, without a picture
    assert rig.preview.picture("1-0") is None  # even its own URL: a frozen frame is never served as current
    assert rig.preview.summary()["counters"]["refused_not_current"] == 1
    assert rig.encoded == ["e1-f0"]


def test_a_frame_from_an_ended_stream_epoch_is_stale_at_once() -> None:
    rig = live_rig(max_encodes_per_s=2.0)  # so that the new epoch's first frame is not due at once
    assert rig.preview.view(with_picture=True)["picture"] is not None
    rig.stamper.disconnect()  # the connection ended a moment after the frame
    view = rig.preview.view(with_picture=True)
    assert view["state"] == "stale" and view["picture"] is None and view["age_ms"] < 2000
    assert view["reason"] == "the stream connection this frame came from has ended; reconnecting"
    assert rig.preview.picture("1-0") is None
    rig.stamper.connect()  # reconnected, no frame from the new epoch yet: the old frame is still not current
    assert rig.preview.view(with_picture=True)["state"] == "stale"
    rig.frame()
    assert rig.preview.view(with_picture=True)["picture"] is None  # the old picture is not current; none due yet
    rig.clock.advance(0.5)
    assert rig.preview.view(with_picture=True)["picture"]["url"] == "/preview/2-0.jpg"


def test_starting_then_offline_without_frames_and_offline_after_a_long_gap() -> None:
    rig = Rig()
    assert rig.preview.view(with_picture=True) | {"bounds": None} == {
        "enabled": True, "state": "starting", "reason": "no frame yet since the runtime started", "age_ms": None,
        "frame": None, "time_label": None, "picture": None, "problem": None, "bounds": None}
    rig.clock.advance(10.0)
    view = rig.preview.view(with_picture=True)
    assert (view["state"], view["reason"]) == ("offline", "no frame received")
    rig.stamper.connect()
    rig.frame()
    assert rig.preview.view()["state"] == "live"
    rig.clock.advance(12.0)
    view = rig.preview.view(with_picture=True)
    assert (view["state"], view["reason"], view["picture"]) == ("offline", "no new frame for 12 s (offline after 10 s)",
                                                                 None)
    assert rig.encoded == []


# ---------------------------------------------------------------- bounds


def encode_times(rig: Rig, requests: int, rate: float) -> list[float]:
    """Frame times (s) of the pictures made while a page asks after every frame, at 15 fps."""
    made = []
    for _ in range(requests):
        frame = rig.frame()
        picture = rig.preview.view(with_picture=True)["picture"]
        if picture is not None and picture["frame_seq"] == frame.frame_seq and f"e1-f{frame.frame_seq}" == rig.encoded[-1]:
            made.append(rig.clock.monotonic_ns() / 1e9)
    return made


def test_encoding_is_bounded_per_second_and_each_frame_is_encoded_once() -> None:
    rig = live_rig()
    for _ in range(5):  # five viewers of the same frame
        assert rig.preview.view(with_picture=True)["picture"]["frame_seq"] == 0
    assert rig.encoded == ["e1-f0"]
    # D75's default, the camera substream's 15 fps: a page asking after every frame gets every frame, once each.
    encode_times(rig, 45, 15.0)
    assert rig.encoded == [f"e1-f{n}" for n in range(46)]
    counters = rig.preview.summary()["counters"]
    assert counters["encodes"] == 46 and counters["reused"] == 4 and "rate_limited" not in counters


@pytest.mark.parametrize("rate", [2.0, 5.0, 7.5])
def test_a_lower_budget_is_never_exceeded_in_any_second_nor_spaced_closer_than_three_quarters(rate: float) -> None:
    rig = live_rig(max_encodes_per_s=rate)
    times = encode_times(rig, 90, rate)  # 6 s at 15 fps
    assert len(set(rig.encoded)) == len(rig.encoded)  # each frame once
    assert all(b - a >= 0.75 / rate - 1e-6 for a, b in zip(times, times[1:]))
    assert all(sum(1 for t in times if start <= t < start + 1.0) <= rate for start in times)
    assert len(times) >= 6 * rate * 0.6  # bounded, not starved: most of the budget is used


def test_a_rate_limited_picture_is_described_by_its_own_frame() -> None:
    rig = live_rig(max_encodes_per_s=2.0)
    rig.preview.view(with_picture=True)
    rig.frame()
    newest = rig.frame()  # 2 frames later, 0.13 s: not due yet
    view = rig.preview.view(with_picture=True)
    assert view["frame"]["frame_seq"] == newest.frame_seq == 2
    assert view["picture"]["frame_seq"] == 0 and view["picture"]["url"] == "/preview/1-0.jpg"
    assert view["picture"]["age_ms"] == 133 and view["age_ms"] == 0
    assert rig.encoded == ["e1-f0"]


def test_status_json_views_never_encode() -> None:
    rig = live_rig()
    for _ in range(10):
        rig.frame()
        assert rig.preview.view()["picture"] is None
    assert rig.encoded == [] and rig.preview.summary()["pictures_kept"] == 0


@pytest.mark.parametrize(("failure", "label"), [
    (EncodeError("jpeg_encode_failed"), "jpeg_encode_failed"),
    (RuntimeError("rtsp://admin:hunter2@192.0.2.10/ failed"), "encode_failed:RuntimeError"),
])
def test_a_failed_encode_is_labelled_and_shows_no_picture(failure: Exception, label: str) -> None:
    def broken(*args: Any, **kwargs: Any) -> EncodedJpeg:
        raise failure

    rig = live_rig(encode=broken)
    view = rig.preview.view(with_picture=True)
    assert view["state"] == "live" and view["picture"] is None and view["problem"] == label
    assert "hunter2" not in repr(view) + repr(rig.preview.summary())
    assert rig.preview.picture("1-0") is None


def test_oversized_or_non_jpeg_pictures_are_dropped() -> None:
    rig = live_rig(picture_bytes=PREVIEW_MAX_JPEG_BYTES + 1)
    view = rig.preview.view(with_picture=True)
    assert view["picture"] is None and view["problem"] == "picture_too_large"
    rig = live_rig(encode=lambda image, fmt, **kw: EncodedJpeg(b"GIF89a", 1, 1))
    assert rig.preview.view(with_picture=True)["problem"] == "not_a_jpeg"


def test_concurrent_viewers_encode_a_frame_once() -> None:
    entered, release = threading.Event(), threading.Event()
    rig = Rig()

    def slow(image: Image, fmt: PixelFormat, **kwargs: Any) -> EncodedJpeg:
        entered.set()
        release.wait(5)
        return rig.encode(image, fmt, **kwargs)

    rig.preview._encode = slow  # type: ignore[attr-defined]
    rig.stamper.connect()
    rig.frame()
    views: list[dict[str, Any]] = []
    threads = [threading.Thread(target=lambda: views.append(rig.preview.view(with_picture=True))) for _ in range(4)]
    threads[0].start()
    assert entered.wait(5)
    for thread in threads[1:]:
        thread.start()
    release.set()
    for thread in threads:
        thread.join(5)
    assert rig.encoded == ["e1-f0"] and [v["picture"]["frame_seq"] for v in views] == [0, 0, 0, 0]


def test_the_preview_bounds_cannot_be_raised() -> None:
    rig = Rig()
    with pytest.raises(ValueError):
        LivePreview(rig.frames, rig.clock, FreshnessConfig(), lambda: None, max_encodes_per_s=PREVIEW_MAX_ENCODES_PER_S + 1)
    with pytest.raises(ValueError):
        LivePreview(rig.frames, rig.clock, FreshnessConfig(), lambda: None, max_jpeg_bytes=PREVIEW_MAX_JPEG_BYTES + 1)
    assert rig.preview.bounds() == {"max_width": 640, "max_height": 480, "jpeg_quality": 70, "max_encodes_per_s": 15.0,
                                    "max_jpeg_bytes": 262_144, "pictures_kept": 2, "yield_encodes_per_s": 2.0,
                                    "stream_wait_s": 1.5, "viewer_idle_s": 3.0, "stale_after_ms": 2000}


def test_the_device_encoder_scales_a_copy_and_leaves_the_frame_unchanged() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from sentinel.media.jpeg import encode_jpeg

    image = np.random.default_rng(7).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    before = image.copy()
    encoded = encode_jpeg(image, PixelFormat.BGR, max_width=640, max_height=480, quality=70)
    assert (encoded.width, encoded.height) == (640, 360) and encoded.data.startswith(SOI)
    assert cv2.imdecode(np.frombuffer(encoded.data, np.uint8), cv2.IMREAD_COLOR).shape == (360, 640, 3)
    assert np.array_equal(image, before)
    small = encode_jpeg(image[:240, :320], PixelFormat.BGR, max_width=640, max_height=480, quality=70)
    assert (small.width, small.height) == (320, 240)
    assert encode_jpeg(image[:, :, 0], PixelFormat.GRAY8, max_width=640, max_height=480, quality=70).width == 640
    for fmt in (PixelFormat.NV12, PixelFormat.RGBA):
        with pytest.raises(EncodeError):
            encode_jpeg(image, fmt, max_width=640, max_height=480, quality=70)
    with pytest.raises(EncodeError):
        encode_jpeg(image[:, :, 0], PixelFormat.BGR, max_width=640, max_height=480, quality=70)


# ---------------------------------------------------------------- the page


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


def get(server: StatusServer, path: str, *, host: str | None = None, method: str = "GET"):
    address, port = server.address
    conn = http.client.HTTPConnection(address, port, timeout=5)
    try:
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", host if host is not None else f"127.0.0.1:{port}")
        conn.endheaders()
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def wait_until(predicate: Any, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_the_page_shows_the_live_picture_and_serves_it(db_path: Path) -> None:
    rig = live_rig()
    with serving(db_path, lambda: rig.preview) as server:
        code, headers, body = get(server, "/")
        page = body.decode()
        assert code == 200 and "img-src 'self'" in headers["Content-Security-Policy"]
        assert '<img class=pic src="/preview/1-0.jpg" width="640" height="480"' in page
        assert "LIVE</span><span class=muted>the newest frame of the connected stream</span>" in page
        assert "2026-10-09T12:00:00.066667Z" in page and "not when the camera captured it" in page
        code, headers, body = get(server, "/preview/1-0.jpg")
        assert (code, headers["Content-Type"], body) == (200, "image/jpeg", SOI + b"e1-f0" + EOI)
        assert headers["Cache-Control"] == "no-store" and headers["X-Content-Type-Options"] == "nosniff"
        document = __import__("json").loads(get(server, "/status.json")[2])
        assert document["preview"]["state"] == "live" and document["preview"]["picture"] is None  # JSON never encodes
    assert rig.encoded == ["e1-f0"]


def test_a_stale_page_shows_no_picture_and_old_picture_urls_are_refused(db_path: Path) -> None:
    rig = live_rig()
    with serving(db_path, lambda: rig.preview) as server:
        assert get(server, "/")[0] == 200
        rig.clock.advance(3.0)
        page = get(server, "/")[2].decode()
        assert "<img" not in page and "<strong>STALE: not current</strong>" in page
        assert "No picture is shown: the last frame is not current." in page
        assert get(server, "/preview/1-0.jpg")[:3:2] == (404, b"no current picture\n")


class Markup(HTMLParser):
    """Each visible text with the classes of every element around it, and the children of each ``fs`` switch."""

    VOID = {"meta", "img", "br", "hr", "input", "link"}

    def __init__(self, page: str) -> None:
        super().__init__()
        self.stack: list[tuple[str, frozenset[str], list[frozenset[str]]]] = []
        self.texts: list[tuple[str, frozenset[str]]] = []
        self.switches: list[list[frozenset[str]]] = []
        self.feed(page)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = frozenset((dict(attrs).get("class") or "").split())
        if self.stack and "fs" in self.stack[-1][1]:
            self.stack[-1][2].append(classes)
        if tag not in self.VOID:
            self.stack.append((tag, classes, []))

    def handle_endtag(self, tag: str) -> None:
        while self.stack:
            name, classes, children = self.stack.pop()
            if "fs" in classes:
                self.switches.append(children)
            if name == tag:
                break

    def handle_data(self, data: str) -> None:
        if data.strip() and not (self.stack and self.stack[-1][0] == "style"):
            self.texts.append((data.strip(), frozenset().union(*(c for _, c, _ in self.stack))))

    def around(self, text: str, *, exact: bool = False) -> list[frozenset[str]]:
        return [c for t, c in self.texts if (t == text if exact else text in t)]


HEALTHY = {**SNAPSHOT, "degraded": []}
CURRENT_CLAIMS = ("This page is current", "All components working", "Page current", "Monitoring",
                  "the newest frame of the connected stream")
NOT_CURRENT_LABELS = (f"NOT CURRENT: this page has not refreshed for {PAGE_CURRENT_S} s",
                      "Last reported: all components working", "NOT CURRENT: last reported LIVE",
                      "Last reported: monitoring")
PICTURE_OLD = "NOT CURRENT: no newer picture has arrived in time; this one was received at"


def test_every_current_claim_is_shown_only_while_the_page_is_fresh() -> None:
    """D71: the page, the runtime's health, LIVE and the picture are current only for PAGE_CURRENT_S after the browser
    shows the page, by CSS alone; hidden by default, so a browser that runs no animation never shows them."""
    rig = live_rig()
    page = render_html(status_page.build_status(HEALTHY, {"state": "available"}, T0, rig.preview.view(with_picture=True)))
    markup = Markup(page)
    for claim in CURRENT_CLAIMS:
        assert markup.around(claim) and all("cur" in c and "old" not in c for c in markup.around(claim)), claim
    assert markup.around("LIVE", exact=True) and all("cur" in c for c in markup.around("LIVE", exact=True))
    for label in NOT_CURRENT_LABELS:
        assert markup.around(label) and all("old" in c and "cur" not in c for c in markup.around(label)), label
    # Each switch holds one current layer and its not-current replacement; green (k-ok) appears only in current layers.
    assert len(markup.switches) >= 5
    assert all(sum("cur" in c for c in s) == 1 and sum("old" in c for c in s) == 1 and len(s) == 2
               for s in markup.switches)
    assert all("cur" in c for t, c in markup.texts if "k-ok" in c), "green outside a current layer"
    # The picture's caption follows the picture's own timer (D73, D75), not the page's.
    assert markup.around(PICTURE_OLD) and all("pic-old" in c and "cur" not in c for c in markup.around(PICTURE_OLD))
    assert ('<figure class=frame data-frame="1-0" data-ingest="2026-10-09T12:00:00.066667Z" data-age-ms="0">'
            '<img class=pic src="/preview/1-0.jpg"') in page
    # The runtime's report is dimmed once the page is not current; the picture is not part of it (its own timer, D75).
    for marker in ('<div class="report live-report" data-region=live-head>', 'class="panel incidents report"',
                   'class="panel health report"', 'class="panel delivery report"'):
        assert marker in page, marker
    assert "<main class=board>" in page
    current = f"{PAGE_CURRENT_S}s linear"
    for rule in (f".cur {{ visibility: hidden; animation: cur {current} var(--fresh-delay, 0s); }}",
                 "@keyframes cur { from, to { visibility: visible; } }",
                 f".old {{ animation: old {current} var(--fresh-delay, 0s); }}",
                 "@keyframes old { from, to { visibility: hidden; } }",
                 f".report {{ opacity: .55; animation: full {current} var(--fresh-delay, 0s); }}",
                 ".live .frame .pic { opacity: .3; animation: pic var(--pic-window, 5s) linear var(--pic-delay, 0s); }",
                 ".live .frame .pic-old { animation: old var(--pic-window, 5s) linear var(--pic-delay, 0s); }",
                 'html[data-fresh="1"] .cur { animation-name: cur1; }', "@keyframes full { from, to { opacity: 1; } }"):
        assert rule in page, rule
    assert "forwards" not in status_page.STYLE and "infinite" not in status_page.STYLE  # no state held past the window
    assert "http-equiv=refresh" not in page and '<script src="/assets/dashboard.js" defer></script>' in page
    assert f'generated at <time datetime="{T0.isoformat()}">' in page  # the page's own time, apart from the runtime's


def test_a_stale_or_degraded_report_makes_no_current_claim_even_on_a_fresh_page() -> None:
    rig = live_rig()
    rig.clock.advance(3.0)
    degraded = {**SNAPSHOT, "degraded": ["capture waiting (open_failed)"]}
    markup = Markup(render_html(status_page.build_status(degraded, {"state": "available"}, T0,
                                                         rig.preview.view(with_picture=True))))
    assert not markup.around("LIVE", exact=True) and not markup.around("All components working")
    assert not markup.around("Last reported: all components working") and not markup.around("last reported LIVE")
    for shown in ("STALE: not current", "Degraded", "capture waiting (open_failed)"):  # last reported, never hidden
        assert markup.around(shown) and all("cur" not in c and "old" not in c for c in markup.around(shown)), shown
    switched = {t for t, c in markup.texts if "cur" in c}
    assert not any("LIVE" in t or "working" in t for t in switched)  # a degraded runtime is still claimed running only
    assert "Monitoring, degraded" in switched and "This page is current" in " ".join(switched)


def test_the_live_view_is_off_or_starting_without_a_preview(db_path: Path) -> None:
    with serving(db_path, None) as server:
        page = get(server, "/")[2].decode()
        assert "the live view is off; start sentinel run with --live-preview to show it" in page and "<img" not in page
        assert get(server, "/preview/1-0.jpg")[0] == 404
        assert "preview" not in __import__("json").loads(get(server, "/status.json")[2])  # unchanged while off
    with serving(db_path, lambda: None) as server:
        assert "STARTING</span><span class=muted>the runtime is starting</span>" in get(server, "/")[2].decode()


@pytest.mark.parametrize("path", [
    "/preview/abc.jpg", "/preview/1-0.png", "/preview/1-0.jpg/", "/preview/../status.json", "/preview/%2e%2e/x.jpg",
    "/preview/1234567890-1.jpg", "/preview/1-0", "/preview/-1-0.jpg", "/preview/1-0.jpg.json",
])
def test_picture_paths_are_validated_and_never_echoed(db_path: Path, path: str) -> None:
    rig = live_rig()
    with serving(db_path, lambda: rig.preview) as server:
        code, _, body = get(server, path)
        assert (code, body) == (404, b"not found\n")
    assert rig.encoded == []


def test_picture_requests_need_get_and_the_loopback_host(db_path: Path) -> None:
    rig = live_rig()
    with serving(db_path, lambda: rig.preview) as server:
        get(server, "/")
        assert get(server, "/preview/1-0.jpg", method="POST")[0] == 405
        assert get(server, "/preview/1-0.jpg", host="evil.example:80")[0] == 421
        assert get(server, "/preview/9-9.jpg")[:3:2] == (404, b"no current picture\n")
        assert server.address[0] == "127.0.0.1"


@contextlib.contextmanager
def stalled_request(port: int, path: str) -> Iterator[socket.socket]:
    """Sends the request, then never reads the response."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    sock.settimeout(5.0)
    try:
        sock.connect(("127.0.0.1", port))
        sock.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
        yield sock
    finally:
        sock.close()


def test_a_stalled_browser_is_disconnected_and_never_holds_the_page(db_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(status_page, "SEND_BUFFER_BYTES", 4096)
    rig = live_rig(picture_bytes=PREVIEW_MAX_JPEG_BYTES)
    with serving(db_path, lambda: rig.preview, handler_timeout_s=0.5) as server:
        port = server.address[1]
        get(server, "/")
        with stalled_request(port, "/preview/1-0.jpg"), stalled_request(port, "/preview/1-0.jpg"):
            time.sleep(0.2)
            assert get(server, "/preview/1-0.jpg")[:3:2] == (503, b"busy: too many picture requests\n")
            assert get(server, "/")[0] == 200 and get(server, "/status.json")[0] == 200  # the page still answers
            assert wait_until(lambda: server.connection_counts()["write_timeouts"] == 2)
            code, _, body = get(server, "/preview/1-0.jpg")  # both picture slots are free again
            assert code == 200 and len(body) == PREVIEW_MAX_JPEG_BYTES
    assert rig.encoded == ["e1-f0"]


def test_connections_are_capped_and_idle_ones_time_out(db_path: Path) -> None:
    with serving(db_path, None, max_connections=2, handler_timeout_s=0.5) as server:
        port = server.address[1]
        idle = [socket.create_connection(("127.0.0.1", port)) for _ in range(2)]
        try:
            time.sleep(0.1)
            with socket.create_connection(("127.0.0.1", port), timeout=5) as third:
                assert third.recv(64).startswith(b"HTTP/1.1 503")
            assert wait_until(lambda: get(server, "/")[0] == 200)  # the idle ones timed out and freed their slots
            assert server.connection_counts()["refused_busy"] >= 1
        finally:
            for sock in idle:
                sock.close()


def test_stop_does_not_wait_for_an_idle_browser(db_path: Path) -> None:
    server = StatusServer(0, lambda: SNAPSHOT, db_path)
    server.start()
    with socket.create_connection(server.address):  # never sends a request
        time.sleep(0.1)
        started = time.monotonic()
        assert server.stop(2.0)
        assert time.monotonic() - started < 1.0
