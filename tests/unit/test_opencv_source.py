"""V2-05 demo form: the OpenCV/FFmpeg source with a fake cv2 module (no decoder, no camera)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from sentinel.config import CaptureConfig
from sentinel.contracts import PixelFormat
from sentinel.media.capture import SourceError
from sentinel.media.opencv_source import (
    FFMPEG_OPTIONS_ENV,
    RTSP_FFMPEG_OPTIONS,
    RTSP_URL_ENV,
    OpenCvSource,
    RtspEndpoint,
    rtsp_endpoint,
    rtsp_url_from_environment,
)

URL = "rtsp://admin:hunter2@192.0.2.10:8554/cam/realmonitor?channel=1&subtype=1"


class FakeImage:
    def __init__(self, shape: tuple[int, ...] = (480, 640, 3), dtype: str = "uint8") -> None:
        self.shape = shape
        self.dtype = dtype


class FakeCapture:
    def __init__(self, target: str, api: int, params: list[int], fake: FakeCv2) -> None:
        self.target, self.api, self.params = target, api, params
        self.options_at_open = os.environ.get(FFMPEG_OPTIONS_ENV)
        self._fake = fake
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - cv2's name
        return self._fake.opens

    def read(self) -> tuple[bool, object]:
        return self._fake.reads.pop(0) if self._fake.reads else (False, None)

    def get(self, prop: int) -> float:
        assert prop == FakeCv2.CAP_PROP_PTS
        return self._fake.pts

    def release(self) -> None:
        self.released = True


class FakeCv2(SimpleNamespace):
    CAP_FFMPEG = 1900
    CAP_PROP_PTS = 71
    CAP_PROP_OPEN_TIMEOUT_MSEC = 53
    CAP_PROP_READ_TIMEOUT_MSEC = 54
    CAP_PROP_N_THREADS = 70

    def __init__(self, *, opens: bool = True) -> None:
        super().__init__()
        self.opens = opens
        self.reads: list[tuple[bool, object]] = []
        self.pts = 6000.0
        self.captures: list[FakeCapture] = []

    def VideoCapture(self, target: str, api: int, params: list[int]) -> FakeCapture:  # noqa: N802
        capture = FakeCapture(target, api, params, self)
        self.captures.append(capture)
        return capture


@pytest.fixture(autouse=True)
def restore_ffmpeg_options(monkeypatch: pytest.MonkeyPatch) -> None:
    # The source sets this process-wide; register it so monkeypatch restores the original.
    monkeypatch.setenv(FFMPEG_OPTIONS_ENV, "operator-value")


def test_rtsp_open_uses_ffmpeg_with_bounded_timeouts_tcp_and_video_only() -> None:
    cv2 = FakeCv2()
    config = CaptureConfig(open_timeout_s=7.5, read_timeout_s=4.0, decode_threads=2)
    source = OpenCvSource(URL, config, cv2=cv2)

    source.open()

    capture = cv2.captures[0]
    assert (capture.target, capture.api) == (URL, FakeCv2.CAP_FFMPEG)
    assert capture.params == [53, 7500, 54, 4000, 70, 2]
    assert capture.options_at_open == RTSP_FFMPEG_OPTIONS == "rtsp_transport;tcp|allowed_media_types;video"
    assert source.endpoint == RtspEndpoint("192.0.2.10", 8554)


def test_a_file_target_gets_no_rtsp_options_and_no_endpoint() -> None:
    cv2 = FakeCv2()
    source = OpenCvSource("/tmp/synthetic.avi", CaptureConfig(), cv2=cv2)
    source.open()
    assert cv2.captures[0].options_at_open == "operator-value"
    assert source.endpoint is None


def test_read_returns_bgr_frames_with_pts_and_none_at_the_end() -> None:
    cv2 = FakeCv2()
    image = FakeImage()
    cv2.reads = [(True, image), (True, FakeImage())]
    source = OpenCvSource(URL, CaptureConfig(), cv2=cv2)
    source.open()

    frame = source.read()
    assert frame is not None
    assert (frame.image, frame.width, frame.height, frame.pixel_format, frame.source_pts) == (
        image, 640, 480, PixelFormat.BGR, 6000,
    )
    assert type(frame.width) is int and type(frame.height) is int
    cv2.pts = -9.223372036854776e18  # AV_NOPTS_VALUE as OpenCV reports it
    assert source.read().source_pts is None  # type: ignore[union-attr]
    assert source.read() is None  # (False, None): ended or no frame within the read timeout


@pytest.mark.parametrize("pts", [float("nan"), float("inf"), -1.0])
def test_unusable_pts_values_become_none(pts: float) -> None:
    cv2 = FakeCv2()
    cv2.reads = [(True, FakeImage())]
    cv2.pts = pts
    source = OpenCvSource(URL, CaptureConfig(), cv2=cv2)
    source.open()
    assert source.read().source_pts is None  # type: ignore[union-attr]


@pytest.mark.parametrize("image", [FakeImage((480, 640)), FakeImage((480, 640, 4)), FakeImage(dtype="uint16")])
def test_unexpected_pixel_layouts_are_refused(image: FakeImage) -> None:
    cv2 = FakeCv2()
    cv2.reads = [(True, image)]
    source = OpenCvSource(URL, CaptureConfig(), cv2=cv2)
    source.open()
    with pytest.raises(SourceError) as caught:
        source.read()
    assert caught.value.reason == "unexpected_frame_format"


def test_a_failed_open_releases_the_capture_and_names_no_url() -> None:
    cv2 = FakeCv2(opens=False)
    source = OpenCvSource(URL, CaptureConfig(), cv2=cv2)
    with pytest.raises(SourceError) as caught:
        source.open()
    assert caught.value.reason == "open_failed"
    assert cv2.captures[0].released
    for text in (str(caught.value), repr(source)):
        assert "hunter2" not in text and "192.0.2.10" not in text
    with pytest.raises(SourceError, match="not_open"):
        source.read()


def test_reopen_and_close_release_the_previous_capture() -> None:
    cv2 = FakeCv2()
    source = OpenCvSource(URL, CaptureConfig(), cv2=cv2)
    source.open()
    source.open()
    assert [c.released for c in cv2.captures] == [True, False]
    source.close()
    source.close()
    assert all(c.released for c in cv2.captures)


def test_missing_opencv_is_a_source_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_cv2(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "cv2":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    source = OpenCvSource(URL, CaptureConfig())
    monkeypatch.setattr(builtins, "__import__", no_cv2)
    with pytest.raises(SourceError) as caught:
        source.open()
    assert caught.value.reason == "opencv_unavailable"


def test_the_url_comes_only_from_the_environment_and_is_validated() -> None:
    assert rtsp_url_from_environment({RTSP_URL_ENV: URL}) == URL
    source = OpenCvSource.from_environment(CaptureConfig(), {RTSP_URL_ENV: URL}, cv2=FakeCv2())
    assert source.endpoint == RtspEndpoint("192.0.2.10", 8554)
    assert rtsp_endpoint("rtsps://cam.local/stream") == RtspEndpoint("cam.local", 322)
    assert rtsp_endpoint("rtsp://[2001:db8::7]/live") == RtspEndpoint("2001:db8::7", 554)

    with pytest.raises(SourceError, match="rtsp_url_missing"):
        rtsp_url_from_environment({RTSP_URL_ENV: "  "})
    with pytest.raises(SourceError, match="rtsp_url_missing"):
        rtsp_url_from_environment({})


@pytest.mark.parametrize(
    "url",
    [
        "http://admin:hunter2@192.0.2.10/live",
        "file:///etc/passwd",
        "rtsp://admin:hunter2@/live",
        "rtsp://admin:hunter2@192.0.2.10:99999/live",
        "rtsp://admin:hunter2@192.0.2.10/live\n-i x",
        "rtsp://admin:hunter2@192.0.2.10/li ve",
    ],
)
def test_invalid_urls_are_refused_without_echoing_them(url: str) -> None:
    with pytest.raises(SourceError) as caught:
        rtsp_url_from_environment({RTSP_URL_ENV: url})
    assert caught.value.reason == "rtsp_url_invalid"
    assert "hunter2" not in str(caught.value) and caught.value.__cause__ is None
