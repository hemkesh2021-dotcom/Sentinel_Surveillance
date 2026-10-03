"""OpenCV/FFmpeg camera source for the Oct 20 demo (D24; V2-05 demo form, full acceptance pending).

Decodes the H.264 substream (profile A, D22) on the CPU with the FFmpeg that
~/onvif_env's OpenCV bundles. Video only: FFmpeg is told to set up only the RTSP
video track, over TCP, so audio is neither received nor decoded. Hardware
decode (NVDEC) and the relay are V2-05 proper.

Device adapter: cv2 is imported on first open, so the portable package imports
without it. The URL comes from SENTINEL_RTSP_URL and never appears in errors,
status or repr. FFmpeg's own log stays at OpenCV's default (errors only); its
connection errors can name the camera's host and port, so the runtime must not
copy decoder stderr into shared logs unredacted.

Source PTS is stream time in microseconds from ``CAP_PROP_POS_MSEC``. OpenCV
4.13's ``CAP_PROP_PTS`` is unsuitable: it rounds the PTS to whole periods of the
stream's estimated average frame rate and repeats the previous value when a
frame has no PTS. On the substream (FFmpeg reports tbr 20 against a 15 fps
average) that stamped 15-21 % of frames without a usable PTS (session 13).
"""

from __future__ import annotations

import ipaddress
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from ..config import CaptureConfig
from ..contracts import PixelFormat
from .capture import DecodedFrame, SourceError

RTSP_URL_ENV = "SENTINEL_RTSP_URL"
# Read by OpenCV's FFmpeg backend at every open; when set it replaces OpenCV's
# default RTSP option (rtsp_flags=prefer_tcp), so it must name the transport too.
FFMPEG_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
RTSP_FFMPEG_OPTIONS = "rtsp_transport;tcp|allowed_media_types;video"
_DEFAULT_PORTS = {"rtsp": 554, "rtsps": 322}


@dataclass(frozen=True)
class RtspEndpoint:
    """Where the camera stream is served; used only to count this host's connections to it."""

    host: str
    port: int

    @property
    def ip(self) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
        """The host as an address, or None for a host name (not resolved here)."""
        try:
            return ipaddress.ip_address(self.host)
        except ValueError:
            return None


def rtsp_endpoint(url: str) -> RtspEndpoint:
    """Host and port of an rtsp:// or rtsps:// URL; SourceError('rtsp_url_invalid') otherwise."""
    if any(ch.isspace() or not ch.isprintable() for ch in url):
        raise SourceError("rtsp_url_invalid")
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        raise SourceError("rtsp_url_invalid") from None
    if parts.scheme not in _DEFAULT_PORTS or not host:
        raise SourceError("rtsp_url_invalid")
    return RtspEndpoint(host, port or _DEFAULT_PORTS[parts.scheme])


def rtsp_url_from_environment(environ: Mapping[str, str] | None = None) -> str:
    url = (os.environ if environ is None else environ).get(RTSP_URL_ENV, "")
    if not url.strip():
        raise SourceError("rtsp_url_missing")
    rtsp_endpoint(url)
    return url


class OpenCvSource:
    """A VideoSource over ``cv2.VideoCapture`` with the FFmpeg backend.

    Open and read are bounded by the capture timeouts (OpenCV's FFmpeg
    interrupt callback). ``target`` is an RTSP URL or, for checks with a local
    file, a path; RTSP targets also get the TCP, video-only FFmpeg options.
    """

    def __init__(self, target: str, config: CaptureConfig, *, cv2: Any = None) -> None:
        self._rtsp = target.startswith(("rtsp://", "rtsps://"))
        self.endpoint = rtsp_endpoint(target) if self._rtsp else None
        self._target = target
        self._config = config
        self._cv2 = cv2
        self._cap: Any = None
        self._reads_since_open = 0

    @classmethod
    def from_environment(
        cls, config: CaptureConfig, environ: Mapping[str, str] | None = None, *, cv2: Any = None
    ) -> OpenCvSource:
        return cls(rtsp_url_from_environment(environ), config, cv2=cv2)

    def __repr__(self) -> str:
        kind = "rtsp" if self._rtsp else "file"
        return f"OpenCvSource({kind}, open={self._cap is not None})"

    def open(self) -> None:
        cv2 = self._module()
        self.close()
        if self._rtsp:
            os.environ[FFMPEG_OPTIONS_ENV] = RTSP_FFMPEG_OPTIONS
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, round(self._config.open_timeout_s * 1000),
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, round(self._config.read_timeout_s * 1000),
            cv2.CAP_PROP_N_THREADS, self._config.decode_threads,
        ]
        cap = cv2.VideoCapture(self._target, cv2.CAP_FFMPEG, params)
        if not cap.isOpened():
            cap.release()
            raise SourceError("open_failed")
        self._cap = cap
        self._reads_since_open = 0

    def read(self) -> DecodedFrame | None:
        cap = self._cap
        if cap is None:
            raise SourceError("not_open")
        ok, image = cap.read()
        if not ok or image is None:
            return None
        shape = getattr(image, "shape", ())
        if len(shape) != 3 or shape[2] != 3 or str(getattr(image, "dtype", "")) != "uint8":
            raise SourceError("unexpected_frame_format")
        first = self._reads_since_open == 0
        self._reads_since_open += 1
        return DecodedFrame(image, int(shape[1]), int(shape[0]), PixelFormat.BGR, self._pts_us(cap, first))

    def _pts_us(self, cap: Any, first: bool) -> int | None:
        """Stream time of the frame just read, in microseconds, or None if unknown.

        OpenCV reports 0 ms for a frame without PTS. Stream time starts at 0, so only
        the first frame after an open can really be at 0.
        """
        msec = cap.get(self._cv2.CAP_PROP_POS_MSEC)
        if not isinstance(msec, float) or not math.isfinite(msec) or msec < 0 or (msec == 0 and not first):
            return None
        return round(msec * 1000)

    def close(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()

    def _module(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError:
                raise SourceError("opencv_unavailable") from None
            self._cv2 = cv2
        return self._cv2
