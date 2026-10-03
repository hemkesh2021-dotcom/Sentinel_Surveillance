"""Camera ingest: the single reader of the upstream stream (guide ch. 6, 7 and 32; V2-05).

CaptureWorker is the only code that reads the camera. It opens a VideoSource,
stamps every decoded frame with FrameStamper (each connection is a new stream
epoch) and hands only the newest frame to the consumer through LatestFrame: a
slow consumer skips frames instead of building a queue, so at most one decoded
image waits. A connection that fails, or delivers no frame within the source's
read timeout, is closed and reopened after a bounded wait that grows while
connections deliver nothing.

Portable: decoders live behind VideoSource in their own modules (the device's
OpenCV/FFmpeg source is media.opencv_source), so importing this one loads none.
Problems are fixed labels or exception class names, never exception text, which
can contain the camera URL and its credentials.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from ..config import CaptureConfig
from ..contracts import FrameRef, PixelFormat, StreamIdentity
from .frames import FrameStamper

_LABEL = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class SourceError(Exception):
    """A source failure described by a fixed snake_case label, never by URL or decoder text."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason if _LABEL.match(reason) else "source_error"


@dataclass(frozen=True)
class DecodedFrame:
    """One decoded picture; the receiver owns ``image`` (e.g. an HxWx3 array).

    ``source_pts`` is the source's stream time in its own unit (OpenCvSource:
    microseconds), or None when unknown. It is advisory (D21).
    """

    image: Any
    width: int
    height: int
    pixel_format: PixelFormat
    source_pts: int | None = None


class VideoSource(Protocol):
    """A camera or file decoder. Every call must return within the source's own timeouts."""

    def open(self) -> None:
        """Connect; raise SourceError if the stream cannot be opened."""

    def read(self) -> DecodedFrame | None:
        """The next decoded frame; None once the stream ended or delivered nothing in time."""

    def close(self) -> None:
        """Release the connection; safe to call when not open."""


@dataclass(frozen=True)
class CapturedFrame:
    frame: FrameRef
    image: Any


class LatestFrame:
    """Single-slot handoff from the capture thread to one consumer.

    Holds at most one undelivered frame: a newer frame replaces it, and take()
    delivers each frame at most once. Counters always satisfy
    published == delivered + replaced + discarded + pending.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending: CapturedFrame | None = None
        self._closed = False
        self._counts = {"published": 0, "delivered": 0, "replaced": 0, "discarded": 0}

    def counts(self) -> dict[str, int]:
        with self._cond:
            return {**self._counts, "pending": int(self._pending is not None)}

    def publish(self, captured: CapturedFrame) -> None:
        with self._cond:
            self._counts["published"] += 1
            if self._pending is not None:
                self._counts["replaced"] += 1
            self._pending = captured
            self._cond.notify_all()

    def discard(self) -> None:
        """Drop the undelivered frame, e.g. because its stream epoch ended."""
        with self._cond:
            if self._pending is not None:
                self._counts["discarded"] += 1
                self._pending = None

    def take(self, timeout_s: float) -> CapturedFrame | None:
        """The newest undelivered frame, waiting up to ``timeout_s``; None on timeout or close."""
        with self._cond:
            self._cond.wait_for(lambda: self._pending is not None or self._closed, timeout_s)
            captured, self._pending = self._pending, None
            if captured is not None:
                self._counts["delivered"] += 1
            return captured

    def close(self) -> None:
        """Wake waiting consumers; later take() calls return without waiting."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class CaptureState(str, Enum):
    STARTING = "starting"  # run() has not begun
    CONNECTING = "connecting"
    STREAMING = "streaming"
    WAITING = "waiting"  # between connections
    STOPPED = "stopped"
    FAILED = "failed"  # the worker itself raised; see problem


@dataclass(frozen=True)
class CaptureStatus:
    state: CaptureState
    stream: StreamIdentity | None  # the connected epoch; None unless streaming
    frames: int  # frames stamped and published, over all connections
    connects: int  # successful opens
    open_failures: int
    stream_ends: int  # connections that ended without a stop request
    retry_delay_s: float | None  # the wait before reopening, while WAITING
    problem: str | None  # the latest problem label


class CaptureWorker:
    """Reads one source until stopped; run() in the calling thread or start() in its own.

    Stopping takes at most one source call (bounded by the source's open or read
    timeout); the source is closed on every path and only by this worker's thread.
    """

    def __init__(
        self,
        source: VideoSource,
        stamper: FrameStamper,
        slot: LatestFrame,
        config: CaptureConfig,
        *,
        wait: Callable[[float], bool] | None = None,
        name: str = "sentinel-capture",
    ) -> None:
        self._source = source
        self._stamper = stamper
        self._slot = slot
        self._config = config
        self._stop = threading.Event()
        # Waits between connections; returns True when stop was requested. Tests inject a fake.
        self._wait = wait or self._stop.wait
        self._name = name
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = CaptureState.STARTING
        self._stream: StreamIdentity | None = None
        self._counts = {"frames": 0, "connects": 0, "open_failures": 0, "stream_ends": 0}
        self._retry_delay_s: float | None = None
        self._problem: str | None = None

    @property
    def connected(self) -> StreamIdentity | None:
        """The stream epoch being read now, for EdgeCore's ``connected`` argument."""
        with self._lock:
            return self._stream

    def status(self) -> CaptureStatus:
        with self._lock:
            return CaptureStatus(
                state=self._state,
                stream=self._stream,
                retry_delay_s=self._retry_delay_s,
                problem=self._problem,
                **self._counts,
            )

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("capture worker already started")
        self._thread = threading.Thread(target=self.run, name=self._name, daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()
        self._slot.close()

    def stop(self, timeout_s: float) -> bool:
        """Request stop and wait; False if the thread is still inside a source call after ``timeout_s``."""
        self.request_stop()
        if self._thread is None:
            return True
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def run(self) -> None:
        initial, longest = self._config.reconnect_initial_s, self._config.reconnect_max_s
        delay = initial
        try:
            while not self._stop.is_set():
                delivered = self._connection()
                if self._stop.is_set():
                    break
                if delivered:
                    delay = initial
                self._set(CaptureState.WAITING, retry_delay_s=delay)
                if self._wait(delay):
                    break
                if not delivered:
                    delay = min(delay * 2, longest)
        except Exception as exc:
            self._set(CaptureState.FAILED, problem=f"worker_error:{type(exc).__name__}")
            raise
        self._set(CaptureState.STOPPED)

    def _connection(self) -> int:
        """One open-read-close cycle; returns the number of frames it delivered."""
        self._set(CaptureState.CONNECTING)
        try:
            self._source.open()
        except Exception as exc:  # noqa: BLE001 - a decoder error must not end ingest
            self._count("open_failures", problem=_problem("open", exc))
            self._close_source()
            return 0
        stream = self._stamper.connect()
        self._set(CaptureState.STREAMING, stream=stream)
        self._count("connects")
        delivered = 0
        size: tuple[int, int] | None = None
        try:
            while not self._stop.is_set():
                try:
                    decoded = self._source.read()
                except Exception as exc:  # noqa: BLE001
                    self._count(None, problem=_problem("read", exc))
                    break
                if decoded is None:
                    self._count(None, problem="no_frame")
                    break
                if not _valid_size(decoded):
                    self._count(None, problem="invalid_frame")
                    break
                # Trackers and boxes assume one geometry per epoch: a new size needs a new connection.
                size = size or (decoded.width, decoded.height)
                if (decoded.width, decoded.height) != size:
                    self._count(None, problem="frame_size_changed")
                    break
                frame = self._stamper.stamp(
                    native_width=decoded.width,
                    native_height=decoded.height,
                    pixel_format=decoded.pixel_format,
                    source_pts=decoded.source_pts,
                )
                self._slot.publish(CapturedFrame(frame, decoded.image))
                delivered += 1
                self._count("frames")
        finally:
            # Withdraw the epoch before dropping its frame, so no consumer sees it as connected.
            self._stamper.disconnect()
            self._set(CaptureState.CONNECTING, stream=None)
            self._slot.discard()
            self._close_source()
        if not self._stop.is_set():
            self._count("stream_ends")
        return delivered

    def _close_source(self) -> None:
        try:
            self._source.close()
        except Exception as exc:  # noqa: BLE001
            self._count(None, problem=_problem("close", exc))

    def _set(
        self,
        state: CaptureState,
        *,
        stream: StreamIdentity | None = None,
        retry_delay_s: float | None = None,
        problem: str | None = None,
    ) -> None:
        with self._lock:
            self._state = state
            self._stream = stream if state is CaptureState.STREAMING else None
            self._retry_delay_s = retry_delay_s
            if problem is not None:
                self._problem = problem

    def _count(self, counter: str | None, *, problem: str | None = None) -> None:
        with self._lock:
            if counter is not None:
                self._counts[counter] += 1
            if problem is not None:
                self._problem = problem


def _problem(phase: str, exc: Exception) -> str:
    if isinstance(exc, SourceError):
        return exc.reason
    return f"{phase}_error:{type(exc).__name__}"


def _valid_size(decoded: DecodedFrame) -> bool:
    return all(type(v) is int and v > 0 for v in (decoded.width, decoded.height))
