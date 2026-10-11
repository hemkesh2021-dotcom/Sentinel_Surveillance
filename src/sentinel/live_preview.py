"""Live view on the D-2 status page (D69; continuous pictures since D75; opt-in: ``sentinel run --live-preview``).

It shows the newest frame ``sentinel run`` already reads, on the loopback-only
page, with its age and state. It is the alpha's stand-in for the guide's Live
view (V2-18, over V2-05/V2-06's compressed relay), not that view.

- **No second camera connection.** The runtime loop puts each frame it takes
  from the one capture slot into RecentFrames (a reference, before
  inference). Nothing here reads the camera.
- **Continuous pictures, one at a time (D75).** The page's player asks
  ``next_frame`` (GET /live/frame) for a live picture newer than the one it
  shows; the request waits at most STREAM_WAIT_S and gets exactly one picture,
  the newest. Nothing is queued for a viewer: a picture made while it was busy
  is replaced by a newer one before it asks again, so a slow viewer gets fewer
  pictures, never older ones. The page's status updates never make a picture.
- **Never in the loop's way.** The loop only swaps a reference (and notifies).
  One encoder thread, started by a viewer's request and ending VIEWER_IDLE_S
  after the last one (so nothing is encoded while nobody watches), encodes the
  newest live frame as it arrives, at a lower OS priority (Linux: nice +10 for
  that thread only). While monitoring work is waiting (``pressure``: incident
  signals not yet recorded, snapshots queued) it encodes at most
  PREVIEW_YIELD_ENCODES_PER_S. The page's own first picture is made in its
  request's thread, under the same budget.
- **Bounded.** Each frame is encoded at most once; at most
  ``max_encodes_per_s`` pictures in any second, never two within 3/4 of that
  interval, one encode at a time, for all viewers together. A picture is
  scaled down to fit PREVIEW_MAX_WIDTH x PREVIEW_MAX_HEIGHT, at a fixed
  quality, and dropped above PREVIEW_MAX_JPEG_BYTES. At most
  PREVIEW_PICTURES_KEPT pictures are kept. While a picture is encoded, the
  encoder holds one frame reference beyond RecentFrames' own.
- **Live, stale, offline, starting,** decided on the injected monotonic clock
  as the runtime decides video freshness (``media.health.assess_frame``):
  live only while the frame belongs to the connected stream epoch and is
  younger than ``freshness.stale_after_s``. A picture is offered, and served,
  only while its own frame is live, with its age when it was sent. Otherwise
  the page shows the state, the age and the frame's identity, and no
  picture: a frozen frame is never shown as current.
- **Times.** A frame's time is its ingest time: when this device received and
  decoded it, not when the camera captured it (``source_time_quality`` says
  what is known about capture time; this build maps it to no clock). Ages are
  monotonic, from ingest to the moment the view was made or the picture sent.
- **Saves nothing.** Pictures exist in memory and on the wire only.

Portable: the default encoder imports cv2 only when it runs (``media.jpeg``).
"""

from __future__ import annotations

import math
import os
import re
import statistics
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import FreshnessConfig
from .contracts import FrameRef, SourceTimeQuality, StreamIdentity
from .incidents.service import utc_text
from .media.clock import NS_PER_SECOND, Clock
from .media.health import VideoFreshness, VideoState, assess_frame
from .media.jpeg import SOI, EncodeError, Encoder, encode_jpeg
from .media.recent import RecentFrames

PREVIEW_MAX_WIDTH = 640
PREVIEW_MAX_HEIGHT = 480
PREVIEW_JPEG_QUALITY = 70
PREVIEW_MAX_ENCODES_PER_S = 15.0  # the camera substream's frame rate: more could only repeat frames (D75)
PREVIEW_YIELD_ENCODES_PER_S = 2.0  # while monitoring work waits (D75)
MIN_SPACING = 0.75  # two encodes are at least this fraction of 1 / rate apart; tolerates frame-arrival jitter
PREVIEW_MAX_JPEG_BYTES = 262_144
PREVIEW_PICTURES_KEPT = 2
ENCODE_WAIT_S = 1.0  # a request waits at most this long for another request's encode, then shows what exists
STREAM_WAIT_S = 1.5  # a stream request waits at most this long for a newer live picture (D75)
VIEWER_IDLE_S = 3.0  # the encoder thread ends this long after the last stream request
ENCODER_WAKE_S = 0.25  # the encoder and waiting requests re-check at least this often (closing, idleness)
ENCODER_NICE = 10  # added to the encoder thread's own nice value (Linux)
ENCODE_TIMES_KEPT = 64  # encode durations kept for the status summary
NS_PER_MS = 1_000_000
TOKEN = re.compile(r"[0-9]{1,9}-[0-9]{1,12}")  # <stream epoch>-<frame sequence>, unique within one runtime
STATE_NAMES = {
    VideoState.FRESH: "live",
    VideoState.STALE: "stale",
    VideoState.OFFLINE: "offline",
    VideoState.STARTING: "starting",
}
_CAPTURE_TIME = {
    SourceTimeQuality.NONE: "the camera's capture time is not known",
    SourceTimeQuality.STREAM_RELATIVE: "the camera's capture time is not known (its stream timestamps map to no clock)",
    SourceTimeQuality.CAPTURE_SYNCED: "a capture time is known through a validated clock sync, but is not shown here",
}


def time_label(quality: SourceTimeQuality) -> str:
    """What a frame's shown time means, given what is known about its capture time."""
    return (f"ingest time: when this device received and decoded the frame, not when the camera captured it; "
            f"{_CAPTURE_TIME[quality]} (source time quality: {quality.value})")


def frame_facts(frame: FrameRef) -> dict[str, Any]:
    """A frame's identity within the run and its ingest time; no boot or run ID."""
    return {
        "stream_epoch": frame.stream_epoch,
        "frame_seq": frame.frame_seq,
        "ingest_utc": utc_text(frame.ingest_utc),
        "source_time_quality": frame.source_time_quality.value,
        "native_width": frame.native_width,
        "native_height": frame.native_height,
    }


def parse_token(text: str) -> tuple[int, int] | None:
    """(stream epoch, frame sequence) from ``<epoch>-<seq>``, or None if it is not one."""
    if not TOKEN.fullmatch(text):
        return None
    epoch, _, seq = text.partition("-")
    return int(epoch), int(seq)


def lower_thread_priority(increment: int = ENCODER_NICE) -> int | None:
    """Lower the calling thread's CPU priority; its new nice value, or None where this is not done.

    Linux only, where the nice value set through a thread ID belongs to that thread alone; never for the process's
    main thread. Raising a nice value needs no privilege.
    """
    if not sys.platform.startswith("linux"):
        return None
    tid = threading.get_native_id()
    if tid == os.getpid():
        return None
    try:
        current = os.getpriority(os.PRIO_PROCESS, tid)
        os.setpriority(os.PRIO_PROCESS, tid, min(19, current + increment))
        return os.getpriority(os.PRIO_PROCESS, tid)
    except OSError:
        return None


@dataclass(frozen=True)
class PreviewPicture:
    frame: FrameRef
    jpeg: bytes
    width: int
    height: int

    @property
    def token(self) -> str:
        return f"{self.frame.stream_epoch}-{self.frame.frame_seq}"

    @property
    def order(self) -> tuple[int, int]:
        return self.frame.stream_epoch, self.frame.frame_seq


@dataclass(frozen=True)
class StreamAnswer:
    """``next_frame``'s answer: a live picture and its age when sent, or, when there is none newer, the view (why)."""

    picture: PreviewPicture | None
    age_ms: int | None
    view: dict[str, Any] | None


class LivePreview:
    """The status page's view of the newest frame; thread-safe (page requests call it from their own threads)."""

    def __init__(
        self,
        frames: RecentFrames,
        clock: Clock,
        freshness: FreshnessConfig,
        connected: Callable[[], StreamIdentity | None],
        *,
        encode: Encoder = encode_jpeg,
        max_encodes_per_s: float = PREVIEW_MAX_ENCODES_PER_S,
        max_jpeg_bytes: int = PREVIEW_MAX_JPEG_BYTES,
        pressure: Callable[[], bool] | None = None,
        threaded: bool = True,
        wall: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 < max_encodes_per_s <= PREVIEW_MAX_ENCODES_PER_S:
            raise ValueError(f"max_encodes_per_s must be above 0 and at most {PREVIEW_MAX_ENCODES_PER_S:g}")
        if not 0 < max_jpeg_bytes <= PREVIEW_MAX_JPEG_BYTES:
            raise ValueError(f"max_jpeg_bytes must be above 0 and at most {PREVIEW_MAX_JPEG_BYTES}")
        self._frames = frames
        self._clock = clock
        self._freshness = freshness
        self._connected = connected
        self._encode = encode
        self._max_encodes_per_s = max_encodes_per_s
        self._max_bytes = max_jpeg_bytes
        self._pressure = pressure
        self._threaded = threaded  # False: no encoder thread (tests drive ``encode_newest`` themselves)
        self._wall = wall  # viewer idleness only; frame ages use ``clock``
        self._started = clock.mono()
        self._lock = threading.Lock()  # pictures, counters, the budget, the problem, the encoder's state
        self._published = threading.Condition(self._lock)  # a new picture was kept, or the preview closed
        self._encoding = threading.Lock()  # one encode at a time
        self._pictures: deque[PreviewPicture] = deque(maxlen=PREVIEW_PICTURES_KEPT)
        self._last_encode_ns: int | None = None
        self._encode_times: deque[int] = deque(maxlen=math.ceil(PREVIEW_MAX_ENCODES_PER_S) + 1)
        self._encode_ms: deque[float] = deque(maxlen=ENCODE_TIMES_KEPT)
        self._problem: str | None = None
        self._counts: Counter[str] = Counter()
        self._encoder: threading.Thread | None = None
        self._encoder_nice: int | None = None
        self._last_viewer: float | None = None
        self._closed = False

    # ------------------------------------------------------------ the page's calls

    def view(self, *, with_picture: bool = False) -> dict[str, Any]:
        """The preview's state now; ``with_picture`` (the HTML page): also a live picture, encoded if due."""
        latest = self._frames.latest()
        frame = latest[0] if latest is not None else None
        freshness = self._assess(frame)
        view: dict[str, Any] = {
            "enabled": True,
            "state": STATE_NAMES[freshness.state],
            "reason": self._reason(freshness),
            "age_ms": None if freshness.last_frame_age_ns is None else freshness.last_frame_age_ns // NS_PER_MS,
            "frame": None if frame is None else frame_facts(frame),
            "time_label": None if frame is None else time_label(frame.source_time_quality),
            "picture": None,
            "problem": None,
            "bounds": self.bounds(),
        }
        if with_picture and latest is not None and freshness.state is VideoState.FRESH:
            picture, _ = self._make(*latest, rate=self._rate())
            del latest  # the image reference ends here
            own = None if picture is None else self._assess(picture.frame)
            if picture is not None and own is not None and own.state is VideoState.FRESH:
                view["picture"] = {
                    "url": f"/preview/{picture.token}.jpg", **frame_facts(picture.frame),
                    "age_ms": own.last_frame_age_ns // NS_PER_MS,  # type: ignore[operator]  # FRESH has an age
                    "width": picture.width, "height": picture.height, "bytes": len(picture.jpeg),
                }
        with self._lock:
            view["problem"] = self._problem
        return view

    def picture(self, token: str) -> PreviewPicture | None:
        """The kept picture named ``token``, only while its own frame is still live; None otherwise."""
        if not TOKEN.fullmatch(token):
            return None
        with self._lock:
            picture = next((p for p in reversed(self._pictures) if p.token == token), None)
            if picture is None:
                self._counts["not_kept"] += 1
                return None
        live = self._assess(picture.frame).state is VideoState.FRESH
        with self._lock:
            self._counts["served" if live else "refused_not_current"] += 1
        return picture if live else None

    def next_frame(self, after: tuple[int, int] | None, wait_s: float = STREAM_WAIT_S) -> StreamAnswer:
        """The newest live picture newer than ``after`` (stream epoch, frame sequence), waiting at most ``wait_s``.

        Starts the encoder if it is not running. Without a newer live picture in time: the view (state and why).
        """
        with self._lock:
            if self._closed:
                return StreamAnswer(None, None, {"enabled": True, "state": "closed",
                                                 "reason": "the runtime is stopping", "frame": None, "age_ms": None})
            self._last_viewer = self._wall()
            self._counts["stream_requests"] += 1
            if self._threaded and self._encoder is None:
                self._encoder = threading.Thread(target=self._run_encoder, name="sentinel-live-encoder", daemon=True)
                self._encoder.start()
        deadline = time.monotonic() + max(0.0, wait_s)
        while True:
            with self._lock:
                newest = self._pictures[-1] if self._pictures else None
            if newest is not None and (after is None or newest.order > after):
                own = self._assess(newest.frame)
                if own.state is VideoState.FRESH:
                    with self._lock:
                        self._counts["stream_pictures_sent"] += 1
                    return StreamAnswer(newest, own.last_frame_age_ns // NS_PER_MS, None)  # type: ignore[operator]
            remaining = deadline - time.monotonic()
            with self._published:
                if remaining <= 0 or self._closed:
                    break
                current = self._pictures[-1] if self._pictures else None
                if current is newest:  # nothing kept since the look above: wait for the encoder (or re-check)
                    self._published.wait(min(remaining, ENCODER_WAKE_S))
        view = self.view()
        with self._lock:
            self._counts["stream_no_newer_picture"] += 1
        return StreamAnswer(None, None, view)

    def encode_newest(self) -> str:
        """The encoder's step: encode the newest frame if it is live, not yet encoded and due. Its outcome label."""
        latest = self._frames.latest()
        if latest is None:
            return "no_frame"
        if self._assess(latest[0]).state is not VideoState.FRESH:
            with self._lock:
                self._counts["stream_not_live"] += 1
            return "not_live"
        _, outcome = self._make(*latest, rate=self._rate())
        del latest
        return outcome

    def close(self, timeout_s: float = 1.0) -> bool:
        """Stop the encoder and answer waiting requests; False if the encoder is still running after ``timeout_s``."""
        with self._published:
            self._closed = True
            encoder = self._encoder
            self._published.notify_all()
        if encoder is not None and encoder is not threading.current_thread():
            encoder.join(timeout_s)
        return encoder is None or not encoder.is_alive()

    def bounds(self) -> dict[str, Any]:
        return {"max_width": PREVIEW_MAX_WIDTH, "max_height": PREVIEW_MAX_HEIGHT, "jpeg_quality": PREVIEW_JPEG_QUALITY,
                "max_encodes_per_s": self._max_encodes_per_s, "max_jpeg_bytes": self._max_bytes,
                "pictures_kept": PREVIEW_PICTURES_KEPT, "yield_encodes_per_s": PREVIEW_YIELD_ENCODES_PER_S,
                "stream_wait_s": STREAM_WAIT_S, "viewer_idle_s": VIEWER_IDLE_S,
                "stale_after_ms": round(self._freshness.stale_after_s * 1000)}

    def summary(self) -> dict[str, Any]:
        """For the runtime status: numbers and fixed labels only."""
        with self._lock:
            spent = list(self._encode_ms)
            return {"state": "enabled", "pictures_kept": len(self._pictures), "problem": self._problem,
                    "encoder": {"running": self._encoder is not None, "nice": self._encoder_nice},
                    "encode_ms": None if not spent else {
                        "last": len(spent), "median": round(statistics.median(spent), 2), "max": round(max(spent), 2)},
                    "counters": dict(sorted(self._counts.items()))}

    # ------------------------------------------------------------ internals

    def _assess(self, frame: FrameRef | None) -> VideoFreshness:
        return assess_frame(frame, self._connected(), self._clock.mono(), self._started, self._freshness)

    def _reason(self, freshness: VideoFreshness) -> str:
        config = self._freshness
        if freshness.state is VideoState.FRESH:
            return "the newest frame of the connected stream"
        if freshness.state is VideoState.STARTING:
            return "no frame yet since the runtime started"
        age = freshness.last_frame_age_ns
        if age is None:
            return "no frame received"
        if freshness.state is VideoState.OFFLINE:
            return f"no new frame for {age / NS_PER_SECOND:.0f} s (offline after {config.offline_after_s:g} s)"
        if age >= config.stale_after_ns:
            return f"no new frame for {age / NS_PER_SECOND:.1f} s (stale after {config.stale_after_s:g} s)"
        return "the stream connection this frame came from has ended; reconnecting"

    def _rate(self) -> float:
        """The encode budget now: the full rate, or the yield rate while monitoring work waits (or cannot be read)."""
        if self._pressure is None:
            return self._max_encodes_per_s
        try:
            waiting = bool(self._pressure())
        except Exception:  # noqa: BLE001 - unknown pressure counts as pressure: monitoring first
            waiting = True
        if not waiting:
            return self._max_encodes_per_s
        with self._lock:
            self._counts["yielded"] += 1
        return min(self._max_encodes_per_s, PREVIEW_YIELD_ENCODES_PER_S)

    def _due(self, now_ns: int, rate: float) -> bool:
        """At most ``rate`` encodes in any second, never two within MIN_SPACING of its interval (failed ones count)."""
        if self._last_encode_ns is not None and now_ns - self._last_encode_ns < MIN_SPACING * NS_PER_SECOND / rate:
            return False
        return sum(1 for t in self._encode_times if now_ns - t < NS_PER_SECOND) + 1 <= rate  # this one included

    def _make(self, frame: FrameRef, image: Any, *, rate: float) -> tuple[PreviewPicture | None, str]:
        """A picture for ``frame``: kept, newly encoded if due, or else the newest kept one (the caller checks it).

        A request that arrives during an encode waits for it (at most ENCODE_WAIT_S), so viewers of one frame share
        one encode. The outcome is reused, encoded, rate_limited, encode_busy or failed.
        """
        with self._lock:
            newest = self._pictures[-1] if self._pictures else None
            if newest is not None and newest.frame.key == frame.key:
                self._counts["reused"] += 1
                return newest, "reused"
        if not self._encoding.acquire(timeout=ENCODE_WAIT_S):
            with self._lock:
                self._counts["encode_busy"] += 1
                return (self._pictures[-1] if self._pictures else None), "encode_busy"
        try:
            with self._lock:  # another request may have encoded it while this one waited
                newest = self._pictures[-1] if self._pictures else None
                if newest is not None and newest.frame.key == frame.key:
                    self._counts["reused"] += 1
                    return newest, "reused"
                now = self._clock.monotonic_ns()
                if not self._due(now, rate):
                    self._counts["rate_limited"] += 1
                    return newest, "rate_limited"
                self._last_encode_ns = now  # a failed encode counts toward the budget too
                self._encode_times.append(now)
            problem = None
            began = time.perf_counter()  # the encode's own duration, for the summary only
            try:
                encoded = self._encode(image, frame.pixel_format, max_width=PREVIEW_MAX_WIDTH,
                                       max_height=PREVIEW_MAX_HEIGHT, quality=PREVIEW_JPEG_QUALITY)
            except EncodeError as exc:
                problem = exc.label
            except Exception as exc:  # noqa: BLE001 - a failed picture must not end the page; the class name only
                problem = f"encode_failed:{type(exc).__name__}"
            else:
                if len(encoded.data) > self._max_bytes:
                    problem = "picture_too_large"
                elif not encoded.data.startswith(SOI):
                    problem = "not_a_jpeg"
            spent_ms = (time.perf_counter() - began) * 1000
            with self._published:
                self._encode_ms.append(spent_ms)
                if problem is not None:
                    self._counts["encode_failures"] += 1
                    self._problem = problem
                    return None, "failed"
                picture = PreviewPicture(frame, encoded.data, encoded.width, encoded.height)
                self._pictures.append(picture)
                self._counts["encodes"] += 1
                self._problem = None
                self._published.notify_all()
                return picture, "encoded"
        finally:
            self._encoding.release()

    def _run_encoder(self) -> None:
        """The encoder thread: the newest live frame as it arrives, until no viewer has asked for VIEWER_IDLE_S."""
        nice = lower_thread_priority()
        with self._lock:
            self._encoder_nice = nice
            self._counts["encoder_starts"] += 1
        version = -1  # RecentFrames counts from 0: the first wait returns at once
        while True:
            with self._lock:
                idle = self._last_viewer is None or self._wall() - self._last_viewer > VIEWER_IDLE_S
                if self._closed or idle:
                    if self._encoder is threading.current_thread():
                        self._encoder = None
                    return
            seen = version
            version, latest = self._frames.wait_newer(version, ENCODER_WAKE_S)
            del latest  # ``encode_newest`` takes the newest itself; no reference is kept while waiting
            if version != seen:
                self.encode_newest()
