"""Live preview on the D-2 status page (D69; opt-in: ``sentinel run --live-preview``).

It shows the newest frame ``sentinel run`` already reads, on the loopback-only
page, with its age and state. It is the alpha's stand-in for the guide's Live
view (V2-18, over V2-05/V2-06's compressed relay), not that view.

- **No second camera connection.** The runtime loop puts each frame it takes
  from the one capture slot into RecentFrames (a reference, before
  inference). Nothing here reads the camera.
- **Never in the loop's way.** The loop only swaps a reference. Encoding runs
  in a page request's thread, only while someone views the page: at most
  ``max_encodes_per_s`` pictures per second for all viewers together, one at a
  time, each frame at most once. A request that comes sooner gets the newest
  picture already encoded, described by its own frame, not the newest frame.
- **Bounded.** A picture is scaled down to fit PREVIEW_MAX_WIDTH x
  PREVIEW_MAX_HEIGHT, at a fixed quality, and dropped above
  PREVIEW_MAX_JPEG_BYTES. At most PREVIEW_PICTURES_KEPT pictures are kept.
  While a picture is encoded, the encoder holds one frame reference beyond
  RecentFrames' own.
- **Live, stale, offline, starting,** decided on the injected monotonic clock
  as the runtime decides video freshness (``media.health.assess_frame``):
  live only while the frame belongs to the connected stream epoch and is
  younger than ``freshness.stale_after_s``. A picture is offered, and served,
  only while its own frame is live. Otherwise the page shows the state, the
  age and the frame's identity, and no picture: a frozen frame is never
  shown as current.
- **Times.** A frame's time is its ingest time: when this device received and
  decoded it, not when the camera captured it (``source_time_quality`` says
  what is known about capture time; this build maps it to no clock). Ages are
  monotonic, from ingest to the moment the view was made.
- **Saves nothing.** Pictures exist in memory and on the wire only.

Portable: the default encoder imports cv2 only when it runs (``media.jpeg``).
"""

from __future__ import annotations

import re
import threading
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
PREVIEW_MAX_ENCODES_PER_S = 2.0
PREVIEW_MAX_JPEG_BYTES = 262_144
PREVIEW_PICTURES_KEPT = 2
ENCODE_WAIT_S = 1.0  # a request waits at most this long for another request's encode, then shows what exists
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


@dataclass(frozen=True)
class PreviewPicture:
    frame: FrameRef
    jpeg: bytes
    width: int
    height: int

    @property
    def token(self) -> str:
        return f"{self.frame.stream_epoch}-{self.frame.frame_seq}"


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
        self._min_interval_ns = round(NS_PER_SECOND / max_encodes_per_s)
        self._max_bytes = max_jpeg_bytes
        self._started = clock.mono()
        self._lock = threading.Lock()  # pictures, counters, the last encode's time, the problem
        self._encoding = threading.Lock()  # one encode at a time
        self._pictures: deque[PreviewPicture] = deque(maxlen=PREVIEW_PICTURES_KEPT)
        self._last_encode_ns: int | None = None
        self._problem: str | None = None
        self._counts: Counter[str] = Counter()

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
            picture = self._picture(*latest)
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

    def bounds(self) -> dict[str, Any]:
        return {"max_width": PREVIEW_MAX_WIDTH, "max_height": PREVIEW_MAX_HEIGHT, "jpeg_quality": PREVIEW_JPEG_QUALITY,
                "max_encodes_per_s": self._max_encodes_per_s, "max_jpeg_bytes": self._max_bytes,
                "pictures_kept": PREVIEW_PICTURES_KEPT}

    def summary(self) -> dict[str, Any]:
        """For the runtime status: numbers and fixed labels only."""
        with self._lock:
            return {"state": "enabled", "pictures_kept": len(self._pictures), "problem": self._problem,
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

    def _due(self) -> bool:
        return self._last_encode_ns is None or self._clock.monotonic_ns() - self._last_encode_ns >= self._min_interval_ns

    def _picture(self, frame: FrameRef, image: Any) -> PreviewPicture | None:
        """A picture for ``frame``: kept, newly encoded if due, or else the newest kept one (the caller checks it).

        A request that arrives during an encode waits for it (at most ENCODE_WAIT_S), so viewers of one frame share
        one encode.
        """
        with self._lock:
            newest = self._pictures[-1] if self._pictures else None
            if newest is not None and newest.frame.key == frame.key:
                self._counts["reused"] += 1
                return newest
        if not self._encoding.acquire(timeout=ENCODE_WAIT_S):
            with self._lock:
                self._counts["encode_busy"] += 1
                return self._pictures[-1] if self._pictures else None
        try:
            with self._lock:  # another request may have encoded it while this one waited
                newest = self._pictures[-1] if self._pictures else None
                if newest is not None and newest.frame.key == frame.key:
                    self._counts["reused"] += 1
                    return newest
                if not self._due():
                    self._counts["rate_limited"] += 1
                    return newest
                self._last_encode_ns = self._clock.monotonic_ns()  # a failed encode counts toward the rate too
            problem = None
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
            with self._lock:
                if problem is not None:
                    self._counts["encode_failures"] += 1
                    self._problem = problem
                    return None
                picture = PreviewPicture(frame, encoded.data, encoded.width, encoded.height)
                self._pictures.append(picture)
                self._counts["encodes"] += 1
                self._problem = None
                return picture
        finally:
            self._encoding.release()
