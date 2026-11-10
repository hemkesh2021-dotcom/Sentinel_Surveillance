"""Zooming into a region of one held picture, from the recorded pixels (session 116, part 28; experimental).

"Can you zoom in the pic?" used to reach a model that only has a 480x360 copy of the picture. A zoomed answer is now a
question about a **crop taken from the decoded picture itself**: on the main-stream profile the 2304x1296 picture the
working copy was made from, cut at the selected rectangle and only then fitted to the model's image size. It is never
the small copy enlarged, and never the browser's own magnification. What it adds is bounded by the recorded pixels: a
thing too small or blurred in the recorded picture stays unreadable, and the answer card says so.

**Pinned frames.** The runtime keeps a reference to the newest frame's pictures in one slot (no copy: the loop's own
arrays, replaced by the next frame). When a viewer asks to zoom or scan, that frame is **pinned**: held with an id for
at most ``PIN_KEEP_S``, at most ``PINS_KEPT`` at once, the oldest pushed out first. A pin holds the working picture and,
on the main-stream profile, the decoded picture (2304x1296 BGR is 8,957,952 bytes; two pins are about 18 MB). No
disk, no snapshot, no archive: expired or pushed out, a pin is gone and a question about it is refused, never answered
from another frame. A pin keeps its own frame's facts (epoch, sequence, ingest time), so a reconnect does not confuse
it: it stays that earlier frame, and is marked historical.

**Regions.** A selection is a rectangle in fractions of the held picture. The working picture is the whole decoded
picture resized (no cropping, media.capture), so the same fractions address both. ``roi_pixels`` maps it to whole
pixels of the source (outward: floor the start, ceil the end, clamped) and refuses a rectangle smaller than
``MIN_ROI_PX`` on a side rather than enlarging a few pixels into false detail. The crop is encoded by the runtime's
bounded JPEG encoder (media.jpeg: ``fit_within``), which gives the size sent: the crop's own aspect ratio, inside the
model's image size, never enlarged. So the image-token cap and the memory profile
of the model's requests are exactly the existing ones, and one crop is asked about at a time, as any question.

**Scanning the visible room** is the same thing ``SCAN_ROWS`` x ``SCAN_COLS`` times: fixed overlapping tiles of one
pinned frame, each asked about as its own question in turn (the page runs it; every tile goes through the usual
acceptance, interval, budget, gate and cancellation). It covers what the camera saw in that one frame: not areas out
of view, and not a camera movement.

Named regions are grounded only in the tracker's own boxes on the pinned frame (``Pin.people``); anything else needs a
selection. Portable: no OpenCV here (the crop is a slice; resizing and encoding are injected).
"""

from __future__ import annotations

import math
import secrets
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..media.clock import NS_PER_SECOND

PINS_KEPT = 2
PIN_KEEP_S = 120.0
MIN_ROI_PX = 64  # a selection smaller than this on a side, in the source's pixels, is refused
SCAN_ROWS, SCAN_COLS = 2, 2
SCAN_OVERLAP = 0.10  # of a tile's size, shared with its neighbour
SCAN_DEADLINE_S = 120.0  # the page's bound for one scan; a tile not analysed by then is reported as such
PEOPLE_KEPT = 8  # tracked people's boxes kept with a pin, for a grounded "zoom on this person"
MAIN_STREAM, WORKING = "main_stream", "working_picture"


class RoiError(ValueError):
    """A selection that cannot be used, by a fixed label."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


def roi_pixels(width: int, height: int, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """The whole-pixel rectangle (x1, y1, x2, y2; end exclusive) of a selection given in fractions of the picture."""
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        raise RoiError("the region needs four numbers") from None
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
        raise RoiError("the region must lie inside the picture, left before right and top before bottom")
    left, top = min(width - 1, math.floor(x1 * width)), min(height - 1, math.floor(y1 * height))
    right, bottom = min(width, math.ceil(x2 * width)), min(height, math.ceil(y2 * height))
    if right - left < MIN_ROI_PX or bottom - top < MIN_ROI_PX:
        raise RoiError(f"the region is smaller than {MIN_ROI_PX} recorded pixels on a side: there is too little in it "
                       "to analyse, and enlarging it would add nothing")
    return left, top, right, bottom


def scan_tiles(rows: int = SCAN_ROWS, cols: int = SCAN_COLS, overlap: float = SCAN_OVERLAP
               ) -> list[tuple[str, tuple[float, float, float, float]]]:
    """The scan's tiles, row by row: (name, region in fractions). Together they cover the whole picture; neighbours
    share ``overlap`` of a tile's size."""
    names_y = ["top", "bottom"] if rows == 2 else [f"row {n + 1}" for n in range(rows)]
    names_x = ["left", "right"] if cols == 2 else [f"column {n + 1}" for n in range(cols)]
    tile_w, tile_h = 1 / (cols - (cols - 1) * overlap), 1 / (rows - (rows - 1) * overlap)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * tile_w * (1 - overlap), r * tile_h * (1 - overlap)
            tiles.append((f"{names_y[r]} {names_x[c]}",
                          (round(max(0.0, x1), 6), round(max(0.0, y1), 6),
                           round(min(1.0, x1 + tile_w), 6), round(min(1.0, y1 + tile_h), 6))))
    return tiles


@dataclass(frozen=True)
class PreparedCrop:
    """A crop already encoded for the model: its JPEG is sent as it is (scene.chat.LlamaChatRequest)."""

    jpeg: bytes
    width: int
    height: int


@dataclass(frozen=True)
class Pin:
    """One held frame: its facts, its pictures and the tracker's people on it."""

    picture: str
    facts: dict[str, Any]  # stream_epoch, frame_seq, ingest_utc
    ingest_mono_ns: int
    pixel_format: Any  # the frame's (contracts.PixelFormat), for the encoder
    working: Any
    working_size: tuple[int, int]
    native: Any  # the decoded main-stream picture; None on the substream profile
    native_size: tuple[int, int] | None
    people: tuple[tuple[int, tuple[float, float, float, float]], ...]  # (track number, its box in fractions)
    pinned_ns: int

    @property
    def source(self) -> tuple[str, Any, tuple[int, int]]:
        """(which picture a crop is cut from, the picture, its width and height)."""
        if self.native is not None and self.native_size is not None:
            return MAIN_STREAM, self.native, self.native_size
        return WORKING, self.working, self.working_size


def _size(image: Any) -> tuple[int, int] | None:
    shape = getattr(image, "shape", None)
    return None if shape is None or len(shape) < 2 else (int(shape[1]), int(shape[0]))


class PinnedFrames:
    """The newest frame's pictures in one slot, and the frames viewers pinned (see the module docstring). ``latest``
    on the runtime's loop thread; everything else on status-page and scene-worker threads. One lock."""

    def __init__(self, kept: int = PINS_KEPT, keep_s: float = PIN_KEEP_S) -> None:
        self._kept, self._keep_ns = kept, round(keep_s * NS_PER_SECOND)
        self._lock = threading.Lock()
        self._latest: tuple[Any, Any, Any, tuple[tuple[int, tuple[float, float, float, float]], ...]] | None = None
        self._pins: OrderedDict[str, Pin] = OrderedDict()

    def latest(self, frame: Any, working: Any, native: Any, people: Any = ()) -> None:
        """The loop's newest frame, its working picture, its decoded picture (or None) and the tracker's people on
        it: references only, replacing the previous frame's."""
        boxes = tuple((int(p.track_id), (p.box.x1, p.box.y1, p.box.x2, p.box.y2)) for p in people)[:PEOPLE_KEPT]
        with self._lock:
            self._latest = (frame, working, native, boxes)

    def pin(self, facts_of: Callable[[Any], dict[str, Any] | None], now_ns: int) -> Pin | None:
        """Pin the newest frame, with the facts ``facts_of`` gives for exactly that frame; None when there is no
        frame, its pictures have no usable size, or ``facts_of`` returns None (it is not live: nothing is pinned)."""
        with self._lock:
            if self._latest is None:
                return None
            frame, working, native, people = self._latest
            working_size = _size(working)
            if working_size is None:
                return None
            self._expire(now_ns)
            facts = facts_of(frame)
            if facts is None:
                return None
            pin = Pin(secrets.token_hex(8), dict(facts), frame.ingest_mono_ns, frame.pixel_format, working,
                      working_size, native, _size(native) if native is not None else None, people, now_ns)
            self._pins[pin.picture] = pin
            while len(self._pins) > self._kept:
                self._pins.popitem(last=False)
            return pin

    def get(self, picture: str, now_ns: int) -> Pin | None:
        with self._lock:
            self._expire(now_ns)
            return self._pins.get(picture)

    def release(self) -> None:
        """Drop every pin and the newest frame's reference (chat turned off, the runtime stopping)."""
        with self._lock:
            self._pins.clear()
            self._latest = None

    def numbers(self, now_ns: int) -> dict[str, int]:
        """Counts and the bytes the pins hold (their decoded and working pictures)."""
        with self._lock:
            self._expire(now_ns)
            held = sum(int(getattr(image, "nbytes", 0) or 0) for pin in self._pins.values()
                       for image in (pin.working, pin.native) if image is not None)
            return {"pinned": len(self._pins), "limit": self._kept, "bytes": held}

    def _expire(self, now_ns: int) -> None:
        for picture in [p for p, pin in self._pins.items() if now_ns - pin.pinned_ns > self._keep_ns]:
            del self._pins[picture]
