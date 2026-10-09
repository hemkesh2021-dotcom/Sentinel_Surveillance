"""The runtime loop's most recent frames, with their images (D69, D70).

``sentinel run``'s loop puts every frame it takes from the capture slot here,
before inference. The live preview reads the newest one; an incident snapshot
looks up the exact frame its rule observation came from, by identity, and
never takes another. Only references are kept: nothing is copied, and the
oldest entry is released when a newer one arrives. Portable.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from ..contracts import FrameKey, FrameRef

RECENT_FRAMES_KEPT = 2  # 640x480 BGR: 921,600 B each; the loop keeps no other reference after its step


class RecentFrames:
    """The last ``capacity`` frames and their images, by frame identity. Thread-safe."""

    def __init__(self, capacity: int = RECENT_FRAMES_KEPT) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._frames: OrderedDict[FrameKey, tuple[FrameRef, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, frame: FrameRef, image: Any) -> None:
        key = frame.key
        with self._lock:
            self._frames[key] = (frame, image)
            self._frames.move_to_end(key)
            while len(self._frames) > self._capacity:
                self._frames.popitem(last=False)

    def get(self, key: FrameKey) -> tuple[FrameRef, Any] | None:
        """That exact frame and its image, or None once it is no longer kept."""
        with self._lock:
            return self._frames.get(key)

    def latest(self) -> tuple[FrameRef, Any] | None:
        with self._lock:
            if not self._frames:
                return None
            return next(reversed(self._frames.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
