from __future__ import annotations

from collections.abc import Callable

import pytest

from sentinel.contracts import FrameRef, PixelFormat
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper

FRAME_INTERVAL_S = 1 / 15


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def stamper(clock: FakeClock) -> FrameStamper:
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    return stamper


@pytest.fixture
def next_frame(clock: FakeClock, stamper: FrameStamper) -> Callable[..., FrameRef]:
    """Advance the clock (default: one 15 fps interval) and ingest a 1080p frame."""

    def _next(*, dt: float = FRAME_INTERVAL_S, pts: int | None = None) -> FrameRef:
        clock.advance(dt)
        return stamper.stamp(
            native_width=1920, native_height=1080, pixel_format=PixelFormat.BGR, source_pts=pts
        )

    return _next
