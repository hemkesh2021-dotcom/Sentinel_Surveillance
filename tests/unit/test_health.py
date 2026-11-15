from __future__ import annotations

from sentinel.config import FreshnessConfig
from sentinel.contracts import PixelFormat
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.media.health import FreshnessMonitor, VideoState

CONFIG = FreshnessConfig()  # stale 2 s, offline 10 s


def test_no_frame_ever_is_starting_then_offline() -> None:
    clock = FakeClock()
    monitor = FreshnessMonitor(CONFIG, clock)
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    assert monitor.assess(stamper.current_stream).state is VideoState.STARTING
    clock.advance(9.999)
    assert monitor.assess(stamper.current_stream).state is VideoState.STARTING
    clock.advance(0.001)
    fresh = monitor.assess(stamper.current_stream)
    assert (fresh.state, fresh.live, fresh.last_frame_age_ns) == (VideoState.OFFLINE, None, None)


def test_only_the_connected_epochs_recent_frame_is_fresh() -> None:
    clock = FakeClock()
    monitor = FreshnessMonitor(CONFIG, clock)
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    frame = stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
    monitor.on_frame(frame)
    assert monitor.assess(stamper.current_stream).live == frame.stream
    # Disconnected, or reconnected with no frame yet: not fresh, even 100 ms later.
    clock.advance(0.1)
    stamper.disconnect()
    assert monitor.assess(stamper.current_stream).state is VideoState.STALE
    stamper.connect()
    reconnected = monitor.assess(stamper.current_stream)
    assert (reconnected.state, reconnected.live) == (VideoState.STALE, None)
    # A late frame from the old epoch cannot make the new one fresh.
    monitor.on_frame(frame)
    assert monitor.assess(stamper.current_stream).state is VideoState.STALE


def test_frames_from_before_a_reboot_do_not_count() -> None:
    clock = FakeClock()
    monitor = FreshnessMonitor(CONFIG, clock)
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    monitor.on_frame(stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR))
    clock.reboot("fake-boot-2")
    assert monitor.assess(None).state is VideoState.OFFLINE
