from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from sentinel.contracts import (
    Applicability,
    FrameRef,
    PixelFormat,
    ResizeTransform,
    SourceTimeQuality,
    stream_relation,
)
from sentinel.media.clock import CrossBootComparisonError, FakeClock
from sentinel.media.frames import FrameNovelty, FrameStamper, frame_novelty

NextFrame = Callable[..., FrameRef]


def stamp_1080p(stamper: FrameStamper) -> FrameRef:
    return stamper.stamp(native_width=1920, native_height=1080, pixel_format=PixelFormat.BGR)


def test_pts_resets_do_not_recycle_frame_identity(
    next_frame: NextFrame, stamper: FrameStamper
) -> None:
    # The RTP clock resets after the third frame, and a PTS value later repeats.
    frames = [next_frame(pts=pts) for pts in (90_000, 93_000, 96_000, 0, 3_000, 90_000)]

    assert len({frame.key for frame in frames}) == len(frames)
    verdicts, last = [], None
    for frame in frames:
        verdicts.append(frame_novelty(frame.key, last, stamper.current_stream))
        last = frame.key
    assert verdicts == [FrameNovelty.NEW_EPOCH] + [FrameNovelty.NEW] * 5


def test_unset_and_non_increasing_pts_at_stream_start_fall_back_to_receive_time(
    next_frame: NextFrame, stamper: FrameStamper, clock: FakeClock
) -> None:
    # Measured on the camera's substream (V2-01): the first packet has no
    # timestamp and the next timestamps can repeat or step back before they
    # settle. Stamping must neither fail nor trust them as stream time.
    pts_in = (None, 0, 0, 6_000, 3_000, 12_000, 18_000)
    frames = [next_frame(pts=pts) for pts in pts_in]

    assert len({frame.key for frame in frames}) == len(frames)
    assert [frame.frame_seq for frame in frames] == list(range(len(frames)))
    assert [frame.source_pts for frame in frames] == list(pts_in)  # kept for diagnostics
    assert [frame.source_time_quality for frame in frames] == [
        SourceTimeQuality.NONE,  # unset
        SourceTimeQuality.STREAM_RELATIVE,  # first timestamp seen
        SourceTimeQuality.NONE,  # repeated
        SourceTimeQuality.STREAM_RELATIVE,
        SourceTimeQuality.NONE,  # stepped back
        SourceTimeQuality.STREAM_RELATIVE,
        SourceTimeQuality.STREAM_RELATIVE,
    ]
    # Receive time orders every frame, whatever its source timestamp.
    ingest = [frame.ingest_mono_ns for frame in frames]
    assert ingest == sorted(ingest) and len(set(ingest)) == len(ingest)
    # An explicit claim cannot upgrade a timestamp that did not advance.
    clock.advance(1 / 15)
    repeated = stamper.stamp(
        native_width=640,
        native_height=480,
        pixel_format=PixelFormat.BGR,
        source_pts=18_000,
        source_time_quality=SourceTimeQuality.CAPTURE_SYNCED,
    )
    assert repeated.source_time_quality is SourceTimeQuality.NONE

    # A new epoch starts over: RTSP timestamps begin at 0 again after reconnect.
    stamper.disconnect()
    stamper.connect()
    assert next_frame(pts=0).source_time_quality is SourceTimeQuality.STREAM_RELATIVE


def test_frozen_latest_frame_is_never_processed_twice(
    next_frame: NextFrame, stamper: FrameStamper, clock: FakeClock
) -> None:
    # v1's FrameReader.get() keeps returning its last frame after the camera stalls.
    frame = next_frame()
    live = stamper.current_stream
    assert frame_novelty(frame.key, None, live) is FrameNovelty.NEW_EPOCH
    for _ in range(3):
        clock.advance(0.5)
        assert frame_novelty(frame.key, frame.key, live) is FrameNovelty.DUPLICATE

    newer = next_frame()
    assert frame_novelty(frame.key, newer.key, live) is FrameNovelty.OUT_OF_ORDER


def test_reconnect_ends_the_old_epoch_for_consumers(
    next_frame: NextFrame, stamper: FrameStamper
) -> None:
    before = next_frame()
    old_stream = stamper.current_stream

    stamper.disconnect()
    assert frame_novelty(before.key, None, stamper.current_stream) is FrameNovelty.NOT_LIVE

    new_stream = stamper.connect()
    after = next_frame()
    # Guide chapter 6: reconnection increments stream_epoch.
    assert (new_stream.run_id, new_stream.stream_epoch) == (
        old_stream.run_id,
        old_stream.stream_epoch + 1,
    )
    assert after.key != before.key
    # A late frame from the old epoch is rejected; the first new one resets per-epoch state.
    assert stream_relation(before.stream, new_stream) is Applicability.SUPERSEDED_EPOCH
    assert frame_novelty(before.key, None, new_stream) is FrameNovelty.NOT_LIVE
    assert frame_novelty(after.key, before.key, new_stream) is FrameNovelty.NEW_EPOCH


def test_restart_in_the_same_boot_never_reuses_a_stream_identity(clock: FakeClock) -> None:
    first_run = FrameStamper("cam-1", clock)
    first_run.connect()
    old = stamp_1080p(first_run)

    # The service restarts in the same boot after the wall clock stepped backwards,
    # at the same monotonic reading, and its counters start over.
    clock.step_utc(timedelta(hours=-2))
    second_run = FrameStamper("cam-1", clock)
    live = second_run.connect()
    new = stamp_1080p(second_run)

    assert (new.stream_epoch, new.frame_seq, new.ingest_mono_ns) == (
        old.stream_epoch,
        old.frame_seq,
        old.ingest_mono_ns,
    )
    assert new.key != old.key
    assert stream_relation(old.stream, live) is Applicability.OTHER_RUN
    assert frame_novelty(old.key, None, live) is FrameNovelty.NOT_LIVE


def test_frames_from_before_a_reboot_are_isolated(
    next_frame: NextFrame, stamper: FrameStamper, clock: FakeClock
) -> None:
    old = next_frame()
    # The new boot's monotonic counter happens to read the same value.
    clock.reboot("fake-boot-2", mono_ns=old.ingest_mono_ns)

    with pytest.raises(RuntimeError, match="boot changed"):
        stamp_1080p(stamper)
    live = FrameStamper("cam-1", clock).connect()
    assert stream_relation(old.stream, live) is Applicability.OTHER_BOOT
    assert frame_novelty(old.key, None, live) is FrameNovelty.NOT_LIVE
    with pytest.raises(CrossBootComparisonError):
        clock.mono().ns_since(old.ingest_instant)


def test_letterbox_transform_round_trips_native_coordinates() -> None:
    transform = ResizeTransform.letterbox(1920, 1080, 640, 640)
    # 1920x1080 content scales to 640x360 with 140 px bars above and below.
    assert transform.to_processed(0, 0) == pytest.approx((0, 140))
    assert transform.to_processed(1920, 1080) == pytest.approx((640, 500))
    for point in [(0.0, 0.0), (960.0, 540.0), (1919.5, 1079.5)]:
        assert transform.to_native(*transform.to_processed(*point)) == pytest.approx(point)


def test_frame_rejects_a_transform_that_does_not_fit_its_output(stamper: FrameStamper) -> None:
    # Portrait content with a landscape letterbox would overflow the 640 px output.
    with pytest.raises(ValidationError, match="outside the processed image"):
        stamper.stamp(
            native_width=1080,
            native_height=1920,
            pixel_format=PixelFormat.BGR,
            resize=ResizeTransform.letterbox(1920, 1080, 640, 640),
        )


IST = timezone(timedelta(hours=5, minutes=30))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"ingest_utc": datetime(2026, 1, 1, 12, 0)}, "timezone info"),
        ({"ingest_utc": datetime(2026, 1, 1, 17, 30, tzinfo=IST)}, "UTC datetime"),
        ({"source_time_quality": SourceTimeQuality.STREAM_RELATIVE}, "requires source_pts"),
    ],
)
def test_frame_times_must_be_unambiguous(
    next_frame: NextFrame, changes: dict[str, object], message: str
) -> None:
    fields = next_frame().model_dump() | changes
    with pytest.raises(ValidationError, match=message):
        FrameRef.model_validate(fields)
