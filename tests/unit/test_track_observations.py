from __future__ import annotations

from collections.abc import Callable

import pytest

from sentinel.contracts import (
    Applicability,
    FrameRef,
    NormalizedBox,
    TrackObservation,
    TrackStatus,
)
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.media.frames import FrameStamper

NextFrame = Callable[..., FrameRef]
BOX = NormalizedBox(x1=0.40, y1=0.20, x2=0.55, y2=0.90)
TRACK_EXPIRY_NS = NS_PER_SECOND  # guide chapter 6 starting value


def detect(frame: FrameRef, *, track_id: int = 7) -> TrackObservation:
    return TrackObservation.detected(
        frame, track_id=track_id, box=BOX, confidence=0.81, status=TrackStatus.CONFIRMED
    )


def test_predictions_do_not_keep_a_track_alive(
    next_frame: NextFrame, stamper: FrameStamper, clock: FakeClock
) -> None:
    # Guide chapter 8: predicted observations are labelled and still expire.
    live = stamper.current_stream
    observation = detect(next_frame())
    observation = observation.predicted_on(next_frame(dt=0.4), box=BOX)
    observation = observation.predicted_on(next_frame(dt=0.4), box=BOX)
    assert observation.applicability(live, clock.mono(), expiry_ns=TRACK_EXPIRY_NS) is (
        Applicability.CURRENT
    )

    observation = observation.predicted_on(next_frame(dt=0.2), box=BOX)  # 1 s after detection
    assert observation.predicted
    assert observation.applicability(live, clock.mono(), expiry_ns=TRACK_EXPIRY_NS) is (
        Applicability.EXPIRED
    )

    refreshed = detect(next_frame())
    assert refreshed.applicability(live, clock.mono(), expiry_ns=TRACK_EXPIRY_NS) is (
        Applicability.CURRENT
    )


def test_track_ids_are_scoped_to_their_stream_epoch(
    next_frame: NextFrame, stamper: FrameStamper, clock: FakeClock
) -> None:
    # A tracker restarted after a reconnect reuses small IDs; v1 keyed face results
    # by the bare ID, so a new person could inherit an old identity.
    before = detect(next_frame(), track_id=1)
    live = stamper.connect()
    after = detect(next_frame(), track_id=1)

    assert after.key != before.key
    assert before.applicability(live, clock.mono(), expiry_ns=TRACK_EXPIRY_NS) is (
        Applicability.SUPERSEDED_EPOCH
    )
    with pytest.raises(ValueError, match="across stream epochs"):
        before.predicted_on(next_frame(), box=BOX)
