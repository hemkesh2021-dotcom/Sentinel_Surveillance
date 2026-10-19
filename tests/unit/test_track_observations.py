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


def test_a_tracks_life_begins_at_its_first_measurement_and_again_after_it_expires(
    next_frame: NextFrame, stamper: FrameStamper, clock: FakeClock
) -> None:
    # Review A, finding 6: identity counts votes only from a track's current life (TrackTable.since), so a tracker
    # that gives an expired track its number back does not join the two lives.
    from sentinel.tracking.tracks import TrackTable

    table = TrackTable(TRACK_EXPIRY_NS)
    live = stamper.current_stream

    def see(frame: FrameRef, *ids: int) -> FrameRef:
        table.update(frame, [detect(frame, track_id=i) for i in ids], live)
        table.current(live, clock.mono())
        return frame

    first = see(next_frame(), 7)
    see(next_frame(dt=0.5), 7)
    see(next_frame(dt=1.0), 7)  # measured again exactly at its expiry age: the same life
    key = detect(first, track_id=7).key
    assert table.since(key) == first.ingest_mono_ns
    see(next_frame(dt=0.6))
    gone = see(next_frame(dt=0.6))  # expired and dropped by current()
    with pytest.raises(KeyError):
        table.since(key)
    back = see(next_frame(dt=0.1), 7)  # the number given back: a new life
    assert table.since(key) == back.ingest_mono_ns > gone.ingest_mono_ns
    again = next_frame(dt=1.5)  # measured again after expiring, with no current() call in between
    table.update(again, [detect(again, track_id=7)], live)
    assert table.since(key) == again.ingest_mono_ns
    stamper.connect()
    other = next_frame()  # a new stream epoch: nothing carried over
    table.update(other, [detect(other, track_id=7)], stamper.current_stream)
    assert table.since(detect(other, track_id=7).key) == other.ingest_mono_ns and table.size == 1
