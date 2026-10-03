"""V2-09/V2-10 demo form: the person tracker boundary with a scripted backend (no model, no GPU)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pytest

from sentinel.config import parse_config
from sentinel.contracts import FrameRef, NormalizedBox, PixelFormat, TrackStatus
from sentinel.live_state import Capability, Occupancy
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.runtime import EdgeCore
from sentinel.tracking.tracker import FrameOutcome, PersonTracker, RawTrack, TrackerError
from sentinel.tracking.tracks import MAX_TRACKS

FRAME_S = 1 / 15


class Image:
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.shape = (height, width, 3)


class ScriptedBackend:
    """Returns one scripted entry per track() call: a list of RawTracks or an exception."""

    def __init__(self, script: Iterable[object] = ()) -> None:
        self.script = list(script)
        self.calls = 0
        self.resets = 0
        self.fail_reset = False

    def track(self, image: object) -> Sequence[RawTrack]:
        self.calls += 1
        entry = self.script.pop(0) if self.script else []
        if isinstance(entry, Exception):
            raise entry
        return entry  # type: ignore[return-value]

    def reset(self) -> None:
        if self.fail_reset:
            raise RuntimeError("reset broke at /dev/nvmap")
        self.resets += 1


def person(track_id: int, x1: float = 100, y1: float = 50, x2: float = 200, y2: float = 400, conf: float = 0.9) -> RawTrack:
    return RawTrack(track_id, x1, y1, x2, y2, conf)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def stamper(clock: FakeClock) -> FrameStamper:
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    return stamper


def next_frame(stamper: FrameStamper, clock: FakeClock, width: int = 640, height: int = 480) -> FrameRef:
    clock.advance(FRAME_S)
    return stamper.stamp(native_width=width, native_height=height, pixel_format=PixelFormat.BGR)


def test_detections_become_normalized_native_image_observations(stamper: FrameStamper, clock: FakeClock) -> None:
    backend = ScriptedBackend([[person(3, 64, 48, 320, 480, 0.75), person(1, 0, 0, 640, 240, 0.5)]])
    tracker = PersonTracker(backend)
    frame = next_frame(stamper, clock)

    result = tracker.process(frame, Image())

    assert result.outcome is FrameOutcome.PROCESSED and backend.resets == 1  # reset before the first frame
    first, second = result.persons  # ordered by track ID
    assert (first.track_id, second.track_id) == (1, 3)
    assert second.box == NormalizedBox(x1=0.1, y1=0.1, x2=0.5, y2=1.0)
    assert first.box == NormalizedBox(x1=0.0, y1=0.0, x2=1.0, y2=0.5)
    assert (second.detector_confidence, second.predicted, second.frame) == (0.75, False, frame.key)
    assert second.observed_mono_ns == second.last_measured_mono_ns == frame.ingest_mono_ns
    assert result.backend_ns is not None and result.backend_ns >= 0


def test_boxes_are_clipped_to_the_image_and_sub_pixel_boxes_dropped(stamper: FrameStamper, clock: FakeClock) -> None:
    backend = ScriptedBackend([[person(1, -20, -5, 700, 500), person(2, 639.5, 10, 660, 90), person(3, 10, 479.2, 50, 481)]])
    tracker = PersonTracker(backend)

    (kept,) = tracker.process(next_frame(stamper, clock), Image()).persons

    assert kept.box == NormalizedBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    assert tracker.counters["boxes_dropped_small"] == 2


def test_confirmation_follows_v1s_score(stamper: FrameStamper, clock: FakeClock) -> None:
    # v1: +1 per detection (cap confirm + 2), -1 per frame without it, visible from 2.
    seen = [True, True, False, True, False, False, True, False, True]
    backend = ScriptedBackend([[person(5)] if s else [] for s in seen])
    tracker = PersonTracker(backend, confirm_detections=2)
    statuses = []
    for _ in seen:
        persons = tracker.process(next_frame(stamper, clock), Image()).persons
        statuses.append(persons[0].status.value[0].upper() if persons else "-")
    # scores:      1    2    1    2    1    0 (forgotten) 1    0    1
    assert statuses == ["T", "C", "-", "C", "-", "-", "T", "-", "T"]


def test_occluded_track_returns_confirmed_and_is_never_reported_while_hidden(
    stamper: FrameStamper, clock: FakeClock
) -> None:
    # Occlusion fixture: four detections, three frames hidden (the backend keeps the ID), back.
    backend = ScriptedBackend([[person(2)]] * 4 + [[]] * 3 + [[person(2, x1=140, x2=240)]])
    tracker = PersonTracker(backend)
    results = [tracker.process(next_frame(stamper, clock), Image()) for _ in range(8)]

    assert [len(r.persons) for r in results] == [1, 1, 1, 1, 0, 0, 0, 1]  # no predictions while hidden
    assert results[-1].persons[0].status is TrackStatus.CONFIRMED  # score 4 - 3 + 1 = 2
    assert results[-1].persons[0].box.x1 == pytest.approx(140 / 640)


def test_a_new_epoch_resets_the_backend_ids_and_confirmations(stamper: FrameStamper, clock: FakeClock) -> None:
    backend = ScriptedBackend([[person(1)], [person(1)], [person(1)]])
    tracker = PersonTracker(backend)
    tracker.process(next_frame(stamper, clock), Image())
    old = tracker.process(next_frame(stamper, clock), Image()).persons[0]
    assert old.status is TrackStatus.CONFIRMED

    stamper.disconnect()
    stamper.connect()  # reconnect: epoch 2; the backend restarts its IDs at 1
    new = tracker.process(next_frame(stamper, clock), Image()).persons[0]

    assert backend.resets == 2 and tracker.counters["epoch_resets"] == 1
    assert new.status is TrackStatus.TENTATIVE  # confirmation does not carry over
    assert new.key != old.key and new.key.stream_epoch == old.key.stream_epoch + 1


def test_a_geometry_change_is_a_new_epoch_and_boxes_use_the_new_size(clock: FakeClock) -> None:
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    backend = ScriptedBackend([[person(1, 0, 0, 320, 240)], [person(1, 0, 0, 320, 240)]])
    tracker = PersonTracker(backend)
    small = tracker.process(next_frame(stamper, clock), Image()).persons[0]
    stamper.disconnect()
    stamper.connect()  # the capture worker ends a connection on a size change (D39)
    large = tracker.process(next_frame(stamper, clock, 1280, 960), Image(1280, 960)).persons[0]
    assert small.box == NormalizedBox(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
    assert large.box == NormalizedBox(x1=0.0, y1=0.0, x2=0.25, y2=0.25)
    assert backend.resets == 2


def test_frames_are_processed_once_and_in_order(stamper: FrameStamper, clock: FakeClock) -> None:
    backend = ScriptedBackend()
    tracker = PersonTracker(backend)
    first = next_frame(stamper, clock)
    second = next_frame(stamper, clock)
    assert tracker.process(second, Image()).outcome is FrameOutcome.PROCESSED
    assert tracker.process(second, Image()).problem == "not_newer"  # e.g. a frozen buffer
    assert tracker.process(first, Image()).problem == "not_newer"
    stamper.disconnect()
    stamper.connect()
    assert tracker.process(next_frame(stamper, clock), Image()).outcome is FrameOutcome.PROCESSED
    assert tracker.process(second, Image()).problem == "older_epoch"
    assert backend.calls == 2


def test_a_backend_failure_is_a_label_and_ids_never_repeat_in_the_epoch(stamper: FrameStamper, clock: FakeClock) -> None:
    backend = ScriptedBackend([[person(1), person(2)], RuntimeError("CUDA error at /home/user/secret"), [person(1)]])
    tracker = PersonTracker(backend)
    before = tracker.process(next_frame(stamper, clock), Image())

    failed = tracker.process(next_frame(stamper, clock), Image())
    after = tracker.process(next_frame(stamper, clock), Image())

    assert (failed.outcome, failed.problem, failed.error_type) == (FrameOutcome.FAILED, "backend_error", "RuntimeError")
    assert "secret" not in repr(failed)
    assert backend.resets == 2 and tracker.counters["failure_resets"] == 1
    # The backend restarted at ID 1, but that ID was already a different person in this epoch.
    assert [p.track_id for p in before.persons] == [1, 2]
    assert [p.track_id for p in after.persons] == [4]
    assert after.persons[0].status is TrackStatus.TENTATIVE


def test_backend_labels_pass_through_and_a_failed_reset_blocks_tracking(stamper: FrameStamper, clock: FakeClock) -> None:
    backend = ScriptedBackend([TrackerError("inconsistent_output")])
    tracker = PersonTracker(backend)
    assert tracker.process(next_frame(stamper, clock), Image()).problem == "inconsistent_output"

    backend.fail_reset = True
    blocked = tracker.process(next_frame(stamper, clock), Image())
    assert (blocked.problem, blocked.error_type) == ("reset_failed", "RuntimeError")
    assert backend.calls == 1  # no tracking on state that could not be reset
    backend.fail_reset = False
    assert tracker.process(next_frame(stamper, clock), Image()).outcome is FrameOutcome.PROCESSED


@pytest.mark.parametrize(
    "tracks",
    [
        [person(1, x1=float("nan"))],
        [person(1, conf=1.5)],
        [person(1, conf=float("inf"))],
        [person(1, x1=300, x2=200)],
        [person(1), person(1, x1=300, x2=400)],
        [RawTrack(-1, 1, 1, 50, 50, 0.9)],
        [RawTrack(True, 1, 1, 50, 50, 0.9)],  # type: ignore[arg-type]
        [RawTrack(2.0, 1, 1, 50, 50, 0.9)],  # type: ignore[arg-type]
    ],
)
def test_malformed_backend_output_fails_the_frame(stamper: FrameStamper, clock: FakeClock, tracks: list[RawTrack]) -> None:
    tracker = PersonTracker(ScriptedBackend([tracks]))
    result = tracker.process(next_frame(stamper, clock), Image())
    assert (result.outcome, result.problem, result.persons) == (FrameOutcome.FAILED, "invalid_output", ())


def test_an_image_that_is_not_the_frames_native_size_fails(stamper: FrameStamper, clock: FakeClock) -> None:
    backend = ScriptedBackend()
    result = PersonTracker(backend).process(next_frame(stamper, clock), Image(640, 640))
    assert result.problem == "image_size_mismatch" and backend.calls == 0


def test_at_most_max_tracks_are_kept_highest_confidence_first(stamper: FrameStamper, clock: FakeClock) -> None:
    crowd = [person(i, x1=i, x2=i + 20, conf=round(0.30 + i / 200, 3)) for i in range(MAX_TRACKS + 6)]
    tracker = PersonTracker(ScriptedBackend([crowd]))
    persons = tracker.process(next_frame(stamper, clock), Image()).persons
    assert len(persons) == MAX_TRACKS and tracker.counters["boxes_dropped_overflow"] == 6
    assert min(p.track_id for p in persons) == 6  # the six lowest-confidence boxes went


def test_tracker_output_drives_edge_core_and_a_failure_reads_unknown_not_empty(
    stamper: FrameStamper, clock: FakeClock
) -> None:
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
    core = EdgeCore(config, clock, None)
    backend = ScriptedBackend([[person(1)], [person(1)], RuntimeError("boom")])  # then nobody
    tracker = PersonTracker(backend)
    states = []
    for _ in range(20):
        frame = next_frame(stamper, clock)
        result = tracker.process(frame, Image())
        # D-1's loop: the detector is unavailable for a failed frame, available otherwise.
        core.set_detector(Capability.UNAVAILABLE if result.outcome is FrameOutcome.FAILED else Capability.AVAILABLE)
        states.append(core.on_frame(frame, result.persons, stamper.current_stream).state)
    # Tentative first (occupancy counts confirmed people), then confirmed; the failed frame is
    # UNKNOWN; the person stays current until one track_expiry (1 s) after the last detection.
    assert [s.occupancy for s in states[:4]] == [Occupancy.EMPTY, Occupancy.OCCUPIED, Occupancy.UNKNOWN, Occupancy.OCCUPIED]
    assert [p.status for p in states[1].people] == [TrackStatus.CONFIRMED]
    assert states[2].occupancy_reason == "person detector unavailable"
    assert states[-1].occupancy is Occupancy.EMPTY and states[-1].people == ()
