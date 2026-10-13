"""D97, the continuity rule (opt-in, ``--face-continuity`` validation runs): an identity is kept only while its track is
followed without a break, never on a track number alone, and for a bounded time.

The monitor's own rules first (gap, jump, overlap, and the two thresholds covering each other), then scenarios on the
real EdgeCore and PersonTracker with a fake clock at 10 frames per second, as test_identity_continuity does for the
plain rules (which these tests leave untouched): a person who looks away stays known while followed, beyond the plain
30 s and never beyond the bound; contradicting faces still end it at once; two people crossing lose their identities
and neither inherits the other's, even when the tracker swaps their numbers; a jump, an occlusion and a late face
result from before a break carry nothing over; a brief detection miss changes nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from sentinel.config import ConfigError, parse_config
from sentinel.contracts import NormalizedBox, PixelFormat, TrackStatus
from sentinel.identity.association import FaceObservation
from sentinel.identity.continuity import NOT_CARRIED, Cause, ContinuityMonitor, ContinuityPolicy, iou
from sentinel.identity.matching import EnrolledIdentity, Enrollment
from sentinel.identity.state import BEFORE_BREAK, Basis, IdentityState
from sentinel.live_state import Capability
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.contracts import TrackObservation
from sentinel.runtime import EdgeCore, break_record
from sentinel.tracking.tracker import PersonTracker, RawTrack

CONFIG = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
POLICY = CONFIG.identity.continuity_policy()
ENROLLED = "idn-00000000000a"
ENROLLMENT = Enrollment(model_revision="synthetic", dimension=4,
                        identities=(EnrolledIdentity(identity_id=ENROLLED, prototypes=((1.0, 0.0, 0.0, 0.0),)),))
MATCH, NOBODY = (0.99, 0.1, 0.0, 0.0), (0.0, 0.1, 0.99, 0.0)
SEAT = (100.0, 100.0, 200.0, 400.0)  # a seated person's box on the 640x480 frame
HEAD = NormalizedBox(x1=0.19, y1=0.22, x2=0.28, y2=0.35)  # inside that box's head region


def box(x1: float, y1: float, x2: float, y2: float) -> NormalizedBox:
    return NormalizedBox(x1=x1 / 640, y1=y1 / 480, x2=x2 / 640, y2=y2 / 480)


# ---------------------------------------------------------------- the monitor


class Frames:
    """Frames and detections for the monitor alone."""

    def __init__(self) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stamper.connect()
        self.monitor = ContinuityMonitor(POLICY)

    def step(self, seconds: float, *people: tuple[int, tuple[float, ...]]) -> list:
        self.clock.advance(seconds)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        persons = [TrackObservation.detected(frame, track_id=track, box=box(*coords), confidence=0.9,
                                             status=TrackStatus.CONFIRMED) for track, coords in people]
        return [(b.key.track_id, b.cause) for b in self.monitor.observe(frame, persons)]


def test_steady_detections_never_break_and_a_new_track_starts_followed() -> None:
    frames = Frames()
    assert frames.step(0.1, (1, SEAT)) == []
    start = frames.monitor.followed_since(next(iter(frames.monitor._tracks)))
    for _ in range(50):  # five seconds, moving a little each frame, as a seated person does
        assert frames.step(0.1, (1, (101.0, 100.0, 201.0, 401.0))) == []
    assert frames.monitor.followed_since(next(iter(frames.monitor._tracks))) == start


def test_a_gap_breaks_only_after_a_missed_frame_and_beyond_the_bound() -> None:
    frames = Frames()
    frames.step(0.1, (1, SEAT))
    assert frames.step(2.0, (1, SEAT)) == []  # no processed frame in between (a stalled loop): no evidence lapsed
    frames.step(0.1)  # a processed frame without the track
    assert frames.step(0.3, (1, SEAT)) == []  # back after 0.4 s: within the 0.5 s bound
    frames.step(0.1)
    frames.step(0.3)
    assert frames.step(0.2, (1, SEAT)) == [(1, Cause.GAP)]  # 0.6 s since its last detection, with frames missed


def test_a_jump_breaks_and_ordinary_movement_does_not() -> None:
    frames = Frames()
    frames.step(0.1, (1, SEAT))
    assert frames.step(0.1, (1, (130.0, 100.0, 230.0, 400.0))) == []  # 30 px across: overlap 0.54
    assert frames.step(0.1, (1, (330.0, 100.0, 430.0, 400.0))) == [(1, Cause.JUMP)]  # nowhere near its last box


def test_overlapping_detections_break_both_and_the_thresholds_cover_any_swap() -> None:
    frames = Frames()
    frames.step(0.1, (1, SEAT), (2, (400.0, 100.0, 500.0, 400.0)))
    # track 2 jumps onto a box overlapping track 1 (intersection over union 0.33): a jump for 2, an overlap for 1
    assert frames.step(0.1, (1, SEAT), (2, (150.0, 100.0, 250.0, 400.0))) == [(2, Cause.JUMP), (1, Cause.OVERLAP)]
    assert frames.step(0.1, (1, SEAT), (2, (150.0, 100.0, 250.0, 400.0))) == [(1, Cause.OVERLAP), (2, Cause.OVERLAP)]
    # whatever box a track is swapped onto, one of the two rules fires: the overlap rule when that box overlapped it
    # by at least overlap_iou, the jump rule otherwise (overlap_iou < min_step_iou)
    for offset in range(0, 101, 5):
        other = (100.0 + offset, 100.0, 200.0 + offset, 400.0)
        overlapping = iou(box(*SEAT), box(*other)) >= POLICY.overlap_iou
        jumping = iou(box(*SEAT), box(*other)) < POLICY.min_step_iou
        assert overlapping or jumping, offset


def test_the_policy_refuses_thresholds_that_leave_a_swap_uncovered() -> None:
    with pytest.raises(ValueError, match="cover|overlap_iou"):
        ContinuityPolicy(overlap_iou=0.5, min_step_iou=0.4)
    with pytest.raises(ValueError):
        ContinuityPolicy(retention_ns=0)
    with pytest.raises(ConfigError, match="cover each other"):
        parse_config({"config_version": 1, "camera": {"id": "cam-1"},
                      "identity": {"continuity_overlap_iou": 0.5, "continuity_min_step_iou": 0.4}})
    assert POLICY.numbers() == {"max_gap_s": 0.5, "overlap_iou": 0.3, "min_step_iou": 0.4, "retention_s": 120.0}


# ---------------------------------------------------------------- scenarios on the real core


class Backend:
    def __init__(self) -> None:
        self.next: list[tuple[int, tuple[float, ...]]] = []

    def track(self, image: Any) -> list[RawTrack]:
        return [RawTrack(track, *coords, 0.9) for track, coords in self.next]

    def reset(self) -> None:
        pass


class Room:
    def __init__(self, continuity: ContinuityPolicy | None = POLICY) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stream = self.stamper.connect()
        self.core = EdgeCore(CONFIG, self.clock, None, face_recognition=Capability.AVAILABLE, enrollment=ENROLLMENT,
                             continuity=continuity)
        self.backend = Backend()
        self.tracker = PersonTracker(self.backend)
        self.transitions: list[dict] = []
        self.breaks: list[dict] = []
        self.state = None
        self.held: list = []  # (frame, persons) kept for a face result applied later

    def frames(self, seconds: float, *people: tuple[int, tuple[float, ...]], face=None, faces_at=None) -> None:
        """``seconds`` at 10 frames per second with these tracks; ``face`` (an embedding, or () for a hidden face) is
        analysed on each whole second's frame and applied at once, as the 1 Hz face worker would."""
        for n in range(round(seconds * 10)):
            self.clock.advance(0.1)
            self.backend.next = list(people)
            frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
            persons = self.tracker.process(frame, object()).persons
            self._keep(self.core.on_frame(frame, persons, self.stream))
            if face is not None and n % 10 == 9 and persons:
                self.apply(frame, persons, face, faces_at)

    def hold(self, *people: tuple[int, tuple[float, ...]]) -> None:
        """One frame whose face result is applied later (results arrive about a second late)."""
        self.clock.advance(0.1)
        self.backend.next = list(people)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        persons = self.tracker.process(frame, object()).persons
        self._keep(self.core.on_frame(frame, persons, self.stream))
        self.held.append((frame, persons))

    def apply(self, frame, persons, face, head: NormalizedBox | None = None):
        faces = () if face == () else (FaceObservation(frame=frame.key, box=head or HEAD, quality=0.95,
                                                       embedding=face),)
        output = self.core.on_face_outcome(frame, persons, faces, self.stream)
        self._keep(output)
        return output

    def person(self, track: int):
        return next((p for p in self.state.people if p.track_id == track), None)

    def _keep(self, output) -> None:
        self.state = output.state
        self.transitions += [t.record() for t in output.identity_transitions]
        self.breaks += [break_record(b) for b in output.identity_breaks]


def known(room: Room, track: int = 1, where: tuple[float, ...] = SEAT) -> Room:
    room.frames(2.0, (track, where), face=MATCH)  # two matches, one a second
    person = room.person(track)
    assert (person.identity, person.identity_id, person.identity_basis) == (IdentityState.KNOWN, ENROLLED, Basis.FRESH)
    return room


def test_a_person_followed_without_a_break_stays_known_retained_beyond_the_plain_expiry_and_never_beyond_the_bound():
    room = known(Room())
    room.frames(45.0, (1, SEAT), face=())  # the face turned away for 45 s, the person followed throughout
    turned = room.person(1)
    assert (turned.identity, turned.identity_basis) == (IdentityState.KNOWN, Basis.RETAINED)  # never "fresh"
    assert turned.identity_followed_ms >= 46_000  # followed since its first detection, without a break
    room.frames(75.0, (1, SEAT), face=())  # the older confirming vote is now over 120 s old: the bound
    assert room.person(1).identity is IdentityState.UNRESOLVED
    assert [t["to"] for t in room.transitions] == ["known", "known", "unresolved"]  # fresh, retained, at the bound
    assert room.breaks == []
    plain = known(Room(continuity=None))  # the same 45 s without the rule: the plain 30 s expiry, as before
    plain.frames(45.0, (1, SEAT), face=())
    assert plain.person(1).identity is IdentityState.UNRESOLVED and plain.person(1).identity_followed_ms is None


def test_a_contradicting_face_still_ends_a_retained_identity_at_once() -> None:
    room = known(Room())
    room.frames(40.0, (1, SEAT), face=())
    room.frames(1.0, (1, SEAT), face=NOBODY)
    assert room.person(1).identity is IdentityState.UNCERTAIN and room.person(1).identity_id is None


def test_two_people_crossing_lose_their_identities_and_neither_inherits_the_others() -> None:
    room = known(Room())
    room.frames(1.0, (1, SEAT), (2, (400.0, 100.0, 500.0, 400.0)))  # someone else, unrecognized, on track 2
    for step in range(1, 11):  # they walk towards each other and cross
        room.frames(0.1, (1, (100.0 + 15 * step, 100.0, 200.0 + 15 * step, 400.0)),
                    (2, (400.0 - 15 * step, 100.0, 500.0 - 15 * step, 400.0)))
    # the tracker swaps their numbers at the crossing: track 1 now follows the other person away to the right
    for step in range(1, 11):
        room.frames(0.1, (1, (250.0 + 15 * step, 100.0, 350.0 + 15 * step, 400.0)),
                    (2, (250.0 - 15 * step, 100.0, 350.0 - 15 * step, 400.0)))
    for track in (1, 2):
        person = room.person(track)
        assert person.identity is IdentityState.UNRESOLVED and person.identity_id is None
        assert person.identity_reason == NOT_CARRIED[Cause.OVERLAP]
    assert {b["track_id"] for b in room.breaks if b["cause"] == "overlap"} == {1, 2}
    assert room.transitions[0]["to"] == "known" and not any(t["to"] == "known" for t in room.transitions[1:])
    right = NormalizedBox(x1=0.66, y1=0.22, x2=0.75, y2=0.35)  # the head region of the box at 400-500 px
    room.frames(2.0, (1, (400.0, 100.0, 500.0, 400.0)), face=NOBODY, faces_at=right)  # only new faces decide again
    assert room.person(1).identity is IdentityState.UNKNOWN and room.person(1).identity_id is None


def test_a_jump_drops_the_identity_even_when_the_track_number_stays() -> None:
    room = known(Room())
    room.frames(0.1, (1, (450.0, 100.0, 550.0, 400.0)))  # the same number, a box nowhere near the last one
    person = room.person(1)
    assert person.identity is IdentityState.UNRESOLVED and person.identity_reason == NOT_CARRIED[Cause.JUMP]
    assert room.transitions[-1]["to"] == "unresolved" and room.transitions[-1]["reason"] == NOT_CARRIED[Cause.JUMP]
    assert room.breaks[-1] == {"identity": "break", "schema": 1, "stream_epoch": 1, "track_id": 1, "cause": "jump",
                               "mono_ns": room.breaks[-1]["mono_ns"]}


def test_an_occlusion_beyond_the_gap_bound_drops_the_identity_even_within_the_track_expiry() -> None:
    room = known(Room())
    room.frames(0.8)  # the chair hides the person for 0.8 s: the track stays current (expiry 1.0 s) but is not seen
    room.frames(1.0, (1, SEAT))
    assert room.person(1).identity is IdentityState.UNRESOLVED
    assert room.person(1).identity_reason == NOT_CARRIED[Cause.GAP]
    plain = known(Room(continuity=None))  # the plain rule keeps it, as test_identity_continuity pins
    plain.frames(0.8)
    plain.frames(1.0, (1, SEAT))
    assert plain.person(1).identity is IdentityState.KNOWN


def test_a_brief_detection_miss_changes_nothing() -> None:
    room = known(Room())
    room.frames(0.3)  # three frames without a detection
    room.frames(5.0, (1, SEAT))
    assert (room.person(1).identity, room.person(1).identity_basis) == (IdentityState.KNOWN, Basis.RETAINED)
    assert room.breaks == []


def test_a_face_result_from_before_a_break_is_not_counted() -> None:
    room = known(Room())
    room.hold((1, SEAT))  # a frame the face worker takes; its result arrives after the break below
    room.frames(0.1, (1, (450.0, 100.0, 550.0, 400.0)))  # the break: a jump
    frame, persons = room.held.pop()
    output = room.apply(frame, persons, MATCH)
    assert output.face_result.persons[0][4] == BEFORE_BREAK and output.face_result.persons[0][2] == "none"
    room.frames(0.9, (1, (450.0, 100.0, 550.0, 400.0)))
    assert room.person(1).identity is IdentityState.UNRESOLVED  # one late match from before the break: not counted
    assert room.core.identity_counters["persons_before_break"] == 1


def test_a_track_lost_beyond_its_expiry_still_loses_everything() -> None:
    room = known(Room())
    room.frames(1.5)
    assert room.person(1) is None and room.transitions[-1]["to"] == "cleared"
    room.frames(1.0, (1, SEAT), face=MATCH)  # the same number again: nothing carried, one match is not enough
    assert room.person(1).identity is IdentityState.UNRESOLVED


def test_without_the_rule_the_core_reports_no_breaks_and_no_followed_time() -> None:
    room = known(Room(continuity=None))
    room.frames(0.1, (1, (450.0, 100.0, 550.0, 400.0)))  # a jump the plain rule ignores, as before
    assert room.breaks == [] and room.person(1).identity is IdentityState.KNOWN
    assert room.core.continuity is None and Room().core.continuity == POLICY
