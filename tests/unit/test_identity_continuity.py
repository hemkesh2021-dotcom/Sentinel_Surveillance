"""Identity continuity through a seated turn, occlusion, track loss, departure, replacement and face reacquisition
(D86: characterization of the current rules, unchanged; guide ch. 9 and 11, audit findings 1 and 2).

The real EdgeCore and PersonTracker run on a fake clock at 10 frames per second; a scripted backend plays the
person detector and tracker (it decides which track IDs a frame has, as ByteTrack would, including an ID it keeps
across a gap). Faces are synthetic 4-D embeddings. What these pin down:
- an identity is kept only on its own track: while the face is hidden it is KNOWN with basis RETAINED until its older
  confirming vote is vote_ttl old, never longer, and a contradicting face ends it at once;
- a track that stops being current (track_expiry after its last detection) loses its identity, even if the tracker
  later publishes the same ID again: nothing carries over a detection gap, so nothing is inferred from a place, a chair
  or a back of a head;
- a replacement person on a new track starts unresolved and never inherits; reacquisition needs fresh matches.
"""

from __future__ import annotations

from typing import Any

from sentinel.config import parse_config
from sentinel.contracts import NormalizedBox, PixelFormat
from sentinel.identity.association import FaceObservation
from sentinel.identity.matching import EnrolledIdentity, Enrollment
from sentinel.identity.state import Basis, IdentityState
from sentinel.live_state import Capability, Occupancy
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.runtime import EdgeCore
from sentinel.tracking.tracker import PersonTracker, RawTrack

CONFIG = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
ENROLLED = "idn-00000000000a"
ENROLLMENT = Enrollment(model_revision="synthetic", dimension=4,
                        identities=(EnrolledIdentity(identity_id=ENROLLED, prototypes=((1.0, 0.0, 0.0, 0.0),)),))
MATCH, NOBODY = (0.99, 0.1, 0.0, 0.0), (0.0, 0.1, 0.99, 0.0)
SEAT = (100.0, 100.0, 200.0, 400.0)  # a seated person's box on the 640x480 frame
HEAD = NormalizedBox(x1=0.19, y1=0.22, x2=0.28, y2=0.35)  # inside that box's head region


class Backend:
    """The detector and tracker: each frame's (track ID, box) list, as the script sets it."""

    def __init__(self) -> None:
        self.next: list[tuple[int, tuple[float, ...]]] = []

    def track(self, image: Any) -> list[RawTrack]:
        return [RawTrack(track, *box, 0.9) for track, box in self.next]

    def reset(self) -> None:
        pass


class Room:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stream = self.stamper.connect()
        self.core = EdgeCore(CONFIG, self.clock, None, face_recognition=Capability.AVAILABLE, enrollment=ENROLLMENT)
        self.backend = Backend()
        self.tracker = PersonTracker(self.backend)
        self.transitions: list[dict] = []
        self.state = None

    def frames(self, seconds: float, *people: tuple[int, tuple[float, ...]], face=None) -> None:
        """``seconds`` of frames at 10 per second with these tracks; ``face`` (an embedding, or () for a hidden
        face) is analysed on each whole second's frame and applied at once, as the 1 Hz face worker would."""
        for n in range(round(seconds * 10)):
            self.clock.advance(0.1)
            self.backend.next = list(people)
            frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
            persons = self.tracker.process(frame, object()).persons
            self._keep(self.core.on_frame(frame, persons, self.stream))
            if face is not None and n % 10 == 9 and persons:
                faces = () if face == () else (FaceObservation(frame=frame.key, box=HEAD, quality=0.95,
                                                               embedding=face),)
                self._keep(self.core.on_face_outcome(frame, persons, faces, self.stream))

    def person(self, track: int):
        return next((p for p in self.state.people if p.track_id == track), None)

    def _keep(self, output) -> None:
        self.state = output.state
        self.transitions += [t.record() for t in output.identity_transitions]


def known(room: Room) -> Room:
    room.frames(2.0, (1, SEAT), face=MATCH)  # two matches, one a second
    person = room.person(1)
    assert (person.identity, person.identity_id, person.identity_basis) == (IdentityState.KNOWN, ENROLLED, Basis.FRESH)
    return room


def test_a_seated_turn_keeps_the_identity_on_its_track_retained_until_its_votes_expire() -> None:
    room = known(Room())
    room.frames(10.0, (1, SEAT), face=())  # the face hidden, the person still detected on the same track
    turned = room.person(1)
    assert (turned.identity, turned.identity_basis) == (IdentityState.KNOWN, Basis.RETAINED)  # never "fresh"
    room.frames(19.0, (1, SEAT), face=())  # the older confirming vote is now 30 s old
    assert room.person(1).identity is IdentityState.UNRESOLVED
    assert [t["to"] for t in room.transitions] == ["known", "known", "unresolved"]  # fresh, retained, expired


def test_a_face_seen_again_during_the_turn_refreshes_it_and_a_contradicting_one_ends_it() -> None:
    room = known(Room())
    room.frames(5.0, (1, SEAT), face=())
    room.frames(1.0, (1, SEAT), face=MATCH)  # reacquired on the same track: fresh again
    assert room.person(1).identity_basis is Basis.FRESH
    room.frames(1.0, (1, SEAT), face=NOBODY)  # a face that matches nobody: no longer known, at once
    assert room.person(1).identity is IdentityState.UNCERTAIN


def test_an_occlusion_shorter_than_the_track_expiry_keeps_the_identity() -> None:
    room = known(Room())
    room.frames(0.8, face=())  # the chair hides the person: no detection for 0.8 s (expiry 1.0 s)
    room.frames(1.0, (1, SEAT))
    assert room.person(1).identity is IdentityState.KNOWN


def test_a_track_lost_beyond_its_expiry_loses_the_identity_even_if_the_tracker_reuses_its_id() -> None:
    room = known(Room())
    room.frames(1.5, face=())  # no detection for 1.5 s: the track is no longer current
    assert room.person(1) is None and room.transitions[-1]["to"] == "cleared"
    room.frames(1.0, (1, SEAT))  # the tracker publishes the same ID again (as a lost-track buffer can)
    assert room.person(1).identity is IdentityState.UNRESOLVED  # nothing carried over the gap


def test_face_reacquisition_after_a_loss_needs_two_fresh_matches() -> None:
    room = known(Room())
    room.frames(1.5)
    room.frames(1.0, (1, SEAT), face=MATCH)
    assert room.person(1).identity is IdentityState.UNRESOLVED  # one match is not enough
    room.frames(1.0, (1, SEAT), face=MATCH)
    assert (room.person(1).identity, room.person(1).identity_basis) == (IdentityState.KNOWN, Basis.FRESH)


def test_departure_clears_the_identity_and_the_room_reads_empty() -> None:
    room = known(Room())
    room.frames(2.0)
    assert room.state.people == () and room.state.occupancy is Occupancy.EMPTY
    assert room.transitions[-1]["to"] == "cleared" and room.transitions[-1]["identity_id"] is None


def test_a_replacement_person_in_the_same_seat_never_inherits_the_identity() -> None:
    room = known(Room())
    room.frames(2.0)  # the known person leaves
    room.frames(1.0, (2, SEAT))  # someone else sits in the same seat: a new track
    assert room.person(2).identity is IdentityState.UNRESOLVED
    room.frames(2.0, (2, SEAT), face=NOBODY)
    replacement = room.person(2)
    assert replacement.identity is IdentityState.UNKNOWN and replacement.identity_id is None
    assert all(t["identity_id"] in (None, ENROLLED) for t in room.transitions)
    assert not any(t["track_id"] == 2 and t["identity_id"] == ENROLLED for t in room.transitions)


def test_a_face_inside_two_overlapping_people_votes_for_neither() -> None:
    room = Room()
    room.frames(3.0, (1, SEAT), (2, (110.0, 105.0, 210.0, 400.0)), face=MATCH)
    assert room.person(1).identity is IdentityState.UNRESOLVED and room.person(2).identity is IdentityState.UNRESOLVED
    assert room.transitions == []
