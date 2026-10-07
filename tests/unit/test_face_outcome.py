"""EdgeCore's 1 Hz face results (V2-25 demo form): applied to their own frame's people, rejected when stale or not
live, with identity retention, clearing and opaque transition records. Synthetic 4-D embeddings only."""

from __future__ import annotations

import pytest

from sentinel.config import parse_config
from sentinel.contracts import FrameRef, NormalizedBox, PixelFormat, TrackObservation, TrackStatus
from sentinel.identity.association import FaceObservation
from sentinel.identity.matching import EnrolledIdentity, Enrollment
from sentinel.identity.state import Basis, IdentityState
from sentinel.live_state import Capability
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.runtime import EdgeCore

CONFIG = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
A, B = (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)
A_FACE, B_FACE, NOBODY = (0.97, 0.1, 0.0, 0.05), (0.05, 0.97, 0.1, 0.0), (0.0, 0.1, 0.99, 0.0)
ENROLLMENT = Enrollment(model_revision="synthetic", dimension=4, identities=(
    EnrolledIdentity(identity_id="idn-00000000000a", prototypes=(A,)),
    EnrolledIdentity(identity_id="idn-00000000000b", prototypes=(B,)),
))
LEFT, RIGHT = (0.05, 0.1, 0.35, 0.9), (0.6, 0.1, 0.9, 0.9)


def head(box: tuple[float, ...]) -> tuple[float, float, float, float]:
    x1, y1, x2, _ = box
    return x1 + 0.08, y1 + 0.02, x2 - 0.08, y1 + 0.15


class World:
    def __init__(self, *, enrollment: Enrollment | None = ENROLLMENT,
                 face: Capability = Capability.AVAILABLE) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stream = self.stamper.connect()
        self.core = EdgeCore(CONFIG, self.clock, None, face_recognition=face, enrollment=enrollment)
        self.transitions: list = []

    def frame(self, *people: tuple[int, tuple[float, ...]], dt: float = 0.5) -> tuple[FrameRef, tuple]:
        self.clock.advance(dt)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        persons = tuple(TrackObservation.detected(frame, track_id=track, box=NormalizedBox(
            x1=box[0], y1=box[1], x2=box[2], y2=box[3]), confidence=0.9, status=TrackStatus.CONFIRMED)
            for track, box in people)
        self._keep(self.core.on_frame(frame, persons, self.stream))
        return frame, persons

    def faces(self, frame: FrameRef, persons, *faces: tuple[tuple[float, ...], tuple[float, ...]] | None,
              connected="live", quality: float = 0.9):
        observations = None if faces == (None,) else tuple(
            FaceObservation(frame=frame.key, box=NormalizedBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]),
                            quality=quality, embedding=e) for b, e in faces)
        output = self.core.on_face_outcome(frame, persons, observations,
                                           self.stream if connected == "live" else connected)
        self._keep(output)
        return output

    def person(self, track: int):
        people = {p.track_id: p for p in self.state.people}
        return people.get(track)

    def _keep(self, output) -> None:
        self.state = output.state
        self.transitions += [t.record() for t in output.identity_transitions]


def test_two_fresh_matches_make_a_known_identity_with_a_fresh_basis() -> None:
    world = World()
    for second in range(2):
        frame, persons = world.frame((1, LEFT), dt=1.0)
        world.faces(frame, persons, (head(LEFT), A_FACE))
        if second == 0:
            assert world.person(1).identity is IdentityState.UNRESOLVED
    known = world.person(1)
    assert (known.identity, known.identity_id, known.identity_basis) == (
        IdentityState.KNOWN, "idn-00000000000a", Basis.FRESH)
    assert known.identity_vote_age_ms == 0
    assert [(t["from"], t["to"], t["identity_id"], t["basis"]) for t in world.transitions] == [
        ("unresolved", "known", "idn-00000000000a", "fresh")]
    counters = world.core.identity_counters
    assert counters["results_applied"] == 2 and counters["votes_match"] == 2 and counters["faces_assigned"] == 2


def test_turning_away_retains_the_identity_on_the_same_track_until_the_older_vote_expires() -> None:
    world = World()
    for _ in range(2):  # matches at 1 s and 2 s of frame time (votes at the frames' own times)
        frame, persons = world.frame((1, LEFT), dt=1.0)
        world.faces(frame, persons, (head(LEFT), A_FACE))
    first_vote_ns = world.clock.monotonic_ns() - 1_000_000_000
    states = []
    while world.clock.monotonic_ns() < first_vote_ns + 31_000_000_000:  # the person stays, facing away
        frame, persons = world.frame((1, LEFT), dt=0.5)
        world.faces(frame, persons)  # a face result with no face on this person
        person = world.person(1)
        states.append((world.clock.monotonic_ns() - first_vote_ns, person.identity, person.identity_basis))
    retained = [ns for ns, state, basis in states if state is IdentityState.KNOWN and basis is Basis.RETAINED]
    cleared = [ns for ns, state, _ in states if state is IdentityState.UNRESOLVED]
    assert retained and min(retained) > 1_000_000_000 + 3_000_000_000  # fresh for 3 s after the newest vote
    assert max(retained) < 30_000_000_000 and min(cleared) >= 30_000_000_000  # until the older vote is 30 s old
    assert all(state is IdentityState.UNRESOLVED for ns, state, _ in states if ns >= 30_000_000_000)
    assert world.person(1).identity_reason == "no face visible"
    assert [(t["from"], t["to"], t["basis"]) for t in world.transitions] == [
        ("unresolved", "known", "fresh"), ("known", "known", "retained"), ("known", "unresolved", None)]
    assert world.core.identity_counters["persons_no_face"] > 0


def test_a_track_that_leaves_loses_its_identity_even_if_its_id_returns() -> None:
    world = World()
    for _ in range(2):
        frame, persons = world.frame((1, LEFT), dt=1.0)
        world.faces(frame, persons, (head(LEFT), A_FACE))
    for _ in range(4):  # 2 s without the person: past the 1 s track expiry
        world.frame(dt=0.5)
    assert world.person(1) is None
    assert world.transitions[-1]["to"] == "cleared" and world.transitions[-1]["reason"] == "track no longer current"
    frame, persons = world.frame((1, LEFT), dt=0.5)  # the tracker reuses ID 1
    world.faces(frame, persons, (head(LEFT), A_FACE))
    assert world.person(1).identity is IdentityState.UNRESOLVED  # one fresh vote is not enough: nothing carried over
    frame, persons = world.frame((1, LEFT), dt=1.0)
    world.faces(frame, persons, (head(LEFT), A_FACE))
    assert world.person(1).identity is IdentityState.KNOWN


def test_stale_failed_and_not_live_results_are_rejected_and_counted() -> None:
    world = World()
    frame, persons = world.frame((1, LEFT), dt=1.0)
    world.frame((1, LEFT), dt=3.5)  # the result arrives 3.5 s after its frame: beyond 3.0 s
    world.faces(frame, persons, (head(LEFT), A_FACE))
    frame, persons = world.frame((1, LEFT), dt=1.0)
    world.faces(frame, persons, None)  # the analysis failed
    world.faces(frame, persons, (head(LEFT), A_FACE), connected=None)  # the stream dropped meanwhile
    world.stamper.disconnect()
    world.stream = world.stamper.connect()  # a reconnect: a new epoch
    world.faces(frame, persons, (head(LEFT), A_FACE))
    counters = world.core.identity_counters
    assert (counters["results_rejected_age"], counters["results_failed"], counters["results_rejected_not_live"],
            counters.get("results_applied", 0), counters.get("votes_match", 0)) == (1, 1, 2, 0, 0)


def test_a_late_result_within_its_bound_counts_from_its_own_frame_time() -> None:
    world = World()
    frame, persons = world.frame((1, LEFT), dt=1.0)
    world.frame((1, LEFT), dt=2.5)
    world.faces(frame, persons, (head(LEFT), A_FACE))
    assert world.core.identity_counters["results_applied"] == 1
    assert world.person(1).identity_vote_age_ms == 2500


def test_results_are_ignored_while_face_recognition_is_unavailable() -> None:
    world = World(face=Capability.UNAVAILABLE)
    for _ in range(2):
        frame, persons = world.frame((1, LEFT), dt=1.0)
        world.faces(frame, persons, (head(LEFT), A_FACE))
    assert world.person(1).identity is IdentityState.UNRESOLVED and world.transitions == []
    assert world.core.identity_counters["results_ignored_face_unavailable"] == 2


def test_each_face_identifies_at_most_one_person_and_overlap_stays_unresolved() -> None:
    world = World()
    for _ in range(2):  # two people, each with their own face
        frame, persons = world.frame((1, LEFT), (2, RIGHT), dt=1.0)
        world.faces(frame, persons, (head(LEFT), A_FACE), (head(RIGHT), B_FACE))
    assert (world.person(1).identity_id, world.person(2).identity_id) == ("idn-00000000000a", "idn-00000000000b")
    other = World()
    overlap = (0.3, 0.1, 0.6, 0.9)
    for _ in range(3):  # one visible face inside two overlapping people
        frame, persons = other.frame((1, LEFT), (2, overlap), dt=1.0)
        other.faces(frame, persons, ((0.32, 0.12, 0.34, 0.22), A_FACE))
    assert {other.person(1).identity, other.person(2).identity} == {IdentityState.UNRESOLVED}
    assert other.core.identity_counters["faces_ambiguous"] == 3 and other.transitions == []


def test_unknown_is_only_a_consistent_non_match_and_an_empty_gallery_resolves_nothing() -> None:
    world = World()
    for _ in range(2):
        frame, persons = world.frame((1, LEFT), dt=1.0)
        world.faces(frame, persons, (head(LEFT), NOBODY))
    assert (world.person(1).identity, world.person(1).identity_basis) == (IdentityState.UNKNOWN, Basis.FRESH)
    frame, persons = world.frame((1, LEFT), dt=1.0)
    world.faces(frame, persons, (head(LEFT), A_FACE))
    assert world.person(1).identity is IdentityState.UNCERTAIN  # a contradicting vote: at once
    empty = World(enrollment=None)
    for _ in range(3):
        frame, persons = empty.frame((1, LEFT), dt=1.0)
        empty.faces(frame, persons, (head(LEFT), A_FACE))
    assert empty.person(1).identity is IdentityState.UNRESOLVED and empty.transitions == []


def test_transition_records_hold_opaque_ids_and_numbers_only() -> None:
    world = World()
    for _ in range(2):
        frame, persons = world.frame((1, LEFT), dt=1.0)
        world.faces(frame, persons, (head(LEFT), A_FACE))
    (record,) = world.transitions
    assert set(record) == {"identity", "schema", "stream_epoch", "track_id", "from", "to", "identity_id", "basis",
                           "last_vote_age_ms", "reason", "mono_ns"}
    assert record["identity"] == "transition" and record["identity_id"] == "idn-00000000000a" and record["schema"] == 1


def test_each_face_result_is_recorded_with_its_frame_time_and_each_persons_vote() -> None:
    world = World()
    frame, persons = world.frame((1, LEFT), (2, RIGHT), dt=1.0)
    world.clock.advance(0.4)
    output = world.faces(frame, persons, (head(LEFT), A_FACE))
    record = output.face_result.record()
    assert set(record) == {"identity", "schema", "stream_epoch", "frame_seq", "frame_mono_ns", "applied_mono_ns",
                           "outcome", "faces", "processing_ms", "error", "persons"}
    assert (record["identity"], record["schema"], record["outcome"], record["faces"]) == ("result", 1, "applied", 1)
    assert record["frame_mono_ns"] == frame.ingest_mono_ns
    assert record["applied_mono_ns"] - record["frame_mono_ns"] == 400_000_000
    assert record["persons"] == [
        {"track_id": 1, "ownership": "assigned", "vote": "match", "identity_id": "idn-00000000000a", "label": "match"},
        {"track_id": 2, "ownership": "no_face", "vote": "none", "identity_id": None, "label": "no_face"}]
    frame, persons = world.frame((1, LEFT), dt=1.0)
    nobody = world.faces(frame, persons, (head(LEFT), NOBODY)).face_result.record()
    assert nobody["persons"][0] | {} == {"track_id": 1, "ownership": "assigned", "vote": "unknown",
                                         "identity_id": None, "label": "unknown"}
    low = world.faces(frame, persons, (head(LEFT), A_FACE), quality=0.2).face_result.record()
    assert low["persons"][0]["vote"] == "none" and low["persons"][0]["label"] == "low_quality"


@pytest.mark.parametrize(("make", "outcome"), [
    (lambda w, f, p: w.faces(f, p, None), "failed"),
    (lambda w, f, p: w.faces(f, p, (head(LEFT), A_FACE), connected=None), "rejected_not_live"),
])
def test_a_rejected_or_failed_result_is_recorded_without_votes(make, outcome) -> None:
    world = World()
    frame, persons = world.frame((1, LEFT), dt=1.0)
    record = make(world, frame, persons).face_result.record()
    assert record["outcome"] == outcome and record["persons"] == []
    stale = World()
    frame, persons = stale.frame((1, LEFT), dt=1.0)
    stale.frame((1, LEFT), dt=3.5)
    assert stale.faces(frame, persons, (head(LEFT), A_FACE)).face_result.record()["outcome"] == "rejected_age"
