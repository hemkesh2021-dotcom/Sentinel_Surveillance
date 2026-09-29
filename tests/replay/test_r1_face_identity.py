"""R1: face ownership and identity states.

Audit findings 1 and 2; guide ch. 2 row 2, ch. 11 and ch. 21 ("One known and one
unknown person overlap; only one face visible; no identity transfer"). Each test
replays synthetic people, faces and 4-D embeddings through v2 and the B0 v1 face
worker on the same frames.
"""

from __future__ import annotations

from identity_harness import ALICE_FACE, BOB_FACE, NOBODY_FACE, run_identity
from timeline_builder import TimelineBuilder

from sentinel.identity.matching import EnrolledIdentity, Enrollment
from sentinel.identity.state import IdentityState

UNRESOLVED, UNKNOWN, KNOWN, UNCERTAIN = (
    IdentityState.UNRESOLVED,
    IdentityState.UNKNOWN,
    IdentityState.KNOWN,
    IdentityState.UNCERTAIN,
)

# Two people whose boxes overlap so much that one face lies in both head regions.
OVERLAPPING = [
    {"track": 1, "box": [0.30, 0.20, 0.55, 0.95]},
    {"track": 2, "box": [0.40, 0.15, 0.65, 0.95]},
]
SHARED_FACE = {"box": [0.44, 0.23, 0.50, 0.31], "embedding": list(ALICE_FACE)}
# The same two people apart: track 1 on the left, track 2 on the right.
APART = [
    {"track": 1, "box": [0.05, 0.20, 0.30, 0.95]},
    {"track": 2, "box": [0.60, 0.20, 0.85, 0.95]},
]


def face_at_left(embedding: tuple[float, ...], quality: float = 0.9) -> dict:
    return {"box": [0.14, 0.24, 0.21, 0.33], "embedding": list(embedding), "quality": quality}


def face_at_right(embedding: tuple[float, ...]) -> dict:
    return {"box": [0.69, 0.24, 0.76, 0.33], "embedding": list(embedding)}


def test_one_face_inside_two_overlapping_people_belongs_to_neither() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(2000, persons=OVERLAPPING, faces=[SHARED_FACE])
        .build()
    )
    run = run_identity(timeline)
    for track in (1, 2):
        assert set(run.history(track)) == {UNRESOLVED}
    people = run.people_at(2000)
    assert {p.identity_reason for p in people.values()} == {"face ownership ambiguous"}
    assert all(p.identity_id is None for p in people.values())
    # B0: both crops contain the face, so v1 names both people Alice.
    assert run.v1.verified == {1: "alice", 2: "alice"}


def test_known_and_unknown_person_with_one_visible_face_do_not_share_identity() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(2000, persons=APART, faces=[face_at_left(ALICE_FACE)])
        .build()
    )
    run = run_identity(timeline)
    assert run.history(1)[:2] == [UNRESOLVED, KNOWN]  # after two consistent matches
    assert set(run.history(1)[1:]) == {KNOWN}
    assert set(run.history(2)) == {UNRESOLVED}
    people = run.people_at(2000)
    assert (people[1].identity_id, people[2].identity_id) == ("alice", None)
    assert people[2].identity_reason == "no face visible"
    # B0: track 2's crop has no face, so it takes the full-frame result: Alice.
    assert run.v1.verified == {1: "alice", 2: "alice"}


def test_after_separating_the_face_attaches_to_exactly_one_person() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(1000, persons=OVERLAPPING, faces=[SHARED_FACE])
        .frames_until(2000, persons=APART, faces=[face_at_left(ALICE_FACE)])
        .build()
    )
    run = run_identity(timeline)
    assert run.people_at(990)[1].identity is UNRESOLVED
    people = run.people_at(2000)
    assert (people[1].identity, people[2].identity) == (KNOWN, UNRESOLVED)


def test_a_reconnect_starts_every_track_unresolved() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(1000, persons=APART, faces=[face_at_left(ALICE_FACE)])
        .disconnect(1000)
        .connect(1500)
        .frames_until(3000, persons=APART, faces=[])  # same tracker IDs, faces turned away
        .build()
    )
    run = run_identity(timeline)
    assert run.people_at(990)[1].identity is KNOWN
    after = run.people_at(3000)
    assert {p.stream_epoch for p in after.values()} == {2}
    assert {p.identity for p in after.values()} == {UNRESOLVED}
    # Bookkeeping for the old epoch's tracks is gone, not just unused.
    assert run.core.diagnostics()["identity_tracks"] == 2


def test_empty_enrollment_leaves_people_unresolved_never_strangers() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(2000, persons=APART, faces=[face_at_left(ALICE_FACE), face_at_right(NOBODY_FACE)])
        .frames_until(2500, persons=APART, faces=[face_at_left(ALICE_FACE)])
        .build()
    )
    for enrollment in (Enrollment(model_revision="stub-embedder-1", dimension=4), None):
        run = run_identity(timeline, enrollment=enrollment, v1_db={})
        people = run.people_at(2000)
        assert {p.identity for p in people.values()} == {UNRESOLVED}
        assert {p.identity_reason for p in people.values()} == {"no identities enrolled"}
        # The more useful reason wins over "no face visible" for the faceless person.
        assert run.people_at(2500)[2].identity_reason == "no identities enrolled"
        # B0: with an empty database every visible face is a face-confirmed "Stranger".
        assert run.v1.verified == {1: "Stranger", 2: "Stranger"}
        assert run.v1.is_stranger(1) and run.v1.is_stranger(2)


def test_a_person_facing_away_stays_unresolved_instead_of_becoming_a_stranger() -> None:
    timeline = TimelineBuilder().connect(0).frames_until(3000, persons=1, faces=[]).build()
    run = run_identity(timeline)
    assert set(run.history(0)) == {UNRESOLVED}
    assert run.people_at(3000)[0].identity_reason == "no face visible"
    # B0: a stranger from the first frame, and labelled "Stranger" after 15 misses.
    assert run.v1.verified == {0: "Stranger"}


def test_unknown_needs_consistent_good_faces_and_contradiction_is_uncertain() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(700, persons=APART[:1], faces=[face_at_left(NOBODY_FACE, quality=0.3)])
        .frames_until(1400, persons=APART[:1], faces=[face_at_left(NOBODY_FACE)])
        .frames_until(2100, persons=APART[:1], faces=[face_at_left(ALICE_FACE)])
        .frames_until(2200, persons=APART[:1], faces=[face_at_left(BOB_FACE)])
        .build()
    )
    run = run_identity(timeline)
    assert run.people_at(660)[1].identity is UNRESOLVED
    assert run.people_at(660)[1].identity_reason == "face quality too low"
    assert run.people_at(1386)[1].identity is UNKNOWN  # good faces, matched nobody
    assert run.people_at(2046)[1].identity is KNOWN
    contradicted = run.people_at(2112)[1]  # one Bob vote after two Alice votes
    assert (contradicted.identity, contradicted.identity_id) == (UNCERTAIN, None)
    # New consistent evidence re-derives the identity (e.g. after a tracker ID swap).
    assert (run.people_at(2178)[1].identity, run.people_at(2178)[1].identity_id) == (KNOWN, "bob")


def test_identity_votes_expire_without_fresh_faces() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(500, persons=APART[:1], faces=[face_at_left(ALICE_FACE)])
        .frames_until(29_000, persons=APART[:1], faces=[])
        .frames_until(31_000, persons=APART[:1], faces=[])
        .build()
    )
    run = run_identity(timeline)
    assert run.people_at(29_000)[1].identity is KNOWN  # votes from 0-462 ms still count
    assert run.people_at(30_990)[1].identity is UNRESOLVED


def test_a_face_close_to_two_enrolled_identities_is_not_a_match() -> None:
    twins = Enrollment(
        model_revision="stub-embedder-1",
        dimension=4,
        identities=(
            EnrolledIdentity(identity_id="alice", prototypes=((1.0, 0.0, 0.0, 0.0),)),
            EnrolledIdentity(identity_id="alice-twin", prototypes=((0.98, 0.2, 0.0, 0.0),)),
        ),
    )
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(1000, persons=APART[:1], faces=[face_at_left((0.99, 0.1, 0.0, 0.0))])
        .build()
    )
    run = run_identity(timeline, enrollment=twins, v1_db={})
    person = run.people_at(1000)[1]
    assert (person.identity, person.identity_reason) == (UNRESOLVED, "face matches more than one identity")
