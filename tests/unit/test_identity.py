from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from sentinel.contracts import FrameRef, NormalizedBox, TrackObservation, TrackStatus
from sentinel.identity.association import FaceObservation, Ownership, associate
from sentinel.identity.matching import EnrolledIdentity, Enrollment, best_match

NextFrame = Callable[..., FrameRef]


def person(frame: FrameRef, track: int, box: tuple[float, float, float, float]) -> TrackObservation:
    return TrackObservation.detected(
        frame, track_id=track, box=NormalizedBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
        confidence=0.9, status=TrackStatus.CONFIRMED,
    )


def face(frame: FrameRef, box: tuple[float, float, float, float]) -> FaceObservation:
    return FaceObservation(
        frame=frame.key, box=NormalizedBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]), quality=0.9
    )


def test_each_face_and_person_take_part_in_at_most_one_pair(next_frame: NextFrame) -> None:
    frame = next_frame()
    a = person(frame, 1, (0.0, 0.1, 0.3, 0.9))
    b = person(frame, 2, (0.5, 0.1, 0.8, 0.9))
    faces = [
        face(frame, (0.10, 0.12, 0.16, 0.20)),  # a's head
        face(frame, (0.60, 0.12, 0.66, 0.20)),  # b's head ...
        face(frame, (0.70, 0.12, 0.76, 0.20)),  # ... and a second face there
        face(frame, (0.40, 0.12, 0.46, 0.20)),  # between them: nobody's head
        face(frame, (0.10, 0.70, 0.16, 0.78)),  # inside a's box, but at knee height
    ]
    result = associate([a, b], faces)
    assert result.face_of == {a.key: 0}
    assert result.person_ownership == {a.key: Ownership.ASSIGNED, b.key: Ownership.AMBIGUOUS}
    assert result.face_ownership == (
        Ownership.ASSIGNED,
        Ownership.AMBIGUOUS,
        Ownership.AMBIGUOUS,
        Ownership.NO_PERSON,
        Ownership.NO_PERSON,
    )


def test_a_face_mostly_outside_the_person_box_is_not_a_candidate(next_frame: NextFrame) -> None:
    frame = next_frame()
    a = person(frame, 1, (0.2, 0.2, 0.4, 0.9))
    edge = face(frame, (0.19, 0.22, 0.29, 0.30))  # centre inside, 90 % inside the box
    outside = face(frame, (0.15, 0.22, 0.27, 0.30))  # centre inside, only 58 % inside
    assert associate([a], [edge]).face_of == {a.key: 0}
    assert associate([a], [outside]).person_ownership[a.key] is Ownership.NO_FACE


def test_faces_and_people_must_share_a_frame(next_frame: NextFrame) -> None:
    first, second = next_frame(), next_frame()
    with pytest.raises(ValueError, match="same frame"):
        associate([person(first, 1, (0, 0, 0.5, 1))], [face(second, (0.1, 0.1, 0.2, 0.2))])


def test_enrollment_is_validated_numbers() -> None:
    ok = EnrolledIdentity(identity_id="a", prototypes=((1.0, 0.0),))
    Enrollment(model_revision="m", dimension=2, identities=(ok,))
    for bad, message in [
        ((ok, ok), "unique"),
        ((EnrolledIdentity(identity_id="b", prototypes=((1.0, 0.0, 0.0),)),), "expected 2"),
        ((EnrolledIdentity(identity_id="b", prototypes=((0.0, 0.0),)),), "zero prototype"),
    ]:
        with pytest.raises(ValidationError, match=message):
            Enrollment(model_revision="m", dimension=2, identities=bad)
    with pytest.raises(ValidationError):
        EnrolledIdentity(identity_id="c", prototypes=((float("nan"), 1.0),))


def test_best_match_reports_similarity_and_runner_up() -> None:
    gallery = Enrollment(
        model_revision="m",
        dimension=2,
        identities=(
            EnrolledIdentity(identity_id="a", prototypes=((1.0, 0.0), (0.8, 0.6))),
            EnrolledIdentity(identity_id="b", prototypes=((0.0, 1.0),)),
        ),
    )
    match = best_match((0.8, 0.6), gallery)
    assert match.identity_id == "a" and match.similarity == pytest.approx(1.0)
    assert match.runner_up == pytest.approx(0.6)
    empty = best_match((1.0, 0.0), Enrollment(model_revision="m", dimension=2))
    assert empty.identity_id is None and empty.runner_up is None
    with pytest.raises(ValueError, match="expected 2"):
        best_match((1.0, 0.0, 0.0), gallery)
    with pytest.raises(ValueError, match="non-zero"):
        best_match((0.0, 0.0), gallery)
