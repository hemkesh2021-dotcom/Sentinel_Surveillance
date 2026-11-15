"""One-to-one association of detected faces with tracked people (guide ch. 11; audit finding 1).

Association is decided over all faces and all people of one frame together.
A face is a candidate for a person when its centre lies in the head region of
the person's box and most of the face lies inside that box. A pair is accepted
only when each is the other's sole candidate, so each face goes to at most one
person and each person gets at most one face. Anything else stays unassigned
with a reason: a face inside two overlapping people is AMBIGUOUS for both, and
never goes to whichever came first.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import Enum

from pydantic import Field, FiniteFloat

from ..contracts import Contract, FrameKey, NormalizedBox, TrackKey, TrackObservation, UnitInterval

MAX_EMBEDDING_DIMENSION = 2048


class FaceObservation(Contract):
    """A detected face on one frame. ``embedding`` is absent when it was not computed."""

    frame: FrameKey
    box: NormalizedBox
    quality: UnitInterval
    embedding: tuple[FiniteFloat, ...] | None = Field(default=None, max_length=MAX_EMBEDDING_DIMENSION)


class Ownership(str, Enum):
    ASSIGNED = "assigned"
    NO_FACE = "no_face"  # person: no face candidate
    NO_PERSON = "no_person"  # face: not inside anyone's head region
    AMBIGUOUS = "ambiguous"  # candidates conflict; nobody gets this face
    # Route B (session 116, part 15): a person the face stage did not analyse this tick (identity.window_stage), and a
    # face that was theirs; never produced by associate(), only by withhold()
    NOT_ANALYSED = "not_analysed"


@dataclass(frozen=True)
class AssociationGeometry:
    head_fraction: float = 0.4  # top part of the person box where a face centre may lie
    min_inside: float = 0.6  # fraction of the face box that must lie inside the person box


@dataclass(frozen=True)
class Association:
    face_of: dict[TrackKey, int]  # accepted pairs: person -> index into the faces given
    person_ownership: dict[TrackKey, Ownership]
    face_ownership: tuple[Ownership, ...]


def associate(
    persons: Sequence[TrackObservation],
    faces: Sequence[FaceObservation],
    geometry: AssociationGeometry = AssociationGeometry(),
) -> Association:
    frames = {p.frame for p in persons} | {f.frame for f in faces}
    if len(frames) > 1:
        raise ValueError("faces and people must come from the same frame")
    candidates_of_face = [
        {p.key for p in persons if _is_candidate(face.box, p.box, geometry)} for face in faces
    ]
    candidates_of_person: dict[TrackKey, set[int]] = {p.key: set() for p in persons}
    for index, keys in enumerate(candidates_of_face):
        for key in keys:
            candidates_of_person[key].add(index)

    face_of: dict[TrackKey, int] = {}
    person_ownership: dict[TrackKey, Ownership] = {}
    for key, indices in candidates_of_person.items():
        if not indices:
            person_ownership[key] = Ownership.NO_FACE
        elif len(indices) == 1 and candidates_of_face[next(iter(indices))] == {key}:
            face_of[key] = next(iter(indices))
            person_ownership[key] = Ownership.ASSIGNED
        else:
            person_ownership[key] = Ownership.AMBIGUOUS
    assigned_faces = set(face_of.values())
    face_ownership = tuple(
        Ownership.ASSIGNED
        if index in assigned_faces
        else Ownership.NO_PERSON
        if not keys
        else Ownership.AMBIGUOUS
        for index, keys in enumerate(candidates_of_face)
    )
    return Association(face_of, person_ownership, face_ownership)


def withhold(association: Association, keys: Collection[TrackKey]) -> Association:
    """Route B: the people in ``keys`` were not analysed this tick (identity.window_stage). The association stays the one
    decided over everyone, so a face that was also theirs stays AMBIGUOUS for all; they get NOT_ANALYSED and no face, and
    a face that was assigned to one of them is NOT_ANALYSED too: it goes to nobody else."""
    withheld = set(keys)
    if not withheld:
        return association
    faces = {association.face_of[k] for k in withheld if k in association.face_of}
    return Association(
        {k: i for k, i in association.face_of.items() if k not in withheld},
        {k: Ownership.NOT_ANALYSED if k in withheld else o for k, o in association.person_ownership.items()},
        tuple(Ownership.NOT_ANALYSED if i in faces else o for i, o in enumerate(association.face_ownership)),
    )


def _is_candidate(face: NormalizedBox, person: NormalizedBox, geometry: AssociationGeometry) -> bool:
    cx, cy = (face.x1 + face.x2) / 2, (face.y1 + face.y2) / 2
    head_bottom = person.y1 + geometry.head_fraction * (person.y2 - person.y1)
    if not (person.x1 <= cx <= person.x2 and person.y1 <= cy <= head_bottom):
        return False
    inter_w = min(face.x2, person.x2) - max(face.x1, person.x1)
    inter_h = min(face.y2, person.y2) - max(face.y1, person.y1)
    face_area = (face.x2 - face.x1) * (face.y2 - face.y1)
    return inter_w > 0 and inter_h > 0 and inter_w * inter_h >= geometry.min_inside * face_area
