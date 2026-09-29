"""Enrollment gallery and matching (guide chapter 11).

Enrollment is plain validated numbers, never pickled objects: a model
revision, a dimension and a few normalized prototypes per identity. An
embedding is compared only with prototypes of the same model revision and
dimension. Similarity is cosine similarity in [-1, 1]; a threshold on it is not
a probability or a "confidence".

This pure-Python matcher is the reference behaviour for small galleries; the
vectorized matcher (V2-25) must give the same decisions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field, FiniteFloat, PositiveInt, StringConstraints, model_validator

from ..contracts import Contract, Identifier

MAX_PROTOTYPES = 8


class EnrolledIdentity(Contract):
    identity_id: Identifier
    prototypes: Annotated[
        tuple[tuple[FiniteFloat, ...], ...], Field(min_length=1, max_length=MAX_PROTOTYPES)
    ]


class Enrollment(Contract):
    model_revision: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    dimension: Annotated[PositiveInt, Field(le=2048)]
    identities: tuple[EnrolledIdentity, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Enrollment:
        ids = [identity.identity_id for identity in self.identities]
        if len(set(ids)) != len(ids):
            raise ValueError("identity IDs must be unique")
        for identity in self.identities:
            for prototype in identity.prototypes:
                if len(prototype) != self.dimension:
                    raise ValueError(
                        f"{identity.identity_id}: prototype has {len(prototype)} values, "
                        f"expected {self.dimension}"
                    )
                if _norm(prototype) == 0.0:
                    raise ValueError(f"{identity.identity_id}: zero prototype")
        return self

    @property
    def is_empty(self) -> bool:
        return not self.identities


@dataclass(frozen=True)
class Match:
    identity_id: str | None  # best identity, or None if the gallery is empty
    similarity: float  # best cosine similarity
    runner_up: float | None  # best similarity of any other identity


def best_match(embedding: Sequence[float], enrollment: Enrollment) -> Match:
    if len(embedding) != enrollment.dimension:
        raise ValueError(f"embedding has {len(embedding)} values, expected {enrollment.dimension}")
    norm = _norm(embedding)
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("embedding must be finite and non-zero")
    scores = sorted(
        (
            (
                max(
                    _dot(embedding, prototype) / (norm * _norm(prototype))
                    for prototype in identity.prototypes
                ),
                identity.identity_id,
            )
            for identity in enrollment.identities
        ),
        reverse=True,
    )
    if not scores:
        return Match(identity_id=None, similarity=-1.0, runner_up=None)
    best, best_id = scores[0]
    return Match(best_id, best, scores[1][0] if len(scores) > 1 else None)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in v))
