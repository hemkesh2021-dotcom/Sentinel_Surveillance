"""Per-track identity state (guide chapters 9 and 11; audit findings 1 and 2).

A track's identity is derived, every time it is read, from that track's own
recent face votes: faces associated to it one-to-one, of sufficient quality,
matched against a non-empty enrollment. There is no stored label to go stale.

- KNOWN: the last ``confirmations`` votes all matched the same identity.
- UNKNOWN: the last ``confirmations`` votes all matched nobody enrolled. It
  says only that; it is not a "stranger" verdict and grants or denies nothing.
- UNCERTAIN: recent votes contradict each other.
- UNRESOLVED: not enough evidence, which is where every track starts and
  stays when no face is visible, ownership is ambiguous, quality is low or
  nobody is enrolled.

Votes expire after ``vote_ttl_ns`` and are scoped to their TrackKey, so a
track in a new stream epoch inherits nothing. Identity is context for rules,
never an access decision by itself.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from ..contracts import TrackKey, TrackObservation
from ..media.clock import MonoInstant
from .association import Association, FaceObservation, Ownership
from .matching import Enrollment, best_match

VOTES_KEPT = 5
UNKNOWN_VOTE = ""  # a good face that matched nobody enrolled


class IdentityState(str, Enum):
    UNRESOLVED = "unresolved"
    UNKNOWN = "unknown"
    KNOWN = "known"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class IdentityPolicy:
    """Starting values only: thresholds need calibration on held-out identities (V2-25)."""

    match_threshold: float = 0.5  # cosine similarity, not a probability
    margin: float = 0.05  # over the runner-up identity
    min_quality: float = 0.6
    confirmations: int = 2
    vote_ttl_ns: int = 30_000_000_000


@dataclass(frozen=True)
class TrackIdentity:
    state: IdentityState
    identity_id: str | None  # only when KNOWN
    reason: str


@dataclass(frozen=True)
class _Vote:
    at_ns: int
    identity: str  # identity ID, or UNKNOWN_VOTE


class IdentityResolver:
    def __init__(self, enrollment: Enrollment | None, policy: IdentityPolicy = IdentityPolicy()) -> None:
        if policy.confirmations < 1:
            raise ValueError("confirmations must be at least 1")
        self._enrollment = enrollment
        self._policy = policy
        self._votes: dict[TrackKey, deque[_Vote]] = {}
        self._last_note: dict[TrackKey, str] = {}

    def observe(
        self,
        persons: Sequence[TrackObservation],
        faces: Sequence[FaceObservation],
        association: Association,
    ) -> None:
        """Record one frame's face evidence for its people."""
        for person in persons:
            key = person.key
            ownership = association.person_ownership.get(key, Ownership.NO_FACE)
            if ownership is not Ownership.ASSIGNED:
                self._last_note[key] = {
                    Ownership.NO_FACE: "no face visible",
                    Ownership.AMBIGUOUS: "face ownership ambiguous",
                }.get(ownership, ownership.value)
                continue
            vote, note = self._vote(faces[association.face_of[key]])
            self._last_note[key] = note
            if vote is not None:
                votes = self._votes.setdefault(key, deque(maxlen=VOTES_KEPT))
                votes.append(_Vote(person.observed_mono_ns, vote))

    def identity(self, key: TrackKey, now: MonoInstant) -> TrackIdentity:
        note = self._last_note.get(key, "no face observed yet")
        if self._enrollment is None or self._enrollment.is_empty:
            return TrackIdentity(IdentityState.UNRESOLVED, None, "no identities enrolled")
        votes = self._current_votes(key, now)
        needed = self._policy.confirmations
        if len(votes) >= needed:
            recent = {vote.identity for vote in votes[-needed:]}
            if len(recent) == 1:
                (value,) = recent
                if value == UNKNOWN_VOTE:
                    return TrackIdentity(IdentityState.UNKNOWN, None, "consistent faces matched nobody")
                return TrackIdentity(IdentityState.KNOWN, value, f"{needed} consistent matches")
        if len({vote.identity for vote in votes}) > 1:
            return TrackIdentity(IdentityState.UNCERTAIN, None, "contradictory face evidence")
        return TrackIdentity(IdentityState.UNRESOLVED, None, note)

    @property
    def tracked(self) -> int:
        """Tracks with identity bookkeeping; bounded by the current tracks."""
        return len(self._votes.keys() | self._last_note.keys())

    def retain_only(self, keys: Iterable[TrackKey]) -> None:
        """Forget tracks that are no longer current."""
        keep = set(keys)
        for table in (self._votes, self._last_note):
            for key in [k for k in table if k not in keep]:
                del table[key]

    def _vote(self, face: FaceObservation) -> tuple[str | None, str]:
        """The face's vote (an identity ID or UNKNOWN_VOTE), or None, and why."""
        enrollment = self._enrollment
        if enrollment is None or enrollment.is_empty:
            return None, "no identities enrolled"
        if face.quality < self._policy.min_quality or face.embedding is None:
            return None, "face quality too low"
        if len(face.embedding) != enrollment.dimension:
            return None, "embedding from an incompatible model"
        match = best_match(face.embedding, enrollment)
        if match.similarity < self._policy.match_threshold:
            return UNKNOWN_VOTE, "face matched nobody"
        if match.runner_up is not None and match.similarity - match.runner_up < self._policy.margin:
            return None, "face matches more than one identity"
        return match.identity_id, "face matched"

    def _current_votes(self, key: TrackKey, now: MonoInstant) -> list[_Vote]:
        if key.boot_id != now.boot_id:
            return []
        votes = self._votes.get(key, ())
        return [v for v in votes if 0 <= now.ns - v.at_ns < self._policy.vote_ttl_ns]
