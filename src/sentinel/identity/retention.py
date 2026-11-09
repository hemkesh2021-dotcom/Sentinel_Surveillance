"""An established identity kept on one continuously followed track, above the raw votes (session 116, part 28).

PROPOSED and experimental; opt-in, validation runs only (``sentinel run --identity-retention``, with
``--face-validation``). Every policy value is a PROPOSED starting value, unvalidated on the device, and some were chosen
with reference to one saved run: that run diagnoses, it does not validate them.

The raw face path (identity.state) keeps its meaning: its KNOWN/UNKNOWN/UNCERTAIN/UNRESOLVED rule, thresholds,
confirmations, vote expiry and transition records are untouched, and with the mode off nothing differs at all. With the
mode on the one raw effect is D97's: a continuity break drops the track's votes and a result from before the break is
not counted (``IdentityResolver.break_track``). This module keeps a separate **established association** per track
above those votes, so a person whose identity was established on a track is not erased by a look away, a blurred face or
a couple of non-matching results. It never changes a vote or a label: it adds, beside the raw state, "this track was
established as an enrolled identity and has not lost that": ``current`` when the raw state is a current qualifying
match (KNOWN, fresh, the same identity), ``retained`` otherwise.

**Established** (the only way an association starts): the raw state is KNOWN (the existing consistent-matches rule)
and its confirming votes are all from this track's current life, after its last continuity break (the resolver drops
everything earlier) and not before the time of any earlier association end on this life. Numbers on the track are no
evidence of continuity: identity.continuity's monitor runs alongside it, with the existing ``identity.continuity_*``
values, and its breaks reach the votes exactly as D97 does (``IdentityResolver.break_track``), while the raw votes keep
the plain expiry (``continuity_retention_ns`` stays None: this is not ``--face-continuity``).

An association **ends**, with a fixed reason, on: a continuity break (gap, jump, overlap); ownership ambiguous on a
face result (also for a person the window stage withheld from that result: the ownership decided over everyone before
withholding counts); the track no longer being current, or beginning a new life with the same number; the video no
longer being live at a step (``video_not_live``: a stall withdraws every track, and at a reconnect the capture withdraws
the epoch first, so that is the only thing a step can know then) or another stream epoch being live; face recognition
turned off; ``max_since_match`` since its last successful match of that identity (**the bound: this is not whole-track
retention**); sustained reliable contradiction (at least ``contradiction_votes`` contradicting votes since the last
match, spanning at least ``contradiction_span``, a match resetting the count); or ``conflict_votes`` consecutive votes
for one different enrolled identity, all of them still current for the raw rule when the last is applied
(``conflict_window``, which is identity.vote_ttl_s: exactly when the raw rule reports that identity, on its own
evidence, not as a transfer). After an end, a vote from before it cannot re-establish it, and nothing carries across a
new TrackKey, a new life or a restart (this ledger is in memory only). The one exception by design: a conflict end
allows the conflicting identity's own confirming votes, which are new evidence for that identity.

Face evidence since the last match is classified, never scored. A **reliable contradiction**: an assigned face in this
track's current life, not rejected as from before its expiry or break, for which the production decision matched nobody
or a different enrolled identity, at least ``min_face_width_fraction`` of the frame's width wide. **Degraded evidence**,
which neither counts nor erases: no face, not analysed, low quality, matches more than one identity, incompatible, or a
non-matching face below that width. A different identity's vote counts toward ``conflict_votes`` whatever its width,
because the raw rule counts it; the width decides only whether it is also a reliable contradiction.

Not thread-safe: EdgeCore calls it on its loop thread. Bounded by the current tracks: an association or an end is
forgotten as soon as its track is no longer current or begins a new life. Numbers and fixed labels only; the identity
ID is opaque and appears only in LiveState and the run's association records, never in the status (D61).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from pydantic import NonNegativeInt, PositiveInt

from ..contracts import Contract, Identifier, StreamIdentity, TrackKey
from ..media.clock import NS_PER_SECOND, MonoInstant
from .association import Ownership
from .continuity import MAX_RETENTION_NS, Cause, ContinuityPolicy
from .state import (
    BEFORE_BREAK,
    BEFORE_EXPIRY,
    UNKNOWN_VOTE,
    Basis,
    IdentityResolver,
    IdentityState,
    TrackIdentity,
    VoteRecord,
)

NS_PER_MS = 1_000_000

NONE = "none"  # no association: the third state of an association change's "from" and "to"


class AssociationState(str, Enum):
    CURRENT = "current"  # the raw state is KNOWN, fresh, the same identity: a current qualifying match
    RETAINED = "retained"  # kept from an earlier match; anything else while the association lives


class Evidence(str, Enum):
    """The newest face result applied to a track's association since it was established, as a fixed label."""

    NONE = "none"  # no result since it was established
    MATCH = "match"  # the newest result matched the established identity
    UNKNOWN = "unknown"  # matched nobody (a wide enough face: a reliable contradiction)
    DIFFERENT_IDENTITY = "different_identity"  # matched another enrolled identity
    TOO_SMALL = "too_small"  # a non-matching face narrower than the width limit (degraded)
    LOW_QUALITY = "low_quality"
    MARGIN = "margin"  # matches more than one identity
    INCOMPATIBLE = "incompatible"
    NO_FACE = "no_face"
    NOT_ANALYSED = "not_analysed"


class EndReason(str, Enum):
    CONTINUITY_GAP = "continuity_gap"
    CONTINUITY_JUMP = "continuity_jump"
    CONTINUITY_OVERLAP = "continuity_overlap"
    AMBIGUOUS_OWNERSHIP = "ambiguous_ownership"
    TRACK_NOT_CURRENT = "track_not_current"
    NEW_TRACK_LIFE = "new_track_life"
    STREAM_EPOCH_CHANGE = "stream_epoch_change"  # another stream epoch was live at that step
    VIDEO_NOT_LIVE = "video_not_live"  # no stream was live at that step (not fresh or withdrawn; says nothing of why)
    RECOGNITION_DISABLED = "recognition_disabled"
    MAX_SINCE_MATCH = "max_since_match"
    SUSTAINED_CONTRADICTION = "sustained_contradiction"
    CONFLICTING_IDENTITY = "conflicting_identity"


# An association record's reason: why it began or changed, or why it ended (EndReason)
ESTABLISHED = "established"
CURRENT_MATCH = "current_match"
NO_CURRENT_MATCH = "no_current_match"
CHANGE_REASONS = (ESTABLISHED, CURRENT_MATCH, NO_CURRENT_MATCH, *(reason.value for reason in EndReason))
RECORD_STATES = (NONE, AssociationState.CURRENT.value, AssociationState.RETAINED.value)

_BREAK_REASONS = {Cause.GAP: EndReason.CONTINUITY_GAP, Cause.JUMP: EndReason.CONTINUITY_JUMP,
                  Cause.OVERLAP: EndReason.CONTINUITY_OVERLAP}
# Of ContinuityPolicy.numbers(), what a retention run reports beside its own rule: the values that decide every break
# that ends an association. Its retention_s is left out: this mode does not use it (the raw votes keep the plain expiry)
MONITOR_NUMBERS = ("max_gap_s", "overlap_iou", "min_step_iou")


@dataclass(frozen=True)
class RetentionPolicy:
    """PROPOSED starting values, unvalidated (see the module docstring); ``conflict_votes`` is identity.confirmations and
    ``conflict_window_ns`` identity.vote_ttl_s (``IdentityConfig.retention_policy``), so that a conflict means what the
    raw rule would report. Numbers chosen with reference to one saved run are diagnosis only."""

    max_since_match_ns: int = 300 * NS_PER_SECOND  # the bound after the last match: at most continuity.MAX_RETENTION_NS
    contradiction_votes: int = 10  # reliable contradicting votes since the last match that end it...
    contradiction_span_ns: int = 10 * NS_PER_SECOND  # ...if they span at least this (a face result is about 1 Hz)
    min_face_width_fraction: float = 0.03  # of the frame's width: a narrower non-matching face is degraded evidence
    conflict_votes: int = 2  # consecutive votes for one different enrolled identity that end it at once...
    conflict_window_ns: int = 30 * NS_PER_SECOND  # ...if all are younger than this when the last is applied

    def __post_init__(self) -> None:
        if not 0 < self.max_since_match_ns <= MAX_RETENTION_NS:
            raise ValueError(f"max_since_match_ns must be positive and at most {MAX_RETENTION_NS}")
        if self.contradiction_votes < 1 or self.conflict_votes < 1:
            raise ValueError("contradiction_votes and conflict_votes must be at least 1")
        if self.contradiction_span_ns <= 0:
            raise ValueError("contradiction_span_ns must be positive")
        if self.conflict_window_ns <= 0:
            raise ValueError("conflict_window_ns must be positive")
        if not 0 <= self.min_face_width_fraction < 1:
            raise ValueError("min_face_width_fraction must be in [0, 1)")

    def numbers(self) -> dict[str, float]:
        """For the status and the startup record: the rule as configured (numbers only)."""
        return {"max_since_match_s": self.max_since_match_ns / NS_PER_SECOND,
                "contradiction_votes": self.contradiction_votes,
                "contradiction_span_s": self.contradiction_span_ns / NS_PER_SECOND,
                "min_face_width_fraction": self.min_face_width_fraction,
                "conflict_votes": self.conflict_votes,
                "conflict_window_s": self.conflict_window_ns / NS_PER_SECOND}


def reported_numbers(policy: RetentionPolicy, monitor: ContinuityPolicy) -> dict[str, float]:
    """What a retention run reports in its status and on its starting line: its own rule and the continuity monitor's
    values that decide every break ending an association (MONITOR_NUMBERS). Numbers only."""
    numbers = monitor.numbers()
    return {**policy.numbers(), **{name: numbers[name] for name in MONITOR_NUMBERS}}


class TrackAssociation(Contract):
    """A track's established association in one publication, separate from the raw identity fields (their meaning is
    unchanged: the latest face evidence). ``identity_id`` is opaque and stays in LiveState and the run's records."""

    state: AssociationState
    identity_id: Identifier
    last_match_age_ms: NonNegativeInt  # since the last match of the established identity on this track
    evidence: Evidence  # the newest face result since it was established, as a fixed label
    evidence_age_ms: NonNegativeInt | None  # None: no result since it was established
    contradictions: NonNegativeInt  # reliable contradicting votes since the last match
    contradiction_limit: PositiveInt
    conflicts: NonNegativeInt  # consecutive votes for one different enrolled identity, those the raw rule still counts
    conflict_limit: PositiveInt
    max_since_match_ms: NonNegativeInt  # the bound


class AssociationEnded(Contract):
    """A track's most recent association end on its current life, kept per current track so that the row can say why."""

    reason: EndReason
    age_ms: NonNegativeInt


@dataclass(frozen=True)
class AssociationChange:
    """One change of a track's association, as a run record's fields (runtime.association_record)."""

    stream_epoch: int
    track_id: int
    from_state: str  # none, current or retained
    to_state: str  # none, current or retained
    identity_id: str | None
    reason: str
    last_match_age_ms: int | None
    contradictions: int
    mono_ns: int


@dataclass
class _Entry:
    identity_id: str
    life_ns: int  # the track's life it was established on (TrackTable.since)
    established_ns: int  # the frame time of its newest confirming vote; nothing from an earlier frame is applied
    last_match_ns: int  # the frame time of the last match of the identity
    evidence: Evidence = Evidence.NONE
    evidence_ns: int | None = None  # the frame time of that evidence (None: none yet)
    contradictions: int = 0
    first_contradiction_ns: int | None = None  # the first and latest reliable contradicting vote since the last match
    last_contradiction_ns: int | None = None
    conflict_identity: str | None = None  # the different identity of the current run of consecutive votes
    conflict_times: tuple[int, ...] = ()  # that run's frame times, oldest first: at most conflict_votes of them
    reported: str = NONE  # the state this association was last published in


@dataclass(frozen=True)
class _Ended:
    at_ns: int  # the frame or decision time of the end: no vote from before it may establish the association again
    reason: EndReason
    life_ns: int  # the life it ended on; a new life keeps nothing
    ended_ns: int  # when it ended, for the row's age
    allow_identity: str | None = None  # a conflict end: the conflicting identity, whose own confirming votes count
    allow_from_ns: int | None = None  # ... from its first conflicting vote


class RetentionLedger:
    def __init__(self, policy: RetentionPolicy = RetentionPolicy()) -> None:
        self._policy = policy
        self._entries: dict[TrackKey, _Entry] = {}
        self._ended: dict[TrackKey, _Ended] = {}
        self._changes: list[AssociationChange] = []
        self.counters: Counter[str] = Counter()  # associations established and ended, by reason; numbers only

    @property
    def policy(self) -> RetentionPolicy:
        return self._policy

    @property
    def tracked(self) -> int:
        """Tracks with an association or a kept end; bounded by the current tracks."""
        return len(self._entries.keys() | self._ended.keys())

    def on_break(self, key: TrackKey, at_ns: int, cause: Cause, now_ns: int) -> None:
        """A continuity break on the track's frame at ``at_ns`` (identity.continuity): its association ends at once."""
        self._end(key, _BREAK_REASONS[cause], at_ns, now_ns)

    def observe(self, record: VoteRecord, at_ns: int, width_fraction: float | None, now_ns: int, *,
                ambiguous: bool = False) -> None:
        """One person's outcome from one applied face result (or an on_frame face stage's) on the frame at ``at_ns``:
        ``width_fraction`` is the assigned face's width as a fraction of the frame's, when it has one. ``ambiguous``: the
        association decided over everyone was AMBIGUOUS for this person, whom the window stage then withheld from this
        result (association.withhold rewrites the ownership, so the vote record, unchanged, says NOT_ANALYSED)."""
        entry = self._entries.get(record.key)
        if entry is None or record.label in (BEFORE_BREAK, BEFORE_EXPIRY):
            return  # no association on this track, or a result the raw path did not count for this life or break
        if at_ns < entry.established_ns or at_ns < entry.last_match_ns:
            self.counters["association_evidence_late"] += 1  # from before the match it stands on: it changes nothing
            return
        if ambiguous or record.ownership is Ownership.AMBIGUOUS:
            self._end(record.key, EndReason.AMBIGUOUS_OWNERSHIP, at_ns, now_ns)
            return
        vote = record.vote
        if vote == entry.identity_id:
            entry.last_match_ns = at_ns
            entry.contradictions, entry.first_contradiction_ns, entry.last_contradiction_ns = 0, None, None
            entry.conflict_identity, entry.conflict_times = None, ()
            entry.evidence, entry.evidence_ns = Evidence.MATCH, at_ns
            return
        wide_enough = width_fraction is not None and width_fraction >= self._policy.min_face_width_fraction
        if vote is not None:  # a vote that is not the established identity: nobody, or a different enrolled identity
            if vote == UNKNOWN_VOTE:
                evidence = Evidence.UNKNOWN if wide_enough else Evidence.TOO_SMALL
                entry.conflict_identity, entry.conflict_times = None, ()
            else:  # a different enrolled identity: the run holds only votes the raw rule still counts
                evidence = Evidence.DIFFERENT_IDENTITY
                earlier = entry.conflict_times if entry.conflict_identity == vote else ()
                entry.conflict_identity = vote
                entry.conflict_times = self._still_current((*earlier, at_ns), now_ns)[-self._policy.conflict_votes:]
            entry.evidence, entry.evidence_ns = evidence, at_ns
            if wide_enough:
                entry.contradictions += 1
                if entry.first_contradiction_ns is None:
                    entry.first_contradiction_ns = at_ns
                entry.last_contradiction_ns = at_ns
            if len(entry.conflict_times) >= self._policy.conflict_votes:
                self._end(record.key, EndReason.CONFLICTING_IDENTITY, at_ns, now_ns, allow_identity=vote,
                          allow_from_ns=entry.conflict_times[0])
            elif (entry.contradictions >= self._policy.contradiction_votes
                  and entry.last_contradiction_ns - entry.first_contradiction_ns  # type: ignore[operator]  # set above
                  >= self._policy.contradiction_span_ns):
                self._end(record.key, EndReason.SUSTAINED_CONTRADICTION, at_ns, now_ns)
            return
        # no vote: degraded evidence, which neither counts nor erases (no_enrollment changes nothing either)
        evidence = {Ownership.NO_FACE: Evidence.NO_FACE, Ownership.NOT_ANALYSED: Evidence.NOT_ANALYSED}.get(
            record.ownership, {"low_quality": Evidence.LOW_QUALITY, "margin": Evidence.MARGIN,
                               "incompatible": Evidence.INCOMPATIBLE}.get(record.label))
        if evidence is not None:
            entry.evidence, entry.evidence_ns = evidence, at_ns

    def forget_all(self, now_ns: int) -> None:
        """Face recognition turned off or on at runtime (D99): every association ends, and no end is kept: nothing
        earlier is presented, and a later start begins with nothing."""
        for key in list(self._entries):
            self._end(key, EndReason.RECOGNITION_DISABLED, now_ns, now_ns, remember=False)
        self._ended.clear()

    def step(
        self, lives: Mapping[TrackKey, int], identities: Mapping[TrackKey, TrackIdentity], resolver: IdentityResolver,
        live: StreamIdentity | None, now: MonoInstant,
    ) -> tuple[dict[TrackKey, TrackAssociation], dict[TrackKey, AssociationEnded], tuple[AssociationChange, ...]]:
        """One publication: ends what is no longer current, establishes what newly holds, and returns each current
        track's view, each current track's most recent end, and the changes since the last step in order."""
        for key, entry in list(self._entries.items()):
            if key not in lives:
                self._end(key, self._gone_reason(key, live), now.ns, now.ns, remember=False)
            elif lives[key] != entry.life_ns:
                self._end(key, EndReason.NEW_TRACK_LIFE, now.ns, now.ns, remember=False)
            elif now.ns - entry.last_match_ns >= self._policy.max_since_match_ns:
                self._end(key, EndReason.MAX_SINCE_MATCH, now.ns, now.ns)
        for key in [k for k, ended in self._ended.items() if k not in lives or ended.life_ns != lives[k]]:
            del self._ended[key]  # kept per current track and life only
        views: dict[TrackKey, TrackAssociation] = {}
        for key, began in lives.items():
            entry = self._entries.get(key)
            resolved = identities.get(key)
            if entry is None:
                entry = self._establish(key, began, resolved, resolver, now)
                if entry is None:
                    continue
            view = self._view(entry, resolved, now)
            if view.state.value != entry.reported:
                reason = (ESTABLISHED if entry.reported == NONE else
                          CURRENT_MATCH if view.state is AssociationState.CURRENT else NO_CURRENT_MATCH)
                self._changes.append(AssociationChange(
                    key.stream_epoch, key.track_id, entry.reported, view.state.value, entry.identity_id, reason,
                    view.last_match_age_ms, entry.contradictions, now.ns))
                entry.reported = view.state.value
            views[key] = view
        ended = {key: AssociationEnded(reason=item.reason, age_ms=max(0, now.ns - item.ended_ns) // NS_PER_MS)
                 for key, item in self._ended.items() if key not in self._entries}
        changes, self._changes = tuple(self._changes), []
        return views, ended, changes

    def _end(self, key: TrackKey, reason: EndReason, at_ns: int, now_ns: int, *, allow_identity: str | None = None,
             allow_from_ns: int | None = None, remember: bool = True) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        self.counters[f"association_ended_{reason.value}"] += 1
        self._changes.append(AssociationChange(
            key.stream_epoch, key.track_id, entry.reported, NONE, entry.identity_id, reason.value,
            max(0, now_ns - entry.last_match_ns) // NS_PER_MS, entry.contradictions, now_ns))
        if remember:
            self._ended[key] = _Ended(at_ns, reason, entry.life_ns, now_ns, allow_identity, allow_from_ns)

    def _still_current(self, times: tuple[int, ...], now_ns: int) -> tuple[int, ...]:
        """Those of a conflict run's frame times the raw rule still counts at ``now_ns`` (``conflict_window``)."""
        return tuple(at for at in times if now_ns - at < self._policy.conflict_window_ns)

    @staticmethod
    def _gone_reason(key: TrackKey, live: StreamIdentity | None) -> EndReason:
        """Why a track with an association is no longer current: as far as this step can tell. No stream is live (a
        stall, or a reconnect, where the capture withdraws the epoch before a new one exists); another epoch is live;
        or its stream is still the live one and the track ended."""
        if live is None:
            return EndReason.VIDEO_NOT_LIVE
        if (key.camera_id, key.boot_id, key.run_id, key.stream_epoch) != (
                live.camera_id, live.boot_id, live.run_id, live.stream_epoch):
            return EndReason.STREAM_EPOCH_CHANGE
        return EndReason.TRACK_NOT_CURRENT

    def _establish(self, key: TrackKey, began: int, resolved: TrackIdentity | None, resolver: IdentityResolver,
                   now: MonoInstant) -> _Entry | None:
        """A new association when Established holds (the module docstring). The resolver has already dropped votes from
        other lives and breaks; here: not before the time of any earlier end on this life (a conflict end allows the
        conflicting identity's own votes)."""
        if resolved is None or resolved.state is not IdentityState.KNOWN or resolved.identity_id is None:
            return None
        times = resolver.confirming_votes(key, now)
        if not times:
            return None
        ended = self._ended.get(key)
        if ended is not None:
            if resolved.identity_id == ended.allow_identity and ended.allow_from_ns is not None:
                if min(times) < ended.allow_from_ns:
                    return None
            elif min(times) < ended.at_ns:
                return None
        newest = max(times)
        entry = _Entry(resolved.identity_id, began, newest, newest)
        self._entries[key] = entry
        self._ended.pop(key, None)  # superseded: the row says why only while no association stands
        self.counters["associations_established"] += 1
        return entry

    def _view(self, entry: _Entry, resolved: TrackIdentity | None, now: MonoInstant) -> TrackAssociation:
        current = (resolved is not None and resolved.state is IdentityState.KNOWN and resolved.basis is Basis.FRESH
                   and resolved.identity_id == entry.identity_id)
        return TrackAssociation(
            state=AssociationState.CURRENT if current else AssociationState.RETAINED,
            identity_id=entry.identity_id,
            last_match_age_ms=max(0, now.ns - entry.last_match_ns) // NS_PER_MS,
            evidence=entry.evidence,
            evidence_age_ms=None if entry.evidence_ns is None else max(0, now.ns - entry.evidence_ns) // NS_PER_MS,
            contradictions=entry.contradictions,
            contradiction_limit=self._policy.contradiction_votes,
            conflicts=len(self._still_current(entry.conflict_times, now.ns)),
            conflict_limit=self._policy.conflict_votes,
            max_since_match_ms=self._policy.max_since_match_ns // NS_PER_MS,
        )
