"""Track retention (session 116, part 28; PROPOSED, ``--identity-retention`` validation runs): an established identity is
kept on a continuously followed track, apart from the raw votes, which do not change, and it says retained.

Scenarios on the real EdgeCore and PersonTracker with a fake clock at 10 frames per second, face results applied as the
1 Hz face worker would, as test_identity_continuity_rule does for D97. Synthetic 4-D embeddings only. The two-non-match
sequence below is a synthetic reproduction of the pattern seen in one saved run: it reproduces that pattern, and it
validates nothing about the policy, which was designed after looking at that run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sentinel.config import ConfigError, parse_config
from sentinel.contracts import NormalizedBox, PixelFormat
from sentinel.identity.association import FaceObservation, Ownership
from sentinel.identity.continuity import NOT_CARRIED, Cause
from sentinel.identity.matching import EnrolledIdentity, Enrollment
from sentinel.identity.retention import (
    CHANGE_REASONS,
    MONITOR_NUMBERS,
    AssociationState,
    EndReason,
    Evidence,
    RetentionLedger,
    RetentionPolicy,
    reported_numbers,
)
from sentinel.identity.state import BEFORE_BREAK, BEFORE_EXPIRY, Basis, IdentityState, VoteRecord
from sentinel.identity.window_stage import WindowNumbers
from sentinel.live_state import Capability
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.runtime import EdgeCore, association_record
from sentinel.tracking.tracker import PersonTracker, RawTrack

CONFIG = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
POLICY = CONFIG.identity.retention_policy()
ENROLLED, OTHER = "idn-00000000000a", "idn-00000000000b"
ENROLLMENT = Enrollment(model_revision="synthetic", dimension=4, identities=(
    EnrolledIdentity(identity_id=ENROLLED, prototypes=((1.0, 0.0, 0.0, 0.0),)),
    EnrolledIdentity(identity_id=OTHER, prototypes=((0.0, 1.0, 0.0, 0.0),))))
MATCH, OTHER_FACE, NOBODY = (0.99, 0.1, 0.0, 0.0), (0.05, 0.99, 0.1, 0.0), (0.0, 0.1, 0.99, 0.0)
MARGIN = (0.7, 0.7, 0.0, 0.0)  # as close to both enrolled identities: no vote
SEAT = (100.0, 100.0, 200.0, 400.0)  # a seated person's box on the 640x480 frame
HEAD = NormalizedBox(x1=0.19, y1=0.22, x2=0.28, y2=0.35)  # inside that box's head region
SMALL_HEAD = NormalizedBox(x1=0.22, y1=0.25, x2=0.24, y2=0.28)  # 0.02 of the frame's width: below the width limit


class Backend:
    def __init__(self) -> None:
        self.next: list[tuple[int, tuple[float, ...]]] = []

    def track(self, image: Any) -> list[RawTrack]:
        return [RawTrack(track, *coords, 0.9) for track, coords in self.next]

    def reset(self) -> None:
        pass


class Room:
    """One person's life on tracks (the real tracker and core), face results applied as the 1 Hz worker applies them."""

    def __init__(self, retention: RetentionPolicy | None = POLICY, config: Any = CONFIG,
                 enrollment: Enrollment = ENROLLMENT) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stream = self.stamper.connect()
        self.core = EdgeCore(config, self.clock, None, face_recognition=Capability.AVAILABLE, enrollment=enrollment,
                             retention=retention)
        self.backend = Backend()
        self.tracker = PersonTracker(self.backend)
        self.state = None
        self.transitions: list[dict] = []
        self.changes: list[tuple[str, str, str]] = []
        self.records: list[dict] = []
        self.breaks = 0
        self.break_causes: list[tuple[int, str]] = []  # what the run's break records say: (track, cause)
        self.held: list = []  # (frame, persons) for a face result applied later

    def frame(self, *people: tuple[int, tuple[float, ...]]) -> tuple[Any, Any]:
        self.clock.advance(0.1)
        self.backend.next = list(people)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        persons = self.tracker.process(frame, object()).persons
        self._keep(self.core.on_frame(frame, persons, self.stream))
        return frame, persons

    def frames(self, seconds: float, *people: tuple[int, tuple[float, ...]], face=None, faces_at=None,
               quality: float = 0.95) -> None:
        """``seconds`` at 10 frames per second with these tracks; ``face`` (an embedding, or () for no face found) is
        analysed on each whole second's frame and applied at once, as the 1 Hz face worker would."""
        for n in range(round(seconds * 10)):
            frame, persons = self.frame(*people)
            if face is not None and n % 10 == 9 and persons:
                self.apply(frame, persons, face, faces_at, quality)

    def hold(self, *people: tuple[int, tuple[float, ...]]) -> None:
        """One frame whose face result is applied later (results arrive about a second late)."""
        self.held.append(self.frame(*people))

    def apply(self, frame, persons, face, head: NormalizedBox | None = None, quality: float = 0.95, windows=None):
        faces = () if face == () else (FaceObservation(frame=frame.key, box=head or HEAD, quality=quality,
                                                       embedding=face),)
        output = self.core.on_face_outcome(frame, persons, faces, self.stream, windows=windows)
        self._keep(output)
        return output

    def person(self, track: int):
        return next((p for p in self.state.people if p.track_id == track), None)

    def _keep(self, output) -> None:
        self.state = output.state
        self.transitions += [t.record() for t in output.identity_transitions]
        self.changes += [(c.from_state, c.to_state, c.reason) for c in output.identity_associations]
        self.records += [association_record(c) for c in output.identity_associations]
        self.breaks += len(output.identity_breaks)
        self.break_causes += [(b.key.track_id, b.cause.value) for b in output.identity_breaks]


def known(room: Room, track: int = 1, where: tuple[float, ...] = SEAT) -> Room:
    room.frames(2.0, (track, where), face=MATCH)  # two matches, one a second
    person = room.person(track)
    assert (person.identity, person.identity_id, person.identity_basis) == (IdentityState.KNOWN, ENROLLED, Basis.FRESH)
    assert person.association.state is AssociationState.CURRENT and person.association.identity_id == ENROLLED
    return room


# ---------------------------------------------------------------- the two non-match results (one saved run's pattern)


def test_two_non_matching_results_leave_the_raw_state_unknown_and_the_identity_retained_with_its_match_age() -> None:
    room = known(Room())
    assert room.person(1).association.evidence is Evidence.NONE and room.person(1).association.evidence_age_ms is None
    room.frames(1.0, (1, SEAT), face=NOBODY)  # the first non-match: contradictory raw evidence, already not a current match
    assert room.person(1).identity is IdentityState.UNCERTAIN
    assert room.person(1).association.state is AssociationState.RETAINED
    assert room.person(1).association.contradictions == 1
    room.frames(1.0, (1, SEAT), face=NOBODY)  # the second: the raw rule reports unknown, as it always did
    person = room.person(1)
    assert person.identity is IdentityState.UNKNOWN and person.identity_id is None
    association = person.association
    assert association.state is AssociationState.RETAINED and association.identity_id == ENROLLED
    assert association.last_match_age_ms == 2000  # the last match was two seconds of frame time ago
    assert (association.evidence, association.evidence_age_ms, association.contradictions) == (Evidence.UNKNOWN, 0, 2)
    assert (association.contradiction_limit, association.conflict_limit, association.max_since_match_ms) == (
        POLICY.contradiction_votes, 2, 300_000)
    assert room.changes == [("none", "current", "established"), ("current", "retained", "no_current_match")]
    assert room.records[0] == {"identity": "association", "schema": 1, "stream_epoch": 1, "track_id": 1,
                               "from": "none", "to": "current", "identity_id": ENROLLED, "reason": "established",
                               "last_match_age_ms": 0, "contradictions": 0, "mono_ns": room.records[0]["mono_ns"]}
    assert set(room.records[1]) == set(room.records[0]) and room.records[1]["contradictions"] == 1
    assert {t["to"] for t in room.transitions} == {"known", "uncertain", "unknown"}  # the raw transitions, unchanged


def test_looking_away_and_low_quality_faces_do_not_erase_or_count_and_the_raw_state_keeps_its_own_expiry() -> None:
    room = known(Room())
    room.frames(45.0, (1, SEAT), face=())  # no usable face for 45 s, the person followed throughout
    person = room.person(1)
    assert person.identity is IdentityState.UNRESOLVED  # the raw votes expired at 30 s as always
    association = person.association
    assert association.state is AssociationState.RETAINED and association.last_match_age_ms == 45_000
    assert (association.evidence, association.contradictions) == (Evidence.NO_FACE, 0)
    assert association.evidence_age_ms == 0
    room.frames(1.0, (1, SEAT), face=MATCH, quality=0.3)  # a low-quality face: no vote, no count, still degraded
    association = room.person(1).association
    assert (association.state, association.evidence, association.contradictions) == (
        AssociationState.RETAINED, Evidence.LOW_QUALITY, 0)
    assert room.core.identity_counters["persons_no_face"] >= 45
    assert room.breaks == 0


def test_a_margin_face_a_person_not_analysed_and_a_narrow_non_matching_face_are_degraded_evidence() -> None:
    room = known(Room())
    room.frames(1.0, (1, SEAT), face=MARGIN)  # matches more than one identity: no vote, so the raw match is still fresh
    association = room.person(1).association
    assert (association.evidence, association.contradictions, association.conflicts) == (Evidence.MARGIN, 0, 0)
    assert association.state is AssociationState.CURRENT
    room = known(Room())
    room.hold((1, SEAT))
    frame, persons = room.held.pop()
    windows = WindowNumbers(1, ((640, 640),), (1.0,), 1, 1, 0, (persons[0].key,), 0, 0, 0, 0)  # route B left this person out
    room.apply(frame, persons, (), windows=windows)
    association = room.person(1).association
    assert (association.evidence, association.contradictions) == (Evidence.NOT_ANALYSED, 0)
    room = known(Room())
    room.frames(1.0, (1, SEAT), face=NOBODY, faces_at=SMALL_HEAD)  # non-matching, too narrow: shown, not counted
    association = room.person(1).association
    assert (association.state, association.evidence, association.contradictions) == (
        AssociationState.RETAINED, Evidence.TOO_SMALL, 0)
    assert room.person(1).identity is IdentityState.UNCERTAIN  # the raw rule counted the vote, as it always does


# ---------------------------------------------------------------- what ends an association


def test_sustained_reliable_contradiction_ends_it_when_the_count_and_the_span_are_both_reached() -> None:
    room = known(Room())
    room.frames(10.0, (1, SEAT), face=NOBODY)  # ten contradicting votes in nine seconds of frame time: not the span yet
    association = room.person(1).association
    assert association.contradictions == 10 and association.state is AssociationState.RETAINED
    room.frames(1.0, (1, SEAT), face=NOBODY)  # the eleventh: span 10 s
    person = room.person(1)
    assert person.association is None and person.association_ended.reason is EndReason.SUSTAINED_CONTRADICTION
    assert person.identity is IdentityState.UNKNOWN  # the raw rule says the same evidence, on its own terms
    assert room.changes[-1] == ("retained", "none", "sustained_contradiction")
    assert room.records[-1]["contradictions"] == 11 and room.records[-1]["identity_id"] == ENROLLED
    assert room.core.identity_counters["association_ended_sustained_contradiction"] == 1
    room.frames(1.0, (1, SEAT), face=MATCH)  # after an end: one new match is not enough
    assert room.person(1).association is None and room.person(1).identity is IdentityState.UNCERTAIN
    room.frames(1.0, (1, SEAT), face=MATCH)  # two new matches: established again, with nothing carried from before
    person = room.person(1)
    assert person.association.state is AssociationState.CURRENT and person.association.contradictions == 0
    assert person.association_ended is None


def test_a_burst_of_contradicting_votes_under_the_span_does_not_end_it() -> None:
    room = known(Room())
    for _ in range(12):  # twelve contradicting results within 1.2 s of frame time: as many votes as you like, no span
        frame, persons = room.frame((1, SEAT))
        room.apply(frame, persons, NOBODY)
    association = room.person(1).association
    assert association.contradictions == 12 and association.state is AssociationState.RETAINED


def test_a_match_of_the_established_identity_resets_the_count() -> None:
    room = known(Room())
    room.frames(9.0, (1, SEAT), face=NOBODY)  # nine contradicting votes
    assert room.person(1).association.contradictions == 9
    room.frames(1.0, (1, SEAT), face=MATCH)  # one match: not enough for a current raw match, but the count starts over
    person = room.person(1)
    assert person.identity is IdentityState.UNCERTAIN
    assert (person.association.state, person.association.contradictions, person.association.evidence,
            person.association.last_match_age_ms) == (AssociationState.RETAINED, 0, Evidence.MATCH, 0)
    room.frames(9.0, (1, SEAT), face=NOBODY)
    assert room.person(1).association.contradictions == 9  # counted from the match: nowhere near the end
    assert room.person(1).association_ended is None


def test_a_different_enrolled_identity_is_flagged_after_one_vote_and_ends_it_after_the_confirmations() -> None:
    room = known(Room())
    room.frames(1.0, (1, SEAT), face=OTHER_FACE)
    person = room.person(1)
    assert person.identity is IdentityState.UNCERTAIN and person.association.state is AssociationState.RETAINED
    assert (person.association.evidence, person.association.conflicts, person.association.identity_id) == (
        Evidence.DIFFERENT_IDENTITY, 1, ENROLLED)
    room.frames(1.0, (1, SEAT), face=OTHER_FACE)  # the second consecutive vote: the raw rule reports OTHER on its own votes
    person = room.person(1)
    assert (person.identity, person.identity_id, person.identity_basis) == (IdentityState.KNOWN, OTHER, Basis.FRESH)
    assert person.association.identity_id == OTHER and person.association.state is AssociationState.CURRENT
    assert person.association_ended is None  # an association stands: the row says nothing about the earlier one
    assert room.changes[-2:] == [("retained", "none", "conflicting_identity"), ("none", "current", "established")]
    assert [r["identity_id"] for r in room.records[-2:]] == [ENROLLED, OTHER]
    assert room.core.identity_counters["association_ended_conflicting_identity"] == 1
    assert room.core.identity_counters["associations_established"] == 2


def test_a_different_identity_on_a_narrow_face_counts_toward_the_conflict_because_the_raw_rule_counts_it() -> None:
    room = known(Room())
    room.frames(1.0, (1, SEAT), face=OTHER_FACE, faces_at=SMALL_HEAD)
    association = room.person(1).association
    assert (association.conflicts, association.contradictions, association.evidence) == (
        1, 0, Evidence.DIFFERENT_IDENTITY)  # flagged, but not a reliable contradiction: too narrow to weigh
    room.frames(1.0, (1, SEAT), face=OTHER_FACE, faces_at=SMALL_HEAD)
    person = room.person(1)
    assert person.identity_id == OTHER and person.association.identity_id == OTHER  # consistent with the raw rule


def test_a_match_or_a_vote_for_nobody_interrupts_a_conflict_and_it_needs_consecutive_votes() -> None:
    room = known(Room())
    room.frames(1.0, (1, SEAT), face=OTHER_FACE)
    assert room.person(1).association.conflicts == 1
    room.frames(1.0, (1, SEAT), face=MATCH)
    assert room.person(1).association.conflicts == 0
    room.frames(1.0, (1, SEAT), face=OTHER_FACE)
    room.frames(1.0, (1, SEAT), face=NOBODY)  # a vote for nobody: the votes for OTHER are no longer consecutive
    assert room.person(1).association.conflicts == 0
    room.frames(1.0, (1, SEAT), face=OTHER_FACE)
    association = room.person(1).association
    assert association.identity_id == ENROLLED and association.conflicts == 1
    assert room.core.identity_counters.get("association_ended_conflicting_identity", 0) == 0


def test_a_conflict_end_re_establishes_only_the_conflicting_identity_and_later_matches_of_the_old_one_need_two_new() -> None:
    room = known(Room())
    room.frames(2.0, (1, SEAT), face=OTHER_FACE)  # ended, and OTHER established on its own two votes
    assert room.person(1).association.identity_id == OTHER
    room.frames(1.0, (1, SEAT), face=NOBODY)  # one contradiction against OTHER
    room.frames(1.0, (1, SEAT), face=MATCH)  # the old identity: one match proves nothing about this track yet
    assert room.person(1).association.identity_id == OTHER
    assert room.person(1).association.evidence is Evidence.DIFFERENT_IDENTITY
    room.frames(1.0, (1, SEAT), face=MATCH)  # a second consecutive match of the old identity: its own confirming votes end OTHER
    person = room.person(1)
    assert person.identity_id == ENROLLED and person.association.identity_id == ENROLLED
    assert person.association.state is AssociationState.CURRENT


def test_a_conflict_counts_only_votes_the_raw_rule_still_counts() -> None:
    room = known(Room())
    room.frames(1.0, (1, SEAT), face=OTHER_FACE)
    assert room.person(1).association.conflicts == 1
    room.frames(40.0, (1, SEAT), face=())  # no usable face for 40 s: that vote has expired, for the raw rule and the row
    assert room.person(1).association.conflicts == 0 and room.person(1).association.state is AssociationState.RETAINED
    room.frames(1.0, (1, SEAT), face=OTHER_FACE)  # one current vote for OTHER: the raw rule reports nothing on one
    person = room.person(1)
    assert person.identity is IdentityState.UNRESOLVED and person.association_ended is None
    assert (person.association.identity_id, person.association.conflicts) == (ENROLLED, 1)
    assert room.core.identity_counters.get("association_ended_conflicting_identity", 0) == 0
    room.frames(1.0, (1, SEAT), face=OTHER_FACE)  # a second, inside the raw window: now the raw rule reports OTHER itself
    person = room.person(1)
    assert (person.identity, person.identity_id, person.association.identity_id) == (IdentityState.KNOWN, OTHER, OTHER)
    assert room.changes[-2:] == [("retained", "none", "conflicting_identity"), ("none", "current", "established")]


def test_a_longer_conflict_run_slides_with_the_raw_window_and_ends_it_exactly_when_the_raw_rule_reports_the_other() -> None:
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "identity": {"confirmations": 3}})
    room = Room(retention=config.identity.retention_policy(), config=config)
    room.frames(3.0, (1, SEAT), face=MATCH)  # established on three matches, at 1, 2 and 3 s
    assert room.person(1).association.identity_id == ENROLLED and room.person(1).association.conflict_limit == 3
    room.frames(1.0, (1, SEAT), face=OTHER_FACE)  # at 4 s
    room.frames(19.0, (1, SEAT), face=())
    room.frames(1.0, (1, SEAT), face=OTHER_FACE)  # at 24 s
    room.frames(19.0, (1, SEAT), face=())
    room.frames(1.0, (1, SEAT), face=OTHER_FACE)  # at 44 s: the vote at 4 s is past the raw expiry, two are still current
    person = room.person(1)
    assert person.identity is IdentityState.UNRESOLVED
    assert (person.association.identity_id, person.association.conflicts) == (ENROLLED, 2)
    room.frames(1.0, (1, SEAT), face=OTHER_FACE)  # at 45 s: three current votes (24, 44, 45 s), and the raw rule says OTHER
    person = room.person(1)
    assert (person.identity, person.identity_id, person.association.identity_id) == (IdentityState.KNOWN, OTHER, OTHER)
    assert room.changes[-2:] == [("retained", "none", "conflicting_identity"), ("none", "current", "established")]


def test_max_since_match_ends_it_and_says_so_even_with_no_contradicting_face_at_all() -> None:
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "identity": {"retention_max_since_match_s": 20.0}})
    room = known(Room(retention=config.identity.retention_policy(), config=config))
    room.frames(19.9, (1, SEAT), face=())  # just inside the bound
    association = room.person(1).association
    assert association.state is AssociationState.RETAINED and association.max_since_match_ms == 20_000
    room.frames(0.2, (1, SEAT), face=())  # the bound passes: negative or missing evidence is never ignored for ever
    person = room.person(1)
    assert person.association is None and person.association_ended.reason is EndReason.MAX_SINCE_MATCH
    assert room.changes[-1] == ("retained", "none", "max_since_match")
    assert person.identity is IdentityState.KNOWN and person.identity_basis is Basis.RETAINED  # raw: its own 30 s TTL
    # a vote from before the end cannot re-establish it, whatever the raw rule still shows
    room.frames(1.0, (1, SEAT), face=MATCH)  # [old, new]: the older confirming vote is from before the end
    assert room.person(1).association is None
    room.frames(1.0, (1, SEAT), face=MATCH)  # both confirming votes after the end: established again
    assert room.person(1).association.state is AssociationState.CURRENT
    assert room.person(1).association_ended is None


def test_ambiguous_ownership_on_a_face_result_ends_it_and_later_matches_must_be_new() -> None:
    room = known(Room())
    other = (170.0, 100.0, 270.0, 400.0)  # overlaps the seat's box by 0.18: no overlap break, but a shared head region
    room.frames(0.5, (1, SEAT), (2, other))
    frame, persons = room.frame((1, SEAT), (2, other))
    assert len(persons) == 2
    shared = NormalizedBox(x1=175.0 / 640, y1=120.0 / 480, x2=195.0 / 640, y2=160.0 / 480)  # in both head regions
    output = room.apply(frame, persons, MATCH, shared)
    assert {p[1] for p in output.face_result.persons} == {Ownership.AMBIGUOUS.value}
    person = room.person(1)
    assert person.association is None and person.association_ended.reason is EndReason.AMBIGUOUS_OWNERSHIP
    assert room.breaks == 0  # not a continuity break: ownership of this face result was ambiguous
    assert room.changes[-1] == ("current", "none", "ambiguous_ownership")
    # the raw state still says known (votes not yet expired); the association stays gone until two new matches
    assert room.person(1).identity is IdentityState.KNOWN
    room.frames(1.0, (1, SEAT), (2, other), face=MATCH, faces_at=HEAD)
    assert room.person(1).association is None  # one new vote: [old, new] has an old confirming vote
    room.frames(1.0, (1, SEAT), (2, other), face=MATCH, faces_at=HEAD)
    assert room.person(1).association.state is AssociationState.CURRENT
    assert room.person(2).association is None and room.person(2).identity_id is None  # never given to the other person


def test_a_person_the_window_stage_withheld_still_loses_it_when_a_face_could_have_been_theirs() -> None:
    room = known(Room())
    other = (170.0, 100.0, 270.0, 400.0)  # overlaps the seat's box by 0.18: no overlap break, but a shared head region
    room.frames(0.5, (1, SEAT), (2, other))
    frame, persons = room.frame((1, SEAT), (2, other))
    shared = NormalizedBox(x1=175.0 / 640, y1=120.0 / 480, x2=195.0 / 640, y2=160.0 / 480)  # in both head regions
    withheld = next(p.key for p in persons if p.track_id == 1)
    windows = WindowNumbers(1, ((640, 640),), (1.0,), 2, 1, 0, (withheld,), 1, 0, 0, 0)  # the window budget left track 1 out
    output = room.apply(frame, persons, MATCH, shared, windows=windows)
    assert {p[:2] for p in output.face_result.persons} == {(1, Ownership.NOT_ANALYSED.value),  # the raw labels as ever:
                                                           (2, Ownership.AMBIGUOUS.value)}  # no vote for anyone
    person = room.person(1)
    assert person.association is None and person.association_ended.reason is EndReason.AMBIGUOUS_OWNERSHIP
    assert room.changes[-1] == ("current", "none", "ambiguous_ownership") and room.breaks == 0
    assert room.core.identity_counters["association_ended_ambiguous_ownership"] == 1
    room = known(Room())  # only ambiguity ends it: this person's own face, withheld from the result, is degraded evidence
    frame, persons = room.frame((1, SEAT))
    room.apply(frame, persons, MATCH, HEAD, windows=WindowNumbers(1, ((640, 640),), (1.0,), 1, 1, 0, (persons[0].key,), 1, 0, 0, 0))
    association = room.person(1).association
    assert (association.evidence, association.contradictions) == (Evidence.NOT_ANALYSED, 0)
    assert room.person(1).association_ended is None


def test_a_continuity_break_ends_it_and_a_late_result_from_before_the_break_cannot_refresh_or_re_establish_it() -> None:
    room = known(Room())
    room.frames(5.0, (1, SEAT), face=())
    assert room.person(1).association.state is AssociationState.RETAINED
    room.hold((1, SEAT))  # a frame the face worker takes; its result arrives after the break below
    room.frames(0.1, (1, (450.0, 100.0, 550.0, 400.0)))  # the same number on a box nowhere near: a jump
    assert room.breaks == 1 and room.changes[-1] == ("retained", "none", "continuity_jump")
    person = room.person(1)
    assert person.association is None and person.association_ended.reason is EndReason.CONTINUITY_JUMP
    assert person.identity_reason == NOT_CARRIED[Cause.JUMP]  # the raw path's own words, unchanged
    frame, persons = room.held.pop()
    output = room.apply(frame, persons, MATCH)  # a match, but from before the break
    assert output.face_result.persons[0][4] == BEFORE_BREAK
    assert room.person(1).association is None and room.person(1).association_ended.reason is EndReason.CONTINUITY_JUMP
    room.frames(1.0, (1, (450.0, 100.0, 550.0, 400.0)), face=MATCH, faces_at=NormalizedBox(x1=0.72, y1=0.22, x2=0.82, y2=0.35))
    assert room.person(1).association is None  # one new match: not established
    room.frames(1.0, (1, (450.0, 100.0, 550.0, 400.0)), face=MATCH, faces_at=NormalizedBox(x1=0.72, y1=0.22, x2=0.82, y2=0.35))
    person = room.person(1)
    assert person.association.state is AssociationState.CURRENT and person.association_ended is None


def test_a_detection_gap_beyond_the_bound_ends_it_and_a_brief_miss_changes_nothing() -> None:
    room = known(Room())
    room.frames(0.3)  # three frames without a detection: within the gap bound
    room.frames(1.0, (1, SEAT))
    assert room.person(1).association is not None and room.breaks == 0
    room.frames(0.8)  # the chair hides the person for 0.8 s: the track stays current, its continuity lapses
    room.frames(0.2, (1, SEAT))
    assert room.breaks == 1 and room.changes[-1][2] == "continuity_gap"
    assert room.person(1).association is None
    assert room.person(1).association_ended.reason is EndReason.CONTINUITY_GAP


def test_a_late_result_from_before_the_establishing_match_changes_nothing() -> None:
    room = Room()
    room.hold((1, SEAT))  # a frame analysed slowly
    room.frames(2.0, (1, SEAT), face=MATCH)  # two later matches establish it
    frame, persons = room.held.pop()
    assert room.person(1).association.state is AssociationState.CURRENT
    room.apply(frame, persons, ())  # its result arrives last: no face, from before the matches it stands on
    association = room.person(1).association
    assert association.evidence is Evidence.NONE and association.evidence_age_ms is None
    assert room.core.identity_counters["association_evidence_late"] == 1


def test_two_people_crossing_lose_their_associations_and_neither_inherits() -> None:
    room = known(Room())
    room.frames(1.0, (1, SEAT), (2, (400.0, 100.0, 500.0, 400.0)))  # someone else, unrecognized, on track 2
    for step in range(1, 11):  # they walk towards each other and cross; the tracker swaps their numbers
        room.frames(0.1, (1, (100.0 + 15 * step, 100.0, 200.0 + 15 * step, 400.0)),
                    (2, (400.0 - 15 * step, 100.0, 500.0 - 15 * step, 400.0)))
    for step in range(1, 11):
        room.frames(0.1, (1, (250.0 + 15 * step, 100.0, 350.0 + 15 * step, 400.0)),
                    (2, (250.0 - 15 * step, 100.0, 350.0 - 15 * step, 400.0)))
    assert room.changes[-1] == ("current", "none", "continuity_overlap")  # at the first frame the boxes overlap
    for track in (1, 2):
        person = room.person(track)
        assert person.association is None and person.identity_id is None
    assert room.person(1).association_ended.reason is EndReason.CONTINUITY_OVERLAP
    assert room.person(2).association_ended is None  # it never had one
    assert not any(r["track_id"] == 2 and r["to"] != "none" for r in room.records)


def test_a_track_that_expires_and_returns_with_its_number_is_a_new_track_and_keeps_nothing() -> None:
    room = known(Room())
    room.frames(1.5)  # gone beyond the track expiry
    assert room.person(1) is None and room.changes[-1] == ("current", "none", "track_not_current")
    room.frames(1.0, (1, SEAT), face=MATCH)  # the same number again: nothing carried, one match is not enough
    person = room.person(1)
    assert person.association is None and person.association_ended is None


def test_a_stalled_loop_that_misses_the_expiry_still_ends_it_as_a_new_life() -> None:
    room = known(Room())
    room.clock.advance(1.5)  # nothing runs for 1.5 s: past the 1 s track expiry, before any step saw it go
    room.frames(0.1, (1, SEAT))
    assert room.changes[-1] == ("current", "none", "new_track_life")
    assert room.person(1).association is None and room.person(1).association_ended is None
    assert room.core.identity_counters["association_ended_new_track_life"] == 1


def test_a_late_result_from_a_tracks_earlier_life_neither_refreshes_nor_establishes() -> None:
    room = known(Room())
    room.hold((1, SEAT))
    room.frames(1.5)  # the track expires
    room.frames(1.0, (1, SEAT))  # and comes back with its number: a new life
    assert room.person(1).association is None
    frame, persons = room.held.pop()
    room.clock.advance(0.5)  # the old frame is still within the face result's age limit
    output = room.apply(frame, persons, MATCH)  # a match, but from the track's earlier life
    assert output.face_result.persons[0][4] == BEFORE_EXPIRY
    assert room.person(1).association is None and room.person(1).association_ended is None
    assert room.core.identity_counters["persons_before_expiry"] == 1


def test_a_stream_reconnect_ends_it_and_nothing_carries_to_the_new_epoch() -> None:
    room = known(Room())
    room.stream = room.stamper.connect()  # a new stream epoch; the old track numbers mean nothing
    room.frames(1.0, (1, SEAT), face=MATCH)
    assert room.changes[-1] == ("current", "none", "stream_epoch_change")
    person = room.person(1)
    assert person.stream_epoch == 2 and person.association is None and person.association_ended is None
    assert person.identity is IdentityState.UNRESOLVED  # one match in the new epoch is not enough


def test_a_real_reconnect_withdraws_the_epoch_first_and_is_recorded_as_video_not_live() -> None:
    room = known(Room())
    room._keep(room.core.tick(None))  # the capture withdraws the epoch before a new one exists: connected is None
    assert room.changes[-1] == ("current", "none", "video_not_live") and not room.state.people
    counters = room.core.identity_counters
    assert counters["association_ended_video_not_live"] == 1 and "association_ended_track_not_current" not in counters
    room.stream = room.stamper.connect()  # the new epoch finds nothing left to end, and nothing carries to it
    room.frames(1.0, (1, SEAT), face=MATCH)
    person = room.person(1)
    assert person.stream_epoch == 2 and person.association is None and person.association_ended is None
    assert [change for change in room.changes if change[1] == "none"] == [("current", "none", "video_not_live")]
    assert "association_ended_stream_epoch_change" not in room.core.identity_counters


def test_a_stalled_camera_is_video_not_live_and_a_track_that_expires_on_live_video_is_not() -> None:
    room = known(Room())
    room.clock.advance(3.0)  # no frame for 3 s: past stale_after_s, so nothing is live though the stream is connected
    room._keep(room.core.tick(room.stream))
    assert room.changes[-1] == ("current", "none", "video_not_live") and not room.state.people
    room = known(Room())
    room.clock.advance(1.5)  # past the 1 s track expiry, inside stale_after_s: the video is live and the track ended
    room._keep(room.core.tick(room.stream))
    assert room.changes[-1] == ("current", "none", "track_not_current")


def test_turning_recognition_off_ends_it_and_on_again_starts_with_nothing() -> None:
    room = known(Room())
    room.frames(4.0, (1, SEAT), face=())  # beyond the fresh window: retained
    room.core.set_face_recognition(Capability.DISABLED)
    room.frames(0.2, (1, SEAT))
    person = room.person(1)
    assert person.association is None and person.association_ended is None
    assert person.identity is IdentityState.UNRESOLVED
    assert room.changes[-1] == ("retained", "none", "recognition_disabled")
    room.core.set_face_recognition(Capability.AVAILABLE)
    room.frames(0.2, (1, SEAT))
    assert room.person(1).association is None and room.person(1).association_ended is None
    room.frames(1.0, (1, SEAT), face=MATCH)
    assert room.person(1).association is None  # one match after the restart proves nothing
    room.frames(1.0, (1, SEAT), face=MATCH)
    assert room.person(1).association.state is AssociationState.CURRENT
    assert room.core.identity_counters["association_ended_recognition_disabled"] == 1


# ---------------------------------------------------------------- bounds, records, the rule without the mode


def test_nothing_is_kept_for_tracks_that_are_gone() -> None:
    room = known(Room())
    room.frames(5.0, (1, SEAT), face=())
    assert room.core.diagnostics()["identity_associations"] == 1
    room.frames(1.5)  # gone beyond the track expiry: association and end forgotten alike
    assert room.core.diagnostics()["identity_associations"] == 0 and not room.state.people
    room = known(Room(), track=7)
    room.frames(0.1, (7, (450.0, 100.0, 550.0, 400.0)))  # a jump: its end is kept while the track lives...
    assert room.core.diagnostics()["identity_associations"] == 1
    room.frames(1.5)  # ... and no longer
    assert room.core.diagnostics()["identity_associations"] == 0


def test_the_records_and_counters_are_fixed_labels_and_numbers_only() -> None:
    room = known(Room())
    room.frames(1.0, (1, SEAT), face=NOBODY)
    room.frames(0.1, (1, (450.0, 100.0, 550.0, 400.0)))  # a jump ends it
    for record in room.records:
        assert set(record) == {"identity", "schema", "stream_epoch", "track_id", "from", "to", "identity_id", "reason",
                               "last_match_age_ms", "contradictions", "mono_ns"}
        assert record["identity"] == "association" and record["reason"] in CHANGE_REASONS
        assert record["from"] in ("none", "current", "retained") and record["to"] in ("none", "current", "retained")
        assert record["identity_id"] in (ENROLLED, OTHER)
    assert room.records[-1]["last_match_age_ms"] == 1100 and room.records[-1]["to"] == "none"
    counters = room.core.identity_counters
    assert counters["associations_established"] == 1 and counters["association_ended_continuity_jump"] == 1
    assert counters["breaks_jump"] == 1


def test_a_sustained_overlap_is_one_break_record_per_track_and_still_breaks_on_every_frame() -> None:
    near, beside = (100.0, 100.0, 300.0, 460.0), (160.0, 100.0, 360.0, 460.0)  # person boxes overlapping by 0.54
    away = (450.0, 100.0, 630.0, 460.0)
    room = Room()
    for _ in range(150):
        room.frame((1, near), (2, beside))
    assert sorted(room.break_causes) == [(1, "overlap"), (2, "overlap")]  # one record per track for the whole stretch
    assert room.core.identity_counters["breaks_overlap"] == 300  # every frame still counts...
    frame, persons = room.frame((1, near), (2, beside))
    room.frame((1, near), (2, beside))
    output = room.apply(frame, persons, ())
    assert {p[4] for p in output.face_result.persons} == {BEFORE_BREAK}  # ... and still breaks: a later frame did
    room.frame((1, near), (2, away))  # track 2 leaves: a jump, another cause, so its own record
    room.frames(0.5, (1, near), (2, away))
    room.frame((1, near), (2, beside))  # and it comes back: a new stretch for both, a jump for track 2
    room.frame((1, near), (2, beside))  # then track 2's overlap: its cause changed; track 1's continues, no new record
    assert sorted(room.break_causes[2:]) == [(1, "overlap"), (2, "jump"), (2, "jump"), (2, "overlap")]


def test_a_row_after_an_end_says_there_is_no_current_vote_and_never_that_the_track_had_no_result() -> None:
    from sentinel.demo_runtime import DemoRuntime
    from sentinel.status_page import _person

    page_policy = {"confirmations": 2, "vote_ttl_s": 30.0, "fresh_window_s": 3.0}

    def row_and_html(room: Room) -> tuple[dict, str]:
        (row,) = DemoRuntime._face_rows(room.state, True, True)["faces"]
        return row, _person(row, room.state.published_utc.isoformat(), page_policy)

    room = known(Room())
    room.frames(0.1, (1, (450.0, 100.0, 550.0, 400.0)))  # a jump: the votes are dropped, so there is no vote age
    row, html = row_and_html(room)
    assert row["vote_age_ms"] is None and row["association_ended"]["reason"] == "continuity_jump"
    assert "No face result on this track yet." not in html
    assert "No current face result on this track since that end." in html
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "identity": {"retention_max_since_match_s": 40.0}})
    room = known(Room(retention=config.identity.retention_policy(), config=config))
    room.frames(41.0, (1, SEAT), face=())  # the raw votes expired at 30 s; the bound ended it at 40 s
    row, html = row_and_html(room)
    assert row["vote_age_ms"] is None and row["association_ended"]["reason"] == "max_since_match"
    assert "No face result on this track yet." not in html
    assert "No current face result on this track since that end." in html


def _script(room: Room, faces: tuple) -> list[tuple]:
    """The raw fields after each step of one scripted scene."""
    out = []
    for second, face in enumerate(faces):
        room.frames(1.0, (1, SEAT), face=face)
        person = room.person(1)
        out.append((second, person.identity, person.identity_id, person.identity_reason, person.identity_basis,
                    person.identity_vote_age_ms))
    return out


def test_with_the_mode_off_the_raw_path_is_exactly_as_it_was_and_with_it_the_raw_fields_do_not_change() -> None:
    faces = (MATCH, MATCH, (), NOBODY, NOBODY, OTHER_FACE, OTHER_FACE, MATCH, MATCH, ())
    plain, retaining = Room(retention=None), Room()
    assert _script(plain, faces) == _script(retaining, faces)
    assert plain.transitions == retaining.transitions  # the raw transition records, byte for byte
    assert [t["to"] for t in plain.transitions] == ["known", "uncertain", "unknown", "uncertain", "known", "uncertain",
                                                    "known"]
    assert plain.records == [] and plain.state.people[0].association is None
    assert plain.state.people[0].identity_followed_ms is None and plain.core.continuity is None
    assert "identity_associations" not in plain.core.diagnostics()
    assert not any(k.startswith("association") for k in plain.core.identity_counters)
    assert retaining.records != []


def test_two_continuity_rules_at_once_are_refused() -> None:
    clock = FakeClock()
    with pytest.raises(ValueError, match="separate rules"):
        EdgeCore(CONFIG, clock, None, retention=POLICY, continuity=CONFIG.identity.continuity_policy())


def test_the_policy_is_validated_and_reported_as_numbers() -> None:
    assert POLICY.numbers() == {"max_since_match_s": 300.0, "contradiction_votes": 10, "contradiction_span_s": 10.0,
                                "min_face_width_fraction": 0.03, "conflict_votes": 2, "conflict_window_s": 30.0}
    with pytest.raises(ValueError):
        RetentionPolicy(max_since_match_ns=601 * NS_PER_SECOND)  # beyond identity.continuity.MAX_RETENTION_NS
    with pytest.raises(ValueError):
        RetentionPolicy(contradiction_votes=0)
    with pytest.raises(ValueError):
        RetentionPolicy(min_face_width_fraction=1.0)
    with pytest.raises(ValueError):
        RetentionPolicy(conflict_window_ns=0)
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "identity": {"confirmations": 3, "vote_ttl_s": 45.0}})
    policy = config.identity.retention_policy()  # a conflict is what the raw rule needs to report another identity
    assert (policy.conflict_votes, policy.conflict_window_ns) == (3, 45 * NS_PER_SECOND)
    assert reported_numbers(POLICY, CONFIG.identity.continuity_policy()) == {
        **POLICY.numbers(), "max_gap_s": 0.5, "overlap_iou": 0.3, "min_step_iou": 0.4}  # not retention_s: it is not used
    with pytest.raises(ConfigError, match="retention_max_since_match_s must exceed"):
        parse_config({"config_version": 1, "camera": {"id": "cam-1"},
                      "identity": {"retention_max_since_match_s": 2.0}})
    with pytest.raises(ConfigError, match="less than or equal to 600"):
        parse_config({"config_version": 1, "camera": {"id": "cam-1"},
                      "identity": {"retention_max_since_match_s": 601.0}})
    with pytest.raises(ConfigError, match="retention_contradiction_span_s must not exceed"):
        parse_config({"config_version": 1, "camera": {"id": "cam-1"},
                      "identity": {"retention_max_since_match_s": 20.0, "retention_contradiction_span_s": 21.0}})


def test_a_ledger_on_its_own_never_retains_without_the_resolvers_evidence() -> None:
    ledger = RetentionLedger(POLICY)
    assert ledger.tracked == 0 and ledger.policy is POLICY
    ledger.observe(VoteRecord(key=_key(), ownership=Ownership.ASSIGNED, vote=ENROLLED, label="match"), 1, 0.1, 2)
    assert ledger.tracked == 0  # no association to update: a match alone starts nothing here


def _key():
    from sentinel.contracts import TrackKey

    return TrackKey(camera_id="cam-1", boot_id="b", run_id="r", stream_epoch=1, track_id=1)


# ---------------------------------------------------------------- readers: a retention run is never an F2 attempt


@pytest.fixture
def runners(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks/runner"))
    import face_check
    import face_evidence

    return face_check, face_evidence


MARKER = reported_numbers(POLICY, CONFIG.identity.continuity_policy())  # the starting line's real marker


def test_every_identity_reading_refuses_a_retention_runs_starting_line(runners, tmp_path) -> None:
    face_check, face_evidence = runners
    path = tmp_path / "run.jsonl"
    association = {"identity": "association", "schema": 3, "stream_epoch": 1, "track_id": 1, "from": "none", "to": "current",
                   "identity_id": ENROLLED, "reason": "established", "last_match_age_ms": 0, "contradictions": 0,
                   "mono_ns": 2_000_000_000}
    path.write_text("".join(json.dumps(line) + "\n" for line in (
        {"run": "starting", "startup": {"face": {"validation_run": True, "identities_enrolled": 1},
                                        "identity_retention": MARKER}}, association)))
    _, problems, starting, _ = face_check.read_records(path)
    assert starting["identity_retention"] == MARKER
    assert any("identity_retention" in problem for problem in problems), problems
    assert any("unknown identity record 'association'" in problem for problem in problems), problems
    assert face_evidence.read(path, "f2").nonconforming == []  # its fields are declared; the reading refuses the run


def test_the_vocabularies_are_the_codes(runners) -> None:
    _, fe = runners
    from sentinel.identity import retention

    assert fe.RETENTION_POLICY == tuple(RetentionPolicy().numbers())
    assert fe.RETENTION_MONITOR == MONITOR_NUMBERS
    assert fe.RETENTION_REPORTED == tuple(reported_numbers(POLICY, CONFIG.identity.continuity_policy()))
    assert set(fe.ASSOCIATION_EVIDENCE) == {e.value for e in retention.Evidence}
    assert set(fe.ASSOCIATION_ENDS) == {r.value for r in retention.EndReason}
    from sentinel.status_page import ASSOCIATION_ENDED

    assert set(ASSOCIATION_ENDED) == {r.value for r in retention.EndReason}  # every end has its plain words on the page
    assert set(fe.ASSOCIATION_REASONS) == set(retention.CHANGE_REASONS)
    assert set(fe.ASSOCIATION_STATES) == set(retention.RECORD_STATES)
    assert all(fe.TRANSITION_REASON.fit(reason)[0] for reason in retention.CHANGE_REASONS)
    assert all(f"association_ended_{reason.value}" in fe.FACE_COUNTERS.values for reason in retention.EndReason)
    assert {"associations_established", "association_evidence_late"} <= fe.FACE_COUNTERS.values
