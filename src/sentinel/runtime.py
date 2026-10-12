"""Portable core of the edge runtime (guide chapters 4, 6, 8 and 9).

EdgeCore turns stamped frames with tracker output, and scene-worker outcomes,
into published live state, evidence and rule candidates. It holds no capture,
decoder or model library: adapters on the device (or a replay driver in tests)
feed it. Every input produces a full LiveState, including frames with nobody
in them and ticks while no frames arrive.

Only a FRESH stream is live for decisions (see media.health). Frames are used
once: repeated or out-of-order frames update freshness bookkeeping only.

Not thread-safe: call it from one loop thread.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .config import SentinelConfig
from .contracts import Evidence, FrameRef, StreamIdentity, TrackKey, TrackObservation, TrackStatus
from .identity.association import AssociationGeometry, FaceObservation, Ownership, associate
from .identity.comparison import ComparisonOutcome, Disposition
from .identity.matching import Enrollment
from .identity.state import UNKNOWN_VOTE, IdentityPolicy, IdentityResolver, IdentityState, TrackIdentity, VoteRecord
from .jobs import WorkerOutcome
from .live_state import Capability, LiveState, Occupancy, PersonState, SceneStatus
from .media.clock import Clock, MonoInstant
from .media.frames import FrameNovelty
from .media.health import FreshnessMonitor, VideoFreshness, VideoState
from .rules.scene_hazard import HazardCandidate, SceneHazardRule
from .rules.zones import ZoneObservation, ZoneRules
from .scene.lane import SceneAnalyzer, SceneLane, SceneLaneSettings
from .scene.state import CurrentScene, Routing, SceneView
from .tracking.tracks import TrackTable

NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class RoutedEvidence:
    evidence: Evidence
    routing: Routing


IDENTITY_RECORD_SCHEMA = 1  # the "identity" stdout records: tick, result and transition
# Schema 2 (D84): schema 1, plus each result person's diagnostic numbers (FaceDiagnostics).
IDENTITY_DIAGNOSTIC_SCHEMA = 2
# Schema 3 (D86), what face validation runs write: schema 2, plus each result's full-scale pass (full_scale_numbers).
IDENTITY_FULL_SCALE_SCHEMA = 3
# Schema 4 (D87), what face validation runs with --face-comparison write: schema 3, plus a "comparison" record for each
# person eligible for the full-scale comparison (ComparisonRecord).
IDENTITY_COMPARISON_SCHEMA = 4
# Schema 5 (D88), what face validation runs with the full-scale detection candidate write: schema 2, plus each
# result's own detection numbers (detection_numbers); no full-scale pass or comparison exists in those runs.
IDENTITY_CANDIDATE_SCHEMA = 5
# Schema 6 (D90), what face validation runs with the candidate write from D90 on: schema 5, plus on each result person
# the position of the prototype that gave the vote's similarity, the assigned face's geometry (face_geometry_numbers)
# and "shadow", the similarity to an experiment's separate working gallery (D91; null without one).
IDENTITY_GEOMETRY_SCHEMA = 6
DIAGNOSTIC_DECIMALS = 3
# D90: what a recorded raw detection may span, in frame widths and heights from the frame's top left corner; a value
# outside is recorded as unavailable (null), never clipped into range.
RAW_BOUNDS = (-1.0, 2.0)
GEOMETRY_PX_DECIMALS = 1
PRODUCTION_OUTCOMES_KEPT = 16  # recent face results' outcomes, by frame, for the comparison records (D87)
# DeepFace 0.0.99's YuNet threshold for production detections (legacy_deepface.YUNET_THRESHOLD; test-tied): the
# full-scale pass's faces are counted at the same score, so its counts compare like for like.
FULL_SCALE_COUNTED = 0.9


def _schema(schema: int) -> int:
    if schema not in (IDENTITY_RECORD_SCHEMA, IDENTITY_DIAGNOSTIC_SCHEMA, IDENTITY_FULL_SCALE_SCHEMA,
                      IDENTITY_COMPARISON_SCHEMA, IDENTITY_CANDIDATE_SCHEMA, IDENTITY_GEOMETRY_SCHEMA):
        raise ValueError(f"unknown identity record schema {schema!r}")
    return schema


def tick_record(frame: FrameRef, persons: Sequence[TrackObservation], tick: str,
                schema: int = IDENTITY_RECORD_SCHEMA) -> dict[str, object]:
    """One 1 Hz face tick: its frame, the track IDs on that frame, and what the worker did with it."""
    return {"identity": "tick", "schema": _schema(schema), "stream_epoch": frame.stream_epoch,
            "frame_seq": frame.frame_seq, "frame_mono_ns": frame.ingest_mono_ns, "tick": tick,
            "tracks": sorted(p.track_id for p in persons)}


@dataclass(frozen=True)
class FaceDiagnostics:
    """One person's numbers on an applied face result, for validation runs (schema 2): where the person's box was and
    how sure the detector was, the assigned face's size and score, and how close it came to an enrolled identity.
    Never an image, an embedding or a name."""

    box: tuple[float, float, float, float]  # the person's box, normalized (x1, y1, x2, y2)
    confidence: float | None  # the person detector's confidence
    face_px: tuple[int, int] | None  # the assigned face's width and height in the frame's pixels; None: no face
    face_score: float | None  # the face detector's confidence for it
    similarity: float | None  # the best cosine similarity to an enrolled identity, when one was computed
    prototype: int | None = None  # D90: that similarity's prototype position in the enrollment (an opaque index)
    geometry: dict[str, Any] | None = None  # D90: face_geometry_numbers for the assigned face (candidate runs)
    shadow: dict[str, Any] | None = None  # D91: similarity to an experiment's working gallery and its prototype

    def fields(self, schema: int = IDENTITY_DIAGNOSTIC_SCHEMA) -> dict[str, object]:
        def r(value: float | None) -> float | None:
            return None if value is None else round(value, DIAGNOSTIC_DECIMALS)

        out = {"box": [r(v) for v in self.box], "confidence": r(self.confidence),
               "face": None if self.face_px is None else {"w_px": self.face_px[0], "h_px": self.face_px[1],
                                                          "score": r(self.face_score)},
               "similarity": r(self.similarity)}
        if schema >= IDENTITY_GEOMETRY_SCHEMA:
            out.update(prototype=self.prototype, geometry=self.geometry, shadow=self.shadow)
        return out


@dataclass(frozen=True)
class FaceResultRecord:
    """One face result and what became of it; votes carry the time of the frame they came from."""

    stream_epoch: int
    frame_seq: int
    frame_mono_ns: int
    applied_mono_ns: int
    outcome: str  # applied, failed, rejected_not_live, rejected_boot, rejected_age, ignored_face_unavailable
    faces: int | None
    persons: tuple[tuple[int, str, str, str | None, str], ...]  # track ID, ownership, vote kind, identity ID, label
    processing_ms: float | None = None
    error: str | None = None
    diagnostics: tuple[FaceDiagnostics, ...] = ()  # one per entry of ``persons``; written with schema 2 and later
    full_scale: dict[str, Any] | None = None  # full_scale_numbers; written with schemas 3 and 4
    detection: dict[str, Any] | None = None  # detection_numbers; written with schemas 5 and 6

    def record(self, schema: int = IDENTITY_RECORD_SCHEMA) -> dict[str, object]:
        persons = [{"track_id": track, "ownership": ownership, "vote": vote, "identity_id": identity, "label": label}
                   for track, ownership, vote, identity, label in self.persons]
        if _schema(schema) >= IDENTITY_DIAGNOSTIC_SCHEMA:
            if len(self.diagnostics) != len(persons):
                raise ValueError("every person on a schema 2 or later result needs its diagnostics")
            persons = [person | numbers.fields(schema) for person, numbers in zip(persons, self.diagnostics)]
        record = {"identity": "result", "schema": schema, "stream_epoch": self.stream_epoch,
                  "frame_seq": self.frame_seq, "frame_mono_ns": self.frame_mono_ns,
                  "applied_mono_ns": self.applied_mono_ns, "outcome": self.outcome, "faces": self.faces,
                  "processing_ms": None if self.processing_ms is None else round(self.processing_ms, 1),
                  "error": self.error, "persons": persons}
        if schema in (IDENTITY_FULL_SCALE_SCHEMA, IDENTITY_COMPARISON_SCHEMA):
            record["full_scale"] = self.full_scale
        elif schema in (IDENTITY_CANDIDATE_SCHEMA, IDENTITY_GEOMETRY_SCHEMA):
            record["detection"] = self.detection
        return record


@dataclass(frozen=True)
class ComparisonRecord:
    """D87: one person eligible for the full-scale comparison on one analysed frame, as numbers and fixed labels: the
    kind (``full_scale_only`` or the ``both`` control), its disposition, the full-scale candidate's score and width, and,
    when completed, the best cosine similarity of its embedding to the enrollment (never a vote, an identity ID, an
    embedding or an image); ``ms`` when it was attempted; ``error`` (a fixed label) when it failed."""

    stream_epoch: int
    frame_seq: int
    frame_mono_ns: int
    track_id: int
    kind: str
    disposition: str
    score: float
    w_px: int
    similarity: float | None
    ms: float | None
    error: str | None

    def record(self, schema: int = IDENTITY_COMPARISON_SCHEMA) -> dict[str, object]:
        if _schema(schema) != IDENTITY_COMPARISON_SCHEMA:
            raise ValueError("comparison records are schema 4 only")
        return {"identity": "comparison", "schema": schema, "stream_epoch": self.stream_epoch,
                "frame_seq": self.frame_seq, "frame_mono_ns": self.frame_mono_ns, "track_id": self.track_id,
                "kind": self.kind, "disposition": self.disposition, "score": round(self.score, DIAGNOSTIC_DECIMALS),
                "w_px": self.w_px,
                "similarity": None if self.similarity is None else round(self.similarity, DIAGNOSTIC_DECIMALS),
                "ms": None if self.ms is None else round(self.ms, 1), "error": self.error}


@dataclass(frozen=True)
class IdentityTransition:
    """A change of one track's identity state, identity ID or basis: opaque IDs and numbers only."""

    stream_epoch: int
    track_id: int
    from_state: str
    to_state: str  # an IdentityState value, or "cleared" when the track is no longer current
    identity_id: str | None  # the opaque gallery ID, only for KNOWN
    basis: str | None
    last_vote_age_ms: int | None
    reason: str
    mono_ns: int

    def record(self, schema: int = IDENTITY_RECORD_SCHEMA) -> dict[str, object]:
        return {"identity": "transition", "schema": _schema(schema), "stream_epoch": self.stream_epoch,
                "track_id": self.track_id,
                "from": self.from_state, "to": self.to_state, "identity_id": self.identity_id, "basis": self.basis,
                "last_vote_age_ms": self.last_vote_age_ms, "reason": self.reason, "mono_ns": self.mono_ns}


@dataclass(frozen=True)
class CoreOutput:
    state: LiveState
    evidence: tuple[RoutedEvidence, ...] = ()  # new evidence, with where it may go
    candidates: tuple[HazardCandidate, ...] = ()
    zones: tuple[ZoneObservation, ...] = ()  # zone rule observations from this step
    identity_transitions: tuple[IdentityTransition, ...] = ()
    face_result: FaceResultRecord | None = None  # set by on_face_outcome


class EdgeCore:
    def __init__(
        self,
        config: SentinelConfig,
        clock: Clock,
        analyzer: SceneAnalyzer | None,
        *,
        detector: Capability = Capability.AVAILABLE,
        face_recognition: Capability = Capability.DISABLED,
        enrollment: Enrollment | None = None,
        scene_id_prefix: str | None = None,
        scene_problem: str | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._freshness = FreshnessMonitor(config.freshness, clock)
        self._tracks = TrackTable(config.freshness.track_expiry_ns)
        # No analyzer: scene analysis is disabled and core monitoring runs without it.
        # With ``scene_problem`` it was requested but is unavailable (failed to start or not admitted).
        self._lane = (
            None
            if analyzer is None
            else SceneLane(
                analyzer, clock, SceneLaneSettings.from_config(config.scene), id_prefix=scene_id_prefix
            )
        )
        self._scene = CurrentScene()
        self._hazard = SceneHazardRule(
            confirmations=config.hazard.confirmations, max_gap_ns=config.hazard.max_gap_ns
        )
        self._zones = ZoneRules(config.zones)
        self._scene_problem = None if analyzer is not None else scene_problem
        self._detector = detector
        self._face_recognition = face_recognition
        identity = config.identity
        self._geometry = AssociationGeometry(identity.head_fraction, identity.min_face_inside)
        self._identities = IdentityResolver(
            enrollment,
            IdentityPolicy(
                match_threshold=identity.match_threshold,
                margin=identity.margin,
                min_quality=identity.min_quality,
                confirmations=identity.confirmations,
                vote_ttl_ns=identity.vote_ttl_ns,
                fresh_window_ns=identity.face_result_max_age_ns,
            ),
        )
        self._face_result_max_age_ns = identity.face_result_max_age_ns
        self.face_counters: Counter[str] = Counter()  # face results: applied or rejected, and why
        self._reported: dict[TrackKey, tuple[str, str | None, str | None]] = {}
        self._last_detected: FrameRef | None = None
        self._sequence = 0
        # D87: recent face results' outcomes by frame (epoch, sequence), so that a comparison is completed only for a
        # frame whose production result was applied; and the comparisons' final dispositions (never identity counters).
        self._production_outcomes: OrderedDict[tuple[int, int], str] = OrderedDict()
        self.comparison_counters: Counter[str] = Counter()

    @property
    def lane(self) -> SceneLane | None:
        return self._lane

    def diagnostics(self) -> dict[str, int]:
        """Sizes of bounded internal state, for health reporting and tests."""
        return {
            "tracks": self._tracks.size,
            "identity_tracks": self._identities.tracked,
            "scene_jobs_in_flight": int(self._lane is not None and self._lane.in_flight is not None),
            "zone_episodes": self._zones.active_episodes,
            "identity_reported": len(self._reported),
        }

    @property
    def identity_counters(self) -> dict[str, int]:
        return dict(sorted((self.face_counters + self._identities.counters).items()))

    def set_detector(self, capability: Capability) -> None:
        self._detector = capability

    def on_frame(
        self,
        frame: FrameRef,
        persons: Sequence[TrackObservation],
        connected: StreamIdentity | None,
        faces: Sequence[FaceObservation] | None = None,
    ) -> CoreOutput:
        """A decoded frame and the tracker's output for it (empty when nobody is there).

        ``faces`` is the face stage's output for this frame, or None when it did
        not run on this frame (it need not run on every frame).
        """
        self._freshness.on_frame(frame)
        freshness = self._freshness.assess(connected)
        new: list[Evidence] = []
        novelty = self._tracks.update(frame, persons, freshness.live)
        if novelty in (FrameNovelty.NEW, FrameNovelty.NEW_EPOCH):
            if self._detector is Capability.AVAILABLE:
                self._last_detected = frame
            if faces is not None and self._face_recognition is Capability.AVAILABLE:
                self._identities.observe(persons, faces, associate(persons, faces, self._geometry))
            if self._lane is not None:
                new += self._lane.on_frame(frame, freshness.live)
        elif self._lane is not None:
            new += self._lane.poll(freshness.live)
        return self._finish(freshness, new)

    def on_face_outcome(
        self,
        frame: FrameRef,
        persons: Sequence[TrackObservation],
        faces: Sequence[FaceObservation] | None,
        connected: StreamIdentity | None,
        *,
        processing_ms: float | None = None,
        error: str | None = None,
        full_scale: Any = None,
        detection: Any = None,
        geometry: Sequence[Any] = (),
    ) -> CoreOutput:
        """A face result for an earlier sampled frame, with the tracker's people on that same frame.

        Its faces are associated with those people only, and its votes carry that frame's time. It is
        rejected, and counted, when face recognition is not available, the analysis failed, the frame's
        stream (camera, boot, run, epoch) is not the live one, or the frame is older than
        ``identity.face_result_max_age_s``. Votes for tracks that are no longer current are dropped by
        the next ``_finish``.
        """
        freshness = self._freshness.assess(connected)
        now = self._clock.mono()
        votes: list = []
        diagnostics: tuple[FaceDiagnostics, ...] = ()
        if self._face_recognition is not Capability.AVAILABLE:
            outcome = "ignored_face_unavailable"
        elif faces is None:
            outcome = "failed"
        elif connected is None or frame.stream != connected:
            outcome = "rejected_not_live"
        elif frame.boot_id != now.boot_id:
            outcome = "rejected_boot"
        elif now.ns - frame.ingest_mono_ns > self._face_result_max_age_ns:
            outcome = "rejected_age"
        else:
            outcome = "applied"
            association = associate(persons, faces, self._geometry)
            for ownership in association.face_ownership:
                self.face_counters[f"faces_{ownership.value}"] += 1
            votes = self._identities.observe(persons, faces, association)
            diagnostics = _diagnostics(frame, persons, faces, association.face_of, votes, geometry)
        self.face_counters[f"results_{outcome}"] += 1
        self._production_outcomes[(frame.stream_epoch, frame.frame_seq)] = outcome
        while len(self._production_outcomes) > PRODUCTION_OUTCOMES_KEPT:
            self._production_outcomes.popitem(last=False)
        record = FaceResultRecord(
            frame.stream_epoch, frame.frame_seq, frame.ingest_mono_ns, now.ns, outcome,
            None if faces is None else len(faces),
            tuple((v.key.track_id, v.ownership.value,
                   "none" if v.vote is None else "unknown" if v.vote == UNKNOWN_VOTE else "match",
                   v.vote if v.vote else None, v.label) for v in votes),
            processing_ms, error, diagnostics, full_scale_numbers(full_scale, frame, persons, self._geometry),
            detection_numbers(detection, frame))
        return dataclasses.replace(self._finish(freshness, []), face_result=record)

    def on_face_comparison(self, outcome: ComparisonOutcome) -> ComparisonRecord:
        """D87: one comparison outcome as a record. A completed comparison gets its best similarity to the enrollment
        (``IdentityResolver.diagnostic_similarity``, read only) when that frame's production result was applied and face
        recognition is available; otherwise it is ``production_not_applied`` and its embedding is discarded. Nothing
        here votes or changes identity state, tracks, counters or live state; the embedding is never kept."""
        disposition, similarity, error = outcome.disposition, None, outcome.error
        if disposition is Disposition.COMPLETED:
            applied = self._production_outcomes.get((outcome.frame.stream_epoch, outcome.frame.frame_seq)) == "applied"
            if not applied or self._face_recognition is not Capability.AVAILABLE or outcome.embedding is None:
                disposition = Disposition.PRODUCTION_NOT_APPLIED
            else:
                similarity = self._identities.diagnostic_similarity(outcome.embedding)
                if similarity is None:
                    disposition, error = Disposition.FAILED, "no_comparable_enrollment"
        self.comparison_counters[f"{outcome.kind.value}:{disposition.value}"] += 1
        frame = outcome.frame
        return ComparisonRecord(frame.stream_epoch, frame.frame_seq, frame.ingest_mono_ns, outcome.track_id,
                                outcome.kind.value, disposition.value, outcome.score, outcome.width_px, similarity,
                                outcome.ms, error)

    def on_scene_outcome(
        self, job_id: str, outcome: WorkerOutcome, connected: StreamIdentity | None
    ) -> CoreOutput:
        freshness = self._freshness.assess(connected)
        new: list[Evidence] = []
        if self._lane is not None:
            completed = self._lane.complete(job_id, outcome)
            new = [completed] if completed is not None else []
            new += self._lane.poll(freshness.live)
        return self._finish(freshness, new)

    def request_enrichment(
        self, frame: FrameRef, incident_id: str, connected: StreamIdentity | None
    ) -> CoreOutput:
        freshness = self._freshness.assess(connected)
        new: list[Evidence] = []
        if self._lane is not None:
            new = self._lane.request_enrichment(frame, incident_id)
            new += self._lane.poll(freshness.live)
        return self._finish(freshness, new)

    def tick(self, connected: StreamIdentity | None) -> CoreOutput:
        """Call regularly (e.g. every 250 ms) so staleness and deadlines show without frames."""
        freshness = self._freshness.assess(connected)
        new = self._lane.poll(freshness.live) if self._lane is not None else []
        return self._finish(freshness, new)

    def _finish(self, freshness: VideoFreshness, new: list[Evidence]) -> CoreOutput:
        now = self._clock.mono()
        live = freshness.live
        routed = tuple(RoutedEvidence(e, self._scene.offer(e, live, now)) for e in new)
        view = self._scene.view(live, now)
        tracks = self._tracks.current(live, now)
        self._identities.retain_only(t.key for t in tracks)
        identities = {t.key: self._identity(t, now) for t in tracks}
        transitions = self._transitions(tracks, identities, now)
        occupancy, reason = self._occupancy(freshness, tracks, now)
        people_count = None if occupancy is Occupancy.UNKNOWN else _confirmed(tracks)
        candidate = self._hazard.evaluate(view, people_count=people_count)
        # Zone rules read tracks only: no identity, no scene verdict (guide ch. 9).
        zones = self._zones.evaluate(tracks, live, now)
        state = self._publish(freshness, tracks, identities, occupancy, reason, view, now)
        return CoreOutput(
            state=state,
            evidence=routed,
            candidates=(candidate,) if candidate is not None else (),
            zones=tuple(zones),
            identity_transitions=transitions,
        )

    def _transitions(
        self, tracks: tuple[TrackObservation, ...], identities: dict[TrackKey, TrackIdentity], now: MonoInstant
    ) -> tuple[IdentityTransition, ...]:
        """Changes since the last step; only while face recognition is available (otherwise all is unresolved)."""
        if self._face_recognition is not Capability.AVAILABLE:
            return ()
        unresolved = (IdentityState.UNRESOLVED.value, None, None)
        changes = []
        current: dict[TrackKey, tuple[str, str | None, str | None]] = {}
        for track in tracks:
            resolved = identities[track.key]
            value = (resolved.state.value, resolved.identity_id, None if resolved.basis is None else resolved.basis.value)
            current[track.key] = value
            before = self._reported.get(track.key, unresolved)
            if value != before:
                changes.append(IdentityTransition(
                    track.key.stream_epoch, track.key.track_id, before[0], value[0], value[1], value[2],
                    None if resolved.last_vote_age_ns is None else resolved.last_vote_age_ns // NS_PER_MS,
                    resolved.reason, now.ns))
        for key, before in self._reported.items():
            if key not in current and before[0] != IdentityState.UNRESOLVED.value:
                changes.append(IdentityTransition(key.stream_epoch, key.track_id, before[0], "cleared", None, None,
                                                  None, "track no longer current", now.ns))
        self._reported = {key: value for key, value in current.items() if value != unresolved}
        return tuple(changes)

    def _occupancy(
        self,
        freshness: VideoFreshness,
        tracks: tuple[TrackObservation, ...],
        now: MonoInstant,
    ) -> tuple[Occupancy, str]:
        if freshness.state is not VideoState.FRESH:
            return Occupancy.UNKNOWN, f"no fresh video ({freshness.state.value})"
        if self._detector is not Capability.AVAILABLE:
            return Occupancy.UNKNOWN, f"person detector {self._detector.value}"
        # Tracks expire track_expiry after their last detection. Without a recent
        # detector frame their absence means "not looked", not "nobody there".
        last = self._last_detected
        if (
            last is None
            or last.stream != freshness.live
            or now.ns - last.ingest_mono_ns >= self._config.freshness.track_expiry_ns
        ):
            return Occupancy.UNKNOWN, "detector has not processed a recent frame"
        count = _confirmed(tracks)
        if count:
            return Occupancy.OCCUPIED, f"{count} confirmed person(s) on fresh video"
        return Occupancy.EMPTY, "nobody detected on fresh video"

    def _publish(
        self,
        freshness: VideoFreshness,
        tracks: tuple[TrackObservation, ...],
        identities: dict[TrackKey, TrackIdentity],
        occupancy: Occupancy,
        occupancy_reason: str,
        view: SceneView,
        now: MonoInstant,
    ) -> LiveState:
        self._sequence += 1
        if self._lane is None:
            scene = SceneStatus.NO_CURRENT_RESULT
            scene_reason = (
                "scene analysis disabled"
                if self._scene_problem is None
                else f"scene analysis unavailable: {self._scene_problem}"
            )
        elif view.report is not None:
            scene, scene_reason = SceneStatus.REPORTED, "current scene report"
        elif view.evidence is not None:
            scene = SceneStatus.UNKNOWN
            scene_reason = f"last scene check {view.evidence.status.value}: {view.evidence.reason}"
        else:
            scene = SceneStatus.NO_CURRENT_RESULT
            scene_reason = (
                "no scene result yet"
                if view.reason is None
                else f"last scene result not current ({view.reason.value})"
            )
        age = freshness.last_frame_age_ns
        return LiveState(
            camera_id=self._config.camera.id,
            sequence=self._sequence,
            boot_id=now.boot_id,
            published_mono_ns=now.ns,
            published_utc=self._clock.utc_now(),
            video=freshness.state,
            last_frame_age_ms=None if age is None else age // NS_PER_MS,
            detector=self._detector,
            face_recognition=self._face_recognition,
            scene_analysis=self._scene_capability(),
            occupancy=occupancy,
            occupancy_reason=occupancy_reason,
            people=tuple(
                PersonState(
                    track_id=t.track_id,
                    stream_epoch=t.frame.stream_epoch,
                    box=t.box,
                    status=t.status,
                    predicted=t.predicted,
                    last_measured_age_ms=(now.ns - t.last_measured_mono_ns) // NS_PER_MS,
                    **self._identity_fields(identities[t.key]),
                )
                for t in tracks
            ),
            scene=scene,
            scene_reason=scene_reason[:300],
            scene_report=view.report,
            scene_evidence_id=view.evidence.evidence_id if view.evidence is not None else None,
        )

    def _scene_capability(self) -> Capability:
        if self._lane is not None:
            return Capability.AVAILABLE
        return Capability.DISABLED if self._scene_problem is None else Capability.UNAVAILABLE

    def _identity(self, track: TrackObservation, now: MonoInstant) -> TrackIdentity:
        if self._face_recognition is not Capability.AVAILABLE:
            return TrackIdentity(IdentityState.UNRESOLVED, None, f"face recognition {self._face_recognition.value}")
        return self._identities.identity(track.key, now)

    @staticmethod
    def _identity_fields(resolved: TrackIdentity) -> dict[str, object]:
        age = resolved.last_vote_age_ns
        return {
            "identity": resolved.state,
            "identity_id": resolved.identity_id,
            "identity_reason": resolved.reason,
            "identity_basis": resolved.basis,
            "identity_vote_age_ms": None if age is None else age // NS_PER_MS,
        }


def _confirmed(tracks: tuple[TrackObservation, ...]) -> int:
    return sum(1 for t in tracks if t.status is TrackStatus.CONFIRMED)


def _diagnostics(
    frame: FrameRef,
    persons: Sequence[TrackObservation],
    faces: Sequence[FaceObservation],
    face_of: dict[TrackKey, int],
    votes: Sequence[VoteRecord],
    geometry: Sequence[Any] = (),
) -> tuple[FaceDiagnostics, ...]:
    """Each vote's person, in the votes' order: its box and confidence, its assigned face's size and score, and the
    vote's similarity. Only numbers are taken from a face; its embedding stays where it is. D90: the similarity's
    prototype position, and the assigned face's geometry when the face stage gave one per face (``geometry``)."""
    by_key = {person.key: person for person in persons}
    out = []
    for vote in votes:
        person = by_key[vote.key]
        index = face_of.get(vote.key)
        face = None if index is None else faces[index]
        out.append(FaceDiagnostics(
            (person.box.x1, person.box.y1, person.box.x2, person.box.y2), person.detector_confidence,
            None if face is None else (round((face.box.x2 - face.box.x1) * frame.native_width),
                                       round((face.box.y2 - face.box.y1) * frame.native_height)),
            None if face is None else face.quality, vote.similarity, vote.prototype,
            None if face is None else face_geometry_numbers(
                face, geometry[index] if len(geometry) == len(faces) else None, frame.native_width,
                frame.native_height)))
    return tuple(out)


def face_geometry_numbers(face: FaceObservation, geometry: Any, width: int, height: int) -> dict[str, Any]:
    """D90: one assigned face's geometry as numbers (validation records; never an image, embedding or name).

    - ``box``: the corrected box the face was reported and associated with (normalized; clipped to the frame);
    - ``raw_box``: the detector's own box, normalized and not clipped, so a value below 0 or above 1 reaches past
      that edge; null when the face stage gave none, or when a value is not finite or lies outside RAW_BOUNDS (it is
      then unavailable, never clipped into range);
    - ``overhang_px``: how far the raw box reaches past the left, top, right and bottom edges, in frame pixels (0 when
      it does not); null exactly when ``raw_box`` is. A raw box inside the frame does not establish that the face was:
      a detector may fit its box to the visible part of a face;
    - ``eyes``: the angle DeepFace's alignment computes from these landmarks, before its integer conversion
      (degrees(atan2(left.y - right.y, left.x - right.x)), DeepFace's left eye being YuNet's columns 6-7), and the
      distance between the eyes in frame pixels; null when either landmark is missing, not finite or out of bounds.
    """
    box = [round(v, DIAGNOSTIC_DECIMALS) for v in (face.box.x1, face.box.y1, face.box.x2, face.box.y2)]
    raw = getattr(geometry, "raw_box", None)
    raw_box = overhang = None
    if raw is not None and _within_bounds(raw[0], raw[1], width, height) \
            and _within_bounds(raw[0] + raw[2], raw[1] + raw[3], width, height) and raw[2] > 0 and raw[3] > 0:
        x, y, w, h = raw
        raw_box = [round(v, DIAGNOSTIC_DECIMALS) for v in (x / width, y / height, (x + w) / width, (y + h) / height)]
        overhang = [round(max(0.0, v), GEOMETRY_PX_DECIMALS) for v in (-x, -y, x + w - width, y + h - height)]
    right, left = getattr(geometry, "right_eye", None), getattr(geometry, "left_eye", None)
    eyes = None
    if right is not None and left is not None and _within_bounds(*right, width, height) \
            and _within_bounds(*left, width, height):
        eyes = {"roll_deg": round(math.degrees(math.atan2(left[1] - right[1], left[0] - right[0])),
                                  GEOMETRY_PX_DECIMALS),
                "spacing_px": round(math.hypot(left[0] - right[0], left[1] - right[1]), GEOMETRY_PX_DECIMALS)}
    return {"box": box, "raw_box": raw_box, "overhang_px": overhang, "eyes": eyes}


def _within_bounds(x: Any, y: Any, width: int, height: int) -> bool:
    low, high = RAW_BOUNDS
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(fx) and math.isfinite(fy) and low <= fx / width <= high and low <= fy / height <= high)


def full_scale_numbers(full_scale: Any, frame: FrameRef, persons: Sequence[TrackObservation],
                       geometry: AssociationGeometry) -> dict[str, Any] | None:
    """D86: the diagnostic full-scale pass of one result, as numbers: how many faces scored at least
    FULL_SCALE_COUNTED, how many of those the one-to-one association would give to a person on this frame, the best
    candidate's score and width in frame pixels, and the pass's time. Never a vote, a box or an image."""
    if full_scale is None:
        return None
    strong = [box for score, box in full_scale.candidates if score >= FULL_SCALE_COUNTED]
    assigned = 0
    if strong and persons:
        faces = [FaceObservation(frame=frame.key, box=box, quality=1.0) for box in strong]
        assigned = sum(o is Ownership.ASSIGNED for o in associate(persons, faces, geometry).face_ownership)
    best = full_scale.candidates[0] if full_scale.candidates else None
    return {"faces": len(strong), "assigned": assigned,
            "best_score": None if best is None else round(best[0], DIAGNOSTIC_DECIMALS),
            "best_w_px": None if best is None else round((best[1].x2 - best[1].x1) * frame.native_width),
            "ms": round(full_scale.ms, 1)}


def detection_numbers(detection: Any, frame: FrameRef) -> dict[str, Any] | None:
    """D88: the full-scale detection candidate's own numbers for one result: the best detection under the 0.9 threshold
    (its score and its width in frame pixels; never a face, a box or an image) and the times to detect and to embed
    (None when no face was kept)."""
    if detection is None:
        return None
    best = detection.best_below
    return {"best_below_score": None if best is None else round(best[0], DIAGNOSTIC_DECIMALS),
            "best_below_w_px": None if best is None else round((best[1].x2 - best[1].x1) * frame.native_width),
            "detect_ms": round(detection.detect_ms, 1),
            "embed_ms": None if detection.embed_ms is None else round(detection.embed_ms, 1)}
