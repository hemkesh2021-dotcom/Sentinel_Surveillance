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

from collections.abc import Sequence
from dataclasses import dataclass

from .config import SentinelConfig
from .contracts import Evidence, FrameRef, StreamIdentity, TrackObservation, TrackStatus
from .identity.association import AssociationGeometry, FaceObservation, associate
from .identity.matching import Enrollment
from .identity.state import IdentityPolicy, IdentityResolver, IdentityState
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


@dataclass(frozen=True)
class CoreOutput:
    state: LiveState
    evidence: tuple[RoutedEvidence, ...] = ()  # new evidence, with where it may go
    candidates: tuple[HazardCandidate, ...] = ()
    zones: tuple[ZoneObservation, ...] = ()  # zone rule observations from this step


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
    ) -> None:
        self._config = config
        self._clock = clock
        self._freshness = FreshnessMonitor(config.freshness, clock)
        self._tracks = TrackTable(config.freshness.track_expiry_ns)
        # No analyzer: scene analysis is disabled and core monitoring runs without it.
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
            ),
        )
        self._last_detected: FrameRef | None = None
        self._sequence = 0

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
        }

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
        occupancy, reason = self._occupancy(freshness, tracks, now)
        people_count = None if occupancy is Occupancy.UNKNOWN else _confirmed(tracks)
        candidate = self._hazard.evaluate(view, people_count=people_count)
        # Zone rules read tracks only: no identity, no scene verdict (guide ch. 9).
        zones = self._zones.evaluate(tracks, live, now)
        state = self._publish(freshness, tracks, occupancy, reason, view, now)
        return CoreOutput(
            state=state,
            evidence=routed,
            candidates=(candidate,) if candidate is not None else (),
            zones=tuple(zones),
        )

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
        occupancy: Occupancy,
        occupancy_reason: str,
        view: SceneView,
        now: MonoInstant,
    ) -> LiveState:
        self._sequence += 1
        if self._lane is None:
            scene, scene_reason = SceneStatus.NO_CURRENT_RESULT, "scene analysis disabled"
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
            scene_analysis=Capability.DISABLED if self._lane is None else Capability.AVAILABLE,
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
                    **self._identity_fields(t, now),
                )
                for t in tracks
            ),
            scene=scene,
            scene_reason=scene_reason[:300],
            scene_report=view.report,
            scene_evidence_id=view.evidence.evidence_id if view.evidence is not None else None,
        )

    def _identity_fields(self, track: TrackObservation, now: MonoInstant) -> dict[str, object]:
        if self._face_recognition is not Capability.AVAILABLE:
            reason = f"face recognition {self._face_recognition.value}"
            return {"identity": IdentityState.UNRESOLVED, "identity_id": None, "identity_reason": reason}
        resolved = self._identities.identity(track.key, now)
        return {
            "identity": resolved.state,
            "identity_id": resolved.identity_id,
            "identity_reason": resolved.reason,
        }


def _confirmed(tracks: tuple[TrackObservation, ...]) -> int:
    return sum(1 for t in tracks if t.status is TrackStatus.CONFIRMED)
