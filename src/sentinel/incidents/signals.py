"""What the incident service accepts: one rule observation, as an IncidentSignal.

Rules produce their own observation contracts (zone observations, scene hazard
candidates). Each is converted here into one signal with a stable
``observation_id`` (the deduplication key of the incident transaction), an
``episode_id`` (one continuous condition), a correlation key (camera, kind,
zone) and its source time on the monotonic clock of its boot. The original
observation is kept as a bounded JSON payload for the incident's evidence.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Annotated

from pydantic import NonNegativeInt, StringConstraints

from ..contracts import Contract, EvidenceKind, Identifier, UtcDatetime
from ..rules.scene_hazard import HazardCandidate, Severity
from ..rules.zones import ZoneObservation, ZonePhase

MAX_PAYLOAD_CHARS = 8192
SEVERITY_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}
HAZARD_RULE_REVISION = "hazard-vlm-only-1"  # D17 semantics


class SignalPhase(str, Enum):
    ENTERED = "entered"  # the condition holds: create or join an incident
    ENDED = "ended"  # the condition stopped holding: note it on its incident


class IncidentSignal(Contract):
    observation_id: Identifier
    episode_id: Identifier
    camera_id: Identifier
    kind: EvidenceKind
    phase: SignalPhase
    zone_id: Identifier | None
    rule_revision: Identifier
    severity: Severity
    title: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    observed_utc: UtcDatetime
    boot_id: Identifier
    observed_mono_ns: NonNegativeInt
    reason: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    payload: Annotated[str, StringConstraints(max_length=MAX_PAYLOAD_CHARS)]

    @property
    def correlation_key(self) -> str:
        return f"{self.camera_id}|{self.kind}|{self.zone_id or '-'}"


def signal_from_zone(observation: ZoneObservation) -> IncidentSignal:
    track = observation.track
    what = "Person in restricted zone" if observation.kind == "zone.restricted_entry" else "Person dwelling in zone"
    return IncidentSignal(
        observation_id=observation.observation_id,
        episode_id=observation.episode_id,
        camera_id=track.camera_id,
        kind=observation.kind,
        phase=SignalPhase.ENTERED if observation.phase is ZonePhase.ENTERED else SignalPhase.ENDED,
        zone_id=observation.zone_id,
        rule_revision=observation.zone_revision,
        severity=observation.severity,
        title=f"{what} {observation.zone_id!r}",
        observed_utc=observation.observed_utc,
        boot_id=track.boot_id,
        observed_mono_ns=observation.last_mono_ns,
        reason=observation.reason,
        payload=observation.model_dump_json(),
    )


def signal_from_hazard(candidate: HazardCandidate) -> IncidentSignal:
    """A VLM-only fire/smoke candidate (D17): at most a warning, titled as unconfirmed."""
    digest = hashlib.sha256("|".join(candidate.evidence_ids).encode()).hexdigest()[:24]
    episode = f"hazard-{digest}"
    source = candidate.last_source
    return IncidentSignal(
        observation_id=f"{episode}.entered",
        episode_id=episode,
        camera_id=source.camera_id,
        kind=candidate.kind,
        phase=SignalPhase.ENTERED,
        zone_id=None,
        rule_revision=HAZARD_RULE_REVISION,
        severity=candidate.severity,
        title="Possible fire or smoke (scene model only, unconfirmed)",
        observed_utc=candidate.observed_utc,
        boot_id=source.boot_id,
        observed_mono_ns=candidate.observed_mono_ns,
        reason=candidate.reason[:300],
        payload=candidate.model_dump_json(),
    )
