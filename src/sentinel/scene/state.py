"""Current scene state and evidence routing (guide chapters 6 and 27; audit finding 3).

Every consumer of scene evidence (status, overlays, rules, dashboard) reads it
through CurrentScene, which only ever holds evidence that was CURRENT when
offered and drops it the first time it is found not current. Once dropped it
never comes back, even if the same stream resumes within its TTL: the scene may
have changed during the gap. Failure evidence (timeout, error) replaces a
previous verdict, so an old answer never stands in for a new frame.

Late or non-current evidence still goes to the incident it was requested for,
and only to that incident.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..contracts import Applicability, Evidence, EvidenceStatus, StreamIdentity
from ..media.clock import MonoInstant
from .report import SceneReport

SCENE_STATE_KIND = "scene.report"  # the only evidence kind that can become scene state


@dataclass(frozen=True)
class Routing:
    applicability: Applicability
    updates_current: bool
    annotates: str | None  # the incident this evidence may enrich, whatever its age


def route_evidence(evidence: Evidence, live: StreamIdentity | None, now: MonoInstant) -> Routing:
    applicability = evidence.applicability(live, now)
    return Routing(
        applicability=applicability,
        updates_current=applicability is Applicability.CURRENT,
        annotates=evidence.incident_id,
    )


@dataclass(frozen=True)
class SceneView:
    """The scene as consumers may use it now."""

    evidence: Evidence | None  # current evidence, observed or failed; None if there is none
    reason: Applicability | None = None  # why the last evidence stopped being current

    @property
    def report(self) -> SceneReport | None:
        """The current report, or None when the scene is unknown right now."""
        if self.evidence is None or self.evidence.status is not EvidenceStatus.OBSERVED:
            return None
        return SceneReport.model_validate(self.evidence.value, strict=False)


class CurrentScene:
    """Holds at most one ``scene.report`` evidence: the newest that is still current."""

    def __init__(self) -> None:
        self._held: Evidence | None = None
        self._reason: Applicability | None = None

    def offer(self, evidence: Evidence, live: StreamIdentity | None, now: MonoInstant) -> Routing:
        routing = route_evidence(evidence, live, now)
        if evidence.kind != SCENE_STATE_KIND:
            routing = replace(routing, updates_current=False)
        self._refresh(live, now)
        if routing.updates_current and not self._holds_newer_than(evidence):
            self._held = evidence
            self._reason = None
        return routing

    def view(self, live: StreamIdentity | None, now: MonoInstant) -> SceneView:
        self._refresh(live, now)
        return SceneView(evidence=self._held, reason=self._reason)

    def _refresh(self, live: StreamIdentity | None, now: MonoInstant) -> None:
        if self._held is None:
            return
        applicability = self._held.applicability(live, now)
        if applicability is not Applicability.CURRENT:
            self._held = None
            self._reason = applicability

    def _holds_newer_than(self, evidence: Evidence) -> bool:
        # Both are current, so both belong to the live stream: frame_seq orders them.
        held = self._held
        return held is not None and held.source.frame_seq > evidence.source.frame_seq
