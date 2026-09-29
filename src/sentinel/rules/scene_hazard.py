"""Fire/smoke candidates from scene reports (guide chapters 1, 9 and 22; audit finding 4).

Scene checks run with or without people, so a hazard can be a candidate in an
empty room. The only fire/smoke signal today is the VLM's scene report. Two
reports from the same model are temporal confirmation, not independent
evidence, so a candidate built from them is capped at WARNING: Sentinel does
not raise a critical fire alarm, or present itself as fire protection, without
a separately evaluated primary fire signal.

Confirmation needs ``confirmations`` distinct positive reports in a row, from
one stream epoch, with no more than ``max_gap_ns`` between their source
frames. Anything else in between resets it: a negative report, a failed or
skipped check, an outage or stall (no current report), an epoch change or a
gap. A candidate is emitted once per episode.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import NonNegativeInt

from ..contracts import Contract, Evidence, FrameKey, Identifier, UtcDatetime
from ..scene.state import SceneView

FIRE_SMOKE_CANDIDATE = "scene.fire_smoke_candidate"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


VLM_ONLY_MAX_SEVERITY = Severity.WARNING


class HazardCandidate(Contract):
    """A candidate incident observation; the incident service decides what to open."""

    kind: Literal["scene.fire_smoke_candidate"] = FIRE_SMOKE_CANDIDATE
    severity: Severity
    primary_signal: bool  # an independent, evaluated fire detector agreed
    evidence_ids: tuple[Identifier, ...]
    first_source: FrameKey
    last_source: FrameKey
    observed_mono_ns: NonNegativeInt  # source time of the confirming report
    observed_utc: UtcDatetime
    people_count: NonNegativeInt | None  # None when occupancy is unknown
    reason: str


class SceneHazardRule:
    def __init__(self, *, confirmations: int, max_gap_ns: int) -> None:
        if confirmations < 2:
            raise ValueError("a VLM-only candidate needs at least two reports")
        if max_gap_ns <= 0:
            raise ValueError("max_gap_ns must be positive")
        self._confirmations = confirmations
        self._max_gap_ns = max_gap_ns
        self._streak: list[Evidence] = []
        self._emitted = False

    def evaluate(self, view: SceneView, *, people_count: int | None) -> HazardCandidate | None:
        """Call on every runtime step with the current scene view."""
        evidence = view.evidence
        report = view.report
        if evidence is None or report is None or not report.fire_or_smoke:
            self._reset()
            return None
        if self._streak:
            last = self._streak[-1]
            if evidence.evidence_id == last.evidence_id:
                return None
            if (
                evidence.source.stream != last.source.stream
                or evidence.observed_mono_ns - last.observed_mono_ns > self._max_gap_ns
                or evidence.observed_mono_ns <= last.observed_mono_ns
            ):
                self._reset()
        self._streak.append(evidence)
        del self._streak[: -self._confirmations]  # keep the bounded window
        if len(self._streak) < self._confirmations or self._emitted:
            return None
        self._emitted = True
        first, last = self._streak[0], self._streak[-1]
        return HazardCandidate(
            severity=VLM_ONLY_MAX_SEVERITY,
            primary_signal=False,
            evidence_ids=tuple(e.evidence_id for e in self._streak),
            first_source=first.source,
            last_source=last.source,
            observed_mono_ns=last.observed_mono_ns,
            observed_utc=last.observed_utc,
            people_count=people_count,
            reason=(
                f"{self._confirmations} consecutive scene reports of fire or smoke from one "
                "model; no independent fire detector, so this is a candidate, not a fire alarm"
            ),
        )

    def _reset(self) -> None:
        self._streak.clear()
        self._emitted = False
