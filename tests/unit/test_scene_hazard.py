from __future__ import annotations

from collections.abc import Callable

import pytest

from sentinel.contracts import Evidence, EvidenceStatus, FrameRef
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.rules.scene_hazard import SceneHazardRule, Severity
from sentinel.scene.state import SceneView

REPORT = {
    "persons_visible": 0,
    "fire_or_smoke": True,
    "threat": "high",
    "observations": [],
    "uncertainty": "medium",
    "summary": "smoke",
}


def evidence(frame: FrameRef, n: int, *, smoke: bool = True) -> SceneView:
    ev = Evidence.observed_on(
        frame,
        evidence_id=f"e-{n}",
        kind="scene.report",
        status=EvidenceStatus.OBSERVED,
        producer="scene-vlm",
        producer_revision="stub",
        ttl_ns=10 * NS_PER_SECOND,
        correlation_group="scene-vlm",
        reason="scene report",
        value={**REPORT, "fire_or_smoke": smoke},
    )
    return SceneView(evidence=ev)


def rule() -> SceneHazardRule:
    return SceneHazardRule(confirmations=2, max_gap_ns=12 * NS_PER_SECOND)


def test_the_same_report_seen_repeatedly_counts_once(next_frame: Callable[..., FrameRef]) -> None:
    hazard = rule()
    view = evidence(next_frame(), 1)
    assert [hazard.evaluate(view, people_count=0) for _ in range(5)] == [None] * 5
    candidate = hazard.evaluate(evidence(next_frame(dt=4), 2), people_count=0)
    assert candidate is not None and candidate.severity is Severity.WARNING
    assert candidate.evidence_ids == ("e-1", "e-2") and candidate.people_count == 0


def test_an_epoch_change_breaks_the_streak_even_without_an_empty_view(
    clock: FakeClock, stamper: FrameStamper, next_frame: Callable[..., FrameRef]
) -> None:
    hazard = rule()
    assert hazard.evaluate(evidence(next_frame(), 1), people_count=0) is None
    stamper.disconnect()
    stamper.connect()
    assert hazard.evaluate(evidence(next_frame(dt=1), 2), people_count=0) is None
    assert hazard.evaluate(evidence(next_frame(dt=4), 3), people_count=0) is not None


def test_a_gap_longer_than_max_gap_breaks_the_streak(next_frame: Callable[..., FrameRef]) -> None:
    hazard = rule()
    assert hazard.evaluate(evidence(next_frame(), 1), people_count=None) is None
    assert hazard.evaluate(evidence(next_frame(dt=12.001), 2), people_count=None) is None
    candidate = hazard.evaluate(evidence(next_frame(dt=12), 3), people_count=None)
    assert candidate is not None and candidate.people_count is None


def test_negative_or_failed_reports_reset(next_frame: Callable[..., FrameRef]) -> None:
    hazard = rule()
    hazard.evaluate(evidence(next_frame(), 1), people_count=0)
    hazard.evaluate(evidence(next_frame(dt=4), 2, smoke=False), people_count=0)
    assert hazard.evaluate(evidence(next_frame(dt=4), 3), people_count=0) is None
    hazard.evaluate(SceneView(evidence=None), people_count=0)
    assert hazard.evaluate(evidence(next_frame(dt=4), 4), people_count=0) is None


def test_a_vlm_only_rule_needs_two_reports() -> None:
    with pytest.raises(ValueError):
        SceneHazardRule(confirmations=1, max_gap_ns=NS_PER_SECOND)
