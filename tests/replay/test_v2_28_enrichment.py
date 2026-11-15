"""V2-28 (demo form): a late scene result only enriches its own incident, in the store.

R3 showed that the scene lane routes a late answer to the incident it was
requested for and never to current scene state. Here the same timeline runs
against the SQLite incident store: incident A is opened by a zone rule, its
enrichment job times out, incident B opens meanwhile, and A's job finally
answers "fire, high threat". The runtime hands every routed evidence to
``IncidentService.annotate()``. v1 on the same answers sends a fire alert at
11.5 s (B0 snapshot); it has no incident record to annotate.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from scene_harness import run_scene

from sentinel.config import IncidentsConfig, NotificationsConfig
from sentinel.contracts import Applicability, Evidence
from sentinel.incidents.service import AnnotationOutcome, IncidentService, IncidentStatus
from sentinel.incidents.signals import IncidentSignal, SignalPhase
from sentinel.media.clock import FakeClock
from sentinel.rules.scene_hazard import Severity
from sentinel.scene.state import Routing
from sentinel.storage.database import Database

UTC_START = datetime(2026, 10, 3, 17, 30, tzinfo=timezone.utc)
OPENED_MS = {"inc-A": 1000, "inc-B": 9500}  # as the timeline's incident events
ZONES = {"inc-A": "door", "inc-B": "window"}


def zone_signal(incident_id: str, source: Evidence) -> IncidentSignal:
    at_ms = OPENED_MS[incident_id]
    episode = f"episode-{ZONES[incident_id]}"
    return IncidentSignal(
        observation_id=f"{episode}.entered",
        episode_id=episode,
        camera_id=source.source.camera_id,
        kind="zone.restricted_entry",
        phase=SignalPhase.ENTERED,
        zone_id=ZONES[incident_id],
        rule_revision="rev1",
        severity=Severity.WARNING,
        title=f"Person in restricted zone {ZONES[incident_id]!r}",
        observed_utc=UTC_START + timedelta(milliseconds=at_ms),
        boot_id=source.source.boot_id,
        observed_mono_ns=at_ms * 1_000_000,
        reason="synthetic zone entry",
        payload="{}",
    )


def test_late_fire_answer_for_a_is_stored_on_a_only(load_timeline, tmp_path: Path) -> None:
    run = run_scene(load_timeline("r3_late_scene_result.jsonl"))
    assert min(run.v1_fire_alert_ms) == 11500  # B0: v1 alerts on the late answer

    db = Database.open(tmp_path / "sentinel.db")
    names = iter(["inc-A", "inc-B"])
    incidents = IncidentService(
        db,
        FakeClock(utc=UTC_START),
        incidents=IncidentsConfig(),
        notifications=NotificationsConfig(channels=["telegram"]),
        new_id=lambda now: next(names),
    )
    first = run.evidence[0][1]

    def hand_over(routed: list[tuple[int, Evidence, Routing]]) -> list[tuple[str, AnnotationOutcome]]:
        # What `sentinel run` does with CoreOutput.evidence (D-1).
        return [(e.evidence_id, incidents.annotate(e, r.applicability).outcome) for _, e, r in routed]

    try:
        outcomes = []
        waiting = sorted(OPENED_MS, key=OPENED_MS.get)
        for item in run.evidence:  # in time order; each incident opens at its timeline time
            while waiting and OPENED_MS[waiting[0]] <= item[0]:
                assert incidents.record(zone_signal(waiting[0], first)).incident_id == waiting.pop(0)
            outcomes += hand_over([item])
        assert waiting == []

        assert outcomes == [
            ("job-1.observed", AnnotationOutcome.NO_INCIDENT),  # periodic: about no incident
            ("job-2.timeout", AnnotationOutcome.ANNOTATED),
            ("job-3.observed", AnnotationOutcome.NO_INCIDENT),
            ("job-2.observed.late", AnnotationOutcome.ANNOTATED),
        ]
        notes = incidents.annotations("inc-A")
        assert [(n.evidence_id, n.applicability) for n in notes] == [
            ("job-2.timeout", Applicability.CURRENT),
            ("job-2.observed.late", Applicability.EXPIRED),
        ]
        assert incidents.annotations("inc-B") == ()

        a, b = incidents.incident("inc-A"), incidents.incident("inc-B")
        # The fire report neither escalated A nor opened a fire incident, and nothing was notified.
        assert (a.status, a.severity, a.kind) == (IncidentStatus.OPEN, Severity.WARNING, "zone.restricted_entry")
        assert (b.status, b.severity, b.revision) == (IncidentStatus.OPEN, Severity.WARNING, 1)
        c = db.connection
        assert c.execute("SELECT COUNT(*) FROM incidents").fetchone()[0] == 2
        assert c.execute("SELECT incident_id, message_kind FROM outbox ORDER BY outbox_id").fetchall() == [
            ("inc-A", "opened"), ("inc-B", "opened")
        ]

        # The runtime restarts before acknowledging and hands everything over again.
        assert {outcome for _, outcome in hand_over(run.evidence)} == {
            AnnotationOutcome.NO_INCIDENT, AnnotationOutcome.DUPLICATE
        }
        assert incidents.incident("inc-A").revision == a.revision
        assert len(incidents.annotations("inc-A")) == 2
    finally:
        db.close()
