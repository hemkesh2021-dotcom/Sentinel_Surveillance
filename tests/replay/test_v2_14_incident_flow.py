"""V2-14 (demo form): rule observations from replays become durable incidents.

The runtime hands every zone observation and scene hazard candidate to the
incident service and may retry any of them (no acknowledgment, restart): the
database must end up the same. v1 had no incident record at all: alerts were
sent directly from the main loop (B0 snapshot), with in-memory cooldowns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from core_harness import EMPTY_ROOM, SMOKE, run_core
from timeline_builder import TimelineBuilder
from zone_harness import IN_ZONE, OUTSIDE, person, run_zones, zone

from sentinel.config import IncidentsConfig, NotificationsConfig
from sentinel.incidents.service import IncidentService, IncidentStatus, RecordOutcome
from sentinel.incidents.signals import signal_from_hazard, signal_from_zone
from sentinel.media.clock import FakeClock
from sentinel.rules.scene_hazard import Severity
from sentinel.storage.database import Database

UTC_START = datetime(2026, 10, 3, 17, 30, tzinfo=timezone.utc)  # 23:00 in Asia/Kolkata


def _service(path: Path) -> tuple[Database, IncidentService]:
    db = Database.open(path)
    return db, IncidentService(
        db,
        FakeClock(utc=UTC_START),
        incidents=IncidentsConfig(),
        notifications=NotificationsConfig(channels=["telegram"]),
    )


def test_zone_episodes_become_one_incident_and_retries_change_nothing(tmp_path: Path) -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(3000, persons=person(IN_ZONE, track=1))
        .frames_until(5000, persons=person(OUTSIDE, track=1))
        # The tracker loses the person and re-identifies them as track 2: a new episode.
        .frames_until(9000, persons=person(IN_ZONE, track=2))
        .frames_until(11_000, persons=person(OUTSIDE, track=2))
        .build()
    )
    run = run_zones(timeline, zones=[zone()], utc_start=UTC_START)
    observations = [o for _, o in run.observations]
    assert [(o.track.track_id, o.phase.value) for o in observations] == [
        (1, "entered"), (1, "ended"), (2, "entered"), (2, "ended")
    ]

    db, incidents = _service(tmp_path / "sentinel.db")
    try:
        outcomes = [incidents.record(signal_from_zone(o)).outcome for o in observations]
        assert outcomes == [RecordOutcome.CREATED, RecordOutcome.ENDED_NOTED, RecordOutcome.JOINED, RecordOutcome.ENDED_NOTED]
        # The runtime never got to acknowledge them, and replays all four.
        retried = [incidents.record(signal_from_zone(o)).outcome for o in observations]
        assert retried == [RecordOutcome.DUPLICATE] * 4
        c = db.connection
        (incident_id,) = c.execute("SELECT incident_id FROM incidents").fetchone()
        assert c.execute("SELECT COUNT(*) FROM incident_evidence").fetchone()[0] == 4
        assert c.execute("SELECT COUNT(*) FROM incident_transitions").fetchone()[0] == 1
        assert c.execute("SELECT channel, message_kind, status FROM outbox").fetchall() == [("telegram", "opened", "pending")]
        incident = incidents.incident(incident_id)
        assert (incident.status, incident.severity, incident.zone_id) == (IncidentStatus.OPEN, Severity.WARNING, "door")
        assert incident.title == "Person in restricted zone 'door'"
    finally:
        db.close()


def test_vlm_only_fire_candidate_opens_an_unconfirmed_warning(tmp_path: Path) -> None:
    timeline = TimelineBuilder().connect(0).frames_until(20_000, persons=0).build()
    smoke = lambda number, job: (1500, SMOKE if number >= 3 else EMPTY_ROOM)  # noqa: E731
    run, _ = run_core(timeline, answer=smoke)
    [(_, candidate)] = run.candidates

    db, incidents = _service(tmp_path / "sentinel.db")
    try:
        first = incidents.record(signal_from_hazard(candidate))
        assert incidents.record(signal_from_hazard(candidate)).outcome is RecordOutcome.DUPLICATE
        incident = incidents.incident(first.incident_id)
        assert incident.severity is Severity.WARNING  # never critical from the scene model alone (D17)
        assert incident.title == "Possible fire or smoke (scene model only, unconfirmed)"
        assert incident.zone_id is None and incident.kind == "scene.fire_smoke_candidate"
    finally:
        db.close()
