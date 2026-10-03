"""V2-14 (demo form): the SQLite incident transaction, migrations and crash safety."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import sentinel
from sentinel.config import IncidentsConfig, NotificationsConfig
from sentinel.incidents.service import (
    IncidentService,
    IncidentStatus,
    InvalidTransition,
    RecordOutcome,
    RevisionConflict,
)
from sentinel.incidents.signals import IncidentSignal, SignalPhase
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.rules.scene_hazard import Severity
from sentinel.storage.database import SCHEMA_VERSION, Database, DatabaseError, WriterBusy, connect_reader

SRC = Path(sentinel.__file__).resolve().parents[1]
T0 = datetime(2026, 10, 3, 17, 30, tzinfo=timezone.utc)
TELEGRAM = NotificationsConfig(channels=["telegram"])


def signal(
    episode: str = "zone-a",
    phase: SignalPhase = SignalPhase.ENTERED,
    *,
    at_s: float = 10.0,
    severity: Severity = Severity.WARNING,
    zone_id: str | None = "door",
    kind: str = "zone.restricted_entry",
    boot_id: str = "boot-1",
) -> IncidentSignal:
    return IncidentSignal(
        observation_id=f"{episode}.{phase.value}",
        episode_id=episode,
        camera_id="cam-1",
        kind=kind,
        phase=phase,
        zone_id=zone_id,
        rule_revision="rev1",
        severity=severity,
        title=f"Person in restricted zone {zone_id!r}",
        observed_utc=T0 + timedelta(seconds=at_s),
        boot_id=boot_id,
        observed_mono_ns=round(at_s * NS_PER_SECOND),
        reason="confirmed track 1 in restricted zone 'door' for 1.0 s",
        payload='{"synthetic": true}',
    )


def count(db: Database | sqlite3.Connection, table: str) -> int:
    connection = db.connection if isinstance(db, Database) else db
    return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "sentinel.db"


@pytest.fixture
def db(db_path: Path):
    database = Database.open(db_path)
    yield database
    database.close()


def service(db: Database, notifications: NotificationsConfig = TELEGRAM, **incidents: float) -> IncidentService:
    return IncidentService(
        db, FakeClock(utc=T0), incidents=IncidentsConfig(**incidents), notifications=notifications
    )


# ---------------------------------------------------------------- database


def test_new_database_is_migrated_with_durable_settings(db: Database, db_path: Path) -> None:
    c = db.connection
    assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 1
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert c.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"incidents", "observations", "incident_evidence", "incident_transitions", "outbox", "delivery_attempts"} <= tables


def test_only_one_writer_and_readers_cannot_write(db: Database, db_path: Path) -> None:
    with pytest.raises(WriterBusy, match="only one Sentinel runtime"):
        Database.open(db_path)
    reader = connect_reader(db_path)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.execute("DELETE FROM incidents")
    reader.close()


def test_reopening_keeps_data_and_a_newer_schema_is_refused(db_path: Path) -> None:
    first = Database.open(db_path)
    service(first).record(signal())
    first.close()
    again = Database.open(db_path)
    assert count(again, "incidents") == 1
    again.connection.execute("PRAGMA user_version = 99")
    again.close()
    with pytest.raises(DatabaseError, match="newer than this build"):
        Database.open(db_path)


# ---------------------------------------------------------------- the transaction


def test_repeated_source_event_produces_one_incident_transition(db: Database) -> None:
    incidents = service(db)
    first = incidents.record(signal())
    assert first.outcome is RecordOutcome.CREATED and first.revision == 1 and len(first.outbox_ids) == 1
    # The runtime retries the same observation (no acknowledgment arrived): nothing new.
    for _ in range(3):
        again = incidents.record(signal())
        assert (again.outcome, again.incident_id, again.revision, again.outbox_ids) == (
            RecordOutcome.DUPLICATE, first.incident_id, 1, ()
        )
    assert [count(db, t) for t in ("incidents", "incident_transitions", "incident_evidence", "outbox")] == [1, 1, 1, 1]
    status, revision = db.connection.execute("SELECT status, revision FROM incidents").fetchone()
    assert (status, revision) == ("open", 1)


def _run_child(db_path: Path, body: str) -> subprocess.CompletedProcess:
    script = textwrap.dedent(
        f"""
        import os, sys
        from datetime import datetime, timezone
        from sentinel.config import IncidentsConfig, NotificationsConfig
        from sentinel.incidents.service import IncidentService
        from sentinel.incidents.signals import IncidentSignal
        from sentinel.media.clock import FakeClock
        from sentinel.storage.database import Database
        db = Database.open({str(db_path)!r})
        service = IncidentService(db, FakeClock(utc=datetime(2026, 10, 3, tzinfo=timezone.utc)),
                                  incidents=IncidentsConfig(), notifications=NotificationsConfig(channels=["telegram"]))
        signal = IncidentSignal.model_validate_json(sys.stdin.read())
        """
    ) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        input=signal().model_dump_json(),
        text=True,
        capture_output=True,
        env={"PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=60,
    )


def test_crash_after_commit_then_retry_deduplicates(db_path: Path) -> None:
    # The process dies right after COMMIT, before it can acknowledge the observation.
    child = _run_child(db_path, "service.record(signal)\nos._exit(9)\n")
    assert child.returncode == 9, child.stderr
    db = Database.open(db_path)
    try:
        result = service(db).record(signal())  # the restarted runtime retries the observation
        assert result.outcome is RecordOutcome.DUPLICATE
        assert [count(db, t) for t in ("incidents", "incident_transitions", "outbox", "observations")] == [1, 1, 1, 1]
    finally:
        db.close()


def test_crash_inside_the_transaction_leaves_nothing_behind(db_path: Path) -> None:
    # The process dies after the incident row is written but before COMMIT.
    body = """
    def die(*args, **kwargs):
        os._exit(9)
    IncidentService._evidence = staticmethod(die)
    service.record(signal)
    """
    child = _run_child(db_path, body)
    assert child.returncode == 9, child.stderr
    db = Database.open(db_path)
    try:
        assert [count(db, t) for t in ("incidents", "observations", "outbox")] == [0, 0, 0]
        assert service(db).record(signal()).outcome is RecordOutcome.CREATED
    finally:
        db.close()


def test_a_failure_at_the_last_step_rolls_back_the_whole_transaction(db: Database) -> None:
    db.connection.execute(
        "CREATE TRIGGER fail_outbox BEFORE INSERT ON outbox BEGIN SELECT RAISE(ABORT, 'disk full'); END"
    )
    incidents = service(db)
    with pytest.raises(sqlite3.IntegrityError, match="disk full"):
        incidents.record(signal())
    assert [count(db, t) for t in ("incidents", "observations", "incident_evidence", "incident_transitions")] == [0, 0, 0, 0]
    db.connection.execute("DROP TRIGGER fail_outbox")
    assert incidents.record(signal()).outcome is RecordOutcome.CREATED


# ---------------------------------------------------------------- correlation and severity


def test_episodes_join_within_the_merge_window_and_open_a_linked_incident_after(db: Database) -> None:
    incidents = service(db, merge_window_s=120.0)
    first = incidents.record(signal("zone-a", at_s=10))
    joined = incidents.record(signal("zone-b", at_s=70))  # a tracker ID switch or a second person
    assert (joined.outcome, joined.incident_id, joined.revision, joined.outbox_ids) == (
        RecordOutcome.JOINED, first.incident_id, 2, ()
    )
    later = incidents.record(signal("zone-c", at_s=70 + 121))  # beyond the window from the latest
    assert later.outcome is RecordOutcome.CREATED and later.incident_id != first.incident_id
    assert incidents.incident(later.incident_id).linked_incident_id == first.incident_id
    other_zone = incidents.record(signal("zone-d", at_s=200, zone_id="garage"))
    assert other_zone.outcome is RecordOutcome.CREATED
    assert count(db, "outbox") == 3  # one "opened" per incident, none for the joined episode


def test_resolved_incidents_are_not_joined(db: Database) -> None:
    incidents = service(db)
    first = incidents.record(signal("zone-a", at_s=10))
    incidents.transition(first.incident_id, IncidentStatus.RESOLVED, expected_revision=1, actor="operator:test", reason="checked")
    again = incidents.record(signal("zone-b", at_s=20))
    assert again.outcome is RecordOutcome.CREATED
    assert incidents.incident(again.incident_id).linked_incident_id == first.incident_id


def test_a_reboot_never_joins_across_boots(db: Database) -> None:
    incidents = service(db)
    first = incidents.record(signal("zone-a", at_s=500))
    # Monotonic time restarts after a reboot; 3 s here is not "3 s after" anything before.
    rebooted = incidents.record(signal("zone-b", at_s=3, boot_id="boot-2"))
    assert rebooted.outcome is RecordOutcome.CREATED and rebooted.incident_id != first.incident_id


def test_severity_takes_the_maximum_and_each_rise_notifies_once(db: Database) -> None:
    incidents = service(db, merge_window_s=600.0)
    first = incidents.record(signal("zone-a", at_s=10, severity=Severity.WARNING))
    for n, severity in enumerate([Severity.WARNING, Severity.INFO, Severity.CRITICAL, Severity.CRITICAL, Severity.WARNING]):
        incidents.record(signal(f"zone-{n}", at_s=20 + n, severity=severity))
    assert incidents.incident(first.incident_id).severity is Severity.CRITICAL
    kinds = [r[0] for r in db.connection.execute("SELECT message_kind FROM outbox ORDER BY outbox_id")]
    assert kinds == ["opened", "escalated-critical"]
    priorities = [r[0] for r in db.connection.execute("SELECT priority FROM outbox ORDER BY outbox_id")]
    assert priorities == [1, 0]


def test_ended_is_evidence_on_its_own_incident_and_changes_no_status(db: Database) -> None:
    incidents = service(db)
    first = incidents.record(signal("zone-a", at_s=10))
    other = incidents.record(signal("zone-x", at_s=11, zone_id="garage"))
    ended = incidents.record(signal("zone-a", SignalPhase.ENDED, at_s=40))
    assert (ended.outcome, ended.incident_id, ended.revision) == (RecordOutcome.ENDED_NOTED, first.incident_id, 2)
    assert incidents.incident(first.incident_id).status is IncidentStatus.OPEN
    phases = db.connection.execute(
        "SELECT incident_id, phase FROM incident_evidence ORDER BY evidence_seq"
    ).fetchall()
    assert phases == [(first.incident_id, "entered"), (other.incident_id, "entered"), (first.incident_id, "ended")]
    orphan = incidents.record(signal("zone-never-entered", SignalPhase.ENDED, at_s=50))
    assert orphan.outcome is RecordOutcome.IGNORED and count(db, "outbox") == 2


def test_channels_are_off_by_default_and_low_severity_is_not_sent(db: Database) -> None:
    quiet = service(db, notifications=NotificationsConfig())
    assert quiet.record(signal("zone-a")).outbox_ids == ()
    info_only = service(db)
    assert info_only.record(signal("zone-b", zone_id="hall", severity=Severity.INFO)).outbox_ids == ()
    assert count(db, "outbox") == 0


def test_a_changed_notification_policy_may_notify_again(db: Database) -> None:
    first = service(db).record(signal("zone-a"))
    stricter = NotificationsConfig(channels=["telegram"], min_severity="info")
    db.connection.execute("UPDATE incidents SET severity = 'info'")
    raised = service(db, notifications=stricter).record(signal("zone-b", at_s=20, severity=Severity.WARNING))
    assert raised.incident_id == first.incident_id and len(raised.outbox_ids) == 1
    revisions = {r[0] for r in db.connection.execute("SELECT policy_revision FROM outbox")}
    assert revisions == {TELEGRAM.revision, stricter.revision}


# ---------------------------------------------------------------- operator transitions


def test_operator_transitions_check_revision_and_lifecycle(db: Database) -> None:
    incidents = service(db)
    created = incidents.record(signal())
    joined = incidents.record(signal("zone-b", at_s=20))  # revision 2 now
    with pytest.raises(RevisionConflict, match="revision 2, not 1"):
        incidents.transition(created.incident_id, IncidentStatus.ACKNOWLEDGED, expected_revision=1, actor="operator:a", reason="seen")
    revision = incidents.transition(
        created.incident_id, IncidentStatus.ACKNOWLEDGED, expected_revision=joined.revision, actor="operator:a", reason="seen"
    )
    # Acknowledged is not resolved: the condition may continue and still joins the incident.
    assert incidents.record(signal("zone-c", at_s=30)).incident_id == created.incident_id
    revision += 1
    revision = incidents.transition(created.incident_id, IncidentStatus.RESOLVED, expected_revision=revision, actor="operator:a", reason="gone")
    with pytest.raises(InvalidTransition, match="from resolved to open"):
        incidents.transition(created.incident_id, IncidentStatus.OPEN, expected_revision=revision, actor="operator:a", reason="again")
    log = db.connection.execute(
        "SELECT from_status, to_status, revision, actor FROM incident_transitions ORDER BY transition_seq"
    ).fetchall()
    assert log == [
        (None, "open", 1, "rule:zone-a.entered"),
        ("open", "acknowledged", 3, "operator:a"),
        ("acknowledged", "resolved", 5, "operator:a"),
    ]


def test_payloads_are_bounded() -> None:
    with pytest.raises(ValueError, match="at most 8192 characters"):
        IncidentSignal.model_validate(signal().model_dump() | {"payload": "x" * 8193})
