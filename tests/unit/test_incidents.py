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
from sentinel.contracts import Applicability, Evidence, EvidenceStatus, FrameRef, PixelFormat
from sentinel.incidents.service import (
    AnnotationOutcome,
    IncidentService,
    IncidentStatus,
    InvalidTransition,
    RecordOutcome,
    RevisionConflict,
)
from sentinel.incidents.signals import IncidentSignal, SignalPhase
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.rules.scene_hazard import Severity
from sentinel.storage import database
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
    assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 2
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert c.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"incidents", "observations", "incident_evidence", "incident_transitions", "outbox", "delivery_attempts",
            "incident_annotations"} <= tables


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


# ---------------------------------------------------------------- enrichment (V2-28)

FIRE_REPORT = {
    "persons_visible": 1, "fire_or_smoke": True, "threat": "high",
    "observations": ["smoke near the ceiling"], "uncertainty": "medium", "summary": "possible smoke",
}


def frame(camera_id: str = "cam-1") -> FrameRef:
    stamper = FrameStamper(camera_id, FakeClock(utc=T0))
    stamper.connect()
    return stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)


def scene_evidence(
    incident_id: str | None,
    *,
    job: str = "job-2",
    status: EvidenceStatus = EvidenceStatus.OBSERVED,
    source: FrameRef | None = None,
    value: object = FIRE_REPORT,
) -> Evidence:
    observed = status is EvidenceStatus.OBSERVED
    return Evidence.observed_on(
        source or frame(),
        evidence_id=f"{job}.{status.value}.late",
        kind="scene.report",
        status=status,
        value=value if observed else None,
        producer="scene.stub",
        producer_revision="test",
        ttl_ns=10 * NS_PER_SECOND,
        correlation_group="scene:cam-1",
        reason="stub result" if observed else "job timed out",
        incident_id=incident_id,
    )


def two_incidents(db: Database) -> tuple[IncidentService, str, str]:
    incidents = service(db)
    a = incidents.record(signal("zone-a", at_s=1.0)).incident_id
    b = incidents.record(signal("zone-b", at_s=9.5, zone_id="window")).incident_id
    assert a is not None and b is not None and a != b
    return incidents, a, b


def snapshot(db: Database) -> dict[str, object]:
    c = db.connection
    return {
        "incidents": c.execute(
            "SELECT incident_id, kind, zone_id, severity, status, title, last_observed_utc, last_mono_ns FROM incidents"
            " ORDER BY incident_id"
        ).fetchall(),
        **{table: count(db, table) for table in
           ("observations", "incident_evidence", "incident_transitions", "outbox", "delivery_attempts")},
    }


def test_late_result_enriches_only_its_own_incident_and_changes_nothing_else(db: Database) -> None:
    # Audit finding 3: v1 applied a late answer to whatever was current. Here the
    # late fire report for A is history on A: no new incident, no escalation, no
    # notification, B untouched.
    incidents, a, b = two_incidents(db)
    before, revision_a, revision_b = snapshot(db), incidents.incident(a).revision, incidents.incident(b).revision

    result = incidents.annotate(scene_evidence(a), Applicability.EXPIRED)

    assert (result.outcome, result.incident_id, result.revision) == (AnnotationOutcome.ANNOTATED, a, revision_a + 1)
    assert snapshot(db) == before
    assert incidents.incident(a).revision == revision_a + 1
    assert incidents.incident(b).revision == revision_b
    [note] = incidents.annotations(a)
    assert (note.evidence_id, note.status, note.applicability) == ("job-2.observed.late", "observed", Applicability.EXPIRED)
    assert '"fire_or_smoke":true' in note.payload
    assert incidents.annotations(b) == ()
    # A screen that showed A before the annotation is stale and must reload.
    with pytest.raises(RevisionConflict):
        incidents.transition(a, IncidentStatus.ACKNOWLEDGED, expected_revision=revision_a, actor="op", reason="seen")


def test_repeated_annotation_after_a_restart_changes_nothing(db_path: Path) -> None:
    db = Database.open(db_path)
    incidents, a, _ = two_incidents(db)
    evidence = scene_evidence(a)
    assert incidents.annotate(evidence, Applicability.EXPIRED).outcome is AnnotationOutcome.ANNOTATED
    revision = incidents.incident(a).revision
    db.close()  # the runtime stopped before acknowledging, and hands the same evidence again

    db = Database.open(db_path)
    try:
        incidents = service(db)
        again = incidents.annotate(evidence, Applicability.EXPIRED)
        assert (again.outcome, again.incident_id, again.revision) == (AnnotationOutcome.DUPLICATE, a, None)
        assert incidents.incident(a).revision == revision
        assert len(incidents.annotations(a)) == 1
    finally:
        db.close()


def test_evidence_naming_no_incident_is_not_stored(db: Database) -> None:
    incidents, a, b = two_incidents(db)
    before = snapshot(db)
    result = incidents.annotate(scene_evidence(None, job="job-1"), Applicability.CURRENT)
    assert result.outcome is AnnotationOutcome.NO_INCIDENT
    assert snapshot(db) == before and count(db, "incident_annotations") == 0
    assert incidents.incident(a).revision == incidents.incident(b).revision == 1


def test_unknown_incident_other_camera_and_oversized_evidence_are_refused(db: Database) -> None:
    incidents, a, _ = two_incidents(db)
    before = snapshot(db)
    unknown = incidents.annotate(scene_evidence("inc-gone"), Applicability.EXPIRED)
    other_camera = incidents.annotate(scene_evidence(a, source=frame("cam-2")), Applicability.OTHER_CAMERA)
    oversized = incidents.annotate(scene_evidence(a, value={"summary": "x" * 9000}), Applicability.EXPIRED)
    assert [unknown.outcome, other_camera.outcome, oversized.outcome] == [
        AnnotationOutcome.UNKNOWN_INCIDENT, AnnotationOutcome.OTHER_CAMERA, AnnotationOutcome.TOO_LARGE
    ]
    assert snapshot(db) == before and count(db, "incident_annotations") == 0
    assert incidents.incident(a).revision == 1


def test_resolved_incident_keeps_its_status_when_a_timeout_or_late_result_arrives(db: Database) -> None:
    # Guide ch. 13: a timeout returns unavailable and leaves the original incident intact.
    incidents, a, _ = two_incidents(db)
    revision = incidents.transition(a, IncidentStatus.RESOLVED, expected_revision=1, actor="op", reason="checked")
    timeout = incidents.annotate(scene_evidence(a, status=EvidenceStatus.TIMEOUT), Applicability.CURRENT)
    late = incidents.annotate(scene_evidence(a), Applicability.EXPIRED)
    assert [timeout.outcome, late.outcome] == [AnnotationOutcome.ANNOTATED] * 2
    incident = incidents.incident(a)
    assert (incident.status, incident.severity, incident.revision) == (IncidentStatus.RESOLVED, Severity.WARNING, revision + 2)
    assert [(n.evidence_id, n.status) for n in incidents.annotations(a)] == [
        ("job-2.timeout.late", "timeout"), ("job-2.observed.late", "observed")
    ]
    assert count(db, "incidents") == 2 and count(db, "outbox") == 2


def test_version_1_database_is_upgraded_in_place(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:  # a database written by the V2-14 build
        patch.setattr(database, "MIGRATIONS", database.MIGRATIONS[:1])
        patch.setattr(database, "SCHEMA_VERSION", 1)
        old = Database.open(db_path)
        incident_id = service(old).record(signal()).incident_id
        assert old.connection.execute("PRAGMA user_version").fetchone()[0] == 1
        old.close()

    db = Database.open(db_path)
    try:
        assert db.connection.execute("PRAGMA user_version").fetchone()[0] == 2
        incidents = service(db)
        assert incidents.incident(incident_id).status is IncidentStatus.OPEN
        assert incidents.annotate(scene_evidence(incident_id), Applicability.EXPIRED).outcome is AnnotationOutcome.ANNOTATED
    finally:
        db.close()
