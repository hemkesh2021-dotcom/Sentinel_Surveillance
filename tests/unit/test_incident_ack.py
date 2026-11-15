"""D68: `sentinel incident ack`, the operator's acknowledgment while monitoring is stopped (real SQLite, real lock)."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import sentinel
from sentinel.alerts.outbox import DeliveryResult, DeliveryStatus, OutboxWorker
from sentinel.cli import ACK_NOTE, main
from sentinel.config import IncidentsConfig, NotificationsConfig
from sentinel.incidents.service import (
    AcknowledgeOutcome,
    IncidentService,
    IncidentStatus,
    InvalidTransition,
    UnknownIncident,
)
from sentinel.incidents.signals import IncidentSignal, SignalPhase
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.rules.scene_hazard import Severity
from sentinel.status_page import StatusServer, read_incident, render_incident_html
from sentinel.storage.database import Database, connect_reader

SRC = Path(sentinel.__file__).resolve().parents[1]
T0 = datetime(2026, 10, 9, 6, 0, tzinfo=timezone.utc)
ACK_AT = T0 + timedelta(minutes=5)
TELEGRAM = NotificationsConfig(channels=["telegram"])


def signal(episode: str, *, at_s: float, zone: str = "door") -> IncidentSignal:
    return IncidentSignal(
        observation_id=f"{episode}.entered", episode_id=episode, camera_id="cam-1", kind="zone.restricted_entry",
        phase=SignalPhase.ENTERED, zone_id=zone, rule_revision="rev1", severity=Severity.WARNING,
        title=f"Person in restricted zone {zone!r}", observed_utc=T0 + timedelta(seconds=at_s), boot_id="boot-1",
        observed_mono_ns=round(at_s * NS_PER_SECOND), reason="confirmed track 1 in restricted zone", payload="{}",
    )


@pytest.fixture
def data(tmp_path: Path) -> Path:
    """A stopped runtime's data directory: two open incidents, each with a queued Telegram alert."""
    directory = tmp_path / "data"
    db = Database.open(directory / "sentinel.db")
    service = IncidentService(db, FakeClock(utc=T0), incidents=IncidentsConfig(), notifications=TELEGRAM)
    service.record(signal("ep-door", at_s=10))
    service.record(signal("ep-gate", at_s=20, zone="gate"))
    db.close()
    return directory


def incident_ids(directory: Path) -> dict[str, str]:
    with closing(connect_reader(directory / "sentinel.db")) as c:
        return dict(c.execute("SELECT zone_id, incident_id FROM incidents").fetchall())


def dump(directory: Path) -> list[str]:
    """Every row of every table, read only."""
    connection = connect_reader(directory / "sentinel.db")
    try:
        return list(connection.iterdump())
    finally:
        connection.close()


def ack(directory: Path, capsys, *args: str, at: datetime = ACK_AT) -> tuple[int, dict | None, str]:
    code = main(["incident", "ack", "--data-dir", str(directory), *args], clock=FakeClock(utc=at))
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if captured.out.strip() else None, captured.err


# ---------------------------------------------------------------- what it records


def test_acknowledging_records_its_time_actor_and_note_and_changes_nothing_else(data, capsys) -> None:
    door = incident_ids(data)["door"]
    before = dump(data)
    code, result, err = ack(data, capsys, door, "--actor", "maintainer", "--note", "seen on the status page")
    assert (code, err) == (0, "")
    assert result == {"incident": "ack", "outcome": "acknowledged", "incident_id": door, "status": "acknowledged",
                      "revision": 2, "acknowledged_utc": "2026-10-09T06:05:00.000000Z", "actor": "operator:maintainer",
                      "note": "seen on the status page"}
    after = dump(data)
    changed = sorted(set(after) - set(before))
    assert len(changed) == 2 and len(set(before) - set(after)) == 1  # one incident row replaced, one transition added
    assert any(line.startswith('INSERT INTO "incident_transitions"') and "'open','acknowledged',2,"
               "'operator:maintainer','seen on the status page','2026-10-09T06:05:00.000000Z'" in line
               for line in changed)
    with closing(connect_reader(data / "sentinel.db")) as c:
        row = c.execute("SELECT status, revision, severity, last_observed_utc, updated_utc FROM incidents "
                        "WHERE incident_id = ?", (door,)).fetchone()
        assert row == ("acknowledged", 2, "warning", "2026-10-09T06:00:10.000000Z", "2026-10-09T06:05:00.000000Z")
        assert c.execute("SELECT count(*), sum(status = 'pending') FROM outbox").fetchone() == (2, 2)  # untouched
    for table in ("outbox", "delivery_attempts", "incident_evidence", "observations", "incident_annotations"):
        assert [l for l in before if f'"{table}"' in l] == [l for l in after if f'"{table}"' in l], table


def test_the_acknowledgment_survives_a_restart_and_the_incident_stays_unresolved(data, capsys) -> None:
    door = incident_ids(data)["door"]
    assert ack(data, capsys, door)[0] == 0
    # The runtime starts again on the same database, and the person is still in the zone.
    db = Database.open(data / "sentinel.db")
    try:
        service = IncidentService(db, FakeClock(utc=ACK_AT), incidents=IncidentsConfig(), notifications=TELEGRAM)
        assert service.incident(door).status is IncidentStatus.ACKNOWLEDGED
        joined = service.record(signal("ep-door-2", at_s=60))  # acknowledged is not resolved: it joins (ch. 9)
        assert (joined.incident_id, joined.outbox_ids) == (door, ())  # no new incident, no new alert
        assert service.incident(door).status is IncidentStatus.ACKNOWLEDGED
    finally:
        db.close()
    detail = read_incident(data / "sentinel.db", door)
    assert detail["incident"]["status"] == "acknowledged"
    assert {k: detail["acknowledgment"][k] for k in ("state", "at_utc", "by", "note")} == {
        "state": "acknowledged", "at_utc": "2026-10-09T06:05:00.000000Z", "by": "operator:local", "note": ACK_NOTE}
    assert "acknowledged 2026-10-09T06:05:00.000000Z by operator:local" in render_incident_html(detail)
    assert read_incident(data / "sentinel.db", incident_ids(data)["gate"])["acknowledgment"]["state"] == (
        "not_acknowledged")


def test_the_incident_page_shows_it_and_stays_read_only_and_loopback(data, capsys) -> None:
    door = incident_ids(data)["door"]
    assert ack(data, capsys, door)[0] == 0
    server = StatusServer(0, lambda: {"runtime": {"state": "starting", "camera_id": "cam-1"}}, data / "sentinel.db")
    server.start()
    try:
        import http.client

        def request(method: str, path: str) -> tuple[int, bytes]:
            conn = http.client.HTTPConnection(*server.address, timeout=5)
            try:
                conn.request(method, path, headers={"Host": f"127.0.0.1:{server.address[1]}"})
                response = conn.getresponse()
                return response.status, response.read()
            finally:
                conn.close()

        assert server.address[0] == "127.0.0.1"
        status, body = request("GET", f"/incidents/{door}.json")
        assert status == 200 and json.loads(body)["acknowledgment"]["state"] == "acknowledged"
        status, body = request("GET", f"/incidents/{door}")
        assert status == 200 and b"acknowledged 2026-10-09T06:05:00.000000Z by operator:local" in body
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            assert request(method, f"/incidents/{door}")[0] == 405, method
    finally:
        assert server.stop(2.0)


class Recorder:
    channel = "telegram"

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, message) -> DeliveryResult:
        self.sent.append(message.incident_id)
        return DeliveryResult(DeliveryStatus.SENT, "sent", provider_message_id=str(len(self.sent)))


def test_acknowledging_neither_sends_nor_withdraws_an_alert(data, capsys) -> None:
    door = incident_ids(data)["door"]
    assert ack(data, capsys, door)[0] == 0
    db = Database.open(data / "sentinel.db")  # the restarted runtime's delivery worker
    try:
        notifier = Recorder()
        worker = OutboxWorker(db, FakeClock(utc=ACK_AT), {"telegram": notifier}, TELEGRAM)
        assert len(worker.run_once()) == 2 and worker.run_once() == []
        assert sorted(notifier.sent) == sorted(incident_ids(data).values())  # each waiting alert once, none added
    finally:
        db.close()


# ---------------------------------------------------------------- idempotence and refusals


def test_repeating_the_acknowledgment_changes_nothing_and_reports_the_first(data, capsys) -> None:
    door = incident_ids(data)["door"]
    first = ack(data, capsys, door)
    before = dump(data)
    code, again, err = ack(data, capsys, door, "--actor", "someone-else", at=ACK_AT + timedelta(hours=1))
    assert (code, err) == (0, "")
    assert again == {**first[1], "outcome": "already_acknowledged"}  # the first time, actor and note; same revision
    assert dump(data) == before


@pytest.mark.parametrize("incident_id", ["", "bad id", "../sentinel.db", "inc/1", "inc-é", "a" * 129, "inc-1;"])
def test_an_invalid_id_is_refused_before_the_database_is_touched(data, capsys, incident_id: str) -> None:
    before = dump(data)
    code, result, err = ack(data, capsys, incident_id)
    assert (code, result) == (1, None)
    assert err.startswith("incident ack: invalid_incident_id")
    assert dump(data) == before


def test_an_unknown_incident_is_refused_without_changes(data, capsys) -> None:
    before = dump(data)
    code, result, err = ack(data, capsys, "inc-20261009T060000Z-000000000000")
    assert (code, result) == (1, None) and "unknown_incident" in err and "nothing changed" in err
    assert dump(data) == before


@pytest.mark.parametrize("final", [IncidentStatus.RESOLVED, IncidentStatus.DISMISSED])
def test_a_resolved_or_dismissed_incident_is_not_acknowledged(data, capsys, final: IncidentStatus) -> None:
    door = incident_ids(data)["door"]
    db = Database.open(data / "sentinel.db")
    service = IncidentService(db, FakeClock(utc=T0), incidents=IncidentsConfig(), notifications=TELEGRAM)
    service.transition(door, final, expected_revision=1, actor="operator:test", reason="closed")
    with pytest.raises(InvalidTransition, match=f"is {final.value}; only an open incident"):
        service.acknowledge(door, actor="operator:test", reason="late")
    with pytest.raises(UnknownIncident):
        service.acknowledge("inc-missing", actor="operator:test", reason="late")
    db.close()
    before = dump(data)
    code, _, err = ack(data, capsys, door)
    assert code == 1 and f"not_acknowledgeable (incident {door} is {final.value}" in err
    assert dump(data) == before


def test_a_missing_database_is_refused_and_nothing_is_created(tmp_path, capsys) -> None:
    absent = tmp_path / "no-such-dir"
    code, _, err = ack(absent, capsys, "inc-1")
    assert code == 1 and "no_database" in err and not absent.exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    code, _, err = ack(empty, capsys, "inc-1")
    assert code == 1 and "no_database" in err and list(empty.iterdir()) == []  # no database, WAL or lock file


def test_the_service_acknowledgment_is_one_transaction_with_the_revision_check(data) -> None:
    door = incident_ids(data)["door"]
    db = Database.open(data / "sentinel.db")
    try:
        service = IncidentService(db, FakeClock(utc=ACK_AT), incidents=IncidentsConfig(), notifications=TELEGRAM)
        outbox = db.connection.execute("SELECT * FROM outbox ORDER BY outbox_id").fetchall()
        first = service.acknowledge(door, actor="operator:a", reason="seen")  # a policy with a channel queues nothing
        again = service.acknowledge(door, actor="operator:b", reason="seen again")
        assert db.connection.execute("SELECT * FROM outbox ORDER BY outbox_id").fetchall() == outbox
        assert (first.outcome, again.outcome) == (AcknowledgeOutcome.ACKNOWLEDGED,
                                                  AcknowledgeOutcome.ALREADY_ACKNOWLEDGED)
        assert first.acknowledgment == again.acknowledgment and again.revision == first.revision == 2
        assert service.transition(door, IncidentStatus.RESOLVED, expected_revision=2, actor="operator:a",
                                  reason="gone") == 3  # the existing lifecycle continues from it
    finally:
        db.close()


# ---------------------------------------------------------------- the single writer, across processes


OWNER = """
import sys
from sentinel.storage.database import Database
db = Database.open(sys.argv[1])  # what `sentinel run` does at startup
print("owner ready", flush=True)
sys.stdin.read()  # hold the database until the test closes stdin
db.close()
"""


def cli(directory: Path, incident_id: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sentinel.cli", "incident", "ack", "--data-dir", str(directory), incident_id],
        capture_output=True, text=True, timeout=60, env={"PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_refused_while_another_process_owns_the_database_then_accepted_once_it_stops(data) -> None:
    door = incident_ids(data)["door"]
    with subprocess.Popen([sys.executable, "-c", OWNER, str(data / "sentinel.db")], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, env={"PYTHONPATH": str(SRC)}) as owner:
        try:
            assert owner.stdout.readline().strip() == "owner ready"
            before = dump(data)
            refused = cli(data, door)
            assert refused.returncode == 1 and refused.stdout == ""
            assert refused.stderr.startswith("incident ack: database_busy") and "stop monitoring first" in refused.stderr
            assert dump(data) == before
        finally:
            owner.stdin.close()
            assert owner.wait(timeout=30) == 0
    accepted = cli(data, door)  # the kernel released the lock with the owner
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(accepted.stdout)["outcome"] == "acknowledged"
