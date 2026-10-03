"""V2-15 (demo form): notification delivery under provider failures and restarts, v2 vs B0 v1.

Guide ch. 10 and 21 ("Telegram returns HTTP error, ok=false, 429, timeout,
success followed by worker crash; correct retry/ambiguity state"). The provider
is a fake; no token is real.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

from b0_v1_snapshot import V1AlertWorker

from sentinel.alerts.outbox import DeliveryResult, DeliveryStatus, OutboxMessage, OutboxWorker
from sentinel.config import IncidentsConfig, NotificationsConfig
from sentinel.incidents.service import IncidentService
from sentinel.incidents.signals import IncidentSignal, SignalPhase
from sentinel.media.clock import FakeClock
from sentinel.rules.scene_hazard import Severity
from sentinel.storage.database import Database

T0 = datetime(2026, 10, 3, 17, 30, tzinfo=timezone.utc)
SETTINGS = NotificationsConfig(channels=["telegram"])
SIGNAL = IncidentSignal(
    observation_id="zone-x.entered", episode_id="zone-x", camera_id="cam-1", kind="zone.restricted_entry",
    phase=SignalPhase.ENTERED, zone_id="door", rule_revision="rev1", severity=Severity.WARNING,
    title="Person in restricted zone 'door'", observed_utc=T0, boot_id="boot-1", observed_mono_ns=1,
    reason="test", payload="{}",
)


class Provider:
    """Scripted provider outcomes, shared by the v1 and v2 paths."""

    def __init__(self, *outcomes: str) -> None:
        self.outcomes = list(outcomes)
        self.accepted = 0

    def next(self) -> str:
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        self.accepted += outcome == "ok"
        return outcome

    # v1: requests.post returns a status or raises on a network failure
    def v1_post(self, caption: str) -> int:
        outcome = self.next()
        if outcome == "down":
            raise ConnectionError("connection refused")
        return {"ok": 200, "unauthorized": 401}[outcome]

    # v2: the Telegram adapter's interpretation of the same outcomes
    @property
    def channel(self) -> str:
        return "telegram"

    def send(self, message: OutboxMessage) -> DeliveryResult:
        outcome = self.next()
        if outcome == "down":
            return DeliveryResult(DeliveryStatus.RETRY, "network error: ConnectionRefusedError")
        if outcome == "unauthorized":
            return DeliveryResult(DeliveryStatus.PERMANENT, "HTTP 401, error 401: Unauthorized")
        return DeliveryResult(DeliveryStatus.SENT, "sent", provider_message_id="9")


def _status(db: Database) -> tuple[str, str | None]:
    return db.connection.execute("SELECT status, last_error FROM outbox").fetchone()


def test_outage_across_a_restart_is_delivered_by_v2_and_lost_by_v1(tmp_path: Path) -> None:
    clock = FakeClock(utc=T0)
    provider = Provider("down", "down")
    db = Database.open(tmp_path / "s.db")
    IncidentService(db, clock, incidents=IncidentsConfig(), notifications=SETTINGS).record(SIGNAL)
    worker = OutboxWorker(db, clock, {"telegram": provider}, SETTINGS, rng=random.Random(3))
    worker.run_once()
    db.close()  # restart while the provider is down
    db = Database.open(tmp_path / "s.db")
    try:
        worker = OutboxWorker(db, clock, {"telegram": provider}, SETTINGS, rng=random.Random(3))
        for _ in range(3):
            clock.advance(60)
            worker.run_once()
        assert _status(db) == ("sent", None) and provider.accepted == 1
    finally:
        db.close()

    v1, v1_provider = V1AlertWorker(), Provider("down", "down")
    v1.send_alert("INTRUDER")
    v1.drain(v1_provider.v1_post)
    v1.restart()
    v1.drain(v1_provider.v1_post)
    assert (v1.dropped, v1_provider.accepted) == (1, 0)  # never delivered, and nothing records it


def test_rejected_credentials_are_dead_lettered_by_v2_and_counted_as_sent_by_v1(tmp_path: Path) -> None:
    clock = FakeClock(utc=T0)
    db = Database.open(tmp_path / "s.db")
    try:
        IncidentService(db, clock, incidents=IncidentsConfig(), notifications=SETTINGS).record(SIGNAL)
        OutboxWorker(db, clock, {"telegram": Provider("unauthorized")}, SETTINGS).run_once()
        assert _status(db) == ("dead", "needs operator action: HTTP 401, error 401: Unauthorized")
    finally:
        db.close()

    v1, provider = V1AlertWorker(), Provider("unauthorized")
    v1.send_alert("INTRUDER")
    v1.drain(provider.v1_post)
    assert (v1.counted_as_sent, provider.accepted) == (1, 0)
