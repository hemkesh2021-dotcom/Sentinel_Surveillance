"""V2-15 (demo form): leased outbox worker and the Telegram adapter, against mocks only.

No test talks to Telegram or uses a real token: the opener is a fake that
returns or raises what the Telegram Bot API would.
"""

from __future__ import annotations

import io
import json
import random
import socket
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sentinel.alerts.outbox import DeliveryResult, DeliveryStatus, OutboxMessage, OutboxWorker
from sentinel.alerts.telegram import NotifierUnavailable, TelegramNotifier
from sentinel.config import IncidentsConfig, NotificationsConfig
from sentinel.incidents.service import IncidentService
from sentinel.incidents.signals import IncidentSignal, SignalPhase
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.rules.scene_hazard import Severity
from sentinel.storage.database import Database

FAKE_TOKEN = "123456789:FAKE-test-token-not-real"
T0 = datetime(2026, 10, 3, 17, 30, tzinfo=timezone.utc)
SETTINGS = NotificationsConfig(channels=["telegram"])


# ---------------------------------------------------------------- fakes


class FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = io.BytesIO(body)

    def read(self, limit: int = -1) -> bytes:
        return self._body.read(limit)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def http_error(code: int, body: dict | None) -> urllib.error.HTTPError:
    data = io.BytesIO(json.dumps(body).encode() if body is not None else b"")
    # Like urllib, the error's URL contains the token: it must never be shown.
    return urllib.error.HTTPError(f"https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage", code, "error", {}, data)


@dataclass
class FakeOpener:
    replies: list[object]  # FakeResponse to return, or an exception to raise
    requests: list[tuple[object, float]] = field(default_factory=list)

    def __call__(self, request: object, *, timeout: float) -> FakeResponse:
        self.requests.append((request, timeout))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def ok(message_id: int = 42) -> FakeResponse:
    return FakeResponse(200, json.dumps({"ok": True, "result": {"message_id": message_id}}).encode())


def telegram(*replies: object) -> tuple[TelegramNotifier, FakeOpener]:
    opener = FakeOpener(list(replies))
    return TelegramNotifier(FAKE_TOKEN, "-100123", timeout_s=7.0, opener=opener), opener


MESSAGE = OutboxMessage(1, "inc-1", "telegram", "opened", 1, "[Sentinel] WARNING: test")


# ---------------------------------------------------------------- Telegram adapter


def test_success_needs_http_200_and_ok_true_with_a_message_id() -> None:
    notifier, opener = telegram(ok(42))
    result = notifier.send(MESSAGE)
    assert (result.status, result.provider_message_id, result.ambiguous) == (DeliveryStatus.SENT, "42", False)
    request, timeout = opener.requests[0]
    assert timeout == 7.0 and request.get_method() == "POST"
    assert json.loads(request.data) == {"chat_id": "-100123", "text": MESSAGE.text, "disable_web_page_preview": True}


@pytest.mark.parametrize(
    ("reply", "status", "ambiguous", "retry_after", "detail"),
    [
        (http_error(502, {"ok": False, "error_code": 502, "description": "Bad Gateway"}), DeliveryStatus.RETRY, False, None, "HTTP 502, error 502: Bad Gateway"),
        (http_error(503, None), DeliveryStatus.RETRY, False, None, "HTTP 503, error 503: no description"),
        (FakeResponse(200, b'{"ok": false, "error_code": 400, "description": "Bad Request: chat not found"}'), DeliveryStatus.PERMANENT, False, None, "chat not found"),
        (http_error(401, {"ok": False, "error_code": 401, "description": "Unauthorized"}), DeliveryStatus.PERMANENT, False, None, "Unauthorized"),
        (http_error(403, {"ok": False, "error_code": 403, "description": "Forbidden: bot was blocked by the user"}), DeliveryStatus.PERMANENT, False, None, "blocked"),
        (http_error(429, {"ok": False, "error_code": 429, "description": "Too Many Requests: retry after 17", "parameters": {"retry_after": 17}}), DeliveryStatus.RETRY, False, 17, "rate limited"),
        (http_error(429, {"ok": False, "error_code": 429, "parameters": {"retry_after": True}}), DeliveryStatus.RETRY, False, None, "rate limited"),
        (FakeResponse(200, b"<html>proxy page</html>"), DeliveryStatus.RETRY, True, None, "unreadable body"),
        (FakeResponse(200, b'{"ok": true}'), DeliveryStatus.RETRY, True, None, "without a message ID"),
        (socket.timeout("timed out"), DeliveryStatus.RETRY, True, None, "timeout after sending"),
        (urllib.error.URLError(TimeoutError()), DeliveryStatus.RETRY, True, None, "timeout after sending"),
        (urllib.error.URLError(ConnectionResetError()), DeliveryStatus.RETRY, True, None, "connection reset"),
        (urllib.error.URLError(ConnectionRefusedError()), DeliveryStatus.RETRY, False, None, "network error: ConnectionRefusedError"),
        (urllib.error.URLError(f"https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage unreachable"), DeliveryStatus.RETRY, False, None, "network error: str"),
    ],
    ids=["502", "503-no-body", "ok-false-400", "401", "403-blocked", "429-retry-after", "429-bad-retry-after",
         "200-not-json", "ok-without-id", "timeout", "url-timeout", "reset", "refused", "url-with-token"],
)
def test_provider_outcomes(reply: object, status: DeliveryStatus, ambiguous: bool, retry_after: float | None, detail: str) -> None:
    notifier, _ = telegram(reply)
    result = notifier.send(MESSAGE)
    assert (result.status, result.ambiguous, result.retry_after_s) == (status, ambiguous, retry_after)
    assert detail in result.detail
    assert "FAKE-test-token" not in result.detail and "123456789" not in result.detail


def test_credentials_come_from_the_environment_and_are_never_shown() -> None:
    with pytest.raises(NotifierUnavailable, match="SENTINEL_TELEGRAM_BOT_TOKEN"):
        TelegramNotifier.from_environment(environ={"SENTINEL_TELEGRAM_CHAT_ID": "1"})
    notifier = TelegramNotifier.from_environment(
        environ={"SENTINEL_TELEGRAM_BOT_TOKEN": FAKE_TOKEN, "SENTINEL_TELEGRAM_CHAT_ID": "-100123"}
    )
    assert "FAKE" not in repr(notifier) and "-100123" not in repr(notifier)


# ---------------------------------------------------------------- the worker


def signal(n: int = 1, severity: Severity = Severity.WARNING, zone: str = "door") -> IncidentSignal:
    return IncidentSignal(
        observation_id=f"zone-{zone}-{n}.entered", episode_id=f"zone-{zone}-{n}", camera_id="cam-1",
        kind="zone.restricted_entry", phase=SignalPhase.ENTERED, zone_id=zone, rule_revision="rev1",
        severity=severity, title=f"Person in restricted zone {zone!r}", observed_utc=T0, boot_id="boot-1",
        observed_mono_ns=n * NS_PER_SECOND, reason="test", payload="{}",
    )


class Scripted:
    """A notifier that returns scripted results and remembers what it sent."""

    def __init__(self, channel: str, *results: DeliveryResult | Exception) -> None:
        self.channel = channel
        self.results = list(results)
        self.sent: list[OutboxMessage] = []

    def send(self, message: OutboxMessage) -> DeliveryResult:
        self.sent.append(message)
        result = self.results.pop(0) if self.results else DeliveryResult(DeliveryStatus.SENT, "sent", provider_message_id="1")
        if isinstance(result, Exception):
            raise result
        return result


RETRY = DeliveryResult(DeliveryStatus.RETRY, "HTTP 502, error 502: Bad Gateway")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(utc=T0)


@pytest.fixture
def db(tmp_path: Path):
    database = Database.open(tmp_path / "sentinel.db")
    yield database
    database.close()


def open_incident(db: Database, clock: FakeClock, *signals: IncidentSignal) -> None:
    incidents = IncidentService(db, clock, incidents=IncidentsConfig(), notifications=SETTINGS)
    for s in signals or (signal(),):
        incidents.record(s)


def worker(db: Database, clock: FakeClock, *notifiers: Scripted | TelegramNotifier, worker_id: str = "w1", **settings: object) -> OutboxWorker:
    config = NotificationsConfig(channels=["telegram"], **settings)
    return OutboxWorker(db, clock, {n.channel: n for n in notifiers}, config, worker_id=worker_id, rng=random.Random(7))


def row(db: Database, outbox_id: int = 1) -> dict:
    cursor = db.connection.execute("SELECT * FROM outbox WHERE outbox_id = ?", (outbox_id,))
    return dict(zip([d[0] for d in cursor.description], cursor.fetchone()))


def attempts(db: Database, outbox_id: int = 1) -> list[tuple]:
    return db.connection.execute(
        "SELECT attempt, lease_owner, outcome FROM delivery_attempts WHERE outbox_id = ? ORDER BY attempt_seq", (outbox_id,)
    ).fetchall()


def test_sent_message_is_recorded_with_the_provider_id(db: Database, clock: FakeClock) -> None:
    open_incident(db, clock)
    notifier, _ = telegram(ok(42))
    assert worker(db, clock, notifier).run_once() == [(1, "sent")]
    r = row(db)
    assert (r["status"], r["provider_message_id"], r["attempts"], r["ambiguous"]) == ("sent", "42", 1, 0)
    assert attempts(db) == [(1, "w1", "sent")]
    assert worker(db, clock, notifier).run_once() == []  # nothing left


def test_message_text_identifies_the_incident_and_carries_no_media(db: Database, clock: FakeClock) -> None:
    open_incident(db, clock)
    fake = Scripted("telegram")
    worker(db, clock, fake).run_once()
    [message] = fake.sent
    (incident_id,) = db.connection.execute("SELECT incident_id FROM incidents").fetchone()
    assert message.text.splitlines()[0] == "[Sentinel] WARNING: Person in restricted zone 'door'"
    assert f"Incident {incident_id} (open) on camera cam-1" in message.text
    assert "at least once" in message.text


def test_transient_failure_backs_off_with_jitter_then_succeeds(db: Database, clock: FakeClock) -> None:
    open_incident(db, clock)
    fake = Scripted("telegram", RETRY)
    w = worker(db, clock, fake, backoff_base_s=5.0)
    assert w.run_once() == [(1, "retry")]
    due = datetime.strptime(row(db)["next_attempt_utc"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    assert T0 + timedelta(seconds=2.5) <= due < T0 + timedelta(seconds=5)
    assert row(db)["last_error"] == "HTTP 502, error 502: Bad Gateway"
    clock.advance(2.4)
    assert w.run_once() == []  # not due yet
    clock.advance(2.6)
    assert w.run_once() == [(1, "sent")] and row(db)["attempts"] == 2


def test_rate_limit_waits_at_least_retry_after(db: Database, clock: FakeClock) -> None:
    open_incident(db, clock)
    notifier, _ = telegram(http_error(429, {"ok": False, "error_code": 429, "parameters": {"retry_after": 30}}), ok())
    w = worker(db, clock, notifier, backoff_base_s=5.0)
    assert w.run_once() == [(1, "retry")]
    clock.advance(29.9)
    assert w.run_once() == []
    clock.advance(0.2)
    assert w.run_once() == [(1, "sent")]


def test_timeout_marks_the_row_ambiguous(db: Database, clock: FakeClock) -> None:
    open_incident(db, clock)
    notifier, _ = telegram(socket.timeout("timed out"), ok())
    w = worker(db, clock, notifier)
    assert w.run_once() == [(1, "retry")]
    assert row(db)["ambiguous"] == 1 and attempts(db)[0][2] == "retry (ambiguous)"
    clock.advance(60)
    assert w.run_once() == [(1, "sent")] and row(db)["ambiguous"] == 1  # stays flagged: a duplicate is possible


def test_permanent_failure_is_dead_lettered_until_an_operator_retries(db: Database, clock: FakeClock) -> None:
    open_incident(db, clock)
    notifier, _ = telegram(http_error(401, {"ok": False, "error_code": 401, "description": "Unauthorized"}), ok())
    w = worker(db, clock, notifier)
    assert w.run_once() == [(1, "dead")]
    assert row(db)["last_error"] == "needs operator action: HTTP 401, error 401: Unauthorized"
    clock.advance(3600)
    assert w.run_once() == []  # dead rows are never retried by themselves
    assert w.retry_dead(1) and not w.retry_dead(1)
    assert w.run_once() == [(1, "sent")]


def test_crash_after_send_retries_after_the_lease_and_flags_a_possible_duplicate(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "sentinel.db"
    db = Database.open(path)
    open_incident(db, clock)
    first = Scripted("telegram")
    [message] = worker(db, clock, first, worker_id="before-crash").lease(10)
    first.send(message)  # Telegram accepted it ... and the process died before complete()
    db.close()

    db = Database.open(path)  # restart
    try:
        second = Scripted("telegram")
        w = worker(db, clock, second, worker_id="after-restart", lease_s=60.0)
        clock.advance(59)
        assert w.run_once() == []  # the lease has not expired
        clock.advance(2)
        assert w.run_once() == [(1, "sent")]
        r = row(db)
        assert (r["attempts"], r["ambiguous"], len(second.sent)) == (2, 1, 1)
        assert attempts(db) == [(1, "before-crash", "abandoned"), (2, "after-restart", "sent")]
        assert f"Incident {message.incident_id}" in second.sent[0].text  # the duplicate is recognisable
    finally:
        db.close()


def test_a_worker_that_lost_its_lease_cannot_overwrite_the_outcome(db: Database, clock: FakeClock) -> None:
    open_incident(db, clock)
    slow = worker(db, clock, Scripted("telegram"), worker_id="slow", lease_s=60.0)
    [message] = slow.lease(10)
    clock.advance(61)
    fast = worker(db, clock, Scripted("telegram"), worker_id="fast", lease_s=60.0)
    assert fast.run_once() == [(1, "sent")]
    late = slow.complete(message, DeliveryResult(DeliveryStatus.RETRY, "HTTP 502"))
    assert late == "lease lost" and row(db)["status"] == "sent"
    assert attempts(db) == [(1, "slow", "late: retry"), (2, "fast", "sent")]


def test_attempts_and_age_are_bounded(db: Database, clock: FakeClock) -> None:
    open_incident(db, clock, signal(1), signal(2, zone="garage"))
    w = worker(db, clock, Scripted("telegram", *[RETRY] * 10), max_attempts=3, backoff_base_s=1.0, backoff_max_s=1.0)
    outcomes = []
    for _ in range(4):
        outcomes += [o for i, o in w.run_once() if i == 1]
        clock.advance(2)
    assert outcomes == ["retry", "retry", "dead"]
    assert row(db, 1)["last_error"].startswith("gave up after 3 attempts")
    # An operator retry gives the row a fresh budget: it is attempted again, not dead at once.
    assert w.retry_dead(1)
    assert [o for i, o in w.run_once() if i == 1] == ["retry"] and row(db, 1)["attempts"] == 1
    aged = worker(db, clock, Scripted("telegram", RETRY), max_age_s=60.0, backoff_base_s=1.0)
    clock.advance(120)
    db.connection.execute("UPDATE outbox SET status = 'pending', attempts = 0 WHERE outbox_id = 2")
    assert aged.run_once() == []  # dead-lettered at lease time: older than max_age_s
    assert row(db, 2)["status"] == "dead"


def test_urgent_rows_first_but_old_rows_are_not_starved(db: Database, clock: FakeClock) -> None:
    every = NotificationsConfig(channels=["telegram"], min_severity="info")
    incidents = IncidentService(db, clock, incidents=IncidentsConfig(), notifications=every)

    def next_sent() -> str:
        fake = Scripted("telegram")
        worker(db, clock, fake, starvation_s=300.0).run_once(limit=1)
        return fake.sent[0].text.splitlines()[0]

    incidents.record(signal(1, Severity.INFO, "hall"))
    clock.advance(10)
    incidents.record(signal(2, Severity.CRITICAL, "door"))
    assert next_sent() == "[Sentinel] CRITICAL: Person in restricted zone 'door'"  # newer but urgent
    assert next_sent() == "[Sentinel] INFO: Person in restricted zone 'hall'"

    incidents.record(signal(3, Severity.INFO, "porch"))
    clock.advance(301)  # the info row has now waited longer than starvation_s
    incidents.record(signal(4, Severity.CRITICAL, "gate"))
    assert next_sent() == "[Sentinel] INFO: Person in restricted zone 'porch'"
    assert next_sent() == "[Sentinel] CRITICAL: Person in restricted zone 'gate'"


def test_channels_fail_independently_and_unconfigured_channels_wait(db: Database, clock: FakeClock) -> None:
    open_incident(db, clock)
    stamp = row(db)["created_utc"]
    for channel in ("webhook", "mqtt"):
        db.connection.execute(
            """INSERT INTO outbox (incident_id, channel, policy_revision, message_kind, priority, status,
                   next_attempt_utc, budget_start_utc, created_utc, updated_utc)
               SELECT incident_id, ?, 'p', 'opened', 1, 'pending', ?, ?, ?, ? FROM outbox WHERE outbox_id = 1""",
            (channel, stamp, stamp, stamp, stamp),
        )
    broken = Scripted("telegram", DeliveryResult(DeliveryStatus.PERMANENT, "HTTP 401"))
    w = OutboxWorker(db, clock, {"telegram": broken, "webhook": Scripted("webhook")}, SETTINGS, worker_id="w", rng=random.Random(1))
    assert sorted(w.run_once()) == [(1, "dead"), (2, "sent")]
    assert row(db, 3)["status"] == "pending" and row(db, 3)["attempts"] == 0  # no mqtt notifier: untouched


def test_outage_survives_a_restart_and_delivers_after_recovery(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "sentinel.db"
    db = Database.open(path)
    open_incident(db, clock)
    notifier, _ = telegram(urllib.error.URLError(ConnectionRefusedError()), urllib.error.URLError(OSError("unreachable")))
    w = worker(db, clock, notifier, backoff_base_s=5.0)
    for _ in range(2):
        assert w.run_once() == [(1, "retry")]
        clock.advance(10)
    db.close()  # the device restarts during the outage; nothing lives only in memory
    db = Database.open(path)
    try:
        notifier, _ = telegram(ok(7))
        clock.advance(30)
        assert worker(db, clock, notifier).run_once() == [(1, "sent")]
        assert row(db)["attempts"] == 3 and row(db)["provider_message_id"] == "7"
    finally:
        db.close()


def test_a_notifier_bug_does_not_stop_the_worker_and_secrets_are_redacted(db: Database, clock: FakeClock) -> None:
    open_incident(db, clock, signal(1), signal(2, zone="garage"))
    leaky = DeliveryResult(DeliveryStatus.RETRY, f"error at https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage")
    fake = Scripted("telegram", RuntimeError("boom"), leaky)
    assert sorted(worker(db, clock, fake).run_once()) == [(1, "retry"), (2, "retry")]
    stored = " ".join(str(v) for r in db.connection.execute("SELECT * FROM outbox") for v in r)
    stored += " ".join(str(v) for r in db.connection.execute("SELECT * FROM delivery_attempts") for v in r)
    assert "FAKE-test-token" not in stored and "notifier error: RuntimeError" in stored and "bot<redacted>" in stored
