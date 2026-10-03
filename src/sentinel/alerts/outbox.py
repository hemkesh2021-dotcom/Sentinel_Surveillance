"""Leased outbox worker (guide chapter 10; V2-15, demo form).

Delivery is **at least once**. A worker leases due rows in one short
transaction (committing the lease), sends each message outside any
transaction, then records the outcome in another, but only while it still
holds the lease. A worker that dies keeps its rows leased until the lease
expires; another worker then retries them and marks the row *ambiguous*,
because the provider may already have accepted the message. Every message
carries its incident ID so a duplicate is recognisable. Nothing here claims
exactly-once delivery.

Order: more urgent rows first, but a row that has been due for longer than
``starvation_s`` is served before newer urgent ones, so low-severity
notifications are not starved. Each channel is independent: a failing channel
never holds back another. Transient failures back off exponentially with
jitter, at least as long as the provider asked (``retry_after``). Rows that run
out of attempts or age, and permanent failures (bad token or chat, blocked
bot), become *dead* and wait for an operator retry. Every error stored or shown
is redacted and bounded.

Lease and retry times are UTC wall-clock times, because they must survive a
restart (monotonic time does not). A wall-clock step can expire a lease early
or late; the result is a retry, which at-least-once delivery already allows.
"""

from __future__ import annotations

import random
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Protocol

from ..config import NotificationsConfig
from ..incidents.service import utc_text
from ..media.clock import Clock
from ..redaction import redact_line
from ..storage.database import Database

MAX_TEXT_CHARS = 4096  # Telegram's message limit
DETAIL_CHARS = 200


class DeliveryStatus(str, Enum):
    SENT = "sent"  # the provider confirmed acceptance
    RETRY = "retry"  # transient: try again later
    PERMANENT = "permanent"  # needs operator action (credentials, destination, blocked)


@dataclass(frozen=True)
class DeliveryResult:
    status: DeliveryStatus
    detail: str  # redacted by the notifier; redacted again before storage
    retry_after_s: float | None = None
    provider_message_id: str | None = None
    ambiguous: bool = False  # the provider may have accepted it although we could not confirm


@dataclass(frozen=True)
class OutboxMessage:
    outbox_id: int
    incident_id: str
    channel: str
    message_kind: str
    attempt: int
    text: str


class Notifier(Protocol):
    channel: str

    def send(self, message: OutboxMessage) -> DeliveryResult: ...


def render(row: dict[str, object], message_kind: str) -> str:
    """Plain text; no images, footage or identity data (guide ch. 10: images only by policy)."""
    severity = str(row["severity"]).upper()
    if message_kind == "opened":
        headline = f"[Sentinel] {severity}: {row['title']}"
    elif message_kind.startswith("escalated-"):
        headline = f"[Sentinel] escalated to {severity}: {row['title']}"
    else:
        headline = f"[Sentinel] {row['title']}"
    lines = [
        headline,
        f"Incident {row['incident_id']} ({row['status']}) on camera {row['camera_id']}",
        f"First seen {row['first_observed_utc']}, latest {row['last_observed_utc']} (UTC)",
        "Delivery is at least once: a repeat with the same incident ID is the same incident.",
    ]
    return "\n".join(lines)[:MAX_TEXT_CHARS]


class OutboxWorker:
    def __init__(
        self,
        db: Database,
        clock: Clock,
        notifiers: dict[str, Notifier],
        settings: NotificationsConfig,
        *,
        worker_id: str | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._db = db
        self._clock = clock
        self._notifiers = dict(notifiers)
        self._settings = settings
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:12]}"
        self._rng = rng or random.Random()
        self.counters: Counter[str] = Counter()

    # ------------------------------------------------------------ one pass

    def run_once(self, limit: int = 10) -> list[tuple[int, str]]:
        """Lease due rows, send them and record outcomes; returns (outbox_id, outcome) pairs."""
        results = []
        for message in self.lease(limit):
            notifier = self._notifiers[message.channel]
            try:
                result = notifier.send(message)
            except Exception as exc:  # noqa: BLE001 - a notifier bug must not stop the worker
                result = DeliveryResult(DeliveryStatus.RETRY, f"notifier error: {type(exc).__name__}", ambiguous=True)
            results.append((message.outbox_id, self.complete(message, result)))
        return results

    # ------------------------------------------------------------ leasing

    def lease(self, limit: int) -> list[OutboxMessage]:
        if not self._notifiers:
            return []
        now = self._clock.utc_now()
        stamp = utc_text(now)
        aged = utc_text(now - timedelta(seconds=self._settings.starvation_s))
        expires = utc_text(now + timedelta(seconds=self._settings.lease_s))
        channels = sorted(self._notifiers)
        messages = []
        with self._db.write() as c:
            rows = c.execute(
                f"""SELECT o.outbox_id, o.incident_id, o.channel, o.message_kind, o.status, o.attempts,
                           o.budget_start_utc, i.severity, i.title, i.status, i.camera_id,
                           i.first_observed_utc, i.last_observed_utc
                    FROM outbox o JOIN incidents i ON i.incident_id = o.incident_id
                    WHERE o.channel IN ({",".join("?" * len(channels))})
                      AND ((o.status = 'pending' AND o.next_attempt_utc <= ?)
                           OR (o.status = 'leased' AND o.lease_expires_utc <= ?))
                    ORDER BY CASE WHEN o.next_attempt_utc <= ? THEN -1 ELSE o.priority END,
                             o.next_attempt_utc, o.outbox_id
                    LIMIT ?""",
                (*channels, stamp, stamp, aged, limit),
            ).fetchall()
            for (outbox_id, incident_id, channel, kind, status, attempts, budget_start, severity, title,
                 incident_status, camera_id, first_seen, last_seen) in rows:
                if status == "leased":
                    # Its worker died or stalled after leasing: the provider may have the message.
                    c.execute(
                        "UPDATE delivery_attempts SET outcome = 'abandoned', finished_utc = ? "
                        "WHERE outbox_id = ? AND outcome = 'started'",
                        (stamp, outbox_id),
                    )
                    c.execute("UPDATE outbox SET ambiguous = 1 WHERE outbox_id = ?", (outbox_id,))
                    self.counters["lease_expired"] += 1
                if self._expired(attempts, budget_start, now):
                    self._dead(c, outbox_id, "gave up: attempts or age exhausted", stamp)
                    continue
                attempt = attempts + 1
                c.execute(
                    """UPDATE outbox SET status = 'leased', lease_owner = ?, lease_expires_utc = ?, attempts = ?,
                           updated_utc = ? WHERE outbox_id = ?""",
                    (self.worker_id, expires, attempt, stamp, outbox_id),
                )
                c.execute(
                    """INSERT INTO delivery_attempts (outbox_id, attempt, lease_owner, started_utc, outcome)
                       VALUES (?, ?, ?, ?, 'started')""",
                    (outbox_id, attempt, self.worker_id, stamp),
                )
                row = {
                    "incident_id": incident_id, "severity": severity, "title": title, "status": incident_status,
                    "camera_id": camera_id, "first_observed_utc": first_seen, "last_observed_utc": last_seen,
                }
                messages.append(OutboxMessage(outbox_id, incident_id, channel, kind, attempt, render(row, kind)))
        return messages

    def _expired(self, attempts: int, budget_start: str, now: datetime) -> bool:
        too_old = budget_start < utc_text(now - timedelta(seconds=self._settings.max_age_s))
        return attempts >= self._settings.max_attempts or too_old

    # ------------------------------------------------------------ outcomes

    def complete(self, message: OutboxMessage, result: DeliveryResult) -> str:
        """Record the outcome if this worker still holds the lease; returns the row's new state."""
        now = self._clock.utc_now()
        stamp = utc_text(now)
        detail = redact_line(result.detail or result.status.value, DETAIL_CHARS)
        with self._db.write() as c:
            held = c.execute(
                "SELECT attempts, budget_start_utc FROM outbox WHERE outbox_id = ? AND status = 'leased' AND lease_owner = ?",
                (message.outbox_id, self.worker_id),
            ).fetchone()
            outcome = result.status.value + (" (ambiguous)" if result.ambiguous else "")
            c.execute(
                """UPDATE delivery_attempts SET outcome = ?, finished_utc = ?, detail = ?
                   WHERE outbox_id = ? AND attempt = ? AND lease_owner = ?""",
                (outcome if held else f"late: {outcome}", stamp, detail, message.outbox_id, message.attempt, self.worker_id),
            )
            if held is None:
                self.counters["late_outcome"] += 1
                return "lease lost"
            attempts, budget_start = held
            if result.ambiguous:
                c.execute("UPDATE outbox SET ambiguous = 1 WHERE outbox_id = ?", (message.outbox_id,))
            if result.status is DeliveryStatus.SENT:
                c.execute(
                    """UPDATE outbox SET status = 'sent', provider_message_id = ?, lease_owner = NULL,
                           lease_expires_utc = NULL, last_error = NULL, updated_utc = ? WHERE outbox_id = ?""",
                    (result.provider_message_id, stamp, message.outbox_id),
                )
                self.counters["sent"] += 1
                return "sent"
            if result.status is DeliveryStatus.PERMANENT:
                self._dead(c, message.outbox_id, f"needs operator action: {detail}", stamp)
                return "dead"
            if self._expired(attempts, budget_start, now):
                self._dead(c, message.outbox_id, f"gave up after {attempts} attempts: {detail}", stamp)
                return "dead"
            delay = self._backoff(attempts, result.retry_after_s)
            c.execute(
                """UPDATE outbox SET status = 'pending', next_attempt_utc = ?, last_error = ?, lease_owner = NULL,
                       lease_expires_utc = NULL, updated_utc = ? WHERE outbox_id = ?""",
                (utc_text(now + timedelta(seconds=delay)), detail, stamp, message.outbox_id),
            )
            self.counters["retry"] += 1
            return "retry"

    def _backoff(self, attempts: int, retry_after_s: float | None) -> float:
        base = min(self._settings.backoff_max_s, self._settings.backoff_base_s * 2 ** (attempts - 1))
        delay = base * (0.5 + 0.5 * self._rng.random())  # jitter in [base/2, base)
        if retry_after_s is not None:
            delay = max(delay, retry_after_s)  # never earlier than the provider asked
        return delay

    def _dead(self, c, outbox_id: int, reason: str, stamp: str) -> None:
        c.execute(
            """UPDATE outbox SET status = 'dead', last_error = ?, lease_owner = NULL, lease_expires_utc = NULL,
                   updated_utc = ? WHERE outbox_id = ?""",
            (redact_line(reason, DETAIL_CHARS), stamp, outbox_id),
        )
        self.counters["dead"] += 1

    # ------------------------------------------------------------ operator action

    def retry_dead(self, outbox_id: int) -> bool:
        """Operator retry of a dead row: a fresh attempt budget, due now."""
        stamp = utc_text(self._clock.utc_now())
        with self._db.write() as c:
            cursor = c.execute(
                """UPDATE outbox SET status = 'pending', attempts = 0, next_attempt_utc = ?, budget_start_utc = ?,
                       updated_utc = ? WHERE outbox_id = ? AND status = 'dead'""",
                (stamp, stamp, stamp, outbox_id),
            )
            return cursor.rowcount == 1
