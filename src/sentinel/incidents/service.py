"""The incident transaction (guide chapter 10; V2-14, demo form).

``record()`` turns one rule observation into durable state in one SQLite
transaction:

    BEGIN IMMEDIATE
      deduplicate the observation ID (a repeated observation changes nothing)
      create the incident, or join the correlated open one with a revision check
      append evidence and, for a new incident, the opening transition
      insert outbox rows, UNIQUE(incident, channel, policy_revision, message_kind)
    COMMIT

Only after it returns may the caller acknowledge the observation. If the
process dies after COMMIT but before that, the retried observation is a
duplicate and creates nothing new.

Correlation: an ENTERED observation joins the latest unresolved incident with
the same camera, rule kind and zone when that incident's latest observation is
from the same boot and at most ``merge_window`` earlier on the monotonic clock
(tracker ID switches and several people make several episodes of one
condition). Otherwise it opens a new incident, linked to the previous one with
that key. Joining never adds severities together: the incident keeps the
highest severity it has seen, and a rise is notified once per level. An ENDED
observation is evidence on the incident its episode joined; it does not change
the incident's status (acknowledged is not resolved, guide ch. 9).

Lifecycle (guide ch. 9): candidate -> open -> acknowledged -> resolved, and
dismissed from any unresolved state. Rule observations open incidents
directly: their persistence already happened in the rule. Operators move
incidents with an expected revision, so a stale screen cannot overwrite a newer
change.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..config import IncidentsConfig, NotificationsConfig
from ..media.clock import Clock, require_utc
from ..rules.scene_hazard import Severity
from ..storage.database import Database
from .signals import SEVERITY_RANK, IncidentSignal, SignalPhase

UNRESOLVED = ("candidate", "open", "acknowledged")
PRIORITY = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}  # lower is more urgent


class IncidentStatus(str, Enum):
    CANDIDATE = "candidate"
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


ALLOWED_TRANSITIONS: dict[IncidentStatus, frozenset[IncidentStatus]] = {
    IncidentStatus.CANDIDATE: frozenset({IncidentStatus.OPEN, IncidentStatus.DISMISSED}),
    IncidentStatus.OPEN: frozenset({IncidentStatus.ACKNOWLEDGED, IncidentStatus.RESOLVED, IncidentStatus.DISMISSED}),
    IncidentStatus.ACKNOWLEDGED: frozenset({IncidentStatus.RESOLVED, IncidentStatus.DISMISSED}),
    IncidentStatus.RESOLVED: frozenset(),
    IncidentStatus.DISMISSED: frozenset(),
}


class IncidentError(Exception):
    pass


class UnknownIncident(IncidentError):
    pass


class RevisionConflict(IncidentError):
    """The incident changed since the caller read it; reload and decide again."""


class InvalidTransition(IncidentError):
    pass


class RecordOutcome(str, Enum):
    CREATED = "created"
    JOINED = "joined"
    ENDED_NOTED = "ended_noted"
    DUPLICATE = "duplicate"  # seen before: nothing changed
    IGNORED = "ignored"  # an ENDED whose episode never reached an incident


@dataclass(frozen=True)
class RecordResult:
    outcome: RecordOutcome
    incident_id: str | None
    revision: int | None
    outbox_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class NotificationPolicy:
    channels: tuple[str, ...]
    min_severity: Severity
    revision: str

    @classmethod
    def from_config(cls, config: NotificationsConfig) -> NotificationPolicy:
        return cls(tuple(config.channels), Severity(config.min_severity), config.revision)

    def notifies(self, severity: Severity) -> bool:
        return SEVERITY_RANK[severity] >= SEVERITY_RANK[self.min_severity]


@dataclass(frozen=True)
class Incident:
    incident_id: str
    camera_id: str
    kind: str
    zone_id: str | None
    severity: Severity
    status: IncidentStatus
    revision: int
    title: str
    linked_incident_id: str | None
    first_observed_utc: str
    last_observed_utc: str


def utc_text(value: datetime) -> str:
    """Fixed-width UTC text, so it sorts in time order."""
    return require_utc(value).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class IncidentService:
    def __init__(
        self,
        db: Database,
        clock: Clock,
        *,
        incidents: IncidentsConfig,
        notifications: NotificationsConfig,
        new_id: Callable[[datetime], str] | None = None,
    ) -> None:
        self._db = db
        self._clock = clock
        self._merge_window_ns = incidents.merge_window_ns
        self._policy = NotificationPolicy.from_config(notifications)
        self._new_id = new_id or (lambda now: f"inc-{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}")

    # ------------------------------------------------------------ rule input

    def record(self, signal: IncidentSignal) -> RecordResult:
        now = self._clock.utc_now()
        with self._db.write() as c:
            seen = c.execute(
                "SELECT incident_id FROM observations WHERE observation_id = ?", (signal.observation_id,)
            ).fetchone()
            if seen is not None:
                revision = None if seen[0] is None else self._load(c, seen[0]).revision
                return RecordResult(RecordOutcome.DUPLICATE, seen[0], revision)
            if signal.phase is SignalPhase.ENDED:
                return self._note_end(c, signal, now)
            incident = self._correlated(c, signal)
            if incident is None:
                return self._create(c, signal, now)
            return self._join(c, incident, signal, now)

    def _create(self, c: sqlite3.Connection, signal: IncidentSignal, now: datetime) -> RecordResult:
        incident_id = self._new_id(now)
        previous = c.execute(
            "SELECT incident_id FROM incidents WHERE correlation_key = ? ORDER BY created_utc DESC, rowid DESC LIMIT 1",
            (signal.correlation_key,),
        ).fetchone()
        observed, stamp = utc_text(signal.observed_utc), utc_text(now)
        c.execute(
            """INSERT INTO incidents (incident_id, camera_id, kind, zone_id, correlation_key, rule_revision,
                   severity, status, revision, title, linked_incident_id, first_observed_utc, last_observed_utc,
                   last_boot_id, last_mono_ns, created_utc, updated_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'open', 1, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                incident_id, signal.camera_id, signal.kind, signal.zone_id, signal.correlation_key,
                signal.rule_revision, signal.severity.value, signal.title,
                previous[0] if previous else None, observed, observed, signal.boot_id,
                signal.observed_mono_ns, stamp, stamp,
            ),
        )
        self._observation(c, signal, incident_id, RecordOutcome.CREATED, stamp)
        self._evidence(c, signal, incident_id)
        self._transition(c, incident_id, None, IncidentStatus.OPEN, 1, f"rule:{signal.observation_id}", signal.reason, stamp)
        outbox = self._notify(c, incident_id, signal.severity, "opened", stamp)
        return RecordResult(RecordOutcome.CREATED, incident_id, 1, outbox)

    def _join(self, c: sqlite3.Connection, incident: sqlite3.Row, signal: IncidentSignal, now: datetime) -> RecordResult:
        old = Severity(incident["severity"])
        severity = signal.severity if SEVERITY_RANK[signal.severity] > SEVERITY_RANK[old] else old
        stamp = utc_text(now)
        newer = signal.observed_mono_ns >= incident["last_mono_ns"]
        cursor = c.execute(
            """UPDATE incidents SET severity = ?, last_observed_utc = ?, last_mono_ns = ?, updated_utc = ?,
                   revision = revision + 1
               WHERE incident_id = ? AND revision = ?""",
            (
                severity.value,
                utc_text(signal.observed_utc) if newer else incident["last_observed_utc"],
                signal.observed_mono_ns if newer else incident["last_mono_ns"],
                stamp,
                incident["incident_id"],
                incident["revision"],
            ),
        )
        if cursor.rowcount != 1:
            raise RevisionConflict(f"incident {incident['incident_id']} changed during the transaction")
        self._observation(c, signal, incident["incident_id"], RecordOutcome.JOINED, stamp)
        self._evidence(c, signal, incident["incident_id"])
        outbox: tuple[int, ...] = ()
        if severity is not old:
            outbox = self._notify(c, incident["incident_id"], severity, f"escalated-{severity.value}", stamp)
        return RecordResult(RecordOutcome.JOINED, incident["incident_id"], incident["revision"] + 1, outbox)

    def _note_end(self, c: sqlite3.Connection, signal: IncidentSignal, now: datetime) -> RecordResult:
        stamp = utc_text(now)
        row = c.execute(
            "SELECT incident_id FROM incident_evidence WHERE episode_id = ? ORDER BY evidence_seq DESC LIMIT 1",
            (signal.episode_id,),
        ).fetchone()
        if row is None:
            self._observation(c, signal, None, RecordOutcome.IGNORED, stamp)
            return RecordResult(RecordOutcome.IGNORED, None, None)
        incident = self._load(c, row[0])
        cursor = c.execute(
            "UPDATE incidents SET updated_utc = ?, revision = revision + 1 WHERE incident_id = ? AND revision = ?",
            (stamp, incident.incident_id, incident.revision),
        )
        if cursor.rowcount != 1:
            raise RevisionConflict(f"incident {incident.incident_id} changed during the transaction")
        self._observation(c, signal, incident.incident_id, RecordOutcome.ENDED_NOTED, stamp)
        self._evidence(c, signal, incident.incident_id)
        return RecordResult(RecordOutcome.ENDED_NOTED, incident.incident_id, incident.revision + 1)

    def _correlated(self, c: sqlite3.Connection, signal: IncidentSignal) -> sqlite3.Row | None:
        cursor = c.cursor()
        cursor.row_factory = sqlite3.Row
        row = cursor.execute(
            f"""SELECT * FROM incidents WHERE correlation_key = ? AND status IN ({",".join("?" * len(UNRESOLVED))})
                ORDER BY created_utc DESC, rowid DESC LIMIT 1""",
            (signal.correlation_key, *UNRESOLVED),
        ).fetchone()
        if row is None or row["last_boot_id"] != signal.boot_id:
            return None  # monotonic times from another boot are not comparable: open a new incident
        if signal.observed_mono_ns - row["last_mono_ns"] > self._merge_window_ns:
            return None
        return row

    @staticmethod
    def _observation(
        c: sqlite3.Connection, signal: IncidentSignal, incident_id: str | None, outcome: RecordOutcome, stamp: str
    ) -> None:
        c.execute(
            "INSERT INTO observations (observation_id, incident_id, outcome, recorded_utc) VALUES (?, ?, ?, ?)",
            (signal.observation_id, incident_id, outcome.value, stamp),
        )

    @staticmethod
    def _evidence(c: sqlite3.Connection, signal: IncidentSignal, incident_id: str) -> None:
        c.execute(
            """INSERT INTO incident_evidence (incident_id, observation_id, episode_id, kind, phase, severity,
                   observed_utc, reason, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                incident_id, signal.observation_id, signal.episode_id, signal.kind, signal.phase.value,
                signal.severity.value, utc_text(signal.observed_utc), signal.reason, signal.payload,
            ),
        )

    @staticmethod
    def _transition(
        c: sqlite3.Connection,
        incident_id: str,
        old: IncidentStatus | None,
        new: IncidentStatus,
        revision: int,
        actor: str,
        reason: str,
        stamp: str,
    ) -> None:
        c.execute(
            """INSERT INTO incident_transitions (incident_id, from_status, to_status, revision, actor, reason, at_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (incident_id, old.value if old else None, new.value, revision, actor, reason, stamp),
        )

    def _notify(
        self, c: sqlite3.Connection, incident_id: str, severity: Severity, message_kind: str, stamp: str
    ) -> tuple[int, ...]:
        if not self._policy.notifies(severity):
            return ()
        ids = []
        for channel in self._policy.channels:
            cursor = c.execute(
                """INSERT OR IGNORE INTO outbox (incident_id, channel, policy_revision, message_kind, priority,
                       status, next_attempt_utc, budget_start_utc, created_utc, updated_utc)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)""",
                (incident_id, channel, self._policy.revision, message_kind, PRIORITY[severity], stamp, stamp, stamp, stamp),
            )
            if cursor.rowcount == 1:
                ids.append(cursor.lastrowid)
        return tuple(ids)

    # ------------------------------------------------------------ operator actions

    def transition(
        self, incident_id: str, to: IncidentStatus, *, expected_revision: int, actor: str, reason: str
    ) -> int:
        """Move an incident; returns its new revision."""
        stamp = utc_text(self._clock.utc_now())
        with self._db.write() as c:
            incident = self._load(c, incident_id)
            if incident.revision != expected_revision:
                raise RevisionConflict(
                    f"incident {incident_id} is at revision {incident.revision}, not {expected_revision}"
                )
            if to not in ALLOWED_TRANSITIONS[incident.status]:
                raise InvalidTransition(f"cannot move incident {incident_id} from {incident.status.value} to {to.value}")
            cursor = c.execute(
                "UPDATE incidents SET status = ?, revision = revision + 1, updated_utc = ? WHERE incident_id = ? AND revision = ?",
                (to.value, stamp, incident_id, expected_revision),
            )
            if cursor.rowcount != 1:
                raise RevisionConflict(f"incident {incident_id} changed during the transaction")
            self._transition(c, incident_id, incident.status, to, expected_revision + 1, actor[:64], reason[:300], stamp)
        return expected_revision + 1

    # ------------------------------------------------------------ reads

    def incident(self, incident_id: str) -> Incident:
        with self._db.read() as c:
            return self._load(c, incident_id)

    @staticmethod
    def _load(c: sqlite3.Connection, incident_id: str) -> Incident:
        row = c.execute(
            """SELECT incident_id, camera_id, kind, zone_id, severity, status, revision, title, linked_incident_id,
                      first_observed_utc, last_observed_utc FROM incidents WHERE incident_id = ?""",
            (incident_id,),
        ).fetchone()
        if row is None:
            raise UnknownIncident(f"no incident {incident_id}")
        return Incident(
            incident_id=row[0], camera_id=row[1], kind=row[2], zone_id=row[3], severity=Severity(row[4]),
            status=IncidentStatus(row[5]), revision=row[6], title=row[7], linked_incident_id=row[8],
            first_observed_utc=row[9], last_observed_utc=row[10],
        )
