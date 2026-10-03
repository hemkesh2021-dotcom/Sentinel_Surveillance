"""SQLite database for incidents and the notification outbox (guide chapter 10).

One process owns writes (an exclusive lock file next to the database), and
every write is one short ``BEGIN IMMEDIATE`` transaction. The database runs in
WAL mode so readers (the status page) do not block the writer, with
``synchronous=FULL`` so a committed incident or outbox row survives a power
cut, a busy timeout, and foreign keys on. Migrations are numbered SQL scripts
applied in order; ``PRAGMA user_version`` records the schema version. A
database written by a newer build is refused rather than guessed at.

Standard library only. Keep the database on local storage, never on a NAS.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:  # POSIX (Linux, macOS); the device and the development hosts
    import fcntl
except ImportError:  # pragma: no cover - Windows is not a supported host
    fcntl = None  # type: ignore[assignment]

BUSY_TIMEOUT_MS = 5000

MIGRATIONS: tuple[str, ...] = (
    # 1: incidents, deduplicated observations, evidence, transitions, outbox, delivery attempts
    """
    CREATE TABLE incidents (
        incident_id TEXT PRIMARY KEY,
        camera_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        zone_id TEXT,
        correlation_key TEXT NOT NULL,
        rule_revision TEXT NOT NULL,
        severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
        status TEXT NOT NULL CHECK (status IN ('candidate', 'open', 'acknowledged', 'resolved', 'dismissed')),
        revision INTEGER NOT NULL CHECK (revision >= 1),
        title TEXT NOT NULL,
        linked_incident_id TEXT REFERENCES incidents (incident_id),
        first_observed_utc TEXT NOT NULL,
        last_observed_utc TEXT NOT NULL,
        last_boot_id TEXT NOT NULL,
        last_mono_ns INTEGER NOT NULL,
        created_utc TEXT NOT NULL,
        updated_utc TEXT NOT NULL
    );
    CREATE INDEX incidents_camera_time ON incidents (camera_id, first_observed_utc);
    CREATE INDEX incidents_status_time ON incidents (status, updated_utc);
    CREATE INDEX incidents_correlation ON incidents (correlation_key, status);

    CREATE TABLE observations (
        observation_id TEXT PRIMARY KEY,
        incident_id TEXT REFERENCES incidents (incident_id),
        outcome TEXT NOT NULL,
        recorded_utc TEXT NOT NULL
    );

    CREATE TABLE incident_evidence (
        evidence_seq INTEGER PRIMARY KEY,
        incident_id TEXT NOT NULL REFERENCES incidents (incident_id),
        observation_id TEXT NOT NULL REFERENCES observations (observation_id),
        episode_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        phase TEXT NOT NULL,
        severity TEXT NOT NULL,
        observed_utc TEXT NOT NULL,
        reason TEXT NOT NULL,
        payload TEXT NOT NULL CHECK (length(payload) <= 8192)
    );
    CREATE INDEX evidence_incident ON incident_evidence (incident_id, evidence_seq);
    CREATE INDEX evidence_episode ON incident_evidence (episode_id);

    CREATE TABLE incident_transitions (
        transition_seq INTEGER PRIMARY KEY,
        incident_id TEXT NOT NULL REFERENCES incidents (incident_id),
        from_status TEXT,
        to_status TEXT NOT NULL,
        revision INTEGER NOT NULL,
        actor TEXT NOT NULL,
        reason TEXT NOT NULL,
        at_utc TEXT NOT NULL
    );
    CREATE INDEX transitions_incident ON incident_transitions (incident_id, transition_seq);

    CREATE TABLE outbox (
        outbox_id INTEGER PRIMARY KEY,
        incident_id TEXT NOT NULL REFERENCES incidents (incident_id),
        channel TEXT NOT NULL,
        policy_revision TEXT NOT NULL,
        message_kind TEXT NOT NULL,
        priority INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'leased', 'sent', 'dead')),
        attempts INTEGER NOT NULL DEFAULT 0,
        next_attempt_utc TEXT NOT NULL,
        lease_owner TEXT,
        lease_expires_utc TEXT,
        last_error TEXT,
        ambiguous INTEGER NOT NULL DEFAULT 0,
        provider_message_id TEXT,
        budget_start_utc TEXT NOT NULL,
        created_utc TEXT NOT NULL,
        updated_utc TEXT NOT NULL,
        UNIQUE (incident_id, channel, policy_revision, message_kind)
    );
    CREATE INDEX outbox_due ON outbox (status, next_attempt_utc, priority);

    CREATE TABLE delivery_attempts (
        attempt_seq INTEGER PRIMARY KEY,
        outbox_id INTEGER NOT NULL REFERENCES outbox (outbox_id),
        attempt INTEGER NOT NULL,
        lease_owner TEXT NOT NULL,
        started_utc TEXT NOT NULL,
        finished_utc TEXT,
        outcome TEXT NOT NULL,
        detail TEXT
    );
    CREATE INDEX attempts_outbox ON delivery_attempts (outbox_id, attempt_seq);
    """,
)
SCHEMA_VERSION = len(MIGRATIONS)


class DatabaseError(Exception):
    """The database cannot be used; the message says why."""


class WriterBusy(DatabaseError):
    """Another process already owns writes to this database."""


def _configure(connection: sqlite3.Connection) -> None:
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA foreign_keys = ON")


def migrate(connection: sqlite3.Connection) -> int:
    """Apply pending migrations, each in its own transaction; returns the schema version."""
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise DatabaseError(
            f"database schema version {version} is newer than this build ({SCHEMA_VERSION}); "
            "use a newer Sentinel or restore a matching backup"
        )
    for number in range(version + 1, SCHEMA_VERSION + 1):
        # executescript would commit implicitly; run the statements inside one transaction.
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in _statements(MIGRATIONS[number - 1]):
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {number}")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return SCHEMA_VERSION


def _statements(script: str) -> list[str]:
    statements, current = [], []
    for line in script.splitlines():
        current.append(line)
        candidate = "\n".join(current).strip()
        if candidate.endswith(";") and sqlite3.complete_statement(candidate):
            statements.append(candidate)
            current = []
    if "\n".join(current).strip():
        raise ValueError("migration ends with an incomplete statement")
    return statements


class Database:
    """The single write owner of one database file.

    One connection, shared by the threads of the owning process (the runtime
    loop records incidents, the outbox worker leases and completes deliveries);
    a lock serializes its use, so every transaction is short and none spans
    network I/O.
    """

    def __init__(self, path: Path, connection: sqlite3.Connection, lock_fd: int | None) -> None:
        self.path = path
        self.connection = connection
        self._lock_fd = lock_fd
        self._lock = threading.RLock()

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> Database:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = _acquire_writer_lock(path)
        try:
            connection = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            _configure(connection)
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if mode.lower() != "wal":
                raise DatabaseError(f"cannot enable WAL mode (got {mode!r}); is the database on local storage?")
            connection.execute("PRAGMA synchronous = FULL")
            migrate(connection)
        except BaseException:
            if lock_fd is not None:
                os.close(lock_fd)
            raise
        return cls(path, connection, lock_fd)

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """One short write transaction; rolled back on any exception."""
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise
            self.connection.execute("COMMIT")

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """A consistent read (one deferred transaction)."""
        with self._lock:
            self.connection.execute("BEGIN")
            try:
                yield self.connection
            finally:
                self.connection.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self.connection.close()
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None


def connect_reader(path: str | os.PathLike[str]) -> sqlite3.Connection:
    """A read-only connection for status pages and tools; it never writes or migrates."""
    connection = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, isolation_level=None)
    _configure(connection)
    return connection


def _acquire_writer_lock(path: Path) -> int | None:
    if fcntl is None:  # pragma: no cover
        return None
    fd = os.open(f"{path}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise WriterBusy(f"another process is writing to {path.name}; only one Sentinel runtime may own it") from None
    return fd
