"""Incident snapshots (D70; opt-in: ``sentinel run --snapshots``).

One JPEG per incident: the frame that the observation opening it came from.

- **The triggering frame, or nothing.** A zone incident's frame is its ENTERED
  observation's ``last_source`` (the in-zone detection that met the rule's
  threshold); a hazard candidate's is its ``last_source`` (the last scene
  report's frame). When the runtime queues the observation, it looks that
  frame up by identity among the images it still holds, and keeps that one
  image until the incident is recorded. If the frame is gone, the snapshot is
  recorded as missing with the reason; no other frame is ever substituted.
- **Recorded.** ``incident_snapshots`` (schema 3) has one row per incident:
  saved, missing or pruned; the reason; the frame's identity (camera, boot,
  run, stream epoch, sequence); its ingest time and source time quality; the
  observation; the stored size and SHA-256. A JPEG comment in the file names
  the incident, frame and time, so a downloaded copy describes itself.
- **Written safely.** In the snapshot directory (0700, not a symlink, opened
  once and used by descriptor): a temporary file (0600) is written and
  fsynced, renamed into place, the directory fsynced, then the row committed.
  At startup, temporary files and files without a saved row (a crash between
  the rename and the commit) are removed; other files are counted and left.
- **Bounded.** A picture is scaled down to fit ``max_width`` x ``max_height``
  and refused above ``max_bytes``. Retention, from guide ch. 14's starting
  values: saved snapshots older than ``retention_s`` are deleted, the oldest
  go first to stay within ``quota_bytes`` and ``max_count``, and nothing is
  saved if the volume's free space would fall below the greater of
  ``reserve_min_bytes`` and ``reserve_fraction`` of the volume. Each deletion
  is kept on its row (pruned, with the reason and time). Retention ages use
  UTC wall time, since they span reboots: a clock step moves them.
- **Off the loop.** SnapshotWriter runs the store on its own thread. The
  runtime loop only queues jobs, without waiting: queued jobs hold the images
  of at most MAX_HELD_FRAMES distinct frames (jobs of one frame share its
  image; a job beyond that is recorded as missing, ``queue_full``), and at
  most MAX_JOBS wait in all (more are counted and get no row).

Portable: the default encoder imports cv2 only when it runs (``media.jpeg``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import secrets
import shutil
import stat
import threading
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from ..contracts import FrameKey, FrameRef
from ..media.clock import NS_PER_SECOND, Clock
from ..media.jpeg import EncodeError, Encoder, encode_jpeg, with_comment
from ..storage.database import Database
from .service import utc_text

SNAPSHOT_DIR = "snapshots"
FILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")  # the incident IDs that can name a file
FILE_NAME = re.compile(r"snap-[A-Za-z0-9][A-Za-z0-9_-]{0,127}\.jpg")
TEMP_NAME = re.compile(r"\.snap-[A-Za-z0-9][A-Za-z0-9_-]{0,127}\.[0-9a-f]{8}\.tmp")
MAX_HELD_FRAMES = 4  # distinct frames whose images queued jobs hold (the one being saved included)
MAX_JOBS = 64  # queued jobs in all
STOP_TIMEOUT_S = 5.0
WAIT_S = 1.0  # how often the idle writer checks for the stop and for retention


def file_name(incident_id: str) -> str:
    return f"snap-{incident_id}.jpg"


@dataclass(frozen=True)
class SnapshotPolicy:
    """Bounds and retention (D70). The defaults are guide ch. 14's starting values, not measured choices."""

    retention_s: float = 7 * 86400.0  # media: 7 days to begin with
    quota_bytes: int = 200_000_000  # decimal bytes, all saved snapshots together
    max_count: int = 2000
    reserve_min_bytes: int = 1_000_000_000  # keep the greater of 1 GB ...
    reserve_fraction: float = 0.10  # ... and 10 % of the volume free
    max_bytes: int = 2_000_000  # one snapshot
    max_width: int = 1920
    max_height: int = 1080
    jpeg_quality: int = 85
    prune_interval_s: float = 3600.0  # retention runs at startup, before each save and this often

    def __post_init__(self) -> None:
        if not self.retention_s > 0 or not self.prune_interval_s > 0:
            raise ValueError("retention_s and prune_interval_s must be positive")
        if min(self.quota_bytes, self.max_count, self.max_bytes, self.max_width, self.max_height) < 1:
            raise ValueError("quota_bytes, max_count, max_bytes, max_width and max_height must be positive")
        if self.max_bytes > self.quota_bytes:
            raise ValueError("max_bytes cannot exceed quota_bytes")
        if self.reserve_min_bytes < 0 or not 0 <= self.reserve_fraction <= 0.5:
            raise ValueError("reserve_min_bytes must be >= 0 and reserve_fraction within [0, 0.5]")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be within [1, 100]")

    def describe(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SnapshotJob:
    """One created incident's snapshot: the observation's source frame, and its image if it was still held."""

    incident_id: str
    observation_id: str
    key: FrameKey  # the triggering observation's source frame
    frame: FrameRef | None  # that frame's record, when still known
    image: Any | None  # that frame's image, when still held; never another frame's
    missing: str | None = None  # why there is no image (a fixed label)


class SnapshotDirectoryError(Exception):
    """The snapshot directory cannot be used; ``label`` says why."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


class SnapshotStore:
    """Saves snapshots and applies retention. One thread uses it (the writer's, after ``open()``); status() is
    thread-safe."""

    def __init__(
        self,
        db: Database,
        directory: Path,
        clock: Clock,
        policy: SnapshotPolicy | None = None,
        *,
        encode: Encoder = encode_jpeg,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    ) -> None:
        self._db = db
        self._dir = Path(directory)
        self._clock = clock
        self.policy = policy or SnapshotPolicy()
        self._encode = encode
        self._disk_usage = disk_usage
        self._fd: int | None = None
        self._lock = threading.Lock()  # counters and the problem, read by status()
        self._counts: Counter[str] = Counter()
        self._problem: str | None = None

    # ------------------------------------------------------------ lifecycle

    def open(self) -> dict[str, int]:
        """Create or check the directory, remove what an interrupted write left, and apply retention."""
        try:
            try:
                info = os.lstat(self._dir)
            except FileNotFoundError:
                os.mkdir(self._dir, 0o700)
                info = os.lstat(self._dir)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise SnapshotDirectoryError("not_a_directory")
            if info.st_uid != os.getuid():
                raise SnapshotDirectoryError("not_owned_by_this_user")
            if info.st_mode & 0o077:
                os.chmod(self._dir, 0o700)  # private footage: owner only
            self._fd = os.open(self._dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except OSError as exc:
            raise SnapshotDirectoryError(f"unusable:{type(exc).__name__}") from None
        with self._db.read() as c:
            saved = {row[0] for row in c.execute("SELECT file_name FROM incident_snapshots WHERE state = 'saved'")}
        found = Counter()
        for name in os.listdir(self._fd):
            if TEMP_NAME.fullmatch(name) or (FILE_NAME.fullmatch(name) and name not in saved):
                self._unlink(name)
                found["removed_temporary" if name.endswith(".tmp") else "removed_unrecorded"] += 1
            elif not FILE_NAME.fullmatch(name):
                found["unexpected_files"] += 1  # not ours: left alone
        return {key: found[key] for key in ("removed_temporary", "removed_unrecorded", "unexpected_files")} | {
            "pruned": self.prune()}

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"problem": self._problem, "counters": dict(sorted(self._counts.items()))}

    # ------------------------------------------------------------ saving

    def save(self, job: SnapshotJob) -> str:
        """Save or record ``job``'s snapshot; returns saved, missing or duplicate."""
        with self._db.read() as c:
            if c.execute("SELECT 1 FROM incident_snapshots WHERE incident_id = ?", (job.incident_id,)).fetchone():
                self._count("duplicate")
                return "duplicate"
        if job.image is None:
            return self._missing(job, job.missing or "frame_not_retained")
        if job.frame is None or job.frame.key != job.key:
            return self._missing(job, "frame_identity_mismatch")
        if not FILE_ID.fullmatch(job.incident_id):
            return self._missing(job, "incident_id_not_file_safe")
        policy = self.policy
        try:
            encoded = self._encode(job.image, job.frame.pixel_format, max_width=policy.max_width,
                                   max_height=policy.max_height, quality=policy.jpeg_quality)
            data = with_comment(encoded.data, self._comment(job.incident_id, job.observation_id, job.frame))
        except EncodeError as exc:
            return self._missing(job, exc.label)
        except Exception as exc:  # noqa: BLE001 - the class name only, never its text
            return self._missing(job, f"encode_failed:{type(exc).__name__}")
        if len(data) > policy.max_bytes:
            return self._missing(job, "too_large")
        self.prune()
        if not self._make_room(len(data)):
            return self._missing(job, "retention_failed", problem=True)
        usage = self._disk_usage(self._dir)
        reserve = max(policy.reserve_min_bytes, int(policy.reserve_fraction * usage.total))
        if usage.free - len(data) < reserve:
            return self._missing(job, "free_space_reserve", problem=True)
        name = file_name(job.incident_id)
        try:
            self._write(name, data)
        except OSError as exc:
            return self._missing(job, f"write_failed:{type(exc).__name__}", problem=True)
        frame = job.frame
        try:
            with self._db.write() as c:
                c.execute(
                    f"INSERT INTO incident_snapshots ({ROW_COLUMNS}) VALUES ({', '.join('?' * len(ROW_FIELDS))})",
                    (job.incident_id, job.observation_id, "saved", None, *_identity(job.key), utc_text(frame.ingest_utc),
                     frame.source_time_quality.value, frame.source_pts, encoded.width, encoded.height, name, len(data),
                     hashlib.sha256(data).hexdigest(), utc_text(self._clock.utc_now()), None),
                )
        except BaseException:
            self._unlink(name)  # no row: the file must not outlive it
            raise
        with self._lock:
            self._counts["saved"] += 1
            self._problem = None
        return "saved"

    @staticmethod
    def _comment(incident_id: str, observation_id: str, frame: FrameRef) -> str:
        return (f"Sentinel incident {incident_id}; camera {frame.camera_id}; stream epoch {frame.stream_epoch} frame "
                f"{frame.frame_seq}; ingest {utc_text(frame.ingest_utc)} (when the device received the frame; "
                f"source time quality {frame.source_time_quality.value}); observation {observation_id}")[:512]

    def _missing(self, job: SnapshotJob, reason: str, *, problem: bool = False) -> str:
        frame = job.frame if job.frame is not None and job.frame.key == job.key else None
        with self._db.write() as c:
            c.execute(
                f"INSERT INTO incident_snapshots ({ROW_COLUMNS}) VALUES ({', '.join('?' * len(ROW_FIELDS))})",
                (job.incident_id, job.observation_id, "missing", reason, *_identity(job.key),
                 None if frame is None else utc_text(frame.ingest_utc),
                 None if frame is None else frame.source_time_quality.value, None if frame is None else frame.source_pts,
                 None, None, None, None, None, utc_text(self._clock.utc_now()), None),
            )
        with self._lock:
            self._counts[f"missing:{reason.partition(':')[0]}"] += 1
            if problem:
                self._problem = reason
        return "missing"

    def _write(self, name: str, data: bytes) -> None:
        """``data`` as ``name`` in the directory: a 0600 temporary file, fsynced, renamed, the directory fsynced."""
        temp = f".{name[:-4]}.{secrets.token_hex(4)}.tmp"
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=self._fd)
        try:
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temp, name, src_dir_fd=self._fd, dst_dir_fd=self._fd)
        except BaseException:
            self._unlink(temp)
            raise
        os.fsync(self._fd)  # type: ignore[arg-type]  # the rename is durable too

    # ------------------------------------------------------------ retention

    def prune(self) -> int:
        """Delete saved snapshots older than the retention; returns how many."""
        cutoff = utc_text(self._clock.utc_now() - timedelta(seconds=self.policy.retention_s))
        with self._db.read() as c:
            rows = c.execute(
                """SELECT incident_id, file_name FROM incident_snapshots WHERE state = 'saved' AND recorded_utc < ?
                   ORDER BY recorded_utc, rowid""", (cutoff,)).fetchall()
        pruned = 0
        for incident_id, name in rows:
            if not self._delete(incident_id, name, "retention_age"):
                break
            pruned += 1
        return pruned

    def _make_room(self, new_bytes: int) -> bool:
        """Delete the oldest saved snapshots until one of ``new_bytes`` fits the quota and the count."""
        while True:
            with self._db.read() as c:
                total, count = c.execute(
                    "SELECT coalesce(sum(bytes), 0), count(*) FROM incident_snapshots WHERE state = 'saved'").fetchone()
                if total + new_bytes <= self.policy.quota_bytes and count + 1 <= self.policy.max_count:
                    return True
                oldest = c.execute(
                    """SELECT incident_id, file_name FROM incident_snapshots WHERE state = 'saved'
                       ORDER BY recorded_utc, rowid LIMIT 1""").fetchone()
            if oldest is None or not self._delete(oldest[0], oldest[1], "retention_quota"):
                return False

    def _delete(self, incident_id: str, name: str | None, reason: str) -> bool:
        """Remove one saved snapshot's file, then mark its row pruned; False (nothing marked) if removal failed."""
        if isinstance(name, str) and FILE_NAME.fullmatch(name):  # never a name that could leave the directory
            try:
                os.unlink(name, dir_fd=self._fd)
            except FileNotFoundError:
                pass
            except OSError as exc:
                with self._lock:
                    self._problem = f"prune_failed:{type(exc).__name__}"
                return False
        with self._db.write() as c:
            c.execute("UPDATE incident_snapshots SET state = 'pruned', reason = ?, pruned_utc = ? "
                      "WHERE incident_id = ? AND state = 'saved'", (reason, utc_text(self._clock.utc_now()), incident_id))
        self._count(f"pruned:{reason}")
        return True

    def _unlink(self, name: str) -> None:
        try:
            os.unlink(name, dir_fd=self._fd)
        except OSError:
            pass

    def _count(self, name: str) -> None:
        with self._lock:
            self._counts[name] += 1


ROW_FIELDS = (
    "incident_id", "observation_id", "state", "reason", "camera_id", "boot_id", "run_id", "stream_epoch", "frame_seq",
    "ingest_utc", "source_time_quality", "source_pts", "width", "height", "file_name", "bytes", "sha256",
    "recorded_utc", "pruned_utc",
)
ROW_COLUMNS = ", ".join(ROW_FIELDS)


def _identity(key: FrameKey) -> tuple[str, str, str, int, int]:
    return key.camera_id, key.boot_id, key.run_id, key.stream_epoch, key.frame_seq


class SnapshotWriter:
    """Runs a SnapshotStore on its own thread; the runtime loop only queues (``submit`` never waits)."""

    def __init__(self, store: SnapshotStore, clock: Clock, *, max_held_frames: int = MAX_HELD_FRAMES,
                 max_jobs: int = MAX_JOBS) -> None:
        if not 1 <= max_held_frames <= max_jobs:
            raise ValueError("need 1 <= max_held_frames <= max_jobs")
        self.store = store
        self._clock = clock
        self._max_held_frames = max_held_frames
        self._max_jobs = max_jobs
        self._cond = threading.Condition()  # also guards the counters and the problem
        self._jobs: deque[SnapshotJob] = deque()
        self._held: Counter[FrameKey] = Counter()  # frames whose images queued (or saving) jobs hold
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._counts: Counter[str] = Counter()
        self._problem: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the snapshot writer can be started once")
        self._thread = threading.Thread(target=self._run, name="sentinel-snapshots", daemon=True)
        self._thread.start()

    def submit(self, job: SnapshotJob) -> bool:
        """Queue ``job``; False (counted) if the writer is stopping or MAX_JOBS wait already."""
        with self._cond:
            if self._stopping or self._thread is None or not self._thread.is_alive():  # never queued: no row
                self._counts["not_queued_stopped"] += 1
                return False
            if len(self._jobs) >= self._max_jobs:
                self._counts["not_queued_full"] += 1
                return False
            if job.image is not None:
                if job.key not in self._held and len(self._held) >= self._max_held_frames:
                    job = dataclasses.replace(job, image=None, missing="queue_full")
                else:
                    self._held[job.key] += 1
            self._jobs.append(job)
            self._counts["queued"] += 1
            self._cond.notify()
            return True

    def stop(self, timeout_s: float) -> bool:
        """Save what is queued, then stop; False if the thread is still running after ``timeout_s``."""
        with self._cond:
            self._stopping = True
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                return False
        self.store.close()
        return True

    def status(self) -> dict[str, Any]:
        """For the runtime status: numbers and fixed labels only."""
        thread = self._thread
        state = "not_started" if thread is None else "running" if thread.is_alive() else "stopped"
        store = self.store.status()
        with self._cond:
            return {"state": state, "queued": len(self._jobs), "held_frames": len(self._held),
                    "problem": self._problem or store["problem"], "writer": dict(sorted(self._counts.items())),
                    "store": store["counters"], "policy": self.store.policy.describe()}

    def _run(self) -> None:
        interval_ns = round(self.store.policy.prune_interval_s * NS_PER_SECOND)
        next_prune = self._clock.monotonic_ns() + interval_ns
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._jobs or self._stopping, WAIT_S)
                job = self._jobs.popleft() if self._jobs else None
                if job is None and self._stopping:
                    return
            if job is not None:
                key = job.key if job.image is not None else None
                try:
                    outcome, problem = self.store.save(job), None
                except Exception as exc:  # noqa: BLE001 - a database or disk error must not end the writer
                    outcome, problem = "errors", f"writer_error:{type(exc).__name__}"
                del job  # the image reference ends here
                with self._cond:
                    self._counts[outcome] += 1
                    self._problem = problem
                    if key is not None:
                        self._held[key] -= 1
                        if self._held[key] <= 0:
                            del self._held[key]
            if self._clock.monotonic_ns() >= next_prune:
                next_prune = self._clock.monotonic_ns() + interval_ns
                try:
                    self.store.prune()
                except Exception as exc:  # noqa: BLE001
                    with self._cond:
                        self._problem = f"retention_error:{type(exc).__name__}"
