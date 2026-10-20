"""The enrolled-identity gallery on disk (V2-20 demo form for V2-25; guide chapter 11).

A gallery is numbers only: a schema version, the face pipeline's full compatibility record, and per identity an
opaque generated ID (``idn-`` and 12 hex digits), the consent date and scope, and at most MAX_PROTOTYPES unit-length
embeddings. It holds no names, photos or file names. It is sealed with the vault (scrypt and AES-256-GCM, a passphrase
typed at startup) in ``<identity dir>/gallery.sealed``; the compatibility record is also the sealed file's
authenticated header, and the two must agree.

**Compatibility** (``FaceCompatibility``) lists every setting that changes an embedding: the DeepFace version, the
recognition model and its weights' SHA-256, the detector and its weights' SHA-256 (YuNet runs through OpenCV, so its
version too), alignment, normalization, crop expansion, colour order, the dimension, the prototype normalization, and
the TensorFlow, tf_keras and NumPy versions. A gallery is used only by a runtime whose record is equal in every field;
otherwise face recognition is unavailable and names the first difference. An upgrade needs re-enrollment.

The identity directory (default ``~/sentinel-data/identity``, on the second drive through its bind mount) must be
0700 and owned by this user; files are 0600. ``inbox/`` takes the photos for one enrollment; they are deleted after a
successful enrollment. ``audit.jsonl`` records enroll, revoke and purge events with the opaque ID, a UTC time and
counts: no embeddings, names or file names.

**One change at a time.** Enrollment, revoke, purge and capture each hold the directory's change lock (``locked``, an
exclusive ``flock``) from reading the gallery to writing it, so concurrent changes from any process wait their turn
(or are refused, ``gallery_busy``) and none is lost or undone by another.

**Audited before and after.** A change goes through ``commit``: sealed into a staging file beside the gallery, read
back and compared in full, checked against the gallery it was based on, recorded as ``<action>_started``, renamed
over the gallery, then recorded as ``<action>`` (completed). A failed rename adds ``<action>_aborted``; a completed
line that cannot be written undoes the change. A completed line is only ever written for a change in effect, and the
gallery never changes without a started line before it. Readers that count completed events (F1 counts ``enroll``)
are unaffected by the started and aborted lines. Purge records ``purge_started`` and ``purge``.

**Not atomic across a stop.** A crash or power loss between a started line and its outcome leaves the started line
alone, with the gallery either unchanged or changed; only the gallery says which. A torn last line (a stop during an
append) stops every change until an operator deals with it (``audit_log_torn``).

Removal: ``revoke`` rewrites the gallery without the identity; ``purge`` deletes the gallery, any staging or previous
file left by a stop, and the inbox. The runtime reads the gallery once at startup, so either takes effect at the next
start. Deleting files on an SSD is not secure erasure.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, FiniteFloat, PositiveInt, StringConstraints, ValidationError, model_validator

from ..contracts import Contract, UtcDatetime
from .matching import MAX_PROTOTYPES, EnrolledIdentity, Enrollment
from .vault import (
    PRIVATE_DIR_MODE,
    PRIVATE_FILE_MODE,
    Sealer,
    Secret,
    VaultError,
    fsync_directory,
    private_directory_problem,
    private_file_problem,
    read_private,
    replace_private,
    write_private_atomic,
)

SCHEMA = "sentinel-identity-gallery"
SCHEMA_VERSION = 1
GALLERY_FILE = "gallery.sealed"
STAGED_FILE = ".gallery.sealed.staged"  # sealed, not yet verified, audited and renamed over the gallery
PREVIOUS_FILE = ".gallery.sealed.previous"  # the gallery as it was, kept during a commit to undo it
LOCK_FILE = ".gallery.lock"  # empty; flock()ed while a change is under way
LOCK_WAIT_S = 60.0  # a change waits this long for one under way (a commit's sealing is a few seconds), then refuses
LOCK_POLL_S = 0.05
AUDIT_FILE = "audit.jsonl"
INBOX_DIR = "inbox"
DEFAULT_IDENTITY_DIR = Path.home() / "sentinel-data" / "identity"
CONSENT_SCOPE = "enrollment on this device for the bounded Sentinel demo; revocable at any time"
UNIT_NORM_TOLERANCE = 1e-6

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Version = Annotated[str, StringConstraints(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9][A-Za-z0-9.+_-]*$")]
Name = Annotated[str, StringConstraints(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
IdentityId = Annotated[str, StringConstraints(pattern=r"^idn-[0-9a-f]{12}$")]
ConsentDate = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]
CONSENT_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")  # ASCII digits only: never wider than ConsentDate


def check_consent_date(text: object) -> str:
    """``text`` if it is a calendar date written YYYY-MM-DD, the one form a gallery stores (ConsentDate and
    GalleryIdentity's check); ValueError otherwise. Every entry point checks with this before anything is loaded or
    collected: ``date.fromisoformat`` alone also takes 20261019 and 2026-W42-1 on Python 3.11+, which the gallery
    would refuse only when it is finally written."""
    if not isinstance(text, str) or CONSENT_DATE.fullmatch(text) is None:
        raise ValueError("consent date must be YYYY-MM-DD")
    date.fromisoformat(text)  # a real calendar date: ValueError for 2026-02-30
    return text


class FaceCompatibility(Contract):
    """Every setting that changes an embedding; two pipelines are compatible only when all are equal."""

    deepface_version: Version
    model_name: Name
    model_sha256: Sha256
    detector_backend: Name
    detector_sha256: Sha256
    opencv_version: Version
    align: bool
    normalization: Name
    expand_percentage: Annotated[int, Field(ge=0, le=100)]
    color_order: Literal["BGR"]
    dimension: Annotated[PositiveInt, Field(le=2048)]
    prototype_normalization: Literal["l2"]
    tensorflow_version: Version
    tf_keras_version: Version
    numpy_version: Version

    def first_difference(self, other: FaceCompatibility) -> str | None:
        for name in type(self).model_fields:
            if getattr(self, name) != getattr(other, name):
                return name
        return None

    @property
    def revision(self) -> str:
        """Enrollment.model_revision: the model, detector and DeepFace version, and a digest of the whole record."""
        digest = hashlib.sha256(_canonical(self.model_dump(mode="json"))).hexdigest()[:16]
        return f"{self.model_name}+{self.detector_backend}+deepface-{self.deepface_version}+{digest}"


class GalleryIdentity(Contract):
    identity_id: IdentityId
    enrolled_utc: UtcDatetime
    consent_date: ConsentDate
    consent_scope: Literal[CONSENT_SCOPE]  # type: ignore[valid-type]
    prototypes: Annotated[tuple[tuple[FiniteFloat, ...], ...], Field(min_length=1, max_length=MAX_PROTOTYPES)]

    @model_validator(mode="after")
    def _valid_date(self) -> GalleryIdentity:
        date.fromisoformat(self.consent_date)
        return self


class GalleryDocument(Contract):
    schema_name: Literal[SCHEMA] = SCHEMA  # type: ignore[valid-type]
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION  # type: ignore[valid-type]
    compatibility: FaceCompatibility
    identities: tuple[GalleryIdentity, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> GalleryDocument:
        ids = [identity.identity_id for identity in self.identities]
        if len(set(ids)) != len(ids):
            raise ValueError("identity IDs must be unique")
        for identity in self.identities:
            for prototype in identity.prototypes:
                if len(prototype) != self.compatibility.dimension:
                    raise ValueError(f"{identity.identity_id}: a prototype does not have the gallery's dimension")
                if abs(math.sqrt(math.fsum(x * x for x in prototype)) - 1.0) > UNIT_NORM_TOLERANCE:
                    raise ValueError(f"{identity.identity_id}: a prototype is not unit length")
        return self

    def enrollment(self) -> Enrollment:
        return Enrollment(
            model_revision=self.compatibility.revision,
            dimension=self.compatibility.dimension,
            identities=tuple(EnrolledIdentity(identity_id=i.identity_id, prototypes=i.prototypes)
                             for i in self.identities),
        )

    def without(self, identity_id: str) -> GalleryDocument:
        return self.model_copy(update={"identities": tuple(i for i in self.identities if i.identity_id != identity_id)})

    def with_identity(self, identity: GalleryIdentity) -> GalleryDocument:
        return GalleryDocument(compatibility=self.compatibility, identities=(*self.identities, identity))


def new_identity_id(existing: set[str], token: Callable[[int], str] = secrets.token_hex) -> str:
    while True:
        candidate = f"idn-{token(6)}"
        if candidate not in existing:
            return candidate


def unit(vector: Sequence[float]) -> tuple[float, ...]:
    """``vector`` at unit length: the one normalization, for the adapter's embeddings (legacy_deepface) and the capture
    experiment's prototypes. ValueError unless its norm is finite and non-zero."""
    try:
        norm = math.sqrt(math.fsum(x * x for x in vector))
    except OverflowError:  # finite squares whose exact sum exceeds the float range
        norm = math.inf
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("embedding must be finite and non-zero")
    return tuple(x / norm for x in vector)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _associated(compatibility: FaceCompatibility) -> dict[str, Any]:
    return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "compatibility": compatibility.model_dump(mode="json")}


class GalleryNotVerified(VaultError):
    """The sealed gallery, decrypted again, is not exactly what was sealed."""

    def __init__(self) -> None:
        super().__init__("gallery_not_verified")


class GalleryChanged(VaultError):
    """The gallery is no longer the one the change was based on (it appeared, went or was replaced since)."""

    def __init__(self) -> None:
        super().__init__("gallery_changed")


class AuditIncomplete(VaultError):
    """The change took effect, its started line is in the audit log, but its completed line could not be written and
    the change could not be undone either."""

    def __init__(self) -> None:
        super().__init__("committed_audit_incomplete")


class IdentityStore:
    """The identity directory: the sealed gallery, the inbox and the audit log."""

    def __init__(self, directory: Path, sealer: Sealer, *,
                 utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 lock_wait_s: float = LOCK_WAIT_S) -> None:
        self.directory = Path(directory)
        self._sealer = sealer
        self._utc_now = utc_now
        self.lock_wait_s = lock_wait_s  # how long a change waits for another one under way (then gallery_busy)
        self._held = threading.local()  # this thread's depth in locked(): re-entrant per thread, never across

    @property
    def gallery_path(self) -> Path:
        return self.directory / GALLERY_FILE

    @property
    def staged_path(self) -> Path:
        return self.directory / STAGED_FILE

    @property
    def previous_path(self) -> Path:
        return self.directory / PREVIOUS_FILE

    @property
    def lock_path(self) -> Path:
        return self.directory / LOCK_FILE

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the directory's change lock: an exclusive ``flock`` on LOCK_FILE, so that one enrollment, revoke,
        purge or capture at a time reads, changes and writes the gallery and the audit log, in any process. Waits up
        to ``lock_wait_s`` for a change under way, then refuses (``gallery_busy``) having changed nothing. The lock
        goes with the process (no stale lock after a stop). Re-entrant within the same thread; another thread, even
        with this store, waits like another process (its own open file description)."""
        depth = getattr(self._held, "depth", 0)
        if depth:
            self._held.depth = depth + 1
            try:
                yield
            finally:
                self._held.depth = depth
            return
        problem = self.directory_problem()
        if problem is not None:
            raise VaultError(problem)
        try:
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, PRIVATE_FILE_MODE)
        except OSError as exc:
            raise VaultError(f"lock_failed:{type(exc).__name__}") from None
        try:
            deadline = time.monotonic() + self.lock_wait_s
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise VaultError("gallery_busy") from None
                    time.sleep(LOCK_POLL_S)
            self._held.depth = 1
            try:
                yield
            finally:
                self._held.depth = 0
        finally:
            os.close(fd)  # releases the lock

    @property
    def audit_path(self) -> Path:
        return self.directory / AUDIT_FILE

    @property
    def inbox(self) -> Path:
        return self.directory / INBOX_DIR

    def ensure_directory(self) -> None:
        """Create the 0700 identity directory and inbox if missing; refuse an existing one that is not private."""
        for directory in (self.directory, self.inbox):
            if not directory.exists():
                directory.mkdir(mode=PRIVATE_DIR_MODE, parents=directory is self.directory)
                directory.chmod(PRIVATE_DIR_MODE)  # mkdir's mode is masked by the umask
            problem = private_directory_problem(directory)
            if problem is not None:
                raise VaultError(problem)

    def directory_problem(self) -> str | None:
        return private_directory_problem(self.directory)

    def exists(self) -> bool:
        return self.gallery_path.exists()

    def load(self, secret: Secret) -> GalleryDocument:
        return self._decode(self._read(self.gallery_path), secret)

    def read_for_update(self, secret: Secret) -> tuple[GalleryDocument | None, str | None]:
        """The gallery and the SHA-256 of its sealed file, (None, None) when there is none: what a change is based on.
        Read it while holding ``locked()``, and pass the digest to ``commit`` as ``base``."""
        if not self.exists():
            return None, None
        sealed = self._read(self.gallery_path)
        return self._decode(sealed, secret), hashlib.sha256(sealed).hexdigest()

    def _read(self, path: Path) -> bytes:
        problem = self.directory_problem()  # the directory first, then the file
        if problem is not None:
            raise VaultError(problem)
        return read_private(path)

    def _decode(self, sealed: bytes, secret: Secret) -> GalleryDocument:
        associated, plaintext = self._sealer.open(sealed, secret)
        try:
            document = GalleryDocument.model_validate_json(plaintext)
        except ValidationError:
            raise VaultError("gallery_invalid") from None
        if associated != _associated(document.compatibility):
            raise VaultError("gallery_inconsistent")  # the authenticated header and the sealed record disagree
        return document

    def _fingerprint(self) -> str | None:
        return hashlib.sha256(read_private(self.gallery_path)).hexdigest() if self.exists() else None

    def save(self, document: GalleryDocument, secret: Secret) -> None:
        """Write ``document`` as the gallery directly: no lock, no audit (to set one up, as tests do). Enrollment,
        revoke and capture change the gallery through ``commit``."""
        self._write(self.gallery_path, document, secret)

    def _write(self, path: Path, document: GalleryDocument, secret: Secret) -> None:
        sealed = self._sealer.seal(_canonical(document.model_dump(mode="json")), secret,
                                   _associated(document.compatibility))
        write_private_atomic(path, sealed)

    def commit(self, document: GalleryDocument, secret: Secret, action: str, identity_id: str, *, base: str | None,
               **counts: int) -> None:
        """Make ``document`` the gallery, under the change lock, with the change audited before and after it.

        1. Seal it into the staging file, decrypt it from there and compare it in full (``GalleryNotVerified``).
        2. Check that the gallery is still the one the change was based on: ``base``, the digest ``read_for_update``
           gave, None for none (``GalleryChanged``).
        3. Keep the current gallery as the previous file (a hard link), then append ``<action>_started`` with the
           identity and counts, and fsync it.
        4. Rename the staging file over the gallery (the directory fsynced).
        5. Append ``<action>``, the completed line.

        A failure in 1-3 leaves the gallery and the audit log as they were. A failed rename (4) adds
        ``<action>_aborted``. A completed line that cannot be written (5) undoes the change from the previous file
        and adds ``<action>_aborted``, both best effort. So a completed line is only ever written for a change in
        effect, and a change never takes effect without its started line before it; only if the undo also fails does
        the change stay with its started line alone (``AuditIncomplete``).

        Not atomic across a stop (a crash or power loss): a stop after 3 leaves a started line with no outcome, the
        gallery either unchanged (before 4) or changed (after 4), and the staging or previous file until the next
        change or purge removes it. The log alone cannot tell which; the gallery can."""
        with self.locked():
            try:
                self._write(self.staged_path, document, secret)
                if self._decode(read_private(self.staged_path), secret) != document:
                    raise GalleryNotVerified()
                if self._fingerprint() != base:
                    raise GalleryChanged()
                self._remove(self.previous_path)
                if base is not None:
                    try:
                        os.link(self.gallery_path, self.previous_path)
                    except OSError as exc:
                        raise VaultError(f"save_failed:{type(exc).__name__}") from None
                self.audit(f"{action}_started", identity_id, **counts)
            except BaseException:
                self._remove(self.staged_path)
                self._remove(self.previous_path)
                raise
            try:
                try:
                    replace_private(self.staged_path, self.gallery_path)
                except VaultError:  # not renamed: the gallery is as it was
                    self._remove(self.staged_path)
                    self._audit_best_effort(f"{action}_aborted", identity_id)
                    raise
                try:
                    self.audit(action, identity_id, **counts)
                except VaultError:
                    self._undo(had_gallery=base is not None)
                    self._audit_best_effort(f"{action}_aborted", identity_id)
                    raise
            finally:
                self._remove(self.previous_path)

    def _undo(self, *, had_gallery: bool) -> None:
        """Put the gallery back as it was before the rename: the previous file, or no gallery."""
        try:
            if had_gallery:
                replace_private(self.previous_path, self.gallery_path)
            else:
                self.gallery_path.unlink()
                fsync_directory(self.directory)
        except (VaultError, OSError):
            raise AuditIncomplete() from None

    def _audit_best_effort(self, action: str, identity_id: str | None) -> None:
        with contextlib.suppress(VaultError):
            self.audit(action, identity_id)

    @staticmethod
    def _remove(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink()

    def audit(self, action: str, identity_id: str | None = None, **counts: int) -> None:
        """Append one complete numbers-only line, or nothing; the identity ID is opaque and nothing else identifies
        anyone. A short or failed write is cut back to the log's previous length (``audit_failed:<class>``). A log
        whose last line is torn (only a stop in the middle of an append leaves one) is refused (``audit_log_torn``):
        nothing is appended to it until an operator has dealt with it."""
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in counts.values()):
            raise ValueError("audit details are counts only")
        line = _canonical({"utc": self._utc_now().isoformat(timespec="seconds"), "action": action,
                           "identity_id": identity_id, **counts}) + b"\n"
        with self.locked():
            if self.audit_path.exists():
                problem = private_file_problem(self.audit_path)
                if problem is not None:
                    raise VaultError(f"audit_{problem}")
            try:
                fd = os.open(self.audit_path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, PRIVATE_FILE_MODE)
            except OSError as exc:
                raise VaultError(f"audit_failed:{type(exc).__name__}") from None
            try:
                length = os.fstat(fd).st_size
                if length and os.pread(fd, 1, length - 1) != b"\n":
                    raise VaultError("audit_log_torn")
                try:
                    view = memoryview(line)
                    while view:
                        view = view[os.write(fd, view):]
                    os.fsync(fd)
                except OSError as exc:
                    with contextlib.suppress(OSError):
                        os.ftruncate(fd, length)
                        os.fsync(fd)
                    raise VaultError(f"audit_failed:{type(exc).__name__}") from None
            finally:
                os.close(fd)

    def revoke(self, identity_id: str, secret: Secret) -> bool:
        """Rewrite the gallery without ``identity_id`` through ``commit`` (under the lock, audited before and after);
        False if it was not enrolled."""
        with self.locked():
            document, base = self.read_for_update(secret)
            if document is None:
                raise VaultError("file_missing")
            if identity_id not in {i.identity_id for i in document.identities}:
                return False
            self.commit(document.without(identity_id), secret, "revoke", identity_id, base=base,
                        identities_left=len(document.identities) - 1)
            return True

    def purge(self) -> dict[str, int]:
        """Delete the gallery, any staging or previous file left behind and everything in the inbox, under the lock,
        between a ``purge_started`` and a ``purge`` line. Deleting cannot be undone: a failure part way leaves the
        started line without its completed one."""
        removed = {"gallery": 0, "inbox_files": 0}
        with self.locked():
            self.audit("purge_started")
            if self.gallery_path.exists():
                self.gallery_path.unlink()
                removed["gallery"] = 1
            for leftover in (self.staged_path, self.previous_path):  # not counted: the purge line's fields stay
                leftover.unlink(missing_ok=True)
            if self.inbox.is_dir():
                for path in sorted(self.inbox.rglob("*"), reverse=True):
                    if path.is_dir() and not path.is_symlink():
                        with contextlib.suppress(OSError):
                            path.rmdir()
                    else:
                        path.unlink()
                        removed["inbox_files"] += 1
            self.audit("purge", None, **removed)
        return removed

    def enrollment_folder(self, folder: Path) -> list[tuple[Path, tuple[int, int, int]]]:
        """The photo copies of one enrollment: a folder directly inside the inbox (not a link) holding regular,
        non-hidden files only; returns each file with its (inode, size, mtime_ns). Anything else is refused."""
        folder = Path(folder)
        if folder.is_symlink() or not folder.is_dir() or folder.resolve().parent != self.inbox.resolve():
            raise VaultError("photos_not_in_inbox")
        entries = []
        for path in sorted(folder.iterdir()):
            info = os.lstat(path)
            if path.name.startswith(".") or not stat.S_ISREG(info.st_mode):
                raise VaultError("photos_folder_not_plain")  # a link, folder, hidden or special file
            entries.append((path, (info.st_ino, info.st_size, info.st_mtime_ns)))
        return entries

    def delete_supplied_copies(self, folder: Path, entries: list[tuple[Path, tuple[int, int, int]]]) -> dict[str, int]:
        """Delete exactly the listed copies, each only if it is still the same file, then the folder if empty."""
        deleted = changed = 0
        for path, identity in entries:
            try:
                info = os.lstat(path)
            except FileNotFoundError:
                changed += 1
                continue
            if not stat.S_ISREG(info.st_mode) or (info.st_ino, info.st_size, info.st_mtime_ns) != identity:
                changed += 1
                continue
            path.unlink()
            deleted += 1
        remaining = sum(1 for _ in Path(folder).iterdir())
        if remaining == 0:
            Path(folder).rmdir()
        return {"photos_deleted": deleted, "photos_changed_not_deleted": changed, "folder_entries_left": remaining}
