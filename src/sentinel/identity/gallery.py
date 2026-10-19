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

Enrollment and capture change the gallery through ``commit``: sealed into a staging file beside it, read back and
compared in full, audited, and only then renamed over it. So the gallery never holds an identity whose event was not
audited; any failure before the rename leaves the gallery and the audit log as they were.

Removal: ``revoke`` rewrites the gallery without the identity; ``purge`` deletes the gallery, any staging file left
by a stop before its rename, and the inbox. The
runtime reads the gallery once at startup, so either takes effect at the next start. Deleting files on an SSD is not
secure erasure.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Callable
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


def unit(vector: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    norm = math.sqrt(math.fsum(x * x for x in vector))
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


class IdentityStore:
    """The identity directory: the sealed gallery, the inbox and the audit log."""

    def __init__(self, directory: Path, sealer: Sealer, *,
                 utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.directory = Path(directory)
        self._sealer = sealer
        self._utc_now = utc_now

    @property
    def gallery_path(self) -> Path:
        return self.directory / GALLERY_FILE

    @property
    def staged_path(self) -> Path:
        return self.directory / STAGED_FILE

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
        return self._open(self.gallery_path, secret)

    def _open(self, path: Path, secret: Secret) -> GalleryDocument:
        problem = self.directory_problem()
        if problem is not None:
            raise VaultError(problem)
        associated, plaintext = self._sealer.open(read_private(path), secret)
        try:
            document = GalleryDocument.model_validate_json(plaintext)
        except ValidationError:
            raise VaultError("gallery_invalid") from None
        if associated != _associated(document.compatibility):
            raise VaultError("gallery_inconsistent")  # the authenticated header and the sealed record disagree
        return document

    def save(self, document: GalleryDocument, secret: Secret) -> None:
        self._write(self.gallery_path, document, secret)

    def _write(self, path: Path, document: GalleryDocument, secret: Secret) -> None:
        sealed = self._sealer.seal(_canonical(document.model_dump(mode="json")), secret,
                                   _associated(document.compatibility))
        write_private_atomic(path, sealed)

    def commit(self, document: GalleryDocument, secret: Secret, action: str, identity_id: str, **counts: int) -> None:
        """Make ``document`` the gallery only once it is verified and its event audited.

        It is sealed into the staging file, decrypted from there and compared in full (GalleryNotVerified), the event
        is audited, and only then is the staging file renamed over the gallery. Any failure before the rename removes
        the staging file and leaves the gallery (or its absence) and the audit log as they were. Only a failed rename,
        or a stop between the audit and the rename, can leave an audit line for a change that did not take effect;
        never a change without its audit line."""
        try:
            self._write(self.staged_path, document, secret)
            if self._open(self.staged_path, secret) != document:
                raise GalleryNotVerified()
            self.audit(action, identity_id, **counts)
            replace_private(self.staged_path, self.gallery_path)
        except BaseException:
            with contextlib.suppress(OSError):
                self.staged_path.unlink()
            raise

    def audit(self, action: str, identity_id: str | None = None, **counts: int) -> None:
        """Append one numbers-only line; the identity ID is opaque and nothing else identifies anyone."""
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in counts.values()):
            raise ValueError("audit details are counts only")
        line = _canonical({"utc": self._utc_now().isoformat(timespec="seconds"), "action": action,
                           "identity_id": identity_id, **counts}) + b"\n"
        if self.audit_path.exists():
            problem = private_file_problem(self.audit_path)
            if problem is not None:
                raise VaultError(f"audit_{problem}")
        try:
            fd = os.open(self.audit_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, PRIVATE_FILE_MODE)
            try:
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            raise VaultError(f"audit_failed:{type(exc).__name__}") from None

    def revoke(self, identity_id: str, secret: Secret) -> bool:
        """Rewrite the gallery without ``identity_id``; False if it was not enrolled."""
        document = self.load(secret)
        if identity_id not in {i.identity_id for i in document.identities}:
            return False
        self.save(document.without(identity_id), secret)
        self.audit("revoke", identity_id, identities_left=len(document.identities) - 1)
        return True

    def purge(self) -> dict[str, int]:
        """Delete the gallery, a staging file left behind and everything in the inbox; the audit log keeps a purge
        line."""
        problem = self.directory_problem()
        if problem is not None:
            raise VaultError(problem)
        removed = {"gallery": 0, "inbox_files": 0}
        if self.gallery_path.exists():
            self.gallery_path.unlink()
            removed["gallery"] = 1
        self.staged_path.unlink(missing_ok=True)  # not counted: the purge line's fields stay as they were
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
