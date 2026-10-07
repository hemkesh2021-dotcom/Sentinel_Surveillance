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

Removal: ``revoke`` rewrites the gallery without the identity; ``purge`` deletes the gallery and the inbox. The
runtime reads the gallery once at startup, so either takes effect at the next start. Deleting files on an SSD is not
secure erasure.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import secrets
import shutil
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
    write_private_atomic,
)

SCHEMA = "sentinel-identity-gallery"
SCHEMA_VERSION = 1
GALLERY_FILE = "gallery.sealed"
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
        problem = self.directory_problem()
        if problem is not None:
            raise VaultError(problem)
        associated, plaintext = self._sealer.open(read_private(self.gallery_path), secret)
        try:
            document = GalleryDocument.model_validate_json(plaintext)
        except ValidationError:
            raise VaultError("gallery_invalid") from None
        if associated != _associated(document.compatibility):
            raise VaultError("gallery_inconsistent")  # the authenticated header and the sealed record disagree
        return document

    def save(self, document: GalleryDocument, secret: Secret) -> None:
        sealed = self._sealer.seal(_canonical(document.model_dump(mode="json")), secret,
                                   _associated(document.compatibility))
        write_private_atomic(self.gallery_path, sealed)

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
        fd = os.open(self.audit_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, PRIVATE_FILE_MODE)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)

    def revoke(self, identity_id: str, secret: Secret) -> bool:
        """Rewrite the gallery without ``identity_id``; False if it was not enrolled."""
        document = self.load(secret)
        if identity_id not in {i.identity_id for i in document.identities}:
            return False
        self.save(document.without(identity_id), secret)
        self.audit("revoke", identity_id, identities_left=len(document.identities) - 1)
        return True

    def purge(self) -> dict[str, int]:
        """Delete the gallery and everything in the inbox; the audit log keeps a purge line."""
        problem = self.directory_problem()
        if problem is not None:
            raise VaultError(problem)
        removed = {"gallery": 0, "inbox_files": 0}
        if self.gallery_path.exists():
            self.gallery_path.unlink()
            removed["gallery"] = 1
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

    def remove_tree(self, directory: Path) -> int:
        """Delete one enrollment's photo folder inside the inbox (never anything outside it); returns files removed."""
        directory = Path(directory)
        inbox = self.inbox.resolve()
        if directory.is_symlink() or inbox not in directory.resolve().parents:
            raise VaultError("photos_not_in_inbox")
        files = sum(1 for path in directory.rglob("*") if not path.is_dir())
        shutil.rmtree(directory)
        return files
