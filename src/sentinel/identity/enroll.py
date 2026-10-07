"""Enrollment from explicitly consented, locally supplied photos (V2-25 demo form; guide chapter 11).

``sentinel identity enroll`` takes one person's photos from a folder inside the identity directory's private
``inbox/`` and the operator's consent confirmation and date. It never reads any other folder, ``face_db.pkl`` or a
pickle. Each photo goes through the runtime's own face pipeline (the same DeepFace version, YuNet, alignment,
normalization and Facenet512, decoded with the same OpenCV), so enrollment and runtime embeddings are compatible by
construction; the gallery's compatibility record must equal the loaded pipeline's.

Checks, all before anything is written:
- every photo has exactly one face (``no_face``, ``several_faces``, ``unreadable_photo`` otherwise) whose YuNet
  confidence reaches ``identity.min_quality`` (``low_quality``); at least MIN_PHOTOS and at most MAX_PROTOTYPES
  photos must pass, and a folder with more photos than MAX_PROTOTYPES is refused rather than sampled;
- the accepted photos agree with each other: each one's best cosine similarity to the others reaches
  ``identity.match_threshold`` (``inconsistent_photos``: a wrong or poor photo);
- the new person is not close to anyone already enrolled (``close_to_enrolled_identity``).

The photos are copies the operator put into one dedicated folder directly inside the inbox; a folder holding
anything but regular, non-hidden files (a link, a subfolder) is refused before any model loads. ``--dry-run``
reports the result per photo and writes and deletes nothing. Otherwise the gallery is saved and reloaded, and the
reloaded gallery must equal exactly what was saved; only then is the event audited (counts and the opaque ID only)
and are exactly the listed copies deleted, each only if it is still the same file (inode, size, mtime). The folder
is removed only if it is then empty. Nothing outside it is ever touched. Reports name photos by index only (file
names can contain names) and never contain embeddings.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .gallery import CONSENT_SCOPE, GalleryDocument, GalleryIdentity, IdentityStore, new_identity_id
from .matching import MAX_PROTOTYPES
from .vault import Secret, VaultError

MIN_PHOTOS = 2  # so that the consistency check has something to compare; a starting rule


class EnrollmentRefused(Exception):
    """Nothing was written; ``label`` and ``report`` (numbers and labels only) say why."""

    def __init__(self, label: str, report: dict[str, Any]) -> None:
        super().__init__(label)
        self.label = label
        self.report = report


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))  # both are unit length


def enroll(
    store: IdentityStore,
    backend: Any,
    folder: Path,
    *,
    secret: Secret,
    consent_date: str,
    min_quality: float,
    match_threshold: float,
    dry_run: bool = False,
    utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc).replace(microsecond=0),
    new_id: Callable[[set[str]], str] = new_identity_id,
) -> dict[str, Any]:
    date.fromisoformat(consent_date)  # ValueError for an invalid date
    store.ensure_directory()
    entries = store.enrollment_folder(folder)
    photos = [path for path, _ in entries]
    report: dict[str, Any] = {"photos": len(photos), "dry_run": dry_run, "results": []}
    if len(photos) > MAX_PROTOTYPES:
        raise EnrollmentRefused("too_many_photos", report)
    backend.load()
    accepted: list[tuple[int, tuple[float, ...]]] = []
    for index, path in enumerate(photos, start=1):
        face, reason = backend.enrollment_photo(path)
        if face is not None and face.quality < min_quality:
            face, reason = None, "low_quality"
        entry: dict[str, Any] = {"photo": index, "result": "accepted" if face is not None else "rejected",
                                 "reason": reason}
        if face is not None:
            entry.update(quality=round(face.quality, 3), face_width_px=face.face_width_px)
            accepted.append((index, face.embedding))
        report["results"].append(entry)
    report["accepted"] = len(accepted)
    if len(accepted) < MIN_PHOTOS:
        raise EnrollmentRefused("too_few_usable_photos", report)
    inconsistent = [index for index, embedding in accepted
                    if max(_cosine(embedding, other) for j, other in accepted if j != index) < match_threshold]
    if inconsistent:
        report["inconsistent_photos"] = inconsistent
        raise EnrollmentRefused("inconsistent_photos", report)
    compatibility = backend.compatibility()
    if store.exists():
        document = store.load(secret)
        difference = document.compatibility.first_difference(compatibility)
        if difference is not None:
            raise EnrollmentRefused(f"gallery_incompatible:{difference}", report)
    else:
        document = GalleryDocument(compatibility=compatibility)
    for identity in document.identities:
        best = max(_cosine(e, p) for _, e in accepted for p in identity.prototypes)
        if best >= match_threshold:
            raise EnrollmentRefused("close_to_enrolled_identity", report)
    if dry_run:
        return report
    identity_id = new_id({i.identity_id for i in document.identities})
    identity = GalleryIdentity(identity_id=identity_id, enrolled_utc=utc_now(), consent_date=consent_date,
                               consent_scope=CONSENT_SCOPE, prototypes=tuple(e for _, e in accepted))
    saved = document.with_identity(identity)
    store.save(saved, secret)
    if store.load(secret) != saved:  # decrypt the file just written and compare everything, not just the ID
        raise VaultError("enrollment_not_verified")
    store.audit("enroll", identity_id, photos_used=len(accepted), photos_rejected=len(photos) - len(accepted))
    report.update(identity_id=identity_id, verified=True, **store.delete_supplied_copies(folder, entries))
    return report
