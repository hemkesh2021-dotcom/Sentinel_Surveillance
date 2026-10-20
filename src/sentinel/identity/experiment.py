"""The camera-side enrollment experiment (D91): a separate working gallery, never the original.

Both parts run only in face validation runs with the full-scale detection candidate (D88), whose faces they use; the
original gallery stays the only one that votes, and is never written.

**Capture** (``--face-capture-gallery DIR``). While the enrolled person works at the seat, the runtime keeps in memory
the embedding of the face assigned to the only person on each applied result whose face the vote compared with the
enrollment (at most CAPTURE_MAX; its similarity is not looked at). At shutdown it selects the prototypes by a rule
fixed here before any capture: the faces whose frames lie at least CAPTURE_WASHOUT_S after the first such face and
before the last are split into CAPTURE_SEGMENTS equal time segments, and in each segment the face nearest its middle is
taken (the earlier on a tie; a segment without a face gives none). No face is chosen or refused by its similarity to
anything. The selection is then held to enrollment's own consistency check (each prototype's best similarity to the
others at least the match threshold; enroll.py's ``inconsistent_photos``). Only then is the working gallery written:
the original's compatibility record and identity ID, the operator's consent date and the selected prototypes, sealed
with the same passphrase into DIR, a new private directory that is not the original's; it is read back and compared,
and the event is audited there (counts only). Nothing is written when any check fails, and the report's ``refused``
says which (``too_few_faces``, ``too_few_prototypes``, ``inconsistent_capture``, ``capture_target_has_gallery``,
``capture_not_verified``, ``capture_write_failed``). ``written`` is true exactly when the working gallery is on disk,
so one refusal comes with it: ``capture_audit_incomplete``, a gallery read back and in place whose completed audit
line could not be written and whose change could not be undone either (``AuditIncomplete``); it is not to be used
(purge it). The embeddings never leave memory otherwise: the report carries counts, and, for the private validation
evidence, each prototype's best similarity to the original and the consistency check's smallest value (numbers only,
as D84's).

**Shadow comparison** (``--face-shadow-gallery DIR``). A working gallery opened read only beside the original (the
same passphrase, the same compatibility record, identities the original also has). Every face whose vote compared it
with the original is also compared with it, and the best similarity and its prototype's position are recorded
(schema 6 "shadow"): never a vote, a label, a state, a transition or anything the page shows.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..media.clock import NS_PER_SECOND
from .gallery import (
    CONSENT_SCOPE,
    AuditIncomplete,
    GalleryChanged,
    GalleryDocument,
    GalleryIdentity,
    GalleryNotVerified,
    IdentityStore,
    check_consent_date,
    unit,
)
from .matching import MAX_PROTOTYPES, Enrollment, best_match, unit_cosine
from .vault import Secret, VaultError

CAPTURE_MAX = 600  # faces kept in memory (each 512 doubles: about 2.5 MB in all); more are counted and dropped
CAPTURE_WASHOUT_S = 15.0  # an assumption, not a measurement: settling after the first face and before the last
CAPTURE_SEGMENTS = MAX_PROTOTYPES  # one prototype per equal time segment
CAPTURE_MIN_FACES = 40  # faces inside the selection span
CAPTURE_MIN_PROTOTYPES = 4  # segments that gave a face
SIMILARITY_DECIMALS = 3
CAPTURE_AUDIT_INCOMPLETE = "capture_audit_incomplete"  # the one refusal whose working gallery was written
CAPTURE_REFUSALS = ("too_few_faces", "too_few_prototypes", "inconsistent_capture", "capture_target_has_gallery",
                    "capture_not_verified", "capture_write_failed", CAPTURE_AUDIT_INCOMPLETE)


class ExperimentRefused(Exception):
    """A fixed label; nothing was opened or written."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


def same_directory(a: Path, b: Path) -> bool:
    return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()


@dataclass
class Shadow:
    """The working gallery opened read only beside the original: only its similarities are recorded."""

    enrollment: Enrollment
    prototypes: int  # in all its identities

    def numbers(self, embedding: Sequence[float] | None) -> dict[str, Any] | None:
        """Its best similarity to ``embedding`` and that prototype's position; None without an embedding or when
        the dimensions differ (a working gallery has the original's compatibility record, so they never should)."""
        if embedding is None or self.enrollment.is_empty or len(embedding) != self.enrollment.dimension:
            return None
        match = best_match(embedding, self.enrollment)
        return {"similarity": round(match.similarity, SIMILARITY_DECIMALS), "prototype": match.prototype}


def open_shadow(store: IdentityStore, secret: Secret, original: GalleryDocument) -> Shadow:
    """The working gallery in ``store``, for comparison only: it must exist, open with the same passphrase, have the
    original's compatibility record exactly, and hold no identity the original does not."""
    if not store.exists():
        raise ExperimentRefused("face_shadow_missing")
    try:
        document = store.load(secret)
    except VaultError as exc:
        raise ExperimentRefused(f"face_shadow_unreadable:{exc.label}") from None
    difference = document.compatibility.first_difference(original.compatibility)
    if difference is not None:
        raise ExperimentRefused(f"face_shadow_incompatible:{difference}")
    known = {identity.identity_id for identity in original.identities}
    if not document.identities or any(identity.identity_id not in known for identity in document.identities):
        raise ExperimentRefused("face_shadow_unknown_identity")
    return Shadow(document.enrollment(), sum(len(identity.prototypes) for identity in document.identities))


def select(times_ns: Sequence[int], *, washout_s: float = CAPTURE_WASHOUT_S,
           segments: int = CAPTURE_SEGMENTS) -> tuple[list[int], int]:
    """The predeclared rule: (the indices of the chosen faces in time order, the faces inside the span). ``times_ns``
    are the faces' frame times in the order they were kept."""
    if not times_ns:
        return [], 0
    order = sorted(range(len(times_ns)), key=lambda i: (times_ns[i], i))
    start = times_ns[order[0]] + round(washout_s * NS_PER_SECOND)
    end = times_ns[order[-1]] - round(washout_s * NS_PER_SECOND)
    span = [i for i in order if start <= times_ns[i] <= end]
    if end <= start or not span:
        return [], len(span)
    length = (end - start) / segments
    chosen = []
    for k in range(segments):
        low, high, middle = start + k * length, start + (k + 1) * length, start + (k + 0.5) * length
        inside = [i for i in span if low <= times_ns[i] < high or (k == segments - 1 and times_ns[i] == end)]
        if inside:
            chosen.append(min(inside, key=lambda i: (abs(times_ns[i] - middle), times_ns[i], i)))
    return chosen, len(span)


@dataclass
class Capture:
    """In-memory collection for one capture run, and the working gallery written once at its end."""

    store: IdentityStore  # the working gallery's directory (never the original's)
    secret: Secret
    original: GalleryDocument  # its compatibility record and single identity ID are the working gallery's
    consent_date: str
    match_threshold: float
    utc_now: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc).replace(microsecond=0),
                                             repr=False)
    _times: list[int] = field(default_factory=list, repr=False)
    _embeddings: list[array] = field(default_factory=list, repr=False)  # never printed
    dropped_over_cap: int = 0
    finished: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        check_consent_date(self.consent_date)  # ValueError unless YYYY-MM-DD, before anything is collected
        if len(self.original.identities) != 1:
            raise ExperimentRefused("face_capture_needs_one_identity")
        if self.store.exists():
            raise ExperimentRefused("face_capture_target_has_gallery")

    @property
    def kept(self) -> int:
        return len(self._times)

    def offer(self, frame_ns: int, embedding: Sequence[float]) -> None:
        if self.finished is not None:
            return
        if len(self._times) >= CAPTURE_MAX:
            self.dropped_over_cap += 1
            return
        self._times.append(frame_ns)
        self._embeddings.append(array("d", embedding))

    def finish(self) -> dict[str, Any]:
        """Select, check and write once; the report (numbers and fixed labels only). The embeddings are discarded.
        Never raises: anything unexpected is refused as ``capture_write_failed``, with nothing written."""
        if self.finished is not None:
            return self.finished
        report: dict[str, Any] = {"identity": "capture", "written": False, "refused": None,
                                  "faces_kept": self.kept, "faces_dropped_over_cap": self.dropped_over_cap,
                                  "faces_in_span": 0, "prototypes": 0, "min_best_to_others": None,
                                  "similarity_to_original": [], "verified": False}
        try:
            chosen, in_span = select(self._times)
            report["faces_in_span"] = in_span
            if in_span < CAPTURE_MIN_FACES:
                report["refused"] = "too_few_faces"
                return self._end(report)
            if len(chosen) < CAPTURE_MIN_PROTOTYPES:
                report["refused"] = "too_few_prototypes"
                report["prototypes"] = len(chosen)
                return self._end(report)
            prototypes = [unit(self._embeddings[i]) for i in chosen]
            report["prototypes"] = len(prototypes)
            best_to_others = [max(unit_cosine(p, q) for j, q in enumerate(prototypes) if j != i)
                              for i, p in enumerate(prototypes)]
            report["min_best_to_others"] = round(min(best_to_others), SIMILARITY_DECIMALS)
            original = self.original.enrollment()
            report["similarity_to_original"] = [round(best_match(p, original).similarity, SIMILARITY_DECIMALS)
                                                for p in prototypes]
            if min(best_to_others) < self.match_threshold:
                report["refused"] = "inconsistent_capture"
                return self._end(report)
            if self.store.exists():
                report["refused"] = "capture_target_has_gallery"
                return self._end(report)
            (identity,) = self.original.identities
            document = GalleryDocument(compatibility=self.original.compatibility, identities=(GalleryIdentity(
                identity_id=identity.identity_id, enrolled_utc=self.utc_now(), consent_date=self.consent_date,
                consent_scope=CONSENT_SCOPE, prototypes=tuple(prototypes)),))
            try:  # read back, audited, and only then in place: a failure leaves no gallery to open as a shadow (only a
                # completed line that cannot be written and then an undo that fails too leave it, AuditIncomplete)
                self.store.ensure_directory()
                self.store.commit(document, self.secret, "capture", identity.identity_id, base=None,
                                  prototypes_used=len(prototypes), faces_in_span=in_span)
            except GalleryNotVerified:
                report["refused"] = "capture_not_verified"
                return self._end(report)
            except GalleryChanged:  # a gallery appeared in the directory after the check above
                report["refused"] = "capture_target_has_gallery"
                return self._end(report)
            except AuditIncomplete:  # read back and in place, its completed line missing: written, not to be used
                report.update(written=True, verified=True, refused=CAPTURE_AUDIT_INCOMPLETE)
                return self._end(report)
            except (VaultError, OSError):
                report["refused"] = "capture_write_failed"
                return self._end(report)
            report.update(written=True, verified=True)
            return self._end(report)
        except Exception:  # noqa: BLE001 - it runs in the runtime's stop: refused there, never raised into it
            report.update(written=False, verified=False, refused="capture_write_failed")
            return self._end(report)
        finally:
            self._times.clear()
            self._embeddings.clear()

    def _end(self, report: dict[str, Any]) -> dict[str, Any]:
        self.finished = report
        return report
