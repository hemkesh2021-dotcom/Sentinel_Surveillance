"""The camera-side enrollment experiment (D91): a separate working gallery, never the original.

Both parts run only in face validation runs: on the substream with the full-scale detection candidate (D88), whose
faces they use, or on route B (main_single) with its own head-window faces (the multi-pose comparison); the original
gallery stays the only one that votes, and is never written.

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
with the original is also compared with it, and the best similarity, its prototype's position and the decision are
recorded (schema 6 and 8 "shadow"). The decision is the production vote decision (state.decide) with the resolver's
own policy, made at full precision on the same face before anything is rounded or written: what the face's vote
would have been had the working gallery voted. It is never a vote, a state, a transition or anything the page shows.

**Raw capture** (``--face-capture-raw DIR``; route B validation runs only, the multi-pose comparison). The same faces as
a capture's (the only person on an applied result, its face compared), kept with their frame time, frame number,
stream epoch, track number and boot, and at the stop sealed with the same passphrase into ``DIR/raw-capture.sealed``
(a new private directory) and read back. Nothing is selected or written as a gallery here: an offline builder
(benchmarks/runner/mpe_build.py) chooses one face per cued pose window by time alone and deletes this file. No label,
similarity, image or crop is kept with a face.
"""

from __future__ import annotations

import hashlib
import json
import os
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
from .state import IdentityPolicy, decide, decision_label
from .vault import (
    PRIVATE_DIR_MODE,
    Secret,
    Sealer,
    VaultError,
    private_directory_problem,
    private_file_problem,
    read_private,
    write_private_atomic,
)

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


def prototypes_digest(prototypes: Sequence[Sequence[float]]) -> str:
    """SHA-256 over a gallery's prototypes' exact values, in order (repr round-trips): which working gallery a run
    compared with, recorded at a route B shadow run's start and in the multi-pose builder's frozen record. A digest
    only: no value can be read back from it."""
    canonical = json.dumps([[float(x) for x in p] for p in prototypes], separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


@dataclass
class Shadow:
    """The working gallery opened read only beside the original: its similarities and decisions are recorded."""

    enrollment: Enrollment
    prototypes: int  # in all its identities
    digest: str = ""  # prototypes_digest over all its identities' prototypes, in order

    def numbers(self, face: Any, policy: IdentityPolicy) -> dict[str, Any] | None:
        """The working gallery's best similarity to ``face`` (rounded, for the private record only), that prototype's
        position and the decision: the production vote decision (``state.decide``) with ``policy``, the resolver's
        own, made here at full precision before anything is rounded. None without an embedding or when the
        dimensions differ (a working gallery has the original's compatibility record, so they never should)."""
        embedding = None if face is None else face.embedding
        if embedding is None or self.enrollment.is_empty or len(embedding) != self.enrollment.dimension:
            return None
        _, note, similarity, prototype = decide(face, self.enrollment, policy)
        return {"similarity": None if similarity is None else round(similarity, SIMILARITY_DECIMALS),
                "prototype": prototype, "decision": decision_label(note)}


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
    every = [p for identity in document.identities for p in identity.prototypes]
    return Shadow(document.enrollment(), len(every), prototypes_digest(every))


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


def write_working_gallery(store: IdentityStore, secret: Secret, original: GalleryDocument, consent_date: str,
                          prototypes: Sequence[Sequence[float]], enrolled_utc: datetime, *,
                          faces_in_span: int) -> tuple[bool, str | None]:
    """D91's one write of a working gallery: the original's compatibility record and single identity ID, the consent
    date and ``prototypes`` (unit length, at most MAX_PROTOTYPES), sealed into ``store`` (never the original's), read
    back, audited (counts only) and only then in place. (written, refusal label or None): ``capture_target_has_gallery``,
    ``capture_not_verified``, ``capture_write_failed``, or written with ``capture_audit_incomplete`` (in place, its
    completed audit line missing: not to be used). The capture and the multi-pose builder both write through here."""
    if store.exists():
        return False, "capture_target_has_gallery"
    (identity,) = original.identities
    try:  # read back, audited, and only then in place: a failure leaves no gallery to open as a shadow (only a
        # completed line that cannot be written and then an undo that fails too leave it, AuditIncomplete)
        document = GalleryDocument(compatibility=original.compatibility, identities=(GalleryIdentity(
            identity_id=identity.identity_id, enrolled_utc=enrolled_utc, consent_date=consent_date,
            consent_scope=CONSENT_SCOPE, prototypes=tuple(tuple(p) for p in prototypes)),))
    except ValueError:  # pydantic's ValidationError too: its message names the identity, so only a fixed label
        return False, "capture_write_failed"
    try:
        store.ensure_directory()
        store.commit(document, secret, "capture", identity.identity_id, base=None,
                     prototypes_used=len(document.identities[0].prototypes), faces_in_span=faces_in_span)
    except GalleryNotVerified:
        return False, "capture_not_verified"
    except GalleryChanged:  # a gallery appeared in the directory after the check above
        return False, "capture_target_has_gallery"
    except AuditIncomplete:  # read back and in place, its completed line missing: written, not to be used
        return True, CAPTURE_AUDIT_INCOMPLETE
    except (VaultError, OSError):
        return False, "capture_write_failed"
    return True, None


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

    def offer(self, frame_ns: int, embedding: Sequence[float], **_frame: Any) -> None:
        """One compared face of the only person (the frame's other numbers are a raw capture's, unused here)."""
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
            written, refused = write_working_gallery(self.store, self.secret, self.original, self.consent_date,
                                                     prototypes, self.utc_now(), faces_in_span=in_span)
            report.update(written=written, verified=written, refused=refused)
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


RAW_FORMAT = "sentinel-mpe-raw-capture-1"
RAW_FILE = "raw-capture.sealed"
RAW_REFUSALS = ("no_faces", "raw_capture_target_exists", "raw_capture_not_verified", "raw_capture_write_failed",
                "raw_capture_left_on_disk")  # the last: refused, and a file this run made could not be removed


def _as_list(value: Any) -> list[float]:
    if isinstance(value, array):
        return list(value)
    raise TypeError("only a face's embedding array is converted")


def _raw_plaintext(document: dict[str, Any]) -> bytes:
    """Canonical JSON; floats keep their exact value (Python's shortest round-trip repr)."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
                      default=_as_list).encode("ascii")


@dataclass
class RawCapture:
    """The multi-pose comparison's raw capture (route B validation runs only): the compared faces of the only person,
    with their frame numbers, sealed once at the stop into a new private directory. No label or similarity is kept,
    nothing is selected and no gallery is written."""

    store: IdentityStore  # only its directory (the raw capture's own, new) and its sealing helper are used
    secret: Secret
    original: GalleryDocument  # a single identity; the file records its ID so the builder can check it
    consent_date: str
    utc_now: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc).replace(microsecond=0),
                                             repr=False)
    _faces: list[dict[str, Any]] = field(default_factory=list, repr=False)  # never printed
    boot_id: str | None = None
    dropped_over_cap: int = 0
    dropped_other_boot: int = 0
    finished: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        check_consent_date(self.consent_date)
        if len(self.original.identities) != 1:
            raise ExperimentRefused("face_capture_needs_one_identity")
        if os.path.lexists(self.store.directory):
            raise ExperimentRefused("face_capture_raw_target_exists")

    @property
    def kept(self) -> int:
        return len(self._faces)

    def offer(self, frame_ns: int, embedding: Sequence[float], *, frame_seq: int | None = None,
              stream_epoch: int | None = None, track_id: int | None = None, boot_id: str | None = None) -> None:
        if self.finished is not None:
            return
        if self.boot_id is None:
            self.boot_id = boot_id
        elif boot_id != self.boot_id:  # one boot per raw capture: the frame times are comparable only within one
            self.dropped_other_boot += 1
            return
        if len(self._faces) >= CAPTURE_MAX:
            self.dropped_over_cap += 1
            return
        self._faces.append({"t": int(frame_ns), "seq": frame_seq, "epoch": stream_epoch, "track": track_id,
                            "e": array("d", embedding)})  # 600 faces of 512 doubles: about 2.5 MB, as Capture

    def finish(self) -> dict[str, Any]:
        """Seal once and read back; the report is counts and fixed labels only. Never raises; the faces are discarded."""
        if self.finished is not None:
            return self.finished
        report: dict[str, Any] = {"identity": "capture_raw", "written": False, "refused": None,
                                  "faces_kept": self.kept, "faces_dropped_over_cap": self.dropped_over_cap,
                                  "faces_dropped_other_boot": self.dropped_other_boot, "verified": False}
        directory = Path(self.store.directory)
        created = False
        try:
            if not self._faces:
                report["refused"] = "no_faces"
                return self._end(report)
            if os.path.lexists(directory):
                report["refused"] = "raw_capture_target_exists"
                return self._end(report)
            (identity,) = self.original.identities
            document = {"format": RAW_FORMAT, "boot_id": self.boot_id, "identity_id": identity.identity_id,
                        "compatibility": self.original.compatibility.revision, "consent_date": self.consent_date,
                        "sealed_utc": self.utc_now().isoformat(timespec="seconds"), "faces": self._faces}
            plaintext = _raw_plaintext(document)
            associated = {"format": RAW_FORMAT, "faces": len(self._faces)}
            directory.mkdir(mode=PRIVATE_DIR_MODE)
            created = True
            directory.chmod(PRIVATE_DIR_MODE)  # mkdir's mode is masked by the umask
            sealer = self.store.sealer
            write_private_atomic(directory / RAW_FILE, sealer.seal(plaintext, self.secret, associated))
            opened_associated, opened = sealer.open(read_private(directory / RAW_FILE), self.secret)
            if opened != plaintext or opened_associated != associated:
                report["refused"] = ("raw_capture_not_verified" if _remove_raw(directory)
                                     else "raw_capture_left_on_disk")
                return self._end(report)
            report.update(written=True, verified=True)
            return self._end(report)
        except Exception:  # noqa: BLE001 - it runs in the runtime's stop: refused there, never raised into it
            gone = _remove_raw(directory) if created else True
            report.update(written=False, verified=False,
                          refused="raw_capture_write_failed" if gone else "raw_capture_left_on_disk")
            return self._end(report)
        finally:
            self._faces.clear()

    def _end(self, report: dict[str, Any]) -> dict[str, Any]:
        self.finished = report
        return report


def _remove_raw(directory: Path) -> bool:
    """Remove a raw capture this run created and could not keep: its file (and a temporary of it), then the directory.
    Never raises; True when the directory is gone."""
    try:
        for child in list(directory.iterdir()) if directory.is_dir() else ():
            if child.name == RAW_FILE or (child.name.startswith(f".{RAW_FILE}.") and child.name.endswith(".tmp")):
                child.unlink(missing_ok=True)
        directory.rmdir()
    except OSError:
        pass
    return not os.path.lexists(directory)


def open_raw_capture(directory: Path, sealer: Sealer, secret: Secret) -> dict[str, Any]:
    """A raw capture read back (for the offline builder): the private directory holding exactly RAW_FILE, opened
    with the passphrase, its format checked. ExperimentRefused with a fixed label otherwise."""
    directory = Path(directory)
    problem = private_directory_problem(directory)
    if problem is not None:
        raise ExperimentRefused(f"raw_capture_directory:{problem}")
    names = sorted(child.name for child in directory.iterdir())
    if names != [RAW_FILE]:
        raise ExperimentRefused("raw_capture_unexpected_files")
    problem = private_file_problem(directory / RAW_FILE)
    if problem is not None:
        raise ExperimentRefused(f"raw_capture_file:{problem}")
    try:
        associated, plaintext = sealer.open(read_private(directory / RAW_FILE), secret)
    except VaultError as exc:
        raise ExperimentRefused(f"raw_capture_unreadable:{exc.label}") from None
    try:
        document = json.loads(plaintext)
    except ValueError:
        raise ExperimentRefused("raw_capture_invalid") from None
    faces = document.get("faces") if isinstance(document, dict) else None
    if (not isinstance(document, dict) or document.get("format") != RAW_FORMAT or associated.get("format") != RAW_FORMAT
            or not isinstance(faces, list) or associated.get("faces") != len(faces)):
        raise ExperimentRefused("raw_capture_invalid")
    for face in faces:
        if not (isinstance(face, dict) and set(face) == {"t", "seq", "epoch", "track", "e"}
                and isinstance(face["t"], int) and isinstance(face["e"], list) and face["e"]):
            raise ExperimentRefused("raw_capture_invalid")
    return document
