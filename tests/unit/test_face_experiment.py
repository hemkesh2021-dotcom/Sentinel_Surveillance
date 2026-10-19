"""D91, the camera-side enrollment experiment's working gallery: the predeclared selection, the capture's checks and
its sealed write (with the real sealing helper; skipped where it is unusable), and the read-only comparison with a
working gallery. Synthetic embeddings only; no image of anyone is used."""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sealer_faults import Tampering, timing_out

from sentinel.identity.experiment import (
    CAPTURE_MAX,
    CAPTURE_MIN_FACES,
    CAPTURE_MIN_PROTOTYPES,
    CAPTURE_SEGMENTS,
    CAPTURE_WASHOUT_S,
    Capture,
    ExperimentRefused,
    open_shadow,
    select,
)
from sentinel.identity.gallery import CONSENT_SCOPE, FaceCompatibility, GalleryDocument, GalleryIdentity, IdentityStore, unit
from sentinel.identity.legacy_deepface import static_compatibility_fields
from sentinel.identity.matching import MAX_PROTOTYPES, best_match
from sentinel.identity.vault import Sealer, Secret, VaultError

SECRET = Secret("a long synthetic passphrase")
NOW = datetime(2026, 10, 14, 9, 0, tzinfo=timezone.utc)
COMPAT = FaceCompatibility(**static_compatibility_fields(), opencv_version="4.13.0", tensorflow_version="2.21.0",
                           tf_keras_version="2.21.0", numpy_version="2.2.6")
ENROLLED = "idn-00000000000a"
NS = 1_000_000_000


def vec(*head: float) -> tuple[float, ...]:
    return unit(tuple(head) + (0.0,) * (512 - len(head)))


ORIGINAL = GalleryDocument(compatibility=COMPAT, identities=(GalleryIdentity(
    identity_id=ENROLLED, enrolled_utc=NOW, consent_date="2026-10-08", consent_scope=CONSENT_SCOPE,
    prototypes=(vec(1.0), vec(1.0, 0.2))),))


@pytest.fixture
def sealer() -> Sealer:
    made = Sealer(kdf_n=2**14)
    try:
        made.check()
    except VaultError as exc:
        pytest.skip(f"the system Python's cryptography is not usable here ({exc.label})")
    return made


def near(seed: int, spread: float = 0.15) -> tuple[float, ...]:
    """A camera face of the enrolled person: the first axis, with some variation."""
    rng = random.Random(seed)
    return vec(1.0, *(rng.uniform(-spread, spread) for _ in range(6)))


# ---------------------------------------------------------------- the predeclared selection


def test_one_face_per_equal_segment_nearest_its_middle_after_the_washouts() -> None:
    times = [round(t * NS) for t in range(0, 301)]  # a face every second for 300 s
    chosen, in_span = select(times)
    start, end = CAPTURE_WASHOUT_S, 300 - CAPTURE_WASHOUT_S  # 15 .. 285 s
    assert in_span == 271
    length = (end - start) / CAPTURE_SEGMENTS
    middles = [start + (k + 0.5) * length for k in range(CAPTURE_SEGMENTS)]
    assert [times[i] / NS for i in chosen] == [float(round(m)) for m in middles]  # 32, 66, 99, ... 268 s
    assert len(chosen) == CAPTURE_SEGMENTS == MAX_PROTOTYPES


def test_a_segment_without_a_face_gives_none_and_order_of_keeping_does_not_matter() -> None:
    times = [round(t * NS) for t in [*range(0, 100), *range(200, 301)]]  # nothing from 100 to 199 s
    chosen, _ = select(times)
    picked = sorted(times[i] / NS for i in chosen)
    assert all(not 100 <= t < 200 for t in picked) and len(picked) < CAPTURE_SEGMENTS
    shuffled = list(times)
    random.Random(1).shuffle(shuffled)
    again, _ = select(shuffled)
    assert sorted(shuffled[i] / NS for i in again) == picked


def test_nothing_is_selected_from_too_short_a_capture() -> None:
    assert select([]) == ([], 0)
    assert select([0, 20 * NS]) == ([], 0)  # the washouts leave no span


# ---------------------------------------------------------------- the capture


def capture(tmp_path: Path, sealer: Sealer, **kwargs) -> Capture:
    store = IdentityStore(tmp_path / "experiment", sealer, utc_now=lambda: NOW)
    return Capture(store, SECRET, kwargs.pop("original", ORIGINAL), kwargs.pop("consent_date", "2026-10-14"),
                   0.70, utc_now=lambda: NOW, **kwargs)


def feed(made: Capture, seconds: int = 300, face=near) -> None:
    for t in range(seconds):
        made.offer(t * NS, face(t))


def test_a_capture_writes_a_sealed_working_gallery_that_reads_back_and_audits_counts_only(tmp_path, sealer) -> None:
    made = capture(tmp_path, sealer)
    feed(made)
    report = made.finish()
    assert report["written"] is True and report["verified"] is True and report["refused"] is None
    assert report["prototypes"] == CAPTURE_SEGMENTS and report["faces_kept"] == 300 and report["faces_in_span"] == 270
    assert len(report["similarity_to_original"]) == CAPTURE_SEGMENTS and report["min_best_to_others"] >= 0.70
    document = made.store.load(SECRET)
    (identity,) = document.identities
    assert document.compatibility == ORIGINAL.compatibility and identity.identity_id == ENROLLED
    assert identity.consent_date == "2026-10-14" and len(identity.prototypes) == CAPTURE_SEGMENTS
    times = list(range(300))
    chosen, _ = select([t * NS for t in times])
    expected = [near(times[i]) for i in chosen]  # the predeclared faces, nothing else (re-normalized: last bits only)
    assert len(identity.prototypes) == len(expected) and all(
        math.isclose(a, b, abs_tol=1e-12) for p, q in zip(identity.prototypes, expected) for a, b in zip(p, q))
    (line,) = made.store.audit_path.read_text().splitlines()
    assert json.loads(line) == {"utc": NOW.isoformat(), "action": "capture", "identity_id": ENROLLED,
                                "prototypes_used": CAPTURE_SEGMENTS, "faces_in_span": 270}
    assert made.kept == 0  # the embeddings are discarded once written
    assert made.finish() is report  # written once
    assert set(report) == {"identity", "written", "refused", "faces_kept", "faces_dropped_over_cap", "faces_in_span",
                           "prototypes", "min_best_to_others", "similarity_to_original", "verified"}
    assert all(isinstance(v, float) for v in report["similarity_to_original"])  # numbers only, never an embedding


@pytest.mark.parametrize("case, reason", [
    ("short", "too_few_faces"),
    ("gappy", "too_few_prototypes"),
    ("someone_else", "inconsistent_capture"),
])
def test_a_capture_that_fails_a_check_writes_nothing(tmp_path, sealer, case, reason) -> None:
    made = capture(tmp_path, sealer)
    if case == "short":
        feed(made, seconds=60)  # 30 faces inside the span: under CAPTURE_MIN_FACES
    elif case == "gappy":
        for t in [*range(0, 20), *range(150, 175), *range(280, 301)]:  # faces in too few segments
            made.offer(t * NS, near(t))
        for t in range(150, 175):
            made.offer(t * NS + NS // 2, near(t + 1000))
    else:  # the faces at one segment's middle belong to someone unlike the rest
        feed(made, face=lambda t: vec(0.0, 0.0, 1.0) if 60 <= t <= 72 else near(t))  # around 65.6 s
    report = made.finish()
    assert (report["written"], report["refused"], report["verified"]) == (False, reason, False)
    assert not made.store.exists() and not made.store.audit_path.exists()
    assert CAPTURE_MIN_FACES == 40 and CAPTURE_MIN_PROTOTYPES == 4


def test_a_capture_never_overwrites_a_gallery_and_needs_one_identity_and_a_date(tmp_path, sealer) -> None:
    existing = IdentityStore(tmp_path / "experiment", sealer, utc_now=lambda: NOW)
    existing.ensure_directory()
    existing.save(ORIGINAL, SECRET)
    with pytest.raises(ExperimentRefused, match="face_capture_target_has_gallery"):
        capture(tmp_path, sealer)
    two = GalleryDocument(compatibility=COMPAT, identities=(
        ORIGINAL.identities[0], GalleryIdentity(identity_id="idn-00000000000b", enrolled_utc=NOW,
                                                consent_date="2026-10-08", consent_scope=CONSENT_SCOPE,
                                                prototypes=(vec(0.0, 1.0),))))
    with pytest.raises(ExperimentRefused, match="face_capture_needs_one_identity"):
        capture(tmp_path / "x", sealer, original=two)
    with pytest.raises(ValueError):
        capture(tmp_path / "y", sealer, consent_date="14 October")
    late = capture(tmp_path / "z", sealer)
    feed(late)
    other = IdentityStore(tmp_path / "z" / "experiment", sealer, utc_now=lambda: NOW)
    other.ensure_directory()
    other.save(ORIGINAL, SECRET)  # a gallery appeared during the run: refused at the end, nothing overwritten
    assert late.finish()["refused"] == "capture_target_has_gallery" and other.load(SECRET) == ORIGINAL


def test_a_capture_keeps_at_most_its_cap_and_counts_the_rest(tmp_path, sealer) -> None:
    made = capture(tmp_path, sealer)
    for t in range(CAPTURE_MAX + 25):
        made.offer(t * NS // 2, near(t))
    assert made.kept == CAPTURE_MAX and made.dropped_over_cap == 25
    assert made.finish()["faces_dropped_over_cap"] == 25


@pytest.mark.parametrize(("failure", "refused"), [
    ("read-back-differs", "capture_not_verified"),
    ("read-back-times-out", "capture_write_failed"),
    ("audit-file-not-private", "capture_write_failed"),
])
def test_a_capture_whose_write_fails_after_sealing_leaves_no_gallery_to_open(tmp_path, sealer, failure,
                                                                             refused) -> None:
    directory = tmp_path / "experiment"
    if failure == "audit-file-not-private":
        IdentityStore(directory, sealer).ensure_directory()
        (directory / "audit.jsonl").touch()
        (directory / "audit.jsonl").chmod(0o644)
    # The sealer's calls: the working gallery is sealed, then read back.
    made = capture(tmp_path, {"read-back-differs": Tampering(nth=1),
                              "read-back-times-out": timing_out(nth=2)}.get(failure, sealer))
    feed(made)
    report = made.finish()
    assert (report["written"], report["refused"], report["verified"]) == (False, refused, False)
    assert {p.name for p in directory.iterdir()} <= {"inbox", "audit.jsonl"}  # no gallery, nothing staged
    assert not (directory / "audit.jsonl").exists() or (directory / "audit.jsonl").read_bytes() == b""
    with pytest.raises(ExperimentRefused, match="face_shadow_missing"):  # a later --face-shadow-gallery run
        open_shadow(IdentityStore(directory, sealer), SECRET, ORIGINAL)


# ---------------------------------------------------------------- the working gallery, compared read only


def test_the_shadow_gives_the_working_gallerys_similarity_and_position_only(tmp_path, sealer) -> None:
    made = capture(tmp_path, sealer)
    feed(made)
    made.finish()
    shadow = open_shadow(made.store, SECRET, ORIGINAL)
    assert shadow.prototypes == CAPTURE_SEGMENTS
    face = near(7)
    expected = best_match(face, made.store.load(SECRET).enrollment())
    assert shadow.numbers(face) == {"similarity": round(expected.similarity, 3), "prototype": expected.prototype}
    assert shadow.numbers(None) is None and shadow.numbers((1.0, 0.0)) is None


def test_a_shadow_must_exist_open_match_and_hold_only_the_originals_identities(tmp_path, sealer) -> None:
    store = IdentityStore(tmp_path / "experiment", sealer, utc_now=lambda: NOW)
    with pytest.raises(ExperimentRefused, match="face_shadow_missing"):
        open_shadow(store, SECRET, ORIGINAL)
    store.ensure_directory()
    store.save(ORIGINAL, SECRET)
    with pytest.raises(ExperimentRefused, match="face_shadow_unreadable"):
        open_shadow(store, Secret("another synthetic passphrase"), ORIGINAL)
    changed = ORIGINAL.model_copy(update={"compatibility": COMPAT.model_copy(update={"numpy_version": "2.3.0"})})
    with pytest.raises(ExperimentRefused, match="face_shadow_incompatible:numpy_version"):
        open_shadow(store, SECRET, changed)
    stranger = GalleryDocument(compatibility=COMPAT, identities=(GalleryIdentity(
        identity_id="idn-00000000000b", enrolled_utc=NOW, consent_date="2026-10-14", consent_scope=CONSENT_SCOPE,
        prototypes=(vec(0.0, 1.0),)),))
    store.save(stranger, SECRET)
    with pytest.raises(ExperimentRefused, match="face_shadow_unknown_identity"):
        open_shadow(store, SECRET, ORIGINAL)
