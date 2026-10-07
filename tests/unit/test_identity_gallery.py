"""The identity gallery (V2-20 demo form for V2-25): numbers only, full compatibility, sealed, revocable.

Store tests seal with the real helper (system Python, synthetic embeddings) and are skipped where it is unusable.
"""

from __future__ import annotations

import json
import math
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from sentinel.identity.gallery import (
    CONSENT_SCOPE,
    FaceCompatibility,
    GalleryDocument,
    GalleryIdentity,
    IdentityStore,
    new_identity_id,
    unit,
)
from sentinel.identity.matching import best_match
from sentinel.identity.vault import Sealer, Secret, VaultError

SECRET = Secret("a long synthetic passphrase")
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
COMPAT = FaceCompatibility(
    deepface_version="0.0.99", model_name="Facenet512", model_sha256="a" * 64, detector_backend="yunet",
    detector_sha256="b" * 64, opencv_version="4.13.0", align=True, normalization="base", expand_percentage=0,
    color_order="BGR", dimension=4, prototype_normalization="l2", tensorflow_version="2.21.0",
    tf_keras_version="2.21.0", numpy_version="2.2.6",
)


def identity(identity_id: str = "idn-000000000001", *prototypes: tuple[float, ...]) -> GalleryIdentity:
    return GalleryIdentity(identity_id=identity_id, enrolled_utc=NOW, consent_date="2026-10-07",
                           consent_scope=CONSENT_SCOPE, prototypes=prototypes or (unit((1.0, 0.0, 0.0, 0.0)),))


def document(*identities: GalleryIdentity) -> GalleryDocument:
    return GalleryDocument(compatibility=COMPAT, identities=identities)


@pytest.fixture
def store(tmp_path: Path) -> IdentityStore:
    sealer = Sealer(kdf_n=2**14)
    try:
        sealer.check()
    except VaultError as exc:
        pytest.skip(f"the system Python's cryptography is not usable here ({exc.label})")
    store = IdentityStore(tmp_path / "identity", sealer, utc_now=lambda: NOW)
    store.ensure_directory()
    return store


# ---------------------------------------------------------------- the document


def test_every_embedding_affecting_setting_must_match() -> None:
    for name, other in [("model_sha256", "c" * 64), ("detector_backend", "retinaface"), ("align", False),
                        ("normalization", "Facenet2018"), ("expand_percentage", 10), ("dimension", 512),
                        ("opencv_version", "4.12.0"), ("tensorflow_version", "2.20.0"), ("deepface_version", "0.0.98"),
                        ("tf_keras_version", "2.20.0"), ("numpy_version", "1.26.4"), ("detector_sha256", "d" * 64)]:
        changed = COMPAT.model_copy(update={name: other})
        assert COMPAT.first_difference(changed) == name
        assert changed.revision != COMPAT.revision
    assert COMPAT.first_difference(COMPAT) is None
    assert COMPAT.revision.startswith("Facenet512+yunet+deepface-0.0.99+")


@pytest.mark.parametrize("make", [
    lambda: document(identity(), identity()),  # duplicate IDs
    lambda: document(identity("idn-000000000001", (1.0, 0.0, 0.0))),  # wrong dimension
    lambda: document(identity("idn-000000000001", (0.5, 0.0, 0.0, 0.0))),  # not unit length
    lambda: document(identity("idn-000000000001", *[unit((1.0, 0.0, 0.0, 0.0))] * 9)),  # over MAX_PROTOTYPES
    lambda: document(identity("Alice")),  # a name is not an identity ID
    lambda: GalleryIdentity.model_validate({**identity().model_dump(), "name": "Alice"}),  # no name field
    lambda: GalleryIdentity.model_validate({**identity().model_dump(), "consent_date": "2026-13-40"}),
    lambda: GalleryIdentity.model_validate({**identity().model_dump(), "consent_scope": "anything"}),
    lambda: document(identity("idn-000000000001", (math.nan, 1.0, 0.0, 0.0))),
])
def test_an_invalid_gallery_is_refused(make) -> None:
    with pytest.raises((ValidationError, ValueError)):
        make()


def test_the_gallery_gives_the_matcher_its_enrollment() -> None:
    doc = document(identity("idn-000000000001", unit((1.0, 0.0, 0.0, 0.0))),
                   identity("idn-000000000002", unit((0.0, 1.0, 0.0, 0.0))))
    enrollment = doc.enrollment()
    assert enrollment.model_revision == COMPAT.revision and enrollment.dimension == 4
    match = best_match((0.9, 0.1, 0.0, 0.0), enrollment)
    assert match.identity_id == "idn-000000000001" and match.runner_up is not None
    assert [i.identity_id for i in doc.without("idn-000000000001").identities] == ["idn-000000000002"]


def test_identity_ids_are_opaque_and_new() -> None:
    tokens = iter(["000000000001", "000000000002"])
    assert new_identity_id({"idn-000000000001"}, lambda n: next(tokens)) == "idn-000000000002"
    assert new_identity_id(set()).startswith("idn-") and len(new_identity_id(set())) == 16


# ---------------------------------------------------------------- the store (real sealing)


def test_a_saved_gallery_loads_only_with_its_secret(store) -> None:
    doc = document(identity())
    store.save(doc, SECRET)
    assert stat.S_IMODE(store.gallery_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700 and stat.S_IMODE(store.inbox.stat().st_mode) == 0o700
    assert store.load(SECRET) == doc
    with pytest.raises(VaultError, match="authentication_failed"):
        store.load(Secret("another long passphrase"))
    raw = store.gallery_path.read_bytes()
    assert b"idn-000000000001" not in raw and b"prototypes" not in raw


def test_the_sealed_header_and_the_record_must_agree(store, tmp_path) -> None:
    doc = document(identity())
    other = COMPAT.model_copy(update={"opencv_version": "4.12.0"})
    sealer = Sealer(kdf_n=2**14)
    plaintext = json.dumps(doc.model_dump(mode="json")).encode()
    store.gallery_path.write_bytes(sealer.seal(plaintext, SECRET, {
        "schema": "sentinel-identity-gallery", "schema_version": 1, "compatibility": other.model_dump(mode="json")}))
    store.gallery_path.chmod(0o600)
    with pytest.raises(VaultError, match="gallery_inconsistent"):
        store.load(SECRET)
    store.gallery_path.write_bytes(sealer.seal(b'{"schema_version": 1}', SECRET, {}))
    with pytest.raises(VaultError, match="gallery_invalid"):
        store.load(SECRET)


def test_revoke_removes_one_identity_and_purge_removes_everything(store) -> None:
    store.save(document(identity("idn-000000000001"), identity("idn-000000000002", unit((0.0, 1.0, 0.0, 0.0)))),
               SECRET)
    assert store.revoke("idn-000000000001", SECRET) is True
    assert [i.identity_id for i in store.load(SECRET).identities] == ["idn-000000000002"]
    assert store.revoke("idn-000000000001", SECRET) is False
    (store.inbox / "batch").mkdir()
    (store.inbox / "batch" / "photo-1.jpg").write_bytes(b"synthetic")
    assert store.purge() == {"gallery": 1, "inbox_files": 1}
    assert not store.gallery_path.exists() and list(store.inbox.iterdir()) == []
    lines = [json.loads(line) for line in store.audit_path.read_text().splitlines()]
    assert [(line["action"], line["identity_id"]) for line in lines] == [
        ("revoke", "idn-000000000001"), ("purge", None)]
    assert stat.S_IMODE(store.audit_path.stat().st_mode) == 0o600
    assert all(set(line) <= {"utc", "action", "identity_id", "identities_left", "gallery", "inbox_files"}
               for line in lines)


def test_audit_lines_take_counts_only(store) -> None:
    with pytest.raises(ValueError):
        store.audit("enroll", "idn-000000000001", label="Alice")  # type: ignore[arg-type]
    assert not store.audit_path.exists()


def test_the_identity_directory_must_be_private(store) -> None:
    store.directory.chmod(0o750)
    with pytest.raises(VaultError, match="identity_dir_not_private"):
        store.load(SECRET)
    with pytest.raises(VaultError, match="identity_dir_not_private"):
        store.ensure_directory()
    store.directory.chmod(0o700)


def test_only_a_folder_inside_the_inbox_can_be_removed(store, tmp_path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "keep.jpg").write_bytes(b"x")
    with pytest.raises(VaultError, match="photos_not_in_inbox"):
        store.remove_tree(outside)
    link = store.inbox / "link"
    link.symlink_to(outside)
    with pytest.raises(VaultError, match="photos_not_in_inbox"):
        store.remove_tree(link)
    assert (outside / "keep.jpg").exists()
    batch = store.inbox / "batch"
    batch.mkdir()
    (batch / "a.jpg").write_bytes(b"x")
    assert store.remove_tree(batch) == 1 and not batch.exists()
