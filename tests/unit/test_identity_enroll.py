"""Consented enrollment and gallery commands (V2-25 demo form). A fake face pipeline returns synthetic embeddings;
the gallery is sealed with the real helper (skipped where it is unusable). No photo of anyone is used."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sentinel.cli import main
from sentinel.identity.enroll import EnrollmentRefused, enroll
from sentinel.identity.gallery import FaceCompatibility, IdentityStore, unit
from sentinel.identity.legacy_deepface import EnrollmentFace, static_compatibility_fields
from sentinel.identity.vault import SECRET_ENV, Sealer, Secret, VaultError

SECRET = Secret("a long synthetic passphrase")
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
COMPAT = FaceCompatibility(**static_compatibility_fields(), opencv_version="4.13.0", tensorflow_version="2.21.0",
                           tf_keras_version="2.21.0", numpy_version="2.2.6")


def vec(*head: float) -> tuple[float, ...]:
    return unit(tuple(head) + (0.0,) * (512 - len(head)))


class Pipeline:
    """A photo's content selects the synthetic result: ``a:<n>`` person A, ``b:<n>`` person B, ``none``, ``two``,
    ``low``, ``other`` (a face unlike the rest)."""

    def __init__(self, compatibility: FaceCompatibility = COMPAT) -> None:
        self.loads = 0
        self._compatibility = compatibility

    def load(self) -> None:
        self.loads += 1

    def compatibility(self) -> FaceCompatibility:
        return self._compatibility

    def enrollment_photo(self, path: Path):
        kind = path.read_text()
        if kind == "none":
            return None, "no_face"
        if kind == "two":
            return None, "several_faces"
        if kind == "low":
            return EnrollmentFace(vec(1.0), 0.3, 40), "face_found"
        if kind == "other":
            return EnrollmentFace(vec(0.0, 0.0, 1.0), 0.9, 120), "face_found"
        person, n = kind.split(":")
        base = (1.0, 0.05 * int(n)) if person == "a" else (0.0, 1.0, 0.05 * int(n))
        return EnrollmentFace(vec(*base), 0.9, 120), "face_found"


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


def photos(store: IdentityStore, *kinds: str, folder: str = "Alice Example") -> Path:
    directory = store.inbox / folder
    directory.mkdir()
    for index, kind in enumerate(kinds):
        (directory / f"Alice Example {index}.jpg").write_text(kind)
    return directory


def run(store, folder, **kwargs):
    defaults = {"secret": SECRET, "consent_date": "2026-10-07", "min_quality": 0.6, "match_threshold": 0.70,
                "utc_now": lambda: NOW}
    return enroll(store, kwargs.pop("pipeline", Pipeline()), folder, **{**defaults, **kwargs})


def test_a_consistent_set_enrolls_one_opaque_identity_and_deletes_the_photos(store) -> None:
    folder = photos(store, "a:1", "a:2", "a:3")
    report = run(store, folder)
    assert report["accepted"] == 3 and report["verified"] is True
    assert (report["photos_deleted"], report["photos_changed_not_deleted"], report["folder_entries_left"]) == (3, 0, 0)
    assert not folder.exists()
    (identity,) = store.load(SECRET).identities
    assert identity.identity_id == report["identity_id"] and len(identity.prototypes) == 3
    assert identity.consent_date == "2026-10-07"
    text = json.dumps(report)
    assert "Alice" not in text and "jpg" not in text and "prototypes" not in text
    (line,) = [json.loads(x) for x in store.audit_path.read_text().splitlines()]
    assert (line["action"], line["identity_id"], line["photos_used"], line["photos_rejected"]) == (
        "enroll", identity.identity_id, 3, 0)


def test_a_dry_run_writes_and_deletes_nothing(store) -> None:
    folder = photos(store, "a:1", "a:2", "none")
    report = run(store, folder, dry_run=True)
    assert report["accepted"] == 2 and [r["reason"] for r in report["results"]] == [
        "face_found", "face_found", "no_face"]
    assert "identity_id" not in report and not store.exists() and len(list(folder.iterdir())) == 3
    assert not store.audit_path.exists()


@pytest.mark.parametrize(("kinds", "label"), [
    (("a:1", "none", "two"), "too_few_usable_photos"),
    (("a:1", "low", "none"), "too_few_usable_photos"),
    (("a:1", "a:2", "other"), "inconsistent_photos"),
])
def test_unusable_or_inconsistent_photos_are_refused_without_writing(store, kinds, label) -> None:
    folder = photos(store, *kinds)
    with pytest.raises(EnrollmentRefused) as refused:
        run(store, folder)
    assert refused.value.label == label and not store.exists() and len(list(folder.iterdir())) == len(kinds)
    if label == "inconsistent_photos":
        assert refused.value.report["inconsistent_photos"] == [3]


def test_more_photos_than_prototypes_are_refused_before_the_model_loads(store) -> None:
    pipeline = Pipeline()
    with pytest.raises(EnrollmentRefused, match="too_many_photos"):
        run(store, photos(store, *[f"a:{i}" for i in range(9)]), pipeline=pipeline)
    assert pipeline.loads == 0


def test_photos_must_be_inside_the_private_inbox(store, tmp_path) -> None:
    outside = tmp_path / "Pictures"
    outside.mkdir()
    (outside / "x.jpg").write_text("a:1")
    with pytest.raises(VaultError, match="photos_not_in_inbox"):
        run(store, outside)
    assert (outside / "x.jpg").exists()


def test_a_second_person_close_to_an_enrolled_one_or_an_incompatible_gallery_is_refused(store) -> None:
    run(store, photos(store, "a:1", "a:2", folder="first"))
    with pytest.raises(EnrollmentRefused, match="close_to_enrolled_identity"):
        run(store, photos(store, "a:3", "a:4", folder="second"))
    other = Pipeline(COMPAT.model_copy(update={"opencv_version": "4.12.0"}))
    with pytest.raises(EnrollmentRefused, match="gallery_incompatible:opencv_version"):
        run(store, photos(store, "b:1", "b:2", folder="third"), pipeline=other)
    report = run(store, store.inbox / "third")
    assert len(store.load(SECRET).identities) == 2 and report["verified"] is True


# ---------------------------------------------------------------- the commands


def cli(store: IdentityStore, tmp_path: Path, *args: str) -> list[str]:
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 1\ncamera: {id: cam-1}\n")
    command, *rest = args
    return ["identity", command, str(config), *rest, "--identity-dir", str(store.directory)]


def call(store, tmp_path, monkeypatch, *args, secret: str | None = SECRET.reveal()):
    if secret is None:
        monkeypatch.delenv(SECRET_ENV, raising=False)
    else:
        monkeypatch.setenv(SECRET_ENV, secret)
    code = main(cli(store, tmp_path, *args), face_backend=lambda weights: Pipeline(),
                identity_store=lambda directory: store)
    return code


def test_enrollment_needs_explicit_consent_and_a_valid_date(store, tmp_path, monkeypatch, capsys) -> None:
    folder = photos(store, "a:1", "a:2")
    assert call(store, tmp_path, monkeypatch, "enroll", str(folder), "--consent-date", "2026-10-07") == 1
    assert "consent_not_confirmed" in capsys.readouterr().err
    assert call(store, tmp_path, monkeypatch, "enroll", str(folder), "--consent-confirmed",
                "--consent-date", "07/10/2026") == 1
    assert "consent_date_invalid" in capsys.readouterr().err
    assert not store.exists() and len(list(folder.iterdir())) == 2


def test_the_commands_enroll_list_revoke_and_purge_without_names(store, tmp_path, monkeypatch, capsys) -> None:
    folder = photos(store, "a:1", "a:2")
    assert call(store, tmp_path, monkeypatch, "enroll", str(folder), "--consent-confirmed",
                "--consent-date", "2026-10-07") == 0
    assert SECRET_ENV not in os.environ
    report = json.loads(capsys.readouterr().out)
    identity_id = report["identity_id"]
    assert call(store, tmp_path, monkeypatch, "list") == 0
    (listed,) = json.loads(capsys.readouterr().out)["identities"]
    assert {k: listed[k] for k in ("identity_id", "prototypes", "consent_date")} == {
        "identity_id": identity_id, "prototypes": 2, "consent_date": "2026-10-07"}
    assert set(listed) == {"identity_id", "prototypes", "consent_date", "enrolled_utc"}
    assert datetime.fromisoformat(listed["enrolled_utc"]).microsecond == 0
    assert call(store, tmp_path, monkeypatch, "list", secret="the wrong long passphrase") == 1
    assert capsys.readouterr().err.strip() == "identity: authentication_failed"
    assert call(store, tmp_path, monkeypatch, "revoke", identity_id) == 0
    assert json.loads(capsys.readouterr().out)["revoked"] is True
    assert call(store, tmp_path, monkeypatch, "purge") == 1
    assert "purge needs --yes" in capsys.readouterr().err
    assert call(store, tmp_path, monkeypatch, "purge", "--yes") == 0
    assert json.loads(capsys.readouterr().out) == {"identity": "purge", "gallery": 1, "inbox_files": 0}
    assert not store.exists()


def test_a_new_gallery_needs_a_long_enough_passphrase(store, tmp_path, monkeypatch, capsys) -> None:
    folder = photos(store, "a:1", "a:2")
    assert call(store, tmp_path, monkeypatch, "enroll", str(folder), "--consent-confirmed",
                "--consent-date", "2026-10-07", secret="short") == 1
    assert capsys.readouterr().err.strip() == "identity: secret_too_short"
    assert not store.exists()


def test_a_refused_enrollment_reports_by_photo_index(store, tmp_path, monkeypatch, capsys) -> None:
    folder = photos(store, "a:1", "none", "two")
    assert call(store, tmp_path, monkeypatch, "enroll", str(folder), "--consent-confirmed",
                "--consent-date", "2026-10-07") == 1
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["refused"] == "too_few_usable_photos" and [r["photo"] for r in report["results"]] == [1, 2, 3]
    assert "Alice" not in captured.out and "Alice" not in captured.err


def test_photos_in_a_nested_folder_or_with_extras_are_refused_before_the_model_loads(store) -> None:
    nested = store.inbox / "outer" / "inner"
    nested.mkdir(parents=True)
    (nested / "x.jpg").write_text("a:1")
    pipeline = Pipeline()
    with pytest.raises(VaultError, match="photos_not_in_inbox"):
        run(store, nested, pipeline=pipeline)
    folder = photos(store, "a:1", "a:2")
    (folder / "sub").mkdir()
    with pytest.raises(VaultError, match="photos_folder_not_plain"):
        run(store, folder, pipeline=pipeline)
    assert pipeline.loads == 0 and len(list(folder.iterdir())) == 3


def test_nothing_is_deleted_unless_the_reloaded_gallery_equals_what_was_saved(store, monkeypatch) -> None:
    folder = photos(store, "a:1", "a:2")
    real_load = store.load
    monkeypatch.setattr(store, "load", lambda secret: real_load(secret).without(
        next(iter(i.identity_id for i in real_load(secret).identities))))
    with pytest.raises(VaultError, match="enrollment_not_verified"):
        run(store, folder)
    assert len(list(folder.iterdir())) == 2 and not store.audit_path.exists()


def test_a_file_added_during_enrollment_is_kept_with_its_folder(store, monkeypatch) -> None:
    folder = photos(store, "a:1", "a:2")
    pipeline = Pipeline()
    real = pipeline.enrollment_photo

    def add_one(path):
        if not (folder / "late.jpg").exists():
            (folder / "late.jpg").write_text("a:9")
        return real(path)

    monkeypatch.setattr(pipeline, "enrollment_photo", add_one)
    report = run(store, folder, pipeline=pipeline)
    assert (report["photos_deleted"], report["folder_entries_left"]) == (2, 1)
    assert [p.name for p in folder.iterdir()] == ["late.jpg"]
