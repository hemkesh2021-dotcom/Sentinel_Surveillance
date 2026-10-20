"""Consented enrollment and gallery commands (V2-25 demo form). A fake face pipeline returns synthetic embeddings;
the gallery is sealed with the real helper (skipped where it is unusable). No photo of anyone is used."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sealer_faults import Pausing, Tampering, timing_out

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
    """A photo's content selects the synthetic result: ``a:<n>`` person A, ``b:<n>`` person B, ``c:<n>`` person C,
    ``none``, ``two``, ``low``, ``other`` (a face unlike the rest)."""

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
        base = {"a": (1.0, 0.05 * int(n)), "b": (0.0, 1.0, 0.05 * int(n)),
                "c": (0.0, 0.0, 0.0, 1.0, 0.05 * int(n))}[person]
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
    lines = [json.loads(x) for x in store.audit_path.read_text().splitlines()]  # started before the rename, then done
    assert [(line["action"], line["identity_id"], line["photos_used"], line["photos_rejected"]) for line in lines] == [
        ("enroll_started", identity.identity_id, 3, 0), ("enroll", identity.identity_id, 3, 0)]


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


def call(store, tmp_path, monkeypatch, *args, secret: str | None = SECRET.reveal(), backend=None):
    if secret is None:
        monkeypatch.delenv(SECRET_ENV, raising=False)
    else:
        monkeypatch.setenv(SECRET_ENV, secret)
    code = main(cli(store, tmp_path, *args), face_backend=lambda weights: backend or Pipeline(),
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


@pytest.mark.parametrize("text", ["20261019", "2026-W42-1", "2026-10-19\n", "2026-02-30"])
def test_enrollment_takes_only_the_gallerys_own_date_form_before_the_model_loads(
        store, tmp_path, monkeypatch, capsys, text) -> None:
    # date.fromisoformat also takes the basic and week forms on Python 3.11+; the gallery stores YYYY-MM-DD only.
    folder = photos(store, "a:1", "a:2")
    assert call(store, tmp_path, monkeypatch, "enroll", str(folder), "--consent-confirmed",
                "--consent-date", text) == 1
    assert "consent_date_invalid" in capsys.readouterr().err
    pipeline = Pipeline()
    with pytest.raises(ValueError):
        run(store, folder, consent_date=text, pipeline=pipeline)
    assert pipeline.loads == 0 and not store.exists() and len(list(folder.iterdir())) == 2


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


def files(store: IdentityStore) -> dict[str, bytes | str]:
    """The identity directory's own entries (not the inbox, nor the empty lock file): bytes, or a link's target."""
    return {p.name: os.readlink(p) if p.is_symlink() else p.read_bytes()
            for p in store.directory.iterdir() if (p.is_symlink() or p.is_file()) and p.name != ".gallery.lock"}


FAILURE_LABELS = {
    "read-back-differs": "enrollment_not_verified",
    "read-back-times-out": "sealer_timeout",
    "audit-file-not-private": "audit_file_not_private",  # e.g. a copy restored with the wrong mode
    "audit-write-fails": "audit_failed:OSError",  # here the log's name is a link, which O_NOFOLLOW refuses
}


@pytest.mark.parametrize("failure", FAILURE_LABELS)
@pytest.mark.parametrize("enrolled_before", [0, 1], ids=["first-person", "second-person"])
def test_a_failure_after_sealing_leaves_gallery_and_audit_unchanged_and_a_retry_enrolls(
        store, enrolled_before, failure) -> None:
    if enrolled_before:
        run(store, photos(store, "b:1", "b:2", folder="earlier"))
    kept_audit = store.audit_path.read_bytes() if store.audit_path.exists() else None
    if failure == "audit-file-not-private":
        store.audit_path.touch()
        store.audit_path.chmod(0o644)
    elif failure == "audit-write-fails":
        store.audit_path.unlink(missing_ok=True)
        store.audit_path.symlink_to(store.directory / "elsewhere")
    # The sealer's calls: an existing gallery is opened first, then the new one is sealed and read back.
    sealer = {"read-back-differs": Tampering(nth=1 + enrolled_before),
              "read-back-times-out": timing_out(nth=2 + enrolled_before)}.get(failure)
    failing = IdentityStore(store.directory, sealer, utc_now=lambda: NOW) if sealer else store
    folder = photos(store, "a:1", "a:2")
    before = files(store)
    with pytest.raises(VaultError) as failed:
        run(failing, folder)
    assert failed.value.label == FAILURE_LABELS[failure]
    assert files(store) == before  # the gallery as it was (or still absent), no audit line, nothing staged
    assert len(list(folder.iterdir())) == 2  # no photo deleted
    # With the cause gone the same photos enroll: no ``close_to_enrolled_identity`` from a half-done attempt.
    if failure == "audit-file-not-private":
        store.audit_path.chmod(0o600)
    elif failure == "audit-write-fails":
        store.audit_path.unlink()
        if kept_audit is not None:
            store.audit_path.write_bytes(kept_audit)
            store.audit_path.chmod(0o600)
    report = run(store, folder)
    enrolled = {identity.identity_id for identity in store.load(SECRET).identities}
    assert report["verified"] is True and len(enrolled) == enrolled_before + 1 and not folder.exists()
    lines = [json.loads(x) for x in store.audit_path.read_text().splitlines()]
    assert sorted(line["identity_id"] for line in lines if line["action"] == "enroll") == sorted(enrolled)


def test_the_command_reports_an_audit_failure_as_one_label_and_enrolls_nobody(
        store, tmp_path, monkeypatch, capsys) -> None:
    folder = photos(store, "a:1", "a:2")
    store.audit_path.symlink_to(store.directory / "elsewhere")
    assert call(store, tmp_path, monkeypatch, "enroll", str(folder), "--consent-confirmed",
                "--consent-date", "2026-10-07") == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "identity: audit_failed:OSError" and captured.out == ""
    assert not store.exists() and len(list(folder.iterdir())) == 2


def test_a_failed_final_rename_leaves_a_started_and_an_aborted_line_never_a_completed_one(store, monkeypatch) -> None:
    # Session 96's gap 3: the enroll line was written before the rename, so a failed rename left an audit line for
    # an enrollment that never took effect. The completed line now follows the rename.
    import sentinel.identity.gallery as gallery

    def fail(source, path):
        raise VaultError("save_failed:OSError")

    monkeypatch.setattr(gallery, "replace_private", fail)
    folder = photos(store, "a:1", "a:2")
    with pytest.raises(VaultError, match="save_failed:OSError"):
        run(store, folder)
    assert not store.exists() and not store.staged_path.exists() and len(list(folder.iterdir())) == 2
    assert [line["action"] for line in audit_lines(store)] == ["enroll_started", "enroll_aborted"]
    assert audited(store) == enrolled(store) == set()


def test_purge_also_removes_a_staged_gallery_left_by_a_stop_before_the_rename(store) -> None:
    run(store, photos(store, "a:1", "a:2"))
    store.staged_path.write_bytes(store.gallery_path.read_bytes())  # as a stop between the audit and the rename leaves
    store.staged_path.chmod(0o600)
    assert store.purge() == {"gallery": 1, "inbox_files": 0}
    assert not store.exists() and not store.staged_path.exists()


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


# ---------------------------------------------------------------- one change at a time, audited before and after


def audit_lines(store: IdentityStore) -> list[dict]:
    """Every audit line; a torn last line fails here."""
    if not store.audit_path.exists():
        return []
    text = store.audit_path.read_text()
    assert text == "" or text.endswith("\n"), "the audit log ends in a torn line"
    return [json.loads(line) for line in text.splitlines()]


def enrolled(store: IdentityStore) -> set[str]:
    return {i.identity_id for i in store.load(SECRET).identities} if store.exists() else set()


def audited(store: IdentityStore) -> set[str]:
    """Who the completed audit lines say is enrolled: enroll adds, revoke removes, purge clears."""
    ids: set[str] = set()
    for line in audit_lines(store):
        if line["action"] == "enroll":
            ids.add(line["identity_id"])
        elif line["action"] == "revoke":
            ids.discard(line["identity_id"])
        elif line["action"] == "purge":
            ids.clear()
    return ids


def under_way(store: IdentityStore, folder: Path, paused: str):
    """An enrollment of ``folder`` in a thread, through its own store on the same directory (as another process
    would), held after it has read the gallery or just before its first audit line; returns (thread, go, outcome)."""
    outcome: dict = {}
    if paused == "after_reading":
        sealer = Pausing()
        other = IdentityStore(store.directory, sealer, utc_now=lambda: NOW)
        reached, go = sealer.read, sealer.go
    else:
        other = IdentityStore(store.directory, Sealer(kdf_n=2**14), utc_now=lambda: NOW)
        reached, go = threading.Event(), threading.Event()
        audit = other.audit

        def held(*args, **kwargs):
            if not reached.is_set():
                reached.set()
                assert go.wait(60)
            return audit(*args, **kwargs)

        other.audit = held  # type: ignore[method-assign]

    def enrol() -> None:
        try:
            outcome["report"] = run(other, folder)
        except (VaultError, EnrollmentRefused) as exc:
            outcome["error"] = exc.label

    thread = threading.Thread(target=enrol)
    thread.start()
    assert reached.wait(60)
    return thread, go, outcome


@pytest.mark.parametrize("paused", ["after_reading", "before_writing"])
@pytest.mark.parametrize("change", ["enroll", "revoke", "purge"])
def test_a_change_made_while_an_enrollment_is_under_way_is_neither_lost_nor_undone(store, change, paused) -> None:
    # Session 96's gaps 2 and 3: each change reads the gallery, changes it and writes it (through one staging file).
    # Interleaved, the later write lost the earlier change, brought back a revoked or purged identity, or wrote an
    # enroll line for an identity whose gallery write then failed.
    run(store, photos(store, "b:1", "b:2", folder="earlier"))
    (x,) = enrolled(store)
    third = photos(store, "c:1", "c:2", folder="third")

    def other_change() -> None:
        if change == "enroll":
            run(store, third)
        elif change == "revoke":
            assert store.revoke(x, SECRET) is True
        else:
            store.purge()

    thread, go, outcome = under_way(store, photos(store, "a:1", "a:2", folder="second"), paused)
    store.lock_wait_s = 0.2  # the other change does not wait long for the enrollment under way
    try:
        other_change()
        first = "done"
    except VaultError as exc:
        first = exc.label
    go.set()
    thread.join(60)
    assert not thread.is_alive()
    if first == "gallery_busy":
        other_change()  # again, now that nothing else is under way
    assert enrolled(store) == audited(store), (first, outcome)  # no audited change lost, none unaudited
    assert first == "gallery_busy" and "report" in outcome  # one change at a time; the enrollment completed
    assert len(enrolled(store)) == {"enroll": 3, "revoke": 1, "purge": 0}[change]
    assert change != "revoke" or x not in enrolled(store)


def test_a_change_waits_for_the_one_under_way_and_then_reads_what_it_left(store) -> None:
    run(store, photos(store, "b:1", "b:2", folder="earlier"))
    holder = IdentityStore(store.directory, Sealer(kdf_n=2**14))
    folder = photos(store, "a:1", "a:2", folder="second")
    outcome: dict = {}
    with holder.locked():  # another change under way
        thread = threading.Thread(target=lambda: outcome.update(report=run(store, folder)))
        thread.start()
        thread.join(0.5)
        assert thread.is_alive() and not outcome and len(enrolled(store)) == 1  # waiting; nothing written
    thread.join(60)
    assert outcome["report"]["verified"] is True and len(enrolled(store)) == 2 == len(audited(store))


def test_a_change_is_refused_while_another_process_holds_the_lock(store, tmp_path, monkeypatch, capsys) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", "import fcntl, os, sys; fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600); "
         "fcntl.flock(fd, fcntl.LOCK_EX); print('locked', flush=True); sys.stdin.readline()",
         str(store.directory / ".gallery.lock")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    folder = photos(store, "a:1", "a:2")
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        store.lock_wait_s = 0.2
        assert call(store, tmp_path, monkeypatch, "enroll", str(folder), "--consent-confirmed",
                    "--consent-date", "2026-10-07") == 1
        assert capsys.readouterr().err.strip() == "identity: gallery_busy"
        assert not store.exists() and audit_lines(store) == [] and len(list(folder.iterdir())) == 2
    finally:
        holder.communicate("\n", timeout=30)
    run(store, folder)  # the holder has exited: its lock went with it
    assert len(enrolled(store)) == 1 == len(audited(store))


@pytest.mark.parametrize("failure", ["audit-file-not-private", "audit-write-fails"])
def test_a_revoke_whose_audit_line_cannot_be_written_changes_nothing(store, failure) -> None:
    # Session 96's gap 1: revoke wrote the gallery, then audited.
    run(store, photos(store, "a:1", "a:2"))
    (x,) = enrolled(store)
    if failure == "audit-file-not-private":
        store.audit_path.chmod(0o644)
    else:
        store.audit_path.unlink()
        store.audit_path.symlink_to(store.directory / "elsewhere")
    before = files(store)
    with pytest.raises(VaultError):
        store.revoke(x, SECRET)
    assert files(store) == before and enrolled(store) == {x}


class ShortWrites:
    """``os`` as gallery.py sees it, with its writes (only the audit's) scripted per call: "ok", "half" (writes half
    the bytes, as a full disk or a size limit can) or "fail" (ENOSPC); later calls are plain. ``fsync_fails``: that
    many fsync calls fail first (EIO)."""

    def __init__(self, *script: str, fsync_fails: int = 0) -> None:
        self._script = list(script)
        self._fsync_fails = fsync_fails

    def __getattr__(self, name: str):
        return getattr(os, name)

    def fsync(self, fd: int) -> None:
        if self._fsync_fails:
            self._fsync_fails -= 1
            raise OSError(errno.EIO, "Input/output error")
        os.fsync(fd)

    def write(self, fd: int, data) -> int:
        step = self._script.pop(0) if self._script else "ok"
        if step == "fail":
            raise OSError(errno.ENOSPC, "No space left on device")
        data = bytes(data)
        return os.write(fd, data[: len(data) // 2] if step == "half" else data)


def test_an_audit_line_cut_short_is_removed_and_nothing_changes(store, monkeypatch) -> None:
    import sentinel.identity.gallery as gallery

    run(store, photos(store, "b:1", "b:2", folder="earlier"))
    before = files(store)
    monkeypatch.setattr(gallery, "os", ShortWrites("half", "fail"))
    folder = photos(store, "a:1", "a:2")
    with pytest.raises(VaultError, match="audit_failed:OSError"):
        run(store, folder)
    assert files(store) == before and len(list(folder.iterdir())) == 2  # no torn line, no change


def test_an_audit_line_that_cannot_be_flushed_is_removed_and_nothing_changes(store, monkeypatch) -> None:
    import sentinel.identity.gallery as gallery

    run(store, photos(store, "b:1", "b:2", folder="earlier"))
    before = files(store)
    monkeypatch.setattr(gallery, "os", ShortWrites(fsync_fails=1))  # written whole, but not durably
    folder = photos(store, "a:1", "a:2")
    with pytest.raises(VaultError, match="audit_failed:OSError"):
        run(store, folder)
    assert files(store) == before and len(list(folder.iterdir())) == 2


def test_the_lock_is_not_shared_by_two_threads_using_one_store(store) -> None:
    store.ensure_directory()
    store.lock_wait_s = 0.2
    held, release, outcome = threading.Event(), threading.Event(), {}

    def hold() -> None:
        with store.locked():
            held.set()
            assert release.wait(30)

    def change() -> None:
        try:
            store.purge()
            outcome["purge"] = "done"
        except VaultError as exc:
            outcome["purge"] = exc.label

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(30)
    other = threading.Thread(target=change)
    other.start()
    other.join(30)
    release.set()
    holder.join(30)
    assert outcome == {"purge": "gallery_busy"} and audit_lines(store) == []


@pytest.mark.parametrize("enrolled_before", [0, 1], ids=["first-person", "second-person"])
def test_a_completion_line_that_cannot_be_written_undoes_the_change(store, monkeypatch, enrolled_before) -> None:
    import sentinel.identity.gallery as gallery

    if enrolled_before:
        run(store, photos(store, "b:1", "b:2", folder="earlier"))
    gallery_before = store.gallery_path.read_bytes() if store.exists() else None
    lines_before = audit_lines(store)
    monkeypatch.setattr(gallery, "os", ShortWrites("ok", "half", "fail"))  # started; completion cut short; aborted
    folder = photos(store, "a:1", "a:2")
    with pytest.raises(VaultError, match="audit_failed:OSError"):
        run(store, folder)
    assert (store.gallery_path.read_bytes() if store.exists() else None) == gallery_before  # undone
    assert [line["action"] for line in audit_lines(store)[len(lines_before):]] == ["enroll_started", "enroll_aborted"]
    assert enrolled(store) == audited(store) and len(list(folder.iterdir())) == 2
    assert {p.name for p in store.directory.iterdir()} <= {"gallery.sealed", "audit.jsonl", "inbox", ".gallery.lock"}


def test_a_change_that_cannot_be_undone_either_says_it_took_effect(store, monkeypatch) -> None:
    import sentinel.identity.gallery as gallery

    run(store, photos(store, "b:1", "b:2", folder="earlier"))
    real = gallery.replace_private
    calls = []

    def replace(source, path):
        calls.append(source)
        if len(calls) == 2:  # the undo
            raise VaultError("save_failed:OSError")
        return real(source, path)

    monkeypatch.setattr(gallery, "replace_private", replace)
    monkeypatch.setattr(gallery, "os", ShortWrites("ok", "half", "fail"))
    with pytest.raises(VaultError, match="committed_audit_incomplete"):
        run(store, photos(store, "a:1", "a:2"))
    lines = audit_lines(store)
    assert len(enrolled(store)) == 2 and lines[-1]["action"] == "enroll_started"  # in effect; started, not completed
    assert lines[-1]["identity_id"] in enrolled(store) - audited(store)


def test_a_torn_last_audit_line_stops_every_change_until_it_is_dealt_with(store) -> None:
    run(store, photos(store, "b:1", "b:2", folder="earlier"))
    (x,) = enrolled(store)
    with store.audit_path.open("ab") as log:
        log.write(b'{"utc":"2026-10-07T12:00:00+00:00","act')  # as a stop in the middle of an append leaves it
    before = files(store)
    folder = photos(store, "a:1", "a:2")
    for change in (lambda: run(store, folder), lambda: store.revoke(x, SECRET), store.purge):
        with pytest.raises(VaultError, match="audit_log_torn"):
            change()
    assert files(store) == before and len(list(folder.iterdir())) == 2


def test_a_commit_finds_the_gallery_as_it_was_read_or_changes_nothing(store) -> None:
    from sentinel.identity.gallery import GalleryChanged

    run(store, photos(store, "b:1", "b:2", folder="earlier"))
    document, base = store.read_for_update(SECRET)
    before = files(store)
    for stale in (None, "0" * 64):  # read as absent, or as another file
        with pytest.raises(GalleryChanged):
            store.commit(document, SECRET, "revoke", document.identities[0].identity_id, base=stale,
                         identities_left=0)
    assert files(store) == before


# ---------------------------------------------------- review A, finding 8: the provider's errors at enrollment


CvError = type("error", (Exception,), {})  # named as OpenCV's cv2.error is


class ProviderRuntime:
    """TensorFlow, OpenCV and DeepFace as the real adapter (LegacyDeepFaceBackend) sees them: it loads, then each
    enrollment photo fails as ``fault`` says, with the photo's path (a person's name) and synthetic secrets in the
    provider's message, or gives the malformed output it names."""

    def __init__(self, fault: str) -> None:
        from sentinel.identity.legacy_deepface import DEEPFACE_VERSION

        self.fault = fault
        self.ns = SimpleNamespace(
            tf=SimpleNamespace(config=SimpleNamespace(get_visible_devices=lambda kind: []), __version__="2.21.0"),
            cv2=SimpleNamespace(__version__="4.13.0", IMREAD_COLOR=1, imread=self._imread),
            numpy=SimpleNamespace(__version__="2.2.6", zeros=lambda shape, dtype: ("warmup", shape), uint8="uint8"),
            tf_keras=SimpleNamespace(__version__="2.21.0"),
            deepface=SimpleNamespace(__version__=DEEPFACE_VERSION),
            DeepFace=SimpleNamespace(build_model=lambda name: None, represent=self._represent))

    def _imread(self, path: str, flag: int):
        if self.fault == "decoder":
            raise CvError(f"imread: cannot decode {path}; token=SYNTH-decoder")
        return ("photo", path)

    def _represent(self, img_path, **kwargs):
        if img_path[0] == "warmup":
            return []
        if self.fault == "provider":
            raise RuntimeError(f"represent failed for {img_path[1]} with {{'api_key': 'SYNTH-provider'}}")
        if self.fault == "memory":
            raise MemoryError(f"cannot pad {img_path[1]} (SYNTH)")
        width = {"infinite-width": float("inf"), "nan-width": float("nan")}.get(self.fault, 10)
        embedding = {"overflowing-value": [1e200] + [1.0] * 511,  # finite; its norm is not (gave all zeros)
                     "overflowing-sum": [1e154] * 512}.get(self.fault, [0.0] * 511 + [1.0])  # OverflowError
        return [{"embedding": embedding, "facial_area": {"x": 0, "y": 0, "w": width, "h": 10},
                 "face_confidence": 0.9}]


def provider(fault: str):
    from sentinel.identity.legacy_deepface import DETECTOR_WEIGHTS, MODEL_WEIGHTS, LegacyDeepFaceBackend

    runtime = ProviderRuntime(fault)
    digests = {MODEL_WEIGHTS[0]: MODEL_WEIGHTS[1], DETECTOR_WEIGHTS[0]: DETECTOR_WEIGHTS[1]}
    return LegacyDeepFaceBackend(Path("/synthetic/weights"), importer=lambda: runtime.ns,
                                 sha256=lambda path: digests[path.name])


@pytest.mark.parametrize(("fault", "label"), [
    ("provider", "enrollment_photo_failed:RuntimeError"),
    ("decoder", "enrollment_photo_failed:error"),
    ("memory", "enrollment_photo_failed:MemoryError"),
    ("infinite-width", "inconsistent_output"),
    ("nan-width", "inconsistent_output"),
    ("overflowing-value", "inconsistent_output"),  # finding 9: was inconsistent_photos, from all-zero embeddings
    ("overflowing-sum", "inconsistent_output"),  # finding 9: was failed:OverflowError
])
def test_a_provider_failure_at_enrollment_is_one_label_with_its_class_never_its_message(
        store, tmp_path, monkeypatch, capsys, fault, label) -> None:
    folder = photos(store, "a:1", "a:2")
    assert call(store, tmp_path, monkeypatch, "enroll", str(folder), "--consent-confirmed", "--consent-date",
                "2026-10-07", backend=provider(fault)) == 1
    captured = capsys.readouterr()
    assert captured.err == f"identity: {label}\n" and captured.out == ""
    assert not store.exists() and audit_lines(store) == [] and len(list(folder.iterdir())) == 2


def test_any_other_failure_in_an_identity_command_is_one_label_too(store, tmp_path, monkeypatch, capsys) -> None:
    class Broken(Pipeline):  # not the provider: a defect anywhere under the command
        def enrollment_photo(self, path: Path):
            raise KeyError(f"{path} SYNTH")

    folder = photos(store, "a:1", "a:2")
    assert call(store, tmp_path, monkeypatch, "enroll", str(folder), "--consent-confirmed", "--consent-date",
                "2026-10-07", backend=Broken()) == 1
    captured = capsys.readouterr()
    assert captured.err == "identity: failed:KeyError\n" and captured.out == ""
    assert not store.exists() and len(list(folder.iterdir())) == 2
