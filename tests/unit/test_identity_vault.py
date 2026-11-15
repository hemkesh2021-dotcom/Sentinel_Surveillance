"""The identity gallery's encryption boundary (option A): the real helper under the system Python, synthetic data only.

These tests run ``sealer.py`` with ``/usr/bin/python3`` and its ``python3-cryptography``, the way the runtime does. They
are skipped, with the reason, where that interpreter or library is missing; fakes are used only to make a save fail.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import stat
import subprocess
from pathlib import Path

import pytest

from sentinel.identity import vault
from sentinel.identity.vault import (
    SECRET_ENV,
    Sealer,
    Secret,
    VaultError,
    prompt_secret,
    read_private,
    take_secret_from_environment,
    write_private_atomic,
)

FAST_N = 2**14  # the helper's minimum; the production cost (2**17) is exercised once below
SECRET = Secret("correct horse battery staple 7")
ASSOCIATED = {"schema": "test", "dimension": 512, "model": "synthetic"}
PLAINTEXT = json.dumps({"identities": [{"identity_id": "idn-0001", "prototypes": [[0.6, 0.8], [1.0, 0.0]]}]}).encode()


def _real_sealer_or_skip(**kwargs) -> Sealer:
    sealer = Sealer(kdf_n=FAST_N, **kwargs)
    try:
        sealer.check()
    except VaultError as exc:
        pytest.skip(f"the system Python's cryptography is not usable here ({exc.label})")
    return sealer


@pytest.fixture
def sealer() -> Sealer:
    return _real_sealer_or_skip()


@pytest.fixture
def private_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "identity"
    directory.mkdir()
    directory.chmod(0o700)
    return directory


def _split(sealed: bytes) -> tuple[dict, bytes, bytes]:
    line, _, body = sealed.partition(b"\n")
    return json.loads(line), line, body


def _rejoin(header: dict, body: bytes) -> bytes:
    line = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return line + b"\n" + body


# ---------------------------------------------------------------- round trip and wrong secret


def test_a_sealed_gallery_opens_with_its_secret_and_hides_the_plaintext(sealer) -> None:
    sealed = sealer.seal(PLAINTEXT, SECRET, ASSOCIATED)
    associated, plaintext = sealer.open(sealed, SECRET)
    assert (associated, plaintext) == (ASSOCIATED, PLAINTEXT)
    assert b"idn-0001" not in sealed and b"prototypes" not in sealed
    assert SECRET.reveal().encode() not in sealed
    header, _, _ = _split(sealed)
    assert header["kdf"]["n"] == FAST_N and (header["kdf"]["r"], header["kdf"]["p"]) == (8, 1)
    assert header["aead"]["name"] == "AES-256-GCM" and header["associated"] == ASSOCIATED
    assert sealed != sealer.seal(PLAINTEXT, SECRET, ASSOCIATED)  # a fresh salt and nonce every time


def test_the_production_kdf_parameters_round_trip() -> None:
    sealer = Sealer()  # n 2**17, r 8, p 1: 128 MiB in the helper
    try:
        sealer.check()
    except VaultError as exc:
        pytest.skip(f"the system Python's cryptography is not usable here ({exc.label})")
    sealed = sealer.seal(PLAINTEXT, SECRET, ASSOCIATED)
    header, _, _ = _split(sealed)
    assert (header["kdf"]["name"], header["kdf"]["n"], header["kdf"]["r"], header["kdf"]["p"]) == ("scrypt", 2**17, 8, 1)
    assert len(base64.b64decode(header["kdf"]["salt"])) == 16 and len(base64.b64decode(header["aead"]["nonce"])) == 12
    assert sealer.open(sealed, SECRET) == (ASSOCIATED, PLAINTEXT)


def test_a_wrong_secret_fails_authentication(sealer) -> None:
    sealed = sealer.seal(PLAINTEXT, SECRET, ASSOCIATED)
    with pytest.raises(VaultError) as raised:
        sealer.open(sealed, Secret("correct horse battery staple 8"))
    assert raised.value.label == "authentication_failed"


# ---------------------------------------------------------------- tampering


def test_any_changed_ciphertext_byte_fails_authentication(sealer) -> None:
    sealed = sealer.seal(PLAINTEXT, SECRET, ASSOCIATED)
    header, line, body = _split(sealed)
    raw = bytearray(base64.b64decode(body.strip()))
    for index in (0, len(raw) // 2, len(raw) - 1):  # first ciphertext byte, the middle, the tag's last byte
        tampered = bytearray(raw)
        tampered[index] ^= 0x01
        with pytest.raises(VaultError) as raised:
            sealer.open(line + b"\n" + base64.b64encode(bytes(tampered)) + b"\n", SECRET)
        assert raised.value.label == "authentication_failed"


@pytest.mark.parametrize("change", [
    lambda h: h["associated"].update(dimension=513),  # the authenticated, non-secret record
    lambda h: h["associated"].pop("model"),
    lambda h: h["kdf"].update(n=2**15),  # a valid but different cost: another key
    lambda h: h["kdf"].update(salt=base64.b64encode(b"\x00" * 16).decode()),
    lambda h: h["aead"].update(nonce=base64.b64encode(b"\x00" * 12).decode()),
])
def test_a_changed_header_fails_authentication(sealer, change) -> None:
    header, _, body = _split(sealer.seal(PLAINTEXT, SECRET, ASSOCIATED))
    change(header)
    with pytest.raises(VaultError) as raised:
        sealer.open(_rejoin(header, body), SECRET)
    assert raised.value.label == "authentication_failed"


@pytest.mark.parametrize(("change", "label"), [
    (lambda h: h["kdf"].update(n=2**30), "kdf_parameters_out_of_range"),  # refused before any allocation
    (lambda h: h["kdf"].update(n=3 * 2**14), "kdf_parameters_out_of_range"),
    (lambda h: h["kdf"].update(r=1), "kdf_parameters_out_of_range"),
    (lambda h: h.update(version=2), "unsupported_format"),
    (lambda h: h.update(extra=1), "malformed_file"),
    (lambda h: h["aead"].update(name="AES-128-CBC"), "malformed_file"),
])
def test_an_invalid_header_is_refused_with_its_label(sealer, change, label) -> None:
    header, _, body = _split(sealer.seal(PLAINTEXT, SECRET, ASSOCIATED))
    change(header)
    with pytest.raises(VaultError) as raised:
        sealer.open(_rejoin(header, body), SECRET)
    assert raised.value.label == label


def test_a_non_canonical_or_truncated_file_is_malformed(sealer) -> None:
    sealed = sealer.seal(PLAINTEXT, SECRET, ASSOCIATED)
    _, line, body = _split(sealed)
    for broken in (line.replace(b":", b": ", 1) + b"\n" + body, sealed[:-5], line + b"\n", b"", b"\n\n"):
        with pytest.raises(VaultError) as raised:
            sealer.open(broken, SECRET)
        assert raised.value.label == "malformed_file"


# ---------------------------------------------------------------- the secret stays out of the process boundary


def test_the_secret_goes_to_the_helper_on_stdin_only(sealer, monkeypatch) -> None:
    seen = []

    def spy(args, **kwargs):
        seen.append((args, kwargs))
        return subprocess.run(args, **kwargs)

    monkeypatch.setenv(SECRET_ENV, "an environment value that must not reach the helper")
    spying = Sealer(kdf_n=FAST_N, run=spy)
    spying.open(spying.seal(PLAINTEXT, SECRET, ASSOCIATED), SECRET)
    for args, kwargs in seen:
        assert SECRET.reveal() not in " ".join(args)
        assert kwargs["env"] == {"LC_ALL": "C.UTF-8"}
        assert SECRET.reveal().encode() in kwargs["input"]
        assert args[1] == "-I"


def test_errors_are_fixed_labels_without_the_secret(sealer) -> None:
    with pytest.raises(VaultError) as raised:
        sealer.open(b"not a sealed file", Secret("the secret text itself"))
    assert "secret text" not in str(raised.value) and raised.value.label == "malformed_file"


def test_a_secret_never_prints_or_pickles() -> None:
    assert "battery" not in repr(SECRET) and "battery" not in str(SECRET) and "battery" not in f"{SECRET}"
    with pytest.raises(TypeError):
        pickle.dumps(SECRET)
    with pytest.raises(VaultError):
        Secret("")


def test_the_secret_is_taken_from_the_environment_once() -> None:
    environ = {SECRET_ENV: "typed at startup", "OTHER": "x"}
    secret = take_secret_from_environment(environ)
    assert secret is not None and secret.reveal() == "typed at startup" and SECRET_ENV not in environ
    assert take_secret_from_environment(environ) is None


def test_a_new_passphrase_is_confirmed_and_needs_a_minimum_length() -> None:
    answers = iter(["long enough passphrase", "long enough passphrase"])
    assert prompt_secret(confirm=True, prompt=lambda _: next(answers)).reveal() == "long enough passphrase"
    with pytest.raises(VaultError, match="secret_too_short"):
        prompt_secret(confirm=True, prompt=lambda _: "short")
    answers = iter(["long enough passphrase", "another long passphrase"])
    with pytest.raises(VaultError, match="secret_mismatch"):
        prompt_secret(confirm=True, prompt=lambda _: next(answers))


def test_a_misbehaving_helper_gives_a_fixed_label() -> None:
    def fake(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout=json.dumps({"ok": False, "error": "x" * 500}).encode(),
                                           stderr=b"Traceback ... secret")
    with pytest.raises(VaultError, match="sealer_bad_response"):
        Sealer(run=fake).check()
    with pytest.raises(VaultError, match="sealer_unavailable:FileNotFoundError"):
        Sealer(python=Path("/nonexistent/python3")).check()


# ---------------------------------------------------------------- failure during saving


def _no_temporaries(directory: Path) -> bool:
    return not [p for p in directory.iterdir() if p.name.endswith(".tmp")]


def _save(sealer: Sealer, path: Path, plaintext: bytes, **kwargs) -> None:
    write_private_atomic(path, sealer.seal(plaintext, SECRET, ASSOCIATED), **kwargs)


def test_a_saved_file_is_private_and_opens(sealer, private_dir) -> None:
    path = private_dir / "gallery.sealed"
    _save(sealer, path, PLAINTEXT)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert sealer.open(read_private(path), SECRET) == (ASSOCIATED, PLAINTEXT)
    assert _no_temporaries(private_dir)


def test_a_helper_failure_while_saving_leaves_the_previous_file(sealer, private_dir) -> None:
    path = private_dir / "gallery.sealed"
    _save(sealer, path, PLAINTEXT)
    before = path.read_bytes()
    with pytest.raises(VaultError, match="sealer_unavailable"):
        _save(Sealer(python=Path("/nonexistent/python3")), path, b"version 2")
    with pytest.raises(VaultError, match="sealer_timeout"):
        _save(Sealer(kdf_n=FAST_N, timeout_s=0.001), path, b"version 2")
    assert path.read_bytes() == before and _no_temporaries(private_dir)
    assert sealer.open(path.read_bytes(), SECRET)[1] == PLAINTEXT


@pytest.mark.parametrize("failing", ["fsync", "replace"])
def test_a_write_failure_while_saving_leaves_the_previous_file(sealer, private_dir, failing) -> None:
    path = private_dir / "gallery.sealed"
    _save(sealer, path, PLAINTEXT)
    before = path.read_bytes()

    def fail(*args):
        raise OSError(28, "No space left on device")

    with pytest.raises(VaultError, match="save_failed:OSError"):
        _save(sealer, path, b"version 2", **{failing: fail})
    assert path.read_bytes() == before and _no_temporaries(private_dir)
    assert sealer.open(path.read_bytes(), SECRET)[1] == PLAINTEXT


def test_an_interrupt_while_saving_removes_the_temporary_and_propagates(sealer, private_dir) -> None:
    path = private_dir / "gallery.sealed"
    _save(sealer, path, PLAINTEXT)
    before = path.read_bytes()

    def interrupt(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _save(sealer, path, b"version 2", replace=interrupt)
    assert path.read_bytes() == before and _no_temporaries(private_dir)


def test_a_partial_write_is_never_visible(sealer, private_dir, monkeypatch) -> None:
    path = private_dir / "gallery.sealed"
    _save(sealer, path, PLAINTEXT)
    before = path.read_bytes()
    real_write, calls = os.write, []

    def short_then_fail(fd, data):
        calls.append(1)
        if len(calls) > 1:
            raise OSError(5, "Input/output error")
        return real_write(fd, bytes(data[:10]))  # a short write first

    sealed = sealer.seal(PLAINTEXT * 50, SECRET, ASSOCIATED)  # before the patch: the helper's pipe uses os.write too
    monkeypatch.setattr(vault.os, "write", short_then_fail)
    with pytest.raises(VaultError, match="save_failed:OSError"):
        write_private_atomic(path, sealed)
    monkeypatch.undo()
    assert len(calls) == 2
    assert path.read_bytes() == before and _no_temporaries(private_dir)


def test_saving_needs_a_private_directory_and_reading_a_private_file(sealer, tmp_path, private_dir) -> None:
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    open_dir.chmod(0o755)
    with pytest.raises(VaultError, match="identity_dir_not_private"):
        _save(sealer, open_dir / "gallery.sealed", PLAINTEXT)
    path = private_dir / "gallery.sealed"
    _save(sealer, path, PLAINTEXT)
    path.chmod(0o640)
    with pytest.raises(VaultError, match="file_not_private"):
        read_private(path)
    path.chmod(0o600)
    link = private_dir / "link.sealed"
    link.symlink_to(path)
    with pytest.raises(VaultError, match="file_not_regular"):
        read_private(link)
