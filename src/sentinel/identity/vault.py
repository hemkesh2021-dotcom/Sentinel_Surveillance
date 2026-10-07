"""The encrypted identity gallery's file boundary (V2-20/V2-25 demo form; option A, maintainer decision 2026-10-07).

Encryption runs in a separate process: the system Python (``/usr/bin/python3``) with its apt-maintained
``python3-cryptography``, executing ``sealer.py`` in isolated mode. The runtime's environment gets no new dependency.
See ``sealer.py`` for the format: scrypt (n 2**17, r 8, p 1, a stored random salt) and AES-256-GCM over the plaintext,
with the exact header line authenticated.

- **The secret** is a passphrase typed at startup and never saved: ``SENTINEL_IDENTITY_PASSPHRASE``, read once and
  removed from this process's environment before anything is spawned, or a hidden prompt. It reaches the helper on
  its stdin only (never its arguments or environment) and is wrapped in ``Secret``, whose repr hides it.
- **Writes are atomic and private:** a new 0600 file in the same 0700 directory, written, fsynced and renamed over
  the old one, then the directory fsynced. Any failure removes the temporary file and leaves the previous file as it
  was.
- **Errors are fixed labels** (``VaultError.label``); the helper's stderr and exception text are never kept.

Limits (disclosed, guide ch. 11): while the runtime runs, the gallery is decrypted in its memory; a compromised account
on the running device can read the process's memory and its environment at startup. Deleting a file on an SSD is not
secure erasure.
"""

from __future__ import annotations

import base64
import contextlib
import getpass
import json
import os
import stat
import subprocess
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

SECRET_ENV = "SENTINEL_IDENTITY_PASSPHRASE"
SYSTEM_PYTHON = Path("/usr/bin/python3")
SEALER_SCRIPT = Path(__file__).with_name("sealer.py")
DEFAULT_KDF_N = 2**17  # scrypt cost: 128 MiB with r 8, in the helper process only
SEALER_TIMEOUT_S = 120.0
MIN_NEW_SECRET_CHARS = 12  # a starting rule for a new gallery's passphrase, not a strength measurement
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


class VaultError(Exception):
    """The gallery cannot be sealed, opened or saved; ``label`` is safe to show."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


class Secret:
    """A passphrase that never prints."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise VaultError("secret_missing")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(<hidden>)"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("a Secret cannot be pickled")


def take_secret_from_environment(environ: MutableMapping[str, str] | None = None) -> Secret | None:
    """Remove SENTINEL_IDENTITY_PASSPHRASE from the environment and return it (None if unset or empty)."""
    environ = os.environ if environ is None else environ
    value = environ.pop(SECRET_ENV, None)
    return Secret(value) if value else None


def prompt_secret(*, confirm: bool, prompt: Callable[[str], str] = getpass.getpass) -> Secret:
    """A hidden prompt; with ``confirm`` (a new gallery) it asks twice and applies MIN_NEW_SECRET_CHARS."""
    value = prompt("identity gallery passphrase: ")
    if not value:
        raise VaultError("secret_missing")
    if confirm:
        if len(value) < MIN_NEW_SECRET_CHARS:
            raise VaultError("secret_too_short")
        if prompt("repeat the passphrase: ") != value:
            raise VaultError("secret_mismatch")
    return Secret(value)


class Sealer:
    """Calls ``sealer.py`` under the system Python; ``run`` is injectable only to simulate a failing helper."""

    def __init__(
        self,
        *,
        python: Path = SYSTEM_PYTHON,
        script: Path = SEALER_SCRIPT,
        kdf_n: int = DEFAULT_KDF_N,
        timeout_s: float = SEALER_TIMEOUT_S,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self._python = Path(python)
        self._script = Path(script)
        self._kdf_n = kdf_n
        self._timeout_s = timeout_s
        self._run = run

    def check(self) -> str:
        """The helper's cryptography version, or VaultError (e.g. ``crypto_unavailable``)."""
        return str(self._call({"op": "check"})["cryptography"])

    def seal(self, plaintext: bytes, secret: Secret, associated: Mapping[str, Any]) -> bytes:
        response = self._call({"op": "seal", "secret": secret.reveal(), "n": self._kdf_n, "associated": dict(associated),
                               "plaintext": base64.b64encode(plaintext).decode("ascii")})
        return self._decode(response.get("sealed"))

    def open(self, sealed: bytes, secret: Secret) -> tuple[dict[str, Any], bytes]:
        response = self._call({"op": "open", "secret": secret.reveal(),
                               "sealed": base64.b64encode(sealed).decode("ascii")})
        associated = response.get("associated")
        if not isinstance(associated, dict):
            raise VaultError("sealer_bad_response")
        return associated, self._decode(response.get("plaintext"))

    def _call(self, request: Mapping[str, Any]) -> dict[str, Any]:
        try:
            done = self._run(
                [str(self._python), "-I", str(self._script)],
                input=json.dumps(request).encode("utf-8"),
                capture_output=True, timeout=self._timeout_s, check=False, close_fds=True,
                env={"LC_ALL": "C.UTF-8"},  # nothing of this process's environment, secrets included
            )
        except subprocess.TimeoutExpired:
            raise VaultError("sealer_timeout") from None
        except OSError as exc:
            raise VaultError(f"sealer_unavailable:{type(exc).__name__}") from None
        try:
            response = json.loads(done.stdout.decode("utf-8").strip().splitlines()[-1])
        except (UnicodeDecodeError, ValueError, IndexError):
            raise VaultError(f"sealer_failed:exit_{done.returncode}") from None
        if not isinstance(response, dict):
            raise VaultError("sealer_bad_response")
        if response.get("ok") is not True:
            error = response.get("error")
            raise VaultError(error if isinstance(error, str) and error.isprintable() and len(error) <= 80
                             else "sealer_bad_response")
        return response

    @staticmethod
    def _decode(text: Any) -> bytes:
        try:
            return base64.b64decode(str(text).encode("ascii"), validate=True)
        except ValueError:
            raise VaultError("sealer_bad_response") from None


def private_directory_problem(directory: Path) -> str | None:
    """None if ``directory`` is a real directory owned by this user with no group or other access."""
    try:
        info = os.lstat(directory)
    except FileNotFoundError:
        return "identity_dir_missing"
    except OSError as exc:
        return f"identity_dir_unreadable:{type(exc).__name__}"
    if not stat.S_ISDIR(info.st_mode):
        return "identity_dir_not_a_directory"
    if info.st_uid != os.getuid():
        return "identity_dir_not_owned"
    if info.st_mode & 0o077:
        return "identity_dir_not_private"
    return None


def private_file_problem(path: Path) -> str | None:
    """None if ``path`` is a regular file (not a link) owned by this user with no group or other access."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return "file_missing"
    except OSError as exc:
        return f"file_unreadable:{type(exc).__name__}"
    if not stat.S_ISREG(info.st_mode):
        return "file_not_regular"
    if info.st_uid != os.getuid():
        return "file_not_owned"
    if info.st_mode & 0o077:
        return "file_not_private"
    return None


def write_private_atomic(
    path: Path,
    data: bytes,
    *,
    fsync: Callable[[int], None] = os.fsync,
    replace: Callable[[Path, Path], None] = os.replace,
) -> None:
    """Replace ``path`` with ``data`` atomically, mode 0600; on failure the old file stays and no temporary is left."""
    directory = path.parent
    problem = private_directory_problem(directory)
    if problem is not None:
        raise VaultError(problem)
    temporary = directory / f".{path.name}.{os.getpid()}.tmp"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, PRIVATE_FILE_MODE)
    except OSError as exc:
        raise VaultError(f"save_failed:{type(exc).__name__}") from None
    try:
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            fsync(fd)
        finally:
            os.close(fd)
        replace(temporary, path)
    except BaseException as exc:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        if isinstance(exc, Exception):
            raise VaultError(f"save_failed:{type(exc).__name__}") from None
        raise
    try:
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return  # the rename is done; durability of the directory entry is best effort here
    try:
        with contextlib.suppress(OSError):
            fsync(dir_fd)
    finally:
        os.close(dir_fd)


def read_private(path: Path) -> bytes:
    problem = private_file_problem(path)
    if problem is not None:
        raise VaultError(problem)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise VaultError(f"read_failed:{type(exc).__name__}") from None

