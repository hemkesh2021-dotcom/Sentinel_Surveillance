"""Sealers that fail on purpose after the gallery is sealed: the real helper with one call made to fail.

Used by the enrollment and capture tests (review A, finding 2) to show that nothing is left behind when the read-back
or the audit fails. Synthetic galleries only.
"""

from __future__ import annotations

import json
import subprocess
import threading

from sentinel.identity.vault import Sealer

KDF_N = 2**14  # the tests' cheap scrypt cost


class Tampering(Sealer):
    """On its ``nth`` open, returns the gallery without its last identity: a read-back that differs."""

    def __init__(self, nth: int) -> None:
        super().__init__(kdf_n=KDF_N)
        self._nth = nth
        self._opens = 0

    def open(self, sealed, secret):
        associated, plaintext = super().open(sealed, secret)
        self._opens += 1
        if self._opens == self._nth:
            document = json.loads(plaintext)
            document["identities"] = document["identities"][:-1]
            plaintext = json.dumps(document).encode("ascii")
        return associated, plaintext


class Pausing(Sealer):
    """The real helper; its first open (an enrollment reading the gallery) returns only after ``go`` is set, and
    ``read`` is set once it has read. Lets a test change the gallery while that enrollment is under way."""

    def __init__(self) -> None:
        super().__init__(kdf_n=KDF_N)
        self.read = threading.Event()
        self.go = threading.Event()
        self._first = True

    def open(self, sealed, secret):
        opened = super().open(sealed, secret)
        if self._first:
            self._first = False
            self.read.set()
            assert self.go.wait(60)
        return opened


def timing_out(nth: int) -> Sealer:
    """The real helper, except that its ``nth`` call (seal or open) times out."""
    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == nth:
            raise subprocess.TimeoutExpired(command, 120.0)
        return subprocess.run(command, **kwargs)

    return Sealer(kdf_n=KDF_N, run=run)
