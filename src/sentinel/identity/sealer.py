"""Seal and open the identity gallery (V2-20/V2-25 demo form; encryption option A, maintainer decision 2026-10-07).

This file is a standalone helper, not a module of the runtime. ``sentinel.identity.vault`` runs it as
``/usr/bin/python3 -I sealer.py`` so that it uses the system Python's apt-maintained ``python3-cryptography`` without
installing anything into the runtime's environment. It imports nothing from Sentinel.

One JSON request on stdin, one JSON response on stdout:

- ``{"op": "check"}`` -> ``{"ok": true, "cryptography": "<version>"}``
- ``{"op": "seal", "secret": str, "plaintext": base64, "associated": object, "n": int}`` -> ``{"ok": true, "sealed":
  base64}``
- ``{"op": "open", "secret": str, "sealed": base64}`` -> ``{"ok": true, "associated": object, "plaintext": base64}``
- any failure -> ``{"ok": false, "error": "<fixed label>"}`` and exit status 1.

The secret is a passphrase typed at startup and never stored. Key derivation is scrypt (RFC 7914) with a fresh 16-byte
salt per seal, ``r`` 8, ``p`` 1 and ``n`` a power of two in [2**14, 2**20] (the vault seals with 2**17: 128 MiB, about
a second); encryption is AES-256-GCM with a fresh 12-byte nonce. A sealed file is one canonical JSON header line (the
format, the KDF parameters and salt, the nonce and the caller's non-secret ``associated`` record), a newline, then the
base64 ciphertext and tag, and a newline. The exact header line is GCM's associated data, so changing any header byte,
the ciphertext or the tag fails authentication, as a wrong secret does; the two are indistinguishable by design.

Nothing here prints, logs or returns the secret, the derived key or the plaintext except the plaintext of a
successful open. Errors are fixed labels; exception text is never written.
"""

import base64
import binascii
import json
import os
import sys
import unicodedata

FORMAT = "sentinel-identity-sealed"
VERSION = 1
KDF_R = 8
KDF_P = 1
KDF_MIN_LOG2_N = 14
KDF_MAX_LOG2_N = 20
SALT_BYTES = 16
NONCE_BYTES = 12
KEY_BYTES = 32
MAX_SECRET_CHARS = 1024
MAX_PLAINTEXT_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = 4 * MAX_PLAINTEXT_BYTES


class Refused(Exception):
    def __init__(self, label):
        super().__init__(label)
        self.label = label


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _b64decode(text):
    if not isinstance(text, str):
        raise Refused("bad_request")
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError):
        raise Refused("bad_request") from None


def _secret_bytes(secret):
    if not isinstance(secret, str) or not secret or len(secret) > MAX_SECRET_CHARS:
        raise Refused("bad_secret")
    return unicodedata.normalize("NFC", secret).encode("utf-8")


def _check_n(n):
    if type(n) is not int or n < 2 or n & (n - 1) or not KDF_MIN_LOG2_N <= n.bit_length() - 1 <= KDF_MAX_LOG2_N:
        raise Refused("kdf_parameters_out_of_range")


def _derive(secret, salt, n):
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    return Scrypt(salt=salt, length=KEY_BYTES, n=n, r=KDF_R, p=KDF_P).derive(_secret_bytes(secret))


def _seal(request):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    n = request.get("n")
    _check_n(n)
    associated = request.get("associated")
    if not isinstance(associated, dict):
        raise Refused("bad_request")
    plaintext = _b64decode(request.get("plaintext"))
    if len(plaintext) > MAX_PLAINTEXT_BYTES:
        raise Refused("plaintext_too_large")
    salt, nonce = os.urandom(SALT_BYTES), os.urandom(NONCE_BYTES)
    header = _canonical({
        "format": FORMAT, "version": VERSION,
        "kdf": {"name": "scrypt", "n": n, "r": KDF_R, "p": KDF_P, "salt": base64.b64encode(salt).decode("ascii")},
        "aead": {"name": "AES-256-GCM", "nonce": base64.b64encode(nonce).decode("ascii")},
        "associated": associated,
    })
    sealed = AESGCM(_derive(request.get("secret"), salt, n)).encrypt(nonce, plaintext, header)
    data = header + b"\n" + base64.b64encode(sealed) + b"\n"
    return {"ok": True, "sealed": base64.b64encode(data).decode("ascii")}


def _parse(data):
    """The header line, its fields and the ciphertext; anything not exactly in the sealed form is malformed."""
    line, newline, rest = data.partition(b"\n")
    if not newline or not rest.endswith(b"\n") or rest.count(b"\n") != 1:
        raise Refused("malformed_file")
    try:
        header = json.loads(line.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        raise Refused("malformed_file") from None
    if not isinstance(header, dict) or set(header) != {"format", "version", "kdf", "aead", "associated"}:
        raise Refused("malformed_file")
    if header["format"] != FORMAT or header["version"] != VERSION:
        raise Refused("unsupported_format")
    kdf, aead = header["kdf"], header["aead"]
    if (not isinstance(kdf, dict) or set(kdf) != {"name", "n", "r", "p", "salt"} or kdf["name"] != "scrypt"
            or not isinstance(aead, dict) or set(aead) != {"name", "nonce"} or aead["name"] != "AES-256-GCM"
            or not isinstance(header["associated"], dict)):
        raise Refused("malformed_file")
    try:
        if _canonical(header) != line:  # one encoding per header: no equivalent rewrites
            raise Refused("malformed_file")
    except (TypeError, ValueError):
        raise Refused("malformed_file") from None
    if kdf["r"] != KDF_R or kdf["p"] != KDF_P:
        raise Refused("kdf_parameters_out_of_range")
    _check_n(kdf["n"])
    try:
        salt = base64.b64decode(kdf["salt"].encode("ascii"), validate=True)
        nonce = base64.b64decode(aead["nonce"].encode("ascii"), validate=True)
        sealed = base64.b64decode(rest[:-1], validate=True)
    except (AttributeError, binascii.Error, UnicodeEncodeError, ValueError):
        raise Refused("malformed_file") from None
    if len(salt) != SALT_BYTES or len(nonce) != NONCE_BYTES or len(sealed) < 16:
        raise Refused("malformed_file")
    return line, header, salt, nonce, sealed


def _open(request):
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    line, header, salt, nonce, sealed = _parse(_b64decode(request.get("sealed")))
    key = _derive(request.get("secret"), salt, header["kdf"]["n"])
    try:
        plaintext = AESGCM(key).decrypt(nonce, sealed, line)
    except InvalidTag:
        raise Refused("authentication_failed") from None
    return {"ok": True, "associated": header["associated"], "plaintext": base64.b64encode(plaintext).decode("ascii")}


def handle(raw):
    try:
        if len(raw) > MAX_REQUEST_BYTES:
            raise Refused("bad_request")
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise Refused("bad_request") from None
        if not isinstance(request, dict):
            raise Refused("bad_request")
        op = request.get("op")
        try:
            import cryptography
        except ImportError:
            raise Refused("crypto_unavailable") from None
        if op == "check":
            return {"ok": True, "cryptography": cryptography.__version__}
        if op == "seal":
            return _seal(request)
        if op == "open":
            return _open(request)
        raise Refused("bad_request")
    except Refused as refused:
        return {"ok": False, "error": refused.label}
    except Exception as exc:  # noqa: BLE001 - a fixed label with the class name; never the exception text
        return {"ok": False, "error": f"internal_error:{type(exc).__name__}"}


def main():
    response = handle(sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1))
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
