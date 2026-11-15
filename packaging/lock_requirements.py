#!/usr/bin/env python3
"""Writes the personal-beta package's hash-pinned requirement file from a list of exact pins. Standard library only.

usage: python3 -B packaging/lock_requirements.py

Reads packaging/pins-runtime.in (one name==version per line: the versions the accepted build ran with on the reference
device, see the header it writes) and, for each, asks PyPI's JSON API for that release's files and their SHA-256.
Every file of the release is listed, as pip-compile --generate-hashes does, so pip may pick the wheel that fits the
device and still refuses any file whose checksum is not listed. A pin is refused here unless the release has a file
that the installer's interpreter can use: a wheel whose tags fit CPython 3.10 on aarch64 Linux (glibc 2.35, JetPack
6's) and whose Requires-Python admits 3.10, or a source distribution. Nothing is downloaded but the JSON documents;
the request names only public package names and versions.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = HERE / "pins-runtime.in"
OUT = HERE / "beta-install" / "manifest" / "requirements-pypi.txt"
HEAD = """# Sentinel personal beta: the runtime's Python packages from PyPI, each pinned to one version and to the SHA-256 of
# that release's files (written by packaging/lock_requirements.py; do not edit by hand).
#
# These are the versions the accepted build ran with on the reference device (its runtime environment, read from the
# installed metadata), limited to what the runtime needs: Ultralytics 8.4.25, DeepFace 0.0.99, TensorFlow 2.21.0 (CPU
# only, for the face model), OpenCV 4.13.0.92, NumPy 2.2.6, pydantic 2.12.5, PyYAML 6.0.3 and everything they
# require. rich, pytz and wheel reached that environment through the system's packages; here they are pinned to the
# same versions. The enrollment records the OpenCV, TensorFlow, tf_keras and NumPy versions: change one and the
# enrollment must be made again.
#
# onnx, onnxslim, onnxruntime, colorama, coloredlogs and humanfriendly are here for build-engine.sh only:
# Ultralytics 8.4.25 exports the detector to ONNX and simplifies it before TensorRT builds the engine (the reference
# environment holds the same versions). onnxruntime 1.23.2 is installed AFTER the Jetson index's onnxruntime-gpu
# 1.23.0 (requirements-jetson.txt), as on the reference device: both use one folder, and this one's files are the
# ones Python imports (the Jetson build was compiled against NumPy 1 and does not import beside NumPy 2.2.6).
#
# NOT here, on purpose: torch, torchvision and onnxruntime-gpu (CUDA builds for JetPack 6, from the Jetson AI Lab
# index: requirements-jetson.txt) and the TensorRT Python binding (JetPack's own python3-libnvinfer, linked in by
# setup-env.sh). Installed with: pip install --no-deps --require-hashes -r requirements-pypi.txt
--index-url https://pypi.org/simple
"""


TARGET_PYTHON = (3, 10)  # the installer's interpreter: CPython 3.10
TARGET_GLIBC = (2, 35)  # JetPack 6's Ubuntu 22.04; the reference device's 24.04 has 2.39


def admits_python(requires: str | None) -> bool:
    """Whether a Requires-Python value admits TARGET_PYTHON (the comparison operators PyPI metadata uses)."""
    for clause in (requires or "").split(","):
        found = re.fullmatch(r"\s*(>=|<=|==|!=|~=|>|<)\s*([0-9]+(?:\.[0-9]+)*)(\.\*)?\s*", clause)
        if not clause.strip():
            continue
        if not found:
            return False  # not understood: never taken for compatible
        operator, version, wildcard = found.groups()
        wanted = tuple(int(part) for part in version.split("."))
        ours = (TARGET_PYTHON + (0,))[:max(len(wanted), 2)] if not wildcard else TARGET_PYTHON[:len(wanted)]
        wanted = (wanted + (0, 0))[:len(ours)] if not wildcard else wanted
        if not {">=": ours >= wanted, "<=": ours <= wanted, ">": ours > wanted, "<": ours < wanted,
                "==": ours == wanted, "!=": ours != wanted, "~=": ours >= wanted and ours[:1] == wanted[:1]}[operator]:
            return False
    return True


def fits_target(filename: str) -> bool:
    """Whether a wheel's tags fit CPython 3.10 on aarch64 Linux with glibc 2.35."""
    parts = filename[:-len(".whl")].split("-")
    if not filename.endswith(".whl") or len(parts) < 5:
        return False
    pythons, abis, platforms = (part.split(".") for part in parts[-3:])
    minor = TARGET_PYTHON[1]

    def python_fits(tag: str, abi: str) -> bool:
        if tag in ("py3", f"py3{minor}", f"cp3{minor}"):
            return abi in ("none", "abi3", f"cp3{minor}")
        found = re.fullmatch(r"cp3(\d+)", tag)  # an older stable-ABI wheel also runs on a newer interpreter
        return bool(found) and abi == "abi3" and int(found.group(1)) <= minor

    def platform_fits(tag: str) -> bool:
        if tag == "any" or tag in ("linux_aarch64", "manylinux2014_aarch64"):
            return True
        found = re.fullmatch(r"manylinux_(\d+)_(\d+)_aarch64", tag)
        return bool(found) and (int(found.group(1)), int(found.group(2))) <= TARGET_GLIBC

    return any(python_fits(tag, abi) for tag in pythons for abi in abis) and any(platform_fits(tag) for tag in platforms)


def usable(files: list[dict]) -> list[str]:
    """The names of the release's files that the installer's interpreter can use."""
    return [item["filename"] for item in files if not item.get("yanked") and admits_python(item.get("requires_python"))
            and (item.get("packagetype") == "sdist" or fits_target(item["filename"]))]


def main() -> int:
    lines = [HEAD]
    for pin in PINS.read_text("utf-8").split():
        name, _, version = pin.partition("==")
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as response:  # noqa: S310
            files = json.load(response)["urls"]
        digests = sorted({item["digests"]["sha256"] for item in files if not item.get("yanked")})
        if not digests:
            raise SystemExit(f"STOP: PyPI lists no file for {pin}")
        if not usable(files):
            raise SystemExit(f"STOP: {pin} has no file that CPython 3.10 on aarch64 Linux can use")
        lines.append(f"{pin} \\\n" + " \\\n".join(f"    --hash=sha256:{digest}" for digest in digests) + "\n")
        print(f"{pin}: {len(digests)} file(s)", flush=True)
    OUT.write_text("".join(lines), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
