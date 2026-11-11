#!/usr/bin/env python3
"""Writes the personal-beta package's hash-pinned requirement file from a list of exact pins. Standard library only.

usage: python3 -B packaging/lock_requirements.py

Reads packaging/pins-runtime.in (one name==version per line: the versions the accepted build ran with on the reference
device, see the header it writes) and, for each, asks PyPI's JSON API for that release's files and their SHA-256.
Every file of the release is listed, as pip-compile --generate-hashes does, so pip may pick the wheel that fits the
device and still refuses any file whose checksum is not listed. Nothing is downloaded but the JSON documents; the
request names only public package names and versions.
"""

from __future__ import annotations

import json
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
# NOT here, on purpose: torch and torchvision (CUDA builds for JetPack 6, from the Jetson AI Lab index:
# requirements-jetson.txt) and the TensorRT Python binding (JetPack's own python3-libnvinfer, linked in by
# setup-env.sh). Installed with: pip install --no-deps --require-hashes -r requirements-pypi.txt
--index-url https://pypi.org/simple
"""


def main() -> int:
    lines = [HEAD]
    for pin in PINS.read_text("utf-8").split():
        name, _, version = pin.partition("==")
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as response:  # noqa: S310
            files = json.load(response)["urls"]
        digests = sorted({item["digests"]["sha256"] for item in files if not item.get("yanked")})
        if not digests:
            raise SystemExit(f"STOP: PyPI lists no file for {pin}")
        usable = [item["filename"] for item in files if item["packagetype"] == "sdist"
                  or any(tag in item["filename"] for tag in ("none-any", "aarch64"))]
        if not usable:
            raise SystemExit(f"STOP: {pin} has no source distribution and no file for aarch64")
        lines.append(f"{pin} \\\n" + " \\\n".join(f"    --hash=sha256:{digest}" for digest in digests) + "\n")
        print(f"{pin}: {len(digests)} file(s)", flush=True)
    OUT.write_text("".join(lines), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
