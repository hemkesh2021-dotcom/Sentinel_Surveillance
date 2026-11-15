"""Synthetic fixtures for the personal-beta installation package's tests (tests/unit/test_beta_install*.py).

A directory tree that stands in for "/", stand-in commands that answer what the scripts ask about the platform, a
home directory, and the package assembled from the working tree as the archive builder assembles it. Nothing here
reads a real device fact, opens a camera, loads a model or uses the network.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packaging"))
import build_archive  # noqa: E402  (the archive builder, imported by path as the runner tools are)

DPKG_REFERENCE = {"cuda-toolkit-12-6": "12.6.11-1", "libnvinfer10": "10.3.0.30-1+cuda12.5", "libcudnn9-cuda-12": "9.3.0.75-1"}
LOADER_REFERENCE = ("libcudart.so.12", "libcublas.so.12", "libcufft.so.11", "libcurand.so.10", "libcusparse.so.12",
                    "libcufile.so.0", "libcupti.so.12", "libcudnn.so.9", "libcudss.so.0", "libopenblas.so.0")
PRESENCE_ONLY = ("git", "cmake", "make", "g++", "ss", "pgrep", "systemctl")  # the preflight only asks whether they exist
REAL = ("bash", "sh", "env", "cat", "awk", "sed", "grep", "sha256sum", "tar", "head", "tail", "tr", "cut", "sort",
        "dirname", "basename", "mktemp", "cp", "rm", "mkdir", "chmod", "ln", "ls", "stat", "date", "id", "readlink",
        "python3")


class World:
    """One synthetic device: its facts, its commands, a home directory and an extracted package."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.root, self.bin, self.home = tmp / "root", tmp / "bin", tmp / "home"
        for folder in (self.root, self.bin, self.home):
            folder.mkdir()
        for tool in REAL:
            found = shutil.which(tool)
            assert found, f"this test needs {tool}"
            (self.bin / tool).symlink_to(found)
        for tool in PRESENCE_ONLY:
            self.command(tool, {"pgrep": "exit 1", "systemctl": "echo inactive"}.get(tool, "exit 0"))  # nothing runs
        self.command("uname", '[ "$1" = -m ] && cat "$FIXTURE/arch"')
        self.command("dpkg-query", 'value="$(awk -v p="${@: -1}" \'$1 == p {print $2}\' "$FIXTURE/dpkg")"; '
                                   '[ -n "$value" ] || exit 1; printf %s "$value"')
        self.command("ldconfig", 'sed "s/^/\\t/; s/$/ (libc6,AArch64) => \\/usr\\/lib\\/x/" "$FIXTURE/loader"')
        self.command("python3.10", 'case "$*" in *version_info*) echo 3.10 ;; esac; exit 0')
        self.fact("proc/device-tree/model", "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super\0")
        self.fact("proc/meminfo", "MemTotal:        7802736 kB\nMemFree:         6000000 kB\n")
        self.fact("etc/nv_tegra_release", "# R36 (release), REVISION: 4.7, GCID: 1, BOARD: generic, EABI: aarch64\n")
        self.fact("etc/os-release", 'ID=ubuntu\nVERSION_ID="24.04"\n')
        self.fact("usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1", "")
        self.fact("usr/lib/python3.10/dist-packages/tensorrt/__init__.py", "")
        self.fact("usr/bin/python3", "#!/bin/sh\nexit 0\n", mode=0o755)  # "import cryptography" succeeds
        (self.root / "usr/local/cuda-12.6").mkdir(parents=True)
        self.fixture("arch", "aarch64\n")
        self.dpkg(DPKG_REFERENCE)
        self.fixture("loader", "".join(f"{name}\n" for name in LOADER_REFERENCE))
        name, members = build_archive.package_members(0, worktree=True)
        self.package = tmp / "extracted" / name
        build_archive.write_tree(members, self.package)
        self.env = {"PATH": str(self.bin), "HOME": str(self.home), "FIXTURE": str(tmp), "LC_ALL": "C",
                    "SENTINEL_BETA_FACTS_ROOT": str(self.root)}

    def command(self, name: str, body: str) -> None:
        path = self.bin / name
        path.unlink(missing_ok=True)
        path.write_text(f"#!/bin/bash\n{body}\n")
        path.chmod(0o755)

    def fact(self, relative: str, text: str, mode: int = 0o644) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(mode)

    def fixture(self, name: str, text: str) -> None:
        (self.tmp / name).write_text(text)

    def dpkg(self, versions: dict[str, str]) -> None:
        self.fixture("dpkg", "".join(f"{package} {version}\n" for package, version in versions.items()))

    def run(self, script: str, *args: str, package: Path | None = None, stdin: str = "", **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str((package or self.package) / script), *args], env={**self.env, **env},
                              input=stdin, capture_output=True, text=True, timeout=120, cwd=self.tmp)

    def install(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return self.run("install.sh", "--prefix", str(self.prefix), "--data-dir", str(self.data), *args, **env)

    @property
    def prefix(self) -> Path:
        return self.tmp / "opt" / "sentinel beta"  # a space: no script may split a path

    @property
    def data(self) -> Path:
        return self.tmp / "var" / "my data"

    @property
    def installed(self) -> Path:
        return self.prefix / "package"


def tree(folder: Path) -> dict[str, Any]:
    """Every file, link and directory under ``folder``: mode and content. Two equal trees are the same on disk."""
    seen: dict[str, Any] = {}
    for path in sorted(folder.rglob("*")):
        key = str(path.relative_to(folder))
        if path.is_symlink():
            seen[key] = ("link", os.readlink(path))
        elif path.is_dir():
            seen[key] = ("dir", oct(path.stat().st_mode & 0o777))
        else:
            seen[key] = ("file", oct(path.stat().st_mode & 0o777), hashlib.sha256(path.read_bytes()).hexdigest())
    return seen


def said(result: subprocess.CompletedProcess) -> str:
    return result.stdout + result.stderr
