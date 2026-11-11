#!/usr/bin/env python3
"""Builds the private personal-beta installation archive from this branch. Standard library only.

usage: python3 -B packaging/build_archive.py --out DIR [--revision N]

What goes in, and nothing else:
  - the package's own files, as committed: packaging/beta-install/ (scripts, helpers, manifests, templates, guide);
  - the application exactly as the accepted commit ACCEPTED has it: src/sentinel (tracked files only) and the four
    files the memory guard runs from (benchmarks/runner), under app/;
  - VERSION (what this package is) and SHA256SUMS (every other file in it).
It refuses to build when the application files in this tree differ from ACCEPTED, or when anything that would be
packaged has uncommitted changes. It reads no gallery, enrollment, observation, key, credential or model file, and
none can be packaged: only files that Git tracks at HEAD are read (through `git show`), never the working directory.

Deterministic: members sorted, one owner, the packaged commit's own time, gzip without a time stamp, so the same
commit and revision always give the same archive and the same SHA-256.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

ACCEPTED = "949da67e8a172e8295528fd756aea30e703d1948"  # the accepted implementation (session 116, part 35)
PACKAGE_SOURCE = "packaging/beta-install"
GUARD_FILES = ("benchmarks/runner/step5_guard.py", "benchmarks/runner/operator_check.py",
               "benchmarks/runner/demo_profile.py", "benchmarks/runner/step4_criteria.py")
APP_TREE = "src/sentinel"
EXECUTABLE = (".sh",)
REPO = Path(__file__).resolve().parents[1]


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True).stdout


def tracked(path: str) -> list[str]:
    return sorted(line for line in git("ls-tree", "-r", "--name-only", "HEAD", "--", path).decode().splitlines() if line)


def read(path: str, worktree: bool) -> bytes:
    return (REPO / path).read_bytes() if worktree else git("show", f"HEAD:{path}")


def package_members(revision: int, *, worktree: bool = False) -> tuple[str, dict[str, bytes]]:
    """(the package's name, its files by relative path). ``worktree``: for the package's own tests only: the files
    as they are in the working directory (still only paths that Git tracks or that the tests name), marked as such."""
    head = git("rev-parse", "HEAD").decode().strip()
    app_files = tracked(APP_TREE) + list(GUARD_FILES)
    package_files = tracked(PACKAGE_SOURCE)
    if worktree:
        package_files = sorted(str(path.relative_to(REPO)) for path in (REPO / PACKAGE_SOURCE).rglob("*")
                               if path.is_file() and "__pycache__" not in path.parts)
    else:
        if git("diff", "--name-only", ACCEPTED, "HEAD", "--", APP_TREE, *GUARD_FILES).strip():
            raise SystemExit(f"STOP: the application files at HEAD differ from the accepted commit {ACCEPTED[:7]}; "
                             "this package ships the accepted code only")
        if git("status", "--porcelain", "--", APP_TREE, *GUARD_FILES, PACKAGE_SOURCE).strip():
            raise SystemExit("STOP: uncommitted changes under the packaged paths; commit them first")
    name = f"sentinel-personal-beta-{ACCEPTED[:7]}-r{revision}" + ("-worktree" if worktree else "")
    members: dict[str, bytes] = {}
    for path in package_files:
        members[path[len(PACKAGE_SOURCE) + 1:]] = read(path, worktree)
    for path in app_files:
        members[f"app/{path}"] = read(path, worktree)
    version = (f"{name}\n"
               f"accepted_code_commit={ACCEPTED}\n"
               f"package_commit={head}{' (working tree, not a release)' if worktree else ''}\n"
               f"package_revision={revision}\n"
               "status=private personal beta; validates and admits nothing; not for redistribution\n"
               "launch=CANNOT LAUNCH ON ANOTHER JETSON YET: the accepted build admits only the reference device's own "
               "detector engine, llama.cpp build and model-file copies (QUICKSTART.md, section 1)\n")
    members["VERSION"] = version.encode()
    sums = "".join(f"{hashlib.sha256(members[path]).hexdigest()}  {path}\n" for path in sorted(members))
    members["SHA256SUMS"] = sums.encode()
    return name, members


def write_tree(members: dict[str, bytes], folder: Path) -> None:
    """The package extracted into ``folder`` (as tar would leave it)."""
    for path, content in members.items():
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target.chmod(0o755 if path.endswith(EXECUTABLE) else 0o644)


def build(out: Path, revision: int) -> Path:
    name, members = package_members(revision)
    stamp = int(git("show", "-s", "--format=%ct", "HEAD").decode().strip())
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(members):
            info = tarfile.TarInfo(f"{name}/{path}")
            info.size = len(members[path])
            info.mtime = stamp
            info.mode = 0o755 if path.endswith(EXECUTABLE) else 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(members[path]))
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    target = out / f"{name}.tar.gz"
    with target.open("wb") as handle, gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as packed:
        packed.write(raw.getvalue())
    target.chmod(0o600)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (out / f"{name}.tar.gz.sha256").write_text(f"{digest}  {target.name}\n")
    print(f"archive={target}")
    print(f"sha256={digest}")
    print(f"files={len(members)} bytes={target.stat().st_size} package_commit={git('rev-parse', 'HEAD').decode().strip()}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="a private directory outside the repository")
    parser.add_argument("--revision", type=int, default=1)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if REPO == out or REPO in out.parents:
        raise SystemExit("STOP: --out must be outside the repository: the archive is private and is never committed")
    build(out, args.revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
