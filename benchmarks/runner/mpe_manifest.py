"""The multi-pose comparison's manifest: every file and directory the experiment's tools make, listed when made, and
the only things its cleanup ever deletes.

``MANIFEST.tsv`` in the experiment root is append-only, one event per line (tab-separated):

- ``add  PATH  KIND  CLASS  CREATOR  UTC  DEV:INODE  SHA256`` when a tool has made PATH (a file, its SHA-256 taken
  before the line is written; a dir, ``-``);
- ``deleted  PATH  UTC`` when the cleanup has removed it.

Classes: ``raw`` (the raw capture), ``candidate`` (the frozen working gallery and its build record),
``working_copy`` (a session's copy of the original gallery), ``evidence`` (sealed sessions) and ``output`` (counts-only
readings). The cleanup deletes only ``raw``, ``candidate`` and ``working_copy`` entries, and only a listed path that is
still the same file or directory it listed (the same device and inode, never a link, and a file still its listed
SHA-256): anything else is reported and left. A cleanup may be limited to one listed path and what lies under it. A listed directory is removed only when nothing unlisted remains in it; an unlisted file there is reported, never
deleted. ``evidence`` and ``output`` are never deleted here: that is the maintainer's explicit step under the retention
decision. Whatever the manifest lists, the cleanup never deletes outside the experiment root (deletable entries are
listed only inside it), at, under or above a protected path (the original enrollment, the sessions), or in a sealed
directory (one holding SHA256SUMS). Nothing outside the manifest is ever deleted: unexpected copies (an unlisted file
with a listed file's SHA-256 under the roots given, a crash report made since the first entry) are only reported, for
review. ``make_working_copy`` makes a session's working copy of the original enrollment inside the root and lists it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "MANIFEST.tsv"
CLASSES = ("raw", "candidate", "working_copy", "evidence", "output")
DELETABLE = ("raw", "candidate", "working_copy")
KINDS = ("file", "dir")


class ManifestError(Exception):
    pass


@dataclass
class Entry:
    path: Path
    kind: str
    cls: str
    creator: str
    utc: str
    identity: tuple[int, int]
    sha256: str | None = None
    deleted: bool = False


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Manifest:
    """The experiment root's manifest (created 0700/0600 on first use)."""

    def __init__(self, root: Path, *, utc_now=_utc) -> None:
        self.root = Path(root)
        self.path = self.root / MANIFEST_NAME
        self._utc_now = utc_now

    def _append(self, *fields: str) -> None:
        if any("\t" in f or "\n" in f for f in fields):
            raise ManifestError("a manifest field holds a tab or a newline")
        if not self.root.exists():
            self.root.mkdir(mode=0o700, parents=True)
            self.root.chmod(0o700)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, ("\t".join(fields) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def entries(self) -> dict[Path, Entry]:
        out: dict[Path, Entry] = {}
        if not self.path.exists():
            return out
        for number, line in enumerate(self.path.read_text("utf-8").splitlines(), 1):
            fields = line.split("\t")
            if fields[0] == "add" and len(fields) == 8 and Path(fields[1]) not in out:
                dev, ino = (int(x) for x in fields[6].split(":"))
                out[Path(fields[1])] = Entry(Path(fields[1]), fields[2], fields[3], fields[4], fields[5], (dev, ino),
                                             None if fields[7] == "-" else fields[7])
            elif fields[0] == "deleted" and len(fields) == 3 and Path(fields[1]) in out:
                out[Path(fields[1])].deleted = True
            else:
                raise ManifestError(f"manifest line {number} is not an event this tool writes")
        return out

    def add(self, path: Path, cls: str, creator: str) -> Entry:
        """List ``path``, which a tool of the experiment has just made: absolute, existing, never a link (nor under
        one), never the manifest or the root itself, not listed already (deleted entries included). A file's SHA-256 is
        taken first and written on the same line."""
        path = Path(path)
        if not path.is_absolute():
            raise ManifestError("a listed path must be absolute")
        if cls not in CLASSES:
            raise ManifestError(f"unknown class {cls!r}")
        if path.resolve() != path:
            raise ManifestError("a listed path must not pass through a link or hold '..'")
        if path in (self.path, self.root):
            raise ManifestError("the manifest and the experiment root are never listed")
        if cls in DELETABLE and self.root not in path.parents:
            raise ManifestError("a deletable entry lies inside the experiment root")
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise ManifestError("a link is never listed")
        kind = "dir" if stat.S_ISDIR(info.st_mode) else "file" if stat.S_ISREG(info.st_mode) else None
        if kind is None:
            raise ManifestError("only regular files and directories are listed")
        if path in self.entries():
            raise ManifestError("already listed")
        digest = sha256_file(path) if kind == "file" else "-"
        self._append("add", str(path), kind, cls, creator, self._utc_now(), f"{info.st_dev}:{info.st_ino}", digest)
        return self.entries()[path]

    def add_tree(self, directory: Path, cls: str, creator: str) -> list[Entry]:
        """List a directory a tool made itself and everything now in it (hidden files too), the directory first. Never
        for a directory the tool did not make: a raw capture is listed path by path after its own checks."""
        made = [self.add(directory, cls, creator)]
        for current, dirs, files in os.walk(directory, followlinks=False):
            for name in sorted(dirs) + sorted(files):
                made.append(self.add(Path(current) / name, cls, creator))
        return made


def _same(entry: Entry, info: os.stat_result) -> bool:
    return (info.st_dev, info.st_ino) == entry.identity and not stat.S_ISLNK(info.st_mode) and (
        stat.S_ISDIR(info.st_mode) if entry.kind == "dir" else stat.S_ISREG(info.st_mode))


SEAL_NAME = "SHA256SUMS"  # a sealed session's manifest (the packages' sealer): such a directory is evidence


def _sealed(path: Path, root: Path) -> bool:
    """True when ``path`` holds a seal or lies under a directory that does (any of its parents, inside the root or
    above it: conservative, a seal anywhere above makes nothing there deletable)."""
    for directory in (path, *path.parents):
        if (directory / SEAL_NAME).exists():
            return True
    return False


def _refusal(entry: Entry, root: Path, protected: tuple[Path, ...]) -> str | None:
    """Why the cleanup must leave a listed entry whatever the manifest says (a manifest is a file; it could be wrong):
    outside the experiment root, at or under a protected path (the original enrollment, a session), or under a seal."""
    if not (root in entry.path.parents):
        return "left_outside_root"
    if any(entry.path == p or p in entry.path.parents or entry.path in p.parents for p in protected):
        return "left_protected"
    if _sealed(entry.path, root):
        return "left_sealed"
    return None


def cleanup(manifest: Manifest, classes: tuple[str, ...], within: Path | None = None,
            protected: tuple[Path, ...] = ()) -> dict:
    """Delete the listed entries of ``classes`` (only raw, candidate, working_copy) that are still what was listed,
    limited to ``within`` and what lies under it when given; files first, then directories deepest first and only when
    empty. Never, whatever is listed: anything outside the experiment root, at, under or holding a ``protected`` path
    (the original enrollment's directory, the sessions), or in a sealed directory (one holding SHA256SUMS). Report
    what was left and why."""
    if not classes or any(cls not in DELETABLE for cls in classes):
        raise ManifestError(f"the cleanup deletes only {', '.join(DELETABLE)} entries")
    root = manifest.root
    protected = tuple(Path(p) for p in protected)
    report = {"deleted": [], "already_absent": [], "left_changed": [], "left_not_empty": [], "unlisted_inside": [],
              "left_outside_root": [], "left_protected": [], "left_sealed": []}
    entries = {}
    for path, entry in manifest.entries().items():
        if entry.cls not in classes or entry.deleted or not (within is None or path == Path(within)
                                                             or Path(within) in path.parents):
            continue
        why = _refusal(entry, root, protected)
        if why is not None:
            report[why].append(str(path))
            continue
        entries[path] = entry
    for entry in sorted((e for e in entries.values() if e.kind == "file"), key=lambda e: str(e.path)):
        try:
            info = os.lstat(entry.path)
        except FileNotFoundError:
            report["already_absent"].append(str(entry.path))
            manifest._append("deleted", str(entry.path), manifest._utc_now())
            continue
        if not _same(entry, info) or entry.sha256 is None or sha256_file(entry.path) != entry.sha256:
            report["left_changed"].append(str(entry.path))
            continue
        os.unlink(entry.path)
        manifest._append("deleted", str(entry.path), manifest._utc_now())
        report["deleted"].append(str(entry.path))
    listed = set(manifest.entries())
    for entry in sorted((e for e in entries.values() if e.kind == "dir"), key=lambda e: -len(e.path.parts)):
        try:
            info = os.lstat(entry.path)
        except FileNotFoundError:
            report["already_absent"].append(str(entry.path))
            manifest._append("deleted", str(entry.path), manifest._utc_now())
            continue
        if not _same(entry, info):
            report["left_changed"].append(str(entry.path))
            continue
        remaining = sorted(entry.path.iterdir())
        if remaining:
            report["left_not_empty"].append(str(entry.path))
            report["unlisted_inside"] += [str(p) for p in remaining if p not in listed]
            continue
        os.rmdir(entry.path)
        manifest._append("deleted", str(entry.path), manifest._utc_now())
        report["deleted"].append(str(entry.path))
    return report


WORKING_COPY_FILES = ("gallery.sealed", "audit.jsonl")  # what a session's working copy of the original holds


def make_working_copy(manifest: Manifest, original_dir: Path, copy_dir: Path, creator: str) -> list[Entry]:
    """A session's working copy of the original enrollment: its sealed gallery (and its audit log, when there is one),
    byte for byte, in a new private directory inside the experiment root, listed as working_copy. The original is only
    read, and its files must be unchanged after the copy. ManifestError when a path is relative or not canonical, the
    original lies inside the root or holds it, the copy is not strictly inside the root, overlaps the original or
    exists already, the original has no sealed gallery, or the copy is not identical."""
    original_dir, copy_dir, root = Path(original_dir), Path(copy_dir), manifest.root
    for path in (original_dir, copy_dir, root):
        if not path.is_absolute() or path.resolve() != path:
            raise ManifestError("a working copy's paths must be absolute and canonical")
    if original_dir == root or root in original_dir.parents or original_dir in root.parents:
        raise ManifestError("the original enrollment never lies inside the experiment root, nor holds it")
    if root not in copy_dir.parents:
        raise ManifestError("a working copy lies inside the experiment root")
    if copy_dir == original_dir or original_dir in copy_dir.parents or copy_dir in original_dir.parents:
        raise ManifestError("a working copy never overlaps the original")
    if os.path.lexists(copy_dir):
        raise ManifestError("a working copy's directory is new")
    names = [n for n in WORKING_COPY_FILES if (original_dir / n).is_file()]
    if "gallery.sealed" not in names:
        raise ManifestError("the original has no sealed gallery")
    before = {n: sha256_file(original_dir / n) for n in names}
    copy_dir.mkdir(mode=0o700)
    copy_dir.chmod(0o700)
    made = [manifest.add(copy_dir, "working_copy", creator)]
    for name in names:
        data = (original_dir / name).read_bytes()
        fd = os.open(copy_dir / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        made.append(manifest.add(copy_dir / name, "working_copy", creator))
    after = {n: sha256_file(original_dir / n) for n in names}
    if after != before or any(sha256_file(copy_dir / n) != before[n] for n in names):
        raise ManifestError("the working copy is not identical to the original")
    return made


def copy_report(manifest: Manifest, roots: list[Path], crash_dir: Path | None = None) -> dict:
    """Read only, for review; nothing is deleted. Unlisted files under ``roots`` whose SHA-256 equals a listed file's,
    deleted ones included (links not followed), unlisted files inside listed directories, and
    crash reports in ``crash_dir`` modified since the manifest's first entry (names and times only, never opened)."""
    entries = manifest.entries()
    listed = set(entries)
    known = {e.sha256 for e in entries.values() if e.sha256 is not None}  # deleted files' digests too
    copies: list[str] = []
    for root in roots:
        for current, dirs, files in os.walk(root, followlinks=False):
            for name in files:
                path = Path(current) / name
                if path in listed or path == manifest.path:
                    continue
                try:
                    info = os.lstat(path)
                except OSError:
                    continue
                if not stat.S_ISREG(info.st_mode) or not known:
                    continue
                try:
                    if sha256_file(path) in known:
                        copies.append(str(path))
                except OSError:
                    continue
    inside = []
    for entry in entries.values():
        if entry.kind == "dir" and not entry.deleted and entry.path.is_dir():
            inside += [str(p) for p in sorted(entry.path.iterdir()) if p not in listed]
    crashes = []
    if crash_dir is not None and crash_dir.is_dir() and entries:
        first = min(datetime.fromisoformat(e.utc) for e in entries.values()).timestamp()
        crashes = sorted(p.name for p in crash_dir.iterdir() if p.is_file() and p.stat().st_mtime >= first)
    return {"copies_with_a_listed_digest": sorted(copies), "unlisted_inside_listed_dirs": sorted(inside),
            "crash_reports_since_first_entry": crashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    clean = sub.add_parser("cleanup", help="delete listed raw, candidate or working_copy entries")
    clean.add_argument("root", type=Path)
    clean.add_argument("--class", dest="classes", action="append", choices=DELETABLE, required=True)
    clean.add_argument("--within", type=Path, default=None)
    clean.add_argument("--protect", type=Path, action="append", default=[],
                       help="never delete at, under or above this path (the original enrollment, a session)")
    copy = sub.add_parser("working-copy", help="copy the original enrollment for one session, listed")
    copy.add_argument("root", type=Path)
    copy.add_argument("--original", type=Path, required=True)
    copy.add_argument("--to", type=Path, required=True)
    rep = sub.add_parser("report", help="read-only report of unexpected copies, for review")
    rep.add_argument("root", type=Path)
    rep.add_argument("--search", type=Path, action="append", default=[])
    rep.add_argument("--crash-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    manifest = Manifest(args.root.absolute())
    if args.command == "working-copy":
        made = make_working_copy(manifest, args.original.absolute(), args.to.absolute(), "mpe_manifest.py working-copy")
        print(f"working_copy_listed={len(made)}")
        return 0
    if args.command == "cleanup":
        result = cleanup(manifest, tuple(args.classes), args.within.absolute() if args.within else None,
                         tuple(p.absolute() for p in args.protect))
    else:
        result = copy_report(manifest, args.search, args.crash_dir)
    for key, values in result.items():
        print(f"{key}={len(values)}")
        for value in values:
            print(f"  {value}")
    left = ("left_changed", "left_not_empty", "left_outside_root", "left_protected", "left_sealed")
    return 0 if args.command == "report" or not any(result[k] for k in left) else 1


if __name__ == "__main__":
    sys.exit(main())
