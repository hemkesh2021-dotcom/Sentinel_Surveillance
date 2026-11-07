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
import re
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


# What each writer may leave in a reserved directory: its own names and its own temporaries, nothing else. A claim
# lists only these (never opening them); any other name is reported and left.
WRITERS = {
    "raw": ({"raw-capture.sealed"}, re.compile(r"\.raw-capture\.sealed\.\d+\.tmp")),  # RawCapture.finish
    "gallery": ({"gallery.sealed", ".gallery.sealed.staged", ".gallery.sealed.previous", ".gallery.lock",
                 "audit.jsonl", "inbox"},  # D91's writer (IdentityStore.commit), and the vault's temporaries
                re.compile(r"\.\.?gallery\.sealed(\.staged)?\.\d+\.tmp")),
    "working_copy": ({"gallery.sealed", "audit.jsonl"}, re.compile(r"(?!)")),  # make_working_copy (O_EXCL, no temp)
}


@dataclass
class Reservation:
    path: Path
    cls: str
    writer: str
    creator: str
    utc: str
    closed: str | None = None  # "claimed" or "absent"


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
            elif fields[0] in ("reserve", "closed"):
                continue  # reservations(): read apart
            else:
                raise ManifestError(f"manifest line {number} is not an event this tool writes")
        return out

    def reservations(self) -> dict[Path, Reservation]:
        out: dict[Path, Reservation] = {}
        if not self.path.exists():
            return out
        for number, line in enumerate(self.path.read_text("utf-8").splitlines(), 1):
            fields = line.split("\t")
            if fields[0] == "reserve" and len(fields) == 6 and Path(fields[1]) not in out:
                out[Path(fields[1])] = Reservation(Path(fields[1]), fields[2], fields[3], fields[4], fields[5])
            elif fields[0] == "closed" and len(fields) == 4 and Path(fields[1]) in out:
                out[Path(fields[1])].closed = fields[3]
            elif fields[0] in ("reserve", "closed"):
                raise ManifestError(f"manifest line {number} is not a reservation event this tool writes")
        return out

    def reserve(self, path: Path, cls: str, writer: str, creator: str) -> Reservation:
        """Before a writer makes ``path`` (a directory): absolute and canonical, strictly inside the root, a
        deletable class, a known writer, not there yet, and never listed or reserved before (each attempt has its
        own name). Whatever that writer then leaves there can be claimed and cleaned up, even if it is killed."""
        path = Path(path)
        if not path.is_absolute() or path.resolve() != path:
            raise ManifestError("a reserved path must be absolute and canonical")
        if cls not in DELETABLE or writer not in WRITERS:
            raise ManifestError("only a deletable class and a known writer are reserved")
        if self.root not in path.parents:
            raise ManifestError("a reserved path lies inside the experiment root")
        if os.path.lexists(path):
            raise ManifestError("a reserved path does not exist yet")
        if path in self.entries() or path in self.reservations():
            raise ManifestError("a path is reserved once: a new attempt takes a new name")
        self._append("reserve", str(path), cls, writer, creator, self._utc_now())
        return self.reservations()[path]

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


def claim(manifest: Manifest, path: Path, *, writer_exited: bool) -> dict:
    """List what the writer of an open reservation left there, once that writer has exited (never while it may still
    write): the directory and, below it, only regular files and directories named as that writer names them (never
    a link, never opened). Anything else is reported and left. Closes the reservation as claimed."""
    path = Path(path)
    reservation = manifest.reservations().get(path)
    if reservation is None or reservation.closed is not None:
        raise ManifestError("no open reservation for this path")
    if not writer_exited:
        raise ManifestError("a reservation is claimed only once its writer has exited")
    info = os.lstat(path)  # FileNotFoundError: close_absent decides
    if not stat.S_ISDIR(info.st_mode):
        raise ManifestError("a reserved path is a directory, never a link or a file")
    names, temporary = WRITERS[reservation.writer]
    listed = [str(manifest.add(path, reservation.cls, reservation.creator).path)]
    unexpected = []
    for current, dirs, files in os.walk(path, followlinks=False):
        for name in sorted(dirs) + sorted(files):
            child = Path(current) / name
            relative = child.relative_to(path)
            child_info = os.lstat(child)
            known = (str(relative) in names or temporary.fullmatch(str(relative)) is not None) and len(relative.parts) == 1
            if known and (stat.S_ISREG(child_info.st_mode) or stat.S_ISDIR(child_info.st_mode)):
                listed.append(str(manifest.add(child, reservation.cls, reservation.creator).path))
            else:
                unexpected.append(str(child))
        dirs[:] = [d for d in dirs if d in names and len(Path(current, d).relative_to(path).parts) == 1]
    manifest._append("closed", str(path), manifest._utc_now(), "claimed")
    return {"listed": listed, "unexpected": unexpected}


def close_absent(manifest: Manifest, path: Path, *, writer_exited: bool, recorded_written: bool) -> None:
    """Close a reservation whose writer made nothing: only once the writer has exited, only when the path is absent,
    and never when the writer's own record says it wrote there (a file that should exist and does not: STOP)."""
    path = Path(path)
    reservation = manifest.reservations().get(path)
    if reservation is None or reservation.closed is not None:
        raise ManifestError("no open reservation for this path")
    if not writer_exited:
        raise ManifestError("a reservation is closed only once its writer has exited")
    if os.path.lexists(path):
        raise ManifestError("the reserved path exists: claim it")
    if recorded_written:
        raise ManifestError("the writer recorded a write that is not there")
    manifest._append("closed", str(path), manifest._utc_now(), "absent")


def status(manifest: Manifest, sessions: Path | None = None, prefix: str = "mp-",
           frozen: Path | None = None) -> dict:
    """What an abandoned or unfinished session left open, for the session-start block to STOP on: reservations not
    yet claimed or closed, raw captures and working copies listed and not deleted, a candidate listed and not
    deleted while no frozen record names it, and session folders of this package not sealed."""
    reservations = manifest.reservations()
    entries = manifest.entries()
    frozen_ok = frozen is not None and frozen.is_file()
    out = {"open_reservations": sorted(str(p) for p, r in reservations.items() if r.closed is None),
           "raw_left": sorted(str(p) for p, e in entries.items() if e.cls == "raw" and not e.deleted),
           "working_copies_left": sorted(str(p) for p, e in entries.items()
                                         if e.cls == "working_copy" and not e.deleted),
           "unfrozen_candidate_left": [] if frozen_ok else sorted(
               str(p) for p, e in entries.items() if e.cls == "candidate" and not e.deleted),
           "unsealed_sessions": []}
    if sessions is not None and sessions.is_dir():
        out["unsealed_sessions"] = sorted(str(d) for d in sessions.iterdir()
                                          if d.is_dir() and d.name.startswith(prefix) and not (d / SEAL_NAME).exists())
    return out


def recover(manifest: Manifest, *, writer_exited: bool, protected: tuple[Path, ...],
            recorded_written: dict[Path, bool] | None = None, frozen: Path | None = None) -> dict:
    """The recovery block's work (an aborted capture, a failed or interrupted build, an abandoned session): claim
    every open reservation (or close it as absent), then clean up what is listed: raw captures and working copies, and
    a candidate only while no frozen record exists (a frozen candidate waits for the retention decision). Never
    outside the root, never a protected path, never a sealed folder."""
    if not writer_exited:
        raise ManifestError("recovery runs only once the writers have exited")
    report: dict = {"claimed": [], "closed_absent": [], "unexpected": []}
    for path, reservation in sorted(manifest.reservations().items()):
        if reservation.closed is not None:
            continue
        if os.path.lexists(path):
            result = claim(manifest, path, writer_exited=True)
            report["claimed"].append(str(path))
            report["unexpected"] += result["unexpected"]
        else:
            close_absent(manifest, path, writer_exited=True,
                         recorded_written=bool((recorded_written or {}).get(path, False)))
            report["closed_absent"].append(str(path))
    classes = ("raw", "working_copy") if frozen is not None and frozen.is_file() else ("raw", "working_copy", "candidate")
    report["cleanup"] = cleanup(manifest, classes, protected=protected)
    return report


def _process_argvs() -> list[list[bytes]]:
    out = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                out.append((entry / "cmdline").read_bytes().split(b"\0"))
            except OSError:
                continue
    return out


def runtime_exited(result: dict | None, processes: list[list[bytes]] | None = None) -> bool:
    """The writer of a session's raw capture (the runtime) has exited: the guard recorded the child's exit, and no
    runtime is left on this machine (its own process group may outlive a killed guard): neither the ``sentinel``
    script's ``run`` nor ``python -m sentinel.cli run``, the package's own launch."""
    if not result or (result.get("child") or {}).get("returncode") is None:
        return False
    for argv in (_process_argvs() if processes is None else processes):
        if b"run" in argv and any(a.endswith(b"sentinel") or a == b"sentinel.cli" for a in argv):
            return False
    return True


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
    manifest.reserve(copy_dir, "working_copy", "working_copy", creator)  # a kill from here on leaves it recoverable
    copy_dir.mkdir(mode=0o700)
    copy_dir.chmod(0o700)
    for name in names:
        data = (original_dir / name).read_bytes()
        fd = os.open(copy_dir / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
    result = claim(manifest, copy_dir, writer_exited=True)  # this process was the writer, and it has written
    made = [manifest.entries()[Path(p)] for p in result["listed"]]
    after = {n: sha256_file(original_dir / n) for n in names}
    if after != before or any(sha256_file(copy_dir / n) != before[n] for n in names) or result["unexpected"]:
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


def _recorded_written(run_dir: Path | None) -> bool:
    """Whether the run's raw capture record says it was written (the runtime's own stop record)."""
    if run_dir is None or not (run_dir / "run.jsonl").is_file():
        return False
    import json

    for line in (run_dir / "run.jsonl").read_text("utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("identity") == "capture_raw" and record.get("written") is True:
            return True
    return False


def _result(run_dir: Path | None) -> dict | None:
    import json

    try:
        return json.loads((run_dir / "result.json").read_text("utf-8")) if run_dir is not None else None
    except (OSError, ValueError):
        return None


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
    res = sub.add_parser("reserve", help="reserve a directory before its writer makes it")
    res.add_argument("root", type=Path)
    res.add_argument("path", type=Path)
    res.add_argument("--class", dest="cls", choices=DELETABLE, required=True)
    res.add_argument("--writer", choices=tuple(WRITERS), required=True)
    for name in ("claim", "close-absent"):
        c = sub.add_parser(name, help=f"{name} a reservation once its writer (the run) has exited")
        c.add_argument("root", type=Path)
        c.add_argument("path", type=Path)
        c.add_argument("--run", type=Path, required=True, help="the run's directory (its result.json and run.jsonl)")
    rec = sub.add_parser("recover", help="claim open reservations, then clean up raw captures, working copies and "
                                         "an unfrozen candidate")
    rec.add_argument("root", type=Path)
    rec.add_argument("--run", type=Path, action="append", default=[], help="each run whose writer must have exited")
    rec.add_argument("--protect", type=Path, action="append", default=[])
    rec.add_argument("--frozen", type=Path, default=None)
    st = sub.add_parser("status", help="what an abandoned or unfinished session left open")
    st.add_argument("root", type=Path)
    st.add_argument("--sessions", type=Path, default=None)
    st.add_argument("--frozen", type=Path, default=None)
    rep = sub.add_parser("report", help="read-only report of unexpected copies, for review")
    rep.add_argument("root", type=Path)
    rep.add_argument("--search", type=Path, action="append", default=[])
    rep.add_argument("--crash-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    manifest = Manifest(args.root.absolute())
    try:
        if args.command == "working-copy":
            made = make_working_copy(manifest, args.original.absolute(), args.to.absolute(),
                                     "mpe_manifest.py working-copy")
            print(f"working_copy_listed={len(made)}")
            return 0
        if args.command == "reserve":
            manifest.reserve(args.path.absolute(), args.cls, args.writer, "mpe_manifest.py reserve")
            print("reserved")
            return 0
        if args.command in ("claim", "close-absent"):
            exited = runtime_exited(_result(args.run))
            if args.command == "claim":
                result = claim(manifest, args.path.absolute(), writer_exited=exited)
                print(f"claimed listed={len(result['listed'])} unexpected={len(result['unexpected'])}")
                for value in result["unexpected"]:
                    print(f"  unexpected (left, for review): {value}")
            else:
                close_absent(manifest, args.path.absolute(), writer_exited=exited,
                             recorded_written=_recorded_written(args.run))
                print("closed_absent")
            return 0
        if args.command == "status":
            result = status(manifest, args.sessions, frozen=args.frozen)
        elif args.command == "recover":
            exited = all(runtime_exited(_result(run)) for run in args.run) if args.run else \
                runtime_exited({"child": {"returncode": 0}})
            any_written = any(_recorded_written(run) for run in args.run)  # then an absent raw capture: STOP
            written = {path: any_written for path, r in manifest.reservations().items() if r.writer == "raw"}
            result = recover(manifest, writer_exited=exited, protected=tuple(p.absolute() for p in args.protect),
                             recorded_written=written, frozen=args.frozen)
            result = {**{k: v for k, v in result.items() if k != "cleanup"}, **result["cleanup"]}
        elif args.command == "cleanup":
            result = cleanup(manifest, tuple(args.classes), args.within.absolute() if args.within else None,
                             tuple(p.absolute() for p in args.protect))
        else:
            result = copy_report(manifest, args.search, args.crash_dir)
    except (ManifestError, FileNotFoundError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    for key, values in result.items():
        print(f"{key}={len(values)}")
        for value in values:
            print(f"  {value}")
    if args.command == "status":
        return 1 if any(result.values()) else 0
    left = ("left_changed", "left_not_empty", "left_outside_root", "left_protected", "left_sealed")
    return 0 if args.command == "report" or not any(result.get(k) for k in left) else 1


if __name__ == "__main__":
    sys.exit(main())
