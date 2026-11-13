#!/usr/bin/env python3
"""Builds the private personal-beta installation archive from this branch. Standard library only.

usage: python3 -B packaging/build_archive.py --out DIR [--revision N]

What goes in, and nothing else:
  - the package's own files, as committed: packaging/beta-install/ (scripts, helpers, manifests, templates, guide);
  - the application, under app/: src/sentinel (tracked files only) and the four files the memory guard runs from
    (benchmarks/runner). Every file is exactly as the accepted commit ACCEPTED has it, except the files that
    CHANGED_SINCE_ACCEPTED names (changed, or new): those are packaged as this branch has them, and the package says
    so: its name carries a modification mark, and VERSION names each one with both SHA-256 values, the reason, and
    what has and has not been run with it. A package with a changed file is never named or described as the accepted
    build;
  - VERSION (what this package is) and SHA256SUMS (every other file in it). VERSION also names every one of the
    package's own files that differs from the previous revision's (PACKAGE_FILES_CHANGED), with both SHA-256 values
    and the reason.
It refuses to build when the application files that differ from ACCEPTED, or the package's own files that differ from
the previous revision, are not exactly the declared ones (one more, or a declared one that does not differ), or when
anything that would be packaged has uncommitted changes. It reads no gallery, enrollment, observation, key,
credential or model file, and none can be packaged: only files that Git tracks at HEAD are read (through
`git show`), never the working directory.

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
# Application files packaged in another form than ACCEPTED has, or that ACCEPTED does not have: path -> (what
# changed, what it has been run with). The builder refuses unless exactly these differ. An empty table means the
# package ships the accepted code only.
_ROUTE_RUN_WITH = ("synthetic tests on the reference device only (temporary directories, synthetic files, stand-ins); "
                   "never on another device, and no run with a camera or a model has used it")
CHANGED_SINCE_ACCEPTED = {
    "benchmarks/runner/step5_guard.py": (
        "process ownership in the memory guard (session 116, part 40): after a run it signals only processes that "
        "descend from it, by number and start time through a process handle; another llama-server of the same user "
        "is reported and never signalled; memory limits, readiness checks and deadlines are unchanged",
        "synthetic and stand-in-process tests on the reference device only; no guarded run with a camera or a model "
        "has used this guard"),
    "src/sentinel/device_artifacts.py": (
        "a new file (session 116, part 41), the device-artifacts route: reads a record of the tester's own detector "
        "engine, llama.cpp build and model files, strictly; refuses a record that is malformed, that claims more "
        "than 'pending' for resource validation, or whose built files were not made by the package's own steps with "
        "the pinned source, options and tools; it admits nothing and measures nothing", _ROUTE_RUN_WITH),
    "src/sentinel/demo_runtime.py": (
        "the device-artifacts route (session 116, part 41): only with a record, and only in a --face-validation run, "
        "the engine, the llama.cpp build, the model files and the face weights are compared with the device's own "
        "record instead of the reference device's records, by the same startup rule; every other admission rule, "
        "the configuration checks against the accepted profile and the memory policy are unchanged; the starting "
        "line then says that resource validation is pending. Without a record nothing differs", _ROUTE_RUN_WITH),
    "src/sentinel/cli.py": (
        "the device-artifacts route (session 116, part 41): the option --device-artifacts FILE of `sentinel run`, "
        "refused without --face --face-validation and for a record that cannot be read; the detector is then given "
        "the recorded engine's SHA-256 as its pin. Without the option the command decides as before, with one "
        "difference in how it is typed: the abbreviation --d, which used to select --data-dir, is now ambiguous",
        _ROUTE_RUN_WITH),
}
# The package's own files (under PACKAGE_SOURCE) that differ from the previous revision's, or that it did not have:
# path -> what changed. The builder refuses unless exactly these differ from PREVIOUS_PACKAGE_COMMIT.
PREVIOUS_PACKAGE_COMMIT = "929b6d8659a007ac885331e1e5e3fbb052cfc436"
PREVIOUS_PACKAGE_NAME = "sentinel-personal-beta-949da67-mod1d12a8b-r2"
PACKAGE_FILES_CHANGED = {
    "QUICKSTART.md": "the guide: the default admission and the opt-in device-artifacts route, its commands, what "
                     "the route does not validate, what was and was not verified",
    "build-engine.sh": "takes the export settings from the asset manifest and, after the export, notes what it "
                       "exported for the device-artifacts record; the settings themselves are unchanged",
    "build-llama.sh": "builds only in a tree it started itself (a marker), refuses compiler flags from the shell, "
                      "removes an earlier run's binaries so that its own build links what it notes, and notes what "
                      "it built for the device-artifacts record. A defect of revisions 1 and 2 is corrected: the "
                      "build command was given the binary's path instead of the target's name (the build had never "
                      "run past its configure step)",
    "check.sh": "passes --device-artifacts FILE on to start.sh --check",
    "install.sh": "its closing note names the route and the two new steps; the installation itself is unchanged",
    "lib/admission_check.py": "with --device-artifacts FILE, makes the runtime's decisions for a run on that record "
                              "and prints the default admission's own answers beside them; without it, unchanged",
    "lib/artifacts.py": "a new file: writes the device-artifacts record from what the package's steps observed, "
                        "and checks every file, the pins and the platform against a record before a launch",
    "lib/beta.py": "notes a download that fetch-assets.sh made itself; prints an asset's settings as JSON for the "
                   "build script",
    "manifest/assets.json": "names the llama.cpp build's libraries, its build target (as cmake_target) and the "
                            "export's tool versions; its notes name the route",
    "record-artifacts.sh": "a new file: the step that writes the record",
    "start.sh": "the opt-in --device-artifacts FILE: the record's check before anything else, the route's notice, "
                "the run's own copy of the record; gates, guard options, the 60-minute limit and START are unchanged",
}
DEFAULT_REVISION = 3
EXECUTABLE = (".sh",)
REPO = Path(__file__).resolve().parents[1]


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True).stdout


def tracked(path: str) -> list[str]:
    return sorted(line for line in git("ls-tree", "-r", "--name-only", "HEAD", "--", path).decode().splitlines() if line)


def read(path: str, worktree: bool) -> bytes:
    return (REPO / path).read_bytes() if worktree else git("show", f"HEAD:{path}")


def working_files(path: str) -> list[str]:
    """For the tests' working-tree form: the files under ``path`` that Git tracks or would track (not ignored)."""
    listed = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", path).decode().split("\0")
    return sorted(name for name in listed if name and (REPO / name).is_file())


def blobs(commit: str, *paths: str) -> dict[str, str]:
    """Git's own content id of every file under ``paths`` at ``commit``, by path."""
    found = {}
    for entry in git("ls-tree", "-r", "-z", commit, "--", *paths).decode().split("\0"):
        if entry:
            meta, _, name = entry.partition("\t")
            found[name] = meta.split()[2]
    return found


def blob_id(content: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()  # noqa: S324 - Git's content id, not security


def differing(base: dict[str, str], files: dict[str, bytes]) -> list[str]:
    """The paths whose packaged content is not ``base``'s: changed, new, or no longer there."""
    return sorted(name for name in set(base) | set(files)
                  if name not in files or base.get(name) != blob_id(files[name]))


def declared_exactly(differs: list[str], declared: dict, what: str) -> None:
    """Never mislabelled: the files that differ are exactly the ones the package declares, or nothing is built."""
    if differs != sorted(declared):
        undeclared = [name for name in differs if name not in declared]
        same = [name for name in declared if name not in differs]
        raise SystemExit(f"STOP: {what} are not exactly the ones this package declares as changed"
                         + (f"; changed but not declared: {' '.join(undeclared)}" if undeclared else "")
                         + (f"; declared but identical: {' '.join(same)}" if same else ""))


def package_members(revision: int, *, worktree: bool = False) -> tuple[str, dict[str, bytes]]:
    """(the package's name, its files by relative path). ``worktree``: for the package's own tests only: the files
    as they are in the working directory (still only paths that Git tracks or that the tests name), marked as such.
    In both forms the application files that differ from ACCEPTED must be exactly CHANGED_SINCE_ACCEPTED, and the
    package's own files that differ from the previous revision's exactly PACKAGE_FILES_CHANGED."""
    head = git("rev-parse", "HEAD").decode().strip()
    if worktree:
        app_files = working_files(APP_TREE) + list(GUARD_FILES)
        package_files = sorted(str(path.relative_to(REPO)) for path in (REPO / PACKAGE_SOURCE).rglob("*")
                               if path.is_file() and "__pycache__" not in path.parts)
    else:
        app_files = tracked(APP_TREE) + list(GUARD_FILES)
        package_files = tracked(PACKAGE_SOURCE)
        if git("status", "--porcelain", "--", APP_TREE, *GUARD_FILES, PACKAGE_SOURCE).strip():
            raise SystemExit("STOP: uncommitted changes under the packaged paths; commit them first")
    application_files = {path: read(path, worktree) for path in app_files}
    own_files = {path[len(PACKAGE_SOURCE) + 1:]: read(path, worktree) for path in package_files}
    # what is packaged (HEAD, or the working tree for the tests) against the accepted commit and against the previous
    # revision of the package: by content, so a new file counts as well as a changed one
    accepted = blobs(ACCEPTED, APP_TREE, *GUARD_FILES)
    declared_exactly(differing(accepted, application_files), CHANGED_SINCE_ACCEPTED,
                     f"the application files that differ from the accepted commit {ACCEPTED[:7]}")
    previous = {path[len(PACKAGE_SOURCE) + 1:]: blob for path, blob in blobs(PREVIOUS_PACKAGE_COMMIT, PACKAGE_SOURCE).items()}
    declared_exactly(differing(previous, own_files), PACKAGE_FILES_CHANGED,
                     f"the package's own files that differ from {PREVIOUS_PACKAGE_NAME}")
    members: dict[str, bytes] = dict(own_files)
    for path, content in application_files.items():
        members[f"app/{path}"] = content
    changed = sorted(CHANGED_SINCE_ACCEPTED)

    def before(commit: str, path: str, known: dict[str, str]) -> str:
        return hashlib.sha256(git("show", f"{commit}:{path}")).hexdigest() if path in known else "absent (a new file)"

    sums = {path: (before(ACCEPTED, path, accepted), hashlib.sha256(application_files[path]).hexdigest())
            for path in changed}
    mark = modification_mark({path: packaged for path, (_, packaged) in sums.items()})
    name = (f"sentinel-personal-beta-{ACCEPTED[:7]}" + (f"-mod{mark[:7]}" if changed else "") + f"-r{revision}"
            + ("-worktree" if worktree else ""))
    if changed:
        application = (f"application=the accepted code with {len(changed)} file(s) changed or added after acceptance "
                       "(below); this is NOT the unchanged accepted build\n"
                       f"modification_mark={mark}\n")
        for path in changed:
            reason, run_with = CHANGED_SINCE_ACCEPTED[path]
            application += (f"changed_file=app/{path} accepted_sha256={sums[path][0]} packaged_sha256={sums[path][1]}\n"
                            f"changed_file_reason=app/{path}: {reason}\n"
                            f"changed_file_run_with=app/{path}: {run_with}\n")
        application += (f"unchanged_from_accepted={len(app_files) - len(changed)} application files: every other "
                        f"file under app/{APP_TREE} and app/{GUARD_FILES[0].rsplit('/', 1)[0]}\n")
    else:
        application = f"application=the accepted code, unchanged ({len(app_files)} files)\n"
    own = (f"package_files_changed_since={PREVIOUS_PACKAGE_NAME} (package_commit {PREVIOUS_PACKAGE_COMMIT}): "
           f"{len(PACKAGE_FILES_CHANGED)} of the package's own files (below); VERSION and SHA256SUMS are written "
           "anew for every package\n")
    for path in sorted(PACKAGE_FILES_CHANGED):
        was = before(PREVIOUS_PACKAGE_COMMIT, f"{PACKAGE_SOURCE}/{path}", {f"{PACKAGE_SOURCE}/{key}": value
                                                                         for key, value in previous.items()})
        own += (f"package_file_changed={path} previous_sha256={was} "
                f"packaged_sha256={hashlib.sha256(own_files[path]).hexdigest()}\n"
                f"package_file_changed_reason={path}: {PACKAGE_FILES_CHANGED[path]}\n")
    version = (f"{name}\n"
               f"accepted_code_commit={ACCEPTED}\n"
               f"{application}"
               f"{own}"
               f"package_commit={head}{' (working tree, not a release)' if worktree else ''}\n"
               f"package_revision={revision}\n"
               "status=private personal beta; validates and admits nothing; not for redistribution\n"
               "launch=the default admission admits only the reference device's own detector engine, llama.cpp build "
               "and model-file copies, and REFUSES on another Jetson. The opt-in device-artifacts route "
               "(QUICKSTART.md, section 1) identifies a tester's own files for experimental validation runs and "
               "validates nothing about that device's resources (resource validation pending)\n"
               "route_verified=in synthetic tests on the reference device only; NEVER RUN ON ANOTHER DEVICE; never "
               "with a camera or a model; build-llama.sh beyond its configure step and build-engine.sh have never "
               "run\n")
    members["VERSION"] = version.encode()
    sums = "".join(f"{hashlib.sha256(members[path]).hexdigest()}  {path}\n" for path in sorted(members))
    members["SHA256SUMS"] = sums.encode()
    return name, members


def modification_mark(packaged: dict[str, str]) -> str:
    """One SHA-256 over the changed files' paths and packaged SHA-256 values: the same changes, the same mark."""
    return hashlib.sha256("".join(f"{packaged[path]}  app/{path}\n" for path in sorted(packaged)).encode()).hexdigest()


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
    parser.add_argument("--revision", type=int, default=DEFAULT_REVISION)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if REPO == out or REPO in out.parents:
        raise SystemExit("STOP: --out must be outside the repository: the archive is private and is never committed")
    build(out, args.revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
