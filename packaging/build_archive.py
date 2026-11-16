#!/usr/bin/env python3
"""Builds the personal-beta installation archive from this branch: the tester package that is attached to an
experimental pre-release. Standard library only.

usage: python3 -B packaging/build_archive.py --out DIR [--revision N]

What goes in, and nothing else:
  - the package's own files, as committed: packaging/beta-install/ (scripts, helpers, manifests, templates, guide);
  - the application, under app/: src/sentinel (tracked files only) and the four files the memory guard runs from
    (benchmarks/runner). Every file is exactly as the accepted commit ACCEPTED has it, except the files that
    CHANGED_SINCE_ACCEPTED names (changed, or new): those are packaged as this branch has them, and the package says
    so: its name carries a modification mark, and VERSION names each one with both SHA-256 values, the reason, and
    what has and has not been run with it. A package with a changed file is never named or described as the accepted
    build;
  - LICENSE: the repository's own licence file, as committed. What the package's scripts download keeps its own
    licences and is not packaged; packaging/beta-install/THIRD_PARTY_NOTICES.md lists it;
  - VERSION (what this package is) and SHA256SUMS (every other file in it). VERSION also names every one of the
    package's own files that differs from the previous revision's (PACKAGE_FILES_CHANGED), with both SHA-256 values
    and the reason, and every one that differs from the revision that ran for real on the reference device.
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
LICENSE_FILE = "LICENSE"  # the repository's licence, packaged at the top of the archive
VERSION_FILE = "src/sentinel/__init__.py"  # holds the application's version string
GUARD_FILES = ("benchmarks/runner/step5_guard.py", "benchmarks/runner/operator_check.py",
               "benchmarks/runner/demo_profile.py", "benchmarks/runner/step4_criteria.py")
APP_TREE = "src/sentinel"
# Application files packaged in another form than ACCEPTED has, or that ACCEPTED does not have: path -> (what
# changed, what it has been run with). The builder refuses unless exactly these differ. An empty table means the
# package ships the accepted code only.
_ROUTE_RUN_WITH = ("synthetic tests on the reference device only (temporary directories, synthetic files, stand-ins); "
                   "never on another device, and no run with a camera or a model has used it")
# The two files that read and judge a record: besides those tests, the same bytes ran once on real files.
_ROUTE_RUN_ON_A_REAL_RECORD = (
    "synthetic tests (temporary directories, synthetic files, stand-ins); and once on the reference device, with "
    "revision 5 and these same bytes: the record of that device's own real builds was read and the route's "
    "decisions made for it by lib/admission_check.py (2026-11-15; no camera, no model loaded, no launch). Never on "
    "another device, and no run with a camera or a model has used it")
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
        "the pinned source, options and tools; it admits nothing and measures nothing", _ROUTE_RUN_ON_A_REAL_RECORD),
    "src/sentinel/demo_runtime.py": (
        "the device-artifacts route (session 116, part 41): only with a record, and only in a --face-validation run, "
        "the engine, the llama.cpp build, the model files and the face weights are compared with the device's own "
        "record instead of the reference device's records, by the same startup rule; every other admission rule, "
        "the configuration checks against the accepted profile and the memory policy are unchanged; the starting "
        "line then says that resource validation is pending. Without a record nothing differs",
        _ROUTE_RUN_ON_A_REAL_RECORD),
    "src/sentinel/cli.py": (
        "the device-artifacts route (session 116, part 41): the option --device-artifacts FILE of `sentinel run`, "
        "refused without --face --face-validation and for a record that cannot be read; the detector is then given "
        "the recorded engine's SHA-256 as its pin. Without the option the command decides as before, with one "
        "difference in how it is typed: the abbreviation --d, which used to select --data-dir, is now ambiguous",
        _ROUTE_RUN_WITH),
    VERSION_FILE: (
        "the version string only (session 116, parts 48 and 49): 2.0.0b1, for the pre-release v2.0.0-beta.1, where "
        "the accepted commit has 2.0.0a2. The beta name is a release designation, not a validation result. "
        "`sentinel --version` prints it and nothing else reads it; no behaviour, threshold, guard or admission rule "
        "is changed",
        "the package's tests only; no run on any device with a camera or a model has used it"),
    "src/sentinel/web/dashboard.js": (
        "the Ask panel's conversation on the dashboard (session 116, part 52): the scroll event of a scroll that "
        "the page made itself is no longer taken for the reader's own scrolling. Before, when the conversation's "
        "list was resized before that event arrived (a window being resized) in a browser that does not keep the "
        "view in place itself, the view could stop following the newest exchange: its question was left above the "
        "view, or the view stayed behind. Nothing else of the page is changed: its texts, style, controls, updates "
        "and every other rule of the conversation's scrolling are the accepted commit's",
        "browser tests on the reference device only (headless Chromium with its own scroll anchoring switched off "
        "for the list, synthetic data); not run in a browser that has no scroll anchoring, and no run with a camera "
        "or a model has used it"),
}
# The package's own files (under PACKAGE_SOURCE) that differ from the previous revision's, or that it did not have:
# path -> what changed. The builder refuses unless exactly these differ from PREVIOUS_PACKAGE_COMMIT.
PREVIOUS_PACKAGE_COMMIT = "705eddb449743551fd993871dfb4812f30f1d623"
PREVIOUS_PACKAGE_NAME = "sentinel-personal-beta-949da67-mod9ff5262-r8"
# The one revision that ran for real (the camera-free rehearsal on the reference device, 2026-11-13 to 2026-11-15).
# VERSION names every file of the package that differs from it: those files have real-device evidence from nobody.
REAL_EVIDENCE_PACKAGE_COMMIT = "34809141654f4d7729e72cb6a5ce0eb6fadf1125"
REAL_EVIDENCE_PACKAGE_NAME = "sentinel-personal-beta-949da67-mod79c18d4-r5"
PACKAGE_FILES_CHANGED = {
    "QUICKSTART.md": "six application files, not five, are named as not the accepted build's: the dashboard's "
                     "script is the sixth, with what was changed in it; this revision's name in the two lines that "
                     "check and unpack the archive; which revision changed what. No step, command, limit or figure "
                     "of the procedure is changed",
    "README.md": "what revisions 6, 7, 8 and 9 changed: this revision's change is one file of the application, the "
                 "dashboard's script, checked in a test browser only",
}
DEFAULT_REVISION = 9
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
        if git("status", "--porcelain", "--", APP_TREE, *GUARD_FILES, PACKAGE_SOURCE, LICENSE_FILE).strip():
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
    if LICENSE_FILE in own_files:
        raise SystemExit(f"STOP: {PACKAGE_SOURCE}/{LICENSE_FILE} would hide the repository's own {LICENSE_FILE}")
    licence = read(LICENSE_FILE, worktree)
    version_line = [line for line in application_files[VERSION_FILE].decode().splitlines() if line.startswith("__version__ = ")]
    if len(version_line) != 1:
        raise SystemExit(f"STOP: {VERSION_FILE} does not hold exactly one __version__ line")
    app_version = version_line[0].split('"')[1]
    members: dict[str, bytes] = dict(own_files)
    members[LICENSE_FILE] = licence
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
    # computed, never declared: every file of the package that the revision with real-device evidence did not have
    # in this form. Whatever is named here has not run for real anywhere
    evidence = {path[len(PACKAGE_SOURCE) + 1:]: blob
                for path, blob in blobs(REAL_EVIDENCE_PACKAGE_COMMIT, PACKAGE_SOURCE).items()}
    unproven = differing(evidence, own_files)
    own += (f"package_files_without_real_device_evidence={len(unproven)} of the package's own files differ from "
            f"{REAL_EVIDENCE_PACKAGE_NAME} (package_commit {REAL_EVIDENCE_PACKAGE_COMMIT}), the only revision that "
            f"ran for real, or are new since: {' '.join(unproven)}; and {LICENSE_FILE}, which no earlier revision "
            "held\n")
    version = (f"{name}\n"
               f"accepted_code_commit={ACCEPTED}\n"
               f"{application}"
               f"{own}"
               f"package_commit={head}{' (working tree, not a release)' if worktree else ''}\n"
               f"package_revision={revision}\n"
               f"application_version={app_version}\n"
               "status=experimental tester pre-release; validates and admits nothing; not accepted for production\n"
               "release_designation=v2.0.0-beta.1: the name of a pre-release only. It records no new validation "
               "result, no acceptance and no change of any feature's status\n"
               f"license=MIT for the files in this package ({LICENSE_FILE}, sha256 {hashlib.sha256(licence).hexdigest()}: "
               "the repository's own licence file at the package commit). The Python packages and model files that "
               "its scripts download, and what is built on the device, are not in this package and keep their own "
               "licences, among them AGPL-3.0, the LFM Open License v1.0 and NVIDIA's agreements "
               "(THIRD_PARTY_NOTICES.md)\n"
               "launch=the default admission admits only the reference device's own detector engine, llama.cpp build "
               "and model-file copies, and REFUSES on another Jetson. The opt-in device-artifacts route "
               "(QUICKSTART.md, section 1) identifies a tester's own files for experimental validation runs and "
               "validates nothing about that device's resources (resource validation pending)\n"
               "route_verified=with revision 5, once, on the reference device's own real builds: the record, its "
               "verification and the route's decisions for that record (2026-11-15; no camera, no model loaded). "
               "check.sh stopped there at the missing enrollment, as it must: the complete check and a launch through "
               "the route have NEVER RUN. NEVER RUN ON ANOTHER DEVICE; never with a camera\n"
               "real_device_evidence=revision 5 only, on the reference device, without a camera: installation, the "
               "Python environment, the downloads, the llama.cpp build (53 minutes, two jobs), the engine export "
               "(187 s, started with 6,950,408,192 B MemFree, lowest 2,622,537,728 B, no swap used; after one failure "
               "and one monitor stop), the record and its verification. One device, one completed export: the least "
               "memory and the time an export needs are not known\n"
               "this_revision_verified=offline only: the package's tests, with temporary directories, synthetic "
               "files and stand-ins, and for the dashboard's script the repository's tests in a test browser. That "
               "holds for revision 6's packaging changes (the installation's own cache locations, guarded-build.sh), "
               "for revision 7's documents, licence and notice files, for revision 8's corrections (the release "
               "designation and version string, how two documents name the licence file) and for this revision's "
               "change (one file of the application, the dashboard's script: how the Ask panel's conversation is "
               "followed while it is resized) alike: no file of this revision that differs from revision 5 "
               "(package_files_without_real_device_evidence above, the application's version file and the "
               "dashboard's script) has run a download, a build, an export, a model or a camera on any device\n")
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
    for existing in (target, out / f"{name}.tar.gz.sha256"):
        if existing.exists() or existing.is_symlink():  # an archive that was handed out is never replaced in place
            raise SystemExit(f"STOP: {existing} already exists and is not replaced; choose another --out or revision")
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
    parser.add_argument("--out", type=Path, required=True, help="a directory outside the repository")
    parser.add_argument("--revision", type=int, default=DEFAULT_REVISION)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if REPO == out or REPO in out.parents:
        raise SystemExit("STOP: --out must be outside the repository: an archive is never committed")
    build(out, args.revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
