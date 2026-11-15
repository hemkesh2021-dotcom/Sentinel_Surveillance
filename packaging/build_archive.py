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
}
# The package's own files (under PACKAGE_SOURCE) that differ from the previous revision's, or that it did not have:
# path -> what changed. The builder refuses unless exactly these differ from PREVIOUS_PACKAGE_COMMIT.
PREVIOUS_PACKAGE_COMMIT = "34809141654f4d7729e72cb6a5ce0eb6fadf1125"
PREVIOUS_PACKAGE_NAME = "sentinel-personal-beta-949da67-mod79c18d4-r5"
_LOCAL = ("PIP_CACHE_DIR, XDG_CACHE_HOME, XDG_CONFIG_HOME, XDG_DATA_HOME, HF_HOME, TORCH_HOME, YOLO_CONFIG_DIR, "
          "DEEPFACE_HOME, CUDA_CACHE_PATH, KERAS_HOME, MPLCONFIGDIR and TMPDIR")
PACKAGE_FILES_CHANGED = {
    "QUICKSTART.md": "rewritten as one sequence for a tester: prerequisites and the TensorRT Python binding (the "
                     "extraction route used on the reference device, with its source, version and SHA-256); archive "
                     "verification and installation; where the installation keeps its files; the environment and the "
                     "downloads; manual memory preparation and the guarded builds, with the reference device's "
                     "observations kept apart from the chosen 5 GiB preparation figure and from the unmeasured "
                     "minimum; enrollment; settings; record and check; launch; limits; uninstall and recovery; which "
                     "steps touch a camera or face data; what ran for real with revision 5 and what was verified "
                     "offline only for this revision",
    "lib/beta.py": "installation-local caches: two more names the installer owns (cache, tmp); a new command, "
                   "local-env, which records, makes and checks <prefix>/cache/... and <prefix>/tmp and prints the "
                   "variables lib/common.sh exports (a link, a directory the installer did not make, or a place that "
                   "resolves outside the installation is refused); uninstall removes both in both of its forms; "
                   "configure shows the two folders. Every other command is as it was",
    "lib/common.sh": f"a new function, beta_local_env: {_LOCAL} are exported as directories inside the installation, "
                     "for the script and for everything it starts, before anything is imported; a value inherited "
                     "from the shell is replaced for that run and named once, and seven variables that would move "
                     "such a place again are removed; HOME is not changed",
    "setup-env.sh": "calls beta_local_env before anything is downloaded or imported, so that pip's download cache "
                    "and the libraries' caches stay inside the installation; the environment is created, and its "
                    "package directory read, without writing bytecode outside it. The pinned requirements, their "
                    "order and the import check are unchanged",
    "fetch-assets.sh": "calls beta_local_env before the downloads; what is downloaded and how it is checked is "
                       "unchanged",
    "build-llama.sh": "calls beta_local_env before cloning and building; states the one measured build (53 minutes "
                      "with two jobs on the reference device) instead of an unmeasured 45 minutes; names "
                      "guarded-build.sh. The repository, the pinned commit, the options and the note are unchanged",
    "build-engine.sh": "calls beta_local_env before the export; its comments and two messages state what was "
                       "observed on the reference device in three attempts (a failure with too little free memory, a "
                       "monitor stop after 136 s, one export completed in 187 s) instead of 'a few minutes', and "
                       "that the least the export needs was not measured; names guarded-build.sh. The export call, "
                       "its settings from the asset manifest, the 1,500,000,000 B start check, the BUILD "
                       "confirmation and the note are unchanged",
    "record-artifacts.sh": "calls beta_local_env before the record is made; the record and its checks are unchanged",
    "identity.sh": "calls beta_local_env before the enrollment command; the face weights' own place is named on the "
                   "command as before",
    "start.sh": "calls beta_local_env after the package check, in both modes, so that the check's temporary "
                "directory and the caches of everything a launch starts are inside the installation; check mode "
                "prints the places in effect (local_env lines). Every gate, limit and prompt, the guard's options "
                "and the runtime's command are unchanged",
    "install.sh": "its closing list of next steps names guarded-build.sh for the two builds; nothing else",
    "preflight.sh": "the failure line for a missing TensorRT Python binding also points to the guide's extraction "
                    "route; no check is changed",
    "uninstall.sh": "its header says that cache and tmp are removed too and what the data directory keeps; the "
                    "command it runs is unchanged",
    "guarded-build.sh": "a new file: runs build-llama.sh or build-engine.sh, unchanged, under the memory monitor "
                        "(lib/guarded_build.py, lib/guarded_run.py), with --check, --status and --stop; it uses no "
                        "sudo, stops no service and frees no memory",
    "lib/guarded_build.py": "a new file: the front end of guarded-build.sh, reduced from the installation "
                            "rehearsal's runner: start conditions (for the engine step a 5 GiB MemFree preparation "
                            "figure, chosen and not measured; no model server, runtime or other build running), a "
                            "typed BUILD, the monitor in a session of its own, a stop relayed to the step's first "
                            "process, the result in <data dir>/builds/. Run with stand-in scripts only",
    "lib/guarded_run.py": "a new file: the installation rehearsal's memory monitor, byte for byte (SHA-256 "
                          "f7f23e4ae5db944d37bcc8db26ec53c4c96a2e9929566bfe0ddc8cb1b721045b): the file that "
                          "supervised the reference device's engine export attempts of 2026-11-14 (stopped at its "
                          "MemFree limit) and 2026-11-15 (completed). It samples memory five times a second, stops "
                          "the step at the limits it is given and signals only the processes that step owns",
}
DEFAULT_REVISION = 6
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
               "files and stand-ins. No file of this revision that differs from revision 5 (above) has run a "
               "download, a build, an export, a model or a camera on any device\n")
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
