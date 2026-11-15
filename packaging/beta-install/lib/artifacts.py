#!/usr/bin/env python3
"""The personal-beta package's device-artifacts record: this device's own files, for experimental validation runs.

Standard library only; any Python 3.9 or newer on Linux. It never opens the camera, loads a model, reads a credential
or uses the network. Sub-commands (the shell scripts call them):

  note    --package DIR --step build-llama|build-engine [--export JSON]
              after a build step of this package has built something: what it built it from, with which settings
              and tools, and the SHA-256 of what it made. Written under <prefix>/assets/provenance/.
  record  --package DIR [--accept-unverified-platform]
              write <data dir>/config/device-artifacts.json: every file the launch uses (name, size, modification
              time, SHA-256), what this package's steps observed about each, and this device's platform versions.
  verify  --package DIR --record FILE
              every file against the record, the record against this package's pins, the platform against the
              record. Changes nothing. Exit 0 only when everything is as recorded.

What a record says, and what it cannot say:
  - Provenance is only what this package's own steps observed. A file made by build-llama.sh or build-engine.sh is
    recorded as built by that step when, and only when, its bytes are the ones the step hashed when it finished. Any
    other file is "inspected_only": its checksum is known, how it was built is not, and no record is written for it.
    A downloaded file is identified by the pinned SHA-256 either way; that this package downloaded it is recorded
    only for the very copy fetch-assets.sh wrote (the same place, the same modification time).
  - What "built by build-llama.sh" rests on: the build tree was started by that script for the pinned commit and
    options (its marker), the source tree is at the pinned commit with no tracked change, the configuration holds the
    pinned options and no extra compiler flags, and the script's own build command linked every file it then hashed.
    Object files left by an interrupted earlier run of the script are reused; a tree that somebody changed by hand
    between two runs of the script cannot be told apart from one that nobody touched.
  - It identifies files. It is not a resource profile and measures nothing: "resource_validation" is always
    "pending". The reference device's measurements are never copied into it.
The runtime reads the same record with its own, stricter reader (app/src/sentinel/device_artifacts.py) and applies
its own pins; this helper's checks come first and hash every file whole.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))  # -I leaves the script's own directory out
import beta  # noqa: E402  (the package's bookkeeping, beside this file and checked with it)
from beta import Stop, say  # noqa: E402

SCHEMA = "sentinel-device-artifacts/1"
PURPOSE = ("identifies this device's own files for experimental validation runs; it is not a resource profile and "
           "measures nothing")
PENDING = "pending"
INSPECTED_STATEMENT = ("only this file's checksum was read; how it was built was not observed by this package's "
                       "build step, so nothing is recorded about its source or settings")
VERIFIED_STATEMENT = "this file equals the pinned SHA-256; this package did not observe its download"
TEST_FIXTURE = "facts were read from a test fixture, not from a device"
BUILD_MARKER = ".sentinel-beta-build"  # build-llama.sh writes it when it starts a build tree: the commit and options
PENDING_NOTE = ("no run on this device has been measured or judged. The accepted resource profile, the launch gates "
                "and the guard's limits come from the reference device: here they are limits, not evidence about "
                "this device")
MAX_RECORD_BYTES = 262_144
BUILT, INSPECTED = "built_by_package_step", "inspected_only"
DOWNLOADED, VERIFIED = "downloaded_by_package_step", "verified_against_pinned_checksum"
RECORD_NAME = "device-artifacts.json"
# role in the record -> the asset in manifest/assets.json
BUILT_ROLES = {"detector_engine": "detector_engine", "llama_server": "llama_server"}
PINNED_ROLES = {"scene_model": "scene_model", "scene_mmproj": "scene_mmproj", "face_model_weights": "face_weights",
                "face_detector_weights": "face_detector"}
ROLES = (*BUILT_ROLES, *PINNED_ROLES)
# What the provenance of a file may hold, by what was observed: exactly these keys (as the runtime's reader requires).
PROVENANCE_KEYS = {
    ("detector_engine", "built_by_package_step"): ("observed", "step", "utc", "package", "weights_sha256", "export", "tools"),
    ("llama_server", "built_by_package_step"): ("observed", "step", "utc", "package", "repository", "commit", "cmake",
                                                "target", "reports", "tools"),
    ("built", "inspected_only"): ("observed", "statement"),
    ("pinned", "downloaded_by_package_step"): ("observed", "step", "utc", "url", "revision"),
    ("pinned", "verified_against_pinned_checksum"): ("observed", "pinned_url", "revision", "statement"),
}
STEPS = {"build-llama": "llama_server", "build-engine": "detector_engine"}
PLATFORM_TEXT = ("device_model", "machine", "kernel", "l4t", "os", "cuda_toolkit", "tensorrt", "cudnn", "python")
PLATFORM_PACKAGES = ("torch", "torchvision", "ultralytics", "tensorrt", "onnx", "onnxslim", "onnxruntime-gpu",
                     "onnxruntime", "tensorflow", "tf_keras", "deepface", "opencv-python", "numpy")
ENGINE_TOOLS = ("ultralytics", "tensorrt", "torch", "onnx", "onnxslim", "onnxruntime-gpu", "onnxruntime")
VERIFIED_COMBINATION, NOT_VERIFIED = "verified_combination", "not_verified_acknowledged"
METADATA = ("import importlib.metadata as m, json, sys\n"
            "out = {'python': '%d.%d.%d' % sys.version_info[:3], 'packages': {}}\n"
            "for name in sys.argv[1:]:\n"
            "    try:\n"
            "        out['packages'][name] = m.version(name)\n"
            "    except m.PackageNotFoundError:\n"
            "        out['packages'][name] = 'missing'\n"
            "print(json.dumps(out))\n")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MTIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+00:00")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,199}")
_LINE = re.compile(r"[ -~]{0,200}")


def real_time(text: object, pattern: re.Pattern) -> bool:
    """A time in the schema's one form that is also a real time."""
    if not isinstance(text, str) or not pattern.fullmatch(text):
        return False
    try:
        datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return False
    return True


def same(one: object, other: object) -> bool:
    """Equal as JSON: true is not 1, and a list's order counts."""
    return json.dumps(one, sort_keys=True) == json.dumps(other, sort_keys=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- files


def facts(path: Path) -> dict:
    """Name, size, modification time and SHA-256 of ``path``, read whole. The pages the read brought into the page
    cache are given back afterwards (a hint the kernel may ignore), so that a check before a launch does not use up
    the free memory the launch itself needs. Stop when the file changes while it is read."""
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
            try:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
        after = path.stat()
    except OSError as error:
        raise Stop(f"{path} cannot be read ({type(error).__name__})") from None
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise Stop(f"{path} changed while it was being read; run this again when nothing is writing it")
    mtime = datetime.fromtimestamp(after.st_mtime, timezone.utc).isoformat(timespec="seconds")
    return {"name": path.name, "bytes": after.st_size, "sha256": digest.hexdigest(), "mtime_utc": mtime}


def assets_by_id(paths: dict[str, str]) -> dict[str, dict]:
    return {asset["id"]: asset for asset in beta.load_assets(Path(paths["PKG"]) / "manifest" / "assets.json")}


def role_paths(paths: dict[str, str], assets: dict[str, dict]) -> dict[str, Path]:
    """Where each role's file is now: the tester's own setting when one names it, else its place in the
    installation (as every other script of the package resolves it)."""
    return {role: beta.asset_target(paths, assets[asset_id]) for role, asset_id in {**BUILT_ROLES, **PINNED_ROLES}.items()}


def library_names(assets: dict[str, dict]) -> list[str]:
    names = assets["llama_server"].get("libraries")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) and "/" not in name for name in names):
        raise Stop("the asset manifest does not list the llama.cpp build's libraries")
    return sorted(names)


def package_version(paths: dict[str, str]) -> str:
    return (Path(paths["PKG"]) / "VERSION").read_text("utf-8").splitlines()[0]


# ---------------------------------------------------------------- what a build step observed


def note_path(paths: dict[str, str], asset_id: str) -> Path:
    return Path(paths["ASSETS"]) / "provenance" / f"{asset_id}.json"


def read_note(paths: dict[str, str], asset_id: str) -> dict | None:
    try:
        note = json.loads(note_path(paths, asset_id).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return note if isinstance(note, dict) else None


def first_line(command: list[str], contains: str = "") -> str:
    """One line of a tool's own version output; "unknown" when it cannot be run. Never a model, never the GPU."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    for line in (done.stdout + done.stderr).splitlines():
        if line.strip() and contains in line:
            return line.strip()[:200]
    return "unknown"


def environment_versions(paths: dict[str, str], names: tuple[str, ...]) -> dict:
    """The runtime environment's Python version and the installed versions of ``names``, read from the installed
    metadata by that environment's own interpreter. Nothing is imported or loaded."""
    interpreter = paths["RT_PY"]
    if not os.access(interpreter, os.X_OK):
        raise Stop(f"the Python environment is not there ({paths['VENV']}). Run: bash {paths['PKG']}/setup-env.sh")
    try:
        done = subprocess.run([interpreter, "-I", "-B", "-c", METADATA, *names], capture_output=True, text=True,
                              timeout=120, stdin=subprocess.DEVNULL)
        found = json.loads(done.stdout)
        versions = {name: str(found["packages"][name]) for name in names}
        python = str(found["python"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        raise Stop(f"the Python environment's package versions could not be read ({interpreter}). "
                   f"Run: bash {paths['PKG']}/setup-env.sh") from None
    missing = [name for name, version in versions.items() if version == "missing"]
    if missing:
        raise Stop(f"the Python environment lacks: {', '.join(missing)}. Run: bash {paths['PKG']}/setup-env.sh")
    return {"python": python, "packages": versions}


def cmake_cache(build: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = (build / "CMakeCache.txt").read_text("utf-8", errors="replace").splitlines()
    except OSError:
        raise Stop(f"{build}/CMakeCache.txt is not there: the build tree was not configured by build-llama.sh") from None
    for line in lines:
        found = re.match(r"([A-Za-z0-9_]+):[A-Z]+=(.*)$", line)
        if found:
            values[found.group(1)] = found.group(2)
    return values


def note_llama(paths: dict[str, str], assets: dict[str, dict]) -> dict:
    """After build-llama.sh's build: the source tree is at the pinned commit with no tracked change, the build tree
    was configured with the pinned options, and these are the files it holds now."""
    asset = assets["llama_server"]
    source = Path(paths["ASSETS"]) / "llama.cpp"
    build = source / "build"

    def git(*args: str) -> str:
        try:
            done = subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True, timeout=60,
                                  stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            raise Stop("git could not be run on the llama.cpp source tree") from None
        if done.returncode != 0:
            raise Stop("git could not read the llama.cpp source tree")
        return done.stdout.strip()

    if git("rev-parse", "HEAD") != asset["commit"]:
        raise Stop(f"the llama.cpp source tree is not at the pinned commit {asset['commit']}; nothing was noted")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise Stop("the llama.cpp source tree has local changes; nothing was noted")
    cache, options = cmake_cache(build), list(asset["cmake"])
    try:
        marker = (build / BUILD_MARKER).read_text("utf-8").strip()
    except OSError:
        marker = ""
    if marker != f"{asset['commit']} {' '.join(options)}":
        raise Stop(f"{build} was not started by build-llama.sh for the pinned commit and options (its marker is missing "
                   "or names others): a build tree made another way may hold other settings; nothing was noted")
    for option in options:
        key, _, value = option[2:].partition("=")
        if not option.startswith("-D") or cache.get(key) != value:
            raise Stop(f"the build tree was not configured with {option} (it has '{cache.get(key)}'); nothing was noted")
    for key in ("CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_CUDA_FLAGS"):
        if cache.get(key, "").strip():
            raise Stop(f"the build tree was configured with extra compiler flags ({key}={cache[key].strip()}): that is "
                       "not the pinned build; nothing was noted")
    binary = build / "bin" / "llama-server"
    outputs = {}
    for name in ["llama-server", *library_names(assets)]:
        target = build / "bin" / name
        if not target.is_file():
            raise Stop(f"the build ended without {target}; nothing was noted")
        found = facts(target)
        outputs[name] = {"sha256": found["sha256"], "bytes": found["bytes"]}
    compilers = {key: cache.get(key, "") for key in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER")}
    return {
        "observed": BUILT, "step": "build-llama.sh", "utc": utc_now(), "package": package_version(paths),
        "repository": asset["repo"], "commit": asset["commit"], "cmake": options, "target": asset["cmake_target"],
        "reports": first_line([str(binary), "--version"], "version:"),
        "tools": {"cmake": first_line(["cmake", "--version"], "version"),
                  # the compilers by name only: a path may lie under the tester's home, and a record holds none
                  **{key.lower(): Path(value).name or "unknown" for key, value in compilers.items()},
                  "cxx": first_line([compilers["CMAKE_CXX_COMPILER"], "--version"]) if compilers["CMAKE_CXX_COMPILER"] else "unknown",
                  "cuda": (first_line([compilers["CMAKE_CUDA_COMPILER"], "--version"], "release")
                           if compilers["CMAKE_CUDA_COMPILER"] else "unknown")},
        "outputs": outputs,
    }


def note_engine(paths: dict[str, str], assets: dict[str, dict], export: str) -> dict:
    """After build-engine.sh's export: the weights are the pinned file, the settings are the ones the export call was
    given, and this is the engine it wrote."""
    weights_asset = assets["detector_weights"]
    weights = beta.asset_target(paths, weights_asset)
    problem = beta.check_file(weights_asset, weights)
    if problem is not None or not weights_asset.get("sha256"):
        raise Stop(f"the detector weights at {weights} are not the pinned file ({problem}); nothing was noted")
    try:
        settings = json.loads(export)
    except ValueError:
        settings = None
    if not isinstance(settings, dict) or not same(settings, assets["detector_engine"].get("export")):
        raise Stop("the export settings given are not the asset manifest's; nothing was noted")
    engine = Path(paths["ASSETS"]) / "engine" / "yolov8n.engine"
    if not engine.is_file():
        raise Stop(f"the export ended without {engine}; nothing was noted")
    found = facts(engine)
    versions = environment_versions(paths, ENGINE_TOOLS)
    return {
        "observed": BUILT, "step": "build-engine.sh", "utc": utc_now(), "package": package_version(paths),
        "weights_sha256": weights_asset["sha256"], "export": settings,
        "tools": {**versions["packages"], "python": versions["python"]},
        "outputs": {engine.name: {"sha256": found["sha256"], "bytes": found["bytes"]}},
    }


def command_note(args: argparse.Namespace) -> int:
    paths = beta.resolved_paths(Path(args.package))
    assets = assets_by_id(paths)
    note = note_llama(paths, assets) if args.step == "build-llama" else note_engine(paths, assets, args.export)
    beta.write_note(paths, STEPS[args.step], note)
    say(f"Noted what this step built (for the device-artifacts record): {note_path(paths, STEPS[args.step])}")
    return 0


# ---------------------------------------------------------------- the platform, as it is now


def platform_facts(paths: dict[str, str]) -> dict:
    """This device's relevant versions, read as preflight.sh reads them (files and the package database; for the
    package's own tests, from the directory tree that stands in for "/"). Nothing is loaded."""
    root = os.environ.get("SENTINEL_BETA_FACTS_ROOT", "")

    def read(relative: str) -> str:
        try:
            return Path(root + relative).read_text("utf-8", errors="replace")
        except OSError:
            return ""

    def dpkg(name: str) -> str:
        try:
            done = subprocess.run(["dpkg-query", "-W", "-f=${Version}", name], capture_output=True, text=True,
                                  timeout=30, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            return ""
        return done.stdout.strip() if done.returncode == 0 else ""

    release = re.search(r"^# R(\d+) \(release\), REVISION: ([0-9.]+)", read("/etc/nv_tegra_release"), re.M)
    system = {key: value.strip().strip('"') for key, _, value in
              (line.partition("=") for line in read("/etc/os-release").splitlines()) if key in ("ID", "VERSION_ID")}
    uname = os.uname()
    found = {
        "device_model": read("/proc/device-tree/model").replace("\0", "").strip(),
        "machine": uname.machine, "kernel": uname.release,
        "l4t": f"{release.group(1)}.{release.group(2)}" if release else "",
        "os": f"{system.get('ID', '')} {system.get('VERSION_ID', '')}".strip(),
        "cuda_toolkit": dpkg("cuda-toolkit-12-6"), "tensorrt": dpkg("libnvinfer10"), "cudnn": dpkg("libcudnn9-cuda-12"),
        **environment_versions(paths, PLATFORM_PACKAGES),
    }
    unread = [key for key in PLATFORM_TEXT if not found[key]]
    if unread:
        raise Stop(f"this device's {', '.join(unread)} could not be read. bash {paths['PKG']}/preflight.sh explains")
    if root:
        found["test_fixture"] = TEST_FIXTURE
    return found


def preflight(paths: dict[str, str]) -> tuple[int, str]:
    environment = dict(os.environ)  # with what install.sh recorded for this installation, as it was checked then
    for variable, key in (("SENTINEL_BETA_PYTHON310", "PYTHON310"), ("SENTINEL_BETA_TENSORRT_PYTHON", "TENSORRT_PYTHON")):
        if paths.get(key):
            environment[variable] = paths[key]
    try:
        done = subprocess.run(["bash", str(Path(paths["PKG"]) / "preflight.sh")], capture_output=True, text=True,
                              timeout=300, stdin=subprocess.DEVNULL, env=environment)
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return done.returncode, done.stdout


# ---------------------------------------------------------------- the record


def artifact_entries(paths: dict[str, str], assets: dict[str, dict]) -> dict:
    """Every role's file as it is now, with what this package's steps observed about it."""
    where = role_paths(paths, assets)
    entries: dict[str, dict] = {}
    for role, asset_id in PINNED_ROLES.items():
        asset, target = assets[asset_id], where[role]
        if not target.is_file():
            raise Stop(f"{role}: MISSING ({target}); nothing was recorded. {asset.get('how', '')}".rstrip())
        found = facts(target)
        if (found["name"], found["bytes"], found["sha256"]) != (Path(asset["path"]).name, asset.get("size"), asset.get("sha256")):
            raise Stop(f"{role}: {target} is not the pinned file {Path(asset['path']).name} (by name, size or "
                       f"SHA-256); nothing was recorded. {asset.get('how', '')}".rstrip())
        note = read_note(paths, asset_id)
        # this package's own download only for the very copy it wrote: the same place and modification time
        if (note is not None and note.get("observed") == DOWNLOADED and note.get("sha256") == found["sha256"]
                and note.get("url") == asset.get("url") and note.get("target") == str(target)
                and note.get("mtime_utc") == found["mtime_utc"] and real_time(note.get("utc"), _UTC)):
            seen = {"observed": DOWNLOADED, "step": "fetch-assets.sh", "utc": note["utc"],
                    "url": asset["url"], "revision": str(asset.get("revision", ""))}
        else:
            seen = {"observed": VERIFIED, "pinned_url": asset.get("url", ""), "revision": str(asset.get("revision", "")),
                    "statement": VERIFIED_STATEMENT}
        entries[role] = {"file": found, "provenance": seen}
    for role, asset_id in BUILT_ROLES.items():
        asset, target = assets[asset_id], where[role]
        if not target.is_file():
            raise Stop(f"{role}: MISSING ({target}); nothing was recorded. {asset.get('how', '')}".rstrip())
        entry: dict = {"file": facts(target)}
        made = {target.name: entry["file"]}
        if role == "llama_server":
            entry["libraries"] = []
            for name in library_names(assets):
                library = target.parent / name
                if not library.is_file():
                    raise Stop(f"llama_server: the library {library} is MISSING; nothing was recorded. "
                               f"{asset.get('how', '')}".rstrip())
                entry["libraries"].append(facts(library))
                made[name] = entry["libraries"][-1]
        note = read_note(paths, asset_id)
        outputs = note.get("outputs") if note is not None and note.get("observed") == BUILT else None
        keys = PROVENANCE_KEYS[(role, BUILT)]
        if (isinstance(outputs, dict) and sorted(outputs) == sorted(made) and all(key in note for key in keys)
                and all(isinstance(outputs[name], dict) and outputs[name].get("sha256") == made[name]["sha256"]
                        for name in made)):
            entry["provenance"] = {key: note[key] for key in keys}  # the schema's keys and nothing else of the note
        else:
            entry["provenance"] = {"observed": INSPECTED, "statement": INSPECTED_STATEMENT}
        entries[role] = entry
    return entries


def canonical(document: dict) -> str:
    return json.dumps(document, indent=1, sort_keys=True) + "\n"


def command_record(args: argparse.Namespace) -> int:
    paths = beta.resolved_paths(Path(args.package))
    assets = assets_by_id(paths)
    code, told = preflight(paths)
    if code not in (0, 2):
        say(told.rstrip())
        raise Stop("the preflight failed on this device (above); nothing was recorded")
    if code == 2 and not args.accept_unverified_platform:
        say("\n".join(line for line in told.splitlines() if line.startswith(("NOT VERIFIED", "  "))))
        raise Stop("this platform was NOT verified (above); nothing was recorded. To record its files anyway, "
                   "knowing that nothing was ever run on this combination: "
                   f"bash {paths['PKG']}/record-artifacts.sh --accept-unverified-platform")
    platform = platform_facts(paths)
    platform["preflight"] = VERIFIED_COMBINATION if code == 0 else NOT_VERIFIED
    document = {"schema": SCHEMA, "created_utc": utc_now(), "package": package_version(paths), "purpose": PURPOSE,
                "resource_validation": PENDING, "platform": platform, "artifacts": artifact_entries(paths, assets)}
    target = Path(paths["DATA_DIR"]) / "config" / RECORD_NAME
    unusable = record_problems(document, assets)
    if unusable:  # said, and not written: a record that no launch could use would only stand in a later one's way
        for role in ROLES:
            entry = document["artifacts"][role]
            say(f"artifact {role}: {entry['file']['name']} SHA-256 {entry['file']['sha256']} [{entry['provenance']['observed']}]")
        for problem in unusable:
            say(f"NOT USABLE FOR A LAUNCH: {problem}")
        raise Stop("nothing was recorded: a record of these files could not be used for a launch (above)")
    kept = False
    if os.path.lexists(target):
        try:
            existing = json.loads(target.read_text("utf-8"))
        except (OSError, ValueError):
            existing = None
        same = isinstance(existing, dict) and {**existing, "created_utc": ""} == {**document, "created_utc": ""}
        if not same or target.is_symlink():
            raise Stop(f"{target} is already there and describes other files, another platform or another package: "
                       f"it is not replaced. Move it away yourself (for example: mv '{target}' '{target}.old'), "
                       "then run this again")
        document, kept = existing, True
    else:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical(document))
    say(f"device_artifacts_record={target}" + (" (already there and unchanged; kept)" if kept else " (written)"))
    for role in ROLES:
        entry = document["artifacts"][role]
        say(f"artifact {role}: {entry['file']['name']} SHA-256 {entry['file']['sha256']} [{entry['provenance']['observed']}]")
    say(f"platform: {platform_line(document['platform'])}")
    say(f"platform_preflight={document['platform']['preflight']}")
    if document["platform"].get("test_fixture"):
        say("TEST FIXTURE: this record's platform facts came from a test fixture, not from a device; no run uses it")
    say(f"resource_validation={PENDING}")
    say(f"resource_validation_note={PENDING_NOTE}")
    say(f"Next: bash {paths['PKG']}/check.sh --device-artifacts '{target}'")
    return 0


def platform_line(platform: dict) -> str:
    return (f"{platform.get('device_model')}; L4T R{platform.get('l4t')}; {platform.get('os')}; kernel "
            f"{platform.get('kernel')}; CUDA {platform.get('cuda_toolkit')}; TensorRT {platform.get('tensorrt')}; "
            f"cuDNN {platform.get('cudnn')}; Python {platform.get('python')}")


# ---------------------------------------------------------------- reading a record, and checking against it


def read_record(data: bytes) -> dict:
    """The record in ``data``, with the shape the schema gives it; Stop for anything else. A record that claims more
    than "pending" for resource validation is not a record."""

    def unique(pairs: list) -> dict:
        found: dict = {}
        for key, value in pairs:
            if key in found:
                raise Stop("the record gives a key twice")
            found[key] = value
        return found

    def shape(value: object, where: str, keys: tuple[str, ...]) -> dict:
        if not isinstance(value, dict) or sorted(value) != sorted(keys):
            raise Stop(f"the record's {where} does not hold exactly: {', '.join(keys)}")
        return value

    def file(value: object, where: str) -> None:
        item = shape(value, where, ("name", "bytes", "sha256", "mtime_utc"))
        if (not isinstance(item["name"], str) or not item["name"] or "/" in item["name"]
                or type(item["bytes"]) is not int or item["bytes"] < 1
                or not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"])
                or not real_time(item["mtime_utc"], _MTIME)):
            raise Stop(f"the record's {where} is not a file entry")

    def provenance(role: str, seen: object) -> None:
        """One of the schema's forms, with exactly its keys: a record has no place for a measurement or a claim."""
        kind = "built" if role in BUILT_ROLES else "pinned"
        allowed = (BUILT, INSPECTED) if kind == "built" else (DOWNLOADED, VERIFIED)
        if not isinstance(seen, dict) or seen.get("observed") not in allowed:
            raise Stop(f"the record's artifacts.{role}.provenance.observed is not one of: {', '.join(allowed)}")
        keys = PROVENANCE_KEYS[(role if seen["observed"] == BUILT else kind, seen["observed"])]
        if sorted(seen) != sorted(keys):
            raise Stop(f"the record's artifacts.{role}.provenance does not hold exactly: {', '.join(keys)}")
        statement = INSPECTED_STATEMENT if seen["observed"] == INSPECTED else VERIFIED_STATEMENT
        tools = seen.get("tools", {})
        if (seen.get("statement", statement) != statement or not real_time(seen.get("utc", "2000-01-01T00:00:00Z"), _UTC)
                or not isinstance(seen.get("package", "x"), str) or not _NAME.fullmatch(seen.get("package", "x"))
                or not isinstance(tools, dict)
                or not all(isinstance(item, str) and _LINE.fullmatch(item) for item in tools.values())
                or not isinstance(seen.get("reports", ""), str) or not _LINE.fullmatch(seen.get("reports", ""))):
            raise Stop(f"the record's artifacts.{role}.provenance is not as the schema requires")

    if len(data) > MAX_RECORD_BYTES:
        raise Stop("the record is larger than a record can be")
    try:
        document = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise Stop("the record is not a JSON document") from None
    top = shape(document, "top level", ("schema", "created_utc", "package", "purpose", "resource_validation",
                                        "platform", "artifacts"))
    if top["schema"] != SCHEMA:
        raise Stop(f"the record's schema is not {SCHEMA}")
    if top["purpose"] != PURPOSE:
        raise Stop("the record's purpose is not the schema's own sentence")
    if top["resource_validation"] != PENDING:
        raise Stop("the record's resource_validation is not \"pending\": a record identifies files and validates nothing")
    if (not real_time(top["created_utc"], _UTC) or not isinstance(top["package"], str)
            or not _NAME.fullmatch(top["package"])):
        raise Stop("the record's created_utc or package is not as the schema requires")
    platform = top["platform"]
    if (not isinstance(platform, dict)
            or any(not isinstance(platform.get(key), str) or not platform[key] or not _LINE.fullmatch(platform[key])
                   for key in PLATFORM_TEXT)
            or not isinstance(platform.get("packages"), dict) or not platform["packages"]
            or not all(isinstance(version, str) and _NAME.fullmatch(version) for version in platform["packages"].values())
            or platform.get("preflight") not in (VERIFIED_COMBINATION, NOT_VERIFIED)
            or platform.get("test_fixture", TEST_FIXTURE) != TEST_FIXTURE
            or not set(platform) <= {*PLATFORM_TEXT, "packages", "preflight", "test_fixture"}):
        raise Stop("the record's platform does not hold exactly its versions, packages and preflight")
    artifacts = shape(top["artifacts"], "artifacts", ROLES)
    for role in ROLES:
        keys = ("file", "libraries", "provenance") if role == "llama_server" else ("file", "provenance")
        entry = shape(artifacts[role], f"artifacts.{role}", keys)
        file(entry["file"], f"artifacts.{role}.file")
        provenance(role, entry["provenance"])
        if role == "llama_server":
            if not isinstance(entry["libraries"], list) or not entry["libraries"]:
                raise Stop("the record's artifacts.llama_server.libraries is not a list of files")
            for library in entry["libraries"]:
                file(library, "artifacts.llama_server.libraries[]")
    return top


def record_problems(document: dict, assets: dict[str, dict]) -> list[str]:
    """Why a launch cannot use this record with this package: a built file whose build was not observed, or settings
    that are not this package's pins. Empty when it can."""
    problems = []
    artifacts = document["artifacts"]
    for role, asset_id in PINNED_ROLES.items():
        asset, recorded = assets[asset_id], artifacts[role]["file"]
        if (recorded["name"], recorded["bytes"], recorded["sha256"]) != (Path(asset["path"]).name, asset.get("size"),
                                                                        asset.get("sha256")):
            problems.append(f"{role}: the recorded file is not the pinned {Path(asset['path']).name}")
    llama, engine = artifacts["llama_server"]["provenance"], artifacts["detector_engine"]["provenance"]
    asset = assets["llama_server"]
    if llama["observed"] != BUILT:
        problems.append("llama_server: its build was not observed by this package's step. A checksum does not show "
                        "how a file was built. Build it with: bash build-llama.sh, then record again")
    elif not same([llama.get("repository"), llama.get("commit"), llama.get("cmake"), llama.get("target")],
                  [asset.get("repo"), asset.get("commit"), asset.get("cmake"), asset.get("cmake_target")]):
        problems.append("llama_server: it was not built from the pinned repository and commit with the pinned options")
    if sorted(library["name"] for library in artifacts["llama_server"]["libraries"]) != library_names(assets):
        problems.append("llama_server: the recorded libraries are not the ones the pinned build consists of")
    asset = assets["detector_engine"]
    if engine["observed"] != BUILT:
        problems.append("detector_engine: its build was not observed by this package's step. A checksum does not "
                        "show how a file was built. Build it with: bash build-engine.sh, then record again")
    else:
        tools, pinned = engine.get("tools"), asset.get("tools")
        if (engine.get("weights_sha256") != assets["detector_weights"].get("sha256")
                or not same(engine.get("export"), asset.get("export"))):
            problems.append("detector_engine: it was not exported from the pinned weights with the pinned settings")
        elif (not isinstance(tools, dict) or not isinstance(pinned, dict) or not pinned
              or any(tools.get(name) != version for name, version in pinned.items())):
            problems.append("detector_engine: it was not exported with the pinned tool versions")
    return problems


def command_verify(args: argparse.Namespace) -> int:
    """Everything a launch on a record relies on, checked without changing anything: the record itself, its fit to
    this package's pins, every file whole against it, and the platform against it."""
    paths = beta.resolved_paths(Path(args.package))
    assets = assets_by_id(paths)
    record = Path(args.record)
    try:
        with record.open("rb") as handle:
            data = handle.read(MAX_RECORD_BYTES + 1)
    except OSError as error:
        raise Stop(f"the device-artifacts record {record} cannot be read ({type(error).__name__}). Make one with: "
                   f"bash {paths['PKG']}/record-artifacts.sh") from None
    document = read_record(data)
    say(f"device_artifacts_record={record}")
    say(f"device_artifacts_record_sha256={hashlib.sha256(data).hexdigest()}")
    say(f"device_artifacts_created_utc={document['created_utc']} by {document['package']}")
    problems = record_problems(document, assets)
    where = role_paths(paths, assets)
    for role in ROLES:
        entry = document["artifacts"][role]
        files = [(where[role], entry["file"])]
        files += [(where[role].parent / library["name"], library) for library in entry.get("libraries", [])]
        wrong = []
        for path, recorded in files:
            if not path.is_file():
                wrong.append(f"{path} is MISSING")
                continue
            now = facts(path)
            if now["name"] != recorded["name"]:
                wrong.append(f"{path} is not named {recorded['name']}")
            elif (now["bytes"], now["sha256"]) != (recorded["bytes"], recorded["sha256"]):
                wrong.append(f"{path} has CHANGED since it was recorded (its SHA-256 is {now['sha256']}, the record "
                             f"has {recorded['sha256']})")
            elif now["mtime_utc"] != recorded["mtime_utc"]:
                wrong.append(f"{path} has the recorded content but another modification time ({now['mtime_utc']}, "
                             f"recorded {recorded['mtime_utc']}): it was rewritten or copied since")
        if wrong:
            problems += [f"{role}: {item}" for item in wrong]
            say(f"artifact {role}: NOT AS RECORDED")
        else:
            say(f"artifact {role}: ok ({entry['file']['name']}, SHA-256 {entry['file']['sha256']}; "
                f"{entry['provenance']['observed']})")
    code, _ = preflight(paths)
    expected = {0: VERIFIED_COMBINATION, 2: NOT_VERIFIED}.get(code)
    recorded_platform = document["platform"]
    if expected is None:
        problems.append(f"platform: the preflight fails on this device now. bash {paths['PKG']}/preflight.sh explains")
    elif expected != recorded_platform["preflight"]:
        problems.append(f"platform: the preflight now reads '{expected}', the record has "
                        f"'{recorded_platform['preflight']}'")
    now_platform = {**platform_facts(paths), "preflight": recorded_platform["preflight"]}
    for key in sorted(set(now_platform) | set(recorded_platform)):
        if now_platform.get(key) != recorded_platform.get(key):
            problems.append(f"platform: {key} is {json.dumps(now_platform.get(key), sort_keys=True)} now; the record "
                            f"has {json.dumps(recorded_platform.get(key), sort_keys=True)}")
    if recorded_platform.get("test_fixture"):
        say("TEST FIXTURE: this record's platform facts came from a test fixture, not from a device")
    say(f"platform_facts={'test_fixture' if recorded_platform.get('test_fixture') else 'this_device'}")
    say(f"platform: {platform_line(recorded_platform)}")
    say(f"platform_preflight={recorded_platform['preflight']}"
        + (" (NOT VERIFIED: nothing was ever run on this combination; acknowledged when the record was made)"
           if recorded_platform["preflight"] == NOT_VERIFIED else ""))
    if problems:
        for problem in problems:
            say(f"REFUSED: {problem}")
        say("artifact_verification=FAILED")
        say("What to do: a changed or rebuilt file needs its own step again (fetch-assets.sh, build-llama.sh, "
            "build-engine.sh) and a new record (move the old one away, then record-artifacts.sh); never edit a "
            "record or a file's time by hand.")
        return 1
    say("artifact_verification=passed (every file equals the record, whole, by SHA-256)")
    say(f"resource_validation={PENDING}")
    say(f"resource_validation_note={PENDING_NOTE}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentinel personal-beta device-artifacts record (see the module docstring).")
    commands = parser.add_subparsers(dest="command", required=True)
    note = commands.add_parser("note")
    note.add_argument("--package", required=True)
    note.add_argument("--step", required=True, choices=sorted(STEPS))
    note.add_argument("--export", default="")
    record = commands.add_parser("record")
    record.add_argument("--package", required=True)
    record.add_argument("--accept-unverified-platform", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--package", required=True)
    verify.add_argument("--record", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"note": command_note, "record": command_record, "verify": command_verify}
    try:
        return handlers[args.command](args)
    except Stop as stop:
        print(f"STOP: {stop}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
