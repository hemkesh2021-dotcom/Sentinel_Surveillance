#!/usr/bin/env python3
"""The personal-beta package's bookkeeping: where things are, what the installer owns, the asset manifest.

Standard library only; any Python 3.9 or newer on Linux. It never opens the camera, loads a model, reads a credential or
touches anything outside the prefix and the data directory it is given. Sub-commands (the shell scripts call them):

  verify-package --package DIR               every listed file equals its checksum and no other file is there
  paths        --package DIR                 print every path the scripts use, one KEY=VALUE per line
  install      --source DIR --prefix DIR --data-dir DIR --python310 FILE [--tensorrt-python DIR]
  own          --prefix DIR NAME             record that the installer made <prefix>/NAME (venv, assets)
  uninstall    --prefix DIR [--keep-assets] [--yes]  remove what the installer owns; never the data directory
  active       --prefix DIR                  this user's processes started from this installation
  watch-run    --guard-pidfile FILE --out DIR --record FILE   record what descends from a run's guard
  after-guard  --record FILE --data-dir DIR --verdict FILE --stop-script FILE   nothing of a run is left running
  stop-run     --observe-root DIR --stop-script FILE          stop.sh: ask the latest observation to stop
  configure    --package DIR [--set KEY=VALUE ...]   show the settings; change one only when asked to
  fetch-assets --package DIR [--manifest FILE] [--only ID]   download the listed files and check them
  verify-assets --package DIR [--manifest FILE]      check every listed asset that is a file; no download
  asset        --package DIR ID FIELD        print one field of one asset (for the build scripts)

The device-artifacts record (what this device's own files are, for experimental validation runs) is lib/artifacts.py's;
this file only notes a download it made itself (write_note), which that record then reports as observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "install-manifest.txt"
MANIFEST_HEAD = "sentinel-beta-install-manifest/1"
CONF_NAME = "install.conf"
DATA_MARKER = ".sentinel-beta-data"
DATA_MARKER_TEXT = "This directory holds one Sentinel personal-beta tester's own data. The installer never deletes it.\n"
OWNED_ALWAYS = ("package", CONF_NAME, MANIFEST_NAME)  # what install itself makes
OWNED_LATER = ("venv", "assets")  # what setup-env.sh, fetch-assets.sh and the build scripts make
OWNED_TRANSIENT = ("package.new", "package.old")  # only while an installation is under way
OWNED_CLEANUP = ("uninstall.sh", "uninstall-helper.py")  # only while an uninstall is unfinished, or assets are kept
OWNED = OWNED_LATER + OWNED_TRANSIENT + OWNED_ALWAYS + OWNED_CLEANUP  # every name the manifest may hold
# The tester's own settings (<data dir>/config/site.conf): where an asset is, when it is not in <prefix>/assets.
SITE_KEYS = ("ENGINE", "LLAMA_SERVER", "SCENE_MODEL", "SCENE_MMPROJ", "DEEPFACE_HOME")
SITE_TEMPLATE = """# Your own settings for the Sentinel personal beta. The installer never overwrites this file.
# Each line is KEY=absolute path, read as plain text (nothing here is executed). Leave a key out to use the file that
# fetch-assets.sh or the build scripts put under <prefix>/assets. Change one with: bash configure.sh --set KEY=/path
#   ENGINE         the TensorRT detector engine built on THIS device (build-engine.sh)
#   LLAMA_SERVER   the llama-server binary built on THIS device at the pinned commit (build-llama.sh)
#   SCENE_MODEL    the scene model file          SCENE_MMPROJ   its own projector
#   DEEPFACE_HOME  the directory that holds .deepface/weights/
# Never put a camera address, a password or a passphrase in this file.
"""


class Stop(Exception):
    """A refusal with its explanation; nothing was half-done silently."""


def say(text: str) -> None:
    print(text, flush=True)


# ---------------------------------------------------------------- plain KEY=VALUE files (read, never executed)

def read_conf(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text("utf-8", errors="replace").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == key and key.replace("_", "").isalnum():
            values[key] = value
    return values


def write_conf(path: Path, values: dict[str, str], head: str = "", mode: int = 0o600) -> None:
    for key, value in values.items():
        if "\n" in value or "\r" in value or "\0" in value:
            raise Stop(f"{key} holds a line break: not written")
    atomic_write(path, head + "".join(f"{key}={value}\n" for key, value in values.items()), mode)


# ---------------------------------------------------------------- paths

def absolute(text: str, what: str) -> Path:
    if not text or not os.path.isabs(os.path.expanduser(text)):
        raise Stop(f"{what} must be an absolute path (got '{text}')")
    return Path(os.path.realpath(os.path.expanduser(text)))


def inside(child: Path, parent: Path) -> bool:
    child, parent = Path(os.path.realpath(child)), Path(os.path.realpath(parent))
    return child == parent or parent in child.parents


def load_assets(manifest: Path) -> list[dict]:
    try:
        document = json.loads(manifest.read_text("utf-8"))
    except (OSError, ValueError) as error:
        raise Stop(f"the asset manifest {manifest} cannot be read ({type(error).__name__})") from None
    assets = document.get("assets") if isinstance(document, dict) else None
    if document.get("schema") != "sentinel-beta-assets/1" or not isinstance(assets, list):
        raise Stop(f"{manifest} is not a sentinel-beta-assets/1 manifest")
    for asset in assets:
        relative = str(asset.get("path", ""))
        if not asset.get("id") or not relative or os.path.isabs(relative) or ".." in Path(relative).parts:
            raise Stop(f"{manifest}: an asset has no id or an unsafe path ('{relative}')")
    return assets


def resolved_paths(package: Path) -> dict[str, str]:
    """Every path the scripts use, from the installed package's place and the two settings files."""
    package = Path(os.path.realpath(package))
    prefix = package.parent
    conf = read_conf(prefix / CONF_NAME)
    if package.name != "package" or "DATA_DIR" not in conf:
        raise Stop("run this from an installed package (<prefix>/package/); install first: bash install.sh")
    data = Path(conf["DATA_DIR"])
    assets_root = prefix / "assets"
    by_id = {asset["id"]: asset for asset in load_assets(package / "manifest" / "assets.json")}
    site = read_conf(data / "config" / "site.conf")

    def asset(key: str, asset_id: str) -> str:
        return site.get(key) or str(assets_root / by_id[asset_id]["path"])

    deepface_home = site.get("DEEPFACE_HOME") or str(assets_root / "deepface-home")
    return {
        "PREFIX": str(prefix), "PKG": str(package), "APP": str(package / "app"), "VENV": str(prefix / "venv"),
        "RT_PY": str(prefix / "venv" / "bin" / "python"), "ASSETS": str(assets_root), "DATA_DIR": str(data),
        "SITE": str(data / "config" / "site.conf"), "CONFIG": str(data / "config" / "live.yaml"),
        "IDENT": str(data / "identity"), "OBS_ROOT": str(data / "observe"),
        "PYTHON310": conf.get("PYTHON310", ""), "TENSORRT_PYTHON": conf.get("TENSORRT_PYTHON", ""),
        "ENGINE": asset("ENGINE", "detector_engine"), "LLAMA_SERVER": asset("LLAMA_SERVER", "llama_server"),
        "SCENE_MODEL": asset("SCENE_MODEL", "scene_model"), "SCENE_MMPROJ": asset("SCENE_MMPROJ", "scene_mmproj"),
        "DEEPFACE_HOME_DIR": deepface_home, "WEIGHTS_DIR": str(Path(deepface_home) / ".deepface" / "weights"),
    }


def command_paths(args: argparse.Namespace) -> int:
    for key, value in resolved_paths(Path(args.package)).items():
        if "\n" in value:
            raise Stop(f"{key} holds a line break")
        say(f"{key}={value}")
    return 0


# ---------------------------------------------------------------- what the installer owns

def atomic_write(path: Path, text: str, mode: int = 0o644) -> None:
    """Replace ``path`` with a new file holding ``text``. The temporary file is created exclusively and never
    through a link; a link at ``path`` itself is replaced as a link, so nothing is ever written through one."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_manifest(prefix: Path) -> list[str] | None:
    path = prefix / MANIFEST_NAME
    if path.is_symlink() or not path.is_file():
        return None
    lines = path.read_text("utf-8", errors="replace").splitlines()
    if not lines or lines[0] != MANIFEST_HEAD:
        raise Stop(f"{path} is not this installer's manifest; nothing was changed")
    names = [line for line in lines[1:] if line]
    for name in names:
        if name not in OWNED:
            raise Stop(f"{path} lists '{name}', which this installer never makes; nothing was changed")
    return names


def write_manifest(prefix: Path, names) -> None:
    """The durable record of what is the installer's in ``prefix``. It is written BEFORE a thing is made and AFTER a
    thing is removed, so an interruption never leaves an installer-made thing that the record does not name."""
    ordered = [name for name in OWNED if name in set(names) | {MANIFEST_NAME}]
    atomic_write(prefix / MANIFEST_NAME, MANIFEST_HEAD + "\n" + "".join(name + "\n" for name in ordered))


def remove_owned(prefix: Path, name: str) -> bool:
    """Remove <prefix>/NAME, which the manifest names. A link is removed as a link: what it points at is not the
    installer's. True if something was there."""
    target = prefix / name
    if name not in OWNED or Path(os.path.realpath(target.parent)) != prefix:
        raise Stop(f"{target} is not directly inside {prefix}; nothing more was removed")
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    else:
        return False
    return True


def package_files(source: Path) -> list[str]:
    """The files a package consists of: the ones its SHA256SUMS lists, and the two that describe it."""
    names = []
    for line in (source / "SHA256SUMS").read_text("utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if len(digest) != 64 or not name or os.path.isabs(name) or ".." in Path(name).parts:
            raise Stop("this package's SHA256SUMS has a line that is not 'checksum  relative/path'")
        names.append(name)
    return names + ["SHA256SUMS"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command_verify_package(args: argparse.Namespace) -> int:
    """The package is exactly its listed files: each is a regular file that equals its checksum, no directory or file
    in it is a link, and nothing else is in it (a stray module or a linked directory beside the checked files would
    be importable ahead of them)."""
    package = Path(os.path.realpath(args.package))
    if not (package / "SHA256SUMS").is_file() or not (package / "VERSION").is_file():
        raise Stop(f"{package} is not a complete package (SHA256SUMS or VERSION is missing)")
    listed = set(package_files(package))
    extra: list[str] = []
    for folder, folders, files in os.walk(package, followlinks=False):
        for name in folders + files:
            path = Path(folder) / name
            relative = str(path.relative_to(package))
            if path.is_symlink():
                extra.append(f"{relative} (a link)")
            elif name in files and relative not in listed:
                extra.append(relative)
    if extra:
        extra.sort()
        raise Stop(f"the package holds files that are not part of it: {', '.join(extra[:5])}"
                   f"{' ...' if len(extra) > 5 else ''}. Remove them, or extract the archive again and reinstall")
    for line in (package / "SHA256SUMS").read_text("utf-8").splitlines():
        digest, _, name = line.partition("  ")
        target = package / name
        if not target.is_file() or file_sha256(target) != digest:
            raise Stop(f"{name} is missing or differs from its checksum. Extract the archive again and reinstall; never "
                       "edit the package's files")
    return 0


DATA_FOLDERS = ("config", "identity", os.path.join("identity", "inbox"), "observe")


def check_data_directory(data: Path) -> None:
    """Refusals only, nothing changed: the data directory is the installer's to use, and nothing in it that the
    installer would write into is a link or something other than a directory."""
    if os.path.lexists(data):
        if not data.is_dir():
            raise Stop(f"{data} is not a directory. Choose another --data-dir (nothing was changed)")
        if any(data.iterdir()) and not (data / DATA_MARKER).is_file():
            raise Stop(f"{data} already holds something that this installer did not make: it is never reused or "
                       "written into. Choose another --data-dir (nothing was changed)")
    for name in DATA_FOLDERS:
        folder = data / name
        if os.path.lexists(folder) and (folder.is_symlink() or not folder.is_dir()):
            raise Stop(f"{folder} is a link or is not a directory: the installer writes nothing through it. Remove it "
                       "yourself (what it points at is left alone), or choose another --data-dir (nothing was changed)")


def prepare_data_directory(data: Path, templates: dict[str, str]) -> bool:
    """The marker first, then the folders, then the two default settings files where none is there. A file or a
    link that is already there is the tester's: it is kept and never written through. True if the directory is new."""
    created = not os.path.lexists(data)
    if created:
        os.makedirs(data, mode=0o700)
        os.chmod(data, 0o700)
    marker = data / DATA_MARKER
    if not os.path.lexists(marker):
        atomic_write(marker, DATA_MARKER_TEXT)  # before anything else: a later attempt then knows the directory
    for name in DATA_FOLDERS:
        folder = data / name
        if not os.path.lexists(folder):
            os.mkdir(folder, 0o700)
            os.chmod(folder, 0o700)
    for name, template in templates.items():
        target = data / "config" / name
        if target.is_symlink():
            say(f"kept: {target} is a link you made; the installer does not write through it")
        elif os.path.lexists(target):
            say(f"kept: {target} (your configuration is never overwritten)")
        else:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(template)
            say(f"wrote: {target}")
    return created


def command_install(args: argparse.Namespace) -> int:
    """One recoverable transaction. Every refusal comes first and changes nothing. Then the ownership record is
    written before each thing it names is made, the new package is copied beside the installed one and checked, and
    only then takes its place: an interruption at any point leaves a state that running install.sh again completes,
    with the previous package still usable until the new one is in place and the tester's data untouched."""
    source = Path(os.path.realpath(args.source))
    prefix = absolute(args.prefix, "--prefix")
    home = Path(os.path.realpath(Path.home()))
    if prefix in (Path("/"), home) or inside(source, prefix):
        raise Stop(f"--prefix {prefix} is the home directory, the file system's root or holds the extracted archive: "
                   "choose a directory of its own, for example ~/sentinel-beta")
    existing = read_manifest(prefix) if prefix.is_dir() else None
    if os.path.lexists(prefix) and existing is None and (not prefix.is_dir() or any(prefix.iterdir())):
        raise Stop(f"{prefix} exists and was not made by this installer: choose another --prefix (nothing was changed)")
    owned = set(existing or [])
    for name in ("package", "package.new", "package.old", CONF_NAME):
        if os.path.lexists(prefix / name) and name not in owned:
            raise Stop(f"{prefix / name} exists and was not made by this installer: it is neither used nor removed. "
                       "Move it away yourself, then install again (nothing was changed)")
    previous = read_conf(prefix / CONF_NAME).get("DATA_DIR") if CONF_NAME in owned else None
    data = absolute(args.data_dir or previous or str(home / "sentinel-beta-data"), "--data-dir")
    if previous and Path(previous) != data:
        raise Stop(f"{prefix} is installed with the data directory {previous}; installing again with another one would "
                   f"leave that data behind. Use --data-dir {previous}, or uninstall first (nothing was changed)")
    if data in (Path("/"), home) or inside(data, prefix) or inside(prefix, data) or inside(source, data):
        raise Stop(f"--data-dir {data} must be a directory of its own, outside {prefix} (nothing was changed)")
    check_data_directory(data)
    files = package_files(source)
    templates = {"live.yaml": (source / "templates" / "live.yaml").read_text("utf-8"), "site.conf": SITE_TEMPLATE}

    # ---- the tester's data directory: only ever added to ----
    created = prepare_data_directory(data, templates)

    # ---- the package: recorded as the installer's, staged, checked, then put in place ----
    prefix.mkdir(parents=True, exist_ok=True)
    write_manifest(prefix, owned | {"package.new"})
    remove_owned(prefix, "package.new")  # only ever a leftover of an interrupted installation: the record names it
    staged = prefix / "package.new"
    os.mkdir(staged)
    for name in files:
        target = staged / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        os.chmod(target, 0o755 if os.access(source / name, os.X_OK) else 0o644)
    for line in (staged / "SHA256SUMS").read_text("utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if file_sha256(staged / name) != digest:
            raise Stop(f"{name} did not copy intact; run install.sh again (the installed package, if any, is unchanged)")
    installed = prefix / "package"
    write_manifest(prefix, owned | {"package.new", "package.old", "package"})
    remove_owned(prefix, "package.old")
    if os.path.lexists(installed):
        os.rename(installed, prefix / "package.old")  # the previous copy stays whole until the new one is in place
    os.rename(staged, installed)
    conf = {"DATA_DIR": str(data), "PYTHON310": args.python310, "TENSORRT_PYTHON": args.tensorrt_python or "",
            "PACKAGE_VERSION": (installed / "VERSION").read_text("utf-8").splitlines()[0]}
    write_manifest(prefix, owned | {"package.old", "package", CONF_NAME})
    for key, value in conf.items():
        if "\n" in value or "\r" in value:
            raise Stop(f"{key} holds a line break: not written")
    atomic_write(prefix / CONF_NAME, "# Written by install.sh; rewritten by every installation. Your own settings are "
                 "in the data directory.\n" + "".join(f"{key}={value}\n" for key, value in conf.items()))
    remove_owned(prefix, "package.old")
    for name in ("uninstall.sh", "uninstall-helper.py"):  # a cleanup entry left by an earlier "--keep-assets"
        if name in owned:
            remove_owned(prefix, name)
    write_manifest(prefix, (owned - set(OWNED_TRANSIENT) - set(OWNED_CLEANUP)) | set(OWNED_ALWAYS))

    if (data / "identity" / "gallery.sealed").exists():
        say(f"kept: {data / 'identity'} (an enrollment is there; the installer never reads or changes it)")
    say(f"installed: {installed}")
    say(f"data directory: {data}" + (" (created)" if created else " (kept)"))
    return 0


def command_own(args: argparse.Namespace) -> int:
    prefix = absolute(args.prefix, "--prefix")
    names = read_manifest(prefix)
    if names is None:
        raise Stop(f"{prefix} holds no installation")
    if args.name not in OWNED_LATER:
        raise Stop(f"'{args.name}' is not something the installer makes")
    if os.path.lexists(prefix / args.name) and args.name not in names:
        raise Stop(f"{prefix / args.name} exists and was not made by this installer: it is neither used nor removed. "
                   "Move it away yourself, then run this again")
    write_manifest(prefix, set(names) | {args.name})
    return 0


CLEANUP_SCRIPT = """#!/bin/bash
# Left here by the Sentinel personal-beta uninstaller. It removes what is still the installer's in this directory:
# the assets you asked it to keep, or the rest of an uninstall that was interrupted. It removes nothing else, and
# never your data directory.   bash uninstall.sh [--keep-assets] [--yes]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
[ "$(id -u)" != 0 ] || { echo "STOP: run as your own user, without sudo" >&2; exit 1; }
cd / && exec python3 -I -B "$HERE/uninstall-helper.py" uninstall --prefix "$HERE" "$@"
"""


def command_uninstall(args: argparse.Namespace) -> int:
    """Removes what the manifest names and nothing else. Restartable: a cleanup entry (uninstall.sh and its helper)
    is put in the prefix first and removed last, so an interrupted uninstall, or assets kept on purpose, can always
    be finished with `bash <prefix>/uninstall.sh`. The data directory is never touched."""
    prefix = absolute(args.prefix, "--prefix")
    names = read_manifest(prefix)
    if names is None:
        raise Stop(f"{prefix} holds no installation of this package (no {MANIFEST_NAME}); nothing was removed")
    running = active_processes(prefix)
    if running:
        raise Stop(f"something from this installation is still running (process {', '.join(map(str, running))}). "
                   f"Stop it first: bash {prefix / 'package' / 'stop.sh'} (nothing was removed)")
    for name in OWNED_CLEANUP:
        if os.path.lexists(prefix / name) and name not in names:
            raise Stop(f"{prefix / name} exists and was not made by this installer: move it away yourself, then "
                       "uninstall again (nothing was removed)")
    data = read_conf(prefix / CONF_NAME).get("DATA_DIR", "")
    keep = {"assets"} if args.keep_assets and "assets" in names else set()
    say(f"This removes what the installer made in {prefix}: "
        + ", ".join(name for name in OWNED_LATER + ("package", CONF_NAME) if name in names and name not in keep)
        + (f". It keeps {prefix / 'assets'}" if keep else "") + ".")
    say(f"It does NOT remove your data{(' at ' + data) if data else ''}, or anything else in {prefix}.")
    if not args.yes:
        say("type UNINSTALL to remove it, or press Enter to keep everything: ")
        if sys.stdin.readline().strip() != "UNINSTALL":
            raise Stop("nothing was removed")

    helper = Path(__file__).read_bytes()
    owned = set(names) | set(OWNED_CLEANUP)
    write_manifest(prefix, owned)
    for name, content, mode in (("uninstall-helper.py", helper, 0o644), ("uninstall.sh", CLEANUP_SCRIPT.encode(), 0o755)):
        if Path(os.path.realpath(__file__)) != prefix / name:
            remove_owned(prefix, name)
            descriptor = os.open(prefix / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
    for name in OWNED_LATER + OWNED_TRANSIENT + ("package", CONF_NAME):
        if name not in owned or name in keep:
            continue
        if remove_owned(prefix, name):
            say(f"removed: {prefix / name}")
        owned.discard(name)
        write_manifest(prefix, owned)
    if keep:
        say(f"kept at your request: {prefix / 'assets'}")
        say(f"To remove it later: bash {prefix / 'uninstall.sh'}")
    else:
        for name in OWNED_CLEANUP:
            remove_owned(prefix, name)
        (prefix / MANIFEST_NAME).unlink()
    ours = set(OWNED_CLEANUP) | {"assets", MANIFEST_NAME} if keep else set()
    left = sorted(p.name for p in prefix.iterdir() if p.name not in ours)
    if left:
        say(f"left in place, not made by this installer: {', '.join(str(prefix / name) for name in left)}")
    elif not keep:
        prefix.rmdir()
        say(f"removed: {prefix}")
    if data:
        say(f"NOT removed, yours to keep or delete: {data} (your configuration, your enrollment and your run records)")
    return 0


# ---------------------------------------------------------------- settings

def command_configure(args: argparse.Namespace) -> int:
    paths = resolved_paths(Path(args.package))
    site = Path(paths["SITE"])
    if args.set:
        if site.is_symlink():
            raise Stop(f"{site} is a link you made: it is not written through or replaced. Edit the file it points at "
                       "yourself (KEY=/absolute/path lines)")
        values = read_conf(site)
        for item in args.set:
            key, separator, value = item.partition("=")
            if not separator or key not in SITE_KEYS:
                raise Stop(f"--set takes KEY=/absolute/path with KEY one of: {', '.join(SITE_KEYS)}")
            if value:
                values[key] = str(absolute(value, key))
            else:
                values.pop(key, None)  # KEY= goes back to the file under <prefix>/assets
        write_conf(site, {key: values[key] for key in SITE_KEYS if key in values}, SITE_TEMPLATE)
        paths = resolved_paths(Path(args.package))
    say(f"configuration file: {paths['CONFIG']}")
    say(f"your settings:      {paths['SITE']}")
    for key in ("ENGINE", "LLAMA_SERVER", "SCENE_MODEL", "SCENE_MMPROJ", "WEIGHTS_DIR", "IDENT"):
        there = "present" if Path(paths[key]).exists() else "MISSING"
        say(f"  {key:<13} {paths[key]}  [{there}]")
    enrolled = (Path(paths["IDENT"]) / "gallery.sealed").exists()
    say(f"  enrollment    {'present' if enrolled else 'MISSING: enroll yourself first (QUICKSTART.md, Enrollment)'}")
    return 0


# ---------------------------------------------------------------- assets

ASSET_SETTINGS = {"scene_model": "SCENE_MODEL", "scene_mmproj": "SCENE_MMPROJ", "llama_server": "LLAMA_SERVER",
                  "detector_engine": "ENGINE"}


def asset_target(paths: dict[str, str], asset: dict) -> Path:
    """Where an asset is: the tester's own setting when one names it, else its place under <prefix>/assets."""
    own = read_conf(Path(paths["SITE"])).get(ASSET_SETTINGS.get(asset["id"], ""))
    if own:
        return Path(own)
    relative = Path(asset["path"])
    if relative.parts[:1] == ("deepface-home",):
        return Path(paths["DEEPFACE_HOME_DIR"]).joinpath(*relative.parts[1:])
    return Path(paths["ASSETS"]) / relative


def check_file(asset: dict, target: Path) -> str | None:
    """None when the file is the one the manifest names; else what is wrong with it."""
    if not target.is_file():
        return "MISSING"
    if asset.get("size") is not None and target.stat().st_size != int(asset["size"]):
        return f"WRONG SIZE ({target.stat().st_size} bytes, expected {asset['size']})"
    if asset.get("sha256") and file_sha256(target) != asset["sha256"]:
        return "WRONG CHECKSUM"
    return None


def write_note(paths: dict[str, str], asset_id: str, note: dict) -> None:
    """What a step of this package observed itself about an asset it has just made or downloaded, kept under
    <prefix>/assets/provenance/ for the device-artifacts record (lib/artifacts.py). A note is only ever written by
    the step that did the work, and names the SHA-256 of what it produced: a file with other bytes is not its."""
    folder = Path(paths["ASSETS"]) / "provenance"
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write(folder / f"{asset_id}.json", json.dumps(note, indent=1, sort_keys=True) + "\n", 0o600)


def download(url: str, target: Path, token: str | None) -> None:
    request = urllib.request.Request(url)
    if token and url.startswith("https://huggingface.co/"):
        request.add_header("Authorization", f"Bearer {token}")  # the tester's own; never printed or stored
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle, 1 << 20)


def command_fetch(args: argparse.Namespace) -> int:
    paths = resolved_paths(Path(args.package))
    manifest = Path(args.manifest) if args.manifest else Path(paths["PKG"]) / "manifest" / "assets.json"
    failed = 0
    for asset in load_assets(manifest):
        if asset.get("kind") != "download" or (args.only and asset["id"] != args.only):
            continue
        target = asset_target(paths, asset)
        if check_file(asset, target) is None:
            say(f"asset {asset['id']}: ok, already there ({target})")
            continue
        if target.exists():
            say(f"asset {asset['id']}: {check_file(asset, target)} at {target}: it is NOT replaced. Move it away yourself, "
                "then run this again")
            failed += 1
            continue
        if not inside(target, Path(paths["PREFIX"])):
            say(f"asset {asset['id']}: MISSING at {target}, a place you set yourself: nothing is downloaded outside the "
                "installation. Put the file there, or clear the setting (configure.sh --set KEY=)")
            failed += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        say(f"asset {asset['id']}: downloading {asset['url']}")
        try:
            download(asset["url"], part, os.environ.get("HF_TOKEN"))
        except Exception as error:  # noqa: BLE001 - every failure is reported the same way, without the request's details
            part.unlink(missing_ok=True)
            say(f"asset {asset['id']}: the download failed ({type(error).__name__}). Check the network and the address in "
                f"{manifest}, then run this again")
            failed += 1
            continue
        problem = check_file(asset, part)
        if problem is not None:
            part.unlink()
            say(f"asset {asset['id']}: the downloaded file is not the pinned one ({problem}); it was deleted. The upstream "
                "file may have changed: do not use another file, report it")
            failed += 1
            continue
        os.replace(part, target)
        try:  # this step downloaded it from the pinned address and it equals the pinned checksum: noted as observed
            write_note(paths, asset["id"], {
                "observed": "downloaded_by_package_step", "step": "fetch-assets.sh", "url": asset["url"],
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sha256": asset.get("sha256"),
                "bytes": target.stat().st_size, "target": str(target),  # this very copy: its place and its time
                "mtime_utc": datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")})
        except OSError as error:
            say(f"asset {asset['id']}: the download could not be noted ({type(error).__name__}); a record will say "
                "that the file equals the pinned checksum, not that this package downloaded it")
        say(f"asset {asset['id']}: ok ({target})")
    return 1 if failed else 0


def command_verify(args: argparse.Namespace) -> int:
    paths = resolved_paths(Path(args.package))
    manifest = Path(args.manifest) if args.manifest else Path(paths["PKG"]) / "manifest" / "assets.json"
    failed = 0
    for asset in load_assets(manifest):
        target = asset_target(paths, asset)
        if asset.get("kind") == "download":
            problem = check_file(asset, target)
        else:  # built on this device: there is no checksum to compare; the scripts that use it check it further
            problem = None if target.is_file() else "MISSING"
        if problem is None:
            say(f"asset {asset['id']}: ok")
        else:
            failed += 1
            say(f"asset {asset['id']}: {problem} ({target}). {asset.get('how', '')}".rstrip())
    return 1 if failed else 0


def command_asset(args: argparse.Namespace) -> int:
    paths = resolved_paths(Path(args.package))
    for asset in load_assets(Path(paths["PKG"]) / "manifest" / "assets.json"):
        if asset["id"] == args.id:
            value = str(asset_target(paths, asset)) if args.field == "target" else asset.get(args.field)
            if value is None:
                raise Stop(f"asset {args.id} has no '{args.field}'")
            say(" ".join(value) if isinstance(value, list) else json.dumps(value, sort_keys=True)
                if isinstance(value, dict) else str(value))
            return 0
    raise Stop(f"no asset '{args.id}' in the manifest")


# ---------------------------------------------------------------- processes: this user's own, by exact identity
#
# No pattern is ever matched against a command line (a path may hold "+", "[" or a space, and one path may be the
# beginning of another). A process is identified by its number together with its start time, its owner, and its
# arguments compared whole; it is signalled through a handle on the process itself, after that identity is checked
# again. Whether a process belongs to a run is known from the record made while the run was alive (it descended from
# the run's guard), never from its name or its port.

PROC = Path("/proc")


def process(pid: int) -> dict | None:
    """One process of this user as /proc shows it now; None if it has gone or is another user's."""
    base = PROC / str(pid)
    try:
        if os.stat(base).st_uid != os.getuid():
            return None
        stat = (base / "stat").read_text("utf-8", errors="replace")
        fields = stat[stat.rindex(")") + 2:].split()
        argv = [part.decode("utf-8", "replace") for part in (base / "cmdline").read_bytes().split(b"\0") if part]
    except (OSError, ValueError):
        return None
    try:
        exe = os.readlink(base / "exe")
    except OSError:
        exe = ""
    try:
        return {"pid": pid, "state": fields[0], "ppid": int(fields[1]), "start": int(fields[19]), "argv": argv,
                "exe": exe, "comm": stat[stat.index("(") + 1:stat.rindex(")")]}
    except (ValueError, IndexError):
        return None


def processes() -> list[dict]:
    found = []
    for entry in os.listdir(PROC):
        if entry.isdigit():
            info = process(int(entry))
            if info is not None and info["state"] != "Z":
                found.append(info)
    return found


def same(info: dict) -> bool:
    """Still that process: the number now belongs to a process with the same start time (not a later one)."""
    now = process(info["pid"])
    return now is not None and now["state"] != "Z" and now["start"] == info["start"]


def has_sequence(argv: list[str], wanted: tuple[str, ...]) -> bool:
    return any(tuple(argv[index:index + len(wanted)]) == wanted for index in range(len(argv) - len(wanted) + 1))


def is_runtime(info: dict, data_dir: str) -> bool:
    """`sentinel run` started for exactly this data directory."""
    return has_sequence(info["argv"], ("-m", "sentinel.cli", "run")) and has_sequence(info["argv"], ("--data-dir", data_dir))


def is_guard(info: dict, out: str) -> bool:
    """The memory guard started for exactly this output directory."""
    return (any(part == "step5_guard.py" or part.endswith("/step5_guard.py") for part in info["argv"])
            and has_sequence(info["argv"], ("--out", out)))


def send_signal(info: dict, signum: int) -> bool:
    """Signal exactly the process ``info`` describes; False if it has gone (a later process with its number is
    never signalled)."""
    handle = None
    try:
        handle = os.pidfd_open(info["pid"])  # Linux 5.3 and Python 3.9: the handle cannot come to mean another process
    except (AttributeError, OSError):
        handle = None
    try:
        if not same(info):
            return False
        if handle is not None:
            signal.pidfd_send_signal(handle, signum)
        else:
            os.kill(info["pid"], signum)
        return True
    except OSError:
        return False
    finally:
        if handle is not None:
            os.close(handle)


def wait_gone(group: list[dict], seconds: float) -> list[dict]:
    """Those of ``group`` still running after at most ``seconds``."""
    deadline = time.monotonic() + seconds
    left = [info for info in group if same(info)]
    while left and time.monotonic() < deadline:
        time.sleep(0.2)
        left = [info for info in left if same(info)]
    return left


def numbers(group: list[dict]) -> str:
    return " ".join(str(info["pid"]) for info in group)


def ancestors() -> set[int]:
    found, pid = set(), os.getpid()
    while pid > 1 and pid not in found:
        found.add(pid)
        info = process(pid)
        pid = info["ppid"] if info else 0
    return found


def active_processes(prefix: Path) -> list[int]:
    """This user's processes started from this installation: an argument names a file inside the prefix (its Python
    environment's interpreter, its scripts, its llama-server). Compared as paths; this command and the shells that
    started it are left out."""
    root = str(prefix) + os.sep
    mine = ancestors()
    return sorted(info["pid"] for info in processes()
                  if info["pid"] not in mine and (any(part.startswith(root) for part in info["argv"])
                                                  or info["exe"].startswith(root)))


def command_active(args: argparse.Namespace) -> int:
    for pid in active_processes(absolute(args.prefix, "--prefix")):
        say(str(pid))
    return 0


def read_record(record: Path) -> list[dict]:
    """The processes the run's own record names that are still the same processes now."""
    alive = []
    try:
        lines = record.read_text("utf-8", errors="replace").splitlines()
    except OSError:
        return alive
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            info = process(int(parts[0]))
            if info is not None and info["state"] != "Z" and info["start"] == int(parts[1]):
                alive.append(info)
    return alive


def command_watch(args: argparse.Namespace) -> int:
    """While a run is alive, records every process that descends from its guard (number, start time, name): the
    runtime and the model server it starts. That record is the only proof of what a run started, since both start
    sessions of their own. Ends when the guard has gone."""
    record, seen, guard = Path(args.record), set(), None
    deadline = time.monotonic() + args.max_s
    while time.monotonic() < deadline:
        if guard is None:
            try:
                info = process(int(Path(args.guard_pidfile).read_text().strip()))
            except (OSError, ValueError):
                info = None
            if info is not None and is_guard(info, args.out):
                guard = info
        elif not same(guard):
            return 0
        if guard is not None:
            children: dict[int, list[dict]] = {}
            for info in processes():
                children.setdefault(info["ppid"], []).append(info)
            stack = [guard["pid"]]
            while stack:
                for child in children.get(stack.pop(), []):
                    stack.append(child["pid"])
                    if (child["pid"], child["start"]) not in seen:
                        seen.add((child["pid"], child["start"]))
                        descriptor = os.open(record, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
                        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                            handle.write(f"{child['pid']} {child['start']} {child['comm']}\n")
        time.sleep(args.interval)
    return 0


def command_after_guard(args: argparse.Namespace) -> int:
    """After the guard has ended: nothing this run started may be left running without it. The run's runtime is asked
    once to stop cleanly; a model server that this run started is stopped once the runtime has gone. A process this
    run did not start is reported and never signalled. Writes RESIDUE=... to the verdict file."""
    owned = read_record(Path(args.record))
    keys = {(info["pid"], info["start"]) for info in owned}
    runtimes = [info for info in owned if is_runtime(info, args.data_dir)]
    for info in processes():  # its arguments name this run's own data directory: this run's, recorded or not
        if is_runtime(info, args.data_dir) and (info["pid"], info["start"]) not in {(r["pid"], r["start"]) for r in runtimes}:
            runtimes.append(info)
    servers = [info for info in owned if info["comm"] == "llama-server"]
    others = [info for info in processes() if info["comm"] == "llama-server" and (info["pid"], info["start"]) not in keys]
    residue = "none"
    if runtimes or servers:
        print("\a", end="", flush=True)
        if runtimes:
            say("WARNING: the guard has ended but this run's runtime is still running; the camera may still be open. "
                "Asking it to stop cleanly, once.")
            for info in runtimes:
                send_signal(info, signal.SIGINT)
            runtimes = wait_gone(runtimes, args.runtime_wait)
        if not runtimes and servers:
            say("A llama-server that this run started is still running after the runtime ended; stopping it.")
            for info in servers:
                send_signal(info, signal.SIGTERM)
            servers = wait_gone(servers, args.server_wait)
        if runtimes:
            residue = "still_running"
            say(f"STILL RUNNING: runtime [{numbers(runtimes)}]. The camera may still be open. Stop it with: bash {args.stop_script}")
        elif servers:
            residue = "still_running"
            say(f"STILL RUNNING: llama-server [{numbers(servers)}]; the runtime has gone, so the camera is closed. "
                f"Stop it with: kill -TERM {numbers(servers)}")
        else:
            residue = "stopped_after_the_guard"
            say("It has stopped.")
    if others:
        say(f"NOTE: another llama-server of yours is running [{numbers(others)}]. This run's record does not name it, so "
            "it is not this run's as far as can be shown: it was not signalled.")
    atomic_write(Path(args.verdict), f"RESIDUE={residue}\n", 0o600)
    return 0


def command_stop(args: argparse.Namespace) -> int:
    """stop.sh: the same request Ctrl-C makes, to the latest observation only. Signals the guard recorded for it, if
    that process is still that guard; else a runtime started for that observation's own data directory. With nothing
    running it signals nothing. It always ends with what is recorded about the desktop."""
    root = Path(args.observe_root)
    folders = sorted(path for path in root.glob("obs-*") if path.is_dir()) if root.is_dir() else []
    if not folders:
        say("No observation folder exists; nothing was signalled.")
        return 0
    latest = folders[-1]
    code, guard = 0, None
    try:
        info = process(int((latest / "guard.pid").read_text().strip()))
    except (OSError, ValueError):
        info = None
    if info is not None and is_guard(info, str(latest / "guard")):
        guard = info
    if guard is not None:
        if not send_signal(guard, signal.SIGINT):
            say("STOP: the signal could not be sent.")
            return 1
        say(f"Asked {latest} to stop; waiting for the clean stop (up to about 2 minutes).")
        if wait_gone([guard], args.guard_wait):
            say(f"It is still stopping after {args.guard_wait:g} s; the guard forces the stop itself at its limit. "
                f"Check again with: bash {args.stop_script}")
            return 1
        say("The guard has stopped.")
    runtimes = [info for info in processes() if is_runtime(info, str(latest / "db"))]
    if runtimes:
        say("A runtime for this observation is still running without its guard; the camera may still be open. "
            "Asking it to stop cleanly, once.")
        for info in runtimes:
            send_signal(info, signal.SIGINT)
        runtimes = wait_gone(runtimes, args.runtime_wait)
        if runtimes:
            say(f"STOP: it did not stop within {args.runtime_wait:g} s. Still running: {numbers(runtimes)}. "
                f"Force it with: kill -TERM {numbers(runtimes)}")
            code = 1
        else:
            say("The runtime has stopped.")
    elif guard is None:
        servers = [info for info in read_record(latest / "processes.txt") if info["comm"] == "llama-server"]
        if servers:
            say(f"No observation is running, but a llama-server that the last run started is ({numbers(servers)}). "
                f"Stop it with: kill -TERM {numbers(servers)}")
        else:
            say("No observation is running; nothing was signalled.")
    # the desktop: what the last run recorded, so that the advice survives a lost launch window
    try:
        desktop = (latest / "desktop.txt").read_text("utf-8", errors="replace")
    except OSError:
        desktop = ""
    if "stopped_by_launcher" in desktop and "restart=ok" not in desktop:
        try:
            state = subprocess.run(["systemctl", "is-active", "display-manager"], capture_output=True, text=True,
                                   timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            state = "unknown"
        if state != "active":
            say("The desktop was stopped for this observation and its restart is not recorded as done"
                + (" (it failed)" if "restart=FAILED" in desktop else "")
                + ". When nothing of the run is left running, restart it with: sudo systemctl start display-manager")
    say(f"Records: {latest}")
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentinel personal-beta package bookkeeping (see the module docstring).")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("paths", "verify-package"):
        commands.add_parser(name).add_argument("--package", required=True)
    install = commands.add_parser("install")
    install.add_argument("--source", required=True)
    install.add_argument("--prefix", required=True)
    install.add_argument("--data-dir", default="")
    install.add_argument("--python310", required=True)
    install.add_argument("--tensorrt-python", default="")
    own = commands.add_parser("own")
    own.add_argument("--prefix", required=True)
    own.add_argument("name")
    uninstall = commands.add_parser("uninstall")
    uninstall.add_argument("--prefix", required=True)
    uninstall.add_argument("--keep-assets", action="store_true")
    uninstall.add_argument("--yes", action="store_true")
    commands.add_parser("active").add_argument("--prefix", required=True)
    watch = commands.add_parser("watch-run")
    watch.add_argument("--guard-pidfile", required=True)
    watch.add_argument("--out", required=True)
    watch.add_argument("--record", required=True)
    watch.add_argument("--interval", type=float, default=0.5)
    watch.add_argument("--max-s", type=float, default=4500.0)
    after = commands.add_parser("after-guard")
    after.add_argument("--record", required=True)
    after.add_argument("--data-dir", required=True)
    after.add_argument("--verdict", required=True)
    after.add_argument("--stop-script", required=True)
    after.add_argument("--runtime-wait", type=float, default=100.0)
    after.add_argument("--server-wait", type=float, default=15.0)
    stop = commands.add_parser("stop-run")
    stop.add_argument("--observe-root", required=True)
    stop.add_argument("--stop-script", required=True)
    stop.add_argument("--guard-wait", type=float, default=150.0)
    stop.add_argument("--runtime-wait", type=float, default=100.0)
    configure = commands.add_parser("configure")
    configure.add_argument("--package", required=True)
    configure.add_argument("--set", action="append", default=[])
    for name in ("fetch-assets", "verify-assets"):
        sub = commands.add_parser(name)
        sub.add_argument("--package", required=True)
        sub.add_argument("--manifest", default="")
        if name == "fetch-assets":
            sub.add_argument("--only", default="")
    asset = commands.add_parser("asset")
    asset.add_argument("--package", required=True)
    asset.add_argument("id")
    asset.add_argument("field")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"paths": command_paths, "verify-package": command_verify_package, "install": command_install, "own": command_own, "uninstall": command_uninstall,
                "active": command_active, "watch-run": command_watch, "after-guard": command_after_guard,
                "stop-run": command_stop, "configure": command_configure, "fetch-assets": command_fetch, "verify-assets": command_verify,
                "asset": command_asset}
    try:
        return handlers[args.command](args)
    except Stop as stop:
        print(f"STOP: {stop}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
