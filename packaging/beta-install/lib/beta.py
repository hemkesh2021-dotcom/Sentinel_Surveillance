#!/usr/bin/env python3
"""The personal-beta package's bookkeeping: where things are, what the installer owns, the asset manifest.

Standard library only; any Python 3.8 or newer. It never opens the camera, loads a model, reads a credential or
touches anything outside the prefix and the data directory it is given. Sub-commands (the shell scripts call them):

  verify-package --package DIR               every listed file equals its checksum and no other file is there
  paths        --package DIR                 print every path the scripts use, one KEY=VALUE per line
  install      --source DIR --prefix DIR --data-dir DIR --python310 FILE [--tensorrt-python DIR]
  own          --prefix DIR NAME             record that the installer made <prefix>/NAME (venv, assets)
  uninstall    --prefix DIR [--keep-assets]  remove what the installer owns; the data directory is never touched
  configure    --package DIR [--set KEY=VALUE ...]   show the settings; change one only when asked to
  fetch-assets --package DIR [--manifest FILE] [--only ID]   download the listed files and check them
  verify-assets --package DIR [--manifest FILE]      check every listed asset that is a file; no download
  asset        --package DIR ID FIELD        print one field of one asset (for the build scripts)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

MANIFEST_NAME = "install-manifest.txt"
MANIFEST_HEAD = "sentinel-beta-install-manifest/1"
CONF_NAME = "install.conf"
DATA_MARKER = ".sentinel-beta-data"
DATA_MARKER_TEXT = "This directory holds one Sentinel personal-beta tester's own data. The installer never deletes it.\n"
OWNED_ALWAYS = ("package", CONF_NAME, MANIFEST_NAME)  # what install itself makes
OWNED_LATER = ("venv", "assets")  # what setup-env.sh, fetch-assets.sh and the build scripts make
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


def write_conf(path: Path, values: dict[str, str], head: str = "") -> None:
    for key, value in values.items():
        if "\n" in value or "\r" in value or "\0" in value:
            raise Stop(f"{key} holds a line break: not written")
    text = head + "".join(f"{key}={value}\n" for key, value in values.items())
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(text, "utf-8")
    os.replace(temporary, path)


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

def read_manifest(prefix: Path) -> list[str] | None:
    path = prefix / MANIFEST_NAME
    if not path.is_file():
        return None
    lines = path.read_text("utf-8", errors="replace").splitlines()
    if not lines or lines[0] != MANIFEST_HEAD:
        raise Stop(f"{path} is not this installer's manifest; nothing was changed")
    names = [line for line in lines[1:] if line]
    for name in names:
        if name not in OWNED_ALWAYS + OWNED_LATER:
            raise Stop(f"{path} lists '{name}', which this installer never makes; nothing was changed")
    return names


def write_manifest(prefix: Path, names: list[str]) -> None:
    ordered = [name for name in OWNED_ALWAYS + OWNED_LATER if name in names]
    temporary = prefix / (MANIFEST_NAME + ".new")
    temporary.write_text(MANIFEST_HEAD + "\n" + "".join(name + "\n" for name in ordered), "utf-8")
    os.replace(temporary, prefix / MANIFEST_NAME)


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
    """The package is exactly its listed files: each equals its checksum, and nothing else is in it (a stray module
    beside the scripts or in the application tree would be importable ahead of the checked ones)."""
    package = Path(os.path.realpath(args.package))
    if not (package / "SHA256SUMS").is_file() or not (package / "VERSION").is_file():
        raise Stop(f"{package} is not a complete package (SHA256SUMS or VERSION is missing)")
    listed = set(package_files(package))
    for line in (package / "SHA256SUMS").read_text("utf-8").splitlines():
        digest, _, name = line.partition("  ")
        target = package / name
        if target.is_symlink() or not target.is_file() or file_sha256(target) != digest:
            raise Stop(f"{name} is missing or differs from its checksum. Extract the archive again and reinstall; never "
                       "edit the package's files")
    extra = sorted(str(path.relative_to(package)) for path in package.rglob("*")
                   if not path.is_dir() and str(path.relative_to(package)) not in listed)
    if extra:
        raise Stop(f"the package holds files that are not part of it: {', '.join(extra[:5])}"
                   f"{' ...' if len(extra) > 5 else ''}. Remove them, or extract the archive again and reinstall")
    return 0


def command_install(args: argparse.Namespace) -> int:
    source = Path(os.path.realpath(args.source))
    prefix = absolute(args.prefix, "--prefix")
    home = Path(os.path.realpath(Path.home()))
    for place, what in ((prefix, "--prefix"),):
        if place in (Path("/"), home) or inside(source, place):
            raise Stop(f"{what} {place} is the home directory, the file system's root or holds the extracted archive: "
                       "choose a directory of its own, for example ~/sentinel-beta")
    existing = read_manifest(prefix) if prefix.is_dir() else None
    if prefix.exists() and existing is None and (not prefix.is_dir() or any(prefix.iterdir())):
        raise Stop(f"{prefix} exists and was not made by this installer: choose another --prefix (nothing was changed)")
    previous = read_conf(prefix / CONF_NAME).get("DATA_DIR") if existing is not None else None
    data = absolute(args.data_dir or previous or str(home / "sentinel-beta-data"), "--data-dir")
    if previous and Path(previous) != data:
        raise Stop(f"{prefix} is installed with the data directory {previous}; installing again with another one would "
                   f"leave that data behind. Use --data-dir {previous}, or uninstall first (nothing was changed)")
    if data in (Path("/"), home) or inside(data, prefix) or inside(prefix, data) or inside(source, data):
        raise Stop(f"--data-dir {data} must be a directory of its own, outside {prefix} (nothing was changed)")
    if data.exists() and (not data.is_dir() or (any(data.iterdir()) and not (data / DATA_MARKER).is_file())):
        raise Stop(f"{data} already holds something that this installer did not make: it is never reused or written "
                   "into. Choose another --data-dir (nothing was changed)")

    files = package_files(source)
    prefix.mkdir(parents=True, exist_ok=True)
    staged = prefix / "package.new"
    if staged.exists():
        shutil.rmtree(staged)
    for name in files:
        target = staged / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        os.chmod(target, 0o755 if os.access(source / name, os.X_OK) else 0o644)
    for line in (staged / "SHA256SUMS").read_text("utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if file_sha256(staged / name) != digest:
            shutil.rmtree(staged)
            raise Stop(f"{name} did not copy intact; nothing was installed")
    installed = prefix / "package"
    if installed.exists():
        shutil.rmtree(installed)  # installer-owned: the previous copy of this package
    os.replace(staged, installed)

    conf = {"DATA_DIR": str(data), "PYTHON310": args.python310, "TENSORRT_PYTHON": args.tensorrt_python or "",
            "PACKAGE_VERSION": (installed / "VERSION").read_text("utf-8").splitlines()[0]}
    write_conf(prefix / CONF_NAME, conf, "# Written by install.sh; rewritten by every installation. Your own settings are in the data directory.\n")
    write_manifest(prefix, sorted(set(existing or []) | set(OWNED_ALWAYS)))

    made = []
    for folder in (data, data / "config", data / "identity", data / "identity" / "inbox", data / "observe"):
        if not folder.exists():
            folder.mkdir(parents=True)
            os.chmod(folder, 0o700)
            made.append(folder)
    if not (data / DATA_MARKER).exists():
        (data / DATA_MARKER).write_text(DATA_MARKER_TEXT, "utf-8")
    for name, template in (("live.yaml", (installed / "templates" / "live.yaml").read_text("utf-8")), ("site.conf", SITE_TEMPLATE)):
        target = data / "config" / name
        if target.exists():
            say(f"kept: {target} (your configuration is never overwritten)")
        else:
            target.write_text(template, "utf-8")
            os.chmod(target, 0o600)
            say(f"wrote: {target}")
    if (data / "identity" / "gallery.sealed").exists():
        say(f"kept: {data / 'identity'} (an enrollment is there; the installer never reads or changes it)")
    say(f"installed: {installed}")
    say(f"data directory: {data}" + (" (created)" if data in made else " (kept)"))
    return 0


def command_own(args: argparse.Namespace) -> int:
    prefix = absolute(args.prefix, "--prefix")
    names = read_manifest(prefix)
    if names is None:
        raise Stop(f"{prefix} holds no installation")
    if args.name not in OWNED_LATER:
        raise Stop(f"'{args.name}' is not something the installer makes")
    write_manifest(prefix, sorted(set(names) | {args.name}))
    return 0


def command_uninstall(args: argparse.Namespace) -> int:
    prefix = absolute(args.prefix, "--prefix")
    names = read_manifest(prefix)
    if names is None:
        raise Stop(f"{prefix} holds no installation of this package (no {MANIFEST_NAME}); nothing was removed")
    data = read_conf(prefix / CONF_NAME).get("DATA_DIR", "")
    kept = []
    for name in [n for n in OWNED_LATER + OWNED_ALWAYS if n in names]:
        if name == "assets" and args.keep_assets:
            kept.append(name)
            continue
        target = prefix / name
        if Path(os.path.realpath(target.parent)) != prefix:
            raise Stop(f"{target} is not directly inside {prefix}; nothing more was removed")
        if target.is_symlink() or target.is_file():
            target.unlink()  # a link is removed as a link: what it points at is not the installer's
        elif target.is_dir():
            shutil.rmtree(target)
        say(f"removed: {target}")
    if kept:
        write_manifest(prefix, kept + [MANIFEST_NAME])
        say(f"kept at your request: {prefix / 'assets'} (remove it by uninstalling again without --keep-assets)")
    ours = {"assets", MANIFEST_NAME} if kept else set()
    left = sorted(p.name for p in prefix.iterdir() if p.name not in ours) if prefix.is_dir() else []
    if left:
        say(f"left in place, not made by this installer: {', '.join(str(prefix / name) for name in left)}")
    elif not kept:
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
        os.chmod(site, 0o600)
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
            say(" ".join(value) if isinstance(value, list) else str(value))
            return 0
    raise Stop(f"no asset '{args.id}' in the manifest")


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)
    handlers = {"paths": command_paths, "verify-package": command_verify_package, "install": command_install, "own": command_own, "uninstall": command_uninstall,
                "configure": command_configure, "fetch-assets": command_fetch, "verify-assets": command_verify,
                "asset": command_asset}
    try:
        return handlers[args.command](args)
    except Stop as stop:
        print(f"STOP: {stop}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
