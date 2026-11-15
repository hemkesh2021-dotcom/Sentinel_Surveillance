"""The installation's own cache, settings and temporary locations (session 116, part 47).

Synthetic only: a temporary prefix and data directory, the package assembled from the working tree, no network, no
model, no camera. What a script exports is read from a shell that sources the package's own lib/common.sh exactly
as the scripts do. The scripts themselves, with what they start, are in test_beta_install_launch.py.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from beta_install_fixture import World, said, tree

VARIABLES = {"PIP_CACHE_DIR": "cache/pip", "XDG_CACHE_HOME": "cache/xdg-cache", "XDG_CONFIG_HOME": "cache/xdg-config",
             "XDG_DATA_HOME": "cache/xdg-data", "HF_HOME": "cache/hf", "TORCH_HOME": "cache/torch",
             "YOLO_CONFIG_DIR": "cache/yolo", "DEEPFACE_HOME": "cache/deepface-home", "CUDA_CACHE_PATH": "cache/cuda",
             "KERAS_HOME": "cache/keras", "MPLCONFIGDIR": "cache/matplotlib", "TMPDIR": "tmp"}
REMOVED = ("PYTHONPYCACHEPREFIX", "TEMP", "TMP", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
           "PYTORCH_KERNEL_CACHE_PATH")
PROBE = 'set -uo pipefail; . "$1/lib/common.sh"; beta_load_paths "$1"; beta_local_env "$1"; env -0'


@pytest.fixture
def world(tmp_path: Path) -> World:
    world = World(tmp_path)
    installed = world.install()
    assert installed.returncode == 0, said(installed)
    return world


def beta(world: World, *args: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(world.bin / "python3"), "-I", "-B", str(world.installed / "lib/beta.py"), *args],
                          env={**world.env, **env}, capture_output=True, text=True, timeout=60)


def exported(world: World, package: Path | None = None, **env: str) -> tuple[subprocess.CompletedProcess, dict[str, str]]:
    """The environment a script has after the package's own two calls, as everything it starts would get it."""
    result = subprocess.run(["bash", "-c", PROBE, "probe", str(package or world.installed)], env={**world.env, **env},
                            capture_output=True, timeout=60, cwd=world.tmp)
    pairs = dict(item.split("=", 1) for item in result.stdout.decode().split("\0") if "=" in item)
    return result, pairs


def manifest(world: World) -> list[str]:
    return (world.prefix / "install-manifest.txt").read_text().splitlines()[1:]


def test_every_cache_settings_and_temporary_location_is_inside_the_installation(world: World) -> None:
    assert " " in str(world.prefix)  # a prefix with a space: no value may be split
    assert not (world.prefix / "cache").exists() and not (world.prefix / "tmp").exists()  # installing makes neither
    home = tree(world.home)
    result, pairs = exported(world)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stderr == b""  # nothing was set in the shell, so nothing is named
    for variable, relative in VARIABLES.items():
        folder = world.prefix / relative
        assert pairs[variable] == str(folder), variable
        assert folder.is_dir() and not folder.is_symlink() and folder.stat().st_mode & 0o777 == 0o700, variable
        assert os.path.realpath(folder).startswith(os.path.realpath(world.prefix) + os.sep), variable
    assert (world.prefix / "cache").stat().st_mode & 0o777 == 0o700
    assert not any(variable in pairs for variable in REMOVED)
    assert pairs["HOME"] == str(world.home) and tree(world.home) == home  # HOME itself is not changed or written to
    # both folders are the installer's, recorded before they were made, so that uninstall removes them
    assert manifest(world) == ["cache", "tmp", "package", "install.conf", "install-manifest.txt"]
    printed = beta(world, "local-env", "--package", str(world.installed))
    assert printed.returncode == 0, said(printed)
    lines = printed.stdout.splitlines()
    assert lines[:len(VARIABLES)] == [f"{variable}={world.prefix / relative}" for variable, relative in VARIABLES.items()]
    assert lines[len(VARIABLES):-1] == [f"{variable}=" for variable in REMOVED]
    assert lines[-1] == f"LOCAL_ENV_COUNT={len(VARIABLES) + len(REMOVED)}"


def test_a_value_set_in_the_shell_is_replaced_for_the_run_and_named_once(world: World) -> None:
    elsewhere = world.tmp / "elsewhere"
    inherited = {variable: str(elsewhere / variable.lower()) for variable in (*VARIABLES, *REMOVED)}
    result, pairs = exported(world, **inherited)
    assert result.returncode == 0, result.stderr.decode()
    for variable, relative in VARIABLES.items():
        assert pairs[variable] == str(world.prefix / relative), variable  # the installation's, not the shell's
    assert not any(variable in pairs for variable in REMOVED)  # removed for the run: none is passed on
    assert result.stderr.decode() == (
        f"note: set in your shell, not used by this installation (it keeps its own under {world.prefix}): "
        + " ".join((*VARIABLES, *REMOVED)) + "\n")
    assert not elsewhere.exists()  # nothing was made or written where the shell pointed
    # a value that already is the installation's own is not a conflict and is not named
    same = {"TMPDIR": str(world.prefix / "tmp"), "HF_HOME": str(world.prefix / "cache/hf")}
    result, pairs = exported(world, **same)
    assert result.returncode == 0 and result.stderr == b"" and pairs["TMPDIR"] == same["TMPDIR"]


def test_an_installation_reached_through_a_link_still_keeps_its_caches_in_the_real_prefix(world: World) -> None:
    link = world.tmp / "a link to the installation"
    link.symlink_to(world.prefix)
    result, pairs = exported(world, package=link / "package")
    assert result.returncode == 0, result.stderr.decode()
    real = Path(os.path.realpath(world.prefix))
    assert pairs["PIP_CACHE_DIR"] == str(real / "cache/pip") and pairs["TMPDIR"] == str(real / "tmp")


@pytest.mark.parametrize("case", ["foreign_cache", "foreign_tmp", "cache_is_a_link", "tmp_became_a_link",
                                  "a_cache_folder_became_a_link", "a_cache_folder_became_a_file"])
def test_a_link_or_something_the_installer_did_not_make_is_refused_and_never_written_through(world: World, case: str) -> None:
    """A symlink escape is refused where it is, and what it points at is left exactly as it was."""
    elsewhere = world.tmp / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "theirs.txt").write_text("another installation's cache")
    if case == "foreign_cache":
        (world.prefix / "cache").mkdir()
        (world.prefix / "cache" / "mine.txt").write_text("made by the tester")
        expected = f"{world.prefix / 'cache'} exists and was not made by this installer"
    elif case == "foreign_tmp":
        (world.prefix / "tmp").mkdir()
        expected = f"{world.prefix / 'tmp'} exists and was not made by this installer"
    elif case == "cache_is_a_link":
        (world.prefix / "cache").symlink_to(elsewhere)
        expected = f"{world.prefix / 'cache'} exists and was not made by this installer"
    else:
        assert exported(world)[0].returncode == 0  # made and recorded as the installer's first
        target = world.prefix / {"tmp_became_a_link": "tmp", "a_cache_folder_became_a_link": "cache/pip",
                                 "a_cache_folder_became_a_file": "cache/yolo"}[case]
        target.rmdir()
        if case == "a_cache_folder_became_a_file":
            target.write_text("not a directory")
        else:
            target.symlink_to(elsewhere)
        expected = f"{target} is a link or is not a directory"
    before, owned = tree(elsewhere), manifest(world)
    result, pairs = exported(world)
    assert result.returncode == 1 and result.stdout == b"", result.stderr.decode()  # the script stops: nothing is started
    assert f"STOP: {expected}" in result.stderr.decode()
    assert tree(elsewhere) == before  # nothing was written through the link
    if case.startswith("foreign") or case == "cache_is_a_link":
        assert manifest(world) == owned  # and the installer does not claim what it did not make
    if case == "foreign_cache":
        assert (world.prefix / "cache" / "mine.txt").read_text() == "made by the tester"


@pytest.mark.parametrize("keep_assets", [False, True])
def test_uninstall_removes_the_caches_and_temporary_files_and_keeps_the_testers_build_records(world: World, keep_assets: bool) -> None:
    assert exported(world)[0].returncode == 0
    (world.prefix / "cache/pip/a wheel").write_text("downloaded")
    (world.prefix / "tmp/left over").write_text("temporary")
    assert beta(world, "own", "--prefix", str(world.prefix), "assets").returncode == 0
    (world.prefix / "assets/models").mkdir(parents=True)
    (world.prefix / "assets/models/a model").write_text("kept when asked")
    (world.data / "builds/engine-20260101T000000Z").mkdir(parents=True)
    (world.data / "builds/engine-20260101T000000Z/RESULT.txt").write_text("OUTCOME=built\n")
    data = tree(world.data)
    result = world.run("uninstall.sh", "--yes", *(["--keep-assets"] if keep_assets else []), package=world.installed)
    assert result.returncode == 0, said(result)
    removes = next(line for line in result.stdout.splitlines() if line.startswith("This removes what the installer made"))
    named = removes.split(": ", 1)[1].split(". It keeps")[0].rstrip(".")
    assert named == ("cache, tmp, package, install.conf" if keep_assets else "assets, cache, tmp, package, install.conf")
    for name in ("cache", "tmp"):
        assert f"removed: {world.prefix / name}" in result.stdout
    if keep_assets:
        assert sorted(path.name for path in world.prefix.iterdir()) == [
            "assets", "install-manifest.txt", "uninstall-helper.py", "uninstall.sh"]
        assert (world.prefix / "assets/models/a model").read_text() == "kept when asked"
        assert world.run("uninstall.sh", "--yes", package=world.prefix).returncode == 0
    assert not world.prefix.exists()
    assert tree(world.data) == data  # the data directory, with the build records, is never touched


def test_the_settings_display_shows_the_two_folders_without_making_them(world: World) -> None:
    shown = world.run("configure.sh", package=world.installed)
    assert shown.returncode == 0, said(shown)
    assert f"library caches and settings: {world.prefix / 'cache'}  [not made yet]" in shown.stdout
    assert f"temporary files:             {world.prefix / 'tmp'}  [not made yet]" in shown.stdout
    assert not (world.prefix / "cache").exists() and not (world.prefix / "tmp").exists()  # shown only
    assert exported(world)[0].returncode == 0
    shown = world.run("configure.sh", package=world.installed)
    assert f"{world.prefix / 'cache'}  [present]" in shown.stdout and f"{world.prefix / 'tmp'}  [present]" in shown.stdout


def test_the_locations_belong_to_an_installation_only(world: World) -> None:
    refused = beta(world, "local-env", "--package", str(world.package))  # the extracted archive, not an installation
    assert refused.returncode == 1 and "run this from an installed package" in refused.stderr
    assert not (world.package.parent / "cache").exists() and not (world.package.parent / "tmp").exists()
    for name in ("cache", "tmp"):  # the names are the installer's own, like venv and assets
        assert beta(world, "own", "--prefix", str(world.prefix), name).returncode == 0
    assert manifest(world)[:2] == ["cache", "tmp"]
    assert beta(world, "own", "--prefix", str(world.prefix), "caches").returncode == 1


def test_every_script_that_downloads_builds_imports_or_launches_sets_the_locations_first(world: World) -> None:
    """The order in each script: the package is checked, the paths are read, the locations are exported, and only
    then is anything downloaded, built, imported or launched. The scripts that start nothing of the kind do not."""
    first_use = {"setup-env.sh": '-m venv --clear "$VENV"', "fetch-assets.sh": "fetch-assets --package",
                 "build-llama.sh": "git clone", "build-engine.sh": 'env PYTHONDONTWRITEBYTECODE=1 "$RT_PY" -B - <<',
                 "record-artifacts.sh": "lib/artifacts.py\" record", "identity.sh": "-m sentinel.cli identity",
                 "start.sh": '"${RT[@]}" config validate'}
    for script, use in first_use.items():
        text = (world.installed / script).read_text()
        call = 'beta_local_env "$PKG"\n' if script == "start.sh" else 'beta_local_env "$HERE"\n'
        assert text.count(call) == 1, script
        assert text.index("beta_load_paths") < text.index(call) < text.index(use), script
        assert text.index('[ "$(id -u)" != 0 ]') < text.index(call), script  # never made as root
    for script in ("install.sh", "preflight.sh", "uninstall.sh", "stop.sh", "configure.sh", "check.sh"):
        assert "beta_local_env" not in (world.installed / script).read_text(), script
    # the two interpreter starts of the environment setup that would otherwise write bytecode outside it
    setup = (world.installed / "setup-env.sh").read_text()
    assert 'env PYTHONDONTWRITEBYTECODE=1 "$PYTHON310" -m venv --clear "$VENV"' in setup
    assert '"$RT_PY" -B -c \'import sysconfig' in setup
    # the launch and the enrollment still name the face weights' own place themselves (the asset manifest's)
    for script in ("start.sh", "identity.sh"):
        assert '"DEEPFACE_HOME=$DEEPFACE_HOME_DIR"' in (world.installed / script).read_text(), script
