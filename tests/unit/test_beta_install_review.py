"""Regression tests for the findings of the independent review of the installer snapshot 26ac5d1 (IR1 to IR8; IR9 is
with the launcher tests). Each case repeats the review's own trigger against the corrected code.

Synthetic only: temporary directories, the package assembled from the working tree, failures injected into the
helper's own functions, and inert sleeper processes started by the test itself (never a Sentinel runtime, a model
server, a camera or anything of another user). A pass says the corrected logic holds for these triggers; it does not
say that an installation on a clean Jetson works.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
from beta_install_fixture import REPO, World, said, tree

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="process identity is read from /proc (Linux)")

SLEEP = "import time; time.sleep(300)"


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def helper(world: World, name: str = "beta_under_test"):
    """The package's helper as a module, so that a failure can be injected into one of its own steps."""
    spec = importlib.util.spec_from_file_location(name, world.package / "lib" / "beta.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and typing look a module up by its name
    spec.loader.exec_module(module)
    return module


def install_arguments(world: World) -> list[str]:
    return ["install", "--source", str(world.package), "--prefix", str(world.prefix), "--data-dir", str(world.data),
            "--python310", "/usr/bin/python3.10"]


class Sleepers:
    """Inert processes of this test: started with chosen arguments, always ended at the end."""

    def __init__(self) -> None:
        self.started: list[subprocess.Popen] = []

    def start(self, *arguments: str, executable: str | None = None) -> subprocess.Popen:
        command = [executable or sys.executable, "-c", SLEEP, *arguments]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.started.append(process)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:  # its arguments are in /proc once the new program is running
            try:
                if SLEEP.encode() in Path(f"/proc/{process.pid}/cmdline").read_bytes():
                    break
            except OSError:
                pass
            time.sleep(0.02)
        return process

    def named(self, folder: Path, name: str) -> subprocess.Popen:
        """A sleeper whose process name is ``name`` (a link to `sleep` run under that name)."""
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / name
        if not link.exists():
            link.symlink_to(shutil.which("sleep"))
        process = subprocess.Popen([str(link), "300"], start_new_session=True)
        self.started.append(process)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and Path(f"/proc/{process.pid}/comm").read_text().strip() != name:
            time.sleep(0.02)
        return process

    def close(self) -> None:
        for process in self.started:
            if process.poll() is None:
                process.kill()
            process.wait()


@pytest.fixture
def sleepers() -> Iterator[Sleepers]:
    group = Sleepers()
    try:
        yield group
    finally:
        group.close()


def gone(process: subprocess.Popen, seconds: float = 5.0) -> bool:
    try:
        process.wait(seconds)
        return True
    except subprocess.TimeoutExpired:
        return False


def start_time(pid: int) -> int:
    stat = Path(f"/proc/{pid}/stat").read_text()
    return int(stat[stat.rindex(")") + 2:].split()[19])


def beta(world: World, *arguments: str, package: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([str(world.bin / "python3"), "-I", "-B", str((package or world.package) / "lib/beta.py"), *arguments],
                          env=world.env, capture_output=True, text=True, timeout=60)


# ---------------------------------------------------------------- IR1, IR2: processes by exact identity

@pytest.mark.parametrize("folder", ["plain", "with space", "data+cache", "data[beta]", "a.b*c?", "(x)|y$"])
def test_a_leftover_runtime_is_found_whatever_characters_its_data_directory_holds_ir1(
        world: World, sleepers: Sleepers, folder: str) -> None:
    observation = world.tmp / folder / "obs-20260101T000000Z"
    observation.mkdir(parents=True)
    runtime = sleepers.start("-m", "sentinel.cli", "run", "live.yaml", "--data-dir", str(observation / "db"))
    neighbour = sleepers.start("-m", "sentinel.cli", "run", "live.yaml", "--data-dir", str(observation / "db-other"))
    unrelated = sleepers.start("--data-dir", str(observation / "db"))  # the same directory, but not `sentinel run`
    result = beta(world, "after-guard", "--record", str(observation / "processes.txt"), "--data-dir", str(observation / "db"),
                  "--verdict", str(observation / "after-guard.txt"), "--stop-script", "stop.sh", "--runtime-wait", "5")
    assert result.returncode == 0, said(result)
    assert "this run's runtime is still running" in result.stdout and "It has stopped." in result.stdout
    assert gone(runtime) and runtime.returncode != 0  # asked once to stop (SIGINT), and it did
    assert neighbour.poll() is None and unrelated.poll() is None  # a path that only begins the same is another run's
    assert (observation / "after-guard.txt").read_text() == "RESIDUE=stopped_after_the_guard\n"


def test_a_runtime_that_does_not_stop_is_reported_and_the_desktop_stays_down_ir1(world: World, sleepers: Sleepers) -> None:
    observation = world.tmp / "obs+1"
    observation.mkdir()
    stubborn = sleepers.start("-m", "sentinel.cli", "run", "--data-dir", str(observation / "db"))
    # it ignores the one clean request, as a stuck runtime would
    stubborn.kill()
    stubborn.wait()
    code = "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(300)"
    stuck = subprocess.Popen([sys.executable, "-c", code, "-m", "sentinel.cli", "run", "--data-dir", str(observation / "db")],
                             start_new_session=True)
    sleepers.started.append(stuck)
    time.sleep(0.5)
    result = beta(world, "after-guard", "--record", str(observation / "none.txt"), "--data-dir", str(observation / "db"),
                  "--verdict", str(observation / "after-guard.txt"), "--stop-script", "/x/stop.sh", "--runtime-wait", "1")
    assert f"STILL RUNNING: runtime [{stuck.pid}]. The camera may still be open. Stop it with: bash /x/stop.sh" in result.stdout
    assert stuck.poll() is None and (observation / "after-guard.txt").read_text() == "RESIDUE=still_running\n"


def test_only_a_model_server_that_the_run_started_is_stopped_and_another_is_reported_ir2(
        world: World, sleepers: Sleepers) -> None:
    observation = world.tmp / "obs"
    observation.mkdir()
    owned = sleepers.named(world.tmp / "run", "llama-server")
    unrelated = sleepers.named(world.tmp / "elsewhere", "llama-server")  # started by the tester during the run
    reused = sleepers.named(world.tmp / "later", "llama-server")  # a later process that got a recorded number
    (observation / "processes.txt").write_text(
        f"{owned.pid} {start_time(owned.pid)} llama-server\n"
        f"{reused.pid} {start_time(reused.pid) - 1} llama-server\n"  # the recorded process started earlier: not this one
        "999999999 5 llama-server\n")  # a recorded process that has gone
    result = beta(world, "after-guard", "--record", str(observation / "processes.txt"), "--data-dir", str(observation / "db"),
                  "--verdict", str(observation / "after-guard.txt"), "--stop-script", "stop.sh", "--server-wait", "5")
    assert result.returncode == 0, said(result)
    assert "A llama-server that this run started is still running after the runtime ended; stopping it." in result.stdout
    assert gone(owned)
    assert unrelated.poll() is None and reused.poll() is None  # never signalled: nothing shows they are this run's
    assert "NOTE: another llama-server of yours is running" in result.stdout and "it was not signalled" in result.stdout
    assert str(unrelated.pid) in result.stdout and str(reused.pid) in result.stdout
    assert (observation / "after-guard.txt").read_text() == "RESIDUE=stopped_after_the_guard\n"


def test_the_runs_record_names_what_descends_from_its_guard_and_nothing_else_ir2(world: World, sleepers: Sleepers) -> None:
    observation = world.tmp / "obs [1]+"
    (observation / "guard").parent.mkdir(parents=True)
    child = ("import subprocess, sys, time; "
             f"subprocess.Popen([{str(world.tmp / 'run' / 'llama-server')!r}, '300'], start_new_session=True); time.sleep(300)")
    (world.tmp / "run").mkdir()
    (world.tmp / "run" / "llama-server").symlink_to(shutil.which("sleep"))
    guard = subprocess.Popen([sys.executable, "-c", child, "benchmarks/runner/step5_guard.py", "--out", str(observation / "guard")],
                             start_new_session=True)
    sleepers.started.append(guard)
    outsider = sleepers.named(world.tmp / "elsewhere", "llama-server")
    (observation / "guard.pid").write_text(str(guard.pid))
    watcher = subprocess.Popen([str(world.bin / "python3"), "-I", "-B", str(world.package / "lib/beta.py"), "watch-run",
                                "--guard-pidfile", str(observation / "guard.pid"), "--out", str(observation / "guard"),
                                "--record", str(observation / "processes.txt"), "--interval", "0.1"], env=world.env)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (observation / "processes.txt").exists():
            time.sleep(0.1)
        lines = [line.split() for line in (observation / "processes.txt").read_text().splitlines()]
        assert [line[2] for line in lines] == ["llama-server"] and int(lines[0][0]) != outsider.pid
        assert int(lines[0][1]) == start_time(int(lines[0][0])) and (observation / "processes.txt").stat().st_mode & 0o777 == 0o600
        started = int(lines[0][0])
        guard.kill()
        guard.wait()
        assert watcher.wait(10) == 0  # it ends with the guard
    finally:
        if watcher.poll() is None:
            watcher.kill()
        for line in (observation / "processes.txt").read_text().splitlines() if (observation / "processes.txt").exists() else []:
            try:
                os.kill(int(line.split()[0]), signal.SIGKILL)
            except (OSError, ValueError):
                pass
    result = beta(world, "after-guard", "--record", str(observation / "processes.txt"), "--data-dir", str(observation / "db"),
                  "--verdict", str(observation / "v.txt"), "--stop-script", "stop.sh")
    assert f"[{started}]" not in result.stdout and outsider.poll() is None  # the recorded one is gone; the outsider is left


def test_stop_signals_only_the_recorded_guard_of_the_latest_observation_ir1(world: World, sleepers: Sleepers) -> None:
    root = world.tmp / "observe+[x]"
    older, latest = root / "obs-20260101T000000Z", root / "obs-20260102T000000Z"
    for folder in (older, latest):
        folder.mkdir(parents=True)
    arguments = ("--step4-headroom", "--out")
    guard = sleepers.start("benchmarks/runner/step5_guard.py", *arguments, str(latest / "guard"))
    older_guard = sleepers.start("benchmarks/runner/step5_guard.py", *arguments, str(older / "guard"))
    similar = sleepers.start("benchmarks/runner/step5_guard.py", *arguments, str(latest / "guard-2"))
    (latest / "guard.pid").write_text(f"{guard.pid}\n")
    result = beta(world, "stop-run", "--observe-root", str(root), "--stop-script", "stop.sh", "--guard-wait", "5")
    assert result.returncode == 0, said(result)
    assert f"Asked {latest} to stop" in result.stdout and "The guard has stopped." in result.stdout
    assert gone(guard) and older_guard.poll() is None and similar.poll() is None
    # A number in guard.pid that now belongs to something else is never signalled.
    (latest / "guard.pid").write_text(f"{similar.pid}\n")
    again = beta(world, "stop-run", "--observe-root", str(root), "--stop-script", "stop.sh", "--guard-wait", "2")
    assert again.returncode == 0 and "No observation is running; nothing was signalled." in again.stdout
    assert similar.poll() is None
    (latest / "guard.pid").write_text("not a number\n")
    assert "nothing was signalled" in beta(world, "stop-run", "--observe-root", str(root), "--stop-script", "s").stdout


@pytest.mark.parametrize("name", ["sentinel beta", "prefix+cache", "prefix[beta]"])
def test_uninstall_refuses_while_something_of_the_installation_runs_whatever_its_path_holds_ir1(
        world: World, sleepers: Sleepers, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setattr(World, "prefix", property(lambda self: self.tmp / "opt" / name))
    assert world.install().returncode == 0
    assert beta(world, "own", "--prefix", str(world.prefix), "venv", package=world.installed).returncode == 0
    (world.prefix / "venv/bin").mkdir(parents=True)
    (world.prefix / "venv/bin/python").symlink_to(sys.executable)
    running = sleepers.start("-m", "sentinel.cli", "run", executable=str(world.prefix / "venv/bin/python"))
    similar = world.tmp / "opt" / (name + "-other") / "venv/bin"  # a path that only begins the same
    similar.mkdir(parents=True)
    (similar / "python").symlink_to(sys.executable)
    neighbour = sleepers.start(executable=str(similar / "python"))
    before = tree(world.prefix)
    refused = world.run("uninstall.sh", "--yes", package=world.installed)
    assert refused.returncode == 1 and "is still running" in said(refused) and str(running.pid) in said(refused)
    assert str(neighbour.pid) not in said(refused) and tree(world.prefix) == before
    running.kill()
    running.wait()
    removed = world.run("uninstall.sh", "--yes", package=world.installed)
    assert removed.returncode == 0, said(removed)
    assert not world.prefix.exists() and neighbour.poll() is None


# ---------------------------------------------------------------- IR3: a recoverable installation

class Injected(Exception):
    pass


def fail_at(module: Any, monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    """Make one step of the installation fail once, as an interruption or a full disk would."""
    calls = {"n": 0}

    def wrap(target: Any, name: str, when: Any) -> None:
        original = getattr(target, name)

        def replacement(*arguments: Any, **keywords: Any) -> Any:
            calls["n"] += 1
            if when(calls["n"], arguments):
                raise Injected(step)
            return original(*arguments, **keywords)

        monkeypatch.setattr(target, name, replacement)

    if step == "first_manifest":
        wrap(module, "write_manifest", lambda n, a: n == 1)
    elif step == "during_copy":
        wrap(module.shutil, "copyfile", lambda n, a: n == 20)
    elif step == "before_swap":
        wrap(module, "write_manifest", lambda n, a: n == 2)
    elif step == "between_renames":
        wrap(module.os, "rename", lambda n, a: str(a[0]).endswith("package.new"))
    elif step == "install_conf":
        wrap(module, "atomic_write", lambda n, a: Path(a[0]).name == "install.conf")
    elif step == "last_manifest":
        wrap(module, "write_manifest", lambda n, a: n == 4)
    elif step == "data_marker":
        wrap(module, "atomic_write", lambda n, a: Path(a[0]).name == module.DATA_MARKER)
    elif step == "data_folders":
        wrap(module.os, "mkdir", lambda n, a: str(a[0]).endswith("identity"))
    elif step == "default_settings":
        wrap(module.os, "open", lambda n, a: str(a[0]).endswith("live.yaml"))
    else:
        raise AssertionError(step)


STEPS = ["first_manifest", "during_copy", "before_swap", "between_renames", "install_conf", "last_manifest",
         "data_marker", "data_folders", "default_settings"]


@pytest.mark.parametrize("step", STEPS)
def test_an_interrupted_first_installation_is_completed_by_installing_again_ir3(
        world: World, monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    monkeypatch.setenv("HOME", str(world.home))
    module = helper(world)
    with monkeypatch.context() as patch:
        fail_at(module, patch, step)
        with pytest.raises(Injected):
            module.command_install(module.build_parser().parse_args(install_arguments(world)))
    retry = world.install()
    assert retry.returncode == 0, said(retry)
    assert tree(world.installed) == tree(world.package)
    assert sorted(path.name for path in world.prefix.iterdir()) == ["install-manifest.txt", "install.conf", "package"]
    assert (world.prefix / "install-manifest.txt").read_text().splitlines()[1:] == ["package", "install.conf", "install-manifest.txt"]
    assert (world.data / "config/live.yaml").read_text() == (world.package / "templates/live.yaml").read_text()
    assert sorted(path.name for path in world.data.iterdir()) == [".sentinel-beta-data", "config", "identity", "observe"]
    assert not list(world.tmp.rglob("*.tmp"))  # no temporary file is left behind


@pytest.mark.parametrize("step", STEPS[:6])
def test_an_interrupted_reinstallation_keeps_the_data_and_a_usable_package_until_the_new_one_is_in_place_ir3(
        world: World, monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    monkeypatch.setenv("HOME", str(world.home))
    assert world.install().returncode == 0
    (world.data / "config/live.yaml").write_text("config_version: 1\ncamera: {id: mine}\n")
    (world.data / "identity/gallery.sealed").write_text("my sealed enrollment")
    (world.data / "observe/obs-1").mkdir()
    (world.data / "observe/obs-1/launch.txt").write_text("a record")
    assert beta(world, "own", "--prefix", str(world.prefix), "venv", package=world.installed).returncode == 0
    (world.prefix / "venv").mkdir()
    (world.prefix / "venv/marker").write_text("the environment")
    data = tree(world.data)
    module = helper(world)
    with monkeypatch.context() as patch:
        fail_at(module, patch, step)
        with pytest.raises(Injected):
            module.command_install(module.build_parser().parse_args(install_arguments(world)))
    assert tree(world.data) == data and (world.prefix / "venv/marker").read_text() == "the environment"
    usable = [path for path in (world.prefix / "package", world.prefix / "package.old") if (path / "start.sh").is_file()]
    assert usable and tree(usable[0]) == tree(world.package)  # a whole, checked copy is still there
    retry = world.install()
    assert retry.returncode == 0, said(retry)
    assert tree(world.data) == data and tree(world.installed) == tree(world.package)
    assert sorted(path.name for path in world.prefix.iterdir()) == ["install-manifest.txt", "install.conf", "package", "venv"]


@pytest.mark.parametrize("name", ["package.new", "package", "package.old", "install.conf"])
def test_something_the_installer_did_not_make_is_never_deleted_or_overwritten_by_an_installation_ir3(
        world: World, name: str) -> None:
    assert world.install().returncode == 0
    assert beta(world, "own", "--prefix", str(world.prefix), "assets", package=world.installed).returncode == 0
    (world.prefix / "assets").mkdir()
    assert world.run("uninstall.sh", "--keep-assets", "--yes", package=world.installed).returncode == 0
    mine = world.prefix / name  # the tester puts something of their own where the installer would write
    if name == "install.conf":
        mine.write_text("my own notes\n")
    else:
        mine.mkdir()
        (mine / "note.txt").write_text("my own notes\n")
    before = tree(world.tmp)
    refused = world.install()
    assert refused.returncode == 1 and f"{mine} exists and was not made by this installer" in said(refused), said(refused)
    assert tree(world.tmp) == before
    for later, owned in (("venv", "setup-env.sh"), ("assets", "fetch-assets.sh")):
        pass
    (world.prefix / "venv").mkdir()
    (world.prefix / "venv/mine").write_text("x")
    cannot_claim = beta(world, "own", "--prefix", str(world.prefix), "venv", package=world.package)
    assert cannot_claim.returncode == 1 and "was not made by this installer" in cannot_claim.stderr


# ---------------------------------------------------------------- IR4: nothing is written through a link

@pytest.mark.parametrize("name", ["live.yaml", "site.conf"])
@pytest.mark.parametrize("dangling", [True, False])
def test_a_linked_settings_file_is_kept_and_never_written_through_ir4(world: World, name: str, dangling: bool) -> None:
    assert world.install().returncode == 0
    outside = world.tmp / "outside" / name
    outside.parent.mkdir()
    if not dangling:
        outside.write_text("the tester's own file elsewhere\n")
    (world.data / "config" / name).unlink()
    (world.data / "config" / name).symlink_to(outside)
    result = world.install()
    assert result.returncode == 0, said(result)
    assert f"kept: {world.data / 'config' / name} is a link you made; the installer does not write through it" in result.stdout
    assert (world.data / "config" / name).is_symlink()
    assert outside.exists() is (not dangling)  # nothing was created outside the data directory
    if not dangling:
        assert outside.read_text() == "the tester's own file elsewhere\n" and outside.stat().st_mode & 0o777 != 0o600
    assert sorted(path.name for path in outside.parent.iterdir()) == ([] if dangling else [name])
    if name == "site.conf":
        refused = world.run("configure.sh", "--set", "ENGINE=/x/engine", package=world.installed)
        assert refused.returncode == 1 and "is a link you made: it is not written through or replaced" in said(refused)
        assert (world.data / "config" / name).is_symlink() and outside.exists() is (not dangling)


@pytest.mark.parametrize("folder", ["config", "identity", "observe", "identity/inbox"])
def test_a_linked_folder_in_the_data_directory_is_refused_before_anything_is_written_ir4(world: World, folder: str) -> None:
    assert world.install().returncode == 0
    outside = world.tmp / "outside"
    outside.mkdir()
    shutil.rmtree(world.data / folder)
    (world.data / folder).symlink_to(outside)
    before = tree(world.tmp)
    result = world.install()
    assert result.returncode == 1 and "is a link or is not a directory: the installer writes nothing through it" in said(result)
    assert tree(world.tmp) == before and list(outside.iterdir()) == []


# ---------------------------------------------------------------- IR5: an uninstall can always be finished

def test_an_interrupted_uninstall_is_finished_by_the_entry_it_leaves_ir5(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    assert world.install().returncode == 0
    for name in ("venv", "assets"):
        assert beta(world, "own", "--prefix", str(world.prefix), name, package=world.installed).returncode == 0
        (world.prefix / name).mkdir()
        (world.prefix / name / "file").write_text(name)
    (world.data / "identity/gallery.sealed").write_text("my sealed enrollment")
    data = tree(world.data)
    module = helper(world)
    original, calls = module.remove_owned, {"n": 0}

    def interrupted(prefix: Path, name: str) -> bool:
        calls["n"] += 1
        if name == "package":
            raise Injected("while removing the package")
        return original(prefix, name)

    with monkeypatch.context() as patch:
        patch.setattr(module, "remove_owned", interrupted)
        with pytest.raises(Injected):
            module.command_uninstall(module.build_parser().parse_args(["uninstall", "--prefix", str(world.prefix), "--yes"]))
    assert (world.prefix / "package").exists() and not (world.prefix / "venv").exists()  # stopped half-way
    assert (world.prefix / "uninstall.sh").is_file() and tree(world.data) == data
    finish = world.run("uninstall.sh", "--yes", package=world.prefix)
    assert finish.returncode == 0, said(finish)
    assert not world.prefix.exists() and tree(world.data) == data


def test_the_extracted_archive_can_remove_a_named_installation_and_refuses_a_place_that_is_not_one_ir5(world: World) -> None:
    assert world.install().returncode == 0
    not_named = world.run("uninstall.sh", "--yes")
    assert not_named.returncode == 1 and "this copy is not installed. Name the installation" in said(not_named)
    empty = world.tmp / "not an installation"
    empty.mkdir()
    (empty / "file").write_text("x")
    refused = world.run("uninstall.sh", "--prefix", str(empty), "--yes")
    assert refused.returncode == 1 and "holds no installation of this package" in said(refused)
    assert (empty / "file").read_text() == "x"
    named = world.run("uninstall.sh", "--prefix", str(world.prefix), "--yes")
    assert named.returncode == 0 and not world.prefix.exists(), said(named)


# ---------------------------------------------------------------- IR6: the export's dependencies are pinned

def test_what_the_engine_export_needs_is_pinned_and_checked_before_any_gpu_work_ir6(world: World) -> None:
    pypi = (world.package / "manifest/requirements-pypi.txt").read_text()
    jetson = (world.package / "manifest/requirements-jetson.txt").read_text()
    for pin in ("onnx==1.20.1", "onnxslim==0.1.90", "onnxruntime==1.23.2", "colorama==0.4.6", "coloredlogs==15.0.1",
                "humanfriendly==10.0"):
        assert f"\n{pin} \\\n    --hash=sha256:" in pypi, pin
    assert "\nonnxruntime-gpu==1.23.0 \\\n    --hash=sha256:4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b" in jetson
    assert "--index-url https://pypi.jetson-ai-lab.io/jp6/cu126" in jetson and "--index-url https://pypi.org/simple" in pypi
    for text in (pypi, jetson):  # every requirement is one exact version with at least one checksum
        for line in text.splitlines():
            if line and line[0].isalnum():
                assert "==" in line and line.endswith("\\"), line
    setup = (world.package / "setup-env.sh").read_text()  # the Jetson build first, PyPI's onnxruntime over it, then the import
    assert setup.index("requirements-jetson.txt") < setup.index("requirements-pypi.txt") < setup.index('onnxruntime.__version__ == "1.23.2"')
    script = (world.package / "build-engine.sh").read_text()
    check, prompt = script.index('"onnxslim": "0.1.90"'), script.index("type BUILD to build the engine now")
    assert check < prompt < script.index("YOLO(sys.argv[1]).export(")  # checked, then asked, then the GPU is used
    assert "simplify=True" in script and "YOLO_OFFLINE=true" in script and "pip install" not in script


def test_the_engine_build_stops_before_asking_when_the_environment_lacks_the_exports_packages_ir6(world: World) -> None:
    assert world.install().returncode == 0
    assert beta(world, "own", "--prefix", str(world.prefix), "venv", package=world.installed).returncode == 0
    (world.prefix / "venv/bin").mkdir(parents=True)
    (world.prefix / "venv/bin/python").symlink_to(sys.executable)  # an environment without onnx, onnxslim, onnxruntime
    assert beta(world, "own", "--prefix", str(world.prefix), "assets", package=world.installed).returncode == 0
    (world.prefix / "assets/engine").mkdir(parents=True)
    (world.prefix / "assets/engine/yolov8n.pt").write_bytes(b"weights")
    world.command("pgrep", "exit 1")
    result = world.run("build-engine.sh", package=world.installed, stdin="BUILD\n")
    assert result.returncode == 1, said(result)
    assert "onnx: needs 1.20.1, installed missing" in result.stdout and "onnxslim: needs 0.1.90" in result.stdout
    assert "the environment lacks what the export needs" in said(result) and "type BUILD" not in said(result)
    assert not (world.prefix / "assets/engine/yolov8n.engine").exists()


# ---------------------------------------------------------------- IR7: the package is exactly its files, and no link

@pytest.mark.parametrize("case", ["linked_directory", "linked_ancestor", "linked_file", "link_in_app"])
def test_a_link_inside_the_package_is_refused_ir7(world: World, case: str) -> None:
    elsewhere = world.tmp / "elsewhere"
    (elsewhere / "sitecustomize").mkdir(parents=True)
    (elsewhere / "sitecustomize" / "__init__.py").write_text("print('not part of the package')\n")
    if case == "linked_directory":
        (world.package / "lib" / "extra").symlink_to(elsewhere / "sitecustomize")
        expected = "lib/extra (a link)"
    elif case == "link_in_app":
        (world.package / "app/src/sitecustomize").symlink_to(elsewhere / "sitecustomize")
        expected = "app/src/sitecustomize (a link)"
    elif case == "linked_ancestor":  # every listed file is still there and equal, but through a linked directory
        shutil.copytree(world.package / "lib", elsewhere / "lib")
        shutil.rmtree(world.package / "lib")
        (world.package / "lib").symlink_to(elsewhere / "lib")
        expected = "lib (a link)"
    else:
        shutil.copy(world.package / "QUICKSTART.md", elsewhere / "QUICKSTART.md")
        (world.package / "QUICKSTART.md").unlink()
        (world.package / "QUICKSTART.md").symlink_to(elsewhere / "QUICKSTART.md")
        expected = "QUICKSTART.md (a link)"
    helper_copy = world.tmp / "beta.py"  # the checker itself is run from a copy: the package under test may be the broken one
    helper_copy.write_bytes((REPO / "packaging/beta-install/lib/beta.py").read_bytes())
    result = subprocess.run([str(world.bin / "python3"), "-I", "-B", str(helper_copy), "verify-package", "--package",
                             str(world.package)], env=world.env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and expected in result.stderr, said(result)
    install = world.install()
    assert install.returncode == 1 and not world.prefix.exists(), said(install)


# ---------------------------------------------------------------- IR8: the lock tool checks that a file fits

def lock_tool():
    spec = importlib.util.spec_from_file_location("lock_requirements_under_test", REPO / "packaging/lock_requirements.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename,fits", [
    ("pkg-1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl", True),
    ("pkg-1.0-cp310-cp310-manylinux_2_28_aarch64.whl", True),
    ("pkg-1.0-cp310-cp310-linux_aarch64.whl", True),
    ("pkg-1.0-py3-none-any.whl", True),
    ("pkg-1.0-py2.py3-none-any.whl", True),
    ("pkg-1.0-cp37-abi3-manylinux_2_28_aarch64.whl", True),
    ("pkg-1.0-cp312-cp312-manylinux_2_28_aarch64.whl", False),  # the review's trigger: another interpreter's wheel
    ("pkg-1.0-cp311-abi3-manylinux_2_28_aarch64.whl", False),
    ("pkg-1.0-cp310-cp310-manylinux_2_39_aarch64.whl", False),  # needs a newer glibc than JetPack 6's
    ("pkg-1.0-cp310-cp310-manylinux_2_28_x86_64.whl", False),
    ("pkg-1.0-cp310-cp310-win_arm64.whl", False),
    ("pkg-1.0.tar.gz", False),
])
def test_only_a_wheel_that_fits_python_3_10_on_aarch64_counts_ir8(filename: str, fits: bool) -> None:
    assert lock_tool().fits_target(filename) is fits


def test_a_release_with_no_usable_file_is_refused_by_the_lock_tool_ir8() -> None:
    tool = lock_tool()
    only_312 = [{"filename": "pkg-1.0-cp312-cp312-manylinux_2_28_aarch64.whl", "packagetype": "bdist_wheel"}]
    assert tool.usable(only_312) == []
    assert tool.usable(only_312 + [{"filename": "pkg-1.0.tar.gz", "packagetype": "sdist"}]) == ["pkg-1.0.tar.gz"]
    fitting = {"filename": "pkg-1.0-cp310-cp310-manylinux_2_28_aarch64.whl", "packagetype": "bdist_wheel"}
    assert tool.usable([fitting]) == [fitting["filename"]]
    assert tool.usable([{**fitting, "yanked": True}]) == []
    for requires, admitted in ((">=3.9", True), (">=3.10,<3.13", True), (">=3.11", False), ("<3.10", False), ("==3.10.*", True),
                               ("!=3.10.*", False), (">=3.8, !=3.9.*", True), ("something else", False), (None, True), ("", True)):
        assert tool.admits_python(requires) is admitted, requires
        assert bool(tool.usable([{**fitting, "requires_python": requires}])) is admitted, requires


# ---------------------------------------------------------------- the changed guard is packaged, and labelled as changed

GUARD = "benchmarks/runner/step5_guard.py"


def builder():
    import beta_install_fixture

    return beta_install_fixture.build_archive


def test_the_package_ships_the_corrected_guard_and_names_it_as_changed() -> None:
    build = builder()
    name, members = build.package_members(0, worktree=True)
    packaged = members[f"app/{GUARD}"]
    accepted = build.git("show", f"{build.ACCEPTED}:{GUARD}")
    assert packaged == (REPO / GUARD).read_bytes() and packaged != accepted
    assert b"PR_SET_CHILD_SUBREAPER" in packaged and b"llama_server_unresolved" in packaged
    assert b"backend.kill(pid, signum)" in accepted and b"backend.kill(" not in packaged  # the cleanup by name is gone
    sha = {"accepted": build.hashlib.sha256(accepted).hexdigest(), "packaged": build.hashlib.sha256(packaged).hexdigest()}
    version = members["VERSION"].decode().splitlines()
    mark = build.modification_mark({GUARD: sha["packaged"]})
    assert name == f"sentinel-personal-beta-949da67-mod{mark[:7]}-r0-worktree" and version[0] == name
    assert f"accepted_code_commit={build.ACCEPTED}" in version and f"modification_mark={mark}" in version
    assert ("application=the accepted code with 1 file(s) changed after acceptance (below); this is NOT the unchanged "
            "accepted build") in version
    assert f"changed_file=app/{GUARD} accepted_sha256={sha['accepted']} packaged_sha256={sha['packaged']}" in version
    assert any(line.startswith(f"changed_file_reason=app/{GUARD}: process ownership") for line in version)
    assert any(line.startswith(f"changed_file_run_with=app/{GUARD}: synthetic and stand-in-process tests") and
               "no guarded run with a camera or a model" in line for line in version)
    unchanged = next(line for line in version if line.startswith("unchanged_from_accepted="))
    assert f"app/{GUARD}" not in unchanged and all(f"app/{path}" in unchanged for path in build.GUARD_FILES if path != GUARD)
    assert f"{sha['packaged']}  app/{GUARD}\n" in members["SHA256SUMS"].decode()
    # every other application file is the accepted commit's
    differs = build.git("diff", "--name-only", build.ACCEPTED, "--", build.APP_TREE, *build.GUARD_FILES).decode().split()
    assert differs == [GUARD]
    # the guide a tester reads names this package, and says what is not the accepted build's
    released, _ = build.package_members(build.DEFAULT_REVISION, worktree=True)
    released = released[:-len("-worktree")]
    guide = members["QUICKSTART.md"].decode()
    assert f"tar -xzf {released}.tar.gz && cd {released}\n" in guide
    assert "**One file is not the accepted build's.**" in guide and "stops every `llama-server`" not in guide
    assert "The accepted guard itself has already stopped" not in members["start.sh"].decode()


@pytest.mark.parametrize(("differs", "declared", "told"), [
    (f"{GUARD}\nsrc/sentinel/cli.py\n", None, "changed but not declared: src/sentinel/cli.py"),
    ("", None, f"declared but identical to the accepted file: {GUARD}"),
    (f"{GUARD}\n", {}, f"changed but not declared: {GUARD}"),
])
def test_the_builder_refuses_a_change_it_does_not_declare_and_a_declaration_without_a_change(
        monkeypatch: pytest.MonkeyPatch, differs: str, declared: dict | None, told: str) -> None:
    build = builder()
    real = build.git
    monkeypatch.setattr(build, "git", lambda *args: differs.encode() if args[0] == "diff" else real(*args))
    if declared is not None:
        monkeypatch.setattr(build, "CHANGED_SINCE_ACCEPTED", declared)
    for worktree in (True, False):
        with pytest.raises(SystemExit) as stop:
            build.package_members(1, worktree=worktree)
        assert "are not exactly the ones this package declares as changed" in str(stop.value) and told in str(stop.value)


def test_only_a_package_without_changed_files_carries_the_accepted_name(monkeypatch: pytest.MonkeyPatch) -> None:
    build = builder()
    real = build.git
    monkeypatch.setattr(build, "git", lambda *args: b"" if args[0] == "diff" else real(*args))
    monkeypatch.setattr(build, "CHANGED_SINCE_ACCEPTED", {})
    name, members = build.package_members(3, worktree=True)
    version = members["VERSION"].decode()
    assert name == "sentinel-personal-beta-949da67-r3-worktree" and "-mod" not in name
    assert "application=the accepted code, unchanged (" in version and "changed_file" not in version
    monkeypatch.undo()
    name, members = build.package_members(3, worktree=True)  # as this branch is: the mark is part of the name
    assert "-mod" in name and "application=the accepted code, unchanged" not in members["VERSION"].decode()
