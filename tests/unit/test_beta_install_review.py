"""Regression tests for the findings of the independent review of the installer snapshot 26ac5d1 (IR1 to IR8; IR9 is
with the launcher tests). Each case repeats the review's own trigger against the corrected code.

Synthetic only: temporary directories, the package assembled from the working tree, failures injected into the
helper's own functions, and inert sleeper processes started by the test itself (never a Sentinel runtime, a model
server, a camera or anything of another user). A pass says the corrected logic holds for these triggers; it does not
say that an installation on a clean Jetson works.
"""

from __future__ import annotations

import importlib.util
import json
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
    """The package's helper as a module, so that a failure can be injected into one of its own steps.

    Imported as the package's scripts run it (``python3 -I -B``): no bytecode is written beside it. A cache folder
    in the package is a file that is not part of it, and the package's own check then refuses to install from it.
    """
    spec = importlib.util.spec_from_file_location(name, world.package / "lib" / "beta.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and typing look a module up by its name
    written, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = written
    assert not (world.package / "lib" / "__pycache__").exists()  # the package is still as it was extracted
    return module


def install_arguments(world: World) -> list[str]:
    return ["install", "--source", str(world.package), "--prefix", str(world.prefix), "--data-dir", str(world.data),
            "--python310", str(world.bin / "python3.10")]  # what install.sh records here: the one its preflight found


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
        """A sleeper whose process name is ``name``: this interpreter, run through a link of that name.

        Not a link to the system's `sleep`. That sleeps only where `sleep` is a program of its own. Where the core
        utilities are one multi-call program, which picks the utility by the name it is run under (the Rust coreutils
        package of Ubuntu 26.04 is built that way; BusyBox is another), a link of another name does not sleep.
        """
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / name
        if not link.exists():
            link.symlink_to(sys.executable)
        process = subprocess.Popen([str(link), "-c", SLEEP], start_new_session=True)
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
             f"subprocess.Popen([{str(world.tmp / 'run' / 'llama-server')!r}, '-c', {SLEEP!r}], start_new_session=True); "
             "time.sleep(300)")
    (world.tmp / "run").mkdir()
    (world.tmp / "run" / "llama-server").symlink_to(sys.executable)  # as Sleepers.named: never the system's `sleep`
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
            watcher.wait()  # reaped here: left to the collector, its warning fails whichever test is running then
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
    assert "YOLO_OFFLINE=true" in script and "pip install" not in script
    # the export's settings are the asset manifest's, given to the export call as they are (and noted as given)
    export = next(asset for asset in json.loads((world.package / "manifest/assets.json").read_text())["assets"]
                  if asset["id"] == "detector_engine")["export"]
    assert export == {"format": "engine", "imgsz": 640, "batch": 1, "half": False, "int8": False, "dynamic": False,
                      "simplify": True, "nms": False, "device": 0}
    assert 'asset --package "$PKG" detector_engine export' in script
    assert "YOLO(sys.argv[1]).export(**json.loads(sys.argv[2]))" in script and '"$RT_PY" - "$WEIGHTS" "$EXPORT"' in script


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


# ---------------------------------------------------------------- what is packaged is named for what it is

GUARD = "benchmarks/runner/step5_guard.py"
ROUTE = ("src/sentinel/cli.py", "src/sentinel/demo_runtime.py", "src/sentinel/device_artifacts.py")
VERSION_FILE = "src/sentinel/__init__.py"  # the fifth changed file: the version string only
NEW_PACKAGE_FILES: tuple = ()  # since revision 7: none; three documents changed
SINCE_REAL_EVIDENCE: tuple = (  # the package's own files that revision 5, which ran for real, did not have in this form
    "QUICKSTART.md", "README.md", "THIRD_PARTY_NOTICES.md", "build-engine.sh", "build-llama.sh", "fetch-assets.sh",
    "guarded-build.sh", "identity.sh", "install.sh", "lib/beta.py", "lib/common.sh", "lib/guarded_build.py",
    "lib/guarded_run.py", "preflight.sh", "record-artifacts.sh", "setup-env.sh", "start.sh", "uninstall.sh")
REAL_RECORD = ("src/sentinel/demo_runtime.py", "src/sentinel/device_artifacts.py")  # ran once on a real record


def builder():
    import beta_install_fixture

    return beta_install_fixture.build_archive


def sha256(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


def test_the_package_ships_the_corrected_guard_and_names_it_as_changed() -> None:
    build = builder()
    name, members = build.package_members(0, worktree=True)
    packaged = members[f"app/{GUARD}"]
    accepted = build.git("show", f"{build.ACCEPTED}:{GUARD}")
    assert packaged == (REPO / GUARD).read_bytes() and packaged != accepted
    assert b"PR_SET_CHILD_SUBREAPER" in packaged and b"llama_server_unresolved" in packaged
    assert b"backend.kill(pid, signum)" in accepted and b"backend.kill(" not in packaged  # the cleanup by name is gone
    version = members["VERSION"].decode().splitlines()
    changed = sorted([GUARD, *ROUTE, VERSION_FILE])
    assert sorted(build.CHANGED_SINCE_ACCEPTED) == changed
    mark = build.modification_mark({path: sha256(members[f"app/{path}"]) for path in changed})
    assert name == f"sentinel-personal-beta-949da67-mod{mark[:7]}-r0-worktree" and version[0] == name
    assert f"accepted_code_commit={build.ACCEPTED}" in version and f"modification_mark={mark}" in version
    assert ("application=the accepted code with 5 file(s) changed or added after acceptance (below); this is NOT the "
            "unchanged accepted build") in version
    assert f"changed_file=app/{GUARD} accepted_sha256={sha256(accepted)} packaged_sha256={sha256(packaged)}" in version
    assert any(line.startswith(f"changed_file_reason=app/{GUARD}: process ownership") for line in version)
    assert any(line.startswith(f"changed_file_run_with=app/{GUARD}: synthetic and stand-in-process tests") and
               "no guarded run with a camera or a model" in line for line in version)
    assert f"{sha256(packaged)}  app/{GUARD}\n" in members["SHA256SUMS"].decode()
    # every other application file is the accepted commit's, by content
    application = {path[len("app/"):]: content for path, content in members.items() if path.startswith("app/")}
    assert build.differing(build.blobs(build.ACCEPTED, build.APP_TREE, *build.GUARD_FILES), application) == changed
    assert (f"unchanged_from_accepted={len(application) - 5} application files: every other file under app/src/sentinel "
            "and app/benchmarks/runner") in version
    # the fifth: one line of one file, the version string, and the package says so
    was, now = build.git("show", f"{build.ACCEPTED}:{VERSION_FILE}").decode(), members[f"app/{VERSION_FILE}"].decode()
    assert now == was.replace('__version__ = "2.0.0a2"', '__version__ = "2.0.0b1"') != was
    assert any(line.startswith(f"changed_file_reason=app/{VERSION_FILE}: the version string only") and
               "nothing else reads it" in line for line in version)
    assert "application_version=2.0.0b1" in version
    # the beta name is a release designation: the package says so, and its status line is what it was
    assert ("release_designation=v2.0.0-beta.1: the name of a pre-release only. It records no new validation result, "
            "no acceptance and no change of any feature's status") in version
    assert "The accepted guard itself has already stopped" not in members["start.sh"].decode()


def test_the_version_names_every_file_of_the_route_and_every_package_file_that_changed() -> None:
    """Nothing changed is passed off as the accepted build or as the previous package: each file is named with both
    checksums and what it has (not) been run with, and the launch line claims no more than was done."""
    build = builder()
    name, members = build.package_members(0, worktree=True)
    version = members["VERSION"].decode().splitlines()
    for path in ROUTE:
        packaged = sha256(members[f"app/{path}"])
        was = ("absent (a new file)" if path.endswith("device_artifacts.py")
               else sha256(build.git("show", f"{build.ACCEPTED}:{path}")))
        assert f"changed_file=app/{path} accepted_sha256={was} packaged_sha256={packaged}" in version, path
        assert any(line.startswith(f"changed_file_reason=app/{path}: ") and "device-artifacts route" in line
                   for line in version), path
        if path in REAL_RECORD:
            # what read and judged the reference device's real record on 2026-11-15 was revision 5: the statement
            # holds only while these are the same bytes that revision 5 packaged
            assert packaged == sha256(build.git("show", f"{build.REAL_EVIDENCE_PACKAGE_COMMIT}:{path}")), path
            run_with = next(line for line in version if line.startswith(f"changed_file_run_with=app/{path}: "))
            assert ("synthetic tests (temporary directories, synthetic files, stand-ins); and once on the reference "
                    "device, with revision 5 and these same bytes:") in run_with, path
            assert "no camera, no model loaded, no launch" in run_with and "Never on another device" in run_with
        else:
            assert (f"changed_file_run_with=app/{path}: synthetic tests on the reference device only (temporary "
                    "directories, synthetic files, stand-ins); never on another device, and no run with a camera or "
                    "a model has used it") in version, path
    # the package's own files against revision 7's (the previous one), and against revision 5's (which ran for real)
    assert build.PREVIOUS_PACKAGE_NAME == "sentinel-personal-beta-949da67-mod191a193-r7"
    assert build.REAL_EVIDENCE_PACKAGE_NAME == "sentinel-personal-beta-949da67-mod79c18d4-r5"
    for commit in (build.PREVIOUS_PACKAGE_COMMIT, build.REAL_EVIDENCE_PACKAGE_COMMIT):
        assert build.git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip() == commit
    assert build.git("merge-base", "--is-ancestor", build.REAL_EVIDENCE_PACKAGE_COMMIT, build.PREVIOUS_PACKAGE_COMMIT) == b""
    own = {path: content for path, content in members.items()  # LICENSE is the repository's file, named apart
           if not path.startswith("app/") and path not in ("VERSION", "SHA256SUMS", "LICENSE")}
    previous = {path[len(build.PACKAGE_SOURCE) + 1:]: blob
                for path, blob in build.blobs(build.PREVIOUS_PACKAGE_COMMIT, build.PACKAGE_SOURCE).items()}
    differs = build.differing(previous, own)
    assert differs == sorted(build.PACKAGE_FILES_CHANGED) and set(NEW_PACKAGE_FILES) <= set(differs)
    assert (f"package_files_changed_since={build.PREVIOUS_PACKAGE_NAME} (package_commit {build.PREVIOUS_PACKAGE_COMMIT}): "
            f"{len(differs)} of the package's own files (below); VERSION and SHA256SUMS are written anew for every "
            "package") in version
    for path in differs:
        was = ("absent (a new file)" if path in NEW_PACKAGE_FILES else
               sha256(build.git("show", f"{build.PREVIOUS_PACKAGE_COMMIT}:{build.PACKAGE_SOURCE}/{path}")))
        assert f"package_file_changed={path} previous_sha256={was} packaged_sha256={sha256(own[path])}" in version, path
        assert any(line.startswith(f"package_file_changed_reason={path}: ") and len(line) > len(path) + 40 for line in version)
    assert differs == ["QUICKSTART.md", "README.md", "THIRD_PARTY_NOTICES.md"]  # documents only: every script, helper and pin is revision 7's
    for path in ("stop.sh", "check.sh", "configure.sh", "lib/hostprep.py", "lib/features.py", "templates/live.yaml",
                 "lib/artifacts.py", "lib/admission_check.py", "manifest/assets.json", "manifest/requirements-pypi.txt",
                 "manifest/requirements-jetson.txt", "start.sh", "install.sh", "uninstall.sh", "setup-env.sh",
                 "fetch-assets.sh", "build-llama.sh", "build-engine.sh", "guarded-build.sh", "record-artifacts.sh",
                 "identity.sh", "preflight.sh", "lib/beta.py", "lib/common.sh", "lib/guarded_build.py", "lib/guarded_run.py"):
        assert path in own and not any(line.startswith(f"package_file_changed={path} ") for line in version), path
    # computed by the builder, not declared: what revision 5, the one revision that ran for real, did not have
    evidence = {path[len(build.PACKAGE_SOURCE) + 1:]: blob
                for path, blob in build.blobs(build.REAL_EVIDENCE_PACKAGE_COMMIT, build.PACKAGE_SOURCE).items()}
    assert build.differing(evidence, own) == sorted(SINCE_REAL_EVIDENCE)
    assert (f"package_files_without_real_device_evidence={len(SINCE_REAL_EVIDENCE)} of the package's own files differ from "
            f"{build.REAL_EVIDENCE_PACKAGE_NAME} (package_commit {build.REAL_EVIDENCE_PACKAGE_COMMIT}), the only revision "
            f"that ran for real, or are new since: {' '.join(sorted(SINCE_REAL_EVIDENCE))}; and LICENSE, which no "
            "earlier revision held") in version
    # the licence travels with the package: the repository's own file, named with its checksum; nothing private is said
    assert members["LICENSE"] == (REPO / "LICENSE").read_bytes() and members["LICENSE"].startswith(b"MIT License\n")
    assert f"{sha256(members['LICENSE'])}  LICENSE\n" in members["SHA256SUMS"].decode()
    licence = next(line for line in version if line.startswith("license="))
    assert licence.startswith(f"license=MIT for the files in this package (LICENSE, sha256 {sha256(members['LICENSE'])}:")
    assert "are not in this package and keep their own licences" in licence and "AGPL-3.0" in licence
    assert "status=experimental tester pre-release; validates and admits nothing; not accepted for production" in version
    for gone in ("private", "redistribution", "maintainer's testers"):
        assert gone not in "\n".join(version), gone
    # the recipe and the record's tools are revision 5's: the pins, the manifest and the record code did not change
    for path in NEW_PACKAGE_FILES:
        assert f"package_file_changed={path} previous_sha256=absent (a new file) packaged_sha256=" in "\n".join(version)
    launch = next(line for line in version if line.startswith("launch="))
    assert "REFUSES on another Jetson" in launch and "validates nothing about that device's resources" in launch
    assert "resource validation pending" in launch
    verified = next(line for line in version if line.startswith("route_verified="))
    assert "with revision 5, once, on the reference device's own real builds" in verified
    assert "check.sh stopped there at the missing enrollment" in verified and "have NEVER RUN" in verified
    assert "NEVER RUN ON ANOTHER DEVICE; never with a camera" in verified
    # what ran for real (revision 5, one device) is kept apart from how this revision was verified (offline only)
    evidence = next(line for line in version if line.startswith("real_device_evidence="))
    assert evidence.startswith("real_device_evidence=revision 5 only, on the reference device, without a camera")
    assert "187 s, started with 6,950,408,192 B MemFree, lowest 2,622,537,728 B, no swap used" in evidence
    assert "the least memory and the time an export needs are not known" in evidence
    offline = next(line for line in version if line.startswith("this_revision_verified="))
    assert offline.startswith("this_revision_verified=offline only") and "on any device" in offline
    assert version.index(evidence) < version.index(offline) == len(version) - 1
    for claim in ("validated on", "verified on another", "admitted", "passes on any", "CANNOT LAUNCH ON ANOTHER JETSON YET"):
        assert claim not in "\n".join(version).replace("validates and admits nothing", ""), claim
    # the guide a tester reads names this package, and says what is not the accepted build's and what is not validated
    released, _ = build.package_members(build.DEFAULT_REVISION, worktree=True)
    released = released[:-len("-worktree")]
    assert released.endswith("-r8") and build.DEFAULT_REVISION == 8
    guide = members["QUICKSTART.md"].decode()
    assert f"tar -xzf {released}.tar.gz && cd {released}\n" in guide
    assert "**Five application files are not the accepted build's.**" in guide and "stops every `llama-server`" not in guide
    for said_in_the_guide in ("**never been run on another device**", "**It does not validate your device.**",
                              "**The launch gates and the guard's limits are limits, not evidence**",
                              "**Only typing START opens the camera.**", "bash record-artifacts.sh",
                              "bash check.sh --device-artifacts ~/sentinel-beta-data/config/device-artifacts.json",
                              "bash ~/sentinel-beta/package/start.sh --device-artifacts ~/sentinel-beta-data/config/device-artifacts.json",
                              "bash ~/sentinel-beta/package/stop.sh", "NOT VERIFIED", "--accept-unverified-platform"):
        assert said_in_the_guide in guide, said_in_the_guide
    assert "CANNOT LAUNCH ON ANOTHER JETSON YET" not in guide and "for when section 1 no longer applies" not in guide


@pytest.mark.parametrize(("table", "change", "told"), [
    ("CHANGED_SINCE_ACCEPTED", "drop:src/sentinel/cli.py",
     "the application files that differ from the accepted commit 949da67 are not exactly the ones this package declares "
     "as changed; changed but not declared: src/sentinel/cli.py"),
    ("CHANGED_SINCE_ACCEPTED", "drop:src/sentinel/device_artifacts.py", "changed but not declared: src/sentinel/device_artifacts.py"),
    ("CHANGED_SINCE_ACCEPTED", "add:src/sentinel/config.py", "declared but identical: src/sentinel/config.py"),
    ("PACKAGE_FILES_CHANGED", "drop:README.md",
     "the package's own files that differ from sentinel-personal-beta-949da67-mod191a193-r7 are not exactly the ones "
     "this package declares as changed; changed but not declared: README.md"),
    ("PACKAGE_FILES_CHANGED", "drop:QUICKSTART.md", "changed but not declared: QUICKSTART.md"),
    ("PACKAGE_FILES_CHANGED", "add:stop.sh", "declared but identical: stop.sh"),
])
def test_the_builder_refuses_a_change_it_does_not_declare_and_a_declaration_without_a_change(
        monkeypatch: pytest.MonkeyPatch, table: str, change: str, told: str) -> None:
    build = builder()
    declared = dict(getattr(build, table))
    action, _, path = change.partition(":")
    if action == "drop":
        del declared[path]
    else:
        declared[path] = ("a declaration without a change", "nothing") if table == "CHANGED_SINCE_ACCEPTED" else "nothing"
    monkeypatch.setattr(build, table, declared)
    with pytest.raises(SystemExit) as stop:
        build.package_members(1, worktree=True)
    assert told in str(stop.value)
    with pytest.raises(SystemExit) as stop:  # the rule itself, as the archive form applies it too
        build.declared_exactly(["a", "b"], {"b": "", "c": ""}, "the files")
    assert str(stop.value) == ("STOP: the files are not exactly the ones this package declares as changed; changed but "
                               "not declared: a; declared but identical: c")
    assert build.declared_exactly(["a", "b"], {"a": "", "b": ""}, "the files") is None


def test_only_a_package_without_changed_files_carries_the_accepted_name(monkeypatch: pytest.MonkeyPatch) -> None:
    build = builder()
    real = build.blobs

    def as_packaged(commit: str, *paths: str) -> dict:
        if commit != build.ACCEPTED:
            return real(commit, *paths)
        return {path: build.blob_id((REPO / path).read_bytes())
                for path in build.working_files(build.APP_TREE) + list(build.GUARD_FILES)}

    monkeypatch.setattr(build, "blobs", as_packaged)  # as if the accepted commit held exactly what is packaged
    monkeypatch.setattr(build, "CHANGED_SINCE_ACCEPTED", {})
    name, members = build.package_members(3, worktree=True)
    version = members["VERSION"].decode()
    assert name == "sentinel-personal-beta-949da67-r3-worktree" and "-mod" not in name
    assert "application=the accepted code, unchanged (" in version and "changed_file=" not in version
    monkeypatch.undo()
    name, members = build.package_members(3, worktree=True)  # as this branch is: the mark is part of the name
    assert "-mod" in name and "application=the accepted code, unchanged" not in members["VERSION"].decode()


# ---------------------------------------------------------------- an archive is written once, and the same again

def test_the_builder_never_replaces_an_archive_and_builds_the_same_bytes_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Earlier archives are kept: a name that is already in the output directory stops the build, whatever is in
    it. The same members give the same archive, byte for byte, in another directory."""
    import tarfile

    build = builder()
    members = {"VERSION": b"a-package-r9\n", "install.sh": b"#!/bin/bash\n", "lib/beta.py": b"pass\n"}
    monkeypatch.setattr(build, "package_members", lambda revision: (f"a-package-r{revision}", dict(members)))
    first = build.build(tmp_path / "one", 9)
    assert first == tmp_path / "one" / "a-package-r9.tar.gz" and first.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "one").stat().st_mode & 0o777 == 0o700
    sidecar = tmp_path / "one" / "a-package-r9.tar.gz.sha256"
    assert sidecar.read_text() == f"{sha256(first.read_bytes())}  a-package-r9.tar.gz\n"
    kept = first.read_bytes()
    members["install.sh"] = b"#!/bin/bash\n# changed\n"
    with pytest.raises(SystemExit) as stop:
        build.build(tmp_path / "one", 9)
    assert "already exists and is not replaced" in str(stop.value) and first.read_bytes() == kept
    first.unlink()  # the checksum file alone also stops it: nothing of an earlier archive is overwritten
    with pytest.raises(SystemExit):
        build.build(tmp_path / "one", 9)
    assert not first.exists() and sidecar.read_text().endswith("  a-package-r9.tar.gz\n")
    members["install.sh"] = b"#!/bin/bash\n"
    again = build.build(tmp_path / "two", 9)
    assert again.read_bytes() == kept  # reproducible: the same inputs, the same archive
    with tarfile.open(again) as archive:
        entries = archive.getmembers()
    assert [entry.name for entry in entries] == ["a-package-r9/VERSION", "a-package-r9/install.sh", "a-package-r9/lib/beta.py"]
    assert all(entry.isfile() and entry.uid == entry.gid == 0 and entry.uname == entry.gname == "" for entry in entries)
    assert {entry.name: entry.mode for entry in entries} == {
        "a-package-r9/VERSION": 0o644, "a-package-r9/install.sh": 0o755, "a-package-r9/lib/beta.py": 0o644}
    assert len({entry.mtime for entry in entries}) == 1
