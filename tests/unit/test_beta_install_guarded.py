"""The package's guarded builds: guarded-build.sh, its front end and the monitor (session 116, part 47).

Stand-ins only. The two build scripts of the installed package are replaced by small scripts that load nothing and
build nothing; the monitor is the package's own file, unchanged. The stops are the package's constants: where a case
needs a stop to happen within seconds, the front end's module is loaded with smaller figures from outside (the
packaged file has no switch for that), and the constants themselves are pinned in the first test. No real build, no
export, no GPU, no model, no camera, no network, nothing that fills or frees memory. A pass says the wiring, the
refusals and the cleanup hold for these stand-ins; it measures no build and no device.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from beta_install_fixture import REPO, World, said

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="process identity is read from /proc (Linux)")

GIB = 1024 ** 3
MONITOR_SHA256 = "f7f23e4ae5db944d37bcc8db26ec53c4c96a2e9929566bfe0ddc8cb1b721045b"  # the rehearsal's monitor
BUILD = r'''#!/bin/bash
# STAND-IN build step (tests of guarded-build.sh only): loads nothing, builds nothing, downloads nothing.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; PREFIX="$(cd "$HERE/.." && pwd -P)"
NAME="$(basename "${BASH_SOURCE[0]}")"
echo "$$" >> "$FIXTURE/standin-pids"
IFS= read -r ANSWER; READ=$?
printf '%s args=[%s] answer=[%s] read=%s tmpdir=[%s]\n' "$NAME" "$*" "$ANSWER" "$READ" "${TMPDIR-}" >> "$FIXTURE/standin-calls"
if [ "$NAME" = build-engine.sh ]; then
  [ "$ANSWER" = BUILD ] || { printf '\a' >&2; echo "STOP: nothing was built" >&2; exit 1; }
  OUT="$PREFIX/assets/engine/yolov8n.engine"
else
  OUT="$PREFIX/assets/llama.cpp/build/bin/llama-server"
fi
case "$(cat "$FIXTURE/behavior")" in
  ok) echo "stand-in: building with $NAME"; mkdir -p "$(dirname "$OUT")"; printf 'a stand-in output' > "$OUT"; echo "Built: $OUT"; exit 0 ;;
  fail) printf '\a' >&2; echo "STOP: the export failed (stand-in)" >&2; exit 1 ;;
  hold) sleep 300 & echo $! >> "$FIXTURE/standin-pids"; echo "stand-in: holding"; wait ;;
esac
'''
DRIVER = r'''import json, sys
sys.path.insert(0, sys.argv[1])
import guarded_build as front_end
over = json.loads(sys.argv[2])
for step, values in over.get("steps", {}).items():
    front_end.STEPS[step].update(values)
for name in ("TERM_GRACE_S", "KILL_GRACE_S"):
    if name in over:
        setattr(front_end, name, over[name])
sys.exit(front_end.main(sys.argv[3:]))
'''
LOW = {"steps": {"engine": {"start_free_bytes": 0, "script_min_free_bytes": 0, "min_free_bytes": 0,
                            "min_available_bytes": 64 * 1024 * 1024, "timeout_s": 60},
                 "llama": {"min_available_bytes": 64 * 1024 * 1024, "timeout_s": 60}},
       "TERM_GRACE_S": 3.0, "KILL_GRACE_S": 3.0}


def sealed_again(package: Path) -> None:
    lines = []
    for line in (package / "SHA256SUMS").read_text().splitlines():
        name = line.partition("  ")[2]
        lines.append(f"{hashlib.sha256((package / name).read_bytes()).hexdigest()}  {name}\n")
    (package / "SHA256SUMS").write_text("".join(lines))


class Guarded(World):
    """An installation whose two build scripts are stand-ins, ready for guarded-build.sh."""

    def __init__(self, tmp: Path) -> None:
        super().__init__(tmp)
        (self.bin / "sleep").symlink_to(shutil.which("sleep"))  # what the holding stand-in starts
        for name in ("build-engine.sh", "build-llama.sh"):
            (self.package / name).write_text(BUILD)
        sealed_again(self.package)
        installed = self.install()
        assert installed.returncode == 0, said(installed)
        self.python = os.path.realpath(self.bin / "python3")
        owned = subprocess.run([self.python, "-I", "-B", str(self.installed / "lib/beta.py"), "own", "--prefix",
                                str(self.prefix), "assets"], env=self.env, capture_output=True, text=True, timeout=60)
        assert owned.returncode == 0, said(owned)
        (self.prefix / "assets/engine").mkdir(parents=True)
        (self.prefix / "venv/bin").mkdir(parents=True)  # only its presence is asked about; nothing is run with it
        (self.prefix / "venv/bin/python").symlink_to(sys.executable)
        self.behave("ok")
        self.started: list[subprocess.Popen] = []

    def behave(self, behavior: str) -> None:
        self.fixture("behavior", behavior + "\n")

    def command_line(self, over: dict, *args: str) -> list[str]:
        return [self.python, "-I", "-B", "-c", DRIVER, str(self.installed / "lib"), json.dumps(over), *args,
                "--package", str(self.installed)]

    def front_end(self, over: dict, *args: str, stdin: str = "BUILD\n") -> subprocess.CompletedProcess:
        return subprocess.run(self.command_line(over, *args), env=self.env, input=stdin, capture_output=True, text=True,
                              timeout=120, cwd=self.tmp)

    def start(self, over: dict, *args: str) -> subprocess.Popen:
        """The front end as a process of its own (its own session, as in a terminal); BUILD is typed."""
        process = subprocess.Popen(self.command_line(over, *args), env=self.env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=self.tmp,
                                   start_new_session=True)
        self.started.append(process)
        process.stdin.write("BUILD\n")
        process.stdin.flush()
        return process

    def pids(self) -> list[int]:
        try:
            return [int(value) for value in (self.tmp / "standin-pids").read_text().split()]
        except OSError:
            return []

    def calls(self) -> list[str]:
        try:
            return (self.tmp / "standin-calls").read_text().splitlines()
        except OSError:
            return []

    def records(self, step: str) -> list[Path]:
        return sorted((self.data / "builds").glob(f"{step}-*"))

    def result(self, step: str) -> str:
        return (self.records(step)[-1] / "RESULT.txt").read_text()

    def close(self) -> None:
        for process in self.started:
            if process.poll() is None:
                process.kill()  # this test's own unreaped child
            process.wait(timeout=10)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        for pid in self.pids():  # a stand-in that a failing case left behind: only if it still works in this test's tree
            try:
                inside = os.readlink(f"/proc/{pid}/cwd").startswith(str(self.tmp) + os.sep)
            except OSError:
                continue
            if inside and running(pid):
                os.kill(pid, signal.SIGKILL)


def running(pid: int) -> bool:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return text[text.rindex(")") + 2:].split()[0] != "Z"


def wait_for(condition, seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return bool(condition())


@pytest.fixture
def guarded(tmp_path: Path):
    world = Guarded(tmp_path)
    yield world
    world.close()


def packaged(name: str):
    spec = importlib.util.spec_from_file_location(f"packaged_{name}", REPO / "packaging/beta-install/lib" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "packaging/beta-install/lib"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(REPO / "packaging/beta-install/lib"))
    return module


# ---------------------------------------------------------------- what is packaged

def test_the_stops_are_the_rehearsals_and_the_monitor_is_its_file_unchanged() -> None:
    front_end = packaged("guarded_build")
    engine, llama = front_end.STEPS["engine"], front_end.STEPS["llama"]
    # the export's stops: below 1 GiB MemFree, below 2 GiB MemAvailable, one hour; the script is the recipe's own
    assert (engine["script"], engine["min_free_bytes"], engine["min_available_bytes"], engine["timeout_s"]) == (
        "build-engine.sh", GIB, 2 * GIB, 3600)
    assert engine["stdin_text"] == "BUILD\n" and engine["script_min_free_bytes"] == 1500000000
    assert engine["start_free_bytes"] == 5 * GIB == 5368709120  # the chosen preparation figure, not a stop
    # the llama.cpp build as the rehearsal ran it: MemAvailable only, three hours
    assert (llama["script"], llama["min_free_bytes"], llama["min_available_bytes"], llama["timeout_s"]) == (
        "build-llama.sh", 0, 2 * GIB, 10800)
    assert llama["stdin_text"] is None and llama["start_free_bytes"] == 0
    assert (front_end.TERM_GRACE_S, front_end.KILL_GRACE_S) == (20.0, 10.0)
    assert front_end.EXIT == {"built": 0, "refused": 10, "active": 11, "stopped_by_monitor": 20, "failed": 21,
                              "stopped_by_operator": 22, "refused_at_monitor_start": 23, "monitor_failed": 24,
                              "leftovers": 5, "error": 50}
    source = (REPO / "packaging/beta-install/lib/guarded_build.py").read_text()
    for never in ("sudo ", "drop_caches", "systemctl", "os.kill(", "killpg", "os.environ[", "getenv"):
        assert never not in source, never  # no service, no memory release, no signal by number alone, no switch
    monitor = REPO / "packaging/beta-install/lib/guarded_run.py"
    assert hashlib.sha256(monitor.read_bytes()).hexdigest() == MONITOR_SHA256
    shell = (REPO / "packaging/beta-install/guarded-build.sh").read_text()
    assert "sudo " not in shell.replace("Nothing here uses sudo", "") and "systemctl" not in shell
    assert shell.index("beta_verify_package") < shell.index("beta_local_env") < shell.index('guarded_build.py" run')


@pytest.mark.parametrize(("argv", "started"), [
    (["/usr/bin/python3", "-I", "-B", "/p/lib/guarded_run.py", "--name", "x"], ("script", "guarded_run.py")),
    (["bash", "/p/package/build-engine.sh"], ("script", "build-engine.sh")),
    (["/bin/bash", "./build-llama.sh", "--jobs", "2"], ("script", "build-llama.sh")),
    (["python3", "-X", "dev", "-W", "ignore", "/p/lib/guarded_run.py"], ("script", "guarded_run.py")),
    (["/v/bin/python", "-B", "-m", "sentinel.cli", "run", "live.yaml"], ("module", "sentinel.cli")),
    (["python3", "-msentinel.cli"], ("module", "sentinel.cli")),
    (["bash", "-c", "sleep 9; : build-engine.sh guarded_run.py -m sentinel.cli run"], ("", "")),
    (["python3", "-c", "import time; time.sleep(9)  # build-engine.sh"], ("", "")),
    (["-bash"], ("", "")),
    (["grep", "-r", "build-engine.sh", "."], ("", "")),
    (["tail", "-f", "/d/builds/engine-x/engine.log"], ("", "")),
    ([], ("", "")),
])
def test_a_program_is_known_by_what_it_runs_not_by_words_in_a_command_line(argv: list[str], started: tuple[str, str]) -> None:
    assert packaged("guarded_build").started_as(argv) == started


# ---------------------------------------------------------------- a build that ends by itself

def test_a_guarded_engine_build_runs_the_script_itself_and_keeps_its_records(guarded: Guarded) -> None:
    result = guarded.front_end(LOW, "run", "--step", "engine")
    assert result.returncode == 0, said(result)
    assert "type BUILD to start the detector engine export under the monitor now" in result.stdout
    assert "stand-in: building with build-engine.sh" in result.stdout  # the step's own output reaches the terminal
    assert guarded.calls() == ["build-engine.sh args=[] answer=[BUILD] read=0 tmpdir=[]"]  # once, BUILD passed on
    (record,) = guarded.records("engine")
    assert record.stat().st_mode & 0o777 == 0o700 and (guarded.data / "builds").stat().st_mode & 0o777 == 0o700
    assert sorted(path.name for path in record.iterdir()) == [
        "RESULT.txt", "conditions.txt", "engine.log", "engine.mem.csv", "engine.result.json", "frontend.log", "monitor.txt"]
    monitor = json.loads((record / "engine.result.json").read_text())
    assert monitor["command"] == ["bash", str(guarded.installed / "build-engine.sh")]  # the script, with nothing added
    assert (monitor["exit_code"], monitor["stopped_by_monitor"], monitor["leftover_processes"]) == (0, None, [])
    low = LOW["steps"]["engine"]  # what the front end holds for the step is what the monitor is given
    assert monitor["thresholds"]["min_free_bytes"] == low["min_free_bytes"]
    assert monitor["thresholds"]["min_available_bytes"] == low["min_available_bytes"]
    assert (monitor["thresholds"]["timeout_s"], monitor["thresholds"]["term_grace_s"], monitor["thresholds"]["kill_grace_s"]) == (60.0, 3.0, 3.0)
    text = guarded.result("engine")
    engine = guarded.prefix / "assets/engine/yolov8n.engine"
    for line in ("OUTCOME=built", "reason=status 0 and the output is there", "exit_code=0 stopped_by_monitor=None",
                 "start_figure_bytes=0 (the preparation figure in use: chosen, not a measured minimum)",
                 f"output={engine} bytes=17 sha256={hashlib.sha256(b'a stand-in output').hexdigest()}",
                 "leftover_processes_of_this_run=none", "next=bash record-artifacts.sh",
                 "these_figures=whole-device readings on this device for this one run: not a requirement, not a profile"):
        assert line in text, line
    assert "memfree_at_start_bytes=" in text and "lowest_memavailable_bytes=" in text and "lowest_swapfree_bytes=" in text
    assert not any(running(pid) for pid in guarded.pids())
    assert list(guarded.home.iterdir()) == []  # nothing under the home directory
    state = json.loads((guarded.data / "builds/state.json").read_text())
    assert state["record"] == str(record) and state["monitor"] is None and state["leader"] is None
    status = guarded.front_end(LOW, "status")
    assert status.returncode == 0 and "guarded build running: no" in status.stdout and "OUTCOME=built" in status.stdout
    again = guarded.front_end(LOW, "run", "--step", "engine")  # the engine is there: refused before BUILD is asked
    assert again.returncode == 10 and "REFUSED no_engine_yet" in again.stdout and "type BUILD" not in again.stdout
    assert len(guarded.calls()) == 1 and len(guarded.records("engine")) == 1


def test_the_llama_step_is_given_its_job_count_and_no_typed_word(guarded: Guarded) -> None:
    result = guarded.front_end(LOW, "run", "--step", "llama", "--jobs", "2")
    assert result.returncode == 0, said(result)
    assert guarded.calls() == ["build-llama.sh args=[--jobs 2] answer=[] read=1 tmpdir=[]"]  # nothing on its input
    monitor = json.loads((guarded.records("llama")[0] / "llama.result.json").read_text())
    assert monitor["command"] == ["bash", str(guarded.installed / "build-llama.sh"), "--jobs", "2"]
    assert monitor["thresholds"]["min_free_bytes"] == 0  # MemFree is not watched for this step
    text = guarded.result("llama")
    assert "OUTCOME=built" in text and "next=bash guarded-build.sh engine" in text and "start_figure_bytes" not in text
    for wrong in (["--jobs", "9"], ["--jobs", "0"], ["--min-start-free-bytes", "2000000000"]):
        refused = guarded.front_end(LOW, "run", "--step", "llama", *wrong)
        assert refused.returncode == 1 and "STOP:" in refused.stderr, said(refused)
    assert guarded.front_end(LOW, "run", "--step", "engine", "--jobs", "2").returncode == 1
    assert len(guarded.calls()) == 1


def test_a_build_that_fails_is_reported_as_the_scripts_own_failure(guarded: Guarded) -> None:
    guarded.behave("fail")
    result = guarded.front_end(LOW, "run", "--step", "engine")
    assert result.returncode == 21, said(result)
    text = guarded.result("engine")
    assert "OUTCOME=failed" in text and "reason=status 1" in text and "output=none" in text
    assert "script_stop_line=STOP: the export failed (stand-in)" in text and "next=" not in text
    assert "leftover_processes_of_this_run=none" in text and not any(running(pid) for pid in guarded.pids())
    assert not (guarded.prefix / "assets/engine/yolov8n.engine").exists()


def test_nothing_starts_without_the_typed_word(guarded: Guarded) -> None:
    for answer in ("\n", "build\n", "yes\n", ""):
        result = guarded.front_end(LOW, "run", "--step", "engine", stdin=answer)
        assert result.returncode == 10 and "STOP: nothing was started" in result.stdout, said(result)
    assert guarded.pids() == [] and guarded.records("engine") == []


# ---------------------------------------------------------------- refusals before anything starts

def test_the_start_conditions_refuse_with_the_measurements_and_start_nothing(guarded: Guarded) -> None:
    """Through guarded-build.sh, with the package's own figures."""
    check = guarded.run("guarded-build.sh", "--check", "engine", package=guarded.installed)
    assert check.returncode in (0, 10), said(check)
    assert ("start figure in use 5368709120 B (5.00 GiB): a chosen preparation condition, not a measured minimum "
            "(the script's own minimum is 1500000000 B; the monitor stops the build below 1073741824 B)") in check.stdout
    assert "the monitor stops the build below 2147483648 B" in check.stdout
    assert "graphical_processes_running " in check.stdout and "closing a desktop is your own step" in check.stdout
    assert "CHECK=would_" in check.stdout and "nothing was started" in check.stdout
    refused = guarded.run("guarded-build.sh", "engine", "--min-start-free-bytes", "99999999999999999",
                          package=guarded.installed, stdin="BUILD\n")
    assert refused.returncode == 10, said(refused)
    assert "REFUSED before anything started: memfree_at_least_the_start_figure" in refused.stdout
    assert "Nothing ran" in refused.stdout and "type BUILD" not in refused.stdout
    (measurements,) = (guarded.data / "builds").glob("refused-engine-*.txt")
    assert "REFUSED memfree_at_least_the_start_figure: MemFree " in measurements.read_text()
    assert "info    meminfo MemTotal=" in measurements.read_text()
    assert guarded.pids() == [] and guarded.records("engine") == []  # no script ran, no run was recorded
    below = guarded.run("guarded-build.sh", "engine", "--min-start-free-bytes", "1000", package=guarded.installed)
    assert below.returncode == 1 and "may not be below build-engine.sh's own minimum (1500000000 B)" in below.stderr
    for wrong in (["nonsense"], [], ["engine", "--force"], ["--status", "extra"], ["engine", "--jobs", "two"]):
        usage = guarded.run("guarded-build.sh", *wrong, package=guarded.installed)
        assert usage.returncode in (1, 2) and guarded.pids() == [], said(usage)
    status = guarded.run("guarded-build.sh", "--status", package=guarded.installed)
    assert status.returncode == 0 and "guarded build running: no" in status.stdout and "latest records: none" in status.stdout
    stop = guarded.run("guarded-build.sh", "--stop", package=guarded.installed)
    assert stop.returncode == 0 and "nothing to stop" in stop.stdout
    extracted = guarded.run("guarded-build.sh", "--check", "engine")  # the archive itself: not an installation
    assert extracted.returncode == 1 and "run this from an installed package" in extracted.stderr


def test_the_shell_entry_runs_a_step_with_the_packages_own_stops_and_locations(guarded: Guarded) -> None:
    """The whole path with nothing lowered: guarded-build.sh, the front end, the monitor, the (stand-in) script."""
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < 3 * GIB:
        pytest.skip("this device has less than 3 GiB of MemAvailable now: the package's own stop would refuse")
    result = guarded.run("guarded-build.sh", "llama", "--jobs", "2", package=guarded.installed, stdin="BUILD\n")
    assert result.returncode == 0, said(result)
    assert guarded.calls() == [f"build-llama.sh args=[--jobs 2] answer=[] read=1 tmpdir=[{guarded.prefix / 'tmp'}]"]
    monitor = json.loads((guarded.records("llama")[0] / "llama.result.json").read_text())
    assert monitor["thresholds"] == {"min_available_bytes": 2 * GIB, "min_free_bytes": 0, "interval_s": 0.2,
                                     "max_gap_s": 0.5, "term_grace_s": 20.0, "kill_grace_s": 10.0, "timeout_s": 10800.0}
    text = guarded.result("llama")
    assert "monitor_stops=memfree_below_0_B memavailable_below_2147483648_B timeout_10800_s" in text
    assert (guarded.prefix / "cache/pip").is_dir()  # the installation's own locations were set before it started


def test_a_model_server_a_runtime_or_another_build_refuses_the_start_and_is_never_signalled(guarded: Guarded) -> None:
    standins = guarded.tmp / "standins"
    (standins / "mod/sentinel").mkdir(parents=True)
    shutil.copy2("/bin/sleep", standins / "llama-server")
    (standins / "mod/sentinel/__init__.py").write_text("")
    (standins / "mod/sentinel/cli.py").write_text("import time\ntime.sleep(300)\n")
    (standins / "build-engine.sh").write_text("sleep 300\n")
    cases = [("model server llama-server", [str(standins / "llama-server"), "300"], standins),
             ("Sentinel runtime", [guarded.python, "-B", "-m", "sentinel.cli", "run"], standins / "mod"),
             ("build-engine.sh", ["bash", str(standins / "build-engine.sh")], standins)]
    for what, command, cwd in cases:
        other = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
        guarded.started.append(other)
        time.sleep(0.5)
        assert other.poll() is None
        result = guarded.front_end(LOW, "run", "--step", "engine")
        assert result.returncode == 10, said(result)
        assert f"REFUSED no_model_server_runtime_or_other_build: running: ['{other.pid}({what})']" in result.stdout
        assert other.poll() is None  # found, named, and left alone
        other.kill()
        other.wait(timeout=10)
    mention = subprocess.Popen(["bash", "-c", "sleep 300; : llama-server build-engine.sh guarded_run.py -m sentinel.cli run"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    guarded.started.append(mention)
    time.sleep(0.5)
    check = guarded.front_end(LOW, "run", "--step", "engine", "--check")
    assert check.returncode == 0 and "ok      no_model_server_runtime_or_other_build: running: none" in check.stdout, said(check)
    assert guarded.pids() == []  # no build script was started in any of this


# ---------------------------------------------------------------- stops, and what is cleaned up

def test_the_monitors_time_limit_ends_everything_the_step_started(guarded: Guarded) -> None:
    guarded.behave("hold")
    over = {**LOW, "steps": {**LOW["steps"], "engine": {**LOW["steps"]["engine"], "timeout_s": 2}}}
    result = guarded.front_end(over, "run", "--step", "engine")
    assert result.returncode == 20, said(result)
    assert len(guarded.pids()) == 2  # the script and the process it started
    assert wait_for(lambda: not any(running(pid) for pid in guarded.pids()), 5)
    text = guarded.result("engine")
    assert "OUTCOME=stopped_by_monitor" in text and "reason=timeout" in text and "stopped_by_monitor=timeout" in text
    assert "leftover_processes_of_this_run=none" in text and "output=none" in text
    monitor = json.loads((guarded.records("engine")[0] / "engine.result.json").read_text())
    assert monitor["leftover_processes"] == [] and any("SIGTERM to 2 owned process(es)" in note for note in monitor["notes"])


def test_a_stop_from_a_second_shell_ends_the_step_through_the_monitor(guarded: Guarded) -> None:
    guarded.behave("hold")
    process = guarded.start(LOW, "run", "--step", "engine")
    assert wait_for(lambda: len(guarded.pids()) == 2), guarded.pids()
    second = guarded.front_end(LOW, "run", "--step", "engine", "--check")  # one guarded build at a time
    assert second.returncode == 11 and "a guarded build is already running" in second.stdout
    status = guarded.front_end(LOW, "status")
    assert "guarded build running: yes" in status.stdout
    stop = guarded.front_end(LOW, "stop")
    assert stop.returncode == 0 and "the request was taken" in stop.stdout, said(stop)
    assert process.wait(timeout=30) == 22
    assert wait_for(lambda: not any(running(pid) for pid in guarded.pids()), 5)
    text = guarded.result("engine")
    assert "OUTCOME=stopped_by_operator" in text and "operator_stop_utc=" in text
    assert "leftover_processes_of_this_run=none" in text
    assert not (guarded.data / "builds/STOP-REQUEST").exists()
    after = guarded.front_end(LOW, "stop")
    assert after.returncode == 0 and "nothing to stop" in after.stdout


def test_a_lost_terminal_does_not_end_the_step_and_an_interrupt_stops_it_cleanly(guarded: Guarded) -> None:
    guarded.behave("hold")
    process = guarded.start(LOW, "run", "--step", "engine")
    assert wait_for(lambda: len(guarded.pids()) == 2), guarded.pids()
    process.send_signal(signal.SIGHUP)  # the terminal went away
    time.sleep(1.0)
    assert process.poll() is None and all(running(pid) for pid in guarded.pids())  # still watched, still running
    process.send_signal(signal.SIGINT)  # Ctrl-C, once
    assert process.wait(timeout=30) == 22
    assert wait_for(lambda: not any(running(pid) for pid in guarded.pids()), 5)
    assert "OUTCOME=stopped_by_operator" in guarded.result("engine")


def test_a_monitor_that_dies_leaves_nothing_of_its_step_running(guarded: Guarded) -> None:
    guarded.behave("hold")
    process = guarded.start(LOW, "run", "--step", "engine")
    assert wait_for(lambda: len(guarded.pids()) == 2), guarded.pids()
    state = guarded.data / "builds/state.json"
    assert wait_for(lambda: state.exists() and json.loads(state.read_text()).get("leader"))
    monitor = json.loads(state.read_text())["monitor"]
    stat = Path(f"/proc/{monitor['pid']}/stat").read_text()
    assert int(stat[stat.rindex(")") + 2:].split()[19]) == monitor["start"]  # still that process, by its start time
    os.kill(monitor["pid"], signal.SIGKILL)
    assert process.wait(timeout=40) == 24
    assert wait_for(lambda: not any(running(pid) for pid in guarded.pids()), 5)
    text = guarded.result("engine")
    assert "OUTCOME=monitor_failed" in text and "leftover_processes_of_this_run=none" in text
    assert "ending its session's processes" in (guarded.records("engine")[0] / "frontend.log").read_text()

