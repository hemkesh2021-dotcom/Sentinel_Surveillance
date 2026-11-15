"""The memory guard signals only what its own run started (session 116, part 40): ownership by descent, never by
name; a number that passed to a later process; a runtime or the guard itself failing. Synthetic /proc trees, small
real processes (none of them a model, a camera or the runtime), and the guard as a real process of its own.

The real cases start processes *named* llama-server (a few lines of Python that sleep). They are skipped when a
llama-server of this user is already running: the guard would refuse to start, and nothing here may meet a real one.
"""

from __future__ import annotations

import errno
import functools
import importlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
STARTING = {"run": "starting", "startup": {"scene_problem": None, "detector_problem": None}}
STOPPED = {"run": "stopped", "shutdown": {"all_stopped": True}, "database_closed": True}


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return importlib.import_module("step5_guard")


# ---------------------------------------------------------------- a synthetic /proc


def put(proc: Path, pid: int, comm: str, ppid: int, start: int, state: str = "S", pgrp: int | None = None) -> None:
    """One process as /proc/<pid>/stat shows it (the fields the guard reads; the rest zero)."""
    folder = proc / str(pid)
    folder.mkdir(parents=True, exist_ok=True)
    pgrp = pid if pgrp is None else pgrp
    (folder / "stat").write_text(f"{pid} ({comm}) {state} {ppid} {pgrp} {pgrp} " + "0 " * 15 + f"{start} 0 0\n")


def gone(proc: Path, pid: int) -> None:
    (proc / str(pid) / "stat").unlink()
    (proc / str(pid)).rmdir()


class Handles:
    """Stand-ins for process handles: a real descriptor each (the guard closes it), every send recorded."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, int]] = []
        self._pids: dict[int, int] = {}

    def open(self, pid: int) -> int:
        handle = os.open(os.devnull, os.O_RDONLY)
        self._pids[handle] = pid
        return handle

    def send(self, handle: int, signum: int) -> None:
        self.sent.append((self._pids[handle], signum))


def ownership(mod, proc: Path, handles: Handles, collected: list | None = None, subreaper=None):
    state = {"on": False, "calls": []}

    def fake_subreaper(enable=None):
        state["calls"].append(enable)
        if enable is not None:
            state["on"] = enable
        return state["on"]

    send = functools.partial(mod.send_to_process, open_handle=handles.open, send_handle=handles.send)
    collect = (lambda pid, flags: collected.append(pid)) if collected is not None else (lambda pid, flags: None)
    own = mod.Ownership(proc, pid=100, subreaper=subreaper or fake_subreaper, send=send, collect=collect)
    return own, state


@pytest.fixture
def proc(tmp_path) -> Path:
    root = tmp_path / "proc"
    put(root, 1, "systemd", 0, 1)
    put(root, 90, "bash", 1, 5)
    put(root, 100, "python", 90, 10)  # the guard
    put(root, 110, "python", 100, 11)  # something the guard's process had before the launch
    (root / "self").mkdir()  # not a number: ignored
    return root


def test_a_process_is_read_with_its_whole_name_and_its_identity(mod, proc) -> None:
    put(proc, 300, "odd) (name", 100, 77, state="R", pgrp=250)
    assert mod.process_info(300, proc) == {"pid": 300, "comm": "odd) (name", "state": "R", "ppid": 100, "pgrp": 250,
                                           "start": 77}
    assert mod.process_info(301, proc) is None  # gone
    assert mod.process_info(300, proc, uid=os.getuid() + 1) is None  # another user's is never this run's
    (proc / "302").mkdir()
    (proc / "302" / "stat").write_text("302 (cut")
    assert mod.process_info(302, proc) is None and 302 not in mod.process_table(proc)
    assert sorted(mod.process_table(proc)) == [1, 90, 100, 110, 300]


def test_owned_is_what_descends_from_the_guard_since_the_launch_and_nothing_else(mod, proc) -> None:
    handles = Handles()
    own, state = ownership(mod, proc, handles)
    assert own.owned() == [] and own.signal({"pid": 110, "start": 11}, signal.SIGTERM) is False  # before begin
    assert own.begin() is None and state["on"] is True
    put(proc, 200, "python", 100, 20)  # the runtime
    own.leader = 200
    put(proc, 210, "llama-server", 200, 30)  # its scene server, in a session of its own
    put(proc, 600, "llama-server", 1, 40)  # somebody else's, started during the run
    put(proc, 610, "llama-server", 90, 41)  # another, from the shell that started the guard
    assert [info["pid"] for info in own.owned()] == [200, 210]  # 110 was there before; 600 and 610 never descend
    assert [info["pid"] for info in own.not_owned("llama-server")] == [600, 610]
    gone(proc, 200)  # the runtime exits; the kernel hands its server to the guard
    put(proc, 210, "llama-server", 100, 30)
    assert [(info["pid"], info["start"], info["comm"]) for info in own.owned()] == [(210, 30, "llama-server")]
    for other in ({"pid": 600, "start": 40}, {"pid": 610, "start": 41}, {"pid": 110, "start": 11}, {"pid": 90, "start": 5}):
        assert own.signal(other, signal.SIGKILL) is False
    assert handles.sent == []
    assert own.signal({"pid": 210, "start": 30, "comm": "llama-server"}, signal.SIGTERM) is True
    assert handles.sent == [(210, signal.SIGTERM)]
    own.end()
    assert state["on"] is False and state["calls"] == [None, True, False]  # as it was before
    assert own.owned() == [] and own.signal({"pid": 210, "start": 30}, signal.SIGTERM) is False
    assert handles.sent == [(210, signal.SIGTERM)]


def test_a_number_that_passed_to_a_later_process_is_never_signalled(mod, proc) -> None:
    handles = Handles()
    own, _ = ownership(mod, proc, handles)
    assert own.begin() is None
    put(proc, 210, "llama-server", 100, 30)
    server = own.owned()[0]
    assert (server["pid"], server["start"]) == (210, 30)
    # it exits and is collected; its number goes to somebody else's llama-server
    put(proc, 210, "llama-server", 1, 95)
    assert own.owned() == [] and own.signal(server, signal.SIGTERM) is False and handles.sent == []
    assert [(info["pid"], info["start"]) for info in own.not_owned("llama-server")] == [(210, 95)]
    # even handed to the guard (a later child with the same number): the identity is the number AND the start time
    put(proc, 210, "llama-server", 100, 95)
    assert own.signal(server, signal.SIGTERM) is False and handles.sent == []
    # the last step alone: the identity is read again after the handle is taken
    assert mod.send_to_process(server, signal.SIGKILL, proc, 100, handles.open, handles.send) is False
    put(proc, 210, "llama-server", 100, 30, state="Z")  # ended, not collected: nothing to signal
    assert mod.send_to_process(server, signal.SIGKILL, proc, 100, handles.open, handles.send) is False
    gone(proc, 210)
    assert mod.send_to_process(server, signal.SIGKILL, proc, 100, handles.open, handles.send) is False
    assert handles.sent == []


def test_a_family_is_never_joined_through_a_number_that_was_reused(mod, proc) -> None:
    own, _ = ownership(mod, proc, Handles())
    assert own.begin() is None
    put(proc, 300, "sh", 100, 80)  # owned
    put(proc, 400, "llama-server", 300, 60)  # says its parent is 300, but started before 300 did: not 300's child
    put(proc, 410, "llama-server", 300, 85)
    assert [info["pid"] for info in own.owned()] == [300, 410]
    assert [info["pid"] for info in own.not_owned("llama-server")] == [400]
    table = mod.process_table(proc)
    assert [info["pid"] for info in mod.descendants(table, 100)] == [110, 300, 410]
    assert [info["pid"] for info in mod.descendants(table, 100, skip={(300, 80)})] == [110]  # and all below it
    assert mod.descendants(table, 999) == []


def test_ended_processes_handed_to_the_guard_are_collected_but_never_the_runtime(mod, proc) -> None:
    collected: list[int] = []
    own, _ = ownership(mod, proc, Handles(), collected)
    assert own.begin() is None
    put(proc, 200, "python", 100, 20, state="Z")  # the runtime, ended: its own Popen collects it
    own.leader = 200
    put(proc, 210, "llama-server", 100, 30, state="Z")
    put(proc, 220, "sh", 100, 31)
    put(proc, 230, "sleep", 220, 32, state="Z")  # another process's child: not the guard's to collect
    put(proc, 110, "python", 100, 11, state="Z")  # there before the launch: not this run's
    assert [info["pid"] for info in own.owned()] == [220] and collected == [210]
    assert own.not_owned("llama-server") == []  # an ended process is not a running llama-server


def test_a_server_whose_first_thread_ended_still_runs_until_its_other_threads_are_gone(mod, proc) -> None:
    """``stat`` shows the first thread only: it can read Z while the others still hold the memory and the port. The
    cleanup must go on waiting, as it did when it waited for the process to leave /proc."""
    results = [(0, 0), (210, 0)]
    own, _ = ownership(mod, proc, Handles())
    own._collect = lambda pid, flags: results.pop(0)
    assert own.begin() is None
    put(proc, 210, "llama-server", 100, 30, state="Z")
    for thread in ("210", "211", "212"):
        (proc / "210" / "task" / thread).mkdir(parents=True)
    put(proc, 600, "llama-server", 1, 40, state="Z")
    (proc / "600" / "task" / "600").mkdir(parents=True)
    (proc / "600" / "task" / "601").mkdir()
    info = mod.process_info(210, proc)
    assert mod.still_running(info, proc) is True and [item["pid"] for item in own.owned()] == [210]
    assert [item["pid"] for item in own.not_owned("llama-server")] == [600]  # somebody else's, likewise still there
    assert len(results) == 2  # nothing was collected while threads were left
    (proc / "210" / "task" / "211").rmdir()
    (proc / "210" / "task" / "212").rmdir()
    assert mod.still_running(info, proc) is False
    assert [item["pid"] for item in own.owned()] == [210] and results == [(210, 0)]  # asked, and not collectable yet
    assert own.owned() == [] and results == []  # collected: gone


@pytest.mark.parametrize("failure", ["raises", "stays_off", "no_guard_in_proc"])
def test_ownership_that_cannot_be_established_is_a_label_and_signals_nothing(mod, proc, failure) -> None:
    handles = Handles()
    calls = []

    def subreaper(enable=None):
        calls.append(enable)
        if failure == "raises":
            raise OSError(errno.EINVAL, "prctl")
        return failure == "no_guard_in_proc" and enable is not False and len(calls) > 1

    if failure == "no_guard_in_proc":
        gone(proc, 100)
    own, _ = ownership(mod, proc, handles, subreaper=subreaper)
    assert own.begin() == "ownership_unavailable" and own.active is False
    put(proc, 210, "llama-server", 100, 30)
    assert own.owned() == [] and own.signal({"pid": 210, "start": 30}, signal.SIGTERM) is False and handles.sent == []
    if failure == "no_guard_in_proc":
        assert calls[-1] is False  # given back as it was


def test_without_process_handles_only_the_guards_own_child_is_signalled(mod, proc) -> None:
    killed = []

    def no_handle(pid):
        raise OSError(errno.ENOSYS, "pidfd_open")

    put(proc, 210, "llama-server", 100, 30)  # the guard's own child: its number is held until the guard collects it
    put(proc, 220, "llama-server", 210, 35)  # deeper: somebody else may collect it, so its number may pass on
    send = functools.partial(mod.send_to_process, proc=proc, own_pid=100, open_handle=no_handle,
                             send_handle=lambda *_: killed.append("handle"), kill=lambda pid, signum: killed.append((pid, signum)))
    assert send({"pid": 210, "start": 30}, signal.SIGTERM) is True
    assert send({"pid": 220, "start": 35}, signal.SIGTERM) is False
    assert send({"pid": 210, "start": 31}, signal.SIGTERM) is False
    assert killed == [(210, signal.SIGTERM)]

    def vanished(pid):
        raise ProcessLookupError(errno.ESRCH, "pidfd_open")

    assert mod.send_to_process({"pid": 210, "start": 30}, signal.SIGTERM, proc, 100, vanished, None, killed.append) is False
    assert killed == [(210, signal.SIGTERM)]


# ---------------------------------------------------------------- real processes


def wait_for(condition, seconds: float = 15.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("not within the wait")


def sleeper(seconds: int = 60, **kwargs) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"], **kwargs)


def test_a_real_process_is_signalled_only_under_its_own_identity(mod) -> None:
    child = sleeper()
    try:
        info = wait_for(lambda: mod.process_info(child.pid))
        assert info["ppid"] == os.getpid()
        assert mod.send_to_process({**info, "start": info["start"] + 1}, signal.SIGKILL) is False
        assert mod.send_to_process({**info, "start": info["start"] - 1}, signal.SIGKILL) is False
        time.sleep(0.2)
        assert child.poll() is None  # untouched
        assert mod.send_to_process(info, signal.SIGTERM) is True and child.wait(10) == -signal.SIGTERM
        assert mod.send_to_process(info, signal.SIGTERM) is False  # collected: gone
    finally:
        child.kill()
        child.wait()


def test_the_kernel_hands_an_orphan_to_the_guard_which_then_owns_it(mod) -> None:
    """A process started in a session of its own by a parent that exits (as the runtime starts its scene server):
    with ownership set up it becomes this process's child, and only then may it be signalled."""
    before = sleeper()  # this process's own, from before: never this run's
    own = mod.Ownership()
    was = mod.child_subreaper()
    try:
        assert own.begin() is None and mod.child_subreaper() is True
        parent = subprocess.Popen([sys.executable, "-c", textwrap.dedent("""
            import subprocess, sys
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            print(child.pid, flush=True)
        """)], stdout=subprocess.PIPE, text=True)
        own.leader = parent.pid
        orphan = int(parent.stdout.readline())
        parent.stdout.close()
        assert parent.wait(10) == 0
        info = wait_for(lambda: (mod.process_info(orphan) or {}).get("ppid") == os.getpid() and mod.process_info(orphan))
        assert info["pgrp"] == orphan  # its own session and group: no group signal would have reached it
        assert [(item["pid"], item["start"]) for item in own.owned()] == [(orphan, info["start"])]
        assert own.signal({"pid": before.pid, "start": mod.process_info(before.pid)["start"]}, signal.SIGKILL) is False
        assert own.signal(info, signal.SIGTERM) is True
        wait_for(lambda: mod.process_info(orphan) is None or not own.owned())  # ended, then collected by owned()
        assert own.owned() == [] and mod.process_info(orphan) is None and before.poll() is None
    finally:
        own.end()
        before.kill()
        before.wait()
    assert mod.child_subreaper() is was


def test_the_runtimes_group_is_signalled_only_while_the_runtime_is_not_collected(mod, tmp_path, monkeypatch) -> None:
    child = mod.Child([sys.executable, "-c", "pass"], tmp_path / "out", tmp_path / "err")
    wait_for(lambda: (mod.process_info(child.pid) or {}).get("state") == "Z")
    assert child.group_alive() is False and child.process.returncode is None  # seen ended, and not collected by that
    assert child.signal_group(0) is True  # still its number, and with it its group's
    assert child.poll() == 0
    sent = []
    monkeypatch.setattr(mod.os, "killpg", lambda pid, signum: sent.append((pid, signum)))
    assert child.signal_group(signal.SIGKILL) is False and sent == []  # the number may now be somebody else's group
    child.signal(signal.SIGKILL)  # and the runtime itself is signalled only while it runs
    assert child.group_alive() is False


SERVER = """#!{python}
import os, signal, sys, time
if sys.argv[2:] == ["stubborn"]:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1] + ".tmp", "w") as handle:
    handle.write(str(os.getpid()))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(120)
"""
RUNTIME = """
import json, os, signal, subprocess, sys, time
server, pidfile, mode = sys.argv[1:4]
stop = []
signal.signal(signal.SIGINT, lambda *_: stop.append(1))
with open(os.path.join(os.path.dirname(pidfile), "runtime.pid.tmp"), "w") as handle:
    handle.write(str(os.getpid()))
os.replace(handle.name, handle.name[:-4])
ends = time.monotonic() + 120  # never outlives a test by much, whatever went wrong
# as sentinel run starts its scene server: a session of its own
child = subprocess.Popen([server, pidfile] + (["stubborn"] if mode == "crash" else []), start_new_session=True)
while not os.path.exists(pidfile):
    time.sleep(0.02)
print({starting!r}, flush=True)
if mode == "crash":
    os._exit(3)
while not stop and time.monotonic() < ends:
    time.sleep(0.05)
print({stopped!r}, flush=True)
sys.exit(0)  # and leaves its server running
"""
DRIVER = """
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, {runners!r})
import step5_guard as mod
BEGAN = time.monotonic()
FAILS_AFTER = float(os.environ["GUARD_FAILS_AFTER"]) if os.environ.get("GUARD_FAILS_AFTER") else None

class Backend(mod.SystemBackend):  # memory, ports and the status page stood in; processes and signals are the real ones
    def sample(self):
        if FAILS_AFTER is not None and time.monotonic() - BEGAN >= FAILS_AFTER:
            raise RuntimeError("the guard's own failure")
        return {{"t_mono": self.clock(), "MemTotal": 7_990_009_856, "MemFree": 4_000_000_000,
                 "MemAvailable": 5_000_000_000, "pswpin": 8, "pswpout": 22}}
    listeners = staticmethod(lambda: [])
    fetch = staticmethod(lambda: ("unavailable:URLError", b""))
    thp_self = staticmethod(lambda: 1)

mod.SystemBackend = Backend
sys.exit(mod.main(sys.argv[1:]))
"""


class Stage:
    """The guard as a process of its own, a runtime stand-in that starts a process named llama-server in its own
    session, and another llama-server that this test starts itself: the same user's, not this run's."""

    def __init__(self, mod, tmp_path: Path) -> None:
        if mod.process_ids("llama-server"):
            pytest.skip("a llama-server of this user is running: the guard would refuse, and none may be met here")
        self.mod, self.tmp = mod, tmp_path
        self.server = tmp_path / "bin" / "llama-server"
        self.server.parent.mkdir()
        self.server.write_text(SERVER.format(python=sys.executable))
        self.server.chmod(0o755)
        (tmp_path / "runtime.py").write_text(RUNTIME.format(starting=json.dumps(STARTING), stopped=json.dumps(STOPPED)))
        (tmp_path / "driver.py").write_text(DRIVER.format(runners=str(RUNNERS)))
        self.out = tmp_path / "part"
        self.started: list[subprocess.Popen] = []
        self.identities: list[dict] = []

    def guard(self, mode: str, *, duration: str = "3", fails_after: str = "") -> subprocess.Popen:
        env = {**os.environ, "GUARD_FAILS_AFTER": fails_after, "PYTHONDONTWRITEBYTECODE": "1"}
        with open(self.tmp / "guard.out", "wb") as out, open(self.tmp / "guard.err", "wb") as err:
            process = subprocess.Popen(
                [sys.executable, "-B", str(self.tmp / "driver.py"), "--out", str(self.out), "--duration-s", duration,
                 "--stop-grace-s", "10", "--", sys.executable, "-B", str(self.tmp / "runtime.py"), str(self.server),
                 str(self.tmp / "owned.pid"), mode], stdout=out, stderr=err, stdin=subprocess.DEVNULL, env=env)
        self.started.append(process)
        wait_for((self.tmp / "runtime.pid").exists)
        runtime = self.mod.process_info(int((self.tmp / "runtime.pid").read_text()))
        if runtime is not None:  # so that a failing case never leaves the stand-in behind
            self.identities.append(runtime)
        return process

    def identity(self, pidfile: str) -> dict:
        path = self.tmp / pidfile
        wait_for(path.exists)
        info = wait_for(lambda: self.mod.process_info(int(path.read_text())))
        assert info["comm"] == "llama-server"  # by name it is exactly what the old cleanup looked for
        self.identities.append(info)
        return info

    def unrelated(self) -> tuple[subprocess.Popen, dict]:
        process = subprocess.Popen([str(self.server), str(self.tmp / "other.pid")], start_new_session=True)
        self.started.append(process)
        return process, self.identity("other.pid")

    def running(self, info: dict) -> bool:
        now = self.mod.process_info(info["pid"])
        return now is not None and now["start"] == info["start"] and now["state"] != "Z"

    def close(self) -> None:
        for process in self.started:  # this test's own children
            if process.poll() is None:
                process.kill()
            process.wait()
        for info in self.identities:  # whatever a failed case left: by identity, through a handle
            self.mod.send_to_process(info, signal.SIGKILL)


@pytest.fixture
def stage(mod, tmp_path):
    made = Stage(mod, tmp_path)
    yield made
    made.close()


def test_a_real_guard_stops_its_own_llama_server_and_leaves_the_same_users_other_one(stage) -> None:
    guard = stage.guard("leave")
    owned = stage.identity("owned.pid")
    assert owned["ppid"] != guard.pid and owned["pgrp"] == owned["pid"]  # the runtime's child, in its own session
    other, other_info = stage.unrelated()
    assert guard.wait(60) == 1  # not clear: this run left a server, and another one's ownership is unresolved
    result = json.loads((stage.out / "result.json").read_text())
    assert result["status"] == "duration_stop" and result["child"]["returncode"] == 0
    assert result["leftovers"] == {"llama_server_after_exit": 1, "llama_server_left": 0, "llama_server_unresolved": 1,
                                   "runtime_group_left": False, "listeners_after": []}
    assert result["cleanup_clear"] is False
    assert not stage.running(owned)  # this run's: stopped
    assert other.poll() is None and stage.running(other_info)  # the other: still running, never signalled


def test_a_real_runtime_crash_leaves_a_stubborn_server_that_is_killed_after_the_grace(stage) -> None:
    guard = stage.guard("crash", duration="30")
    owned = stage.identity("owned.pid")
    other, other_info = stage.unrelated()
    began = time.monotonic()
    assert guard.wait(60) == 1
    result = json.loads((stage.out / "result.json").read_text())
    assert result["status"] == "child_exited" and result["child"]["returncode"] == 3
    assert result["leftovers"]["llama_server_after_exit"] == 1 and result["leftovers"]["llama_server_left"] == 0
    assert result["leftovers"]["llama_server_unresolved"] == 1 and result["cleanup_clear"] is False
    assert not stage.running(owned) and other.poll() is None and stage.running(other_info)
    assert result["child"]["ran_s"] < 5.0 and time.monotonic() - began >= 4.0  # SIGTERM ignored: SIGKILL 5 s later


def test_a_real_guard_that_fails_stops_its_own_run_and_leaves_the_other_server(stage) -> None:
    guard = stage.guard("leave", duration="60", fails_after="2.5")
    owned = stage.identity("owned.pid")
    other, other_info = stage.unrelated()
    assert guard.wait(60) == 1
    error = (stage.tmp / "guard.err").read_text()
    assert "RuntimeError: the guard's own failure" in error and not (stage.out / "result.json").exists()
    assert json.loads((stage.out / "run.jsonl").read_text().splitlines()[-1]) == STOPPED  # asked to stop, cleanly
    assert not stage.running(owned)
    assert other.poll() is None and stage.running(other_info)
