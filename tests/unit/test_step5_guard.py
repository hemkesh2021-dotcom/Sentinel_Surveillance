"""Checklist step 5's guarded wrapper (D59): the existing memory guard around `sentinel run`, a bounded duration, the
runtime's own Ctrl-C path, leftover cleanup and numbers-only evidence. Fake backends; one real subprocess."""

from __future__ import annotations

import importlib
import json
import signal
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
GB = 1_000_000_000
HEALTHY = {"MemTotal": 7_990_009_856, "MemFree": 4 * GB, "MemAvailable": 5 * GB, "pswpin": 8, "pswpout": 22}
REFUSAL = ("scene_not_admitted: llama-lfm2-vl-scene: resource profile step4cand-demo-20261007T090339Z measured memory "
           "policy (THP workload_disabled, model-file release post_load), not the runtime's (THP system, model-file "
           "release none)")
STARTING = {"run": "starting", "startup": {
    "memory_policy": {"thp": "workload_disabled", "model_file_release": "post_load"},
    "thp_scope": {"before_launch": {"runtime": 1}, "llama_ready": {"runtime": 1, "scene_server": 1},
                  "workload_verified": {"runtime": 0, "scene_server": 1}},
    "thp_disable": {"verified": True, "reason": None, "set_rc": 0, "thp_enabled": 0, "t_mono": 1.0},
    "releases": {"scene": {"files": {"llm": {"result": "returned_0", "bytes": 1}, "mmproj": {"result": "returned_0"}}},
                 "detector": {"files": {"engine": {"result": "returned_0"}}}},
    "scene_server": {"state": "ready", "problem": None, "layers": "offloaded 17/17 layers to GPU",
                     "vision_on_gpu": True},
    "scene_problem": None, "detector_problem": None, "notifier_problems": {},
    "status_page": "http://127.0.0.1:18090/"}}
STOPPED = {"run": "stopped", "shutdown": {"stopped": {"capture": True, "scene_server": True}, "all_stopped": True,
                                          "signals_not_recorded": 0}, "database_closed": True}


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return importlib.import_module("step5_guard")


class FakeChild:
    """A scripted runtime: ready after ``ready_s``, stops ``stop_s`` after SIGINT, or exits at ``exit_s``."""

    def __init__(self, backend, argv, stdout: Path, stderr: Path) -> None:
        self.b, self.stdout, self.stderr = backend, stdout, stderr
        self.pid, self.argv = 4242, argv
        self.started = backend.now
        self.rc = None
        self.stop_at = None
        self.alive_group = True
        self.signals = []
        stdout.write_bytes(b"")
        stderr.write_bytes(b"")
        self.ready_written = False

    def _write(self, item) -> None:
        with open(self.stdout, "a") as handle:
            handle.write((item if isinstance(item, str) else json.dumps(item)) + "\n")

    def poll(self):
        b = self.b
        age = b.now - self.started
        if self.rc is None:
            if b.script.get("exit_s") is not None and age >= b.script["exit_s"]:
                if b.script.get("stderr"):
                    self.stderr.write_text(b.script["stderr"] + "\n")
                self.rc, self.alive_group = b.script.get("exit_rc", 1), False
            elif b.script.get("ready_s") is not None and age >= b.script["ready_s"] and not self.ready_written:
                for line in b.script.get("noise", ()):
                    self._write(line)
                self._write(STARTING)
                self.ready_written = True
            if self.stop_at is not None and b.now >= self.stop_at:
                self._write(STOPPED)
                self.rc, self.alive_group = 0, False
        return self.rc

    def signal(self, signum) -> None:
        self.signals.append(signum)
        if signum == signal.SIGINT and self.stop_at is None and self.b.script.get("stop_s") is not None:
            self.stop_at = self.b.now + self.b.script["stop_s"]

    def signal_group(self, signum) -> bool:
        self.signals.append(("group", signum))
        if signum == signal.SIGKILL:
            self.alive_group = False
            self.rc = -9 if self.rc is None else self.rc
        return self.alive_group

    def group_alive(self) -> bool:
        return self.alive_group


class FakeBackend:
    def __init__(self, **script) -> None:
        self.now = 100.0
        self.script = script
        self.children = []
        self.killed = []
        self.llama = list(script.get("llama_before", ()))
        self.fetched = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def utc(self):
        return "2026-10-07T12:00:00Z"

    def sample(self):
        values = dict(HEALTHY)
        for at, changes in self.script.get("memory", ()):
            if self.children and self.now - self.children[0].started >= at:
                values.update(changes)
        return {"t_mono": self.now, **values}

    def spawn(self, argv, stdout, stderr):
        child = FakeChild(self, argv, stdout, stderr)
        self.children.append(child)
        if self.script.get("orphan_after_exit"):
            self.llama_after = [5151]
        return child

    def llama_pids(self):
        if self.children and self.children[0].rc is not None:
            return [pid for pid in getattr(self, "llama_after", []) if pid not in self.killed]
        return self.llama

    def kill(self, pid, signum):
        self.killed.append(pid)
        return True

    def listeners(self):
        return list(self.script.get("listeners", ()))

    def fetch(self):
        self.fetched.append(self.now)
        return "saved", b'{"runtime": {"state": "running"}}'

    def thp_self(self):
        return self.script.get("thp", 1)


def run(mod, tmp_path, backend, **kwargs):
    out = tmp_path / "part"
    params = dict(duration_s=10.0, stop_grace_s=30.0, ready_timeout_s=60.0)
    params.update(kwargs)
    return mod.supervise(["sentinel", "run"], out, backend=backend, **params), out


@pytest.mark.parametrize(("script", "reason"), [
    ({"thp": 0}, "thp_enabled_not_1"),
    ({"llama_before": [77]}, "llama_server_running"),
    ({"listeners": [{"port": 18090, "family": "tcp", "scope": "loopback"}]}, "port_in_use"),
])
def test_preflight_refuses_before_anything_starts(mod, tmp_path, script, reason) -> None:
    backend = FakeBackend(ready_s=1, stop_s=1, **script)
    result, out = run(mod, tmp_path, backend)
    assert result["status"] == f"refused:{reason}" and backend.children == [] and not out.exists()


def test_preflight_refuses_an_existing_directory_and_a_guard_already_met(mod, tmp_path, monkeypatch) -> None:
    (tmp_path / "part").mkdir()
    backend = FakeBackend(ready_s=1, stop_s=1)
    assert run(mod, tmp_path, backend)[0]["status"] == "refused:out_exists"
    (tmp_path / "part").rmdir()
    monkeypatch.setattr(backend, "sample", lambda: {"t_mono": 1.0, **HEALTHY, "MemAvailable": 1 * GB})
    assert run(mod, tmp_path, backend)[0]["status"] == "refused:memory_guard_already_met"
    assert backend.children == []


def test_the_duration_counts_from_readiness_and_stops_with_sigint_to_the_runtime_alone(mod, tmp_path) -> None:
    backend = FakeBackend(ready_s=40.0, stop_s=3.0)
    result, out = run(mod, tmp_path, backend, status_at=((5.0, "start"), (8.0, "end")))
    child = backend.children[0]
    assert child.signals == [signal.SIGINT]  # the runtime's Ctrl-C path; never its group, never llama-server
    assert result["status"] == "duration_stop" and result["child"]["returncode"] == 0
    assert result["stop"]["requested_by"] == "duration_elapsed" and not result["stop"]["forced"]
    assert 50.0 <= result["stop"]["requested_at_s"] <= 50.8 and 2.8 <= result["stop"]["exited_after_stop_s"] <= 3.4
    assert 40.0 <= result["child"]["ready_after_s"] <= 40.6  # seen on the next 0.2 s sample at most
    assert sorted(c["name"] for c in result["captures"]) == ["end", "start"]
    assert all(c["status"] == "saved" for c in result["captures"])
    assert json.loads((out / "status-start.json").read_text()) == {"runtime": {"state": "running"}}
    assert result["guard"]["samples"] > 250 and result["guard"]["trigger"] is None
    assert result["guard"]["largest_gap_s"] <= 0.5 and result["guard"]["min_mem_available_bytes"] == 5 * GB
    assert (out / "guard.jsonl").read_text().count("\n") == result["guard"]["samples"]
    summary = result["run_output"]["starting"]
    assert summary["memory_policy"] == {"thp": "workload_disabled", "model_file_release": "post_load"}
    assert summary["thp_scope"]["workload_verified"] == {"runtime": 0, "scene_server": 1}
    assert summary["thp_disable"] == {"verified": True, "reason": None, "set_rc": 0, "thp_enabled": 0}
    assert summary["releases"] == {"scene": {"llm": "returned_0", "mmproj": "returned_0"},
                                   "detector": {"engine": "returned_0"}}
    assert summary["scene_server"]["state"] == "ready" and summary["scene_server"]["vision_on_gpu"] is True
    assert result["run_output"]["stopped"] == {"all_stopped": True, "stopped": {"capture": True, "scene_server": True},
                                               "database_closed": True, "signals_not_recorded": 0}
    assert result["run_output"]["non_json_lines"] == 0 and result["cleanup_clear"] is True
    assert "status_page" not in json.dumps(result)  # only the summarised keys are copied


def test_from_launch_counts_the_duration_and_the_offsets_from_launch(mod, tmp_path) -> None:
    backend = FakeBackend(ready_s=40.0, stop_s=1.0)
    result, _ = run(mod, tmp_path, backend, duration_s=60.0, from_launch=True, status_at=((30.0, "idle"),))
    assert 60.0 <= result["stop"]["requested_at_s"] <= 60.4
    assert 30.0 <= result["captures"][0]["taken_after_launch_s"] <= 30.4  # before readiness, as the operator timed it


def test_the_guard_stops_the_runtime_the_same_way_and_records_its_trigger(mod, tmp_path) -> None:
    backend = FakeBackend(ready_s=5.0, stop_s=2.0, memory=((20.0, {"MemTotal": 7_990_009_856,
                                                                    "MemAvailable": 3_100_000_000}),))
    result, _ = run(mod, tmp_path, backend, duration_s=600.0)
    assert result["status"] == "guard_stop" and backend.children[0].signals == [signal.SIGINT]
    assert result["guard"]["trigger"]["stop"] == "sampled_pressure_stop"
    assert result["guard"]["trigger_after_stop_request"] is False
    assert 20.0 <= result["stop"]["requested_at_s"] <= 20.4 and result["guard"]["peak_pressure_bytes"] >= 4_800_000_000


def test_a_swap_counter_change_is_a_guard_stop_as_in_step_4(mod, tmp_path) -> None:
    backend = FakeBackend(ready_s=5.0, stop_s=2.0, memory=((12.0, {"pswpout": 23}),))
    result, _ = run(mod, tmp_path, backend, duration_s=600.0)
    assert result["status"] == "guard_stop" and result["guard"]["trigger"]["stop"] == "swap_counter_change"


def test_a_runtime_that_refuses_exits_on_its_own_and_its_label_is_kept(mod, tmp_path) -> None:
    backend = FakeBackend(exit_s=2.0, exit_rc=1, stderr="some diagnostic\nrun: " + REFUSAL)
    result, out = run(mod, tmp_path, backend, duration_s=60.0, from_launch=True)
    assert result["status"] == "child_exited" and result["child"]["returncode"] == 1
    assert result["run_error_label"] == REFUSAL and result["stop"]["requested_by"] is None
    assert result["run_output"]["starting"] is None and backend.children[0].signals == []
    assert result["cleanup_clear"] is True and not (out / "sentinel.db").exists()


def test_a_runtime_that_ignores_the_stop_is_terminated_then_killed_after_the_grace(mod, tmp_path) -> None:
    backend = FakeBackend(ready_s=1.0)  # no stop_s: SIGINT is ignored
    result, _ = run(mod, tmp_path, backend, duration_s=5.0, stop_grace_s=20.0)
    signals = backend.children[0].signals
    assert signals[0] == signal.SIGINT and ("group", signal.SIGTERM) in signals and ("group", signal.SIGKILL) in signals
    assert result["status"] == "forced_stop" and result["stop"]["forced"] is True


def test_a_llama_server_left_after_exit_is_stopped_and_reported(mod, tmp_path) -> None:
    backend = FakeBackend(ready_s=1.0, stop_s=1.0, orphan_after_exit=True)
    result, _ = run(mod, tmp_path, backend, duration_s=2.0)
    assert result["leftovers"]["llama_server_after_exit"] == 1 and result["leftovers"]["llama_server_left"] == 0
    assert 5151 in backend.killed and result["cleanup_clear"] is False


def test_readiness_has_a_bound_and_an_operator_interrupt_stops_cleanly(mod, tmp_path) -> None:
    backend = FakeBackend(ready_s=None, stop_s=1.0)
    result, _ = run(mod, tmp_path, backend, duration_s=60.0, ready_timeout_s=30.0)
    assert result["status"] == "ready_timeout" and 30.0 <= result["stop"]["requested_at_s"] <= 30.4
    backend = FakeBackend(ready_s=1.0, stop_s=1.0)
    calls = {"n": 0}

    def interrupted():
        calls["n"] += 1
        return calls["n"] > 20

    result, _ = mod.supervise(["sentinel", "run"], tmp_path / "second", backend=backend, duration_s=60.0,
                              interrupted=interrupted), None
    assert result["status"] == "operator_interrupt" and backend.children[0].signals == [signal.SIGINT]


def test_listeners_are_classified_by_scope_without_addresses(mod, tmp_path) -> None:
    net = tmp_path / "net"
    net.mkdir()
    head = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
    (net / "tcp").write_text(head + "".join(f"   0: {local} 00000000:0000 {state} 0 0 0 0 0 0\n" for local, state in (
        ("0100007F:46AA", "0A"),  # 127.0.0.1:18090
        ("00000000:46A1", "0A"),  # 0.0.0.0:18081
        ("6400A8C0:46AA", "0A"),  # another address:18090
        ("0100007F:46AA", "01"),  # established, not a listener
        ("0100007F:0016", "0A"),  # another port
    )))
    (net / "tcp6").write_text(head + "".join(f"   0: {local} {'0' * 32}:0000 0A 0 0 0 0 0 0\n" for local in (
        "00000000000000000000000001000000:46AA",  # ::1
        "00000000000000000000000000000000:46A1",  # ::
        "0000000000000000FFFF00000100007F:46AA",  # ::ffff:127.0.0.1
    )))
    found = mod.listeners(net)
    assert found == [
        {"port": 18081, "family": "tcp", "scope": "any"}, {"port": 18081, "family": "tcp6", "scope": "any"},
        {"port": 18090, "family": "tcp", "scope": "loopback"}, {"port": 18090, "family": "tcp", "scope": "other"},
        {"port": 18090, "family": "tcp6", "scope": "loopback"}, {"port": 18090, "family": "tcp6", "scope": "loopback"},
    ]
    assert "127.0.0.1" not in json.dumps(found) and mod.listeners(tmp_path / "missing") == []


def test_run_output_counts_non_json_lines_for_d48(mod, tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text("engine banner\n" + json.dumps(STARTING) + "\n" + json.dumps({"state": "running"}) + "\n")
    output = mod.read_run_output(path)
    assert (output["lines"], output["json_lines"], output["non_json_lines"]) == (3, 2, 1)
    assert output["starting"]["detector_problem"] is None and output["stopped"] is None


def test_the_command_line_validates_its_offsets_and_needs_a_command(mod, tmp_path, capsys) -> None:
    with pytest.raises(SystemExit):
        mod.main(["--out", str(tmp_path / "x"), "--duration-s", "10", "--status-at", "nope"])
    with pytest.raises(SystemExit):
        mod.main(["--out", str(tmp_path / "x"), "--duration-s", "901", "--", "true"])
    with pytest.raises(SystemExit):
        mod.main(["--out", str(tmp_path / "x"), "--duration-s", "10"])
    assert not (tmp_path / "x").exists()


def test_a_real_runtime_stand_in_stops_on_sigint_and_leaves_nothing(mod, tmp_path, monkeypatch) -> None:
    """A real child process in its own session: SIGINT to it alone, its clean exit, numbers-only result.json."""
    script = tmp_path / "runtime.py"
    script.write_text(textwrap.dedent(f"""
        import json, signal, sys, time
        stop = []
        signal.signal(signal.SIGINT, lambda *_: stop.append(1))
        print("not json: a library banner", flush=True)
        print({json.dumps(json.dumps(STARTING))}, flush=True)
        while not stop:
            time.sleep(0.05)
        print({json.dumps(json.dumps(STOPPED))}, flush=True)
        sys.exit(0)
    """))

    class Backend(mod.SystemBackend):
        def sample(self):
            return {"t_mono": self.clock(), **HEALTHY}

        @staticmethod
        def llama_pids():
            return []

        @staticmethod
        def listeners():
            return []

        @staticmethod
        def fetch():
            return "unavailable:URLError", b""

        @staticmethod
        def thp_self():
            return 1

    monkeypatch.setattr(mod, "SystemBackend", Backend)
    out = tmp_path / "part"
    code = mod.main(["--out", str(out), "--duration-s", "1", "--stop-grace-s", "10", "--status-at", "0.2:start",
                     "--", sys.executable, str(script)])
    result = json.loads((out / "result.json").read_text())
    assert code == 0 and result["status"] == "duration_stop" and result["child"]["returncode"] == 0
    assert result["run_output"]["non_json_lines"] == 1 and result["run_output"]["stopped"]["database_closed"] is True
    assert result["captures"][0]["status"] == "unavailable:URLError" and not (out / "status-start.json").exists()
    assert result["stop"]["exited_after_stop_s"] < 5 and result["cleanup_clear"] is True
    assert sorted(p.name for p in out.iterdir()) == ["guard.jsonl", "result.json", "run.err", "run.jsonl"]


FRESH_BOOT = {"MemTotal": 7_990_009_856, "MemFree": 5_600_000_000, "MemAvailable": 6_900_000_000, "pswpin": 8,
              "pswpout": 22}


@pytest.mark.parametrize(("changes", "admitted"), [
    ({}, True),
    ({"MemAvailable": 5_990_009_856}, False),  # pressure exactly 2.0 GB
    ({"MemFree": 3_499_999_999}, False),
    ({"MemAvailable": 3_999_999_999}, False),
])
def test_step4_headroom_refuses_before_launch_with_the_unchanged_admission_values(mod, tmp_path, monkeypatch,
                                                                                  changes, admitted) -> None:
    backend = FakeBackend(ready_s=1.0, stop_s=1.0)
    monkeypatch.setattr(backend, "sample", lambda: {"t_mono": backend.now, **FRESH_BOOT, **changes})
    result, out = run(mod, tmp_path, backend, duration_s=2.0, step4_headroom=True)
    if admitted:
        assert result["status"] == "duration_stop" and result["parameters"]["step4_headroom"] is True
    else:
        assert result["status"] == "refused:initial_headroom_refused" and backend.children == [] and not out.exists()
        assert result["preflight_sample"]["MemFree"] == {**FRESH_BOOT, **changes}["MemFree"]  # recorded, numbers only


def test_without_step4_headroom_only_the_guard_limits_apply_before_launch(mod, tmp_path) -> None:
    backend = FakeBackend(ready_s=1.0, stop_s=1.0)  # HEALTHY: pressure 2.99 GB, under the guard's 4.8 GB
    assert run(mod, tmp_path, backend, duration_s=2.0)[0]["status"] == "duration_stop"
    backend = FakeBackend(ready_s=1.0, stop_s=1.0)
    assert run(mod, tmp_path / "x", backend, duration_s=2.0, step4_headroom=True)[0]["status"] == (
        "refused:initial_headroom_refused")
