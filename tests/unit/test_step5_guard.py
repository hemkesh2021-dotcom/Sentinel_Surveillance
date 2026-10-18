"""Checklist step 5's guarded wrapper (D59): the existing memory guard around `sentinel run`, a bounded duration, the
runtime's own Ctrl-C path, leftover cleanup and numbers-only evidence. Fake backends; one real subprocess."""

from __future__ import annotations

import argparse
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
    "scene_server": {"state": "ready", "problem": None, "layers": "17/17", "vision_on_gpu": True},
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
                self._write(b.script.get("starting", STARTING))
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
    assert result["status"] == f"refused:{reason}" and backend.children == []
    assert out.is_dir() and list(out.iterdir()) == []  # the directory holds the refusal's result.json (main)


def test_preflight_refuses_an_existing_directory_and_a_guard_already_met(mod, tmp_path, monkeypatch) -> None:
    (tmp_path / "part").mkdir()
    backend = FakeBackend(ready_s=1, stop_s=1)
    assert run(mod, tmp_path, backend)[0]["status"] == "refused:out_exists"
    assert list((tmp_path / "part").iterdir()) == []
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
        assert result["status"] == "refused:initial_headroom_refused" and backend.children == [] and out.is_dir()
        assert result["preflight_sample"]["MemFree"] == {**FRESH_BOOT, **changes}["MemFree"]  # recorded, numbers only


def test_without_step4_headroom_only_the_guard_limits_apply_before_launch(mod, tmp_path) -> None:
    backend = FakeBackend(ready_s=1.0, stop_s=1.0)  # HEALTHY: pressure 2.99 GB, under the guard's 4.8 GB
    assert run(mod, tmp_path, backend, duration_s=2.0)[0]["status"] == "duration_stop"
    backend = FakeBackend(ready_s=1.0, stop_s=1.0)
    (tmp_path / "x").mkdir()
    assert run(mod, tmp_path / "x", backend, duration_s=2.0, step4_headroom=True)[0]["status"] == (
        "refused:initial_headroom_refused")


def test_main_saves_a_refusal_but_never_writes_into_an_existing_directory(mod, tmp_path, monkeypatch, capsys) -> None:
    class Refusing(FakeBackend):
        def __init__(self):
            super().__init__(thp=0)

    monkeypatch.setattr(mod, "SystemBackend", Refusing)
    out = tmp_path / "part"
    assert mod.main(["--out", str(out), "--duration-s", "5", "--", "true"]) == 1
    assert json.loads((out / "result.json").read_text())["status"] == "refused:thp_enabled_not_1"
    (out / "result.json").write_text("earlier evidence")
    monkeypatch.setattr(mod, "SystemBackend", lambda: FakeBackend())
    assert mod.main(["--out", str(out), "--duration-s", "5", "--", "true"]) == 1
    assert (out / "result.json").read_text() == "earlier evidence"


def test_cues_print_at_offsets_from_readiness_and_record_when(mod, tmp_path) -> None:
    backend = FakeBackend(ready_s=5, stop_s=1)
    said = []
    result, _ = run(mod, tmp_path, backend, duration_s=10.0,
                    cues=((0.0, "t0", "START THE STOPWATCH"), (4.0, "turn", "turn away"), (30.0, "late", "never")),
                    say=said.append)
    ready = result["child"]["ready_mono_s"]
    assert ready == pytest.approx(100.0 + result["child"]["ready_after_s"], abs=1e-6)
    assert [line.split("] ", 1)[1] for line in said] == ["START THE STOPWATCH", "turn away"]
    assert said[0].startswith("\a[0:00] ") and said[1].startswith("\a[0:04] ")
    t0, turn, late = result["cues"]
    assert t0["name"] == "t0" and 0 <= t0["printed_mono_s"] - ready < 0.25
    assert 4.0 <= turn["printed_mono_s"] - ready < 4.25
    assert late == {"name": "late", "at_s": 30.0, "printed_after_launch_s": None, "printed_mono_s": None}
    assert result["parameters"]["cues"] == [{"at_s": 0.0, "name": "t0"}, {"at_s": 4.0, "name": "turn"},
                                            {"at_s": 30.0, "name": "late"}]
    assert "START" not in json.dumps(result)  # cue texts are printed, not stored


def test_no_cue_before_readiness_and_none_without_cues(mod, tmp_path) -> None:
    said = []
    result, _ = run(mod, tmp_path, FakeBackend(stop_s=1), ready_timeout_s=3.0, cues=((0.0, "t0", "go"),),
                    say=said.append)
    assert said == [] and result["status"] == "ready_timeout" and result["cues"][0]["printed_mono_s"] is None
    (tmp_path / "plain").mkdir()
    plain, _ = run(mod, tmp_path / "plain", FakeBackend(ready_s=1, stop_s=1))
    assert plain["cues"] == [] and plain["parameters"]["cues"] == []


@pytest.mark.parametrize("bad", ["5:t0", "5::text", "x:t0:text", "5:t 0:text", "5:t0:" + "x" * 161,
                                 "5:t0:café", "901:t0:late"])
def test_cue_arguments_are_checked(mod, bad) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        mod._cue_at(bad)
    assert mod._cue_at("120:k1:sit at the Mac, facing the camera") == (120.0, "k1", "sit at the Mac, facing the camera")


def test_the_starting_summary_keeps_the_face_fields(mod, tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    startup = {**STARTING["startup"], "face": {"validation_run": True, "identities_enrolled": 1},
               "face_problem": None, "face_memory_before": {"MemFree": 5}, "releases": {
                   **STARTING["startup"]["releases"],
                   "face": {"files": {"facenet512_weights.h5": {"result": "returned_0"},
                                      "face_detection_yunet_2023mar.onnx": {"result": "returned_0"}}}}}
    path.write_text(json.dumps({"run": "starting", "startup": startup}) + "\n")
    starting = mod.read_run_output(path)["starting"]
    assert starting["face"] == {"validation_run": True, "identities_enrolled": 1} and starting["face_problem"] is None
    assert starting["releases"]["face"] == {"facenet512_weights.h5": "returned_0",
                                            "face_detection_yunet_2023mar.onnx": "returned_0"}


# ---------------------------------------------------------------- session 80: readiness before T0 (--require-ready)

TRACKER = {"conf": 0.4, "match_thresh": 0.8, "track_high_thresh": 0.25, "track_low_thresh": 0.1,
           "new_track_thresh": 0.25, "track_buffer": 30, "max_time_lost": 30, "fuse_score": True,
           "unconfirmed_thresh": 0.7, "shadow_conf": 0.1, "shadow_max_nms": 30000, "max_det": 300, "iou": 0.7}
BOUNDS = {"shadow_conf": 0.1, "max_records": 20000, "max_bytes": 32000000, "max_line_bytes": 4096, "max_det": 16,
          "max_shadow": 12, "max_lost": 8, "max_assign_tracks": 8, "max_assign_dets": 16}
# The device run trk-20261016T032337Z's starting line (its detector and face entries as sealed): the detector refused
# the diagnostic, the marker's tracker is null, and the session-79 guard then printed every cue against no detector.
DEVICE_NOT_READY = {"run": "starting", "startup": {
    **STARTING["startup"], "detector_problem": "diagnostic_unsupported (callback_order)",
    "face": {"validation_run": True, "identities_enrolled": 1, "tracking_diagnostic": {**BOUNDS, "tracker": None}},
    "face_problem": None}}
READY = {"run": "starting", "startup": {
    **STARTING["startup"],
    "face": {"validation_run": True, "identities_enrolled": 1, "tracking_diagnostic": {**BOUNDS, "tracker": TRACKER}},
    "face_problem": None}}
REQUIRE = "tracking_diagnostic"


def test_the_devices_starting_line_is_refused_before_t0_no_cue_printed_no_status_saved(mod, tmp_path) -> None:
    backend = FakeBackend(ready_s=5, stop_s=1, starting=DEVICE_NOT_READY)
    said = []
    result, out = run(mod, tmp_path, backend, duration_s=60.0, require_ready=REQUIRE, status_at=((2.0, "e1"),),
                      cues=((0.0, "t0", "T0 - leave the camera view now"), (4.0, "k1", "come back")), say=said.append)
    assert result["status"] == "readiness_refused" and result["stop"]["requested_by"] == "readiness_refused"
    assert result["readiness"] == {"required": REQUIRE, "met": False,
                                   "reason": "detector_problem:diagnostic_unsupported (callback_order)",
                                   "starting_after_s": result["stop"]["requested_at_s"]}
    assert 5.0 <= result["stop"]["requested_at_s"] < 5.5  # stopped on the tick that read the starting line
    assert backend.children[0].signals == [signal.SIGINT]  # the runtime's own Ctrl-C path, at once
    assert said == [] and all(c["printed_mono_s"] is None for c in result["cues"])
    assert "ready_mono_s" not in result["child"] and "ready_after_s" not in result["child"]
    assert [c["status"] for c in result["captures"]] == ["not_reached"] and backend.fetched == []
    assert result["child"]["returncode"] == 0 and result["cleanup_clear"] is True  # a clean exit, still refused
    assert result["run_output"]["starting"]["detector_problem"] == "diagnostic_unsupported (callback_order)"
    assert result["parameters"]["require_ready"] == REQUIRE and not (out / "status-e1.json").exists()


def test_readiness_met_counts_the_cues_from_the_starting_line_as_before(mod, tmp_path) -> None:
    backend = FakeBackend(ready_s=5, stop_s=1, starting=READY)
    said = []
    result, _ = run(mod, tmp_path, backend, duration_s=10.0, require_ready=REQUIRE,
                    cues=((0.0, "t0", "go"), (4.0, "k1", "sit")), say=said.append)
    assert result["status"] == "duration_stop" and result["stop"]["requested_by"] == "duration_elapsed"
    assert result["readiness"] == {"required": REQUIRE, "met": True, "reason": None,
                                   "starting_after_s": result["child"]["ready_after_s"]}
    assert [line.split("] ", 1)[1] for line in said] == ["go", "sit"]
    assert 0 <= result["cues"][0]["printed_mono_s"] - result["child"]["ready_mono_s"] < 0.25


def test_without_the_requirement_the_devices_starting_line_still_counts_as_ready(mod, tmp_path) -> None:
    said = []
    result, _ = run(mod, tmp_path, FakeBackend(ready_s=5, stop_s=1, starting=DEVICE_NOT_READY), duration_s=10.0,
                    cues=((0.0, "t0", "go"),), say=said.append)  # as the session-79 package ran it: unchanged
    assert result["status"] == "duration_stop" and len(said) == 1 and "readiness" not in result
    assert result["parameters"]["require_ready"] is None


@pytest.mark.parametrize(("startup", "reason"), [
    (None, "startup_missing"),
    (STARTING["startup"], "marker_missing"),
    ({**STARTING["startup"], "face": {"validation_run": True, "identities_enrolled": 1}}, "marker_missing"),
    (DEVICE_NOT_READY["startup"], "detector_problem:diagnostic_unsupported (callback_order)"),
    ({**READY["startup"], "detector_problem": "memfree_below_minimum"}, "detector_problem:memfree_below_minimum"),
    ({**READY["startup"], "face": {**READY["startup"]["face"], "tracking_diagnostic": {**BOUNDS, "tracker": None}}},
     "tracker_settings_missing"),
    ({**READY["startup"], "face": {**READY["startup"]["face"], "tracking_diagnostic": {
        **BOUNDS, "tracker": {k: v for k, v in TRACKER.items() if k != "iou"}}}}, "tracker_settings_incomplete:iou"),
    ({**READY["startup"], "face": {**READY["startup"]["face"], "tracking_diagnostic": {
        **BOUNDS, "tracker": {**TRACKER, "fuse_score": 1}}}}, "tracker_settings_incomplete:fuse_score"),
    ({**READY["startup"], "face": {**READY["startup"]["face"], "tracking_diagnostic": {
        **BOUNDS, "tracker": {**TRACKER, "conf": "0.4"}}}}, "tracker_settings_incomplete:conf"),
])
def test_each_unmet_readiness_reason_is_a_fixed_label(mod, startup, reason) -> None:
    assert mod.readiness_problem(REQUIRE, startup) == reason
    assert mod.readiness_problem(None, startup) is None  # no requirement: the starting line is readiness


def test_readiness_met_needs_the_whole_marker_and_the_keys_are_the_runtimes(mod) -> None:
    from sentinel.tracking.diagnostic import SETTING_KEYS

    assert mod.readiness_problem(REQUIRE, READY["startup"]) is None
    assert mod.readiness_problem("something_else", READY["startup"]) == "unknown_requirement"
    assert mod.TRACKING_SETTING_KEYS == SETTING_KEYS
    assert mod.READINESS == ("tracking_diagnostic", "landmark_diagnostic")  # session 90 added the landmark diagnostic's


def test_a_clean_exit_before_the_starting_line_never_meets_the_requirement(mod, tmp_path, monkeypatch) -> None:
    result, _ = run(mod, tmp_path, FakeBackend(exit_s=2.0, exit_rc=0), duration_s=60.0, require_ready=REQUIRE)
    assert result["status"] == "child_exited" and result["child"]["returncode"] == 0
    assert result["readiness"] == {"required": REQUIRE, "met": False, "reason": "no_starting_line",
                                   "starting_after_s": None}
    # main: exit 1 whenever the requirement is unmet (a clean process exit alone never implies readiness); 0 when met
    for name, script, expected in (("exit0", dict(exit_s=1.0, exit_rc=0), 1),
                                   ("device", dict(ready_s=1, stop_s=1, starting=DEVICE_NOT_READY), 1),
                                   ("ready", dict(ready_s=1, stop_s=1, starting=READY), 0)):
        monkeypatch.setattr(mod, "SystemBackend", lambda script=script: FakeBackend(**script))
        out = tmp_path / name
        assert mod.main(["--out", str(out), "--duration-s", "5", "--require-ready", REQUIRE, "--cue-at", "0:t0:go",
                         "--", "true"]) == expected, name
        assert json.loads((out / "result.json").read_text())["readiness"]["met"] is (expected == 0)
    monkeypatch.setattr(mod, "SystemBackend", lambda: FakeBackend(exit_s=1.0, exit_rc=0))
    assert mod.main(["--out", str(tmp_path / "plain"), "--duration-s", "5", "--", "true"]) == 0  # unchanged without it


def test_the_requirement_is_a_fixed_choice_on_the_command_line(mod, tmp_path, monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit):
        mod.main(["--out", str(tmp_path / "x"), "--duration-s", "5", "--require-ready", "detector", "--", "true"])
    assert "invalid choice" in capsys.readouterr().err
    seen = {}
    monkeypatch.setattr(mod, "supervise", lambda command, out, **kw: seen.update(kw) or {"status": "x"})
    mod.main(["--out", str(tmp_path / "y"), "--duration-s", "5", "--require-ready", REQUIRE, "--", "true"])
    assert seen["require_ready"] == REQUIRE
