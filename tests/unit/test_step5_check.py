"""Checklist step 5's predeclared readings (D59): each part validates only on complete, expected evidence."""

from __future__ import annotations

import copy
import importlib
import io
import json
import time
from pathlib import Path

import pytest

from sentinel.scene.server import LlamaServerProcess

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
LOOP = {"port": 18090, "family": "tcp", "scope": "loopback"}
SCENE_PORT = {"port": 18081, "family": "tcp", "scope": "loopback"}
CLEAN = {"guard": {"trigger": None, "peak_pressure_bytes": 1, "min_mem_free_bytes": 2, "min_mem_available_bytes": 3},
         "stop": {"forced": False}, "cleanup_clear": True,
         "leftovers": {"llama_server_after_exit": 0, "llama_server_left": 0, "runtime_group_left": False,
                       "listeners_after": []}}
STOPPED = {"all_stopped": True, "database_closed": True, "stopped": {}, "signals_not_recorded": 0}
CANDIDATE = {"thp": "workload_disabled", "model_file_release": "post_load"}
NO_DELIVERY = {"queued": 0, "attempted": 0, "delivered": 0, "failed": 0, "ambiguous": 0}


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return importlib.import_module("step5_check")


def write(directory: Path, result: dict, statuses: dict) -> Path:
    directory.mkdir()
    (directory / "result.json").write_text(json.dumps(result))
    for name, status in statuses.items():
        (directory / f"status-{name}.json").write_text(json.dumps(status))
    return directory


def refusal(mod) -> dict:
    return {**copy.deepcopy(CLEAN), "status": "child_exited", "child": {"returncode": 1},
            "run_error_label": mod.EXPECTED_REFUSAL, "run_output": {"starting": None, "stopped": None}, "captures": []}


def core() -> tuple[dict, dict]:
    result = {**copy.deepcopy(CLEAN), "status": "duration_stop", "child": {"returncode": 0, "ready_after_s": 31.2},
              "run_output": {"lines": 9, "non_json_lines": 0, "stopped": dict(STOPPED), "starting": {
                  "memory_policy": dict(CANDIDATE),
                  "thp_scope": {"before_launch": {"runtime": 1}, "workload_verified": {"runtime": 0}},
                  "thp_disable": {"verified": True, "reason": None, "set_rc": 0, "thp_enabled": 0},
                  "releases": {"detector": {"engine": "returned_0"}}, "detector_problem": None,
                  "notifier_problems": {}}},
              "captures": [{"name": "idle", "listeners": [LOOP]}, {"name": "alert", "listeners": [LOOP]}]}
    statuses = {
        "idle": {"runtime": {"state": "running"}, "live": {"video": "fresh", "occupancy": "empty"},
                 "incidents": {"recent": []}, "delivery": {"totals": dict(NO_DELIVERY)}},
        "alert": {"runtime": {"state": "running"}, "live": {"video": "fresh", "occupancy": "occupied"},
                  "incidents": {"recent": [{"incident_id": "inc-1"}]},
                  "delivery": {"totals": {**NO_DELIVERY, "delivered": 1}}},
    }
    return result, statuses


def scene() -> tuple[dict, dict]:
    result = {**copy.deepcopy(CLEAN), "status": "duration_stop", "child": {"returncode": 0},
              "run_output": {"lines": 30, "non_json_lines": 0, "stopped": dict(STOPPED), "starting": {
                  "memory_policy": dict(CANDIDATE),
                  "thp_scope": {"before_launch": {"runtime": 1}, "llama_ready": {"runtime": 1, "scene_server": 1},
                                "workload_verified": {"runtime": 0, "scene_server": 1}},
                  "thp_disable": {"verified": True, "reason": None},
                  "releases": {"scene": {"llm": "returned_0", "mmproj": "returned_0"},
                               "detector": {"engine": "returned_0"}},
                  "scene_server": {"state": "ready", "problem": None, "layers": "17/17", "vision_on_gpu": True},
                  "scene_problem": None, "detector_problem": None,
                  "scene_memory_before": {"MemFree": 4}, "detector_memory_before": {"MemFree": 3}}},
              "captures": [{"name": "scene-start", "taken_after_launch_s": 110.0, "listeners": [SCENE_PORT, LOOP]},
                           {"name": "scene-end", "taken_after_launch_s": 620.0, "listeners": [SCENE_PORT, LOOP]}]}

    def status(delivered: int) -> dict:
        return {"runtime": {"state": "running"}, "live": {"video": "fresh"},
                "components": {"scene": {"state": "available", "worker": {"delivered": delivered}}},
                "incidents": {"recent": [{"incident_id": "inc-1"}]}, "delivery": {"totals": dict(NO_DELIVERY)}}

    return result, {"scene-start": status(14), "scene-end": status(140)}


def test_each_part_validates_on_complete_expected_evidence(mod, tmp_path, capsys) -> None:
    data = tmp_path / "data"
    data.mkdir()
    assert mod.main(["refusal", str(write(tmp_path / "r", refusal(mod), {})), "--data-dir", str(data)]) == 0
    assert mod.main(["core", str(write(tmp_path / "c", *core())), "--data-dir", str(data)]) == 0
    assert mod.main(["scene", str(write(tmp_path / "s", *scene())), "--data-dir", str(data)]) == 0
    out = capsys.readouterr().out
    assert out.count("validated") == 3 and "not validated" not in out and "FAIL" not in out and "MISSING" not in out
    assert "DESCRIPTIVE scene reports between the captures; per 4 s: [126, 0.99]" in out


def test_missing_evidence_never_passes(mod, tmp_path, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    for part in ("refusal", "core", "scene"):
        assert mod.main([part, str(empty), "--data-dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert out.count("not validated") == 3 and "MISSING" in out and "\nPASS" not in "\n" + out.split("PASS")[0]


@pytest.mark.parametrize("change", [
    lambda r: r.update(run_error_label="scene_not_admitted: something else"),
    lambda r: r["child"].update(returncode=0),
    lambda r: r["run_output"].update(starting={"memory_policy": None}),
    lambda r: r["leftovers"].update(llama_server_after_exit=1),
    lambda r: r["guard"].update(trigger={"stop": "sampled_pressure_stop"}),
])
def test_a_refusal_part_fails_on_any_deviation(mod, tmp_path, change) -> None:
    result = refusal(mod)
    change(result)
    data = tmp_path / "data"
    data.mkdir()
    assert not mod.read_refusal(write(tmp_path / "r", result, {}), data).ok


def test_a_refusal_that_created_a_database_fails(mod, tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "sentinel.db").write_bytes(b"")
    assert not mod.read_refusal(write(tmp_path / "r", refusal(mod), {}), data).ok


@pytest.mark.parametrize("change", [
    lambda r, s: s["alert"]["incidents"].update(recent=[{"incident_id": "a"}, {"incident_id": "b"}]),
    lambda r, s: s["alert"]["delivery"].update(totals={**NO_DELIVERY, "delivered": 2}),
    lambda r, s: s["alert"]["delivery"].update(totals={**NO_DELIVERY, "delivered": 1, "ambiguous": 1}),
    lambda r, s: s["alert"]["delivery"].update(totals={**NO_DELIVERY, "failed": 1}),
    lambda r, s: s["idle"]["live"].update(occupancy="occupied"),
    lambda r, s: s["idle"]["live"].update(video="stale"),
    lambda r, s: r["run_output"]["starting"].update(notifier_problems={"telegram": "credentials_missing"}),
    lambda r, s: r["run_output"]["starting"]["thp_disable"].update(verified=False),
    lambda r, s: r["run_output"]["starting"].update(releases={"detector": {"engine": "returned_error"}}),
    lambda r, s: r["run_output"]["starting"].update(memory_policy={"thp": "system", "model_file_release": "none"}),
    lambda r, s: r["run_output"].update(non_json_lines=1),
    lambda r, s: r["run_output"]["stopped"].update(database_closed=False),
    lambda r, s: r["captures"][0].update(listeners=[{**LOOP, "scope": "any"}]),
    lambda r, s: r.update(status="forced_stop"),
    lambda r, s: s.pop("alert"),
])
def test_the_core_part_fails_on_any_deviation(mod, tmp_path, change) -> None:
    result, statuses = core()
    change(result, statuses)
    assert not mod.read_core(write(tmp_path / "c", result, statuses), tmp_path).ok


@pytest.mark.parametrize("change", [
    lambda r, s: r["run_output"]["starting"]["scene_server"].update(state="failed", problem="timeout"),
    lambda r, s: r["run_output"]["starting"]["scene_server"].update(vision_on_gpu=False),
    lambda r, s: r["run_output"]["starting"]["scene_server"].update(layers="16/17"),
    lambda r, s: r["run_output"]["starting"]["scene_server"].update(layers="18/18"),
    lambda r, s: r["run_output"]["starting"]["scene_server"].update(layers=None),
    lambda r, s: r["run_output"]["starting"]["scene_server"].pop("layers"),
    lambda r, s: r["run_output"]["starting"]["scene_server"].update(layers="offloaded 17/17 layers to GPU"),
    lambda r, s: r["run_output"]["starting"].update(scene_problem="server_failed:timeout"),
    lambda r, s: r["run_output"]["starting"]["thp_scope"].update(llama_ready={"runtime": 1, "scene_server": 0}),
    lambda r, s: r["run_output"]["starting"]["releases"].pop("scene"),
    lambda r, s: s["scene-end"]["components"]["scene"].update(state="unavailable"),
    lambda r, s: s["scene-end"]["components"]["scene"]["worker"].update(delivered=14),
    lambda r, s: s["scene-end"]["delivery"].update(totals={**NO_DELIVERY, "queued": 1}),
    lambda r, s: r["captures"][1].update(listeners=[LOOP]),
    lambda r, s: r.update(status="guard_stop"),
])
def test_the_scene_part_fails_on_any_deviation(mod, tmp_path, change) -> None:
    result, statuses = scene()
    change(result, statuses)
    assert not mod.read_scene(write(tmp_path / "s", result, statuses), tmp_path).ok


L4T = "/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1"
OFFLOAD_OUTPUT = [b"load_tensors: offloaded 17/17 layers to GPU\n", b"clip_model_loader: CLIP using CUDA0 backend\n"]


class ServerChild:
    """llama-server as LlamaServerProcess sees it: its output lines, then stopped on request."""

    pid = 4242

    def __init__(self, lines: list[bytes]) -> None:
        self.stdout = io.BytesIO(b"".join(lines))
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


def runtime_scene_server(tmp_path: Path, lines: list[bytes]) -> dict:
    """The starting line's ``scene_server`` as ``sentinel run`` records it, from the real server's own parse."""
    files = [tmp_path / name for name in ("llama-server", "model.gguf", "mmproj.gguf")]
    for path in files:
        path.write_bytes(b"x")
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds
        time.sleep(0.001)  # lets the output pump run

    server = LlamaServerProcess(
        *files, 18081, environ={"PATH": "/usr/bin"}, popen=lambda argv, **kwargs: ServerChild(lines),
        health=lambda port: True, in_use=lambda port: False, libcuda=lambda pid: [L4T],
        libraries=lambda pid: [str(tmp_path / "libllama.so")], sleep=sleep, monotonic=lambda: now[0])
    status = server.start(30.0)
    server.stop(1.0)
    return {"state": status.state.value, "problem": status.problem, "layers": status.layers,
            "vision_on_gpu": status.vision_on_gpu}


def test_the_scene_server_item_reads_the_layers_the_runtime_reports(mod, tmp_path, capsys) -> None:
    """Step 5's first run (step5-20261007T153831Z) failed this item with the server ready, 17/17 layers and the vision
    encoder on GPU: it expected the profiler's whole log line, not the runtime's "N/M". Here the value comes from the
    real server's parse of llama-server's offload line, through the guard's summary."""
    reported = runtime_scene_server(tmp_path, OFFLOAD_OUTPUT)
    assert reported == {"state": "ready", "problem": None, "layers": "17/17", "vision_on_gpu": True}
    run_output = tmp_path / "run.jsonl"
    run_output.write_text(json.dumps({"run": "starting", "startup": {"scene_server": reported}}) + "\n")
    summary = importlib.import_module("step5_guard").read_run_output(run_output)["starting"]["scene_server"]
    result, statuses = scene()
    result["run_output"]["starting"]["scene_server"] = summary
    data = tmp_path / "data"
    data.mkdir()
    assert mod.main(["scene", str(write(tmp_path / "s", result, statuses)), "--data-dir", str(data)]) == 0
    assert "PASS scene server ready, 17/17 layers, vision encoder on GPU; scene admitted: " in capsys.readouterr().out
