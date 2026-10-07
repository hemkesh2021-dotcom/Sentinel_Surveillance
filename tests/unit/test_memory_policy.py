"""D58: the memory policies shared by `sentinel run` and the profiler's workload.

Covers the policy record; the shared THP disable and its read-only readers; the shared release call, its eligible
files and its rule; `sentinel run`'s order (scene server, its release, the THP disable, the detector, its release),
its scope checks, its refusals and their cleanup; the scene server's guard against inheriting a disabled state,
including a later spawn (a restart) after the runtime disabled THP; the profile's recorded policy and identity,
configuration matching and the unchanged admission (D46). Fakes only, except the test that calls the real prctl in
a throwaway child process (skipped where the kernel lacks THP_enabled): no test process ever changes its own flag.
"""

from __future__ import annotations

import dataclasses
import importlib
import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sentinel import adapters, cli, memory_policy
from sentinel.adapters import RESOURCE_PROFILES, STEP4_CRITERIA_ID, accepted_profile_problem, resolve
from sentinel.config import parse_config
from sentinel.demo_runtime import (
    Devices,
    MemoryOps,
    RunOptions,
    SceneOptions,
    StartupRefused,
    assemble,
    profile_mismatch,
)
from sentinel.media.clock import FakeClock
from sentinel.memory_policy import (
    CANDIDATE_POLICY,
    DEFAULT_POLICY,
    RELEASE_FILE_ROLES,
    RELEASE_SETTLE_S,
    MemoryPolicy,
    release_component_files,
    release_problem,
)
from sentinel.scene.server import LlamaServerProcess, ServerState

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
SERVER_PID = 4242
REAL_THP = (sys.platform.startswith("linux") and os.path.exists("/proc/self/status")
            and "THP_enabled:" in Path("/proc/self/status").read_text())


@pytest.fixture
def runners(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return SimpleNamespace(profile=importlib.import_module("demo_profile"),
                           workload=importlib.import_module("demo_workload"),
                           criteria=importlib.import_module("step4_criteria"),
                           operator=importlib.import_module("operator_check"))


# ---------------------------------------------------------------- the policy record


def test_the_default_policy_changes_nothing_and_the_candidate_names_both_policies() -> None:
    assert DEFAULT_POLICY == MemoryPolicy() == MemoryPolicy("system", "none")
    assert CANDIDATE_POLICY == MemoryPolicy("workload_disabled", "post_load")
    assert memory_policy.policy_from_flags(workload_thp_disable=False, post_load_release=False) == DEFAULT_POLICY
    assert memory_policy.policy_from_flags(workload_thp_disable=True, post_load_release=True) == CANDIDATE_POLICY
    assert memory_policy.policy_from_flags(workload_thp_disable=True, post_load_release=False).labels() == {
        "thp": "workload_disabled", "model_file_release": "none"}
    assert CANDIDATE_POLICY.describe() == "THP workload_disabled, model-file release post_load"
    for bad in ({"thp": "never"}, {"model_file_release": "always"}, {"thp": ""}):
        with pytest.raises(ValueError):
            MemoryPolicy(**bad)
    assert RunOptions(Path("d"), Path("e")).memory_policy == DEFAULT_POLICY


# ---------------------------------------------------------------- the shared THP disable and readers


def proc_files(files: dict[str, str | None]):
    def opener(path, mode):
        assert mode == "rb"  # read-only
        if files.get(str(path)) is None:
            raise FileNotFoundError(path)
        return io.BytesIO(files[str(path)].encode())

    return opener


SELF = {"/proc/self/status": "Name:\tx\nTHP_enabled:\t0\n", "/proc/self/smaps_rollup": "AnonHugePages:\t2048 kB\n"}


def fake_prctl(calls: list):
    def call(option, arg2):
        calls.append((option, arg2))
        return (0, 0) if option == 41 else (1, 0)

    return call


def test_the_workload_and_the_runtime_make_the_same_verified_disable(runners) -> None:
    shared_calls, workload_calls = [], []
    shared = memory_policy.disable_thp_for_this_process(prctl=fake_prctl(shared_calls), modules={},
                                                        opener=proc_files(SELF), clock=lambda: 3.0)
    workload = runners.workload.disable_thp_for_this_process(prctl=fake_prctl(workload_calls), modules={},
                                                             opener=proc_files(SELF), clock=lambda: 3.0)
    assert shared == workload and shared["verified"] is True and shared["anon_huge_pages_bytes"] == 2 << 20
    assert shared_calls == workload_calls == [(41, 1), (42, 0)]
    assert MemoryOps().disable_thp is memory_policy.disable_thp_for_this_process  # sentinel run's default
    assert runners.workload.WTD_MODEL_MODULES == memory_policy.MODEL_MODULES == runners.profile.WTD_MODEL_MODULES


def test_a_missing_binding_or_a_loaded_model_library_sets_nothing() -> None:
    record = memory_policy.disable_thp_for_this_process(binding=lambda: None, modules={}, opener=proc_files(SELF))
    assert (record["verified"], record["reason"], record["set_rc"]) == (False, "prctl_unavailable", None)
    calls: list = []
    record = memory_policy.disable_thp_for_this_process(prctl=fake_prctl(calls), modules={"torch": 1},
                                                        opener=proc_files(SELF))
    assert calls == [] and record["reason"] == "model_modules_loaded"


@pytest.mark.parametrize(("status", "rollup", "thp", "huge"), [
    ("THP_enabled:\t1\n", "AnonHugePages:\t0 kB\n", 1, 0),
    ("THP_enabled:\t0\n", "AnonHugePages:\t4096 kB\n", 0, 4 << 20),
    ("Name:\tx\n", "Rss:\t1 kB\n", None, None),
    ("THP_enabled:\tmaybe\n", "AnonHugePages:\tlots\n", None, None),
    ("THP_enabled:\t1\n" + "x" * (20 << 10), "AnonHugePages:\t0 kB\n" + "x" * (20 << 10), None, None),
    (None, None, None, None),
])
def test_the_readers_are_read_only_bounded_and_unreadable_is_none(status, rollup, thp, huge) -> None:
    opener = proc_files({"/proc/77/status": status, "/proc/77/smaps_rollup": rollup})
    assert memory_policy.read_thp_enabled(77, opener=opener) == thp
    assert memory_policy.read_anon_huge_pages(77, opener=opener) == huge
    assert memory_policy.read_thp_enabled(None) is None and memory_policy.read_anon_huge_pages(None) is None


# ---------------------------------------------------------------- the shared release


def releaser(results: dict[str, str] | None = None, calls: list | None = None):
    results = results or {}

    def release(path: Path) -> dict:
        if calls is not None:
            calls.append(Path(path).name)
        result = results.get(Path(path).name, "returned_0")
        return {"name": Path(path).name, "bytes": 10 if result == "returned_0" else None, "result": result,
                "returncode": 0 if result == "returned_0" else 22, "error": None if result == "returned_0" else "EINVAL",
                "elapsed_s": 0.001, "call": memory_policy.FADVISE_CALL}

    return release


def test_a_component_releases_exactly_its_eligible_files_in_their_fixed_order() -> None:
    calls: list = []
    record = release_component_files("scene", {"mmproj": Path("/m/p.gguf"), "llm": Path("/m/l.gguf")},
                                     release=releaser(calls=calls))
    assert calls == ["l.gguf", "p.gguf"]  # RELEASE_FILE_ROLES order (llm, then mmproj), whatever the mapping order
    assert record["verified"] is True and record["problem"] is None and list(record["files"]) == ["llm", "mmproj"]
    assert dict(RELEASE_FILE_ROLES) == {"scene": ("llm", "mmproj"), "detector": ("engine",),
                                        "face": ("facenet512_weights.h5", "face_detection_yunet_2023mar.onnx")}


@pytest.mark.parametrize("result", ["returned_error", "open_failed", "unsupported", "something_else"])
def test_any_call_that_did_not_return_0_fails_the_release_with_its_file_and_result(result) -> None:
    record = release_component_files("scene", {"llm": Path("l.gguf"), "mmproj": Path("p.gguf")},
                                     release=releaser({"p.gguf": result}))
    label = result if result in memory_policy.RELEASE_RESULTS else "unknown"
    assert record["verified"] is False and record["problem"] == f"mmproj:{label}"


@pytest.mark.parametrize(("component", "files", "problem"), [
    ("scene", {"llm": Path("l")}, "files_not_eligible"),
    ("detector", {"engine": Path("e"), "llm": Path("l")}, "files_not_eligible"),
    ("detector", {}, "files_not_eligible"),
    ("tokenizer", {"engine": Path("e")}, "unknown_component"),
])
def test_files_that_are_not_exactly_the_eligible_ones_are_refused_before_any_call(component, files, problem) -> None:
    calls: list = []
    record = release_component_files(component, files, release=releaser(calls=calls))
    assert calls == [] and record == {"component": component, "files": {}, "verified": False, "problem": problem}


def test_the_release_rule_never_reads_missing_as_met() -> None:
    assert release_problem("detector", {"engine": {"result": "returned_0"}}) is None
    assert release_problem("detector", {"engine": {}}) == "engine:unknown"
    assert release_problem("detector", {"engine": "returned_0"}) == "engine:unknown"
    assert release_problem("detector", None) == "files_not_eligible"
    assert release_problem("face", {"facenet512_weights.h5": {"result": "returned_0"}}) == "files_not_eligible"


def test_the_profiler_and_the_runtime_share_the_call_the_files_the_reader_and_the_settle(runners, monkeypatch,
                                                                                        tmp_path) -> None:
    profile, operator = runners.profile, runners.operator
    assert profile.release_file_cache is memory_policy.release_file_cache is MemoryOps().release
    assert profile.MR1_RELEASE_FILES is RELEASE_FILE_ROLES and profile.DEEPFACE_WEIGHTS == RELEASE_FILE_ROLES["face"]
    assert profile.read_thp_enabled is memory_policy.read_thp_enabled is MemoryOps().thp_enabled
    files = profile.mr1_files(profile.parse_args([]))
    assert {component: tuple(roles) for component, roles in files.items()} == dict(RELEASE_FILE_ROLES)
    # The settle before each release: the candidate's command passes 15 s, and the profiler hands it to the workload.
    argv = operator.step4cand_argv(tmp_path, Path("clip.mp4"))
    assert float(argv[argv.index("--settle-s") + 1]) == RELEASE_SETTLE_S
    launched = []
    monkeypatch.setattr(profile.subprocess, "Popen", lambda argv, **kwargs: launched.append(argv))
    profile.start_workload(profile.parse_args(argv[2:]), tmp_path)
    assert float(launched[0][launched[0].index("--settle-s") + 1]) == RELEASE_SETTLE_S


# ---------------------------------------------------------------- sentinel run: order, scope and records


class Ops:
    """A fake MemoryOps that records every call in one shared log; it never touches this process's flag."""

    def __init__(self, log: list, *, own_before=1, own_after=0, server=1, disable_reason=None, release_results=None,
                 interrupt_sleep=False):
        self.log, self.disabled = log, False
        self.own_before, self.own_after, self.server = own_before, own_after, server
        self.disable_reason, self.interrupt_sleep = disable_reason, interrupt_sleep
        self.release_results = release_results or {}

    def disable(self) -> dict:
        self.log.append("disable_thp")
        self.disabled = self.disable_reason is None
        return {"t_mono": 5.0, "requested": True, "model_modules_loaded": [], "set_rc": 0, "set_errno": None,
                "get_value": 1, "thp_enabled": 0, "anon_huge_pages_bytes": 0, "verified": self.disabled,
                "reason": self.disable_reason}

    def read(self, pid) -> int | None:
        self.log.append(f"thp_enabled:{pid}")
        if pid == "self":
            return self.own_after if self.disabled else self.own_before
        return self.server if pid == SERVER_PID else None

    def sleep(self, seconds: float) -> None:
        self.log.append(f"sleep:{seconds:g}")
        if self.interrupt_sleep:
            raise KeyboardInterrupt

    def release(self, path: Path) -> dict:
        self.log.append(f"release:{Path(path).name}")
        return releaser(self.release_results)(path)

    def memory(self) -> MemoryOps:
        return MemoryOps(disable_thp=self.disable, thp_enabled=self.read, release=self.release, sleep=self.sleep)


class World:
    """Fake devices for assemble(): every component records into one ordered log."""

    def __init__(self, tmp_path: Path, **ops) -> None:
        self.log: list[str] = []
        self.ops = Ops(self.log, **ops)
        self.server_state = "ready"
        self.engine = tmp_path / "yolov8n.engine"

    def devices(self) -> Devices:
        world = self

        class Backend:
            def load(self) -> None:
                world.log.append("detector_load")

            def track(self, image):
                return []

            def reset(self) -> None:
                pass

        def scene_server(options: SceneOptions, port: int) -> Any:
            world.log.append(f"scene_server:require_system_thp={options.require_system_thp}")
            status = SimpleNamespace(state=SimpleNamespace(value=world.server_state), problem=None, layers="17/17",
                                     vision_on_gpu=True)

            def start(timeout):
                world.log.append("scene_start")
                return status

            return SimpleNamespace(pid=SERVER_PID, start=start, status=lambda: status,
                                   stop=lambda grace: world.log.append("scene_stop") or True)

        def scene_request(port: int, timeout: float) -> Any:
            world.log.append("scene_request")
            return lambda job, image: None

        return Devices(capture_source=lambda config: object(), tracker_backend=lambda engine: Backend(),
                       scene_server=scene_server, scene_request=scene_request,
                       meminfo=lambda: {"MemFree": 7_000_000_000, "MemAvailable": 7_000_000_000},
                       notifiers=lambda config: ({}, {}), memory=self.ops.memory())


def scene_config(fixture) -> Any:
    return parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "adapters": [fixture.manifest]})


def candidate_fixture(accepted_scene, **changes):
    """The SYNTHETIC accepted profile, recording the candidate policy under STEP4_CRITERIA_ID (a fixture only).

    accepted_profile_problem() refuses such a record (its identity measures the default policy), so these tests pass
    an admission stand-in that accepts it: they exercise the runtime's order and checks, not admission."""
    return accepted_scene(memory_policy=CANDIDATE_POLICY, **changes)


@pytest.fixture
def admit_any(monkeypatch):
    """Admission stand-in for order tests only: the scene adapter is admitted whatever the profile records."""
    import sentinel.demo_runtime as runtime

    monkeypatch.setattr(runtime, "scene_admission", lambda config, scene, profiles, policy: (None, "rev"))


def build(world: World, tmp_path: Path, policy: MemoryPolicy, scene=None, config=None, profiles=RESOURCE_PROFILES):
    options = RunOptions(tmp_path / "data", world.engine, scene=scene, memory_policy=policy)
    config = config or parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
    return assemble(config, options, world.devices(), FakeClock(), profiles=profiles)


def test_with_both_policies_the_runtime_follows_the_profilers_order(tmp_path, accepted_scene, admit_any) -> None:
    fixture = candidate_fixture(accepted_scene)
    world = World(tmp_path)
    assembly = build(world, tmp_path, CANDIDATE_POLICY, fixture.options, scene_config(fixture), fixture.profiles)
    assert world.log == [
        "thp_enabled:self",  # before_launch: 1, before anything starts
        "scene_server:require_system_thp=True", "scene_start",  # spawned first, keeping the system setting
        "thp_enabled:self", f"thp_enabled:{SERVER_PID}",  # llama_ready: both 1
        "sleep:15", f"release:{fixture.model.name}", f"release:{fixture.mmproj.name}",  # scene: settle, then release
        "scene_request",
        "disable_thp", "thp_enabled:self", f"thp_enabled:{SERVER_PID}",  # this process 0, the server still 1
        "detector_load",  # the D27 cuInit and the model libraries come after the verified disable
        "sleep:15", "release:yolov8n.engine",  # detector: settle, then release
    ]
    startup = assembly.startup
    assert startup["memory_policy"] == {"thp": "workload_disabled", "model_file_release": "post_load"}
    assert startup["thp_scope"] == {"before_launch": {"runtime": 1}, "llama_ready": {"runtime": 1, "scene_server": 1},
                                    "workload_verified": {"runtime": 0, "scene_server": 1}}
    assert startup["thp_disable"]["verified"] is True and startup["thp_disable"]["reason"] is None
    assert set(startup["releases"]) == {"scene", "detector"} and startup["releases"]["scene"]["settle_s"] == 15.0
    assert startup["releases"]["detector"]["files"] == {"engine": {"result": "returned_0", "returncode": 0, "error": None,
                                                                   "bytes": 10, "elapsed_s": 0.001}}
    text = json.dumps(startup)
    assert str(tmp_path) not in text and fixture.model.name not in text  # numbers and fixed labels only
    assert assembly.runtime.snapshot()["components"]["scene"]["state"] == "available"
    assembly.database.close()


def test_without_scene_the_disable_comes_before_the_detector_and_its_release_after(tmp_path) -> None:
    world = World(tmp_path)
    assembly = build(world, tmp_path, CANDIDATE_POLICY)
    assert world.log == ["thp_enabled:self", "disable_thp", "thp_enabled:self", "detector_load", "sleep:15",
                         "release:yolov8n.engine"]
    assert assembly.startup["thp_scope"]["workload_verified"] == {"runtime": 0}  # no server to read
    assembly.database.close()


def test_each_policy_alone_does_only_its_own_part(tmp_path) -> None:
    world = World(tmp_path)
    build(world, tmp_path / "thp", MemoryPolicy("workload_disabled", "none")).database.close()
    assert world.log == ["thp_enabled:self", "disable_thp", "thp_enabled:self", "detector_load"]
    world = World(tmp_path)
    assembly = build(world, tmp_path / "release", MemoryPolicy("system", "post_load"))
    assert world.log == ["detector_load", "sleep:15", "release:yolov8n.engine"]
    assert "thp_scope" not in assembly.startup and "thp_disable" not in assembly.startup
    assembly.database.close()


def test_the_default_policy_touches_no_thp_state_and_releases_nothing(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()  # the default policy, as measured: admitted exactly as before D58
    world = World(tmp_path)
    assembly = build(world, tmp_path, DEFAULT_POLICY, fixture.options, scene_config(fixture), fixture.profiles)
    assert world.log == ["scene_server:require_system_thp=False", "scene_start", "scene_request", "detector_load"]
    assert assembly.startup["memory_policy"] == {"thp": "system", "model_file_release": "none"}
    assert not {"thp_scope", "thp_disable", "releases"} & set(assembly.startup)
    assembly.database.close()


def test_a_scene_server_that_fails_leaves_the_policy_to_core_monitoring(tmp_path, accepted_scene, admit_any) -> None:
    fixture = candidate_fixture(accepted_scene)
    world = World(tmp_path)
    world.server_state = "failed"
    assembly = build(world, tmp_path, CANDIDATE_POLICY, fixture.options, scene_config(fixture), fixture.profiles)
    assert world.log == ["thp_enabled:self", "scene_server:require_system_thp=True", "scene_start", "disable_thp",
                         "thp_enabled:self", "detector_load", "sleep:15", "release:yolov8n.engine"]
    assert assembly.startup["scene_problem"].startswith("server_failed")  # no scene release, no server to read
    assembly.database.close()


# ---------------------------------------------------------------- sentinel run: refusals and their cleanup


def assert_closed(tmp_path: Path) -> None:
    from sentinel.storage.database import Database

    Database.open(tmp_path / "data" / "sentinel.db").close()  # a database left open would hold the writer lock


@pytest.mark.parametrize(("ops", "label"), [
    ({"own_before": 0}, "thp_enabled_not_1_before_launch"),
    ({"own_before": None}, "thp_enabled_not_1_before_launch"),  # unreadable is never as expected
])
def test_a_process_whose_thp_is_already_off_refuses_before_anything_starts(tmp_path, accepted_scene, admit_any, ops,
                                                                           label) -> None:
    fixture = candidate_fixture(accepted_scene)
    world = World(tmp_path, **ops)
    with pytest.raises(StartupRefused) as error:
        build(world, tmp_path, CANDIDATE_POLICY, fixture.options, scene_config(fixture), fixture.profiles)
    assert error.value.label == label and world.log == ["thp_enabled:self"]
    assert not (tmp_path / "data" / "sentinel.db").exists()  # refused before the database opens


@pytest.mark.parametrize(("ops", "label", "last"), [
    ({"disable_reason": "status_mismatch"}, "thp_disable_not_verified:status_mismatch", "disable_thp"),
    ({"disable_reason": "model_modules_loaded"}, "thp_disable_not_verified:model_modules_loaded", "disable_thp"),
    ({"own_after": 1}, "thp_scope_not_verified:workload_verified", f"thp_enabled:{SERVER_PID}"),
    ({"own_after": None}, "thp_scope_not_verified:workload_verified", f"thp_enabled:{SERVER_PID}"),
])
def test_an_unverified_disable_or_scope_refuses_stops_the_server_and_loads_no_detector(
    tmp_path, accepted_scene, admit_any, ops, label, last,
) -> None:
    fixture = candidate_fixture(accepted_scene)
    world = World(tmp_path, **ops)
    with pytest.raises(StartupRefused) as error:
        build(world, tmp_path, CANDIDATE_POLICY, fixture.options, scene_config(fixture), fixture.profiles)
    assert error.value.label == label
    assert world.log[-2:] == [last, "scene_stop"] and "detector_load" not in world.log
    assert_closed(tmp_path)


def test_a_server_that_reads_thp_disabled_after_the_runtimes_disable_refuses(tmp_path, accepted_scene,
                                                                             admit_any) -> None:
    fixture = candidate_fixture(accepted_scene)
    world = World(tmp_path)

    def read(pid):
        world.log.append(f"thp_enabled:{pid}")
        if pid == "self":
            return 0 if world.ops.disabled else 1
        return 0 if world.ops.disabled else 1  # the server "changed" after the disable: never as expected

    world.ops.read = read
    with pytest.raises(StartupRefused) as error:
        build(world, tmp_path, CANDIDATE_POLICY, fixture.options, scene_config(fixture), fixture.profiles)
    assert error.value.label == "thp_scope_not_verified:workload_verified" and world.log[-1] == "scene_stop"
    assert_closed(tmp_path)


def test_a_server_that_does_not_read_1_once_ready_refuses_and_is_stopped(tmp_path, accepted_scene, admit_any) -> None:
    fixture = candidate_fixture(accepted_scene)
    world = World(tmp_path, server=0)
    with pytest.raises(StartupRefused) as error:
        build(world, tmp_path, CANDIDATE_POLICY, fixture.options, scene_config(fixture), fixture.profiles)
    assert error.value.label == "thp_scope_not_verified:llama_ready"
    assert world.log[-1] == "scene_stop" and "disable_thp" not in world.log
    assert_closed(tmp_path)


def test_a_failed_scene_release_refuses_before_the_disable_and_the_detector(tmp_path, accepted_scene,
                                                                            admit_any) -> None:
    fixture = candidate_fixture(accepted_scene)
    world = World(tmp_path, release_results={fixture.mmproj.name: "returned_error"})
    with pytest.raises(StartupRefused) as error:
        build(world, tmp_path, CANDIDATE_POLICY, fixture.options, scene_config(fixture), fixture.profiles)
    assert error.value.label == "post_load_release_failed:scene:mmproj:returned_error"
    assert world.log[-3:] == [f"release:{fixture.model.name}", f"release:{fixture.mmproj.name}", "scene_stop"]
    assert "disable_thp" not in world.log and "detector_load" not in world.log
    assert_closed(tmp_path)


@pytest.mark.parametrize("result", ["open_failed", "returned_error", "unsupported"])
def test_a_failed_detector_release_refuses_and_stops_the_scene_server(tmp_path, accepted_scene, admit_any,
                                                                       result) -> None:
    fixture = candidate_fixture(accepted_scene)
    world = World(tmp_path, release_results={"yolov8n.engine": result})
    with pytest.raises(StartupRefused) as error:
        build(world, tmp_path, CANDIDATE_POLICY, fixture.options, scene_config(fixture), fixture.profiles)
    assert error.value.label == f"post_load_release_failed:detector:engine:{result}"
    assert world.log[-3:] == ["sleep:15", "release:yolov8n.engine", "scene_stop"]
    assert_closed(tmp_path)


def test_an_interrupt_during_a_settle_stops_the_server_and_closes_the_database(tmp_path, accepted_scene,
                                                                               admit_any) -> None:
    fixture = candidate_fixture(accepted_scene)
    world = World(tmp_path, interrupt_sleep=True)
    with pytest.raises(KeyboardInterrupt):
        build(world, tmp_path, CANDIDATE_POLICY, fixture.options, scene_config(fixture), fixture.profiles)
    assert world.log[-2:] == ["sleep:15", "scene_stop"] and "disable_thp" not in world.log
    assert_closed(tmp_path)


# ---------------------------------------------------------------- the scene server's guard (startup and restarts)


@pytest.fixture
def server_files(tmp_path: Path) -> dict[str, Path]:
    paths = {name: tmp_path / name for name in ("llama-server", "model.gguf", "mmproj.gguf")}
    for path in paths.values():
        path.write_bytes(b"x")
    return paths


class FakeProcess:
    def __init__(self) -> None:
        self.stdout = io.BytesIO(b"load_tensors: offloaded 17/17 layers to GPU\nclip: CLIP using CUDA0 backend\n")
        self.pid, self.returncode, self.signals = SERVER_PID, None, []

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.signals.append("TERM")
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def guarded(files: dict[str, Path], reads: dict, *, require: bool = True, popen_calls: list | None = None):
    process = FakeProcess()
    calls = [] if popen_calls is None else popen_calls

    def read(pid):
        if pid not in reads:
            raise AssertionError(f"unexpected THP read of {pid}")
        return reads[pid]

    now = [0.0]  # a clock that advances with each sleep, so a bounded wait always ends
    server = LlamaServerProcess(
        files["llama-server"], files["model.gguf"], files["mmproj.gguf"], 18081, environ={"PATH": "/usr/bin"},
        popen=lambda argv, **kwargs: calls.append(argv) or process, health=lambda port: True,
        in_use=lambda port: False, libcuda=lambda pid: ["/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1"],
        libraries=lambda pid: [str(files["llama-server"].parent / "libllama.so.0.0.8932")],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds), monotonic=lambda: now[0],
        require_system_thp=require, thp_enabled=read,
    )
    return server, process, calls


def test_a_server_is_never_spawned_from_a_process_whose_thp_is_disabled(server_files) -> None:
    """A restart (a new server object, as V2-19 would make one) after the runtime's disable cannot inherit it."""
    for parent in (0, None):  # disabled, or unreadable: never as expected
        server, process, calls = guarded(server_files, {"self": parent})
        status = server.start(5.0)
        assert (status.state, status.problem) == (ServerState.FAILED, "parent_thp_disabled") and calls == []
        assert server.pid is None


def test_a_started_server_must_keep_the_system_setting_or_it_is_stopped(server_files) -> None:
    server, process, calls = guarded(server_files, {"self": 1, SERVER_PID: 0})
    status = server.start(5.0)
    assert (status.state, status.problem) == (ServerState.FAILED, "server_thp_not_system")
    assert len(calls) == 1 and process.signals == ["TERM"]
    server, process, _ = guarded(server_files, {"self": 1, SERVER_PID: 1})
    assert server.start(5.0).state is ServerState.READY and server.pid == SERVER_PID


def test_without_the_requirement_the_server_reads_no_thp_state(server_files) -> None:
    server, _, _ = guarded(server_files, {}, require=False)  # any read would fail the test
    assert server.start(5.0).state is ServerState.READY


def test_the_cli_builds_the_server_with_the_requirement_the_options_carry(server_files) -> None:
    options = SceneOptions(server_files["llama-server"], server_files["model.gguf"], server_files["mmproj.gguf"])
    assert cli._scene_server(options, 18081)._require_system_thp is False
    assert cli._scene_server(dataclasses.replace(options, require_system_thp=True), 18081)._require_system_thp is True


@pytest.mark.skipif(not REAL_THP, reason="needs a Linux kernel that reports THP_enabled")
def test_on_this_kernel_only_children_started_after_the_disable_inherit_it_and_the_guard_refuses_them(tmp_path) -> None:
    """The real call, in a throwaway child only: a server started before keeps 1; one started after would read 0,
    which is why a later spawn is refused; this test process is unaffected."""
    before = Path("/proc/self/status").read_text()
    script = textwrap.dedent("""
        import json, subprocess, sys
        from pathlib import Path
        from sentinel import memory_policy
        from sentinel.scene.server import LlamaServerProcess
        sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
        early = subprocess.Popen(sleeper)  # spawned before the disable, as sentinel run spawns its scene server
        record = memory_policy.disable_thp_for_this_process(modules={})
        late = subprocess.Popen(sleeper)  # spawned after it
        found = {"record": record, "early": memory_policy.read_thp_enabled(early.pid),
                 "late": memory_policy.read_thp_enabled(late.pid), "self": memory_policy.read_thp_enabled("self")}
        for child in (early, late):
            child.kill()
            child.wait()
        files = [Path(sys.argv[1]) / name for name in ("llama-server", "model.gguf", "mmproj.gguf")]
        for path in files:
            path.write_bytes(b"x")
        calls = []
        server = LlamaServerProcess(*files, 18099, environ={"PATH": "/usr/bin"}, in_use=lambda port: False,
                                    popen=lambda argv, **kwargs: calls.append(argv), require_system_thp=True)
        status = server.start(1.0)
        found.update(guard=status.problem, spawned=len(calls))
        print(json.dumps(found))
    """)
    done = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=60,
                          check=True)
    found = json.loads(done.stdout)
    assert found["record"]["verified"] is True and found["self"] == 0
    assert (found["early"], found["late"]) == (1, 0)
    assert (found["guard"], found["spawned"]) == ("parent_thp_disabled", 0)
    after = Path("/proc/self/status").read_text()
    assert [line for line in after.splitlines() if line.startswith("THP_enabled")] == [
        line for line in before.splitlines() if line.startswith("THP_enabled")]


# ---------------------------------------------------------------- the profile's identity, matching and admission


def test_every_registry_profile_records_the_default_policy_and_none_has_a_variant_identity() -> None:
    assert adapters.ResourceProfile.__dataclass_fields__["memory_policy"].default == DEFAULT_POLICY
    for profile in RESOURCE_PROFILES.values():
        assert profile.memory_policy == DEFAULT_POLICY
        assert profile.criteria_id not in (adapters.STEP4PLR_CRITERIA_ID, adapters.STEP4_CANDIDATE_CRITERIA_ID)


def test_the_identity_and_policy_tables_agree_between_the_runtime_and_the_runner(runners) -> None:
    criteria = runners.criteria
    assert (adapters.STEP4_CRITERIA_ID, adapters.STEP4PLR_CRITERIA_ID, adapters.STEP4_CANDIDATE_CRITERIA_ID) == (
        criteria.CRITERIA_ID, criteria.PLR_CRITERIA_ID, criteria.CANDIDATE_CRITERIA_ID)
    assert {key: (policy.thp, policy.model_file_release)
            for key, policy in adapters.CRITERIA_MEMORY_POLICIES.items()} == criteria.MEMORY_POLICIES
    assert adapters.CRITERIA_MEMORY_POLICIES[criteria.CANDIDATE_CRITERIA_ID] == CANDIDATE_POLICY


def test_a_step4_profile_that_records_another_policy_is_refused_as_an_identity_mismatch(tmp_path,
                                                                                        accepted_scene) -> None:
    for policy in (CANDIDATE_POLICY, MemoryPolicy("system", "post_load"), MemoryPolicy("workload_disabled", "none")):
        fixture = accepted_scene(memory_policy=policy)
        problem = accepted_profile_problem(fixture.profile.profile_id, fixture.profiles)
        assert problem == (f"resource profile {fixture.profile.profile_id} records memory policy ({policy.describe()}), "
                           f"not the one {STEP4_CRITERIA_ID} measures (THP system, model-file release none)")


def test_a_candidate_identity_profile_is_never_admitted_while_d46_is_unchanged(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene(criteria_id=adapters.STEP4_CANDIDATE_CRITERIA_ID, memory_policy=CANDIDATE_POLICY)
    problem = accepted_profile_problem(fixture.profile.profile_id, fixture.profiles)
    assert problem == f"resource profile {fixture.profile.profile_id} was not judged against {STEP4_CRITERIA_ID}"
    (status,) = resolve([adapters.AdapterManifest(**fixture.manifest)], profiles=fixture.profiles)
    assert status.state is adapters.AdapterState.UNAVAILABLE
    world = World(tmp_path)
    with pytest.raises(StartupRefused) as error:
        build(world, tmp_path, CANDIDATE_POLICY, fixture.options, scene_config(fixture), fixture.profiles)
    assert error.value.label.startswith("scene_not_admitted: ") and "not judged against" in error.value.label
    assert world.log == [] and not (tmp_path / "data" / "sentinel.db").exists()


@pytest.mark.parametrize("policy", [CANDIDATE_POLICY, MemoryPolicy("workload_disabled", "none"),
                                    MemoryPolicy("system", "post_load")])
def test_a_runtime_policy_the_profile_did_not_measure_is_refused_before_anything_starts(tmp_path, accepted_scene,
                                                                                         policy) -> None:
    fixture = accepted_scene()  # accepted and measured with the default policy
    world = World(tmp_path)
    with pytest.raises(StartupRefused) as error:
        build(world, tmp_path, policy, fixture.options, scene_config(fixture), fixture.profiles)
    assert error.value.label == (
        f"scene_not_admitted: llama-lfm2-vl-scene: resource profile {fixture.profile.profile_id} measured memory "
        f"policy (THP system, model-file release none), not the runtime's ({policy.describe()})")
    assert world.log == [] and not (tmp_path / "data" / "sentinel.db").exists()
    config = scene_config(fixture)
    assert profile_mismatch(fixture.profile, config, fixture.options) is None  # the default still matches
    assert profile_mismatch(fixture.profile, config, fixture.options, policy=DEFAULT_POLICY) is None


# ---------------------------------------------------------------- the command line


@pytest.mark.parametrize(("flags", "policy"), [
    ([], DEFAULT_POLICY),
    (["--workload-thp-disable"], MemoryPolicy("workload_disabled", "none")),
    (["--post-load-release"], MemoryPolicy("system", "post_load")),
    (["--workload-thp-disable", "--post-load-release"], CANDIDATE_POLICY),
])
def test_the_run_flags_select_the_policy_and_default_to_none(tmp_path, monkeypatch, capsys, flags, policy) -> None:
    seen = []

    def refuse(config, options, devices, clock, **kwargs):
        seen.append(options)
        raise StartupRefused("stand-in")

    monkeypatch.setattr(cli, "assemble", refuse)
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 1\ncamera: {id: cam-1}\n")
    code = cli.main(["run", str(config), "--data-dir", str(tmp_path / "data"), "--engine", "e", "--status-port", "0",
                     *flags], devices=World(tmp_path).devices())
    assert code == 1 and capsys.readouterr().err.strip() == "run: stand-in"
    assert seen[0].memory_policy == policy and seen[0].scene is None
