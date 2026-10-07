"""The D58 candidate (opt-in): step 4 with the post-load release and the workload's verified THP disable, without
MA1's allocator, smaps or THP-counter sampling; judged by D47's rules and thresholds under its own identity.

Fakes only: no GPU, model, camera, journal, real cache release or THP change. Covers the identity (unchanged rules
and thresholds; R's release and THP inputs, missing never met), the workload's flag without attribution, the
profiler's flags, command and manifest, the release handshake that stops the run on a failed call, the cheap scope
reads, a full fake run of the profiler's orchestration (order, checkpoints, effect readings, aborts), and the
candidate record with its four inputs.
"""

from __future__ import annotations

import copy
import importlib
import io
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel import memory_policy

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
STEP4_RUN_INPUTS = ("guard_completed", "cleanup_clear", "check9_ok", "drop_declared", "profile_complete", "headless",
                    "no_dev_tools", "no_tracked_changes")
CANDIDATE_INPUTS = ("thp_disable_verified", "thp_scope_verified", "thp_effect_verified", "thp_settings_unchanged")
KERNEL = {"status": "observed", "oom_candidates": 0, "nvmap_candidates": 0}
MIB = 1 << 20
SNAPSHOT = {  # every THP setting key the profiler compares (21), read before and after
    "t_mono": 1.0, "kernel": {"release": "5.15.148-tegra", "version": "#1 SMP PREEMPT"}, "base_page_bytes": 4096,
    "thp_pmd_bytes": 2 * MIB, "hugetlb_default_bytes": 2 * MIB, "enabled": "always", "defrag": "madvise",
    "shmem_enabled": "never", "use_zero_page": 1,
    "khugepaged": {"defrag": 1, "scan_sleep_millisecs": 10000, "alloc_sleep_millisecs": 60000, "pages_to_scan": 4096,
                   "max_ptes_none": 511, "max_ptes_swap": 64, "max_ptes_shared": 256},
    "config": {"HZ": 250, "TRANSPARENT_HUGEPAGE": "y", "TRANSPARENT_HUGEPAGE_ALWAYS": "y",
               "TRANSPARENT_HUGEPAGE_MADVISE": "n", "READ_ONLY_THP_FOR_FS": "n"},
    "read_ms": 3.0,
}


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return SimpleNamespace(criteria=importlib.import_module("step4_criteria"),
                           profile=importlib.import_module("demo_profile"),
                           workload=importlib.import_module("demo_workload"),
                           operator=importlib.import_module("operator_check"))


def good_profile() -> dict:
    """Profile-side evidence that passes every D47 criterion (the same shape as test_step4plr's)."""
    return {
        "status": "complete",
        "steady_interval": {"status": "complete", "duration_s": 600.0, "coverage": 0.99, "max_gap_s": 0.3,
                            "ended_by": "workload_end", "max_bytes": 4_500_000_000, "seconds_above_target": 0.0,
                            "share_above_target": 0.0, "pswpin_delta": 0},
        "combined": {"run_peak_bytes": 4_700_000_000},
        "steady_trend": {"used": {"slope_bytes_per_min": 1_000_000}},
        "workload_steady": {
            "detector": {"unique_fps": 14.9, "processed_frames": 8940, "source_frames": 9000,
                         "windows": {"min_fps": 14.3}, "schedule_age_ms": {"p95": 70.0, "p99": 90.0}},
            "face": {"runs": 590, "achieved_hz": 0.98, "errors": {}, "error_count": 0},
            "scene": {"attempts": 150, "completed": 150, "errors": {}, "over_d16_timeout": 0, "valid_reports": 149,
                      "invalid_reports": 1, "finish_reasons": {"stop": 150}, "client_timeouts": 0, "http_errors": 0,
                      "transport_errors": 0},
        },
        "gpu_evidence": {"llama": {"ok": True}, "workload_cuda": {"cuinit": 0, "ok": True}},
        "provenance": {"llama_server": {"flags": ["--n-gpu-layers", "999", "--ctx-size", "2048", "--parallel", "1",
                                                  "--cache-ram", "0"]}},
        "cache_evidence": {"verdict": "disabled_verified", "missing": []},
    }


def run_inputs(**changes) -> dict:
    run = dict.fromkeys(STEP4_RUN_INPUTS, True)
    run.update(releases_recorded=True, release_calls_returned_0=True, **dict.fromkeys(CANDIDATE_INPUTS, True))
    run.update(changes)
    return run


# ---------------------------------------------------------------- identity, rules and thresholds


def test_the_candidate_identity_is_distinct_and_its_procedure_and_admission_are_stated(mods) -> None:
    c = mods.criteria
    assert c.CANDIDATE_CRITERIA_ID == "step4cand-wtd-plr-combined-cache-off-v2"
    assert len({c.CRITERIA_ID, c.PLR_CRITERIA_ID, c.CANDIDATE_CRITERIA_ID}) == 3
    assert c.CANDIDATE_RUN_INPUTS == CANDIDATE_INPUTS
    result = c.evaluate(good_profile(), run=run_inputs(), kernel=KERNEL, identity={"status": "verified"},
                        criteria_id=c.CANDIDATE_CRITERIA_ID)
    assert result["criteria_id"] == c.CANDIDATE_CRITERIA_ID and result["eligible_for_maintainer_review"] is True
    assert result["accepted"] is False and "not admissible" in result["admission"]
    assert c.CRITERIA_ID in result["admission"] and "same memory policy" in result["admission"]
    assert "unchanged" in result["criteria_rules"] and "four THP-disable inputs" in result["criteria_rules"]
    assert "transparent huge pages disabled in the workload process alone" in result["procedure"]
    assert c.MEMORY_POLICIES[c.CANDIDATE_CRITERIA_ID] == ("workload_disabled", "post_load")


@pytest.mark.parametrize("change", [
    lambda p: None,
    lambda p: p["steady_interval"].update(max_bytes=5_000_000_000),  # at the steady target: pass
    lambda p: p["steady_interval"].update(max_bytes=5_000_000_001, seconds_above_target=0.2),
    lambda p: p["combined"].update(run_peak_bytes=5_400_000_001),
    lambda p: p["steady_trend"]["used"].update(slope_bytes_per_min=10_000_000),  # at M3's limit: pass
    lambda p: p["steady_trend"]["used"].update(slope_bytes_per_min=10_000_001),
    lambda p: p["workload_steady"]["detector"].update(unique_fps=14.49),
    lambda p: p["workload_steady"]["detector"]["windows"].update(min_fps=13.49),
    lambda p: p["workload_steady"]["detector"]["schedule_age_ms"].update(p99=250.1),
    lambda p: p["workload_steady"]["face"].update(achieved_hz=0.94),
    lambda p: p["workload_steady"]["scene"].update(over_d16_timeout=1),
    lambda p: p["steady_interval"].update(coverage=0.94),
    lambda p: p["cache_evidence"].update(verdict="unverified"),
])
def test_every_criterion_and_threshold_reads_the_same_under_the_candidate_identity(mods, change) -> None:
    c = mods.criteria
    profile = good_profile()
    change(profile)
    kwargs = dict(kernel=KERNEL, identity={"status": "verified"})
    d47 = c.evaluate(copy.deepcopy(profile), run=dict.fromkeys(STEP4_RUN_INPUTS, True), **kwargs)
    cand = c.evaluate(copy.deepcopy(profile), run=run_inputs(), criteria_id=c.CANDIDATE_CRITERIA_ID, **kwargs)
    assert {k: v["status"] for k, v in cand["criteria"].items()} == {k: v["status"] for k, v in d47["criteria"].items()}
    assert {k: v["requirement"] for k, v in cand["criteria"].items() if k != "R_valid_run"} == {
        k: v["requirement"] for k, v in d47["criteria"].items() if k != "R_valid_run"}
    assert cand["blocking"] == d47["blocking"]


@pytest.mark.parametrize("key", [*CANDIDATE_INPUTS, "releases_recorded", "release_calls_returned_0"])
@pytest.mark.parametrize("value", [False, None, "yes", 1, "missing"])
def test_candidate_validity_needs_every_release_and_thp_input_and_missing_is_never_met(mods, key, value) -> None:
    c = mods.criteria
    run = run_inputs()
    if value == "missing":
        del run[key]
    else:
        run[key] = value
    result = c.evaluate(good_profile(), run=run, kernel=KERNEL, identity={"status": "verified"},
                        criteria_id=c.CANDIDATE_CRITERIA_ID)
    assert result["criteria"]["R_valid_run"]["status"] == "fail" and not result["eligible_for_maintainer_review"]
    assert all(item["status"] == "unavailable" for name, item in result["criteria"].items()
               if name not in ("R_valid_run", "K_kernel", "I_identity"))


def test_step4_and_plr_validity_never_read_the_thp_inputs(mods) -> None:
    c = mods.criteria
    off = dict.fromkeys(CANDIDATE_INPUTS, False)
    plr = c.evaluate(good_profile(), run=run_inputs(**off), kernel=KERNEL, identity={"status": "verified"},
                     criteria_id=c.PLR_CRITERIA_ID)
    d47 = c.evaluate(good_profile(), run={**dict.fromkeys(STEP4_RUN_INPUTS, True), **off}, kernel=KERNEL,
                     identity={"status": "verified"})
    assert plr["eligible_for_maintainer_review"] is True and d47["eligible_for_maintainer_review"] is True
    assert "candidate" not in plr["criteria"]["R_valid_run"]["requirement"] and "THP" not in plr["procedure"]


# ---------------------------------------------------------------- the workload's flag


def run_main(w, monkeypatch, argv):
    order, events = [], []
    monkeypatch.setattr(w, "event", lambda kind, /, **fields: events.append((kind, fields)))
    monkeypatch.setattr(w, "disable_thp_for_this_process",
                        lambda: order.append("thp_disable") or {"verified": True, "reason": None, "t_mono": 1.0})
    monkeypatch.setattr(w, "check_cuda_driver", lambda: order.append("cuda_driver") or False)
    monkeypatch.setattr(w, "load_detector", lambda engine: order.append("detector") or None)
    return w.main(argv), order, events


def test_the_workload_disables_thp_with_the_release_alone_before_the_driver_and_any_model(mods, monkeypatch) -> None:
    code, order, events = run_main(mods.workload, monkeypatch,
                                   ["--port", "1", "--engine", "e", "--post-load-release", "--thp-disable"])
    assert order[:2] == ["thp_disable", "cuda_driver"] and code == 3  # (the fake driver check fails next)
    assert events[0][0] == "thp_disable" and not any(name == "memattr_config" for name, _ in events)


@pytest.mark.parametrize("argv", [
    ["--thp-disable"], ["--memory-attribution", "--thp-disable"], ["--mr1-release-check", "--thp-disable"],
])
def test_the_workload_refuses_the_disable_without_the_post_load_release(mods, monkeypatch, argv) -> None:
    with pytest.raises(SystemExit):
        run_main(mods.workload, monkeypatch, ["--port", "1", "--engine", "e", *argv])
    with pytest.raises(SystemExit):
        run_main(mods.workload, monkeypatch, ["--port", "1", "--scene-only", "--thp-disable"])


# ---------------------------------------------------------------- the profiler's flags, command and manifest


@pytest.fixture
def quiet_host(mods, monkeypatch):
    p = mods.profile
    monkeypatch.setattr(p, "scan_processes", lambda: [])
    monkeypatch.setattr(p, "command_output", lambda argv, env=None: "inactive")
    monkeypatch.setattr(p, "port_in_use", lambda port: False)
    return p


CANDIDATE_FLAGS = ["--post-load-release", "--workload-thp-disable", "--candidate"]


@pytest.mark.parametrize(("flags", "fragment"), [
    (["--candidate"], "--candidate (D58) is the post-load release with the workload's THP disable"),
    (["--post-load-release", "--candidate"], "--candidate (D58) is the post-load release"),
    (["--workload-thp-disable", "--candidate"], "--candidate (D58) is the post-load release"),
    ([*CANDIDATE_FLAGS, "--memory-attribution"], "--candidate (D58) runs without MA1's sampling"),
    ([*CANDIDATE_FLAGS, "--memory-attribution", "--thp-observation"], "--candidate (D58) runs without MA1's sampling"),
    (["--post-load-release", "--workload-thp-disable"], "only with --thp-observation (MA1-WTD, D57) or --candidate"),
])
def test_the_candidate_is_off_by_default_and_refuses_any_other_combination(quiet_host, flags, fragment) -> None:
    p = quiet_host
    assert p.parse_args([]).candidate is False
    problems, _ = p.preconditions(p.parse_args(flags))
    assert any(fragment in problem for problem in problems), problems
    problems, _ = p.preconditions(p.parse_args(CANDIDATE_FLAGS))
    assert not any("--candidate" in problem or "--workload-thp-disable" in problem for problem in problems)


def test_the_candidates_workload_command_is_step4s_plus_the_release_and_the_disable_only(mods, monkeypatch,
                                                                                        tmp_path) -> None:
    p = mods.profile
    launched = []
    monkeypatch.setattr(p.subprocess, "Popen", lambda argv, **kwargs: launched.append(argv))
    step4 = mods.operator.step4_argv(tmp_path, Path("clip.mp4"))[2:]
    p.start_workload(p.parse_args(step4), tmp_path)
    p.start_workload(p.parse_args(mods.operator.step4cand_argv(tmp_path, Path("clip.mp4"))[2:]), tmp_path)
    assert launched[1] == [*launched[0], "--post-load-release", "--thp-disable"]
    assert "--memory-attribution" not in launched[1]


def test_the_candidate_manifest_records_its_policy_flag_settings_and_note(mods, monkeypatch, tmp_path) -> None:
    p = mods.profile
    monkeypatch.setattr(p, "command_output", lambda argv, env=None: "")
    monkeypatch.setattr(p, "llama_build_files", lambda binary: {})
    monkeypatch.setattr(p, "build_has_option", lambda files: None)
    monkeypatch.setattr(p, "thp_settings_snapshot", lambda: dict(SNAPSHOT))
    monkeypatch.setattr(p, "read_meminfo", lambda *a, **k: {})
    context = {"display_manager": "inactive", "desktop_processes": [], "dev_tools_running": []}
    candidate = p.provenance(p.parse_args(["--sanitized-logs", *CANDIDATE_FLAGS]), context, "demo-profile-x")
    assert candidate["memory_policy"] == {"thp": "workload_disabled", "model_file_release": "post_load"}
    assert candidate["parameters"]["candidate"] is True and candidate["thp_settings"]["before"] == SNAPSHOT
    assert "candidate_note" in candidate and "plr_note" not in candidate and "wtd_note" not in candidate
    assert "ma1_note" not in candidate and "thp_note" not in candidate
    step4 = p.provenance(p.parse_args(["--sanitized-logs"]), context, "demo-profile-y")
    assert step4["memory_policy"] == {"thp": "system", "model_file_release": "none"}
    assert "candidate" not in step4["parameters"] and "thp_settings" not in step4
    plr = p.provenance(p.parse_args(["--sanitized-logs", "--post-load-release"]), context, "demo-profile-z")
    assert plr["memory_policy"] == {"thp": "system", "model_file_release": "post_load"} and "plr_note" in plr


# ---------------------------------------------------------------- the release handshake and the scope reads


def handler_world(mods, results: dict[str, str] | None = None, *, raise_on: str | None = None):
    p = mods.profile
    results = results or {}
    written, terminated, records = [], [], []
    work = SimpleNamespace(pid=300, stdin=SimpleNamespace(write=written.append, flush=lambda: None),
                           terminate=lambda: terminated.append(True))
    events = SimpleNamespace(add=lambda source, name, **fields: records.append((name, fields)))

    def release(stage, files, pids, settle_s):
        if stage == raise_on:
            raise RuntimeError("boom")
        return {"component": stage, "outcome": "x",
                "files": {label: {"result": results.get(label, "returned_0")} for label in files}}

    files = {component: {label: Path(label) for label in labels}
             for component, labels in memory_policy.RELEASE_FILE_ROLES.items()}
    return p, {"llama": None, "work": work}, events, release, files, written, terminated, records


def test_with_the_candidate_rule_a_good_release_is_acknowledged_as_before(mods) -> None:
    p, procs, events, release, files, written, terminated, records = handler_world(mods)
    handle = p.mr1_checkpoint_handler(procs, events, lambda name: None, files, 0.0, release=release,
                                      require_returned_0=True)
    for stage in ("scene", "detector", "face"):
        handle(stage)
    assert written == ["ack detector\n", "ack face\n"] and terminated == [] and handle.failures == []
    assert [name for name, _ in records] == ["mr1_release"] * 3


@pytest.mark.parametrize(("stage", "results", "raise_on", "problem"), [
    ("detector", {"engine": "returned_error"}, None, "engine:returned_error"),
    ("face", {"face_detection_yunet_2023mar.onnx": "open_failed"}, None,
     "face_detection_yunet_2023mar.onnx:open_failed"),
    ("detector", {}, "detector", "error:RuntimeError"),
])
def test_a_failed_release_is_never_acknowledged_and_stops_the_waiting_workload(mods, stage, results, raise_on,
                                                                               problem) -> None:
    p, procs, events, release, files, written, terminated, records = handler_world(mods, results, raise_on=raise_on)
    handle = p.mr1_checkpoint_handler(procs, events, lambda name: None, files, 0.0, release=release,
                                      require_returned_0=True)
    handle(stage)
    assert written == [] and terminated == [True] and handle.failures == [stage]
    assert ("release_failed", {"component": stage, "problem": problem}) in records


def test_a_failed_scene_release_is_listed_for_the_main_thread_with_nothing_to_terminate(mods) -> None:
    p, procs, events, release, files, written, terminated, records = handler_world(mods, {"mmproj": "unsupported"})
    handle = p.mr1_checkpoint_handler(procs, events, lambda name: None, files, 0.0, release=release,
                                      require_returned_0=True)
    handle("scene")
    assert handle.failures == ["scene"] and terminated == [] and written == []


def test_without_the_candidate_rule_a_failed_call_stays_descriptive_as_in_plr(mods) -> None:
    p, procs, events, release, files, written, terminated, records = handler_world(mods, {"engine": "returned_error"})
    handle = p.mr1_checkpoint_handler(procs, events, lambda name: None, files, 0.0, release=release)
    handle("detector")
    assert written == ["ack detector\n"] and terminated == [] and handle.failures == []
    assert "release_failed" not in [name for name, _ in records]


def test_the_pump_reads_at_the_steady_start_only_when_asked(mods, tmp_path) -> None:
    p = mods.profile
    lines = "".join(p.EVENT_PREFIX + json.dumps(record) + "\n" for record in (
        {"event": "steady_boundary", "edge": "start", "boundary_t_mono": 10.0},
        {"event": "steady_boundary", "edge": "end", "boundary_t_mono": 20.0}))
    for asked in (False, True):
        called = []
        events = p.Events(tmp_path / f"events-{asked}.jsonl", 0.0)
        p.pump_workload(SimpleNamespace(stdout=io.StringIO(lines)), tmp_path, events, lambda name, source="": None,
                        sanitized=True, on_wtd=called.append, **({"on_steady_start": called.append} if asked else {}))
        events.close()
        assert called == (["steady_start", "steady_end"] if asked else ["steady_end"])


# ---------------------------------------------------------------- a full fake run of the profiler's orchestration


class FakeLines:
    """A workload's stdout: scripted lines, then EOF; a terminated workload writes nothing more."""

    def __init__(self, lines: list[str], done: threading.Event) -> None:
        self.lines, self.done, self.stopped = list(lines), done, False

    def readline(self, limit: int = -1) -> str:
        if self.stopped or not self.lines:
            self.done.set()
            return ""
        return self.lines.pop(0)


class FakeWorkload:
    def __init__(self, lines: list[str], log: list) -> None:
        self.pid, self.returncode, self.log = 300, None, log
        self.done = threading.Event()
        self.stdout = FakeLines(lines, self.done)
        self.stdin = SimpleNamespace(write=lambda text: log.append(text.strip()), flush=lambda: None)

    def terminate(self) -> None:
        self.log.append("workload_terminated")
        self.stdout.stopped = True
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.done.wait(10)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class FakeServer:
    def __init__(self) -> None:
        self.pid, self.returncode = 200, None

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def workload_lines(p, *, anon: int = 0) -> list[str]:
    def line(event: str, **fields) -> str:
        return p.EVENT_PREFIX + json.dumps({"event": event, **fields}) + "\n"

    return [
        line("thp_disable", t_mono=1.0, requested=True, model_modules_loaded=[], set_rc=0, set_errno=None, get_value=1,
             thp_enabled=0, anon_huge_pages_bytes=anon, verified=True, reason=None),
        line("cuda_driver", cuinit=0, libcuda=[], ok=True), line("phase", name="detector_load"),
        line("detector_loaded", seconds=8.0), line("phase", name="detector_settle"),
        line("mr1_checkpoint", stage="detector"), line("phase", name="face_load"), line("face_loaded", seconds=14.0),
        line("phase", name="face_settle"), line("mr1_checkpoint", stage="face"), line("phase", name="warmup"),
        line("phase", name="steady"), line("steady_boundary", edge="start", boundary_t_mono=20.0),
        line("steady_boundary", edge="end", boundary_t_mono=30.0), line("phase", name="stopping"),
        line("workload_stats", seconds=10.0, clip_loops=1, clip_loop_t_mono=[25.0]),
    ]


def fake_run(mods, monkeypatch, tmp_path, *, release_results: dict[str, str] | None = None, anon: int = 0):
    """Drive demo_profile.run() for the candidate with every device, process and file read faked; nothing sleeps."""
    p = mods.profile
    log: list[str] = []
    release_results = release_results or {}
    own = os.getpid()
    monkeypatch.setattr(p, "preconditions", lambda args: ([], {"display_manager": "inactive", "desktop_processes": [],
                                                                "dev_tools_running": []}))
    monkeypatch.setattr(p, "provenance", lambda args, context, run_id: {
        "run_id": run_id, "parameters": {"candidate": True}, "thp_settings": {"before": dict(SNAPSHOT)},
        "memory_policy": {"thp": "workload_disabled", "model_file_release": "post_load"}})

    class Quiet:
        def __init__(self, *args, **kwargs) -> None:
            self.phase, self.pids, self.floor_hit = "start", {}, False
            log.append(f"sampler:thp={kwargs.get('thp', False)}") if args and str(args[0]).endswith(".csv") else None

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr(p, "Sampler", Quiet)
    monkeypatch.setattr(p, "Tegrastats", Quiet)
    monkeypatch.setattr(p, "MapsSampler", lambda *a, **k: pytest.fail("the candidate never reads smaps periodically"))
    monkeypatch.setattr(p.signal, "signal", lambda *args: None)
    monkeypatch.setattr(p.time, "sleep", lambda seconds: log.append(f"sleep:{seconds:g}"))
    monkeypatch.setattr(p, "read_meminfo", lambda *a, **k: {"MemFree": 7_000_000_000})
    monkeypatch.setattr(p, "start_llama", lambda args, run_dir: (log.append("llama_start") or FakeServer(),
                                                                 SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(p, "wait_ready", lambda port, proc, timeout: True)
    monkeypatch.setattr(p, "llama_gpu_check", lambda proc, path: {"ok": True})
    monkeypatch.setattr(p, "cmdline_evidence", lambda pid: {"seen": True})
    monkeypatch.setattr(p, "mr1_snapshot", lambda pids: {"t_mono": 1.0, "meminfo": {}, "pressure_bytes": None,
                                                         "processes": {}})

    def release(path):
        log.append(f"release:{Path(path).name}")
        result = release_results.get(Path(path).name, "returned_0")
        return {"name": Path(path).name, "bytes": 1, "result": result, "returncode": 0, "error": None,
                "elapsed_s": 0.0}

    monkeypatch.setattr(p, "release_file_cache", release)
    workload = FakeWorkload(workload_lines(p, anon=anon), log)
    monkeypatch.setattr(p, "start_workload", lambda args, run_dir: log.append("workload_start") or workload)
    monkeypatch.setattr(p, "read_thp_enabled", lambda pid: log.append(f"thp:{pid}") or {own: 1, 200: 1, 300: 0}[pid])
    monkeypatch.setattr(p, "read_anon_huge_pages", lambda pid: log.append(f"anon:{pid}") or anon)
    monkeypatch.setattr(p, "thp_settings_snapshot", lambda: log.append("settings") or dict(SNAPSHOT))
    monkeypatch.setattr(p, "scan_processes", lambda: [])
    monkeypatch.setattr(p, "port_in_use", lambda port: False)
    monkeypatch.setattr(p, "sha256_of", lambda path: None)
    monkeypatch.setattr(p, "llama_build_files", lambda binary: {})
    monkeypatch.setattr(p, "summarize", lambda run_dir: "")
    argv = mods.operator.step4cand_argv(tmp_path, tmp_path / "clip.mp4")[2:]
    argv[argv.index("--release-settle-s") + 1] = "0"  # release_component's own wait is a real sleep
    code = p.run(p.parse_args(argv))
    (run_dir,) = list(tmp_path.glob("demo-profile-*"))
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return code, log, events, manifest


def test_the_candidate_run_keeps_the_profilers_order_and_reads_its_scope_cheaply(mods, monkeypatch, tmp_path) -> None:
    code, log, events, manifest = fake_run(mods, monkeypatch, tmp_path)
    own = os.getpid()
    assert code == 0 and manifest["status"] == "complete"
    assert log[:log.index("workload_start") + 1] == [
        "sampler:thp=False",  # no THP counters in memory.csv
        f"thp:{own}",  # before_launch
        "sleep:30",  # baseline
        "llama_start", f"thp:{own}", "thp:200",  # llama_ready
        "sleep:15",  # llama_settle
        "release:LFM2-VL-1.6B-Q4_0.gguf", "release:mmproj-LFM2-VL-1.6B-Q8_0.gguf",  # the scene's, before the workload
        "workload_start",
    ]
    after = log[log.index("workload_start") + 1:]
    assert after[:4] == [f"thp:{own}", "thp:200", "thp:300", "anon:300"]  # workload_verified, then its effect read
    assert after.index("release:yolov8n.engine") < after.index("ack detector") < after.index("ack face")
    assert after.index("ack face") < after.index("anon:300", 4)  # steady_start: the effect read alone
    assert after[-7:-3] == [f"thp:{own}", "thp:200", "thp:300", "anon:300"]  # steady_end, then its effect read
    assert after[-3:] == ["sleep:15", "sleep:15", "settings"]  # the two unloads, then the settings after the run
    scope = [e for e in events if e["event"] == "wtd_scope"]
    assert [e["checkpoint"] for e in scope] == list(mods.profile.WTD_CHECKPOINTS) and all(e["ok"] for e in scope)
    effect = [(e["checkpoint"], e["anon_huge_pages_bytes"]) for e in events if e["event"] == "cand_effect"]
    assert effect == [("workload_verified", 0), ("steady_start", 0), ("steady_end", 0)]
    assert manifest["thp_settings"]["after"] == SNAPSHOT and not [e for e in events if e["event"] == "memattr_maps"]


def test_a_failed_scene_release_stops_the_candidate_before_the_workload_starts(mods, monkeypatch, tmp_path) -> None:
    code, log, events, manifest = fake_run(mods, monkeypatch, tmp_path,
                                           release_results={"mmproj-LFM2-VL-1.6B-Q8_0.gguf": "returned_error"})
    assert code == 1 and "workload_start" not in log
    assert manifest["status"] == ("aborted: the scene model files' post-load release did not return 0 (D58); the "
                                  "workload was not started")
    assert [e["problem"] for e in events if e["event"] == "release_failed"] == ["mmproj:returned_error"]


def test_a_failed_detector_release_terminates_the_workload_and_aborts(mods, monkeypatch, tmp_path) -> None:
    code, log, events, manifest = fake_run(mods, monkeypatch, tmp_path,
                                           release_results={"yolov8n.engine": "open_failed"})
    assert code == 1 and "workload_terminated" in log and "ack detector" not in log and "ack face" not in log
    assert manifest["status"] == "aborted: the detector model files' post-load release did not return 0 (D58)"
    assert [e["component"] for e in events if e["event"] == "release_failed"] == ["detector"]


# ---------------------------------------------------------------- the candidate record and its four inputs


def write_events(run_dir: Path, *, startup: dict | None = None, load_t: float = 2.0, scope_ok: dict | None = None,
                 effect: dict | None = None, releases=("scene", "detector", "face")) -> None:
    scope_ok = scope_ok or {}
    effect = {"workload_verified": 0, "steady_start": 0, "steady_end": 0} if effect is None else effect
    records = [{"t_mono": 1.0, "event": "thp_disable", "requested": True, "model_modules_loaded": [], "set_rc": 0,
                "set_errno": None, "get_value": 1, "thp_enabled": 0, "anon_huge_pages_bytes": 0, "verified": True,
                "reason": None, **(startup or {})},
               {"t_mono": load_t, "event": "phase", "name": "detector_load"}]
    for name in ("before_launch", "llama_ready", "workload_verified", "steady_end"):
        records.append({"t_mono": 1.0, "event": "wtd_scope", "checkpoint": name, "thp_enabled": {},
                        "expected": {}, "ok": scope_ok.get(name, True)})
    for name, value in effect.items():
        records.append({"t_mono": 3.0, "event": "cand_effect", "checkpoint": name, "anon_huge_pages_bytes": value})
    for component in releases:
        roles = memory_policy.RELEASE_FILE_ROLES[component]
        records.append({"t_mono": 1.5, "event": "mr1_release", "component": component, "outcome": "memfree_rose_cached_fell",
                        "files": {role: {"result": "returned_0"} for role in roles}, "files_total_bytes": 1,
                        "calls_elapsed_s": 0.1, "settle_s": 5.0, "before": {}, "after": {}, "deltas": {}})
    (run_dir / "events.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))


def report(mods, run_dir: Path, *, after=SNAPSHOT, before=SNAPSHOT, stats=None) -> dict:
    manifest = {"thp_settings": {"before": before, "after": after},
                "memory_policy": {"thp": "workload_disabled", "model_file_release": "post_load"}}
    return mods.profile.candidate_report(run_dir, "complete", manifest, stats or {"clip_loops": 1,
                                                                                  "clip_loop_t_mono": [9.0]})


def test_a_verified_candidate_records_all_four_inputs_true(mods, tmp_path) -> None:
    write_events(tmp_path)
    checked = report(mods, tmp_path)
    assert checked["run_inputs"] == dict.fromkeys(CANDIDATE_INPUTS, True)
    assert checked["criteria_id"] == "step4cand-wtd-plr-combined-cache-off-v2"
    assert checked["release_problems"] == dict.fromkeys(("scene", "detector", "face")) and checked["release_failed"] == []
    assert checked["intervention"]["startup"]["before_model_load"] is True
    assert checked["intervention"]["effect"] == {"reference_bytes": 0, "readings": {
        "workload_verified": 0, "steady_start": 0, "steady_end": 0}}
    assert any("sentinel run behaves the same" in item for item in checked["cannot_establish"])


@pytest.mark.parametrize(("events", "settings", "failed"), [
    ({"startup": {"verified": False, "reason": "status_mismatch"}}, {}, "thp_disable_verified"),
    ({"load_t": 0.5}, {}, "thp_disable_verified"),  # the check came after the detector load began
    ({"scope_ok": {"llama_ready": False}}, {}, "thp_scope_verified"),
    ({"scope_ok": {"steady_end": False}}, {}, "thp_scope_verified"),
    ({"effect": {"workload_verified": 0, "steady_start": 0, "steady_end": 2 * MIB}}, {}, "thp_effect_verified"),
    ({"effect": {"workload_verified": 0, "steady_start": None, "steady_end": 0}}, {}, "thp_effect_verified"),
    ({"effect": {"workload_verified": 0, "steady_end": 0}}, {}, "thp_effect_verified"),  # a reading missing
    ({"startup": {"anon_huge_pages_bytes": None}}, {}, "thp_effect_verified"),  # no reference
    ({}, {"after": {**SNAPSHOT, "enabled": "madvise"}}, "thp_settings_unchanged"),
    ({}, {"after": None}, "thp_settings_unchanged"),
    ({}, {"after": {**SNAPSHOT, "config": None}}, "thp_settings_unchanged"),  # not comparable is never met
])
def test_each_unverified_part_fails_its_own_input_and_only_it(mods, tmp_path, events, settings, failed) -> None:
    write_events(tmp_path, **events)
    checked = report(mods, tmp_path, **settings)
    assert checked["run_inputs"] == {key: key != failed for key in CANDIDATE_INPUTS}


def test_huge_pages_made_before_the_check_may_stay_but_never_grow(mods, tmp_path) -> None:
    write_events(tmp_path, startup={"anon_huge_pages_bytes": 4 * MIB},
                 effect={"workload_verified": 4 * MIB, "steady_start": 2 * MIB, "steady_end": 4 * MIB})
    assert report(mods, tmp_path)["run_inputs"]["thp_effect_verified"] is True


def test_a_release_that_was_not_reached_or_failed_is_named(mods, tmp_path) -> None:
    write_events(tmp_path, releases=("scene",))
    checked = report(mods, tmp_path)
    assert checked["release_problems"] == {"scene": None, "detector": "not_reached", "face": "not_reached"}


def write_summary_run(p, run_dir: Path, *, candidate: bool) -> None:
    """A minimal run directory the summary can read (memory rows, events, manifest, logs)."""
    parameters = {"face_hz": 1.0, "scene_interval_s": 4.0, "steady_s": 600.0, "post_load_release": True,
                  "release_settle_s": 5.0, **({"workload_thp_disable": True, "candidate": True} if candidate else {})}
    manifest = {"run_id": "demo-profile-20991231T000000Z", "input": {"synthetic": False, "fps": 15.0},
                "parameters": parameters, "repository": {"commit": "a" * 40, "tracked_changes": False},
                "llama_server": {"cache_ram_mib": 0, "build_has_cache_ram_option": True,
                                 "flags": ["--n-gpu-layers", "999", "--ctx-size", "2048", "--parallel", "1",
                                           "--cache-ram", "0"]},
                "memory_policy": {"thp": "workload_disabled" if candidate else "system", "model_file_release": "post_load"},
                **({"thp_settings": {"before": SNAPSHOT, "after": SNAPSHOT}} if candidate else {})}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with (run_dir / "memory.csv").open("w") as handle:
        handle.write(",".join(p.Sampler.COLUMNS) + "\n")
        for t, phase in [(t / 5, "warmup") for t in range(50)] + [(10 + t / 5, "steady") for t in range(3001)]:
            row = dict.fromkeys(p.Sampler.COLUMNS, "")
            row.update(t_mono=f"{t:.3f}", phase=phase, mem_total=8_000_000_000, mem_available=4_000_000_000,
                       mem_free=2_000_000_000, cached=1_000_000_000, swap_total=0, swap_free=0, pswpin=0, pswpout=0,
                       work_rss=2_000_000_000)
            handle.write(",".join(str(row[c]) for c in p.Sampler.COLUMNS) + "\n")
    (run_dir / "tegrastats.log").write_text("")
    (run_dir / "llama-server.log").write_text("t_mono=2.000 prompt cache is disabled\n")
    write_events(run_dir)
    with (run_dir / "events.jsonl").open("a") as handle:
        for record in ({"t_mono": 10.0, "event": "steady_boundary", "edge": "start", "boundary_t_mono": 10.0},
                       {"t_mono": 610.0, "event": "steady_boundary", "edge": "end", "boundary_t_mono": 610.0},
                       {"t_mono": 700.0, "event": "run_end", "status": "complete"}):
            handle.write(json.dumps(record) + "\n")


def test_the_summary_writes_the_candidate_record_and_judges_under_its_identity(mods, tmp_path) -> None:
    p = mods.profile
    write_summary_run(p, tmp_path, candidate=True)
    text = p.summarize(tmp_path)
    written = json.loads((tmp_path / "profile.json").read_text())
    assert written["criteria_id"] == "step4cand-wtd-plr-combined-cache-off-v2" and written["candidate"] == "candidate.json"
    assert written["memory_policy"] == {"thp": "workload_disabled", "model_file_release": "post_load"}
    assert (tmp_path / "plr.json").exists() and not (tmp_path / "wtd.json").exists()
    checked = json.loads((tmp_path / "candidate.json").read_text())
    assert checked["run_inputs"] == dict.fromkeys(CANDIDATE_INPUTS, True)
    assert "D58 candidate (post-load release and the workload's THP disable" in text
    assert "(step4cand-wtd-plr-combined-cache-off-v2: D47's rules and thresholds under the D58 candidate" in text
    assert "MA1 memory attribution" not in text and "THP observation (D56" not in text


def test_without_the_candidate_flag_a_plr_summary_is_unchanged(mods, tmp_path) -> None:
    p = mods.profile
    write_summary_run(p, tmp_path, candidate=False)
    text = p.summarize(tmp_path)
    written = json.loads((tmp_path / "profile.json").read_text())
    assert written["criteria_id"] == "step4plr-combined-cache-off-v2" and "candidate" not in written
    assert not (tmp_path / "candidate.json").exists() and "D58 candidate" not in text
