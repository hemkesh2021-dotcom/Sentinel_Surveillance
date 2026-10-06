"""Step-4 PLR (D54, opt-in): step 4 with MR1's per-model release after each load and settle, then the full
warm-up and steady phases, judged by D47's criteria under its own identity.

Fakes only: no GPU, model, camera, journal or real cache release. Covers the distinct identity with
unchanged rules and thresholds, R's release inputs (missing is never met), the scene admission that
refuses the PLR identity, the workload's checkpoints before the unchanged warm-up/steady loop, the
profiler's command, budget and summary, and that step 4's and MR1's paths are unchanged.
"""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel import adapters

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
STEP4_RUN_INPUTS = ("guard_completed", "cleanup_clear", "check9_ok", "drop_declared", "profile_complete", "headless",
                    "no_dev_tools", "no_tracked_changes")
KERNEL = {"status": "observed", "oom_candidates": 0, "nvmap_candidates": 0}


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return SimpleNamespace(criteria=importlib.import_module("step4_criteria"),
                           profile=importlib.import_module("demo_profile"),
                           workload=importlib.import_module("demo_workload"))


def good_profile() -> dict:
    """Profile-side evidence that passes every D47 criterion (the same shape as test_step4's)."""
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


def run_inputs(plr: bool = True, **changes) -> dict:
    run = dict.fromkeys(STEP4_RUN_INPUTS, True)
    if plr:
        run.update(releases_recorded=True, release_calls_returned_0=True)
    run.update(changes)
    return run


# ---------------------------------------------------------------- identity, rules and thresholds


def test_the_plr_identity_is_distinct_and_its_admission_and_procedure_are_stated(mods) -> None:
    c = mods.criteria
    assert c.PLR_CRITERIA_ID == "step4plr-combined-cache-off-v2" != c.CRITERIA_ID == "step4-combined-cache-off-v2"
    result = c.evaluate(good_profile(), run=run_inputs(), kernel=KERNEL, identity={"status": "verified"},
                        criteria_id=c.PLR_CRITERIA_ID)
    assert result["criteria_id"] == c.PLR_CRITERIA_ID and result["eligible_for_maintainer_review"] is True
    assert result["accepted"] is False and "not admissible" in result["admission"]
    assert c.CRITERIA_ID in result["criteria_rules"] and "unchanged" in result["criteria_rules"]
    assert "120 s warm-up and 600 s steady" in result["procedure"]
    default = c.evaluate(good_profile(), run=run_inputs(plr=False), kernel=KERNEL, identity={"status": "verified"})
    assert default["criteria_id"] == c.CRITERIA_ID and not {"procedure", "admission", "criteria_rules"} & set(default)
    with pytest.raises(ValueError):
        c.evaluate(good_profile(), run=run_inputs(), kernel=KERNEL, identity={"status": "verified"},
                   criteria_id="step4-combined-cache-off-v3")


@pytest.mark.parametrize("change", [
    lambda p: None,
    lambda p: p["steady_interval"].update(max_bytes=5_000_000_000),  # at the steady target: pass
    lambda p: p["steady_interval"].update(max_bytes=5_000_000_001, seconds_above_target=0.2),
    lambda p: p["combined"].update(run_peak_bytes=5_400_000_000),  # at the peak ceiling: pass
    lambda p: p["combined"].update(run_peak_bytes=5_400_000_001),
    lambda p: p["steady_trend"]["used"].update(slope_bytes_per_min=10_000_001),
    lambda p: p["workload_steady"]["detector"].update(unique_fps=14.49),
    lambda p: p["workload_steady"]["detector"]["windows"].update(min_fps=13.49),
    lambda p: p["workload_steady"]["detector"]["schedule_age_ms"].update(p95=150.1),
    lambda p: p["workload_steady"]["face"].update(achieved_hz=0.94),
    lambda p: p["workload_steady"]["scene"].update(attempts=139),
    lambda p: p["workload_steady"]["scene"].update(valid_reports=142),
    lambda p: p["steady_interval"].update(coverage=0.94),
    lambda p: p["cache_evidence"].update(verdict="unverified"),
])
def test_every_criterion_and_threshold_reads_the_same_under_both_identities(mods, change) -> None:
    c = mods.criteria
    profile = good_profile()
    change(profile)
    kwargs = dict(kernel=KERNEL, identity={"status": "verified"})
    d47 = c.evaluate(copy.deepcopy(profile), run=run_inputs(plr=False), **kwargs)
    plr = c.evaluate(copy.deepcopy(profile), run=run_inputs(), criteria_id=c.PLR_CRITERIA_ID, **kwargs)
    assert {k: v["status"] for k, v in plr["criteria"].items()} == {k: v["status"] for k, v in d47["criteria"].items()}
    assert {k: v["requirement"] for k, v in plr["criteria"].items() if k != "R_valid_run"} == {
        k: v["requirement"] for k, v in d47["criteria"].items() if k != "R_valid_run"}
    assert plr["blocking"] == d47["blocking"]


@pytest.mark.parametrize("change", [
    {"releases_recorded": False}, {"release_calls_returned_0": False}, {"releases_recorded": None},
    {"release_calls_returned_0": "yes"}, {"releases_recorded": 1},
])
def test_plr_validity_needs_its_releases_and_missing_is_never_met(mods, change) -> None:
    c = mods.criteria
    result = c.evaluate(good_profile(), run=run_inputs(**change), kernel=KERNEL, identity={"status": "verified"},
                        criteria_id=c.PLR_CRITERIA_ID)
    assert result["criteria"]["R_valid_run"]["status"] == "fail" and not result["eligible_for_maintainer_review"]
    assert all(item["status"] == "unavailable" for name, item in result["criteria"].items()
               if name not in ("R_valid_run", "K_kernel", "I_identity"))
    without = {k: v for k, v in run_inputs().items() if k not in change}
    missing = c.evaluate(good_profile(), run=without, kernel=KERNEL, identity={"status": "verified"},
                         criteria_id=c.PLR_CRITERIA_ID)
    assert missing["criteria"]["R_valid_run"]["status"] == "fail"


def test_step4s_own_r_is_unchanged_and_ignores_the_release_inputs(mods) -> None:
    c = mods.criteria
    result = c.evaluate(good_profile(), run=run_inputs(plr=False, releases_recorded=False), kernel=KERNEL,
                        identity={"status": "verified"})
    assert result["criteria"]["R_valid_run"]["status"] == "pass"
    assert result["criteria"]["R_valid_run"]["requirement"] == (
        "guard completed, cleanup clear, same-boot Check 9, cache drop declared, profile complete, headless, "
        "no dev tools, no tracked changes")


def test_a_profile_judged_under_the_plr_identity_never_admits_scene(accepted_scene, mods) -> None:
    passing = accepted_scene()
    assert adapters.accepted_profile_problem(passing.profile.profile_id, passing.profiles) is None
    fixture = accepted_scene(criteria_id=mods.criteria.PLR_CRITERIA_ID)
    problem = adapters.accepted_profile_problem(fixture.profile.profile_id, fixture.profiles)
    assert problem == f"resource profile {fixture.profile.profile_id} was not judged against step4-combined-cache-off-v2"
    (status,) = adapters.resolve([adapters.AdapterManifest(**fixture.manifest)], profiles=fixture.profiles)
    assert status.state is adapters.AdapterState.UNAVAILABLE and "not judged against" in status.reason
    assert all(p.criteria_id != mods.criteria.PLR_CRITERIA_ID for p in adapters.RESOURCE_PROFILES.values())


# ---------------------------------------------------------------- workload


@pytest.mark.parametrize("flags", [[], ["--post-load-release"], ["--mr1-release-check"]])
def test_the_workload_releases_at_both_checkpoints_then_runs_the_unchanged_loop_only_with_plr(
    mods, monkeypatch, flags,
) -> None:
    workload = mods.workload
    calls = []

    class FakeSampler:
        def __init__(self, provider):
            self.phase = "detector_settle"

        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")

    monkeypatch.setattr(workload, "check_cuda_driver", lambda: True)
    monkeypatch.setattr(workload, "load_detector", lambda engine: calls.append("detector") or "model")
    monkeypatch.setattr(workload, "load_face", lambda: calls.append("face") or "deepface")
    monkeypatch.setattr(workload, "AllocatorSampler", FakeSampler)
    monkeypatch.setattr(workload, "run_workload", lambda model, face, args, stats, allocator: calls.append(
        f"workload {args.warmup_s:g}/{args.steady_s:g}"))
    monkeypatch.setattr(workload, "mr1_checkpoint", lambda stage: calls.append(f"checkpoint {stage}"))
    monkeypatch.setattr(workload, "run_smoke", lambda model, face, args: calls.append("smoke") or {"detector": {}})
    monkeypatch.setattr(workload, "event", lambda kind, **fields: calls.append(f"event {kind}"))
    monkeypatch.setattr(workload.time, "sleep", lambda seconds: None)
    assert workload.main(["--engine", "e", "--port", "1", *flags]) == 0
    steps = [c for c in calls if not c.startswith("event phase") and c != "event cuda_driver"]
    if flags == ["--post-load-release"]:
        assert steps == ["detector", "start", "checkpoint detector", "face", "checkpoint face", "workload 120/600",
                         "stop"]
    elif flags == ["--mr1-release-check"]:  # MR1 unchanged
        assert steps == ["detector", "start", "checkpoint detector", "face", "checkpoint face", "smoke",
                         "event mr1_smoke", "checkpoint after_smoke", "stop"]
    else:  # step 4 unchanged
        assert steps == ["detector", "start", "face", "workload 120/600", "stop"]


@pytest.mark.parametrize("argv", [
    ["--scene-only", "--port", "1", "--post-load-release"],
    ["--engine", "e", "--port", "1", "--post-load-release", "--mr1-release-check"],
])
def test_the_workload_refuses_plr_with_scene_only_or_mr1(mods, argv) -> None:
    with pytest.raises(SystemExit):
        mods.workload.main(argv)


# ---------------------------------------------------------------- profiler


@pytest.mark.parametrize("flags", [[], ["--post-load-release"], ["--mr1-release-check"]])
def test_the_workload_command_gains_only_the_plr_flag_and_a_stdin_pipe(mods, monkeypatch, tmp_path, flags) -> None:
    profile = mods.profile
    launched = []
    monkeypatch.setattr(profile.subprocess, "Popen", lambda argv, **kwargs: launched.append((argv, kwargs)))
    args = profile.parse_args(["--clip", "c.mp4", *flags])
    profile.start_workload(args, tmp_path)
    (argv, kwargs), = launched
    default = [str(args.python), str(profile.WORKLOAD), "--engine", str(args.engine), "--port", "18081",
               "--fps", "15.0", "--face-hz", "1.0", "--scene-interval-s", "4.0", "--settle-s", "15.0",
               "--warmup-s", "120.0", "--steady-s", "600.0", "--clip", "c.mp4"]
    assert argv == default + flags
    assert kwargs.get("stdin") == (subprocess.PIPE if flags else None)


def test_plr_is_off_by_default_has_the_release_settle_and_is_refused_with_scene_only_or_mr1(mods, monkeypatch) -> None:
    profile = mods.profile
    args = profile.parse_args([])
    assert args.post_load_release is False and args.mr1_release_check is False and args.mr1_release_settle_s == 5.0
    assert profile.parse_args(["--release-settle-s", "4.0"]).mr1_release_settle_s == 4.0
    assert profile.parse_args(["--mr1-release-settle-s", "3.0"]).mr1_release_settle_s == 3.0
    monkeypatch.setattr(profile, "scan_processes", lambda: [])
    monkeypatch.setattr(profile, "command_output", lambda argv, env=None: "inactive")
    monkeypatch.setattr(profile, "port_in_use", lambda port: False)
    for flags, text in ((["--scene-only", "--post-load-release"], "--post-load-release needs the detector"),
                        (["--post-load-release", "--mr1-release-check"], "separate procedures")):
        problems, _ = profile.preconditions(profile.parse_args(flags))
        assert any(text in problem for problem in problems)


def test_the_plr_budget_adds_only_the_two_releases_and_fits_step4s_deadline(mods, monkeypatch) -> None:
    profile = mods.profile
    monkeypatch.syspath_prepend(str(RUNNERS))
    operator = importlib.import_module("operator_check")
    step4 = profile.parse_args(operator.step4_argv(Path("/out"), Path("/clip.mp4"))[2:])
    plr = profile.parse_args(operator.step4plr_argv(Path("/out"), Path("/clip.mp4"))[2:])
    mr1 = profile.parse_args(operator.mr1_argv(Path("/out"), Path("/clip.mp4"))[2:])
    assert profile.workload_budget(step4) == 90 + 2 * 15 + 120 + 600 + 60  # unchanged
    assert profile.workload_budget(mr1) == 90 + 2 * 15 + 3 * (5 + 10) + 60 + 60  # unchanged
    assert profile.workload_budget(plr) == profile.workload_budget(step4) + 2 * (5 + 10)
    # Worst case inside the child: baseline, llama-server's ready timeout, its settle and release window, the
    # workload's whole budget, two unload settles. It stays under step 4's deadline, which PLR keeps.
    worst = plr.baseline_s + plr.llama_timeout_s + plr.settle_s + plr.mr1_release_settle_s + 10 \
        + profile.workload_budget(plr) + 2 * plr.settle_s
    assert worst < operator.STEP4_DEADLINE_S == 1200.0
    assert (plr.warmup_s, plr.steady_s, plr.face_hz, plr.scene_interval_s, plr.llama_cache_ram, plr.min_free_gb) == (
        step4.warmup_s, step4.steady_s, step4.face_hz, step4.scene_interval_s, step4.llama_cache_ram, step4.min_free_gb)


def snapshot(free: int, cached: int) -> dict:
    meminfo = {"MemTotal": 8_000_000_000, "MemFree": free, "MemAvailable": free + cached, "Cached": cached,
               "AnonPages": 1_500_000_000, "Mapped": 600_000_000}
    return {"t_mono": 1.0, "meminfo": meminfo, "pressure_bytes": 8_000_000_000 - free - cached,
            "processes": {"llama": {"rss_bytes": 2_000_000_000, "hwm_bytes": None, "pss_bytes": None}}}


def write_plr_run(profile, run_dir: Path, *, releases=("scene", "detector", "face"), steady=True,
                  flag: bool = True) -> None:
    parameters = {"face_hz": 1.0, "scene_interval_s": 4.0, "steady_s": 600.0}
    if flag:
        parameters.update(post_load_release=True, release_settle_s=5.0)
    manifest = {"run_id": "demo-profile-20991231T000000Z", "input": {"synthetic": False, "fps": 15.0},
                "parameters": parameters,
                "llama_server": {"cache_ram_mib": 0, "build_has_cache_ram_option": True,
                                 "flags": ["--n-gpu-layers", "999", "--ctx-size", "2048", "--parallel", "1",
                                           "--cache-ram", "0"]},
                "repository": {"commit": "a" * 40, "tracked_changes": False}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    series = [(t / 5, "baseline", 1_000_000_000, 6_000_000_000, 500_000_000) for t in range(50)]
    series += [(10 + t / 5, "warmup", 4_000_000_000, 2_400_000_000, 1_700_000_000) for t in range(600)]
    if steady:
        series += [(130 + t / 5, "steady", 4_100_000_000, 2_300_000_000 - t * 1000, 1_700_000_000 + t * 1000)
                   for t in range(3001)]
    with (run_dir / "memory.csv").open("w") as handle:
        handle.write(",".join(profile.Sampler.COLUMNS) + "\n")
        for t, phase, used, free, cached in series:
            row = dict.fromkeys(profile.Sampler.COLUMNS, "")
            row.update(t_mono=f"{t:.3f}", phase=phase, mem_total=8_000_000_000, mem_available=8_000_000_000 - used,
                       mem_free=free, cached=cached, swap_total=0, swap_free=0, pswpin=0, pswpout=0)
            handle.write(",".join(str(row[c]) for c in profile.Sampler.COLUMNS) + "\n")
    (run_dir / "tegrastats.log").write_text("")
    events, free, cached = [], 1_000_000_000, 2_700_000_000
    for component in releases:
        states = iter([snapshot(free, cached), snapshot(free + 400_000_000, cached - 400_000_000)])
        record = profile.release_component(
            component, {"model": Path("/home/someone/models/private.gguf")}, {}, 5.0, snapshot=lambda: next(states),
            release=lambda path: {"name": "private.gguf", "bytes": 10, "result": "returned_0", "returncode": 0,
                                  "error": None, "elapsed_s": 0.001},
            sleep=lambda seconds: None)
        events.append({"t_mono": 5.0, "source": "orchestrator", "event": "mr1_release", **record})
        free, cached = free + 400_000_000, cached - 400_000_000
    events.append({"t_mono": 1.0, "event": "llama_cmdline", "seen": True, "cache_ram": "0", "flags": []})
    if steady:
        events += [{"t_mono": 130.0, "event": "steady_boundary", "edge": "start", "boundary_t_mono": 130.0},
                   {"t_mono": 730.0, "event": "steady_boundary", "edge": "end", "boundary_t_mono": 730.0}]
    events.append({"t_mono": 900.0, "event": "run_end", "status": "complete" if steady else "aborted: x"})
    (run_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run_dir / "llama-server.log").write_text("t_mono=2.000 prompt cache is disabled\n")


def test_a_plr_summary_records_its_releases_and_trajectory_and_judges_under_its_identity(mods, tmp_path) -> None:
    profile = mods.profile
    write_plr_run(profile, tmp_path)
    text = profile.summarize(tmp_path)
    report = json.loads((tmp_path / "plr.json").read_text())
    written = json.loads((tmp_path / "profile.json").read_text())
    assert written["criteria_id"] == "step4plr-combined-cache-off-v2" and written["post_load_release"] == "plr.json"
    assert written["step4_profile_criteria"]["S_steady_interval"]["status"] == "pass"
    assert report["label"] == "step4plr-post-load-release" and report["profile_status"] == "complete"
    assert report["release_outcomes"] == dict.fromkeys(("scene", "detector", "face"), "memfree_rose_cached_fell")
    assert report["after_last_release"] == {"MemFree": 2_200_000_000, "Cached": 1_500_000_000}
    assert report["warmup"] == {"samples": 600, "mem_free_min_bytes": 2_400_000_000,
                                "cached_first_bytes": 1_700_000_000, "cached_last_bytes": 1_700_000_000}
    assert report["steady"]["mem_free_min_bytes"] == 2_300_000_000 - 3000 * 1000  # inside the monotonic interval
    assert report["steady"]["cached_last_bytes"] == 1_700_000_000 + 3000 * 1000
    assert any("same-boot arm" in item for item in report["cannot_establish"])
    assert "Post-load release (step-4 PLR, D54" in text and "(step4plr-combined-cache-off-v2: D47's rules" in text
    assert "MR1 release check" not in text and not (tmp_path / "mr1.json").exists()


def test_a_partial_plr_run_reports_what_was_not_reached(mods, tmp_path) -> None:
    profile = mods.profile
    write_plr_run(profile, tmp_path, releases=("scene",), steady=False)
    text = profile.summarize(tmp_path)
    report = json.loads((tmp_path / "plr.json").read_text())
    assert report["release_outcomes"] == {"scene": "memfree_rose_cached_fell", "detector": "not_reached",
                                          "face": "not_reached"}
    assert report["after_last_release"] is None and report["steady"] is None and report["profile_status"] == "aborted"
    assert "detector: not reached" in text and "steady interval: not reached" in text
    assert json.loads((tmp_path / "profile.json").read_text())["step4_profile_criteria"]["S_steady_interval"][
        "status"] == "unavailable"


def test_without_the_flag_the_summary_is_step4s_with_no_plr_record(mods, tmp_path) -> None:
    profile = mods.profile
    write_plr_run(profile, tmp_path, flag=False)
    text = profile.summarize(tmp_path)
    written = json.loads((tmp_path / "profile.json").read_text())
    assert not (tmp_path / "plr.json").exists() and "criteria_id" not in written and "post_load_release" not in written
    assert "Step-4 criteria decidable from this run (step4-combined-cache-off-v2; demo profile" in text
    assert "Post-load release" not in text and "step4plr" not in text
