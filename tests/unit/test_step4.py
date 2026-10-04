"""D47: step-4 criteria v2 and the guarded step4 mode, with fakes only (no GPU, model, camera or journal).

Covers the failure paths: steady boundaries (teardown excluded, missing or early
boundaries), cache evidence that is missing or contradictory, Check 9 and identity
prerequisites, the post-run wait and bounded kernel queries with uncertain
coverage, identity changes, and the startup identity checks. Nothing here
accepts a profile.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib
import json
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel import adapters
from sentinel.demo_runtime import profile_mismatch
from sentinel.config import parse_config
from sentinel.scene.llama_server import SCENE_REQUEST_SHA256

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
COMMIT = "a" * 40
BOOT = "11111111-2222-4333-8444-555555555555"
SERVICES = (
    "ActiveState=inactive\nUnitFileState=enabled\nId=ollama.service\nLoadState=loaded\n\n"
    "Id=gdm.service\nNames=gdm.service display-manager.service\nActiveState=inactive\n"
    "UnitFileState=disabled\nLoadState=loaded\n"
)
VALID_REPORT = json.dumps({"persons_visible": 0, "fire_or_smoke": False, "threat": "none", "observations": [],
                           "uncertainty": "low", "summary": "An empty room."})


@pytest.fixture
def runners(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return SimpleNamespace(criteria=importlib.import_module("step4_criteria"),
                           operator=importlib.import_module("operator_check"),
                           profile=importlib.import_module("demo_profile"),
                           workload=importlib.import_module("demo_workload"))


# ---------------------------------------------------------------- thresholds in one place


def test_registry_thresholds_equal_the_runner_criteria(runners) -> None:
    c = runners.criteria
    assert adapters.STEP4_CRITERIA_ID == c.CRITERIA_ID == "step4-combined-cache-off-v2"
    pairs = [
        (adapters.STEP4_TARGET_STEADY_BYTES, c.TARGET_STEADY_BYTES), (adapters.STEP4_MAX_PEAK_BYTES, c.TARGET_PEAK_BYTES),
        (adapters.STEP4_MAX_STEADY_SLOPE_BYTES_PER_MIN, c.MAX_STEADY_SLOPE_BYTES_PER_MIN),
        (adapters.STEP4_MIN_STEADY_COVERAGE, c.MIN_STEADY_COVERAGE), (adapters.STEP4_MIN_UNIQUE_FPS, c.MIN_UNIQUE_FPS),
        (adapters.STEP4_MIN_WINDOW_FPS, c.MIN_WINDOW_FPS),
        (adapters.STEP4_MAX_SCHEDULE_AGE_P95_MS, c.MAX_SCHEDULE_AGE_P95_MS),
        (adapters.STEP4_MAX_SCHEDULE_AGE_P99_MS, c.MAX_SCHEDULE_AGE_P99_MS), (adapters.STEP4_MIN_FACE_HZ, c.MIN_FACE_HZ),
        (adapters.STEP4_MIN_SCENE_ATTEMPTS, c.MIN_SCENE_ATTEMPTS),
        (adapters.STEP4_MIN_SCENE_STRICT_VALID_SHARE, c.MIN_SCENE_STRICT_VALID_SHARE),
    ]
    assert all(a == b for a, b in pairs)
    assert c.TARGET_STEADY_BYTES == 5_000_000_000 and c.TARGET_PEAK_BYTES == 5_400_000_000


def test_guard_thresholds_and_cleanup_are_unchanged(runners) -> None:
    op = runners.operator
    assert (op.PRESSURE_STOP, op.FREE_FLOOR, op.AVAILABLE_FLOOR, op.SAMPLE_S) == (4_800_000_000, 1 << 30, 2 << 30, 0.2)
    assert op.STEP4_DEADLINE_S == 1200.0


# ---------------------------------------------------------------- steady interval


def rows(start: float, end: float, used: int = 4_000_000_000, step: float = 0.2, phase: str = "steady") -> list[dict]:
    out, t = [], start
    while t <= end + 1e-9:
        out.append({"t": round(t, 3), "phase": phase, "used": used, "mem_free": 2_000_000_000, "pswpin": 5, "pswpout": 7})
        t += step
    return out


def boundary(edge: str, t: float) -> dict:
    return {"event": "steady_boundary", "edge": edge, "boundary_t_mono": t}


def test_teardown_rows_labelled_steady_are_excluded(runners) -> None:
    samples = rows(100.0, 700.0) + rows(700.2, 740.0, used=6_000_000_000)  # still labelled "steady" while exiting
    info, inside = runners.criteria.steady_interval([boundary("start", 100.0), boundary("end", 700.0)], samples)
    assert info["status"] == "complete" and info["ended_by"] == "workload_end"
    assert info["max_bytes"] == 4_000_000_000 and info["seconds_above_target"] == 0
    assert all(r["t"] <= 700.0 for r in inside) and info["coverage"] >= 0.99 and info["max_gap_s"] <= 0.21
    assert info["pswpin_delta"] == 0


def test_time_above_the_steady_target_is_measured_not_hidden_by_a_percentile(runners) -> None:
    samples = rows(0.0, 600.0)
    for row in samples[100:110]:  # 2 s above 5.0 GB: p95 stays under the target
        row["used"] = 5_100_000_000
    info, _ = runners.criteria.steady_interval([boundary("start", 0.0), boundary("end", 600.0)], samples)
    assert info["p95_bytes"] == 4_000_000_000 and info["max_bytes"] == 5_100_000_000
    assert info["seconds_above_target"] == pytest.approx(2.0, abs=0.01)


def test_an_earlier_stop_boundary_truncates_and_missing_boundaries_are_unavailable(runners) -> None:
    c = runners.criteria
    events = [boundary("start", 0.0), {"event": "stop_boundary", "reason": "interrupted", "boundary_t_mono": 300.0},
              boundary("end", 600.0)]
    info, inside = c.steady_interval(events, rows(0.0, 600.0))
    assert (info["status"], info["ended_by"], info["duration_s"]) == ("truncated", "stop_boundary", 300.0)
    assert max(r["t"] for r in inside) <= 300.0
    assert c.steady_interval([boundary("end", 600.0)], rows(0.0, 600.0))[0] == {
        "status": "unavailable", "reason": "no steady start boundary"}
    assert c.steady_interval([boundary("start", 0.0)], rows(0.0, 600.0))[0]["status"] == "unavailable"
    no_swap = [{k: v for k, v in r.items() if k not in ("pswpin", "pswpout")} for r in rows(0.0, 600.0)]
    info, _ = c.steady_interval([boundary("start", 0.0), boundary("end", 600.0)], no_swap)
    assert info["pswpin_delta"] is None  # unavailable, never zero
    short, _ = c.steady_interval([boundary("start", 0.0), boundary("end", 500.0)], rows(0.0, 500.0))
    assert short["status"] == "truncated"


# ---------------------------------------------------------------- cache evidence


MANIFEST = {"llama_server": {"cache_ram_mib": 0, "build_has_cache_ram_option": True,
                             "flags": ["--n-gpu-layers", "999", "--ctx-size", "2048", "--parallel", "1", "--cache-ram", "0"]}}
CMDLINE = {"seen": True, "cache_ram": "0", "flags": []}
STARTUP_OFF = {"startup": {"enabled": False, "limit_mib": None, "t_mono": 5.0}, "steady_state_updates": None}


def test_cache_off_needs_every_independent_source(runners) -> None:
    c = runners.criteria
    assert c.cache_evidence(MANIFEST, CMDLINE, STARTUP_OFF, telemetry_timestamped=True)["verdict"] == "disabled_verified"
    no_stamp = c.cache_evidence(MANIFEST, CMDLINE, STARTUP_OFF, telemetry_timestamped=False)
    assert no_stamp["verdict"] == "unverified" and no_stamp["steady_state_updates"] is None
    assert c.cache_evidence(MANIFEST, None, STARTUP_OFF, telemetry_timestamped=True)["verdict"] == "unverified"
    assert c.cache_evidence(MANIFEST, CMDLINE, None, telemetry_timestamped=True)["verdict"] == "unverified"
    unknown_build = {"llama_server": {**MANIFEST["llama_server"], "build_has_cache_ram_option": None}}
    assert c.cache_evidence(unknown_build, CMDLINE, STARTUP_OFF, telemetry_timestamped=True)["verdict"] == "unverified"
    running_default = {"seen": True, "cache_ram": None, "flags": []}
    assert c.cache_evidence(MANIFEST, running_default, STARTUP_OFF, telemetry_timestamped=True)["verdict"] == "enabled"
    on = {"startup": {"enabled": True, "limit_mib": 8192, "t_mono": 5.0}, "steady_state_updates": 3}
    assert c.cache_evidence(MANIFEST, CMDLINE, on, telemetry_timestamped=True)["verdict"] == "enabled"
    updates = {**STARTUP_OFF, "steady_state_updates": 2}
    assert c.cache_evidence(MANIFEST, CMDLINE, updates, telemetry_timestamped=True)["verdict"] == "unverified"


def test_running_command_line_and_build_option_are_read_from_the_process_and_files(runners, tmp_path) -> None:
    p = runners.profile
    argv = b"\0".join([b"/x/llama-server", b"--model", b"/home/u/m.gguf", b"--mmproj", b"/home/u/p.gguf",
                       b"--host", b"127.0.0.1", b"--port", b"18081", b"--cache-ram", b"0"]) + b"\0"
    evidence = p.cmdline_evidence(1, read=lambda pid: argv)
    assert evidence == {"seen": True, "cache_ram": "0", "flags": ["--host", "127.0.0.1", "--port", "18081",
                                                                  "--cache-ram", "0"]}
    assert p.cmdline_evidence(1, read=lambda pid: (_ for _ in ()).throw(OSError()))["seen"] is False
    (tmp_path / "llama-server").write_bytes(b"\0usage")
    (tmp_path / "libllama-common.so.0.0.8932").write_bytes(b"x --cache-ram N x")
    (tmp_path / "libllama-common.so.0").symlink_to("libllama-common.so.0.0.8932")
    files = p.llama_build_files(tmp_path / "llama-server")
    assert sorted(files) == ["libllama-common.so.0.0.8932", "llama-server"]  # real files, not links
    assert p.build_has_option(files) is True
    (tmp_path / "libllama-common.so.0.0.8932").write_bytes(b"old build")
    assert p.build_has_option(files) is False
    assert p.build_has_option({"llama-server": tmp_path / "absent"}) is None


# ---------------------------------------------------------------- workload: boundaries before joins


def test_the_workload_ends_steady_before_joining_and_excludes_work_finished_in_teardown(runners, monkeypatch) -> None:
    w = runners.workload
    events: list[tuple[str, dict]] = []
    completions: list[float] = []

    class Frames:
        def __init__(self, clip, fps):
            self.loops = 0

        def read(self):
            return object()

    class Model:
        def track(self, frame, **kwargs):
            time.sleep(0.005)
            return []

    class Face:
        def represent(self, **kwargs):
            return []

    def scene_request(port, frame):
        if any(name == "steady_boundary" for name, _ in events):
            time.sleep(0.5)  # still running when the steady interval ends: it completes during teardown
        completions.append(time.monotonic())
        return {"choices": [{"finish_reason": "stop", "message": {"content": VALID_REPORT}}]}

    monkeypatch.setattr(w, "Frames", Frames)
    monkeypatch.setattr(w, "scene_request", scene_request)
    def fake_event(kind, /, **fields):
        events.append((kind, fields))

    monkeypatch.setattr(w, "event", fake_event)
    args = SimpleNamespace(clip=None, fps=30.0, face_hz=5.0, scene_interval_s=0.05, port=1, warmup_s=0.2, steady_s=0.6)
    allocator = SimpleNamespace(phase="face_settle")
    w.run_workload(Model(), Face(), args, w.Stats(), allocator)
    names = [name for name, _ in events]
    end_index = names.index("steady_boundary", names.index("steady_boundary") + 1)
    assert events[end_index + 1] == ("phase", {"name": "stopping"}) and names[-1] == "workload_stats"
    late = [fields for name, fields in events[end_index:] if name == "scene_progress"]
    assert late and all(fields["phase"] == "stopping" for fields in late)  # teardown work is never labelled steady
    start = events[names.index("steady_boundary")][1]["boundary_t_mono"]
    end = events[end_index][1]["boundary_t_mono"]
    stats = events[-1][1]
    assert stats["seconds"] == pytest.approx(end - start, abs=0.01)
    in_steady = sum(1 for t in completions if start <= t <= end + 0.001)
    assert stats["scene"]["completed"] == in_steady < sum(1 for t in completions if t >= start)
    assert stats["scene"]["attempts"] == stats["scene"]["completed"]
    det = stats["detector"]
    assert det["unique_fps"] == det["processed_fps"] and det["schedule_age_ms"]["n"] == det["processed_frames"]
    assert det["decode_to_result_age_ms"]["p95"] <= det["schedule_age_ms"]["max"] + 1
    assert stats["request_sha256"] == SCENE_REQUEST_SHA256  # the workload sends what the runtime would


def test_workload_counters_separate_timeouts_http_and_transport_errors_and_windows(runners) -> None:
    w = runners.workload
    stats = w.Stats()
    stats.vlm_ms = [1000.0, 9000.0]
    stats.vlm_errors = {"timeout": 2, "HTTP 500": 1, "ConnectionRefusedError": 1}
    stats.det_done = [100.0 + i / 15 for i in range(15 * 20)]  # 20 s at 15 fps
    stats.det_ms = [10.0] * len(stats.det_done)
    summary = w.Stats.summary(stats, 20.0, window_start=100.0)
    scene = summary["scene"]
    assert (scene["attempts"], scene["client_timeouts"], scene["http_errors"], scene["transport_errors"]) == (6, 2, 1, 1)
    assert scene["over_d16_timeout"] == 1
    assert summary["detector"]["windows"] == {"window_s": 10, "count": 2, "min_fps": 15.0, "max_fps": 15.0}


def test_errors_with_names_the_sanitizer_drops_still_fail_face_and_scene(runners) -> None:
    # Sanitized logs keep only recognised error names; the totals counted before sanitizing must still gate.
    w = runners.workload
    stats = w.Stats()
    stats.face_ms = [900.0] * 600
    stats.face_errors = {"OutOfMemory": 3}
    stats.vlm_ms = [4000.0] * 148
    stats.vlm_errors = {"RemoteDisconnected": 1, "IncompleteRead": 1}
    stats.vlm_finish_reasons = {"stop": 148}
    stats.vlm_valid = 148
    summary = runners.profile.sanitize_diagnostic(w.Stats.summary(stats, 600.0, window_start=0.0))
    assert summary["face"]["errors"] == {} and summary["scene"]["errors"] == {}  # the names were dropped
    assert summary["face"]["error_count"] == 3 and summary["scene"]["transport_errors"] == 2
    profile = good_profile()
    profile["workload_steady"]["face"] = summary["face"]
    profile["workload_steady"]["scene"] = summary["scene"]
    result = runners.criteria.evaluate_profile(profile)
    assert result["F_face"]["status"] == "fail"
    assert result["V1_scene_requests"]["status"] == "fail"


# ---------------------------------------------------------------- profile evaluation


def good_profile() -> dict:
    return {
        "status": "complete",
        "steady_interval": {"status": "complete", "duration_s": 600.0, "coverage": 0.99, "max_gap_s": 0.3,
                            "ended_by": "workload_end", "max_bytes": 4_500_000_000, "seconds_above_target": 0.0,
                            "share_above_target": 0.0, "pswpin_delta": 0},
        "combined": {"run_peak_bytes": 4_700_000_000},
        "steady_trend": {"used": {"slope_bytes_per_min": 1_000_000}},
        "workload_steady": {
            "detector": {"unique_fps": 14.9, "processed_frames": 8940, "source_frames": 9000,
                         "windows": {"min_fps": 14.3}, "schedule_age_ms": {"p95": 70.0, "p99": 90.0},
                         "decode_to_result_age_ms": {"p95": 60.0}},
            "face": {"runs": 590, "achieved_hz": 0.98, "errors": {}, "error_count": 0, "latency_ms": {"p95": 922.0}},
            "scene": {"attempts": 150, "completed": 150, "errors": {}, "over_d16_timeout": 0, "valid_reports": 149,
                      "invalid_reports": 1, "finish_reasons": {"stop": 150}, "rejected_reports_by_reason": {"invalid_report": 1},
                      "client_timeouts": 0, "http_errors": 0, "transport_errors": 0, "latency_ms": {"p95": 4000.0}},
        },
        "gpu_evidence": {"llama": {"ok": True, "layers": "offloaded 17/17 layers to GPU"},
                         "workload_cuda": {"cuinit": 0, "ok": True}},
        "provenance": MANIFEST,
        "cache_evidence": {"verdict": "disabled_verified", "missing": []},
    }


def test_a_good_profile_passes_the_profile_side_criteria(runners) -> None:
    result = runners.criteria.evaluate_profile(good_profile())
    assert {name: item["status"] for name, item in result.items()} == dict.fromkeys(result, "pass")


@pytest.mark.parametrize(
    ("change", "criterion", "status"),
    [
        (lambda p: p["steady_interval"].update(status="truncated"), "S_steady_interval", "fail"),
        (lambda p: p["steady_interval"].update(coverage=None), "S_steady_interval", "unavailable"),
        (lambda p: p["steady_interval"].update(max_bytes=5_000_000_001, seconds_above_target=0.2), "M1_steady_pressure", "fail"),
        (lambda p: p["combined"].update(run_peak_bytes=5_400_000_001), "M2_run_peak", "fail"),
        (lambda p: p["steady_interval"].update(pswpin_delta=None), "M3_trend_swap", "unavailable"),
        (lambda p: p["workload_steady"]["detector"]["windows"].update(min_fps=12.0), "T1_unique_throughput", "fail"),
        (lambda p: p["workload_steady"]["detector"]["schedule_age_ms"].update(p99=300.0), "T2_frame_age", "fail"),
        (lambda p: p["workload_steady"]["face"].update(errors={"ValueError": 1}), "F_face", "fail"),
        (lambda p: p["workload_steady"]["scene"].update(errors={"timeout": 1}), "V1_scene_requests", "fail"),
        (lambda p: p["workload_steady"]["face"].update(error_count=1), "F_face", "fail"),
        (lambda p: p["workload_steady"]["face"].pop("error_count"), "F_face", "unavailable"),
        (lambda p: p["workload_steady"]["scene"].update(transport_errors=1), "V1_scene_requests", "fail"),
        (lambda p: p["workload_steady"]["scene"].pop("http_errors"), "V1_scene_requests", "unavailable"),
        (lambda p: p["workload_steady"]["scene"].update(finish_reasons={"stop": 149, "length": 1}), "V2_scene_completion", "fail"),
        (lambda p: p["workload_steady"]["scene"].update(valid_reports=140), "V2_scene_completion", "fail"),
        (lambda p: p["gpu_evidence"].update(workload_cuda={}), "G_gpu", "unavailable"),
        (lambda p: p["cache_evidence"].update(verdict="unverified"), "C_prompt_cache", "unavailable"),
    ],
)
def test_each_profile_criterion_fails_or_is_unavailable_on_its_own(runners, change, criterion, status) -> None:
    profile = good_profile()
    change(profile)
    assert runners.criteria.evaluate_profile(profile)[criterion]["status"] == status


def test_an_invalid_run_or_uncertain_evidence_is_never_eligible_and_nothing_is_accepted(runners) -> None:
    c = runners.criteria
    run = dict.fromkeys(("guard_completed", "cleanup_clear", "check9_ok", "drop_declared", "profile_complete",
                         "headless", "no_dev_tools", "no_tracked_changes"), True)
    kernel = {"status": "observed", "oom_candidates": 0, "nvmap_candidates": 0}
    good = c.evaluate(good_profile(), run=run, kernel=kernel, identity={"status": "verified"})
    assert good["eligible_for_maintainer_review"] is True and good["accepted"] is False
    stopped = c.evaluate(good_profile(), run={**run, "guard_completed": False}, kernel=kernel,
                         identity={"status": "verified"})
    assert stopped["criteria"]["R_valid_run"]["status"] == "fail"
    assert all(item["status"] == "unavailable" for name, item in stopped["criteria"].items()
               if name not in ("R_valid_run", "K_kernel", "I_identity"))
    for kernel_case in ({"status": "uncertain", "oom_candidates": None}, {"status": "unavailable"},
                        {"status": "truncated"}):
        result = c.evaluate(good_profile(), run=run, kernel=kernel_case, identity={"status": "verified"})
        assert result["criteria"]["K_kernel"]["status"] == "unavailable" and not result["eligible_for_maintainer_review"]
    changed = c.evaluate(good_profile(), run=run, kernel=kernel, identity={"status": "changed"})
    assert changed["criteria"]["I_identity"]["status"] == "fail"
    assert c.evaluate(None, run=run, kernel=kernel, identity={"status": "verified"})["blocking"]


# ---------------------------------------------------------------- profiler summary with boundaries


def write_run(tmp_path: Path, profile_module, *, end: float | None = 610.0) -> Path:
    run = tmp_path / "demo-profile-20991231T000000Z"
    run.mkdir()
    manifest = {
        "run_id": run.name, "input": {"synthetic": False, "fps": 15.0}, "status": "complete",
        "parameters": {"face_hz": 1.0, "scene_interval_s": 4.0, "steady_s": 600.0},
        "repository": {"commit": COMMIT, "tracked_changes": False}, "boot_id": BOOT,
        "llama_server": MANIFEST["llama_server"],
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    with (run / "memory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(profile_module.Sampler.COLUMNS)
        series = [(t / 5, "baseline", 1_000_000_000) for t in range(0, 50)]
        series += [(10 + t / 5, "steady", 4_000_000_000) for t in range(0, 3001)]
        series += [(611 + t / 5, "steady", 6_000_000_000) for t in range(0, 50)]  # teardown still labelled steady
        for t, phase, used in series:
            row = {c: "" for c in profile_module.Sampler.COLUMNS}
            row.update(t_mono=f"{t:.3f}", phase=phase, mem_total=8_000_000_000, mem_available=8_000_000_000 - used,
                       mem_free=2_000_000_000, swap_total=0, swap_free=0, pswpin=0, pswpout=0)
            writer.writerow([row[c] for c in profile_module.Sampler.COLUMNS])
    events = [{"t_mono": 10.0, "event": "steady_boundary", "edge": "start", "boundary_t_mono": 10.0}]
    if end is not None:
        events.append({"t_mono": end, "event": "steady_boundary", "edge": "end", "boundary_t_mono": end})
    events += [{"t_mono": 1.0, "event": "llama_cmdline", **CMDLINE},
               {"t_mono": 900.0, "event": "run_end", "status": "complete"}]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run / "llama-server.log").write_text("t_mono=2.000 prompt cache is disabled\n")
    return run


def test_the_summary_uses_the_monotonic_interval_and_excludes_teardown(runners, tmp_path) -> None:
    run = write_run(tmp_path, runners.profile)
    text = runners.profile.summarize(run)
    profile = json.loads((run / "profile.json").read_text())
    steady = profile["steady_interval"]
    assert (steady["status"], steady["max_bytes"], steady["seconds_above_target"]) == ("complete", 4_000_000_000, 0.0)
    assert profile["combined"]["steady_p95_bytes"] == 4_000_000_000  # teardown rows at 6 GB excluded
    assert profile["steady_trend"]["seconds"] == 600.0  # the trend spans the interval, not the exit rows
    assert profile["steady_trend"]["used"]["last"] == 4_000_000_000
    assert profile["combined"]["run_peak_bytes"] == 6_000_000_000  # the run peak keeps every phase
    assert profile["cache_evidence"]["verdict"] == "disabled_verified"
    assert profile["step4_profile_criteria"]["M1_steady_pressure"]["status"] == "pass"
    assert "steady basis: monotonic boundaries" in text and "never acceptance" in text


def test_a_run_without_a_steady_end_has_no_steady_verdict(runners, tmp_path) -> None:
    run = write_run(tmp_path, runners.profile, end=None)
    text = runners.profile.summarize(run)
    profile = json.loads((run / "profile.json").read_text())
    assert profile["steady_interval"]["status"] == "unavailable" and "no steady verdict" in text
    assert profile["step4_profile_criteria"]["S_steady_interval"]["status"] == "unavailable"


def test_the_sanitized_pump_keeps_boundaries_and_fingerprints_only(runners, tmp_path) -> None:
    p = runners.profile
    assert p.sanitize_diagnostic({"event": "steady_boundary", "edge": "end", "boundary_t_mono": 5.5, "x": "secret"}) == {
        "event": "steady_boundary", "edge": "end", "boundary_t_mono": 5.5}
    assert p.sanitize_diagnostic({"request_sha256": "f" * 64})["request_sha256"] == "f" * 64
    assert p.sanitize_diagnostic({"request_sha256": "not a hash"})["request_sha256"] is None
    assert "stopping" in p.PHASES_IN_ORDER


# ---------------------------------------------------------------- operator: identity and step4


class Child:
    def __init__(self, backend, argv, *, duration=0.0, returncode=0, output=b"", on_done=None):
        self.backend, self.argv, self.end = backend, argv, backend.now + duration
        self.returncode, self.chunks, self.on_done, self.killed, self.closed = returncode, [output], on_done, False, False

    def poll(self):
        done = self.killed or self.backend.now >= self.end
        if done and self.on_done:
            self.on_done()
            self.on_done = None
        return self.returncode if done else None

    def read(self):
        return self.chunks.pop(0) if self.chunks else b""

    def exists(self):
        return not self.killed and self.poll() is None

    def signal(self, signum):
        if self.exists() and signum == signal.SIGKILL:
            self.killed, self.returncode = True, -9
        elif self.exists() and signum == signal.SIGTERM:
            self.killed, self.returncode = True, -15

    def close(self):
        self.closed = True


class Backend:
    def __init__(self, output: Path | None = None):
        self.now = 1000.0
        self.children: list[Child] = []
        self.output = output
        self.boot = BOOT
        self.pressure = 1_500_000_000  # below the guard's 2.0 GB admission limit
        self.journal = {"kernel": [], "markers": None, "journald": []}
        self.journal_status = {}
        self.profile = good_profile()
        self.manifest_overrides = {}
        self.child_plan = {}
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append((self.now, seconds))
        self.now += seconds

    def boot_id(self):
        return self.boot

    def sample(self):
        return {"t_mono": self.now, "MemTotal": 8_000_000_000, "MemAvailable": 8_000_000_000 - self.pressure,
                "MemFree": 4_000_000_000, "pswpin": 0, "pswpout": 0}

    def context(self):
        return {"boot_id": self.boot, "root": False, "port_18081_in_use": False, "cached_assets_available": True,
                "process_inspection_unavailable": False,
                "workloads": {k: {"count": 0, "pids": []} for k in
                              ("model_servers", "desktop", "dev_tools", "python_unclassified", "media_or_gpu_tools")}}

    def _journal_output(self, argv):
        if "-k" in argv and "-n" in argv:
            return {"output": json.dumps({"MESSAGE": "boot ok"}).encode()}
        kind = "kernel" if "-k" in argv else "markers" if "-t" in argv else "journald"
        plan = dict(self.journal_status.get(kind, {}))
        records = self.journal[kind]
        if records is None:  # markers: written by this run, at the logger calls' times
            records = [{"MESSAGE": f"step4 {edge} {self.output.name}", "__REALTIME_TIMESTAMP": str(int(t * 1e6))}
                       for edge, t in self.markers]
        plan.setdefault("output", "\n".join(json.dumps(r) for r in records).encode())
        return plan

    def spawn(self, argv, env):
        if "systemctl" in argv[0]:
            plan = {"output": SERVICES.encode()}
        elif "journalctl" in argv[0]:
            plan = self._journal_output(argv)
        elif argv[0] == "git":
            plan = {"output": COMMIT.encode()}
        elif "logger" in argv[0]:
            edge = argv[-1].split()[1]
            self.markers = [*getattr(self, "markers", []), (edge, self.wall())]
            plan = {}
        elif "--api" in argv:
            api = argv[argv.index("--api") + 1]
            plan = {"duration": 0.4, "output": json.dumps({
                "api": api, "status": "bounded_smoke_complete", "allocated_bytes": 256 << 20,
                "cap_bytes": 256 << 20, "chunk_bytes": 32 << 20, "cleanup_clear": True}).encode()}
        else:
            plan = {"duration": 900.0, "on_done": self._write_profiler_files, **self.child_plan}
        child = Child(self, argv, **plan)
        self.children.append(child)
        return child

    def wall(self):
        return 1_900_000_000 + self.now

    def _write_profiler_files(self):
        run = self.output / "demo-profile-20991231T000000Z"
        run.mkdir(exist_ok=True)
        start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.wall() - 900))
        finish = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.wall()))
        manifest = {"boot_id": BOOT, "started_utc": start, "finished_utc": finish, "display_manager": "inactive",
                    "desktop_processes": [], "dev_tools_running": [],
                    "repository": {"commit": COMMIT, "tracked_changes": False},
                    "sha256": {k.split(":", 1)[1]: v for k, v in self.identity_hashes.items() if k.startswith("model:")},
                    "sha256_build": {k.split(":", 1)[1]: v for k, v in self.identity_hashes.items() if k.startswith("build:")},
                    "sha256_clip": self.identity_hashes.get("clip"), **self.manifest_overrides}
        (run / "manifest.json").write_text(json.dumps(manifest))
        (run / "profile.json").write_text(json.dumps(self.profile))


def identity_tree(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "assets"
    root.mkdir(exist_ok=True)
    files = {}
    for label, size in (("clip", 30), ("build:llama-server", 20), ("build:libllama.so.0.0.8932", 10),
                        ("model:llm", 40), ("model:mmproj", 25), ("model:engine", 15)):
        path = root / label.replace(":", "_")
        path.write_bytes(bytes([size]) * size)
        files[label] = path
    return files


def private_dir(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"sentinel-operator-{name}"
    path.mkdir(mode=0o700)
    return path


def save(path: Path, report: dict) -> Path:
    target = path / "result.json"
    target.write_text(json.dumps(report))
    target.chmod(0o600)
    return target


def prepare(runners, tmp_path, *, clip_sha=None):
    op = runners.operator
    files = identity_tree(tmp_path)
    backend = Backend()
    clip_sha = clip_sha or hashlib.sha256(files["clip"].read_bytes()).hexdigest()
    identity = op.step4_identity(backend, op.ProcessRunner(backend), files["clip"], clip_sha, files=files)
    identity["finished_utc"] = "2099-01-01T00:00:00Z"
    identity_path = save(private_dir(tmp_path, "identity"), {"schema_version": 1, "mode": "step4_identity",
                                                             "step4_identity": identity})
    check9 = op.execute("check9", Backend(), private_dir(tmp_path, "check9"))
    check9["finished_utc"] = "2099-01-01T00:10:00Z"
    check9_path = save(tmp_path / "sentinel-operator-check9", check9)
    return SimpleNamespace(files=files, identity=identity, identity_path=identity_path, check9_path=check9_path)


def run_step4(runners, tmp_path, prep, backend=None, **overrides):
    op = runners.operator
    output = private_dir(tmp_path, "step4")
    backend = backend or Backend()
    backend.output = output
    backend.identity_hashes = {label: entry["sha256"] for label, entry in prep.identity["files"].items()}
    kwargs = dict(check9_report=prep.check9_path, identity_report=prep.identity_path, clip=prep.files["clip"],
                  confirm_step4=True, dropped_caches=True, identity_files_fn=lambda clip: prep.files)
    kwargs.update(overrides)
    return op.execute("step4", backend, output, **kwargs), backend


def test_the_identity_snapshot_hashes_and_times_every_file_and_checks_the_clip(runners, tmp_path) -> None:
    op = runners.operator
    files = identity_tree(tmp_path)
    good = hashlib.sha256(files["clip"].read_bytes()).hexdigest()
    backend = Backend()
    identity = op.step4_identity(backend, op.ProcessRunner(backend), files["clip"], good, files=files)
    assert identity["status"] == "complete" and identity["clip_matches_recorded"] is True
    assert set(identity["files"]) == set(files) and all(e["sha256"] and "hash_seconds" in e for e in identity["files"].values())
    assert identity["started_utc"] <= identity["finished_utc"] and "before the D37 cache drop" in identity["note"]
    assert op.step4_identity(backend, op.ProcessRunner(backend), files["clip"], "0" * 64, files=files)["status"] == "clip_mismatch"
    files["model:llm"].unlink()
    assert op.step4_identity(backend, op.ProcessRunner(backend), files["clip"], good, files=files)["status"] == "incomplete"


@pytest.mark.parametrize("case", ["latest_check9", "no_identity", "identity_after_check9", "identity_other_boot",
                                  "file_changed", "drop_undeclared", "unconfirmed", "failed_check9", "clip_mismatch"])
def test_step4_refuses_before_any_process_when_a_prerequisite_fails(runners, tmp_path, case) -> None:
    prep = prepare(runners, tmp_path, clip_sha="0" * 64 if case == "clip_mismatch" else None)
    overrides = {}
    if case == "latest_check9":
        overrides["explicit_check9"] = False
    elif case == "no_identity":
        overrides["identity_report"] = None
    elif case in ("identity_after_check9", "identity_other_boot"):
        report = json.loads(prep.identity_path.read_text())
        if case == "identity_after_check9":
            report["step4_identity"]["finished_utc"] = "2099-01-01T01:00:00Z"
        else:
            report["step4_identity"]["boot_id"] = "other-boot"
        prep.identity_path.write_text(json.dumps(report))
    elif case == "file_changed":
        prep.files["model:mmproj"].write_bytes(b"x" * 26)
    elif case == "drop_undeclared":
        overrides["dropped_caches"] = False
    elif case == "unconfirmed":
        overrides["confirm_step4"] = False
    elif case == "failed_check9":
        report = json.loads(prep.check9_path.read_text())
        report["check9"]["status"] = "inconclusive"
        prep.check9_path.write_text(json.dumps(report))
    result, backend = run_step4(runners, tmp_path, prep, **overrides)
    assert result["step4"]["status"] == "refused" and result["step4"]["refusals"]
    assert not [c for c in backend.children if "demo_profile.py" in " ".join(c.argv) or "logger" in c.argv[0]]


def test_a_completed_step4_waits_then_queries_the_recorded_boot_and_interval(runners, tmp_path) -> None:
    prep = prepare(runners, tmp_path)
    result, backend = run_step4(runners, tmp_path, prep)
    section = result["step4"]
    (child,) = [c for c in backend.children if "demo_profile.py" in " ".join(c.argv)]
    argv = child.argv
    assert argv[argv.index("--steady-s") + 1] == "600" and argv[argv.index("--llama-cache-ram") + 1] == "0"
    assert argv[argv.index("--clip") + 1] == str(prep.files["clip"]) and "--allow-dev-tools" not in argv
    assert section["status"] == "completed" and section["post_run_wait_s"] == 60.0
    queries = [c.argv for c in backend.children if "journalctl" in c.argv[0] and "-n" not in c.argv]
    assert len(queries) == 3 and all(q[q.index("-b") + 1] == BOOT for q in queries)
    kernel_query = next(q for q in queries if "-k" in q)
    manifest = json.loads(next(backend.output.glob("demo-profile-*/manifest.json")).read_text())
    started, finished = (runners.operator._utc(manifest[k]).timestamp() for k in ("started_utc", "finished_utc"))
    assert kernel_query[kernel_query.index("--since") + 1] == f"@{int(started) - 1}"
    assert kernel_query[kernel_query.index("--until") + 1] == f"@{int(finished + 60) + 1}"  # recorded run end + 60 s
    end_marker_time = backend.markers[-1][1]
    child_end = 1_900_000_000 + 1000.0 + 900.0
    assert end_marker_time >= child_end + 60  # the wait really happened before the end marker and the queries
    assert section["kernel"]["status"] == "observed" and section["kernel"]["oom_candidates"] == 0
    assert section["identity"]["status"] == "verified"
    assert section["criteria"]["eligible_for_maintainer_review"] is True and section["criteria"]["accepted"] is False
    assert result["hardware_acceptance"] == "PENDING"


@pytest.mark.parametrize(
    ("setup", "status", "reason"),
    [
        (lambda b: b.journal.update(markers=[]), "uncertain", "run markers not readable"),
        (lambda b: b.journal_status.update(kernel={"output": b"x" * (1 << 21)}), "truncated", "kernel query output truncated"),
        (lambda b: b.journal_status.update(kernel={"returncode": 1}), "unavailable", "kernel query unavailable"),
        (lambda b: b.journal_status.update(journald={"output": b"not json"}), "unavailable", "journald query unavailable"),
        (lambda b: b.journal.update(journald=[{"MESSAGE": "Missed 12 kernel messages"}]), "uncertain", "possible loss"),
        (lambda b: b.manifest_overrides.update(boot_id="22222222-2222-4333-8444-555555555555"), "unavailable",
         "not the current boot"),
    ],
)
def test_uncertain_kernel_coverage_is_never_zero_errors(runners, tmp_path, setup, status, reason) -> None:
    prep = prepare(runners, tmp_path)
    backend = Backend()
    setup(backend)
    result, _ = run_step4(runners, tmp_path, prep, backend=backend)
    kernel = result["step4"]["kernel"]
    assert kernel["status"] == status and any(reason in r for r in kernel["reasons"])
    assert kernel["oom_candidates"] is None and kernel["nvmap_candidates"] is None
    assert result["step4"]["criteria"]["criteria"]["K_kernel"]["status"] == "unavailable"
    assert result["step4"]["criteria"]["eligible_for_maintainer_review"] is False


def test_an_interrupted_post_run_wait_leaves_kernel_evidence_unavailable(runners, tmp_path) -> None:
    prep = prepare(runners, tmp_path)
    backend = Backend()
    stop = {"at": None}

    def interrupted():
        child_done = any("demo_profile.py" in " ".join(c.argv) and c.poll() is not None for c in backend.children)
        if child_done and stop["at"] is None:
            stop["at"] = backend.now + 10
        return stop["at"] is not None and backend.now >= stop["at"]

    result, _ = run_step4(runners, tmp_path, prep, backend=backend, interrupted=interrupted)
    assert result["step4"]["post_run_wait_s"] < 60
    assert result["step4"]["kernel"]["status"] == "unavailable"


def test_a_guard_stop_kills_the_owned_group_and_the_run_is_not_eligible(runners, tmp_path) -> None:
    prep = prepare(runners, tmp_path)
    backend = Backend()
    original = backend.sample

    def rising():
        sample = original()
        if backend.now > 1100:
            sample["MemAvailable"] = 8_000_000_000 - 4_900_000_000
        return sample

    backend.sample = rising
    result, _ = run_step4(runners, tmp_path, prep, backend=backend)
    section = result["step4"]
    assert section["status"] == "sampled_pressure_stop" and section["cleanup_clear"] is True
    assert section["criteria"]["criteria"]["R_valid_run"]["status"] == "fail"
    assert section["criteria"]["eligible_for_maintainer_review"] is False


def test_a_file_changed_during_the_run_fails_identity(runners, tmp_path) -> None:
    prep = prepare(runners, tmp_path)
    backend = Backend()
    original = backend._write_profiler_files

    def changed():
        backend.identity_hashes = {**backend.identity_hashes, "model:llm": "f" * 64}
        original()

    backend._write_profiler_files = changed
    result, _ = run_step4(runners, tmp_path, prep, backend=backend)
    assert result["step4"]["identity"]["status"] == "changed" and result["step4"]["identity"]["changed"] == ["model:llm"]
    assert result["step4"]["criteria"]["criteria"]["I_identity"]["status"] == "fail"


def test_the_cli_exit_status_reflects_eligibility_not_acceptance(runners, tmp_path, capsys, monkeypatch) -> None:
    op = runners.operator
    monkeypatch.setattr(op, "OUTPUT_ROOT", tmp_path)
    files = identity_tree(tmp_path)
    monkeypatch.setattr(op, "identity_files", lambda clip: files)
    clip_sha = hashlib.sha256(files["clip"].read_bytes()).hexdigest()
    assert op.main(["--step4-identity", "--step4-clip", str(files["clip"]), "--step4-clip-sha256", clip_sha],
                   backend=Backend()) == 0
    printed = capsys.readouterr().out
    assert clip_sha not in printed and json.loads(printed)["status"] == "complete"  # the hash stays in result.json
    assert clip_sha in Path(json.loads(printed)["result_file"]).read_text()
    assert op.main(["--step4-identity", "--step4-clip", str(files["clip"]), "--step4-clip-sha256", "0" * 64],
                   backend=Backend()) == 1
    with pytest.raises(SystemExit):
        op.main(["--step4-identity", "--step4-clip", str(files["clip"]), "--step4-clip-sha256", "short"])
    capsys.readouterr()
    assert op.main(["--execute-workload", "step4", "--latest-check9-report", "--confirm-step4-prerequisites",
                    "--operator-dropped-caches", "--step4-clip", str(files["clip"])], backend=Backend()) == 1
    assert "step4_requires_explicit_check9_report" in capsys.readouterr().out


# ---------------------------------------------------------------- startup identity checks


def test_startup_checks_the_request_fingerprint_binary_and_libraries(accepted_scene, monkeypatch) -> None:
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
    fixture = accepted_scene()
    assert profile_mismatch(fixture.profile, config, fixture.options) is None
    stale_request = dataclasses.replace(fixture.profile, scene_request_sha256="0" * 64)
    assert "another scene request" in profile_mismatch(stale_request, config, fixture.options)
    fixture.binary.write_bytes(b"\1" * 8)  # same size, new content: the binary is hashed
    assert profile_mismatch(fixture.profile, config, fixture.options) == "llama-server llama-server does not have the profiled SHA-256"
    fixture.binary.write_bytes(b"\0" * 8)
    fixture.libraries[0].write_bytes(b"\1" * 16)  # a small library is hashed too
    assert "libllama.so.0.0.8932 does not have the profiled SHA-256" in profile_mismatch(fixture.profile, config, fixture.options)
    fixture.libraries[0].write_bytes(b"\0" * 16)
    import os

    monkeypatch.setattr("sentinel.demo_runtime.STARTUP_HASH_LIMIT_BYTES", 20)  # make the 24 B library "large"
    before = fixture.libraries[1].stat()
    fixture.libraries[1].write_bytes(b"\1" * 24)  # same size, new content, and the old mtime restored:
    os.utime(fixture.libraries[1], ns=(before.st_atime_ns, before.st_mtime_ns))
    assert profile_mismatch(fixture.profile, config, fixture.options) is None  # undetected at startup (the limitation)
    fixture.libraries[1].write_bytes(b"\1" * 25)  # a size change is caught
    assert "libggml-cuda.so.0.10.0 (25 B" in profile_mismatch(fixture.profile, config, fixture.options)
    fixture.libraries[1].unlink()
    assert profile_mismatch(fixture.profile, config, fixture.options).endswith("libggml-cuda.so.0.10.0 is missing")
    assert adapters.STARTUP_IDENTITY_LIMITATION in fixture.profile.limitations
