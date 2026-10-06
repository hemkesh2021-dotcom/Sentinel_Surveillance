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
import errno
import hashlib
import importlib
import json
import re
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
GIB = 1 << 30
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


def _boot_descriptor_parses(text: str) -> bool:
    """systemd 255 ``parse_boot_descriptor``: only the first 32 characters are tried as an ID, then an offset."""
    if text == "all":
        return True
    if len(text) >= 32 and re.fullmatch(r"[0-9a-fA-F]{32}", text[:32]):
        text = text[32:]
    elif len(text) >= 32 and text[:1] not in ("-", "+"):
        return False
    return text == "" or re.fullmatch(r"[+-]?\d+", text) is not None


def journalctl_argument_error(argv: list[str]) -> tuple[int, bytes] | None:
    """How journalctl 255 rejects step 4's arguments, or None if they parse (the fake journal's model).

    Modelled on systemd 255's journalctl.c and the maintainer's parsing test on the device (session 39):
    the dashed /proc boot ID after a bare ``-b`` left ``-b`` meaning the current boot and became a match
    (rc 1, invalid_match), while the 32-hex form parsed (rc 0). ``--boot=`` with a bad value fails itself.
    """
    words, positional, index = argv[1:], [], 0
    while index < len(words):
        word = words[index]
        if word.startswith("--boot="):
            value = word.split("=", 1)[1]
            if not _boot_descriptor_parses(value):
                return 1, f"Failed to parse boot descriptor '{value}'\n".encode()
        elif word == "-b":
            if index + 1 < len(words) and _boot_descriptor_parses(words[index + 1]):
                index += 1  # journalctl takes the next word as the boot only if it parses
        elif word in ("--since", "--until", "-t", "-n", "-o"):
            index += 1
        elif not word.startswith("-"):
            positional.append(word)
        index += 1
    for word in positional:
        if "=" not in word and not word.startswith("/"):
            return 1, f"Failed to add match '{word}': Invalid argument\n".encode()
    return None


class Child:
    def __init__(self, backend, argv, *, duration=0.0, returncode=0, output=b"", on_done=None, errors=b"",
                 stubborn=False):
        self.backend, self.argv, self.end = backend, argv, backend.now + duration
        self.returncode, self.chunks, self.on_done, self.killed, self.closed = returncode, [output], on_done, False, False
        self.errors, self.stubborn, self.signals = errors, stubborn, []

    def poll(self):
        done = self.killed or self.backend.now >= self.end
        if done and self.on_done:
            self.on_done()
            self.on_done = None
        return self.returncode if done else None

    def read(self):
        return self.chunks.pop(0) if self.chunks else b""

    def error_output(self):
        return self.errors

    def exists(self):
        return self.stubborn or (not self.killed and self.poll() is None)

    def signal(self, signum):
        self.signals.append(signum)
        if self.stubborn:
            return
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
        self.mr1_report = None  # MR1: the profiler's mr1.json, written with the other files when set
        self.plr_report = None  # step-4 PLR: the profiler's plr.json, likewise
        self.memattr_report = None  # MA1: the profiler's memattr.json, likewise
        self.thp_report = None  # MA1-THP: the profiler's thp.json, likewise
        self.wtd_report = None  # MA1-WTD: the profiler's wtd.json, likewise
        self.thp_flag = 1  # MA1-WTD: this (operator) process's THP_enabled
        self.sleeps = []

    def thp_enabled(self):
        return self.thp_flag

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
        rejected = journalctl_argument_error(argv)
        if rejected:
            return {"returncode": rejected[0], "errors": rejected[1]}
        if "-k" in argv and "-n" in argv:
            return {"output": json.dumps({"MESSAGE": "boot ok"}).encode()}
        kind = "kernel" if "-k" in argv else "markers" if "-t" in argv else "journald"
        plan = dict(self.journal_status.get(kind, {}))
        records = self.journal[kind]
        if records is None:  # markers: written by this run under the queried tag, at the logger calls' times
            tag = argv[argv.index("-t") + 1]
            records = [{"MESSAGE": message, "__REALTIME_TIMESTAMP": str(int(t * 1e6))}
                       for message, t, marker_tag in self.markers if marker_tag == tag]
        plan.setdefault("output", "\n".join(json.dumps(r) for r in records).encode())
        return plan

    def spawn(self, argv, env, capture_stderr=False):
        if "systemctl" in argv[0]:
            plan = {"output": SERVICES.encode()}
        elif "journalctl" in argv[0]:
            plan = self._journal_output(argv)
        elif argv[0] == "git":
            plan = {"output": COMMIT.encode()}
        elif "logger" in argv[0]:
            self.markers = [*getattr(self, "markers", []), (argv[-1], self.wall(), argv[argv.index("-t") + 1])]
            plan = {}
        elif "--api" in argv:
            api = argv[argv.index("--api") + 1]
            plan = {"duration": 0.4, "output": json.dumps({
                "api": api, "status": "bounded_smoke_complete", "allocated_bytes": 256 << 20,
                "cap_bytes": 256 << 20, "chunk_bytes": 32 << 20, "cleanup_clear": True}).encode()}
        else:
            if "--out" in argv:  # the CLI makes its own private output directory
                self.output = Path(argv[argv.index("--out") + 1])
            self.child_started = self.wall()
            plan = {"duration": 900.0, "on_done": self._write_profiler_files, **self.child_plan}
        if not capture_stderr:
            plan.pop("errors", None)  # stderr goes to DEVNULL unless the caller captures it
        child = Child(self, argv, **plan)
        self.children.append(child)
        return child

    def wall(self):
        return 1_900_000_000 + self.now

    def _write_profiler_files(self):
        run = self.output / "demo-profile-20991231T000000Z"
        run.mkdir(exist_ok=True)
        start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(getattr(self, "child_started", self.wall() - 900)))
        finish = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.wall()))
        manifest = {"boot_id": BOOT, "started_utc": start, "finished_utc": finish, "display_manager": "inactive",
                    "desktop_processes": [], "dev_tools_running": [],
                    "repository": {"commit": COMMIT, "tracked_changes": False},
                    "sha256": {k.split(":", 1)[1]: v for k, v in self.identity_hashes.items() if k.startswith("model:")},
                    "sha256_build": {k.split(":", 1)[1]: v for k, v in self.identity_hashes.items() if k.startswith("build:")},
                    "sha256_clip": self.identity_hashes.get("clip"), **self.manifest_overrides}
        (run / "manifest.json").write_text(json.dumps(manifest))
        (run / "profile.json").write_text(json.dumps(self.profile))
        if self.mr1_report is not None:
            (run / "mr1.json").write_text(json.dumps(self.mr1_report))
        if self.plr_report is not None:
            (run / "plr.json").write_text(json.dumps(self.plr_report))
        if self.memattr_report is not None:
            (run / "memattr.json").write_text(json.dumps(self.memattr_report))
        if self.thp_report is not None:
            (run / "thp.json").write_text(json.dumps(self.thp_report))
        if self.wtd_report is not None:
            (run / "wtd.json").write_text(json.dumps(self.wtd_report))


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
    assert len(queries) == 3 and all(f"--boot={BOOT.replace('-', '')}" in q and "-b" not in q for q in queries)
    kernel_query = next(q for q in queries if "-k" in q)
    manifest = json.loads(next(backend.output.glob("demo-profile-*/manifest.json")).read_text())
    started, finished = (runners.operator._utc(manifest[k]).timestamp() for k in ("started_utc", "finished_utc"))
    assert kernel_query[kernel_query.index("--since") + 1] == f"@{int(started) - 1}"
    assert kernel_query[kernel_query.index("--until") + 1] == f"@{int(finished + 60) + 1}"  # recorded run end + 60 s
    end_marker_time = backend.markers[-1][1]
    child_end = 1_900_000_000 + 1000.0 + 900.0
    assert end_marker_time >= child_end + 60  # the wait really happened before the end marker and the queries
    assert section["kernel"]["status"] == "observed" and section["kernel"]["oom_candidates"] == 0
    assert {q["error_class"] for q in section["kernel"]["queries"].values()} == {"success"}
    assert section["guard_trigger"] is None and section["guard_stopped_at_s"] is None
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (BOOT, BOOT.replace("-", "")),
        (BOOT.replace("-", ""), BOOT.replace("-", "")),
        (None, None), (12345, None), ("", None), ("test-boot", None),
        ("ABCDEF00-1111-4222-8333-444444444444", None),  # /proc prints lowercase; anything else is refused
        (BOOT + "\n", None), (BOOT[:-1], None), (BOOT.replace("-", "", 1), None),  # dashes all or none
        ("11111111222243338444555555555555-1", None),  # an offset is not a boot ID
        ("00000000-0000-0000-0000-000000000000", None), ("0" * 32, None),  # journalctl: no ID = current boot
    ],
)
def test_boot_ids_are_validated_and_normalized_for_journalctl(runners, value, expected) -> None:
    assert runners.operator.journal_boot_id(value) == expected


def test_the_dashed_boot_after_a_bare_b_is_a_match_error_and_the_new_form_parses() -> None:
    """The fake journal's model reproduces the device result before step4 code relies on it."""
    old = ["/usr/bin/journalctl", "-k", "-b", BOOT, "--since", "@1", "--until", "@2", "-o", "json", "--quiet"]
    assert journalctl_argument_error(old) == (1, f"Failed to add match '{BOOT}': Invalid argument\n".encode())
    assert journalctl_argument_error(["/usr/bin/journalctl", "-k", f"--boot={BOOT}"])[0] == 1
    assert journalctl_argument_error(["/usr/bin/journalctl", "-k", f"--boot={BOOT.replace('-', '')}", "--since", "@1"]) is None
    assert journalctl_argument_error(["/usr/bin/journalctl", "-k", "-b", "-n", "1000", "-o", "json"]) is None


@pytest.mark.parametrize(
    ("kind", "plan", "status", "returncode", "error_class"),
    [
        ("kernel", {"returncode": 1, "errors": f"Failed to add match '{BOOT}': Invalid argument\n".encode()},
         "child_failed", 1, "invalid_match"),
        ("markers", {"returncode": 1, "errors": b"Failed to parse boot descriptor 'x'\n"},
         "child_failed", 1, "invalid_boot_descriptor"),
        ("journald", {"returncode": 1, "errors": b"No journal files were opened due to insufficient permissions.\n"},
         "child_failed", 1, "permission_denied"),
        ("kernel", {"returncode": 1, "errors": b"camera rtsp://user:secret@cam failed\n"},
         "child_failed", 1, "nonzero_exit_unclassified"),
        ("kernel", {"duration": 30.0}, "timeout", -15, "timeout"),
        ("journald", {"output": b"not json"}, "completed", 0, "unparsable_output"),
    ],
)
def test_a_failed_query_keeps_status_return_code_and_a_fixed_error_class(
    runners, tmp_path, kind, plan, status, returncode, error_class,
) -> None:
    prep = prepare(runners, tmp_path)
    backend = Backend()
    backend.journal_status[kind] = plan
    result, _ = run_step4(runners, tmp_path, prep, backend=backend)
    kernel = result["step4"]["kernel"]
    assert kernel["queries"][kind] == {"status": status, "returncode": returncode, "records": None,
                                       "error_class": error_class}
    assert all(q["error_class"] == "success" for name, q in kernel["queries"].items() if name != kind)
    assert kernel["status"] == "unavailable" and kernel["reasons"] == [f"{kind} query unavailable"]
    assert kernel["oom_candidates"] is None and kernel["nvmap_candidates"] is None and "records" not in kernel
    assert result["step4"]["criteria"]["criteria"]["K_kernel"]["status"] == "unavailable"
    assert result["step4"]["criteria"]["criteria"]["K_kernel"]["value"]["records"] is None
    assert "secret" not in json.dumps(result) and "Failed to" not in json.dumps(result)  # classes, never text


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        (lambda b: b.manifest_overrides.update(boot_id="test-boot"), "recorded boot ID missing or invalid"),
        (lambda b: b.manifest_overrides.update(boot_id="00000000-0000-0000-0000-000000000000"),
         "recorded boot ID missing or invalid"),
        (lambda b: b.manifest_overrides.update(boot_id=None), "recorded boot ID missing or invalid"),
        (lambda b: setattr(b, "boot", "ABCDEF00-1111-4222-8333-444444444444"), "current boot ID unreadable or invalid"),
    ],
)
def test_an_invalid_boot_id_runs_no_journal_query_and_never_falls_back_to_the_current_boot(
    runners, tmp_path, setup, reason,
) -> None:
    prep = prepare(runners, tmp_path)
    backend = Backend()
    original = backend.context
    setup(backend)
    backend.context = lambda: {**original(), "boot_id": BOOT}  # admission sees the real boot; only the journal differs
    result, backend = run_step4(runners, tmp_path, prep, backend=backend)
    kernel = result["step4"]["kernel"]
    assert [c for c in backend.children if "journalctl" in c.argv[0] and "-n" not in c.argv] == []
    assert kernel["status"] == "unavailable" and any(reason in r for r in kernel["reasons"])
    assert kernel["queries"] == {} and kernel["oom_candidates"] is None and kernel["nvmap_candidates"] is None
    assert result["step4"]["criteria"]["criteria"]["K_kernel"]["status"] == "unavailable"


def test_an_unnormalized_boot_in_the_attached_form_fails_loudly_not_as_the_current_boot(
    runners, tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(runners.operator, "journal_boot_id", lambda value: value)  # skip the normalization
    prep = prepare(runners, tmp_path)
    result, _ = run_step4(runners, tmp_path, prep)
    kernel = result["step4"]["kernel"]
    assert kernel["status"] == "unavailable" and kernel["oom_candidates"] is None
    assert {name: (q["returncode"], q["error_class"]) for name, q in kernel["queries"].items()} == {
        name: (1, "invalid_boot_descriptor") for name in ("kernel", "markers", "journald")}


def test_a_free_floor_stop_reports_its_condition_sample_and_time_apart_from_cleanup(runners, tmp_path) -> None:
    prep = prepare(runners, tmp_path)
    backend = Backend()
    original = backend.sample

    def falling():
        sample = original()
        if backend.now > 1100:  # pressure and MemAvailable stay inside their limits; only MemFree crosses
            sample["MemFree"] = (1 << 30) - 6_340_608
        return sample

    backend.sample = falling
    result, _ = run_step4(runners, tmp_path, prep, backend=backend)
    section = result["step4"]
    trigger = section["guard_trigger"]
    assert section["status"] == trigger["stop"] == "free_or_available_stop"
    assert trigger["conditions"] == [{"stop": "free_or_available_stop", "condition": "MemFree < 1073741824",
                                      "field": "MemFree", "comparison": "<", "threshold": 1 << 30,
                                      "value": (1 << 30) - 6_340_608}]
    lines = (backend.output / "guard.jsonl").read_text().splitlines()
    assert trigger["guard_jsonl_line"] == len(lines) == section["samples"]
    assert json.loads(lines[trigger["guard_jsonl_line"] - 1]) == trigger["sample"]
    assert trigger["t_mono"] == trigger["sample"]["t_mono"] and 1100 < trigger["t_mono"] < 1100.5
    assert section["guard_stopped_at_basis"] == "post_cleanup_clock"  # fake cleanup takes no time
    assert section["guard_stopped_at_s"] >= round(trigger["t_mono"], 3)
    assert section["criteria"]["criteria"]["R_valid_run"]["status"] == "fail"


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


def test_the_step4_child_command_is_exactly_the_reviewed_one(runners, tmp_path) -> None:
    prep = prepare(runners, tmp_path)
    _, backend = run_step4(runners, tmp_path, prep)
    (child,) = [c for c in backend.children if "demo_profile.py" in " ".join(c.argv)]
    assert child.argv == [
        "/usr/bin/python3", str(RUNNERS / "demo_profile.py"), "--out", str(backend.output), "--clip", str(prep.files["clip"]),
        "--no-evict", "--sanitized-logs", "--llama-cache-ram", "0", "--face-hz", "1", "--scene-interval-s", "4",
        "--baseline-s", "30", "--settle-s", "15", "--warmup-s", "120", "--steady-s", "600",
        "--llama-timeout-s", "60", "--load-timeout-s", "90", "--min-free-gb", "3.5",
    ]
    assert [m[2] for m in backend.markers] == ["sentinel-step4", "sentinel-step4"]


# ---------------------------------------------------------------- MR1 operator mode (step 4's guard and evidence)


def fake_mr1_report(runners, tmp_path, *, result="returned_0", moved=True, smoke_ok=True,
                    reached=("scene", "detector", "face")) -> dict:
    """The profiler's own mr1.json, built by its functions from fake samples and calls."""
    profile = runners.profile

    def snap(free, cached):
        meminfo = dict.fromkeys(profile.MEMINFO_KEYS)
        meminfo.update(MemTotal=8_000_000_000, MemFree=free, MemAvailable=free + cached, Cached=cached, Mapped=600_000_000)
        return {"t_mono": 1.0, "meminfo": meminfo, "pressure_bytes": 8_000_000_000 - free - cached,
                "processes": {"llama": {"rss_bytes": 2_000_000_000, "hwm_bytes": None, "pss_bytes": None}}}

    call = {"name": "LFM2-private-name.gguf", "bytes": 10, "call": profile.MR1_FADVISE_CALL, "result": result,
            "returncode": 0 if result == "returned_0" else errno.ENOENT,
            "error": None if result == "returned_0" else "ENOENT", "elapsed_s": 0.001}
    events = []
    for component in reached:
        states = iter([snap(1_000_000_000, 2_700_000_000),
                       snap(1_400_000_000, 2_300_000_000) if moved else snap(1_000_000_000, 2_700_000_000)])
        record = profile.release_component(component, {"model": Path("/home/someone/models/x.gguf")}, {}, 5.0,
                                           snapshot=lambda: next(states), release=lambda path: dict(call),
                                           sleep=lambda seconds: None)
        events.append({"event": "mr1_release", **record})
    events.append({"event": "mr1_smoke", **profile.sanitize_mr1_smoke({
        "detector": {"frames_requested": 15, "processed": 15 if smoke_ok else 12, "errors": {} if smoke_ok else {"ValueError": 3}},
        "face": {"runs": 1, "completed": 1, "errors": {}},
        "scene": {"attempts": 1, "completed": 1, "finish_reason": "stop", "valid": True}})})
    events.append({"event": "mr1_snapshot", "stage": "after_smoke", **snap(1_350_000_000, 2_350_000_000)})
    run = tmp_path / "mr1-source"
    run.mkdir(exist_ok=True)
    (run / "events.jsonl").write_text("\n".join(json.dumps(item) for item in events) + "\n")
    return profile.mr1_report(run, "complete")


def run_mr1(runners, tmp_path, prep, backend=None, **overrides):
    op = runners.operator
    output = private_dir(tmp_path, "mr1")
    backend = backend or Backend()
    backend.output = output
    backend.identity_hashes = {}  # MR1 hashes nothing; the fake manifest still lists none
    backend.child_plan.setdefault("duration", 200.0)
    if backend.mr1_report is None:
        backend.mr1_report = fake_mr1_report(runners, tmp_path)
    kwargs = dict(check9_report=prep.check9_path, clip=prep.files["clip"], confirm_mr1=True, dropped_caches=True)
    kwargs.update(overrides)
    return op.execute("mr1", backend, output, **kwargs), backend


@pytest.mark.parametrize(("case", "refusal"), [
    ("unconfirmed", "mr1_operator_prerequisites_unconfirmed"), ("drop_undeclared", "mr1_cache_drop_not_declared"),
    ("no_clip", "mr1_clip_required"), ("missing_clip", "mr1_clip_required"),
    ("latest_check9", "mr1_requires_explicit_check9_report"), ("no_check9", "check9_report_required"),
    ("failed_check9", "check9_report_mismatch"),
])
def test_mr1_refuses_before_any_process_when_a_prerequisite_fails(runners, tmp_path, case, refusal) -> None:
    prep = prepare(runners, tmp_path)
    overrides = {"unconfirmed": {"confirm_mr1": False}, "drop_undeclared": {"dropped_caches": False},
                 "no_clip": {"clip": None}, "missing_clip": {"clip": tmp_path / "absent.mp4"},
                 "latest_check9": {"explicit_check9": False}, "no_check9": {"check9_report": None}}.get(case, {})
    if case == "failed_check9":
        report = json.loads(prep.check9_path.read_text())
        report["check9"]["status"] = "inconclusive"
        prep.check9_path.write_text(json.dumps(report))
    result, backend = run_mr1(runners, tmp_path, prep, **overrides)
    assert result["mr1"]["status"] == "refused" and refusal in result["mr1"]["refusals"]
    assert not [c for c in backend.children if "demo_profile.py" in " ".join(c.argv) or "logger" in c.argv[0]]
    assert not (backend.output / "guard.jsonl").exists()


def test_a_completed_mr1_run_keeps_step4_settings_and_evidence_and_is_never_acceptance(runners, tmp_path) -> None:
    op = runners.operator
    prep = prepare(runners, tmp_path)
    result, backend = run_mr1(runners, tmp_path, prep)
    section = result["mr1"]
    (child,) = [c for c in backend.children if "demo_profile.py" in " ".join(c.argv)]
    argv = child.argv
    assert argv == op.mr1_argv(backend.output, prep.files["clip"])
    assert argv[argv.index("--llama-cache-ram") + 1] == "0" and argv[argv.index("--clip") + 1] == str(prep.files["clip"])
    assert argv[argv.index("--min-free-gb") + 1] == "3.5" and "--no-evict" in argv and "--sanitized-logs" in argv
    assert argv[argv.index("--warmup-s") + 1] == argv[argv.index("--steady-s") + 1] == "0"
    assert argv[-3:] == ["--mr1-release-check", "--mr1-release-settle-s", "5.0"] and "--allow-dev-tools" not in argv
    assert section["status"] == "completed" and section["deadline_s"] == op.MR1_DEADLINE_S == 540.0
    assert section["guard_trigger"] is None and section["post_run_wait_s"] == 60.0
    assert [(m[0].split()[:2], m[2]) for m in backend.markers] == [(["mr1", "start"], "sentinel-mr1"),
                                                                   (["mr1", "end"], "sentinel-mr1")]
    queries = [c.argv for c in backend.children if "journalctl" in c.argv[0] and "-n" not in c.argv]
    assert all(f"--boot={BOOT.replace('-', '')}" in q for q in queries) and len(queries) == 3
    assert section["kernel"]["status"] == "observed" and section["kernel"]["oom_candidates"] == 0
    reading = section["interpretation"]
    assert reading["execution_valid"] is True and reading["functional_smoke"] == "pass"
    assert reading["release_outcomes"] == dict.fromkeys(("scene", "detector", "face"), "memfree_rose_cached_fell")
    assert reading["acceptance"] == "none" and "never accepted" in section["scope"]
    excerpt = section["release_check"]
    assert excerpt["status"] == "recorded" and excerpt["releases"]["scene"]["deltas"]["MemFree"] == 400_000_000
    assert excerpt["releases"]["scene"]["files"]["model"]["result"] == "returned_0"
    text = json.dumps(result)
    assert "private-name" not in text and "/home/someone" not in text  # numbers and fixed labels only
    assert result["step4"]["status"] == "PENDING" and result["hardware_acceptance"] == "PENDING"


@pytest.mark.parametrize("failure", ["free_floor", "timeout", "cleanup_failed", "no_record"])
def test_an_mr1_guard_stop_timeout_cleanup_failure_or_missing_record_is_never_a_valid_execution(
    runners, tmp_path, failure,
) -> None:
    op = runners.operator
    prep = prepare(runners, tmp_path)
    backend = Backend()
    if failure == "free_floor":
        original = backend.sample

        def falling():
            sample = original()
            if backend.now > 1100:
                sample["MemFree"] = GIB - 1
            return sample

        backend.sample = falling
    elif failure == "timeout":
        backend.child_plan["duration"] = op.MR1_DEADLINE_S + 100
    elif failure == "cleanup_failed":
        backend.child_plan.update(duration=op.MR1_DEADLINE_S + 100, stubborn=True)
    else:
        backend.mr1_report = {}
    result, backend = run_mr1(runners, tmp_path, prep, backend=backend)
    section = result["mr1"]
    (child,) = [c for c in backend.children if "demo_profile.py" in " ".join(c.argv)]
    reading = section["interpretation"]
    assert reading["execution_valid"] is False and reading["acceptance"] == "none"
    if failure == "free_floor":
        assert section["status"] == section["guard_trigger"]["stop"] == "free_or_available_stop"
        assert section["guard_trigger"]["conditions"][0]["field"] == "MemFree"
        assert child.signals[:1] == [signal.SIGTERM] and section["cleanup_clear"] is True
    elif failure == "timeout":
        assert section["status"] == "timeout" and section["guard_trigger"] is None and section["cleanup_clear"] is True
    elif failure == "cleanup_failed":
        assert section["status"] == "cleanup_failed" and section["cleanup_clear"] is False
        assert child.signals == [signal.SIGTERM, signal.SIGKILL]
        assert section["release_check"] == {"status": "unavailable"}
        assert reading["release_outcomes"] == dict.fromkeys(("scene", "detector", "face"), "not_reached")
    else:
        assert section["status"] == "completed" and reading["functional_smoke"] == "unavailable"
    if failure == "no_record":
        assert section["guard_stopped_at_s"] is None
    else:
        assert section["guard_stopped_at_basis"] == "post_cleanup_clock"


@pytest.mark.parametrize(("report_kwargs", "exit_status", "outcome"), [
    ({}, 0, "memfree_rose_cached_fell"),
    ({"moved": False}, 0, "ineffective"),  # a memory outcome never sets the exit status
    ({"result": "open_failed"}, 0, "call_failed"),
    ({"smoke_ok": False}, 1, "memfree_rose_cached_fell"),
])
def test_the_mr1_cli_exit_status_reflects_execution_and_smoke_never_memory(
    runners, tmp_path, monkeypatch, capsys, report_kwargs, exit_status, outcome,
) -> None:
    op = runners.operator
    monkeypatch.setattr(op, "OUTPUT_ROOT", tmp_path)
    prep = prepare(runners, tmp_path)
    backend = Backend()
    backend.child_plan["duration"] = 200.0
    backend.mr1_report = fake_mr1_report(runners, tmp_path, **report_kwargs)
    backend.identity_hashes = {}
    status = op.main(["--execute-workload", "mr1", "--check9-report", str(prep.check9_path), "--mr1-clip",
                      str(prep.files["clip"]), "--confirm-mr1-prerequisites", "--operator-dropped-caches"],
                     backend=backend)
    printed = json.loads(capsys.readouterr().out)
    assert status == exit_status
    assert set(printed["mr1"]["interpretation"]["release_outcomes"].values()) == {outcome}
    if "result" in report_kwargs:
        assert printed["mr1"]["release_check"]["releases"]["face"]["files"]["model"]["error"] == "ENOENT"
    with pytest.raises(SystemExit):
        op.main(["--execute-workload", "step4", "--mr1-clip", str(prep.files["clip"])], backend=Backend())


# ---------------------------------------------------------------- step-4 PLR (D54): step 4 plus post-load releases


def fake_plr_report(runners, tmp_path, *, result="returned_0", moved=True, reached=("scene", "detector", "face")) -> dict:
    """The profiler's own plr.json, built by its functions from fake samples and calls."""
    source = fake_mr1_report(runners, tmp_path, result=result, moved=moved, reached=reached)
    run = tmp_path / "plr-source"
    run.mkdir(exist_ok=True)
    events = [{"event": "mr1_release", "component": name, **record} for name, record in source["releases"].items()]
    (run / "events.jsonl").write_text("".join(json.dumps(item) + "\n" for item in events))
    rows = [{"t": 1.0, "phase": "warmup", "mem_free": 1_300_000_000, "cached": 2_300_000_000}]
    return runners.profile.plr_report(run, "complete", rows, rows)


def run_step4plr(runners, tmp_path, prep, backend=None, *, record=True, **overrides):
    op = runners.operator
    output = private_dir(tmp_path, "step4plr")
    backend = backend or Backend()
    backend.output = output
    backend.identity_hashes = {label: entry["sha256"] for label, entry in prep.identity["files"].items()}
    if record and backend.plr_report is None:  # record=False: the profiler wrote no plr.json
        backend.plr_report = fake_plr_report(runners, tmp_path)
    kwargs = dict(check9_report=prep.check9_path, identity_report=prep.identity_path, clip=prep.files["clip"],
                  confirm_step4plr=True, dropped_caches=True, identity_files_fn=lambda clip: prep.files)
    kwargs.update(overrides)
    return op.execute("step4plr", backend, output, **kwargs), backend


def test_the_step4plr_child_is_step4s_reviewed_command_plus_the_release_only(runners, tmp_path) -> None:
    op = runners.operator
    prep = prepare(runners, tmp_path)
    result, backend = run_step4plr(runners, tmp_path, prep)
    section = result["step4plr"]
    (child,) = [c for c in backend.children if "demo_profile.py" in " ".join(c.argv)]
    step4 = op.step4_argv(backend.output, prep.files["clip"])
    assert child.argv == [*step4, "--post-load-release", "--release-settle-s", "5.0"]
    assert step4[step4.index("--warmup-s") + 1] == "120" and step4[step4.index("--steady-s") + 1] == "600"
    assert "--mr1-release-check" not in child.argv and "--allow-dev-tools" not in child.argv
    assert section["deadline_s"] == op.STEP4_DEADLINE_S == 1200.0  # step 4's deadline, not MR1's 540 s
    assert [(m[0].split()[:2], m[2]) for m in backend.markers] == [(["step4plr", "start"], "sentinel-step4plr"),
                                                                   (["step4plr", "end"], "sentinel-step4plr")]
    assert "--post-load-release" not in step4  # step 4's own command is pinned by the reviewed-command test


def test_a_completed_step4plr_is_eligible_only_under_its_own_identity_and_never_accepted(runners, tmp_path) -> None:
    prep = prepare(runners, tmp_path)
    result, backend = run_step4plr(runners, tmp_path, prep)
    section = result["step4plr"]
    criteria = section["criteria"]
    assert section["status"] == "completed" and section["post_run_wait_s"] == 60.0 and section["guard_trigger"] is None
    assert criteria["criteria_id"] == "step4plr-combined-cache-off-v2" and criteria["eligible_for_maintainer_review"]
    assert criteria["accepted"] is False and "not admissible" in criteria["admission"]
    assert criteria["criteria"]["R_valid_run"]["value"]["releases_recorded"] is True
    assert criteria["criteria"]["R_valid_run"]["value"]["release_calls_returned_0"] is True
    assert section["status_detail"].startswith("eligible for maintainer review as a step-4 PLR result; not D47's")
    assert section["identity"]["status"] == "verified" and section["kernel"]["status"] == "observed"
    assert "never D47's step-4 result" in section["scope"] and section["acceptance"].startswith("never")
    excerpt = section["release_check"]
    assert excerpt["status"] == "recorded" and excerpt["criteria_id"] == "step4plr-combined-cache-off-v2"
    assert excerpt["release_outcomes"] == dict.fromkeys(("scene", "detector", "face"), "memfree_rose_cached_fell")
    assert excerpt["releases"]["face"]["deltas"]["MemFree"] == 400_000_000 and excerpt["steady"]["samples"] == 1
    text = json.dumps(result)
    assert "private-name" not in text and "/home/someone" not in text  # numbers and fixed labels only
    assert result["step4"]["status"] == "PENDING" and result["hardware_acceptance"] == "PENDING"


@pytest.mark.parametrize(("overrides", "refusal"), [
    ({"confirm_step4plr": False, "confirm_step4": True}, "step4plr_operator_prerequisites_unconfirmed"),
    ({"dropped_caches": False}, "step4plr_cache_drop_not_declared"),
    ({"explicit_check9": False}, "step4plr_requires_explicit_check9_report"),
    ({"clip": None}, "step4plr_clip_required"),
    ({"identity_report": None}, "step4_identity_report_required"),  # the step-4 identity snapshot
])
def test_step4plr_refuses_before_any_process_when_a_step4_prerequisite_fails(
    runners, tmp_path, overrides, refusal,
) -> None:
    prep = prepare(runners, tmp_path)
    result, backend = run_step4plr(runners, tmp_path, prep, **overrides)
    assert result["step4plr"]["status"] == "refused" and refusal in result["step4plr"]["refusals"]
    assert not [c for c in backend.children if "demo_profile.py" in " ".join(c.argv) or "logger" in c.argv[0]]


@pytest.mark.parametrize("case", ["no_record", "call_failed", "scene_only_reached", "ineffective"])
def test_a_step4plr_run_without_its_declared_releases_is_never_eligible(runners, tmp_path, case) -> None:
    prep = prepare(runners, tmp_path)
    backend = Backend()
    backend.plr_report = {"no_record": None, "call_failed": fake_plr_report(runners, tmp_path, result="open_failed"),
                          "scene_only_reached": fake_plr_report(runners, tmp_path, reached=("scene",)),
                          "ineffective": fake_plr_report(runners, tmp_path, moved=False)}[case]
    result, _ = run_step4plr(runners, tmp_path, prep, backend=backend, record=case != "no_record")
    criteria = result["step4plr"]["criteria"]
    r = criteria["criteria"]["R_valid_run"]
    if case == "ineffective":  # a memory outcome is descriptive: it never blocks eligibility
        assert r["status"] == "pass" and criteria["eligible_for_maintainer_review"] is True
        assert set(result["step4plr"]["release_check"]["release_outcomes"].values()) == {"ineffective"}
        return
    assert r["status"] == "fail" and criteria["eligible_for_maintainer_review"] is False
    assert criteria["accepted"] is False and "R_valid_run" in criteria["blocking"]
    if case == "no_record":
        assert result["step4plr"]["release_check"] == {"status": "unavailable"}
        assert r["value"]["releases_recorded"] is False and r["value"]["release_calls_returned_0"] is False
    elif case == "call_failed":
        assert r["value"] == {**r["value"], "releases_recorded": True, "release_calls_returned_0": False}
    else:
        assert r["value"]["releases_recorded"] is False


def test_a_step4plr_guard_stop_is_not_eligible_and_names_its_stopping_sample(runners, tmp_path) -> None:
    prep = prepare(runners, tmp_path)
    backend = Backend()
    original = backend.sample

    def falling():
        sample = original()
        if backend.now > 1100:
            sample["MemFree"] = GIB - 1
        return sample

    backend.sample = falling
    result, backend = run_step4plr(runners, tmp_path, prep, backend=backend)
    section = result["step4plr"]
    (child,) = [c for c in backend.children if "demo_profile.py" in " ".join(c.argv)]
    assert section["status"] == section["guard_trigger"]["stop"] == "free_or_available_stop"
    assert section["guard_trigger"]["conditions"][0]["field"] == "MemFree" and child.signals[:1] == [signal.SIGTERM]
    assert section["criteria"]["criteria"]["R_valid_run"]["status"] == "fail"
    assert section["criteria"]["eligible_for_maintainer_review"] is False and section["cleanup_clear"] is True


def test_the_step4plr_cli_exit_status_reflects_eligibility_and_needs_its_own_confirmation(
    runners, tmp_path, monkeypatch, capsys,
) -> None:
    op = runners.operator
    monkeypatch.setattr(op, "OUTPUT_ROOT", tmp_path)
    prep = prepare(runners, tmp_path)
    monkeypatch.setattr(op, "identity_files", lambda clip: prep.files)

    def backend():
        b = Backend()
        b.identity_hashes = {label: entry["sha256"] for label, entry in prep.identity["files"].items()}
        b.plr_report = fake_plr_report(runners, tmp_path)
        return b

    common = ["--execute-workload", "step4plr", "--check9-report", str(prep.check9_path), "--identity-report",
              str(prep.identity_path), "--step4-clip", str(prep.files["clip"]), "--operator-dropped-caches"]
    assert op.main([*common, "--confirm-step4plr-prerequisites"], backend=backend()) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["step4plr"]["criteria"]["criteria_id"] == "step4plr-combined-cache-off-v2"
    assert printed["step4plr"]["criteria"]["accepted"] is False
    assert op.main([*common, "--confirm-step4-prerequisites"], backend=backend()) == 1  # step 4's confirmation is not PLR's
    assert "step4plr_operator_prerequisites_unconfirmed" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        op.main(["--execute-workload", "step4", "--confirm-step4plr-prerequisites"], backend=Backend())


# ---------------------------------------------------------------- MA1 (D55): the instrumented PLR diagnostic


def fake_memattr_report(runners, tmp_path, *, malloc=True, maps=True) -> dict:
    """The profiler's own memattr.json, built by its function from a few synthetic events."""
    profile = runners.profile
    run = tmp_path / "memattr-source"
    run.mkdir(exist_ok=True)
    events = [{"t_mono": 9.0, "event": "memattr_config", "python": "3.10.14"}]
    for t in (110.0, 190.0):
        events.append({"t_mono": t, "event": "memattr_process", "malloc_status": "observed" if malloc else "unsupported",
                       "malloc": dict.fromkeys(profile.MALLINFO2_REPORTED, int(t)) if malloc else None,
                       "pymalloc_blocks": 5, "gc_collections": [1, 1, 1], "gc_full": [[150.0, 2.0, 1, 0]]})
        events.append({"t_mono": t, "event": "memattr_maps", "status": "observed" if maps else "unavailable",
                       "reason": None if maps else "read_failed", "read_ms": 3.0, "vmas": 9,
                       **({"categories": {name: {"rss_bytes": int(t), "private_dirty_bytes": 0, "anonymous_bytes": 0}
                                          for name in profile.MAPS_CATEGORIES}} if maps else {})})
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    samples = [{"t": t, "work_rss": 2_000_000_000, "anon_pages": 1, "work_pss": 1, "llama_rss": 1}
               for t in (100.0, 150.0, 200.0)]
    report = profile.memattr_report(run, "complete", samples, None, (100.0, 200.0))
    report["config"] = {**(report["config"] or {}), "note": "/home/someone/private-name"}  # never relayed
    return report


def run_ma1(runners, tmp_path, prep, backend=None, *, record=True, **overrides):
    op = runners.operator
    output = private_dir(tmp_path, "ma1")
    backend = backend or Backend()
    backend.output = output
    backend.identity_hashes = {label: entry["sha256"] for label, entry in prep.identity["files"].items()}
    backend.plr_report = backend.plr_report or fake_plr_report(runners, tmp_path)
    if record and backend.memattr_report is None:
        backend.memattr_report = fake_memattr_report(runners, tmp_path)
    kwargs = dict(check9_report=prep.check9_path, identity_report=prep.identity_path, clip=prep.files["clip"],
                  confirm_ma1=True, dropped_caches=True, identity_files_fn=lambda clip: prep.files)
    kwargs.update(overrides)
    return op.execute("ma1", backend, output, **kwargs), backend


def test_the_ma1_child_is_the_plr_command_plus_the_attribution_flags_only(runners, tmp_path) -> None:
    op = runners.operator
    prep = prepare(runners, tmp_path)
    result, backend = run_ma1(runners, tmp_path, prep)
    (child,) = [c for c in backend.children if "demo_profile.py" in " ".join(c.argv)]
    plr = op.step4plr_argv(backend.output, prep.files["clip"])
    assert child.argv == [*plr, "--memory-attribution", "--maps-interval-s", "5.0"]
    assert result["ma1"]["deadline_s"] == op.STEP4_DEADLINE_S == 1200.0
    assert [(m[0].split()[:2], m[2]) for m in backend.markers] == [(["ma1", "start"], "sentinel-ma1"),
                                                                   (["ma1", "end"], "sentinel-ma1")]
    args = runners.profile.parse_args(child.argv[2:])  # models, rates, warm-up, duration and guard inputs unchanged
    step4 = runners.profile.parse_args(op.step4_argv(backend.output, prep.files["clip"])[2:])
    assert (args.warmup_s, args.steady_s, args.face_hz, args.scene_interval_s, args.fps, args.llama_cache_ram,
            args.min_free_gb, args.settle_s, args.baseline_s, args.evict) == (
        step4.warmup_s, step4.steady_s, step4.face_hz, step4.scene_interval_s, step4.fps, step4.llama_cache_ram,
        step4.min_free_gb, step4.settle_s, step4.baseline_s, step4.evict)
    assert runners.profile.workload_budget(args) == runners.profile.workload_budget(
        runners.profile.parse_args(plr[2:]))


def test_a_completed_ma1_run_is_never_eligible_and_its_criteria_are_reference_only(runners, tmp_path) -> None:
    prep = prepare(runners, tmp_path)
    result, _ = run_ma1(runners, tmp_path, prep)
    section = result["ma1"]
    assert section["status"] == "completed" and section["acceptance"].startswith("never: instrumented diagnostic")
    assert "criteria" not in section and "status_detail" not in section
    assert section["reference_criteria"]["M3_trend_swap"]["status"] == "pass"
    assert all(set(item) == {"status", "value"} for item in section["reference_criteria"].values())
    assert "never eligible" in section["reference_note"]
    reading = section["interpretation"]
    assert reading["execution_valid"] is True and reading["attribution_recorded"] is True
    assert reading["eligible_for_maintainer_review"] is False and reading["acceptance"] == "none"
    excerpt = section["attribution"]
    assert excerpt["status"] == "recorded" and excerpt["label"] == "ma1-memory-attribution"
    assert excerpt["windows"]["steady"]["malloc"]["arena"]["delta"] == 80 and excerpt["full_gc_count"] == 2
    assert "steps" not in excerpt and '"listed"' not in json.dumps(excerpt) and '"note"' not in json.dumps(excerpt)
    text = json.dumps(result)
    assert "/home/someone" not in text and "private-name" not in text
    assert result["step4"]["status"] == "PENDING" and "step4plr" not in result


@pytest.mark.parametrize("case", ["m3_fails", "no_record", "maps_unavailable", "malloc_unsupported", "cleanup"])
def test_ma1_execution_and_recording_never_depend_on_the_memory_readings(runners, tmp_path, case) -> None:
    prep = prepare(runners, tmp_path)
    backend = Backend()
    if case == "m3_fails":
        backend.profile["steady_trend"]["used"]["slope_bytes_per_min"] = 11_886_599
    elif case == "maps_unavailable":
        backend.memattr_report = fake_memattr_report(runners, tmp_path, maps=False)
    elif case == "malloc_unsupported":
        backend.memattr_report = fake_memattr_report(runners, tmp_path, malloc=False)
    elif case == "cleanup":
        backend.child_plan = {"stubborn": True}
    result, _ = run_ma1(runners, tmp_path, prep, backend=backend, record=case != "no_record")
    reading = result["ma1"]["interpretation"]
    expected = {"m3_fails": (True, True), "no_record": (True, False), "maps_unavailable": (True, False),
                "malloc_unsupported": (True, False), "cleanup": (False, True)}[case]
    assert (reading["execution_valid"], reading["attribution_recorded"]) == expected
    if case == "m3_fails":  # a failing reference criterion is descriptive only
        assert result["ma1"]["reference_criteria"]["M3_trend_swap"]["status"] == "fail"
    if case == "no_record":
        assert result["ma1"]["attribution"] == {"status": "unavailable"}
    assert reading["eligible_for_maintainer_review"] is False


@pytest.mark.parametrize(("overrides", "refusal"), [
    ({"confirm_ma1": False, "confirm_step4plr": True}, "ma1_operator_prerequisites_unconfirmed"),
    ({"dropped_caches": False}, "ma1_cache_drop_not_declared"),
    ({"explicit_check9": False}, "ma1_requires_explicit_check9_report"),
    ({"clip": None}, "ma1_clip_required"),
    ({"identity_report": None}, "step4_identity_report_required"),
])
def test_ma1_refuses_before_any_process_when_a_step4_prerequisite_fails(runners, tmp_path, overrides, refusal) -> None:
    prep = prepare(runners, tmp_path)
    result, backend = run_ma1(runners, tmp_path, prep, **overrides)
    assert result["ma1"]["status"] == "refused" and refusal in result["ma1"]["refusals"]
    assert not [c for c in backend.children if "demo_profile.py" in " ".join(c.argv) or "logger" in c.argv[0]]


def test_the_ma1_cli_exit_reflects_execution_and_recording_and_needs_its_own_confirmation(
    runners, tmp_path, monkeypatch, capsys,
) -> None:
    op = runners.operator
    monkeypatch.setattr(op, "OUTPUT_ROOT", tmp_path)
    prep = prepare(runners, tmp_path)
    monkeypatch.setattr(op, "identity_files", lambda clip: prep.files)

    def backend(record=True):
        b = Backend()
        b.identity_hashes = {label: entry["sha256"] for label, entry in prep.identity["files"].items()}
        b.plr_report = fake_plr_report(runners, tmp_path)
        b.memattr_report = fake_memattr_report(runners, tmp_path) if record else None
        b.profile["steady_trend"]["used"]["slope_bytes_per_min"] = 11_886_599  # M3 fails: still exit 0
        return b

    common = ["--execute-workload", "ma1", "--check9-report", str(prep.check9_path), "--identity-report",
              str(prep.identity_path), "--step4-clip", str(prep.files["clip"]), "--operator-dropped-caches"]
    assert op.main([*common, "--confirm-ma1-prerequisites"], backend=backend()) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ma1"]["interpretation"]["eligible_for_maintainer_review"] is False
    assert op.main([*common, "--confirm-ma1-prerequisites"], backend=backend(record=False)) == 1
    capsys.readouterr()
    assert op.main(common, backend=backend()) == 1
    assert "ma1_operator_prerequisites_unconfirmed" in capsys.readouterr().out
    for wrong in (["--confirm-step4plr-prerequisites"], ["--execute-workload", "step4plr", "--confirm-ma1-prerequisites"]):
        with pytest.raises(SystemExit):  # each confirmation belongs to its own mode
            op.main([*common, *wrong] if wrong[0].startswith("--confirm") else wrong, backend=Backend())


# ---------------------------------------------------------------- MA1-THP (D56): MA1 plus read-only THP observation


THP_SNAPSHOT = {"kernel": {"release": "5.15.148-tegra", "version": "#1 SMP PREEMPT Thu Sep 18 /home/someone"},
                "base_page_bytes": 4096, "thp_pmd_bytes": 2 << 20, "hugetlb_default_bytes": 2 << 20,
                "enabled": "always", "defrag": "madvise", "shmem_enabled": "never", "use_zero_page": 1,
                "khugepaged": {"pages_to_scan": 4096, "scan_sleep_millisecs": 10000, "max_ptes_none": 511},
                "config": {"HZ": 250, "TRANSPARENT_HUGEPAGE": "y"}, "read_ms": 2.0}


def fake_thp_report(runners, tmp_path, *, reading="supported", after=THP_SNAPSHOT) -> dict:
    """The profiler's own thp.json, built by its function from a few synthetic rows and smaps readings."""
    profile = runners.profile
    run = tmp_path / f"thp-source-{reading}"
    run.mkdir(exist_ok=True)
    stepped, collapsed = (lambda t: t >= 150.0), (lambda t: reading == "supported" and t >= 150.0)
    samples = [{"t": t, "phase": "steady", "work_rss": 2_000_000_000 + (4 << 20 if stepped(t) else 0),
                "llama_rss": 1, "anon_pages": 1,
                **({"thp_t_mono": t, "khugepaged_pages_collapsed": 5 + collapsed(t), "thp_collapse_alloc": 5 + collapsed(t),
                    "thp_split_page": 0, "thp_split_pmd": 0, "anon_huge_pages": 1}
                   if reading != "inconclusive" else {})}
               for t in (149.6, 149.8, 150.0, 150.2)]
    events = [{"t_mono": t, "event": "memattr_maps", "status": "observed",
               "categories": {name: {"rss_bytes": 1 + (4 << 20 if name == "heap" and stepped(t) else 0),
                                     "anon_huge_pages_bytes": (2 << 20 if name == "heap" and collapsed(t) else 0)}
                              for name in profile.MAPS_CATEGORIES}} for t in (145.0, 150.0, 155.0)]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    manifest = {"thp_settings": {"before": THP_SNAPSHOT, "after": after}}
    return profile.thp_report(run, "complete", samples, manifest, None, (140.0, 160.0))


def run_ma1thp(runners, tmp_path, prep, backend=None, *, reading="supported", record=True, **overrides):
    op = runners.operator
    output = private_dir(tmp_path, "ma1thp")
    backend = backend or Backend()
    backend.output = output
    backend.identity_hashes = {label: entry["sha256"] for label, entry in prep.identity["files"].items()}
    backend.plr_report = backend.plr_report or fake_plr_report(runners, tmp_path)
    backend.memattr_report = backend.memattr_report or fake_memattr_report(runners, tmp_path)
    if record and backend.thp_report is None:
        backend.thp_report = fake_thp_report(runners, tmp_path, reading=reading)
    kwargs = dict(check9_report=prep.check9_path, identity_report=prep.identity_path, clip=prep.files["clip"],
                  confirm_ma1thp=True, dropped_caches=True, identity_files_fn=lambda clip: prep.files)
    kwargs.update(overrides)
    return op.execute("ma1thp", backend, output, **kwargs), backend


def test_the_ma1thp_child_is_ma1s_command_plus_the_thp_flag_only(runners, tmp_path) -> None:
    op = runners.operator
    prep = prepare(runners, tmp_path)
    result, backend = run_ma1thp(runners, tmp_path, prep)
    (child,) = [c for c in backend.children if "demo_profile.py" in " ".join(c.argv)]
    assert child.argv == [*op.ma1_argv(backend.output, prep.files["clip"]), "--thp-observation"]
    assert op.ma1_argv(backend.output, prep.files["clip"])[-1] == "5.0"  # MA1's own command is unchanged
    assert result["ma1thp"]["deadline_s"] == op.STEP4_DEADLINE_S
    assert [(m[0].split()[:2], m[2]) for m in backend.markers] == [(["ma1thp", "start"], "sentinel-ma1thp"),
                                                                   (["ma1thp", "end"], "sentinel-ma1thp")]
    args = runners.profile.parse_args(child.argv[2:])
    ma1 = runners.profile.parse_args(op.ma1_argv(backend.output, prep.files["clip"])[2:])
    assert args.thp_observation is True and ma1.thp_observation is False
    assert {k: v for k, v in vars(args).items() if k != "thp_observation"} == {
        k: v for k, v in vars(ma1).items() if k != "thp_observation"}  # rates, durations, guard inputs, interval


def test_a_completed_ma1thp_run_records_thp_and_is_never_eligible(runners, tmp_path) -> None:
    prep = prepare(runners, tmp_path)
    result, _ = run_ma1thp(runners, tmp_path, prep)
    section = result["ma1thp"]
    assert section["status"] == "completed" and section["acceptance"].startswith("never: instrumented diagnostic")
    assert "criteria" not in section and "PLR and MA1 runs only" in section["reference_note"]
    reading = section["interpretation"]
    assert reading["execution_valid"] and reading["attribution_recorded"] and reading["thp_recorded"] is True
    assert reading["thp_reading"] == {"label": "supported", "reason": None}
    assert reading["eligible_for_maintainer_review"] is False and "unconfirmed" in reading["thp"]
    excerpt = section["thp"]
    assert excerpt["status"] == "recorded" and excerpt["label"] == "ma1thp-thp-observation"
    assert excerpt["settings"]["before"]["enabled"] == "always" and excerpt["settings"]["before"]["config"]["HZ"] == 250
    assert excerpt["settings"]["before"]["kernel_release"] == "5.15.148-tegra"
    assert excerpt["steps"]["by_outcome"] == {"supported": 1, "unsupported": 0, "inconclusive": 0}
    text = json.dumps(result)
    assert "/home/someone" not in text and "PREEMPT" not in text  # the version string stays in the private file
    assert '"listed"' not in json.dumps(excerpt) and "rule" not in excerpt["outcome"]
    assert "ma1" not in result and result["step4"]["status"] == "PENDING"


@pytest.mark.parametrize(("case", "exit_code"), [
    ("supported", 0), ("unsupported", 0), ("inconclusive_counters", 1), ("no_record", 1), ("settings_after_missing", 1),
])
def test_the_ma1thp_exit_needs_recorded_thp_readings_and_never_their_reading(
    runners, tmp_path, monkeypatch, capsys, case, exit_code,
) -> None:
    op = runners.operator
    monkeypatch.setattr(op, "OUTPUT_ROOT", tmp_path)
    prep = prepare(runners, tmp_path)
    monkeypatch.setattr(op, "identity_files", lambda clip: prep.files)
    backend = Backend()
    backend.identity_hashes = {label: entry["sha256"] for label, entry in prep.identity["files"].items()}
    backend.plr_report = fake_plr_report(runners, tmp_path)
    backend.memattr_report = fake_memattr_report(runners, tmp_path)
    backend.profile["steady_trend"]["used"]["slope_bytes_per_min"] = 16_235_751  # M3 fails: descriptive only
    backend.thp_report = None if case == "no_record" else fake_thp_report(
        runners, tmp_path, reading={"inconclusive_counters": "inconclusive"}.get(case, case.split("_")[0]),
        after=None if case == "settings_after_missing" else THP_SNAPSHOT)
    common = ["--execute-workload", "ma1thp", "--check9-report", str(prep.check9_path), "--identity-report",
              str(prep.identity_path), "--step4-clip", str(prep.files["clip"]), "--operator-dropped-caches"]
    assert op.main([*common, "--confirm-ma1thp-prerequisites"], backend=backend) == exit_code
    reading = json.loads(capsys.readouterr().out)["ma1thp"]["interpretation"]
    assert reading["eligible_for_maintainer_review"] is False
    if case == "unsupported":  # an unsupported reading is recorded, not a failure and not a fix
        assert reading["thp_recorded"] is True and reading["thp_reading"]["label"] == "unsupported"


@pytest.mark.parametrize(("overrides", "refusal"), [
    ({"confirm_ma1thp": False, "confirm_ma1": True}, "ma1thp_operator_prerequisites_unconfirmed"),
    ({"dropped_caches": False}, "ma1thp_cache_drop_not_declared"),
    ({"explicit_check9": False}, "ma1thp_requires_explicit_check9_report"),
    ({"clip": None}, "ma1thp_clip_required"),
    ({"identity_report": None}, "step4_identity_report_required"),
])
def test_ma1thp_refuses_before_any_process_when_a_step4_prerequisite_fails(runners, tmp_path, overrides,
                                                                          refusal) -> None:
    prep = prepare(runners, tmp_path)
    result, backend = run_ma1thp(runners, tmp_path, prep, **overrides)
    assert result["ma1thp"]["status"] == "refused" and refusal in result["ma1thp"]["refusals"]
    assert not [c for c in backend.children if "demo_profile.py" in " ".join(c.argv) or "logger" in c.argv[0]]


def test_each_ma1_confirmation_belongs_to_its_own_mode(runners, tmp_path) -> None:
    op = runners.operator
    for wrong in (["--execute-workload", "ma1", "--confirm-ma1thp-prerequisites"],
                  ["--execute-workload", "ma1thp", "--confirm-ma1-prerequisites"],
                  ["--confirm-ma1thp-prerequisites"],
                  ["--execute-workload", "ma1thp", "--confirm-ma1wtd-prerequisites"],
                  ["--execute-workload", "ma1wtd", "--confirm-ma1thp-prerequisites"],
                  ["--confirm-ma1wtd-prerequisites"]):
        with pytest.raises(SystemExit):
            op.main(wrong, backend=Backend())


# ---------------------------------------------------------------- MA1-WTD (D57): MA1-THP with the workload's THP disabled


def fake_wtd_report(runners, tmp_path, *, outcome="supported") -> dict:
    """The profiler's own wtd.json, built by its function from a few synthetic rows, readings and events."""
    profile = runners.profile
    run = tmp_path / f"wtd-source-{outcome}"
    run.mkdir(exist_ok=True)
    grown = outcome == "contrary"
    samples = [{"t": round(130.0 + 0.2 * i, 3), "phase": "steady",
                "work_rss": 2_000_000_000 + (4 << 20 if grown and i >= 100 else 0), "llama_rss": 1, "anon_pages": 1,
                "thp_t_mono": 1.0, "khugepaged_pages_collapsed": 5 + (i >= 100) * (outcome != "no_opportunity"),
                "thp_collapse_alloc": 5 + (i >= 100) * (outcome != "no_opportunity"), "khugepaged_full_scans": 2,
                "thp_split_page": 0, "thp_split_pmd": 0, "anon_huge_pages": 1} for i in range(200)]
    events = [{"t_mono": 100.0, "event": "thp_disable", "requested": True, "model_modules_loaded": [], "set_rc": 0,
               "set_errno": None, "get_value": 1, "thp_enabled": 0, "anon_huge_pages_bytes": 0,
               "verified": outcome != "not_verified", "reason": None},
              {"t_mono": 101.0, "event": "phase", "name": "detector_load"},
              *({"t_mono": 100.0, "event": "wtd_scope", "checkpoint": name, "thp_enabled": profile.WTD_EXPECTED[name],
                 "expected": profile.WTD_EXPECTED[name], "ok": True} for name in profile.WTD_CHECKPOINTS)]
    for t in range(130, 171, 5):
        rss = (600 << 20) + (4 << 20 if grown and t >= 150 else 0)
        events.append({"t_mono": float(t), "event": "memattr_maps", "status": "observed",
                       "categories": {name: {"size_bytes": (800 << 20) if name == "heap" else 1,
                                             "rss_bytes": rss if name == "heap" else 1, "anon_huge_pages_bytes": 0}
                                      for name in profile.MAPS_CATEGORIES}})
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    manifest = {"thp_settings": {"before": THP_SNAPSHOT, "after": THP_SNAPSHOT}}
    stats = {"clip_loops": 1, "clip_loop_t_mono": [100.5], "detector": {"unique_fps": 15.0}, "face": {}, "scene": {}}
    report = profile.wtd_report(run, "complete", samples, manifest, (130.0, 168.0), stats)
    report["intervention"]["startup"]["note"] = "/home/someone/private"  # never relayed
    return report


def run_ma1wtd(runners, tmp_path, prep, backend=None, *, outcome="supported", record=True, **overrides):
    op = runners.operator
    output = private_dir(tmp_path, "ma1wtd")
    backend = backend or Backend()
    backend.output = output
    backend.identity_hashes = {label: entry["sha256"] for label, entry in prep.identity["files"].items()}
    backend.plr_report = backend.plr_report or fake_plr_report(runners, tmp_path)
    backend.memattr_report = backend.memattr_report or fake_memattr_report(runners, tmp_path)
    backend.thp_report = backend.thp_report or fake_thp_report(runners, tmp_path, reading="unsupported")
    if record and backend.wtd_report is None:
        backend.wtd_report = fake_wtd_report(runners, tmp_path, outcome=outcome)
    kwargs = dict(check9_report=prep.check9_path, identity_report=prep.identity_path, clip=prep.files["clip"],
                  confirm_ma1wtd=True, dropped_caches=True, identity_files_fn=lambda clip: prep.files)
    kwargs.update(overrides)
    return op.execute("ma1wtd", backend, output, **kwargs), backend


def test_the_ma1wtd_child_is_ma1thps_command_plus_the_disable_flag_only(runners, tmp_path) -> None:
    op = runners.operator
    prep = prepare(runners, tmp_path)
    result, backend = run_ma1wtd(runners, tmp_path, prep)
    (child,) = [c for c in backend.children if "demo_profile.py" in " ".join(c.argv)]
    assert child.argv == [*op.ma1thp_argv(backend.output, prep.files["clip"]), "--workload-thp-disable"]
    assert result["ma1wtd"]["deadline_s"] == op.STEP4_DEADLINE_S
    assert [(m[0].split()[:2], m[2]) for m in backend.markers] == [(["ma1wtd", "start"], "sentinel-ma1wtd"),
                                                                   (["ma1wtd", "end"], "sentinel-ma1wtd")]
    args = runners.profile.parse_args(child.argv[2:])
    ma1thp = runners.profile.parse_args(op.ma1thp_argv(backend.output, prep.files["clip"])[2:])
    assert args.workload_thp_disable is True and ma1thp.workload_thp_disable is False
    assert {k: v for k, v in vars(args).items() if k != "workload_thp_disable"} == {
        k: v for k, v in vars(ma1thp).items() if k != "workload_thp_disable"}  # models, rates, durations, guard inputs
    assert result["ma1wtd"]["operator_thp_enabled"] == 1


def test_a_completed_ma1wtd_run_records_the_intervention_and_reading_and_is_never_eligible(runners, tmp_path) -> None:
    prep = prepare(runners, tmp_path)
    result, _ = run_ma1wtd(runners, tmp_path, prep)
    section = result["ma1wtd"]
    assert section["status"] == "completed" and section["acceptance"].startswith("never: instrumented diagnostic")
    assert "criteria" not in section and "fault-time huge pages" in section["reference_note"]
    reading = section["interpretation"]
    assert reading["execution_valid"] and reading["attribution_recorded"] and reading["thp_recorded"]
    assert reading["intervention_verified"] is True and reading["wtd_reading"] == {"label": "supported", "reason": None}
    assert reading["thp_reading"]["label"] == "unsupported"  # D56's reading is still recorded beside it
    assert reading["eligible_for_maintainer_review"] is False and "not a fix" in reading["wtd"]
    excerpt = section["wtd"]
    assert excerpt["status"] == "recorded" and excerpt["label"] == "ma1wtd-workload-thp-disable"
    assert excerpt["intervention"]["scope"]["llama_ready"]["thp_enabled"] == {"profiler": 1, "llama": 1}
    assert excerpt["growth"]["opportunity"]["comparable"] is True and excerpt["clip_loops"]["complete"] is True
    assert excerpt["growth"]["net_steps"]["by_class"]["net_step"] == 0
    text = json.dumps(result)
    assert '"listed"' not in json.dumps(excerpt) and "/home/someone" not in text and "rule" not in excerpt["outcome"]
    assert "ma1thp" not in result and result["step4"]["status"] == "PENDING"


@pytest.mark.parametrize(("case", "exit_code", "label"), [
    ("supported", 0, "supported"), ("contrary", 0, "contrary"), ("no_opportunity", 0, "inconclusive"),
    ("not_verified", 1, "inconclusive"), ("no_record", 1, "unavailable"),
])
def test_the_ma1wtd_exit_needs_a_verified_intervention_and_never_its_reading(
    runners, tmp_path, monkeypatch, capsys, case, exit_code, label,
) -> None:
    op = runners.operator
    monkeypatch.setattr(op, "OUTPUT_ROOT", tmp_path)
    prep = prepare(runners, tmp_path)
    monkeypatch.setattr(op, "identity_files", lambda clip: prep.files)
    backend = Backend()
    backend.identity_hashes = {label: entry["sha256"] for label, entry in prep.identity["files"].items()}
    backend.plr_report = fake_plr_report(runners, tmp_path)
    backend.memattr_report = fake_memattr_report(runners, tmp_path)
    backend.thp_report = fake_thp_report(runners, tmp_path, reading="unsupported")
    backend.wtd_report = None if case == "no_record" else fake_wtd_report(runners, tmp_path, outcome=case)
    backend.profile["steady_trend"]["used"]["slope_bytes_per_min"] = 19_846_506  # M3 fails: descriptive only
    common = ["--execute-workload", "ma1wtd", "--check9-report", str(prep.check9_path), "--identity-report",
              str(prep.identity_path), "--step4-clip", str(prep.files["clip"]), "--operator-dropped-caches"]
    assert op.main([*common, "--confirm-ma1wtd-prerequisites"], backend=backend) == exit_code
    reading = json.loads(capsys.readouterr().out)["ma1wtd"]["interpretation"]
    assert reading["wtd_reading"]["label"] == label and reading["eligible_for_maintainer_review"] is False
    assert reading["intervention_verified"] is (case not in ("not_verified", "no_record"))


@pytest.mark.parametrize(("overrides", "flag", "refusal"), [
    ({"confirm_ma1wtd": False, "confirm_ma1thp": True}, 1, "ma1wtd_operator_prerequisites_unconfirmed"),
    ({"dropped_caches": False}, 1, "ma1wtd_cache_drop_not_declared"),
    ({"explicit_check9": False}, 1, "ma1wtd_requires_explicit_check9_report"),
    ({"clip": None}, 1, "ma1wtd_clip_required"),
    ({"identity_report": None}, 1, "step4_identity_report_required"),
    ({}, 0, "ma1wtd_operator_thp_enabled_not_1"),  # everything it starts would inherit the disable
    ({}, None, "ma1wtd_operator_thp_enabled_not_1"),  # unreadable is never as expected
])
def test_ma1wtd_refuses_before_any_process_when_a_prerequisite_fails(runners, tmp_path, overrides, flag,
                                                                    refusal) -> None:
    prep = prepare(runners, tmp_path)
    backend = Backend()
    backend.thp_flag = flag
    result, backend = run_ma1wtd(runners, tmp_path, prep, backend=backend, **overrides)
    assert result["ma1wtd"]["status"] == "refused" and refusal in result["ma1wtd"]["refusals"]
    assert not [c for c in backend.children if "demo_profile.py" in " ".join(c.argv) or "logger" in c.argv[0]]


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
