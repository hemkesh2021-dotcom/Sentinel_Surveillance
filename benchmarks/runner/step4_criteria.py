"""Step-4 combined-profile criteria ``step4-combined-cache-off-v2`` (operator checklist step 4; D46, D47).

Standard library only: demo_profile.py (system Python) computes the run-side
evidence with it, and operator_check.py combines that with the guard, Check 9,
identity and kernel evidence. Every criterion is ``pass``, ``fail`` or
``unavailable``; unavailable is never a pass, and missing telemetry is never
read as zero. Nothing here accepts a profile: acceptance is only a separate,
maintainer-approved registry commit (D46).

These are demo criteria for a 640x480 replay at 15 fps with detector, face
(1 Hz) and scene (every 4 s). They do not test the guide's beta gates, such
as at least 15 unique detected frames/s sustained with 1080p input.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

CRITERIA_ID = "step4-combined-cache-off-v2"
STEADY_S = 600.0
SAMPLE_INTERVAL_S = 0.2
MIN_STEADY_COVERAGE = 0.95  # share of the expected 0.2 s samples present in the steady interval
MAX_SAMPLE_GAP_S = 1.0  # a longer gap inside the steady interval makes it uncertain
TARGET_STEADY_BYTES = 5_000_000_000  # guide ch. 12/22: steady pressure, every steady sample
TARGET_PEAK_BYTES = 5_400_000_000  # guide ch. 12/22: sampled cold-load/runtime peak
MAX_STEADY_SLOPE_BYTES_PER_MIN = 10_000_000
MIN_UNIQUE_FPS = 14.5
WINDOW_S = 10
MIN_WINDOW_FPS = 13.5
MAX_SCHEDULE_AGE_P95_MS = 150.0
MAX_SCHEDULE_AGE_P99_MS = 250.0
MIN_PROCESSED_RATIO = 0.99  # supplementary
MIN_FACE_HZ = 0.95
MIN_SCENE_ATTEMPTS = 140
MIN_SCENE_STRICT_VALID_SHARE = 0.95
MAX_SCENE_LATENCY_MS = 8000.0  # D16 job timeout
POST_RUN_WAIT_S = 60.0
EXPECTED_LLAMA_FLAGS = ("--n-gpu-layers", "999", "--ctx-size", "2048", "--parallel", "1", "--cache-ram", "0")
# Step-4 PLR (opt-in, D54): step 4's procedure with each model's file cache released after its load and settle
# (MR1's release), then step 4's full warm-up and steady phases. Every rule and threshold above applies unchanged,
# under this distinct identity: a PLR result is never D47's step-4 result, and D46's admission
# (sentinel.adapters.accepted_profile_problem) requires CRITERIA_ID. R also needs the declared releases to have
# happened: all three recorded, every call returning 0. The release outcomes themselves stay descriptive.
PLR_CRITERIA_ID = "step4plr-combined-cache-off-v2"
PLR_RUN_INPUTS = ("releases_recorded", "release_calls_returned_0")
PLR_PROCEDURE = ("step 4 with each model's file cache released (posix_fadvise DONTNEED) after its load and settle, "
                 "then step 4's 120 s warm-up and 600 s steady phases (D54)")
PLR_ADMISSION = (f"not admissible: D46 scene admission requires {CRITERIA_ID} from step 4's procedure, and "
                 "sentinel run performs no post-load release")
# D58 candidate (opt-in): the PLR procedure plus the workload's verified THP disable (D57's call, shared with sentinel
# run), without MA1's sampling. Every rule and threshold above applies unchanged under this identity; R also needs
# PLR's two release inputs and these four, which the profiler's candidate.json records (missing is never met).
CANDIDATE_CRITERIA_ID = "step4cand-wtd-plr-combined-cache-off-v2"
CANDIDATE_RUN_INPUTS = ("thp_disable_verified", "thp_scope_verified", "thp_effect_verified", "thp_settings_unchanged")
CANDIDATE_PROCEDURE = ("step 4 with each model's file cache released after its load and settle (D54) and transparent "
                       "huge pages disabled in the workload process alone, verified (D57's call), without MA1's "
                       "allocator, smaps or THP-counter sampling; step 4's 120 s warm-up and 600 s steady phases (D58)")
CANDIDATE_ADMISSION = (f"not admissible: D46 scene admission requires {CRITERIA_ID}; admitting this identity needs a "
                       "separate maintainer decision and acceptance commit, and sentinel run must apply the same "
                       "memory policy (thp workload_disabled, model-file release post_load)")
# The memory policy (sentinel.memory_policy: thp, model_file_release) each identity's procedure measures.
MEMORY_POLICIES = {
    CRITERIA_ID: ("system", "none"),
    PLR_CRITERIA_ID: ("system", "post_load"),
    CANDIDATE_CRITERIA_ID: ("workload_disabled", "post_load"),
}

PASS, FAIL, UNAVAILABLE = "pass", "fail", "unavailable"


def _boundary(events: Sequence[Mapping[str, Any]], name: str, **match: Any) -> float | None:
    for event in events:
        if event.get("event") == name and all(event.get(k) == v for k, v in match.items()):
            value = event.get("boundary_t_mono")
            if type(value) in (int, float):
                return float(value)
    return None


def steady_interval(
    events: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]], steady_s: float = STEADY_S
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    """The steady interval from monotonic boundaries, and the samples inside it.

    Start: the workload's ``steady_boundary`` start. End: the earliest of the
    workload's ``steady_boundary`` end (stamped before its worker threads are
    joined) and any orchestrator ``stop_boundary`` (memory floor, abort,
    interruption). CSV phase labels are not used. Without a start, or without
    any end, the interval is unavailable.
    """
    start = _boundary(events, "steady_boundary", edge="start")
    end_mark = _boundary(events, "steady_boundary", edge="end")
    stops = [float(e["boundary_t_mono"]) for e in events
             if e.get("event") == "stop_boundary" and type(e.get("boundary_t_mono")) in (int, float)]
    stop = min(stops) if stops else None
    if start is None:
        return {"status": UNAVAILABLE, "reason": "no steady start boundary"}, []
    ends = [t for t in (end_mark, stop) if t is not None]
    if not ends:
        return {"status": UNAVAILABLE, "reason": "no steady end or stop boundary"}, []
    end = min(ends)
    duration = max(0.0, end - start)
    completed = end_mark is not None and end_mark == end and duration >= steady_s - 1.0
    rows = [s for s in samples if start <= s["t"] <= end]
    info: dict[str, Any] = {
        "status": "complete" if completed else "truncated",
        "basis": "monotonic boundaries",
        "start_t_mono": start,
        "end_t_mono": end,
        "ended_by": "workload_end" if end_mark is not None and end_mark == end else "stop_boundary",
        "duration_s": round(duration, 3),
        "expected_s": steady_s,
        "samples": len(rows),
    }
    expected = duration / SAMPLE_INTERVAL_S if duration else 0.0
    info["coverage"] = round(len(rows) / expected, 4) if expected else None
    edges = [start, *(r["t"] for r in rows), end]
    info["max_gap_s"] = round(max(b - a for a, b in zip(edges, edges[1:])), 3) if len(edges) > 1 else None
    used = [r["used"] for r in rows if r.get("used") is not None]
    if used:
        above = 0.0
        for index, row in enumerate(rows):
            if row.get("used") is not None and row["used"] > TARGET_STEADY_BYTES:
                following = rows[index + 1]["t"] if index + 1 < len(rows) else end
                above += following - row["t"]
        ordered = sorted(used)
        info.update(
            max_bytes=max(used),
            median_bytes=int(statistics.median(used)),
            p95_bytes=ordered[max(0, -(-95 * len(ordered) // 100) - 1)],  # reported, never a criterion
            seconds_above_target=round(above, 3),
            share_above_target=round(above / duration, 4) if duration else None,
            mem_free_min_bytes=min((r["mem_free"] for r in rows if r.get("mem_free") is not None), default=None),
        )
    for key in ("pswpin", "pswpout"):
        values = [r[key] for r in rows if r.get(key) is not None]
        info[f"{key}_delta"] = values[-1] - values[0] if len(values) > 1 else None  # None: unavailable, not zero
    return info, rows


def cache_evidence(
    manifest: Mapping[str, Any],
    cmdline: Mapping[str, Any] | None,
    cache_summary: Mapping[str, Any] | None,
    *,
    telemetry_timestamped: bool,
) -> dict[str, Any]:
    """Is llama-server's prompt cache verifiably off? Several independent sources must agree.

    - the flag the profiler passed (manifest ``llama_server.cache_ram_mib``);
    - the running server's own command line (``/proc/<pid>/cmdline``, captured once ready);
    - the installed build knows the option (its string is in the build files; llama-server
      also exits on an unknown option, so a ready server with it in its command line accepted it);
    - the server's startup log line;
    - zero cache-state updates in the steady interval, counted only when the startup line was
      captured with receipt times (otherwise the count is unavailable, never zero).
    """
    server = manifest.get("llama_server") or {}
    flag = server.get("cache_ram_mib")
    build = server.get("build_has_cache_ram_option")
    running = (cmdline or {}).get("cache_ram")
    startup = (cache_summary or {}).get("startup")
    startup_state = None if not startup else ("enabled" if startup.get("enabled") else "disabled")
    updates = None
    if startup_state is not None and telemetry_timestamped:
        updates = (cache_summary or {}).get("steady_state_updates") or 0
    evidence = {
        "manifest_cache_ram_mib": flag,
        "running_cmdline_cache_ram": running,
        "build_has_cache_ram_option": build,
        "startup_log": startup_state,
        "steady_state_updates": updates,
    }
    # A running server without --cache-ram, or a manifest without the flag, uses llama-server's default cache.
    if (startup_state == "enabled" or ((cmdline or {}).get("seen") and running != "0")
            or ("cache_ram_mib" in server and flag != 0)):
        verdict = "enabled"
    elif flag == 0 and running == "0" and build is True and startup_state == "disabled" and updates == 0:
        verdict = "disabled_verified"
    else:
        verdict = "unverified"
    evidence["missing"] = [k for k, ok in (
        ("manifest flag", flag == 0), ("running command line", running == "0"), ("installed build", build is True),
        ("startup log", startup_state == "disabled"), ("steady update count", updates == 0),
    ) if not ok]
    evidence["verdict"] = verdict
    return evidence


def _criterion(status: str, requirement: str, value: Any = None) -> dict[str, Any]:
    return {"status": status, "requirement": requirement, "value": value}


def _check(ok: bool | None, requirement: str, value: Any = None) -> dict[str, Any]:
    return _criterion(UNAVAILABLE if ok is None else PASS if ok else FAIL, requirement, value)


def _num(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) else None


def evaluate_profile(profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The criteria the profiler's own evidence can decide (S, M1-M3, T1-T3, F, V1-V2, G, C)."""
    steady = profile.get("steady_interval") or {}
    work = profile.get("workload_steady") or {}
    det, face, scene = work.get("detector") or {}, work.get("face") or {}, work.get("scene") or {}
    complete = steady.get("status") == "complete"
    coverage = _num(steady.get("coverage"))
    gap = _num(steady.get("max_gap_s"))
    out: dict[str, dict[str, Any]] = {}
    out["S_steady_interval"] = _check(
        None if steady.get("status") in (None, UNAVAILABLE) or coverage is None or gap is None
        else complete and coverage >= MIN_STEADY_COVERAGE and gap <= MAX_SAMPLE_GAP_S,
        f"complete {STEADY_S:g} s interval from monotonic boundaries, coverage >= {MIN_STEADY_COVERAGE}, "
        f"no gap over {MAX_SAMPLE_GAP_S:g} s",
        {k: steady.get(k) for k in ("status", "duration_s", "coverage", "max_gap_s", "ended_by")},
    )
    usable = out["S_steady_interval"]["status"] == PASS
    smax, above = _num(steady.get("max_bytes")), _num(steady.get("seconds_above_target"))
    out["M1_steady_pressure"] = _check(
        None if not usable or smax is None or above is None else smax <= TARGET_STEADY_BYTES and above == 0,
        f"every steady sample <= {TARGET_STEADY_BYTES} B; 0 s above",
        {"max_bytes": steady.get("max_bytes"), "seconds_above_target": steady.get("seconds_above_target"),
         "share_above_target": steady.get("share_above_target")},
    )
    peak = _num((profile.get("combined") or {}).get("run_peak_bytes"))
    out["M2_run_peak"] = _check(None if peak is None else peak <= TARGET_PEAK_BYTES,
                                f"sampled run peak <= {TARGET_PEAK_BYTES} B", peak)
    slope = _num(((profile.get("steady_trend") or {}).get("used") or {}).get("slope_bytes_per_min"))
    swapin = steady.get("pswpin_delta")
    out["M3_trend_swap"] = _check(
        None if not usable or slope is None or swapin is None else slope <= MAX_STEADY_SLOPE_BYTES_PER_MIN and swapin == 0,
        f"steady slope <= {MAX_STEADY_SLOPE_BYTES_PER_MIN} B/min; no pages swapped in during steady",
        {"slope_bytes_per_min": slope, "pswpin_delta": swapin},
    )
    fps = _num(det.get("unique_fps"))
    window = _num((det.get("windows") or {}).get("min_fps"))
    out["T1_unique_throughput"] = _check(
        None if not usable or fps is None or window is None else fps >= MIN_UNIQUE_FPS and window >= MIN_WINDOW_FPS,
        f"mean >= {MIN_UNIQUE_FPS} unique detected frames/s; every complete {WINDOW_S} s window >= {MIN_WINDOW_FPS}",
        {"unique_fps": fps, "min_window_fps": window},
    )
    age = det.get("schedule_age_ms") or {}
    p95, p99 = _num(age.get("p95")), _num(age.get("p99"))
    out["T2_frame_age"] = _check(
        None if not usable or p95 is None or p99 is None else p95 <= MAX_SCHEDULE_AGE_P95_MS and p99 <= MAX_SCHEDULE_AGE_P99_MS,
        f"replay scheduling age (scheduled arrival to result) p95 <= {MAX_SCHEDULE_AGE_P95_MS:g} ms, "
        f"p99 <= {MAX_SCHEDULE_AGE_P99_MS:g} ms; decode-to-result age is reported, not gated",
        {"schedule_age_ms": age or None, "decode_to_result_age_ms": det.get("decode_to_result_age_ms")},
    )
    processed, source = _num(det.get("processed_frames")), _num(det.get("source_frames"))
    out["T3_processed_ratio"] = _check(
        None if not usable or not processed or not source else processed / source >= MIN_PROCESSED_RATIO,
        f"supplementary: processed/decoded >= {MIN_PROCESSED_RATIO}",
        None if not processed or not source else round(processed / source, 4),
    )
    # Error totals gate F and V1: the workload counts them before sanitizing, which drops error
    # names it does not recognise (e.g. RemoteDisconnected), so an empty ``errors`` map alone is not zero.
    hz, face_errors, face_error_count = _num(face.get("achieved_hz")), face.get("errors"), face.get("error_count")
    out["F_face"] = _check(
        None if not usable or hz is None or not isinstance(face_errors, dict) or type(face_error_count) is not int
        else hz >= MIN_FACE_HZ and face_error_count == 0 and not face_errors,
        f"face workload executed: >= {MIN_FACE_HZ} Hz achieved, 0 errors",
        {"runs": face.get("runs"), "achieved_hz": hz, "error_count": face_error_count, "errors": face_errors,
         "latency_ms": face.get("latency_ms")},
    )
    attempts = scene.get("attempts")
    errors = scene.get("errors")
    finish = scene.get("finish_reasons")
    error_counts = [scene.get(key) for key in ("client_timeouts", "http_errors", "transport_errors")]
    out["V1_scene_requests"] = _check(
        None if not usable or type(attempts) is not int or not isinstance(errors, dict)
        or type(scene.get("over_d16_timeout")) is not int or any(type(n) is not int for n in error_counts)
        else attempts >= MIN_SCENE_ATTEMPTS and not any(error_counts) and not errors and scene["over_d16_timeout"] == 0,
        f">= {MIN_SCENE_ATTEMPTS} attempts; 0 HTTP, transport or client-timeout errors; 0 completions over "
        f"{MAX_SCENE_LATENCY_MS:g} ms",
        {k: scene.get(k) for k in ("attempts", "completed", "client_timeouts", "http_errors", "transport_errors",
                                   "over_d16_timeout", "latency_ms")},
    )
    valid, completed = scene.get("valid_reports"), scene.get("completed")
    out["V2_scene_completion"] = _check(
        None if not usable or type(attempts) is not int or attempts == 0 or not isinstance(finish, dict)
        or type(valid) is not int or type(completed) is not int
        else (set(finish) <= {"stop"} and finish.get("stop", 0) == completed
              and valid >= MIN_SCENE_STRICT_VALID_SHARE * attempts),
        f"every completion finished 'stop' (0 truncated/other); strict-valid >= {MIN_SCENE_STRICT_VALID_SHARE} of "
        "attempts; structural validity only, not scene accuracy",
        {"valid_reports": valid, "invalid_reports": scene.get("invalid_reports"), "finish_reasons": finish,
         "rejected_reports_by_reason": scene.get("rejected_reports_by_reason")},
    )
    gpu = profile.get("gpu_evidence") or {}
    llama, cuda = gpu.get("llama") or {}, gpu.get("workload_cuda") or {}
    out["G_gpu"] = _check(
        None if not llama or not cuda else bool(llama.get("ok")) and cuda.get("cuinit") == 0 and cuda.get("ok") is True,
        "llama-server: every layer and the vision encoder on CUDA0, only L4T libcuda; workload cuInit 0",
        {"llama_ok": llama.get("ok"), "layers": llama.get("layers"), "workload_cuinit": cuda.get("cuinit")},
    )
    flags = tuple(((profile.get("provenance") or {}).get("llama_server") or {}).get("flags") or ())
    cache = profile.get("cache_evidence") or {}
    out["C_prompt_cache"] = _check(
        None if cache.get("verdict") in (None, "unverified") else cache["verdict"] == "disabled_verified"
        and flags == EXPECTED_LLAMA_FLAGS,
        "flags exactly " + " ".join(EXPECTED_LLAMA_FLAGS) + "; cache off by manifest, running command line, "
        "installed build, startup log and 0 steady updates",
        {"verdict": cache.get("verdict"), "missing": cache.get("missing"), "flags": list(flags)},
    )
    return out


def evaluate(
    profile: Mapping[str, Any] | None,
    *,
    run: Mapping[str, Any],
    kernel: Mapping[str, Any],
    identity: Mapping[str, Any],
    criteria_id: str = CRITERIA_ID,
) -> dict[str, Any]:
    """Every criterion, and whether the run may go to the maintainer for review (never acceptance).

    ``criteria_id`` is CRITERIA_ID for step 4, PLR_CRITERIA_ID for the step-4 PLR variant, or CANDIDATE_CRITERIA_ID
    for the D58 candidate: the same rules and thresholds, plus R's two release inputs (both variants) and the four
    THP-disable inputs (the candidate), reported under the variant's own identity."""
    if criteria_id not in (CRITERIA_ID, PLR_CRITERIA_ID, CANDIDATE_CRITERIA_ID):
        raise ValueError(f"unknown criteria identity {criteria_id!r}")
    candidate = criteria_id == CANDIDATE_CRITERIA_ID
    plr = criteria_id == PLR_CRITERIA_ID or candidate
    criteria: dict[str, dict[str, Any]] = {}
    criteria["R_valid_run"] = _check(
        bool(run.get("guard_completed") and run.get("cleanup_clear") and run.get("check9_ok")
             and run.get("drop_declared") and run.get("profile_complete") and run.get("headless")
             and run.get("no_dev_tools") and run.get("no_tracked_changes")
             and (not plr or all(run.get(key) is True for key in PLR_RUN_INPUTS))
             and (not candidate or all(run.get(key) is True for key in CANDIDATE_RUN_INPUTS))),
        "guard completed, cleanup clear, same-boot Check 9, cache drop declared, profile complete, headless, "
        "no dev tools, no tracked changes"
        + ("; PLR: all three post-load releases recorded and every release call returned 0" if plr else "")
        + ("; candidate: the workload's THP disable verified before the detector load, THP_enabled as expected at "
           "all four checkpoints, its AnonHugePages never above its value at the check, and the THP settings "
           "unchanged" if candidate else ""),
        dict(run),
    )
    if profile is None:
        for name in ("S_steady_interval", "M1_steady_pressure", "M2_run_peak", "M3_trend_swap", "T1_unique_throughput",
                     "T2_frame_age", "T3_processed_ratio", "F_face", "V1_scene_requests", "V2_scene_completion",
                     "G_gpu", "C_prompt_cache"):
            criteria[name] = _criterion(UNAVAILABLE, "profiler evidence", None)
    else:
        criteria.update(evaluate_profile(profile))
    if criteria["R_valid_run"]["status"] != PASS:  # a stopped or invalid run cannot pass the run criteria
        for name, item in criteria.items():
            if name != "R_valid_run" and item["status"] == PASS:
                item["status"] = UNAVAILABLE
    kernel_ok = kernel.get("status") == "observed"
    criteria["K_kernel"] = _check(
        None if not kernel_ok else kernel.get("oom_candidates") == 0 and kernel.get("nvmap_candidates") == 0,
        "kernel log coverage observed for the recorded boot over the run plus "
        f"{POST_RUN_WAIT_S:g} s; 0 OOM and 0 NvMap candidate lines; unavailable is never zero",
        {k: kernel.get(k) for k in ("status", "reasons", "oom_candidates", "nvmap_candidates", "records")},
    )
    criteria["I_identity"] = _check(
        None if identity.get("status") is None else identity.get("status") == "verified",
        "identity snapshot before the cache drop: files hashed, clip hash equals the recorded check 8 clip, "
        "files unchanged at start and rehashed equal at the end",
        dict(identity),
    )
    blocking = [name for name, item in criteria.items() if item["status"] != PASS]
    result = {
        "criteria_id": criteria_id,
        "criteria": criteria,
        "eligible_for_maintainer_review": not blocking,
        "blocking": blocking,
        "accepted": False,
        "acceptance": "only a separate, maintainer-approved registry commit can accept a profile (D46)",
        "scope": "demo profile (640x480 replay, 15 fps); not the guide's 1080p beta gates",
    }
    if candidate:
        result.update(procedure=CANDIDATE_PROCEDURE, admission=CANDIDATE_ADMISSION,
                      criteria_rules=f"{CRITERIA_ID} (D47 with the session 18 amendment): every rule and threshold "
                                     "unchanged; R adds the two release inputs and the four THP-disable inputs")
    elif plr:
        result.update(procedure=PLR_PROCEDURE, admission=PLR_ADMISSION,
                      criteria_rules=f"{CRITERIA_ID} (D47 with the session 18 amendment): every rule and threshold "
                                     "unchanged; R adds the two release inputs")
    return result
