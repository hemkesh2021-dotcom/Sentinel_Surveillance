from __future__ import annotations

import csv
import importlib.util
import io
import json
import subprocess
import sys
import textwrap
import threading
import weakref
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
NEW_FIELDS = {
    "Shmem": "shmem", "Unevictable": "unevictable", "Mlocked": "mlocked",
    "SUnreclaim": "s_unreclaim", "KReclaimable": "k_reclaimable", "CmaFree": "cma_free",
}
ATTRIBUTION_FIELDS = {
    "AnonPages": "anon_pages", "Mapped": "mapped", "Active(file)": "active_file", "Inactive(file)": "inactive_file",
}
LEGACY_COLUMNS = [
    "t_mono", "phase", "mem_total", "mem_free", "mem_available", "cached", "swap_total", "swap_free",
    "llama_rss", "llama_hwm", "llama_pss", "work_rss", "work_hwm", "work_pss", "pswpin", "pswpout",
]
ALLOCATOR_VALUES = {
    "allocated_bytes.all.current": 128, "allocated_bytes.all.peak": 256,
    "reserved_bytes.all.current": 512, "reserved_bytes.all.peak": 768,
}


def load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNERS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def profile():
    return load_runner("demo_profile")


@pytest.fixture
def workload():
    return load_runner("demo_workload")


def test_runner_imports_need_no_ml_or_hardware_libraries() -> None:
    script = textwrap.dedent(
        f"""
        import ctypes, importlib.abc, importlib.util, sys
        from pathlib import Path
        blocked = {{"torch", "numpy", "cv2", "tensorflow", "deepface", "ultralytics",
                   "tensorrt", "pycuda", "cuda", "gi", "onnxruntime", "jtop"}}
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] in blocked:
                    raise AssertionError("ML import attempted: " + name)
        def no_cuda(*args, **kwargs):
            raise AssertionError("hardware library load attempted")
        sys.meta_path.insert(0, Block())
        ctypes.CDLL = no_cuda
        sys.path.insert(0, {str(RUNNERS)!r})
        for name in ("demo_profile", "demo_workload", "operator_check", "gpu_alloc_probe"):
            spec = importlib.util.spec_from_file_location(name, Path({str(RUNNERS)!r}) / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        assert not blocked.intersection(sys.modules)
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_meminfo_converts_all_six_fields_from_kib_to_bytes(profile, tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    path.write_text("\n".join(f"{key}: {index} kB" for index, key in enumerate(NEW_FIELDS, start=1)))
    assert profile.read_meminfo(path) == {key: index * 1024 for index, key in enumerate(NEW_FIELDS, start=1)}


@pytest.mark.parametrize("value", ["", "8", "8 MB", "bad kB", "-1 kB"])
def test_absent_or_invalid_meminfo_is_unavailable_not_zero(profile, tmp_path: Path, value: str) -> None:
    path = tmp_path / "meminfo"
    path.write_text(f"Shmem: 0 kB\nUnevictable: {value}\n")
    assert profile.read_meminfo(path) == {"Shmem": 0}


@pytest.mark.parametrize("missing_cma", [False, True])
def test_sampler_new_fields_round_trip_with_missing_values(
    profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_cma: bool,
) -> None:
    meminfo = tmp_path / "meminfo"
    lines = [
        "MemTotal: 8000000 kB", "MemFree: 4000000 kB", "MemAvailable: 6000000 kB",
        "Cached: 2000000 kB", "SwapTotal: 0 kB", "SwapFree: 0 kB",
    ]
    lines += [f"{key}: {index} kB" for index, key in enumerate(NEW_FIELDS) if key != "CmaFree" or not missing_cma]
    meminfo.write_text("\n".join(lines))
    read_meminfo = profile.read_meminfo
    monkeypatch.setattr(profile, "read_meminfo", lambda: read_meminfo(meminfo))
    monkeypatch.setattr(profile, "read_swap_counters", lambda: {"pswpin": 0, "pswpout": 2})
    monkeypatch.setattr(profile.time, "monotonic", lambda: 123.456)
    path = tmp_path / "memory.csv"
    sampler = profile.Sampler(path, lambda: pytest.fail("unexpected memory-floor callback"))
    sampler.phase = "steady"
    monkeypatch.setattr(sampler._halt, "wait", lambda timeout: sampler._halt.set())
    sampler.run()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    sample = profile.load_samples(path)[0]
    assert (sample["t"], sample["phase"]) == (123.456, "steady")
    for index, (key, column) in enumerate(NEW_FIELDS.items()):
        expected = None if key == "CmaFree" and missing_cma else index * 1024
        assert sample[column] == expected
        assert rows[0][column] == ("" if expected is None else str(expected))
    assert sample["used"] == 2_000_000 * 1024
    assert sample["swap_used"] == 0


def test_attribution_columns_follow_every_existing_column_in_place(profile) -> None:
    before_session_39 = [*LEGACY_COLUMNS[:8], *NEW_FIELDS.values(), *LEGACY_COLUMNS[8:]]
    assert profile.Sampler.COLUMNS == [*before_session_39, *ATTRIBUTION_FIELDS.values()]
    assert profile.MEMINFO_KEYS == (*profile.BASE_MEMINFO_KEYS, *ATTRIBUTION_FIELDS)
    assert profile.BASE_MEMINFO_KEYS[:6] == ("MemTotal", "MemFree", "MemAvailable", "Cached", "SwapTotal", "SwapFree")


@pytest.mark.parametrize("present", [True, False])
def test_sampler_records_attribution_fields_or_marks_them_unavailable(
    profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, present: bool,
) -> None:
    meminfo = tmp_path / "meminfo"
    lines = ["MemTotal: 8000000 kB", "MemFree: 4000000 kB", "MemAvailable: 6000000 kB", "Cached: 2000000 kB",
             "SwapTotal: 0 kB", "SwapFree: 0 kB", "Active(anon): 11 kB", "Inactive(anon): 12 kB"]
    if present:
        lines += ["AnonPages: 100 kB", "Mapped: 200 kB", "Active(file): 300 kB", "Inactive(file): 400 kB"]
    meminfo.write_text("\n".join(lines))
    read_meminfo = profile.read_meminfo
    assert ("Active(anon)" in read_meminfo(meminfo)) is False  # only the named fields are kept
    monkeypatch.setattr(profile, "read_meminfo", lambda: read_meminfo(meminfo))
    monkeypatch.setattr(profile, "read_swap_counters", lambda: {"pswpin": 0, "pswpout": 0})
    path = tmp_path / "memory.csv"
    sampler = profile.Sampler(path, lambda: pytest.fail("unexpected memory-floor callback"))
    monkeypatch.setattr(sampler._halt, "wait", lambda timeout: sampler._halt.set())
    sampler.run()
    with path.open(newline="") as handle:
        header, row = list(csv.reader(handle))
    assert header == profile.Sampler.COLUMNS and len(row) == len(header)
    values = dict(zip(header, row))
    sample = profile.load_samples(path)[0]
    for index, column in enumerate(ATTRIBUTION_FIELDS.values(), start=1):
        expected = index * 100 * 1024 if present else None
        assert sample[column] == expected and values[column] == ("" if expected is None else str(expected))
    assert sample["mem_free"] == 4_000_000 * 1024 and sample["used"] == 2_000_000 * 1024
    assert values["pswpout"] == "0" and values["cma_free"] == ""


def test_legacy_csv_marks_attribution_fields_unavailable(profile, tmp_path: Path) -> None:
    path = tmp_path / "memory.csv"
    write_legacy_csv(path)
    assert all(sample[column] is None for sample in profile.load_samples(path) for column in ATTRIBUTION_FIELDS.values())


def write_legacy_csv(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEGACY_COLUMNS)
        writer.writeheader()
        for timestamp, phase in ((1, "baseline"), (2, "steady"), (3, "steady")):
            writer.writerow({
                "t_mono": timestamp, "phase": phase, "mem_total": 8_000_000_000,
                "mem_free": 4_000_000_000, "mem_available": 7_000_000_000, "cached": 2_000_000_000,
                "swap_total": 0, "swap_free": 0, "pswpin": 0, "pswpout": 0,
            })


def test_legacy_csv_keeps_original_accounting_and_marks_new_fields_unavailable(profile, tmp_path: Path) -> None:
    path = tmp_path / "memory.csv"
    write_legacy_csv(path)
    samples = profile.load_samples(path)
    assert len(samples) == 3
    assert samples[0]["t"] == 1.0
    for sample in samples:
        assert sample["used"] == 1_000_000_000
        assert sample["swap_used"] == 0
        assert all(sample[column] is None for column in NEW_FIELDS.values())


def test_allocator_samples_use_injected_monotonic_time_values_and_lifetime_peaks(workload, clock) -> None:
    records = []
    readings = dict(ALLOCATOR_VALUES)
    start = clock.mono().ns / 1e9
    sampler = workload.AllocatorSampler(
        lambda: readings, clock=lambda: clock.mono().ns / 1e9,
        emit=lambda name, **fields: records.append({"event": name, **fields}),
    )
    assert sampler.sample()
    assert records[0] == {
        "event": "torch_allocator", "t_mono": start, "phase": "detector_settle", "clock": "time.monotonic",
        "units": "bytes", "device": 0, "peak_scope": "allocator_lifetime", "status": "observed", "error": None,
        "allocated_bytes": 128, "allocated_peak_bytes": 256, "reserved_bytes": 512, "reserved_peak_bytes": 768,
    }
    clock.step_utc(timedelta(days=1))
    assert not sampler.sample()
    clock.advance(1)
    sampler.phase = "steady"
    readings["allocated_bytes.all.current"] = 64
    assert sampler.sample()
    assert records[-1]["t_mono"] == start + 1.0
    assert records[-1]["phase"] == "steady"
    assert records[-1]["allocated_bytes"] == 64
    assert records[-1]["allocated_peak_bytes"] == 256
    assert records[-1]["reserved_peak_bytes"] == 768


def test_runtime_provider_reads_only_device_zero_allocator_statistics(workload, monkeypatch) -> None:
    calls = []

    def memory_stats(device):
        calls.append(device)
        return ALLOCATOR_VALUES

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(memory_stats=memory_stats)))
    assert workload.torch_allocator_stats() is ALLOCATOR_VALUES
    assert calls == [0]


def test_allocator_source_timestamp_survives_orchestrator_relay(profile, workload, tmp_path, monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(workload.sys, "stdout", output)
    sampler = workload.AllocatorSampler(lambda: ALLOCATOR_VALUES, clock=lambda: 123.456, emit=workload.event)
    sampler.sample()
    monkeypatch.setattr(profile.time, "monotonic", lambda: 200.0)
    events = profile.Events(tmp_path / "events.jsonl", 200.0)
    try:
        profile.pump_workload(
            SimpleNamespace(stdout=io.StringIO(output.getvalue())), tmp_path, events,
            lambda *args, **kwargs: pytest.fail("allocator event changed orchestrator phase"),
        )
    finally:
        events.close()
    record = json.loads((tmp_path / "events.jsonl").read_text())
    assert record["t_mono"] == 123.456
    assert record["clock"] == "time.monotonic"
    assert record["units"] == "bytes"
    assert record["allocated_bytes"] == 128
    assert record["allocated_peak_bytes"] == 256
    assert record["reserved_peak_bytes"] == 768


@pytest.mark.parametrize("value", [None, -1, True, "256", 256.0])
def test_missing_or_invalid_allocator_values_stay_unavailable(workload, value) -> None:
    records = []
    sampler = workload.AllocatorSampler(
        lambda: {"allocated_bytes.all.current": 0, "allocated_bytes.all.peak": value},
        clock=lambda: 5.0, emit=lambda name, **fields: records.append(fields),
    )
    sampler.sample()
    assert records[0]["allocated_bytes"] == 0
    assert records[0]["allocated_peak_bytes"] is None
    assert records[0]["reserved_bytes"] is None
    assert records[0]["reserved_peak_bytes"] is None
    assert records[0]["status"] == "unavailable"


def test_allocator_provider_failure_is_unavailable_and_contains_no_exception_text(workload) -> None:
    records = []

    def fail():
        raise RuntimeError("private provider detail")

    sampler = workload.AllocatorSampler(fail, clock=lambda: 5.0, emit=lambda name, **fields: records.append(fields))
    sampler.sample()
    assert records[0]["error"] == "RuntimeError"
    assert records[0]["status"] == "unavailable"
    assert all(records[0][field] is None for field in workload.ALLOCATOR_FIELDS)
    assert "private provider detail" not in json.dumps(records)


def test_allocator_cadence_skips_missed_ticks_retains_no_history_and_stops(workload, clock) -> None:
    count = 0
    reading_ref = None

    class Reading(dict):
        pass

    def provider():
        nonlocal reading_ref
        reading = Reading(ALLOCATOR_VALUES)
        reading_ref = weakref.ref(reading)
        return reading

    def emit(name, **fields):
        nonlocal count
        count += 1
        assert name == "torch_allocator"
        assert fields["allocated_bytes"] == 128

    sampler = workload.AllocatorSampler(provider, clock=lambda: clock.mono().ns / 1e9, emit=emit)
    for _ in range(4000):
        sampler.sample()
        assert reading_ref() is None
        clock.advance(0.25)
    assert count == 1000
    clock.advance(100)
    assert sampler.sample()
    assert not sampler.sample()
    assert count == 1001
    sampler.stop()
    clock.advance(100)
    assert not sampler.sample()
    assert count == 1001


def test_allocator_thread_stops_during_its_wait(workload) -> None:
    sampled = threading.Event()
    sampler = workload.AllocatorSampler(lambda: ALLOCATOR_VALUES, emit=lambda name, **fields: sampled.set())
    sampler.start()
    try:
        assert sampled.wait(timeout=2)
    finally:
        sampler.stop()
    assert not sampler.is_alive()


@pytest.mark.parametrize("load_failure", [False, True])
def test_workload_main_wires_allocator_and_stops_it_on_success_or_failure(workload, monkeypatch, load_failure) -> None:
    calls = []
    detector, face = object(), object()

    class FakeSampler:
        def __init__(self, provider):
            assert provider is workload.torch_allocator_stats
            self.phase = "detector_settle"

        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")

    def load_face():
        if load_failure:
            raise RuntimeError("fake load failure")
        return face

    def run(model, deepface, args, stats, allocator):
        assert model is detector and deepface is face
        assert allocator.phase == "face_settle"
        calls.append("workload")

    monkeypatch.setattr(workload, "check_cuda_driver", lambda: True)
    monkeypatch.setattr(workload, "load_detector", lambda engine: detector)
    monkeypatch.setattr(workload, "load_face", load_face)
    monkeypatch.setattr(workload, "AllocatorSampler", FakeSampler)
    monkeypatch.setattr(workload, "run_workload", run)
    monkeypatch.setattr(workload, "event", lambda *args, **fields: None)
    monkeypatch.setattr(workload.time, "sleep", lambda seconds: None)
    arguments = ["--engine", "fake.engine", "--port", "18081"]
    if load_failure:
        with pytest.raises(RuntimeError, match="fake load failure"):
            workload.main(arguments)
        assert calls == ["start", "stop"]
    else:
        assert workload.main(arguments) == 0
        assert calls == ["start", "workload", "stop"]


def test_summary_streams_allocator_history_and_preserves_legacy_device_accounting(profile, tmp_path: Path) -> None:
    write_legacy_csv(tmp_path / "memory.csv")
    (tmp_path / "tegrastats.log").write_text("")
    manifest = {
        "run_id": "synthetic-test", "input": {"synthetic": True, "fps": 15},
        "parameters": {"face_hz": 1, "scene_interval_s": 4}, "repository": {"commit": "fake", "tracked_changes": False},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with (tmp_path / "events.jsonl").open("w") as handle:
        for index in range(4000):
            handle.write(json.dumps({"event": "torch_allocator", "t_mono": index, "allocated_bytes": 999_000_000_000}) + "\n")
            handle.write(json.dumps({"event": "run_end", "status": f"fake-{index}"}) + "\n")
    events = profile.load_latest_events(tmp_path / "events.jsonl")
    assert events == {"run_end": {"event": "run_end", "status": "fake-3999"}}
    summary = profile.summarize(tmp_path)
    result = json.loads((tmp_path / "profile.json").read_text())
    assert result["status"] == "fake-3999"
    assert result["combined"]["steady_median_bytes"] == 1_000_000_000
    assert result["combined"]["run_peak_bytes"] == 1_000_000_000
    assert "do not measure total GPU/device usage or prove unload" in summary


B8932_CACHE_LOG = [  # b8932 SRV_WRN formats (tools/server/server-context.cpp, server-task.cpp)
    "srv    load_model: prompt cache is enabled, size limit: 8192 MiB\n",
    "srv    load_model: use `--cache-ram 0` to disable the prompt cache\n",
    "srv  update_slots: updating prompt cache\n",
    "srv          load:  - looking for better prompt, base f_keep = -1.000, sim = 0.000\n",
    "srv        update:  - cache state: 3 prompts, 18.750 MiB (limits: 8192.000 MiB, 2048 tokens, 89478 est)\n",
    "srv        update:    - prompt 0x55d0c0ffee: private 450 tokens, checkpoints:  0,     6.250 MiB\n",
    "srv         alloc:  - prompt is already in the cache, skipping\n",
    "srv        update:  - cache size limit reached, removing oldest entry (size = 6.250 MiB)\n",
    "srv         alloc: failed to allocate memory for prompt cache state: private std::bad_alloc\n",
    "srv    load_model: prompt cache is disabled - use `--cache-ram N` to enable it\n",
    "srv    load_model: prompt cache is enabled, size limit: no limit\n",
    "secret://user:password@camera/ private description text\n",
]


@pytest.mark.parametrize("given", [None, 0, 512])
def test_cache_ram_flag_is_passed_only_when_given(profile, monkeypatch, tmp_path: Path, given) -> None:
    arguments = [] if given is None else ["--llama-cache-ram", str(given)]
    args = profile.parse_args(arguments)
    launched = []
    monkeypatch.setattr(profile.subprocess, "Popen", lambda argv, **kwargs: launched.append(argv) or SimpleNamespace())
    profile.start_llama(args, tmp_path)[1].close()
    argv = launched[0]
    assert argv[-len(profile.LLAMA_FLAGS) - (0 if given is None else 2):][:len(profile.LLAMA_FLAGS)] == profile.LLAMA_FLAGS
    if given is None:
        assert "--cache-ram" not in argv and args.llama_cache_ram is None
    else:
        assert argv[-2:] == ["--cache-ram", str(given)] and args.llama_cache_ram == given
    assert profile.llama_flags(args) == argv[argv.index("--n-gpu-layers"):]


@pytest.mark.parametrize("value", ["-1", "no-limit"])
def test_cache_ram_refuses_unlimited_or_malformed_values(profile, value) -> None:
    with pytest.raises(SystemExit):
        profile.parse_args(["--llama-cache-ram", value])


def test_scene_only_needs_only_the_scene_model_and_refuses_a_clip(profile, monkeypatch, tmp_path: Path) -> None:
    args = profile.parse_args(["--scene-only"])
    assert set(profile.model_files(args)) == {"llm", "mmproj"}
    assert "engine" in profile.model_files(profile.parse_args([]))
    launched = []
    monkeypatch.setattr(profile.subprocess, "Popen", lambda argv, **kwargs: launched.append(argv))
    profile.start_workload(args, tmp_path)
    assert launched[0][-1] == "--scene-only" and "--clip" not in launched[0]
    monkeypatch.setattr(profile, "scan_processes", lambda: [])
    monkeypatch.setattr(profile, "command_output", lambda argv, env=None: "inactive")
    monkeypatch.setattr(profile, "port_in_use", lambda port: False)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    args = profile.parse_args(["--scene-only", "--clip", str(clip)])
    problems, _ = profile.preconditions(args)
    assert any("--scene-only" in problem for problem in problems)


def test_prompt_cache_lines_keep_numbers_and_fixed_labels_only(profile) -> None:
    server = SimpleNamespace(stdout=io.BytesIO("".join(B8932_CACHE_LOG).encode()))
    output = io.BytesIO()
    ticks = iter(range(100, 200))
    profile.pump_llama_diagnostics(server, output, clock=lambda: float(next(ticks)))
    saved = output.getvalue().decode().splitlines()
    stamps = [line.split(" ", 1)[0] for line in saved]
    assert [line.split(" ", 1)[1] for line in saved] == [
        "prompt cache is enabled, size limit: 8192 MiB",
        "prompt cache state: 3 prompts, 18.750 MiB (limits: 8192.000 MiB, 2048 tokens, 89478 est)",
        "prompt cache event: duplicate_skipped",
        "prompt cache event: evicted",
        "prompt cache event: allocation_failed",
        "prompt cache is disabled",
        "prompt cache is enabled, size limit: no limit",
    ]
    assert stamps == ["t_mono=100.000", "t_mono=104.000", "t_mono=106.000", "t_mono=107.000",
                      "t_mono=108.000", "t_mono=109.000", "t_mono=110.000"]
    text = output.getvalue().decode()
    assert "private" not in text and "secret" not in text and "0x55d0c0ffee" not in text


@pytest.mark.parametrize("sanitized", [True, False])
def test_prompt_cache_summary_counts_updates_and_the_steady_window(profile, tmp_path: Path, sanitized) -> None:
    log = tmp_path / "llama-server.log"
    if sanitized:
        log.write_text("\n".join([
            "offloaded 17/17 layers to GPU",
            "t_mono=5.000 prompt cache is enabled, size limit: 8192 MiB",
            *(f"t_mono={t:.3f} prompt cache state: {n} prompts, {6.25 * n:.3f} MiB "
              f"(limits: 8192.000 MiB, 2048 tokens, 89478 est)" for n, t in enumerate((8.0, 12.0, 16.0, 24.0), 1)),
            "t_mono=17.000 prompt cache event: duplicate_skipped",
        ]) + "\n")
    else:
        log.write_text("".join(B8932_CACHE_LOG[:9]))
    result = profile.prompt_cache_summary(log, (10.0, 20.0))
    assert result["startup"] == {"enabled": True, "limit_mib": 8192, "t_mono": 5.0 if sanitized else None}
    if sanitized:
        assert result["state_updates"] == 4 and result["steady_state_updates"] == 2
        assert (result["steady_first"]["prompts"], result["steady_last"]["prompts"]) == (2, 3)
        assert result["last"] == {"t_mono": 24.0, "prompts": 4, "size_mib": 25.0, "limit_mib": 8192.0,
                                  "limit_tokens": 2048, "estimated_tokens": 89478}
        assert result["max_size_mib"] == 25.0 and result["duplicate_skipped"] == 1
    else:  # a raw log has no receipt times, so nothing is attributed to the steady phase
        assert result["state_updates"] == 1 and result["steady_state_updates"] is None
        assert (result["duplicate_skipped"], result["evicted"], result["allocation_failed"]) == (1, 1, 1)
    legacy = tmp_path / "legacy.log"
    legacy.write_text("offloaded 17/17 layers to GPU\n")
    assert profile.prompt_cache_summary(legacy, None) is None
    assert profile.prompt_cache_summary(tmp_path / "absent.log", None) is None


def test_scene_only_summary_reports_trend_cache_and_progress(profile, tmp_path: Path) -> None:
    manifest = {
        "run_id": "synthetic-s1", "input": {"synthetic": True, "fps": 15},
        "parameters": {"face_hz": 1, "scene_interval_s": 4, "scene_only": True},
        "llama_server": {"cache_ram_mib": 0}, "repository": {"commit": "fake", "tracked_changes": False},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with (tmp_path / "memory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEGACY_COLUMNS)
        writer.writeheader()
        rows = [(0, "baseline", 0), (1, "llama_load", 0), (2, "llama_settle", 0), (3, "warmup", 0)]
        rows += [(10 + second, "steady", second) for second in range(61)]
        for timestamp, phase, step in rows:
            writer.writerow({
                "t_mono": timestamp, "phase": phase, "mem_total": 8_000_000_000, "mem_free": 5_000_000_000,
                "mem_available": 6_000_000_000 - step * 1_000_000, "cached": 1_000_000_000, "swap_total": 0,
                "swap_free": 0, "llama_pss": 500_000_000 + step * 2_000_000, "pswpin": 0, "pswpout": 0,
            })
    (tmp_path / "tegrastats.log").write_text("")
    (tmp_path / "llama-server.log").write_text(
        "t_mono=1.500 prompt cache is disabled\n" "t_mono=9.000 prompt cache state: 0 prompts, 0.000 MiB "
        "(limits: 0.000 MiB, 2048 tokens, 2048 est)\n"
    )
    progress = {"event": "scene_progress", "phase": "steady", "requests": 53, "completed": 53, "valid_reports": 53,
                "invalid_reports": 0, "errors": {"http": 0, "timeout": 0, "other": 0},
                "finish_reasons": {"stop": 53, "length": 0, "other": 0, "missing": 0}, "synthetic_images_issued": 53}
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(item) for item in (
        progress, {"event": "workload_stats", "seconds": 60.0, "input": "synthetic noise, distinct per request",
                   "scene": {"completed": 15}, "synthetic_images_issued": 53},
        {"event": "run_end", "status": "complete"},
    )) + "\n")
    summary = profile.summarize(tmp_path)
    result = json.loads((tmp_path / "profile.json").read_text())
    trend = result["steady_trend"]
    assert trend["seconds"] == 60.0
    assert trend["used"]["slope_bytes_per_min"] == 60_000_000 and trend["llama_pss"]["slope_bytes_per_min"] == 120_000_000
    assert (trend["used"]["first"], trend["used"]["last"]) == (2_000_000_000, 2_060_000_000)
    assert trend["work_pss"]["n"] == 0 and trend["work_pss"]["slope_bytes_per_min"] is None
    assert {k: v for k, v in result["prompt_cache"]["startup"].items() if k != "t_mono"} == {
        "enabled": False, "limit_mib": None}
    assert result["scene_progress_last"]["requests"] == 53
    assert result["provenance"]["parameters"]["scene_only"] is True
    assert "scene-only, one distinct synthetic noise image per request" in summary
    assert "--cache-ram 0 MiB" in summary
    assert "detector: not run (scene-only)" in summary and "detector and face: not run (scene-only)" in summary
    assert "requests 53, completed 53" in summary and "startup: disabled" in summary


def test_sanitized_pump_keeps_scene_progress_counters_but_no_text(profile, tmp_path: Path) -> None:
    record = {"event": "scene_progress", "t_mono": 1.5, "phase": "steady", "requests": 2, "completed": 1,
              "valid_reports": 1, "invalid_reports": 0, "errors": {"http": 1, "timeout": 0, "other": 0},
              "finish_reasons": {"stop": 1, "length": 0, "other": 0, "missing": 0},
              "rejected_reports_by_reason": {"truncated": 0}, "latency_ms": 812.4, "prompt_tokens": 305,
              "completion_tokens": 120, "synthetic_images_issued": 2, "summary": "private model text"}
    events = []
    sink = SimpleNamespace(add=lambda source, name, **fields: events.append((name, fields)))
    proc = SimpleNamespace(stdout=io.StringIO("@@EVENT " + json.dumps(record) + "\n"))
    profile.pump_workload(proc, tmp_path, sink, lambda *args, **kwargs: None, sanitized=True)
    expected = {key: value for key, value in record.items() if key not in ("event", "summary")}
    assert events == [("scene_progress", expected)]
