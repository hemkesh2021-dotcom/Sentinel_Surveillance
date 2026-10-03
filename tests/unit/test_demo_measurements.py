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
