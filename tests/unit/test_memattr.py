"""MA1 (D55, opt-in diagnostic): the workload's allocator and garbage-collector counters, its mapping
categories, their sanitizing and relay, the command and the summary record.

Portable: no GPU, model, camera or journal. The allocator and smaps tests read this test process's own
counters and /proc/self/smaps (skipped where the C library or /proc lacks them); everything about the
workload process is synthetic. Nothing here collects garbage in the workload, trims an allocator or changes a
setting; the one test that calls gc.collect() does so in the test process to produce a full collection.
"""

from __future__ import annotations

import gc
import importlib
import io
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
MIB = 1 << 20
H = 64 * MIB  # glibc's non-main arena heap size and alignment on 64-bit


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return SimpleNamespace(profile=importlib.import_module("demo_profile"),
                           workload=importlib.import_module("demo_workload"))


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class ReadOnlyGc:
    """The collector interface the recorder and sampler may use; anything that would change it fails the test."""

    def __init__(self) -> None:
        self.callbacks: list = []

    def get_threshold(self):
        return (700, 10, 10)

    def isenabled(self):
        return True

    def get_freeze_count(self):
        return 0

    def get_count(self):
        return (5, 1, 0)

    def __getattr__(self, name):  # collect, disable, enable, set_threshold, freeze, unfreeze, ...
        raise AssertionError(f"the diagnostic must not call gc.{name}")


def collect_events():
    emitted = []
    return emitted, lambda kind, **fields: emitted.append({"event": kind, **fields})


# ---------------------------------------------------------------- allocator counters (workload)


def test_the_reported_allocator_fields_are_the_same_on_both_sides(mods) -> None:
    w, p = mods.workload, mods.profile
    assert w.MALLINFO2_REPORTED == p.MALLINFO2_REPORTED and "usmblks" not in w.MALLINFO2_REPORTED
    assert w.MALLINFO2_FIELDS[:2] == ("arena", "ordblks") and len(w.MALLINFO2_FIELDS) == 10
    assert w.MEMATTR_FULL_GC_PER_SAMPLE == p.MEMATTR_FULL_GC_PER_SAMPLE


def test_mallinfo2_sums_the_arenas_and_counts_large_allocations_as_mmapped(mods) -> None:
    read = mods.workload.glibc_mallinfo2()
    if read is None:
        pytest.skip("this C library has no mallinfo2 (glibc 2.33 or later)")
    before = read()
    assert set(before) == set(mods.workload.MALLINFO2_REPORTED) and all(type(v) is int for v in before.values())
    assert before["uordblks"] + before["fordblks"] == before["arena"]  # in use + kept free = obtained from the system
    block = bytearray(64 * MIB)  # above glibc's largest mmap threshold (32 MiB): always its own mapping
    during = read()
    assert during["hblkhd"] - before["hblkhd"] >= 64 * MIB and during["hblks"] > before["hblks"]
    del block
    assert read()["hblkhd"] < during["hblkhd"]
    assert mods.workload.glibc_version()  # e.g. "2.39"


@pytest.mark.parametrize("error", [AttributeError, OSError])
def test_a_c_library_without_mallinfo2_is_unsupported_and_never_reads_as_zero(mods, error) -> None:
    w = mods.workload

    def load(name):
        raise error("no such symbol")

    assert w.glibc_mallinfo2(load) is None and w.glibc_version(load) is None
    emitted, emit = collect_events()
    sampler = w.ProcessMemorySampler(lambda: "steady", mallinfo=None, module=ReadOnlyGc(), clock=FakeClock(),
                                     emit=emit)
    assert sampler.sample()
    (record,) = emitted
    assert record["malloc"] is None and record["malloc_status"] == "unsupported" and record["malloc_call_us"] is None
    assert record["status"] == "partial" and type(record["pymalloc_blocks"]) is int


def test_failed_readings_are_unavailable_with_their_class_never_zero(mods) -> None:
    w = mods.workload
    emitted, emit = collect_events()

    def broken():
        raise RuntimeError("boom")

    def blocks():
        raise ValueError("no")

    sampler = w.ProcessMemorySampler(lambda: "warmup", mallinfo=broken, blocks=blocks, module=ReadOnlyGc(),
                                     clock=FakeClock(), emit=emit)
    sampler.sample()
    partial = w.ProcessMemorySampler(lambda: "warmup", mallinfo=lambda: {"arena": 5, "uordblks": -1},
                                     module=ReadOnlyGc(), clock=FakeClock(), emit=emit)
    partial.sample()
    failed, incomplete = emitted
    assert failed["status"] == "unavailable" and failed["malloc_error"] == "RuntimeError"
    assert failed["malloc"] is None and failed["pymalloc_blocks"] is None
    assert incomplete["malloc_status"] == "unavailable" and incomplete["malloc"]["arena"] == 5
    assert incomplete["malloc"]["uordblks"] is None and incomplete["malloc"]["fordblks"] is None


def test_the_sampler_streams_once_per_interval_with_the_current_phase(mods) -> None:
    w = mods.workload
    emitted, emit = collect_events()
    clock, phase = FakeClock(), {"name": "warmup"}
    sampler = w.ProcessMemorySampler(lambda: phase["name"], mallinfo=lambda: dict.fromkeys(w.MALLINFO2_REPORTED, 7),
                                     blocks=lambda: 11, module=ReadOnlyGc(), clock=clock, emit=emit)
    assert sampler.sample() and not sampler.sample()  # nothing more within the interval
    clock.now += 1.0
    phase["name"] = "steady"
    assert sampler.sample()
    first, second = emitted
    assert (first["phase"], second["phase"], second["t_mono"]) == ("warmup", "steady", 101.0)
    assert first["status"] == "observed" and first["malloc"] == dict.fromkeys(w.MALLINFO2_REPORTED, 7)
    assert first["gc_counts"] == [5, 1, 0] and first["gc_collections"] == [0, 0, 0] and first["gc_full"] == []
    assert first["gc_enabled"] is True and first["malloc_call_us"] >= 0 and first["sample_us"] >= 0


# ---------------------------------------------------------------- garbage-collector events (workload)


def test_the_recorder_counts_every_collection_and_times_full_ones_within_its_bound(mods) -> None:
    w = mods.workload
    module, clock = ReadOnlyGc(), FakeClock(50.0)
    recorder = w.GcRecorder(clock=clock, module=module, limit=2)
    recorder.install()
    recorder.install()
    assert module.callbacks == [recorder]
    for generation, (start, end) in ((0, (50.0, 50.001)), (2, (51.0, 51.25)), (1, (52.0, 52.01)),
                                     (2, (53.0, 53.5)), (2, (54.0, 54.1))):
        clock.now = start
        recorder("start", {"generation": generation})
        clock.now = end
        recorder("stop", {"generation": generation, "collected": 3, "uncollectable": 0})
    assert recorder.collections == [1, 1, 3] and recorder.collected == [3, 3, 9]
    assert recorder.seconds[2] == pytest.approx(0.85)
    assert recorder.drain() == [[51.0, 250.0, 3, 0], [53.0, 500.0, 3, 0]] and recorder.full_dropped == 1
    assert recorder.drain() == []
    recorder("stop", {"generation": 2})  # a stop without its start, without counts: None, never a made-up value
    assert recorder.drain() == [[None, None, None, None]] and recorder.collections[2] == 4
    recorder.remove()
    assert module.callbacks == []


def test_a_real_full_collection_is_recorded_and_the_collector_is_left_as_it_was(mods) -> None:
    w = mods.workload
    callbacks, thresholds, enabled = list(gc.callbacks), gc.get_threshold(), gc.isenabled()
    emitted, emit = collect_events()
    sampler = w.ProcessMemorySampler(lambda: "steady", mallinfo=w.glibc_mallinfo2(), emit=emit,
                                     libc_version=w.glibc_version())
    sampler.start()
    try:
        gc.collect(2)  # the test's own full collection
    finally:
        sampler.stop()
    assert gc.callbacks == callbacks and gc.get_threshold() == thresholds and gc.isenabled() == enabled
    config = emitted[0]
    assert config["event"] == "memattr_config" and config["gc_thresholds"] == list(thresholds)
    assert config["mallinfo2"] is (w.glibc_mallinfo2() is not None)
    full = [item for record in emitted[1:] for item in record["gc_full"]]
    full += sampler._recorder.drain()  # any collection after the last sample
    assert any(start is not None and duration >= 0 and type(collected) is int
               for start, duration, collected, _ in full)


# ---------------------------------------------------------------- workload command line and order


@pytest.mark.parametrize("argv", [
    ["--scene-only", "--port", "1", "--memory-attribution"],
    ["--engine", "e", "--port", "1", "--mr1-release-check", "--memory-attribution"],
])
def test_the_workload_refuses_attribution_without_a_combined_steady_phase(mods, argv) -> None:
    with pytest.raises(SystemExit):
        mods.workload.main(argv)


@pytest.mark.parametrize("flags", [["--post-load-release"], ["--post-load-release", "--memory-attribution"]])
def test_attribution_wraps_the_unchanged_plr_order_and_only_reads(mods, monkeypatch, flags) -> None:
    workload = mods.workload
    calls = []

    class FakeAllocator:
        def __init__(self, provider):
            self.phase = "detector_settle"

        def start(self):
            calls.append("allocator start")

        def stop(self):
            calls.append("allocator stop")

    class FakeMemattr:
        def __init__(self, phase, **kwargs):
            calls.append(f"memattr {phase()}")

        def start(self):
            calls.append("memattr start")

        def stop(self):
            calls.append("memattr stop")

    monkeypatch.setattr(workload, "check_cuda_driver", lambda: True)
    monkeypatch.setattr(workload, "load_detector", lambda engine: calls.append("detector") or "model")
    monkeypatch.setattr(workload, "load_face", lambda: calls.append("face") or "deepface")
    monkeypatch.setattr(workload, "AllocatorSampler", FakeAllocator)
    monkeypatch.setattr(workload, "ProcessMemorySampler", FakeMemattr)
    monkeypatch.setattr(workload, "run_workload", lambda model, face, args, stats, allocator: calls.append(
        f"workload {args.warmup_s:g}/{args.steady_s:g} {args.face_hz:g} Hz {args.scene_interval_s:g} s"))
    monkeypatch.setattr(workload, "mr1_checkpoint", lambda stage: calls.append(f"checkpoint {stage}"))
    monkeypatch.setattr(workload, "event", lambda kind, **fields: None)
    monkeypatch.setattr(workload.time, "sleep", lambda seconds: None)
    assert workload.main(["--engine", "e", "--port", "1", *flags]) == 0
    plr = ["detector", "allocator start", "checkpoint detector", "face", "checkpoint face",
           "workload 120/600 1 Hz 4 s", "allocator stop"]
    if "--memory-attribution" in flags:
        assert calls == [*plr[:2], "memattr detector_settle", "memattr start", *plr[2:-1], "memattr stop", plr[-1]]
    else:
        assert calls == plr


# ---------------------------------------------------------------- mapping categories (profiler)


def vma(start: int, end: int, perms: str, path: str = "", rss_kb: int = 4, *, skip: str | None = None,
        bad: str | None = None) -> str:
    lines = [f"{start:x}-{end:x} {perms} 00000000 00:00 0" + (f"                          {path}" if path else "")]
    fields = {"Size": (end - start) // 1024, "KernelPageSize": 4, "Rss": rss_kb, "Pss": rss_kb, "Shared_Clean": 0,
              "Private_Dirty": rss_kb, "Anonymous": 0 if path else rss_kb, "Swap": 0, "SwapPss": 0, "Locked": 0}
    for name, value in fields.items():
        if name == skip:
            continue
        lines.append(f"{name}:{' ' * (16 - len(name))}{'x' if name == bad else value} kB")
    lines.append("VmFlags: rd wr mr mw me ac")
    return "\n".join(lines) + "\n"


BASE = 0x7F00_0000_0000  # a multiple of 64 MiB

SMAPS = "".join([
    vma(0xAAAA_0000_0000, 0xAAAA_0000_1000, "r-xp", "/home/someone/private/python3.10", 4),
    vma(0xAAAA_0100_0000, 0xAAAA_0200_0000, "rw-p", "[heap]", 4096),
    vma(BASE, BASE + 8 * MIB, "rw-p", "", 2048),                    # arena heap: used prefix ...
    vma(BASE + 8 * MIB, BASE + H, "---p", "", 0),                   # ... and its no-access remainder
    vma(BASE + 2 * H, BASE + 3 * H, "rw-p", "", 65536),             # a fully grown arena heap
    vma(BASE + 3 * H + 4096, BASE + 3 * H + 8 * MIB, "rw-p", "", 100),  # unaligned anonymous: other
    vma(BASE + 4 * H, BASE + 4 * H + MIB, "rw-p", "", 8),          # aligned, but its tail ends off a boundary
    vma(BASE + 4 * H + MIB, BASE + 4 * H + 3 * MIB, "---p", "", 0),
    vma(BASE + 7 * H - MIB, BASE + 7 * H + 4 * MIB, "rw-p", "", 512),  # a heap merged with 1 MiB below it ...
    vma(BASE + 7 * H + 4 * MIB, BASE + 8 * H, "---p", "", 0),       # ... and its no-access remainder
    vma(BASE + 8 * H + MIB, BASE + 8 * H + 2 * MIB, "rw-p", "", 16),  # block start (8H) below it: not a heap
    vma(BASE + 8 * H + 2 * MIB, BASE + 9 * H, "---p", "", 0),
    vma(BASE + 5 * H, BASE + 5 * H + 4096, "rw-s", "/dev/nvmap", 0),
    vma(BASE + 5 * H + MIB, BASE + 5 * H + 2 * MIB, "rw-s", "/memfd:secret-name (deleted)", 16),
    vma(BASE + 5 * H + 3 * MIB, BASE + 5 * H + 4 * MIB, "rw-s", "/dev/zero (deleted)", 4),
    vma(BASE + 5 * H + 5 * MIB, BASE + 5 * H + 6 * MIB, "rw-s", "/dev/shm/private-frame", 8),
    vma(0xFFFF_F000_0000, 0xFFFF_F002_1000, "rw-p", "[stack]", 132),
    vma(0xFFFF_F100_0000, 0xFFFF_F100_2000, "r-xp", "[vdso]", 4),
])


def test_mappings_are_summed_by_category_and_no_path_survives(mods) -> None:
    result = mods.profile.parse_smaps(SMAPS)
    c = result["categories"]
    assert result["vmas"] == 18 and set(c) == set(mods.profile.MAPS_CATEGORIES)
    assert c["heap"]["rss_bytes"] == 4096 * 1024 and c["stack"]["rss_bytes"] == 132 * 1024
    assert c["arena_like"]["vmas"] == 3 and c["arena_like"]["rss_bytes"] == (2048 + 65536 + 512) * 1024
    assert c["arena_like"]["size_bytes"] == 8 * MIB + H + 5 * MIB  # the merged neighbour's MiB included
    assert c["anon_noaccess"]["vmas"] == 4 and c["anon_noaccess"]["rss_bytes"] == 0
    assert c["anon_other"]["vmas"] == 3 and c["anon_other"]["rss_bytes"] == 124 * 1024  # incl. both decoys
    assert (c["file"]["vmas"], c["device"]["vmas"], c["shmem_like"]["vmas"], c["special"]["vmas"]) == (1, 1, 3, 1)
    assert c["file"]["anonymous_bytes"] == 0 and c["anon_other"]["anonymous_bytes"] == 124 * 1024
    assert c["shmem_like"]["private_dirty_bytes"] == 28 * 1024
    text = json.dumps(result)
    assert not any(fragment in text for fragment in ("/", "secret", "private-frame", "someone", "python3", "nvmap",
                                                      "memfd", "deleted"))


@pytest.mark.parametrize("change", [{"skip": "Swap"}, {"bad": "Swap"}])
def test_a_missing_or_unparseable_field_is_unavailable_for_its_category_only(mods, change) -> None:
    text = SMAPS + vma(BASE + 6 * H, BASE + 6 * H + MIB, "rw-p", "/usr/lib/libmissing.so", 4, **change)
    c = mods.profile.parse_smaps(text)["categories"]
    assert c["file"]["swap_bytes"] is None and c["file"]["rss_bytes"] == 8 * 1024
    assert c["heap"]["swap_bytes"] == 0 and c["arena_like"]["swap_bytes"] == 0


@pytest.mark.parametrize("text", ["", "\n", "Rss: 4 kB\n", "abc-def rw-p\n"])
def test_text_without_a_parseable_mapping_is_refused(mods, text) -> None:
    with pytest.raises(ValueError):
        mods.profile.parse_smaps(text)


def test_reading_smaps_reports_status_reason_and_its_own_cost(mods, monkeypatch) -> None:
    p = mods.profile
    if not Path("/proc/self/smaps").exists():
        pytest.skip("no /proc/<pid>/smaps here")
    import os

    observed = p.read_maps_aggregate(os.getpid())
    assert observed["status"] == "observed" and observed["vmas"] > 0 and observed["bytes_read"] > 0
    assert observed["read_ms"] >= 0 and observed["parse_ms"] >= 0
    assert sum(item["vmas"] for item in observed["categories"].values()) == observed["vmas"]

    def missing(path, mode):
        raise FileNotFoundError(2, "gone")

    gone = p.read_maps_aggregate(123, opener=missing)
    assert gone["status"] == "unavailable" and gone["reason"] == "read_failed" and gone["error"] == "ENOENT"
    assert "categories" not in gone
    assert p.read_maps_aggregate(1, opener=lambda path, mode: io.BytesIO(b""))["reason"] == "no_mappings"
    assert p.read_maps_aggregate(1, opener=lambda path, mode: io.BytesIO(b"zz\n"))["reason"] == "unparsed"
    monkeypatch.setattr(p, "MAPS_READ_LIMIT_BYTES", 10)
    big = p.read_maps_aggregate(1, opener=lambda path, mode: io.BytesIO(SMAPS.encode()))
    assert big == {"status": "unavailable", "reason": "too_large", "bytes_read": 11, "read_ms": big["read_ms"]}


def test_the_maps_sampler_keeps_its_interval_and_never_reads_without_a_process(mods) -> None:
    p = mods.profile
    clock, pid, emitted, reads = FakeClock(10.0), {"value": None}, [], []
    sampler = p.MapsSampler(lambda: pid["value"], lambda: "steady",
                            lambda source, name, **fields: emitted.append((source, name, fields)),
                            interval_s=5.0, read=lambda value: reads.append(value) or {"status": "observed", "vmas": 1},
                            clock=clock)
    assert sampler.sample() and reads == []
    pid["value"] = 4242
    clock.now = 14.9
    assert not sampler.sample()
    clock.now = 15.0
    assert sampler.sample() and reads == [4242]
    (s1, n1, f1), (_, _, f2) = emitted
    assert (s1, n1) == ("orchestrator", "memattr_maps") and f1 == {"t_mono": 10.0, "phase": "steady",
                                                                     "status": "unavailable", "reason": "no_process"}
    assert f2 == {"t_mono": 15.0, "phase": "steady", "status": "observed", "vmas": 1}


# ---------------------------------------------------------------- sanitizing and relay (profiler)


def process_record(**changes) -> dict:
    record = {"event": "memattr_process", "t_mono": 120.5, "phase": "steady", "status": "observed",
              "malloc_status": "observed", "malloc_error": None, "malloc": {"arena": 10, "uordblks": 6, "fordblks": 4},
              "malloc_call_us": 3.2, "pymalloc_blocks": 99, "gc_enabled": True, "gc_counts": [1, 2, 3],
              "gc_collections": [10, 1, 0], "gc_seconds": [0.01, 0.002, 0.0], "gc_collected": [5, 0, 0],
              "gc_uncollectable": [0, 0, 0], "gc_full": [[120.1, 4.5, 7, 0]], "gc_full_dropped": 0, "sample_us": 40.0}
    record.update(changes)
    return record


def test_the_sanitizer_keeps_fixed_numeric_fields_and_drops_everything_else(mods) -> None:
    p = mods.profile
    clean = p.sanitize_memattr(process_record(extra="/home/someone/x", malloc_error="ignored text"))
    assert "extra" not in clean and clean["malloc_error"] is None
    assert clean["malloc"]["arena"] == 10 and clean["malloc"]["hblkhd"] is None  # missing stays unavailable
    assert clean["gc_full"] == [[120.1, 4.5, 7, 0]] and clean["gc_collections"] == [10, 1, 0]
    odd = p.sanitize_memattr(process_record(phase="/tmp/x", status="weird", pymalloc_blocks=-1,
                                            malloc={"arena": True, "uordblks": 1 << 70}, malloc_call_us=float("nan"),
                                            gc_counts=[1, 2], gc_full=[[1.0, 2.0, 3, 4]] * 40, gc_enabled=1))
    assert odd["phase"] is None and odd["status"] == "unavailable" and odd["pymalloc_blocks"] is None
    assert odd["malloc"]["arena"] is None and odd["malloc"]["uordblks"] is None and odd["malloc_call_us"] is None
    assert odd["gc_counts"] is None and len(odd["gc_full"]) == 16 and odd["gc_enabled"] is None
    config = p.sanitize_memattr({"event": "memattr_config", "glibc_version": "2.39", "python": "3.10.14",
                                 "gc_thresholds": [700, 10, 10], "mallinfo2": True, "interval_s": 1.0,
                                 "full_gc_per_sample": 16, "gc_enabled": True, "gc_frozen": 0, "path": "/x"})
    assert config["glibc_version"] == "2.39" and config["python"] == "3.10.14" and "path" not in config
    assert p.sanitize_memattr({"event": "memattr_config", "glibc_version": "glibc /x"})["glibc_version"] is None
    assert p.sanitize_memattr({"event": "other"}) is None


@pytest.mark.parametrize("sanitized", [True, False])
def test_the_relay_records_attribution_events_sanitized_in_either_mode(mods, tmp_path, sanitized) -> None:
    p = mods.profile
    lines = [p.EVENT_PREFIX + json.dumps(process_record(extra="drop me")) + "\n",
             p.EVENT_PREFIX + json.dumps({"event": "memattr_config", "python": "3.10.14", "note": "drop"}) + "\n"]
    proc = SimpleNamespace(stdout=io.StringIO("".join(lines)))
    events = p.Events(tmp_path / "events.jsonl", 0.0)
    p.pump_workload(proc, tmp_path, events, lambda name, source="x": None, sanitized=sanitized)
    events.close()
    saved = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [r["event"] for r in saved] == ["memattr_process", "memattr_config"]
    assert "extra" not in saved[0] and "note" not in saved[1] and saved[0]["t_mono"] == 120.5


# ---------------------------------------------------------------- command, preconditions and summary (profiler)


@pytest.mark.parametrize("flags", [["--post-load-release"], ["--post-load-release", "--memory-attribution"]])
def test_the_workload_command_gains_only_the_attribution_flag(mods, monkeypatch, tmp_path, flags) -> None:
    p = mods.profile
    launched = []
    monkeypatch.setattr(p.subprocess, "Popen", lambda argv, **kwargs: launched.append((argv, kwargs)))
    args = p.parse_args(["--clip", "c.mp4", *flags])
    p.start_workload(args, tmp_path)
    (argv, kwargs), = launched
    default = [str(args.python), str(p.WORKLOAD), "--engine", str(args.engine), "--port", "18081",
               "--fps", "15.0", "--face-hz", "1.0", "--scene-interval-s", "4.0", "--settle-s", "15.0",
               "--warmup-s", "120.0", "--steady-s", "600.0", "--clip", "c.mp4"]
    assert argv == default + flags and kwargs.get("stdin") == subprocess.PIPE


def test_attribution_is_off_by_default_bounded_and_defined_only_with_plr(mods, monkeypatch) -> None:
    p = mods.profile
    args = p.parse_args([])
    assert args.memory_attribution is False and args.maps_interval_s == 5.0
    assert p.parse_args(["--maps-interval-s", "2"]).maps_interval_s == 2.0
    for bad in ("1.9", "61", "nan", "inf"):
        with pytest.raises(SystemExit):
            p.parse_args(["--maps-interval-s", bad])
    monkeypatch.setattr(p, "scan_processes", lambda: [])
    monkeypatch.setattr(p, "command_output", lambda argv, env=None: "inactive")
    monkeypatch.setattr(p, "port_in_use", lambda port: False)
    problems, _ = p.preconditions(p.parse_args(["--memory-attribution"]))
    assert any("only with --post-load-release" in problem for problem in problems)
    problems, _ = p.preconditions(p.parse_args(["--memory-attribution", "--post-load-release"]))
    assert not any("--memory-attribution" in problem for problem in problems)


def write_ma1_run(profile, run_dir: Path, *, malloc: bool = True, attribution: bool = True) -> None:
    parameters = {"face_hz": 1.0, "scene_interval_s": 4.0, "steady_s": 600.0, "post_load_release": True,
                  "release_settle_s": 5.0}
    if attribution:
        parameters.update(memory_attribution=True, maps_interval_s=5.0)
    manifest = {"run_id": "demo-profile-20991231T000000Z", "input": {"synthetic": False, "fps": 15.0},
                "parameters": parameters, "llama_server": {"cache_ram_mib": 0, "flags": []},
                "repository": {"commit": "a" * 40, "tracked_changes": False}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    rows = [(10 + t / 5, "warmup") for t in range(600)] + [(130 + t / 5, "steady") for t in range(3001)]
    with (run_dir / "memory.csv").open("w") as handle:
        handle.write(",".join(profile.Sampler.COLUMNS) + "\n")
        for t, phase in rows:
            stepped = t >= 300.0  # one 12 MiB step in the workload's RSS at 300 s, all anonymous
            row = dict.fromkeys(profile.Sampler.COLUMNS, "")
            row.update(t_mono=f"{t:.3f}", phase=phase, mem_total=8_000_000_000, mem_available=4_000_000_000,
                       mem_free=2_000_000_000, cached=1_000_000_000, swap_total=0, swap_free=0, pswpin=0, pswpout=0,
                       work_rss=2_000_000_000 + (12 * MIB if stepped else 0),
                       anon_pages=1_800_000_000 + (12 * MIB if stepped else 0))
            handle.write(",".join(str(row[c]) for c in profile.Sampler.COLUMNS) + "\n")
    (run_dir / "tegrastats.log").write_text("")
    events = [{"t_mono": 130.0, "event": "steady_boundary", "edge": "start", "boundary_t_mono": 130.0},
              {"t_mono": 730.0, "event": "steady_boundary", "edge": "end", "boundary_t_mono": 730.0},
              {"t_mono": 900.0, "event": "run_end", "status": "complete"}]
    if attribution:
        events.append({"t_mono": 9.0, "event": "memattr_config", "source": "workload", "python": "3.10.14",
                       "mallinfo2": malloc, "gc_thresholds": [700, 10, 10]})
        for second in range(10, 731):
            grown = second >= 300
            events.append({"t_mono": float(second), "event": "memattr_process", "phase": "steady",
                           "status": "observed" if malloc else "partial",
                           "malloc_status": "observed" if malloc else "unsupported",
                           "malloc": {**dict.fromkeys(profile.MALLINFO2_REPORTED, 1000),
                                      "arena": 500 * MIB + (12 * MIB if grown else 0), "uordblks": 400 * MIB,
                                      "fordblks": 100 * MIB + (12 * MIB if grown else 0)} if malloc else None,
                           "malloc_call_us": 4.0 if malloc else None, "pymalloc_blocks": 1000,
                           "gc_collections": [second * 10, second, second // 10],
                           "gc_full": [[299.5, 30.0, 100, 0]] if second == 300 else [], "sample_us": 50.0})
        for t in range(10, 731, 5):
            events.append({"t_mono": float(t), "event": "memattr_maps", "phase": "steady", "status": "observed",
                           "bytes_read": 1000, "read_ms": 2.0, "parse_ms": 1.0, "vmas": 10,
                           "categories": {name: {"rss_bytes": (300 * MIB + (12 * MIB if t >= 300 else 0)
                                                               if name == "arena_like" else MIB),
                                                 "private_dirty_bytes": MIB, "anonymous_bytes": MIB}
                                          for name in profile.MAPS_CATEGORIES}})
        events.append({"t_mono": 731.0, "event": "memattr_maps", "phase": "stopping", "status": "unavailable",
                       "reason": "read_failed", "error": "ESRCH"})
    (run_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run_dir / "llama-server.log").write_text("t_mono=2.000 prompt cache is disabled\n")


def test_the_ma1_summary_attributes_each_window_and_step_descriptively(mods, tmp_path) -> None:
    p = mods.profile
    write_ma1_run(p, tmp_path)
    text = p.summarize(tmp_path)
    report = json.loads((tmp_path / "memattr.json").read_text())
    written = json.loads((tmp_path / "profile.json").read_text())
    assert written["memory_attribution"] == "memattr.json" and "never eligibility" in written["instrumentation"]
    assert written["criteria_id"] == "step4plr-combined-cache-off-v2"  # never step 4's own identity
    steady = report["windows"]["steady"]
    assert steady["malloc"]["fordblks"]["delta"] == 12 * MIB and steady["malloc"]["uordblks"]["delta"] == 0
    assert steady["maps"]["arena_like"]["rss_bytes"]["delta"] == 12 * MIB
    assert steady["maps"]["heap"]["rss_bytes"]["delta"] == 0
    assert steady["process_memory_csv"]["work_rss"]["delta"] == 12 * MIB
    assert steady["gc_collections_by_generation"] == {"gen0": 6000, "gen1": 600, "gen2": 60}
    assert report["steps"]["count"] == 1
    (step,) = report["steps"]["listed"]
    assert step["rss_delta_bytes"] == 12 * MIB and step["nearest_full_gc_offset_s"] == pytest.approx(0.5)
    assert step["maps_bracket"]["rss_delta_bytes"]["arena_like"] == 12 * MIB
    assert step["malloc_bracket"]["delta"]["fordblks"] == 12 * MIB
    assert report["sources"]["maps"]["unavailable_by_reason"] == {"read_failed": 1}
    assert report["cost"]["maps_read_ms"] == {"n": 145, "p50": 2.0, "max": 2.0}  # the failed read had no read_ms
    assert any("leak or bounded" in item for item in report["cannot_establish"])
    assert "MA1 memory attribution (instrumented diagnostic, D55" in text
    assert "instrumented MA1 run: descriptive only, never eligible" in text


def test_unsupported_allocator_counters_stay_unavailable_in_the_summary(mods, tmp_path) -> None:
    p = mods.profile
    write_ma1_run(p, tmp_path, malloc=False)
    p.summarize(tmp_path)
    report = json.loads((tmp_path / "memattr.json").read_text())
    malloc = report["windows"]["steady"]["malloc"]
    assert all(item == {"n": 0, "first": None, "last": None, "delta": None, "min": None, "max": None}
               for item in malloc.values())
    assert report["sources"]["process"]["malloc_unsupported"] > 0 and report["steps"]["listed"][0]["malloc_bracket"] is None
    assert report["sources"]["process"]["full_gc_dropped"] is None  # never recorded: unavailable, not 0
    assert report["cost"]["malloc_call_us"] == {"n": 0, "p50": None, "max": None}


def test_a_plr_run_without_the_flag_writes_no_attribution_record(mods, tmp_path) -> None:
    p = mods.profile
    write_ma1_run(p, tmp_path, attribution=False)
    text = p.summarize(tmp_path)
    assert not (tmp_path / "memattr.json").exists() and "MA1" not in text
    written = json.loads((tmp_path / "profile.json").read_text())
    assert "memory_attribution" not in written and "instrumentation" not in written


def test_the_maps_sampler_thread_stops_promptly(mods) -> None:
    p = mods.profile
    done = threading.Event()
    sampler = p.MapsSampler(lambda: None, lambda: "steady", lambda *a, **k: done.set(), interval_s=60.0)
    sampler.start()
    assert done.wait(5)
    sampler.stop()
    assert not sampler.is_alive()


def test_periodic_attribution_readings_are_recorded_but_not_echoed(mods, tmp_path, capsys) -> None:
    p = mods.profile
    events = p.Events(tmp_path / "events.jsonl", 0.0)
    events.add("workload", "memattr_process", t_mono=1.0, status="observed")
    events.add("orchestrator", "memattr_maps", t_mono=1.0, status="observed")
    events.add("workload", "memattr_config", python="3.10.14")
    events.close()
    printed = capsys.readouterr().out
    assert "memattr_process" not in printed and "memattr_maps" not in printed and "memattr_config" in printed
    saved = [json.loads(line)["event"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert saved == ["memattr_process", "memattr_maps", "memattr_config"]


def test_small_allocations_in_another_thread_count_in_arena_and_in_a_heap_mapping(mods) -> None:
    """mallinfo2 sums every arena on this C library (its manual's BUGS note says only the main one): the
    allocations of a new thread, which glibc serves from that thread's arena, raise ``arena`` and the RSS of a
    heap-like mapping ([heap] or an arena_like one)."""
    read = mods.workload.glibc_mallinfo2()
    if read is None or not Path("/proc/self/smaps").exists():
        pytest.skip("needs glibc's mallinfo2 and /proc/<pid>/smaps")
    import os

    def heaps() -> int:
        categories = mods.profile.read_maps_aggregate(os.getpid())["categories"]
        return categories["heap"]["rss_bytes"] + categories["arena_like"]["rss_bytes"]

    kept: list = []
    before, mapped_before = read(), heaps()
    worker = threading.Thread(target=lambda: kept.append([bytearray(2000) for _ in range(20_000)]))  # ~40 MB
    worker.start()
    worker.join()
    after, mapped_after = read(), heaps()
    assert after["arena"] - before["arena"] >= 30_000_000 and after["uordblks"] - before["uordblks"] >= 30_000_000
    assert after["hblkhd"] - before["hblkhd"] < 1_000_000  # small chunks: not individually mmapped
    assert mapped_after - mapped_before >= 30_000_000


def test_the_pymalloc_count_follows_small_python_objects(mods) -> None:
    import sys

    before = sys.getallocatedblocks()
    kept = [(i, str(i)) for i in range(20_000)]  # a tuple and a str each: small objects in pymalloc's pools
    assert sys.getallocatedblocks() - before >= 20_000 and len(kept) == 20_000
