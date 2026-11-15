"""MR1 (opt-in): per-model page-cache release after each load and settle, then bounded smoke checks.

Fakes only: no GPU, model, camera, journal or real cache release. Covers the release record
(returned, failed, unsupported, ineffective), missing telemetry kept unavailable, the
workload's checkpoint handshake and smoke checks, the orchestrator's relay, the summary,
and that the default (non-MR1) path is unchanged.
"""

from __future__ import annotations

import errno
import importlib
import io
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
GIB = 1 << 30


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return SimpleNamespace(profile=importlib.import_module("demo_profile"),
                           workload=importlib.import_module("demo_workload"))


def snapshot(free, cached, *, available=None, mapped=600_000_000, llama_rss=2_000_000_000, **missing):
    """A fake mr1_snapshot result; a key passed in missing=... is unavailable (None)."""
    if available is None and free is not None and cached is not None:
        available = free + cached
    meminfo = {"MemTotal": 8_000_000_000, "MemFree": free, "MemAvailable": available,
               "Cached": cached, "AnonPages": 1_500_000_000, "Mapped": mapped, "Active(file)": 400_000_000,
               "Inactive(file)": cached - 400_000_000 if cached is not None else None}
    meminfo.update({key: None for key in missing})
    total, avail = meminfo["MemTotal"], meminfo["MemAvailable"]
    return {"t_mono": 1.0, "meminfo": meminfo, "pressure_bytes": total - avail if avail is not None else None,
            "processes": {"llama": {"rss_bytes": llama_rss, "hwm_bytes": llama_rss, "pss_bytes": None},
                          "work": {"rss_bytes": None, "hwm_bytes": None, "pss_bytes": None}}}


def returned(size=100):
    return lambda path: {"name": Path(path).name, "bytes": size, "call": "posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)",
                         "result": "returned_0", "returncode": 0, "error": None, "elapsed_s": 0.001}


def release(mods, before, after, release_fn=None, files=None):
    states = iter([before, after])
    sleeps = []
    record = mods.profile.release_component(
        "scene", files or {"llm": Path("/m/llm.gguf"), "mmproj": Path("/m/mmproj.gguf")}, {"llama": 1, "work": None},
        5.0, snapshot=lambda: next(states), release=release_fn or returned(), sleep=sleeps.append)
    return record, sleeps


# ---------------------------------------------------------------- the release call


def test_a_release_records_the_call_its_return_value_size_and_time(mods, tmp_path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"x" * 1234)
    calls = []
    record = mods.profile.release_file_cache(path, advise=lambda *args: calls.append(args))
    assert record["result"] == "returned_0" and record["returncode"] == 0 and record["error"] is None
    assert record["bytes"] == 1234 and record["name"] == "model.gguf" and record["elapsed_s"] >= 0
    assert record["call"] == "posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)"
    (fd, offset, length, advice), = calls
    assert (offset, length, advice) == (0, 0, getattr(os, "POSIX_FADV_DONTNEED", 4))
    with pytest.raises(OSError):
        os.fstat(fd)  # the descriptor was closed


def test_the_real_call_on_a_temporary_file_returns_0_where_the_platform_has_it(mods, tmp_path) -> None:
    path = tmp_path / "scratch.bin"
    path.write_bytes(b"x" * 8192)
    record = mods.profile.release_file_cache(path)
    expected = "returned_0" if hasattr(os, "posix_fadvise") else "unsupported"
    assert record["result"] == expected and path.read_bytes() == b"x" * 8192  # the file itself is unchanged


@pytest.mark.parametrize("failure", ["advise_error", "missing_file", "unsupported"])
def test_a_failed_or_unsupported_release_keeps_its_exact_status(mods, tmp_path, monkeypatch, failure) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"x")

    def refuse(*args):
        raise OSError(errno.EINVAL, "Invalid argument")

    if failure == "advise_error":
        record = mods.profile.release_file_cache(path, advise=refuse)
        assert (record["result"], record["returncode"], record["error"], record["bytes"]) == (
            "returned_error", errno.EINVAL, "EINVAL", 1)
    elif failure == "missing_file":
        record = mods.profile.release_file_cache(tmp_path / "absent", advise=refuse)
        assert (record["result"], record["returncode"], record["error"], record["bytes"]) == (
            "open_failed", errno.ENOENT, "ENOENT", None)
    else:
        monkeypatch.delattr(mods.profile.os, "posix_fadvise", raising=False)
        record = mods.profile.release_file_cache(path)
        assert (record["result"], record["returncode"], record["bytes"]) == ("unsupported", None, None)


# ---------------------------------------------------------------- samples, deltas and outcomes


def test_a_snapshot_keeps_missing_memory_and_process_fields_unavailable_not_zero(mods) -> None:
    profile = mods.profile
    partial = profile.mr1_snapshot({"llama": 7, "work": None}, meminfo=lambda: {"MemTotal": 8, "MemFree": 3},
                                   process_memory=lambda pid: (None, None), pss=lambda pid: None, clock=lambda: 2.0)
    assert partial["meminfo"]["MemFree"] == 3 and partial["meminfo"]["Mapped"] is None
    assert set(partial["meminfo"]) == set(profile.MEMINFO_KEYS) and partial["pressure_bytes"] is None
    assert partial["processes"] == {"llama": {"rss_bytes": None, "hwm_bytes": None, "pss_bytes": None},
                                    "work": {"rss_bytes": None, "hwm_bytes": None, "pss_bytes": None}}

    def unreadable():
        raise OSError("no /proc")

    empty = profile.mr1_snapshot({}, meminfo=unreadable)
    assert all(value is None for value in empty["meminfo"].values()) and empty["pressure_bytes"] is None
    full = profile.mr1_snapshot({"llama": 7}, meminfo=lambda: {"MemTotal": 8, "MemAvailable": 5, "Mapped": -1},
                                process_memory=lambda pid: (10, 12), pss=lambda pid: 4)
    assert full["pressure_bytes"] == 3 and full["meminfo"]["Mapped"] is None  # invalid, not kept
    assert full["processes"]["llama"] == {"rss_bytes": 10, "hwm_bytes": 12, "pss_bytes": 4}


@pytest.mark.parametrize(("after", "outcome"), [
    (snapshot(2_200_000_000, 1_500_000_000), "memfree_rose_cached_fell"),
    (snapshot(1_000_000_000, 2_700_000_000), "ineffective"),
    (snapshot(990_000_000, 2_710_000_000), "ineffective"),  # moved the other way
    (snapshot(1_100_000_000, 2_700_000_000), "partial"),
    (snapshot(1_000_000_000, 2_600_000_000), "partial"),
    (snapshot(2_200_000_000, None), "telemetry_unavailable"),
    (snapshot(None, 1_500_000_000), "telemetry_unavailable"),
])
def test_release_outcomes_are_directional_labels_with_the_measured_deltas(mods, after, outcome) -> None:
    record, sleeps = release(mods, snapshot(1_000_000_000, 2_700_000_000), after)
    assert record["outcome"] == outcome and sleeps == [5.0]
    deltas = record["deltas"]
    for key in ("MemFree", "Cached"):
        a, b = 2_700_000_000 if key == "Cached" else 1_000_000_000, after["meminfo"][key]
        assert deltas[key] == (None if b is None else b - a)  # unavailable, never zero
    assert deltas["Mapped"] == 0 and deltas["processes"]["work"] == {"rss_bytes": None, "hwm_bytes": None,
                                                                     "pss_bytes": None}
    assert record["files_total_bytes"] == 200 and "not how much" in record["files_total_bytes_note"]
    assert "neither per-file residency nor which pages" in record["basis"]


@pytest.mark.parametrize("bad", [
    {"result": "open_failed", "returncode": errno.ENOENT, "error": "ENOENT", "bytes": None},
    {"result": "returned_error", "returncode": errno.EBADF, "error": "EBADF", "bytes": 5},
    {"result": "unsupported", "returncode": None, "error": None, "bytes": None},
])
def test_a_failed_call_is_call_failed_even_when_memory_moved(mods, bad) -> None:
    def mixed(path):
        return {**returned()(path), **bad} if Path(path).name == "mmproj.gguf" else returned()(path)

    record, _ = release(mods, snapshot(1_000_000_000, 2_700_000_000), snapshot(2_000_000_000, 1_700_000_000), mixed)
    assert record["outcome"] == "call_failed"
    assert record["files"]["mmproj"]["result"] == bad["result"] and record["files"]["llm"]["result"] == "returned_0"
    if bad["bytes"] is None:
        assert record["files_total_bytes"] is None


def test_the_model_files_released_per_component_are_fixed(mods) -> None:
    profile = mods.profile
    args = profile.parse_args([])
    files = profile.mr1_files(args)
    assert {c: set(f) for c, f in files.items()} == {
        "scene": {"llm", "mmproj"}, "detector": {"engine"},
        "face": {"facenet512_weights.h5", "face_detection_yunet_2023mar.onnx"}}
    assert files["scene"]["llm"] == args.model and files["detector"]["engine"] == args.engine


# ---------------------------------------------------------------- the workload's checkpoints and smoke checks


def test_a_checkpoint_waits_for_its_own_acknowledgement_or_stops_the_workload(mods, monkeypatch) -> None:
    workload = mods.workload
    emitted = []
    monkeypatch.setattr(workload, "event", lambda kind, **fields: emitted.append((kind, fields)))
    timeouts = []
    workload.mr1_checkpoint("detector", lambda timeout: timeouts.append(timeout) or "ack detector")
    assert emitted == [("mr1_checkpoint", {"stage": "detector"})] and timeouts == [workload.MR1_ACK_TIMEOUT_S]
    for reply in (None, "ack face", "", "ack detector extra"):
        with pytest.raises(SystemExit) as stop:
            workload.mr1_checkpoint("detector", lambda timeout, reply=reply: reply)
        assert stop.value.code == 4 and emitted[-1] == ("fatal", {"reason": "mr1 checkpoint not acknowledged"})


def test_the_acknowledgement_reader_is_bounded_and_ends_on_eof(mods) -> None:
    read_line = mods.workload.read_ack_line
    reader, writer = os.pipe()
    try:
        os.write(writer, b"ack face\n")
        assert read_line(1.0, reader) == "ack face"
        started = time.monotonic()
        assert read_line(0.05, reader) is None and time.monotonic() - started < 1.0
        os.write(writer, b"x" * 300)
        assert read_line(1.0, reader) is None  # overlong
    finally:
        os.close(writer)
    assert read_line(1.0, reader) is None  # EOF: the orchestrator is gone
    os.close(reader)


class FakeFrames:
    def __init__(self, fail_after=None):
        self.reads, self.fail_after = 0, fail_after

    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise RuntimeError("clip ended")
        return f"frame-{self.reads}"


class FakeModel:
    def __init__(self, failing=()):
        self.frames, self.failing = [], set(failing)

    def track(self, frame, **kwargs):
        self.frames.append(frame)
        if len(self.frames) in self.failing:
            raise ValueError("bad frame")


class FakeFace:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def represent(self, img_path, **kwargs):
        self.calls.append(img_path)
        if self.fail:
            raise MemoryError()
        return [{"face_confidence": 0.9}]


def scene_ok(port, frame, stats):
    return {"latency_ms": 2500.0, "finish_reason": "stop", "valid": True, "prompt_tokens": 325, "completion_tokens": 100}


def test_smoke_checks_run_each_component_once_bounded_and_pass_when_all_complete(mods) -> None:
    workload, profile = mods.workload, mods.profile
    frames, model, face = FakeFrames(), FakeModel(), FakeFace()
    seen = []
    result = workload.run_smoke(model, face, SimpleNamespace(port=18081), frames=frames,
                                scene=lambda port, frame, stats: seen.append((port, frame)) or scene_ok(port, frame, stats))
    assert frames.reads == len(model.frames) == workload.MR1_DETECTOR_FRAMES == 15
    assert face.calls == ["frame-15"] and seen == [(18081, "frame-15")]  # the last detector frame
    assert result["detector"]["processed"] == 15 and result["detector"]["errors"] == {}
    assert result["face"] == {"runs": 1, "completed": 1, "runs_with_face": 1, "errors": {},
                              "latency_ms": result["face"]["latency_ms"]}
    assert result["scene"]["completed"] == 1 and result["scene"]["finish_reason"] == "stop" and result["scene"]["valid"]
    clean = profile.sanitize_mr1_smoke(json.loads(json.dumps(result)))
    assert profile.functional_smoke(clean) == "pass"


@pytest.mark.parametrize("failure", ["detector", "face", "scene_error", "scene_invalid", "scene_length", "no_frames"])
def test_any_failed_smoke_check_is_a_fail_with_its_error_class(mods, failure) -> None:
    workload, profile = mods.workload, mods.profile
    model = FakeModel(failing={3} if failure == "detector" else ())
    face = FakeFace(fail=failure == "face")
    frames = FakeFrames(fail_after=0 if failure == "no_frames" else None)

    def scene(port, frame, stats):
        if failure == "scene_error":
            with stats.lock:
                stats.vlm_errors["timeout"] = 1
            return {"error": "timeout"}
        if failure == "scene_invalid":
            return {"latency_ms": 2000.0, "finish_reason": "stop", "rejection": "invalid_report"}
        if failure == "scene_length":
            return {"latency_ms": 2000.0, "finish_reason": "length", "rejection": "truncated"}
        return scene_ok(port, frame, stats)

    result = profile.sanitize_mr1_smoke(json.loads(json.dumps(
        workload.run_smoke(model, face, SimpleNamespace(port=1), frames=frames, scene=scene))))
    assert profile.functional_smoke(result) == "fail"
    if failure == "detector":
        assert result["detector"]["processed"] == 14 and result["detector"]["errors"] == {"ValueError": 1}
    if failure == "face":
        assert result["face"]["completed"] == 0 and result["face"]["errors"] == {"MemoryError": 1}
    if failure == "scene_error":
        assert result["scene"]["error"] == "timeout" and result["scene"]["errors"] == {"timeout": 1}
    if failure == "no_frames":
        assert result["detector"]["errors"] == {"RuntimeError": 15} and result["scene"]["attempts"] == 0


def test_the_smoke_record_keeps_only_fixed_keys_numbers_and_labels(mods) -> None:
    profile = mods.profile
    raw = {"detector": {"frames_requested": 15, "processed": 15, "errors": {"secret rtsp://u:p@cam": 1, "OSError": 2},
                        "latency_ms": {"n": 15, "p50": 9.5, "max": float("inf"), "text": "x"}, "boxes": "private"},
           "face": {"runs": 1, "completed": 1, "errors": {}, "latency_ms": 900.0, "names": ["someone"]},
           "scene": {"attempts": 1, "completed": 1, "finish_reason": "a private summary", "valid": "yes",
                     "rejection": "truncated", "error": None, "errors": {"HTTP 503": 1}, "latency_ms": 2000.0},
           "summary": "private model text"}
    clean = profile.sanitize_mr1_smoke(raw)
    assert set(clean) == {"detector", "face", "scene"} and "private" not in json.dumps(clean)
    assert clean["detector"]["errors"] == {"OSError": 2} and clean["detector"]["latency_ms"] == {
        "n": 15, "p50": 9.5, "max": None}
    assert clean["scene"]["finish_reason"] is None and clean["scene"]["valid"] is None
    assert clean["scene"]["rejection"] == "truncated" and clean["scene"]["errors"] == {"HTTP 503": 1}
    assert profile.functional_smoke(None) == "unavailable" and profile.functional_smoke(clean) == "fail"


@pytest.mark.parametrize("mr1", [False, True])
def test_the_workload_main_takes_the_mr1_path_only_when_asked(mods, monkeypatch, mr1) -> None:
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
    monkeypatch.setattr(workload, "run_workload", lambda *args: calls.append("workload"))
    monkeypatch.setattr(workload, "mr1_checkpoint", lambda stage: calls.append(f"checkpoint {stage}"))
    monkeypatch.setattr(workload, "run_smoke", lambda model, face, args: calls.append("smoke") or {"detector": {}})
    monkeypatch.setattr(workload, "event", lambda kind, **fields: calls.append(f"event {kind}"))
    monkeypatch.setattr(workload.time, "sleep", lambda seconds: None)
    arguments = ["--engine", "e", "--port", "1"] + (["--mr1-release-check"] if mr1 else [])
    assert workload.main(arguments) == 0
    steps = [c for c in calls if not c.startswith("event phase") and c != "event cuda_driver"]
    if mr1:
        assert steps == ["detector", "start", "checkpoint detector", "face", "checkpoint face", "smoke",
                         "event mr1_smoke", "checkpoint after_smoke", "stop"]
    else:
        assert steps == ["detector", "start", "face", "workload", "stop"]
    with pytest.raises(SystemExit):
        workload.main(["--scene-only", "--port", "1", "--mr1-release-check"])


# ---------------------------------------------------------------- the orchestrator's relay and defaults


def lines(*records):
    return io.StringIO("".join("@@EVENT " + json.dumps(record) + "\n" for record in records))


def test_the_pump_hands_checkpoints_to_the_release_and_sanitizes_the_smoke(mods, tmp_path) -> None:
    profile = mods.profile
    stages, added = [], []
    sink = SimpleNamespace(add=lambda source, name, **fields: added.append((name, fields)))
    proc = SimpleNamespace(stdout=lines(
        {"event": "phase", "name": "detector_settle"},
        {"event": "mr1_checkpoint", "stage": "detector"},
        {"event": "mr1_checkpoint", "stage": "rm -rf /"},
        {"event": "mr1_smoke", "detector": {"processed": 15, "frames_requested": 15, "errors": {}},
         "scene": {"finish_reason": "secret text"}, "summary": "private"},
        {"event": "mr1_checkpoint", "stage": "after_smoke"},
    ))
    phases = []
    profile.pump_workload(proc, tmp_path, sink, lambda name, source: phases.append(name), sanitized=True,
                          on_checkpoint=stages.append)
    assert stages == ["detector", "after_smoke"] and phases == ["detector_settle"]
    (name, smoke), = added
    assert name == "mr1_smoke" and "private" not in json.dumps(smoke) and "secret" not in json.dumps(smoke)


class BrokenPipe:
    def write(self, text):
        raise BrokenPipeError()

    def flush(self):
        raise BrokenPipeError()


def test_the_checkpoint_handler_releases_samples_and_acknowledges_each_stage(mods) -> None:
    profile = mods.profile
    added, phases, released, sampled = [], [], [], []
    events = SimpleNamespace(add=lambda source, name, **fields: added.append((name, fields)))
    procs = {"llama": SimpleNamespace(pid=11), "work": None}
    files = {"scene": {"llm": Path("a")}, "detector": {"engine": Path("b")}, "face": {"w": Path("c")}}

    def release(stage, stage_files, pids, settle_s):
        released.append((stage, stage_files, dict(pids), settle_s))
        return {"component": stage, "outcome": "ineffective"}

    handle = profile.mr1_checkpoint_handler(procs, events, phases.append, files, 5.0, release=release,
                                            snapshot=lambda pids: sampled.append(dict(pids)) or {"t_mono": 1.0})
    handle("scene")  # before the workload exists: no acknowledgement
    procs["work"] = SimpleNamespace(pid=22, stdin=io.StringIO())
    handle("detector")
    handle("face")
    handle("after_smoke")
    assert phases == ["scene_release", "detector_release", "face_release"]
    assert released == [("scene", files["scene"], {"llama": 11, "work": None}, 5.0),
                        ("detector", files["detector"], {"llama": 11, "work": 22}, 5.0),
                        ("face", files["face"], {"llama": 11, "work": 22}, 5.0)]
    assert sampled == [{"llama": 11, "work": 22}]
    assert [name for name, _ in added] == ["mr1_release", "mr1_release", "mr1_release", "mr1_snapshot"]
    assert procs["work"].stdin.getvalue() == "ack detector\nack face\nack after_smoke\n"


def test_a_failed_release_step_is_recorded_by_class_and_never_acknowledged(mods) -> None:
    profile = mods.profile
    added = []
    events = SimpleNamespace(add=lambda source, name, **fields: added.append((name, fields)))
    stdin = io.StringIO()
    procs = {"llama": SimpleNamespace(pid=1), "work": SimpleNamespace(pid=2, stdin=stdin)}

    def broken(stage, files, pids, settle_s):
        raise RuntimeError("private path /home/someone/models")

    handle = profile.mr1_checkpoint_handler(procs, events, lambda name: None, {"detector": {}}, 5.0, release=broken)
    handle("detector")
    assert added == [("mr1_error", {"stage": "detector", "error": "RuntimeError"})] and stdin.getvalue() == ""
    procs["work"] = SimpleNamespace(pid=2, stdin=BrokenPipe())
    handle = profile.mr1_checkpoint_handler(procs, events, lambda name: None, {}, 5.0,
                                            snapshot=lambda pids: {"t_mono": 1.0})
    handle("after_smoke")  # the workload is gone: no exception escapes the relay thread


@pytest.mark.parametrize("sanitized", [True, False])
def test_without_mr1_the_pump_never_releases_and_relays_as_before(mods, tmp_path, sanitized) -> None:
    profile = mods.profile
    added = []
    sink = SimpleNamespace(add=lambda source, name, **fields: added.append(name))
    proc = SimpleNamespace(stdout=lines({"event": "mr1_checkpoint", "stage": "detector"},
                                        {"event": "workload_stats", "seconds": 1.0}))
    profile.pump_workload(proc, tmp_path, sink, lambda *a, **k: None, sanitized=sanitized)
    assert added == (["workload_stats"] if sanitized else ["mr1_checkpoint", "workload_stats"])


@pytest.mark.parametrize("mr1", [False, True])
def test_the_workload_command_and_stdin_change_only_with_mr1(mods, monkeypatch, tmp_path, mr1) -> None:
    profile = mods.profile
    launched = []
    monkeypatch.setattr(profile.subprocess, "Popen", lambda argv, **kwargs: launched.append((argv, kwargs)))
    args = profile.parse_args(["--clip", "c.mp4"] + (["--mr1-release-check"] if mr1 else []))
    profile.start_workload(args, tmp_path)
    (argv, kwargs), = launched
    default = [str(args.python), str(profile.WORKLOAD), "--engine", str(args.engine), "--port", "18081",
               "--fps", "15.0", "--face-hz", "1.0", "--scene-interval-s", "4.0", "--settle-s", "15.0",
               "--warmup-s", "120.0", "--steady-s", "600.0", "--clip", "c.mp4"]
    assert argv == default + (["--mr1-release-check"] if mr1 else [])
    assert ("stdin" in kwargs) is mr1 and (kwargs.get("stdin") == subprocess.PIPE if mr1 else True)


def test_mr1_is_off_by_default_and_refused_with_scene_only(mods, monkeypatch) -> None:
    profile = mods.profile
    args = profile.parse_args([])
    assert args.mr1_release_check is False and args.mr1_release_settle_s == 5.0
    assert profile.PHASES_IN_ORDER[:3] == ("baseline", "llama_load", "llama_settle")
    default_phases = [p for p in profile.PHASES_IN_ORDER if not p.endswith("_release") and p != "smoke"]
    assert default_phases == ["baseline", "llama_load", "llama_settle", "detector_load", "detector_settle",
                              "face_load", "face_settle", "warmup", "steady", "stopping", "unload_workload",
                              "unload_llama"]
    monkeypatch.setattr(profile, "scan_processes", lambda: [])
    monkeypatch.setattr(profile, "command_output", lambda argv, env=None: "inactive")
    monkeypatch.setattr(profile, "port_in_use", lambda port: False)
    problems, _ = profile.preconditions(profile.parse_args(["--scene-only", "--mr1-release-check"]))
    assert any("--mr1-release-check" in problem for problem in problems)


def write_mr1_run(profile, run_dir: Path, *, smoke=True, releases=("scene", "detector", "face")) -> None:
    manifest = {"run_id": "mr1-fake", "input": {"synthetic": False, "fps": 15.0},
                "parameters": {"face_hz": 1.0, "scene_interval_s": 4.0, "mr1_release_check": True,
                               "mr1_release_settle_s": 5.0},
                "llama_server": {"cache_ram_mib": 0}, "repository": {"commit": "fake", "tracked_changes": False}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "memory.csv").write_text(",".join(profile.Sampler.COLUMNS) + "\n")
    (run_dir / "tegrastats.log").write_text("")
    events = []
    free, cached = 1_000_000_000, 2_700_000_000
    for component in releases:
        record, _ = release(SimpleNamespace(profile=profile), snapshot(free, cached),
                            snapshot(free + 400_000_000, cached - 400_000_000))
        events.append({"event": "mr1_release", "source": "orchestrator", **{**record, "component": component}})
        free, cached = free + 400_000_000, cached - 400_000_000
    if smoke:
        events.append({"event": "mr1_smoke", "source": "workload", **profile.sanitize_mr1_smoke({
            "detector": {"frames_requested": 15, "processed": 15, "errors": {}},
            "face": {"runs": 1, "completed": 1, "errors": {}},
            "scene": {"attempts": 1, "completed": 1, "finish_reason": "stop", "valid": True}})})
        events.append({"event": "mr1_snapshot", "source": "orchestrator", "stage": "after_smoke",
                       **snapshot(free - 50_000_000, cached + 50_000_000)})
    events.append({"event": "run_end", "status": "complete"})
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(item) for item in events) + "\n")


def test_an_mr1_summary_writes_its_record_and_no_step4_criteria(mods, tmp_path) -> None:
    profile = mods.profile
    write_mr1_run(profile, tmp_path)
    text = profile.summarize(tmp_path)
    report = json.loads((tmp_path / "mr1.json").read_text())
    assert json.loads((tmp_path / "profile.json").read_text())["step4_profile_criteria"] is None
    assert "Step-4 criteria" not in text and "MR1 release check (descriptive; not step-4 evidence" in text
    assert report["profile_status"] == "complete" and report["functional_smoke"] == "pass"
    assert {c: r["outcome"] for c, r in report["releases"].items()} == dict.fromkeys(
        ("scene", "detector", "face"), "memfree_rose_cached_fell")
    assert report["after_smoke_vs_face_release"]["MemFree"] == -50_000_000
    assert any("per-file page-cache residency" in item for item in report["cannot_establish"])


def test_a_partial_mr1_run_reports_what_was_not_reached(mods, tmp_path) -> None:
    profile = mods.profile
    write_mr1_run(profile, tmp_path, smoke=False, releases=("scene",))
    text = profile.summarize(tmp_path)
    report = json.loads((tmp_path / "mr1.json").read_text())
    assert set(report["releases"]) == {"scene"} and report["functional_smoke"] == "unavailable"
    assert report["after_smoke"] is None and report["after_smoke_vs_face_release"] is None
    assert "detector: not reached" in text and "face: not reached" in text
