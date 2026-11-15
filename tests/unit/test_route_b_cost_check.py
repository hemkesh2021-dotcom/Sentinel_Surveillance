"""Route B's offline cost check (V2, session 116, part 17): its phases, memory reading, refusals and reading, with fakes. No NumPy,
OpenCV or model runs here; one test makes the synthetic pictures where NumPy exists (the device's environment)."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import sys
from pathlib import Path

import pytest

from sentinel.media.clock import FakeClock

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
sys.path.insert(0, str(RUNNERS))  # as the other runner tools' tests import theirs; removed again at once
try:
    rc = importlib.import_module("route_b_cost_check")
finally:
    sys.path.remove(str(RUNNERS))


class Picture:
    """A picture's stand-in: shape and size only."""

    def __init__(self, width: int, height: int) -> None:
        self.shape = (height, width, 3)
        self.nbytes = width * height * 3

    def __getitem__(self, key):
        rows, cols = key
        return Picture(cols.stop - cols.start, rows.stop - rows.start)

    def copy(self) -> Picture:
        return Picture(self.shape[1], self.shape[0])

    def setflags(self, write: bool) -> None:
        pass


class Detector:
    def __init__(self) -> None:
        self.sizes: list = []

    def setInputSize(self, size) -> None:  # noqa: N802
        self.sizes.append(size)

    def setScoreThreshold(self, value) -> None:  # noqa: N802
        assert value == rc.FULL_SCALE_FLOOR

    def detect(self, image):
        assert image.shape[:2] == (self.sizes[-1][1], self.sizes[-1][0])  # each window at its own input size
        return 1, [[0.0] * 15]  # one candidate, to see it counted


VERIFIED = {"verified": True, "reason": None}


def run(arguments, made, **kwargs):
    return rc.run(arguments, made, thp=lambda: VERIFIED, emit=lambda line: None, **kwargs)


def args(tmp_path, **over) -> argparse.Namespace:
    base = {"out": tmp_path / "route-b-cost.json", "weights": None, "stand_in": True, "pictures": 2, "resize_calls": 5,
            "window_calls": 3, "ticks": 4}
    return argparse.Namespace(**{**base, **over})


def pieces(**over):
    ticks = itertools.count(0, 1_000_000)  # every reading of the clock is 1 ms later
    cpu = itertools.count(0, 2_000_000)  # and 2 ms of CPU: 2 cores
    rss = itertools.count(100_000_000, 1_000_000)
    detector = Detector()

    def memory():
        value = next(rss)
        return {"VmRSS": value, "VmHWM": value + 500_000, "RssAnon": value // 2, "RssFile": value // 2, "RssShmem": 0,
                "Pss": value - 1_000, "Pss_Anon": value // 2, "Pss_File": value // 2}

    made = {"numpy": None, "cv2": None, "resize": lambda image, w, h: Picture(w, h), "perf": lambda: next(ticks),
            "cpu": lambda: next(cpu), "memory": memory, "reset_peak": lambda: True,
            "pictures": lambda numpy, count: [Picture(rc.WIDTH, rc.HEIGHT) for _ in range(count)],
            "clock": FakeClock(), "detector": lambda: detector, "sensors": lambda: {"zones_c": {"cpu": 40.0},
                                                                                   "cpu_khz": [1_000], "gpu_hz": None},
            "system": lambda: {"MemAvailable": 9_000_000_000, "MemFree": 8_000_000_000},
            "versions": {"numpy": "fake", "opencv": "fake", "opencv_threads": 6}}
    made.update(over)
    return made, detector


def test_every_phase_runs_in_order_and_each_window_is_detected_at_its_own_size(tmp_path) -> None:
    made, detector = pieces()
    result = run(args(tmp_path, stand_in=False), made)
    assert result["verdict"] == "complete" and result["status"] == "finished" and result["problems"] == []
    assert [p["phase"] for p in result["phases"]] == list(rc.EXPECTED)
    windows = {p["phase"]: p for p in result["phases"]}["windows"]["layouts"]
    assert windows["one_far"]["windows"] == [[640, 640]] and len(windows["four_far"]["windows"]) == 4
    near = windows["near_and_far"]
    assert len(near["windows"]) == 2 and min(near["scales"]) < 1.0 and max(max(s) for s in near["windows"]) == 640
    steady = {p["phase"]: p for p in result["phases"]}["steady:four_far"]
    assert steady["ticks"]["n"] == 4 and steady["first_tick_ms"] is not None  # the first tick timed apart
    assert steady["candidates"] == 5 * 4  # every window's candidates counted, the first tick's too
    assert {tuple(s) for s in detector.sizes} >= {(640, 640)}
    on_disk = json.loads((tmp_path / "route-b-cost.json").read_text())
    assert on_disk["verdict"] == "complete" and on_disk["stand_in"] is False


def test_each_phase_has_its_own_memory_and_cpu_and_the_model_cost_is_a_difference(tmp_path) -> None:
    made, _ = pieces()
    result = run(args(tmp_path), made)
    for phase in result["phases"]:
        assert phase["peak_reset"] is True and phase["memory_before"]["VmRSS"] < phase["memory_after"]["VmRSS"]
        assert phase["peak_bytes"] == phase["memory_after"]["VmHWM"] and phase["cores"] > 0  # CPU over wall
    derived = result["derived"]
    load = {p["phase"]: p for p in result["phases"]}
    assert derived["model_bytes_rss"] == (load["first_detection"]["memory_after"]["VmRSS"]
                                          - load["model_load"]["memory_before"]["VmRSS"])
    assert "assumes" in derived["resize_at_20_fps"] and "assumes" in derived["detection_per_tick"]


def test_the_weights_are_checked_before_any_detector_is_built(tmp_path) -> None:
    built = []
    missing = tmp_path / "absent.onnx"
    with pytest.raises(rc.Refused, match="weights_unreadable"):
        rc.checked_detector(missing, built.append)
    other = tmp_path / "other.onnx"
    other.write_bytes(b"not the pinned detector")
    with pytest.raises(rc.Refused, match="weights_hash_mismatch"):
        rc.checked_detector(other, built.append)
    assert built == []
    pinned = tmp_path / "pinned.onnx"
    pinned.write_bytes(b"synthetic stand-in for the pinned weights")
    rc.checked_detector(pinned, built.append, expected=hashlib.sha256(pinned.read_bytes()).hexdigest())
    assert built == [pinned]
    assert rc.DETECTOR_WEIGHTS == ("face_detection_yunet_2023mar.onnx",
                                   "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4")


def test_a_refused_model_keeps_what_was_measured_and_reads_failed(tmp_path) -> None:
    def refuse():
        raise rc.Refused("weights_hash_mismatch")

    made, _ = pieces(detector=refuse)
    result = run(args(tmp_path, stand_in=False), made)
    assert result["status"] == "refused:weights_hash_mismatch" and result["verdict"] == "failed"
    assert [p["phase"] for p in result["phases"]][:4] == ["baseline", "inputs", "resize", "windows"]
    lines, code = rc.reading(tmp_path)
    assert code == 1 and any("execution: failed" in line for line in lines)


def test_a_phase_that_raises_is_kept_and_an_unreset_peak_is_named(tmp_path) -> None:
    calls = itertools.count()

    def resize(image, w, h):
        if next(calls) == 12:
            raise RuntimeError("never kept")
        return Picture(w, h)

    made, _ = pieces(resize=resize)
    result = run(args(tmp_path), made)
    assert result["status"] == "failed:RuntimeError" and result["verdict"] == "incomplete"
    resize_phase = {p["phase"]: p for p in result["phases"]}["resize"]
    assert resize_phase["ended"] == "error:RuntimeError" and "never kept" not in json.dumps(result)
    made, _ = pieces(reset_peak=lambda: False)
    result = run(args(tmp_path), made)
    assert result["verdict"] == "incomplete" and "peak_not_reset:baseline" in result["problems"]


def test_the_reading_says_stand_in_first_and_names_its_limits_and_estimates(tmp_path) -> None:
    made, _ = pieces()
    run(args(tmp_path), made)
    (tmp_path / "result.json").write_text(json.dumps({"status": "child_exited", "guard": {"trigger": None},
                                                      "child": {"returncode": 0}}))
    lines, code = rc.reading(tmp_path)
    assert code == 0 and lines[1].startswith("STAND-IN DETECTOR")
    assert sum(line.startswith("LIMIT:") for line in lines) == len(rc.LIMITS) >= 5
    assert any(line.startswith("ESTIMATE resize_at_20_fps") for line in lines)
    assert lines[-1].startswith("route B cost check: execution complete") and "no limit, gate or admission" in lines[-1]
    text = "\n".join(lines)
    assert "inputs:" in text and "scaffolding, not a runtime cost" in text
    (tmp_path / "result.json").write_text(json.dumps({"status": "refused:memfree_floor"}))
    lines, code = rc.reading(tmp_path)
    assert code == 1 and lines[0].startswith("REFUSED the guard")


def test_memory_fields_are_read_in_bytes_and_an_unreadable_field_is_none(tmp_path) -> None:
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmHWM:\t  204800 kB\nVmRSS:\t  102400 kB\nRssAnon:\t 51200 kB\n")
    rollup = tmp_path / "rollup"
    rollup.write_text("Rss:  1 kB\nPss:   99 kB\nPss_Anon:  9 kB\n")
    memory = rc.process_memory(status, rollup)
    assert memory["VmRSS"] == 102400 * 1024 and memory["VmHWM"] == 204800 * 1024 and memory["Pss"] == 99 * 1024
    assert memory["RssFile"] is None and memory["Pss_File"] is None
    assert rc.process_memory(tmp_path / "none", tmp_path / "none")["VmRSS"] is None
    target = tmp_path / "clear_refs"
    assert rc.reset_peak(target) is True and target.read_text() == "5"
    assert rc.reset_peak(tmp_path) is False  # a directory: not permitted, reported, never raised


def test_the_synthetic_pictures_are_seeded_and_faceless_noise_and_gradients() -> None:
    numpy = pytest.importorskip("numpy")
    first, second = rc.synthetic_pictures(numpy, 2), rc.synthetic_pictures(numpy, 2)
    assert all(p.shape == (rc.HEIGHT, rc.WIDTH, 3) and p.dtype == numpy.uint8 for p in first)
    assert all(numpy.array_equal(a, b) for a, b in zip(first, second, strict=True))  # the same seed, the same pictures


def test_the_memory_policy_is_set_and_verified_before_anything_is_measured(tmp_path) -> None:
    made, _ = pieces()
    lines: list = []
    result = rc.run(args(tmp_path), made, thp=lambda: {"verified": False, "reason": "model_modules_loaded"},
                    emit=lines.append)
    assert result["status"] == "refused:thp_disable_not_verified:model_modules_loaded" and result["phases"] == []
    assert result["verdict"] == "failed" and not any('"run": "starting"' in line for line in lines)  # never ready
    lines.clear()
    rc.run(args(tmp_path), made, thp=lambda: VERIFIED, emit=lines.append)
    assert json.loads(lines[0]) == {"run": "starting", "startup": {"tool": "route_b_cost_check", "stand_in": True}}
    assert json.loads(lines[-1])["route_b_cost_check"] == "complete"
    assert rc.main(["guard-duration"]) == 0 and rc.GUARD_DURATION_S <= 900  # within the guard's own bounds


def test_each_steady_phase_is_on_disk_before_the_next_and_measured_from_its_own_start(tmp_path) -> None:
    """Review findings (session 116, part 17): each steady phase's record is written after the phase ends, so an
    interruption keeps it; its peak is read against its own start, which holds the tool's window copies, counted apart."""
    class Boom(BaseException):
        pass

    made, detector = pieces()
    calls = itertools.count()
    original = detector.detect

    def detect(image):
        if next(calls) == 2 * (4 + 1) + 1:  # first_detection, one_far's 5 ticks, then the second layout's first window
            raise Boom
        return original(image)

    detector.detect = detect
    with pytest.raises(Boom):
        run(args(tmp_path), made)
    on_disk = json.loads((tmp_path / "route-b-cost.json").read_text())
    steady = {p["phase"]: p for p in on_disk["phases"]}.get("steady:one_far")
    assert steady is not None and steady["ticks"]["n"] == 4 and steady["held_copy_bytes"] > 0
    made, _ = pieces()
    result = run(args(tmp_path), made)
    assert "steady_growth_over_phase_start_bytes" in result["derived"]
    assert "steady_peak_over_after_first_bytes" not in result["derived"]
