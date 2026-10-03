"""V2-09/V2-10 demo form: the legacy Ultralytics backend with fake torch/YOLO modules (no GPU, no engine)."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sentinel.adapters import AdapterManifest, AdapterState, load, resolve
from sentinel.inference.legacy_ultralytics import (
    L4T_LIBCUDA_DIR,
    TRACK_ARGS,
    LegacyUltralyticsTracker,
    cuda_driver_problem,
)
from sentinel.tracking.tracker import RawTrack, TrackerError

ROOT = Path(__file__).resolve().parents[2]
L4T = f"{L4T_LIBCUDA_DIR}libcuda.so.1.1"
UBUNTU_535 = "/usr/lib/aarch64-linux-gnu/libcuda.so.535.309.01"


def maps_line(path: str) -> str:
    return f"ffff8a000000-ffff8a100000 r-xp 00000000 b3:02 1234 {path}"


class Tensor:
    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def cpu(self) -> Tensor:
        return self

    def tolist(self) -> list[Any]:
        return self.values


class FakeByteTrack:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


class FakeYolo:
    instances: list[FakeYolo] = []

    def __init__(self, path: str, task: str) -> None:
        self.path, self.task = path, task
        self.calls: list[tuple[object, dict]] = []
        self.outputs: list[object] = []
        self.predictor: Any = None
        FakeYolo.instances.append(self)

    def track(self, image: object, **kwargs: object) -> list[object]:
        self.calls.append((image, kwargs))
        if self.predictor is None:  # Ultralytics creates its trackers on the first track() call
            self.predictor = SimpleNamespace(trackers=[FakeByteTrack()])
        return self.outputs.pop(0) if self.outputs else [SimpleNamespace(boxes=None)]


def boxes(xyxy: list[list[float]], conf: list[float], ids: list[float] | None) -> list[object]:
    return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor(xyxy), conf=Tensor(conf), id=None if ids is None else Tensor(ids)))]


class Runtime:
    """What the default importer returns: (torch, YOLO, numpy)."""

    def __init__(self, cuda: bool = True) -> None:
        self.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda))
        self.numpy = SimpleNamespace(uint8="uint8", zeros=lambda shape, dtype: ("blank", shape, dtype))
        self.imported = 0
        self.offline_at_import: str | None = None

    def __call__(self) -> tuple[Any, Any, Any]:
        self.imported += 1
        self.offline_at_import = os.environ.get("YOLO_OFFLINE")
        return self.torch, FakeYolo, self.numpy


@pytest.fixture
def engine(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "yolov8n.engine"
    path.write_bytes(b"not a real engine")
    return path, hashlib.sha256(b"not a real engine").hexdigest()


@pytest.fixture(autouse=True)
def clean_yolo(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeYolo.instances = []
    monkeypatch.setenv("YOLO_OFFLINE", "operator-value")  # restored after each test


def backend(engine: tuple[Path, str], runtime: Runtime, **kwargs: Any) -> LegacyUltralyticsTracker:
    path, digest = engine
    return LegacyUltralyticsTracker(path, expected_sha256=digest, driver_check=lambda: None, importer=runtime, **kwargs)


def test_load_builds_the_engine_with_v1s_arguments_warms_up_then_resets(engine: tuple[Path, str]) -> None:
    runtime = Runtime()
    tracker = backend(engine, runtime)

    tracker.load()

    (model,) = FakeYolo.instances
    assert (model.path, model.task) == (str(engine[0]), "detect")
    assert [kwargs for _, kwargs in model.calls] == [TRACK_ARGS] * 3
    assert model.calls[0][0] == ("blank", (480, 640, 3), "uint8")
    assert model.predictor.trackers[0].resets == 1  # warm-up frames leave no tracker state behind
    assert runtime.offline_at_import == "true"  # no analytics or online checks from Ultralytics


def test_track_args_match_the_arguments_check_8_profiled() -> None:
    spec = importlib.util.spec_from_file_location("demo_workload_args", ROOT / "benchmarks/runner/demo_workload.py")
    workload = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(workload)  # type: ignore[union-attr]
    assert TRACK_ARGS == workload.TRACK_ARGS
    assert TRACK_ARGS["conf"] == 0.4 and TRACK_ARGS["classes"] == [0] and TRACK_ARGS["persist"] is True


def test_the_driver_guard_runs_before_anything_is_imported_or_hashed(engine: tuple[Path, str]) -> None:
    runtime = Runtime()
    path, digest = engine
    tracker = LegacyUltralyticsTracker(path, expected_sha256=digest, driver_check=lambda: "libcuda_not_l4t", importer=runtime)
    with pytest.raises(TrackerError) as caught:
        tracker.load()
    assert caught.value.label == "libcuda_not_l4t" and runtime.imported == 0


def test_only_the_profiled_engine_is_loaded(engine: tuple[Path, str], tmp_path: Path) -> None:
    runtime = Runtime()
    wrong = LegacyUltralyticsTracker(engine[0], expected_sha256="0" * 64, driver_check=lambda: None, importer=runtime)
    with pytest.raises(TrackerError) as caught:
        wrong.load()
    assert caught.value.label == "engine_hash_mismatch"
    missing = LegacyUltralyticsTracker(tmp_path / "absent.engine", driver_check=lambda: None, importer=runtime)
    with pytest.raises(TrackerError) as caught:
        missing.load()
    assert (caught.value.label, caught.value.error_type) == ("engine_unreadable", "FileNotFoundError")
    assert runtime.imported == 0 and FakeYolo.instances == []


def test_missing_runtime_or_cuda_and_load_errors_are_labels(engine: tuple[Path, str]) -> None:
    def no_runtime() -> tuple[Any, Any, Any]:
        raise ImportError("No module named 'ultralytics' in /home/user/env")

    with pytest.raises(TrackerError) as caught:
        backend(engine, no_runtime).load()  # type: ignore[arg-type]
    assert (caught.value.label, caught.value.error_type) == ("runtime_unavailable", "ImportError")

    with pytest.raises(TrackerError) as caught:
        backend(engine, Runtime(cuda=False)).load()
    assert caught.value.label == "torch_cuda_unavailable" and FakeYolo.instances == []

    class Broken(Runtime):
        def __call__(self) -> tuple[Any, Any, Any]:
            torch, _, numpy = super().__call__()

            def explode(path: str, task: str) -> None:
                raise RuntimeError(f"failed to deserialize {path}")

            return torch, explode, numpy

    with pytest.raises(TrackerError) as caught:
        backend(engine, Broken()).load()
    assert (caught.value.label, caught.value.error_type) == ("load_failed", "RuntimeError")
    assert "deserialize" not in str(caught.value)


def test_track_returns_tracked_boxes_only(engine: tuple[Path, str]) -> None:
    tracker = backend(engine, Runtime())
    with pytest.raises(TrackerError) as caught:
        tracker.track("frame")
    assert caught.value.label == "not_loaded"
    tracker.load()
    model = FakeYolo.instances[0]
    model.outputs = [
        boxes([[10.0, 20.0, 110.0, 220.0], [300.0, 40.0, 360.0, 300.0]], [0.91, 0.47], [4.0, 7.0]),
        boxes([[10.0, 20.0, 110.0, 220.0]], [0.35], None),  # detections the tracker did not take
        [SimpleNamespace(boxes=None)],
        [],
    ]
    assert tracker.track("frame-1") == [RawTrack(4, 10.0, 20.0, 110.0, 220.0, 0.91), RawTrack(7, 300.0, 40.0, 360.0, 300.0, 0.47)]
    assert type(tracker.track("frame-1b")) is list and tracker.track("frame-2") == [] and tracker.track("frame-3") == []
    assert model.calls[3] == ("frame-1", TRACK_ARGS)


@pytest.mark.parametrize(
    "output",
    [
        boxes([[1.0, 2.0, 3.0, 4.0]], [0.9, 0.8], [1.0]),  # lengths differ
        boxes([[1.0, 2.0, 3.0, 4.0]], [0.9], [1.5]),  # a fractional track ID
    ],
)
def test_inconsistent_ultralytics_output_is_refused(engine: tuple[Path, str], output: list[object]) -> None:
    tracker = backend(engine, Runtime())
    tracker.load()
    FakeYolo.instances[0].outputs = [output]
    with pytest.raises(TrackerError) as caught:
        tracker.track("frame")
    assert caught.value.label == "inconsistent_output"


def test_reset_clears_ultralytics_trackers_and_is_safe_before_load(engine: tuple[Path, str]) -> None:
    tracker = backend(engine, Runtime())
    tracker.reset()  # nothing loaded: nothing to reset
    tracker.load()
    tracker.reset()
    assert FakeYolo.instances[0].predictor.trackers[0].resets == 2


@pytest.mark.parametrize(
    ("rc", "mapped", "problem"),
    [
        (0, [L4T], None),
        (0, [L4T, L4T], None),
        (100, [UBUNTU_535], "libcuda_not_l4t"),  # check 3a: the desktop-GPU library shadows L4T's
        (0, [L4T, UBUNTU_535], "libcuda_not_l4t"),
        (100, [L4T], "cuinit_failed"),
        (0, [], "libcuda_not_l4t"),
    ],
)
def test_the_d27_driver_guard(rc: int, mapped: list[str], problem: str | None) -> None:
    text = "\n".join([maps_line("/usr/lib/libc.so.6"), *map(maps_line, mapped)])
    assert cuda_driver_problem(cuinit=lambda: rc, maps=lambda: text) == problem


def test_the_driver_guard_reports_a_missing_library() -> None:
    def absent() -> int:
        raise OSError("libcuda.so.1: cannot open shared object file")

    assert cuda_driver_problem(cuinit=absent, maps=lambda: "") == "libcuda_unavailable"


def test_the_registry_admits_the_legacy_detector_only_with_the_provisional_profile() -> None:
    manifest = {
        "adapter_id": "legacy-yolov8n-bytetrack",
        "contract_version": 1,
        "implementation_revision": "1",
        "enabled": True,
        "input_kinds": ["frame"],
        "output_kinds": ["person.track"],
        "model_revision": "yolov8n.engine-08370639",
        "resource_profile_id": "provisional-demo-20261003T085010Z",
        "timeout_ms": 1000,
    }
    (status,) = resolve([AdapterManifest.model_validate(manifest)])
    assert status.state is AdapterState.ENABLED
    implementation, loaded = load(status)  # imports the module only: no torch, no Ultralytics
    assert implementation is LegacyUltralyticsTracker and loaded.state is AdapterState.ENABLED

    (unknown,) = resolve([AdapterManifest.model_validate({**manifest, "resource_profile_id": "unmeasured"})])
    assert unknown.state is AdapterState.UNAVAILABLE
    assert unknown.reason == "no measured resource profile; unknown profiles are unavailable"
