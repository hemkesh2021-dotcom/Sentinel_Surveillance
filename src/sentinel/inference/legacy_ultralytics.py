"""Legacy parity detector and tracker for the Oct 20 demo (D24; V2-09/V2-10 demo form, full acceptance pending).

Runs v1's model the way v1 runs it: ``yolov8n.engine`` (TensorRT 10.3, FP32,
640x640, check 5) through Ultralytics ``YOLO.track()`` with ByteTrack and v1's
arguments. TRACK_ARGS equal the arguments check 8 measured for the provisional
demo profile (D33). This is the parity reference: it is not V2-09 proper (a
TensorRT adapter without torch, with fixed buffers and a cached context) or
V2-10 proper (ByteTrack separated from Ultralytics, low-score boxes kept for
its second association stage). With ``conf`` 0.4, boxes below 0.4 never reach
ByteTrack, as in v1.

Device adapter: nothing heavy is imported until load(), which first checks:

- **D27:** ``libcuda.so.1`` must be L4T's and ``cuInit`` must return 0. This is
  checked with ctypes before torch or TensorRT touch CUDA, because TensorRT
  aborts the whole process with the wrong library (U13).
- **The engine file:** its SHA-256 must equal the engine that check 6 hashed
  and check 8 profiled. The provisional profile and parity apply to that file
  only, and it is the only engine this adapter deserializes.
- **No network:** ``YOLO_OFFLINE=true`` is set before Ultralytics is imported,
  so it sends no usage analytics and attempts no online checks or installs.
  Its settings on this device have analytics on, which v1 and check 8 ran with.

Errors are fixed labels with an exception class name; exception text is never kept.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..tracking.tracker import RawTrack, TrackerError

TRACK_ARGS = {  # as surveillance4_1.py@2b2d639 line 501 calls yolo.track()
    "imgsz": 640,
    "device": 0,
    "half": True,  # no effect: the engine is FP32 (check 5)
    "verbose": False,
    "classes": [0],
    "conf": 0.4,
    "persist": True,
    "tracker": "bytetrack.yaml",
}
# yolov8n.engine as hashed in session 3 (check 6) and profiled by check 8 (D33).
LEGACY_ENGINE_SHA256 = "08370639f961d2c67148c19562718ef80527c7085e88d2d923176180f1b98637"
L4T_LIBCUDA_DIR = "/usr/lib/aarch64-linux-gnu/nvidia/"
WARMUP_FRAMES = 3
WARMUP_SHAPE = (480, 640, 3)


def cuda_driver_problem(
    cuinit: Callable[[], int] | None = None, maps: Callable[[], str] | None = None
) -> str | None:
    """D27 guard: None if L4T's libcuda is the only one mapped and cuInit succeeds, else a label."""
    try:
        rc = cuinit() if cuinit is not None else ctypes.CDLL("libcuda.so.1").cuInit(0)
    except OSError:
        return "libcuda_unavailable"
    text = maps() if maps is not None else Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
    mapped = {line.split()[-1] for line in text.splitlines() if "libcuda.so" in line}
    if not mapped or not all(path.startswith(L4T_LIBCUDA_DIR) for path in mapped):
        return "libcuda_not_l4t"
    return None if rc == 0 else "cuinit_failed"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_runtime() -> tuple[Any, Any, Any]:
    import numpy
    import torch
    from ultralytics import YOLO

    return torch, YOLO, numpy


class LegacyUltralyticsTracker:
    """A TrackerBackend over Ultralytics ``YOLO.track()``; call load() before track()."""

    def __init__(
        self,
        engine: Path,
        *,
        expected_sha256: str = LEGACY_ENGINE_SHA256,
        driver_check: Callable[[], str | None] = cuda_driver_problem,
        importer: Callable[[], tuple[Any, Any, Any]] = _import_runtime,
    ) -> None:
        self._engine = Path(engine)
        self._expected = expected_sha256
        self._driver_check = driver_check
        self._importer = importer
        self._model: Any = None

    def __repr__(self) -> str:
        return f"LegacyUltralyticsTracker(loaded={self._model is not None})"

    def load(self) -> None:
        problem = self._driver_check()
        if problem is not None:
            raise TrackerError(problem)
        try:
            digest = file_sha256(self._engine)
        except OSError as exc:
            raise TrackerError("engine_unreadable", type(exc).__name__) from None
        if digest != self._expected:
            raise TrackerError("engine_hash_mismatch")
        os.environ["YOLO_OFFLINE"] = "true"  # read when Ultralytics is first imported
        try:
            torch, yolo, numpy = self._importer()
        except ImportError as exc:
            raise TrackerError("runtime_unavailable", type(exc).__name__) from None
        if not torch.cuda.is_available():
            raise TrackerError("torch_cuda_unavailable")
        try:
            model = yolo(str(self._engine), task="detect")
            blank = numpy.zeros(WARMUP_SHAPE, numpy.uint8)
            for _ in range(WARMUP_FRAMES):
                model.track(blank, **TRACK_ARGS)
        except Exception as exc:
            raise TrackerError("load_failed", type(exc).__name__) from None
        self._model = model
        self.reset()

    def track(self, image: Any) -> list[RawTrack]:
        if self._model is None:
            raise TrackerError("not_loaded")
        results = self._model.track(image, **TRACK_ARGS)
        boxes = results[0].boxes if results else None
        # Without tracks, Ultralytics returns the untracked detections with id None (v1 skips them too).
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        ids = boxes.id.cpu().tolist()
        if not len(xyxy) == len(confidences) == len(ids):
            raise TrackerError("inconsistent_output")
        tracks = []
        for (x1, y1, x2, y2), confidence, track_id in zip(xyxy, confidences, ids):
            if not float(track_id).is_integer():
                raise TrackerError("inconsistent_output")
            tracks.append(RawTrack(int(track_id), x1, y1, x2, y2, confidence))
        return tracks

    def reset(self) -> None:
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or ():
            tracker.reset()  # clears tracked/lost tracks and restarts IDs
