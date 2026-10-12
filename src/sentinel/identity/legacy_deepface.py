"""Interim face adapter for the Oct 20 demo (V2-25 demo form, D34; full acceptance pending).

Runs the face path the accepted replay profile measured (``step4cand-demo-20261007T090339Z``; demo_workload.py's
FACE_ARGS): DeepFace 0.0.99 ``represent()`` on the whole frame, YuNet detection, alignment, base normalization and
Facenet512 embeddings, with TensorFlow on the CPU, once per second (D34). Nothing heavy is imported until load(),
which fails closed with a fixed label unless:

- both weight files exist and have the SHA-256 that profile recorded (the only weights this adapter loads);
- TensorFlow sees no GPU after ``set_visible_devices([], "GPU")`` (v1 kept it on the CPU to leave the GPU to the
  detector and llama-server; the profile measured it that way);
- the DeepFace version is the profiled one.

Output: one ``FaceObservation`` per detected face. With ``enforce_detection=False`` DeepFace returns the whole image
as a face with confidence 0 when it finds none (deepface/modules/detection.py); such results are dropped and counted,
because that full-frame fallback is v1's identity-transfer defect. Quality is YuNet's detection confidence (a
starting gate, not a calibrated quality measure); embeddings are scaled to unit length. Exception text is never kept.

The compatibility record of the loaded pipeline (``compatibility()``) must equal the gallery's in every field before
any matching (gallery.FaceCompatibility).

**Full-scale pass (D86, validation runs only, diagnostic).** With ``align=True`` DeepFace 0.0.99 pads the frame with a
black border of half its height and width before detection (``detection.detect_faces``), and its YuNet wrapper then
scales anything wider or taller than 640 pixels down to 640 (``YuNetClient.detect_faces``): a 640x480 frame becomes
1280x960 and is detected at half scale, keeping only scores of at least 0.9. The embedding itself is cut from the
padded native-resolution picture, so only detection is affected. ``enable_full_scale()`` runs the same pinned YuNet
weights once more on the frame itself, at native scale, and reports every candidate scoring at least
FULL_SCALE_FLOOR (a recording floor, never a decision). Its result never becomes a vote; it says whether faces the
production pass missed are there at full scale, and at what score and size.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..contracts import FrameKey, NormalizedBox
from .association import FaceObservation
from .gallery import FaceCompatibility

DEEPFACE_VERSION = "0.0.99"
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "yunet"
DIMENSION = 512
# As the accepted replay profile's provenance recorded them (session 46 run, files unchanged on 2026-10-07).
MODEL_WEIGHTS = ("facenet512_weights.h5", "3f76b5117a9ca574d536af8199e6720089eb4ad3dc7e93534496d88265de864f")
DETECTOR_WEIGHTS = ("face_detection_yunet_2023mar.onnx",
                    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4")
FACE_ARGS: Mapping[str, Any] = {  # demo_workload.py's FACE_ARGS with DeepFace's defaults written out
    "model_name": MODEL_NAME, "detector_backend": DETECTOR_BACKEND, "enforce_detection": False,
    "align": True, "normalization": "base", "expand_percentage": 0,
}
WARMUP_SHAPE = (480, 640, 3)
YUNET_THRESHOLD = 0.9  # DeepFace 0.0.99's default yunet_score_threshold: what production detection keeps
YUNET_MAX_SIDE = 640  # DeepFace's YuNet wrapper scales anything larger down to this
FULL_SCALE_FLOOR = 0.3  # D86: what the diagnostic full-scale pass records; never a decision threshold
FULL_SCALE_KEPT = 8  # candidates kept per frame, highest score first


class FaceBackendError(Exception):
    """A fixed label and, at most, an exception class name."""

    def __init__(self, label: str, error_type: str | None = None) -> None:
        super().__init__(label if error_type is None else f"{label} ({error_type})")
        self.label = label
        self.error_type = error_type


@dataclass(frozen=True)
class FullScale:
    """D86: the pinned YuNet on the frame itself, at native scale: (score, box) candidates at or above
    FULL_SCALE_FLOOR, highest first, and how long the pass took. Diagnostic only."""

    candidates: tuple[tuple[float, NormalizedBox], ...]
    ms: float


@dataclass(frozen=True)
class FaceRun:
    """One frame's faces and what was dropped."""

    faces: tuple[FaceObservation, ...]
    detected: int  # results with a confidence above 0
    fallback_dropped: int  # DeepFace's whole-image "face" with confidence 0
    full_scale: FullScale | None = None  # D86, validation runs only


@dataclass(frozen=True)
class EnrollmentFace:
    embedding: tuple[float, ...]  # unit length
    quality: float
    face_width_px: int


def default_weights_dir(environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    return Path(environ.get("DEEPFACE_HOME", str(Path.home()))) / ".deepface" / "weights"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def static_compatibility_fields() -> dict[str, Any]:
    """The pinned part of the record, known without importing anything."""
    return {"deepface_version": DEEPFACE_VERSION, "model_name": MODEL_NAME, "model_sha256": MODEL_WEIGHTS[1],
            "detector_backend": DETECTOR_BACKEND, "detector_sha256": DETECTOR_WEIGHTS[1],
            "align": FACE_ARGS["align"], "normalization": FACE_ARGS["normalization"],
            "expand_percentage": FACE_ARGS["expand_percentage"], "color_order": "BGR", "dimension": DIMENSION,
            "prototype_normalization": "l2"}


def static_difference(compatibility: FaceCompatibility) -> str | None:
    """The first pinned field a gallery does not share with this adapter (checked before any model loads)."""
    recorded = compatibility.model_dump()
    for name, value in static_compatibility_fields().items():
        if recorded[name] != value:
            return name
    return None


def _unit(values: Sequence[Any]) -> tuple[float, ...] | None:
    try:
        vector = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if len(vector) != DIMENSION or not all(math.isfinite(v) for v in vector):
        return None
    norm = math.sqrt(math.fsum(v * v for v in vector))
    if norm == 0.0:
        return None
    return tuple(v / norm for v in vector)


def _confidence(item: Mapping[str, Any]) -> float:
    try:
        value = float(item.get("face_confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, value) if math.isfinite(value) and value > 0 else 0.0


def _pixel_box(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
    area = item.get("facial_area")
    if not isinstance(area, Mapping):
        raise FaceBackendError("inconsistent_output")
    try:
        x, y, w, h = (float(area[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError):
        raise FaceBackendError("inconsistent_output") from None
    return x, y, w, h


def to_face_run(frame: FrameKey, width: int, height: int, raw: Sequence[Mapping[str, Any]]) -> FaceRun:
    """DeepFace ``represent()`` output for one frame as FaceObservations; confidence-0 fallbacks are dropped."""
    faces, detected, dropped = [], 0, 0
    for item in raw:
        confidence = _confidence(item)
        if confidence == 0.0:
            dropped += 1
            continue
        detected += 1
        x, y, w, h = _pixel_box(item)
        x1, y1 = max(0.0, x / width), max(0.0, y / height)
        x2, y2 = min(1.0, (x + w) / width), min(1.0, (y + h) / height)
        embedding = _unit(item.get("embedding") or ())
        if embedding is None:
            raise FaceBackendError("inconsistent_output")
        if not (x1 < x2 and y1 < y2):
            continue  # nothing of it inside the frame
        faces.append(FaceObservation(frame=frame, box=NormalizedBox(x1=x1, y1=y1, x2=x2, y2=y2),
                                     quality=confidence, embedding=embedding))
    return FaceRun(tuple(faces), detected, dropped)


def enrollment_face(raw: Sequence[Mapping[str, Any]]) -> tuple[EnrollmentFace | None, str]:
    """The single face of an enrollment photo, or None and why (``no_face``, ``several_faces``)."""
    found = [item for item in raw if _confidence(item) > 0.0]
    if not found:
        return None, "no_face"
    if len(found) > 1:
        return None, "several_faces"
    item = found[0]
    embedding = _unit(item.get("embedding") or ())
    if embedding is None:
        raise FaceBackendError("inconsistent_output")
    return EnrollmentFace(embedding, _confidence(item), round(_pixel_box(item)[2])), "face_found"


def _import_runtime() -> SimpleNamespace:
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # before any TensorFlow operation: the CPU only, as profiled
    import cv2
    import numpy
    import tf_keras
    from deepface import DeepFace

    import deepface

    return SimpleNamespace(tf=tf, cv2=cv2, numpy=numpy, tf_keras=tf_keras, deepface=deepface, DeepFace=DeepFace)


class LegacyDeepFaceBackend:
    """DeepFace's face path as the replay profile ran it; call load() first."""

    def __init__(
        self,
        weights_dir: Path | None = None,
        *,
        pins: Sequence[tuple[str, str]] = (MODEL_WEIGHTS, DETECTOR_WEIGHTS),
        importer: Callable[[], SimpleNamespace] = _import_runtime,
        sha256: Callable[[Path], str] = file_sha256,
    ) -> None:
        self._weights_dir = default_weights_dir() if weights_dir is None else Path(weights_dir)
        self._pins = tuple(pins)
        self._importer = importer
        self._sha256 = sha256
        self._runtime: SimpleNamespace | None = None
        self._compatibility: FaceCompatibility | None = None
        self._full_scale_wanted = False
        self._full_scale: Any = None  # cv2.FaceDetectorYN on the pinned detector weights, when enabled

    def __repr__(self) -> str:
        return f"LegacyDeepFaceBackend(loaded={self._runtime is not None})"

    @property
    def weight_files(self) -> dict[str, Path]:
        """The files sentinel.memory_policy releases after the load (its ``face`` roles)."""
        return {name: self._weights_dir / name for name, _ in self._pins}

    def enable_full_scale(self) -> None:
        """D86: also run the full-scale diagnostic pass (call before load(); validation runs only)."""
        if self._runtime is not None:
            raise FaceBackendError("already_loaded")
        self._full_scale_wanted = True

    def load(self) -> None:
        for name, expected in self._pins:
            try:
                digest = self._sha256(self._weights_dir / name)
            except OSError as exc:
                raise FaceBackendError("weights_unreadable", type(exc).__name__) from None
            if digest != expected:
                raise FaceBackendError("weights_hash_mismatch")
        try:
            runtime = self._importer()
        except ImportError as exc:
            raise FaceBackendError("runtime_unavailable", type(exc).__name__) from None
        except Exception as exc:  # set_visible_devices raises once TensorFlow has initialized devices
            raise FaceBackendError("tensorflow_cpu_pin_failed", type(exc).__name__) from None
        if runtime.tf.config.get_visible_devices("GPU"):
            raise FaceBackendError("tensorflow_gpu_visible")
        if getattr(runtime.deepface, "__version__", None) != DEEPFACE_VERSION:
            raise FaceBackendError("deepface_version_mismatch")
        try:
            runtime.DeepFace.build_model(MODEL_NAME)
            runtime.DeepFace.represent(img_path=runtime.numpy.zeros(WARMUP_SHAPE, runtime.numpy.uint8), **FACE_ARGS)
            compatibility = FaceCompatibility(
                **static_compatibility_fields(),
                opencv_version=str(runtime.cv2.__version__), tensorflow_version=str(runtime.tf.__version__),
                tf_keras_version=str(runtime.tf_keras.__version__), numpy_version=str(runtime.numpy.__version__),
            )
            if self._full_scale_wanted:  # the same weights file, its SHA-256 checked above
                self._full_scale = runtime.cv2.FaceDetectorYN_create(
                    str(self._weights_dir / DETECTOR_WEIGHTS[0]), "", (0, 0), FULL_SCALE_FLOOR)
        except Exception as exc:
            raise FaceBackendError("load_failed", type(exc).__name__) from None
        self._runtime = runtime
        self._compatibility = compatibility

    def compatibility(self) -> FaceCompatibility:
        if self._compatibility is None:
            raise FaceBackendError("not_loaded")
        return self._compatibility

    def faces(self, frame: FrameKey, width: int, height: int, image: Any) -> FaceRun:
        """One represent() call on the whole frame (BGR, as decoded)."""
        if self._runtime is None:
            raise FaceBackendError("not_loaded")
        run = to_face_run(frame, width, height, self._runtime.DeepFace.represent(img_path=image, **FACE_ARGS))
        if self._full_scale is None:
            return run
        return dataclasses.replace(run, full_scale=full_scale_pass(self._full_scale, width, height, image))

    def enrollment_photo(self, path: Path) -> tuple[EnrollmentFace | None, str]:
        """Decode a photo with the same OpenCV (EXIF orientation applied) and find its single face."""
        if self._runtime is None:
            raise FaceBackendError("not_loaded")
        image = self._runtime.cv2.imread(str(path), self._runtime.cv2.IMREAD_COLOR)
        if image is None:
            return None, "unreadable_photo"
        return enrollment_face(self._runtime.DeepFace.represent(img_path=image, **FACE_ARGS))


def full_scale_pass(detector: Any, width: int, height: int, image: Any) -> FullScale | None:
    """D86: one YuNet detection on the frame at native scale; None when the frame is larger than YuNet's 640-pixel
    limit (it would be scaled again) or the pass fails: a diagnostic never changes the production result."""
    if width > YUNET_MAX_SIDE or height > YUNET_MAX_SIDE:
        return None
    started = time.perf_counter()
    try:
        detector.setInputSize((width, height))
        detector.setScoreThreshold(FULL_SCALE_FLOOR)
        _, found = detector.detect(image)
    except Exception:  # noqa: BLE001 - the diagnostic is dropped, never the face result
        return None
    candidates = []
    for row in () if found is None else found:
        try:
            x, y, w, h, score = (float(row[i]) for i in (0, 1, 2, 3, -1))
        except (TypeError, ValueError, IndexError):
            return None
        x1, y1 = max(0.0, x / width), max(0.0, y / height)
        x2, y2 = min(1.0, (x + w) / width), min(1.0, (y + h) / height)
        if math.isfinite(score) and x1 < x2 and y1 < y2:
            candidates.append((min(1.0, max(0.0, score)), NormalizedBox(x1=x1, y1=y1, x2=x2, y2=y2)))
    candidates.sort(key=lambda c: -c[0])
    return FullScale(tuple(candidates[:FULL_SCALE_KEPT]), (time.perf_counter() - started) * 1000)
