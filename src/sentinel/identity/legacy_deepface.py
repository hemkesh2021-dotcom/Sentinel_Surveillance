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

**Full-scale comparison (D87, validation runs only, diagnostic).** ``enable_comparison()`` (after
``enable_full_scale()``) lets ``embed_full_scale()`` embed one full-scale candidate through the production pipeline
from detection on: the candidate's YuNet row becomes DeepFace's region exactly as ``YuNetClient.detect_faces`` reports
a detection on an unresized picture (``int()`` of the box and eyes, x and y clamped at 0, YuNet's columns 6-7 as
DeepFace's ``left_eye`` and 4-5 as ``right_eye``); it is moved into the frame padded as ``detection.detect_faces``
pads it (``copyMakeBorder``, half the height and width, black) and cut and aligned by DeepFace's own
``detection.extract_face`` (align, expand 0, the borders, ``yunet``); the crop goes to ``DeepFace.represent`` with
``detector_backend="skip"`` divided by 255, because the skip path, unlike ``extract_faces``, does not divide: from
there the crop takes production's exact steps (BGR to RGB and back, ``resize_image`` to 160x160, ``base``
normalization, the same ``Facenet512.forward``). Only the detection's scale and the landmarks' precision differ.
``represent`` takes the model from DeepFace's cache (``modeling.build_model``); the load fails closed unless the
production warm-up put Facenet512 there and these functions have the parameters read here, and every comparison checks
the cache first, so a second recognition model is never loaded. The embedding is returned in memory only; it is
compared with the enrollment elsewhere and never becomes a vote.

**Full-scale detection candidate (D88, validation runs only, selected explicitly).**
``select_full_scale_detection()`` replaces DeepFace's detection stage, for every face of every frame, instead of adding
a pass: the YuNet DeepFace's warm-up built from the pinned weights (taken from its model cache, never created again)
runs once on the frame at native scale, and only its detections scoring at least YUNET_THRESHOLD, production's own
threshold, become faces. Each is moved into the frame padded exactly as ``detection.detect_faces`` pads it, converted
as ``YuNetClient.detect_faces`` would convert a detection on that padded picture without resizing it (``candidate_region``:
``int()`` in the padded picture's coordinates, where its clamp at 0 never acts on a face inside the frame), cut and
aligned by DeepFace's own ``extract_face``, reported with ``extract_faces``' own clamps and its 2-decimal confidence,
divided by 255, and embedded in one ``represent(detector_backend="skip")`` call, which batches the frame's faces as
production's call does, with the cached Facenet512. From the aligned cut on, every step is production's; only the
detection's scale and its landmarks' precision differ (production's 640x480 frame is padded to 1280x960 and scaled to
half, so its boxes and eyes come in 2-pixel steps). A frame without a face at the threshold is never embedded:
production's whole-frame stand-in with confidence 0, which the adapter drops, does not exist on this path. Thresholds,
association, ownership, retention, expiry and ambiguity are downstream and unchanged, and so is ``compatibility()``:
the enrollment's model, alignment and normalization are this path's too. Failures are FaceBackendError labels, never a
partial result. D86's diagnostic pass and D87's comparison would repeat this detection and are refused with it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
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
YUNET_ROW_VALUES = 15  # x, y, w, h, five landmarks (x, y) and the score (cv2.FaceDetectorYN)
# D87: the DeepFace 0.0.99 functions the comparison calls, with the parameters it passes (checked at load), and the
# fields of the region it builds (deepface.models.Detector.FacialAreaRegion).
COMPARISON_PARAMETERS = {
    "extract_face": {"facial_area", "img", "align", "expand_percentage", "width_border", "height_border",
                     "detector_backend"},
    "represent": {"img_path", "model_name", "detector_backend", "enforce_detection", "align", "expand_percentage",
                  "normalization"},
}
REGION_FIELDS = {"x", "y", "w", "h", "left_eye", "right_eye", "confidence"}
# D88: the face detection stage. DETECTION_DEEPFACE is production's (DeepFace's own, on the padded frame);
# DETECTION_FULL_SCALE is the candidate, selected explicitly for validation runs only.
DETECTION_DEEPFACE = "deepface"
DETECTION_FULL_SCALE = "full_scale"
DETECTOR_METHODS = ("setInputSize", "setScoreThreshold", "detect")  # cv2.FaceDetectorYN, as DeepFace's client uses it
CONFIDENCE_DECIMALS = 2  # extract_faces reports round(confidence, 2), and represent passes that on


class FaceBackendError(Exception):
    """A fixed label and, at most, an exception class name."""

    def __init__(self, label: str, error_type: str | None = None) -> None:
        super().__init__(label if error_type is None else f"{label} ({error_type})")
        self.label = label
        self.error_type = error_type


@dataclass(frozen=True)
class FullScaleRow:
    """D87: one full-scale candidate's raw YuNet geometry in frame pixels, for the comparison only (memory, never a
    record): the box (columns 0-3) and the two eye landmarks (columns 4-5 and 6-7)."""

    x: float
    y: float
    w: float
    h: float
    eye_4: tuple[float, float]  # YuNet's columns 4-5: DeepFace's right_eye
    eye_6: tuple[float, float]  # YuNet's columns 6-7: DeepFace's left_eye
    score: float


@dataclass(frozen=True)
class FullScale:
    """D86: the pinned YuNet on the frame itself, at native scale: (score, box) candidates at or above
    FULL_SCALE_FLOOR, highest first, and how long the pass took. Diagnostic only. D87: ``rows`` holds each candidate's
    raw geometry in the same order (None where its row had no usable landmarks); empty when not recorded."""

    candidates: tuple[tuple[float, NormalizedBox], ...]
    ms: float
    rows: tuple[FullScaleRow | None, ...] = ()


@dataclass(frozen=True)
class CandidateDetection:
    """D88: the full-scale detection candidate's own numbers for one frame (validation records only): the best
    detection under YUNET_THRESHOLD and at least FULL_SCALE_FLOOR (never a face), how long detection took, and how long
    cutting, aligning and embedding the kept faces took (None when none was kept)."""

    best_below: tuple[float, NormalizedBox] | None
    detect_ms: float
    embed_ms: float | None


@dataclass(frozen=True)
class FaceRun:
    """One frame's faces and what was dropped."""

    faces: tuple[FaceObservation, ...]
    detected: int  # results with a confidence above 0
    fallback_dropped: int  # DeepFace's whole-image "face" with confidence 0
    full_scale: FullScale | None = None  # D86, validation runs only
    detection: CandidateDetection | None = None  # D88, the candidate's own numbers


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
    from deepface.models.Detector import FacialAreaRegion  # D87: imported by DeepFace itself; nothing new loads
    from deepface.modules import detection, modeling

    import deepface

    return SimpleNamespace(tf=tf, cv2=cv2, numpy=numpy, tf_keras=tf_keras, deepface=deepface, DeepFace=DeepFace,
                           detection=detection, modeling=modeling, FacialAreaRegion=FacialAreaRegion)


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
        self._comparison_wanted = False
        self._recognition_model: Any = None  # D87: the cached Facenet512 the warm-up built, when comparing
        self._detection = DETECTION_DEEPFACE
        self._candidate: tuple[Any, Any] | None = None  # D88: DeepFace's cached YuNet and Facenet512, when selected

    def __repr__(self) -> str:
        return f"LegacyDeepFaceBackend(loaded={self._runtime is not None})"

    @property
    def detection(self) -> str:
        """The face detection stage: DETECTION_DEEPFACE (production) or DETECTION_FULL_SCALE (D88)."""
        return self._detection

    def select_full_scale_detection(self) -> None:
        """D88: the full-scale detection candidate replaces DeepFace's detection (call before load(); validation runs
        only). It excludes D86's diagnostic pass and D87's comparison, which would only repeat its detection."""
        if self._runtime is not None:
            raise FaceBackendError("already_loaded")
        if self._full_scale_wanted:
            raise FaceBackendError("candidate_excludes_diagnostics")
        self._detection = DETECTION_FULL_SCALE

    @property
    def weight_files(self) -> dict[str, Path]:
        """The files sentinel.memory_policy releases after the load (its ``face`` roles)."""
        return {name: self._weights_dir / name for name, _ in self._pins}

    def enable_full_scale(self) -> None:
        """D86: also run the full-scale diagnostic pass (call before load(); validation runs only)."""
        if self._runtime is not None:
            raise FaceBackendError("already_loaded")
        if self._detection == DETECTION_FULL_SCALE:
            raise FaceBackendError("candidate_excludes_diagnostics")
        self._full_scale_wanted = True

    def enable_comparison(self) -> None:
        """D87: allow embed_full_scale() (call after enable_full_scale() and before load(); validation runs only)."""
        if self._runtime is not None:
            raise FaceBackendError("already_loaded")
        if not self._full_scale_wanted:
            raise FaceBackendError("comparison_needs_full_scale")
        self._comparison_wanted = True

    @property
    def comparison_enabled(self) -> bool:
        return self._recognition_model is not None

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
        recognition_model = candidate = None
        if self._comparison_wanted:
            recognition_model = _comparison_ready(runtime)  # raises a FaceBackendError label when not
        if self._detection == DETECTION_FULL_SCALE:
            candidate = _candidate_ready(runtime)  # D88: the warm-up's own models, or a FaceBackendError label
        self._runtime = runtime
        self._compatibility = compatibility
        self._recognition_model = recognition_model
        self._candidate = candidate

    def compatibility(self) -> FaceCompatibility:
        if self._compatibility is None:
            raise FaceBackendError("not_loaded")
        return self._compatibility

    def faces(self, frame: FrameKey, width: int, height: int, image: Any) -> FaceRun:
        """One represent() call on the whole frame (BGR, as decoded)."""
        if self._runtime is None:
            raise FaceBackendError("not_loaded")
        if self._candidate is not None:
            return self._candidate_faces(frame, width, height, image)
        run = to_face_run(frame, width, height, self._runtime.DeepFace.represent(img_path=image, **FACE_ARGS))
        if self._full_scale is None:
            return run
        return dataclasses.replace(run, full_scale=full_scale_pass(self._full_scale, width, height, image,
                                                                   rows=self._recognition_model is not None))

    def embed_full_scale(self, width: int, height: int, image: Any, row: FullScaleRow) -> tuple[float, ...]:
        """D87: the unit-length Facenet512 embedding of one full-scale candidate, cut, aligned, normalized and embedded
        as production would from that detection (see the module's comparison note). Raises a FaceBackendError with a
        fixed label (and at most an exception class name) on any failure; never loads a model."""
        runtime, model = self._runtime, self._recognition_model
        if runtime is None or model is None:
            raise FaceBackendError("comparison_not_enabled")
        if _cached_recognition_model(runtime) is not model:
            raise FaceBackendError("model_not_cached")  # represent() would build a second model: never called
        region, width_border, height_border = padded_region(deepface_region(row), width, height)
        try:
            padded = runtime.cv2.copyMakeBorder(image, height_border, height_border, width_border, width_border,
                                                runtime.cv2.BORDER_CONSTANT, value=[0, 0, 0])
            detected = runtime.detection.extract_face(
                facial_area=runtime.FacialAreaRegion(**region), img=padded, align=FACE_ARGS["align"],
                expand_percentage=FACE_ARGS["expand_percentage"], width_border=width_border,
                height_border=height_border, detector_backend=DETECTOR_BACKEND)
            del padded
            crop = detected.img
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                raise FaceBackendError("empty_crop")
            raw = runtime.DeepFace.represent(
                img_path=crop / 255, model_name=MODEL_NAME, detector_backend="skip", enforce_detection=False,
                align=FACE_ARGS["align"], expand_percentage=FACE_ARGS["expand_percentage"],
                normalization=FACE_ARGS["normalization"])
        except FaceBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - a fixed label; the production result is never touched
            raise FaceBackendError("comparison_failed", type(exc).__name__) from None
        if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], Mapping):
            raise FaceBackendError("inconsistent_output")
        embedding = _unit(raw[0].get("embedding") or ())
        if embedding is None:
            raise FaceBackendError("inconsistent_output")
        return embedding

    def _candidate_faces(self, frame: FrameKey, width: int, height: int, image: Any) -> FaceRun:
        """D88: one frame through the candidate (the module's note): detection at native scale, then production's
        own cut, alignment, normalization and model. Raises a FaceBackendError label on any failure; never a partial
        result, and never a model built or loaded."""
        runtime = self._runtime
        assert runtime is not None and self._candidate is not None
        detector, model = self._candidate
        if _cached_detector(runtime) is not detector:
            raise FaceBackendError("detector_not_cached")
        if _cached_recognition_model(runtime) is not model:
            raise FaceBackendError("model_not_cached")  # represent() would build a second model: never called
        if width > YUNET_MAX_SIDE or height > YUNET_MAX_SIDE:
            raise FaceBackendError("frame_too_large")  # YuNet would need it scaled: that is not this candidate
        started = time.perf_counter()
        try:
            detector.setInputSize((width, height))
            detector.setScoreThreshold(FULL_SCALE_FLOOR)  # DeepFace's client sets its own before each use
            _, found = detector.detect(image)
        except Exception as exc:  # noqa: BLE001 - a fixed label
            raise FaceBackendError("candidate_failed", type(exc).__name__) from None
        rows = candidate_rows(found)
        detect_ms = (time.perf_counter() - started) * 1000
        # Production's detections: OpenCV keeps a float32 score above its float 0.9, which is the same set as these
        # scores at or above 0.9 in double precision; greedy NMS never lets a lower score suppress a higher one, so
        # the recording floor leaves this set as a 0.9 threshold would give it.
        kept = [row for row in rows if row.score >= YUNET_THRESHOLD]
        below = [(row.score, box) for row in rows if row.score < YUNET_THRESHOLD
                 for box in (_row_box(row, width, height),) if box is not None]
        items: list[dict[str, Any]] = []
        embed_ms = None
        if kept:
            started = time.perf_counter()
            items = self._embed_candidates(runtime, width, height, image, kept)
            embed_ms = (time.perf_counter() - started) * 1000
        run = to_face_run(frame, width, height, items)
        return dataclasses.replace(run, detection=CandidateDetection(
            max(below, key=lambda c: c[0]) if below else None, detect_ms, embed_ms))

    def _embed_candidates(self, runtime: SimpleNamespace, width: int, height: int, image: Any,
                          rows: Sequence[FullScaleRow]) -> list[dict[str, Any]]:
        """D88: the kept detections cut and aligned by DeepFace's extract_face in the padded frame, reported as
        extract_faces reports them, and embedded in one represent(skip) call: represent's own output items."""
        width_border, height_border = int(0.5 * width), int(0.5 * height)
        try:
            padded = runtime.cv2.copyMakeBorder(image, height_border, height_border, width_border, width_border,
                                                runtime.cv2.BORDER_CONSTANT, value=[0, 0, 0])
            crops, reported = [], []
            for row in rows:
                detected = runtime.detection.extract_face(
                    facial_area=runtime.FacialAreaRegion(**candidate_region(row, width_border, height_border)),
                    img=padded, align=FACE_ARGS["align"], expand_percentage=FACE_ARGS["expand_percentage"],
                    width_border=width_border, height_border=height_border, detector_backend=DETECTOR_BACKEND)
                if detected.img.shape[0] == 0 or detected.img.shape[1] == 0:
                    continue  # extract_faces skips an empty cut too
                crops.append(detected.img / 255)  # extract_faces divides; represent's skip path does not
                reported.append((reported_area(detected.facial_area, width, height),
                                 round(float(row.score), CONFIDENCE_DECIMALS)))
            del padded
            if not crops:
                return []
            raw = runtime.DeepFace.represent(
                img_path=crops, model_name=MODEL_NAME, detector_backend="skip", enforce_detection=False,
                align=FACE_ARGS["align"], expand_percentage=FACE_ARGS["expand_percentage"],
                normalization=FACE_ARGS["normalization"])
        except FaceBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - a fixed label; nothing partial is returned
            raise FaceBackendError("candidate_failed", type(exc).__name__) from None
        per_crop = [raw] if len(crops) == 1 else raw  # represent unwraps the list of a single image
        if not (isinstance(per_crop, list) and len(per_crop) == len(crops)
                and all(isinstance(r, list) and len(r) == 1 and isinstance(r[0], Mapping) for r in per_crop)):
            raise FaceBackendError("inconsistent_output")
        return [{"facial_area": area, "face_confidence": confidence, "embedding": result[0].get("embedding")}
                for (area, confidence), result in zip(reported, per_crop)]

    def enrollment_photo(self, path: Path) -> tuple[EnrollmentFace | None, str]:
        """Decode a photo with the same OpenCV (EXIF orientation applied) and find its single face."""
        if self._runtime is None:
            raise FaceBackendError("not_loaded")
        image = self._runtime.cv2.imread(str(path), self._runtime.cv2.IMREAD_COLOR)
        if image is None:
            return None, "unreadable_photo"
        return enrollment_face(self._runtime.DeepFace.represent(img_path=image, **FACE_ARGS))


def _cached_recognition_model(runtime: SimpleNamespace) -> Any:
    """Facenet512 in DeepFace's model cache (``modeling.cached_models``, created by ``build_model``), or None."""
    cache = getattr(runtime.modeling, "cached_models", None)
    if not isinstance(cache, Mapping) or not isinstance(cache.get("facial_recognition"), Mapping):
        return None
    return cache["facial_recognition"].get(MODEL_NAME)


def _cached_detector(runtime: SimpleNamespace) -> Any:
    """D88: the cv2.FaceDetectorYN inside DeepFace's cached YuNet client (``modeling.cached_models``, which the
    warm-up's represent() filled from the pinned weights file), or None."""
    cache = getattr(runtime.modeling, "cached_models", None)
    if not isinstance(cache, Mapping) or not isinstance(cache.get("face_detector"), Mapping):
        return None
    detector = getattr(cache["face_detector"].get(DETECTOR_BACKEND), "model", None)
    if detector is None or not all(callable(getattr(detector, name, None)) for name in DETECTOR_METHODS):
        return None
    return detector


def _check_calls(runtime: SimpleNamespace, label: str) -> None:
    """That the DeepFace functions the comparison (D87) and the candidate (D88) call take the parameters they pass,
    and that FacialAreaRegion has the fields they set; a FaceBackendError ``label`` otherwise."""
    try:
        functions = {"extract_face": runtime.detection.extract_face, "represent": runtime.DeepFace.represent}
        for name, wanted in COMPARISON_PARAMETERS.items():
            if not wanted <= set(inspect.signature(functions[name]).parameters):
                raise FaceBackendError(label, name)
        if not REGION_FIELDS <= {f.name for f in dataclasses.fields(runtime.FacialAreaRegion)}:
            raise FaceBackendError(label, "FacialAreaRegion")
    except FaceBackendError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise FaceBackendError(label, type(exc).__name__) from None


def _comparison_ready(runtime: SimpleNamespace) -> Any:
    """D87, at load: the cached Facenet512 the warm-up built, after checking that the DeepFace functions the
    comparison calls take the parameters it passes. A FaceBackendError label otherwise (the load fails closed)."""
    _check_calls(runtime, "comparison_unavailable")
    model = _cached_recognition_model(runtime)
    if model is None:
        raise FaceBackendError("model_not_cached")
    return model


def _candidate_ready(runtime: SimpleNamespace) -> tuple[Any, Any]:
    """D88, at load: (DeepFace's cached YuNet detector, its cached Facenet512), both built by the production warm-up,
    after the same check of DeepFace's functions. A FaceBackendError label otherwise (the load fails closed)."""
    _check_calls(runtime, "candidate_unavailable")
    detector = _cached_detector(runtime)
    if detector is None:
        raise FaceBackendError("detector_not_cached")
    model = _cached_recognition_model(runtime)
    if model is None:
        raise FaceBackendError("model_not_cached")
    return detector, model


def candidate_rows(found: Any) -> list[FullScaleRow]:
    """D88: every YuNet row, in YuNet's order, as its box, eyes and score in frame pixels. A malformed row is an error
    (DeepFace's own ``int()`` would raise on it), never a face dropped quietly."""
    rows = []
    for row in () if found is None else found:
        try:
            if len(row) < YUNET_ROW_VALUES:
                raise ValueError("short row")
            x, y, w, h, e4x, e4y, e6x, e6y = (float(row[i]) for i in range(8))
            score = float(row[-1])
        except (TypeError, ValueError, IndexError):
            raise FaceBackendError("inconsistent_output") from None
        if not all(math.isfinite(v) for v in (x, y, w, h, e4x, e4y, e6x, e6y, score)):
            raise FaceBackendError("inconsistent_output")
        rows.append(FullScaleRow(x, y, w, h, (e4x, e4y), (e6x, e6y), score))
    return rows


def candidate_region(row: FullScaleRow, width_border: int, height_border: int) -> dict[str, Any]:
    """D88: what DeepFace 0.0.99's ``YuNetClient.detect_faces`` would report for this detection had it detected on the
    padded picture without scaling it: the row moved by the borders, then ``int()`` of the box and the eyes, x and y
    clamped at 0 (the padded picture's edge, which a face inside the frame never reaches; a box reaching past the
    frame's top or left edge keeps its extent into the black border, as production's does), YuNet's columns 6-7 as
    ``left_eye`` and 4-5 as ``right_eye``. For a box and eyes inside the frame this is D87's
    ``padded_region(deepface_region(row))`` exactly; D87 clamped at the frame's edge instead."""
    return {"x": max(int(row.x + width_border), 0), "y": max(int(row.y + height_border), 0),
            "w": int(row.w), "h": int(row.h), "confidence": float(row.score),
            "left_eye": (int(row.eye_6[0] + width_border), int(row.eye_6[1] + height_border)),
            "right_eye": (int(row.eye_4[0] + width_border), int(row.eye_4[1] + height_border))}


def reported_area(area: Any, width: int, height: int) -> dict[str, int]:
    """D88: DeepFace 0.0.99's ``extract_faces`` report of a cut face (its ``facial_area``) on a frame of this size:
    ``extract_face`` has already taken the borders off x and y; then x and y at least 0, and the width and height cut
    at the frame's last column and row (``min(width - x - 1, w)``), as integers."""
    x, y = max(0, int(area.x)), max(0, int(area.y))
    return {"x": x, "y": y, "w": min(width - x - 1, int(area.w)), "h": min(height - y - 1, int(area.h))}


def _row_box(row: FullScaleRow, width: int, height: int) -> NormalizedBox | None:
    """A row's box normalized and clamped to the frame as full_scale_pass makes it; None when nothing is inside."""
    x1, y1 = max(0.0, row.x / width), max(0.0, row.y / height)
    x2, y2 = min(1.0, (row.x + row.w) / width), min(1.0, (row.y + row.h) / height)
    return NormalizedBox(x1=x1, y1=y1, x2=x2, y2=y2) if x1 < x2 and y1 < y2 else None


def full_scale_pass(detector: Any, width: int, height: int, image: Any, *, rows: bool = False) -> FullScale | None:
    """D86: one YuNet detection on the frame at native scale; None when the frame is larger than YuNet's 640-pixel
    limit (it would be scaled again) or the pass fails: a diagnostic never changes the production result. With
    ``rows`` (D87, comparison runs) each kept candidate's raw geometry is kept beside it, in memory only."""
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
            candidates.append((min(1.0, max(0.0, score)), NormalizedBox(x1=x1, y1=y1, x2=x2, y2=y2),
                               _row_geometry(row, x, y, w, h, score) if rows else None))
    candidates.sort(key=lambda c: -c[0])
    kept = candidates[:FULL_SCALE_KEPT]
    return FullScale(tuple((score, box) for score, box, _ in kept), (time.perf_counter() - started) * 1000,
                     tuple(geometry for _, _, geometry in kept) if rows else ())


def _row_geometry(row: Any, x: float, y: float, w: float, h: float, score: float) -> FullScaleRow | None:
    """D87: the row's box and eyes in frame pixels; None when it has no landmarks or they are not finite."""
    try:
        if len(row) < YUNET_ROW_VALUES:
            return None
        eyes = [float(row[i]) for i in (4, 5, 6, 7)]
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for v in eyes):
        return None
    return FullScaleRow(x, y, w, h, (eyes[0], eyes[1]), (eyes[2], eyes[3]), score)


def deepface_region(row: FullScaleRow) -> dict[str, Any]:
    """D87: what DeepFace 0.0.99's ``YuNetClient.detect_faces`` reports for this row on a picture of at most 640
    pixels (no resize): ``int()`` of the box and the eyes, x and y clamped at 0 (the eyes are not), YuNet's columns
    6-7 as ``left_eye`` and 4-5 as ``right_eye`` (DeepFace names the eyes from the person's side), and the score."""
    return {"x": max(int(row.x), 0), "y": max(int(row.y), 0), "w": int(row.w), "h": int(row.h),
            "confidence": float(row.score), "left_eye": (int(row.eye_6[0]), int(row.eye_6[1])),
            "right_eye": (int(row.eye_4[0]), int(row.eye_4[1]))}


def padded_region(region: Mapping[str, Any], width: int, height: int) -> tuple[dict[str, Any], int, int]:
    """D87: the region moved into the frame as ``detection.detect_faces`` pads it with ``align=True`` (a black border
    of ``int(0.5 * height)`` above and below and ``int(0.5 * width)`` left and right), and the two borders."""
    width_border, height_border = int(0.5 * width), int(0.5 * height)
    moved = dict(region)
    moved["x"], moved["y"] = region["x"] + width_border, region["y"] + height_border
    for eye in ("left_eye", "right_eye"):
        moved[eye] = (region[eye][0] + width_border, region[eye][1] + height_border)
    return moved, width_border, height_border
