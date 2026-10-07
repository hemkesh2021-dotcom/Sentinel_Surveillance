"""The interim face adapter (V2-25 demo form): pins, the CPU pin, DeepFace's confidence-0 fallback, output checks.

A fake runtime stands in for TensorFlow, OpenCV and DeepFace; no model is loaded.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel.contracts import FrameKey
from sentinel.identity.gallery import FaceCompatibility
from sentinel.identity.legacy_deepface import (
    DEEPFACE_VERSION,
    DETECTOR_WEIGHTS,
    FACE_ARGS,
    MODEL_WEIGHTS,
    FaceBackendError,
    LegacyDeepFaceBackend,
    enrollment_face,
    static_compatibility_fields,
    static_difference,
    to_face_run,
)
from sentinel.memory_policy import RELEASE_FILE_ROLES

FRAME = FrameKey(camera_id="cam-1", boot_id="boot-1", run_id="run-1", stream_epoch=0, frame_seq=7)
EMBEDDING = [0.0] * 511 + [2.0]


def face(x=100, y=50, w=80, h=100, confidence=0.9, embedding=EMBEDDING) -> dict:
    return {"embedding": list(embedding), "facial_area": {"x": x, "y": y, "w": w, "h": h},
            "face_confidence": confidence}


def fallback() -> dict:  # what DeepFace returns for "no face" with enforce_detection=False
    return face(x=0, y=0, w=640, h=480, confidence=0)


class FakeRuntime:
    def __init__(self, *, gpus=(), version=DEEPFACE_VERSION, represent=None) -> None:
        self.calls = []
        visible = list(gpus)
        config = SimpleNamespace(get_visible_devices=lambda kind: visible)
        self.ns = SimpleNamespace(
            tf=SimpleNamespace(config=config, __version__="2.21.0"),
            cv2=SimpleNamespace(__version__="4.13.0", IMREAD_COLOR=1, imread=self._imread),
            numpy=SimpleNamespace(__version__="2.2.6", zeros=lambda shape, dtype: ("zeros", shape), uint8="uint8"),
            tf_keras=SimpleNamespace(__version__="2.21.0"),
            deepface=SimpleNamespace(__version__=version),
            DeepFace=SimpleNamespace(build_model=lambda name: self.calls.append(("build", name)),
                                     represent=self._represent),
        )
        self.next_result = [fallback()]
        self._represent_impl = represent

    def _imread(self, path, flag):
        return None if "unreadable" in path else ("image", path)

    def _represent(self, img_path, **kwargs):
        self.calls.append(("represent", kwargs))
        if self._represent_impl is not None:
            return self._represent_impl(img_path)
        return self.next_result


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    for name, _ in (MODEL_WEIGHTS, DETECTOR_WEIGHTS):
        (tmp_path / name).write_bytes(b"synthetic")
    return tmp_path


def backend(weights: Path, runtime: FakeRuntime, *, digests=None) -> LegacyDeepFaceBackend:
    digests = digests or {MODEL_WEIGHTS[0]: MODEL_WEIGHTS[1], DETECTOR_WEIGHTS[0]: DETECTOR_WEIGHTS[1]}
    return LegacyDeepFaceBackend(weights, importer=lambda: runtime.ns, sha256=lambda path: digests[path.name])


def test_it_loads_the_profiled_path_on_the_cpu_and_reports_its_compatibility(weights) -> None:
    runtime = FakeRuntime()
    face_backend = backend(weights, runtime)
    face_backend.load()
    assert ("build", "Facenet512") in runtime.calls
    assert runtime.calls[-1] == ("represent", dict(FACE_ARGS))  # the warm-up uses the same arguments
    assert dict(FACE_ARGS) == {"model_name": "Facenet512", "detector_backend": "yunet", "enforce_detection": False,
                               "align": True, "normalization": "base", "expand_percentage": 0}
    compatibility = face_backend.compatibility()
    assert isinstance(compatibility, FaceCompatibility)
    assert (compatibility.opencv_version, compatibility.tensorflow_version, compatibility.tf_keras_version,
            compatibility.numpy_version, compatibility.dimension) == ("4.13.0", "2.21.0", "2.21.0", "2.2.6", 512)
    assert static_difference(compatibility) is None
    assert set(face_backend.weight_files) == set(RELEASE_FILE_ROLES["face"])  # what the D58 release expects


@pytest.mark.parametrize(("runtime", "digests", "label"), [
    (FakeRuntime(gpus=["/physical_device:GPU:0"]), None, "tensorflow_gpu_visible"),
    (FakeRuntime(version="0.0.98"), None, "deepface_version_mismatch"),
    (FakeRuntime(), {MODEL_WEIGHTS[0]: "0" * 64, DETECTOR_WEIGHTS[0]: DETECTOR_WEIGHTS[1]}, "weights_hash_mismatch"),
    (FakeRuntime(), {MODEL_WEIGHTS[0]: MODEL_WEIGHTS[1], DETECTOR_WEIGHTS[0]: "0" * 64}, "weights_hash_mismatch"),
])
def test_it_fails_closed(weights, runtime, digests, label) -> None:
    with pytest.raises(FaceBackendError) as raised:
        backend(weights, runtime, digests=digests).load()
    assert raised.value.label == label


def test_a_failed_cpu_pin_or_missing_library_is_a_label(weights) -> None:
    def pin_fails():
        raise RuntimeError("Visible devices cannot be modified after being initialized")

    def missing():
        raise ImportError("No module named 'deepface'")

    for importer, label in ((pin_fails, "tensorflow_cpu_pin_failed"), (missing, "runtime_unavailable")):
        face_backend = LegacyDeepFaceBackend(weights, importer=importer, sha256=lambda p: dict(
            [MODEL_WEIGHTS, DETECTOR_WEIGHTS])[p.name])
        with pytest.raises(FaceBackendError) as raised:
            face_backend.load()
        assert raised.value.label == label and "initialized" not in str(raised.value)
    with pytest.raises(FaceBackendError, match="weights_unreadable"):
        LegacyDeepFaceBackend(weights / "nowhere").load()


def test_the_confidence_zero_whole_image_fallback_is_dropped() -> None:
    run = to_face_run(FRAME, 640, 480, [fallback()])
    assert (run.faces, run.detected, run.fallback_dropped) == ((), 0, 1)


def test_faces_become_normalized_observations() -> None:
    run = to_face_run(FRAME, 640, 480, [face(), face(x=600, y=-10, w=80, h=60, confidence=1.7)])
    assert run.detected == 2 and len(run.faces) == 2
    first, clipped = run.faces
    assert (first.box.x1, first.box.y1, first.box.x2, first.box.y2) == (100 / 640, 50 / 480, 180 / 640, 150 / 480)
    assert (clipped.box.x2, clipped.box.y1, clipped.quality) == (1.0, 0.0, 1.0)  # clipped to the frame; capped at 1
    assert first.quality == 0.9 and first.frame == FRAME
    assert math.isclose(math.fsum(v * v for v in first.embedding), 1.0) and first.embedding[-1] == 1.0


@pytest.mark.parametrize("bad", [
    face(embedding=[0.0] * 512),  # zero
    face(embedding=[1.0] * 128),  # another model's dimension
    face(embedding=[float("nan")] + [1.0] * 511),
    {"embedding": EMBEDDING, "face_confidence": 0.9},  # no facial area
])
def test_inconsistent_output_is_an_error_not_a_face(bad) -> None:
    with pytest.raises(FaceBackendError, match="inconsistent_output"):
        to_face_run(FRAME, 640, 480, [bad])


def test_an_enrollment_photo_needs_exactly_one_face(weights) -> None:
    assert enrollment_face([fallback()]) == (None, "no_face")
    assert enrollment_face([face(), face(x=300)]) == (None, "several_faces")
    single, reason = enrollment_face([face(w=81), fallback()])
    assert reason == "face_found" and single is not None and single.face_width_px == 81 and single.quality == 0.9
    runtime = FakeRuntime()
    face_backend = backend(weights, runtime)
    with pytest.raises(FaceBackendError, match="not_loaded"):
        face_backend.enrollment_photo(Path("photo.jpg"))
    face_backend.load()
    assert face_backend.enrollment_photo(Path("unreadable.jpg")) == (None, "unreadable_photo")
    runtime.next_result = [face()]
    assert face_backend.enrollment_photo(Path("photo.jpg"))[1] == "face_found"


def test_a_gallery_from_another_pipeline_is_seen_before_loading() -> None:
    base = FaceCompatibility(**static_compatibility_fields(), opencv_version="4.13.0", tensorflow_version="2.21.0",
                             tf_keras_version="2.21.0", numpy_version="2.2.6")
    assert static_difference(base) is None
    assert static_difference(base.model_copy(update={"detector_backend": "retinaface"})) == "detector_backend"
    assert static_difference(base.model_copy(update={"align": False})) == "align"
