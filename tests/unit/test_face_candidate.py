"""D88, the full-scale face detection candidate (validation runs only, selected explicitly).

Fakes stand in for OpenCV, DeepFace and TensorFlow; no model is loaded. What the candidate mirrors was read from the
installed DeepFace 0.0.99 sources (session 61): ``YuNetClient.detect_faces`` (the region, here on the padded picture
without scaling), ``detection.detect_faces`` (the border), ``extract_face`` (cut and alignment), ``extract_faces``
(the reported area, the 2-decimal confidence, the /255 and the dropped empty cut) and ``represent`` (skip, one batch,
the cached model). Identity tests drive the real EdgeCore with the candidate's FaceRuns: matching and non-matching
faces, no face, and ambiguous ownership.
"""

from __future__ import annotations

import dataclasses
import json
import math
import struct
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from sentinel.config import parse_config
from sentinel.contracts import FrameKey, NormalizedBox, PixelFormat, TrackObservation, TrackStatus
from sentinel.identity.gallery import unit
from sentinel.identity.legacy_deepface import (
    CONFIDENCE_DECIMALS,
    DETECTION_DEEPFACE,
    DETECTION_FULL_SCALE,
    DETECTOR_WEIGHTS,
    FACE_ARGS,
    FULL_SCALE_FLOOR,
    MODEL_NAME,
    MODEL_WEIGHTS,
    YUNET_THRESHOLD,
    FaceBackendError,
    FaceRun,
    FullScaleRow,
    LegacyDeepFaceBackend,
    candidate_region,
    candidate_rows,
    deepface_region,
    padded_region,
    reported_area,
    to_face_run,
)
from sentinel.identity.matching import EnrolledIdentity, Enrollment
from sentinel.identity.state import IdentityState
from sentinel.live_state import Capability
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.runtime import IDENTITY_CANDIDATE_SCHEMA, EdgeCore

FRAME = FrameKey(camera_id="cam-1", boot_id="b", run_id="r", stream_epoch=0, frame_seq=1)


def vec(*head: float) -> tuple[float, ...]:
    return unit(tuple(head) + (0.0,) * (512 - len(head)))


ENROLLED = vec(1.0)
MATCHING, NOT_MATCHING = vec(0.97, 0.1, 0.0, 0.05), vec(0.0, 0.1, 0.99)  # cosine 0.993 and 0.0 with ENROLLED
ENROLLMENT = Enrollment(model_revision="synthetic", dimension=512, identities=(
    EnrolledIdentity(identity_id="idn-00000000000a", prototypes=(ENROLLED,)),))


def yunet(x, y, w, h, score, eyes=None):
    """One YuNet row (x, y, w, h, right eye, left eye, nose, mouth corners, score) in frame pixels."""
    rx, ry, lx, ly = eyes if eyes is not None else (x + 0.3 * w, y + 0.4 * h, x + 0.7 * w, y + 0.41 * h)
    return [x, y, w, h, rx, ry, lx, ly, x + 0.5 * w, y + 0.6 * h, x + 0.35 * w, y + 0.8 * h, x + 0.65 * w,
            y + 0.8 * h, score]


# People as the tracker reports them (normalized), and faces inside their head regions (frame pixels, 640x480).
LEFT = NormalizedBox(x1=0.05, y1=0.1, x2=0.35, y2=0.9)  # 32-224 x 48-432 px; head region down to 201.6 px
RIGHT = NormalizedBox(x1=0.6, y1=0.1, x2=0.9, y2=0.9)  # 384-576 x 48-432 px
LEFT_FACE = yunet(60.6, 60.3, 58.7, 62.2, 0.93)
RIGHT_FACE = yunet(440.2, 61.9, 57.4, 63.8, 0.95)


class Cut:
    """An aligned cut's stand-in: its shape, its place in the frame's cuts, and division recorded."""

    def __init__(self, shape=(80, 64, 3), index=0) -> None:
        self.shape, self.index, self.divided_by = shape, index, None

    def __truediv__(self, other):
        divided = Cut(self.shape, self.index)
        divided.divided_by = other
        return divided


@dataclass
class Region:  # deepface.models.Detector.FacialAreaRegion's fields
    x: int
    y: int
    w: int
    h: int
    left_eye: tuple | None = None
    right_eye: tuple | None = None
    confidence: float | None = None
    nose: tuple | None = None
    mouth_right: tuple | None = None
    mouth_left: tuple | None = None


class Detector:
    """cv2.FaceDetectorYN as DeepFace's YuNet client holds it: every call recorded."""

    def __init__(self, rows, error=None) -> None:
        self.rows, self.error, self.calls = rows, error, []

    def setInputSize(self, size) -> None:  # noqa: N802 - OpenCV's name
        self.calls.append(("size", size))

    def setScoreThreshold(self, value) -> None:  # noqa: N802
        self.calls.append(("threshold", value))

    def detect(self, image):
        self.calls.append(("detect", image))
        if self.error is not None:
            raise self.error
        return 1, self.rows


class Runtime:
    """TensorFlow, OpenCV and DeepFace stand-ins. DeepFace's own represent() with yunet (the warm-up, production)
    builds its YuNet client into the model cache, as modeling.build_model does; every call is recorded."""

    def __init__(self, rows=(), *, embeddings=(MATCHING,), cut_shapes=None, detect_error=None, extract_error=None,
                 represent_output=None, cache_detector=True, cache_model=True) -> None:
        self.calls: list = []
        self.detector = Detector(list(rows), detect_error)
        self.model = object()
        self.embeddings, self.cut_shapes = list(embeddings), cut_shapes
        self.extract_error, self.represent_output = extract_error, represent_output
        self.cache_detector, self.cache_model = cache_detector, cache_model
        self.cache: dict = {"facial_recognition": {}, "face_detector": {}}
        self.ns = SimpleNamespace(
            tf=SimpleNamespace(config=SimpleNamespace(get_visible_devices=lambda kind: []), __version__="2.21.0"),
            cv2=SimpleNamespace(__version__="4.13.0", BORDER_CONSTANT=0, FaceDetectorYN_create=self._create,
                                copyMakeBorder=self._border),
            numpy=SimpleNamespace(__version__="2.2.6", zeros=lambda shape, dtype: ("zeros", shape), uint8="uint8"),
            tf_keras=SimpleNamespace(__version__="2.21.0"), deepface=SimpleNamespace(__version__="0.0.99"),
            DeepFace=SimpleNamespace(build_model=self._build, represent=self.represent),
            detection=SimpleNamespace(extract_face=self.extract_face),
            modeling=SimpleNamespace(cached_models=self.cache), FacialAreaRegion=Region)

    def _build(self, name):
        self.calls.append(("build", name))
        if self.cache_model:
            self.cache["facial_recognition"][name] = self.model

    def _create(self, *args):
        self.calls.append(("create_detector", *args))
        return Detector([])

    def _border(self, image, top, bottom, left, right, kind, value):
        self.calls.append(("border", image, top, bottom, left, right, kind, value))
        return ("padded", image)

    def extract_face(self, facial_area, img, align, expand_percentage, width_border, height_border, detector_backend):
        index = sum(1 for c in self.calls if c[0] == "extract")
        self.calls.append(("extract", dataclasses.asdict(facial_area), img, align, expand_percentage, width_border,
                           height_border, detector_backend))
        if self.extract_error is not None:
            raise self.extract_error
        shape = (80, 64, 3) if self.cut_shapes is None else self.cut_shapes[index]
        # extract_face takes the borders off x and y (and the eyes) of the region it returns; w and h are kept
        return SimpleNamespace(img=Cut(shape, index), facial_area=Region(
            x=facial_area.x - width_border, y=facial_area.y - height_border, w=facial_area.w, h=facial_area.h))

    def represent(self, img_path, model_name="VGG-Face", enforce_detection=True, detector_backend="opencv",
                  align=True, expand_percentage=0, normalization="base", anti_spoofing=False, max_faces=None,
                  l2_normalize=False, minmax_normalize=False, return_face=False, cryptosystem=None):
        kwargs = {"model_name": model_name, "enforce_detection": enforce_detection,
                  "detector_backend": detector_backend, "align": align, "expand_percentage": expand_percentage,
                  "normalization": normalization}
        self.calls.append(("represent", img_path, kwargs))
        if detector_backend != "skip":  # DeepFace's own detection: it builds (and caches) its YuNet client
            if self.cache_detector:
                self.cache["face_detector"]["yunet"] = SimpleNamespace(model=self.detector)
            return [{"embedding": [0.0] * 511 + [1.0], "facial_area": {"x": 0, "y": 0, "w": 640, "h": 480},
                     "face_confidence": 0}]
        if self.represent_output is not None:
            return self.represent_output
        images = img_path if isinstance(img_path, list) else [img_path]
        out = [[{"embedding": list(self.embeddings[image.index % len(self.embeddings)]), "facial_area": {},
                 "face_confidence": 0}] for image in images]
        return out[0] if len(images) == 1 else out  # represent unwraps the list of a single image


def backend(tmp_path, runtime: Runtime, *, select: bool = True) -> LegacyDeepFaceBackend:
    for name, _ in (MODEL_WEIGHTS, DETECTOR_WEIGHTS):
        (tmp_path / name).write_bytes(b"synthetic")
    digests = {MODEL_WEIGHTS[0]: MODEL_WEIGHTS[1], DETECTOR_WEIGHTS[0]: DETECTOR_WEIGHTS[1]}
    made = LegacyDeepFaceBackend(tmp_path, importer=lambda: runtime.ns, sha256=lambda p: digests[p.name])
    if select:
        made.select_full_scale_detection()
    made.load()
    return made


def after_load(runtime: Runtime, calls_at_load: int) -> list:
    return runtime.calls[calls_at_load:]


# ---------------------------------------------------------------- coordinate mapping


def yunet_client_on_padded(face: list[float], width_border: int, height_border: int) -> dict:
    """DeepFace 0.0.99 ``YuNetClient.detect_faces``, transcribed, for a detection on the padded picture that is not
    scaled: YuNet's row in padded coordinates, then int(), x and y clamped at 0, columns 6-7 as left_eye."""
    shifted = [v + (width_border if i % 2 == 0 else height_border) if i in (0, 1, 4, 5, 6, 7) else v
               for i, v in enumerate(face[:8])]
    (x, y, w, h, x_le, y_le, x_re, y_re) = list(map(int, shifted))
    x = max(x, 0)
    y = max(y, 0)
    return {"x": x, "y": y, "w": w, "h": h, "confidence": float(face[-1]), "left_eye": (x_re, y_re),
            "right_eye": (x_le, y_le)}


INSIDE = [128.7, 72.2, 64.9, 72.1, 150.4, 100.9, 176.6, 101.2, 160.0, 115.0, 152.0, 128.0, 175.0, 128.0, 0.93]
PAST_THE_EDGE = [-3.7, -0.4, 40.2, 50.8, 5.9, 12.1, 25.3, 11.7, 15.0, 20.0, 8.0, 30.0, 22.0, 30.0, 0.91]
BOTTOM_RIGHT = [600.0, 400.5, 39.9, 79.99, 610.1, 420.9, 630.9, 421.2, 620.0, 440.0, 612.0, 455.0, 628.0, 455.0, 0.9]


@pytest.mark.parametrize("face", [INSIDE, PAST_THE_EDGE, BOTTOM_RIGHT])
def test_the_region_is_what_yunet_client_reports_on_the_padded_picture_unscaled(face) -> None:
    (row,) = candidate_rows([face])
    assert candidate_region(row, 320, 240) == yunet_client_on_padded(face, 320, 240)
    region = candidate_region(row, 320, 240)  # YuNet's columns 6-7 are the person's left eye (DeepFace's naming)
    assert region["left_eye"] == (int(face[6] + 320), int(face[7] + 240))
    assert region["right_eye"] == (int(face[4] + 320), int(face[5] + 240))


def test_inside_the_frame_it_equals_d87s_conversion_and_past_the_edge_it_keeps_productions_extent() -> None:
    inside, past = candidate_rows([INSIDE, PAST_THE_EDGE])
    d87, _, _ = padded_region(deepface_region(inside), 640, 480)
    assert candidate_region(inside, 320, 240) == d87  # what the D87 comparison measured, for boxes in the frame
    # A box starting 3.7 px left of the frame: production (detecting on the padded picture) keeps it reaching 4 px
    # into the black border; D87 clamped it at the frame's edge, moving the cut 4 px right.
    d87_past, _, _ = padded_region(deepface_region(past), 640, 480)
    assert (candidate_region(past, 320, 240)["x"], d87_past["x"]) == (316, 320)
    assert (candidate_region(past, 320, 240)["y"], d87_past["y"]) == (239, 240)
    assert candidate_region(past, 320, 240)["w"] == d87_past["w"] == 40  # the width is not adjusted by either


def test_the_reported_area_is_what_extract_faces_reports() -> None:
    # extract_face has already taken the borders off; extract_faces clamps x and y at 0 and cuts at the last column
    assert reported_area(Region(x=-4, y=-1, w=40, h=50), 640, 480) == {"x": 0, "y": 0, "w": 40, "h": 50}
    assert reported_area(Region(x=600, y=400, w=39, h=79), 640, 480) == {"x": 600, "y": 400, "w": 39, "h": 79}
    assert reported_area(Region(x=605, y=430, w=39, h=79), 640, 480) == {"x": 605, "y": 430, "w": 34, "h": 49}


def test_malformed_rows_are_errors_never_dropped_faces() -> None:
    assert candidate_rows(None) == [] and candidate_rows([]) == []
    for bad in ([1.0, 2.0, 3.0, 4.0, 0.95], INSIDE[:8] + [math.nan] * 7, ["x"] * 15,
                INSIDE[:4] + [math.inf] + INSIDE[5:]):
        with pytest.raises(FaceBackendError, match="inconsistent_output"):
            candidate_rows([bad])


# ---------------------------------------------------------------- one detection, production's cut and model


def test_detection_runs_once_at_native_scale_on_deepfaces_own_cached_detector(tmp_path) -> None:
    runtime = Runtime([LEFT_FACE])
    made = backend(tmp_path, runtime)
    at_load = len(runtime.calls)
    assert made.detection == DETECTION_FULL_SCALE
    run = made.faces(FRAME, 640, 480, "frame")
    assert runtime.detector.calls == [("size", (640, 480)), ("threshold", FULL_SCALE_FLOOR), ("detect", "frame")]
    calls = after_load(runtime, at_load)
    assert [c[0] for c in calls] == ["border", "extract", "represent"]  # no DeepFace detection on this path
    assert not any(c[0] == "represent" and c[2]["detector_backend"] != "skip" for c in calls)
    assert not any(c[0] in ("create_detector", "build") for c in runtime.calls[at_load:])
    assert len(run.faces) == 1 and run.detected == 1 and run.fallback_dropped == 0


def test_each_face_is_cut_and_aligned_by_extract_face_in_the_padded_frame_and_divided_by_255(tmp_path) -> None:
    runtime = Runtime([LEFT_FACE])
    made = backend(tmp_path, runtime)
    at_load = len(runtime.calls)
    made.faces(FRAME, 640, 480, "frame")
    border, extract, represent = after_load(runtime, at_load)
    assert border == ("border", "frame", 240, 240, 320, 320, 0, [0, 0, 0])  # detect_faces's black border
    (row,) = candidate_rows([LEFT_FACE])
    assert extract[1] == {**candidate_region(row, 320, 240), "nose": None, "mouth_right": None, "mouth_left": None}
    assert extract[2:] == (("padded", "frame"), True, 0, 320, 240, "yunet")
    assert [cut.divided_by for cut in represent[1]] == [255]  # extract_faces divides; the skip path does not
    assert represent[2] == {"model_name": MODEL_NAME, "enforce_detection": False, "detector_backend": "skip",
                            "align": FACE_ARGS["align"], "expand_percentage": FACE_ARGS["expand_percentage"],
                            "normalization": FACE_ARGS["normalization"]}


def test_a_face_becomes_the_observation_production_would_report_for_that_detection(tmp_path) -> None:
    runtime = Runtime([LEFT_FACE])
    (face,) = backend(tmp_path, runtime).faces(FRAME, 640, 480, "frame").faces
    # production's report of the same detection: extract_faces' area and 2-decimal confidence, through to_face_run
    area = reported_area(Region(x=int(LEFT_FACE[0] + 320) - 320, y=int(LEFT_FACE[1] + 240) - 240,
                                w=int(LEFT_FACE[2]), h=int(LEFT_FACE[3])), 640, 480)
    (expected,) = to_face_run(FRAME, 640, 480, [{"facial_area": area, "face_confidence": round(0.93, 2),
                                                  "embedding": list(MATCHING)}]).faces
    assert face == expected
    assert face.box == NormalizedBox(x1=60 / 640, y1=60 / 480, x2=118 / 640, y2=122 / 480)
    assert face.quality == 0.93 and CONFIDENCE_DECIMALS == 2
    assert math.isclose(math.fsum(v * v for v in face.embedding), 1.0)  # unit length, as production's
    (rounded,) = backend(tmp_path, Runtime([yunet(60.6, 60.3, 58.7, 62.2, 0.9372)])).faces(
        FRAME, 640, 480, "frame").faces
    assert rounded.quality == 0.94  # extract_faces' round(confidence, 2), which production's quality is


def float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def test_only_detections_at_productions_threshold_become_faces_and_the_best_below_is_recorded(tmp_path) -> None:
    # OpenCV compares a float32 score with its float 0.9: 0.9 itself (0.89999997...) is not kept, as here
    rows = [yunet(60.0, 60.0, 58.0, 62.0, float32(0.9)), yunet(440.0, 62.0, 57.0, 64.0, 0.95),
            yunet(300.0, 300.0, 30.0, 34.0, 0.62), yunet(700.0, 10.0, 30.0, 30.0, 0.85)]  # the last outside the frame
    runtime = Runtime(rows)
    run = backend(tmp_path, runtime).faces(FRAME, 640, 480, "frame")
    assert [round(f.quality, 2) for f in run.faces] == [0.95]
    assert sum(1 for c in runtime.calls if c[0] == "extract") == 1
    score, box = run.detection.best_below
    assert score == float32(0.9) and box.x1 == 60 / 640  # the best one under 0.9 with anything inside the frame
    assert run.detection.detect_ms >= 0 and run.detection.embed_ms >= 0
    assert YUNET_THRESHOLD == 0.9 and float32(0.9) < 0.9 < float32(0.9000001)


def test_a_frame_without_a_face_at_the_threshold_is_never_embedded(tmp_path) -> None:
    runtime = Runtime([yunet(60.0, 60.0, 58.0, 62.0, 0.71)])
    made = backend(tmp_path, runtime)
    at_load = len(runtime.calls)
    run = made.faces(FRAME, 640, 480, "frame")
    assert run.faces == () and run.detected == 0 and run.fallback_dropped == 0  # no whole-frame stand-in exists
    assert after_load(runtime, at_load) == []  # no border, cut or embedding
    assert run.detection.embed_ms is None and run.detection.best_below[0] == 0.71
    empty = Runtime([])
    assert backend(tmp_path, empty).faces(FRAME, 640, 480, "frame").detection.best_below is None


def test_several_faces_are_embedded_in_one_batch_in_yunets_order(tmp_path) -> None:
    runtime = Runtime([LEFT_FACE, RIGHT_FACE], embeddings=(MATCHING, NOT_MATCHING))
    made = backend(tmp_path, runtime)
    at_load = len(runtime.calls)
    run = made.faces(FRAME, 640, 480, "frame")
    represents = [c for c in after_load(runtime, at_load) if c[0] == "represent"]
    assert len(represents) == 1 and [cut.index for cut in represents[0][1]] == [0, 1]  # one forward pass
    assert [round(f.box.x1 * 640) for f in run.faces] == [60, 440]
    assert [f.embedding[:2] for f in run.faces] == [MATCHING[:2], NOT_MATCHING[:2]]


def test_an_empty_cut_is_skipped_as_extract_faces_skips_it(tmp_path) -> None:
    runtime = Runtime([LEFT_FACE, RIGHT_FACE], cut_shapes=[(0, 64, 3), (80, 64, 3)])
    run = backend(tmp_path, runtime).faces(FRAME, 640, 480, "frame")
    assert [round(f.box.x1 * 640) for f in run.faces] == [440] and run.detected == 1
    all_empty = Runtime([LEFT_FACE], cut_shapes=[(80, 0, 3)])
    made = backend(tmp_path, all_empty)
    at_load = len(all_empty.calls)
    assert made.faces(FRAME, 640, 480, "frame").faces == ()
    assert [c[0] for c in after_load(all_empty, at_load)] == ["border", "extract"]  # nothing to embed


@pytest.mark.parametrize(("runtime", "size", "label"), [
    (Runtime([LEFT_FACE], detect_error=RuntimeError("cv2 says no")), (640, 480), "candidate_failed"),
    (Runtime([[1.0, 2.0, 3.0]]), (640, 480), "inconsistent_output"),
    (Runtime([LEFT_FACE], extract_error=ValueError("cv2 says no")), (640, 480), "candidate_failed"),
    (Runtime([LEFT_FACE, RIGHT_FACE], represent_output=[[{"embedding": [1.0] * 512}]]), (640, 480),
     "inconsistent_output"),
    (Runtime([LEFT_FACE], represent_output=[{"embedding": [0.0] * 512}]), (640, 480), "inconsistent_output"),
    (Runtime([LEFT_FACE], represent_output=[{"embedding": [1.0] * 10}]), (640, 480), "inconsistent_output"),
    (Runtime([LEFT_FACE]), (1280, 720), "frame_too_large"),
])
def test_failures_are_fixed_labels_never_a_partial_result(tmp_path, runtime, size, label) -> None:
    made = backend(tmp_path, runtime)
    with pytest.raises(FaceBackendError) as raised:
        made.faces(FRAME, *size, "frame")
    assert raised.value.label == label and "cv2 says no" not in str(raised.value)


# ---------------------------------------------------------------- no second model, selection, compatibility


def test_no_detector_or_recognition_model_is_built_or_loaded_twice(tmp_path) -> None:
    runtime = Runtime([LEFT_FACE])
    made = backend(tmp_path, runtime)
    assert [c for c in runtime.calls if c[0] == "build"] == [("build", MODEL_NAME)]  # production's own warm-up
    assert not any(c[0] == "create_detector" for c in runtime.calls)  # D86's separate detector is never made
    for _ in range(3):
        made.faces(FRAME, 640, 480, "frame")
    assert [c for c in runtime.calls if c[0] in ("build", "create_detector")] == [("build", MODEL_NAME)]
    del runtime.cache["face_detector"]["yunet"]  # evicted: DeepFace would build another
    calls = len(runtime.calls)
    with pytest.raises(FaceBackendError, match="detector_not_cached"):
        made.faces(FRAME, 640, 480, "frame")
    other = Runtime([LEFT_FACE])
    made = backend(tmp_path, other)
    other.cache["facial_recognition"].clear()  # evicted: represent() would build another
    with pytest.raises(FaceBackendError, match="model_not_cached"):
        made.faces(FRAME, 640, 480, "frame")
    assert runtime.calls[calls:] == [] and other.detector.calls == []  # nothing called at all
    for kwargs, label in (({"cache_detector": False}, "detector_not_cached"),
                          ({"cache_model": False}, "model_not_cached")):
        with pytest.raises(FaceBackendError, match=label):
            backend(tmp_path, Runtime([LEFT_FACE], **kwargs))  # the load fails closed


def test_the_load_checks_deepfaces_functions(tmp_path) -> None:
    runtime = Runtime([LEFT_FACE])
    runtime.ns.detection.extract_face = lambda facial_area, img, align: None  # an older or newer signature
    with pytest.raises(FaceBackendError, match="candidate_unavailable"):
        backend(tmp_path, runtime)


def test_the_candidate_is_selected_explicitly_before_the_load_and_excludes_the_diagnostics(tmp_path) -> None:
    runtime = Runtime([LEFT_FACE])
    plain = backend(tmp_path, runtime, select=False)
    assert plain.detection == DETECTION_DEEPFACE
    at_load = len(runtime.calls)
    plain.faces(FRAME, 640, 480, "frame")  # production unchanged: DeepFace's own represent() on the frame
    assert after_load(runtime, at_load) == [("represent", "frame", {k: v for k, v in FACE_ARGS.items()})]
    assert runtime.detector.calls == []
    with pytest.raises(FaceBackendError, match="already_loaded"):
        plain.select_full_scale_detection()
    fresh = LegacyDeepFaceBackend(tmp_path)
    fresh.enable_full_scale()
    with pytest.raises(FaceBackendError, match="candidate_excludes_diagnostics"):
        fresh.select_full_scale_detection()
    chosen = LegacyDeepFaceBackend(tmp_path)
    chosen.select_full_scale_detection()
    with pytest.raises(FaceBackendError, match="candidate_excludes_diagnostics"):
        chosen.enable_full_scale()
    with pytest.raises(FaceBackendError, match="comparison_needs_full_scale"):
        chosen.enable_comparison()


def test_the_candidate_keeps_the_enrollments_compatibility(tmp_path) -> None:
    plain = backend(tmp_path, Runtime([LEFT_FACE]), select=False).compatibility()
    candidate = backend(tmp_path, Runtime([LEFT_FACE])).compatibility()
    assert candidate == plain and candidate.first_difference(plain) is None  # the gallery's record still matches
    assert (candidate.detector_backend, candidate.align, candidate.normalization, candidate.expand_percentage) == (
        "yunet", True, "base", 0)


# ---------------------------------------------------------------- the identity rules, unchanged


CONFIG = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})


class World:
    """The real EdgeCore, fed the candidate adapter's FaceRuns for frames whose people the test places."""

    def __init__(self, tmp_path, runtime: Runtime) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stream = self.stamper.connect()
        self.core = EdgeCore(CONFIG, self.clock, None, face_recognition=Capability.AVAILABLE, enrollment=ENROLLMENT)
        self.runtime = runtime
        self.backend = backend(tmp_path, runtime)
        self.transitions: list = []
        self.records: list = []

    def step(self, rows, embeddings=(MATCHING,), people=(LEFT,), dt: float = 1.0):
        self.runtime.detector.rows = list(rows)
        self.runtime.embeddings = list(embeddings)
        self.clock.advance(dt)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        persons = tuple(TrackObservation.detected(frame, track_id=i + 1, box=box, confidence=0.9,
                                                  status=TrackStatus.CONFIRMED) for i, box in enumerate(people))
        self._keep(self.core.on_frame(frame, persons, self.stream))
        run = self.last_run = self.backend.faces(frame.key, 640, 480, "frame")
        output = self.core.on_face_outcome(frame, persons, run.faces, self.stream, processing_ms=900.0,
                                           detection=run.detection)
        self._keep(output)
        self.records.append(output.face_result.record(IDENTITY_CANDIDATE_SCHEMA))
        return self.records[-1]

    def person(self, track: int):
        return {p.track_id: p for p in self.state.people}.get(track)

    def _keep(self, output) -> None:
        self.state = output.state
        self.transitions += [(t.from_state, t.to_state, t.reason) for t in output.identity_transitions]


def test_matching_faces_need_two_confirmations_and_a_face_that_does_not_match_votes_unknown(tmp_path) -> None:
    world = World(tmp_path, Runtime())
    first = world.step([LEFT_FACE])
    assert world.person(1).identity is IdentityState.UNRESOLVED
    world.step([LEFT_FACE])
    assert world.person(1).identity is IdentityState.KNOWN  # two fresh matches, as for production's faces
    assert first["persons"][0]["label"] == "match" and first["persons"][0]["similarity"] == 0.993
    nonmatch = world.step([LEFT_FACE], embeddings=(NOT_MATCHING,))  # a face found and assigned, but not the enrolled one
    person = nonmatch["persons"][0]
    assert (person["ownership"], person["vote"], person["label"], person["identity_id"]) == (
        "assigned", "unknown", "unknown", None)
    assert person["similarity"] == 0.0 and person["face"]["score"] == 0.93
    assert world.person(1).identity is IdentityState.UNCERTAIN  # contradicting evidence ends KNOWN
    for _ in range(2):
        world.step([LEFT_FACE], embeddings=(NOT_MATCHING,))
    assert world.person(1).identity is IdentityState.UNKNOWN  # a face-confirmed stranger only from faces seen
    assert [t[:2] for t in world.transitions] == [("unresolved", "known"), ("known", "uncertain"),
                                                   ("uncertain", "unknown")]


def test_no_face_gives_no_vote_and_retains_an_identity_by_the_unchanged_rule(tmp_path) -> None:
    world = World(tmp_path, Runtime())
    for _ in range(2):
        world.step([LEFT_FACE])
    record = world.step([yunet(60.0, 60.0, 58.0, 62.0, 0.8)])  # only a detection under 0.9: no face
    person = record["persons"][0]
    assert (person["ownership"], person["vote"], person["face"], person["similarity"]) == (
        "no_face", "none", None, None)
    assert record["faces"] == 0 and record["detection"]["best_below_score"] == 0.8
    assert record["detection"]["embed_ms"] is None
    assert world.person(1).identity is IdentityState.KNOWN  # retained on its own track, as the rule says
    nobody = World(tmp_path, Runtime())
    nobody.step([])
    assert nobody.person(1).identity is IdentityState.UNRESOLVED  # never a stranger without a face


def test_a_face_two_people_could_own_stays_unresolved_and_two_faces_for_one_person_too(tmp_path) -> None:
    overlap = (NormalizedBox(x1=0.05, y1=0.1, x2=0.35, y2=0.9), NormalizedBox(x1=0.08, y1=0.1, x2=0.4, y2=0.9))
    world = World(tmp_path, Runtime())
    record = world.step([LEFT_FACE], people=overlap)
    assert [(p["ownership"], p["vote"]) for p in record["persons"]] == [("ambiguous", "none")] * 2
    two = World(tmp_path, Runtime())
    record = two.step([LEFT_FACE, yunet(130.0, 70.0, 50.0, 60.0, 0.94)], embeddings=(MATCHING, MATCHING))
    assert [(p["ownership"], p["vote"]) for p in record["persons"]] == [("ambiguous", "none")]
    assert two.person(1).identity is IdentityState.UNRESOLVED  # never whichever face came first


def test_each_person_gets_their_own_face_one_to_one(tmp_path) -> None:
    world = World(tmp_path, Runtime())
    record = world.step([LEFT_FACE, RIGHT_FACE], embeddings=(MATCHING, NOT_MATCHING), people=(LEFT, RIGHT))
    assert [(p["track_id"], p["ownership"], p["label"]) for p in record["persons"]] == [
        (1, "assigned", "match"), (2, "assigned", "unknown")]


def test_the_same_faces_give_the_same_votes_whichever_detection_produced_them(tmp_path) -> None:
    """The candidate changes only how a FaceRun is made: the same observations through production's FaceRun give the
    same votes, labels, states and transitions."""
    candidate = World(tmp_path, Runtime())
    production = World(tmp_path, Runtime())
    for embeddings in ((MATCHING,), (MATCHING,), (NOT_MATCHING,), (MATCHING,)):
        mine = candidate.step([LEFT_FACE], embeddings=embeddings)
        theirs_run = FaceRun(candidate.last_run.faces, 1, 1)  # as DeepFace's own detection would have given them
        production.clock.advance(1.0)
        frame = production.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        persons = (TrackObservation.detected(frame, track_id=1, box=LEFT, confidence=0.9,
                                             status=TrackStatus.CONFIRMED),)
        production._keep(production.core.on_frame(frame, persons, production.stream))
        faces = tuple(f.model_copy(update={"frame": frame.key}) for f in theirs_run.faces)
        output = production.core.on_face_outcome(frame, persons, faces, production.stream, processing_ms=900.0)
        production._keep(output)
        theirs = output.face_result.record(2)
        assert mine["persons"] == theirs["persons"] and mine["faces"] == theirs["faces"]
        assert candidate.person(1).identity is production.person(1).identity
    assert candidate.transitions == production.transitions


def test_the_candidates_records_carry_its_numbers_and_nothing_else(tmp_path) -> None:
    world = World(tmp_path, Runtime())
    record = world.step([LEFT_FACE, yunet(400.0, 300.0, 30.0, 33.0, 0.44)])
    assert record["schema"] == IDENTITY_CANDIDATE_SCHEMA and "full_scale" not in record
    assert set(record["detection"]) == {"best_below_score", "best_below_w_px", "detect_ms", "embed_ms"}
    assert record["detection"]["best_below_score"] == 0.44 and record["detection"]["best_below_w_px"] == 30
    text = json.dumps(record)
    assert "embedding" not in text and "0.97" not in text  # numbers only: never an embedding
