"""D87, the full-scale embedding comparison (validation runs only, diagnostic, never a vote).

Fakes stand in for OpenCV, DeepFace and TensorFlow; no model is loaded. The DeepFace 0.0.99 steps the comparison
mirrors were read from the installed sources (session 60): ``YuNetClient.detect_faces`` (the region), ``detect_faces``
(the border), ``extract_face`` (cut and alignment), ``extract_faces`` (/255) and ``represent`` (skip, resize,
normalization, the cached model).
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from sentinel.config import parse_config
from sentinel.contracts import FrameKey, NormalizedBox, PixelFormat, TrackObservation, TrackStatus
from sentinel.identity import comparison as comparison_module
from sentinel.identity.association import AssociationGeometry, FaceObservation
from sentinel.identity.comparison import (
    COMPARISON_CAP,
    COMPARISON_INTERVAL_S,
    CONTROL_SHARE,
    ComparisonOutcome,
    ComparisonPolicy,
    Disposition,
    FullScaleComparator,
    Kind,
    comparison_targets,
    error_label,
)
from sentinel.identity.legacy_deepface import (
    COMPARISON_PARAMETERS,
    DETECTOR_WEIGHTS,
    MODEL_NAME,
    MODEL_WEIGHTS,
    FaceBackendError,
    FaceRun,
    FullScale,
    FullScaleRow,
    LegacyDeepFaceBackend,
    deepface_region,
    full_scale_pass,
    padded_region,
)
from sentinel.identity.matching import EnrolledIdentity, Enrollment, best_match
from sentinel.identity.state import IdentityPolicy, IdentityResolver
from sentinel.identity.worker import COMPARISON_OUTBOX, FaceWorker
from sentinel.live_state import Capability
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.media.frames import FrameStamper
from sentinel import runtime as runtime_module
from sentinel.runtime import IDENTITY_COMPARISON_SCHEMA, EdgeCore

GEOMETRY = AssociationGeometry()
PERSON_BOX = NormalizedBox(x1=0.1, y1=0.1, x2=0.4, y2=0.9)  # head region: y 0.1 .. 0.42
FACE_BOX = NormalizedBox(x1=0.2, y1=0.15, x2=0.3, y2=0.3)  # inside it
OUTSIDE_BOX = NormalizedBox(x1=0.2, y1=0.6, x2=0.3, y2=0.75)  # below the head region
ROW = FullScaleRow(128.0, 72.0, 64.0, 72.0, (150.4, 100.9), (176.6, 101.2), 0.93)


def frame_and_person(clock: FakeClock, stamper: FrameStamper, *, boxes=(PERSON_BOX,), dt: float = 1.0):
    clock.advance(dt)
    frame = stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
    persons = tuple(TrackObservation.detected(frame, track_id=i + 1, box=box, confidence=0.9,
                                              status=TrackStatus.CONFIRMED) for i, box in enumerate(boxes))
    return frame, persons


def production_face(frame, box=FACE_BOX, embedding=(1.0, 0.0, 0.0, 0.0)):
    return FaceObservation(frame=frame.key, box=box, quality=0.92, embedding=embedding)


def full(*candidates: tuple[float, NormalizedBox], rows=None):
    return FullScale(tuple(candidates), 48.0, tuple(ROW for _ in candidates) if rows is None else tuple(rows))


@pytest.fixture
def world():
    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    return SimpleNamespace(clock=clock, stamper=stamper)


# ---------------------------------------------------------------- the adapter: region, border, calls


def yunet_client_region(face: list[float]) -> dict:
    """DeepFace 0.0.99 ``YuNetClient.detect_faces``, transcribed for a picture of at most 640 pixels (no resize)."""
    (x, y, w, h, x_le, y_le, x_re, y_re) = list(map(int, face[:8]))
    x = max(x, 0)
    y = max(y, 0)
    confidence = float(face[-1])
    return {"x": x, "y": y, "w": w, "h": h, "confidence": confidence, "left_eye": (x_re, y_re),
            "right_eye": (x_le, y_le)}


@pytest.mark.parametrize("face", [
    [128.7, 72.2, 64.9, 72.1, 150.4, 100.9, 176.6, 101.2, 160.0, 115.0, 152.0, 128.0, 175.0, 128.0, 0.93],
    [-3.7, -0.4, 40.2, 50.8, 5.9, 12.1, 25.3, 11.7, 15.0, 20.0, 8.0, 30.0, 22.0, 30.0, 0.91],  # past the edge
    [600.0, 400.5, 39.9, 79.99, 610.1, 420.9, 630.9, 421.2, 620.0, 440.0, 612.0, 455.0, 628.0, 455.0, 0.9],
])
def test_the_region_is_what_deepfaces_yunet_client_reports_unresized(face) -> None:
    detector = FakeYuNet([face])
    run = full_scale_pass(detector, 640, 480, "frame", rows=True)
    (row,) = run.rows
    assert deepface_region(row) == yunet_client_region(face)
    region = deepface_region(row)
    assert region["left_eye"] == (int(face[6]), int(face[7])) and region["right_eye"] == (int(face[4]), int(face[5]))


def test_the_border_is_detect_faces_border_and_moves_box_and_eyes() -> None:
    region = deepface_region(ROW)
    moved, wb, hb = padded_region(region, 640, 480)
    assert (wb, hb) == (320, 240)
    assert (moved["x"], moved["y"], moved["w"], moved["h"]) == (128 + 320, 72 + 240, 64, 72)
    assert moved["left_eye"] == (176 + 320, 101 + 240) and moved["right_eye"] == (150 + 320, 100 + 240)
    assert padded_region(region, 641, 481)[1:] == (320, 240)  # int(0.5 * size), as detect_faces computes it


class FakeYuNet:
    def __init__(self, rows) -> None:
        self.rows = rows

    def setInputSize(self, size) -> None:  # noqa: N802 - OpenCV's name
        pass

    def setScoreThreshold(self, value) -> None:  # noqa: N802
        pass

    def detect(self, image):
        return 1, self.rows


def test_rows_are_kept_only_for_comparison_runs_in_the_candidates_order() -> None:
    rows = [[10, 10, 20, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0.5], [300, 100, 60, 70, 310, 120, 340, 121, 0, 0, 0, 0,
                                                                     0, 0, 0.95], [5, 5, 9, 9, 0.92]]
    plain = full_scale_pass(FakeYuNet(rows), 640, 480, "f")
    assert plain.rows == () and [round(s, 2) for s, _ in plain.candidates] == [0.95, 0.92, 0.5]  # D86 unchanged
    kept = full_scale_pass(FakeYuNet(rows), 640, 480, "f", rows=True)
    assert [round(s, 2) for s, _ in kept.candidates] == [round(s, 2) for s, _ in plain.candidates]
    assert kept.rows[0].eye_4 == (310.0, 120.0) and kept.rows[0].eye_6 == (340.0, 121.0)
    assert kept.rows[1] is None  # a row without landmarks: never compared
    assert kept.rows[2].x == 10.0


class Crop:
    """An array stand-in: shape, and division recorded (the comparison divides by 255, as extract_faces does)."""

    def __init__(self, shape=(80, 64, 3)) -> None:
        self.shape, self.divided_by = shape, None

    def __truediv__(self, other):
        divided = Crop(self.shape)
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


class ComparisonRuntime:
    """TensorFlow, OpenCV and DeepFace stand-ins: the production calls, the model cache and the comparison's calls."""

    def __init__(self, *, cached=True, crop=None, represent_output=None, extract_error=None) -> None:
        self.calls: list = []
        self.model = object()
        self.crop = Crop() if crop is None else crop
        self.represent_output = represent_output
        self.extract_error = extract_error
        cache = {"facial_recognition": {}}
        self.ns = SimpleNamespace(
            tf=SimpleNamespace(config=SimpleNamespace(get_visible_devices=lambda kind: []), __version__="2.21.0"),
            cv2=SimpleNamespace(__version__="4.13.0", BORDER_CONSTANT=0, FaceDetectorYN_create=self._create,
                                copyMakeBorder=self._border),
            numpy=SimpleNamespace(__version__="2.2.6", zeros=lambda shape, dtype: ("zeros", shape), uint8="uint8"),
            tf_keras=SimpleNamespace(__version__="2.21.0"), deepface=SimpleNamespace(__version__="0.0.99"),
            DeepFace=SimpleNamespace(build_model=self._build, represent=self.represent),
            detection=SimpleNamespace(extract_face=self.extract_face),
            modeling=SimpleNamespace(cached_models=cache), FacialAreaRegion=Region)
        self._cache, self._cached = cache, cached

    def _build(self, name):
        self.calls.append(("build", name))
        if self._cached:
            self._cache["facial_recognition"][name] = self.model

    def _create(self, *args):
        return FakeYuNet([[300, 100, 60, 70, 310, 120, 340, 121, 0, 0, 0, 0, 0, 0, 0.95]])

    def _border(self, image, top, bottom, left, right, kind, value):
        self.calls.append(("border", image, top, bottom, left, right, kind, value))
        return ("padded", image)

    def extract_face(self, facial_area, img, align, expand_percentage, width_border, height_border, detector_backend):
        self.calls.append(("extract", dataclasses.asdict(facial_area), img, align, expand_percentage, width_border,
                           height_border, detector_backend))
        if self.extract_error is not None:
            raise self.extract_error
        return SimpleNamespace(img=self.crop)

    def represent(self, img_path, model_name="VGG-Face", enforce_detection=True, detector_backend="opencv",
                  align=True, expand_percentage=0, normalization="base", anti_spoofing=False, max_faces=None,
                  l2_normalize=False, minmax_normalize=False, return_face=False, cryptosystem=None):
        kwargs = {"model_name": model_name, "enforce_detection": enforce_detection,
                  "detector_backend": detector_backend, "align": align, "expand_percentage": expand_percentage,
                  "normalization": normalization}
        self.calls.append(("represent", img_path, kwargs))
        if detector_backend == "skip":
            return self.represent_output if self.represent_output is not None else [
                {"embedding": [3.0] + [0.0] * 511, "facial_area": {}, "face_confidence": 0}]
        return [{"embedding": [0.0] * 512, "facial_area": {"x": 0, "y": 0, "w": 640, "h": 480},
                 "face_confidence": 0}]


def comparison_backend(tmp_path, runtime: ComparisonRuntime, *, comparison=True) -> LegacyDeepFaceBackend:
    for name, _ in (MODEL_WEIGHTS, DETECTOR_WEIGHTS):
        (tmp_path / name).write_bytes(b"synthetic")
    digests = {MODEL_WEIGHTS[0]: MODEL_WEIGHTS[1], DETECTOR_WEIGHTS[0]: DETECTOR_WEIGHTS[1]}
    backend = LegacyDeepFaceBackend(tmp_path, importer=lambda: runtime.ns, sha256=lambda p: digests[p.name])
    backend.enable_full_scale()
    if comparison:
        backend.enable_comparison()
    backend.load()
    return backend


def test_the_comparison_cuts_aligns_and_embeds_as_production_with_the_cached_model(tmp_path) -> None:
    runtime = ComparisonRuntime()
    backend = comparison_backend(tmp_path, runtime)
    assert backend.comparison_enabled
    before = [c for c in runtime.calls if c[0] == "build"]
    embedding = backend.embed_full_scale(640, 480, "frame", ROW)
    assert embedding == (1.0,) + (0.0,) * 511  # unit length, as production's
    border, extract, represent = [c for c in runtime.calls if c[0] in ("border", "extract", "represent")][-3:]
    assert border == ("border", "frame", 240, 240, 320, 320, 0, [0, 0, 0])  # detect_faces's black border
    region = deepface_region(ROW)
    assert extract[1] == {"x": region["x"] + 320, "y": region["y"] + 240, "w": region["w"], "h": region["h"],
                          "left_eye": (176 + 320, 101 + 240), "right_eye": (150 + 320, 100 + 240),
                          "confidence": 0.93, "nose": None, "mouth_right": None, "mouth_left": None}
    assert extract[2:] == (("padded", "frame"), True, 0, 320, 240, "yunet")
    assert represent[1].divided_by == 255  # extract_faces divides by 255; represent's skip path does not
    assert represent[2] == {"model_name": MODEL_NAME, "enforce_detection": False, "detector_backend": "skip",
                            "align": True, "expand_percentage": 0, "normalization": "base"}
    assert [c for c in runtime.calls if c[0] == "build"] == before  # no model is built for the comparison
    assert set(COMPARISON_PARAMETERS["represent"]) <= set(represent[2]) | {"img_path"}


def test_the_production_result_and_the_full_scale_numbers_are_unchanged_by_the_comparison(tmp_path) -> None:
    frame = FrameKey(camera_id="cam-1", boot_id="b", run_id="r", stream_epoch=0, frame_seq=1)
    with_rows = comparison_backend(tmp_path, ComparisonRuntime()).faces(frame, 640, 480, "frame")
    without = comparison_backend(tmp_path, ComparisonRuntime(), comparison=False).faces(frame, 640, 480, "frame")
    assert (with_rows.faces, with_rows.detected, with_rows.fallback_dropped) == (
        without.faces, without.detected, without.fallback_dropped)
    assert with_rows.full_scale.candidates == without.full_scale.candidates
    assert without.full_scale.rows == () and with_rows.full_scale.rows[0] is not None


@pytest.mark.parametrize(("runtime", "label"), [
    (ComparisonRuntime(crop=Crop((0, 64, 3))), "empty_crop"),
    (ComparisonRuntime(extract_error=RuntimeError("cv2 says no")), "comparison_failed"),
    (ComparisonRuntime(represent_output=[{"embedding": [1.0] * 512}] * 2), "inconsistent_output"),
    (ComparisonRuntime(represent_output=[{"embedding": [0.0] * 512}]), "inconsistent_output"),
    (ComparisonRuntime(represent_output=[{"embedding": [1.0] * 10}]), "inconsistent_output"),
])
def test_a_failed_comparison_is_a_fixed_label_never_an_embedding(tmp_path, runtime, label) -> None:
    backend = comparison_backend(tmp_path, runtime)
    with pytest.raises(FaceBackendError) as raised:
        backend.embed_full_scale(640, 480, "frame", ROW)
    assert raised.value.label == label and "cv2 says no" not in str(raised.value)


def test_no_second_model_is_ever_loaded(tmp_path) -> None:
    runtime = ComparisonRuntime()
    backend = comparison_backend(tmp_path, runtime)
    runtime.ns.modeling.cached_models["facial_recognition"].clear()  # evicted: represent() would build another
    calls = len(runtime.calls)
    with pytest.raises(FaceBackendError, match="model_not_cached"):
        backend.embed_full_scale(640, 480, "frame", ROW)
    assert runtime.calls[calls:] == []  # nothing called at all
    with pytest.raises(FaceBackendError, match="model_not_cached"):
        comparison_backend(tmp_path, ComparisonRuntime(cached=False))  # the load fails closed


def test_the_load_checks_deepfaces_functions_and_the_order_of_enabling(tmp_path) -> None:
    runtime = ComparisonRuntime()
    runtime.ns.detection.extract_face = lambda facial_area, img, align: None  # an older or newer signature
    with pytest.raises(FaceBackendError, match="comparison_unavailable"):
        comparison_backend(tmp_path, runtime)
    plain = LegacyDeepFaceBackend(tmp_path)
    with pytest.raises(FaceBackendError, match="comparison_needs_full_scale"):
        plain.enable_comparison()
    loaded = comparison_backend(tmp_path, ComparisonRuntime(), comparison=False)
    with pytest.raises(FaceBackendError, match="comparison_not_enabled"):
        loaded.embed_full_scale(640, 480, "frame", ROW)
    with pytest.raises(FaceBackendError, match="already_loaded"):
        loaded.enable_comparison()


# ---------------------------------------------------------------- eligibility and the budget


def test_eligibility_uses_production_and_full_scale_association_on_the_same_frame(world) -> None:
    frame, persons = frame_and_person(world.clock, world.stamper)
    only = comparison_targets(frame, persons, (), full((0.93, FACE_BOX)), GEOMETRY)
    assert [(t.kind, t.candidate, t.score, t.width_px) for t in only] == [(Kind.FULL_SCALE_ONLY, 0, 0.93, 64)]
    both = comparison_targets(frame, persons, (production_face(frame),), full((0.93, FACE_BOX)), GEOMETRY)
    assert [t.kind for t in both] == [Kind.BOTH]
    assert comparison_targets(frame, persons, (), full((0.89, FACE_BOX)), GEOMETRY) == []  # under the 0.9 count
    assert comparison_targets(frame, persons, (), full((0.95, OUTSIDE_BOX)), GEOMETRY) == []  # not in a head region
    assert comparison_targets(frame, persons, (), None, GEOMETRY) == []
    assert comparison_targets(frame, (), (), full((0.95, FACE_BOX)), GEOMETRY) == []
    assert comparison_module.FULL_SCALE_COUNTED == runtime_module.FULL_SCALE_COUNTED


def test_a_production_face_owned_ambiguously_is_not_eligible(world) -> None:
    overlap = NormalizedBox(x1=0.12, y1=0.1, x2=0.42, y2=0.9)  # a second person over the same head
    frame, persons = frame_and_person(world.clock, world.stamper, boxes=(PERSON_BOX, overlap))
    assert comparison_targets(frame, persons, (production_face(frame),), full((0.93, FACE_BOX)), GEOMETRY) == []


def target(frame, kind=Kind.FULL_SCALE_ONLY, track=1):
    key = TrackObservation.detected(frame, track_id=track, box=PERSON_BOX, confidence=0.9,
                                    status=TrackStatus.CONFIRMED).key
    return comparison_module.Target(key, kind, 0, 0.93, 64)


def test_the_budget_one_per_frame_primary_first_interval_cap_and_quota(world) -> None:
    comparator = FullScaleComparator(lambda f, i, r: (1.0,), GEOMETRY, ComparisonPolicy(cap=7))
    rows = (ROW,)
    frame, _ = frame_and_person(world.clock, world.stamper)
    now = frame.ingest_mono_ns
    chosen, skipped = comparator.plan(frame, [target(frame, Kind.BOTH, 1), target(frame, Kind.FULL_SCALE_ONLY, 2)],
                                      rows, now)
    assert chosen.kind is Kind.FULL_SCALE_ONLY and chosen.key.track_id == 2
    assert [(t.key.track_id, d) for t, d in skipped] == [(1, Disposition.SKIPPED_ONE_PER_RESULT)]
    chosen, skipped = comparator.plan(frame, [target(frame)], rows, now + NS_PER_SECOND)
    assert chosen is None and skipped[0][1] is Disposition.SKIPPED_INTERVAL
    plans = []
    for step in range(1, 10):  # one eligible person every 4 s: the interval never refuses
        f, _ = frame_and_person(world.clock, world.stamper, dt=COMPARISON_INTERVAL_S)
        kind = Kind.BOTH if step in (1, 2, 5) else Kind.FULL_SCALE_ONLY
        chosen, skipped = comparator.plan(f, [target(f, kind)], rows, f.ingest_mono_ns)
        plans.append(chosen.kind.value if chosen else skipped[0][1].value)
    # a control waits until it would be at most one in four attempts (counting itself); then the cap (7) stops all
    assert plans == ["skipped_quota", "skipped_quota", "full_scale_only", "full_scale_only", "both",
                     "full_scale_only", "full_scale_only", "full_scale_only", "skipped_cap"]
    assert comparator.counters["attempted_both"] == 1 and comparator.counters["attempted_full_scale_only"] == 6
    assert (COMPARISON_INTERVAL_S, COMPARISON_CAP, CONTROL_SHARE) == (4.0, 75, 4)


def test_stale_frames_missing_landmarks_and_a_blocked_worker_are_skipped(world) -> None:
    comparator = FullScaleComparator(lambda f, i, r: (1.0,), GEOMETRY)
    frame, _ = frame_and_person(world.clock, world.stamper)
    stale = frame.ingest_mono_ns + 3 * NS_PER_SECOND + 1
    assert comparator.plan(frame, [target(frame)], (ROW,), stale)[1][0][1] is Disposition.SKIPPED_STALE
    assert comparator.plan(frame, [target(frame)], (None,), frame.ingest_mono_ns)[1][0][1] is \
        Disposition.SKIPPED_NO_LANDMARKS
    for blocked in (Disposition.SKIPPED_STOPPING, Disposition.SKIPPED_PRODUCTION_WAITING):
        assert comparator.plan(frame, [target(frame)], (ROW,), frame.ingest_mono_ns, blocked)[1][0][1] is blocked
    assert comparator.counters["attempted_full_scale_only"] == 0


def test_a_failed_embedding_is_failed_with_its_label_and_time_never_a_nonmatch(world) -> None:
    def broken(frame, image, row):
        raise FaceBackendError("empty_crop")

    ticks = iter([1.0, 1.25])
    comparator = FullScaleComparator(broken, GEOMETRY, perf_counter=lambda: next(ticks))
    frame, _ = frame_and_person(world.clock, world.stamper)
    outcome = comparator.compare(frame, "image", target(frame), (ROW,))
    assert (outcome.disposition, outcome.error, outcome.embedding, outcome.ms) == (
        Disposition.FAILED, "empty_crop", None, 250.0)
    comparator.record(outcome)
    assert comparator.counters["failed_full_scale_only"] == 1 and comparator.counters["failed:empty_crop"] == 1
    assert error_label(ValueError("x")) == "comparison_failed:ValueError"
    assert error_label(FaceBackendError("comparison_failed", "RuntimeError")) == "comparison_failed:RuntimeError"
    nan = FullScaleComparator(lambda f, i, r: (float("nan"),), GEOMETRY).compare(frame, "i", target(frame), (ROW,))
    assert nan.disposition is Disposition.FAILED and nan.error == "comparison_failed:ValueError"


# ---------------------------------------------------------------- the worker


class Image:
    def __init__(self, tag: int = 0) -> None:
        self.tag = tag

    def copy(self) -> "Image":
        return Image(self.tag)

    def setflags(self, *, write: bool) -> None:
        pass


def wait_for(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.002)


class Embedder:
    """An embedding that blocks until released (or fails), recording which frame it saw."""

    def __init__(self, error: Exception | None = None) -> None:
        self.entered, self.release = threading.Semaphore(0), threading.Semaphore(0)
        self.seen, self.error = [], error

    def __call__(self, frame, image, row):
        self.seen.append(frame.frame_seq)
        self.entered.release()
        assert self.release.acquire(timeout=5)
        if self.error is not None:
            raise self.error
        return (1.0, 0.0, 0.0, 0.0)


def full_scale_only(frame, image):
    return FaceRun((), 0, 1, full((0.93, FACE_BOX)))


def ticked(worker: FaceWorker, world, dt: float = 1.0):
    frame, persons = frame_and_person(world.clock, world.stamper, dt=dt)
    assert worker.consider(frame, persons, Image(frame.frame_seq)) == "offered"
    return frame


def test_production_is_published_first_and_a_tick_offered_meanwhile_is_yielded(world) -> None:
    embed = Embedder()
    worker = FaceWorker(full_scale_only, world.clock, comparator=FullScaleComparator(embed, GEOMETRY))
    worker.start()
    first = ticked(worker, world)
    assert embed.entered.acquire(timeout=5)  # comparing frame 1 now
    result = worker.drain()
    assert result is not None and result.frame == first  # production's result was published before it
    assert worker.status()["busy"] is True
    ticked(worker, world)  # offered while the comparison runs
    embed.release.release()
    wait_for(lambda: not worker.status()["busy"])
    counters = worker.status()["counters"]
    assert counters["yielded_to_comparison"] == 1 and counters["started"] == 1 and not worker.status()["waiting"]
    outcomes = worker.drain_comparisons()
    assert [(o.frame, o.disposition) for o in outcomes] == [(first, Disposition.COMPLETED)]
    third = ticked(worker, world)  # the next tick runs normally, on time
    wait_for(lambda: worker.status()["counters"].get("completed") == 2 and not worker.status()["busy"])
    assert worker.drain().frame == third
    assert [o.disposition for o in worker.drain_comparisons()] == [Disposition.SKIPPED_INTERVAL]
    assert embed.seen == [first.frame_seq]  # within the 4 s interval: no second comparison
    assert worker.stop(5)
    assert third.frame_seq == first.frame_seq + 2


def test_a_frames_comparisons_wait_until_its_production_result_is_taken(world) -> None:
    embed = Embedder()
    embed.release.release()  # does not block
    worker = FaceWorker(full_scale_only, world.clock, comparator=FullScaleComparator(embed, GEOMETRY))
    worker.start()
    ticked(worker, world)
    wait_for(lambda: not worker.status()["busy"] and embed.seen)
    assert worker.drain_comparisons() == []  # its production result is still waiting for the loop
    assert worker.drain() is not None
    assert [o.disposition for o in worker.drain_comparisons()] == [Disposition.COMPLETED]
    assert worker.stop(5)


def test_no_comparison_starts_while_a_production_job_waits(world) -> None:
    gate = threading.Semaphore(0)

    def slow(frame, image):
        assert gate.acquire(timeout=5)
        return full_scale_only(frame, image)

    embed = Embedder()
    worker = FaceWorker(slow, world.clock, comparator=FullScaleComparator(embed, GEOMETRY))
    worker.start()
    ticked(worker, world)
    wait_for(lambda: worker.status()["busy"])
    ticked(worker, world)  # waits behind the running production analysis
    gate.release()  # the first finishes; a production job is waiting
    wait_for(lambda: worker.status()["counters"].get("started") == 2)
    worker.drain()
    assert [o.disposition for o in worker.drain_comparisons()] == [Disposition.SKIPPED_PRODUCTION_WAITING]
    embed.release.release()
    gate.release()
    wait_for(lambda: not worker.status()["busy"])
    assert worker.stop(5)


def test_a_failing_comparison_leaves_production_and_the_worker_running(world) -> None:
    embed = Embedder(error=RuntimeError("boom"))
    embed.release.release()
    embed.release.release()  # two comparisons below, neither blocks
    worker = FaceWorker(full_scale_only, world.clock, comparator=FullScaleComparator(embed, GEOMETRY))
    worker.start()
    ticked(worker, world)
    assert embed.entered.acquire(timeout=5)
    wait_for(lambda: not worker.status()["busy"])
    assert worker.drain().faces == ()
    (outcome,) = worker.drain_comparisons()
    assert (outcome.disposition, outcome.error, outcome.embedding) == (
        Disposition.FAILED, "comparison_failed:RuntimeError", None)
    status = worker.status()
    assert status["counters"]["completed"] == 1 and status["comparison"]["counters"]["failed_full_scale_only"] == 1
    ticked(worker, world, dt=5.0)
    wait_for(lambda: worker.status()["counters"]["completed"] == 2)
    assert worker.stop(5)


def test_the_outbox_is_bounded_and_its_overflow_counted(world) -> None:
    worker = FaceWorker(full_scale_only, world.clock, comparator=FullScaleComparator(lambda f, i, r: (1.0,), GEOMETRY))
    frame, _ = frame_and_person(world.clock, world.stamper)
    worker._keep_outcomes([ComparisonOutcome(frame, 1, Kind.BOTH, Disposition.SKIPPED_QUOTA, 0.93, 64)] *
                          (COMPARISON_OUTBOX + 3))
    assert len(worker.drain_comparisons()) == COMPARISON_OUTBOX
    assert worker.status()["counters"]["comparison_outcomes_dropped"] == 3


def test_a_worker_without_a_comparator_is_unchanged(world) -> None:
    worker = FaceWorker(full_scale_only, world.clock)
    worker.start()
    ticked(worker, world)
    wait_for(lambda: worker.drain() is not None)
    status = worker.status()
    assert "comparison" not in status and "yielded_to_comparison" not in status["counters"]
    assert worker.drain_comparisons() == []
    assert worker.stop(5)


# ---------------------------------------------------------------- the runtime: records, and no interference


A_VEC = (1.0, 0.0, 0.0, 0.0)
ENROLLMENT = Enrollment(model_revision="synthetic", dimension=4, identities=(
    EnrolledIdentity(identity_id="idn-00000000000a", prototypes=(A_VEC,)),))
CONFIG = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})


class Core:
    def __init__(self, face=Capability.AVAILABLE) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stream = self.stamper.connect()
        self.core = EdgeCore(CONFIG, self.clock, None, face_recognition=face, enrollment=ENROLLMENT)
        self.transitions: list = []

    def step(self, embedding, *, compare=None, connected="live"):
        self.clock.advance(1.0)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        persons = (TrackObservation.detected(frame, track_id=1, box=PERSON_BOX, confidence=0.9,
                                             status=TrackStatus.CONFIRMED),)
        out = self.core.on_frame(frame, persons, self.stream)
        self.transitions += [t.record() for t in out.identity_transitions]
        faces = () if embedding is None else (production_face(frame, embedding=embedding),)
        out = self.core.on_face_outcome(frame, persons, faces, self.stream if connected == "live" else connected)
        self.transitions += [t.record() for t in out.identity_transitions]
        records = []
        if compare is not None:
            outcome = ComparisonOutcome(frame, 1, Kind.FULL_SCALE_ONLY, Disposition.COMPLETED, 0.93, 64,
                                        embedding=compare, ms=812.0)
            records.append(self.core.on_face_comparison(outcome).record())
        return frame, out, records


def test_a_completed_comparison_gets_its_similarity_and_a_schema_4_record() -> None:
    core = Core()
    frame, _, (record,) = core.step(None, compare=(0.8, 0.6, 0.0, 0.0))
    assert record == {"identity": "comparison", "schema": IDENTITY_COMPARISON_SCHEMA, "stream_epoch": frame.stream_epoch,
                      "frame_seq": frame.frame_seq, "frame_mono_ns": frame.ingest_mono_ns, "track_id": 1,
                      "kind": "full_scale_only", "disposition": "completed", "score": 0.93, "w_px": 64,
                      "similarity": round(best_match((0.8, 0.6, 0.0, 0.0), ENROLLMENT).similarity, 3), "ms": 812.0,
                      "error": None}
    assert record["similarity"] == 0.8 and "idn-" not in json.dumps(record)
    with pytest.raises(ValueError):
        runtime_module.ComparisonRecord(0, 0, 0, 1, "both", "completed", 0.9, 1, None, None, None).record(3)


def test_without_an_applied_production_result_the_embedding_is_discarded() -> None:
    core = Core()
    _, _, (record,) = core.step(None, compare=A_VEC, connected=None)  # production rejected: not live
    assert (record["disposition"], record["similarity"]) == ("production_not_applied", None)
    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    for _ in range(20):  # a frame sequence number the core has not seen
        unseen, _ = frame_and_person(clock, stamper)
    outcome = ComparisonOutcome(unseen, 1, Kind.BOTH, Disposition.COMPLETED, 0.93, 64, embedding=A_VEC, ms=1.0)
    assert core.core.on_face_comparison(outcome).disposition == "production_not_applied"
    failed = ComparisonOutcome(unseen, 1, Kind.BOTH, Disposition.FAILED, 0.93, 64, ms=1.0, error="empty_crop")
    assert core.core.on_face_comparison(failed).record()["error"] == "empty_crop"
    assert core.core.comparison_counters == {"full_scale_only:production_not_applied": 1,
                                              "both:production_not_applied": 1, "both:failed": 1}


def test_comparisons_never_change_identity_state_votes_counters_or_transitions() -> None:
    script = [A_VEC, None, A_VEC, (0.0, 1.0, 0.0, 0.0), None, None, A_VEC, A_VEC, None]
    plain, compared = Core(), Core()
    for embedding in script:
        _, out_plain, _ = plain.step(embedding)
        _, out_compared, records = compared.step(embedding, compare=(0.0, 1.0, 0.0, 0.0))  # a nonmatching shadow
        assert records[0]["disposition"] == "completed"
        assert out_plain.state.people == out_compared.state.people
        assert out_plain.face_result.record(4)["persons"] == out_compared.face_result.record(4)["persons"]
    assert plain.transitions == compared.transitions
    assert plain.core.identity_counters == compared.core.identity_counters
    assert plain.core.diagnostics() == compared.core.diagnostics()


def test_the_diagnostic_similarity_reads_the_enrollment_only() -> None:
    resolver = IdentityResolver(ENROLLMENT, IdentityPolicy(match_threshold=0.7))
    assert resolver.diagnostic_similarity((0.6, 0.8, 0.0, 0.0)) == pytest.approx(0.6)
    assert resolver.diagnostic_similarity((1.0, 0.0)) is None  # another dimension
    assert IdentityResolver(None).diagnostic_similarity(A_VEC) is None
    assert resolver.counters == {} and resolver.tracked == 0
