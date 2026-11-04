"""Session 116, part 11: the door-range check (benchmarks/runner/door_range_check.py).

No camera and no model: fakes stand in for OpenCV, DeepFace and the streams; the frames' stamping, the vote rule, the
hand-off and the capture worker are the runtime's own. What is tested:
- **The configurations:** ``native`` is D88's steps, tied to D88 on a frame it accepts, and also takes the main stream's
  frame D88 refuses; ``production`` is the runtime's call, unchanged.
- **The thresholds are the runtime's:** 0.9 for a face, the vote rule for a match. A candidate under 0.9 is a size and a
  score only, never embedded or voted.
- **The timeline:** readiness only after both streams gave a frame; samples only inside the declared windows; the
  operator's cues fit the guard's limits and leave walking time before each window.
- **Staleness and ambiguity:** a stale pair is not analysed; a frame with several faces gives no size or vote.
- **Numbers only:** no embedding, identity ID, frame or URL in the result, and similarity only on request.
- **Execution is all it judges:** a short window is incomplete, a guard's stop keeps what was measured.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sentinel.config import CaptureConfig, IdentityConfig
from sentinel.contracts import FrameKey, PixelFormat
from sentinel.identity.gallery import unit
from sentinel.identity.legacy_deepface import (
    DETECTOR_WEIGHTS,
    FACE_ARGS,
    FULL_SCALE_FLOOR,
    MODEL_NAME,
    MODEL_WEIGHTS,
    FaceBackendError,
    FaceRun,
    to_face_run,
)
from sentinel.identity.matching import EnrolledIdentity, Enrollment
from sentinel.identity.state import IdentityResolver
from sentinel.identity.yunet import YUNET_THRESHOLD
from sentinel.media.capture import CapturedFrame, DecodedFrame
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.media.frames import FrameStamper

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
FRAME = FrameKey(camera_id="cam-1", boot_id="b", run_id="r", stream_epoch=0, frame_seq=1)
IDENTITY = "idn-00000000000a"
SIZES = {"sub": (640, 480), "main": (2304, 1296)}
URLS = {"SENTINEL_RTSP_URL": "rtsp://operator:hunter2@192.0.2.10:554/sub",
        "SENTINEL_RTSP_MAIN_URL": "rtsp://operator:hunter2@192.0.2.10:554/main"}


def vec(*head: float) -> tuple[float, ...]:
    return unit(tuple(head) + (0.0,) * (512 - len(head)))


ENROLLED = vec(1.0)
MATCHING, NOT_MATCHING = vec(0.97, 0.1, 0.0, 0.05), vec(0.0, 0.1, 0.99)
ENROLLMENT = Enrollment(model_revision="synthetic", dimension=512, identities=(
    EnrolledIdentity(identity_id=IDENTITY, prototypes=(ENROLLED,)),))


@pytest.fixture(scope="module")
def dr():
    sys.path.insert(0, str(RUNNERS))
    try:
        yield importlib.import_module("door_range_check")
    finally:
        sys.path.remove(str(RUNNERS))


def yunet(x, y, w, h, score):
    """One YuNet row (x, y, w, h, right eye, left eye, nose, mouth corners, score) in frame pixels."""
    return [x, y, w, h, x + 0.3 * w, y + 0.4 * h, x + 0.7 * w, y + 0.4 * h, x + 0.5 * w, y + 0.6 * h, x + 0.35 * w,
            y + 0.8 * h, x + 0.65 * w, y + 0.8 * h, score]


class Cut:
    def __init__(self, index=0) -> None:
        self.shape, self.index = (80, 64, 3), index

    def __truediv__(self, other):
        return self


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
    def __init__(self, rows) -> None:
        self.rows, self.calls = rows, []

    def setInputSize(self, size) -> None:  # noqa: N802 - OpenCV's name
        self.calls.append(("size", size))

    def setScoreThreshold(self, value) -> None:  # noqa: N802
        self.calls.append(("threshold", value))

    def detect(self, image):
        self.calls.append(("detect", image))
        return 1, self.rows


class Runtime:
    """OpenCV, DeepFace and TensorFlow stand-ins, as test_face_candidate's: represent() with yunet fills the model
    cache, and a represent(skip) answers per cut."""

    def __init__(self, rows=(), production=None) -> None:
        self.calls: list = []
        self.detector, self.model = Detector(list(rows)), object()
        self.production = production
        self.cache: dict = {"facial_recognition": {}, "face_detector": {}}
        self.ns = SimpleNamespace(
            tf=SimpleNamespace(config=SimpleNamespace(get_visible_devices=lambda kind: []), __version__="2.21.0"),
            cv2=SimpleNamespace(__version__="4.13.0", BORDER_CONSTANT=0, FaceDetectorYN_create=lambda *a: Detector([]),
                                copyMakeBorder=self._border),
            numpy=SimpleNamespace(__version__="2.2.6", zeros=lambda shape, dtype: ("zeros", shape), uint8="uint8"),
            tf_keras=SimpleNamespace(__version__="2.21.0"), deepface=SimpleNamespace(__version__="0.0.99"),
            DeepFace=SimpleNamespace(build_model=self._build, represent=self.represent),
            detection=SimpleNamespace(extract_face=self.extract_face),
            modeling=SimpleNamespace(cached_models=self.cache), FacialAreaRegion=Region)

    def _build(self, name):
        self.cache["facial_recognition"][name] = self.model

    def _border(self, image, top, bottom, left, right, kind, value):
        self.calls.append(("border", top, left))
        return ("padded", image)

    def extract_face(self, facial_area, img, align, expand_percentage, width_border, height_border, detector_backend):
        index = sum(1 for c in self.calls if c[0] == "extract")
        self.calls.append(("extract", dataclasses.asdict(facial_area), width_border, height_border))
        return SimpleNamespace(img=Cut(index), facial_area=Region(
            x=facial_area.x - width_border, y=facial_area.y - height_border, w=facial_area.w, h=facial_area.h))

    def represent(self, img_path, model_name="VGG-Face", enforce_detection=True, detector_backend="opencv",
                  align=True, expand_percentage=0, normalization="base", **_):
        self.calls.append(("represent", detector_backend, dict(
            model_name=model_name, enforce_detection=enforce_detection, detector_backend=detector_backend, align=align,
            expand_percentage=expand_percentage, normalization=normalization)))
        if detector_backend != "skip":
            self.cache["face_detector"]["yunet"] = SimpleNamespace(model=self.detector)
            if self.production is not None and img_path != "warm":
                return self.production
            return [{"embedding": [0.0] * 511 + [1.0], "facial_area": {"x": 0, "y": 0, "w": 640, "h": 480},
                     "face_confidence": 0}]
        images = img_path if isinstance(img_path, list) else [img_path]
        out = [[{"embedding": list(MATCHING), "facial_area": {}, "face_confidence": 0}] for _ in images]
        return out[0] if len(images) == 1 else out


def loaded(dr, tmp_path, runtime: Runtime):
    for name, _ in (MODEL_WEIGHTS, DETECTOR_WEIGHTS):
        (tmp_path / name).write_bytes(b"synthetic")
    digests = {MODEL_WEIGHTS[0]: MODEL_WEIGHTS[1], DETECTOR_WEIGHTS[0]: DETECTOR_WEIGHTS[1]}
    backend = dr.RangeBackend(tmp_path, importer=lambda: runtime.ns, sha256=lambda p: digests[p.name])
    backend.select_full_scale_detection()
    backend.load()
    runtime.calls.clear()
    runtime.detector.calls.clear()
    return backend


# ---------------------------------------------------------------- the two configurations


def test_native_is_the_candidate_on_a_frame_the_candidate_accepts(dr, tmp_path) -> None:
    rows = [yunet(60.6, 60.3, 58.7, 62.2, 0.93), yunet(300.0, 80.0, 20.0, 22.0, 0.4)]
    native_rt, candidate_rt = Runtime(rows), Runtime(rows)
    native = loaded(dr, tmp_path, native_rt)
    candidate = loaded(dr, tmp_path, candidate_rt)
    got, under = native.native_faces(FRAME, 640, 480, "image")
    want = candidate.faces(FRAME, 640, 480, "image")
    assert got.faces == want.faces and got.geometry == want.geometry and got.detected == want.detected
    assert got.detection.best_below == want.detection.best_below
    assert native_rt.calls == candidate_rt.calls and native_rt.detector.calls == candidate_rt.detector.calls
    assert [row.score for row in under] == [0.4]


def test_native_takes_the_main_streams_frame_that_the_candidate_refuses(dr, tmp_path) -> None:
    runtime = Runtime([yunet(1000.0, 400.0, 44.0, 50.0, 0.95), yunet(1500.0, 500.0, 30.0, 33.0, 0.89)])
    backend = loaded(dr, tmp_path, runtime)
    with pytest.raises(FaceBackendError, match="frame_too_large"):
        backend.faces(FRAME, 2304, 1296, "image")  # D88 stays as it was
    runtime.detector.calls.clear()
    run, under = backend.native_faces(FRAME, 2304, 1296, "image")
    assert ("size", (2304, 1296)) in runtime.detector.calls and ("threshold", FULL_SCALE_FLOOR) in runtime.detector.calls
    assert len(run.faces) == 1 and run.faces[0].quality == 0.95
    assert [row.score for row in under] == [0.89]
    assert sum(1 for c in runtime.calls if c[0] == "extract") == 1  # the one under 0.9 is never cut or embedded
    assert run.detection.best_below[0] == 0.89


def test_production_is_the_runtimes_call_unchanged(dr, tmp_path) -> None:
    face = {"embedding": list(MATCHING), "face_confidence": 0.97,
            "facial_area": {"x": 100, "y": 90, "w": 40, "h": 44, "left_eye": (125, 100), "right_eye": (112, 101)}}
    fallback = {"embedding": list(MATCHING), "face_confidence": 0, "facial_area": {"x": 0, "y": 0, "w": 640, "h": 480}}
    runtime = Runtime(production=[face, fallback])
    backend = loaded(dr, tmp_path, runtime)
    run = backend.production_faces(FRAME, 640, 480, "image")
    assert runtime.calls[0][1] == FACE_ARGS["detector_backend"]
    assert runtime.calls[0][2] == {key: FACE_ARGS[key] for key in runtime.calls[0][2]}
    assert runtime.detector.calls == []  # the candidate's native pass does not run here
    assert len(run.faces) == 1 and run.fallback_dropped == 1
    assert dr.eye_spacing_px(run.geometry[0]) == round(math.dist((125, 100), (112, 101)), 1)
    assert run.faces[0] == to_face_run(FRAME, 640, 480, [face, fallback]).faces[0]


def test_thresholds_are_the_runtimes_own(dr) -> None:
    identity = IdentityConfig()
    assert (YUNET_THRESHOLD, identity.match_threshold, identity.margin, identity.min_quality) == (0.9, 0.70, 0.05, 0.6)
    policy = dr.policy_from(identity)
    assert (policy.match_threshold, policy.margin, policy.min_quality) == (0.70, 0.05, 0.6)


# ---------------------------------------------------------------- votes and records


def one_face(embedding, quality=0.95, box=(100, 100, 40, 40)):
    x, y, w, h = box
    return to_face_run(FRAME, 640, 480, [{"embedding": list(embedding), "face_confidence": quality,
                                          "facial_area": {"x": x, "y": y, "w": w, "h": h}}]).faces[0]


def test_a_vote_is_the_runtime_rule_and_a_fixed_label(dr) -> None:
    resolver = IdentityResolver(ENROLLMENT, dr.policy_from(IdentityConfig()))
    matched = dr.face_record(one_face(MATCHING), None, 640, 480, resolver)
    assert matched["vote"] == "match" and matched["similarity"] > 0.7 and matched["w_px"] == 40.0
    assert dr.face_record(one_face(NOT_MATCHING), None, 640, 480, resolver)["vote"] == "unknown"
    assert dr.face_record(one_face(MATCHING, quality=0.5), None, 640, 480, resolver)["vote"] == "low_quality"
    two = Enrollment(model_revision="synthetic", dimension=512, identities=(
        EnrolledIdentity(identity_id=IDENTITY, prototypes=(ENROLLED,)),
        EnrolledIdentity(identity_id="idn-00000000000b", prototypes=(vec(0.98, 0.2),))))
    assert dr.face_record(one_face(vec(1.0, 0.1)), None, 640, 480, IdentityResolver(two, dr.policy_from(
        IdentityConfig())))["vote"] == "margin"
    assert IDENTITY not in json.dumps(matched)
    assert resolver.counters == {}  # read only: the rule's counters and votes are not touched


# ---------------------------------------------------------------- the two streams' relation


def test_affine_facts_tell_a_crop_from_a_squeeze(dr) -> None:
    crop = dr.affine_facts([[2.7, 0.0, 288.0], [0.0, 2.7, 0.0]], (640, 480), (2304, 1296))
    assert (crop["scale_x"], crop["scale_y"]) == (2.7, 2.7)
    assert crop["left"] == 0.125 and crop["right"] == 0.875 and crop["top"] == 0.0 and crop["bottom"] == 1.0
    squeeze = dr.affine_facts([[3.6, 0.0, 0.0], [0.0, 2.7, 0.0]], (640, 480), (2304, 1296))
    assert squeeze["left"] == 0.0 and squeeze["right"] == 1.0 and squeeze["scale_x"] != squeeze["scale_y"]


def test_fov_estimate_recovers_a_known_mapping_with_real_opencv(dr) -> None:
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    generator = numpy.random.default_rng(116)
    scene = cv2.GaussianBlur(generator.integers(0, 255, (1296, 2304, 3), dtype=numpy.uint8), (7, 7), 0)
    sub = cv2.resize(scene[:, 288:2016], (640, 480), interpolation=cv2.INTER_AREA)  # a 4:3 centre crop
    got = dr.fov_estimate(cv2, numpy, sub, scene)
    assert got["status"] == "fitted" and abs(got["scale_x"] - 2.7) < 0.05 and abs(got["left"] - 0.125) < 0.02


# ---------------------------------------------------------------- the timeline, on fake streams


class FakeStreams:
    """Frames stamped by the runtime's FrameStamper at each stream's size; taking one costs a frame interval of fake
    time. ``silent`` streams give nothing."""

    def __init__(self, clock: FakeClock, *, silent=(), stale_after_s: float | None = None) -> None:
        self.clock, self.silent, self.stale_after_s = clock, set(silent), stale_after_s
        self.running: dict[str, int] = {}
        self.started: list[tuple[str, int]] = []
        self.stampers = {name: FrameStamper(f"cam-{name}", clock) for name in SIZES}
        self.frames = {name: 0 for name in SIZES}
        self.last: dict[str, CapturedFrame] = {}
        self.t0 = clock.monotonic_ns()

    def start(self, stream: str, threads: int) -> None:
        assert stream not in self.running
        self.running[stream] = threads
        self.started.append((stream, threads))
        self.stampers[stream].connect()

    def stop(self, stream: str) -> bool:
        del self.running[stream]
        self.stampers[stream].disconnect()
        return True

    def stop_all(self) -> bool:
        for stream in list(self.running):
            self.stop(stream)
        return True

    def take(self, stream: str, timeout_s: float):
        assert stream in self.running
        self.clock.advance(min(timeout_s, 0.02))
        if stream in self.silent:
            return None
        stale = self.stale_after_s is not None and (
            self.clock.monotonic_ns() - self.t0) / NS_PER_SECOND >= self.stale_after_s and stream in self.last
        if stale:
            return self.last[stream]  # as a stream that stopped delivering new frames would look, for the age check
        width, height = SIZES[stream]
        frame = self.stampers[stream].stamp(native_width=width, native_height=height, pixel_format=PixelFormat.BGR,
                                            source_pts=self.frames[stream] * 50_000)
        self.frames[stream] += 1
        self.last[stream] = CapturedFrame(frame, f"PIXELS-{stream}")
        return self.last[stream]

    def counts(self, stream: str) -> dict[str, Any]:
        return {"hand_off": {}, "state": "streaming", "frames": self.frames[stream], "connects": 1, "open_failures": 0,
                "stream_ends": 0, "problem": "no_frame" if stream in self.silent else None}

    def connections(self) -> int:
        return len(self.running)


class Faces:
    """A scripted backend: faces by the mark the fake clock is in, on the main stream only at the door when ``door_main``.
    Each call costs fake time, as inference does."""

    def __init__(self, clock: FakeClock, t0: dict, *, several_at: str | None = None, door_sub_under: bool = True) -> None:
        self.clock, self.t0, self.several_at, self.door_sub_under = clock, t0, several_at, door_sub_under
        self.images: list[str] = []
        self.calls: list[tuple[str, int]] = []

    def _mark(self) -> str | None:
        if "ns" not in self.t0:
            return None
        t = (self.clock.monotonic_ns() - self.t0["ns"]) / NS_PER_SECOND
        for name, start, end in (("empty", 35, 68), ("seat", 90, 133), ("mid", 150, 193), ("door", 210, 253)):
            if start <= t < end:
                return name
        return None

    def _run(self, frame, width, height, image, config):
        self.clock.advance(0.5)
        self.images.append(image)
        self.calls.append((config, width))
        mark = self._mark()
        size = {"seat": 60, "mid": 30, "door": 12}.get(mark)
        if mark is None or size is None:
            return to_face_run(frame, width, height, []), []
        if width > 640:
            size = round(size * 3.6)
        if mark == "door" and width == 640:
            under = [dr_row(size)] if self.door_sub_under and config == "native" else []
            return to_face_run(frame, width, height, []), under
        faces = [{"embedding": list(MATCHING if mark != "door" else NOT_MATCHING), "face_confidence": 0.95,
                  "facial_area": {"x": 100, "y": 90, "w": size, "h": size}}]
        if mark == self.several_at:
            faces.append({"embedding": list(MATCHING), "face_confidence": 0.92,
                          "facial_area": {"x": 300, "y": 90, "w": size, "h": size}})
        return to_face_run(frame, width, height, faces), []

    def production_faces(self, frame, width, height, image) -> FaceRun:
        return self._run(frame, width, height, image, "production")[0]

    def native_faces(self, frame, width, height, image):
        return self._run(frame, width, height, image, "native")


def dr_row(size):
    from sentinel.identity.legacy_deepface import FullScaleRow

    return FullScaleRow(100.0, 90.0, float(size), float(size), (0.0, 0.0), (1.0, 0.0), 0.6)


def session(dr, tmp_path, *, silent=(), stale_after_s=None, several_at=None, interrupt_at=None, environ=None):
    clock = FakeClock()
    streams = FakeStreams(clock, silent=silent, stale_after_s=stale_after_s)
    t0: dict = {}
    faces = Faces(clock, t0, several_at=several_at)
    lines: list[str] = []

    def emit(line: str) -> None:
        lines.append(line)
        if json.loads(line).get("run") == "starting":
            t0["ns"] = clock.monotonic_ns()

    def sleep(seconds: float) -> None:
        clock.advance(seconds)
        if interrupt_at is not None and "ns" in t0 and (clock.monotonic_ns() - t0["ns"]) / NS_PER_SECOND >= interrupt_at:
            raise KeyboardInterrupt

    args = argparse.Namespace(identity_dir=tmp_path / "identity", out=tmp_path / "door-range.json", seat_m=0.8, mid_m=2.5,
                              door_m=4.6)
    result = dr.run(args, environ=URLS if environ is None else environ, clock=clock, streams=streams,
                    recognition=lambda: (faces, SimpleNamespace(enrollment=lambda: ENROLLMENT)), emit=emit, sleep=sleep,
                    fov=lambda sub, main: {"status": "fitted", "matches": 40, "inliers": 30, **dr.affine_facts(
                        [[3.6, 0.0, 0.0], [0.0, 2.7, 0.0]], (640, 480), (2304, 1296))})
    return result, lines, streams, faces, args.out


def test_cues_fit_the_guards_limits_and_leave_time_before_each_window(dr) -> None:
    assert all(len(text) <= 160 and all(32 <= ord(c) < 127 for c in text) for _, _, text in dr.CUES)
    at = {name: when for when, name, _ in dr.CUES}
    assert at["t0"] == 0.0
    windows = {w.name: w for w in dr.WINDOWS}
    assert windows["empty"].start_s - at["t0"] >= 30 and at["seat"] - windows["empty"].end_s >= 5  # gone and not back
    for mark in dr.MARKS:
        assert windows[mark].start_s - at[mark] >= 15  # walking time, then the window
        assert windows[mark].end_s - windows[mark].start_s >= 30
    assert windows["door"].end_s == at["stop"] == dr.MARKS_END_S
    assert dr.TIMELINE_S <= 480  # the block's --duration-s, counted from the same readiness line


def test_the_guard_accepts_the_cues_it_is_given(dr) -> None:
    import step5_guard

    for when, name, text in dr.CUES:
        assert step5_guard._cue_at(f"{when:g}:{name}:{text}") == (when, name, text)


def test_a_whole_session_samples_only_inside_the_windows(dr, tmp_path) -> None:
    result, lines, streams, faces, out = session(dr, tmp_path)
    assert result["verdict"] == "complete" and result["status"] == "finished" and result["problems"] == []
    assert json.loads(lines[0]) == {"run": "starting", "startup": {"tool": "door_range_check"}}
    assert sum(1 for line in lines if '"starting"' in line) == 1
    windows = {w.name: w for w in dr.WINDOWS}
    assert {s["window"] for s in result["samples"]} == set(windows)
    for sample in result["samples"]:
        window = windows[sample["window"]]
        assert window.start_s <= sample["t_s"] < window.end_s
        assert [(r["stream"], r["config"]) for r in sample["results"]] == [
            ("sub", "production"), ("sub", "native"), ("main", "production"), ("main", "native")]
        assert sample["frames"]["sub"]["width"] == 640 and sample["frames"]["main"]["width"] == 2304
        assert ("fov" in sample) == (sample["window"] == "empty")
    assert result["streams"]["main"]["height"] == 1296 and result["connections_at_ready"] == 2
    assert result["marks_m"] == {"seat": 0.8, "mid": 2.5, "door": 4.6}
    assert result["thresholds"] == {"yunet": 0.9, "recording_floor": 0.3, "match": 0.70, "margin": 0.05, "min_quality": 0.6}
    assert result["workers_stopped"] is True and streams.running == {}
    on_disk = json.loads(out.read_text())
    assert on_disk["verdict"] == "complete" and len(on_disk["samples"]) == len(result["samples"])
    assert out.stat().st_mode & 0o777 == 0o600


def test_the_decode_phases_have_fresh_connections_and_no_inference(dr, tmp_path) -> None:
    result, _, streams, faces, _ = session(dr, tmp_path)
    assert streams.started == [("sub", 1), ("main", 2), ("sub", 1), ("main", 2), ("main", 1), ("main", 2), ("sub", 1)]
    names = [phase["name"] for phase in result["decode"]]
    assert names == ["both", "main_threads_1", "main_threads_2", "sub_only"]
    assert {phase["name"]: phase["connections"] for phase in result["decode"]} == {
        "both": 2, "main_threads_1": 1, "main_threads_2": 1, "sub_only": 1}
    calls_after_marks = len(faces.calls)
    assert calls_after_marks == 4 * len(result["samples"])  # every inference belongs to a sample
    both = result["decode"][0]["streams"]
    assert both["main"]["seen"] > 0 and both["main"]["pts_span_s"] is not None and both["sub"]["fps_seen"] > 0


def test_no_frame_or_embedding_or_identity_or_url_is_kept(dr, tmp_path) -> None:
    result, lines, _, faces, out = session(dr, tmp_path)
    assert "PIXELS-main" in faces.images  # the frames did reach the backend
    text = out.read_text() + "".join(lines)
    for private in ("PIXELS", IDENTITY, "embedding", "hunter2", "192.0.2.10", "operator:", str(MATCHING[0])):
        assert private not in text


def test_source_saves_no_image_or_video(dr) -> None:
    source = Path(dr.__file__).read_text()
    for call in ("imwrite", "VideoWriter", "imencode", '"wb"', "'wb'", "tofile", "numpy.save"):
        assert call not in source


def test_readiness_needs_a_frame_from_both_streams(dr, tmp_path) -> None:
    result, lines, streams, _, out = session(dr, tmp_path, silent=("main",))
    assert result["status"] == "refused:stream_not_ready:main:no_frame" and result["verdict"] == "failed"
    assert not any('"starting"' in line for line in lines)  # so the guard's cues never start
    assert streams.running == {} and json.loads(out.read_text())["samples"] == []


@pytest.mark.parametrize("environ, label", [
    ({"SENTINEL_RTSP_URL": URLS["SENTINEL_RTSP_URL"]}, "url_missing:main"),
    ({**URLS, "SENTINEL_RTSP_MAIN_URL": "http://nope"}, "url_invalid:main"),
    ({**URLS, "SENTINEL_RTSP_MAIN_URL": URLS["SENTINEL_RTSP_URL"]}, "urls_identical"),
])
def test_bad_urls_are_refused_by_label_only(dr, tmp_path, environ, label) -> None:
    result, lines, streams, _, out = session(dr, tmp_path, environ=environ)
    assert result["status"] == f"refused:{label}" and streams.started == []
    assert "hunter2" not in out.read_text() + "".join(lines)


def test_a_stale_pair_is_not_analysed(dr, tmp_path) -> None:
    result, _, _, faces, _ = session(dr, tmp_path, stale_after_s=100.0)  # frames stop being new during the seat window
    assert result["counters"]["stale_pair"] > 0
    assert all(s["t_s"] < 103 for s in result["samples"])
    assert len(faces.calls) == 4 * len(result["samples"])
    state, problems = dr.verdict(result)
    assert state == "incomplete" and any(p.startswith("window_mid_samples:0") for p in problems)


def test_the_guards_stop_keeps_what_was_measured(dr, tmp_path) -> None:
    result, _, streams, _, out = session(dr, tmp_path, interrupt_at=140.0)  # the wait before the middle window
    assert result["status"] == "stopped" and streams.running == {} and result["workers_stopped"] is True
    on_disk = json.loads(out.read_text())
    assert {s["window"] for s in on_disk["samples"]} == {"empty", "seat"} and on_disk["verdict"] == "incomplete"


def test_several_faces_stay_unresolved(dr, tmp_path) -> None:
    result, _, _, _, _ = session(dr, tmp_path, several_at="mid")
    inside = [s for s in result["samples"] if s["window"] == "mid"]
    facts = dr.cell(inside, "sub", "native")
    assert facts["several_faces"] == facts["n"] > 0
    assert facts["one_face"] == 0 and facts["w_px"] is None and facts["votes"] == {}


def test_a_candidate_under_the_threshold_is_never_a_face(dr, tmp_path) -> None:
    result, _, _, _, _ = session(dr, tmp_path)
    inside = [s for s in result["samples"] if s["window"] == "door"]
    facts = dr.cell(inside, "sub", "native")
    assert facts["one_face"] == 0 and facts["no_face"] == facts["n"] > 0 and facts["votes"] == {}
    assert facts["no_face_with_candidate_under_threshold"] == facts["n"] and facts["under_w_px_p50"] == 12.0
    main = dr.cell(inside, "main", "native")
    assert main["one_face"] == main["n"] and main["w_px"]["p50"] == 43.0 and main["votes"] == {"unknown": main["n"]}
    assert all("similarity" not in c for s in inside for r in s["results"] for c in r.get("under", []))


# ---------------------------------------------------------------- the capture wiring


class Source:
    """A VideoSource giving a fixed-size frame until it is closed."""

    def __init__(self, size) -> None:
        self.size, self.opened, self.closed = size, 0, 0
        self.pulse = threading.Event()

    def open(self) -> None:
        self.opened += 1

    def read(self):
        self.pulse.wait(0.005)
        return DecodedFrame("pixels", self.size[0], self.size[1], PixelFormat.BGR, None)

    def close(self) -> None:
        self.closed += 1


def test_live_streams_use_the_runtimes_worker_and_one_frame_hand_off(dr, tmp_path) -> None:
    from sentinel.media.clock import SystemClock

    made: list[tuple[str, CaptureConfig, Source]] = []

    def source(url: str, config: CaptureConfig) -> Source:
        made.append((url, config, Source(SIZES["main" if url.endswith("main") else "sub"])))
        return made[-1][2]

    net = tmp_path / "net"
    net.mkdir()
    # 192.0.2.10:554, established, in /proc/net/tcp's little-endian words
    address = (192 | 0 << 8 | 2 << 16 | 10 << 24)
    (net / "tcp").write_text(f"header\n 0: 00000000:1234 {address:08X}:022A 01 x\n")
    (net / "tcp6").write_text("header\n")
    live = dr.LiveStreams({"sub": URLS["SENTINEL_RTSP_URL"], "main": URLS["SENTINEL_RTSP_MAIN_URL"]}, SystemClock(),
                          source=source, proc_net=net)
    assert repr(live).count("hunter2") == 0
    live.start("sub", 1)
    live.start("main", 2)
    assert [(config.decode_threads) for _, config, _ in made] == [1, 2]
    with pytest.raises(RuntimeError):
        live.start("main", 2)
    first = live.take("main", 2.0)
    assert first.frame.native_width == 2304 and first.frame.stream_epoch == 1
    assert live.take("sub", 2.0).frame.native_width == 640
    counts = live.counts("main")
    assert counts["connects"] == 1 and counts["hand_off"]["pending"] <= 1 and counts["state"] == "streaming"
    assert "hunter2" not in json.dumps(counts)
    assert live.connections() == 1
    assert live.stop("main") is True
    live.start("main", 1)
    assert live.take("main", 2.0).frame.stream_epoch == 2  # the same ingest run's next epoch
    assert live.stop_all() is True
    assert all(s.closed == 1 for _, _, s in made)


# ---------------------------------------------------------------- recognition's loading


def test_recognition_is_refused_when_the_gallery_is_not_usable(dr, tmp_path, monkeypatch) -> None:
    import sentinel.demo_runtime as demo

    monkeypatch.setattr(demo, "open_gallery", lambda store, secret: (None, "no_identities_enrolled"))
    with pytest.raises(dr.Refused, match="gallery:no_identities_enrolled"):
        dr.load_recognition(tmp_path, "secret", backend_factory=lambda: pytest.fail("no model for an unusable gallery"))


def test_recognition_is_refused_when_the_model_differs_from_the_gallerys(dr, tmp_path, monkeypatch) -> None:
    import sentinel.demo_runtime as demo

    document = SimpleNamespace(compatibility="gallery's")

    class Backend:
        selected_before_load = False
        selected = False

        def select_full_scale_detection(self):
            self.selected = True

        def load(self):
            self.selected_before_load = self.selected

        def compatibility(self):
            return SimpleNamespace(first_difference=lambda other: "model_sha256")

    monkeypatch.setattr(demo, "open_gallery", lambda store, secret: (document, None))
    backend = Backend()
    with pytest.raises(dr.Refused, match="gallery_incompatible:model_sha256"):
        dr.load_recognition(tmp_path, "secret", backend_factory=lambda: backend)
    assert backend.selected_before_load


# ---------------------------------------------------------------- the reading


def folder_with(dr, tmp_path, account: dict[str, str] | None = None, *, guard: dict | None = None):
    session(dr, tmp_path)
    (tmp_path / "result.json").write_text(json.dumps(guard if guard is not None else {
        "status": "child_exited", "child": {"returncode": 0},
        "guard": {"trigger": None, "peak_pressure_bytes": 3_000_000_000, "min_mem_free_bytes": 2_000_000_000}}))
    if account is not None:
        (tmp_path / "account.txt").write_text("".join(f"{k}={v}\n" for k, v in account.items()))
    return tmp_path


PLANNED = {"left_view_for_empty": "yes", "at_marks_in_windows": "yes", "faced_camera": "yes", "others_in_view": "no"}


def test_the_reading_shows_each_mark_stream_and_configuration(dr, tmp_path) -> None:
    lines, code = dr.reading(folder_with(dr, tmp_path, PLANNED))
    text = "\n".join(lines)
    assert code == 0 and "PASS the guard" in text and "execution: complete" in text
    assert "--- door (4.6 m)" in text and "--- seat (0.8 m)" in text and "--- empty (no person expected)" in text
    assert "main native" in text and "sub  production" in text
    assert "USER-REPORTED account" in text and "CONDITION not as planned" not in text
    assert "frame sizes read from the streams: sub 640x480, main 2304x1296" in text
    assert "scale_x" in text and "decode main_threads_1" in text and "NVDEC is not measured" in text
    assert "validates nothing" in text
    for claim in ("recognition passed", "requirement met", "accuracy:"):
        assert claim not in text


def test_the_reading_keeps_similarity_private_unless_asked(dr, tmp_path) -> None:
    folder = folder_with(dr, tmp_path, PLANNED)
    plain = "\n".join(dr.reading(folder)[0])
    assert "similarity" not in plain.replace("door_range", "")
    assert "PRIVATE similarity" in "\n".join(dr.reading(folder, with_similarity=True)[0])


def test_the_reading_says_when_the_account_differs_or_is_missing(dr, tmp_path) -> None:
    text = "\n".join(dr.reading(folder_with(dr, tmp_path, {**PLANNED, "others_in_view": "yes"}))[0])
    assert "CONDITION not as planned (others_in_view)" in text
    (tmp_path / "account.txt").unlink()
    assert "MISSING the operator's account" in "\n".join(dr.reading(tmp_path)[0])


def test_the_reading_of_a_refused_launch_judges_nothing(dr, tmp_path) -> None:
    lines, code = dr.reading(folder_with(dr, tmp_path, PLANNED, guard={"status": "refused:baseline"}))
    assert code == 1 and lines == ["REFUSED the guard: refused:baseline, before launch: nothing was started and nothing is judged"]


def test_the_reading_of_a_missing_result_judges_nothing(dr, tmp_path) -> None:
    (tmp_path / "result.json").write_text(json.dumps({"status": "guard_stop", "guard": {"trigger": "mem_free"}, "child": {}}))
    lines, code = dr.reading(tmp_path)
    assert code == 1 and lines[0].startswith("FAIL the guard: status guard_stop") and "MISSING the check's result" in lines[1]


def test_an_incomplete_session_reads_as_incomplete_never_complete(dr, tmp_path) -> None:
    session(dr, tmp_path, interrupt_at=140.0)
    (tmp_path / "result.json").write_text(json.dumps({"status": "guard_stop", "child": {}, "guard": {"trigger": "x"}}))
    lines, code = dr.reading(tmp_path)
    text = "\n".join(lines)
    assert code == 3 and "execution: incomplete" in text and "FAIL the guard" in text and "status:stopped" in text


def test_the_command_line_prints_the_guards_cue_values(dr, capsys) -> None:
    assert dr.main(["cues"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == len(dr.CUES) and printed[0].startswith("0:t0:T0 - ")


@pytest.mark.parametrize("bad", ["0.1", "16", "nan", "inf"])
def test_a_typed_distance_must_be_plausible(dr, bad) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        dr.distance(bad)


def test_module_names_no_model_it_does_not_load(dr, tmp_path) -> None:
    assert MODEL_NAME == "Facenet512" and dr.MAX_SAMPLES_PER_WINDOW >= 40
    assert dr.READY_TIMEOUT_S <= 300  # within the guard's readiness wait
