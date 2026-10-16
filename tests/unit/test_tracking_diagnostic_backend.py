"""The tracking diagnostic on the legacy Ultralytics backend (session 79; the callback check corrected in session 80),
with fakes that keep the installed 8.4.25 shape: the model's callback dict is seeded with Ultralytics' own no-op for
each event (``get_default_callbacks``), the first ``track()`` appends a ``functools.partial`` of the tracker's
callback (``register_tracker``) and makes the predictor, which shares the dict; the predictor runs inference,
post-processing, then the event's callbacks in order (the no-op, ours, the tracker's, which calls the tracker's update
and replaces the results with the published rows); the tracker calls its own get_dists before assigning and keeps
tracked, lost and removed lists. Session 81: inference, post-processing and the callbacks run in torch's inference mode
(``smart_inference_mode`` around ``stream_inference``), so a clone made there is an inference tensor, and an in-place
write to one outside the mode raises, as torch 2.9.1 does (the device run trk-20261016T100149Z's shadow pass failed on
every frame for that). No GPU, engine or Ultralytics import here; test_tracking_diagnostic_installed.py runs the same
path on the installed torch and Ultralytics, CPU only."""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sentinel.inference.legacy_ultralytics import CALLBACK, TRACK_ARGS, LegacyUltralyticsTracker, callback_problem
from sentinel.tracking.diagnostic import SETTING_KEYS, TrackerDiagnostics, settings_problem
from sentinel.tracking.tracker import PersonTracker, TrackerError

LOST, TRACKED = 2, 1


class Mode:
    """torch's inference mode as the shadow pass meets it: what is made while it is on is an inference tensor, and an
    in-place write to one while it is off raises (torch: "Inplace update to inference tensor outside InferenceMode is
    not allowed"). ``inference_mode`` is what the tools hand the adapter (torch.inference_mode on the device)."""

    enabled = False
    entered = 0

    @classmethod
    @contextlib.contextmanager
    def inference_mode(cls):
        previous, cls.enabled = cls.enabled, True
        cls.entered += 1
        try:
            yield
        finally:
            cls.enabled = previous

    @classmethod
    def write(cls, tensor: Any) -> None:
        if getattr(tensor, "inference", False) and not cls.enabled:
            raise RuntimeError("Inplace update to inference tensor outside InferenceMode is not allowed.")


class Tensor:
    def __init__(self, values: list[Any], inference: bool = False) -> None:
        self.values = values
        self.inference = inference

    def cpu(self) -> Tensor:
        return self

    def tolist(self) -> list[Any]:
        return copy.deepcopy(self.values)


class Boxes:
    """Ultralytics' Boxes as the adapter reads it (xyxy, conf, cls, id tensors)."""

    def __init__(self, rows: list[tuple[float, float, float, float, float]], ids: list[int] | None = None) -> None:
        self.rows = rows
        self.xyxy = Tensor([list(r[:4]) for r in rows])
        self.conf = Tensor([r[4] for r in rows])
        self.cls = Tensor([0.0 for _ in rows])
        self.id = None if ids is None else Tensor([float(i) for i in ids])


class Row:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __gt__(self, threshold: float) -> Row:
        return Row([1.0 if v > threshold else 0.0 for v in self.values])

    def sum(self) -> Row:
        return Row([sum(self.values)])

    def item(self) -> float:
        return self.values[0]


class Raw:
    """The head's raw output (1, 84, anchors): anchors as (x1, y1, x2, y2, person score)."""

    def __init__(self, anchors: list[tuple[float, float, float, float, float]], inference: bool = False) -> None:
        self.anchors = anchors
        self.ndim, self.shape = 3, (1, 84, len(anchors))
        self.clones = 0
        self.inference = inference

    def detach(self) -> Raw:
        return self

    def clone(self) -> Raw:
        self.clones += 1
        return Raw(list(self.anchors), inference=Mode.enabled)

    def __getitem__(self, key: Any) -> Row:
        assert key[0] == 0 and key[1] == 4
        return Row([a[4] for a in self.anchors])


class Det:
    """What non_max_suppression returns for one image: det[:, :4] boxes, det[:, 4] scores."""

    def __init__(self, rows: list[tuple[float, float, float, float, float]], inference: bool = False) -> None:
        self.rows = rows
        self.inference = inference

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: Any) -> Tensor:
        if key[1] == 4:
            return Tensor([r[4] for r in self.rows], self.inference)
        return Tensor([list(r[:4]) for r in self.rows], self.inference)


class Tools:
    """(non_max_suppression, scale_boxes, TrackState, inference_mode) stand-ins; the NMS filters the raw anchors by
    score and mutates what it is given (so a clone must have been handed to it), and scale_boxes writes into the boxes
    it is given, as 8.4.25's do: each write raises on an inference tensor outside inference mode (Mode.write).
    ``inference_mode`` replaces what is handed to the adapter (``contextlib.nullcontext`` = the code before session 81,
    which ran both outside the mode)."""

    def __init__(self, *, fail: bool = False, inference_mode: Any = None) -> None:
        self.calls: list[dict] = []
        self.fail = fail
        self.inference_mode = inference_mode or Mode.inference_mode
        self.modes: list[bool] = []  # whether inference mode was on at each NMS call

    def nms(self, raw: Raw, **kwargs: Any) -> list[Det]:
        self.calls.append(kwargs)
        self.modes.append(Mode.enabled)
        if self.fail:
            raise RuntimeError("nms broke")
        Mode.write(raw)  # 8.4.25: prediction[..., :4] = xywh2xyxy(...) on the transposed view of its input
        kept = [a for a in raw.anchors if a[4] >= kwargs["conf_thres"]][:kwargs["max_det"]]
        raw.anchors.clear()  # the mutation the adapter must never let reach production
        return [Det(kept, inference=Mode.enabled)]

    @staticmethod
    def scale_boxes(im_shape: tuple[int, int], boxes: Tensor, img0_shape: tuple[int, ...]) -> Tensor:
        Mode.write(boxes)  # 8.4.25: boxes[..., 0] -= pad_x ... and clip_boxes' clamp_, in place
        return boxes

    def __call__(self) -> tuple[Any, Any, Any, Any]:
        return self.nms, self.scale_boxes, SimpleNamespace(Lost=LOST, Tracked=TRACKED), self.inference_mode


class STrack:
    def __init__(self, track_id: int, idx: int, box: tuple[float, ...], state: int, activated: bool,
                 end_frame: int) -> None:
        self.track_id, self.idx, self.box, self.state, self.is_activated = track_id, idx, box, state, activated
        self.end_frame = end_frame

    @property
    def xyxy(self) -> Tensor:
        return Tensor(list(self.box))


class Matrix:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def tolist(self) -> list[list[float]]:
        return [list(r) for r in self.rows]


class ByteTrack:
    """A scripted BYTETracker: each update() consumes one step (the pool its stage-1 get_dists sees, the matrix,
    and the lists after the update) and returns the published rows."""

    def __init__(self) -> None:
        self.resets = 0
        self.frame_id = 0
        self.max_time_lost = 30
        self.args = SimpleNamespace(match_thresh=0.8, track_high_thresh=0.25, track_low_thresh=0.1,
                                    new_track_thresh=0.25, track_buffer=30, fuse_score=True)
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.steps: list[dict] = []
        self.inputs: list[Boxes] = []
        self.matrices: list[Matrix] = []
        self.dists_seen: list[Any] = []

    def get_dists(self, tracks: list[STrack], detections: list[Any]) -> Matrix:
        matrix = Matrix([[0.5] * len(detections) for _ in tracks])
        self.matrices.append(matrix)
        return matrix

    def update(self, det: Boxes, img: Any = None, feats: Any = None) -> list[list[float]]:
        self.frame_id += 1
        self.inputs.append(det)
        step = self.steps.pop(0) if self.steps else {"tracked": [], "lost": []}
        detections = [SimpleNamespace(idx=i, score=r[4]) for i, r in enumerate(det.rows)]
        pool = step.get("pool", self.tracked_stracks + self.lost_stracks)
        if pool or detections:
            self.dists_seen.append(self.get_dists(pool, detections))  # through the instance: the wrapper sees it
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        if unconfirmed:
            self.dists_seen.append(self.get_dists(unconfirmed, detections))
        self.tracked_stracks = list(step["tracked"])
        self.lost_stracks = list(step["lost"])
        return [[*t.box, t.track_id, 0.9, 0.0, t.idx] for t in self.tracked_stracks if t.is_activated]

    def reset(self) -> None:
        self.resets += 1
        self.tracked_stracks, self.lost_stracks, self.removed_stracks, self.frame_id = [], [], [], 0


class Predictor:
    def __init__(self, callbacks: dict[str, list], tracker: ByteTrack) -> None:
        self.callbacks = callbacks
        self.trackers = [tracker]
        self.args = SimpleNamespace(iou=0.7, max_det=300, conf=0.4)
        self.model = SimpleNamespace(end2end=False)
        self.results: list[Any] = []
        self.raw: list[Raw] = []
        self.inference_calls = 0

    def inference(self, im: Any, *args: Any, **kwargs: Any) -> Raw:
        self.inference_calls += 1
        return self.raw.pop(0) if self.raw else Raw([])


def default_postprocess_end(predictor: Any) -> None:
    """Ultralytics' own no-op for the event (utils/callbacks/base.py), with which it seeds every model's list."""


def tracker_postprocess_end(predictor: Any, persist: bool = False) -> None:
    """Ultralytics' tracker callback (trackers/track.py): the tracker's update, then the results replaced."""
    det = predictor.results[0].boxes
    if det is None:  # already replaced by an earlier tracker callback on this frame (a doubled registration)
        det = Boxes([])
    rows = predictor.trackers[0].update(det, None, None)
    predictor.results[0] = SimpleNamespace(  # Ultralytics' rows: x1, y1, x2, y2, id, conf, cls, idx
        boxes=Boxes([(*r[:4], r[5]) for r in rows], [int(r[4]) for r in rows]) if rows else None)


def other_callback(predictor: Any) -> None:
    """A callback that is neither ours, Ultralytics' default nor its tracker's."""


def references() -> tuple[tuple[Any, ...], Any]:
    """What _import_callback_references reads from the installed package: (the defaults, the tracker's function)."""
    return (default_postprocess_end,), tracker_postprocess_end


class Yolo:
    """YOLO as the diagnostic relies on it (8.4.25): the callback dict seeded with the default no-op per event
    (``get_default_callbacks``); the first ``track()`` registers a ``partial`` of the tracker's callback after any
    callback added before (``register_tracker``) and makes the predictor, which shares the dict and the trackers; then
    inference, post-processing and the event's callbacks in order, all in inference mode (``stream_inference`` under
    ``smart_inference_mode``)."""

    instances: list[Yolo] = []
    register: Any = None  # a test's own registration in place of register_tracker's (None: as installed)

    def __init__(self, path: str, task: str) -> None:
        self.path, self.task = path, task
        self.callbacks: dict[str, list] = {CALLBACK: [default_postprocess_end]}
        self.predictor: Predictor | None = None
        self.detections: list[list[tuple[float, float, float, float, float]]] = []
        self.calls = 0
        Yolo.instances.append(self)

    def add_callback(self, event: str, func: Any) -> None:
        self.callbacks.setdefault(event, []).append(func)

    def track(self, image: Any, **kwargs: Any) -> list[Any]:
        self.calls += 1
        if self.predictor is None:
            if Yolo.register is None:
                self.add_callback(CALLBACK, functools.partial(tracker_postprocess_end, persist=kwargs.get("persist")))
            else:
                Yolo.register(self)
            self.predictor = Predictor(self.callbacks, ByteTrack())
        with Mode.inference_mode():
            self.predictor.inference(SimpleNamespace(shape=(1, 3, 640, 640)))
            self.predictor.results = [SimpleNamespace(boxes=Boxes(self.detections.pop(0) if self.detections else []))]
            for callback in list(self.callbacks.get(CALLBACK, [])):
                callback(self.predictor)
        return self.predictor.results


class Runtime:
    def __init__(self) -> None:
        self.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        self.numpy = SimpleNamespace(uint8="uint8", zeros=lambda shape, dtype: ("blank", shape, dtype))

    def __call__(self) -> tuple[Any, Any, Any]:
        return self.torch, Yolo, self.numpy


@pytest.fixture(autouse=True)
def clean(monkeypatch: pytest.MonkeyPatch) -> None:
    Yolo.instances = []
    Yolo.register = None
    Mode.enabled, Mode.entered = False, 0
    monkeypatch.setenv("YOLO_OFFLINE", "operator-value")


@pytest.fixture
def engine(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "yolov8n.engine"
    path.write_bytes(b"not a real engine")
    return path, hashlib.sha256(b"not a real engine").hexdigest()


def backend(engine: tuple[Path, str], tools: Tools | None = None, *, diagnostic: bool = True,
            refs: Any = references) -> LegacyUltralyticsTracker:
    path, digest = engine
    tracker = LegacyUltralyticsTracker(path, expected_sha256=digest, driver_check=lambda: None, importer=Runtime(),
                                       diagnostic_tools=tools or Tools(), callback_references=refs)
    if diagnostic:
        tracker.enable_diagnostic()
    return tracker


PERSON = (100.0, 80.0, 220.0, 470.0, 0.9)
TORSO = (110.0, 250.0, 215.0, 470.0, 0.45)
IMAGE = SimpleNamespace(shape=(480, 640, 3))


def loaded(engine: tuple[Path, str], tools: Tools | None = None, *, diagnostic: bool = True) -> tuple[LegacyUltralyticsTracker, Yolo, ByteTrack]:
    adapter = backend(engine, tools, diagnostic=diagnostic)
    adapter.load()
    (model,) = Yolo.instances
    return adapter, model, model.predictor.trackers[0]


# ---------------------------------------------------------------- the disabled path installs nothing


def test_without_the_diagnostic_no_callback_wrapper_or_reading_exists(engine) -> None:
    adapter, model, tracker = loaded(engine, diagnostic=False)
    default, trackers = model.callbacks[CALLBACK]  # Ultralytics' own no-op and its tracker's partial, nothing else
    assert default is default_postprocess_end and trackers.func is tracker_postprocess_end
    assert trackers.keywords == {"persist": True}
    assert "inference" not in vars(model.predictor) and "get_dists" not in vars(tracker)
    model.detections.append([PERSON])
    tracker.steps.append({"tracked": [STrack(1, 0, PERSON[:4], TRACKED, True, 1)], "lost": []})
    assert [t.track_id for t in adapter.track(IMAGE)] == [1]
    assert adapter.last_diagnostics() is None and adapter.diagnostic_settings() is None


def test_the_diagnostic_is_enabled_before_load_only_and_below_the_production_confidence(engine) -> None:
    adapter = backend(engine, diagnostic=False)
    with pytest.raises(TrackerError, match="diagnostic_shadow_conf"):
        adapter.enable_diagnostic(0.4)
    adapter.load()
    with pytest.raises(TrackerError, match="diagnostic_after_load"):
        adapter.enable_diagnostic()


# ---------------------------------------------------------------- load: verification, hooks and settings


def test_load_registers_the_capture_before_the_trackers_callback_and_wraps_the_instances(engine) -> None:
    adapter, model, tracker = loaded(engine)
    order = model.callbacks[CALLBACK]
    # 8.4.25's list: Ultralytics' default no-op first, then the capture, then the tracker's partial. Session 80: the
    # device run trk-20261016T032337Z was refused "callback_order" by a check that required the capture at index 0.
    assert order[0] is default_postprocess_end and order[1] is adapter._capture_callback and len(order) == 3
    assert isinstance(order[2], functools.partial) and order[2].func is tracker_postprocess_end
    assert "inference" in vars(model.predictor) and "get_dists" in vars(tracker)
    assert adapter.diagnostic_settings() == {
        "conf": 0.4, "match_thresh": 0.8, "track_high_thresh": 0.25, "track_low_thresh": 0.1,
        "new_track_thresh": 0.25, "track_buffer": 30, "max_time_lost": 30, "fuse_score": True,
        "unconfirmed_thresh": 0.7, "shadow_conf": 0.1, "shadow_max_nms": 30000, "max_det": 300, "iou": 0.7}
    assert tracker.resets == 1 and model.calls == 3  # the warm-up and the reset as before


@pytest.mark.parametrize("break_what, label", [
    ("tracker_first", "callback_order"), ("tracker_missing", "callback_order"), ("tracker_twice", "callback_order"),
    ("tracker_bare", "callback_order"), ("capture_twice", "callback_order"), ("unknown_before", "callback_unknown"),
    ("unknown_between", "callback_unknown"), ("tracker", "tracker_state"), ("predictor", "predictor"),
    ("tools", "RuntimeError"), ("tools_shape", "tools"), ("references", "ImportError")])
def test_load_refuses_when_the_installed_objects_are_not_what_the_diagnostic_reads(engine, monkeypatch, break_what,
                                                                                   label) -> None:
    partial = functools.partial(tracker_postprocess_end, persist=True)
    registrations = {  # in place of register_tracker's, on the first track(); the list holds the default, then ours
        "tracker_first": lambda m: m.callbacks[CALLBACK].insert(0, partial),
        "tracker_missing": lambda m: None,
        "tracker_twice": lambda m: (m.add_callback(CALLBACK, partial),
                                    m.add_callback(CALLBACK, functools.partial(tracker_postprocess_end, persist=True))),
        "tracker_bare": lambda m: m.add_callback(CALLBACK, tracker_postprocess_end),
        "capture_twice": lambda m: (m.add_callback(CALLBACK, m.callbacks[CALLBACK][1]),
                                    m.add_callback(CALLBACK, partial)),
        "unknown_before": lambda m: (m.callbacks[CALLBACK].insert(0, other_callback),
                                     m.add_callback(CALLBACK, partial)),
        "unknown_between": lambda m: (m.add_callback(CALLBACK, other_callback), m.add_callback(CALLBACK, partial)),
    }
    refs: Any = references
    if break_what in registrations:
        monkeypatch.setattr(Yolo, "register", registrations[break_what])
    elif break_what == "references":
        def missing():
            raise ImportError("no ultralytics.trackers.track")
        refs = missing
    elif break_what == "tracker":
        monkeypatch.delattr(ByteTrack, "get_dists")
    elif break_what == "predictor":
        original_init = Predictor.__init__

        def init(self, callbacks, tracker):  # the predictor's arguments lack what the shadow pass needs
            original_init(self, callbacks, tracker)
            self.args = SimpleNamespace(iou=0.7)
        monkeypatch.setattr(Predictor, "__init__", init)
    tools = Tools()
    if break_what == "tools":
        def failing():
            raise RuntimeError("no ultralytics")
        tools = failing  # type: ignore[assignment]
    elif break_what == "tools_shape":  # the tools as handed out before session 81: no inference mode
        three = Tools()
        tools = lambda: three()[:3]  # type: ignore[assignment]  # noqa: E731
    adapter = backend(engine, tools, refs=refs)
    with pytest.raises(TrackerError) as caught:
        adapter.load()
    assert caught.value.label == "diagnostic_unsupported" and caught.value.error_type == label
    assert repr(adapter) == "LegacyUltralyticsTracker(loaded=False)"


def test_the_callback_check_accepts_the_installed_lifecycle_and_refuses_every_other_shape() -> None:
    """Session 80: callback_problem on the lists themselves. The installed 8.4.25 shape is [default no-op, capture,
    partial(tracker)]; the session-79 check required the capture at index 0 and refused the device run."""
    def capture(predictor):
        pass

    partial = functools.partial(tracker_postprocess_end, persist=True)

    def check(order):
        return callback_problem(order, capture, (default_postprocess_end,), tracker_postprocess_end)

    installed = [default_postprocess_end, capture, partial]
    assert installed[0] is not capture  # what the device run's list looked like to the session-79 check: refused
    assert check(installed) is None
    assert check([capture, partial]) is None  # an Ultralytics that seeded no default
    assert check([default_postprocess_end, partial, capture]) == "callback_order"  # ours after the tracker's: too late
    assert check([default_postprocess_end, capture]) == "callback_order"  # no tracker callback
    assert check([default_postprocess_end, partial]) == "callback_order"  # no capture
    assert check([*installed, functools.partial(tracker_postprocess_end)]) == "callback_order"  # the tracker's twice
    assert check([default_postprocess_end, capture, capture, partial]) == "callback_order"  # ours twice
    assert check([default_postprocess_end, capture, tracker_postprocess_end]) == "callback_order"  # bare, no partial
    assert check([default_postprocess_end, capture, functools.partial(other_callback)]) == "callback_order"
    assert check([other_callback, capture, partial]) == "callback_unknown"  # could change the results before ours
    assert check([default_postprocess_end, capture, other_callback, partial]) == "callback_unknown"
    assert check([*installed, other_callback]) == "callback_unknown"
    assert check(None) == "callback_order" and check({}) == "callback_order" and check([]) == "callback_order"


def test_the_settings_carry_every_key_the_runtime_and_the_guard_require_before_t0(engine) -> None:
    adapter, _, _ = loaded(engine)
    settings = adapter.diagnostic_settings()
    assert set(settings) == set(SETTING_KEYS) and settings_problem(settings) is None
    assert settings_problem(None) == "tracker_settings_missing"
    assert settings_problem({k: v for k, v in settings.items() if k != "iou"}) == "tracker_settings_incomplete:iou"
    assert settings_problem({**settings, "fuse_score": 1}) == "tracker_settings_incomplete:fuse_score"
    assert settings_problem({**settings, "conf": True}) == "tracker_settings_incomplete:conf"
    assert settings_problem({**settings, "max_det": None}) == "tracker_settings_incomplete:max_det"


# ---------------------------------------------------------------- a frame: inputs, outcomes, costs, shadow


def test_a_frames_reading_keeps_the_inputs_and_outcomes_apart_and_changes_nothing(engine) -> None:
    tools = Tools()
    adapter, model, tracker = loaded(engine, tools)
    model.detections.append([PERSON, TORSO])
    model.predictor.raw.append(Raw([(*PERSON[:4], 0.9), (*TORSO[:4], 0.45), (300.0, 300.0, 340.0, 360.0, 0.3),
                                    (10.0, 10.0, 20.0, 20.0, 0.05)]))
    seat = STrack(3, 0, PERSON[:4], TRACKED, True, 1)
    new = STrack(4, 1, TORSO[:4], TRACKED, False, 1)
    tracker.steps.append({"tracked": [seat, new], "lost": []})
    published = adapter.track(IMAGE)
    assert [(t.track_id, t.confidence) for t in published] == [(3, 0.9)]  # the production output, as before
    d = adapter.last_diagnostics()
    assert d is not None and d.problems == () and d.tracker_frame == 1
    assert [(x.idx, x.score, x.box) for x in d.detections] == [(0, 0.9, PERSON[:4]), (1, 0.45, TORSO[:4])]
    assert [(p.track_id, p.idx, p.outcome) for p in d.published] == [(3, 0, "confirmed")]
    assert [(u.idx, u.outcome, u.track_id) for u in d.unpublished] == [(1, "new_unconfirmed", 4)]
    assert d.lost == () and d.removed == () and d.second_stage_detections == 0
    # the production inputs the tracker received are exactly the captured ones, and untouched
    assert tracker.inputs[-1].rows == [PERSON, TORSO]
    # the shadow: below the confidence only, the agreement counted, its own limits read, the raw output cloned
    assert [(s.score, s.box) for s in d.shadow] == [(0.3, (300.0, 300.0, 340.0, 360.0))]
    assert (d.shadow_high, d.shadow_agree, d.shadow_candidates, d.shadow_limited) == (2, 2, 3, False)
    assert tools.calls[-1] == {"conf_thres": 0.1, "iou_thres": 0.7, "classes": [0], "max_det": 300, "end2end": False}
    assert set(d.timings_ns) == {"capture", "clone", "snapshot", "shadow"}


def test_the_shadow_pass_runs_on_a_clone_and_the_inference_output_is_returned_unchanged(engine) -> None:
    adapter, model, tracker = loaded(engine)
    raw = Raw([(*PERSON[:4], 0.9)])
    model.predictor.raw.append(raw)
    model.detections.append([PERSON])
    tracker.steps.append({"tracked": [STrack(1, 0, PERSON[:4], TRACKED, True, 1)], "lost": []})
    adapter.track(IMAGE)
    assert raw.clones == 1 and raw.anchors == [(*PERSON[:4], 0.9)]  # the NMS stand-in emptied the clone, not this


def test_the_shadow_pass_runs_in_inference_mode_like_the_clone_it_reads(engine) -> None:
    """Session 81: the clone is made inside stream_inference's inference mode, so it is an inference tensor, and the
    NMS's and scale_boxes' in-place writes raise outside the mode. The device run trk-20261016T100149Z recorded
    shadow_failed on 242 of 242 frames; the shadow pass as it ran before the fix (outside the mode) fails here the
    same way, and the fixed pass reads the same clone, once, with production's output unchanged either way."""
    def frame(tools: Tools) -> tuple[list[Any], TrackerDiagnostics, Raw]:
        Yolo.instances = []
        adapter, model, tracker = loaded(engine, tools)
        raw = Raw([(*PERSON[:4], 0.9), (300.0, 300.0, 340.0, 360.0, 0.3)])
        model.predictor.raw.append(raw)
        model.detections.append([PERSON])
        tracker.steps.append({"tracked": [STrack(1, 0, PERSON[:4], TRACKED, True, 1)], "lost": []})
        published = adapter.track(IMAGE)
        return published, adapter.last_diagnostics(), raw

    old = Tools(inference_mode=contextlib.nullcontext)
    before, failed, _ = frame(old)
    assert old.modes == [False] and failed.problems == ("shadow_failed",)  # the device run's failure, reproduced
    assert failed.shadow == () and failed.shadow_limited is None and failed.shadow_candidates == 2
    tools = Tools()
    after, d, raw = frame(tools)
    assert tools.modes == [True] and d.problems == ()  # the NMS ran in inference mode, and so did scale_boxes
    assert [(s.score, s.box) for s in d.shadow] == [(0.3, (300.0, 300.0, 340.0, 360.0))]
    assert (d.shadow_high, d.shadow_agree, d.shadow_candidates, d.shadow_limited) == (1, 1, 2, False)
    assert raw.clones == 1 and raw.anchors == [(*PERSON[:4], 0.9), (300.0, 300.0, 340.0, 360.0, 0.3)]  # one copy
    assert after == before and [t.track_id for t in after] == [1]  # production's tracks the same either way
    assert Mode.enabled is False  # the mode is left as it was found


def test_an_empty_shadow_result_is_a_completed_pass_and_an_exception_is_not(engine) -> None:
    """A shadow pass that finds nothing below the confidence completed (no problem; shadow_limited a flag); one that
    raises is shadow_failed, with shadow_limited left None (the reading tells them apart, session 81)."""
    adapter, model, tracker = loaded(engine)
    model.predictor.raw.append(Raw([(*PERSON[:4], 0.9)]))
    model.detections.append([PERSON])
    tracker.steps.append({"tracked": [STrack(1, 0, PERSON[:4], TRACKED, True, 1)], "lost": []})
    adapter.track(IMAGE)
    d = adapter.last_diagnostics()
    assert d.problems == () and d.shadow == () and d.shadow_limited is False and d.shadow_high == 1


def test_the_association_costs_are_the_trackers_own_matrices_at_the_pre_assignment_stage(engine) -> None:
    adapter, model, tracker = loaded(engine)
    lost = STrack(3, 0, PERSON[:4], LOST, True, 1)
    tracker.frame_id = 4  # the tracker's fifth update follows
    model.detections.append([TORSO])
    tracker.steps.append({"pool": [lost], "tracked": [STrack(5, 0, TORSO[:4], TRACKED, False, 5)], "lost": [lost]})
    adapter.track(IMAGE)
    d = adapter.last_diagnostics()
    assert d is not None
    (pool,) = d.assignments
    assert pool.stage == "pool" and pool.thresh == 0.8 and pool.dets == (0,) and pool.cost == ((0.5,),)
    assert pool.tracks == ((3, "lost", 4),)  # lost for four updates at that point (frame_id 5 - end_frame 1)
    assert d.lost[0].track_id == 3 and d.lost[0].frames_lost == 4
    assert tracker.dists_seen[-1] is tracker.matrices[-1]  # the matrix the tracker assigned with, untouched
    assert [(u.idx, u.outcome, u.track_id) for u in d.unpublished] == [(0, "new_unconfirmed", 5)]


def test_refound_continued_and_removed_follow_the_trackers_lists_across_frames_and_a_reset(engine) -> None:
    adapter, model, tracker = loaded(engine)
    seat = STrack(3, 0, PERSON[:4], TRACKED, True, 1)
    model.detections += [[PERSON], [], [PERSON], [], [PERSON]]
    tracker.steps += [
        {"tracked": [seat], "lost": []},
        {"tracked": [], "lost": [STrack(3, 0, PERSON[:4], LOST, True, 1)]},
        {"tracked": [STrack(3, 0, PERSON[:4], TRACKED, True, 3)], "lost": []},
        {"tracked": [], "lost": []},  # the tracker dropped it
        {"tracked": [STrack(3, 0, PERSON[:4], TRACKED, True, 5)], "lost": []},
    ]
    outcomes = []
    for _ in range(5):
        adapter.track(IMAGE)
        d = adapter.last_diagnostics()
        assert d is not None
        outcomes.append(([(p.track_id, p.outcome) for p in d.published], [x.track_id for x in d.lost], d.removed))
    assert outcomes == [([(3, "confirmed")], [], ()), ([], [3], ()), ([(3, "refound")], [], ()), ([], [], (3,)),
                        ([(3, "confirmed")], [], ())]
    adapter.reset()
    model.detections.append([PERSON])
    tracker.steps.append({"tracked": [STrack(3, 0, PERSON[:4], TRACKED, True, 1)], "lost": []})
    adapter.track(IMAGE)
    assert [(p.track_id, p.outcome) for p in adapter.last_diagnostics().published] == [(3, "confirmed")]


# ---------------------------------------------------------------- failures become named problems, never exceptions


def test_each_part_that_fails_is_a_named_problem_on_the_record_and_production_goes_on(engine) -> None:
    adapter, model, tracker = loaded(engine, Tools(fail=True))
    model.detections.append([PERSON])
    model.predictor.raw.append(Raw([(*PERSON[:4], 0.9)]))
    tracker.steps.append({"tracked": [STrack(1, 0, PERSON[:4], TRACKED, True, 1)], "lost": []})
    assert [t.track_id for t in adapter.track(IMAGE)] == [1]
    assert adapter.last_diagnostics().problems == ("shadow_failed",)
    # the raw output missing (the wrapper got nothing to clone), the tracker replaced behind the diagnostic's back
    model.predictor.raw.append(None)  # type: ignore[arg-type]
    model.predictor.trackers = [ByteTrack()]
    model.detections.append([PERSON])
    adapter.track(IMAGE)
    d = adapter.last_diagnostics()
    assert d.problems == ("tracker_replaced", "clone_failed") and d.published == () and d.lost == ()
    assert [(x.idx, x.score) for x in d.detections] == [(0, 0.9)]  # the capture still read the inputs


def test_a_frame_without_the_capture_reports_it_missing(engine, monkeypatch) -> None:
    adapter, model, tracker = loaded(engine)
    monkeypatch.setattr(Yolo, "track", lambda self, image, **kw: [SimpleNamespace(boxes=None)])  # no callbacks ran
    adapter.track(IMAGE)
    d = adapter.last_diagnostics()
    assert "capture_missing" in d.problems and "raw_missing" in d.problems and d.detections == ()


# ---------------------------------------------------------------- the boundary passes it through with its offsets


def test_the_person_tracker_passes_the_reading_through_with_this_epochs_id_offset(engine) -> None:
    adapter, model, tracker = loaded(engine)
    from sentinel.contracts import PixelFormat
    from sentinel.media.clock import FakeClock
    from sentinel.media.frames import FrameStamper

    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    person = PersonTracker(adapter, diagnostics=True)
    model.detections += [[PERSON], [PERSON]]
    tracker.steps += [{"tracked": [STrack(1, 0, PERSON[:4], TRACKED, True, 1)], "lost": []},
                      {"tracked": [STrack(1, 0, PERSON[:4], TRACKED, True, 1)], "lost": []}]

    def frame():
        clock.advance(0.1)
        return stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)

    first = person.process(frame(), IMAGE)
    assert first.diagnostics is not None and [p.track_id for p in first.diagnostics.published] == [1]
    model.detections.append([PERSON])
    tracker.steps.append({"tracked": [STrack(1, 0, PERSON[:4], TRACKED, True, 1)], "lost": []})
    original = Yolo.track

    def broken(self, image, **kwargs):
        raise TrackerError("backend_error")
    Yolo.track = broken  # type: ignore[method-assign]
    try:
        failed = person.process(frame(), IMAGE)
    finally:
        Yolo.track = original  # type: ignore[method-assign]
    assert failed.outcome.value == "failed" and failed.diagnostics is None
    after = person.process(frame(), IMAGE)  # the backend reset: its IDs restart, the boundary offsets them
    assert [p.track_id for p in after.persons] == [3]  # past every ID used before the reset (1), as published
    assert [p.track_id for p in after.diagnostics.published] == [3]  # the reading carries the same offset


def test_without_a_reading_backend_the_boundary_records_that_and_nothing_else() -> None:
    from sentinel.contracts import PixelFormat
    from sentinel.media.clock import FakeClock
    from sentinel.media.frames import FrameStamper

    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    plain = SimpleNamespace(track=lambda image: [], reset=lambda: None)
    person = PersonTracker(plain, diagnostics=True)
    clock.advance(0.1)
    result = person.process(stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR), IMAGE)
    assert result.diagnostics == TrackerDiagnostics(None, problems=("no_backend_diagnostics",))
    assert PersonTracker(plain).process  # the default: nothing read


def test_track_args_are_unchanged_by_the_diagnostic() -> None:
    assert TRACK_ARGS["conf"] == 0.4 and TRACK_ARGS["tracker"] == "bytetrack.yaml" and TRACK_ARGS["persist"] is True
