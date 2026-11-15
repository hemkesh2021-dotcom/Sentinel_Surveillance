"""The tracking diagnostic on the installed torch and Ultralytics, CPU only, synthetic tensors (session 81).

The device run trk-20261016T100149Z recorded ``shadow_failed`` on all 242 frames: the raw output is cloned inside
Ultralytics' ``stream_inference``, which runs under ``smart_inference_mode``, so the clone is an inference tensor, and
8.4.25's NMS (and ``scale_boxes``) write into what they are given, which torch refuses outside inference mode. The
portable fakes (test_tracking_diagnostic_backend.py) model that; this module runs the installed objects themselves:
``get_default_callbacks``, ``register_tracker``, ``on_predict_start`` (a real BYTETracker from the bundled
bytetrack.yaml), ``get_cfg``, ``smart_inference_mode``, ``BasePredictor.inference``, DetectionPredictor's
``postprocess`` (the real NMS, ``scale_boxes`` and ``Results``), the tracker's own callback and update, and the
adapter's load, hooks and shadow pass with its real tools and callback references. Only the forward pass (a synthetic
head output), the preprocessed input tensor and the data loader are stand-ins; no engine, weights, camera or GPU.

Skipped where torch or Ultralytics is not installed (the repository's .venv and CI). On the device, CPU only and
without installing anything (the .venv's pytest appended after the device environment's own packages):

    CUDA_VISIBLE_DEVICES= YOLO_OFFLINE=true PYTHONPATH=$PWD/src ~/onvif_env/bin/python -c "import sys; \\
      sys.path.append('/home/villain8001/sentinel-surveillance/.venv/lib/python3.10/site-packages'); import pytest; \\
      sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', 'tests/unit/test_tracking_diagnostic_installed.py']))"
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

with warnings.catch_warnings():  # the device environment's own import-time warnings (NVML, matplotlib) are not ours
    warnings.simplefilter("ignore")
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    os.environ["YOLO_OFFLINE"] = "true"  # as load() sets it before Ultralytics is first imported
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.trackers import register_tracker
    from ultralytics.utils import nms
    from ultralytics.utils.callbacks.base import get_default_callbacks
    from ultralytics.utils.torch_utils import smart_inference_mode

from sentinel.inference.legacy_ultralytics import CALLBACK, LegacyUltralyticsTracker, _import_diagnostic_tools
from sentinel.tracking.diagnostic import settings_problem

INPUT = (640, 640)  # the engine's fixed input: the 480x640 frame letterboxed with 80 rows above and below
FRAME_SHAPE = (480, 640, 3)
PERSON = (160.0, 355.0, 120.0, 390.0, 0.9)  # cx, cy, w, h in the input, person score: (100, 80, 220, 470) in the frame
FAINT = (480.0, 300.0, 40.0, 60.0, 0.2)  # below the production confidence, above the shadow's: (460, 190, 500, 250)
NOISE = (50.0, 50.0, 10.0, 10.0, 0.05)  # below both


def head(*anchors: tuple[float, float, float, float, float], n: int = 8) -> Any:
    """A synthetic head output (1, 84, n): per anchor cx, cy, w, h in the input, then 80 class scores (person = 0)."""
    raw = torch.zeros((1, 84, n))
    for i, (cx, cy, w, h, score) in enumerate(anchors):
        raw[0, :4, i] = torch.tensor([cx, cy, w, h])
        raw[0, 4, i] = score
    return raw


class Engine:
    """The forward pass's stand-in (AutoBackend's place): the next synthetic head output, made where a real forward
    pass makes it, inside the predictor's inference mode."""

    def __init__(self) -> None:
        self.heads: list[Any] = []
        self.returned: list[Any] = []
        self.names = {i: f"class{i}" for i in range(80)}
        self.end2end = False

    def __call__(self, im: Any, *args: Any, **kwargs: Any) -> Any:
        out = (self.heads.pop(0) if self.heads else head()).clone()
        self.returned.append(out)
        return out


class Predictor(DetectionPredictor):
    """DetectionPredictor's own inference, postprocess, construct_results, construct_result and run_callbacks, on a
    predictor made without BasePredictor.__init__ (which would set up a model, a source and a save directory); the
    attributes those methods read are set here, the arguments by the installed get_cfg."""

    def __init__(self, callbacks: dict, engine: Engine, args: Any) -> None:  # deliberately not super().__init__
        self.callbacks, self.model, self.args = callbacks, engine, args  # the model's dict, shared, as installed
        self.dataset = SimpleNamespace(bs=1, mode="image")
        self.save_dir = Path("unused")  # the tracker's callback joins a result's path to it; nothing is written
        self.batch = None
        self.results: list[Any] = []

    @smart_inference_mode()
    def stream(self, frame: Any) -> list[Any]:
        """stream_inference's order for one frame (8.4.25 engine/predictor.py 276-360): on_predict_start,
        preprocess, inference, postprocess, then on_predict_postprocess_end, all under smart_inference_mode."""
        self.run_callbacks("on_predict_start")
        self.batch = (["frame.jpg"], [frame], [""])
        im = torch.zeros((1, 3, *INPUT))  # the letterboxed input's shape; its pixels are never read
        preds = self.inference(im)
        self.results = self.postprocess(preds, im, [frame])
        self.run_callbacks("on_predict_postprocess_end")
        return self.results


class Yolo:
    """The parts of ultralytics.YOLO the adapter drives, on the installed objects: the callbacks from
    get_default_callbacks(), add_callback as Model's (append), track() as Model.track (register_tracker on the first
    call), the predictor made with the model's dict and get_cfg's arguments."""

    instances: list[Yolo] = []

    def __init__(self, path: str, task: str) -> None:
        self.callbacks = get_default_callbacks()
        self.predictor: Predictor | None = None
        self.engine = Engine()
        Yolo.instances.append(self)

    def add_callback(self, event: str, func: Any) -> None:
        self.callbacks[event].append(func)

    def track(self, source: Any = None, stream: bool = False, persist: bool = False, **kwargs: Any) -> list[Any]:
        if not hasattr(self.predictor, "trackers"):
            register_tracker(self, persist)
        if self.predictor is None:
            overrides = {k: v for k, v in kwargs.items() if k != "device"}  # CPU only: the engine's device is unused
            args = get_cfg(overrides={**overrides, "mode": "track", "batch": 1, "save": False, "rect": True})
            self.predictor = Predictor(self.callbacks, self.engine, args)
        return self.predictor.stream(source)


@pytest.fixture(autouse=True)
def clean() -> None:
    Yolo.instances = []


@pytest.fixture
def engine_file(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "yolov8n.engine"
    path.write_bytes(b"not a real engine")
    return path, hashlib.sha256(b"not a real engine").hexdigest()


def adapter(engine_file: tuple[Path, str], *, diagnostic: bool = True,
            tools: Any = _import_diagnostic_tools) -> tuple[LegacyUltralyticsTracker, Yolo]:
    path, digest = engine_file
    backend = LegacyUltralyticsTracker(
        path, expected_sha256=digest, driver_check=lambda: None, diagnostic_tools=tools,
        importer=lambda: (SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)), Yolo, numpy))
    if diagnostic:
        backend.enable_diagnostic()
    backend.load()  # the warm-up track() calls register the tracker's callback, then the diagnostic's hooks
    return backend, Yolo.instances[-1]


def one_frame(backend: LegacyUltralyticsTracker, model: Yolo, *anchors: Any) -> list[Any]:
    model.engine.heads.append(head(*anchors))
    return backend.track(numpy.zeros(FRAME_SHAPE, numpy.uint8))


def test_installed_inference_mode_refuses_the_nms_write_outside_it() -> None:
    """The cause, on the installed torch and NMS: a clone made under smart_inference_mode is an inference tensor,
    and the NMS's in-place conversion of it raises outside inference mode and runs inside it."""
    @smart_inference_mode()
    def clone_like_the_wrapper(raw: Any) -> Any:
        return raw.detach().clone()

    clone = clone_like_the_wrapper(head(PERSON))
    assert clone.is_inference()
    with pytest.raises(RuntimeError, match="[Ii]nference tensor outside InferenceMode"):
        nms.non_max_suppression(clone, conf_thres=0.1, iou_thres=0.7, classes=[0], max_det=300)
    with torch.inference_mode():
        out = nms.non_max_suppression(clone_like_the_wrapper(head(PERSON)), conf_thres=0.1, iou_thres=0.7,
                                      classes=[0], max_det=300)
    assert out[0][:, :5].tolist() == [[100.0, 160.0, 220.0, 550.0, pytest.approx(0.9)]]


def test_the_shadow_pass_completes_on_the_installed_lifecycle_and_production_is_unchanged(engine_file,
                                                                                          monkeypatch) -> None:
    backend, model = adapter(engine_file)
    order = model.callbacks[CALLBACK]  # the installed shape: Ultralytics' no-op, the capture, the tracker's partial
    assert len(order) == 3 and order[1] is backend._capture_callback
    assert settings_problem(backend.diagnostic_settings()) is None
    sources: list[int] = []
    original_clone = torch.Tensor.clone

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:  # which tensor each clone copies, by its storage
        sources.append(self.data_ptr())
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counted)
    published = one_frame(backend, model, PERSON, FAINT, NOISE)
    monkeypatch.undo()
    d = backend.last_diagnostics()
    assert d is not None and d.problems == ()
    preds = model.engine.returned[-1]  # what the forward pass handed production
    assert sources.count(preds.data_ptr()) == 1  # the raw output copied once, by the wrapper; no second copy
    clone = backend._raw[0]
    assert clone.is_inference() and clone.data_ptr() != preds.data_ptr()  # made in the mode; production's apart
    assert [(x.idx, round(x.score, 3), x.box) for x in d.detections] == [(0, 0.9, (100.0, 80.0, 220.0, 470.0))]
    assert [(round(s.score, 3), s.box) for s in d.shadow] == [(0.2, (460.0, 190.0, 500.0, 250.0))]
    assert (d.shadow_high, d.shadow_agree, d.shadow_candidates, d.shadow_limited) == (1, 1, 2, False)
    assert [(p.track_id, p.idx, p.outcome) for p in d.published] == [(1, 0, "confirmed")]
    assert not torch.is_inference_mode_enabled()  # the mode is left as it was found

    plain, plain_model = adapter(engine_file, diagnostic=False)
    assert one_frame(plain, plain_model, PERSON, FAINT, NOISE) == published  # the same tracks, boxes and scores
    assert torch.equal(plain_model.engine.returned[-1], preds)  # production's own tensor, as production left it
    assert torch.equal(plain_model.predictor.results[0].boxes.data, model.predictor.results[0].boxes.data)
    assert plain.last_diagnostics() is None


def test_the_shadow_pass_as_it_ran_before_the_fix_fails_on_the_installed_lifecycle(engine_file) -> None:
    """The tools without inference mode (the code before session 81, which ran both outside it): every frame's shadow
    pass raises into shadow_failed, as on the device, while production's tracks are unaffected."""
    old = lambda: (*_import_diagnostic_tools()[:3], contextlib.nullcontext)  # noqa: E731
    backend, model = adapter(engine_file, tools=old)
    published = one_frame(backend, model, PERSON, FAINT, NOISE)
    d = backend.last_diagnostics()
    assert d.problems == ("shadow_failed",) and d.shadow == () and d.shadow_limited is None
    assert [(t.track_id, t.x1, t.y1, t.x2, t.y2) for t in published] == [(1, 100.0, 80.0, 220.0, 470.0)]


def test_an_empty_shadow_result_on_the_installed_lifecycle_is_a_completed_pass(engine_file) -> None:
    """Nothing below the production confidence: the pass completed (no problem, shadow_limited a flag), which the
    reading counts as a successful pass; and a frame with no person at all completes too."""
    backend, model = adapter(engine_file)
    one_frame(backend, model, PERSON)
    d = backend.last_diagnostics()
    assert d.problems == () and d.shadow == () and d.shadow_limited is False and d.shadow_high == 1
    assert one_frame(backend, model) == []
    d = backend.last_diagnostics()
    assert d.problems == () and d.detections == () and d.shadow == () and d.shadow_limited is False
