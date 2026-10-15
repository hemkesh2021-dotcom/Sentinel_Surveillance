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

The tracking diagnostic (session 79; ``enable_diagnostic`` before ``load``, only for ``sentinel run
--face-validation --tracking-diagnostic``) reads, and never changes, what this path computes on each frame: the
detections handed to the tracker (a callback registered before Ultralytics registers its own, so it runs first on the
post-processed, pre-tracking results), the tracker's own association costs (its ``get_dists`` wrapped on the
instance; the matrix it returns is handed back unchanged), its track lists after the update, and separately a
second NMS at a lower confidence on a clone of the raw head output (``inference`` wrapped on the predictor; the
original output object is returned untouched). Production boxes, IDs and tracker state are the same with or without
it; without it no callback or wrapper is installed. Every part of it fails into a named problem on the frame's record,
never into the production path.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..tracking.diagnostic import (
    STAGES,
    Assignment,
    Detection,
    Lost,
    Published,
    ShadowBox,
    TrackerDiagnostics,
    Unpublished,
)
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
# Ultralytics 8.4.25 (installed, pinned by the environment): the unconfirmed-track assignment's limit
# (trackers/byte_tracker.py, BYTETracker.update: ``linear_assignment(dists, thresh=0.7)``) and NMS's candidate cap
# (utils/nms.py, ``max_nms``). Read from the installed source; recorded with every run so a reading never assumes them.
UNCONFIRMED_THRESH = 0.7
SHADOW_MAX_NMS = 30000
CALLBACK = "on_predict_postprocess_end"


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


def _import_diagnostic_tools() -> tuple[Any, Any, Any]:
    """(non_max_suppression, scale_boxes, TrackState) from the installed Ultralytics; only with the diagnostic."""
    from ultralytics.trackers.basetrack import TrackState
    from ultralytics.utils import nms, ops

    return nms.non_max_suppression, ops.scale_boxes, TrackState


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    inter = width * height if width > 0 and height > 0 else 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class LegacyUltralyticsTracker:
    """A TrackerBackend over Ultralytics ``YOLO.track()``; call load() before track()."""

    def __init__(
        self,
        engine: Path,
        *,
        expected_sha256: str = LEGACY_ENGINE_SHA256,
        driver_check: Callable[[], str | None] = cuda_driver_problem,
        importer: Callable[[], tuple[Any, Any, Any]] = _import_runtime,
        diagnostic_tools: Callable[[], tuple[Any, Any, Any]] = _import_diagnostic_tools,
    ) -> None:
        self._engine = Path(engine)
        self._expected = expected_sha256
        self._driver_check = driver_check
        self._importer = importer
        self._model: Any = None
        # The tracking diagnostic (off unless enable_diagnostic() ran before load())
        self._diagnostic_tools = diagnostic_tools
        self._shadow_conf: float | None = None
        self._capture_callback: Callable[[Any], None] = self._capture  # one bound object: registered and checked
        self._tools: tuple[Any, Any, Any] | None = None
        self._settings: dict[str, Any] | None = None
        self._predictor: Any = None
        self._tracker_ref: Any = None
        self._slot: tuple[list, list, list] | None = None
        self._slot_problem: str | None = None
        self._slot_ns = 0
        self._raw: tuple[Any, tuple[int, ...]] | None = None
        self._raw_problem: str | None = None
        self._raw_ns = 0
        self._assign: list[Assignment] = []
        self._assign_problem: str | None = None
        self._prev_activated: set[int] = set()
        self._prev_lost: set[int] = set()
        self._prev_tracked: set[int] = set()
        self._last: TrackerDiagnostics | None = None

    def __repr__(self) -> str:
        return f"LegacyUltralyticsTracker(loaded={self._model is not None})"

    # ---------------------------------------------------------------- the diagnostic's switch (before load)

    def enable_diagnostic(self, shadow_conf: float = 0.1) -> None:
        """Read the tracker's inputs, costs and state on every frame, and a shadow NMS at ``shadow_conf``; before
        load(), which then refuses (``diagnostic_unsupported``) unless the installed Ultralytics exposes what it reads."""
        if self._model is not None:
            raise TrackerError("diagnostic_after_load")
        if not 0.0 < float(shadow_conf) < float(TRACK_ARGS["conf"]):
            raise TrackerError("diagnostic_shadow_conf")
        self._shadow_conf = float(shadow_conf)

    def diagnostic_settings(self) -> dict[str, Any] | None:
        """The tracker's settings as the loaded objects hold them (numbers; ``fuse_score`` a flag), or None."""
        return None if self._settings is None else dict(self._settings)

    def last_diagnostics(self) -> TrackerDiagnostics | None:
        """What the diagnostic read on the last track() call; None without the diagnostic."""
        return self._last

    # ---------------------------------------------------------------- load and track

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
            if self._shadow_conf is not None:  # before the first track(): Ultralytics registers its tracker's then
                model.add_callback(CALLBACK, self._capture_callback)
            blank = numpy.zeros(WARMUP_SHAPE, numpy.uint8)
            for _ in range(WARMUP_FRAMES):
                model.track(blank, **TRACK_ARGS)
        except Exception as exc:
            raise TrackerError("load_failed", type(exc).__name__) from None
        if self._shadow_conf is not None:
            self._install(model)  # refuses before this backend is usable; nothing of the diagnostic is left half set
        self._model = model
        self.reset()

    def track(self, image: Any) -> list[RawTrack]:
        if self._model is None:
            raise TrackerError("not_loaded")
        if self._shadow_conf is not None:
            self._begin_frame()
        results = self._model.track(image, **TRACK_ARGS)
        boxes = results[0].boxes if results else None
        tracks: list[RawTrack] = []
        # Without tracks, Ultralytics returns the untracked detections with id None (v1 skips them too).
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            ids = boxes.id.cpu().tolist()
            if not len(xyxy) == len(confidences) == len(ids):
                raise TrackerError("inconsistent_output")
            for (x1, y1, x2, y2), confidence, track_id in zip(xyxy, confidences, ids):
                if not float(track_id).is_integer():
                    raise TrackerError("inconsistent_output")
                tracks.append(RawTrack(int(track_id), x1, y1, x2, y2, confidence))
        if self._shadow_conf is not None:
            self._last = self._collect(tracks, image)  # reads only; any failure is a named problem on the record
        return tracks

    def reset(self) -> None:
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or ():
            tracker.reset()  # clears tracked/lost tracks and restarts IDs
        self._prev_activated, self._prev_lost, self._prev_tracked = set(), set(), set()
        self._last = None

    # ---------------------------------------------------------------- the diagnostic's parts (read only)

    def _install(self, model: Any) -> None:
        """Verify what the diagnostic reads on the loaded objects, then wrap the tracker's get_dists and the
        predictor's inference on the instances; TrackerError(``diagnostic_unsupported``, what) otherwise."""
        predictor = getattr(model, "predictor", None)
        callbacks = getattr(predictor, "callbacks", None)
        order = callbacks.get(CALLBACK) if isinstance(callbacks, dict) else None
        if not order or order[0] is not self._capture_callback or len(order) < 2:
            raise TrackerError("diagnostic_unsupported", "callback_order")
        trackers = getattr(predictor, "trackers", None)
        tracker = trackers[0] if trackers else None
        needed = ("tracked_stracks", "lost_stracks", "removed_stracks", "frame_id", "max_time_lost", "args",
                  "get_dists", "update")
        if tracker is None or not all(hasattr(tracker, name) for name in needed):
            raise TrackerError("diagnostic_unsupported", "tracker_state")
        args = getattr(predictor, "args", None)
        if args is None or not all(hasattr(args, name) for name in ("iou", "max_det")) \
                or not callable(getattr(predictor, "inference", None)):
            raise TrackerError("diagnostic_unsupported", "predictor")
        try:
            self._tools = self._diagnostic_tools()
        except Exception as exc:  # noqa: BLE001 - an import problem is a label, never text
            raise TrackerError("diagnostic_unsupported", type(exc).__name__) from None
        targs = tracker.args
        try:
            self._settings = {
                "conf": float(TRACK_ARGS["conf"]), "match_thresh": float(targs.match_thresh),
                "track_high_thresh": float(targs.track_high_thresh), "track_low_thresh": float(targs.track_low_thresh),
                "new_track_thresh": float(targs.new_track_thresh), "track_buffer": int(targs.track_buffer),
                "max_time_lost": int(tracker.max_time_lost), "fuse_score": bool(targs.fuse_score),
                "unconfirmed_thresh": UNCONFIRMED_THRESH, "shadow_conf": float(self._shadow_conf or 0.0),
                "shadow_max_nms": SHADOW_MAX_NMS, "max_det": int(args.max_det), "iou": float(args.iou),
            }
        except (AttributeError, TypeError, ValueError):
            raise TrackerError("diagnostic_unsupported", "tracker_args") from None
        self._predictor, self._tracker_ref = predictor, tracker
        original_dists = tracker.get_dists
        tracker.get_dists = lambda tracks, detections: self._on_dists(original_dists, tracks, detections)
        original_inference = predictor.inference
        predictor.inference = lambda im, *a, **k: self._on_inference(original_inference, im, *a, **k)

    def _begin_frame(self) -> None:
        self._slot, self._slot_problem, self._slot_ns = None, None, 0
        self._raw, self._raw_problem, self._raw_ns = None, None, 0
        self._assign, self._assign_problem = [], None

    def _capture(self, predictor: Any) -> None:
        """The callback: the post-processed results before the tracker's callback replaces them."""
        started = time.perf_counter_ns()
        try:
            results = getattr(predictor, "results", None)
            boxes = results[0].boxes if results else None
            if boxes is None:
                self._slot = ([], [], [])
            else:
                self._slot = (boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())
        except Exception:  # noqa: BLE001 - never into the production path
            self._slot, self._slot_problem = None, "capture_failed"
        self._slot_ns = time.perf_counter_ns() - started

    def _on_inference(self, original: Callable[..., Any], im: Any, *args: Any, **kwargs: Any) -> Any:
        preds = original(im, *args, **kwargs)
        started = time.perf_counter_ns()
        try:
            raw = preds[0] if isinstance(preds, (list, tuple)) else preds
            self._raw = (raw.detach().clone(), tuple(int(x) for x in im.shape[2:]))
        except Exception:  # noqa: BLE001
            self._raw, self._raw_problem = None, "clone_failed"
        self._raw_ns = time.perf_counter_ns() - started
        return preds

    def _on_dists(self, original: Callable[..., Any], tracks: Any, detections: Any) -> Any:
        dists = original(tracks, detections)
        try:
            tracker, state = self._tracker_ref, self._tools[2] if self._tools else None
            stage = STAGES[min(len(self._assign), 1)]
            thresh = float(tracker.args.match_thresh) if stage == "pool" else UNCONFIRMED_THRESH
            frame_id = int(tracker.frame_id)
            rows = tuple((int(t.track_id), "lost" if state is not None and t.state == state.Lost else "tracked",
                          max(0, frame_id - int(t.end_frame))) for t in tracks)
            dets = tuple(int(d.idx) for d in detections)
            matrix = dists.tolist() if hasattr(dists, "tolist") else dists
            cost = tuple(tuple(round(float(c), 4) for c in row) for row in matrix)
            self._assign.append(Assignment(stage, thresh, rows, dets, cost))
        except Exception:  # noqa: BLE001
            self._assign_problem = "assign_failed"
        return dists

    def _collect(self, tracks: list[RawTrack], image: Any) -> TrackerDiagnostics:
        problems: list[str] = []
        timings = {"capture": self._slot_ns, "clone": self._raw_ns}
        tracker = None
        trackers = getattr(self._predictor, "trackers", None)
        if not trackers or trackers[0] is not self._tracker_ref:
            problems.append("tracker_replaced")
        else:
            tracker = self._tracker_ref
        detections: tuple[Detection, ...] = ()
        if self._slot is None:
            problems.append(self._slot_problem or "capture_missing")
        else:
            xyxy, conf, _ = self._slot
            detections = tuple(Detection(i, float(s), (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
                               for i, (b, s) in enumerate(zip(xyxy, conf)))
        started = time.perf_counter_ns()
        published: tuple[Published, ...] = ()
        unpublished: tuple[Unpublished, ...] = ()
        lost: tuple[Lost, ...] = ()
        removed: tuple[int, ...] = ()
        tracker_frame = None
        second = 0
        if tracker is not None:
            try:
                frame_id = int(tracker.frame_id)
                tracker_frame = frame_id
                tracked, lost_list = list(tracker.tracked_stracks), list(tracker.lost_stracks)
                activated = {int(t.track_id) for t in tracked if t.is_activated}
                lost_ids = {int(t.track_id) for t in lost_list}
                pub = []
                for t in tracked:
                    if not t.is_activated:
                        continue
                    n = int(t.track_id)
                    outcome = ("refound" if n in self._prev_lost else "continued" if n in self._prev_activated
                               else "confirmed")
                    pub.append(Published(n, int(t.idx), outcome, tuple(float(v) for v in t.xyxy.tolist())))
                consumed = {p.idx for p in pub}
                unconfirmed = {int(t.idx): int(t.track_id) for t in tracked if not t.is_activated}
                unpub = [Unpublished(d.idx, "new_unconfirmed", unconfirmed[d.idx]) if d.idx in unconfirmed
                         else Unpublished(d.idx, "unmatched", None) for d in detections if d.idx not in consumed]
                lost = tuple(Lost(int(t.track_id), max(0, frame_id - int(t.end_frame)),
                                  tuple(float(v) for v in t.xyxy.tolist())) for t in lost_list)
                removed = tuple(sorted((self._prev_lost | self._prev_tracked) - activated - lost_ids
                                       - set(unconfirmed.values())))
                self._prev_activated, self._prev_lost = activated, lost_ids
                self._prev_tracked = activated | set(unconfirmed.values())
                published, unpublished = tuple(pub), tuple(unpub)
                low, high = float(tracker.args.track_low_thresh), float(tracker.args.track_high_thresh)
                second = sum(1 for d in detections if low < d.score < high)
            except Exception:  # noqa: BLE001
                problems.append("snapshot_failed")
        timings["snapshot"] = time.perf_counter_ns() - started
        if self._assign_problem:
            problems.append(self._assign_problem)
        started = time.perf_counter_ns()
        shadow: tuple[ShadowBox, ...] = ()
        high_count, agree, candidates, limited = 0, 0, None, None
        if self._raw is None:
            problems.append(self._raw_problem or "raw_missing")
        elif self._tools is None:
            problems.append("shadow_failed")
        else:
            try:
                nms_fn, scale_boxes, _ = self._tools
                raw, im_shape = self._raw
                args = self._predictor.args
                conf = float(TRACK_ARGS["conf"])
                end2end = bool(getattr(getattr(self._predictor, "model", None), "end2end", False))
                try:  # the anchors above the shadow confidence, counted on the clone before its NMS touches it
                    if int(raw.ndim) == 3 and int(raw.shape[1]) >= 5:
                        candidates = int((raw[0, 4, :] > float(self._shadow_conf or 0.0)).sum().item())
                    else:
                        problems.append("layout_unknown")
                except Exception:  # noqa: BLE001
                    problems.append("layout_unknown")
                out = nms_fn(raw, conf_thres=float(self._shadow_conf or 0.0), iou_thres=float(args.iou),
                             classes=list(TRACK_ARGS["classes"]), max_det=int(args.max_det), end2end=end2end)
                det = out[0]
                rows: list[tuple[float, tuple[float, float, float, float]]] = []
                if len(det):
                    boxes = scale_boxes(im_shape, det[:, :4], image.shape)
                    scores = det[:, 4]
                    rows = [(float(s), tuple(float(v) for v in b))
                            for b, s in zip(boxes.cpu().tolist(), scores.cpu().tolist())]
                shadow = tuple(ShadowBox(s, b) for s, b in rows if s < conf)
                high = [b for s, b in rows if s >= conf]
                high_count = len(high)
                agree = sum(1 for d in detections if any(_iou(d.box, b) >= 0.9 for b in high))
                limited = (candidates is not None and candidates > SHADOW_MAX_NMS) or len(rows) >= int(args.max_det)
            except Exception:  # noqa: BLE001
                problems.append("shadow_failed")
        timings["shadow"] = time.perf_counter_ns() - started
        return TrackerDiagnostics(tracker_frame, detections, published, unpublished, lost, removed,
                                  tuple(self._assign), second, shadow, high_count, agree, candidates, limited,
                                  timings, tuple(problems))
