"""Demo workload for the resource-profile run (U17 option a, decision D28).

Started by ``demo_profile.py`` inside ``~/onvif_env`` (decision D24) with the
L4T libcuda preloaded (decision D27). Do not run it by hand.

It loads the two in-process demo models the way v1 and the demo's interim
adapters use them, then runs them together with scene requests to the
llama-server that ``demo_profile.py`` started:

- detector: the legacy ``yolov8n.engine`` through Ultralytics ``track()`` with
  ByteTrack and v1's arguments, on every frame at the source rate;
- face: DeepFace Facenet512 with the YuNet detector on whole frames, TensorFlow
  pinned to the CPU as v1 does, in a thread at a sampled rate;
- scene: one chat request at a time to llama-server, started every 4 s (D16),
  with v1's image shape (480x360 JPEG, quality 60) and the v2 SceneReport prompt.

Frames come from a replay clip (``--clip``) decoded by OpenCV/FFmpeg on the CPU,
as the demo decodes the camera (D24), paced at the source rate with
latest-frame semantics, or from synthetic noise frames without ``--clip``.

``--scene-only`` (S1, prompt-cache A/B) runs the scene requests alone: no
detector, face model, torch or CUDA driver in this process. Each request then
carries its own deterministic synthetic noise image (index i is the same in
every run), so no two requests send the same image.

``--mr1-release-check`` (MR1, opt-in) stops after the loads: after the
detector's and the face model's settles it emits ``mr1_checkpoint`` and waits,
bounded, for ``ack <stage>`` on stdin while the orchestrator releases that
model's file cache and samples memory. It then runs bounded smoke checks
(detector frames, one face analysis, one scene request), emits ``mr1_smoke``
with counts and fixed labels, waits for a last acknowledgement and exits. It
runs no warm-up or steady phase.

Structured results go to stdout as ``@@EVENT <json>`` lines. Model output text
is never written anywhere: the footage is private. Only counts and timings are.
After every scene request a ``scene_progress`` event carries cumulative
fixed-key counters since the workload started, so an interrupted run keeps its
counts up to the last request.

Torch allocator samples stream at most once per second, without retaining a
sample history. t_mono is sample-start time.monotonic() seconds, comparable to
memory.csv within the manifest's boot. Values are bytes for this process's
torch CUDA allocator on device 0; peaks cover its lifetime and are not reset.
Missing/failed readings are None. These exclude other processes and allocations
outside torch's allocator, so they neither measure total GPU/device memory nor
prove memory was reclaimed after unload. Sampling does not synchronize CUDA,
empty its cache or change allocation policy.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import math
import os
import select
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping

EVENT_PREFIX = "@@EVENT "
ALLOCATOR_INTERVAL_S = 1.0
ALLOCATOR_FIELDS = {
    "allocated_bytes": "allocated_bytes.all.current",
    "allocated_peak_bytes": "allocated_bytes.all.peak",
    "reserved_bytes": "reserved_bytes.all.current",
    "reserved_peak_bytes": "reserved_bytes.all.peak",
}
TRACK_ARGS = {  # as surveillance4_1.py calls yolo.track()
    "imgsz": 640,
    "device": 0,
    "half": True,
    "verbose": False,
    "classes": [0],
    "conf": 0.4,
    "persist": True,
    "tracker": "bytetrack.yaml",
}
FACE_ARGS = {"model_name": "Facenet512", "detector_backend": "yunet", "enforce_detection": False}
VLM_IMAGE_SIZE = (480, 360)
VLM_JPEG_QUALITY = 60
VLM_MAX_TOKENS = 200
VLM_MAX_RESPONSE_BYTES = 1_000_000
VLM_TIMEOUT_S = 30.0
MR1_ACK_TIMEOUT_S = 60.0  # the orchestrator's release and samples take a few seconds
MR1_DETECTOR_FRAMES = 15  # one second of the 15 fps replay, unpaced
WINDOW_S = 10  # unique-frame throughput windows (step-4 criterion T1)
MAX_FRAME_SAMPLES = 20_000  # 600 s at 15 fps is 9,000
DEMO_SCENE_JOB_TIMEOUT_S = 8.0  # D16; latencies above it are counted, not cut off
SYNTHETIC_SHAPE = (480, 640, 3)
SCENE_ONLY_SEED = b"sentinel-s1-scene-only-v1"
SCENE_ONLY_INPUT = "synthetic noise, distinct per request"
PROGRESS_FINISH_REASONS = ("stop", "length", "other", "missing")
PROGRESS_REJECTIONS = (
    "truncated", "invalid_report", "incomplete_completion", "malformed_response", "server_error",
    "unsupported_completion",
)
PROGRESS_ERRORS = ("http", "timeout", "other")
SCENE_SYSTEM = (
    "You are the scene-analysis component of a home security camera. "
    "Reply with exactly one JSON object and nothing else."
)
SCENE_PROMPT = (
    "Describe this camera image as JSON with exactly these fields: "
    '"persons_visible" (integer 0-50), "fire_or_smoke" (true or false), '
    '"threat" ("none", "low", "medium" or "high"), '
    '"observations" (up to 3 short phrases, aim under 48 characters each), '
    '"uncertainty" ("low", "medium" or "high"), '
    '"summary" (one short complete sentence, aim under 100 characters). '
    "Finish descriptions well before the schema limits; do not fill the available space."
)


def event(kind: str, /, **fields: object) -> None:
    sys.stdout.write(EVENT_PREFIX + json.dumps({"event": kind, **fields}) + "\n")
    sys.stdout.flush()


def torch_allocator_stats() -> Mapping[str, int]:
    """Lazy runtime provider; importing this module does not import torch."""
    import torch

    return torch.cuda.memory_stats(0)


class AllocatorSampler(threading.Thread):
    """Stream fixed-size allocator snapshots at 1 Hz, with no retained history."""

    def __init__(
        self,
        provider: Callable[[], Mapping[str, int]],
        *,
        clock: Callable[[], float] = time.monotonic,
        emit: Callable[..., None] = event,
    ) -> None:
        super().__init__(name="allocator-sampler", daemon=True)
        self.phase = "detector_settle"
        self._provider = provider
        self._clock = clock
        self._emit = emit
        self._next_sample = 0.0
        self._halt = threading.Event()

    def sample(self) -> bool:
        now = self._clock()
        if self._halt.is_set() or now < self._next_sample:
            return False
        self._next_sample = now + ALLOCATOR_INTERVAL_S
        phase = self.phase
        error = None
        try:
            readings = self._provider()
            values = {}
            for field, key in ALLOCATOR_FIELDS.items():
                value = readings.get(key)
                values[field] = value if type(value) is int and value >= 0 else None
        except Exception as exc:
            error = type(exc).__name__
            values = dict.fromkeys(ALLOCATOR_FIELDS)
        self._emit(
            "torch_allocator", t_mono=round(now, 3), phase=phase, clock="time.monotonic",
            units="bytes", device=0, peak_scope="allocator_lifetime",
            status="observed" if all(value is not None for value in values.values()) else "unavailable",
            error=error, **values,
        )
        return True

    def run(self) -> None:
        while not self._halt.is_set():
            self.sample()
            self._halt.wait(max(0.0, self._next_sample - self._clock()))

    def stop(self) -> None:
        self._halt.set()
        if self.ident is not None:
            self.join(timeout=5)


def percentiles(values: list[float]) -> dict[str, float | int | None]:
    """Nearest-rank p50/p95/p99/max of millisecond values."""
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)

    def rank(p: float) -> float:
        return round(ordered[max(1, math.ceil(p * len(ordered))) - 1], 1)

    return {"n": len(ordered), "p50": rank(0.50), "p95": rank(0.95), "p99": rank(0.99), "max": round(ordered[-1], 1)}


def scene_request_sha256() -> str | None:
    """Fingerprint of the request this workload sends (prompts, image shape, limits, schema), or None."""
    try:
        from sentinel.scene.llama_server import request_fingerprint
    except ImportError:
        return None
    return request_fingerprint(system=SCENE_SYSTEM, prompt=SCENE_PROMPT, image_size=VLM_IMAGE_SIZE,
                               jpeg_quality=VLM_JPEG_QUALITY, max_tokens=VLM_MAX_TOKENS, temperature=0.05,
                               model="lfm2-vl")


def check_cuda_driver() -> bool:
    """Decision D27: GPU work only with L4T's libcuda and a working cuInit."""
    cuda = ctypes.CDLL("libcuda.so.1")
    rc = cuda.cuInit(0)
    with open("/proc/self/maps") as maps:
        mapped = sorted({line.split()[-1] for line in maps if "libcuda" in line})
    ok = rc == 0 and bool(mapped) and all("/nvidia/" in path for path in mapped)
    event("cuda_driver", cuinit=rc, libcuda=mapped, ok=ok)
    return ok


class Frames:
    """Replay clip (looped) or synthetic frames, paced as a live source."""

    def __init__(self, clip: str | None, fps: float) -> None:
        import cv2
        import numpy as np

        self._cv2 = cv2
        self.clip = clip
        self.fps = fps
        self.loops = 0
        self.decoded = 0
        if clip:
            self._cap = cv2.VideoCapture(clip)
            if not self._cap.isOpened():
                raise SystemExit("cannot open the replay clip")
        else:
            rng = np.random.default_rng(0)
            self._synthetic = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(16)]

    def read(self):
        self.decoded += 1
        if not self.clip:
            return self._synthetic[self.decoded % len(self._synthetic)]
        ok, frame = self._cap.read()
        if not ok:
            self._cap.release()
            self._cap = self._cv2.VideoCapture(self.clip)
            self.loops += 1
            ok, frame = self._cap.read()
            if not ok:
                raise SystemExit("replay clip became unreadable")
        return frame


class Latest:
    """Latest decoded frame, shared with the face and scene threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame = None

    def put(self, frame) -> None:
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame


def synthetic_scene_bytes(index: int) -> bytes:
    """Deterministic noise pixels for request ``index``: the same bytes in every run, a different image per index."""
    return hashlib.shake_256(SCENE_ONLY_SEED + index.to_bytes(8, "big")).digest(math.prod(SYNTHETIC_SHAPE))


class SyntheticScenes:
    """Scene-only image source: each get() is a new image; nothing is retained after it is returned."""

    def __init__(self) -> None:
        import numpy as np

        self._np = np
        self.issued = 0

    def get(self):
        data = synthetic_scene_bytes(self.issued)
        self.issued += 1
        return self._np.frombuffer(bytearray(data), dtype=self._np.uint8).reshape(SYNTHETIC_SHAPE)


def load_detector(engine: str):
    started = time.monotonic()
    import numpy as np
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        event("fatal", reason="torch sees no CUDA device")
        raise SystemExit(3)
    model = YOLO(engine, task="detect")
    blank = np.zeros((480, 640, 3), np.uint8)
    for _ in range(3):
        model.track(blank, **TRACK_ARGS)
    event("detector_loaded", seconds=round(time.monotonic() - started, 2), torch=torch.__version__)
    return model


def load_face():
    started = time.monotonic()
    import numpy as np
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")  # v1: TensorFlow/DeepFace on the CPU
    except Exception as exc:  # noqa: BLE001 - recorded, as v1 only warns
        event("warning", detail=f"could not pin TensorFlow to the CPU: {type(exc).__name__}")
    from deepface import DeepFace

    DeepFace.build_model(FACE_ARGS["model_name"])
    DeepFace.represent(img_path=np.zeros((480, 640, 3), np.uint8), **FACE_ARGS)
    event(
        "face_loaded",
        seconds=round(time.monotonic() - started, 2),
        tensorflow=tf.__version__,
        tf_visible_gpus=len(tf.config.get_visible_devices("GPU")),
    )
    return DeepFace


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._clear()

    def reset(self) -> None:
        """Start counting afresh (at the start of the steady phase)."""
        with self.lock:
            self._clear()

    def _clear(self) -> None:
        self.det_ms: list[float] = []
        self.det_done: list[float] = []  # time.monotonic() when each unique frame's result was ready
        self.det_decode_age_ms: list[float] = []  # decode return -> result (actual ingest-to-result in the replay)
        self.det_schedule_age_ms: list[float] = []  # scheduled live arrival -> result (includes any backlog)
        self.det_person_frames = 0
        self.det_max_persons = 0
        self.source_frames = 0
        self.face_ms: list[float] = []
        self.face_runs_with_face = 0
        self.face_errors: dict[str, int] = {}
        self.vlm_ms: list[float] = []
        self.vlm_errors: dict[str, int] = {}
        self.vlm_valid = 0
        self.vlm_invalid = 0
        self.vlm_unchecked = 0
        self.vlm_rejections: dict[str, int] = {}
        self.vlm_finish_reasons: dict[str, int] = {}
        self.vlm_summary_at_limit = 0
        self.vlm_observations_at_limit = 0
        self.vlm_prompt_tokens: list[int] = []
        self.vlm_completion_tokens: list[int] = []

    def summary(self, seconds: float, window_start: float | None = None) -> dict[str, object]:
        with self.lock:
            over = sum(1 for ms in self.vlm_ms if ms > DEMO_SCENE_JOB_TIMEOUT_S * 1000)
            windows = None
            if window_start is not None:
                counts = [0] * int(seconds // WINDOW_S)
                for done in self.det_done:
                    index = int((done - window_start) // WINDOW_S)
                    if 0 <= index < len(counts):
                        counts[index] += 1
                windows = {"window_s": WINDOW_S, "count": len(counts),
                           "min_fps": round(min(counts) / WINDOW_S, 2) if counts else None,
                           "max_fps": round(max(counts) / WINDOW_S, 2) if counts else None}
            errors = dict(self.vlm_errors)
            http = sum(n for name, n in errors.items() if name.startswith("HTTP "))
            timeouts = errors.get("timeout", 0)

            def mean(values: list[int]) -> float | None:
                return round(sum(values) / len(values), 1) if values else None

            return {
                "seconds": round(seconds, 1),
                "detector": {
                    "source_frames": self.source_frames,
                    "processed_frames": len(self.det_ms),
                    "processed_fps": round(len(self.det_ms) / seconds, 2) if seconds else None,
                    "unique_fps": round(len(self.det_ms) / seconds, 2) if seconds else None,  # each frame once
                    "windows": windows,
                    "latency_ms": percentiles(self.det_ms),
                    "schedule_age_ms": percentiles(self.det_schedule_age_ms),
                    "decode_to_result_age_ms": percentiles(self.det_decode_age_ms),
                    "frames_with_person": self.det_person_frames,
                    "max_persons": self.det_max_persons,
                },
                "face": {
                    "runs": len(self.face_ms),
                    "achieved_hz": round(len(self.face_ms) / seconds, 2) if seconds else None,
                    "latency_ms": percentiles(self.face_ms),
                    "runs_with_face": self.face_runs_with_face,
                    "errors": dict(self.face_errors),
                    "error_count": sum(self.face_errors.values()),  # survives sanitizing that drops unknown names
                },
                "scene": {
                    "attempts": len(self.vlm_ms) + sum(errors.values()),
                    "client_timeouts": timeouts,
                    "http_errors": http,
                    "transport_errors": sum(errors.values()) - http - timeouts,
                    "completed": len(self.vlm_ms),
                    "latency_ms": percentiles(self.vlm_ms),
                    "over_d16_timeout": over,
                    "errors": errors,
                    "valid_reports": self.vlm_valid,
                    "invalid_reports": self.vlm_invalid,
                    "unchecked_reports": self.vlm_unchecked,
                    "rejected_reports_by_reason": dict(self.vlm_rejections),
                    "finish_reasons": dict(self.vlm_finish_reasons),
                    "valid_summaries_at_limit": self.vlm_summary_at_limit,
                    "valid_observations_at_limit": self.vlm_observations_at_limit,
                    "accuracy": "not evaluated; structural validity is not scene accuracy",
                    "prompt_tokens_mean": mean(self.vlm_prompt_tokens),
                    "completion_tokens_mean": mean(self.vlm_completion_tokens),
                },
            }


class SceneProgress:
    """Cumulative scene counters since the workload started (never reset), emitted after every request.

    Keys are fixed: finish reasons, rejection reasons and error classes are counted
    under fixed labels, with no exception text or model output.
    """

    def __init__(self, source=None, *, clock: Callable[[], float] = time.monotonic,
                 emit: Callable[..., None] | None = None) -> None:
        self.phase = "warmup"
        self._source = source
        self._clock = clock
        self._emit = emit
        self._lock = threading.Lock()
        self.requests = self.completed = self.valid = self.invalid = 0
        self.finish_reasons = dict.fromkeys(PROGRESS_FINISH_REASONS, 0)
        self.rejections = dict.fromkeys(PROGRESS_REJECTIONS, 0)
        self.errors = dict.fromkeys(PROGRESS_ERRORS, 0)

    def record(self, *, latency_ms: float | None = None, finish_reason: str | None = None,
               rejection: str | None = None, valid: bool = False, error: str | None = None,
               prompt_tokens: int | None = None, completion_tokens: int | None = None) -> None:
        with self._lock:
            self.requests += 1
            self.completed += int(latency_ms is not None)
            self.valid += int(valid)
            if finish_reason is not None:
                key = finish_reason if finish_reason in self.finish_reasons else "other"
                self.finish_reasons[key] += 1
            if rejection is not None:
                self.invalid += 1
                if rejection in self.rejections:
                    self.rejections[rejection] += 1
            if error is not None:
                self.errors[error if error in self.errors else "other"] += 1
            snapshot = {
                "t_mono": round(self._clock(), 3), "phase": self.phase, "requests": self.requests,
                "completed": self.completed, "valid_reports": self.valid, "invalid_reports": self.invalid,
                "finish_reasons": dict(self.finish_reasons),
                "rejected_reports_by_reason": dict(self.rejections), "errors": dict(self.errors),
                "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "synthetic_images_issued": getattr(self._source, "issued", None),
            }
        (self._emit or event)("scene_progress", **snapshot)


def face_attempt(deepface, frame, stats: Stats) -> None:
    """One face analysis of a frame, counted in stats (errors by class, never raised)."""
    started = time.perf_counter()
    try:
        faces = deepface.represent(img_path=frame, **FACE_ARGS)
        found = sum(1 for face in faces if float(face.get("face_confidence") or 0) > 0)
        with stats.lock:
            stats.face_ms.append((time.perf_counter() - started) * 1000)
            stats.face_runs_with_face += 1 if found else 0
    except Exception as exc:  # noqa: BLE001 - counted by class, never raised
        with stats.lock:
            name = type(exc).__name__
            stats.face_errors[name] = stats.face_errors.get(name, 0) + 1


def face_loop(deepface, latest: Latest, stats: Stats, hz: float, stop: threading.Event) -> None:
    period = 1.0 / hz
    next_start = time.monotonic()
    while not stop.is_set():
        frame = latest.get()
        if frame is not None:
            face_attempt(deepface, frame, stats)
        next_start += period
        stop.wait(max(0.0, next_start - time.monotonic()))


def scene_request(port: int, frame) -> dict:
    from sentinel.scene.completion import SceneCompletionError, scene_response_format

    import cv2

    small = cv2.resize(frame, VLM_IMAGE_SIZE)
    ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, VLM_JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    image = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")
    body = {
        "model": "lfm2-vl",
        "max_tokens": VLM_MAX_TOKENS,
        "temperature": 0.05,
        "stream": False,
        "response_format": scene_response_format(),
        "messages": [
            {"role": "system", "content": SCENE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": SCENE_PROMPT},
                ],
            },
        ],
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=VLM_TIMEOUT_S) as response:
        data = response.read(VLM_MAX_RESPONSE_BYTES + 1)
    if len(data) > VLM_MAX_RESPONSE_BYTES:
        raise SceneCompletionError("response_too_large")
    return json.loads(data)


def scene_attempt(port: int, frame, stats: Stats) -> dict[str, object]:
    """One scene request for a frame, counted in stats; returns the fixed-key outcome SceneProgress records."""
    started = time.perf_counter()
    outcome: dict[str, object] = {}
    try:
        from sentinel.scene.completion import (
            SceneCompletionError, parse_scene_completion, scene_completion_finish_reason,
        )
        from sentinel.scene.report import SceneReportError

        payload = scene_request(port, frame)
        elapsed = (time.perf_counter() - started) * 1000
        usage = payload.get("usage") if isinstance(payload, Mapping) else None
        usage = usage if isinstance(usage, Mapping) else {}
        finish_reason = scene_completion_finish_reason(payload)
        tokens = {key: usage[key] if type(usage.get(key)) is int and usage[key] >= 0 else None
                  for key in ("prompt_tokens", "completion_tokens")}
        outcome.update(latency_ms=elapsed, finish_reason=finish_reason, **tokens)
        with stats.lock:
            stats.vlm_ms.append(elapsed)
            stats.vlm_finish_reasons[finish_reason] = stats.vlm_finish_reasons.get(finish_reason, 0) + 1
            if tokens["prompt_tokens"] is not None:
                stats.vlm_prompt_tokens.append(tokens["prompt_tokens"])
            if tokens["completion_tokens"] is not None:
                stats.vlm_completion_tokens.append(tokens["completion_tokens"])
        try:
            report = parse_scene_completion(payload)
        except SceneReportError as exc:
            rejection = exc.reason if isinstance(exc, SceneCompletionError) else "invalid_report"
            outcome["rejection"] = rejection
            with stats.lock:
                stats.vlm_invalid += 1
                stats.vlm_rejections[rejection] = stats.vlm_rejections.get(rejection, 0) + 1
        else:
            outcome["valid"] = True
            schema = report.model_json_schema()["properties"]
            with stats.lock:
                stats.vlm_valid += 1
                stats.vlm_summary_at_limit += int(len(report.summary) == schema["summary"]["maxLength"])
                stats.vlm_observations_at_limit += sum(
                    len(observation) == schema["observations"]["items"]["maxLength"]
                    for observation in report.observations
                )
    except Exception as exc:  # noqa: BLE001 - counted by class, never raised or echoed
        if isinstance(exc, urllib.error.HTTPError):
            name, outcome["error"] = f"HTTP {exc.code}", "http"
        elif isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError):
            name, outcome["error"] = "timeout", "timeout"
        else:
            name, outcome["error"] = type(exc).__name__, "other"
        with stats.lock:
            stats.vlm_errors[name] = stats.vlm_errors.get(name, 0) + 1
    return outcome


def scene_loop(port: int, latest: Latest, stats: Stats, interval: float, stop: threading.Event,
               progress: SceneProgress | None = None) -> None:
    next_start = time.monotonic()
    while not stop.is_set():
        frame = latest.get()
        if frame is not None:
            outcome = scene_attempt(port, frame, stats)
            if progress is not None:
                progress.record(**outcome)
        next_start = max(next_start + interval, time.monotonic())
        stop.wait(max(0.0, next_start - time.monotonic()))


def run_workload(model, deepface, args, stats: Stats, allocator: AllocatorSampler) -> None:
    frames = Frames(args.clip, args.fps)
    latest = Latest()
    stop = threading.Event()
    progress = SceneProgress()
    workers = [
        threading.Thread(target=face_loop, args=(deepface, latest, stats, args.face_hz, stop), daemon=True),
        threading.Thread(target=scene_loop, args=(args.port, latest, stats, args.scene_interval_s, stop, progress),
                         daemon=True),
    ]
    period = 1.0 / args.fps
    started = time.monotonic()
    phase_end = started + args.warmup_s
    allocator.phase = "warmup"
    event("phase", name="warmup")
    for worker in workers:
        worker.start()
    in_steady = False
    steady_started = started
    shown = 0  # source frames made available so far
    while True:
        now = time.monotonic()
        if now >= phase_end:
            if in_steady:
                break
            in_steady = True
            stats.reset()
            steady_started = now
            phase_end = now + args.steady_s
            allocator.phase = progress.phase = "steady"
            event("phase", name="steady")
            event("steady_boundary", edge="start", boundary_t_mono=round(now, 3))
        due = int((now - started) / period) + 1  # frames a live source has delivered by now
        if due <= shown:
            time.sleep(max(0.0, started + shown * period - time.monotonic()))
            continue
        frame = None
        while shown < due:  # a live decoder decodes every frame; keep only the latest
            frame = frames.read()
            shown += 1
            with stats.lock:
                stats.source_frames += 1
        decoded_at = time.monotonic()
        scheduled_at = started + (shown - 1) * period  # when a live camera would have delivered this frame
        latest.put(frame)
        t0 = time.perf_counter()
        results = model.track(frame, **TRACK_ARGS)
        elapsed = (time.perf_counter() - t0) * 1000
        done = time.monotonic()
        persons = len(results[0].boxes) if results and results[0].boxes is not None else 0
        with stats.lock:
            stats.det_ms.append(elapsed)
            if len(stats.det_done) < MAX_FRAME_SAMPLES:
                stats.det_done.append(done)
                stats.det_decode_age_ms.append((done - decoded_at) * 1000)
                stats.det_schedule_age_ms.append((done - scheduled_at) * 1000)
            stats.det_person_frames += 1 if persons else 0
            stats.det_max_persons = max(stats.det_max_persons, persons)
    steady_ended = time.monotonic()
    # The steady interval ends here, before any worker is joined: nothing finished during teardown counts.
    event("steady_boundary", edge="end", boundary_t_mono=round(steady_ended, 3))
    allocator.phase = progress.phase = "stopping"
    event("phase", name="stopping")
    summary = stats.summary(steady_ended - steady_started, window_start=steady_started)
    summary["request_sha256"] = scene_request_sha256()
    stop.set()
    for worker in workers:
        worker.join(timeout=VLM_TIMEOUT_S + 5)
    summary["input"] = "replay clip" if args.clip else "synthetic noise"
    summary["clip_loops"] = frames.loops
    event("workload_stats", **summary)


def read_ack_line(timeout: float, fd: int = 0) -> str | None:
    """One line from the orchestrator within the timeout; None on timeout, EOF or an overlong line."""
    deadline = time.monotonic() + timeout
    data = b""
    while not data.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or len(data) > 256:
            return None
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            return None
        chunk = os.read(fd, 64)
        if not chunk:
            return None
        data += chunk
    return data.decode(errors="replace").strip()


def mr1_checkpoint(stage: str, read_line: Callable[[float], str | None] = read_ack_line) -> None:
    """Pause until the orchestrator acknowledges this stage; stop the workload if it does not."""
    event("mr1_checkpoint", stage=stage)
    if read_line(MR1_ACK_TIMEOUT_S) != f"ack {stage}":
        event("fatal", reason="mr1 checkpoint not acknowledged")
        raise SystemExit(4)


def run_smoke(model, deepface, args, *, frames=None, scene=None) -> dict[str, object]:
    """Bounded functional checks after the releases: detector frames, one face analysis, one scene request.

    Counts, latencies and fixed labels only. It checks that each component still runs, not accuracy.
    """
    frames = frames or Frames(args.clip, args.fps)
    scene = scene or scene_attempt
    stats = Stats()
    detector: dict[str, object] = {"frames_requested": MR1_DETECTOR_FRAMES, "processed": 0, "errors": {}}
    latencies: list[float] = []
    frame = None
    for _ in range(MR1_DETECTOR_FRAMES):
        try:
            frame = frames.read()
            started = time.perf_counter()
            model.track(frame, **TRACK_ARGS)
            latencies.append((time.perf_counter() - started) * 1000)
            detector["processed"] += 1
        except Exception as exc:  # noqa: BLE001 - counted by class, never raised
            name = type(exc).__name__
            detector["errors"][name] = detector["errors"].get(name, 0) + 1
    detector["latency_ms"] = percentiles(latencies)
    face: dict[str, object] = {"runs": 0, "completed": 0, "runs_with_face": 0, "errors": {}, "latency_ms": None}
    outcome: dict[str, object] = {}
    if frame is not None:
        face["runs"] = 1
        face_attempt(deepface, frame, stats)
        outcome = scene(args.port, frame, stats)
    with stats.lock:
        face.update(completed=len(stats.face_ms), runs_with_face=stats.face_runs_with_face, errors=dict(stats.face_errors),
                    latency_ms=round(stats.face_ms[0], 1) if stats.face_ms else None)
        scene_errors = dict(stats.vlm_errors)
    latency = outcome.get("latency_ms")
    return {
        "detector": detector,
        "face": face,
        "scene": {
            "attempts": 1 if frame is not None else 0, "completed": int(latency is not None),
            "finish_reason": outcome.get("finish_reason"), "valid": bool(outcome.get("valid")),
            "rejection": outcome.get("rejection"), "error": outcome.get("error"), "errors": scene_errors,
            "latency_ms": round(latency, 1) if isinstance(latency, float) else None,
        },
    }


def run_scene_only(args, stats: Stats) -> None:
    """S1: scene requests alone, each with a new deterministic image; no detector, face or CUDA here."""
    source = SyntheticScenes()
    progress = SceneProgress(source)
    stop = threading.Event()
    worker = threading.Thread(
        target=scene_loop, args=(args.port, source, stats, args.scene_interval_s, stop, progress), daemon=True,
    )
    event("phase", name="warmup")
    worker.start()
    time.sleep(args.warmup_s)
    progress.phase = "steady"
    stats.reset()
    event("phase", name="steady")
    started = time.monotonic()
    time.sleep(args.steady_s)
    steady_seconds = time.monotonic() - started  # before waiting for an in-flight request
    stop.set()
    worker.join(timeout=VLM_TIMEOUT_S + 5)
    summary = stats.summary(steady_seconds)
    event("workload_stats", seconds=summary["seconds"], scene=summary["scene"], input=SCENE_ONLY_INPUT,
          synthetic_images_issued=source.issued)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine")
    parser.add_argument("--scene-only", action="store_true",
                        help="S1: scene requests only, one distinct synthetic image per request; no detector/face/CUDA")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--clip")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--face-hz", type=float, default=1.0)  # D34
    parser.add_argument("--scene-interval-s", type=float, default=4.0)
    parser.add_argument("--settle-s", type=float, default=15.0)
    parser.add_argument("--warmup-s", type=float, default=120.0)
    parser.add_argument("--steady-s", type=float, default=600.0)
    parser.add_argument("--mr1-release-check", action="store_true",
                        help="MR1: pause after each settle for the orchestrator's cache release, then smoke checks only")
    args = parser.parse_args(argv)
    if args.scene_only:
        if args.clip:
            parser.error("--scene-only uses its own synthetic images; --clip is not allowed")
        if args.mr1_release_check:
            parser.error("--mr1-release-check needs the detector and face models; not with --scene-only")
        run_scene_only(args, Stats())
        return 0
    if not args.engine:
        parser.error("--engine is required unless --scene-only is given")

    if not check_cuda_driver():
        event("fatal", reason="libcuda is not L4T's or cuInit failed (decision D27)")
        return 3
    event("phase", name="detector_load")
    model = load_detector(args.engine)
    allocator = AllocatorSampler(torch_allocator_stats)
    allocator.start()
    try:
        event("phase", name="detector_settle")
        time.sleep(args.settle_s)
        if args.mr1_release_check:
            mr1_checkpoint("detector")
        allocator.phase = "face_load"
        event("phase", name="face_load")
        deepface = load_face()
        allocator.phase = "face_settle"
        event("phase", name="face_settle")
        time.sleep(args.settle_s)
        if args.mr1_release_check:
            mr1_checkpoint("face")
            allocator.phase = "smoke"
            event("phase", name="smoke")
            event("mr1_smoke", **run_smoke(model, deepface, args))
            mr1_checkpoint("after_smoke")
        else:
            run_workload(model, deepface, args, Stats(), allocator)
    finally:
        allocator.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
