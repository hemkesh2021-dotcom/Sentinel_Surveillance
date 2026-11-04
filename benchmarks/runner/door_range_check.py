"""The door-range check (session 116, part 11): what the camera's two streams give at the required distances.

The maintainer's requirement is recognition at the front door, about 4-5 m. This one guarded live-CCTV session measures,
at three marks whose distances the operator measured and typed (the seat, about 2.5 m, the doorway):
- **Actual face sizes**, in each stream's pixels: 640x480 H.264 substream and 2304x1296 HEVC main stream (read from the
  frames, never assumed).
- **Detection outcomes** on both streams, frame pair by frame pair, in two configurations:
  - ``production``: DeepFace's represent() on the whole frame, exactly as the runtime calls it, unchanged. On a large
    frame DeepFace pads it and its YuNet client scales it to 640 pixels (half scale on the substream, about 0.14 on the
    main stream).
  - ``native``: D88's steps on the frame at its own size: the same weights and YuNet at the frame's size, then
    production's cut, alignment, normalization and Facenet512. On the main stream this goes past D88's 640-pixel limit,
    which is why it is this tool's own method (``RangeBackend.native_faces``; the runtime and D88 are unchanged, and a
    test ties the two together on a frame D88 accepts).
- **Matching outcomes** against the existing gallery (a working copy), through the runtime's own vote rule.
- **The two streams' relation**: on empty-room pairs, an affine estimate from ORB features; on the marks, the ratio of
  the two streams' face widths.
- **Decode cost** with no inference: frames, process CPU and whether the stream keeps pace, with each stream alone and
  with both.

**Thresholds are the runtime's own, unchanged**: YuNet 0.9, match 0.70, margin 0.05, quality 0.6. A candidate between
the recording floor (0.3) and 0.9 is reported as a size and a score only: it is never embedded, matched or called a
face.

**Numbers only.** No frame, crop, embedding or identity ID leaves memory; this module saves no image or video and has
no way to. Frames are dropped when a sample ends. Similarities stay in the private result file; the reading shows them
only when asked (``read --with-similarity``), and they never go into Git.

**What it is not.** One person, one session, facing the camera on purpose: a best case for presentation. So every count
is "k of n samples at this mark", never a rate or an accuracy, and never an F2. A negative at a mark says that
configuration gave nothing there in this session; a positive validates nothing. Without a person detector here, the
runtime's ownership rule (a face votes only inside a person's box) is not exercised, and neither is a non-enrolled
person. A frame with more than one kept face stays unresolved: its faces are listed but counted apart, and no size or
vote is taken from it. Inference timings include the concurrent decoding of both streams, so they are not the runtime's.
It changes no limit, gate, threshold or admission.

Usage: ``door_range_check.py run --identity-dir DIR --out FILE --seat-m M --mid-m M --door-m M`` (the camera URLs from
SENTINEL_RTSP_URL and SENTINEL_RTSP_MAIN_URL, the passphrase from SENTINEL_IDENTITY_PASSPHRASE), under step5_guard, which
prints ``cues`` from this tool's ``{"run": "starting"}`` line; ``door_range_check.py read SESSION_DIR``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sentinel.config import CaptureConfig, IdentityConfig
from sentinel.identity.legacy_deepface import (
    FACE_ARGS,
    FULL_SCALE_FLOOR,
    CandidateDetection,
    FaceBackendError,
    FaceGeometry,
    FaceRun,
    FullScaleRow,
    LegacyDeepFaceBackend,
    _cached_detector,
    _cached_recognition_model,
    _row_box,
    candidate_rows,
    to_face_run,
)
from sentinel.identity.state import _NOTE_LABELS, IdentityPolicy, IdentityResolver
from sentinel.identity.yunet import YUNET_THRESHOLD
from sentinel.media.capture import CaptureWorker, LatestFrame
from sentinel.media.clock import NS_PER_SECOND, Clock, SystemClock
from sentinel.media.frames import FrameStamper
from sentinel.media.opencv_source import RTSP_URL_ENV, OpenCvSource, rtsp_endpoint
from sentinel.media.probe import established_connections

MAIN_URL_ENV = "SENTINEL_RTSP_MAIN_URL"
SUB, MAIN = "sub", "main"
STREAMS = (SUB, MAIN)
PRODUCTION, NATIVE = "production", "native"
CONFIGS = (PRODUCTION, NATIVE)
MARKS = ("seat", "mid", "door")
EMPTY = "empty"
# Decoding threads: the substream as the runtime's default; the main stream with two during the marks, so that a slow
# decoder cannot leave its frames behind the substream's. FFmpeg's frame threading delays a frame by threads - 1 frames.
SUB_THREADS = 1
MAIN_THREADS = 2

MAX_FRAME_AGE_NS = 2 * NS_PER_SECOND  # a pair with an older frame is not analysed (counted: stale_pair)
READY_TIMEOUT_S = 60.0  # both streams must deliver a frame within this, after the models have loaded
TAKE_S = 1.0
MAX_FACES_KEPT = 8  # per frame, in a record; the counts are of all of them
MAX_SAMPLES_PER_WINDOW = 80  # far above what the windows allow: a bound, not a target
MIN_SAMPLES = 5  # fewer in a window makes the execution incomplete; nothing is extended or repeated to reach it
EXIT = {"complete": 0, "incomplete": 3, "failed": 1}
PRIVACY_COUNT = re.compile(r"(\S+) ([a-z_]+)=(\d+)")


@dataclass(frozen=True)
class Window:
    name: str
    start_s: float
    end_s: float


# Seconds from T0, the tool's readiness (the guard's cues count from the same line). The operator is told all of this
# before the launch (D50); nothing asks for a command while it runs.
WINDOWS = (Window(EMPTY, 35.0, 65.0), Window("seat", 90.0, 130.0), Window("mid", 150.0, 190.0),
           Window("door", 210.0, 250.0))
MARKS_END_S = 250.0
CUES = (
    (0.0, "t0", "T0 - start your phone timer for 1:10 and leave the camera's view now; stay out of view until it rings"),
    (70.0, "seat", "1:10 - come back to your seat and face the camera lens; stay there until the next beep"),
    (130.0, "mid", "2:10 - stand on the middle mark and face the camera lens; stay there until the next beep"),
    (190.0, "door", "3:10 - stand on the doorway mark and face the camera lens; stay there until the next beep"),
    (250.0, "stop", "4:10 - the marks are done: come back to the Mac and type nothing; it ends by itself"),
)
# Decode cost, with no inference: (name, {stream: decoding threads}, seconds measured). Each starts with fresh
# connections and skips DECODE_SETTLE_S first.
DECODE_PHASES = (("both", {SUB: SUB_THREADS, MAIN: MAIN_THREADS}, 30.0), ("main_threads_1", {MAIN: 1}, 30.0),
                 ("main_threads_2", {MAIN: 2}, 30.0), ("sub_only", {SUB: SUB_THREADS}, 30.0))
DECODE_SETTLE_S = 5.0
DECODE_RSS_KEYS = ("VmRSS", "VmHWM")
TIMELINE_S = MARKS_END_S + sum(seconds + DECODE_SETTLE_S + 8.0 for _, _, seconds in DECODE_PHASES) + 15.0

SCOPE = ("scope: one person, one session, facing the camera on purpose at measured marks: a best case for presentation; "
         "counts are k of n samples at a mark, never a rate or an accuracy, never an F2; a negative says this "
         "configuration gave nothing there in this session, a positive validates nothing; no person detector, so the "
         "ownership rule is not exercised; no non-enrolled person; thresholds unchanged; numbers only, no image or video")


class Refused(Exception):
    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


# ---------------------------------------------------------------- the two detection configurations


class RangeBackend(LegacyDeepFaceBackend):
    """The runtime's backend with the D88 candidate's models loaded, and both configurations callable on any frame."""

    def production_faces(self, frame: Any, width: int, height: int, image: Any) -> FaceRun:
        """production: the runtime's call, unchanged, whichever detection was selected at load."""
        runtime = self._runtime
        if runtime is None:
            raise FaceBackendError("not_loaded")
        raw = runtime.DeepFace.represent(img_path=image, **FACE_ARGS)
        return to_face_run(frame, width, height, raw, [_area_geometry(item) for item in raw])

    def native_faces(self, frame: Any, width: int, height: int, image: Any) -> tuple[FaceRun, list[FullScaleRow]]:
        """native: _candidate_faces' steps on a frame of any size (D88 itself refuses a frame over 640 pixels and stays
        so); the same detector and model, thresholds and cut. Also returns every row under the threshold, which is
        never embedded. A FaceBackendError label on any failure; never a partial result, never a model built."""
        runtime = self._runtime
        if runtime is None or self._candidate is None:
            raise FaceBackendError("not_loaded")
        detector, model = self._candidate
        if _cached_detector(runtime) is not detector:
            raise FaceBackendError("detector_not_cached")
        if _cached_recognition_model(runtime) is not model:
            raise FaceBackendError("model_not_cached")
        started = time.perf_counter()
        try:
            detector.setInputSize((width, height))
            detector.setScoreThreshold(FULL_SCALE_FLOOR)
            _, found = detector.detect(image)
        except Exception as exc:  # noqa: BLE001 - a fixed label
            raise FaceBackendError("candidate_failed", type(exc).__name__) from None
        rows = candidate_rows(found)
        detect_ms = (time.perf_counter() - started) * 1000
        kept = [row for row in rows if row.score >= YUNET_THRESHOLD]
        under = [row for row in rows if row.score < YUNET_THRESHOLD]
        best = max(((row.score, box) for row in under for box in (_row_box(row, width, height),) if box is not None),
                   key=lambda c: c[0], default=None)
        items: list[dict[str, Any]] = []
        embed_ms = None
        if kept:
            started = time.perf_counter()
            items = self._embed_candidates(runtime, width, height, image, kept)
            embed_ms = (time.perf_counter() - started) * 1000
        run = to_face_run(frame, width, height, items, [item.pop("geometry") for item in items])
        return replace(run, detection=CandidateDetection(best, detect_ms, embed_ms)), under


def _area_geometry(item: Any) -> FaceGeometry | None:
    """DeepFace's reported eyes for one represent() item, when it gives both."""
    area = item.get("facial_area") if isinstance(item, Mapping) else None
    if not isinstance(area, Mapping):
        return None
    left, right = area.get("left_eye"), area.get("right_eye")
    try:
        return FaceGeometry(None, (float(right[0]), float(right[1])), (float(left[0]), float(left[1])))
    except (TypeError, ValueError, IndexError):
        return None


def eye_spacing_px(geometry: FaceGeometry | None) -> float | None:
    if geometry is None or geometry.left_eye is None or geometry.right_eye is None:
        return None
    return round(math.dist(geometry.left_eye, geometry.right_eye), 1)


# ---------------------------------------------------------------- one sample


def policy_from(identity: IdentityConfig) -> IdentityPolicy:
    """The runtime's vote settings (runtime.py builds them the same way); only these three decide a vote."""
    return IdentityPolicy(match_threshold=identity.match_threshold, margin=identity.margin,
                          min_quality=identity.min_quality)


def face_record(face: Any, geometry: FaceGeometry | None, width: int, height: int,
                resolver: IdentityResolver) -> dict[str, Any]:
    """One kept face: its size in this stream's pixels, its place, and the vote's fixed label. Never the identity ID or
    the embedding; the similarity is for the private result file."""
    box = face.box
    vote, why, similarity, _ = resolver._vote(face)  # the runtime's rule, read only: nothing is counted or stored
    del vote
    return {"w_px": round((box.x2 - box.x1) * width, 1), "h_px": round((box.y2 - box.y1) * height, 1),
            "cx": round((box.x1 + box.x2) / 2, 4), "cy": round((box.y1 + box.y2) / 2, 4),
            "score": round(float(face.quality), 3), "eye_px": eye_spacing_px(geometry),
            "vote": _NOTE_LABELS[why], "similarity": None if similarity is None else round(similarity, 3)}


def analyse(backend: RangeBackend, resolver: IdentityResolver, stream: str, config: str, captured: Any,
            perf_counter: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """One frame through one configuration: a record of numbers and fixed labels."""
    frame = captured.frame
    width, height = frame.native_width, frame.native_height
    record: dict[str, Any] = {"stream": stream, "config": config}
    started = perf_counter()
    under: list[FullScaleRow] | None = None
    try:
        if config == PRODUCTION:
            run = backend.production_faces(frame.key, width, height, captured.image)
        else:
            run, under = backend.native_faces(frame.key, width, height, captured.image)
    except FaceBackendError as exc:
        return {**record, "error": exc.label, "elapsed_ms": round((perf_counter() - started) * 1000, 1)}
    except Exception as exc:  # noqa: BLE001 - a provider's message can name a file: its class only
        return {**record, "error": f"backend:{type(exc).__name__}", "elapsed_ms": round((perf_counter() - started) * 1000, 1)}
    record["elapsed_ms"] = round((perf_counter() - started) * 1000, 1)
    geometry: Sequence[FaceGeometry | None] = run.geometry or (None,) * len(run.faces)
    record["faces_n"] = len(run.faces)
    record["faces"] = [face_record(face, shape, width, height, resolver)
                       for face, shape in list(zip(run.faces, geometry, strict=True))[:MAX_FACES_KEPT]]
    if under is not None:
        ordered = sorted(under, key=lambda row: -row.score)
        record["under_n"] = len(ordered)
        record["under"] = [{"w_px": round(row.w, 1), "h_px": round(row.h, 1), "score": round(row.score, 3)}
                           for row in ordered[:MAX_FACES_KEPT]]
    if run.detection is not None:
        record["detect_ms"] = round(run.detection.detect_ms, 1)
        record["embed_ms"] = None if run.detection.embed_ms is None else round(run.detection.embed_ms, 1)
    return record


def frame_facts(captured: Any, now_ns: int) -> dict[str, Any]:
    frame = captured.frame
    return {"epoch": frame.stream_epoch, "seq": frame.frame_seq, "ingest_mono_ns": frame.ingest_mono_ns,
            "source_pts": frame.source_pts, "width": frame.native_width, "height": frame.native_height,
            "age_ms": round((now_ns - frame.ingest_mono_ns) / 1e6, 1)}


def affine_facts(matrix: Sequence[Sequence[float]], sub_size: tuple[int, int],
                 main_size: tuple[int, int]) -> dict[str, float]:
    """A substream-to-main-stream affine matrix as numbers a person can read: the two scales, the shear terms and where
    the substream's frame lands in the main stream's, as fractions of its width and height."""
    (a, b, tx), (c, d, ty) = matrix
    sub_w, sub_h = sub_size
    main_w, main_h = main_size
    return {"scale_x": round(a, 3), "scale_y": round(d, 3), "shear_xy": round(b, 3), "shear_yx": round(c, 3),
            "tx_px": round(tx, 1), "ty_px": round(ty, 1), "left": round(tx / main_w, 3),
            "right": round((tx + a * sub_w) / main_w, 3), "top": round(ty / main_h, 3),
            "bottom": round((ty + d * sub_h) / main_h, 3)}


def fov_estimate(cv2: Any, numpy: Any, sub: Any, main: Any) -> dict[str, Any]:
    """The substream-to-main-stream mapping from one pair of frames: ORB features, cross-checked matches, a RANSAC
    affine fit. The main stream is first scaled to the substream's width, because ORB's pyramid does not reach their
    ratio; the fit is made there and reported in the main stream's pixels. Numbers and fixed labels only; nothing is
    kept."""
    sub_h, sub_w = sub.shape[:2]
    main_h, main_w = main.shape[:2]
    scale = sub_w / main_w
    small = cv2.resize(main, (sub_w, round(main_h * scale)), interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(2000)
    points_sub, described_sub = orb.detectAndCompute(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY), None)
    points_main, described_main = orb.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
    if described_sub is None or described_main is None:
        return {"status": "no_features"}
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(described_sub, described_main)
    if len(matches) < 12:
        return {"status": "too_few_matches", "matches": len(matches)}
    source = numpy.float32([points_sub[m.queryIdx].pt for m in matches])
    target = numpy.float32([points_main[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffine2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None or inliers is None:
        return {"status": "no_fit", "matches": len(matches)}
    return {"status": "fitted", "matches": len(matches), "inliers": int(inliers.sum()),
            **affine_facts((matrix / scale).tolist(), (sub_w, sub_h), (main_w, main_h))}


# ---------------------------------------------------------------- the two live streams


class LiveStreams:
    """Each stream's production capture worker and one-frame hand-off: a slow consumer skips frames, so at most one
    decoded image of each stream waits, whatever the inference takes. URLs never appear in anything it returns."""

    def __init__(self, urls: Mapping[str, str], clock: Clock, *,
                 source: Callable[[str, CaptureConfig], Any] | None = None,
                 proc_net: Path = Path("/proc/net")) -> None:
        self._urls, self._clock, self._proc_net = dict(urls), clock, proc_net
        self._source = source or (lambda url, config: OpenCvSource(url, config))
        self._running: dict[str, tuple[CaptureWorker, LatestFrame, Any]] = {}
        self._stampers: dict[str, FrameStamper] = {}

    def start(self, stream: str, threads: int) -> None:
        if stream in self._running:
            raise RuntimeError("already_running")
        config = CaptureConfig(decode_threads=threads)
        # one stamper per stream for the whole run: a restart is the next epoch of the same ingest run
        stamper = self._stampers.setdefault(stream, FrameStamper(f"cam-{stream}", self._clock))
        slot = LatestFrame()
        source = self._source(self._urls[stream], config)
        worker = CaptureWorker(source, stamper, slot, config, name=f"sentinel-range-{stream}")
        worker.start()
        self._running[stream] = (worker, slot, source)

    def stop(self, stream: str) -> bool:
        worker, slot, _ = self._running.pop(stream)
        stopped = worker.stop(CaptureConfig().open_timeout_s + CaptureConfig().read_timeout_s + 1.0)
        slot.close()
        return stopped

    def stop_all(self) -> bool:
        return all([self.stop(stream) for stream in list(self._running)])

    def take(self, stream: str, timeout_s: float) -> Any:
        return self._running[stream][1].take(timeout_s)

    def counts(self, stream: str) -> dict[str, Any]:
        worker, slot, _ = self._running[stream]
        status = worker.status()
        return {"hand_off": slot.counts(), "state": status.state.value, "frames": status.frames,
                "connects": status.connects, "open_failures": status.open_failures, "stream_ends": status.stream_ends,
                "problem": status.problem}

    def connections(self) -> int | None:
        """This host's established connections to the camera (the two URLs name one endpoint, or the count is None)."""
        endpoints = {rtsp_endpoint(url) for url in self._urls.values()}
        if len(endpoints) != 1:
            return None
        return established_connections(next(iter(endpoints)), self._proc_net)


# ---------------------------------------------------------------- the session


def proc_memory(path: Path = Path("/proc/self/status")) -> dict[str, int]:
    """This process's resident memory in bytes (fixed keys only)."""
    found: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            if key in DECODE_RSS_KEYS:
                found[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {}
    return found


@dataclass
class Session:
    """Everything the timeline needs, injected: the streams, the backend, the clock and how to wait."""

    streams: Any
    backend: Any
    resolver: IdentityResolver
    clock: Clock
    sleep: Callable[[float], None]
    write: Callable[[dict[str, Any]], None]
    fov: Callable[[Any, Any], dict[str, Any]]
    process_time: Callable[[], float] = time.process_time
    memory: Callable[[], dict[str, int]] = proc_memory


def take_pair(session: Session, counters: Counter[str]) -> dict[str, Any] | None:
    """One frame of each stream, newest first; None (and why, counted) when a stream gave none or a frame is stale."""
    pair = {stream: session.streams.take(stream, TAKE_S) for stream in STREAMS}
    if any(captured is None for captured in pair.values()):
        counters["no_frame"] += 1
        return None
    now = session.clock.monotonic_ns()
    if any(now - captured.frame.ingest_mono_ns > MAX_FRAME_AGE_NS for captured in pair.values()):
        counters["stale_pair"] += 1
        return None
    return pair


def run_sample(session: Session, window: str, t_s: float, pair: Mapping[str, Any],
               counters: Counter[str]) -> dict[str, Any]:
    """One frame pair through both configurations, and on an empty-room pair the two streams' relation. The frames are
    not kept: the sample holds numbers and fixed labels."""
    now = session.clock.monotonic_ns()
    sample: dict[str, Any] = {
        "window": window, "t_s": round(t_s, 3),
        "frames": {stream: frame_facts(pair[stream], now) for stream in STREAMS},
        "pair_skew_ms": round(abs(pair[SUB].frame.ingest_mono_ns - pair[MAIN].frame.ingest_mono_ns) / 1e6, 1),
        "results": [analyse(session.backend, session.resolver, stream, config, pair[stream])
                    for stream in STREAMS for config in CONFIGS],
    }
    counters["samples"] += 1
    errors = [r["error"] for r in sample["results"] if "error" in r]
    if errors:
        counters["sample_with_error"] += 1
    if window == EMPTY:
        try:
            sample["fov"] = session.fov(pair[SUB].image, pair[MAIN].image)
        except Exception as exc:  # noqa: BLE001 - a fixed label: the estimate never ends the session
            sample["fov"] = {"status": f"failed:{type(exc).__name__}"}
    return sample


def run_window(session: Session, window: Window, t0_ns: int, result: dict[str, Any], counters: Counter[str]) -> None:
    """Samples, one after another, each starting inside the window. Waits for its start; nothing runs between windows
    but the capture workers."""
    now_s = (session.clock.monotonic_ns() - t0_ns) / NS_PER_SECOND
    if now_s < window.start_s:
        session.sleep(window.start_s - now_s)
    done = 0
    while done < MAX_SAMPLES_PER_WINDOW:
        t_s = (session.clock.monotonic_ns() - t0_ns) / NS_PER_SECOND
        if t_s >= window.end_s:
            break
        pair = take_pair(session, counters)
        if pair is None:
            continue
        result["samples"].append(run_sample(session, window.name, t_s, pair, counters))
        done += 1
        del pair
    session.write(result)


def decode_phase(session: Session, name: str, threads: Mapping[str, int], seconds: float) -> dict[str, Any]:
    """Fresh connections and no inference: every frame the hand-off gives is timed, its frame dropped at once. The
    process's CPU covers FFmpeg's threads; the loaded models are idle. ``pts_span_s`` against ``ingest_span_s`` shows
    whether decoding kept the stream's pace (PTS is advisory, D24: a stream without usable PTS shows null)."""
    for stream, count in threads.items():
        session.streams.start(stream, count)
    try:
        deadline = session.clock.monotonic_ns() + round(DECODE_SETTLE_S * NS_PER_SECOND)
        while session.clock.monotonic_ns() < deadline:
            for stream in threads:
                session.streams.take(stream, 0.05)
        started_ns = session.clock.monotonic_ns()
        cpu_before = session.process_time()
        before = {stream: session.streams.counts(stream) for stream in threads}
        seen: dict[str, list[tuple[int, int | None]]] = {stream: [] for stream in threads}
        end_ns = started_ns + round(seconds * NS_PER_SECOND)
        connections, counted = None, False
        while session.clock.monotonic_ns() < end_ns:
            for stream in threads:
                captured = session.streams.take(stream, 0.02)
                if captured is not None:
                    seen[stream].append((captured.frame.ingest_mono_ns, captured.frame.source_pts))
            if not counted and session.clock.monotonic_ns() >= started_ns + NS_PER_SECOND:
                connections, counted = session.streams.connections(), True
        wall_s = (session.clock.monotonic_ns() - started_ns) / NS_PER_SECOND
        cpu_s = session.process_time() - cpu_before
        after = {stream: session.streams.counts(stream) for stream in threads}
    finally:
        for stream in threads:
            session.streams.stop(stream)
    return {"name": name, "threads": dict(threads), "wall_s": round(wall_s, 3), "cpu_s": round(cpu_s, 3),
            "cpu_cores_used": round(cpu_s / wall_s, 3) if wall_s > 0 else None, "connections": connections,
            "memory": session.memory(), "streams": {stream: stream_pace(seen[stream], before[stream], after[stream])
                                                    for stream in threads}}


def stream_pace(seen: Sequence[tuple[int, int | None]], before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Frames decoded and handed over in a phase, the frame rate by ingest time, and the PTS span beside it."""
    decoded = None
    if isinstance(before.get("frames"), int) and isinstance(after.get("frames"), int):
        decoded = after["frames"] - before["frames"]
    facts: dict[str, Any] = {"decoded": decoded, "seen": len(seen), "stream_ends": after.get("stream_ends"),
                             "problem": after.get("problem")}
    if len(seen) >= 2:
        ingest = [t for t, _ in seen]
        steps = sorted(b - a for a, b in zip(ingest, ingest[1:], strict=False))
        span = (ingest[-1] - ingest[0]) / NS_PER_SECOND
        facts["ingest_span_s"] = round(span, 3)
        facts["fps_seen"] = round((len(seen) - 1) / span, 2) if span > 0 else None
        facts["step_ms"] = {"p50": round(steps[len(steps) // 2] / 1e6, 1),
                            "p95": round(steps[min(len(steps) - 1, math.ceil(len(steps) * 0.95) - 1)] / 1e6, 1),
                            "max": round(steps[-1] / 1e6, 1)}
        pts = [p for _, p in seen if p is not None]
        facts["pts_span_s"] = round((pts[-1] - pts[0]) / 1e6, 3) if len(pts) >= 2 else None
    return facts


def wait_for_streams(session: Session, timeout_s: float) -> dict[str, dict[str, Any]]:
    """The first frame of each stream, or a Refused label naming which stream gave none."""
    deadline = session.clock.monotonic_ns() + round(timeout_s * NS_PER_SECOND)
    first: dict[str, dict[str, Any]] = {}
    while len(first) < len(STREAMS):
        for stream in STREAMS:
            if stream in first:
                continue
            captured = session.streams.take(stream, 0.1)
            if captured is not None:
                first[stream] = {"width": captured.frame.native_width, "height": captured.frame.native_height,
                                 "boot_id": captured.frame.boot_id, "run_id": captured.frame.run_id,
                                 "epoch": captured.frame.stream_epoch}
        if len(first) < len(STREAMS) and session.clock.monotonic_ns() >= deadline:
            missing = next(stream for stream in STREAMS if stream not in first)
            raise Refused(f"stream_not_ready:{missing}:{session.streams.counts(missing)['problem']}")
    return first


def timeline(session: Session, result: dict[str, Any], emit: Callable[[str], None]) -> None:
    """From both streams' first frames to the end: the readiness line (T0, the guard's cues), the four windows, then the
    decode phases. The caller stops whatever still runs."""
    counters: Counter[str] = Counter()
    result["counters"] = counters
    for stream, threads in ((SUB, SUB_THREADS), (MAIN, MAIN_THREADS)):
        session.streams.start(stream, threads)
    result["streams"] = wait_for_streams(session, READY_TIMEOUT_S)
    if result["streams"][SUB]["boot_id"] != result["streams"][MAIN]["boot_id"]:
        raise Refused("boot_id_mismatch")
    sub, main = result["streams"][SUB], result["streams"][MAIN]
    if sub["width"] * sub["height"] >= main["width"] * main["height"]:
        raise Refused("main_stream_not_larger_than_substream")  # the URLs swapped, or not the two streams this compares
    result["connections_at_ready"] = session.streams.connections()
    t0_ns = session.clock.monotonic_ns()
    result["t0_mono_ns"] = t0_ns
    result["status"] = "running"
    session.write(result)
    emit(json.dumps({"run": "starting", "startup": {"tool": "door_range_check"}}))
    for window in WINDOWS:
        run_window(session, window, t0_ns, result, counters)
    now_s = (session.clock.monotonic_ns() - t0_ns) / NS_PER_SECOND
    if now_s < MARKS_END_S:
        session.sleep(MARKS_END_S - now_s)
    result["counts_after_marks"] = {stream: session.streams.counts(stream) for stream in STREAMS}
    session.streams.stop_all()
    for name, threads, seconds in DECODE_PHASES:
        result["decode"].append(decode_phase(session, name, threads, seconds))
        session.write(result)
    result["status"] = "finished"


def verdict(result: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Execution only, never recognition: whether each window has its samples, every configuration ran and the decode
    phases finished. Nothing is extended or repeated to reach this."""
    problems: list[str] = []
    if result.get("status") != "finished":
        problems.append(f"status:{result.get('status')}")
    samples = result.get("samples") or []
    for window in WINDOWS:
        n = sum(1 for s in samples if s["window"] == window.name)
        if n < MIN_SAMPLES:
            problems.append(f"window_{window.name}_samples:{n}<{MIN_SAMPLES}")
    errors = Counter(r["error"] for s in samples for r in s["results"] if "error" in r)
    problems.extend(f"backend_error:{label}:{n}" for label, n in sorted(errors.items()))
    done = [phase["name"] for phase in result.get("decode") or []]
    missing = [name for name, _, _ in DECODE_PHASES if name not in done]
    if missing:
        problems.append("decode_missing:" + ",".join(missing))
    if not samples:
        return "failed", problems
    return ("incomplete" if problems else "complete"), problems


def atomic_writer(path: Path) -> Callable[[dict[str, Any]], None]:
    def write(result: dict[str, Any]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")
        os.replace(temporary, path)
    return write


def distance(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and 0.2 <= value <= 15.0):
        raise argparse.ArgumentTypeError("a measured distance in metres, 0.2 to 15")
    return value


def new_result(args: argparse.Namespace, identity: IdentityConfig, clock: Clock) -> dict[str, Any]:
    return {"check": "door_range", "scope": SCOPE, "started_utc": clock.utc_now().isoformat(), "boot_id": clock.boot_id,
            "status": "starting", "marks_m": {"seat": args.seat_m, "mid": args.mid_m, "door": args.door_m},
            "thresholds": {"yunet": YUNET_THRESHOLD, "recording_floor": FULL_SCALE_FLOOR,
                           "match": identity.match_threshold, "margin": identity.margin,
                           "min_quality": identity.min_quality},
            "windows": [{"name": w.name, "start_s": w.start_s, "end_s": w.end_s} for w in WINDOWS],
            "samples": [], "decode": [], "problems": []}


def load_recognition(identity_dir: Path, secret: Any, weights_dir: Path | None = None,
                     backend_factory: Callable[[], RangeBackend] | None = None) -> tuple[RangeBackend, Any]:
    """The working copy's gallery and both configurations' models, compared as the runtime compares them. A Refused
    label otherwise; nothing here writes to the identity directory."""
    from sentinel.demo_runtime import StartupRefused, open_gallery
    from sentinel.identity.gallery import IdentityStore
    from sentinel.identity.vault import Sealer

    try:
        document, problem = open_gallery(IdentityStore(identity_dir, Sealer()), secret)
    except StartupRefused as exc:
        raise Refused(f"gallery:{exc}") from None
    if problem is not None or document is None:
        raise Refused(f"gallery:{problem}")
    backend = RangeBackend(weights_dir) if backend_factory is None else backend_factory()
    backend.select_full_scale_detection()  # DeepFace's cached detector and model, native_faces' own
    try:
        backend.load()
    except FaceBackendError as exc:
        raise Refused(f"backend:{exc.label}") from None
    difference = backend.compatibility().first_difference(document.compatibility)
    if difference is not None:
        raise Refused(f"gallery_incompatible:{difference}")
    return backend, document


def run(args: argparse.Namespace, *, environ: Mapping[str, str] | None = None, clock: Clock | None = None,
        streams: Any = None, recognition: Callable[[], tuple[Any, Any]] | None = None,
        emit: Callable[[str], None] = lambda line: print(line, flush=True), sleep: Callable[[float], None] = time.sleep,
        fov: Callable[[Any, Any], dict[str, Any]] | None = None) -> dict[str, Any]:
    """The whole session; the result file is written after every window and however it ends."""
    environment = os.environ if environ is None else environ
    clock = clock or SystemClock()
    identity = IdentityConfig()
    result = new_result(args, identity, clock)
    write = atomic_writer(args.out)
    write(result)
    live = streams
    try:
        urls = {SUB: environment.get(RTSP_URL_ENV, ""), MAIN: environment.get(MAIN_URL_ENV, "")}
        for stream, url in urls.items():
            if not url.strip():
                raise Refused(f"url_missing:{stream}")
            try:
                rtsp_endpoint(url)
            except Exception:  # noqa: BLE001 - never the URL or its error
                raise Refused(f"url_invalid:{stream}") from None
        if urls[SUB] == urls[MAIN]:
            raise Refused("urls_identical")
        if recognition is None:
            from sentinel.identity.vault import take_secret_from_environment

            secret = take_secret_from_environment()
            if secret is None:
                raise Refused("passphrase_missing")
            backend, document = load_recognition(args.identity_dir, secret)
        else:
            backend, document = recognition()
        if live is None:
            live = LiveStreams(urls, clock)
        if fov is None:
            cv2, numpy = backend._runtime.cv2, backend._runtime.numpy
            fov = lambda sub, main: fov_estimate(cv2, numpy, sub, main)  # noqa: E731
        session = Session(live, backend, IdentityResolver(document.enrollment(), policy_from(identity)), clock, sleep,
                          write, fov)
        timeline(session, result, emit)
    except Refused as exc:
        result["status"] = f"refused:{exc.label}"
    except KeyboardInterrupt:  # the guard's stop
        result["status"] = "stopped"
    except Exception as exc:  # noqa: BLE001 - a fixed label: messages can name the camera
        result["status"] = f"failed:{type(exc).__name__}"
    finally:
        if live is not None:
            try:
                result["workers_stopped"] = live.stop_all()
            except Exception as exc:  # noqa: BLE001
                result["workers_stopped"] = f"failed:{type(exc).__name__}"
    result["finished_utc"] = clock.utc_now().isoformat()
    result["verdict"], result["problems"] = verdict(result)
    write(result)
    emit(json.dumps({"door_range_check": result["verdict"], "status": result["status"],
                     "samples": len(result["samples"])}))
    return result


# ---------------------------------------------------------------- the reading


def _p50(values: Sequence[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def cell(samples: Sequence[Mapping[str, Any]], stream: str, config: str) -> dict[str, Any]:
    """One window, stream and configuration: counts over its samples. Only frames with exactly one kept face give a size
    or a vote; a frame with more is counted apart, unresolved."""
    rows = [r for s in samples for r in s["results"] if r["stream"] == stream and r["config"] == config]
    ran = [r for r in rows if "error" not in r]
    single = [r["faces"][0] for r in ran if r.get("faces_n") == 1]
    votes = Counter(face["vote"] for face in single)
    under = [c for r in ran for c in (r.get("under") or [])]
    no_face = [r for r in ran if r.get("faces_n") == 0]
    facts: dict[str, Any] = {
        "n": len(rows), "errors": len(rows) - len(ran), "one_face": len(single), "no_face": len(no_face),
        "several_faces": sum(1 for r in ran if (r.get("faces_n") or 0) > 1),
        "votes": dict(sorted(votes.items())), "match": votes.get("match", 0),
        "w_px": _spread([face["w_px"] for face in single]), "h_px": _spread([face["h_px"] for face in single]),
        "eye_px": _spread([face["eye_px"] for face in single if face.get("eye_px") is not None]),
        "score_p50": _p50([face["score"] for face in single]), "elapsed_ms_p50": _p50([r["elapsed_ms"] for r in ran]),
    }
    if any("under" in r for r in ran):
        facts["no_face_with_candidate_under_threshold"] = sum(1 for r in no_face if r.get("under_n"))
        facts["under_w_px_p50"] = _p50([c["w_px"] for c in under])
        facts["under_score_p50"] = _p50([c["score"] for c in under])
    return facts


def _spread(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"p50": round(statistics.median(values), 1), "min": round(min(values), 1), "max": round(max(values), 1)}


def similarity_spread(samples: Sequence[Mapping[str, Any]], stream: str, config: str) -> dict[str, float] | None:
    values = [r["faces"][0]["similarity"] for s in samples for r in s["results"]
              if r["stream"] == stream and r["config"] == config and r.get("faces_n") == 1
              and r["faces"][0].get("similarity") is not None]
    return None if not values else {"p50": round(statistics.median(values), 3), "min": min(values), "max": max(values)}


def width_ratios(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Main-stream face width over substream face width, on the pairs where both native configurations kept exactly one
    face: the two streams' scale, measured on the same moment's faces."""
    ratios = []
    for sample in samples:
        found = {r["stream"]: r for r in sample["results"] if r["config"] == NATIVE and r.get("faces_n") == 1}
        if SUB in found and MAIN in found and found[SUB]["faces"][0]["w_px"] > 0:
            ratios.append(found[MAIN]["faces"][0]["w_px"] / found[SUB]["faces"][0]["w_px"])
    return None if not ratios else {"n": len(ratios), "p50": round(statistics.median(ratios), 2)}


def privacy_line(directory: Path) -> tuple[str, bool]:
    """The session's privacy counts (face_privacy.py's ``<file> <key>=<n>`` lines), summed; file names only."""
    counted, hits = 0, []
    for path in sorted(directory.glob("secret-counts-*.txt")):
        for line in path.read_text().splitlines():
            found = PRIVACY_COUNT.fullmatch(line)
            if found:
                counted += 1
                if int(found[3]):
                    hits.append(found[1])
    if not counted:
        return "MISSING privacy counts: no camera credential, passphrase or name was counted", True
    if hits:
        return f"FAIL privacy: a camera credential, the passphrase or a name was found in {sorted(set(hits))}", False
    return f"PASS privacy: no camera credential, passphrase or name in {counted} counts", True


def read_account(directory: Path) -> dict[str, str]:
    try:
        lines = (directory / "account.txt").read_text().splitlines()
    except OSError:
        return {}
    return dict(line.split("=", 1) for line in lines if "=" in line)


def reading(directory: Path, *, with_similarity: bool = False) -> tuple[list[str], int]:
    """What the session folder shows: the guard, the operator's account, then descriptive tables. It judges execution
    only; nothing here says the requirement is met."""
    lines: list[str] = []
    guard_path = directory / "result.json"
    try:
        guard = json.loads(guard_path.read_text())
    except (OSError, ValueError):
        lines.append(f"MISSING the guard's result, {guard_path.name} (nothing is judged)")
        guard = None
    if guard is not None:
        status = str(guard.get("status"))
        if status.startswith("refused:"):
            lines.append(f"REFUSED the guard: {status}, before launch: nothing was started and nothing is judged")
            return lines, 1
        gm = guard.get("guard") or {}
        word = "PASS" if status == "child_exited" else "FAIL"
        lines.append(f"{word} the guard: status {status}, trigger {gm.get('trigger')}, returncode "
                     f"{(guard.get('child') or {}).get('returncode')}; peak pressure {gm.get('peak_pressure_bytes')} B, "
                     f"lowest MemFree {gm.get('min_mem_free_bytes')} B")
    try:
        result = json.loads((directory / "door-range.json").read_text())
    except (OSError, ValueError):
        lines.append("MISSING the check's result, door-range.json (nothing is judged)")
        return lines, 1
    lines.append(SCOPE)
    marks = result.get("marks_m") or {}
    lines.append(f"marks typed before the launch (metres, camera to the face's position): {marks}")
    lines.append(f"thresholds (the runtime's, unchanged): {result.get('thresholds')}")
    account = read_account(directory)
    if not account:
        lines.append("MISSING the operator's account (account.txt): the conditions below are unconfirmed")
    else:
        lines.append(f"USER-REPORTED account: {account}")
        expected = {"left_view_for_empty": "yes", "at_marks_in_windows": "yes", "faced_camera": "yes",
                    "others_in_view": "no"}
        off = [key for key, want in expected.items() if account.get(key) != want]
        if off:
            lines.append(f"CONDITION not as planned ({', '.join(off)}): the windows concerned are read with that in mind")
    privacy, private = privacy_line(directory)
    lines.append(privacy)
    state, problems = verdict(result)
    lines.append(f"execution: {state}{' (' + '; '.join(problems) + ')' if problems else ''}; status {result.get('status')}; "
                 f"counters {dict(result.get('counters') or {})}")
    streams = result.get("streams") or {}
    lines.append("frame sizes read from the streams: " + ", ".join(
        f"{name} {facts.get('width')}x{facts.get('height')}" for name, facts in streams.items()))
    lines.append(f"this host's connections to the camera at readiness: {result.get('connections_at_ready')}")
    samples = result.get("samples") or []
    for window in (w["name"] for w in result.get("windows") or []):
        inside = [s for s in samples if s["window"] == window]
        title = "no person expected" if window == EMPTY else f"{marks.get(window)} m"
        lines.append(f"--- {window} ({title}): {len(inside)} frame pairs, pair skew p50 "
                     f"{_p50([s['pair_skew_ms'] for s in inside])} ms")
        for stream in STREAMS:
            for config in CONFIGS:
                facts = cell(inside, stream, config)
                text = (f"{stream:4} {config:10} {facts['one_face']} of {facts['n']} one face, {facts['no_face']} no face, "
                        f"{facts['several_faces']} several (unresolved), {facts['errors']} errors; votes {facts['votes']}; "
                        f"width px {facts['w_px']}, height px {facts['h_px']}, eyes px {facts['eye_px']}, "
                        f"score p50 {facts['score_p50']}, time p50 {facts['elapsed_ms_p50']} ms")
                if "under_w_px_p50" in facts:
                    text += (f"; {facts['no_face_with_candidate_under_threshold']} no-face frames had a candidate under "
                             f"{(result.get('thresholds') or {}).get('yunet')} (width p50 {facts['under_w_px_p50']} px, "
                             f"score p50 {facts['under_score_p50']}; never embedded, never a face)")
                if with_similarity:
                    text += f"; PRIVATE similarity {similarity_spread(inside, stream, config)}"
                lines.append(text)
        ratio = width_ratios(inside)
        if ratio is not None:
            lines.append(f"main-stream width / substream width, native on both: {ratio}")
    fits = [s["fov"] for s in samples if s.get("fov", {}).get("status") == "fitted"]
    others = Counter(s["fov"]["status"] for s in samples if "fov" in s and s["fov"].get("status") != "fitted")
    if fits:
        keys = ("scale_x", "scale_y", "shear_xy", "shear_yx", "left", "right", "top", "bottom", "inliers")
        lines.append(f"the two streams on empty-room pairs ({len(fits)} fitted, others {dict(others)}), medians (the "
                     "substream's frame in the main stream's, as fractions): "
                     + str({key: round(statistics.median(f[key] for f in fits), 3) for key in keys}))
    else:
        lines.append(f"the two streams on empty-room pairs: no fit ({dict(others)})")
    for phase in result.get("decode") or []:
        lines.append(f"decode {phase['name']} (no inference): threads {phase['threads']}, process CPU "
                     f"{phase['cpu_cores_used']} cores over {phase['wall_s']} s, connections {phase['connections']}, "
                     f"memory {phase['memory']}, streams {phase['streams']}")
    lines.append("CPU decoding with OpenCV's FFmpeg; NVDEC is not measured here")
    lines.append(f"door-range check: execution {state} (descriptive numbers; it validates nothing and admits nothing)")
    return lines, EXIT[state] if private else EXIT["failed"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--identity-dir", type=Path, required=True)
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--seat-m", type=distance, required=True)
    go.add_argument("--mid-m", type=distance, required=True)
    go.add_argument("--door-m", type=distance, required=True)
    look = sub.add_parser("read")
    look.add_argument("directory", type=Path)
    look.add_argument("--with-similarity", action="store_true", help="PRIVATE values: never paste them into Git")
    sub.add_parser("cues", help="the guard's --cue-at values, one per line")
    args = parser.parse_args(argv)
    if args.command == "cues":
        for at, name, text in CUES:
            print(f"{at:g}:{name}:{text}")
        return 0
    if args.command == "read":
        lines, code = reading(args.directory, with_similarity=args.with_similarity)
        print("\n".join(lines))
        return code
    result = run(args)
    return EXIT[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
