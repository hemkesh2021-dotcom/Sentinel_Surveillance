"""Route B's offline cost check (V2; session 116, part 17): what the main_single profile's own steps cost on this device's CPU,
measured on synthetic pictures, with the process's memory by phase. No camera, no stream, no GPU, no face and no embedding.

One process, the runtime's own functions, the device environment's NumPy and OpenCV, in this order:

- **baseline:** the process after its imports (Python, NumPy, OpenCV and the runtime's portable modules), before anything is made.
- **inputs:** synthetic 2304x1296 BGR pictures made here (seeded noise and smooth gradients): scaffolding, counted apart, never a
  runtime cost. Nothing is read from a camera, a stream or an image file, and nothing but numbers is written.
- **resize:** ``media.opencv_source.area_resize`` to the 640x480 working copy, which the capture thread makes for every decoded
  picture under main_single.
- **windows:** ``identity.window_stage.prepare`` for fixed layouts of people: the window copies an offer makes.
- **model load:** the pinned YuNet weights, their SHA-256 checked first as the face backend checks them, built as DeepFace 0.0.99's
  YuNet client builds its detector (``cv2.FaceDetectorYN_create(weights, "", (0, 0))``). Then its first detection, timed apart.
- **steady state:** each layout's windows detected as ``LegacyDeepFaceBackend.window_faces`` detects them (``setInputSize`` to the
  window's input, the recording floor, ``detect``), tick after tick, the first tick of each layout timed apart.

**Memory, by phase.** Before each phase the kernel's peak resident size is reset to the current size (``/proc/self/clear_refs``,
value 5, this process only), so each phase's own peak is read from ``VmHWM`` afterwards, beside ``VmRSS``, the anonymous and file
parts and the proportional set size (``smaps_rollup``). The model's cost is the difference across its load and first detection;
the steady cost is each steady phase's peak over that phase's own start, which already holds the tool's window copies
for every synthetic picture (``held_copy_bytes``, scaffolding: the runtime holds at most two jobs' windows). These are this process's numbers: they are not
the runtime's, whose process already holds TensorFlow, Facenet512, the tracker's CUDA context and the decoder.

**CPU.** Wall time (``perf_counter_ns``) per call and the process's CPU time per phase (``process_time_ns``), so ``cores`` is CPU
over wall: OpenCV parallelizes both the resize and YuNet over its threads (``cv2.getNumThreads``, recorded, left at its default as
the runtime leaves it).

**Memory policy, as validation runs have it.** Before NumPy or OpenCV is imported, THP is disabled for this process alone and
verified (``memory_policy.disable_thp_for_this_process``, D58), as ``sentinel run --workload-thp-disable`` does; the run is refused
when that cannot be verified. Nothing system-wide changes.

**Under the guard.** The readiness line (``{"run": "starting", ...}``) is printed once the imports and the THP setting are done;
``guard-duration`` gives the guard's bound. A stop (the guard's SIGINT) keeps every phase already measured.

``--stand-in`` replaces YuNet by a stand-in that detects nothing at once and checks no weights: the package's dry run, never a
measurement of YuNet. The result says so, and the reading prints it first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sentinel.contracts import NormalizedBox, PixelFormat, ResizeTransform, TrackObservation, TrackStatus
from sentinel.identity.legacy_deepface import DETECTOR_WEIGHTS, FULL_SCALE_FLOOR, default_weights_dir
from sentinel.identity.window_stage import WindowSettings, prepare
from sentinel.media.clock import SystemClock
from sentinel.media.frames import FrameStamper
from sentinel.memory_policy import disable_thp_for_this_process
from stream_timing_check import read_sensors, sensor_summary

CHECK = "route_b_cost"
WIDTH, HEIGHT = 2304, 1296  # the main stream's decoded size (dr-20261105T085421Z)
WORKING = (640, 480)
SQUEEZE = ResizeTransform(output_width=WORKING[0], output_height=WORKING[1], scale_x=WORKING[0] / WIDTH,
                          scale_y=WORKING[1] / HEIGHT)
PICTURES = 4  # distinct synthetic pictures, used in turn
SEED = 20261106
RESIZE_CALLS = 300
WARMUP_CALLS = 10
WINDOW_CALLS = 100
TICKS = 150
FACE_INTERVAL_S = 1.0  # the face stage's tick (identity.face_interval_s)
MAIN_FPS = 20.0  # the main stream's nominal rate (dr-20261105T085421Z), for the derived per-second figures only
# People as the tracker reports them (normalized): the layouts the windows are measured for.
LAYOUTS: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "one_far": ((0.45, 0.30, 0.50, 0.62),),  # one window of 640x640 at scale 1
    "two_far": ((0.20, 0.30, 0.25, 0.62), (0.70, 0.30, 0.75, 0.62)),
    "four_far": ((0.05, 0.30, 0.10, 0.62), (0.30, 0.30, 0.35, 0.62), (0.55, 0.30, 0.60, 0.62), (0.80, 0.30, 0.85, 0.62)),
    "near_and_far": ((0.02, 0.05, 0.40, 1.0), (0.45, 0.30, 0.50, 0.62)),  # a scaled window beside a 640x640 one
}
CLEAR_REFS = Path("/proc/self/clear_refs")
STATUS = Path("/proc/self/status")
ROLLUP = Path("/proc/self/smaps_rollup")
MEMINFO = Path("/proc/meminfo")
STATUS_FIELDS = ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem")
ROLLUP_FIELDS = ("Pss", "Pss_Anon", "Pss_File")
EXIT = {"complete": 0, "incomplete": 3, "failed": 1}
GUARD_DURATION_S = 600.0  # the guard's bound; the defaults finish in a few minutes on this device (an estimate until measured)
SCOPE = ("scope: this process on this device's CPU, synthetic pictures, no camera, no face, no embedding, no other workload of the "
         "runtime; numbers for planning, never the runtime's cost")
LIMITS = (
    "LIMIT: no face is in these pictures. Detection time is YuNet's network on the window's input size, which does not depend on "
    "the content, plus its post-processing, which grows with candidates: real faces and textured backgrounds give more candidates "
    "than noise. The candidates seen here are counted.",
    "LIMIT: no embedding runs. A kept face adds DeepFace's cut, alignment and Facenet512 per face (about 785-810 ms per call on the "
    "substream path, dr-20261105T085421Z); a tick with no face adds nothing, which is route B's saving over the fallback embedding.",
    "LIMIT: one process alone. In the runtime the same CPU also decodes the main stream (0.734 cores with 2 threads, decoding only, "
    "dr-), runs the face embedding, the tracker's host side and the scene and chat clients, and OpenCV's threads share it; nothing "
    "here measures that contention, the GPU or memory pressure.",
    "LIMIT: memory is this process's, phase by phase. The runtime's process already holds TensorFlow, Facenet512, the decoder and "
    "the tracker's CUDA context; YuNet's own load and the window copies are what main_single adds, but how much of that the "
    "allocator returns, and the combined peak, only a combined run measures.",
    "LIMIT: no real HEVC decode, no real frames, no camera timing, and minutes, not an hour: no thermal steady state.",
    "LIMIT: recognition, person detection and range say nothing here; neither does any timing of a face's detection score.",
)


class Refused(Exception):
    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


# ---------------------------------------------------------------- memory and time


def _kib_fields(text: str, fields: Sequence[str]) -> dict[str, int | None]:
    values: dict[str, int | None] = {name: None for name in fields}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in values:
            parts = rest.split()
            if len(parts) == 2 and parts[1] == "kB" and parts[0].isdigit():
                values[name] = int(parts[0]) * 1024
    return values


def process_memory(status: Path = STATUS, rollup: Path = ROLLUP) -> dict[str, int | None]:
    """This process's memory in bytes; a field that cannot be read is None."""
    try:
        values = _kib_fields(status.read_text(), STATUS_FIELDS)
    except OSError:
        values = dict.fromkeys(STATUS_FIELDS)
    try:
        values.update(_kib_fields(rollup.read_text(), ROLLUP_FIELDS))
    except OSError:
        values.update(dict.fromkeys(ROLLUP_FIELDS))
    return values


def system_memory(meminfo: Path = MEMINFO) -> dict[str, int | None]:
    try:
        return _kib_fields(meminfo.read_text(), ("MemAvailable", "MemFree"))
    except OSError:
        return {"MemAvailable": None, "MemFree": None}


def reset_peak(clear_refs: Path = CLEAR_REFS) -> bool:
    """Reset this process's peak resident size to its current one (clear_refs value 5); False when not permitted."""
    try:
        clear_refs.write_text("5")
        return True
    except OSError:
        return False


def summary_ms(samples_ns: Sequence[int]) -> dict[str, Any]:
    if not samples_ns:
        return {"n": 0}
    ordered = sorted(samples_ns)

    def rank(q: float) -> float:
        return round(ordered[max(0, math.ceil(q * len(ordered)) - 1)] / 1e6, 3)

    return {"n": len(ordered), "mean_ms": round(sum(ordered) / len(ordered) / 1e6, 3), "min_ms": round(ordered[0] / 1e6, 3),
            "p50_ms": rank(0.50), "p95_ms": rank(0.95), "p99_ms": rank(0.99), "max_ms": round(ordered[-1] / 1e6, 3)}


class Phase:
    """One phase: its memory before, its own peak and after, its wall and CPU time."""

    def __init__(self, name: str, pieces: Mapping[str, Any], phases: list[dict[str, Any]]) -> None:
        self.name, self.pieces, self.phases = name, pieces, phases

    def __enter__(self) -> Phase:
        self.record: dict[str, Any] = {"phase": self.name, "memory_before": self.pieces["memory"]()}
        self.record["peak_reset"] = self.pieces["reset_peak"]()
        self.wall0, self.cpu0 = self.pieces["perf"](), self.pieces["cpu"]()
        return self

    def __exit__(self, kind: Any, value: Any, trace: Any) -> None:
        wall, cpu = self.pieces["perf"]() - self.wall0, self.pieces["cpu"]() - self.cpu0
        after = self.pieces["memory"]()
        self.record.update(memory_after=after, peak_bytes=after.get("VmHWM"), wall_ms=round(wall / 1e6, 3),
                           cpu_ms=round(cpu / 1e6, 3), cores=round(cpu / wall, 2) if wall > 0 else None)
        if kind is not None:
            self.record["ended"] = f"error:{kind.__name__}"
        self.phases.append(self.record)  # also when the phase raised: what it measured is kept


# ---------------------------------------------------------------- the pieces measured


def synthetic_pictures(numpy: Any, count: int = PICTURES, seed: int = SEED) -> list[Any]:
    """Seeded noise and smooth gradients, 2304x1296 BGR: no face, no file, no camera."""
    rng = numpy.random.default_rng(seed)
    pictures = []
    for index in range(count):
        if index % 2 == 0:
            pictures.append(rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=numpy.uint8))
        else:
            x = numpy.linspace(0, 200, WIDTH, dtype=numpy.float32)[None, :, None]
            y = numpy.linspace(0, 55, HEIGHT, dtype=numpy.float32)[:, None, None]
            grain = rng.integers(0, 16, size=(HEIGHT, WIDTH, 3)).astype(numpy.float32)
            pictures.append((x + y + grain).astype(numpy.uint8))
    return pictures


def layout_people(clock: Any, layout: Sequence[tuple[float, float, float, float]]) -> tuple[Any, tuple[TrackObservation, ...]]:
    stamper = FrameStamper("cost-check", clock)
    stamper.connect()
    frame = stamper.stamp(native_width=WIDTH, native_height=HEIGHT, pixel_format=PixelFormat.BGR, resize=SQUEEZE)
    persons = tuple(TrackObservation.detected(frame, track_id=i + 1, box=NormalizedBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]),
                                              confidence=0.9, status=TrackStatus.CONFIRMED) for i, b in enumerate(layout))
    return frame, persons


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class StandIn:
    """The dry run's detector: YuNet's three calls, nothing detected, no weights."""

    def __init__(self) -> None:
        self.calls = 0

    def setInputSize(self, size: Any) -> None:  # noqa: N802 - OpenCV's name
        pass

    def setScoreThreshold(self, value: float) -> None:  # noqa: N802
        pass

    def detect(self, image: Any) -> tuple[int, Any]:
        self.calls += 1
        return 1, None


def checked_detector(weights: Path, build: Callable[[Path], Any], expected: str = DETECTOR_WEIGHTS[1]) -> Any:
    """The detector built from ``weights`` only after their SHA-256 is the pinned one, as the face backend checks it."""
    try:
        digest = file_sha256(weights)
    except OSError:
        raise Refused("weights_unreadable") from None
    if digest != expected:
        raise Refused("weights_hash_mismatch")
    return build(weights)


def yunet(cv2: Any, weights: Path) -> Any:
    """The detector exactly as DeepFace 0.0.99's YuNet client builds it."""
    return cv2.FaceDetectorYN_create(str(weights), "", (0, 0))


def detect_windows(detector: Any, images: Any, perf: Callable[[], int]) -> tuple[list[tuple[tuple[int, int], int]], int]:
    """One tick's detection, as window_faces runs it: per window (its input size, its time in ns); and the candidates seen."""
    timed, candidates = [], 0
    for window, image in zip(images.windows, images.inputs, strict=True):
        size = window.input_size
        started = perf()
        detector.setInputSize(size)
        detector.setScoreThreshold(FULL_SCALE_FLOOR)
        _, found = detector.detect(image)
        timed.append((size, perf() - started))
        candidates += 0 if found is None else len(found)
    return timed, candidates


# ---------------------------------------------------------------- the run


def measure(args: argparse.Namespace, pieces: Mapping[str, Any], result: dict[str, Any], write: Callable[[], None]) -> None:
    numpy, cv2, resize, perf = pieces["numpy"], pieces["cv2"], pieces["resize"], pieces["perf"]
    phases = result["phases"]
    with Phase("baseline", pieces, phases) as phase:
        pass
    with Phase("inputs", pieces, phases) as phase:
        pictures = pieces["pictures"](numpy, args.pictures)
        phase.record["bytes_made"] = sum(int(p.nbytes) for p in pictures)
    write()
    with Phase("resize", pieces, phases) as phase:
        for i in range(WARMUP_CALLS):
            resize(pictures[i % len(pictures)], *WORKING)
        times = []
        for i in range(args.resize_calls):
            started = perf()
            copy = resize(pictures[i % len(pictures)], *WORKING)
            times.append(perf() - started)
            if tuple(copy.shape[:2]) != (WORKING[1], WORKING[0]):
                raise Refused("working_copy_size")
            del copy
        phase.record["calls"] = summary_ms(times)
    write()
    settings = WindowSettings()
    clock = pieces["clock"]
    plans = {name: layout_people(clock, layout) for name, layout in LAYOUTS.items()}
    with Phase("windows", pieces, phases) as phase:
        per_layout = {}
        for name, (frame, persons) in plans.items():
            times = []
            images = None
            for i in range(WARMUP_CALLS + args.window_calls):
                started = perf()
                images = prepare(frame, persons, pictures[i % len(pictures)], resize=resize, settings=settings)
                if i >= WARMUP_CALLS:
                    times.append(perf() - started)
            assert images is not None
            per_layout[name] = {"windows": [list(w.input_size) for w in images.windows],
                                "scales": [round(w.scale, 4) for w in images.windows], "uncovered": len(images.uncovered),
                                "copy_bytes": images.nbytes, "calls": summary_ms(times)}
        phase.record["layouts"] = per_layout
    write()
    first_images = prepare(*plans["one_far"], pictures[0], resize=resize, settings=settings)  # before: never the model's
    with Phase("model_load", pieces, phases) as phase:
        detector = pieces["detector"]()
    write()
    with Phase("first_detection", pieces, phases) as phase:
        timed, candidates = detect_windows(detector, first_images, perf)
        phase.record.update(input=list(timed[0][0]), ms=round(timed[0][1] / 1e6, 3), candidates=candidates)
    write()
    images: list[Any] = []
    for name, (frame, persons) in plans.items():
        images = []  # the previous layout's copies released first
        images = [prepare(frame, persons, picture, resize=resize, settings=settings) for picture in pictures]
        with Phase(f"steady:{name}", pieces, phases) as phase:
            # the tool's own copies, one window set per synthetic picture: scaffolding the phase starts with, counted apart
            phase.record["held_copy_bytes"] = sum(i.nbytes for i in images)
            first_tick, candidates_seen = None, 0
            ticks, by_size = [], {}
            for tick in range(args.ticks + 1):
                started = perf()
                timed, candidates = detect_windows(detector, images[tick % len(images)], perf)
                elapsed = perf() - started
                candidates_seen += candidates
                if tick == 0:
                    first_tick = elapsed
                    continue
                ticks.append(elapsed)
                for size, ns in timed:
                    by_size.setdefault(f"{size[0]}x{size[1]}", []).append(ns)
                if tick % 25 == 0:
                    result["sensor_samples"].append(pieces["sensors"]())
            phase.record.update(windows=[list(w.input_size) for w in images[0].windows], ticks=summary_ms(ticks),
                                first_tick_ms=None if first_tick is None else round(first_tick / 1e6, 3),
                                per_window=dict(sorted((k, summary_ms(v)) for k, v in by_size.items())),
                                candidates=candidates_seen)
        write()  # after the phase: its record is appended, and the write is outside its time and peak
    del images, first_images, pictures


def default_pieces(stand_in: bool, weights: Path) -> dict[str, Any]:
    import cv2
    import numpy

    from sentinel.media.opencv_source import area_resize

    def detector() -> Any:
        return StandIn() if stand_in else checked_detector(weights, lambda path: yunet(cv2, path))

    return {"numpy": numpy, "cv2": cv2, "resize": area_resize, "perf": time.perf_counter_ns, "cpu": time.process_time_ns,
            "memory": process_memory, "reset_peak": reset_peak, "pictures": synthetic_pictures, "clock": SystemClock(),
            "detector": detector, "sensors": read_sensors, "system": system_memory,
            "versions": {"numpy": numpy.__version__, "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads()}}


def atomic_write(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace, pieces: Mapping[str, Any] | None = None,
        thp: Callable[[], Mapping[str, Any]] = disable_thp_for_this_process,
        emit: Callable[[str], None] = lambda line: print(line, flush=True)) -> dict[str, Any]:
    """The whole check; the result file is written after every phase and however it ends."""
    weights = args.weights or (default_weights_dir() / DETECTOR_WEIGHTS[0])
    result: dict[str, Any] = {"check": CHECK, "scope": SCOPE, "stand_in": bool(args.stand_in), "status": "starting",
                              "phases": [], "sensor_samples": [],
                              "settings": {"pictures": args.pictures, "picture": [WIDTH, HEIGHT], "working": list(WORKING),
                                           "resize_calls": args.resize_calls, "window_calls": args.window_calls,
                                           "ticks": args.ticks, "warmup_calls": WARMUP_CALLS, "seed": SEED,
                                           "layouts": {k: [list(b) for b in v] for k, v in LAYOUTS.items()},
                                           "window_settings": WindowSettings().numbers(),
                                           "weights": DETECTOR_WEIGHTS[0], "weights_sha256": DETECTOR_WEIGHTS[1]}}
    out = args.out
    try:
        policy = dict(thp())  # before NumPy and OpenCV, as the runtime's validation runs set it
        result["thp_disable"] = policy
        if policy.get("verified") is not True:
            raise Refused(f"thp_disable_not_verified:{policy.get('reason')}")
        pieces = pieces or default_pieces(bool(args.stand_in), weights)
        result.update(environment={"python": platform.python_version(), "machine": platform.machine(),
                                   "cpus": os.cpu_count(), "ld_preload": bool(os.environ.get("LD_PRELOAD")),
                                   "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                                   **pieces["versions"]},
                      boot_id=pieces["clock"].boot_id, started_utc=pieces["clock"].utc_now().isoformat(),
                      system_before=pieces["system"](), loadavg_before=list(os.getloadavg()))
        result["sensor_samples"].append(pieces["sensors"]())
        atomic_write(out, result)
        emit(json.dumps({"run": "starting", "startup": {"tool": "route_b_cost_check", "stand_in": bool(args.stand_in)}}))
        measure(args, pieces, result, lambda: atomic_write(out, result))
        result["status"] = "finished"
    except Refused as exc:
        result["status"] = f"refused:{exc.label}"
    except KeyboardInterrupt:  # the guard's stop: what was measured is kept
        result["status"] = "stopped"
    except Exception as exc:  # noqa: BLE001 - a fixed label and the class only
        result["status"] = f"failed:{type(exc).__name__}"
    if pieces is not None:
        result["sensor_samples"].append(pieces["sensors"]())
        result["sensors"] = sensor_summary(result["sensor_samples"])
        result["system_after"] = pieces["system"]()
        result["loadavg_after"] = list(os.getloadavg())
        result["finished_utc"] = pieces["clock"].utc_now().isoformat()
    result["derived"] = derived(result)
    result["verdict"], result["problems"] = verdict(result)
    atomic_write(out, result)
    emit(json.dumps({"route_b_cost_check": result["verdict"], "status": result["status"], "stand_in": result["stand_in"]}))
    return result


# ---------------------------------------------------------------- what follows from the numbers, and the reading


EXPECTED = ("baseline", "inputs", "resize", "windows", "model_load", "first_detection", *(f"steady:{n}" for n in LAYOUTS))


def _phase(result: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    return next((p for p in result.get("phases") or [] if p.get("phase") == name), None)


def _delta(after: Mapping[str, Any] | None, before: Mapping[str, Any] | None, field: str) -> int | None:
    if not after or not before or after.get(field) is None or before.get(field) is None:
        return None
    return int(after[field]) - int(before[field])


def derived(result: Mapping[str, Any]) -> dict[str, Any]:
    """Planning figures computed from the measurements, each labelled with its assumption; never measured themselves."""
    out: dict[str, Any] = {}
    load, first = _phase(result, "model_load"), _phase(result, "first_detection")
    if load and first:
        out["model_bytes_rss"] = _delta(first.get("memory_after"), load.get("memory_before"), "VmRSS")
        out["model_bytes_pss"] = _delta(first.get("memory_after"), load.get("memory_before"), "Pss")
    steady = [p for p in result.get("phases") or [] if str(p.get("phase", "")).startswith("steady:")]
    growth = [p["peak_bytes"] - p["memory_before"]["VmRSS"] for p in steady
              if p.get("peak_bytes") is not None and (p.get("memory_before") or {}).get("VmRSS") is not None]
    if steady:  # each steady phase against its own start, which already holds the tool's copies (held_copy_bytes)
        out["steady_growth_over_phase_start_bytes"] = max(growth) if growth else None
    resize = _phase(result, "resize")
    if resize and (resize.get("calls") or {}).get("mean_ms") is not None:
        mean = resize["calls"]["mean_ms"]
        out["resize_at_20_fps"] = {"assumes": f"every decoded picture resized at {MAIN_FPS:g} fps, alone on the CPU",
                                   "wall_ms_per_s": round(mean * MAIN_FPS, 1),
                                   "cores": None if resize.get("cores") is None else round(
                                       resize["cores"] * mean * MAIN_FPS / 1000, 3)}
    ticks = {}
    for p in steady:
        mean = (p.get("ticks") or {}).get("mean_ms")
        if mean is not None:
            ticks[p["phase"].split(":", 1)[1]] = {"share_of_the_1_s_tick": round(mean / (FACE_INTERVAL_S * 1000), 3),
                                                  "cores_while_detecting": p.get("cores")}
    if ticks:
        out["detection_per_tick"] = {"assumes": "one face tick per second, no face kept (no embedding)", **ticks}
    return out


def verdict(result: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Execution only: every phase ran to its end. A slow or a large number is a result, never a failure."""
    status = str(result.get("status"))
    problems = [] if status == "finished" else [f"status:{status}"]
    for name in EXPECTED:
        phase = _phase(result, name)
        if phase is None:
            problems.append(f"phase_missing:{name}")
        elif phase.get("ended"):
            problems.append(f"phase_{phase['ended']}:{name}")
        elif phase.get("peak_reset") is not True:
            problems.append(f"peak_not_reset:{name}")
    if status.startswith("refused:") or not result.get("phases"):
        return "failed", problems
    return ("incomplete" if problems else "complete"), problems


def _mb(value: Any) -> str:
    return "n/a" if value is None else f"{value / 1e6:.1f} MB"


def reading(directory: Path) -> tuple[list[str], int]:
    lines: list[str] = []
    try:
        guard: Any = json.loads((directory / "result.json").read_text())
    except (OSError, ValueError):
        guard = None
        lines.append("MISSING the guard's result, result.json")
    if guard is not None:
        status = str(guard.get("status"))
        if status.startswith("refused:"):
            return [f"REFUSED the guard: {status}, before launch: nothing was started and nothing is judged"], 1
        g = guard.get("guard") or {}
        lines.append(f"{'PASS' if status == 'child_exited' else 'FAIL'} the guard: status {status}, trigger {g.get('trigger')}, "
                     f"returncode {(guard.get('child') or {}).get('returncode')}; peak pressure {g.get('peak_pressure_bytes')} B")
    try:
        result = json.loads((directory / "route-b-cost.json").read_text())
    except (OSError, ValueError):
        lines.append("MISSING the check's result, route-b-cost.json (nothing is judged)")
        return lines, 1
    if result.get("stand_in"):
        lines.append("STAND-IN DETECTOR: a dry run of the tool; no detection time or model memory here is YuNet's")
    lines.append(SCOPE)
    state, problems = verdict(result)
    lines.append(f"execution: {state}{' (' + '; '.join(problems) + ')' if problems else ''}; status {result.get('status')}")
    policy = result.get("thp_disable") or {}
    lines.append(f"memory policy: THP disabled for this process {'and verified' if policy.get('verified') else 'NOT verified'} "
                 f"(reason {policy.get('reason')}), as validation runs set it; LD_PRELOAD "
                 f"{'set' if (result.get('environment') or {}).get('ld_preload') else 'not set'}")
    env = result.get("environment") or {}
    lines.append(f"environment: Python {env.get('python')}, NumPy {env.get('numpy')}, OpenCV {env.get('opencv')} with "
                 f"{env.get('opencv_threads')} threads, {env.get('cpus')} CPUs ({env.get('affinity')} usable); load average before "
                 f"{result.get('loadavg_before')}, after {result.get('loadavg_after')}")
    for phase in result.get("phases") or []:
        name = phase["phase"]
        before, after = phase.get("memory_before") or {}, phase.get("memory_after") or {}
        memory = (f"RSS {_mb(before.get('VmRSS'))} -> {_mb(after.get('VmRSS'))}, phase peak {_mb(phase.get('peak_bytes'))}, "
                  f"PSS {_mb(before.get('Pss'))} -> {_mb(after.get('Pss'))}")
        cost = f"{phase.get('wall_ms')} ms wall, {phase.get('cores')} cores"
        if name == "inputs":
            detail = f"synthetic pictures made: {_mb(phase.get('bytes_made'))} (scaffolding, not a runtime cost)"
        elif name == "resize":
            detail = f"working copy per picture: {phase.get('calls')}"
        elif name == "windows":
            detail = "; ".join(f"{k}: windows {v['windows']} (scales {v['scales']}), copies {_mb(v['copy_bytes'])}, {v['calls']}"
                               for k, v in (phase.get("layouts") or {}).items())
        elif name == "first_detection":
            detail = f"first detection on {phase.get('input')}: {phase.get('ms')} ms, candidates {phase.get('candidates')}"
        elif name.startswith("steady:"):
            detail = (f"windows {phase.get('windows')}; the tool's copies held {_mb(phase.get('held_copy_bytes'))} (scaffolding); "
                      f"per tick {phase.get('ticks')}; first tick {phase.get('first_tick_ms')} ms; "
                      f"per window {phase.get('per_window')}; candidates seen {phase.get('candidates')}")
        else:
            detail = ""
        lines.append(f"{name}: {memory}; {cost}{'; ' + detail if detail else ''}")
    d = result.get("derived") or {}
    lines.append(f"model: RSS {_mb(d.get('model_bytes_rss'))}, PSS {_mb(d.get('model_bytes_pss'))} from before the load to after "
                 f"the first detection; steady growth {_mb(d.get('steady_growth_over_phase_start_bytes'))} at most over a steady "
                 "phase's own start (which holds the tool's window copies, scaffolding)")
    for name in ("resize_at_20_fps", "detection_per_tick"):
        if name in d:
            lines.append(f"ESTIMATE {name}: {d[name]}")
    lines.append(f"temperature and clocks: {result.get('sensors')}; memory available before {_mb((result.get('system_before') or {}).get('MemAvailable'))}, "
                 f"after {_mb((result.get('system_after') or {}).get('MemAvailable'))}")
    lines.extend(LIMITS)
    lines.append(f"route B cost check: execution {state}; numbers for planning only; no limit, gate or admission follows from them")
    return lines, EXIT[state]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--weights", type=Path, default=None, help="the YuNet weights; default DeepFace's weights directory")
    go.add_argument("--stand-in", action="store_true", help="the dry run: a stand-in detector, no weights, no model")
    go.add_argument("--pictures", type=int, default=PICTURES)
    go.add_argument("--resize-calls", type=int, default=RESIZE_CALLS)
    go.add_argument("--window-calls", type=int, default=WINDOW_CALLS)
    go.add_argument("--ticks", type=int, default=TICKS)
    look = sub.add_parser("read")
    look.add_argument("directory", type=Path)
    sub.add_parser("guard-duration", help="the guard's --duration-s")
    args = parser.parse_args(argv)
    if args.command == "guard-duration":
        print(f"{GUARD_DURATION_S:g}")
        return 0
    if args.command == "read":
        lines, code = reading(args.directory)
        print("\n".join(lines))
        return code
    if min(args.pictures, args.resize_calls, args.window_calls, args.ticks) < 1:
        parser.error("counts must be at least 1")
    return EXIT[run(args)["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
