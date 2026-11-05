"""The two-stream timing check (session 116, part 14): how far apart the substream's and the main stream's frames of one moment
reach this device, on its own ingest clock. It decides between two sessions and one stream for the recognition-range path.

**What it estimates, and what it does not.** Each frame is stamped when its decode finishes (FrameStamper's ingest time). The same
movement appears in both streams; correlating that movement gives the difference between the two streams' capture-to-ingest
delays, for this device, connection and decoder setting. **It is an alignment estimate, not proof of capture-time
synchronization:** it does not show when either frame was captured, only how the arrivals line up for what can be matched, and
compression, 15 and 20 fps sampling and repetitive movement can bias or confuse it. PTS gives no cross-stream link, because each
session's PTS starts at its own zero. PTS is used only within a stream, for delivery jitter.

**Method.** Each frame becomes a 6x4 grid of mean brightness, held in memory only. Per cell, the change over time is resampled to
20 ms and normalized cross-correlation is computed for main-stream lags within 2 s (positive: the main stream arrives later). The
6x4 cells keep position, so that someone walking left and right does not look like a repeating signal.

**Uncertainty, predeclared.** For each walking block:
- a 95 % interval from a block bootstrap over 5 s blocks;
- the peak's width at 90 % of its height;
- the second-highest separated peak, as a ratio;
- a null: the same correlation with the main stream shifted far outside the search range, which keeps both signals' structure and
  destroys their alignment;
- how many of the 5 s blocks agree alone, within 100 ms;
- the motion over the same stream's still block.

A block is labelled ``estimated`` only when nothing below applies; otherwise it gives **no estimate**, and its candidate is shown
as a candidate. These labels describe the measurement. They are not pass or fail, and collection is never extended to change them:
``too_few_frames``, ``insufficient_motion``, ``ambiguous_at_search_edge``, ``ambiguous_low_peak``, ``ambiguous_multiple_peaks``,
``ambiguous_wide_interval``.

**The timeline** (seconds from T0, both streams delivering; the guard prints the cues): six connections, 80 s each, with two main
stream decoder settings, quiet and under a synthetic CPU load (a stand-in for the face embedding's bursts, never the combined
workload), never pooled. Optionally one block of light-switch steps: an independent event, read the same way with its own
limitations. Each connection is new, so differences between connections are visible.

**Numbers only.** No frame or grid is saved. Saved: labels, estimates, per-frame times with one brightness and one motion number
(``stream-series.json``), temperatures and clocks. It runs no model and changes no limit, gate, threshold or admission.

**No tolerance is adopted.** The largest magnitude inside an estimate's interval is reported, never compared with a limit. Whether a
difference is safe for ownership depends on measured boxes and speeds (``sentinel.identity.head_windows.safe_skew_s``).

Usage: ``stream_timing_check.py run --out FILE --light yes|no`` (URLs from SENTINEL_RTSP_URL and SENTINEL_RTSP_MAIN_URL), under
step5_guard, which prints ``cues --light yes|no``; ``stream_timing_check.py read SESSION_DIR``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from operator import mul
from pathlib import Path
from typing import Any

from door_range_check import (
    EXIT,
    MAIN,
    MAIN_URL_ENV,
    STREAMS,
    SUB,
    LiveStreams,
    Refused,
    atomic_writer,
    privacy_line,
    proc_memory,
    read_account,
)
from sentinel.media.clock import NS_PER_SECOND, Clock, SystemClock
from sentinel.media.opencv_source import RTSP_URL_ENV, rtsp_endpoint

GRID = (6, 4)  # columns, rows: position survives, a picture does not; never saved
STEP_S = 0.02
MAX_LAG_S = 2.0
BLOCK_S = 5.0
NULL_SHIFTS_S = (8.0, 14.0, 20.0, 26.0)  # all beyond the search range and inside the window
BOOTSTRAP_REPLICATES = 300
BOOTSTRAP_SEED = 20261105
BLOCK_AGREEMENT_MS = 100.0
# What a label means, set before any data
MIN_FRAME_SHARE = 0.8  # of the frames a block should hold at the stream's nominal rate
MIN_MOTION_RATIO = 3.0  # walking block over still block, per stream
PEAK_OVER_NULL = 1.5
SECOND_PEAK_RATIO = 0.85
MAX_INTERVAL_MS = 200.0
NOMINAL_FPS = {SUB: 15.0, MAIN: 20.0}  # measured in dr-20261105T085421Z
UNBOUNDED_RATIO = 999999.0  # a still block with no change at all; JSON has no infinity

CLOSE_S = 75.0
STILL_WINDOW = (10.0, 25.0)
WALK_CUES = (25.0, 70.0)
WALK_WINDOW = (27.0, 68.0)  # inside the cues: cues are not movement times
SETTLE_S = 5.0
READY_TIMEOUT_S = 60.0
POLL_S = 0.004
SENSOR_EVERY_S = 2.0
MAX_FRAMES_PER_TRACK = 3000
LOAD_PROCESSES = 6  # the device's six cores, as a whole-frame embedding's threads may use them
LOAD_ON_S, LOAD_OFF_S = 0.8, 0.2  # fci3c's face path was busy about this share of each second; a stand-in, never the workload
BURN = ("import sys,time\nend=time.monotonic()+float(sys.argv[1])\nwhile time.monotonic()<end:\n t=time.monotonic()\n"
        f" while time.monotonic()-t<{LOAD_ON_S}: pass\n time.sleep({LOAD_OFF_S})\n")
LIGHT_TOGGLES = (10.0, 20.0, 30.0, 40.0)
LIGHT_WINDOW = (5.0, 48.0)
STEP_WINDOW_S = 0.3
STEP_MIN_LEVELS = 3.0
STEP_NOISE_FACTOR = 6.0
STEP_RESOLUTION_MS = round((1000 / NOMINAL_FPS[SUB] + 1000 / NOMINAL_FPS[MAIN]) / 2, 1)  # one step: half a frame of each stream
THERMAL_ROOT = Path("/sys/class/thermal")
CPU_ROOT = Path("/sys/devices/system/cpu")
GPU_FREQ = Path("/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq")

SCOPE = ("scope: an alignment estimate on this device's ingest clock, for these connections and settings, one session; not proof of "
         "capture-time synchronization; no model, no frame or grid saved; no tolerance adopted and no route chosen by this reading")
LIMITS = (
    "limit: ingest time is when decoding finished; the lag combines capture, encoding, network and decoding differences and cannot "
    "separate them",
    "limit: compression, the two frame rates and repeated movement can bias or confuse a correlation; an ambiguous block gives no estimate",
    "limit: the load blocks use a synthetic CPU burner, not the combined face, scene and chat workload",
    "limit: the interval covers variation inside a block; differences between connections show only these connections",
    "limit: the search step is 20 ms and frames arrive 50 and 67 ms apart: a narrower interval shows blocks agreeing, not that accuracy",
    "limit: ownership is judged from measured boxes and speeds (head_windows.safe_skew_s), never from these numbers alone",
)


@dataclass(frozen=True)
class Slot:
    name: str
    main_threads: int
    load: bool
    start_s: float

    @property
    def condition(self) -> str:
        return f"{'load' if self.load else 'quiet'}_main_threads_{self.main_threads}"


SLOTS = tuple(Slot(name, threads, load, index * 80.0) for index, (name, threads, load) in enumerate(
    (("c1", 1, False), ("c2", 2, False), ("c3", 1, False), ("c4", 2, False), ("l1", 1, True), ("l2", 2, True))))
LIGHT_SLOT = Slot("light", 1, False, 480.0)
WALKS_END_S = SLOTS[-1].start_s + WALK_CUES[1]
LIGHT_END_S = LIGHT_SLOT.start_s + LIGHT_WINDOW[1] + 2.0
ANALYSIS_BOUND_S = 240.0  # what the guard must allow after the timeline; measured far below this on synthetic data
GUARD_DURATION_S = LIGHT_END_S + ANALYSIS_BOUND_S


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def cues(light: bool) -> list[tuple[float, str, str]]:
    """The operator's beeps, from T0. Fixed text; the guard prints them."""
    out = [(0.0, "t0", "T0 - start your phone stopwatch and stand still in the camera's view until the next beep")]
    for index, slot in enumerate(SLOTS, 1):
        walk, stop = slot.start_s + WALK_CUES[0], slot.start_s + WALK_CUES[1]
        out.append((walk, f"walk{index}", f"{_mmss(walk)} - walk back and forth across the camera's view, varying your pace, until the next beep"))
        if index == len(SLOTS) and not light:
            out.append((stop, "stop", f"{_mmss(stop)} - finished: come back to the Mac and type nothing; it ends by itself"))
        elif index == len(SLOTS):
            out.append((stop, "tolight", f"{_mmss(stop)} - go to the light switch and stand still until the next beep"))
        else:
            out.append((stop, f"still{index}", f"{_mmss(stop)} - stop and stand still until the next beep"))
    if light:
        for index, offset in enumerate(LIGHT_TOGGLES, 1):
            at = LIGHT_SLOT.start_s + offset
            out.append((at, f"switch{index}", f"{_mmss(at)} - switch the light once, then stand still"))
        out.append((LIGHT_END_S, "stop", f"{_mmss(LIGHT_END_S)} - finished: come back to the Mac and type nothing; it ends by itself"))
    return out


# ---------------------------------------------------------------- frames to numbers


def coarse_grid(image: Any) -> tuple[float, ...]:
    """A frame's mean brightness (its green channel) on the grid. In memory only."""
    import cv2

    small = cv2.resize(image[::4, ::4, 1].astype("float32"), GRID, interpolation=cv2.INTER_AREA)
    return tuple(float(v) for v in small.ravel())


@dataclass
class Track:
    """One stream's frames in one slot: times in seconds from the slot's start, identities, and the grids, in memory only."""

    times: list[float] = field(default_factory=list)
    pts_us: list[int | None] = field(default_factory=list)
    seq: list[int] = field(default_factory=list)
    epoch: list[int] = field(default_factory=list)
    grids: list[array] = field(default_factory=list)
    overflow: int = 0

    def add(self, t_s: float, pts_us: int | None, seq: int, epoch: int, grid: Sequence[float]) -> None:
        if len(self.times) >= MAX_FRAMES_PER_TRACK:
            self.overflow += 1
            return
        self.times.append(t_s)
        self.pts_us.append(pts_us)
        self.seq.append(seq)
        self.epoch.append(epoch)
        self.grids.append(array("f", grid))

    def indices(self, start_s: float, end_s: float) -> list[int]:
        return [i for i, t in enumerate(self.times) if start_s <= t < end_s]


def frame_facts(track: Track, start_s: float, end_s: float, stream: str) -> dict[str, Any]:
    """The frames in a window: how many, at what rate, how many the hand-off skipped, and delivery jitter within the stream."""
    chosen = track.indices(start_s, end_s)
    expected = (end_s - start_s) * NOMINAL_FPS[stream]
    facts: dict[str, Any] = {"frames": len(chosen), "expected": round(expected), "share": round(len(chosen) / expected, 3)}
    if len(chosen) < 2:
        return facts
    times = [track.times[i] for i in chosen]
    steps = sorted(b - a for a, b in zip(times, times[1:], strict=False))
    span = times[-1] - times[0]
    facts.update(fps=round((len(times) - 1) / span, 2) if span > 0 else None,
                 step_ms={"p50": round(steps[len(steps) // 2] * 1000, 1),
                          "p95": round(steps[min(len(steps) - 1, math.ceil(len(steps) * 0.95) - 1)] * 1000, 1),
                          "max": round(steps[-1] * 1000, 1)},
                 skipped_by_hand_off=sum(max(0, track.seq[b] - track.seq[a] - 1) for a, b in zip(chosen, chosen[1:], strict=False)
                                         if track.epoch[a] == track.epoch[b]))
    facts["delivery_ms"] = delivery_jitter([track.times[i] for i in chosen], [track.pts_us[i] for i in chosen])
    return facts


def delivery_jitter(times: Sequence[float], pts_us: Sequence[int | None]) -> dict[str, float] | None:
    """Within one stream only: how much later than the earliest-delivered frame each frame arrived, against its own PTS, and the
    drift. PTS is stream-relative, so this says nothing about the other stream."""
    pairs = [(t, p / 1e6) for t, p in zip(times, pts_us, strict=True) if p is not None]
    if len(pairs) < 10:
        return None
    residual = sorted((t - p, t) for t, p in pairs)
    lowest = residual[0][0]
    delays = sorted((value - lowest) * 1000 for value, _ in residual)
    first, last = pairs[0], pairs[-1]
    span = last[0] - first[0]
    drift = ((last[0] - last[1]) - (first[0] - first[1])) * 1000 / (span / 60) if span > 0 else None
    return {"p50": round(delays[len(delays) // 2], 1), "p95": round(delays[min(len(delays) - 1, math.ceil(len(delays) * 0.95) - 1)], 1),
            "max": round(delays[-1], 1), "drift_ms_per_min": None if drift is None else round(drift, 1)}


def _differences(track: Track, chosen: Sequence[int]) -> tuple[list[float], list[list[float]]]:
    """Per cell, the brightness change per second between consecutive frames, placed at their midpoint. A pair across a
    reconnection or with a long gap is left out, never bridged."""
    times: list[float] = []
    values: list[list[float]] = []
    for a, b in zip(chosen, chosen[1:], strict=False):
        dt = track.times[b] - track.times[a]
        if track.epoch[a] != track.epoch[b] or not 0 < dt <= 1.0:
            continue
        times.append((track.times[a] + track.times[b]) / 2)
        values.append([(y - x) / dt for x, y in zip(track.grids[a], track.grids[b], strict=True)])
    return times, values


def _resample(times: Sequence[float], values: Sequence[Sequence[float]], start_s: float, count: int) -> list[list[float]]:
    """Each cell on a uniform grid by linear interpolation, zero outside the frames' support, its mean removed."""
    cells = len(values[0]) if values else GRID[0] * GRID[1]
    out = [[0.0] * count for _ in range(cells)]
    if len(times) < 2:
        return out
    position = 0
    for index in range(count):
        t = start_s + index * STEP_S
        if t < times[0] or t > times[-1]:
            continue
        while position + 1 < len(times) - 1 and times[position + 1] < t:
            position += 1
        a, b = times[position], times[position + 1]
        weight = (t - a) / (b - a) if b > a else 0.0
        low, high = values[position], values[position + 1]
        for cell in range(cells):
            out[cell][index] = low[cell] + (high[cell] - low[cell]) * weight
    for cell in out:
        mean = sum(cell) / count
        cell[:] = [v - mean for v in cell]
    return out


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(map(mul, a, b))


def _sweep(x: Sequence[Sequence[float]], y: Sequence[Sequence[float]], max_lag: int, blocks: Sequence[tuple[int, int]]
           ) -> list[list[float]]:
    """For each lag from -max_lag to +max_lag (y later for positive), the sum over the cells of x(t)y(t+lag), per block."""
    count = len(x[0])
    out = []
    for lag in range(-max_lag, max_lag + 1):
        row = []
        for start, end in blocks:
            first, last = max(start, -lag, 0), min(end, count - lag, count)
            row.append(0.0 if last <= first else sum(_dot(xs[first:last], ys[first + lag:last + lag])
                                                     for xs, ys in zip(x, y, strict=True)))
        out.append(row)
    return out


def _quantile(ordered: Sequence[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe_peak(curve: Sequence[float], step_ms: float, max_lag: int) -> dict[str, Any]:
    """The highest point, its width at 90 %, and the highest other local maximum beyond where the curve first falls under half."""
    best = max(range(len(curve)), key=lambda i: (curve[i], -abs(i - max_lag)))
    peak = curve[best]
    left = right = best
    while left > 0 and curve[left - 1] >= 0.9 * peak:
        left -= 1
    while right + 1 < len(curve) and curve[right + 1] >= 0.9 * peak:
        right += 1
    low = high = best
    while low > 0 and curve[low - 1] >= 0.5 * peak:
        low -= 1
    while high + 1 < len(curve) and curve[high + 1] >= 0.5 * peak:
        high += 1
    others = [curve[i] for i in range(1, len(curve) - 1) if not low <= i <= high and curve[i - 1] <= curve[i] >= curve[i + 1]]
    second = max(others, default=None)
    return {"lag_ms": round((best - max_lag) * step_ms, 1), "peak": round(peak, 4), "width_ms": round((right - left) * step_ms, 1),
            "second_ratio": None if second is None or peak <= 0 else round(max(second, 0.0) / peak, 3),
            "at_edge": best in (0, len(curve) - 1)}


def estimate_lag(sub: Track, main: Track, start_s: float, end_s: float) -> dict[str, Any]:
    """The main stream's lag behind the substream over a window, with the predeclared uncertainty. Positive: main arrives later."""
    count = int((end_s - start_s) / STEP_S)
    x = _resample(*_differences(sub, sub.indices(start_s, end_s)), start_s, count)
    y = _resample(*_differences(main, main.indices(start_s, end_s)), start_s, count)
    energy_x = sum(_dot(c, c) for c in x)
    energy_y = sum(_dot(c, c) for c in y)
    if energy_x <= 0 or energy_y <= 0:
        return {"lag_ms": None, "peak": None, "reason": "no_signal"}
    max_lag = round(MAX_LAG_S / STEP_S)
    per_block = round(BLOCK_S / STEP_S)
    blocks = [(a, a + per_block) for a in range(0, count - per_block + 1, per_block)] or [(0, count)]
    blocks[-1] = (blocks[-1][0], count)
    table = _sweep(x, y, max_lag, blocks)
    norm = math.sqrt(energy_x * energy_y)
    curve = [sum(row) / norm for row in table]
    facts = describe_peak(curve, STEP_S * 1000, max_lag)
    # the bootstrap resamples whole blocks, because neighbouring moments are not independent
    rng = random.Random(BOOTSTRAP_SEED)
    replicas = []
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = [rng.randrange(len(blocks)) for _ in blocks]
        replica = [sum(row[b] for b in chosen) for row in table]
        replicas.append((max(range(len(replica)), key=replica.__getitem__) - max_lag) * STEP_S * 1000)
    replicas.sort()
    facts["interval_ms"] = [round(_quantile(replicas, 0.025), 1), round(_quantile(replicas, 0.975), 1)]
    facts["largest_magnitude_ms"] = round(max(abs(v) for v in facts["interval_ms"]), 1)
    alone = [(max(range(len(table)), key=lambda i: table[i][b]) - max_lag) * STEP_S * 1000 for b in range(len(blocks))]
    facts["blocks_agreeing"] = sum(abs(v - facts["lag_ms"]) <= BLOCK_AGREEMENT_MS for v in alone)
    facts["blocks"] = len(blocks)
    # the null: the same signals, the alignment destroyed by a shift far outside the search range
    nulls = []
    whole = [(0, count)]
    for shift_s in NULL_SHIFTS_S:
        shift = round(shift_s / STEP_S)
        moved = [c[shift:] + c[:shift] for c in y]
        nulls.append(max(row[0] for row in _sweep(x, moved, max_lag, whole)) / norm)
    facts["null_peak"] = round(max(nulls), 4)
    return facts


def motion_level(track: Track, start_s: float, end_s: float) -> float | None:
    """The root-mean-square change per second over the cells in a window."""
    _, values = _differences(track, track.indices(start_s, end_s))
    if not values:
        return None
    return math.sqrt(sum(v * v for row in values for v in row) / (len(values) * len(values[0])))


def classify(frames: Mapping[str, Mapping[str, Any]], ratio: Mapping[str, float | None], facts: Mapping[str, Any]) -> tuple[str, list[str]]:
    """The label and every reason that applies. ``estimated`` only when none does."""
    reasons = []
    if any(f.get("share", 0) < MIN_FRAME_SHARE for f in frames.values()):
        reasons.append("too_few_frames")
    if any(r is None or r < MIN_MOTION_RATIO for r in ratio.values()):
        reasons.append("insufficient_motion")
    if facts.get("lag_ms") is None or (facts.get("peak") or 0) <= 0:
        reasons.append("ambiguous_low_peak")
    else:
        if facts["at_edge"]:
            reasons.append("ambiguous_at_search_edge")
        if facts["peak"] < PEAK_OVER_NULL * facts["null_peak"]:
            reasons.append("ambiguous_low_peak")
        if facts["second_ratio"] is not None and facts["second_ratio"] >= SECOND_PEAK_RATIO:
            reasons.append("ambiguous_multiple_peaks")
        interval = facts["interval_ms"]
        if interval[1] - interval[0] > MAX_INTERVAL_MS:
            reasons.append("ambiguous_wide_interval")
    return (reasons[0], reasons) if reasons else ("estimated", [])


def analyse_slot(slot: Slot, tracks: Mapping[str, Track]) -> dict[str, Any]:
    """One connection's walking block: the frames, the motion over the still block, the estimate and its label."""
    frames = {s: frame_facts(tracks[s], *WALK_WINDOW, s) for s in STREAMS}
    ratio: dict[str, float | None] = {}
    for stream in STREAMS:
        still = motion_level(tracks[stream], *STILL_WINDOW)
        walk = motion_level(tracks[stream], *WALK_WINDOW)
        ratio[stream] = None if still is None or walk is None else round(walk / still, 2) if still > 0 else UNBOUNDED_RATIO
    facts = estimate_lag(tracks[SUB], tracks[MAIN], *WALK_WINDOW)
    label, reasons = classify(frames, ratio, facts)
    return {"label": label, "reasons": reasons, **facts, "motion_ratio": ratio, "frames": frames,
            "still_frames": {s: frame_facts(tracks[s], *STILL_WINDOW, s) for s in STREAMS}}


def step_events(times: Sequence[float], levels: Sequence[float]) -> list[dict[str, float]]:
    """Sudden changes of the whole picture's brightness, each at its midpoint crossing. The threshold is at least
    STEP_MIN_LEVELS and STEP_NOISE_FACTOR times the window's own typical change, so a still scene gives none."""
    n = len(times)
    if n < 8:
        return []
    def mean(start: float, end: float) -> float | None:
        chosen = [levels[i] for i in range(n) if start <= times[i] < end]
        return sum(chosen) / len(chosen) if chosen else None

    scores = []
    for i in range(n):
        before, after = mean(times[i] - STEP_WINDOW_S, times[i]), mean(times[i], times[i] + STEP_WINDOW_S)
        scores.append(0.0 if before is None or after is None else after - before)
    typical = statistics.median(abs(s) for s in scores)
    threshold = max(STEP_MIN_LEVELS, STEP_NOISE_FACTOR * typical)
    events: list[dict[str, float]] = []
    index = 0
    while index < n:
        if abs(scores[index]) < threshold:
            index += 1
            continue
        end = index
        while end + 1 < n and abs(scores[end + 1]) >= threshold:
            end += 1
        top = max(range(index, end + 1), key=lambda i: abs(scores[i]))
        before = mean(times[top] - STEP_WINDOW_S, times[top])
        after = mean(times[top], times[top] + STEP_WINDOW_S)
        middle = (before + after) / 2
        crossing = next((i for i in range(max(1, index - 3), min(n, end + 4))
                         if (levels[i - 1] - middle) * (levels[i] - middle) <= 0 and levels[i] != levels[i - 1]), None)
        if crossing is not None:
            weight = (middle - levels[crossing - 1]) / (levels[crossing] - levels[crossing - 1])
            events.append({"t_s": times[crossing - 1] + weight * (times[crossing] - times[crossing - 1]),
                           "size": round(after - before, 1)})
        index = end + 1
    return events


def analyse_light(tracks: Mapping[str, Track]) -> dict[str, Any]:
    """The light switch's steps, paired by direction within the search range: each pair's main-stream lag. An independent event,
    with its own limits: exposure control and encoding can shape each stream's step differently."""
    found: dict[str, list[dict[str, float]]] = {}
    for stream in STREAMS:
        track = tracks[stream]
        chosen = track.indices(*LIGHT_WINDOW)
        found[stream] = step_events([track.times[i] for i in chosen], [sum(track.grids[i]) / len(track.grids[i]) for i in chosen])
    lags = []
    for event in found[SUB]:
        same = [e for e in found[MAIN] if e["size"] * event["size"] > 0 and abs(e["t_s"] - event["t_s"]) <= MAX_LAG_S]
        if same:
            nearest = min(same, key=lambda e: abs(e["t_s"] - event["t_s"]))
            lags.append(round((nearest["t_s"] - event["t_s"]) * 1000, 1))
    return {"label": "steps_found" if len(lags) >= 2 else "no_usable_steps", "steps": {s: len(found[s]) for s in STREAMS},
            "paired": len(lags), "lags_ms": lags, "median_ms": round(statistics.median(lags), 1) if lags else None,
            "resolution_ms": STEP_RESOLUTION_MS,
            "frames": {s: frame_facts(tracks[s], *LIGHT_WINDOW, s) for s in STREAMS}}


# ---------------------------------------------------------------- the device's sensors, and the load


def _sensor_int(path: Path) -> int | None:
    """A sensor file's number, or None. Reading this device's cv0-cv2 thermal zones, which have no value, fails with TypeError
    rather than OSError (the dry run on the real files found it), and a sensor must never end a measurement."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError, TypeError):
        return None


def read_sensors(thermal: Path = THERMAL_ROOT, cpu: Path = CPU_ROOT, gpu: Path = GPU_FREQ) -> dict[str, Any]:
    """Temperatures in degrees Celsius by zone, each core's clock in kHz and the GPU's in Hz, from world-readable files. A zone
    that cannot be read is left out."""
    zones: dict[str, float] = {}
    for zone in sorted(thermal.glob("thermal_zone*")):
        value = _sensor_int(zone / "temp")
        try:
            name = (zone / "type").read_text().strip()
        except (OSError, ValueError, TypeError):
            continue
        if value is not None:
            zones[name] = value / 1000
    clocks = [v for v in (_sensor_int(p) for p in sorted(cpu.glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))) if v is not None]
    return {"zones_c": zones, "cpu_khz": clocks, "gpu_hz": _sensor_int(gpu)}


def sensor_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not samples:
        return None
    zones: dict[str, list[float]] = {}
    for sample in samples:
        for name, value in sample["zones_c"].items():
            zones.setdefault(name, []).append(value)
    cpu = [v for s in samples for v in s["cpu_khz"]]
    gpu = [s["gpu_hz"] for s in samples if s["gpu_hz"] is not None]
    return {"samples": len(samples), "max_c": {k: round(max(v), 1) for k, v in sorted(zones.items())},
            "cpu_khz": [min(cpu), max(cpu)] if cpu else None, "gpu_hz": [min(gpu), max(gpu)] if gpu else None}


def spawn_load(seconds: float) -> list[Any]:
    """The synthetic CPU load: LOAD_PROCESSES self-limiting processes, ending by themselves whatever happens to this one."""
    return [subprocess.Popen([sys.executable, "-I", "-c", BURN, str(seconds)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(LOAD_PROCESSES)]


def stop_load(processes: Sequence[Any]) -> None:
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()


# ---------------------------------------------------------------- the session


@dataclass
class Session:
    streams: Any
    clock: Clock
    sleep: Callable[[float], None]
    write: Callable[[dict[str, Any]], None]
    reduce: Callable[[Any], Sequence[float]] = coarse_grid
    spawn: Callable[[float], list[Any]] = spawn_load
    stop: Callable[[Sequence[Any]], None] = stop_load
    sensors: Callable[[], dict[str, Any]] = read_sensors
    memory: Callable[[], dict[str, int]] = proc_memory
    samples: list[dict[str, Any]] = field(default_factory=list)
    sensor_errors: int = 0
    next_sensor_ns: int = 0


def _now_s(session: Session, t0_ns: int) -> float:
    return (session.clock.monotonic_ns() - t0_ns) / NS_PER_SECOND


def _collect(session: Session, tracks: Mapping[str, Track], t0_ns: int, slot: Slot, until_s: float,
             series: Mapping[str, list[list[float]]]) -> None:
    """Take every frame the hand-off gives until a time from T0; times are ingest times, so taking one late costs nothing unless a
    frame is replaced, and that is counted."""
    while _now_s(session, t0_ns) < until_s:
        for stream in STREAMS:
            captured = session.streams.take(stream, POLL_S)
            if captured is None:
                continue
            frame = captured.frame
            t = (frame.ingest_mono_ns - t0_ns) / NS_PER_SECOND - slot.start_s
            grid = session.reduce(captured.image)
            del captured
            track = tracks[stream]
            level = sum(grid) / len(grid)
            motion = 0.0
            if track.grids:
                dt = t - track.times[-1]
                if dt > 0:
                    motion = sum(abs(a - b) for a, b in zip(grid, track.grids[-1], strict=True)) / len(grid) / dt
            track.add(t, frame.source_pts, frame.frame_seq, frame.stream_epoch, grid)
            series[stream].append([round(t * 1000), round(level, 1), round(motion, 1)])
        now = session.clock.monotonic_ns()
        if now >= session.next_sensor_ns:
            session.next_sensor_ns = now + round(SENSOR_EVERY_S * NS_PER_SECOND)
            try:
                session.samples.append(session.sensors())
            except Exception:  # noqa: BLE001 - counted: a sensor must never end a measurement
                session.sensor_errors += 1


def _start_streams(session: Session, slot: Slot) -> None:
    session.streams.start(SUB, 1)
    session.streams.start(MAIN, slot.main_threads)


def _first_frames(session: Session, timeout_s: float) -> dict[str, dict[str, Any]]:
    """Both streams' first frames, or a Refused label naming the stream. Their sizes must be the two this check compares."""
    deadline = session.clock.monotonic_ns() + round(timeout_s * NS_PER_SECOND)
    first: dict[str, dict[str, Any]] = {}
    while len(first) < len(STREAMS):
        for stream in STREAMS:
            if stream not in first:
                captured = session.streams.take(stream, 0.1)
                if captured is not None:
                    first[stream] = {"width": captured.frame.native_width, "height": captured.frame.native_height,
                                     "boot_id": captured.frame.boot_id, "run_id": captured.frame.run_id}
        if len(first) < len(STREAMS) and session.clock.monotonic_ns() >= deadline:
            missing = next(s for s in STREAMS if s not in first)
            raise Refused(f"stream_not_ready:{missing}:{session.streams.counts(missing)['problem']}")
    if first[SUB]["boot_id"] != first[MAIN]["boot_id"]:
        raise Refused("boot_id_mismatch")
    if first[SUB]["width"] * first[SUB]["height"] >= first[MAIN]["width"] * first[MAIN]["height"]:
        raise Refused("main_stream_not_larger_than_substream")
    return first


def timeline(session: Session, result: dict[str, Any], emit: Callable[[str], None], light: bool,
             series_out: Callable[[dict[str, Any]], None]) -> None:
    """Connect, T0, every slot in turn, then the analysis, which needs the grids and so happens before they are dropped."""
    schedule = [*SLOTS, LIGHT_SLOT] if light else list(SLOTS)
    held: list[tuple[Slot, dict[str, Track]]] = []
    all_series: dict[str, Any] = {}
    t0_ns = None
    for index, slot in enumerate(schedule):
        if t0_ns is not None:
            wait = slot.start_s - _now_s(session, t0_ns)
            if wait > 0:
                session.sleep(wait)
        _start_streams(session, slot)
        if index == 0:
            result["streams"] = _first_frames(session, READY_TIMEOUT_S)
            t0_ns = session.clock.monotonic_ns()
            result["t0_mono_ns"] = t0_ns
            result["status"] = "running"
            session.write(result)
            emit(json.dumps({"run": "starting", "startup": {"tool": "stream_timing_check"}}))
        tracks = {s: Track() for s in STREAMS}
        series: dict[str, list[list[float]]] = {s: [] for s in STREAMS}
        load: list[Any] = []
        record: dict[str, Any] = {"name": slot.name, "condition": slot.condition, "main_threads": slot.main_threads, "load": slot.load}
        end_s = slot.start_s + (LIGHT_END_S - LIGHT_SLOT.start_s if slot is LIGHT_SLOT else CLOSE_S)
        try:
            if slot.load:
                _collect(session, tracks, t0_ns, slot, slot.start_s + SETTLE_S, series)
                load = session.spawn(end_s - slot.start_s - SETTLE_S)
            _collect(session, tracks, t0_ns, slot, slot.start_s + WALK_CUES[0] + 1.0, series)
            record["connections"] = session.streams.connections()
            _collect(session, tracks, t0_ns, slot, end_s, series)
            record["counts"] = {s: session.streams.counts(s) for s in STREAMS}
            record["loadavg"] = [round(v, 2) for v in os.getloadavg()]
            record["memory"] = session.memory()
        finally:  # however the slot ends, a guard's stop included, what it collected is kept
            if load:
                session.stop(load)
            session.streams.stop_all()
            record["collected"] = {s: len(tracks[s].times) for s in STREAMS}
            record["frames_overflow"] = {s: tracks[s].overflow for s in STREAMS}
            result["slots"].append(record)
            all_series[slot.name] = series
            held.append((slot, tracks))
            session.write(result)
            series_out(all_series)
    result["status"] = "analysing"
    session.write(result)
    started = session.clock.monotonic_ns()
    for slot, tracks in held:
        record = next(r for r in result["slots"] if r["name"] == slot.name)
        record["analysis"] = analyse_light(tracks) if slot is LIGHT_SLOT else analyse_slot(slot, tracks)
        tracks.clear()
    result["analysis_s"] = round((session.clock.monotonic_ns() - started) / NS_PER_SECOND, 1)
    result["status"] = "finished"


def summarize(result: Mapping[str, Any]) -> dict[str, Any]:
    """Per condition, never pooled across conditions: the estimates, how far apart they are, and the labels of the others."""
    conditions: dict[str, Any] = {}
    for record in result.get("slots") or []:
        if record["name"] == LIGHT_SLOT.name or "analysis" not in record:
            continue
        item = conditions.setdefault(record["condition"], {"estimates": [], "no_estimate": []})
        analysis = record["analysis"]
        if analysis["label"] == "estimated":
            item["estimates"].append({"slot": record["name"], "lag_ms": analysis["lag_ms"], "interval_ms": analysis["interval_ms"]})
        else:
            item["no_estimate"].append({"slot": record["name"], "label": analysis["label"]})
    for item in conditions.values():
        lags = [e["lag_ms"] for e in item["estimates"]]
        item["between_connections_ms"] = round(max(lags) - min(lags), 1) if len(lags) >= 2 else None
        item["largest_magnitude_in_an_interval_ms"] = (
            round(max(abs(v) for e in item["estimates"] for v in e["interval_ms"]), 1) if lags else None)
    return conditions


def verdict(result: Mapping[str, Any], light: bool) -> tuple[str, list[str]]:
    """Execution only: every block collected and analysed. An ambiguous measurement is a result here, not an incomplete run."""
    problems: list[str] = []
    if result.get("status") != "finished":
        problems.append(f"status:{result.get('status')}")
    slots = {r["name"]: r for r in result.get("slots") or []}
    for slot in [*SLOTS, *([LIGHT_SLOT] if light else [])]:
        record = slots.get(slot.name)
        if record is None:
            problems.append(f"slot_missing:{slot.name}")
        elif "analysis" not in record:
            problems.append(f"slot_not_analysed:{slot.name}")
        elif any((record["analysis"].get("frames") or {}).get(s, {}).get("share", 0) < MIN_FRAME_SHARE for s in STREAMS):
            problems.append(f"slot_short_of_frames:{slot.name}")
    if not slots:
        return "failed", problems
    return ("incomplete" if problems else "complete"), problems


def run(args: argparse.Namespace, *, environ: Mapping[str, str] | None = None, clock: Clock | None = None, streams: Any = None,
        emit: Callable[[str], None] = lambda line: print(line, flush=True), sleep: Callable[[float], None] = time.sleep,
        **pieces: Any) -> dict[str, Any]:
    """The whole session; the result file is written after every slot and however it ends."""
    environment = os.environ if environ is None else environ
    clock = clock or SystemClock()
    light = args.light == "yes"
    result: dict[str, Any] = {"check": "stream_timing", "scope": SCOPE, "started_utc": clock.utc_now().isoformat(),
                              "boot_id": clock.boot_id, "status": "starting", "light_block": light, "slots": [],
                              "method": {"grid": list(GRID), "step_ms": STEP_S * 1000, "max_lag_ms": MAX_LAG_S * 1000, "block_s": BLOCK_S,
                                         "bootstrap": BOOTSTRAP_REPLICATES, "min_motion_ratio": MIN_MOTION_RATIO,
                                         "peak_over_null": PEAK_OVER_NULL, "second_peak_ratio": SECOND_PEAK_RATIO,
                                         "max_interval_ms": MAX_INTERVAL_MS, "min_frame_share": MIN_FRAME_SHARE}}
    write = atomic_writer(args.out)
    series_write = atomic_writer(args.out.with_name("stream-series.json"))
    write(result)
    live = streams
    session = None
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
        if live is None:
            live = LiveStreams(urls, clock)
        session = Session(live, clock, sleep, write, **pieces)
        timeline(session, result, emit, light, series_write)
    except Refused as exc:
        result["status"] = f"refused:{exc.label}"
    except KeyboardInterrupt:  # the guard's stop: what was collected is kept; the analysis it interrupted is not forced
        result["status"] = "stopped"
    except Exception as exc:  # noqa: BLE001 - a fixed label: messages can name the camera
        result["status"] = f"failed:{type(exc).__name__}"
    finally:
        if live is not None:
            try:
                result["workers_stopped"] = live.stop_all()
            except Exception as exc:  # noqa: BLE001
                result["workers_stopped"] = f"failed:{type(exc).__name__}"
    result["sensors"] = sensor_summary(session.samples) if session is not None else None
    result["sensor_samples"] = session.samples if session is not None else []
    result["sensor_errors"] = session.sensor_errors if session is not None else 0
    result["finished_utc"] = clock.utc_now().isoformat()
    result["summary"] = summarize(result)
    result["verdict"], result["problems"] = verdict(result, light)
    write(result)
    emit(json.dumps({"stream_timing_check": result["verdict"], "status": result["status"]}))
    return result


# ---------------------------------------------------------------- the reading


def _line_for(record: Mapping[str, Any]) -> str:
    a = record["analysis"]
    head = (f"{record['name']} {record['condition']}: ")
    if a["label"] == "estimated":
        text = (f"estimated main-stream lag {a['lag_ms']} ms, 95 % interval {a['interval_ms']}, peak {a['peak']} (null {a['null_peak']}), "
                f"width {a['width_ms']} ms, second peak ratio {a['second_ratio']}, {a['blocks_agreeing']} of {a['blocks']} blocks agree")
    else:
        candidate = "" if a.get("lag_ms") is None else (
            f"; candidate only {a['lag_ms']} ms, interval {a.get('interval_ms')}, peak {a.get('peak')} (null {a.get('null_peak')}), "
            f"second peak ratio {a.get('second_ratio')}, {a.get('blocks_agreeing')} of {a.get('blocks')} blocks agree")
        text = f"NO ESTIMATE ({', '.join(a['reasons'])}){candidate}"
    rates = {s: (a["frames"][s].get("fps"), a["frames"][s].get("skipped_by_hand_off")) for s in STREAMS}
    delivery = {s: a["frames"][s].get("delivery_ms") for s in STREAMS}
    return (f"{head}{text}; motion over still {a['motion_ratio']}; fps and frames skipped by the hand-off {rates}; delivery within each "
            f"stream {delivery}; connections {record.get('connections')}; load average {record.get('loadavg')}")


def reading(directory: Path) -> tuple[list[str], int]:
    """The guard, the account, the privacy counts, then every block. It judges execution only and picks no route."""
    lines: list[str] = []
    try:
        guard: Any = json.loads((directory / "result.json").read_text())
    except (OSError, ValueError):
        lines.append("MISSING the guard's result, result.json (nothing is judged)")
        guard = None
    if guard is not None:
        status = str(guard.get("status"))
        if status.startswith("refused:"):
            return [f"REFUSED the guard: {status}, before launch: nothing was started and nothing is judged"], 1
        g = guard.get("guard") or {}
        lines.append(f"{'PASS' if status == 'child_exited' else 'FAIL'} the guard: status {status}, trigger {g.get('trigger')}, returncode "
                     f"{(guard.get('child') or {}).get('returncode')}; peak pressure {g.get('peak_pressure_bytes')} B, largest sampling gap "
                     f"{g.get('largest_gap_s')} s")
    try:
        result = json.loads((directory / "stream-timing.json").read_text())
    except (OSError, ValueError):
        lines.append("MISSING the check's result, stream-timing.json (nothing is judged)")
        return lines, 1
    lines.append(SCOPE)
    account = read_account(directory)
    if account:
        lines.append(f"USER-REPORTED account: {account}")
        planned = {"walked_in_walking_blocks": "yes", "still_in_still_blocks": "yes"}
        off = [k for k, v in planned.items() if account.get(k) != v]
        if off:
            lines.append(f"CONDITION not as planned ({', '.join(off)}): those blocks' labels are read with that in mind")
    else:
        lines.append("MISSING the operator's account (account.txt): the conditions below are unconfirmed")
    privacy, private = privacy_line(directory)
    lines.append(privacy)
    light = bool(result.get("light_block"))
    state, problems = verdict(result, light)
    lines.append(f"execution: {state}{' (' + '; '.join(problems) + ')' if problems else ''}; status {result.get('status')}; analysis took "
                 f"{result.get('analysis_s')} s")
    sizes = result.get("streams") or {}
    lines.append("frame sizes read from the streams: " + ", ".join(f"{k} {v.get('width')}x{v.get('height')}" for k, v in sizes.items()))
    for record in result.get("slots") or []:
        if "analysis" not in record:
            lines.append(f"{record['name']} {record['condition']}: collected {record.get('collected')} frames, not analysed")
        elif record["name"] == LIGHT_SLOT.name:
            a = record["analysis"]
            lines.append(f"light switch ({a['label']}): steps {a['steps']}, paired {a['paired']}, main-stream lags {a['lags_ms']} ms, median "
                         f"{a['median_ms']} ms, each step resolved no better than +/-{a['resolution_ms']} ms; an independent event with its "
                         "own limits, never pooled with the walking blocks")
        else:
            lines.append(_line_for(record))
    for name, item in (result.get("summary") or {}).items():
        lines.append(f"--- {name}: estimates {[e['lag_ms'] for e in item['estimates']]} ms, apart by {item['between_connections_ms']} ms; largest "
                     f"magnitude inside an interval {item['largest_magnitude_in_an_interval_ms']} ms; no estimate: "
                     f"{[(e['slot'], e['label']) for e in item['no_estimate']]}")
    lines.append(f"temperature and clocks: {result.get('sensors')}")
    lines.extend(LIMITS)
    lines.append(f"stream timing check: execution {state}; no tolerance adopted; no route chosen; alignment estimate only, not proof of "
                 "capture-time synchronization")
    return lines, EXIT[state] if private else EXIT["failed"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--light", choices=("yes", "no"), required=True)
    look = sub.add_parser("read")
    look.add_argument("directory", type=Path)
    cue = sub.add_parser("cues", help="the guard's --cue-at values, one per line")
    cue.add_argument("--light", choices=("yes", "no"), required=True)
    sub.add_parser("guard-duration", help="the guard's --duration-s")
    args = parser.parse_args(argv)
    if args.command == "cues":
        for at, name, text in cues(args.light == "yes"):
            print(f"{at:g}:{name}:{text}")
        return 0
    if args.command == "guard-duration":
        print(f"{GUARD_DURATION_S:g}")
        return 0
    if args.command == "read":
        lines, code = reading(args.directory)
        print("\n".join(lines))
        return code
    return EXIT[run(args)["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
