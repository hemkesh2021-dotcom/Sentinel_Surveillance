"""Session 116, part 14: the two-stream timing check (benchmarks/runner/stream_timing_check.py). No camera and no model: simulated
streams with a known lag. What is tested:
- **The estimate:** a known lag is recovered with its sign, inside its interval, through different frame rates, jitter and lost
  frames.
- **Ambiguity:** repeating, unrelated or absent movement, a lag beyond the search range and too few frames never give an
  estimate; the label says why, and execution is still complete.
- **The session:** T0 once, after both streams; a new connection per block; the load only where declared and always stopped;
  conditions never pooled; a guard's stop keeps what was collected.
- **Numbers only:** no grid or frame saved; no URL anywhere; no tolerance and no route in the reading.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel.media.clock import NS_PER_SECOND, FakeClock

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
URLS = {"SENTINEL_RTSP_URL": "rtsp://operator:hunter2@192.0.2.10:554/sub",
        "SENTINEL_RTSP_MAIN_URL": "rtsp://operator:hunter2@192.0.2.10:554/main"}
CELLS = 8  # fewer than the real grid, to keep the tests quick; the code takes the grid's own length
FPS = {"sub": 15.0, "main": 20.0}
SIZES = {"sub": (640, 480), "main": (2304, 1296)}


@pytest.fixture(scope="module")
def st():
    sys.path.insert(0, str(RUNNERS))
    try:
        yield importlib.import_module("stream_timing_check")
    finally:
        sys.path.remove(str(RUNNERS))


def wander(t: float) -> float:
    """A person's position: smooth and not periodic within the search range."""
    return 0.5 + 0.28 * math.sin(2 * math.pi * t / 7.3) + 0.16 * math.sin(2 * math.pi * t / 3.1 + 1.0) + 0.05 * math.sin(2 * math.pi * t / 1.37)


def repeat(t: float) -> float:
    """A strictly repeating movement, with a period inside the search range."""
    return 0.5 + 0.4 * math.sin(2 * math.pi * t / 1.2)


def still(t: float) -> float:
    return 0.5


def grid_at(position: float, rng: random.Random, noise: float = 0.3) -> tuple[float, ...]:
    """Mean brightness per cell of a picture with a dark person at ``position`` across it."""
    return tuple(120.0 - 60.0 * math.exp(-(((i + 0.5) / CELLS - position) / 0.09) ** 2) + rng.gauss(0, noise) for i in range(CELLS))


def track_of(st, path, fps: float, latency: float, start: float, end: float, *, seed: int = 1, jitter: float = 0.003, drop: float = 0.0,
             noise: float = 0.3):
    """A stream's Track: frames captured at ``fps``, each reaching ingest ``latency`` later, plus jitter."""
    rng = random.Random(seed)
    track = st.Track()
    seq, t = 0, start
    while t < end:
        seq += 1
        if rng.random() >= drop:
            track.add(t + latency + rng.uniform(0, jitter), round(t * 1e6), seq, 1, grid_at(path(t), rng, noise))
        t += 1 / fps
    return track


def estimate(st, sub_path=wander, main_path=None, lag: float = 0.0, **options):
    sub = track_of(st, sub_path, 15.0, 0.1, 0.0, 45.0, seed=1, **options)
    main = track_of(st, main_path or sub_path, 20.0, 0.1 + lag, 0.0, 45.0, seed=2, **options)
    return st.estimate_lag(sub, main, 2.0, 43.0)


# ---------------------------------------------------------------- the estimate


@pytest.mark.parametrize("lag_ms", [-300, 0, 120, 180, 700])
def test_a_known_lag_is_recovered_with_its_sign_inside_its_interval(st, lag_ms) -> None:
    got = estimate(st, lag=lag_ms / 1000)
    assert abs(got["lag_ms"] - lag_ms) <= 40.0
    assert got["interval_ms"][0] - 20.0 <= lag_ms <= got["interval_ms"][1] + 20.0
    assert got["second_ratio"] is None or got["second_ratio"] < st.SECOND_PEAK_RATIO
    assert got["peak"] > st.PEAK_OVER_NULL * got["null_peak"]
    assert got["blocks_agreeing"] >= got["blocks"] - 2
    assert got["largest_magnitude_ms"] == max(abs(v) for v in got["interval_ms"])


def test_jitter_widens_nothing_into_a_false_lag(st) -> None:
    got = estimate(st, lag=0.15, jitter=0.04)
    assert abs(got["lag_ms"] - 170) <= 60.0  # the mean of 0-40 ms added to both; the frames' own jitter stays small


def test_lost_frames_do_not_move_the_estimate(st) -> None:
    got = estimate(st, lag=0.2, drop=0.1)
    assert abs(got["lag_ms"] - 200) <= 40.0


def test_a_repeating_movement_is_flagged_never_estimated(st) -> None:
    got = estimate(st, sub_path=repeat, lag=0.2)
    label, reasons = st.classify(ok_frames(st), {"sub": 9.0, "main": 9.0}, got)
    assert label != "estimated" and "ambiguous_multiple_peaks" in reasons


def test_unrelated_movement_is_flagged_never_estimated(st) -> None:
    def other(t: float) -> float:
        return 0.5 + 0.3 * math.sin(2 * math.pi * t / 5.3 + 2.0) + 0.1 * math.sin(2 * math.pi * t / 2.2)

    got = estimate(st, sub_path=wander, main_path=other)
    label, reasons = st.classify(ok_frames(st), {"sub": 9.0, "main": 9.0}, got)
    assert label != "estimated" and set(reasons) & {"ambiguous_low_peak", "ambiguous_multiple_peaks", "ambiguous_wide_interval"}


def test_a_lag_beyond_the_search_range_never_reads_as_a_good_estimate(st) -> None:
    got = estimate(st, lag=3.0)
    label, _ = st.classify(ok_frames(st), {"sub": 9.0, "main": 9.0}, got)
    assert label != "estimated"


def test_no_change_at_all_gives_no_signal(st) -> None:
    got = estimate(st, sub_path=still, noise=0.0)
    assert got == {"lag_ms": None, "peak": None, "reason": "no_signal"}
    label, reasons = st.classify(ok_frames(st), {"sub": 1.0, "main": 1.0}, got)
    assert label == "insufficient_motion" and reasons == ["insufficient_motion", "ambiguous_low_peak"]


def ok_frames(st):
    return {"sub": {"share": 1.0}, "main": {"share": 1.0}}


@pytest.mark.parametrize("frames, ratio, change, reason", [
    ({"sub": {"share": 0.5}, "main": {"share": 1.0}}, {"sub": 9.0, "main": 9.0}, {}, "too_few_frames"),
    ({"sub": {"share": 1.0}, "main": {"share": 1.0}}, {"sub": 2.9, "main": 9.0}, {}, "insufficient_motion"),
    ({"sub": {"share": 1.0}, "main": {"share": 1.0}}, {"sub": None, "main": 9.0}, {}, "insufficient_motion"),
    ({"sub": {"share": 1.0}, "main": {"share": 1.0}}, {"sub": 9.0, "main": 9.0}, {"at_edge": True}, "ambiguous_at_search_edge"),
    ({"sub": {"share": 1.0}, "main": {"share": 1.0}}, {"sub": 9.0, "main": 9.0}, {"null_peak": 0.5}, "ambiguous_low_peak"),
    ({"sub": {"share": 1.0}, "main": {"share": 1.0}}, {"sub": 9.0, "main": 9.0}, {"second_ratio": 0.85}, "ambiguous_multiple_peaks"),
    ({"sub": {"share": 1.0}, "main": {"share": 1.0}}, {"sub": 9.0, "main": 9.0}, {"interval_ms": [0.0, 201.0]}, "ambiguous_wide_interval"),
])
def test_each_label_has_its_predeclared_reason(st, frames, ratio, change, reason) -> None:
    good = {"lag_ms": 100.0, "peak": 0.6, "null_peak": 0.1, "second_ratio": 0.2, "interval_ms": [80.0, 120.0], "at_edge": False}
    assert st.classify(ok_frames(st), {"sub": 9.0, "main": 9.0}, good) == ("estimated", [])
    label, reasons = st.classify(frames, ratio, {**good, **change})
    assert label == reason and reason in reasons


def test_a_pair_across_a_reconnection_is_not_bridged(st) -> None:
    track = st.Track()
    track.add(1.0, None, 1, 1, (10.0,) * CELLS)
    track.add(1.05, None, 2, 1, (11.0,) * CELLS)
    track.add(1.1, None, 1, 2, (90.0,) * CELLS)  # another epoch
    track.add(9.0, None, 2, 2, (91.0,) * CELLS)  # a long gap
    times, values = st._differences(track, [0, 1, 2, 3])
    assert len(times) == 1 and values[0][0] == pytest.approx(1.0 / 0.05)


def test_delivery_jitter_is_within_a_stream_and_shows_a_stall_and_a_drift(st) -> None:
    times = [i * 0.05 for i in range(100)]
    pts = [round(i * 0.05 * 1e6) for i in range(100)]
    even = st.delivery_jitter(times, pts)
    assert even["p95"] == pytest.approx(0.0, abs=0.1) and even["drift_ms_per_min"] == pytest.approx(0.0, abs=0.1)
    stalled = st.delivery_jitter([t + (0.3 if 50 <= i < 55 else 0.0) for i, t in enumerate(times)], pts)
    assert stalled["max"] == pytest.approx(300.0, abs=0.5) and stalled["p50"] == pytest.approx(0.0, abs=0.1)
    drifting = st.delivery_jitter([t * 1.01 for t in times], pts)
    assert drifting["drift_ms_per_min"] == pytest.approx(594.0, abs=5.0)  # 1 % per second of the stretched clock
    assert st.delivery_jitter(times, [None] * 100) is None


def test_the_peak_description_reads_width_and_a_second_peak(st) -> None:
    curve = [0.0, 0.1, 0.8, 1.0, 0.8, 0.1, 0.0, 0.1, 0.9, 0.1, 0.0]
    got = st.describe_peak(curve, 20.0, 5)
    assert got["lag_ms"] == -40.0 and got["peak"] == 1.0 and got["width_ms"] == 0.0 and got["second_ratio"] == 0.9 and not got["at_edge"]
    assert st.describe_peak([1.0, 0.5, 0.0], 20.0, 1)["at_edge"] is True


def test_the_light_steps_are_paired_and_a_still_scene_has_none(st) -> None:
    def tracks(lag: float, toggles):
        out = {}
        for stream, fps, latency in (("sub", 15.0, 0.1), ("main", 20.0, 0.1 + lag)):
            track = st.Track()
            t, seq = 5.0, 0
            rng = random.Random(4)
            while t < 48.0:
                seq += 1
                level = 100.0 + 30.0 * sum(1 for at in toggles if t >= at) % 60.0
                track.add(t + latency, None, seq, 1, tuple(level + rng.gauss(0, 0.1) for _ in range(CELLS)))
                t += 1 / fps
            out[stream] = track
        return out

    got = st.analyse_light(tracks(0.25, (10.013, 20.041, 30.007, 40.029)))
    assert got["label"] == "steps_found" and got["paired"] == 4
    assert abs(got["median_ms"] - 250.0) <= st.STEP_RESOLUTION_MS and got["resolution_ms"] == st.STEP_RESOLUTION_MS
    none = st.analyse_light(tracks(0.25, ()))
    assert none["label"] == "no_usable_steps" and none["paired"] == 0 and none["median_ms"] is None


# ---------------------------------------------------------------- the session, on simulated streams


class Sim:
    """Two simulated streams on a fake clock: frames are captured at each stream's rate and arrive at their latency for that
    connection. Times are exact: frames carry their own ingest stamps, as the capture thread gives them."""

    def __init__(self, clock: FakeClock, lags, *, silent=(), sizes=SIZES, paths=None) -> None:
        self.clock, self.lags, self.silent, self.sizes = clock, list(lags), set(silent), sizes
        self.paths = paths or {}
        self.running: dict[str, int] = {}
        self.started: list[tuple[str, int]] = []
        self.connection = -1
        self.next_capture: dict[str, int] = {}
        self.seq = {s: 0 for s in FPS}
        self.epoch = {s: 0 for s in FPS}
        self.rng = {s: random.Random(i + 7) for i, s in enumerate(FPS)}
        self.t0_ns: int | None = None

    def start(self, stream: str, threads: int) -> None:
        assert stream not in self.running
        if stream == "sub":
            self.connection += 1
        self.running[stream] = threads
        self.started.append((stream, threads))
        self.epoch[stream] += 1
        self.next_capture[stream] = self.clock.monotonic_ns() + 500_000_000

    def stop(self, stream: str) -> bool:
        del self.running[stream]
        return True

    def stop_all(self) -> bool:
        for stream in list(self.running):
            self.stop(stream)
        return True

    def _latency_ns(self, stream: str) -> int:
        base = 100_000_000
        return base if stream == "sub" else base + round(self.lags[min(self.connection, len(self.lags) - 1)] * 1e9)

    def _position(self, capture_ns: int) -> float:
        if self.t0_ns is None:
            return 0.5
        t = (capture_ns - self.t0_ns) / NS_PER_SECOND
        slot = int(t // 80) if t < 480 else 6
        offset = t - slot * 80
        path = self.paths.get(slot, wander)
        return path(t) if 25.0 <= offset < 70.0 else path(25.0 + slot * 80)

    def take(self, stream: str, timeout_s: float):
        assert stream in self.running
        now = self.clock.monotonic_ns()
        if stream in self.silent:
            self.clock.advance(timeout_s)
            return None
        capture = self.next_capture[stream]
        ingest = capture + self._latency_ns(stream)
        if ingest > now + round(timeout_s * 1e9):
            self.clock.advance(timeout_s)
            return None
        if ingest > now:
            self.clock.advance((ingest - now) / 1e9)
        self.next_capture[stream] = capture + round(1e9 / FPS[stream])
        self.seq[stream] += 1
        width, height = self.sizes[stream]
        frame = SimpleNamespace(native_width=width, native_height=height, ingest_mono_ns=ingest, source_pts=(capture // 1000),
                                frame_seq=self.seq[stream], stream_epoch=self.epoch[stream], boot_id="b", run_id=f"r-{stream}")
        return SimpleNamespace(frame=frame, image=grid_at(self._position(capture), self.rng[stream]))

    def counts(self, stream: str):
        return {"hand_off": {}, "state": "streaming", "frames": self.seq[stream], "connects": self.epoch[stream], "open_failures": 0,
                "stream_ends": 0, "problem": "no_frame" if stream in self.silent else None}

    def connections(self) -> int:
        return len(self.running)


def session(st, tmp_path, *, lags=(0.18,), light="no", silent=(), sizes=SIZES, paths=None, interrupt_at=None, environ=None, spawn=None):
    clock = FakeClock()
    sim = Sim(clock, lags, silent=silent, sizes=sizes, paths=paths)
    lines: list[str] = []
    loads: list[dict] = []

    def emit(line: str) -> None:
        lines.append(line)
        if json.loads(line).get("run") == "starting":
            sim.t0_ns = clock.monotonic_ns()

    def sleep(seconds: float) -> None:
        clock.advance(seconds)
        if interrupt_at is not None and sim.t0_ns is not None and (clock.monotonic_ns() - sim.t0_ns) / NS_PER_SECOND >= interrupt_at:
            raise KeyboardInterrupt

    def fake_spawn(seconds: float):
        record = {"seconds": seconds, "stopped": False, "at_s": (clock.monotonic_ns() - sim.t0_ns) / NS_PER_SECOND}
        loads.append(record)
        return [record]

    def fake_stop(handles) -> None:
        for handle in handles:
            handle["stopped"] = True

    args = argparse.Namespace(out=tmp_path / "stream-timing.json", light=light)
    result = st.run(args, environ=URLS if environ is None else environ, clock=clock, streams=sim, emit=emit, sleep=sleep,
                    reduce=lambda grid: grid, spawn=spawn or fake_spawn, stop=fake_stop,
                    sensors=lambda: {"zones_c": {"cpu-thermal": 50.0}, "cpu_khz": [1420800], "gpu_hz": 306000000},
                    memory=lambda: {"VmRSS": 1})
    return SimpleNamespace(result=result, lines=lines, sim=sim, loads=loads, out=args.out, series=tmp_path / "stream-series.json")


@pytest.fixture(scope="module")
def whole(st, tmp_path_factory):
    """One whole simulated session with a different lag per connection; shared because it runs the full analysis."""
    return session(st, tmp_path_factory.mktemp("whole"), lags=(0.14, 0.20, 0.16, 0.24, 0.30, 0.36, 0.20), light="no")


def test_a_whole_session_gives_each_connection_its_own_estimate(st, whole) -> None:
    result = whole.result
    assert result["verdict"] == "complete" and result["status"] == "finished" and result["problems"] == []
    want = {"c1": 140, "c2": 200, "c3": 160, "c4": 240, "l1": 300, "l2": 360}
    for record in result["slots"]:
        analysis = record["analysis"]
        assert analysis["label"] == "estimated", record["name"]
        assert abs(analysis["lag_ms"] - want[record["name"]]) <= 40.0
        assert analysis["interval_ms"][0] - 20 <= want[record["name"]] <= analysis["interval_ms"][1] + 20
        assert analysis["motion_ratio"]["sub"] >= st.MIN_MOTION_RATIO
        assert analysis["frames"]["main"]["fps"] == pytest.approx(20.0, abs=0.5)


def test_conditions_are_summarized_apart_never_pooled(st, whole) -> None:
    summary = whole.result["summary"]
    assert set(summary) == {"quiet_main_threads_1", "quiet_main_threads_2", "load_main_threads_1", "load_main_threads_2"}
    assert len(summary["quiet_main_threads_1"]["estimates"]) == 2 and len(summary["load_main_threads_2"]["estimates"]) == 1
    assert summary["quiet_main_threads_1"]["between_connections_ms"] == pytest.approx(20.0, abs=30.0)
    assert summary["load_main_threads_2"]["between_connections_ms"] is None
    assert summary["load_main_threads_2"]["largest_magnitude_in_an_interval_ms"] >= 300


def test_readiness_is_once_and_after_both_streams(st, whole) -> None:
    assert json.loads(whole.lines[0]) == {"run": "starting", "startup": {"tool": "stream_timing_check"}}
    assert sum('"starting"' in line for line in whole.lines) == 1
    assert whole.result["streams"]["main"]["width"] == 2304


def test_every_block_is_a_new_connection_with_its_declared_decoder_setting(st, whole) -> None:
    pairs = whole.sim.started
    assert pairs[::2] == [("sub", 1)] * 6
    assert [threads for _, threads in pairs[1::2]] == [1, 2, 1, 2, 1, 2]
    assert [r["connections"] for r in whole.result["slots"]] == [2] * 6
    assert whole.sim.running == {} and whole.result["workers_stopped"] is True


def test_the_load_runs_only_in_the_load_blocks_and_is_stopped(st, whole) -> None:
    assert [round(load["at_s"]) for load in whole.loads] == [325, 405]
    assert all(load["stopped"] and load["seconds"] == 70.0 for load in whole.loads)
    assert {r["name"]: r["load"] for r in whole.result["slots"]} == {"c1": False, "c2": False, "c3": False, "c4": False, "l1": True, "l2": True}


def test_no_grid_frame_or_url_is_saved(st, whole) -> None:
    text = whole.out.read_text() + whole.series.read_text() + "".join(whole.lines)
    for private in ("hunter2", "192.0.2.10", "operator:", '"grid"', '"grids"'):
        assert private not in text.replace('"grid": [', "")  # only the method's declared grid size
    series = json.loads(whole.series.read_text())
    row = series["c1"]["main"][100]
    assert len(row) == 3 and all(isinstance(v, (int, float)) for v in row)  # a time and two numbers, never cells
    assert whole.out.stat().st_mode & 0o777 == 0o600


def test_sensors_are_recorded_and_summarized(st, whole) -> None:
    sensors = whole.result["sensors"]
    assert sensors["samples"] > 100 and sensors["max_c"] == {"cpu-thermal": 50.0} and sensors["gpu_hz"] == [306000000, 306000000]


def test_source_saves_no_image(st) -> None:
    source = Path(st.__file__).read_text()
    for call in ("imwrite", "VideoWriter", "imencode", '"wb"', "tofile", "numpy.save", "DeepFace", "FaceDetectorYN"):
        assert call not in source


def test_an_ambiguous_block_gives_no_estimate_and_the_run_is_still_complete(st, tmp_path) -> None:
    got = session(st, tmp_path, lags=(0.18,), paths={0: repeat, 1: still})
    result = got.result
    assert result["verdict"] == "complete"
    labels = {r["name"]: r["analysis"]["label"] for r in result["slots"]}
    assert labels["c1"] != "estimated" and labels["c2"] == "insufficient_motion" and labels["c3"] == "estimated"
    assert [e["slot"] for e in result["summary"]["quiet_main_threads_1"]["estimates"]] == ["c3"]
    assert {e["slot"] for e in result["summary"]["quiet_main_threads_1"]["no_estimate"]} == {"c1"}


def test_the_light_block_is_only_there_when_asked_for(st, whole) -> None:
    assert [r["name"] for r in whole.result["slots"]] == ["c1", "c2", "c3", "c4", "l1", "l2"]


@pytest.mark.parametrize("environ, label", [
    ({"SENTINEL_RTSP_URL": URLS["SENTINEL_RTSP_URL"]}, "url_missing:main"),
    ({**URLS, "SENTINEL_RTSP_MAIN_URL": "http://nope"}, "url_invalid:main"),
    ({**URLS, "SENTINEL_RTSP_MAIN_URL": URLS["SENTINEL_RTSP_URL"]}, "urls_identical"),
])
def test_bad_urls_are_refused_by_label_only(st, tmp_path, environ, label) -> None:
    got = session(st, tmp_path, environ=environ)
    assert got.result["status"] == f"refused:{label}" and got.result["verdict"] == "failed" and got.sim.started == []
    assert "hunter2" not in got.out.read_text() + "".join(got.lines)


def test_a_silent_stream_is_refused_before_t0(st, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(st, "READY_TIMEOUT_S", 2.0)
    got = session(st, tmp_path, silent=("main",))
    assert got.result["status"] == "refused:stream_not_ready:main:no_frame" and not any('"starting"' in line for line in got.lines)
    assert got.sim.running == {}


def test_swapped_urls_are_refused_before_t0(st, tmp_path) -> None:
    got = session(st, tmp_path, sizes={"sub": (2304, 1296), "main": (640, 480)})
    assert got.result["status"] == "refused:main_stream_not_larger_than_substream" and not any('"starting"' in line for line in got.lines)


def test_the_guards_stop_keeps_what_was_collected_and_stops_the_streams(st, tmp_path) -> None:
    got = session(st, tmp_path, interrupt_at=79.0)
    result = got.result
    assert result["status"] == "stopped" and got.sim.running == {} and result["workers_stopped"] is True
    assert [r["name"] for r in json.loads(got.out.read_text())["slots"]] == ["c1"] and result["verdict"] == "incomplete"
    assert "analysis" not in result["slots"][0]  # nothing is forced after a stop


def test_a_stop_during_a_load_block_stops_the_load(st, tmp_path, monkeypatch) -> None:
    original = st._collect

    def collect(session_, tracks, t0_ns, slot, until_s, series):
        if slot.load and until_s > slot.start_s + st.SETTLE_S:
            raise KeyboardInterrupt
        return original(session_, tracks, t0_ns, slot, until_s, series)

    monkeypatch.setattr(st, "_collect", collect)
    got = session(st, tmp_path)
    assert got.result["status"] == "stopped" and len(got.loads) == 1 and got.loads[0]["stopped"] and got.sim.running == {}


# ---------------------------------------------------------------- the operator's cues and the real helpers


@pytest.mark.parametrize("light", [False, True])
def test_the_cues_fit_the_guard_and_leave_each_block_inside_them(st, light) -> None:
    import step5_guard

    cues = st.cues(light)
    assert all(step5_guard._cue_at(f"{at:g}:{name}:{text}") == (at, name, text) for at, name, text in cues)
    assert [c[0] for c in cues] == sorted(c[0] for c in cues) and cues[0][1] == "t0" and cues[-1][1] == "stop"
    assert len({c[1] for c in cues}) == len(cues)
    for slot in st.SLOTS:
        at = {round(c[0]) for c in cues}
        assert slot.start_s + st.WALK_CUES[0] in at and slot.start_s + st.WALK_CUES[1] in at
        assert st.WALK_CUES[0] < st.WALK_WINDOW[0] < st.WALK_WINDOW[1] < st.WALK_CUES[1] and st.STILL_WINDOW[1] <= st.WALK_CUES[0]
    assert cues[-1][0] <= st.GUARD_DURATION_S - st.ANALYSIS_BOUND_S
    assert ("switch1" in {c[1] for c in cues}) == light


def test_the_command_line_prints_the_guards_values(st, capsys) -> None:
    assert st.main(["cues", "--light", "no"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("0:t0:T0 - ") and printed[-1].startswith("470:stop:")
    assert st.main(["guard-duration"]) == 0
    assert float(capsys.readouterr().out) == st.GUARD_DURATION_S


def test_the_real_load_ends_by_itself_and_can_be_stopped(st, monkeypatch) -> None:
    monkeypatch.setattr(st, "LOAD_PROCESSES", 2)
    processes = st.spawn_load(0.3)
    deadline = time.monotonic() + 10
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert all(p.poll() == 0 for p in processes)
    long_running = st.spawn_load(60.0)
    st.stop_load(long_running)
    assert all(p.poll() is not None for p in long_running)


def test_sensors_read_world_readable_files_and_skip_what_cannot_be_read(st, tmp_path) -> None:
    thermal, cpu = tmp_path / "thermal", tmp_path / "cpu"
    for index, (name, temp) in enumerate((("cpu-thermal", "52875"), ("cv0-thermal", None))):
        zone = thermal / f"thermal_zone{index}"
        zone.mkdir(parents=True)
        (zone / "type").write_text(name + "\n")
        if temp:
            (zone / "temp").write_text(temp + "\n")
    (cpu / "cpu0" / "cpufreq").mkdir(parents=True)
    (cpu / "cpu0" / "cpufreq" / "scaling_cur_freq").write_text("1420800\n")
    gpu = tmp_path / "gpu"
    gpu.write_text("306000000\n")
    got = st.read_sensors(thermal, cpu, gpu)
    assert got == {"zones_c": {"cpu-thermal": 52.875}, "cpu_khz": [1420800], "gpu_hz": 306000000}
    assert st.read_sensors(thermal, cpu, tmp_path / "missing")["gpu_hz"] is None
    assert st.sensor_summary([]) is None


def test_the_real_grid_on_real_arrays_when_opencv_is_here(st) -> None:
    pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    frame = numpy.zeros((1296, 2304, 3), numpy.uint8)
    frame[:, 1152:, 1] = 200
    got = st.coarse_grid(frame)
    assert len(got) == st.GRID[0] * st.GRID[1]
    assert got[0] == pytest.approx(0.0, abs=1.0) and got[5] == pytest.approx(200.0, abs=1.0)


# ---------------------------------------------------------------- the reading


GUARD = {"status": "child_exited", "child": {"returncode": 0}, "guard": {"trigger": None, "peak_pressure_bytes": 1_400_000_000, "largest_gap_s": 0.2}}
PLANNED = {"walked_in_walking_blocks": "yes", "still_in_still_blocks": "yes", "others_in_view": "no"}


def folder(st, tmp_path, whole, *, account=PLANNED, guard=GUARD, counts=True) -> Path:
    (tmp_path / "stream-timing.json").write_text(whole.out.read_text())
    (tmp_path / "result.json").write_text(json.dumps(guard))
    if account is not None:
        (tmp_path / "account.txt").write_text("".join(f"{k}={v}\n" for k, v in account.items()))
    if counts:
        (tmp_path / "secret-counts-camera.txt").write_text("stream-timing.json userinfo=0\n")
    return tmp_path


def test_the_reading_says_what_it_is_and_what_it_is_not(st, tmp_path, whole) -> None:
    lines, code = st.reading(folder(st, tmp_path, whole))
    text = "\n".join(lines)
    assert code == 0 and "PASS the guard" in text and "execution: complete" in text and "PASS privacy" in text
    assert "estimated main-stream lag" in text and "--- quiet_main_threads_1" in text and "temperature and clocks" in text
    assert "not proof of capture-time synchronization" in text and "no tolerance adopted" in text and "no route chosen" in text
    assert all(limit in text for limit in st.LIMITS)
    assert lines[-1] == ("stream timing check: execution complete; no tolerance adopted; no route chosen; alignment estimate only, not proof of "
                         "capture-time synchronization")
    for claim in ("are synchronized", "tolerance met", "within tolerance", "route chosen:"):
        assert claim not in text


def test_the_reading_shows_an_ambiguous_block_as_no_estimate(st, tmp_path) -> None:
    got = session(st, tmp_path, paths={0: repeat})
    lines, code = st.reading(folder(st, tmp_path, got))
    line = next(line for line in lines if line.startswith("c1 "))
    assert code == 0 and "NO ESTIMATE" in line and "candidate only" in line and "estimated main-stream lag" not in line


def test_the_reading_flags_an_account_that_differs_or_is_missing(st, tmp_path, whole) -> None:
    text = "\n".join(st.reading(folder(st, tmp_path, whole, account={**PLANNED, "still_in_still_blocks": "no"}))[0])
    assert "CONDITION not as planned (still_in_still_blocks)" in text
    (tmp_path / "account.txt").unlink()
    assert "MISSING the operator's account" in "\n".join(st.reading(tmp_path)[0])


def test_the_reading_of_a_refused_launch_judges_nothing(st, tmp_path, whole) -> None:
    lines, code = st.reading(folder(st, tmp_path, whole, guard={"status": "refused:llama_server_running"}))
    assert code == 1 and lines == ["REFUSED the guard: refused:llama_server_running, before launch: nothing was started and nothing is judged"]


def test_the_reading_of_a_missing_result_judges_nothing(st, tmp_path) -> None:
    (tmp_path / "result.json").write_text(json.dumps({"status": "guard_stop", "child": {}, "guard": {"trigger": "x"}}))
    lines, code = st.reading(tmp_path)
    assert code == 1 and lines[0].startswith("FAIL the guard") and "MISSING the check's result" in lines[1]


def test_a_privacy_hit_fails_the_reading(st, tmp_path, whole) -> None:
    root = folder(st, tmp_path, whole)
    (root / "secret-counts-camera-main.txt").write_text("run.err userinfo=1\n")
    lines, code = st.reading(root)
    assert code == 1 and any(line.startswith("FAIL privacy") and "run.err" in line for line in lines)


def test_an_interrupted_session_reads_incomplete(st, tmp_path) -> None:
    got = session(st, tmp_path, interrupt_at=79.0)
    lines, code = st.reading(folder(st, tmp_path, got, guard={"status": "guard_stop", "child": {}, "guard": {"trigger": "x"}}))
    text = "\n".join(lines)
    assert code == 3 and "execution: incomplete" in text and "collected, not analysed" in text
