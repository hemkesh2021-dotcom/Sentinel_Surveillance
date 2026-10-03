"""V2-09/V2-10 demo form: `sentinel track probe` with fake capture and tracker backends (no camera, no GPU)."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from sentinel.cli import main
from sentinel.contracts import PixelFormat
from sentinel.media.capture import CapturedFrame, DecodedFrame
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.tracking.probe import TrackProbe, read_meminfo
from sentinel.tracking.tracker import PersonTracker, RawTrack, TrackerError

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
PLENTY = {"MemFree": 4_000_000_000, "MemAvailable": 5_000_000_000}


class PacedSource:
    """About 100 fps of tiny frames until closed."""

    endpoint = None

    def __init__(self) -> None:
        self.opens = 0
        self._pts = 0

    def open(self) -> None:
        self.opens += 1

    def read(self) -> DecodedFrame:
        time.sleep(0.01)
        self._pts += 66_667
        return DecodedFrame(object(), 640, 480, PixelFormat.BGR, self._pts)

    def close(self) -> None:
        pass


class FakeBackend:
    def __init__(self, *, load_error: TrackerError | None = None, fail_every: int = 0, delay_s: float = 0.0) -> None:
        self.load_error = load_error
        self.fail_every = fail_every
        self.delay_s = delay_s
        self.loaded = False
        self.calls = 0

    def load(self) -> None:
        if self.load_error is not None:
            raise self.load_error
        self.loaded = True

    def track(self, image: object) -> Sequence[RawTrack]:
        assert self.loaded
        self.calls += 1
        time.sleep(self.delay_s)
        if self.fail_every and self.calls % self.fail_every == 0:
            raise RuntimeError("CUDA failure with private detail")
        return [RawTrack(1, 64, 48, 320, 480, 0.8)] if self.calls % 2 else []

    def reset(self) -> None:
        pass


def run_cli(backend: FakeBackend, *, meminfo: dict[str, int] | None = PLENTY, seconds: str = "1") -> tuple[int, PacedSource]:
    source = PacedSource()
    code = main(
        ["track", "probe", str(DEFAULT_CONFIG), "--engine", "/models/yolov8n.engine", "--seconds", seconds],
        capture_source=lambda config: source,
        tracker_backend=lambda engine: backend,
        meminfo=lambda: meminfo,
    )
    return code, source


def test_meminfo_reads_free_and_available_bytes(tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    path.write_text("MemTotal:  7802744 kB\nMemFree:   3000000 kB\nMemAvailable:  4500000 kB\nCached: 1 kB\n")
    assert read_meminfo(path) == {"MemFree": 3_072_000_000, "MemAvailable": 4_608_000_000}
    path.write_text("MemTotal:  7802744 kB\n")
    assert read_meminfo(path) is None
    assert read_meminfo(tmp_path / "absent") is None


def test_track_probe_counts_outcomes_tracks_and_timings() -> None:
    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    backend = FakeBackend(fail_every=5)
    backend.load()
    probe = TrackProbe(PersonTracker(backend), clock)
    frames = []
    for _ in range(7):
        clock.advance(0.066)
        frames.append(stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR))
    for frame in frames + [frames[2]]:  # the last is a repeat: skipped
        probe.observe(CapturedFrame(frame, object()))
    summary = probe.summary()
    tracking = summary["tracking"]
    assert (tracking["processed"], tracking["failed"], tracking["skipped"]) == (6, 1, 1)
    assert tracking["failures"] == {"backend_error": 1} and tracking["error_types"] == {"RuntimeError": 1}
    assert tracking["failure_resets"] == 1
    # Calls 1 and 3 detect backend ID 1; call 5 fails; call 7's ID 1 is published as 2 after the reset.
    assert summary["persons"] == {"frames_with_persons": 3, "max_per_frame": 1, "track_ids": 2, "confirmed_track_ids": 0}
    assert summary["backend_ms"]["count"] == 7 and summary["result_age_ms"]["count"] == 6
    assert "private" not in json.dumps(summary)


def test_cli_track_probe_loads_the_backend_and_prints_a_numbers_only_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = FakeBackend()
    code, source = run_cli(backend)
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and source.opens == 1
    assert summary["probe"] == "track" and summary["status"] == "frames_received"
    assert summary["tracking"]["processed"] > 10 and summary["tracking"]["failed"] == 0
    assert summary["persons"]["frames_with_persons"] > 0
    assert summary["load"]["memory_before"] == PLENTY and summary["load"]["min_free_bytes"] == 1_500_000_000
    assert summary["frames"]["captured"] >= summary["tracking"]["processed"]
    assert "yolov8n" not in json.dumps(summary)


def test_a_slow_detector_shows_as_replaced_frames_not_a_queue(capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = run_cli(FakeBackend(delay_s=0.05))
    summary = json.loads(capsys.readouterr().out)
    frames = summary["frames"]
    assert code == 0 and frames["replaced"] > 0
    assert frames["captured"] == frames["delivered"] + frames["replaced"] + frames["discarded"]
    assert summary["backend_ms"]["p50"] >= 50


def test_cli_track_probe_refusals_load_nothing_and_open_no_camera(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend()
    code, source = run_cli(backend, meminfo={"MemFree": 1_000_000_000, "MemAvailable": 4_000_000_000})
    refusal = json.loads(capsys.readouterr().out)
    assert code == 1 and refusal["status"] == "refused" and refusal["reason"] == "memfree_below_minimum"
    assert not backend.loaded and source.opens == 0

    code, source = run_cli(FakeBackend(load_error=TrackerError("engine_hash_mismatch")))
    assert code == 1 and source.opens == 0
    assert capsys.readouterr().err.strip() == "track probe: engine_hash_mismatch"
    code, _ = run_cli(FakeBackend(load_error=TrackerError("load_failed", "RuntimeError")))
    assert capsys.readouterr().err.strip() == "track probe: load_failed (RuntimeError)"

    monkeypatch.delenv("SENTINEL_RTSP_URL", raising=False)
    constructed = []
    code = main(
        ["track", "probe", str(DEFAULT_CONFIG), "--engine", "/models/yolov8n.engine"],
        tracker_backend=lambda engine: constructed.append(engine),
        meminfo=lambda: PLENTY,
    )
    assert code == 1 and constructed == []
    assert capsys.readouterr().err.strip() == "track probe: rtsp_url_missing"

    with pytest.raises(SystemExit):
        main(["track", "probe", str(DEFAULT_CONFIG), "--engine", "x", "--min-free-gb", "0.1"])
    with pytest.raises(SystemExit):
        main(["track", "probe", str(DEFAULT_CONFIG)])  # --engine is required


def test_tracking_failures_make_the_probe_exit_non_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = run_cli(FakeBackend(fail_every=3))
    summary = json.loads(capsys.readouterr().out)
    assert code == 1 and summary["tracking"]["failed"] > 0
    assert summary["tracking"]["failures"] == {"backend_error": summary["tracking"]["failed"]}
