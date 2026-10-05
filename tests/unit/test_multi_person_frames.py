"""Session 36: `sentinel track probe --multi-person-frames`, the bounded records of frames with two or more persons.

Synthetic frames and scripted tracker results only: no camera, GPU, model or
image. The records must describe exactly the counted frames, stay bounded, hold
no image, and leave every existing count unchanged.
"""

from __future__ import annotations

import gc
import json
import time
import weakref
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from sentinel.cli import main
from sentinel.contracts import NormalizedBox, PixelFormat
from sentinel.media.capture import CapturedFrame, DecodedFrame
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.tracking.box_summary import BoxSummary
from sentinel.tracking.multi_person import MAX_SAMPLES, MultiPersonFrames, overlap
from sentinel.tracking.probe import TrackProbe
from sentinel.tracking.tracker import PersonTracker, RawTrack, TrackingResult

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
PLENTY = {"MemFree": 4_000_000_000, "MemAvailable": 5_000_000_000}
# Pixels of a 640x480 image; normalized: A [0.1, 0.1, 0.5, 1.0], B [0.2, 0.2, 0.4, 0.9] (inside A),
# C [0.7, 0.0, 1.0, 0.5] (apart from both).
A = RawTrack(1, 64, 48, 320, 480, 0.91)
B = RawTrack(5, 128, 96, 256, 432, 0.43)
C = RawTrack(9, 448, 0, 640, 240, 0.6)


class Image:
    """A stand-in for a decoded frame that can be weakly referenced."""

    shape = (480, 640, 3)


class ScriptedBackend:
    def __init__(self, script: list[object]) -> None:
        self.script = list(script)

    def load(self) -> None:
        pass

    def track(self, image: object) -> Sequence[RawTrack]:
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]

    def reset(self) -> None:
        pass


def probe_with(script: list[object], observers: Sequence[Callable[[CapturedFrame, Any], None]] = (),
               *, repeat: int | None = None) -> tuple[TrackProbe, list[CapturedFrame]]:
    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    probe = TrackProbe(PersonTracker(ScriptedBackend(script)), clock, observers)
    captured = []
    for index in range(len(script)):
        clock.advance(0.4)
        frame = stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR,
                              source_pts=(index + 1) * 66_667)
        captured.append(CapturedFrame(frame, Image()))
    for item in captured:
        probe.observe(item)
    if repeat is not None:
        probe.observe(captured[repeat])  # not newer: skipped, so neither counted nor recorded
    return probe, captured


def ms(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def test_overlap_is_iou_and_the_share_of_the_smaller_box() -> None:
    left = NormalizedBox(x1=0.0, y1=0.0, x2=0.5, y2=1.0)
    assert overlap(left, left) == (1.0, 1.0)
    iou, of_smaller = overlap(left, NormalizedBox(x1=0.25, y1=0.0, x2=0.75, y2=1.0))
    assert iou == pytest.approx(1 / 3) and of_smaller == pytest.approx(0.5)
    iou, of_smaller = overlap(left, NormalizedBox(x1=0.1, y1=0.1, x2=0.2, y2=0.2))  # inside
    assert iou == pytest.approx(0.01 / 0.5) and of_smaller == pytest.approx(1.0)
    assert overlap(left, NormalizedBox(x1=0.5, y1=0.0, x2=1.0, y2=1.0)) == (0.0, 0.0)  # touching edges
    assert overlap(left, NormalizedBox(x1=0.6, y1=0.6, x2=0.9, y2=0.9)) == (0.0, 0.0)


def test_each_record_is_one_counted_frame_with_two_or_more_persons() -> None:
    multi = MultiPersonFrames()
    probe, captured = probe_with([[], [A], [A, B], [A, B, C], [B]], [multi.observe])
    block = multi.summary()["multi_person_frames"]
    assert (block["frames"], len(block["samples"]), block["samples_omitted"]) == (2, 2, 0)
    assert block["omitted_first_utc"] is None and block["omitted_last_utc"] is None
    assert block["persons_not_recorded"] == 0 and block["max_samples"] == MAX_SAMPLES

    first, second = block["samples"]
    frame = captured[2].frame
    assert first["frame"] == {"camera_id": "cam-1", "boot_id": frame.boot_id, "run_id": frame.run_id,
                              "stream_epoch": frame.stream_epoch, "frame_seq": frame.frame_seq}
    assert first["ingest_utc"] == ms(frame.ingest_utc) and first["ingest_mono_ns"] == frame.ingest_mono_ns
    assert first["source_pts"] == 3 * 66_667 and first["source_time_quality"] == frame.source_time_quality.value
    assert first["persons"] == [
        {"track_id": 1, "status": "confirmed", "confidence": 0.91, "box": [0.1, 0.1, 0.5, 1.0]},
        {"track_id": 5, "status": "tentative", "confidence": 0.43, "box": [0.2, 0.2, 0.4, 0.9]},
    ]
    assert first["pairs"] == [{"tracks": [1, 5], "iou": 0.389, "of_smaller": 1.0}]  # B lies inside A
    assert second["frame"]["frame_seq"] == captured[3].frame.frame_seq
    assert [p["track_id"] for p in second["persons"]] == [1, 5, 9]
    assert second["pairs"] == [{"tracks": [1, 5], "iou": 0.389, "of_smaller": 1.0},
                               {"tracks": [1, 9], "iou": 0.0, "of_smaller": 0.0},
                               {"tracks": [5, 9], "iou": 0.0, "of_smaller": 0.0}]
    # The records are the frames behind the timeline's max_persons >= 2, and plain JSON.
    timeline = probe.summary()["timeline"]["seconds"]
    seconds = {s["ingest_utc"][:19] + "Z" for s in block["samples"]}
    assert {e["utc"] for e in timeline if e["max_persons"] >= 2} == seconds
    assert json.loads(json.dumps(block)) == block


def test_records_are_bounded_and_omissions_are_counted() -> None:
    multi = MultiPersonFrames(max_samples=2, max_persons=2)
    _, captured = probe_with([[A, B, C], [A, B], [], [A, B], [A, B, C]], [multi.observe])
    block = multi.summary()["multi_person_frames"]
    assert (block["frames"], len(block["samples"]), block["samples_omitted"]) == (4, 2, 2)
    assert block["omitted_first_utc"] == ms(captured[3].frame.ingest_utc)
    assert block["omitted_last_utc"] == ms(captured[4].frame.ingest_utc)
    first = block["samples"][0]
    assert [p["track_id"] for p in first["persons"]] == [1, 5] and len(first["pairs"]) == 1
    assert block["persons_not_recorded"] == 1  # track 9 on the first retained frame
    assert (block["max_samples"], block["max_persons_per_sample"]) == (2, 2)

    many = MultiPersonFrames(max_samples=3)
    probe_with([[A, B]] * 400, [many.observe])
    block = many.summary()["multi_person_frames"]
    assert (block["frames"], len(block["samples"]), block["samples_omitted"]) == (400, 3, 397)
    with pytest.raises(ValueError):
        MultiPersonFrames(max_samples=0)
    with pytest.raises(ValueError):
        MultiPersonFrames(max_persons=1)


def test_failed_and_skipped_frames_are_not_recorded() -> None:
    multi = MultiPersonFrames()
    probe, _ = probe_with([[A, B], RuntimeError("private detail"), [A, B], [A]], [multi.observe], repeat=0)
    block = multi.summary()["multi_person_frames"]
    assert block["frames"] == 2 and len(block["samples"]) == 2
    tracking = probe.summary()["tracking"]
    assert (tracking["processed"], tracking["failed"], tracking["skipped"]) == (3, 1, 1)
    assert "private detail" not in json.dumps(block)


def test_the_records_change_no_count() -> None:
    script: list[object] = [[], [A], [A, B], RuntimeError("x"), [A, B, C], [C], [], [A, C]]
    results: list[TrackingResult] = []
    plain, _ = probe_with(list(script), repeat=2)
    multi = MultiPersonFrames()
    observed, _ = probe_with(list(script), [lambda c, r: results.append(r), multi.observe, BoxSummary().observe],
                             repeat=2)
    without = {k: v for k, v in plain.summary().items() if not k.endswith("_ms")}
    with_observers = {k: v for k, v in observed.summary().items() if not k.endswith("_ms")}
    assert without == with_observers
    assert multi.summary()["multi_person_frames"]["frames"] == sum(
        r.outcome.value == "processed" and len(r.persons) >= 2 for r in results) == 3


def test_no_image_is_kept() -> None:
    multi = MultiPersonFrames()
    _, captured = probe_with([[A, B], [A, B, C], [A]], [multi.observe])
    images = [weakref.ref(item.image) for item in captured]
    del captured
    gc.collect()
    assert all(ref() is None for ref in images)
    assert len(multi.summary()["multi_person_frames"]["samples"]) == 2


# ------------------------------------------------------------ the command


class PacedSource:
    endpoint = None

    def __init__(self) -> None:
        self.reads = 0

    def open(self) -> None:
        pass

    def read(self) -> DecodedFrame:
        time.sleep(0.01)
        self.reads += 1
        return DecodedFrame(object(), 640, 480, PixelFormat.BGR, self.reads * 66_667)

    def close(self) -> None:
        pass


class TwoPersons:
    def load(self) -> None:
        pass

    def track(self, image: object) -> Sequence[RawTrack]:
        return [A, B]

    def reset(self) -> None:
        pass


def run(*flags: str) -> int:
    return main(
        ["track", "probe", str(DEFAULT_CONFIG), "--engine", "/models/yolov8n.engine", "--seconds", "1", *flags],
        capture_source=lambda config: PacedSource(),
        tracker_backend=lambda engine: TwoPersons(),
        meminfo=lambda: PLENTY,
    )


def test_the_command_adds_only_the_multi_person_block(capsys: pytest.CaptureFixture[str]) -> None:
    code = run("--multi-person-frames", "--box-summary")
    summary = json.loads(capsys.readouterr().out)
    assert code == 0
    block, persons = summary["multi_person_frames"], summary["persons"]
    assert block["frames"] == persons["frames_with_persons"] == summary["tracking"]["processed"] > 0
    assert persons["max_per_frame"] == 2
    assert len(block["samples"]) == min(block["frames"], MAX_SAMPLES)
    assert block["samples_omitted"] == block["frames"] - len(block["samples"])
    assert all(s["pairs"] == [{"tracks": [1, 5], "iou": 0.389, "of_smaller": 1.0}] for s in block["samples"])
    assert [t["frames"] for t in summary["track_boxes"]["tracks"]] == [block["frames"]] * 2

    code = run()
    plain = json.loads(capsys.readouterr().out)
    assert code == 0 and "multi_person_frames" not in plain
    assert set(plain) == set(summary) - {"multi_person_frames", "track_boxes"}
