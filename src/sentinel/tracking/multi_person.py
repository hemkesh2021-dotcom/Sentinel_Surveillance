"""Opt-in, numbers-only records of frames with two or more persons, for `sentinel track probe --multi-person-frames`.

Added in session 36 after 3TC run 2, where three timeline seconds had
``max_persons`` 2. ``track_boxes`` keeps per-track statistics only, so it cannot
say where two boxes were relative to each other on the same frame. This
observer records exactly those frames: each record is one processed frame on
which the tracker published two or more persons at once. It describes what was
counted and judges nothing; it holds no image and no reference to one.

Bounded:
- at most ``max_samples`` records, the first ones seen. Later frames with two or
  more persons are counted in ``samples_omitted``, with the ingest times of the
  first and last omitted frame;
- at most ``max_persons`` persons per record, in published order (track ID).
  Persons beyond that are counted in ``persons_not_recorded``, and pairs are
  formed among the recorded persons only.

Each record has:
- ``frame``: camera_id, boot_id, run_id, stream_epoch and frame_seq;
- ``ingest_utc`` (ms), ``ingest_mono_ns``, ``source_pts`` (the source's stream
  time in its own unit, or null) and ``source_time_quality``;
- ``persons``: track_id, status, confidence (3 dp) and box [x1, y1, x2, y2],
  normalized to the native image (3 dp);
- ``pairs``: for each pair of recorded persons, their track IDs, ``iou``
  (intersection over union) and ``of_smaller`` (intersection over the smaller
  box's area), 3 dp, computed from the unrounded boxes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..contracts import NormalizedBox
from ..media.capture import CapturedFrame
from .tracker import FrameOutcome, TrackingResult

MAX_SAMPLES = 64
MAX_PERSONS = 8


def _utc_ms(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def overlap(a: NormalizedBox, b: NormalizedBox) -> tuple[float, float]:
    """(intersection over union, intersection over the smaller box's area) of two boxes."""
    width = min(a.x2, b.x2) - max(a.x1, b.x1)
    height = min(a.y2, b.y2) - max(a.y1, b.y1)
    inter = width * height if width > 0 and height > 0 else 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter), inter / min(area_a, area_b)


class MultiPersonFrames:
    """A TrackProbe observer that keeps bounded records of frames with two or more persons."""

    def __init__(self, max_samples: int = MAX_SAMPLES, max_persons: int = MAX_PERSONS) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1")
        if max_persons < 2:
            raise ValueError("max_persons must be at least 2")
        self._max_samples = max_samples
        self._max_persons = max_persons
        self._frames = 0
        self._samples: list[dict[str, Any]] = []
        self._omitted = 0
        self._omitted_first: datetime | None = None
        self._omitted_last: datetime | None = None
        self._persons_not_recorded = 0

    def observe(self, captured: CapturedFrame, result: TrackingResult) -> None:
        if result.outcome is not FrameOutcome.PROCESSED or len(result.persons) < 2:
            return
        frame = captured.frame
        self._frames += 1
        if len(self._samples) >= self._max_samples:
            self._omitted += 1
            if self._omitted_first is None:
                self._omitted_first = frame.ingest_utc
            self._omitted_last = frame.ingest_utc
            return
        persons = result.persons[: self._max_persons]
        self._persons_not_recorded += len(result.persons) - len(persons)
        pairs = []
        for i, first in enumerate(persons):
            for second in persons[i + 1:]:
                iou, of_smaller = overlap(first.box, second.box)
                pairs.append({"tracks": [first.track_id, second.track_id],
                              "iou": round(iou, 3), "of_smaller": round(of_smaller, 3)})
        self._samples.append({
            "frame": {"camera_id": frame.camera_id, "boot_id": frame.boot_id, "run_id": frame.run_id,
                      "stream_epoch": frame.stream_epoch, "frame_seq": frame.frame_seq},
            "ingest_utc": _utc_ms(frame.ingest_utc),
            "ingest_mono_ns": frame.ingest_mono_ns,
            "source_pts": frame.source_pts,
            "source_time_quality": frame.source_time_quality.value,
            "persons": [
                {"track_id": person.track_id, "status": person.status.value,
                 "confidence": round(float(person.detector_confidence or 0.0), 3),
                 "box": [round(v, 3) for v in (person.box.x1, person.box.y1, person.box.x2, person.box.y2)]}
                for person in persons
            ],
            "pairs": pairs,
        })

    def summary(self) -> dict[str, Any]:
        return {
            "multi_person_frames": {
                "coordinates": "normalized to the native image: [x1, y1, x2, y2]",
                "overlap": "iou: intersection over union; of_smaller: intersection over the smaller box's area",
                "frames": self._frames,
                "max_samples": self._max_samples,
                "max_persons_per_sample": self._max_persons,
                "samples_omitted": self._omitted,
                "omitted_first_utc": None if self._omitted_first is None else _utc_ms(self._omitted_first),
                "omitted_last_utc": None if self._omitted_last is None else _utc_ms(self._omitted_last),
                "persons_not_recorded": self._persons_not_recorded,
                "samples": self._samples,
            }
        }
