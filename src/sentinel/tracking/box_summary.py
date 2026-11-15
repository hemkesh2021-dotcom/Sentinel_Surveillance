"""Opt-in, numbers-only box statistics per track for `sentinel track probe --box-summary` (P1, session 26).

The track probe's normal report holds counts and IDs, never boxes. With
``--box-summary`` it also reports, for each counted ``[stream_epoch, track_id]``,
where its boxes were, so a record can say where person boxes appeared without
any image. It summarizes exactly the observations the probe counts.

Fixed size: at most MAX_BOX_TRACKS rows, one per track in the order first seen.
Observations of later tracks are counted in ``observations_not_summarized``.
Each row has the same fields:
- ``track``: [stream_epoch, track_id];
- ``first_utc`` and ``last_utc``: ingest times of its first and last counted frame (ms);
- ``frames`` and ``confirmed_frames``;
- ``confidence``: min, mean and max (3 dp);
- ``box_first``, ``box_last``, ``box_mean`` and ``box_union``: [x1, y1, x2, y2],
  normalized to the native image (3 dp).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..contracts import TrackStatus
from ..media.capture import CapturedFrame
from .tracker import FrameOutcome, TrackingResult

MAX_BOX_TRACKS = 64


def _utc_ms(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _rounded(values: list[float] | tuple[float, ...]) -> list[float]:
    return [round(v, 3) for v in values]


class BoxSummary:
    """A TrackProbe observer that keeps bounded per-track box statistics."""

    def __init__(self, max_tracks: int = MAX_BOX_TRACKS) -> None:
        if max_tracks < 1:
            raise ValueError("max_tracks must be at least 1")
        self._max = max_tracks
        self._rows: dict[tuple[int, int], dict[str, Any]] = {}
        self._not_summarized = 0

    def observe(self, captured: CapturedFrame, result: TrackingResult) -> None:
        if result.outcome is not FrameOutcome.PROCESSED:
            return
        frame = captured.frame
        for person in result.persons:
            key = (frame.stream_epoch, person.track_id)
            box = (person.box.x1, person.box.y1, person.box.x2, person.box.y2)
            confidence = float(person.detector_confidence or 0.0)
            row = self._rows.get(key)
            if row is None:
                if len(self._rows) >= self._max:
                    self._not_summarized += 1
                    continue
                row = self._rows[key] = {
                    "first_utc": frame.ingest_utc, "frames": 0, "confirmed_frames": 0,
                    "conf_min": confidence, "conf_max": confidence, "conf_sum": 0.0,
                    "box_first": box, "box_sum": [0.0, 0.0, 0.0, 0.0], "box_union": list(box),
                }
            row["last_utc"] = frame.ingest_utc
            row["frames"] += 1
            row["confirmed_frames"] += person.status is TrackStatus.CONFIRMED
            row["conf_min"] = min(row["conf_min"], confidence)
            row["conf_max"] = max(row["conf_max"], confidence)
            row["conf_sum"] += confidence
            row["box_last"] = box
            row["box_sum"] = [total + value for total, value in zip(row["box_sum"], box)]
            union = row["box_union"]
            row["box_union"] = [min(union[0], box[0]), min(union[1], box[1]),
                                max(union[2], box[2]), max(union[3], box[3])]

    def summary(self) -> dict[str, Any]:
        tracks = []
        for (epoch, track_id), row in self._rows.items():
            frames = row["frames"]
            tracks.append({
                "track": [epoch, track_id],
                "first_utc": _utc_ms(row["first_utc"]),
                "last_utc": _utc_ms(row["last_utc"]),
                "frames": frames,
                "confirmed_frames": row["confirmed_frames"],
                "confidence": {"min": round(row["conf_min"], 3), "mean": round(row["conf_sum"] / frames, 3),
                               "max": round(row["conf_max"], 3)},
                "box_first": _rounded(row["box_first"]),
                "box_last": _rounded(row["box_last"]),
                "box_mean": _rounded([total / frames for total in row["box_sum"]]),
                "box_union": _rounded(row["box_union"]),
            })
        return {
            "track_boxes": {
                "coordinates": "normalized to the native image: [x1, y1, x2, y2]",
                "max_tracks": self._max,
                "tracks": tracks,
                "observations_not_summarized": self._not_summarized,
            }
        }
