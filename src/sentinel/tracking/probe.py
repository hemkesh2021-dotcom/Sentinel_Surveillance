"""Track probe: what `sentinel track probe` adds to the capture probe (V2-09/V2-10 demo-form device check).

The capture probe's loop hands every frame it takes to a PersonTracker, so a
detector slower than the camera shows up as replaced frames. The summary adds
detector throughput and timing, track and failure counts, and the memory seen
around the model load. Numbers and fixed labels only: no images, boxes, URL or
host. Timing is ingest-based (U3), so "result age" is ingest-to-result, not
capture-to-result.

The timeline splits the same counts by UTC second of frame ingest, so a timed
operator check can compare periods (for example nobody in view, then one
person, then nobody again) against its own prompt times. It is bounded and
holds numbers only.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from ..contracts import TrackStatus
from ..media.capture import CapturedFrame
from ..media.clock import Clock
from ..media.probe import MAX_SAMPLES, percentiles_ms
from .tracker import FrameOutcome, PersonTracker

MAX_TRACK_IDS = 10_000
MAX_TIMELINE_SECONDS = 320  # the longest probe (300 s) plus slack; later seconds are counted as dropped
LIMITATIONS = (
    "legacy parity adapter (Ultralytics track, ByteTrack, v1 arguments) on software-decoded "
    "frames; ingest-based timing; counts, not accuracy; not hardware acceptance"
)


def read_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int] | None:
    """MemFree and MemAvailable in bytes, or None where /proc/meminfo is unavailable."""
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    values = {}
    for line in lines:
        name, _, rest = line.partition(":")
        if name in ("MemFree", "MemAvailable"):
            values[name] = int(rest.split()[0]) * 1024
    return values if len(values) == 2 else None


class TrackProbe:
    """A FrameConsumer for run_probe that tracks every frame it is given."""

    def __init__(self, tracker: PersonTracker, clock: Clock) -> None:
        self._tracker = tracker
        self._clock = clock
        self._outcomes = {outcome.value: 0 for outcome in FrameOutcome}
        self._failures: Counter[str] = Counter()
        self._error_types: Counter[str] = Counter()
        self._backend_ns: list[int] = []
        self._result_age_ns: list[int] = []
        self._ids: set[tuple[int, int]] = set()
        self._confirmed: set[tuple[int, int]] = set()
        self._frames_with_persons = 0
        self._max_persons = 0
        self._first_mono_ns: int | None = None
        self._last_mono_ns: int | None = None
        self._timeline: dict[datetime, dict[str, Any]] = {}
        self._timeline_dropped = 0

    def observe(self, captured: CapturedFrame) -> None:
        frame = captured.frame
        result = self._tracker.process(frame, captured.image)
        done = self._clock.mono().ns
        self._outcomes[result.outcome.value] += 1
        if result.backend_ns is not None and len(self._backend_ns) < MAX_SAMPLES:
            self._backend_ns.append(result.backend_ns)
        second = self._second(frame.ingest_utc) if result.outcome is not FrameOutcome.SKIPPED else None
        if result.outcome is FrameOutcome.FAILED:
            self._failures[result.problem or "unknown"] += 1
            if result.error_type:
                self._error_types[result.error_type] += 1
            if second is not None:
                second["failed"] += 1
            return
        if result.outcome is not FrameOutcome.PROCESSED:
            return
        if second is not None:
            confirmed = sum(person.status is TrackStatus.CONFIRMED for person in result.persons)
            second["processed"] += 1
            second["frames_with_persons"] += bool(result.persons)
            second["max_persons"] = max(second["max_persons"], len(result.persons))
            second["max_confirmed"] = max(second["max_confirmed"], confirmed)
            second["track_ids"].update((frame.stream_epoch, person.track_id) for person in result.persons)
        if self._first_mono_ns is None:
            self._first_mono_ns = done
        self._last_mono_ns = done
        if len(self._result_age_ns) < MAX_SAMPLES:
            self._result_age_ns.append(done - frame.ingest_mono_ns)
        if result.persons:
            self._frames_with_persons += 1
            self._max_persons = max(self._max_persons, len(result.persons))
        for person in result.persons:
            key = (frame.stream_epoch, person.track_id)
            if len(self._ids) < MAX_TRACK_IDS:
                self._ids.add(key)
            if person.status is TrackStatus.CONFIRMED and len(self._confirmed) < MAX_TRACK_IDS:
                self._confirmed.add(key)

    def _second(self, ingest_utc: datetime) -> dict[str, Any] | None:
        key = ingest_utc.replace(microsecond=0)
        entry = self._timeline.get(key)
        if entry is None:
            if len(self._timeline) >= MAX_TIMELINE_SECONDS:
                self._timeline_dropped += 1
                return None
            entry = self._timeline[key] = {"processed": 0, "failed": 0, "frames_with_persons": 0,
                                           "max_persons": 0, "max_confirmed": 0, "track_ids": set()}
        return entry

    def summary(self) -> dict[str, Any]:
        processed = self._outcomes[FrameOutcome.PROCESSED.value]
        span_ns = (self._last_mono_ns or 0) - (self._first_mono_ns or 0)
        counters = self._tracker.counters
        return {
            "probe": "track",
            "tracking": {
                **self._outcomes,
                "processed_fps": round((processed - 1) / (span_ns / 1e9), 3) if processed > 1 and span_ns > 0 else None,
                "failures": dict(sorted(self._failures.items())),
                "error_types": dict(sorted(self._error_types.items())),
                **{
                    name: counters[name]
                    for name in ("epoch_resets", "failure_resets", "boxes_dropped_small", "boxes_dropped_overflow")
                },
            },
            "backend_ms": percentiles_ms(self._backend_ns),
            "result_age_ms": percentiles_ms(self._result_age_ns),
            "persons": {
                "frames_with_persons": self._frames_with_persons,
                "max_per_frame": self._max_persons,
                "track_ids": len(self._ids),
                "confirmed_track_ids": len(self._confirmed),
            },
            "timeline": {
                "bucket": "UTC second of frame ingest; track_ids are [stream_epoch, track_id]",
                "dropped_frames": self._timeline_dropped,
                "seconds": [
                    {"utc": key.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     **{name: value for name, value in entry.items() if name != "track_ids"},
                     "track_ids": [list(pair) for pair in sorted(entry["track_ids"])]}
                    for key, entry in sorted(self._timeline.items())
                ],
            },
            "track_limitations": LIMITATIONS,
        }
