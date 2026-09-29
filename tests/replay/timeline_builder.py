"""Builds inline replay timelines on a 15 fps frame grid, so events never overlap frame runs."""

from __future__ import annotations

import json

from sentinel.replay import Timeline

FRAME_MS = 66  # the substream delivers ~15 fps


def frame_at(at_ms: int) -> int:
    """The first frame time on the grid at or after ``at_ms``."""
    return -(-at_ms // FRAME_MS) * FRAME_MS


def people(count: int) -> list[dict]:
    """``count`` side-by-side people with non-overlapping boxes."""
    return [
        {"track": n, "box": [0.05 + 0.3 * n, 0.2, 0.25 + 0.3 * n, 0.9]} for n in range(count)
    ]


class TimelineBuilder:
    def __init__(self, name: str = "inline", description: str = "inline replay variant") -> None:
        self._lines = [{"event": "timeline", "name": name, "description": description}]
        self._last_ms = 0
        self._next_frame_ms = 0

    def event(self, at_ms: int, event: str, **fields: object) -> TimelineBuilder:
        if at_ms < self._last_ms:
            raise ValueError(f"{event} at {at_ms} ms is before {self._last_ms} ms")
        self._lines.append({"at_ms": at_ms, "event": event, **fields})
        self._last_ms = at_ms
        return self

    def connect(self, at_ms: int) -> TimelineBuilder:
        return self.event(at_ms, "connect")

    def disconnect(self, at_ms: int) -> TimelineBuilder:
        return self.event(at_ms, "disconnect")

    def tick(self, at_ms: int) -> TimelineBuilder:
        return self.event(at_ms, "tick")

    def incident(self, at_ms: int, incident_id: str) -> TimelineBuilder:
        return self.event(at_ms, "incident", incident_id=incident_id)

    def result(self, at_ms: int, job: int, report: dict | None = None, **fields: object) -> TimelineBuilder:
        if report is not None:
            fields["text"] = json.dumps(report)
        return self.event(at_ms, "result", job=job, **fields)

    def frames_until(self, end_ms: int, *, persons: int = 1, **fields: object) -> TimelineBuilder:
        """Frames on the grid from the next slot (not before the last event) up to, excluding, end_ms."""
        start = max(self._next_frame_ms, frame_at(self._last_ms))
        count = len(range(start, end_ms, FRAME_MS))
        if count == 0:
            return self
        self.event(start, "frame", repeat=count, every_ms=FRAME_MS, persons=people(persons), **fields)
        self._last_ms = start + (count - 1) * FRAME_MS
        self._next_frame_ms = self._last_ms + FRAME_MS
        return self

    def build(self) -> Timeline:
        return Timeline.parse("\n".join(json.dumps(line) for line in self._lines))
