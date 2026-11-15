"""Deterministic replay of synthetic timelines (V2-03; guide chapters 21 and 25).

A timeline is a JSON-lines file with one event per line. The first line is a
``timeline`` header; every later event has an ``at_ms`` offset from the start
of the replay, and offsets never decrease. Timelines contain no images or
footage: a ``frame`` event states what the detector and face stages would have
reported for that frame, in normalized native-image coordinates.

The driver advances a FakeClock to each event, keeps stream identity with a
FrameStamper and yields one step per event to the code under test, which
decides what the event means for it (for example, which job a ``result``
completes).
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    TypeAdapter,
    ValidationError,
)

from .contracts import (
    FrameRef,
    Identifier,
    NormalizedBox,
    PixelFormat,
    StreamIdentity,
    TrackObservation,
    TrackStatus,
    UnitInterval,
)
from .media.clock import FakeClock, MonoInstant
from .media.frames import FrameStamper

MAX_EXPANDED_FRAMES = 100_000

Box = tuple[UnitInterval, UnitInterval, UnitInterval, UnitInterval]


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class PersonSpec(_Spec):
    """A tracked person as the detector/tracker would report it on one frame."""

    track: NonNegativeInt
    box: Box
    confidence: UnitInterval = 0.9
    status: TrackStatus = TrackStatus.CONFIRMED


class FaceSpec(_Spec):
    """A detected face; ``embedding`` is a synthetic vector, never real face data."""

    box: Box
    quality: UnitInterval = 0.9
    embedding: tuple[float, ...] | None = None


class TimelineHeader(_Spec):
    event: Literal["timeline"]
    name: Identifier
    description: str
    camera_id: Identifier = "cam-1"
    native_width: PositiveInt = 640
    native_height: PositiveInt = 480


class _Timed(_Spec):
    at_ms: NonNegativeInt
    note: str | None = None


class ConnectEvent(_Timed):
    event: Literal["connect"]


class DisconnectEvent(_Timed):
    event: Literal["disconnect"]


class FrameEvent(_Timed):
    """One frame, or ``repeat`` identical frames ``every_ms`` apart starting at ``at_ms``."""

    event: Literal["frame"]
    persons: tuple[PersonSpec, ...] = ()
    faces: tuple[FaceSpec, ...] = ()
    pts: int | None = None
    repeat: PositiveInt = 1
    every_ms: PositiveInt | None = None


class TickEvent(_Timed):
    """Time passes with no new frame; evaluate time-dependent state."""

    event: Literal["tick"]


class ResultEvent(_Timed):
    """An asynchronous worker delivers an outcome for ``job`` (the Nth job it was given, from 1)."""

    event: Literal["result"]
    job: PositiveInt
    text: str | None = None
    failure: Literal["timeout", "error"] | None = None
    claimed_job: str | None = None  # job ID the response claims, if it is not the job's own


class IncidentEvent(_Timed):
    """Something outside the scene lane opened an incident, e.g. a zone rule."""

    event: Literal["incident"]
    incident_id: Identifier


TimedEvent = Annotated[
    Union[ConnectEvent, DisconnectEvent, FrameEvent, TickEvent, ResultEvent, IncidentEvent],
    Field(discriminator="event"),
]
_EVENT = TypeAdapter(TimedEvent)


class TimelineError(ValueError):
    """The timeline file is malformed; the message names the line."""


@dataclass(frozen=True)
class Timeline:
    header: TimelineHeader
    events: tuple[TimedEvent, ...]

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Timeline:
        return cls.parse(Path(path).read_text(encoding="utf-8"), source=os.fspath(path))

    @classmethod
    def parse(cls, text: str, *, source: str = "<timeline>") -> Timeline:
        header: TimelineHeader | None = None
        events: list[TimedEvent] = []
        last_ms = 0
        expanded = 0
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{source}:{number}"
            try:
                if header is None:
                    header = TimelineHeader.model_validate_json(line)
                    continue
                event = _EVENT.validate_json(line)
            except ValidationError as exc:
                first = exc.errors()[0]
                loc = ".".join(str(part) for part in first["loc"]) or "<line>"
                raise TimelineError(f"{where}: {loc}: {first['msg']}") from None
            if event.at_ms < last_ms:
                raise TimelineError(f"{where}: at_ms {event.at_ms} is earlier than {last_ms}")
            last_ms = event.at_ms
            if isinstance(event, FrameEvent):
                if event.repeat > 1 and event.every_ms is None:
                    raise TimelineError(f"{where}: a repeated frame needs every_ms")
                last_ms = event.at_ms + (event.repeat - 1) * (event.every_ms or 0)
                expanded += event.repeat
                if expanded > MAX_EXPANDED_FRAMES:
                    raise TimelineError(f"{where}: more than {MAX_EXPANDED_FRAMES} frames")
            events.append(event)
        if header is None:
            raise TimelineError(f"{source}: empty timeline; the first line must be a header")
        return cls(header=header, events=tuple(events))


@dataclass(frozen=True)
class ReplayStep:
    """One event as the code under test sees it."""

    event: TimedEvent
    now: MonoInstant
    stream: StreamIdentity | None  # the connected stream epoch, if any
    frame: FrameRef | None = None  # set for frame events
    persons: tuple[TrackObservation, ...] = ()
    faces: tuple[FaceSpec, ...] = ()


class ReplayDriver:
    """Plays a timeline against a FakeClock; ``clock`` and ``stamper`` stay inspectable."""

    def __init__(self, timeline: Timeline, *, clock: FakeClock | None = None) -> None:
        self.timeline = timeline
        self.clock = clock or FakeClock()
        self.stamper = FrameStamper(timeline.header.camera_id, self.clock)
        self._origin_ns = self.clock.monotonic_ns()

    def steps(self) -> Iterator[ReplayStep]:
        for event in self.timeline.events:
            if isinstance(event, FrameEvent):
                for index in range(event.repeat):
                    self._advance_to(event.at_ms + index * (event.every_ms or 0))
                    yield self._frame_step(event)
                continue
            self._advance_to(event.at_ms)
            if isinstance(event, ConnectEvent):
                self.stamper.connect()
            elif isinstance(event, DisconnectEvent):
                self.stamper.disconnect()
            yield ReplayStep(event=event, now=self.clock.mono(), stream=self.stamper.current_stream)

    def _advance_to(self, at_ms: int) -> None:
        target = self._origin_ns + at_ms * 1_000_000
        self.clock.advance(ns=target - self.clock.monotonic_ns())

    def _frame_step(self, event: FrameEvent) -> ReplayStep:
        header = self.timeline.header
        if self.stamper.current_stream is None:
            raise TimelineError(f"frame at {event.at_ms} ms while disconnected")
        frame = self.stamper.stamp(
            native_width=header.native_width,
            native_height=header.native_height,
            pixel_format=PixelFormat.BGR,
            source_pts=event.pts,
        )
        persons = tuple(
            TrackObservation.detected(
                frame,
                track_id=person.track,
                box=_box(person.box),
                confidence=person.confidence,
                status=person.status,
            )
            for person in event.persons
        )
        return ReplayStep(
            event=event,
            now=self.clock.mono(),
            stream=self.stamper.current_stream,
            frame=frame,
            persons=persons,
            faces=event.faces,
        )


def _box(values: Sequence[float]) -> NormalizedBox:
    x1, y1, x2, y2 = values
    return NormalizedBox(x1=x1, y1=y1, x2=x2, y2=y2)
