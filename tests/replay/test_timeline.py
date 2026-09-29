from __future__ import annotations

import pytest

from sentinel.contracts import TrackStatus
from sentinel.media.clock import NS_PER_SECOND
from sentinel.replay import (
    FrameEvent,
    ReplayDriver,
    ResultEvent,
    TickEvent,
    Timeline,
    TimelineError,
)

HEADER = '{"event": "timeline", "name": "t", "description": "test"}'


def parse(*lines: str) -> Timeline:
    return Timeline.parse("\n".join((HEADER, *lines)))


def test_driver_plays_events_at_their_offsets_with_stream_identity() -> None:
    timeline = parse(
        '{"at_ms": 0, "event": "connect"}',
        '{"at_ms": 100, "event": "frame", "repeat": 3, "every_ms": 50,'
        ' "persons": [{"track": 7, "box": [0.1, 0.2, 0.3, 0.9]}]}',
        '{"at_ms": 300, "event": "disconnect"}',
        '{"at_ms": 2300, "event": "tick"}',
        '{"at_ms": 2400, "event": "connect"}',
        '{"at_ms": 2400, "event": "frame", "pts": 0}',
        '{"at_ms": 2500, "event": "result", "job": 1, "failure": "timeout"}',
    )
    driver = ReplayDriver(timeline)
    origin = driver.clock.monotonic_ns()
    steps = list(driver.steps())

    offsets_ms = [(step.now.ns - origin) // 1_000_000 for step in steps]
    assert offsets_ms == [0, 100, 150, 200, 300, 2300, 2400, 2400, 2500]
    frames = [step.frame for step in steps if step.frame is not None]
    assert [(f.stream_epoch, f.frame_seq) for f in frames] == [(1, 0), (1, 1), (1, 2), (2, 0)]
    assert (frames[0].native_width, frames[0].native_height) == (640, 480)
    assert steps[5].stream is None and isinstance(steps[5].event, TickEvent)
    assert steps[6].stream is not None and steps[6].stream.stream_epoch == 2

    person = steps[1].persons[0]
    assert person.frame == frames[0].key and person.track_id == 7
    assert person.status is TrackStatus.CONFIRMED and not person.predicted
    assert person.box.y2 == 0.9
    assert isinstance(steps[-1].event, ResultEvent) and steps[-1].event.failure == "timeout"


def test_a_repeated_frame_run_advances_time_for_the_whole_run() -> None:
    timeline = parse(
        '{"at_ms": 0, "event": "connect"}',
        '{"at_ms": 0, "event": "frame", "repeat": 150, "every_ms": 66}',
    )
    driver = ReplayDriver(timeline)
    start = driver.clock.monotonic_ns()
    assert sum(1 for _ in driver.steps()) == 151
    assert driver.clock.monotonic_ns() - start == 149 * 66 * NS_PER_SECOND // 1000


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ((), "first line must be a header"),
        (('{"at_ms": 10, "event": "tick"}', '{"at_ms": 5, "event": "tick"}'), ":3: at_ms 5"),
        (
            (
                '{"at_ms": 0, "event": "frame", "repeat": 3, "every_ms": 100}',
                '{"at_ms": 150, "event": "tick"}',
            ),
            "earlier than 200",
        ),
        (('{"at_ms": 0, "event": "frame", "repeat": 2}',), "needs every_ms"),
        (('{"at_ms": 0, "event": "explode"}',), ":2:"),
        (('{"at_ms": 0, "event": "tick", "colour": 1}',), "Extra inputs are not permitted"),
        (('{"at_ms": 0, "event": "frame", "persons": [{"track": 1, "box": [0, 0, 2, 1]}]}',), ""),
        (("not json",), ":2:"),
    ],
)
def test_malformed_timelines_are_rejected_with_their_line(
    lines: tuple[str, ...], message: str
) -> None:
    text = "\n".join(lines) if lines == () else "\n".join((HEADER, *lines))
    with pytest.raises(TimelineError, match=message or None):
        Timeline.parse(text, source="t.jsonl")


def test_a_frame_while_disconnected_is_a_timeline_error() -> None:
    driver = ReplayDriver(parse('{"at_ms": 0, "event": "frame"}'))
    with pytest.raises(TimelineError, match="while disconnected"):
        list(driver.steps())


def test_timeline_events_are_typed() -> None:
    timeline = parse('{"at_ms": 0, "event": "frame", "faces": [{"box": [0, 0, 0.1, 0.1]}]}')
    (event,) = timeline.events
    assert isinstance(event, FrameEvent)
    assert event.faces[0].quality == 0.9 and event.faces[0].embedding is None
