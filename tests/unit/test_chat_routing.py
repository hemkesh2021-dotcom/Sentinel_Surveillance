"""Which questions are answered from records and which from the picture (session 116, part 28): fixed patterns."""

from __future__ import annotations

import pytest

from sentinel.scene.chat_routing import (
    ALERTS,
    DEFAULT_ALERTS_S,
    DEFAULT_HISTORY_S,
    HISTORY,
    MAX_PERIOD_S,
    VISUAL,
    WHO,
    Route,
    route,
)


@pytest.mark.parametrize(("question", "expected"), [
    # the maintainer's three questions, as asked
    ("What happened in the last 10 minutes?", Route(HISTORY, 600.0, False, True)),
    ("Who is in the room?", Route(WHO)),
    ("Can you check whether any alerts were triggered?", Route(ALERTS, DEFAULT_ALERTS_S, False, False)),
    # near variants
    ("what happened?", Route(HISTORY, DEFAULT_HISTORY_S, False, False)),
    ("What has happened in the past hour", Route(HISTORY, 3600.0, False, True)),
    ("what's been happening over the last 30 min?", Route(HISTORY, 1800.0, False, True)),
    ("Did anything happen in the last half an hour?", Route(HISTORY, 1800.0, False, True)),
    ("Any recent activity?", Route(HISTORY, DEFAULT_HISTORY_S, False, False)),
    ("Give me a summary of the last two hours", Route(HISTORY, 7200.0, False, True)),
    ("what happened since startup?", Route(HISTORY, None, True, True)),
    ("What happened today?", Route(HISTORY, None, True, True)),
    ("who's here?", Route(WHO)),
    ("Is anyone home?", Route(WHO)),
    ("Is there anybody in the room?", Route(WHO)),
    ("How many people are in the room?", Route(WHO)),
    ("Who is that?", Route(WHO)),
    ("Were there any incidents in the last 15 minutes?", Route(ALERTS, 900.0, False, True)),
    ("Any notifications sent?", Route(ALERTS, DEFAULT_ALERTS_S, False, False)),
    ("was I notified on telegram", Route(ALERTS, DEFAULT_ALERTS_S, False, False)),
    ("Has anything been triggered since the start?", Route(ALERTS, None, True, True)),
    ("What happened with the alerts in the last hour?", Route(ALERTS, 3600.0, False, True)),  # alerts first
])
def test_a_question_about_records_is_routed_to_them_with_its_period(question: str, expected: Route) -> None:
    assert route(question) == expected


@pytest.mark.parametrize("question", [
    "Is the door open?", "Is anyone at the door?", "What is on the desk?", "Is there a phone on the desk?",
    "near the water bottle?", "Can you zoom in the pic?", "How many chairs are there?", "Is the light on?",
    "What colour is the jacket?", "Is somebody sitting on the chair?", "Describe the room", "Is the person standing?",
    "Is there an alarm clock on the shelf?", "What is happening in this picture?", "who left this bag on the floor",
    "Is the window open since this morning", "What's happening?", "Who is in this picture?",
    "what happened in this image", "is anyone in the frame",
])
def test_a_question_about_the_picture_stays_one(question: str) -> None:
    assert route(question) == Route(VISUAL)


def test_a_period_is_bounded_and_never_below_a_second() -> None:
    assert route("what happened in the last 9999 days").period_s == MAX_PERIOD_S
    assert route("what happened in the last 0 seconds").period_s == 1.0
    assert route("what happened in the last minute").period_s == 60.0
    assert route("WHAT HAPPENED  in the LAST 5 Minutes??").period_s == 300.0
