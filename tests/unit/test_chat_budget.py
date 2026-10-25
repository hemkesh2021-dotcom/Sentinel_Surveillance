"""Session 114: how long an answer a question has time for (chat_budget), with numbers only (no model, no camera).

Covered: the window's budget from chat's own measured pace, after the reserve and the time before decoding; the
120-token cap never exceeded; the conservative fallback (cold start, no usable timing) never below the minimum and
never waiting; a window too short with a measured pace waiting, or given up once it may not; the server's timings
checked (missing, malformed, inconsistent, too short to time); the pace from the latest samples, reset when the load
changed; an overrun; and trimming to complete sentences without an empty answer or a silently dropped rest.
"""

from __future__ import annotations

import math

import pytest

from sentinel.scene.chat import CHAT_MAX_TOKENS
from sentinel.scene.chat_budget import (
    CHAT_MIN_TOKENS,
    FALLBACK,
    FALLBACK_MS_PER_TOKEN,
    FALLBACK_OVERHEAD_MS,
    PACE_MIN_TOKENS,
    PACE_SAMPLES,
    RELEASE_RESERVE_MS,
    AnswerPlan,
    ChatPace,
    Pace,
    ServerTiming,
    complete_sentences,
    overran,
    pace_sample,
    plan_answer,
    server_timing,
)

MEASURED = Pace(overhead_ms=600.0, ms_per_token=20.0, measured=True, samples=3)


def test_the_cap_stays_120_and_the_proposed_values_are_as_recorded() -> None:
    assert CHAT_MAX_TOKENS == 120  # unchanged: no plan goes above it
    assert (CHAT_MIN_TOKENS, RELEASE_RESERVE_MS, FALLBACK_OVERHEAD_MS, FALLBACK_MS_PER_TOKEN) == (30, 500.0, 1500.0, 30.0)
    assert FALLBACK_OVERHEAD_MS + CHAT_MAX_TOKENS * FALLBACK_MS_PER_TOKEN == 5100.0  # a full answer: about 5.1 s


def test_a_measured_pace_sizes_the_answer_to_its_window_after_the_reserve_and_the_time_before_decoding() -> None:
    plan = plan_answer(2800.0, MEASURED, cap=CHAT_MAX_TOKENS, may_wait=True)
    assert plan == AnswerPlan("start", math.floor((2800 - 500 - 600) / 20), "measured", 2800.0)  # 85 tokens
    assert plan_answer(60_000.0, MEASURED, cap=CHAT_MAX_TOKENS, may_wait=True).max_tokens == CHAT_MAX_TOKENS
    edge = 500 + 600 + CHAT_MIN_TOKENS * 20  # exactly the minimum fits
    assert plan_answer(float(edge), MEASURED, cap=CHAT_MAX_TOKENS, may_wait=False) == AnswerPlan(
        "start", CHAT_MIN_TOKENS, "measured", float(edge))


def test_a_window_too_short_for_a_useful_answer_waits_for_the_next_gap_or_is_given_up_never_started() -> None:
    short = 500 + 600 + (CHAT_MIN_TOKENS - 1) * 20.0
    assert plan_answer(short, MEASURED, cap=CHAT_MAX_TOKENS, may_wait=True).action == "wait"
    assert plan_answer(short, MEASURED, cap=CHAT_MAX_TOKENS, may_wait=False).action == "give_up"
    assert plan_answer(0.0, MEASURED, cap=CHAT_MAX_TOKENS, may_wait=False) == AnswerPlan("give_up", 0, "measured", 0.0)


def test_with_no_periodic_job_expected_the_answer_has_the_full_cap() -> None:
    for pace in (MEASURED, FALLBACK):
        assert plan_answer(None, pace, cap=CHAT_MAX_TOKENS, may_wait=False) == AnswerPlan(
            "start", CHAT_MAX_TOKENS, "unbounded", None)


def test_the_fallback_is_deliberately_slow_never_below_the_minimum_and_never_waits() -> None:
    assert FALLBACK.measured is False
    for window in (0.0, 1500.0, 2800.0):  # the grace alone, a typical gap: the minimum, at once
        assert plan_answer(window, FALLBACK, cap=CHAT_MAX_TOKENS, may_wait=True) == AnswerPlan(
            "start", CHAT_MIN_TOKENS, "fallback", window)
    assert plan_answer(5000.0, FALLBACK, cap=CHAT_MAX_TOKENS, may_wait=True).max_tokens == math.floor(
        (5000 - 500 - 1500) / 30)  # a long window: still sized by the slow pace (100)
    assert plan_answer(5000.0, FALLBACK, cap=20, may_wait=True).max_tokens == 20  # never above its cap


def timings(**values: object) -> dict[str, object]:
    return {"timings": {"prompt_ms": 410.5, "predicted_ms": 900.0, "predicted_n": 45, **values}}


@pytest.mark.parametrize("payload", [
    {}, {"timings": None}, {"timings": []}, "not a mapping", None,
    timings(prompt_ms="410"), timings(prompt_ms=True), timings(prompt_ms=-1), timings(prompt_ms=math.nan),
    timings(predicted_ms=math.inf), timings(predicted_ms=60_001), timings(predicted_n=45.0), timings(predicted_n=True),
    timings(predicted_n=-1), timings(predicted_n=CHAT_MAX_TOKENS + 1), timings(predicted_n=None),
])
def test_missing_or_malformed_timings_give_none_and_never_raise(payload: object) -> None:
    assert server_timing(payload, CHAT_MAX_TOKENS) is None


def test_the_servers_timings_are_kept_as_numbers_within_the_requested_limit() -> None:
    assert server_timing(timings(), CHAT_MAX_TOKENS) == ServerTiming(410.5, 900.0, 45)
    assert server_timing(timings(predicted_n=45), 40) is None  # more tokens than were asked for: malformed


def test_a_sample_takes_the_time_before_decoding_never_less_than_the_prompts_and_rejects_what_cannot_be() -> None:
    assert pace_sample(1500.0, ServerTiming(410.0, 900.0, 45)) == ((600.0, 20.0), None)
    assert pace_sample(1300.0, ServerTiming(410.0, 900.0, 45)) == ((410.0, 20.0), None)  # within the slack
    assert pace_sample(1000.0, ServerTiming(410.0, 900.0, 45)) == (None, "inconsistent")  # more than the request
    assert pace_sample(1500.0, ServerTiming(410.0, 0.0, 45)) == (None, "inconsistent")
    assert pace_sample(1500.0, ServerTiming(410.0, 100.0, PACE_MIN_TOKENS - 1)) == (None, "too_short")
    assert pace_sample(90_000.0, ServerTiming(0.0, 59_000.0, 10)) == (None, "inconsistent")  # 5.9 s a token
    assert pace_sample(math.nan, ServerTiming(410.0, 900.0, 45)) == (None, "inconsistent")


def test_the_pace_is_the_largest_of_the_latest_samples_and_a_reset_returns_to_the_fallback() -> None:
    pace = ChatPace()
    assert pace.current() is FALLBACK and pace.reset() is False
    pace.add((900.0, 15.0))
    pace.add((500.0, 25.0))
    assert pace.current() == Pace(900.0, 25.0, measured=True, samples=2)  # each the slowest seen
    for _ in range(PACE_SAMPLES):
        pace.add((400.0, 10.0))
    assert pace.current() == Pace(400.0, 10.0, measured=True, samples=PACE_SAMPLES)  # bounded: older ones dropped
    assert pace.reset() is True and pace.current() is FALLBACK
    with pytest.raises(ValueError):
        ChatPace(keep=0)


def test_an_overrun_is_a_sized_question_still_running_at_its_planned_end() -> None:
    plan = AnswerPlan("start", 85, "measured", 2800.0)
    assert not overran(plan, 2299.0) and overran(plan, 2300.0) and overran(plan, 3000.0)
    assert not overran(AnswerPlan("start", CHAT_MAX_TOKENS, "unbounded", None), 30_000.0) and not overran(None, 1e9)


@pytest.mark.parametrize(("text", "expected"), [
    ("A chair stands by the window. A person", ("A chair stands by the window.", "A person")),
    ("Nobody is visible. It is hard to tell because the image", ("Nobody is visible.",
                                                                  "It is hard to tell because the image")),
    ('He said "stop." Then the', ('He said "stop."', "Then the")),
    ("Is the door open? Maybe (the light is low.) But", ("Is the door open? Maybe (the light is low.)", "But")),
    ("The door is open.", ("The door is open.", "")),
    ("The value is 3.5 metres and", ("", "The value is 3.5 metres and")),
    ("1. A chair", ("", "1. A chair")),
    ("There is no", ("", "There is no")),
])
def test_trimming_keeps_complete_sentences_and_returns_the_rest_never_an_empty_answer(
        text: str, expected: tuple[str, str]) -> None:
    complete, rest = complete_sentences(text)
    assert (complete, rest) == expected
    assert (complete + " " + rest).split() == text.split()  # nothing lost: the rest is returned, never dropped
