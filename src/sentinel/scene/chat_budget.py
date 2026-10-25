"""How long an answer a question has time for before the next periodic scene job ends it (session 114; D74, D81).

A question runs on the scene worker between scene jobs. A periodic job that arrives while it runs lets it finish within
the grace (analyzer.SIDE_GRACE_S), then ends it: the question is cancelled and its answer lost. Here each question gets
an answer limit (``max_tokens``) that fits the time it has, so that it usually ends by itself first. Monitoring is not
changed: scene jobs still go first, an incident's job still ends a question at once, and the grace's cut-off stays the
safety net when an estimate is wrong.

- **The window** is the scene worker's (analyzer.SideWindow): from when the question could start to the end of the next
  periodic job's grace. None while no periodic job is expected; the answer then has the full cap (CHAT_MAX_TOKENS).
- **The budget** keeps RELEASE_RESERVE_MS of the window free, takes off the question's non-decoding time (encoding the
  picture, processing the prompt with it, the transfer: never less than the server's own prompt time), and divides the
  rest by the time per answer token. Never more than the cap, which stays 120.
- **Measured from chat only.** An answered question's time on the worker, with llama-server's own timings for it
  (``timings``: ``prompt_ms``, ``predicted_n``, ``predicted_ms``), is one sample; ChatPace keeps the latest PACE_SAMPLES
  and uses the largest overhead and the largest time per token among them. Scene jobs are not a measure: their output
  is schema-constrained JSON from another prompt.
- **Fallback.** With no sample (a cold start; timings missing, malformed, inconsistent or too short to time decoding
  give none), and after a sized question still ran at its planned end and was ended for a scene job (the samples are
  then discarded: the load has changed), the budget uses FALLBACK_OVERHEAD_MS and FALLBACK_MS_PER_TOKEN, deliberately
  slow, and never less than CHAT_MIN_TOKENS. Such a question starts without waiting, so that a sample can be taken.
- **Too short.** With a measured pace, a window that does not fit CHAT_MIN_TOKENS makes the question wait for the next
  gap, within its time to start; after that it is given up, never started only to be cut off.
- **Shortened answers.** An answer stopped by a limit below the cap is trimmed to its complete sentences and labelled;
  the unfinished rest is kept apart and shown as such, never dropped silently. With no complete sentence there is no
  answer (``incomplete``), never an empty one.

The PROPOSED values below are for the maintainer's decision: they shape answers, not monitoring.
"""

from __future__ import annotations

import math
import re
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

CHAT_MIN_TOKENS = 30  # PROPOSED: about one or two short sentences, a quarter of the 120-token cap
# PROPOSED: a planned answer ends this long before the grace does. It is D81's allowance for an abort to end a request
# (MAX_CHAT_DELAY_MS 2000 less the 1500 ms grace); a planned answer that overruns by less still ends by itself, without
# the cut-off and llama-server's own cancellation (it notices a closed request about once a second).
RELEASE_RESERVE_MS = 500.0
# PROPOSED, deliberately slow: with these, a full 120-token answer would take 5.1 s, about the longest chat answer
# recorded (session 104: 5,255 ms, which includes its wait for the worker).
FALLBACK_OVERHEAD_MS = 1500.0
FALLBACK_MS_PER_TOKEN = 30.0
PACE_SAMPLES = 8
PACE_MIN_TOKENS = 8  # a shorter answer gives no sample: too few tokens to time decoding
PACE_MAX_MS = 60_000.0  # a server timing above this is malformed (requests are bounded by their 20 s timeout)
PACE_MAX_MS_PER_TOKEN = 1_000.0
TIMING_SLACK_MS = 50.0  # how far the server's prompt and decoding time may exceed the worker's measure of the request
_SENTENCE_END = re.compile(r"[.!?…][\"'’”)\]]*(?=\s|$)")
_LETTER = re.compile(r"[^\W\d_]")


@dataclass(frozen=True)
class ServerTiming:
    """llama-server's own timings for one completed request, checked: numbers only."""

    prompt_ms: float
    decode_ms: float
    tokens: int


@dataclass(frozen=True)
class Pace:
    """What an answer costs: the time before decoding (ms) and per answer token (ms); measured, or the fallback."""

    overhead_ms: float
    ms_per_token: float
    measured: bool
    samples: int = 0

    def tokens_within(self, ms: float) -> int:
        return max(0, math.floor((ms - self.overhead_ms) / self.ms_per_token))


FALLBACK = Pace(FALLBACK_OVERHEAD_MS, FALLBACK_MS_PER_TOKEN, measured=False)


@dataclass(frozen=True)
class AnswerPlan:
    """What a question does when the worker could start it: ``start`` with ``max_tokens``, ``wait`` for the next gap,
    or ``give_up``; ``basis`` is ``unbounded`` (no periodic job expected), ``measured`` or ``fallback``."""

    action: str
    max_tokens: int
    basis: str
    window_ms: float | None  # the time it had, from the scene worker; None when unbounded


def _number(value: object, low: float, high: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and low <= number <= high else None


def server_timing(payload: object, max_tokens: int) -> ServerTiming | None:
    """The checked ``timings`` of a completion payload, or None when they are missing or malformed; never raises."""
    timings = payload.get("timings") if isinstance(payload, Mapping) else None
    if not isinstance(timings, Mapping):
        return None
    prompt_ms = _number(timings.get("prompt_ms"), 0.0, PACE_MAX_MS)
    decode_ms = _number(timings.get("predicted_ms"), 0.0, PACE_MAX_MS)
    tokens = timings.get("predicted_n")
    if prompt_ms is None or decode_ms is None or isinstance(tokens, bool) or not isinstance(tokens, int) \
            or not 0 <= tokens <= max_tokens:
        return None
    return ServerTiming(prompt_ms, decode_ms, tokens)


def pace_sample(run_ms: float, timing: ServerTiming) -> tuple[tuple[float, float] | None, str | None]:
    """((overhead ms, ms per token), None) from an answered question's time on the worker and the server's timings
    for it, or (None, why there is no sample: ``too_short`` or ``inconsistent``)."""
    if timing.tokens < PACE_MIN_TOKENS:
        return None, "too_short"
    if timing.decode_ms <= 0 or not math.isfinite(run_ms) \
            or timing.prompt_ms + timing.decode_ms > run_ms + TIMING_SLACK_MS:
        return None, "inconsistent"  # the server's time cannot exceed the request it was part of
    per_token = timing.decode_ms / timing.tokens
    if per_token > PACE_MAX_MS_PER_TOKEN:
        return None, "inconsistent"
    return (max(run_ms - timing.decode_ms, timing.prompt_ms), per_token), None


class ChatPace:
    """The pace chat's own answers were measured at: the largest overhead and time per token among the latest
    ``keep`` samples, or FALLBACK while there are none. Thread-safe."""

    def __init__(self, keep: int = PACE_SAMPLES) -> None:
        if keep < 1:
            raise ValueError("keep must be at least 1")
        self._samples: deque[tuple[float, float]] = deque(maxlen=keep)
        self._lock = threading.Lock()

    def current(self) -> Pace:
        with self._lock:
            if not self._samples:
                return FALLBACK
            return Pace(max(overhead for overhead, _ in self._samples), max(per for _, per in self._samples),
                        measured=True, samples=len(self._samples))

    def add(self, sample: tuple[float, float]) -> None:
        with self._lock:
            self._samples.append(sample)

    def reset(self) -> bool:
        """Discard every sample (the load changed); True if there were any."""
        with self._lock:
            had = bool(self._samples)
            self._samples.clear()
            return had


def plan_answer(window_ms: float | None, pace: Pace, *, cap: int, may_wait: bool) -> AnswerPlan:
    """The answer limit a question has time for, or whether it waits for the next gap (``may_wait``: still within its
    time to start) or is given up. ``cap`` is the answer's own limit, which no plan exceeds."""
    if window_ms is None:
        return AnswerPlan("start", cap, "unbounded", None)
    tokens = min(cap, pace.tokens_within(window_ms - RELEASE_RESERVE_MS))
    if not pace.measured:
        return AnswerPlan("start", max(tokens, min(CHAT_MIN_TOKENS, cap)), "fallback", window_ms)
    if tokens >= CHAT_MIN_TOKENS:
        return AnswerPlan("start", tokens, "measured", window_ms)
    return AnswerPlan("wait" if may_wait else "give_up", tokens, "measured", window_ms)


def overran(plan: AnswerPlan | None, run_ms: float) -> bool:
    """Whether a sized question was still running at its planned end (the window less the reserve)."""
    return plan is not None and plan.window_ms is not None and run_ms >= plan.window_ms - RELEASE_RESERVE_MS


def complete_sentences(text: str) -> tuple[str, str]:
    """(its complete sentences, the unfinished rest) of an answer a limit stopped; ("", text) when no sentence is
    complete."""
    end = None
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
    if end is None or not _LETTER.search(text[:end]):
        return "", text
    return text[:end].rstrip(), text[end:].strip()
