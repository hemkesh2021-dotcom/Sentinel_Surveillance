"""Which kind of question the operator asked (session 116, part 28): fixed patterns, no model.

A question about what was **recorded** (what happened lately, who is in the room, whether alerts were triggered) is
answered from the system's own records (``sentinel.chat_evidence``), never by the scene model, which sees one still
picture and remembers nothing. Everything else is a question about the picture and takes the existing path unchanged.

Conservative on purpose: a question is routed away from the picture only when it clearly asks for records. "Is the
door open?", "is anyone at the door?" or "what is on the desk?" stay questions about the picture. A wrong guess toward
the picture costs a model answer that says what it sees; a wrong guess toward records would answer a different
question than the one asked.

Pure functions over the cleaned question text; nothing here is an instruction to anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VISUAL, HISTORY, WHO, ALERTS = "visual", "history", "who", "alerts"
INTENTS = (VISUAL, HISTORY, WHO, ALERTS)
RECORDED = (HISTORY, WHO, ALERTS)

DEFAULT_HISTORY_S = 600.0  # "what happened?" with no period: the last 10 minutes
DEFAULT_ALERTS_S = 86400.0  # "any alerts?" with no period: the last 24 hours
MAX_PERIOD_S = 7 * 86400.0  # a longer period is read as this; the answer states what was actually covered

_UNITS = {"second": 1.0, "sec": 1.0, "minute": 60.0, "min": 60.0, "hour": 3600.0, "hr": 3600.0, "day": 86400.0}
_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15,
          "twenty": 20, "thirty": 30, "half an": 0.5, "half a": 0.5}
_PERIOD = re.compile(
    r"\b(?:last|past|previous|within|in the last|in the past|over the last|over the past)\s+"
    r"(?:(\d{1,4}|half an|half a|an|a|one|two|three|four|five|ten|fifteen|twenty|thirty)\s+)?"
    r"(second|sec|minute|min|hour|hr|day)s?\b")
_SINCE_START = re.compile(r"\bsince (?:the )?(?:start(?:up)?|boot|launch|it started|you started|this run started)\b|"
                          r"\bthis (?:run|session)\b")
_TODAY = re.compile(r"\b(?:today|tonight|so far)\b")
_PICTURE = re.compile(r"\b(?:picture|image|photo|frame|pic|snapshot|screenshot)\b")  # names the picture: about it

_ALERTS = re.compile(r"\b(?:alerts?|incidents?|notifications?|notified|telegram)\b|"
                     r"\b(?:was|were|has|have|did|is|are)\b.*\btriggered\b|\bany(?:thing)? triggered\b")
_WHO = re.compile(
    r"^(?:who(?:'s| is| was|s| are)?\b.*\b(?:room|here|there|home|present|in view|on camera|around)\b|"
    r"who(?:'s| is)? (?:it|that|there)\b|"
    r"(?:is|are) (?:there )?(?:anyone|anybody|someone|somebody|any people|people) "
    r"(?:here|home|present|in the room|in view|on camera|around)\b|"
    r"how many (?:people|persons) (?:are |is )?(?:there |here |present |in the room|in view|on camera))")
# the past, or "has been happening"; "what is happening?" asks about the picture now
_HISTORY = re.compile(
    r"\bwhat(?:'s| has| have)? happened\b|\bwhat(?:'s| has| have| is) been happening\b|"
    r"\b(?:did|has) any ?thing happen(?:ed)?\b|\bany ?thing happen(?:ed)?\b|"
    r"\b(?:recent|any|latest) (?:activity|events?)\b|\b(?:summary|recap|timeline|log) of\b|"
    r"\bwhat (?:did|have) (?:you|we|the camera) (?:see|seen|record|recorded)\b")


@dataclass(frozen=True)
class Route:
    """``period_s``: the period the question named, or its intent's default; ``since_start``: it asked for everything
    since the runtime started; ``period_stated``: the question named the period itself."""

    intent: str
    period_s: float | None = None
    since_start: bool = False
    period_stated: bool = False


def period(text: str) -> tuple[float | None, bool]:
    """(seconds, since the start) a question names; (None, False) when it names neither."""
    if _SINCE_START.search(text) or _TODAY.search(text):
        return None, True
    found = _PERIOD.search(text)
    if not found:
        return None, False
    count, unit = found.group(1), found.group(2)
    number = 1.0 if count is None else float(_WORDS[count]) if count in _WORDS else float(count)
    return min(MAX_PERIOD_S, max(1.0, number * _UNITS[unit])), False


def route(question: str) -> Route:
    """The intent of one cleaned question. Alerts first (an alert question may also say "what happened"), then who,
    then history; anything else is about the picture."""
    text = " ".join(question.lower().replace("’", "'").split())
    seconds, since_start = period(text)
    stated = seconds is not None or since_start
    if _ALERTS.search(text):
        return Route(ALERTS, None if since_start else seconds or DEFAULT_ALERTS_S, since_start, stated)
    if _PICTURE.search(text):
        return Route(VISUAL)
    if _WHO.search(text):
        return Route(WHO)
    if _HISTORY.search(text):
        return Route(HISTORY, None if since_start else seconds or DEFAULT_HISTORY_S, since_start, stated)
    return Route(VISUAL)


# ---------------------------------------------------------------- follow-ups to a question about the picture

SUBJECT_MAX = 80
_PLACES = ("near", "next to", "by", "beside", "on", "in", "inside", "under", "underneath", "behind", "above", "below",
           "left of", "right of", "to the left of", "to the right of", "around", "at", "in front of", "on top of",
           "close to")
# a state asked of the subject ("is the door open?"): not part of the subject
_STATES = ("open", "closed", "shut", "off", "empty", "full", "visible", "locked", "unlocked", "running", "lit", "dark",
           "clean", "dirty", "present", "standing", "sitting", "moving", "wet", "dry", "broken", "there", "here")
_PLACE = re.compile(r"^(?:and |or |what about |how about )?((?:" + "|".join(sorted(_PLACES, key=len, reverse=True))
                    + r") .{1,120}?)\??$")
_OTHER = re.compile(r"^(?:and|or|what about|how about)(?: is there| are there)? (?:an? |the |any |some )?(.{1,80}?)\??$")
_PRONOUN = re.compile(r"\b(it|they|them|that one|this one)\b")
# the earlier question's subject: "is there a phone on the desk?" -> "a phone"; the place after it is dropped
_SUBJECT = re.compile(
    r"^(?:is there|are there|do you see|can you see|is|are|where is|where are|where's)\s+"
    r"((?:an? |the |any |some |my |his |her |their )?[\w' -]{1,60}?)"
    r"(?:\s+(?:" + "|".join(sorted(_PLACES, key=len, reverse=True)) + r")\b.*|\s+(?:" + "|".join(_STATES) + r")\b.*)?\??$")
_EXISTENCE = re.compile(r"^(is there|are there|do you see|can you see)\b")


def subject_of(question: str) -> str | None:
    """The thing an earlier question about the picture asked about, by fixed patterns; None when none can be read."""
    text = " ".join(question.lower().replace("’", "'").split())
    found = _SUBJECT.match(text)
    if not found:
        return None
    subject = found.group(1).strip()
    return subject[:SUBJECT_MAX] if subject and subject not in ("it", "there", "anyone", "anybody") else None


def resolve_followup(question: str, earlier: str | None) -> tuple[str | None, str | None]:
    """(how to read a follow-up, the active subject), from the question before it about the picture; (None, subject)
    when the question stands by itself. Deterministic and conservative:

    - a place alone ("near the water bottle?") continues the earlier question at that place:
      "Is there a phone on the desk?" then "near the water bottle?" reads "Is there a phone near the water bottle?";
    - "what about X?" / "and X?" asks the earlier question about X instead (said in those words, never rebuilt
      into a sentence of its own);
    - a pronoun ("is it on the table?") is read with the earlier subject named.

    The earlier *answer* is never used here: only the operator's own earlier question decides how the next is read."""
    if not earlier:
        return None, None
    subject = subject_of(earlier)
    text = " ".join(question.lower().replace("’", "'").split())
    if subject is None:
        return None, None
    lead = _EXISTENCE.match(" ".join(earlier.lower().split()))
    opener = lead.group(1).capitalize() if lead else "Is"
    place = _PLACE.match(text)
    if place and len(text) <= 140:
        return f"{opener} {subject} {place.group(1)}?", subject
    other = _OTHER.match(text)
    if other and lead and not _PLACE.match(text):
        thing = other.group(1).strip()
        return f"The earlier question (“{earlier.strip()[:200]}”), asked about {thing} instead.", thing[:SUBJECT_MAX]
    if _PRONOUN.search(text) and len(text) <= 160:
        named = _PRONOUN.sub(subject if subject.split()[0] in ("the", "my", "his", "her", "their")
                             else "the " + subject.split(" ", 1)[-1] if subject.split()[0] in ("a", "an", "any", "some")
                             else subject, text, count=1)
        return named[:1].upper() + named[1:] + ("" if named.endswith("?") else "?"), subject
    return None, subject
