"""Rule schedules in an IANA timezone (guide chapter 9).

A schedule is a set of daily local-time windows ``[start, end)`` in one IANA
timezone. A window whose end is not after its start spans midnight: 22:00–06:00
is active from 22:00 until 06:00 the next morning. Membership is decided from
the source frame's UTC ingest time converted to local wall-clock time, so
daylight-saving changes follow the zone's rules and a late evaluation does not
change the answer. This is the one place where wall-clock time takes part in a
decision: a schedule is wall-clock policy. Durations, persistence and gaps are
still measured on the monotonic clock.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..media.clock import require_utc

_HHMM = re.compile(r"^([01][0-9]|2[0-3]):([0-5][0-9])$")
MINUTES_PER_DAY = 24 * 60


def parse_hhmm(text: str) -> int:
    """Minutes after local midnight for 'HH:MM' (00:00 to 23:59)."""
    match = _HHMM.match(text)
    if match is None:
        raise ValueError(f"expected a 24-hour time 'HH:MM' from 00:00 to 23:59, got {text!r}")
    return int(match.group(1)) * 60 + int(match.group(2))


def load_timezone(name: str) -> ZoneInfo:
    """An IANA timezone such as 'Asia/Kolkata'. Fixed offsets and abbreviations are rejected."""
    if "/" not in name and name != "UTC":
        raise ValueError(f"expected an IANA timezone such as 'Asia/Kolkata', got {name!r}")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"unknown IANA timezone {name!r}") from None


@dataclass(frozen=True)
class Window:
    start_minute: int
    end_minute: int

    def __post_init__(self) -> None:
        if self.start_minute == self.end_minute:
            raise ValueError("a window must not start and end at the same time")

    @property
    def spans_midnight(self) -> bool:
        return self.end_minute < self.start_minute

    def contains(self, minute: int) -> bool:
        if self.spans_midnight:
            return minute >= self.start_minute or minute < self.end_minute
        return self.start_minute <= minute < self.end_minute


@dataclass(frozen=True)
class Schedule:
    timezone: ZoneInfo
    windows: tuple[Window, ...]

    @classmethod
    def parse(cls, timezone: str, windows: Sequence[tuple[str, str]]) -> Schedule:
        return cls(
            timezone=load_timezone(timezone),
            windows=tuple(Window(parse_hhmm(start), parse_hhmm(end)) for start, end in windows),
        )

    def local(self, utc: datetime) -> datetime:
        return require_utc(utc).astimezone(self.timezone)

    def active_at(self, utc: datetime) -> bool:
        local = self.local(utc)
        minute = local.hour * 60 + local.minute
        return any(window.contains(minute) for window in self.windows)
