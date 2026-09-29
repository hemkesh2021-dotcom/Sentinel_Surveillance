"""Injectable clocks (guide chapter 6).

Monotonic time decides ages, TTLs and dwell within one boot; UTC is for display
and audit only. A monotonic reading means nothing outside the boot that produced
it, so readings carry their boot ID and refuse arithmetic across boots.
"""

from __future__ import annotations

import functools
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

NS_PER_SECOND = 1_000_000_000
LINUX_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


class CrossBootComparisonError(ValueError):
    """Monotonic readings from different boots were compared."""


@dataclass(frozen=True, slots=True)
class MonoInstant:
    """A monotonic clock reading, comparable only with readings from the same boot."""

    boot_id: str
    ns: int

    def ns_since(self, earlier: MonoInstant) -> int:
        """Nanoseconds from ``earlier`` to this reading (negative if ``earlier`` is later)."""
        if self.boot_id != earlier.boot_id:
            raise CrossBootComparisonError(
                f"monotonic readings from boot {earlier.boot_id!r} and boot "
                f"{self.boot_id!r} are not comparable"
            )
        return self.ns - earlier.ns


def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"expected a timezone-aware UTC datetime, got {value.isoformat()}")
    return value


class Clock(ABC):
    """Time source injected into ingest, rules, workers and tests."""

    @property
    @abstractmethod
    def boot_id(self) -> str:
        """ID of the boot that this clock's monotonic readings belong to."""

    @abstractmethod
    def monotonic_ns(self) -> int:
        """Monotonic nanoseconds; never decreases within a boot."""

    @abstractmethod
    def utc_now(self) -> datetime:
        """Timezone-aware UTC wall-clock time, for display and audit only."""

    def mono(self) -> MonoInstant:
        return MonoInstant(self.boot_id, self.monotonic_ns())


@functools.cache
def _process_boot_id() -> str:
    return f"process-{uuid.uuid4()}"


def read_boot_id(path: Path = LINUX_BOOT_ID_PATH) -> str:
    """Return the kernel's per-boot ID, or a per-process ID where there is none.

    The fallback (macOS/Windows development hosts) is stricter than a real boot
    ID: monotonic readings kept from an earlier process count as another boot.
    """
    try:
        boot_id = path.read_text(encoding="ascii").strip()
    except OSError:
        return _process_boot_id()
    return boot_id or _process_boot_id()


class SystemClock(Clock):
    """Host clock: CLOCK_MONOTONIC on Linux, system time for UTC."""

    def __init__(self) -> None:
        self._boot_id = read_boot_id()

    @property
    def boot_id(self) -> str:
        return self._boot_id

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def utc_now(self) -> datetime:
        return datetime.now(timezone.utc)


class FakeClock(Clock):
    """Deterministic clock for tests and replay.

    ``advance`` moves monotonic and wall-clock time together. ``step_utc`` moves
    only the wall clock, like an NTP step or a manual change. ``reboot`` starts
    a new boot whose monotonic readings are unrelated to the previous boot's.
    """

    def __init__(
        self,
        *,
        boot_id: str = "fake-boot-1",
        mono_ns: int = NS_PER_SECOND,
        utc: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc),
    ) -> None:
        if mono_ns < 0:
            raise ValueError("monotonic time cannot be negative")
        self._boot_id = boot_id
        self._mono_ns = mono_ns
        self._utc = require_utc(utc)

    @property
    def boot_id(self) -> str:
        return self._boot_id

    def monotonic_ns(self) -> int:
        return self._mono_ns

    def utc_now(self) -> datetime:
        return self._utc

    def advance(self, seconds: float = 0.0, *, ns: int = 0) -> None:
        delta_ns = round(seconds * NS_PER_SECOND) + ns
        if delta_ns < 0:
            raise ValueError("monotonic time cannot go backwards")
        self._mono_ns += delta_ns
        self._utc += timedelta(microseconds=delta_ns / 1_000)

    def step_utc(self, delta: timedelta) -> None:
        self._utc += delta

    def reboot(
        self,
        boot_id: str,
        *,
        mono_ns: int = NS_PER_SECOND,
        downtime: timedelta = timedelta(seconds=30),
    ) -> None:
        if boot_id == self._boot_id:
            raise ValueError("a reboot must produce a new boot ID")
        if mono_ns < 0:
            raise ValueError("monotonic time cannot be negative")
        self._boot_id = boot_id
        self._mono_ns = mono_ns
        self._utc += downtime
