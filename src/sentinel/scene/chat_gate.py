"""When chat may run (D77): the chat part of V2-29's pressure response (guide chapter 12), for the alpha.

Chat comes last in the guide's priority (capture and tracking, policy events, verification, incident summary, then
interactive chat). This gate closes chat to new questions, and asks for a running one to be cancelled, by the guide's
initial thresholds, on the same whole-device measure the operator guard and the step-4 limits use: pressure =
MemTotal - MemAvailable, in bytes (kB x 1024), decimal thresholds.

- **No new question:** pressure above PRESSURE_STOP_NEW_BYTES (4.7 GB) for PRESSURE_STOP_SAMPLES consecutive samples,
  or scene analysis missing its deadline MISSES_TO_STOP times within MISSES_WINDOW_S ("sustained deadline misses"; the
  guide gives no number: these two values are PROPOSED and need the maintainer's decision before a device check).
- **Cancel the running question:** pressure above PRESSURE_CANCEL_BYTES (5.0 GB): the guide's "cancel optional work".
- **Open again:** after pressure has stayed below PRESSURE_READMIT_BYTES (4.5 GB) for READMIT_S (60 s) with no scene
  deadline missed meanwhile ("healthy queues").
- **Fail closed:** a pressure that cannot be read closes chat until readings return and the re-admission holds.

Only chat is governed here. The guide's further steps (unloading the VLM, reducing face sampling, stopping the other
optional workers) are V2-29 proper and are not built: this gate never touches monitoring. Portable: the runtime feeds
it samples (``observe``) at most once a second, on its loop thread; the status page reads it from other threads.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from pathlib import Path
from typing import Any

from ..media.clock import NS_PER_SECOND

PRESSURE_STOP_NEW_BYTES = 4_700_000_000
PRESSURE_STOP_SAMPLES = 3
PRESSURE_CANCEL_BYTES = 5_000_000_000
PRESSURE_READMIT_BYTES = 4_500_000_000
READMIT_S = 60.0
MISSES_WINDOW_S = 60.0  # PROPOSED (D77): needs the maintainer's decision
MISSES_TO_STOP = 2  # PROPOSED (D77): needs the maintainer's decision
SAMPLE_INTERVAL_S = 1.0


def read_pressure(path: Path = Path("/proc/meminfo")) -> int | None:
    """MemTotal - MemAvailable in bytes (kB x 1024), or None where it cannot be read."""
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in lines:
        name, _, rest = line.partition(":")
        if name in ("MemTotal", "MemAvailable"):
            try:
                values[name] = int(rest.split()[0]) * 1024
            except (IndexError, ValueError):
                return None
    if len(values) != 2 or values["MemAvailable"] > values["MemTotal"]:
        return None
    return values["MemTotal"] - values["MemAvailable"]


class ChatGate:
    """Whether new questions may be asked, and whether a running one must be cancelled. Thread-safe."""

    def __init__(self, *, stop_new_bytes: int = PRESSURE_STOP_NEW_BYTES, stop_samples: int = PRESSURE_STOP_SAMPLES,
                 cancel_bytes: int = PRESSURE_CANCEL_BYTES, readmit_bytes: int = PRESSURE_READMIT_BYTES,
                 readmit_s: float = READMIT_S, misses_window_s: float = MISSES_WINDOW_S,
                 misses_to_stop: int = MISSES_TO_STOP) -> None:
        if not readmit_bytes < stop_new_bytes < cancel_bytes or stop_samples < 1 or misses_to_stop < 1:
            raise ValueError("the thresholds must order re-admission < stop new < cancel, with positive counts")
        self._stop_new = stop_new_bytes
        self._stop_samples = stop_samples
        self._cancel = cancel_bytes
        self._readmit = readmit_bytes
        self._readmit_ns = round(readmit_s * NS_PER_SECOND)
        self._window_ns = round(misses_window_s * NS_PER_SECOND)
        self._misses_to_stop = misses_to_stop
        self._lock = threading.Lock()
        self._open = True
        self._reason: str | None = None
        self._above = 0  # consecutive samples above stop_new
        self._calm_since: int | None = None  # when pressure was last seen at or above readmit, or unreadable
        self._misses: deque[int] = deque()  # when each deadline miss was seen
        self._seen_misses: int | None = None
        self._last_miss_ns: int | None = None
        self._cancel_reason: str | None = None
        self._above_cancel = False  # one cancellation request per spell above the cancel threshold
        self._pressure: int | None = None
        self._peak: int | None = None
        self.counters: Counter[str] = Counter()

    def observe(self, now_ns: int, pressure_bytes: int | None, deadline_misses: int) -> None:
        """One sample: the whole-device pressure (None if unreadable) and the scene lane's total deadline misses."""
        with self._lock:
            self.counters["samples"] += 1
            self._pressure = pressure_bytes
            if pressure_bytes is not None:
                self._peak = pressure_bytes if self._peak is None else max(self._peak, pressure_bytes)
            if self._seen_misses is not None and deadline_misses > self._seen_misses:
                for _ in range(min(deadline_misses - self._seen_misses, 100)):
                    self._misses.append(now_ns)
                self._last_miss_ns = now_ns
            self._seen_misses = deadline_misses
            while self._misses and now_ns - self._misses[0] > self._window_ns:
                self._misses.popleft()
            if pressure_bytes is None:
                self._above = 0
                self._calm_since = None
                self._close("memory pressure cannot be read, so chat waits until it can")
                return
            self._above = self._above + 1 if pressure_bytes > self._stop_new else 0
            if pressure_bytes >= self._readmit:
                self._calm_since = None
            elif self._calm_since is None:
                self._calm_since = now_ns
            if pressure_bytes > self._cancel:
                self._close(f"device memory pressure is above {self._cancel / 1e9:g} GB, so chat is stopped "
                            "and a running question cancelled")
                if not self._above_cancel:
                    self._above_cancel = True
                    self._cancel_reason = "device memory pressure: monitoring comes first"
                    self.counters["cancel_requests"] += 1
                return
            self._above_cancel = False
            if self._above >= self._stop_samples:
                self._close(f"device memory pressure has been above {self._stop_new / 1e9:g} GB for "
                            f"{self._stop_samples} samples, so new questions wait")
            elif len(self._misses) >= self._misses_to_stop:
                self._close(f"scene analysis missed its deadline {len(self._misses)} times within "
                            f"{self._window_ns / NS_PER_SECOND:g} s, so new questions wait")
            elif not self._open and self._calm_since is not None and now_ns - self._calm_since >= self._readmit_ns \
                    and (self._last_miss_ns is None or now_ns - self._last_miss_ns >= self._readmit_ns):
                self._open, self._reason = True, None
                self.counters["reopened"] += 1

    def allows_new(self) -> tuple[bool, str | None]:
        with self._lock:
            return self._open, self._reason

    def take_cancel(self) -> str | None:
        """The reason a running question must be cancelled, once per request for it; None otherwise."""
        with self._lock:
            reason, self._cancel_reason = self._cancel_reason, None
            return reason

    def summary(self) -> dict[str, Any]:
        """Numbers and fixed labels only."""
        with self._lock:
            return {"open": self._open, "reason": self._reason, "pressure_bytes": self._pressure,
                    "peak_pressure_bytes": self._peak, "misses_in_window": len(self._misses),
                    "limits": {"stop_new_bytes": self._stop_new, "stop_samples": self._stop_samples,
                               "cancel_bytes": self._cancel, "readmit_bytes": self._readmit,
                               "readmit_s": self._readmit_ns / NS_PER_SECOND,
                               "misses_window_s": self._window_ns / NS_PER_SECOND,
                               "misses_to_stop": self._misses_to_stop},
                    "counters": dict(sorted(self.counters.items()))}

    def _close(self, reason: str) -> None:
        # Called with the lock held.
        if self._open:
            self.counters["closed"] += 1
        self._open, self._reason = False, reason
