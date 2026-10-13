"""Feature controls (D99; opt-in: ``sentinel run --feature-controls``): turning face recognition, scene analysis, chat,
the live view and incident snapshots on and off while monitoring keeps running.

Only what this runtime started at launch can be switched, and only between running and paused. Nothing is loaded,
started, admitted or unloaded at runtime (there is no runtime admission or unload in this build: V2-19, V2-29), no
second model or camera connection is ever started, and core monitoring (capture, detection, tracking, rules, incidents,
alerts) has no switch here. A feature this run did not start says that a restart is needed, with the operator's steps;
one that is not admitted stays unavailable and says why, whatever is requested.

- **Requested apart from actual.** An accepted request changes ``requested`` at once; the runtime loop applies it on its
  own thread at its next step, and the actual state is always read from the component itself (running, starting,
  stopping, off, busy, unavailable, error), never inferred from the request.
- **Serialized and bounded.** One loop thread applies changes, in the order they were accepted. At most one change per
  feature waits: a newer request for that feature replaces the waiting one, and a request for the state already
  requested changes nothing, so repeated clicks coalesce. At most RATE_LIMIT changes are accepted in any RATE_WINDOW_S;
  more are refused as busy.
- **Logged, bounded.** The latest LOG_KEPT entries (the change's number, its time, the feature, the requested state and
  the result) are shown on the page, and each change the loop applies is written once as a ``control`` line on the runtime's
  stdout (``drain_records``). No key, address, question, picture or identity is ever logged; refused attempts are
  counted, never logged one by one.

Authorization is the status page's (status_page: the operator's control key, the page's token, same origin). This
module trusts no caller and accepts only exact feature names and states. Thread-safe.
"""

from __future__ import annotations

import threading
from collections import Counter, OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .media.clock import NS_PER_SECOND, Clock

FEATURES = ("face", "scene", "chat", "live_view", "snapshots")
STATES = ("on", "off")
RATE_LIMIT = 6  # accepted changes, all features together, per RATE_WINDOW_S
RATE_WINDOW_S = 10.0
LOG_KEPT = 20
RECORDS_KEPT = 200  # control lines waiting for the loop's stdout; more are counted, not kept

# Results in the log and the control lines (fixed labels).
ACCEPTED = "accepted"  # recorded as requested; the loop applies it at its next step
REPLACED = "replaced"  # a waiting change replaced by a newer request for the same feature
APPLIED = "applied"  # the loop changed the feature
UNCHANGED = "unchanged"  # the loop found the feature already in the requested state
REFUSED = "refused"  # the loop could not apply it (the runtime was stopping)
OUTCOMES = (ACCEPTED, REPLACED, APPLIED, UNCHANGED, REFUSED)


@dataclass(frozen=True)
class Change:
    seq: int
    feature: str
    on: bool


class FeatureControls:
    """``switchable``: the features this run started and may pause and resume (the others are read only)."""

    def __init__(self, clock: Clock, switchable: Mapping[str, bool]) -> None:
        unknown = set(switchable) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown features {sorted(unknown)}")
        self._clock = clock
        self._lock = threading.Lock()
        self._switchable = {name: bool(switchable.get(name)) for name in FEATURES}
        self._requested = dict(self._switchable)  # what was started starts on
        self._waiting: OrderedDict[str, Change] = OrderedDict()
        self._seq = 0
        self._accepted_ns: deque[int] = deque()
        self._log: deque[dict[str, Any]] = deque(maxlen=LOG_KEPT)
        self._records: deque[dict[str, Any]] = deque()
        self.counters: Counter[str] = Counter()

    # ------------------------------------------------------------ request threads

    def request(self, feature: object, state: object) -> tuple[int, dict[str, Any]]:
        """(HTTP status, document) for an authorized request; feature and state must be exact."""
        if feature not in FEATURES or state not in STATES:
            with self._lock:
                self.counters["refused_invalid"] += 1
            return 400, {"state": "invalid", "reason": "unknown feature or state"}
        assert isinstance(feature, str)
        on = state == "on"
        now = self._clock.monotonic_ns()
        with self._lock:
            if not self._switchable[feature]:
                self.counters["refused_not_switchable"] += 1
                return 409, {"state": "refused", "feature": feature,
                             "reason": "this run did not start it: a restart is needed to change it"}
            if self._requested[feature] == on:
                self.counters["unchanged"] += 1
                return 200, {"state": UNCHANGED, "feature": feature, "requested": state}
            while self._accepted_ns and now - self._accepted_ns[0] >= RATE_WINDOW_S * NS_PER_SECOND:
                self._accepted_ns.popleft()
            if len(self._accepted_ns) >= RATE_LIMIT:
                self.counters["refused_busy"] += 1
                return 429, {"state": "busy", "feature": feature,
                             "reason": f"at most {RATE_LIMIT} changes every {RATE_WINDOW_S:g} s; try again shortly"}
            self._accepted_ns.append(now)
            self._seq += 1
            change = Change(self._seq, feature, on)
            replaced = self._waiting.pop(feature, None)
            if replaced is not None:
                self.counters["replaced"] += 1
                self._entry(replaced, REPLACED)
            self._waiting[feature] = change
            self._requested[feature] = on
            self.counters["accepted"] += 1
            self._entry(change, ACCEPTED)
            return 202, {"state": ACCEPTED, "change": change.seq, "feature": feature, "requested": state}

    def refused(self, why: str) -> None:
        """An attempt the status page refused before it got here (a wrong key, another origin): counted only."""
        with self._lock:
            self.counters[f"refused_{why}"] += 1

    # ------------------------------------------------------------ the loop thread

    def take(self) -> list[Change]:
        """The waiting changes, oldest first; each is applied (or refused) by the caller, then ``done``."""
        with self._lock:
            changes = list(self._waiting.values())
            self._waiting.clear()
            return changes

    def done(self, change: Change, outcome: str) -> None:
        if outcome not in (APPLIED, UNCHANGED, REFUSED):
            raise ValueError(f"not a loop outcome: {outcome!r}")
        with self._lock:
            self.counters[outcome] += 1
            entry = self._entry(change, outcome)
            if len(self._records) >= RECORDS_KEPT:
                self.counters["records_dropped"] += 1
            else:
                self._records.append({"control": "feature", **entry, "mono_ns": self._clock.monotonic_ns()})

    def drain_records(self) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._records)
            self._records.clear()
            return records

    # ------------------------------------------------------------ reading

    def requested(self, feature: str) -> bool:
        with self._lock:
            return self._requested[feature]

    def waiting(self, feature: str) -> bool:
        with self._lock:
            return feature in self._waiting

    def switchable(self, feature: str) -> bool:
        return self._switchable[feature]

    def view(self) -> dict[str, Any]:
        with self._lock:
            return {"log": list(self._log), "counters": dict(sorted(self.counters.items())),
                    "rate": {"limit": RATE_LIMIT, "window_s": RATE_WINDOW_S}}

    def _entry(self, change: Change, outcome: str) -> dict[str, Any]:
        entry = {"change": change.seq, "at_utc": self._clock.utc_now().isoformat(), "feature": change.feature,
                 "requested": "on" if change.on else "off", "result": outcome}
        self._log.append(entry)
        return dict(entry)
