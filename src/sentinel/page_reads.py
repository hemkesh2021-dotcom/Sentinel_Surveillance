"""When a status page first read an identity change (session 116, part 28; measurement only, PROPOSED).

A track becoming KNOWN, or an association being established, reaches a viewer in two further steps after the core
decided it: the runtime publishes its next status (``DemoRuntime._refresh_status``), and a page's request reads that
status (``DemoRuntime.page_snapshot``; the page asks every ``status_page.REFRESH_S``). This class times those two steps,
so that a run's records can tell the status refresh and the page's polling apart from tracking, the face worker and the
confirmation rule (``benchmarks/runner/identity_latency.py``). It changes nothing: no state, label or schedule depends
on it, and it exists only in ``--identity-retention`` validation runs.

What a record says, and no more: the first status published after the change, and the first page request that read a
status at or after that publication. It is the time the request read the status on this device, not the time a browser
drew it. A change superseded by another change of the same track before any status was published is dropped and
counted, since no page could have read it; a change no page read within ``UNREAD_AFTER_NS`` is dropped and counted too
(no page was polling: that is not a latency).

Bounded: at most ``PENDING_MAX`` waiting changes and ``DONE_MAX`` finished records; what does not fit is dropped,
oldest first, and counted. Numbers and fixed labels only: a track number and a state name, never an identity ID.

Threads: ``changed``, ``published`` and ``drain`` on the runtime's loop thread; ``read`` on a status-page thread. One
lock, held for a few list operations; a page request never waits for the loop beyond that.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from typing import Any

from .media.clock import NS_PER_SECOND

PENDING_MAX = 16  # changes waiting for a status publication or a page read
DONE_MAX = 64  # finished records waiting for the loop to write them
UNREAD_AFTER_NS = 30 * NS_PER_SECOND  # a published change no page read within this: no page was polling

IDENTITY = "identity"  # a raw identity transition (runtime.IdentityTransition)
ASSOCIATION = "association"  # an established association's change (identity.retention.AssociationChange)
CHANGES = (IDENTITY, ASSOCIATION)
COUNTERS = ("watched", "completed", "superseded", "unread", "dropped_pending", "dropped_done")


class PageReadLatency:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # [stream_epoch, track_id, change, to, changed_ns, published_ns or None]
        self._pending: deque[list[Any]] = deque()
        self._done: deque[dict[str, Any]] = deque()
        self._counters: Counter[str] = Counter()

    def changed(self, stream_epoch: int, track_id: int, change: str, to: str, mono_ns: int, *, watch: bool,
                holds: bool) -> None:
        """A change of one track's raw identity or association at ``mono_ns`` (the loop thread, once the live state
        holds it). ``watch``: this change is one to time (a track becoming KNOWN, an association established).
        ``holds``: what a watched change showed still holds after this one (the track is still KNOWN, the association
        still stands); when it does not, a waiting change of the same track and kind that no status has published yet
        is superseded: no page could have read it."""
        if change not in CHANGES:
            raise ValueError("change must be identity or association")
        with self._lock:
            if not holds:
                kept = [item for item in self._pending
                        if not (item[5] is None and item[:3] == [stream_epoch, track_id, change])]
                self._counters["superseded"] += len(self._pending) - len(kept)
                self._pending = deque(kept)
            if not watch:
                return
            self._counters["watched"] += 1
            self._pending.append([stream_epoch, track_id, change, to, mono_ns, None])
            while len(self._pending) > PENDING_MAX:
                self._pending.popleft()
                self._counters["dropped_pending"] += 1

    def published(self, now_ns: int) -> None:
        """The runtime swapped in a new status at ``now_ns`` (the loop thread, under the same lock as the swap): it
        holds every waiting change."""
        with self._lock:
            for item in self._pending:
                if item[5] is None:
                    item[5] = now_ns

    def read(self, now_ns: int) -> None:
        """A page's request read the current status at ``now_ns`` (a status-page thread, under the same lock as that
        read): every published change is now read."""
        with self._lock:
            waiting: deque[list[Any]] = deque()
            for item in self._pending:
                epoch, track, change, to, changed_ns, published_ns = item
                if published_ns is None or published_ns > now_ns:
                    waiting.append(item)
                elif now_ns - published_ns > UNREAD_AFTER_NS:
                    self._counters["unread"] += 1
                else:
                    self._counters["completed"] += 1
                    self._done.append({"stream_epoch": epoch, "track_id": track, "watched": change, "to": to,
                                       "changed_mono_ns": changed_ns, "published_mono_ns": published_ns,
                                       "read_mono_ns": now_ns})
                    while len(self._done) > DONE_MAX:
                        self._done.popleft()
                        self._counters["dropped_done"] += 1
            self._pending = waiting

    def drain(self) -> list[dict[str, Any]]:
        """The finished records since the last call, oldest first (the loop thread)."""
        with self._lock:
            done, self._done = list(self._done), deque()
        return done

    def numbers(self) -> dict[str, int]:
        """For the status: counts only."""
        with self._lock:
            return {"pending": len(self._pending), **{name: self._counters[name] for name in COUNTERS}}


def page_read_record(item: dict[str, Any], schema: int) -> dict[str, Any]:
    """One finished record as a run's identity record (kind ``page_read``): numbers and fixed labels only."""
    return {"identity": "page_read", "schema": schema, **item}
