"""A bounded, structured history of what the runtime itself observed, for recorded-fact answers (session 116, part 28).

Chat is asked "what happened in the last 10 minutes?". The scene model cannot say: it sees one still picture. The
incident database cannot say either: it holds incidents, not every observation. This keeps the missing part: a small,
in-memory list of the runtime's own state changes, so that an answer can cite what was recorded and say exactly what
it covers.

**What is kept** (fixed kinds, numbers and fixed labels; never a picture, an embedding, a similarity value or an
identity ID): the runtime's start; the video state changing (fresh, stale, offline) and a new stream epoch; occupancy
changing; a confirmed track number appearing and going; a track's raw identity state changing (an enrolled identity is
"an enrolled person", never a name or an ID); an established association beginning, changing or ending
(--identity-retention runs); each current scene report (its fields, with its summary cleaned and cut to
``SUMMARY_MAX`` characters: the model's own words, unverified, kept as data).

**Capacity and retention**: at most ``EVENTS_MAX`` events and ``SCENES_MAX`` scene reports, nothing older than
``KEEP_S`` (60 minutes); the oldest are dropped first and counted. In memory only: it starts empty at each start of the
runtime, and it says so. At roughly 300 bytes an item that is under a megabyte.

**Coverage**, reported with every reading: where the history begins (the runtime's start, or the oldest moment still
fully kept after drops), the period asked for, and whether the two meet. "Nothing recorded in the covered interval" is
therefore distinguishable from "not covered", and neither says that nothing happened: what no detector, tracker or
scene check noticed is not here.

Threads: ``observe`` and the ``note_*`` methods on the runtime's loop thread; ``window`` on status-page threads. One
lock.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from .media.clock import NS_PER_SECOND, Clock

EVENTS_MAX = 1000
SCENES_MAX = 900
KEEP_S = 3600.0
SUMMARY_MAX = 200
KINDS = ("runtime_started", "video", "stream_epoch", "occupancy", "track_appeared", "track_gone", "identity",
         "association", "scene_report")


def _utc(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class RecentHistory:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque()
        self._scenes: deque[dict[str, Any]] = deque()
        self._started_ns = clock.monotonic_ns()
        self._started_utc = clock.utc_now()
        self._dropped = {"events": 0, "scenes": 0}
        self._dropped_until_ns: int | None = None  # the newest moment something was dropped from: not covered before
        # what the last publication showed, to record changes only
        self._video: str | None = None
        self._epoch: int | None = None
        self._occupancy: str | None = None
        self._tracks: set[tuple[int, int]] = set()
        self._scene_id: str | None = None
        self._add("runtime_started", {})

    @property
    def started_utc(self) -> datetime:
        """When this history began: the runtime's start."""
        return self._started_utc

    # ------------------------------------------------------------ writing (the loop thread)

    def observe(self, live: Any) -> None:
        """One publication of the runtime's live state (live_state.LiveState): records what changed since the last."""
        video = live.video.value
        if video != self._video:
            self._add("video", {"from": self._video, "to": video})
            self._video = video
        confirmed = {(p.stream_epoch, p.track_id) for p in live.people if p.status.value == "confirmed"}
        epochs = {epoch for epoch, _ in confirmed}
        if epochs and max(epochs) != self._epoch:
            if self._epoch is not None:
                self._add("stream_epoch", {"from": self._epoch, "to": max(epochs)})
            self._epoch = max(epochs)
        for epoch, track in sorted(confirmed - self._tracks):
            self._add("track_appeared", {"stream_epoch": epoch, "track_id": track})
        for epoch, track in sorted(self._tracks - confirmed):
            self._add("track_gone", {"stream_epoch": epoch, "track_id": track})
        self._tracks = confirmed
        occupancy = live.occupancy.value
        if occupancy != self._occupancy:
            self._add("occupancy", {"from": self._occupancy, "to": occupancy, "confirmed_people": len(confirmed)})
            self._occupancy = occupancy
        report, evidence_id = live.scene_report, live.scene_evidence_id
        if report is not None and evidence_id is not None and evidence_id != self._scene_id:
            self._scene_id = evidence_id
            self._add("scene_report", {
                "evidence_id": str(evidence_id), "persons_visible": report.persons_visible,
                "fire_or_smoke": report.fire_or_smoke, "threat": report.threat.value,
                "uncertainty": report.uncertainty.value,
                "summary": " ".join(str(report.summary).split())[:SUMMARY_MAX]}, scene=True)

    def note_identity(self, stream_epoch: int, track_id: int, from_state: str, to_state: str) -> None:
        """A track's raw identity state changed (runtime.IdentityTransition): states only, never an identity ID."""
        if from_state != to_state:
            self._add("identity", {"stream_epoch": stream_epoch, "track_id": track_id, "from": from_state,
                                   "to": to_state})

    def note_association(self, stream_epoch: int, track_id: int, from_state: str, to_state: str, reason: str) -> None:
        """An established association changed (identity.retention.AssociationChange): states and its fixed reason."""
        self._add("association", {"stream_epoch": stream_epoch, "track_id": track_id, "from": from_state,
                                  "to": to_state, "reason": reason})

    def _add(self, kind: str, fields: dict[str, Any], *, scene: bool = False) -> None:
        now = self._clock.monotonic_ns()
        item = {"kind": kind, "mono_ns": now, "utc": _utc(self._clock.utc_now()), **fields}
        with self._lock:
            target, limit, name = ((self._scenes, SCENES_MAX, "scenes") if scene
                                   else (self._events, EVENTS_MAX, "events"))
            target.append(item)
            while len(target) > limit:
                self._drop(target, name)
            self._prune(now)

    def _drop(self, target: deque[dict[str, Any]], name: str) -> None:
        gone = target.popleft()
        self._dropped[name] += 1
        if self._dropped_until_ns is None or gone["mono_ns"] > self._dropped_until_ns:
            self._dropped_until_ns = gone["mono_ns"]

    def _prune(self, now: int) -> None:
        floor = now - round(KEEP_S * NS_PER_SECOND)
        for target, name in ((self._events, "events"), (self._scenes, "scenes")):
            while target and target[0]["mono_ns"] < floor:
                self._drop(target, name)

    # ------------------------------------------------------------ reading (status-page threads)

    def window(self, period_s: float | None) -> dict[str, Any]:
        """What was recorded in the last ``period_s`` seconds (None: since the runtime started), oldest first, with
        what that reading covers."""
        now = self._clock.monotonic_ns()
        now_utc = self._clock.utc_now()
        with self._lock:
            self._prune(now)
            kept_from = self._started_ns if self._dropped_until_ns is None else self._dropped_until_ns
            wanted_from = self._started_ns if period_s is None else now - round(period_s * NS_PER_SECOND)
            start = max(kept_from, wanted_from)
            events = [dict(item) for item in self._events if item["mono_ns"] >= start]
            scenes = [dict(item) for item in self._scenes if item["mono_ns"] >= start]
            dropped = dict(self._dropped)

        def at(ns: int) -> str:
            return _utc(now_utc - timedelta(microseconds=(now - ns) // 1000))

        return {
            "now_utc": _utc(now_utc), "runtime_started_utc": _utc(self._started_utc),
            "requested_from_utc": None if period_s is None else at(wanted_from),
            "covered_from_utc": at(start),
            # the whole requested period is in the history: nothing was dropped from it, and it began after the start
            "complete": wanted_from >= kept_from,
            "limited_by": (None if wanted_from >= kept_from
                           else "runtime_start" if self._dropped_until_ns is None else "capacity_or_age"),
            "events": events, "scenes": scenes, "dropped": dropped,
            "capacity": {"events": EVENTS_MAX, "scenes": SCENES_MAX, "keep_s": KEEP_S},
        }

    def numbers(self) -> dict[str, int]:
        """Counts only (tests and tools; the status is unchanged by this history)."""
        with self._lock:
            return {"events": len(self._events), "scenes": len(self._scenes),
                    "dropped_events": self._dropped["events"], "dropped_scenes": self._dropped["scenes"]}


def by_time(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: item["mono_ns"])
