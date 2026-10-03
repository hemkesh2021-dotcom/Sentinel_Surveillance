"""Zone rules over current person tracks (guide chapter 9; V2-13, demo form).

A zone rule watches confirmed, current tracks. A track is *in* a zone on a
frame when it was really detected on that frame (not a tracker prediction),
the anchor point of its box lies inside the zone polygon, and the frame's UTC
ingest time falls inside the zone's schedule. One continuous presence of one
track in one zone is an *episode*:

- it starts at the track's first in-zone detection;
- it becomes an observation (phase ENTERED) once in-zone detections span
  ``min_duration`` (entry persistence for ``restricted``, the dwell threshold
  for ``dwell``), at most once per episode;
- in-zone detections further apart than ``gap_tolerance`` start a new
  episode, and so does a new stream epoch (tracker IDs are per epoch);
- it ends (phase ENDED, only if it was ENTERED) when the track has not been in
  the zone for longer than ``gap_tolerance``, the track is no longer current,
  the video is not fresh (stall, outage) or the stream reconnected.

Durations come from source ingest times on the monotonic clock, so they do not
move with wall-clock steps or late evaluation. Only schedule membership reads
wall-clock time (see ``rules.schedule``).

The rule reads no identity and no scene verdict. A known person in a restricted
zone is still observed; an identity can add context to the incident but never
removes the rule observation (guide ch. 9: identity is separate from access
rules). ENTERED creates a candidate incident; the incident service (V2-14)
decides what to open, and deduplicates by ``observation_id``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import NonNegativeInt

from ..config import ZoneConfig
from ..contracts import (
    Contract,
    FrameKey,
    Identifier,
    StreamIdentity,
    TrackKey,
    TrackObservation,
    TrackStatus,
    UtcDatetime,
)
from ..media.clock import MonoInstant
from .geometry import anchor_point, contains
from .scene_hazard import Severity

NS_PER_MS = 1_000_000
ZONE_KINDS = {"restricted": "zone.restricted_entry", "dwell": "zone.dwell"}


class ZonePhase(str, Enum):
    ENTERED = "entered"
    ENDED = "ended"


class ZoneObservation(Contract):
    """A rule observation; the incident service turns ENTERED into a candidate incident."""

    contract_version: Literal[1] = 1
    observation_id: Identifier  # unique per episode and phase; the incident service deduplicates on it
    episode_id: Identifier  # shared by the ENTERED and ENDED observations of one presence
    kind: Literal["zone.restricted_entry", "zone.dwell"]
    phase: ZonePhase
    zone_id: Identifier
    zone_revision: Identifier
    severity: Severity
    track: TrackKey
    first_source: FrameKey  # first in-zone detection of the episode
    last_source: FrameKey  # latest in-zone detection
    first_mono_ns: NonNegativeInt
    last_mono_ns: NonNegativeInt
    observed_utc: UtcDatetime  # of the latest in-zone detection
    duration_ms: NonNegativeInt  # between the first and latest in-zone detections
    reason: str


@dataclass
class _Episode:
    episode_id: str
    first: TrackObservation
    last: TrackObservation  # latest in-zone detection
    entered: bool = False
    out_reason: str = "left the zone"  # why the latest detection did not count as in the zone


class ZoneRule:
    """One configured zone. Bounded: one episode per current track."""

    def __init__(self, zone: ZoneConfig) -> None:
        self.zone = zone
        self._kind = ZONE_KINDS[zone.rule]
        self._polygon = zone.points
        self._anchor = zone.anchor_kind
        self._schedule = zone.schedule.build() if zone.schedule is not None else None
        self._severity = Severity(zone.severity)
        self._revision = zone.revision
        self._episodes: dict[TrackKey, _Episode] = {}
        self._seen: dict[TrackKey, int] = {}  # latest processed detection time per current track

    @property
    def active_episodes(self) -> int:
        return len(self._episodes)

    def evaluate(
        self,
        tracks: Sequence[TrackObservation],
        live: StreamIdentity | None,
        now: MonoInstant,
    ) -> list[ZoneObservation]:
        """Call on every runtime step with the current tracks (empty while video is not live)."""
        out: list[ZoneObservation] = []
        current = {track.key: track for track in tracks}
        for key in list(self._episodes):
            if key not in current:
                episode = self._episodes.pop(key)
                out += self._ended(episode, self._gone_reason(key, live))
        self._seen = {key: ns for key, ns in self._seen.items() if key in current}
        for key, track in current.items():
            if track.predicted or track.last_measured_mono_ns <= self._seen.get(key, -1):
                continue  # predictions are not measurements; each detection counts once
            self._seen[key] = track.last_measured_mono_ns
            out += self._detection(key, track)
        for key, episode in list(self._episodes.items()):
            if now.ns - episode.last.observed_mono_ns > self.zone.gap_tolerance_ns:
                del self._episodes[key]
                out += self._ended(episode, episode.out_reason)
        return out

    def _detection(self, key: TrackKey, track: TrackObservation) -> list[ZoneObservation]:
        out: list[ZoneObservation] = []
        why_out = self._why_not_in_zone(track)
        episode = self._episodes.get(key)
        if why_out is not None:
            if episode is not None:
                episode.out_reason = why_out
            return out
        if episode is not None and track.observed_mono_ns - episode.last.observed_mono_ns > self.zone.gap_tolerance_ns:
            del self._episodes[key]
            out += self._ended(episode, episode.out_reason)
            episode = None
        if episode is None:
            episode = _Episode(episode_id=self._episode_id(track), first=track, last=track)
            self._episodes[key] = episode
        episode.last = track
        episode.out_reason = "left the zone"
        if not episode.entered and track.observed_mono_ns - episode.first.observed_mono_ns >= self.zone.min_duration_ns:
            episode.entered = True
            out.append(self._observation(episode, ZonePhase.ENTERED, self._entered_reason(episode)))
        return out

    def _why_not_in_zone(self, track: TrackObservation) -> str | None:
        if track.status is not TrackStatus.CONFIRMED:
            return "track not confirmed"
        if not contains(self._polygon, anchor_point(track.box, self._anchor)):
            return "left the zone"
        if self._schedule is not None and not self._schedule.active_at(track.observed_utc):
            return "outside the zone's schedule"
        return None

    @staticmethod
    def _gone_reason(key: TrackKey, live: StreamIdentity | None) -> str:
        if live is None:
            return "no fresh video (stall, outage or disconnect)"
        if (key.camera_id, key.boot_id, key.run_id, key.stream_epoch) != (
            live.camera_id,
            live.boot_id,
            live.run_id,
            live.stream_epoch,
        ):
            return "stream reconnected (new epoch)"
        return "track expired (not detected for the track expiry period)"

    def _ended(self, episode: _Episode, why: str) -> list[ZoneObservation]:
        if not episode.entered:
            return []  # a presence too short to count leaves no observation
        return [self._observation(episode, ZonePhase.ENDED, f"presence in zone {self.zone.zone_id!r} ended: {why}")]

    def _entered_reason(self, episode: _Episode) -> str:
        seconds = (episode.last.observed_mono_ns - episode.first.observed_mono_ns) / 1e9
        what = "restricted zone" if self.zone.rule == "restricted" else "zone"
        when = "" if self._schedule is None else " during its schedule"
        return (
            f"confirmed track {episode.last.track_id} in {what} {self.zone.zone_id!r}{when} "
            f"for {seconds:.1f} s (threshold {self.zone.min_duration_s:g} s); identity not considered"
        )

    def _episode_id(self, first: TrackObservation) -> str:
        frame = first.frame
        digest = hashlib.sha256(
            "|".join(
                (
                    self.zone.zone_id,
                    self._revision,
                    frame.camera_id,
                    frame.boot_id,
                    frame.run_id,
                    str(frame.stream_epoch),
                    str(frame.frame_seq),
                    str(first.track_id),
                )
            ).encode()
        ).hexdigest()[:24]
        return f"zone-{digest}"

    def _observation(self, episode: _Episode, phase: ZonePhase, reason: str) -> ZoneObservation:
        first, last = episode.first, episode.last
        return ZoneObservation(
            observation_id=f"{episode.episode_id}.{phase.value}",
            episode_id=episode.episode_id,
            kind=self._kind,
            phase=phase,
            zone_id=self.zone.zone_id,
            zone_revision=self._revision,
            severity=self._severity,
            track=last.key,
            first_source=first.frame,
            last_source=last.frame,
            first_mono_ns=first.observed_mono_ns,
            last_mono_ns=last.observed_mono_ns,
            observed_utc=last.observed_utc,
            duration_ms=(last.observed_mono_ns - first.observed_mono_ns) // NS_PER_MS,
            reason=reason[:300],
        )


class ZoneRules:
    """All enabled zones of one camera."""

    def __init__(self, zones: Sequence[ZoneConfig]) -> None:
        self._rules = [ZoneRule(zone) for zone in zones if zone.enabled]

    @property
    def active_episodes(self) -> int:
        return sum(rule.active_episodes for rule in self._rules)

    def evaluate(
        self,
        tracks: Sequence[TrackObservation],
        live: StreamIdentity | None,
        now: MonoInstant,
    ) -> list[ZoneObservation]:
        out: list[ZoneObservation] = []
        for rule in self._rules:
            out += rule.evaluate(tracks, live, now)
        return out
