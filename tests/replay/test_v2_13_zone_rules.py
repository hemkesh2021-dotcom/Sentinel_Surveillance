"""V2-13 (demo form): restricted-zone and dwell rules on deterministic replays.

Guide ch. 9 ("fresh confirmed track inside zone during configured schedule";
anchor, persistence, gap tolerance; IANA timezone; intervals across midnight;
"Unauthorized zone entry should create an incident independently of
recognition/VLM"; a known identity "must not automatically bypass every
restricted-zone rule"). Timelines are synthetic. Where it matters, each test
also shows v1's only person alert (B0 snapshot, ``v1_intruder_alert``) for the
same situation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from b0_v1_snapshot import v1_intruder_alert, v1_is_restricted_time
from identity_harness import ALICE_FACE, ENROLLMENT
from timeline_builder import FRAME_MS, TimelineBuilder, frame_at
from zone_harness import BESIDE_ZONE, IN_ZONE, OUTSIDE, person, run_zones, zone

from sentinel.identity.state import IdentityState
from sentinel.live_state import Capability
from sentinel.rules.scene_hazard import Severity
from sentinel.rules.zones import ZoneObservation

KOLKATA = ZoneInfo("Asia/Kolkata")
NIGHT = {"timezone": "Asia/Kolkata", "windows": [{"start": "22:00", "end": "06:00"}]}


def local(*args: int) -> datetime:
    """A Kolkata wall-clock time as the UTC instant the device clock would show."""
    return datetime(*args, tzinfo=KOLKATA).astimezone(timezone.utc)


def local_time(observation: ZoneObservation) -> str:
    return observation.observed_utc.astimezone(KOLKATA).strftime("%H:%M:%S")


def test_presence_across_midnight_is_one_episode() -> None:
    leaves = 180_000  # in the zone from 23:59:00 to 00:02:00 local
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(leaves, persons=person(IN_ZONE))
        .frames_until(leaves + 3000, persons=person(OUTSIDE))
        .build()
    )
    run = run_zones(timeline, zones=[zone(schedule=NIGHT)], utc_start=local(2026, 10, 3, 23, 59, 0))

    [(entered_ms, entered)] = run.entered
    assert 1000 <= entered_ms < 1000 + FRAME_MS
    assert local_time(entered) == "23:59:01"
    assert entered.kind == "zone.restricted_entry" and entered.severity is Severity.WARNING
    # Midnight does not split the presence: it ends only when the person leaves.
    last_in_zone = frame_at(leaves) - FRAME_MS
    [(ended_ms, ended)] = run.ended
    assert last_in_zone + 1000 < ended_ms <= last_in_zone + 1000 + FRAME_MS
    assert ended.episode_id == entered.episode_id
    assert local_time(ended) == "00:01:59" and ended.duration_ms == last_in_zone
    assert ended.reason.endswith("ended: left the zone")
    assert run.core.diagnostics()["zone_episodes"] == 0


def test_presence_ends_when_the_window_closes_at_6am() -> None:
    timeline = TimelineBuilder().connect(0).frames_until(60_000, persons=person(IN_ZONE)).build()
    run = run_zones(timeline, zones=[zone(schedule=NIGHT)], utc_start=local(2026, 10, 4, 5, 59, 30))

    [(entered_ms, entered)] = run.entered
    assert 1000 <= entered_ms < 1000 + FRAME_MS and local_time(entered) == "05:59:31"
    # From 06:00:00 the detections are outside the schedule; one gap tolerance later it ends.
    last_in_window = frame_at(30_000) - FRAME_MS
    [(ended_ms, ended)] = run.ended
    assert last_in_window + 1000 < ended_ms <= last_in_window + 1000 + FRAME_MS
    assert local_time(ended) == "05:59:59"
    assert ended.reason.endswith("outside the zone's schedule")


def test_window_opening_at_10pm_counts_persistence_from_22_00() -> None:
    timeline = TimelineBuilder().connect(0).frames_until(60_000, persons=person(IN_ZONE)).build()
    run = run_zones(timeline, zones=[zone(schedule=NIGHT)], utc_start=local(2026, 10, 3, 21, 59, 30))

    first_in_window = frame_at(30_000)
    [(entered_ms, entered)] = run.entered
    assert first_in_window + 1000 <= entered_ms < first_in_window + 1000 + FRAME_MS
    assert entered.first_source.frame_seq == first_in_window // FRAME_MS
    assert local_time(entered) == "22:00:01"
    assert run.ended == []


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [(NIGHT, 0), (None, 1)],
    ids=["night-schedule-at-noon", "always-active-at-noon"],
)
def test_schedule_gates_daytime_entries(schedule: dict | None, expected: int) -> None:
    timeline = TimelineBuilder().connect(0).frames_until(5000, persons=person(IN_ZONE)).build()
    run = run_zones(timeline, zones=[zone(schedule=schedule)], utc_start=local(2026, 10, 3, 12, 0, 0))
    assert len(run.entered) == expected


def test_calm_entry_with_scene_analysis_off_is_still_observed() -> None:
    timeline = TimelineBuilder().connect(0).frames_until(5000, persons=person(IN_ZONE)).build()
    run = run_zones(timeline, zones=[zone(schedule=NIGHT)], utc_start=local(2026, 10, 3, 23, 0, 0))

    state = run.at(5000)
    assert state.scene_analysis is Capability.DISABLED and state.scene_report is None
    [(_, entered)] = run.entered
    assert "identity not considered" in entered.reason
    # B0: v1 alerts on a person only with a medium/high VLM threat; with no scene
    # verdict (or a calm one) the same unrecognised person at 23:00 never alerts.
    assert v1_is_restricted_time(23)
    assert not v1_intruder_alert(any_stranger=True, scene_threat=None, local_hour=23)
    assert not v1_intruder_alert(any_stranger=True, scene_threat="none", local_hour=23)


def test_known_person_still_triggers_the_restricted_zone() -> None:
    face = {"box": [0.70, 0.22, 0.80, 0.32], "embedding": list(ALICE_FACE)}
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(5000, persons=person(IN_ZONE), faces=[face])
        .build()
    )
    run = run_zones(
        timeline,
        zones=[zone(schedule=NIGHT)],
        utc_start=local(2026, 10, 3, 23, 0, 0),
        enrollment=ENROLLMENT,
    )

    [(entered_ms, entered)] = run.entered
    [alice] = run.at(entered_ms).people
    assert (alice.identity, alice.identity_id) == (IdentityState.KNOWN, "alice")
    # The rule observation carries no identity: identity is context for the incident.
    assert not {"identity", "identity_id"} & set(ZoneObservation.model_fields)
    # B0: v1 never alerts on a recognised person, whatever the VLM says.
    assert not v1_intruder_alert(any_stranger=False, scene_threat="high", local_hour=23)


def test_stall_ends_the_observation_and_a_reconnect_starts_a_new_episode() -> None:
    last = frame_at(5000) - FRAME_MS  # last frame before the camera stalls
    resume = last + 3500
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(5000, persons=person(IN_ZONE))
        .tick(last + 999)
        .tick(last + 1000)
        .tick(last + 2100)
        .disconnect(last + 3000)
        .connect(resume)
        .frames_until(resume + 4000, persons=person(IN_ZONE))
        .build()
    )
    run = run_zones(timeline, zones=[zone()], utc_start=local(2026, 10, 3, 23, 0, 0))

    (_, first), (second_ms, second) = run.entered
    [(ended_ms, ended)] = run.ended
    # The person is no longer seen 1 s after the last frame (track expiry), so the
    # presence ends there; it is not carried through the stall.
    assert ended_ms == last + 1000 and ended.episode_id == first.episode_id
    assert ended.reason.endswith("track expired (not detected for the track expiry period)")
    # After the reconnect the same tracker ID is a new track in a new epoch: a new episode.
    assert second.track.stream_epoch > first.track.stream_epoch and second.track.track_id == 1
    assert second.episode_id != first.episode_id
    assert frame_at(resume) + 1000 <= second_ms < frame_at(resume) + 1000 + FRAME_MS
    assert [ms for ms, _ in run.observations if last < ms < resume] == [last + 1000]


def test_disconnect_while_in_the_zone_ends_the_observation() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(3000, persons=person(IN_ZONE))
        .disconnect(3010)
        .connect(3020)
        .frames_until(6000, persons=person(IN_ZONE))
        .build()
    )
    run = run_zones(timeline, zones=[zone()], utc_start=local(2026, 10, 3, 23, 0, 0))

    [(ended_ms, ended)] = run.ended
    assert ended_ms == 3010 and ended.reason.endswith("no fresh video (stall, outage or disconnect)")
    assert len(run.entered) == 2 and run.entered[1][0] >= frame_at(3020) + 1000


def test_entry_persistence_and_gap_tolerance() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(800, persons=person(IN_ZONE))  # 0.73 s in the zone: too short
        .frames_until(3000, persons=person(OUTSIDE))
        .frames_until(6000, persons=person(IN_ZONE))
        .frames_until(6600, persons=person(OUTSIDE))  # steps out for 0.6 s: same presence
        .frames_until(9000, persons=person(IN_ZONE))
        .frames_until(12_000, persons=person(OUTSIDE))  # out for 3 s: presence ends
        .frames_until(14_000, persons=person(IN_ZONE))
        .build()
    )
    run = run_zones(timeline, zones=[zone()], utc_start=local(2026, 10, 3, 23, 0, 0))

    (first_ms, first), (second_ms, second) = run.entered
    assert frame_at(3000) + 1000 <= first_ms < frame_at(3000) + 1000 + FRAME_MS
    [(ended_ms, ended)] = run.ended
    last_in_zone = frame_at(9000) - FRAME_MS
    assert ended.episode_id == first.episode_id and ended.last_mono_ns - first.first_mono_ns == (
        last_in_zone - frame_at(3000)
    ) * 1_000_000
    assert last_in_zone + 1000 < ended_ms <= last_in_zone + 1000 + FRAME_MS
    assert second.episode_id != first.episode_id and second_ms >= frame_at(12_000) + 1000


def test_zone_membership_uses_the_bottom_centre_anchor() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(4000, persons=person(BESIDE_ZONE))  # body overlaps the zone, feet outside
        .frames_until(6000, persons=person(IN_ZONE))
        .build()
    )
    run = run_zones(timeline, zones=[zone()], utc_start=local(2026, 10, 3, 23, 0, 0))

    [(entered_ms, entered)] = run.entered
    assert entered_ms >= frame_at(4000) + 1000 and entered.first_source.frame_seq == frame_at(4000) // FRAME_MS


def test_dwell_rule_fires_only_after_its_threshold() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(15_000, persons=person(IN_ZONE))  # 15 s: below the 20 s dwell threshold
        .frames_until(20_000, persons=person(OUTSIDE))
        .frames_until(45_000, persons=person(IN_ZONE))
        .build()
    )
    dwell = zone(zone_id="sofa", rule="dwell", min_duration_s=20.0, severity="info")
    run = run_zones(timeline, zones=[dwell], utc_start=local(2026, 10, 3, 15, 0, 0))

    [(entered_ms, entered)] = run.entered
    assert (entered.kind, entered.severity, entered.zone_id) == ("zone.dwell", Severity.INFO, "sofa")
    assert frame_at(20_000) + 20_000 <= entered_ms < frame_at(20_000) + 20_000 + FRAME_MS


def test_tentative_tracks_do_not_count() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(5000, persons=person(IN_ZONE, status="tentative"))
        .build()
    )
    run = run_zones(timeline, zones=[zone()], utc_start=local(2026, 10, 3, 23, 0, 0))
    assert run.observations == []


def test_each_zone_observes_independently_and_disabled_zones_not_at_all() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(3000, persons=person(IN_ZONE, track=1) + person(OUTSIDE, track=2))
        .build()
    )
    room = zone(zone_id="room", polygon=[[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]])
    off = zone(zone_id="off", enabled=False)
    run = run_zones(timeline, zones=[zone(), room, off], utc_start=local(2026, 10, 3, 23, 0, 0))

    seen = sorted((o.zone_id, o.track.track_id) for _, o in run.entered)
    assert seen == [("door", 1), ("room", 1), ("room", 2)]
    assert len({o.observation_id for _, o in run.entered}) == 3


def test_observation_ids_are_unique_per_phase_and_stable_per_episode() -> None:
    timeline = (
        TimelineBuilder()
        .connect(0)
        .frames_until(3000, persons=person(IN_ZONE))
        .frames_until(6000, persons=person(OUTSIDE))
        .build()
    )
    run = run_zones(timeline, zones=[zone()], utc_start=local(2026, 10, 3, 23, 0, 0))
    [(_, entered)] = run.entered
    [(_, ended)] = run.ended
    assert entered.observation_id == f"{entered.episode_id}.entered"
    assert ended.observation_id == f"{entered.episode_id}.ended"
    assert entered.zone_revision == ended.zone_revision
