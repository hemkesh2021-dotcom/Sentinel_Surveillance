"""Zone geometry, IANA schedules, zone configuration and the zone rule's edge cases (V2-13)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sentinel.config import ConfigError, load_config, parse_config
from sentinel.contracts import FrameRef, NormalizedBox, TrackObservation, TrackStatus
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.rules.geometry import Anchor, anchor_point, contains, polygon_problem
from sentinel.rules.schedule import Schedule, load_timezone, parse_hhmm
from sentinel.rules.zones import ZonePhase, ZoneRule

SQUARE = [(0.6, 0.6), (1.0, 0.6), (1.0, 1.0), (0.6, 1.0)]
IN_ZONE = NormalizedBox(x1=0.65, y1=0.2, x2=0.85, y2=0.9)
OUTSIDE = NormalizedBox(x1=0.05, y1=0.2, x2=0.25, y2=0.9)


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# ---------------------------------------------------------------- geometry


def test_polygon_validation_explains_each_problem() -> None:
    assert polygon_problem(SQUARE) is None
    assert "at least 3" in polygon_problem(SQUARE[:2])
    assert "outside the image" in polygon_problem([(0.0, 0.0), (1.2, 0.0), (0.5, 0.5)])
    assert "the same" in polygon_problem([(0.1, 0.1), (0.1, 0.1), (0.5, 0.5), (0.1, 0.5)])
    assert "no area" in polygon_problem([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)])
    # An asymmetric bow tie: two edges cross.
    assert "cross" in polygon_problem([(0.1, 0.1), (0.9, 0.9), (0.9, 0.3), (0.1, 0.8)])
    assert "at most 32" in polygon_problem([(0.5 + 0.4 * (i % 2), i / 40) for i in range(33)])


def test_containment_and_anchor_points() -> None:
    concave = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.4), (0.0, 1.0)]  # notch from below
    assert contains(concave, (0.2, 0.2)) and contains(concave, (0.97, 0.9))  # right prong
    assert not contains(concave, (0.9, 0.9))  # just inside the notch edge (x = 0.917 at y = 0.9)
    assert not contains(concave, (0.5, 0.8))  # inside the notch
    box = NormalizedBox(x1=0.2, y1=0.1, x2=0.4, y2=0.9)
    assert anchor_point(box, Anchor.BOTTOM_CENTER) == pytest.approx((0.3, 0.9))
    assert anchor_point(box, Anchor.CENTER) == pytest.approx((0.3, 0.5))


# ---------------------------------------------------------------- schedules


def test_windows_across_midnight_and_window_edges() -> None:
    night = Schedule.parse("Asia/Kolkata", [("22:00", "06:00")])
    kolkata = timedelta(hours=5, minutes=30)  # no daylight saving in Asia/Kolkata
    at = lambda h, m, s=0: utc(2026, 10, 3, 0, 0) + timedelta(hours=h, minutes=m, seconds=s) - kolkata  # noqa: E731
    assert not night.active_at(at(21, 59, 59))
    assert night.active_at(at(22, 0))
    assert night.active_at(at(23, 59, 59)) and night.active_at(at(24, 0)) and night.active_at(at(29, 59, 59))
    assert not night.active_at(at(30, 0))  # 06:00 the next day: windows are half-open
    split = Schedule.parse("UTC", [("08:00", "09:00"), ("17:30", "18:00")])
    assert split.active_at(utc(2026, 1, 1, 8, 59)) and not split.active_at(utc(2026, 1, 1, 9, 0))
    assert split.active_at(utc(2026, 1, 1, 17, 45)) and not split.active_at(utc(2026, 1, 1, 12, 0))


def test_daylight_saving_follows_local_wall_time() -> None:
    # New York springs forward at 02:00 EST on 2026-03-08 (07:00 UTC) to 03:00 EDT.
    early = Schedule.parse("America/New_York", [("01:00", "03:30")])
    assert early.active_at(utc(2026, 3, 8, 6, 59))  # 01:59 EST
    assert early.active_at(utc(2026, 3, 8, 7, 0))  # 03:00 EDT
    assert not early.active_at(utc(2026, 3, 8, 7, 30))  # 03:30 EDT: only 1.5 h of window that night
    # London falls back at 02:00 BST on 2026-10-25 (01:00 UTC): 01:00-02:00 local happens twice.
    repeated = Schedule.parse("Europe/London", [("01:00", "02:00")])
    assert repeated.active_at(utc(2026, 10, 25, 0, 30))  # 01:30 BST
    assert repeated.active_at(utc(2026, 10, 25, 1, 30))  # 01:30 GMT
    assert not repeated.active_at(utc(2026, 10, 25, 2, 0))  # 02:00 GMT


def test_schedule_inputs_are_validated() -> None:
    assert parse_hhmm("00:00") == 0 and parse_hhmm("23:59") == 1439
    for bad in ("24:00", "7:00", "07:60", "07:00:00", ""):
        with pytest.raises(ValueError, match="HH:MM"):
            parse_hhmm(bad)
    for bad in ("IST", "+05:30", "Asia/Nowhere", "../etc/passwd"):
        with pytest.raises(ValueError, match="timezone"):
            load_timezone(bad)
    with pytest.raises(ValueError, match="same time"):
        Schedule.parse("UTC", [("10:00", "10:00")])
    with pytest.raises(ValueError, match="UTC"):
        Schedule.parse("UTC", [("10:00", "11:00")]).active_at(datetime(2026, 1, 1, 10, 30))  # naive


# ---------------------------------------------------------------- configuration


def test_zone_config_errors_are_located(tmp_path: Path) -> None:
    path = tmp_path / "zones.yaml"
    path.write_text(
        """\
config_version: 1
camera: {id: cam-1}
zones:
  - zone_id: door
    polygon: [[0.1, 0.1], [0.9, 0.9], [0.9, 0.3], [0.1, 0.8]]
    severty: critical
    schedule: {timezone: IST, windows: [{start: "22:00", end: "6:00"}]}
  - zone_id: door
    polygon: [[0.6, 0.6], [1.0, 0.6], [1.0, 1.0]]
    rule: loiter
    min_duration_s: 0
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    problems = caught.value.problems
    assert "zones.0.polygon: edges 0 and 2 cross; the polygon must not intersect itself" in problems
    assert "zones.0.severty: unknown setting (did you mean 'severity'?)" in problems
    assert any(p.startswith("zones.0.schedule.timezone: expected an IANA timezone") for p in problems)
    assert any(p.startswith("zones.0.schedule.windows.0.end: expected a 24-hour time") for p in problems)
    assert any(p.startswith("zones.1.rule:") for p in problems)
    assert any(p.startswith("zones.1.min_duration_s:") for p in problems)


def test_duplicate_zone_ids_are_rejected_and_revisions_track_settings() -> None:
    base = {"config_version": 1, "camera": {"id": "cam-1"}}
    door = {"zone_id": "door", "polygon": [list(p) for p in SQUARE]}
    with pytest.raises(ConfigError, match="duplicate zone_id 'door'"):
        parse_config({**base, "zones": [door, door]})
    one = parse_config({**base, "zones": [door]}).zones[0]
    again = parse_config({**base, "zones": [dict(door)]}).zones[0]
    longer = parse_config({**base, "zones": [{**door, "min_duration_s": 2.0}]}).zones[0]
    assert one.revision == again.revision != longer.revision


# ---------------------------------------------------------------- the rule itself


def _rule(**settings: object) -> ZoneRule:
    door = {"zone_id": "door", "polygon": [list(p) for p in SQUARE], **settings}
    return ZoneRule(parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "zones": [door]}).zones[0])


def _detect(frame: FrameRef, box: NormalizedBox = IN_ZONE) -> TrackObservation:
    return TrackObservation.detected(frame, track_id=7, box=box, confidence=0.9, status=TrackStatus.CONFIRMED)


def test_predictions_neither_build_nor_extend_a_presence(
    clock: FakeClock, stamper: FrameStamper, next_frame: Callable[..., FrameRef]
) -> None:
    rule = _rule()
    seen = _detect(next_frame())
    assert rule.evaluate([seen], stamper.current_stream, clock.mono()) == []
    # The tracker keeps predicting the person in the zone for 2 s: no entry, and
    # the presence lapses one gap tolerance after the last real detection.
    out = []
    for _ in range(30):
        frame = next_frame()
        out += rule.evaluate([seen.predicted_on(frame, box=IN_ZONE)], stamper.current_stream, clock.mono())
    assert out == [] and rule.active_episodes == 0
    # A prediction for a track the rule never saw detected does not place it in the zone.
    fresh = _rule()
    unseen = seen.predicted_on(next_frame(), box=IN_ZONE)
    assert fresh.evaluate([unseen], stamper.current_stream, clock.mono()) == []
    assert fresh.active_episodes == 0


def test_sparse_evaluation_still_splits_presences_at_a_gap(
    clock: FakeClock, stamper: FrameStamper, next_frame: Callable[..., FrameRef]
) -> None:
    rule = _rule(min_duration_s=0.5, gap_tolerance_s=1.0)
    out = []
    for _ in range(10):
        out += rule.evaluate([_detect(next_frame())], stamper.current_stream, clock.mono())
    [entered] = out
    # The next in-zone detection arrives 3 s later, with no evaluation in between.
    out = rule.evaluate([_detect(next_frame(dt=3.0))], stamper.current_stream, clock.mono())
    [ended] = out
    assert ended.phase is ZonePhase.ENDED and ended.episode_id == entered.episode_id
    assert rule.active_episodes == 1  # the late detection starts a new presence, not yet entered


def test_reason_when_a_new_epoch_replaces_the_tracks(
    clock: FakeClock, stamper: FrameStamper, next_frame: Callable[..., FrameRef]
) -> None:
    rule = _rule()
    for _ in range(20):
        out = rule.evaluate([_detect(next_frame())], stamper.current_stream, clock.mono())
    assert rule.active_episodes == 1
    stamper.connect()  # reconnect: the first frame of the new epoch arrives directly
    out = rule.evaluate([_detect(next_frame())], stamper.current_stream, clock.mono())
    [ended] = [o for o in out if o.phase is ZonePhase.ENDED]
    assert ended.reason.endswith("stream reconnected (new epoch)")
    assert rule.active_episodes == 1  # the new epoch's track started its own episode


def test_wall_clock_steps_do_not_change_persistence(
    clock: FakeClock, stamper: FrameStamper, next_frame: Callable[..., FrameRef]
) -> None:
    rule = _rule(min_duration_s=1.0)
    out = rule.evaluate([_detect(next_frame())], stamper.current_stream, clock.mono())
    clock.step_utc(timedelta(hours=1))  # NTP step forwards
    for _ in range(13):  # 13 more frames: 0.87 s of monotonic time
        out += rule.evaluate([_detect(next_frame())], stamper.current_stream, clock.mono())
    assert out == []
    clock.step_utc(timedelta(hours=-3))  # and backwards
    for _ in range(3):
        out += rule.evaluate([_detect(next_frame())], stamper.current_stream, clock.mono())
    [entered] = out
    assert entered.phase is ZonePhase.ENTERED and entered.duration_ms == 1000


def test_late_evaluation_ends_a_presence_by_source_time(
    clock: FakeClock, stamper: FrameStamper, next_frame: Callable[..., FrameRef]
) -> None:
    rule = _rule(min_duration_s=0.1, gap_tolerance_s=1.0)
    for _ in range(5):
        rule.evaluate([_detect(next_frame())], stamper.current_stream, clock.mono())
    last = _detect(next_frame(dt=0.0))
    clock.advance(1.0)  # exactly one gap tolerance: still the same presence
    assert rule.evaluate([last], stamper.current_stream, clock.mono()) == []
    clock.advance(ns=1)
    [ended] = rule.evaluate([last], stamper.current_stream, clock.mono())
    assert ended.phase is ZonePhase.ENDED and ended.reason.endswith("left the zone")


def test_config_validate_lists_zones(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from sentinel.cli import main

    path = tmp_path / "zones.yaml"
    path.write_text(
        """\
config_version: 1
camera: {id: cam-1}
zones:
  - zone_id: door
    polygon: [[0.6, 0.6], [1.0, 0.6], [1.0, 1.0], [0.6, 1.0]]
    schedule: {timezone: Asia/Kolkata, windows: [{start: "22:00", end: "06:00"}]}
  - zone_id: sofa
    rule: dwell
    polygon: [[0.0, 0.5], [0.4, 0.5], [0.4, 1.0]]
    severity: info
    enabled: false
""",
        encoding="utf-8",
    )
    assert main(["config", "validate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "  zone door: restricted, enabled, warning, 22:00-06:00 Asia/Kolkata" in out
    assert "  zone sofa: dwell, disabled, info, always active" in out
