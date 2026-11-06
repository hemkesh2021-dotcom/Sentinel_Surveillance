"""Session 116, part 14: the standalone head-window geometry (identity/head_windows.py); no camera, no model, no wiring.

- **Windows:** inside the frame, the head region and its margin covered, a constant input size in the usual case; a near person
  scaled down, never up; people sharing a window when they fit; people beyond the bound reported, never dropped quietly.
- **Each face once:** partial faces at a window's inner edge dropped, duplicates across windows suppressed, then the real
  ``associate()`` decides the owner.
- **The ownership bound:** while a moved face stays its own person's candidate, the real ``associate()`` never gives it to
  anyone else, checked across many random layouts; past that it may be lost or ambiguous.
- **Standalone:** nothing in the runtime imports it.
"""

from __future__ import annotations

import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sentinel.contracts import FrameKey, NormalizedBox, TrackKey, TrackObservation, TrackStatus
from sentinel.identity.association import AssociationGeometry, FaceObservation, Ownership, associate
from sentinel.identity.head_windows import (
    DEFAULT_MARGIN,
    DEFAULT_SIDE,
    FaceBox,
    Window,
    detection_to_frame,
    head_region,
    own_candidate_margin,
    plan_windows,
    safe_skew_s,
    select_faces,
    to_normalized,
    touches_window_edge,
)

W, H = 2304, 1296  # the main stream measured in dr-20261105T085421Z
FRAME = FrameKey(camera_id="cam-1", boot_id="b", run_id="r", stream_epoch=1, frame_seq=7)


def box(x1: float, y1: float, x2: float, y2: float) -> NormalizedBox:
    return NormalizedBox(x1=x1, y1=y1, x2=x2, y2=y2)


FAR = box(0.48, 0.30, 0.54, 0.75)  # a person standing far off: 138 px wide, 583 px high
NEAR = box(0.05, 0.02, 0.75, 0.98)  # filling most of the frame


def person(track: int, b: NormalizedBox) -> TrackObservation:
    return TrackObservation(frame=FRAME, observed_mono_ns=1, observed_utc=datetime(2026, 11, 5, tzinfo=timezone.utc), track_id=track,
                            box=b, status=TrackStatus.CONFIRMED, predicted=False, detector_confidence=0.9,
                            last_measured_mono_ns=1)


def face(b: NormalizedBox) -> FaceObservation:
    return FaceObservation(frame=FRAME, box=b, quality=0.95, embedding=None)


def covers(window: Window, person_box: NormalizedBox, margin: float = DEFAULT_MARGIN) -> bool:
    region = head_region(person_box)
    pad = margin * (person_box.x2 - person_box.x1) * W
    return (window.x <= max(0.0, region.x1 * W - pad) and window.y <= max(0.0, region.y1 * H - pad)
            and window.x + window.w >= min(float(W), region.x2 * W + pad)
            and window.y + window.h >= min(float(H), region.y2 * H + pad))


# ---------------------------------------------------------------- the windows


def test_the_head_region_is_the_rule_of_association() -> None:
    region = head_region(FAR)
    assert (region.x1, region.y1, region.x2) == (FAR.x1, FAR.y1, FAR.x2)
    assert region.y2 == pytest.approx(FAR.y1 + AssociationGeometry().head_fraction * (FAR.y2 - FAR.y1))


def test_a_far_person_gets_a_constant_window_of_native_pixels() -> None:
    plan = plan_windows([FAR], W, H)
    window = plan.windows[0]
    assert (window.w, window.h, window.scale) == (DEFAULT_SIDE, DEFAULT_SIDE, 1.0)
    assert window.input_size == (DEFAULT_SIDE, DEFAULT_SIDE) and window.owners == (0,) and plan.uncovered == ()
    assert covers(window, FAR)
    centre = (window.x + window.w / 2) / W
    assert centre == pytest.approx((FAR.x1 + FAR.x2) / 2, abs=1 / W)


def test_a_near_person_is_scaled_down_and_never_up() -> None:
    window = plan_windows([NEAR], W, H).windows[0]
    assert window.scale < 1.0 and max(window.input_size) == DEFAULT_SIDE and covers(window, NEAR)


def test_a_window_moves_to_stay_inside_the_frame_and_keeps_its_size() -> None:
    for corner in (box(0.0, 0.0, 0.05, 0.3), box(0.95, 0.0, 1.0, 0.3), box(0.0, 0.7, 0.05, 1.0), box(0.95, 0.7, 1.0, 1.0)):
        window = plan_windows([corner], W, H).windows[0]
        assert (window.w, window.h) == (DEFAULT_SIDE, DEFAULT_SIDE)
        assert 0 <= window.x and window.x + window.w <= W and 0 <= window.y and window.y + window.h <= H
        assert covers(window, corner)


def test_a_frame_smaller_than_the_side_is_used_whole() -> None:
    window = plan_windows([box(0.4, 0.2, 0.6, 0.9)], 640, 480).windows[0]
    assert (window.x, window.y, window.w, window.h, window.scale) == (0, 0, 640, 480, 1.0)


def test_people_who_fit_share_a_window_and_others_get_their_own() -> None:
    together = plan_windows([box(0.40, 0.3, 0.45, 0.7), box(0.47, 0.3, 0.52, 0.7)], W, H)
    assert len(together.windows) == 1 and together.windows[0].owners == (0, 1)
    apart = plan_windows([box(0.05, 0.3, 0.10, 0.7), box(0.85, 0.3, 0.90, 0.7)], W, H)
    assert sorted(w.owners for w in apart.windows) == [(0,), (1,)]


def test_people_beyond_the_bound_are_reported_not_dropped() -> None:
    people = [box(0.02 + 0.2 * i, 0.3, 0.06 + 0.2 * i, 0.7) for i in range(5)]
    plan = plan_windows(people, W, H, max_windows=2)
    assert len(plan.windows) == 2 and len(plan.uncovered) == 3
    assert sorted(i for w in plan.windows for i in w.owners) + list(plan.uncovered) != []
    assert sorted([*plan.uncovered, *(i for w in plan.windows for i in w.owners)]) == [0, 1, 2, 3, 4]


def test_the_largest_people_are_covered_first_and_the_order_given_does_not_matter() -> None:
    small, large = box(0.05, 0.4, 0.07, 0.5), box(0.8, 0.1, 0.95, 0.9)
    plan = plan_windows([small, large], W, H, max_windows=1)
    assert plan.windows[0].owners == (1,) and plan.uncovered == (0,)
    swapped = plan_windows([large, small], W, H, max_windows=1)
    assert swapped.windows[0].owners == (0,) and swapped.uncovered == (1,)


def test_no_people_means_no_windows() -> None:
    assert plan_windows([], W, H).windows == () and plan_windows([], W, H).uncovered == ()


@pytest.mark.parametrize("args, kwargs", [((0, H), {}), ((W, 0), {}), ((W, H), {"side": 0}), ((W, H), {"max_windows": 0}),
                                          ((W, H), {"head_fraction": 0.0}), ((W, H), {"margin": -0.1})])
def test_impossible_settings_are_refused(args, kwargs) -> None:
    with pytest.raises(ValueError):
        plan_windows([FAR], *args, **kwargs)


def test_every_window_is_valid_across_random_layouts() -> None:
    rng = random.Random(116)
    for _ in range(2000):
        count = rng.randint(1, 6)
        people = []
        for _ in range(count):
            x1, y1 = rng.uniform(0, 0.95), rng.uniform(0, 0.9)
            people.append(box(x1, y1, min(1.0, x1 + rng.uniform(0.01, 0.8)), min(1.0, y1 + rng.uniform(0.05, 0.9))))
        plan = plan_windows(people, W, H)
        seen = [i for w in plan.windows for i in w.owners]
        assert sorted([*seen, *plan.uncovered]) == list(range(count)) and len(seen) == len(set(seen))
        for window in plan.windows:
            assert 0 <= window.x and window.x + window.w <= W and 0 <= window.y and window.y + window.h <= H
            assert max(window.input_size) <= DEFAULT_SIDE and 0 < window.scale <= 1.0
            assert all(covers(window, people[i]) for i in window.owners)


# ---------------------------------------------------------------- each face once


def test_a_detection_returns_to_frame_pixels() -> None:
    native = Window(1000, 200, 640, 640, 1.0, (0,))
    got = detection_to_frame(native, 0, 300.0, 100.0, 40.0, 50.0, 0.93)
    assert (got.x, got.y, got.w, got.h, got.score, got.window) == (1300.0, 300.0, 40.0, 50.0, 0.93, 0)
    scaled = Window(100, 0, 1280, 1280, 0.5, (0,))
    got = detection_to_frame(scaled, 0, 300.0, 100.0, 40.0, 50.0, 0.93)
    assert (got.x, got.y, got.w, got.h) == (700.0, 200.0, 80.0, 100.0)


def test_a_partial_face_at_an_inner_edge_is_dropped_but_not_at_the_frames_edge() -> None:
    inner = Window(1000, 200, 640, 640, 1.0, (0,))
    assert touches_window_edge(FaceBox(1000.0, 400.0, 40, 50, 0.9, 0), inner, W, H)
    assert touches_window_edge(FaceBox(1600.0, 400.0, 40, 50, 0.9, 0), inner, W, H)
    assert not touches_window_edge(FaceBox(1300.0, 400.0, 40, 50, 0.9, 0), inner, W, H)
    corner = Window(0, 0, 640, 640, 1.0, (0,))
    assert not touches_window_edge(FaceBox(0.0, 0.0, 40, 50, 0.9, 0), corner, W, H)  # the frame's own edge: a whole face may sit there
    selected = select_faces([FaceBox(1000.0, 400.0, 40, 50, 0.95, 0), FaceBox(1300.0, 400.0, 40, 50, 0.91, 0)], [inner], W, H)
    assert [f.score for f in selected.faces] == [0.91] and selected.dropped_at_window_edge == 1


def test_a_face_seen_in_two_windows_is_kept_once_the_higher_score() -> None:
    windows = [Window(600, 200, 640, 640, 1.0, (0,)), Window(900, 200, 640, 640, 1.0, (1,))]
    same = [FaceBox(1000.0, 400.0, 60, 70, 0.92, 0), FaceBox(1002.0, 401.0, 60, 70, 0.95, 1)]
    selected = select_faces(same, windows, W, H)
    assert [(f.score, f.window) for f in selected.faces] == [(0.95, 1)] and selected.dropped_duplicate == 1


def test_two_real_faces_are_both_kept_in_a_deterministic_order() -> None:
    window = Window(600, 200, 640, 640, 1.0, (0, 1))
    faces = [FaceBox(800.0, 400.0, 60, 70, 0.92, 0), FaceBox(1000.0, 400.0, 60, 70, 0.95, 0)]
    assert [f.score for f in select_faces(faces, [window], W, H).faces] == [0.95, 0.92]
    assert select_faces(faces, [window], W, H) == select_faces(list(reversed(faces)), [window], W, H)


def test_to_normalized_clamps_and_refuses_nothing_inside() -> None:
    got = to_normalized(FaceBox(-10.0, 100.0, 110.0, 100.0, 0.9, 0), W, H)
    assert got == box(0.0, 100 / H, 100 / W, 200 / H)
    assert to_normalized(FaceBox(W + 5.0, 10.0, 20.0, 20.0, 0.9, 0), W, H) is None


def test_a_mapped_face_reaches_the_real_association_and_its_person() -> None:
    """The whole geometry to the owner: the person's window, a detection in it, frame pixels, normalized, associate()."""
    people = [person(1, FAR)]
    window = plan_windows([FAR], W, H).windows[0]
    centre_x, centre_y = (FAR.x1 + FAR.x2) / 2 * W, (FAR.y1 + 0.1 * (FAR.y2 - FAR.y1)) * H
    detection = detection_to_frame(window, 0, centre_x - window.x - 30, centre_y - window.y - 35, 60, 70, 0.94)
    kept = select_faces([detection], [window], W, H).faces
    result = associate(people, [face(to_normalized(kept[0], W, H))])
    assert result.face_of == {people[0].key: 0} and result.person_ownership[people[0].key] is Ownership.ASSIGNED


def test_two_people_over_one_face_stay_ambiguous_after_the_mapping() -> None:
    a, b = box(0.40, 0.3, 0.50, 0.8), box(0.46, 0.3, 0.56, 0.8)
    people = [person(1, a), person(2, b)]
    plan = plan_windows([a, b], W, H)
    window = plan.windows[0]
    assert window.owners == (0, 1)
    detection = detection_to_frame(window, 0, 0.48 * W - window.x - 30, 0.4 * H - window.y - 35, 60, 70, 0.94)
    kept = select_faces([detection], plan.windows, W, H).faces
    result = associate(people, [face(to_normalized(kept[0], W, H))])
    assert result.face_of == {} and result.face_ownership == (Ownership.AMBIGUOUS,)


# ---------------------------------------------------------------- the ownership bound


AT_DOOR = box(0.47, 0.30, 0.53, 0.80)
FACE_AT_DOOR = box(0.49, 0.32, 0.51, 0.37)


def test_the_margin_is_the_distance_to_the_rule_s_boundary() -> None:
    mx, my = own_candidate_margin(FACE_AT_DOOR, AT_DOOR)
    assert 0.0 < mx <= 0.03 + 1e-6 and 0.0 < my
    moved_in = box(FACE_AT_DOOR.x1 + mx * 0.99, FACE_AT_DOOR.y1, FACE_AT_DOOR.x2 + mx * 0.99, FACE_AT_DOOR.y2)
    moved_out = box(FACE_AT_DOOR.x1 + mx * 1.01, FACE_AT_DOOR.y1, FACE_AT_DOOR.x2 + mx * 1.01, FACE_AT_DOOR.y2)
    assert associate([person(1, AT_DOOR)], [face(moved_in)]).face_of != {}
    assert associate([person(1, AT_DOOR)], [face(moved_out)]).face_of == {}


def test_a_face_that_is_not_a_candidate_has_no_margin() -> None:
    assert own_candidate_margin(box(0.1, 0.1, 0.12, 0.15), AT_DOOR) == (0.0, 0.0)
    assert safe_skew_s(box(0.1, 0.1, 0.12, 0.15), AT_DOOR, (0.1, 0.0)) == 0.0


def test_the_safe_skew_scales_with_speed_and_a_still_person_has_no_limit() -> None:
    slow = safe_skew_s(FACE_AT_DOOR, AT_DOOR, (0.05, 0.0))
    fast = safe_skew_s(FACE_AT_DOOR, AT_DOOR, (0.20, 0.0))
    assert slow == pytest.approx(4 * fast, rel=1e-3)
    assert safe_skew_s(FACE_AT_DOOR, AT_DOOR, (0.0, 0.0)) == math.inf


def test_the_safe_skew_takes_the_worse_direction_of_an_off_centre_face() -> None:
    off = box(0.50, 0.32, 0.52, 0.37)
    assert safe_skew_s(off, AT_DOOR, (0.1, 0.0)) == safe_skew_s(off, AT_DOOR, (-0.1, 0.0))  # the sign of the skew is unknown


def test_non_finite_velocity_is_refused() -> None:
    with pytest.raises(ValueError):
        safe_skew_s(FACE_AT_DOOR, AT_DOOR, (math.nan, 0.0))


def test_within_the_safe_skew_a_face_never_goes_to_another_person() -> None:
    """The bound's claim, against the real associate(): random pairs of people, the first one's face moved by v * t with t up to
    the bound and either sign. It may be ambiguous; it is never assigned to the other person, and never lost by the first."""
    rng = random.Random(1160)
    exercised = ambiguous = 0
    for _ in range(4000):
        width = rng.uniform(0.04, 0.2)
        x = rng.uniform(0.1, 0.7)
        y = rng.uniform(0.1, 0.4)
        own = box(x, y, x + width, min(0.99, y + rng.uniform(0.3, 0.55)))
        gap = rng.uniform(-0.5, 1.5) * width
        other_x = min(0.8, x + width + gap)
        other = box(other_x, y + rng.uniform(-0.05, 0.05), other_x + width, min(0.99, y + rng.uniform(0.3, 0.55)))
        size = rng.uniform(0.25, 0.6) * width
        cx, cy = rng.uniform(own.x1 + size / 2, own.x2 - size / 2), own.y1 + rng.uniform(0.05, 0.3) * (own.y2 - own.y1)
        f = box(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)
        velocity = (rng.uniform(-0.3, 0.3), rng.uniform(-0.05, 0.05))
        bound = safe_skew_s(f, own, velocity)
        if not math.isfinite(bound) or bound <= 0:
            continue
        t = rng.choice((-1, 1)) * rng.uniform(0, bound)
        moved = box(f.x1 + velocity[0] * t, f.y1 + velocity[1] * t, f.x2 + velocity[0] * t, f.y2 + velocity[1] * t)
        people = [person(1, own), person(2, other)]
        result = associate(people, [face(moved)])
        exercised += 1
        assert people[1].key not in result.face_of
        assert result.face_ownership[0] in (Ownership.ASSIGNED, Ownership.AMBIGUOUS)
        ambiguous += result.face_ownership[0] is Ownership.AMBIGUOUS
    assert exercised > 3000 and ambiguous > 50  # both outcomes really occurred


def test_past_the_bound_a_wrong_owner_is_possible_which_is_why_it_is_a_bound() -> None:
    """A face carried past its own box's edge into a neighbour's: the mechanism the bound guards against, shown once."""
    own, neighbour = box(0.40, 0.30, 0.46, 0.80), box(0.47, 0.30, 0.53, 0.80)
    f = box(0.42, 0.32, 0.44, 0.37)
    velocity = (0.30, 0.0)
    bound = safe_skew_s(f, own, velocity)
    moved = box(f.x1 + velocity[0] * 0.2, f.y1, f.x2 + velocity[0] * 0.2, f.y2)
    assert 0.2 > bound
    result = associate([person(1, own), person(2, neighbour)], [face(moved)])
    assert list(result.face_of) == [TrackKey(camera_id="cam-1", boot_id="b", run_id="r", stream_epoch=1, track_id=2)]


# ---------------------------------------------------------------- standalone


def test_only_the_window_stage_imports_the_helper_and_the_skew_bound_stays_unwired() -> None:
    """Route B (session 116, part 15) uses the windows through identity.window_stage only; the ownership bound for a time
    difference between two pictures stays a validation measure, because route B takes a face and its people from one frame."""
    root = Path(__file__).resolve().parents[2] / "src" / "sentinel"
    importers = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py") if p.name != "head_windows.py"
                       and re.search(r"from \.head_windows import|head_windows import", p.read_text(encoding="utf-8")))
    assert importers == ["identity/window_stage.py"]
    users = [p for p in root.rglob("*.py") if p.name != "head_windows.py"
             and re.search(r"safe_skew_s|own_candidate_margin", p.read_text(encoding="utf-8"))]
    assert users == []
