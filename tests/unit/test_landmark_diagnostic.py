"""The landmark diagnostic (session 90): its measurements, its association with the production vote's person, the
backend's opt-in landmark parse, the schema-7 records, the guard's readiness, the launch-memory block letter, and the
values the runner scripts keep equal to the runtime's."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel.config import parse_config
from sentinel.contracts import NormalizedBox, PixelFormat, TrackObservation, TrackStatus
from sentinel.identity import landmarks as lm
from sentinel.identity.association import AssociationGeometry, FaceObservation
from sentinel.identity.legacy_deepface import (
    FULL_SCALE_KEPT,
    YUNET_THRESHOLD,
    FaceBackendError,
    FullScale,
    full_scale_pass,
)
from sentinel.identity.matching import EnrolledIdentity, Enrollment
from sentinel.live_state import Capability
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.runtime import (
    FULL_SCALE_COUNTED,
    IDENTITY_FULL_SCALE_SCHEMA,
    IDENTITY_GEOMETRY_SCHEMA,
    IDENTITY_LANDMARK_SCHEMA,
    EdgeCore,
)

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
CONFIG = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
GEOMETRY = AssociationGeometry()
A_FACE, NOBODY = (0.97, 0.1, 0.0, 0.05), (0.0, 0.1, 0.99, 0.0)
ENROLLMENT = Enrollment(model_revision="synthetic", dimension=4, identities=(
    EnrolledIdentity(identity_id="idn-00000000000a", prototypes=((1.0, 0.0, 0.0, 0.0),)),))
LEFT, RIGHT = (0.05, 0.1, 0.35, 0.9), (0.6, 0.1, 0.9, 0.9)


@pytest.fixture
def runners(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return SimpleNamespace(guard=importlib.import_module("step5_guard"), check=importlib.import_module("face_check"),
                           evidence=importlib.import_module("face_evidence"),
                           memory=importlib.import_module("launch_memory"),
                           landmark=importlib.import_module("landmark_check"),
                           account=importlib.import_module("face_account"))


def points(cx: float = 128.0, cy: float = 110.0, d: float = 22.0, *, lateral: float = 0.0, nose: float = 0.55,
           mouth: float = 1.1, mouth_lateral: float = 0.0, angle: float = 0.0, box_w: float | None = None
           ) -> lm.Landmarks:
    """A synthetic upright face's landmarks (person's right eye on the image's left), then turned by ``angle`` degrees
    clockwise as displayed about the eye midpoint; the box spans 2.6 eye spacings unless ``box_w`` says otherwise."""
    theta = math.radians(angle)

    def turn(x: float, y: float) -> tuple[float, float]:
        dx, dy = x - cx, y - cy
        return (cx + dx * math.cos(theta) - dy * math.sin(theta), cy + dx * math.sin(theta) + dy * math.cos(theta))

    w = 2.6 * d if box_w is None else box_w
    return lm.Landmarks((cx - 1.3 * d, cy - d, w, 3.0 * d), turn(cx - d / 2, cy), turn(cx + d / 2, cy),
                        turn(cx + lateral * d, cy + nose * d),
                        turn(cx - 0.4 * d + mouth_lateral * d, cy + mouth * d),
                        turn(cx + 0.4 * d + mouth_lateral * d, cy + mouth * d))


def yunet_row(landmarks: lm.Landmarks, score: float) -> list[float]:
    return [*landmarks.box, *(v for point in landmarks.points for v in point), score]


# ---------------------------------------------------------------- the measurements


def test_an_upright_symmetric_face_measures_as_its_construction() -> None:
    values, reason = lm.measure(points(), 640, 480)
    assert reason is None
    assert values == {"nose_lateral": 0.0, "nose_vertical": 0.55, "mouth_vertical": 1.1, "nose_position": 0.5,
                      "mouth_lateral": 0.0, "eye_line_deg": 0.0, "eye_spacing_px": 22.0, "eye_spacing_ratio": 0.385}
    assert set(values) == set(lm.MEASURES)


def test_the_shape_measures_ignore_position_size_and_in_plane_rotation() -> None:
    base, _ = lm.measure(points(lateral=0.12, nose=0.6, mouth=1.2, mouth_lateral=0.05), 640, 480)
    moved, _ = lm.measure(points(300.0, 250.0, 44.0, lateral=0.12, nose=0.6, mouth=1.2, mouth_lateral=0.05,
                                 angle=25.0), 640, 480)
    for key in ("nose_lateral", "nose_vertical", "mouth_vertical", "nose_position", "mouth_lateral",
                "eye_spacing_ratio"):
        assert moved[key] == pytest.approx(base[key], abs=0.001), key
    assert moved["eye_line_deg"] == 25.0 and moved["eye_spacing_px"] == 44.0 and base["eye_line_deg"] == 0.0


def test_the_signs_follow_the_documented_frame() -> None:
    toward_left_eye, _ = lm.measure(points(lateral=0.2), 640, 480)  # the nose toward the image's right
    toward_right_eye, _ = lm.measure(points(lateral=-0.2), 640, 480)
    assert toward_left_eye["nose_lateral"] == 0.2 and toward_right_eye["nose_lateral"] == -0.2
    lower_left_eye, _ = lm.measure(points(angle=10.0), 640, 480)  # the left eye lower on screen
    assert lower_left_eye["eye_line_deg"] == 10.0
    assert lm.measure(points(nose=0.9, mouth=1.0), 640, 480)[0]["nose_position"] == 0.9


@pytest.mark.parametrize(("landmarks", "reason"), [
    (None, "malformed_row"),
    (points(cx=5.0), "landmark_outside_frame"),  # the right eye left of the frame
    (points(cy=470.0), "landmark_outside_frame"),  # the mouth below it
    (points(d=3.9), "eye_spacing_too_small"),
    (points(angle=180.0), "eyes_not_left_to_right"),  # upside down: the eye line beyond 90 degrees
    (points(angle=90.0), "eyes_not_left_to_right"),  # exactly vertical
    (points(mouth=-0.5, nose=-0.2), "mouth_not_below_eyes"),
    (points(box_w=0.0), "malformed_row"),
])
def test_invalid_observations_have_a_fixed_reason_and_no_numbers(landmarks, reason) -> None:
    assert lm.measure(landmarks, 640, 480) == (None, reason)
    assert reason in lm.INVALID_REASONS


def test_a_row_maps_to_the_documented_columns_and_a_malformed_row_to_none() -> None:
    row = [float(v) for v in range(1, 16)]
    parsed = lm.from_row(row)
    assert parsed.box == (1.0, 2.0, 3.0, 4.0) and parsed.right_eye == (5.0, 6.0) and parsed.left_eye == (7.0, 8.0)
    assert parsed.nose == (9.0, 10.0) and parsed.mouth_right == (11.0, 12.0) and parsed.mouth_left == (13.0, 14.0)
    assert lm.from_row(row[:14]) is None and lm.from_row([*row[:9], math.nan, *row[10:]]) is None
    assert lm.from_row(["x"] * 15) is None and lm.from_row(None) is None


# ---------------------------------------------------------------- the association with the vote's person


class Frames:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stream = self.stamper.connect()

    def frame(self, *people: tuple[int, tuple[float, ...]]):
        self.clock.advance(1.0)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        persons = tuple(TrackObservation.detected(frame, track_id=track, box=NormalizedBox(
            x1=b[0], y1=b[1], x2=b[2], y2=b[3]), confidence=0.9, status=TrackStatus.CONFIRMED) for track, b in people)
        return frame, persons


def box_of(landmarks: lm.Landmarks) -> NormalizedBox:
    x, y, w, h = landmarks.box
    return NormalizedBox(x1=x / 640, y1=y / 480, x2=(x + w) / 640, y2=(y + h) / 480)


def native(*faces: tuple[float, lm.Landmarks | None], box: NormalizedBox | None = None) -> FullScale:
    return FullScale(tuple((score, box or box_of(f or points())) for score, f in faces), 1.0,
                     landmarks=tuple(f for _, f in faces))


def records(frame, persons, full, production=()):
    faces = tuple(FaceObservation(frame=frame.key, box=b, quality=0.9) for b in production)
    face_of = {p.key: i for i, p in enumerate(persons) if i < len(faces)}
    return {k.track_id: v for k, v in lm.landmark_records(frame, persons, faces, face_of, full, GEOMETRY).items()}


def test_the_persons_own_candidate_is_measured_with_its_overlap_against_the_production_face() -> None:
    frame, persons = Frames().frame((1, LEFT))
    face = points()
    out = records(frame, persons, native((0.93, face)), production=(box_of(face),))
    assert out[1]["status"] == "measured" and out[1]["reason"] is None and out[1]["score"] == 0.93
    assert out[1]["face_iou"] == 1.0 and out[1]["nose_position"] == 0.5
    assert set(out[1]) == set(lm.RECORD_KEYS)
    alone = records(frame, persons, native((0.93, face)))  # production found no face for this person
    assert alone[1]["status"] == "measured" and alone[1]["face_iou"] is None


def test_weaker_candidates_never_attach_and_no_nearest_face_is_taken() -> None:
    frame, persons = Frames().frame((1, LEFT))
    weak = records(frame, persons, native((0.89, points())))
    assert weak[1]["status"] == "no_candidate" and all(weak[1][k] is None for k in lm.MEASURES)
    elsewhere = points(cx=500.0, cy=400.0)  # a strong face outside the person's head region
    assert records(frame, persons, native((0.97, elsewhere)))[1]["status"] == "no_candidate"


def test_conflicts_are_ambiguous_for_everyone_involved() -> None:
    frame, persons = Frames().frame((1, LEFT))
    two = records(frame, persons, native((0.95, points()), (0.94, points(cx=140.0))))
    assert two[1] == {**dict.fromkeys(lm.RECORD_KEYS), "status": "ambiguous"}
    frame, persons = Frames().frame((1, LEFT), (2, (0.1, 0.05, 0.4, 0.9)))  # overlapping people, one face
    shared = records(frame, persons, native((0.95, points())))
    assert shared[1]["status"] == shared[2]["status"] == "ambiguous"


def test_a_failed_pass_an_unrecorded_pass_and_the_person_bound_are_explicit() -> None:
    frame, persons = Frames().frame((1, LEFT))
    assert records(frame, persons, None)[1] == {**dict.fromkeys(lm.RECORD_KEYS), "status": "unavailable",
                                                "reason": "pass_failed"}
    unrecorded = FullScale(((0.95, box_of(points())),), 1.0)  # landmarks not recorded (None)
    assert records(frame, persons, unrecorded)[1]["status"] == "unavailable"
    people = [(t, (0.02 * t, 0.1, 0.02 * t + 0.3, 0.9)) for t in (7, 3, 9, 1, 5)]
    frame, persons = Frames().frame(*people)
    out = records(frame, persons, native((0.95, points(cx=500.0, cy=400.0))))
    assert out[9] == {**dict.fromkeys(lm.RECORD_KEYS), "status": "not_recorded", "reason": "persons_bound"}
    assert sorted(t for t, v in out.items() if v["status"] != "not_recorded") == [1, 3, 5, 7]


def test_an_invalid_candidate_keeps_its_score_and_reason_and_a_displaced_face_shows_a_low_overlap() -> None:
    frame, persons = Frames().frame((1, LEFT))
    out = records(frame, persons, native((0.92, points(d=3.0))), production=(box_of(points()),))
    assert out[1]["status"] == "invalid" and out[1]["reason"] == "eye_spacing_too_small" and out[1]["score"] == 0.92
    assert all(out[1][k] is None for k in lm.MEASURES) and out[1]["face_iou"] is not None
    shifted = NormalizedBox(x1=0.25, y1=0.16, x2=0.34, y2=0.3)  # production's face beside the native one
    low = records(frame, persons, native((0.95, points())), production=(shifted,))
    assert low[1]["status"] == "measured" and low[1]["face_iou"] < 0.5


# ---------------------------------------------------------------- the backend's opt-in parse


class FakeYuNet:
    def __init__(self, rows) -> None:
        self.rows = rows

    def setInputSize(self, size) -> None:  # noqa: N802 - OpenCV's name
        pass

    def setScoreThreshold(self, value) -> None:  # noqa: N802
        pass

    def detect(self, image):
        return 1, self.rows


def test_the_native_pass_keeps_landmarks_in_its_candidates_order_only_when_asked() -> None:
    low, high = points(), points(cx=300.0, lateral=0.2)
    rows = [yunet_row(low, 0.91), yunet_row(high, 0.97), [*yunet_row(points(cx=400.0), 0.95)[:9], math.nan,
                                                          *yunet_row(points(cx=400.0), 0.95)[10:]]]
    plain = full_scale_pass(FakeYuNet(rows), 640, 480, "frame")
    assert plain.landmarks is None and len(plain.candidates) == 3  # unchanged without the diagnostic
    kept = full_scale_pass(FakeYuNet(rows), 640, 480, "frame", landmarks=True)
    assert kept.candidates == plain.candidates
    assert kept.landmarks[0] == lm.from_row(yunet_row(high, 0.97))  # best first, with its own landmarks
    assert kept.landmarks[1] is None  # the row with a non-finite landmark: its candidate stays, its landmarks None
    assert kept.landmarks[2] == lm.from_row(yunet_row(low, 0.91))
    many = full_scale_pass(FakeYuNet([yunet_row(points(), 0.9)] * 12), 640, 480, "f", landmarks=True)
    assert len(many.landmarks) == len(many.candidates) == FULL_SCALE_KEPT


def test_the_backend_enables_landmarks_only_after_the_full_scale_pass_and_before_the_load(tmp_path) -> None:
    from sentinel.identity.legacy_deepface import LegacyDeepFaceBackend

    backend = LegacyDeepFaceBackend(tmp_path)
    with pytest.raises(FaceBackendError, match="landmarks_need_full_scale"):
        backend.enable_landmarks()
    backend.enable_full_scale()
    backend.enable_landmarks()
    assert backend.landmarks_enabled is False  # not loaded: no detector yet
    backend._runtime = object()  # as after load()
    with pytest.raises(FaceBackendError, match="already_loaded"):
        backend.enable_landmarks()


# ---------------------------------------------------------------- the schema-7 records


def outcome(full: FullScale | None, *, landmarks: bool, embedding=A_FACE, production: bool = True):
    frames = Frames()
    core = EdgeCore(CONFIG, frames.clock, None, face_recognition=Capability.AVAILABLE, enrollment=ENROLLMENT)
    frame, persons = frames.frame((1, LEFT))
    core.on_frame(frame, persons, frames.stream)
    face = points()
    faces = (FaceObservation(frame=frame.key, box=box_of(face), quality=0.91, embedding=embedding),) if production else ()
    return core.on_face_outcome(frame, persons, faces, frames.stream, full_scale=full, landmarks=landmarks).face_result


def test_schema_7_adds_the_landmark_record_and_time_and_changes_no_vote() -> None:
    full = native((0.93, points()))
    with_landmarks = outcome(full, landmarks=True).record(IDENTITY_LANDMARK_SCHEMA)
    without = outcome(full, landmarks=False).record(IDENTITY_FULL_SCALE_SCHEMA)
    person = with_landmarks["persons"][0]
    assert person["landmarks"]["status"] == "measured" and person["landmarks"]["face_iou"] == 1.0
    assert isinstance(with_landmarks["landmark_ms"], float) and with_landmarks["landmark_ms"] >= 0
    assert {k: v for k, v in person.items() if k != "landmarks"} == without["persons"][0]  # the same vote and numbers
    assert {k: v for k, v in with_landmarks.items() if k not in ("persons", "landmark_ms", "schema",
                                                                 "applied_mono_ns", "processing_ms")} \
        == {k: v for k, v in without.items() if k not in ("persons", "schema", "applied_mono_ns", "processing_ms")}
    assert "landmarks" not in json.dumps(without) and "landmark_ms" not in without
    geometry = outcome(full, landmarks=True).record(IDENTITY_GEOMETRY_SCHEMA)  # schema 6 is unchanged by schema 7
    assert "landmarks" not in geometry["persons"][0] and "prototype" in geometry["persons"][0]
    failed = outcome(None, landmarks=True).record(IDENTITY_LANDMARK_SCHEMA)
    assert failed["persons"][0]["landmarks"]["status"] == "unavailable" and failed["full_scale"] is None
    nobody = outcome(full, landmarks=True, embedding=NOBODY).record(IDENTITY_LANDMARK_SCHEMA)
    assert nobody["persons"][0]["vote"] == "unknown" and nobody["persons"][0]["landmarks"]["status"] == "measured"


# ---------------------------------------------------------------- runner scripts: readiness, block letter, ties


STARTUP = {"detector_problem": None, "face_problem": None,
           "face": {"validation_run": True, "identities_enrolled": 1, "landmark_diagnostic": lm.marker()}}


def test_the_guard_requires_the_pinned_marker_and_no_face_or_detector_problem(runners) -> None:
    problem = runners.guard.readiness_problem
    assert runners.guard.LANDMARK_MARKER == lm.marker() and "landmark_diagnostic" in runners.guard.READINESS
    assert problem("landmark_diagnostic", STARTUP) is None
    assert problem("landmark_diagnostic", {**STARTUP, "face_problem": "load_failed"}) == "face_problem:load_failed"
    assert problem("landmark_diagnostic", {**STARTUP, "detector_problem": "x"}) == "detector_problem:x"
    assert problem("landmark_diagnostic", {**STARTUP, "face": {"validation_run": True}}) == "marker_missing"
    changed = {**STARTUP, "face": {**STARTUP["face"], "landmark_diagnostic": {**lm.marker(), "score": 0.8}}}
    assert problem("landmark_diagnostic", changed) == "marker_not_pinned"
    assert problem("landmark_diagnostic", None) == "startup_missing"
    assert problem("tracking_diagnostic", STARTUP) == "marker_missing"  # the other requirement is unchanged
    verdict = importlib.import_module("guard_verdict")
    assert verdict.READINESS == runners.guard.READINESS
    result = {"mode": "step5_guard", "status": "duration_stop", "child": {"returncode": 0}, "guard": {"trigger": None},
              "stop": {"forced": False},
              "cleanup_clear": True, "readiness": {"required": "landmark_diagnostic", "met": True, "reason": None},
              "parameters": {"cues": [{"name": "t0"}, {"name": "stop"}]},
              "cues": [{"name": "t0", "printed_mono_s": 1.0}, {"name": "stop", "printed_mono_s": 621.0}]}
    judged = dict(verdict.verdict(result, 0, ("t0", "stop"), "landmark_diagnostic"))
    assert judged["readiness"] == "met" and judged["cues_printed"] == "2/2"


PROVENANCE_L = ["LB_exit=0", "LF_exit=0", "LM_attempt=1 utc=2026-10-19T10:00:00Z", "LM_sync_exit=0", "LM_drop_exit=0",
                "LM_after_record_exit=0", "LM_exit=0", "LO_observation=hm-x", "LO_memory_record=memory-prepared-20261019T100100Z.json",
                "LO_prepared=complete", "LO_exit=0"]


def test_the_launch_check_reads_the_named_blocks_and_keeps_the_f2_default(runners, tmp_path) -> None:
    provenance = tmp_path / "provenance.txt"
    provenance.write_text("\n".join(PROVENANCE_L) + "\n")
    memory = runners.memory
    assert memory.preparation(provenance, "L") == "memory-prepared-20261019T100100Z.json"
    with pytest.raises(memory.Refused, match="drop_not_attempted"):
        memory.preparation(provenance)  # the F2 package's letter: these lines are not its blocks'
    provenance.write_text("\n".join(line.replace("L", "F", 1) if line[0] == "L" else line for line in PROVENANCE_L)
                          + "\n")
    assert memory.preparation(provenance) == "memory-prepared-20261019T100100Z.json"
    with pytest.raises(SystemExit):
        memory.main(["check", "--session-dir", str(tmp_path), "--reference-result", "r", "--out", "o",
                     "--block", "LL"])


def test_the_runner_scripts_keep_the_runtimes_values(runners) -> None:
    check, evidence, landmark = runners.check, runners.evidence, runners.landmark
    assert lm.LANDMARK_SCORE == YUNET_THRESHOLD == FULL_SCALE_COUNTED
    assert check.LANDMARK_SCHEMA == IDENTITY_LANDMARK_SCHEMA and check.LANDMARK_SCHEMA in check.SCHEMAS
    assert check.LANDMARK_RECORD_KEYS == set(lm.RECORD_KEYS) and check.LANDMARK_MEASURES == lm.MEASURES
    assert check.LANDMARK_INVALID == lm.INVALID_REASONS and check.LANDMARK_MIN_EYE_SPACING_PX == lm.MIN_EYE_SPACING_PX
    assert {k: v for k, v in check.LANDMARK_FIXED.items() if v is not None} == lm.OTHER_REASONS
    assert set(check.LANDMARK_FIXED) | {"measured", "invalid"} == set(lm.STATUSES)
    assert evidence.LANDMARK_STATUSES == lm.STATUSES and evidence.LANDMARK_MEASURES == lm.MEASURES
    assert evidence.LANDMARK_REASONS == (*lm.INVALID_REASONS, *lm.OTHER_REASONS.values())
    assert evidence.LANDMARK_MARKER == tuple(lm.marker()) and landmark.PINNED_MARKER == lm.marker()
    assert landmark.BAND_TAIL == pytest.approx((1 - landmark.BAND_COVERAGE) / len(landmark.PRIMARY) / 2)
    assert set(landmark.PRIMARY + landmark.SECONDARY) <= set(lm.MEASURES)
    assert runners.account.FIELDS_V10 == runners.account.FIELDS_V6 and 10 in runners.account.SEPARATE_FORMATS


@pytest.mark.parametrize("change", [
    lambda r: r.update(status="measured", reason="eye_spacing_too_small"),
    lambda r: r.update(status="invalid", reason="eye_spacing_too_small"),  # numbers left on an invalid record
    lambda r: r.update(status="no_candidate"),  # numbers on a fixed status
    lambda r: r.update(score=0.85),  # a candidate under production's threshold
    lambda r: r.update(face_iou=None),  # production assigned a face, so the overlap must be there
    lambda r: r.update(eye_spacing_px=3.0),
    lambda r: r.update(mouth_vertical=0.0),
    lambda r: r.update(eye_line_deg=95.0),
    lambda r: r.update(extra=1),
    lambda r: r.pop("nose_lateral"),
])
def test_face_check_refuses_a_landmark_record_that_does_not_fit_its_status(runners, change) -> None:
    record = outcome(native((0.93, points())), landmarks=True).record(IDENTITY_LANDMARK_SCHEMA)
    assert runners.check._record_problem("result", record) is None
    change(record["persons"][0]["landmarks"])
    assert "landmark record" in runners.check._record_problem("result", record)


def test_face_check_accepts_every_status_and_ties_the_landmark_time_to_the_outcome(runners) -> None:
    for full, production in ((native((0.93, points())), False), (native((0.92, points(d=3.0))), True),
                             (native((0.5, points())), True), (None, True)):
        record = outcome(full, landmarks=True, production=production).record(IDENTITY_LANDMARK_SCHEMA)
        assert runners.check._record_problem("result", record) is None, record["persons"][0]["landmarks"]
    record = outcome(native((0.93, points())), landmarks=True).record(IDENTITY_LANDMARK_SCHEMA)
    record["landmark_ms"] = None
    assert "landmark time" in runners.check._record_problem("result", record)
    record = outcome(None, landmarks=True).record(IDENTITY_LANDMARK_SCHEMA)
    record["persons"][0]["landmarks"] = {**dict.fromkeys(lm.RECORD_KEYS), "status": "no_candidate"}
    assert "landmark record" in runners.check._record_problem("result", record)  # no pass, so nothing was looked at


def test_a_candidate_at_productions_threshold_attaches_and_one_just_under_it_does_not() -> None:
    frame, persons = Frames().frame((1, LEFT))
    assert records(frame, persons, native((0.9, points())))[1]["status"] == "measured"
    assert records(frame, persons, native((0.8999, points())))[1]["status"] == "no_candidate"
