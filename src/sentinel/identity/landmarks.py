"""Landmark measurements for the validation-only landmark diagnostic (session 90; diagnostic only, never a vote).

What it measures, and from what. The D86 native-scale pass runs the pinned YuNet (``face_detection_yunet_2023mar.onnx``
through ``cv2.FaceDetectorYN``) once more on the analysed frame itself: 640x480, no padding, no scaling. OpenCV
documents each detection as one row of 15 float32 values: the box ``x, y, w, h``, then five landmarks ``(x, y)`` in the
order right eye, left eye, nose tip, right mouth corner, left mouth corner ("right" and "left" are the person's: for a
face toward the camera the right eye appears on the image's left), then the score. DeepFace 0.0.99's YuNet client reads
the same columns (4-5 as its ``right_eye``, 6-7 as its ``left_eye``). This diagnostic keeps, for each detection scoring
at least LANDMARK_SCORE that the one-to-one association gives to a person, a few numbers derived from those five points.
No image, crop, embedding or landmark coordinate is kept; production detection, embeddings and votes never see them.

Coordinates. Frame pixels at native resolution: origin at the frame's top-left corner, x to the right, y downward;
YuNet's values unrounded and unclipped.

Derived measurements, in the face's own eye-line frame (E_r, E_l the right and left eye, N the nose tip, R and L the
right and left mouth corners):

- ``M = (E_r + E_l) / 2``; ``v = E_l - E_r``; ``d = |v|`` (the eye spacing, px); ``u = v / d`` (along the eye line,
  toward the left eye); ``n = (-u_y, u_x)`` (perpendicular to it, pointing down the face when the face is upright);
  ``P = (R + L) / 2`` (the mouth midpoint);
- ``nose_lateral = ((N - M) . u) / d``: the nose tip's offset along the eye line from the eye midpoint, in eye
  spacings (positive toward the left eye, which is the image's right for an upright face toward the camera);
- ``nose_vertical = ((N - M) . n) / d``: the nose tip's distance below the eye line, in eye spacings;
- ``mouth_vertical = ((P - M) . n) / d``: the mouth midpoint's distance below the eye line, in eye spacings;
- ``nose_position = nose_vertical / mouth_vertical``: where the nose tip lies between the eye line (0) and the mouth
  line (1);
- ``mouth_lateral = ((P - M) . u) / d``: the mouth midpoint's offset along the eye line, in eye spacings;
- ``eye_line_deg = atan2(v_y, v_x)`` in degrees: the eye line's angle in the image (0 level; positive when the left eye
  is lower on screen, a clockwise turn as displayed);
- ``eye_spacing_px = d``, and ``eye_spacing_ratio = d / w`` (w the detection box's width).

The first five are invariant to the face's position, size and in-plane rotation in the image by construction; the last
three are not. All are proxies: shape descriptors of five regressed points. They are not head angles and are not
calibrated against any measured pose; the reading never converts them into yaw, pitch or roll and never calls any range
of them "frontal". Head rotation is one thing that moves them; so do the face's own shape, expression (the mouth
points), glasses or anything partly covering the face, the camera's height and angle (a constant offset for one seat),
lens distortion away from the frame's centre, and the detector's own landmark error. That error is not documented by
OpenCV and is not measured here; at this camera's face size (about 57 px wide, about 22 px between the eyes) one or two
pixels of error are several hundredths of an eye spacing, and the error may itself grow when a face is turned.

Invalid observations (no measurement; a fixed reason): a row with fewer than 15 values or a non-finite value
(``malformed_row``); any landmark outside the frame (``landmark_outside_frame``: YuNet extrapolated it); an eye spacing
below MIN_EYE_SPACING_PX (``eye_spacing_too_small``; a guard against undefined directions, not a calibrated limit); the
left eye not to the right of the right eye in the image (``eyes_not_left_to_right``: the eye line turned beyond 90
degrees, or the points swapped); the mouth midpoint not below the eye line (``mouth_not_below_eyes``).

Association (``landmark_records``). The native candidates scoring at least LANDMARK_SCORE (production's own detection
threshold, so the two passes are compared like for like) are associated with the frame's people by the same one-to-one
rule production's faces are (``association.associate``: the face's centre in the person's head region, most of the face
inside the box, each face to at most one person and each person at most one face, conflicts ambiguous). Weaker
candidates never attach. Each person on an applied result then gets one record with an explicit status, never the
nearest face: ``measured``; ``invalid`` (with the reason above); ``no_candidate`` (no native candidate for this person);
``ambiguous`` (the association found a conflict); ``unavailable`` (the native pass gave nothing for this frame:
``pass_failed``); ``not_recorded`` (``persons_bound``: beyond LANDMARK_PERSONS_MAX people on the frame, by track ID).
When production also assigned a face to the same person on the same frame, ``face_iou`` is the overlap of the two face
boxes (intersection over union), so a reading can tell whether both passes found the same face region; it is null when
production assigned none. Geometry only: it never shows what either box covers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..contracts import FrameRef, TrackKey, TrackObservation
from .association import AssociationGeometry, FaceObservation, Ownership, associate
from .continuity import iou
from .yunet import YUNET_THRESHOLD

LANDMARK_SCHEMA_VERSION = 1
ROW_VALUES = 15  # x, y, w, h, five landmarks (x, y), score (cv2.FaceDetectorYN)
LANDMARK_SCORE = YUNET_THRESHOLD  # production detection's own threshold: candidates at least this strong take part
MIN_EYE_SPACING_PX = 4.0  # below this the eye line's direction is undefined in practice; a guard, not a calibration
LANDMARK_PERSONS_MAX = 4  # people per result that get a record; the others are counted as not_recorded
RATIO_DECIMALS = 3
DEGREE_DECIMALS = 1
PX_DECIMALS = 1
MEASURES = ("nose_lateral", "nose_vertical", "mouth_vertical", "nose_position", "mouth_lateral", "eye_line_deg",
            "eye_spacing_px", "eye_spacing_ratio")
STATUSES = ("measured", "invalid", "no_candidate", "ambiguous", "unavailable", "not_recorded")
INVALID_REASONS = ("malformed_row", "landmark_outside_frame", "eye_spacing_too_small", "eyes_not_left_to_right",
                   "mouth_not_below_eyes")
OTHER_REASONS = {"unavailable": "pass_failed", "not_recorded": "persons_bound"}
RECORD_KEYS = ("status", "reason", "score", "face_iou", *MEASURES)


@dataclass(frozen=True)
class Landmarks:
    """One YuNet detection's box and five landmarks in native frame pixels, unrounded (a row's columns 0-13)."""

    box: tuple[float, float, float, float]  # x, y, w, h
    right_eye: tuple[float, float]  # columns 4-5
    left_eye: tuple[float, float]  # columns 6-7
    nose: tuple[float, float]  # columns 8-9
    mouth_right: tuple[float, float]  # columns 10-11
    mouth_left: tuple[float, float]  # columns 12-13

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        return (self.right_eye, self.left_eye, self.nose, self.mouth_right, self.mouth_left)


def from_row(row: Any) -> Landmarks | None:
    """A YuNet row's box and landmarks, or None when it has fewer than ROW_VALUES values or a non-finite one."""
    try:
        if len(row) < ROW_VALUES:
            return None
        values = [float(row[i]) for i in range(ROW_VALUES)]
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    pairs = [(values[i], values[i + 1]) for i in range(4, 14, 2)]
    return Landmarks((values[0], values[1], values[2], values[3]), *pairs)


def measure(landmarks: Landmarks | None, width: int, height: int) -> tuple[dict[str, float] | None, str | None]:
    """The derived measurements (the module's formulas), or None and the reason it is invalid."""
    if landmarks is None:
        return None, "malformed_row"
    if any(not (0.0 <= x <= width and 0.0 <= y <= height) for x, y in landmarks.points):
        return None, "landmark_outside_frame"
    (rx, ry), (lx, ly) = landmarks.right_eye, landmarks.left_eye
    vx, vy = lx - rx, ly - ry
    d = math.hypot(vx, vy)
    if d < MIN_EYE_SPACING_PX:
        return None, "eye_spacing_too_small"
    if vx <= 0:
        return None, "eyes_not_left_to_right"
    ux, uy = vx / d, vy / d
    nx, ny = -uy, ux
    mx, my = (rx + lx) / 2, (ry + ly) / 2
    px = (landmarks.mouth_right[0] + landmarks.mouth_left[0]) / 2
    py = (landmarks.mouth_right[1] + landmarks.mouth_left[1]) / 2
    nose = (landmarks.nose[0] - mx, landmarks.nose[1] - my)
    mouth = (px - mx, py - my)
    mouth_vertical = (mouth[0] * nx + mouth[1] * ny) / d
    if mouth_vertical <= 0:
        return None, "mouth_not_below_eyes"
    nose_vertical = (nose[0] * nx + nose[1] * ny) / d
    w = landmarks.box[2]
    values = {
        "nose_lateral": (nose[0] * ux + nose[1] * uy) / d,
        "nose_vertical": nose_vertical,
        "mouth_vertical": mouth_vertical,
        "nose_position": nose_vertical / mouth_vertical,
        "mouth_lateral": (mouth[0] * ux + mouth[1] * uy) / d,
        "eye_line_deg": math.degrees(math.atan2(vy, vx)),
        "eye_spacing_px": d,
        "eye_spacing_ratio": d / w if w > 0 and math.isfinite(w) else None,
    }
    if values["eye_spacing_ratio"] is None:
        return None, "malformed_row"
    return {k: round(v, DEGREE_DECIMALS if k == "eye_line_deg" else PX_DECIMALS if k == "eye_spacing_px"
                     else RATIO_DECIMALS) for k, v in values.items()}, None


def _record(status: str, reason: str | None = None, score: float | None = None, face_iou: float | None = None,
            values: Mapping[str, float] | None = None) -> dict[str, Any]:
    return {"status": status, "reason": reason,
            "score": None if score is None else round(score, RATIO_DECIMALS),
            "face_iou": None if face_iou is None else round(face_iou, RATIO_DECIMALS),
            **{k: None if values is None else values[k] for k in MEASURES}}


def landmark_records(frame: FrameRef, persons: Sequence[TrackObservation], production_faces: Sequence[FaceObservation],
                     production_face_of: Mapping[TrackKey, int], full_scale: Any,
                     geometry: AssociationGeometry) -> dict[TrackKey, dict[str, Any]]:
    """One record per person on an applied result (the module's association; never the nearest face). ``full_scale`` is
    the native pass's FullScale for this frame (legacy_deepface), or None when the pass gave nothing."""
    order = sorted(persons, key=lambda p: p.track_id)
    recorded = {p.key for p in order[:LANDMARK_PERSONS_MAX]}
    out: dict[TrackKey, dict[str, Any]] = {}
    landmarks = None if full_scale is None else full_scale.landmarks
    if full_scale is None or landmarks is None or len(landmarks) != len(full_scale.candidates):
        for p in persons:
            out[p.key] = (_record("unavailable", OTHER_REASONS["unavailable"]) if p.key in recorded
                          else _record("not_recorded", OTHER_REASONS["not_recorded"]))
        return out
    strong = [(index, score, box) for index, (score, box) in enumerate(full_scale.candidates) if score >= LANDMARK_SCORE]
    faces = [FaceObservation(frame=frame.key, box=box, quality=min(1.0, max(0.0, score))) for _, score, box in strong]
    association = associate(persons, faces, geometry)
    for p in persons:
        if p.key not in recorded:
            out[p.key] = _record("not_recorded", OTHER_REASONS["not_recorded"])
            continue
        ownership = association.person_ownership.get(p.key, Ownership.NO_FACE)
        if ownership is Ownership.AMBIGUOUS:
            out[p.key] = _record("ambiguous")
            continue
        if ownership is not Ownership.ASSIGNED:
            out[p.key] = _record("no_candidate")
            continue
        index, score, box = strong[association.face_of[p.key]]
        production = production_face_of.get(p.key)
        overlap = None if production is None else iou(production_faces[production].box, box)
        values, reason = measure(landmarks[index], frame.native_width, frame.native_height)
        out[p.key] = (_record("invalid", reason, score, overlap) if values is None
                      else _record("measured", None, score, overlap, values))
    return out


def marker() -> dict[str, Any]:
    """The startup line's face.landmark_diagnostic entry: the version and the pinned values a reading checks."""
    return {"version": LANDMARK_SCHEMA_VERSION, "score": LANDMARK_SCORE, "min_eye_spacing_px": MIN_EYE_SPACING_PX,
            "persons_max": LANDMARK_PERSONS_MAX}
