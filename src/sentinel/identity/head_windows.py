"""Head windows: where to look for a face in a larger frame, and how far a face may drift and keep its owner (session 116,
part 14). Portable. Route B (session 116, part 15) uses the windows and the selection through ``identity.window_stage``, the only
runtime module that imports this one; ``own_candidate_margin`` and ``safe_skew_s`` stay unwired (route B takes each face and its
people from one frame, so no time difference enters). It decides no vote and changes no threshold.

**The windows.** A tracked person's box (normalized) gives a head region, the part of the box in which ``association`` lets a
face centre lie. ``plan_windows`` turns the regions of a frame's people into rectangles of that frame's pixels, for a
detector whose input is at most ``side`` pixels on a side:
- A window is ``side`` x ``side`` of native pixels (less only where the frame itself is smaller), centred on what it must
  cover and moved, never cut, to stay inside the frame. The input size is therefore constant in the usual case.
- A person so near that the region and its margin exceed ``side`` gets a larger window, scaled down for detection
  (``scale`` < 1). Such a face is large; the face's cut can still come from the native pixels.
- People whose regions fit one window share it. Anything more than ``max_windows`` windows leaves people ``uncovered``, by
  index, so that a caller reports them instead of treating them as having no face.

**Each face once.** Windows may overlap. ``select_faces`` works in frame pixels: it drops a detection touching a window's edge
where that edge is not the frame's (a partial face), then suppresses overlapping detections across windows, the highest score
kept. The owner is still ``association.associate``'s to decide, over all faces and all people together.

**Coordinates.** A person box comes from one frame and the window is cut from another, possibly another stream's. Both are
normalized to the whole picture, which ``dr-20261105T085421Z`` measured for this camera (the substream is the main stream's
whole picture, squeezed: left -0.001, right 1.000, top 0.000, bottom 1.001). That is this camera's, at those settings, and it
says nothing about when each frame was captured.

**The ownership bound.** With a time difference between the person box and the face, the face has moved relative to the box.
``own_candidate_margin`` measures how far a face can move before it stops being a candidate of its own person under
``association``'s rule. While it stays a candidate, the face cannot be assigned to anyone else: ``associate`` assigns a face only
to its sole candidate, so at worst the pair is AMBIGUOUS. This is what ``safe_skew_s`` turns into a time for a given velocity. It
is a sufficient condition for no wrong owner, per person and per moment, and no fixed tolerance follows from it: the inputs are
measured boxes and speeds, and a track's velocity is an estimate of the head's. With one tracked person there is no other owner,
and drifting out of the box only loses a vote.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..contracts import NormalizedBox
from .association import AssociationGeometry, _is_candidate

DEFAULT_SIDE = 640  # YuNet's own limit in DeepFace's client; a constant input bounds its memory and time
DEFAULT_MARGIN = 0.5  # of the person box's width, around the head region: room for a face reaching past the box
DEFAULT_MAX_WINDOWS = 4
DUPLICATE_IOU = 0.3  # YuNet's own non-maximum suppression default, applied across windows
EDGE_TOLERANCE_PX = 1.0  # in the detector's input pixels


@dataclass(frozen=True)
class Window:
    """A rectangle of frame pixels and how it reaches the detector. ``owners`` are the indices of the people it covers."""

    x: int
    y: int
    w: int
    h: int
    scale: float
    owners: tuple[int, ...]

    @property
    def input_size(self) -> tuple[int, int]:
        """The detector's input (width, height): the window itself at scale 1."""
        return max(1, round(self.w * self.scale)), max(1, round(self.h * self.scale))

    @property
    def scale_x(self) -> float:
        """The input's width over the window's: ``scale`` itself unless rounding the input's size moved it."""
        return self.input_size[0] / self.w

    @property
    def scale_y(self) -> float:
        return self.input_size[1] / self.h

    def to_frame(self, x: float, y: float) -> tuple[float, float]:
        """A point of the detector's input, in frame pixels (each axis by its own exact scale)."""
        return self.x + x / self.scale_x, self.y + y / self.scale_y


@dataclass(frozen=True)
class WindowPlan:
    windows: tuple[Window, ...]
    uncovered: tuple[int, ...]  # people beyond max_windows: not looked at this tick, never "no face"


@dataclass(frozen=True)
class FaceBox:
    """A detection in frame pixels, with the window it came from."""

    x: float
    y: float
    w: float
    h: float
    score: float
    window: int


@dataclass(frozen=True)
class Selected:
    faces: tuple[FaceBox, ...]
    dropped_at_window_edge: int
    dropped_duplicate: int


@dataclass(frozen=True)
class _Rect:
    """Whole frame pixels: rounded outward once, so that a window of this size always covers what was asked."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    def union(self, other: _Rect) -> _Rect:
        return _Rect(min(self.x1, other.x1), min(self.y1, other.y1), max(self.x2, other.x2), max(self.y2, other.y2))


def head_region(person: NormalizedBox, head_fraction: float = AssociationGeometry.head_fraction) -> NormalizedBox:
    """The part of a person's box where ``association`` lets a face centre lie."""
    return NormalizedBox(x1=person.x1, y1=person.y1, x2=person.x2,
                         y2=person.y1 + head_fraction * (person.y2 - person.y1))


def _required(person: NormalizedBox, width: int, height: int, head_fraction: float, margin: float) -> _Rect:
    """The head region and its margin in frame pixels, inside the frame."""
    region = head_region(person, head_fraction)
    pad = margin * (person.x2 - person.x1) * width
    return _Rect(max(0, math.floor(region.x1 * width - pad)), max(0, math.floor(region.y1 * height - pad)),
                 min(width, math.ceil(region.x2 * width + pad)), min(height, math.ceil(region.y2 * height + pad)))


def _placed(need: _Rect, width: int, height: int, side: int) -> tuple[int, int, int, int, float]:
    """A window covering ``need``: (x, y, w, h, scale). Native pixels when it fits ``side``, else larger and scaled down.
    Centred, then moved to stay inside the frame; moving never uncovers ``need``, which lies inside the frame."""
    longest = max(need.w, need.h)
    span = side if longest <= side else longest
    w, h = min(span, width), min(span, height)
    x = min(max(0, need.x1 - (w - need.w) // 2), width - w)
    y = min(max(0, need.y1 - (h - need.h) // 2), height - h)
    largest = max(w, h)
    return x, y, w, h, 1.0 if largest <= side else side / largest


def plan_windows(persons: Sequence[NormalizedBox], width: int, height: int, *, side: int = DEFAULT_SIDE,
                 head_fraction: float = AssociationGeometry.head_fraction, margin: float = DEFAULT_MARGIN,
                 max_windows: int = DEFAULT_MAX_WINDOWS) -> WindowPlan:
    """The windows for one frame's people. Deterministic: the largest people first, and each joins the first group it fits."""
    if width <= 0 or height <= 0 or side <= 0 or max_windows <= 0:
        raise ValueError("frame size, side and max_windows must be positive")
    if not 0.0 < head_fraction <= 1.0 or margin < 0.0:
        raise ValueError("head_fraction must be in (0, 1] and margin not negative")
    order = sorted(range(len(persons)), key=lambda i: (-(persons[i].x2 - persons[i].x1) * (persons[i].y2 - persons[i].y1), i))
    groups: list[tuple[_Rect, list[int]]] = []
    uncovered: list[int] = []
    for index in order:
        need = _required(persons[index], width, height, head_fraction, margin)
        for position, (rect, owners) in enumerate(groups):
            joined = rect.union(need)
            if joined.w <= side and joined.h <= side:
                groups[position] = (joined, [*owners, index])
                break
        else:
            if len(groups) < max_windows:
                groups.append((need, [index]))
            else:
                uncovered.append(index)
    windows = []
    for rect, owners in groups:
        x, y, w, h, scale = _placed(rect, width, height, side)
        windows.append(Window(x, y, w, h, scale, tuple(sorted(owners))))
    return WindowPlan(tuple(windows), tuple(sorted(uncovered)))


def detection_to_frame(window: Window, index: int, x: float, y: float, w: float, h: float, score: float) -> FaceBox:
    """A detector row (in the window's input pixels) as a box in frame pixels."""
    left, top = window.to_frame(x, y)
    return FaceBox(left, top, w / window.scale_x, h / window.scale_y, score, index)


def touches_window_edge(face: FaceBox, window: Window, width: int, height: int) -> bool:
    """Whether a detection reaches an edge of its window that is not an edge of the frame: part of a face may lie beyond."""
    tolerance = EDGE_TOLERANCE_PX / window.scale
    return ((window.x > 0 and face.x <= window.x + tolerance)
            or (window.y > 0 and face.y <= window.y + tolerance)
            or (window.x + window.w < width and face.x + face.w >= window.x + window.w - tolerance)
            or (window.y + window.h < height and face.y + face.h >= window.y + window.h - tolerance))


def _iou(a: FaceBox, b: FaceBox) -> float:
    w = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    h = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    if w <= 0 or h <= 0:
        return 0.0
    shared = w * h
    return shared / (a.w * a.h + b.w * b.h - shared)


def select_faces(faces: Sequence[FaceBox], windows: Sequence[Window], width: int, height: int,
                 duplicate_iou: float = DUPLICATE_IOU) -> Selected:
    """Each face once: partial detections at a window's edge dropped, then duplicates across windows, the highest score
    kept (ties: the earlier window). The order is deterministic."""
    whole = [f for f in faces if not touches_window_edge(f, windows[f.window], width, height)]
    kept = suppress_duplicates(whole, duplicate_iou)
    return Selected(tuple(kept), len(faces) - len(whole), len(whole) - len(kept))


def suppress_duplicates(faces: Sequence[FaceBox], duplicate_iou: float = DUPLICATE_IOU) -> list[FaceBox]:
    """Overlapping detections (IoU at least ``duplicate_iou``) as one, the highest score kept (ties: the earlier window, then
    position); deterministic."""
    kept: list[FaceBox] = []
    for face in sorted(faces, key=lambda f: (-f.score, f.window, f.x, f.y)):
        if all(_iou(face, other) < duplicate_iou for other in kept):
            kept.append(face)
    return kept


def inside_share(face: FaceBox, other: FaceBox) -> float:
    """How much of ``face``'s area lies inside ``other`` (0 to 1)."""
    w = min(face.x + face.w, other.x + other.w) - max(face.x, other.x)
    h = min(face.y + face.h, other.y + other.h) - max(face.y, other.y)
    area = face.w * face.h
    return 0.0 if w <= 0 or h <= 0 or area <= 0 else (w * h) / area


def to_normalized(face: FaceBox, width: int, height: int) -> NormalizedBox | None:
    """A frame-pixel box as the normalized box ``association`` takes, clamped to the frame; None when nothing is inside."""
    x1, y1 = max(0.0, face.x / width), max(0.0, face.y / height)
    x2, y2 = min(1.0, (face.x + face.w) / width), min(1.0, (face.y + face.h) / height)
    return NormalizedBox(x1=x1, y1=y1, x2=x2, y2=y2) if x1 < x2 and y1 < y2 else None


def _shifted(face: NormalizedBox, dx: float, dy: float) -> NormalizedBox | None:
    """The face moved by (dx, dy), not clamped: leaving the frame is treated as leaving the box."""
    x1, y1, x2, y2 = face.x1 + dx, face.y1 + dy, face.x2 + dx, face.y2 + dy
    if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0:
        return None
    return NormalizedBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _largest_step(face: NormalizedBox, person: NormalizedBox, geometry: AssociationGeometry, dx: float, dy: float) -> float:
    """The largest t in [0, 1] (to 1e-6) with the face moved by t*(dx, dy) still its person's candidate; the rule is convex
    along a line from a candidate position, so bisection finds the boundary."""
    def holds(t: float) -> bool:
        moved = _shifted(face, t * dx, t * dy)
        return moved is not None and _is_candidate(moved, person, geometry)

    if not holds(0.0):
        return 0.0
    if holds(1.0):
        return 1.0
    low, high = 0.0, 1.0
    while high - low > 1e-6:
        middle = (low + high) / 2
        low, high = (middle, high) if holds(middle) else (low, middle)
    return low


def own_candidate_margin(face: NormalizedBox, person: NormalizedBox,
                         geometry: AssociationGeometry = AssociationGeometry()) -> tuple[float, float]:
    """How far the face can move left or right, and up or down, and stay ``person``'s candidate (normalized): the smaller of
    each pair of directions. (0.0, 0.0) when it is not a candidate now."""
    if not _is_candidate(face, person, geometry):
        return 0.0, 0.0
    x = min(_largest_step(face, person, geometry, -1.0, 0.0), _largest_step(face, person, geometry, 1.0, 0.0))
    y = min(_largest_step(face, person, geometry, 0.0, -1.0), _largest_step(face, person, geometry, 0.0, 1.0))
    return x, y


def safe_skew_s(face: NormalizedBox, person: NormalizedBox, velocity: tuple[float, float],
                geometry: AssociationGeometry = AssociationGeometry()) -> float:
    """The time difference (seconds) within which a face moving at ``velocity`` (normalized per second, in frame axes) stays
    ``person``'s candidate, in the direction of travel: inf for a still person, 0.0 when it is not a candidate now. Within it the
    face cannot go to anyone else. A measure for validation records, never a tolerance by itself."""
    vx, vy = velocity
    if not all(math.isfinite(v) for v in velocity):
        raise ValueError("velocity must be finite")
    if not _is_candidate(face, person, geometry):
        return 0.0
    speed = math.hypot(vx, vy)
    if speed == 0.0:
        return math.inf
    both = _largest_step(face, person, geometry, vx / speed, vy / speed)
    reverse = _largest_step(face, person, geometry, -vx / speed, -vy / speed)
    return min(both, reverse) / speed  # the sign of the time difference is not assumed: either direction
