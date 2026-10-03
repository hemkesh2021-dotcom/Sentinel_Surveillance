"""Zone geometry in normalized native-image coordinates (guide chapter 9).

Coordinates are fractions of the native image: x to the right, y downwards,
both in [0, 1]. Zones are simple polygons. A person is placed in a zone by one
documented anchor point of their box, not by box overlap: for a floor zone the
bottom-centre of the box approximates where the person stands.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from ..contracts import NormalizedBox

Point = tuple[float, float]

MAX_VERTICES = 32
_MIN_AREA = 1e-6  # of the whole image; rules out degenerate (collinear) polygons


class Anchor(str, Enum):
    BOTTOM_CENTER = "bottom_center"  # where a standing person meets the floor
    CENTER = "center"


def anchor_point(box: NormalizedBox, anchor: Anchor) -> Point:
    x = (box.x1 + box.x2) / 2
    if anchor is Anchor.BOTTOM_CENTER:
        return x, box.y2
    return x, (box.y1 + box.y2) / 2


def polygon_problem(points: Sequence[Point]) -> str | None:
    """Why ``points`` is not a usable zone polygon, or None if it is."""
    if len(points) < 3:
        return "a zone polygon needs at least 3 points"
    if len(points) > MAX_VERTICES:
        return f"a zone polygon has at most {MAX_VERTICES} points"
    for index, (x, y) in enumerate(points):
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return f"point {index} ({x}, {y}) is outside the image; use fractions in [0, 1]"
    n = len(points)
    for index in range(n):
        if points[index] == points[(index + 1) % n]:
            return f"points {index} and {(index + 1) % n} are the same"
    if abs(_signed_area(points)) < _MIN_AREA:
        return "the polygon has no area"
    edges = [(points[i], points[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # neighbouring edges share a vertex
            if _segments_intersect(*edges[i], *edges[j]):
                return f"edges {i} and {j} cross; the polygon must not intersect itself"
    return None


def contains(polygon: Sequence[Point], point: Point) -> bool:
    """Even-odd ray casting. Points exactly on an edge are classified deterministically
    but either way; zones should not depend on single-pixel boundaries."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        (x1, y1), (x2, y2) = polygon[i], polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            crossing = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing:
                inside = not inside
    return inside


def _signed_area(points: Sequence[Point]) -> float:
    n = len(points)
    return sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1] for i in range(n)) / 2


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1, o2, o3, o4 = _orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and 0 not in (o1, o2, o3, o4):
        return True
    return (
        (o1 == 0 and _on_segment(a, b, c))
        or (o2 == 0 and _on_segment(a, b, d))
        or (o3 == 0 and _on_segment(c, d, a))
        or (o4 == 0 and _on_segment(c, d, b))
    )
