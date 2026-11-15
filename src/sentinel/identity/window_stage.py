"""Route B's face stage (session 116, part 15): faces found in head windows of the decoded picture itself, for the people the
tracker found on its working copy of that same picture. PROPOSED: used only by ``sentinel run`` under the main_single stream
profile, in validation runs, until a validation run passes on it. Portable: nothing here imports an image library.

**One frame.** Under main_single the capture stamps each decoded picture once (media.capture): the tracker is handed its
working copy, a squeeze of the whole picture (``FrameRef.whole_picture``), so a normalized person box means the same place in
the decoded picture; the windows are cut from that decoded picture, under the same FrameRef, and every face is stamped with
that frame's key. ``association.associate`` refuses faces and people from different frames, so no time difference between two
pictures can enter. ``prepare`` refuses a frame whose processed image is not the whole picture, an image of another size, and
people from another frame.

**At the offer** (the loop thread, ``prepare``): the windows are planned from that frame's people (``head_windows.plan_windows``,
the configured head fraction), and each window is copied at its detector input size, read-only: at most ``max_windows`` x
``side`` x ``side`` x 3 bytes (4 x 1,228,800 B with the defaults), never the whole decoded picture. A window larger than
``side`` (a person near the camera) is scaled down whole (``resize``, an area resize on the device); its detection, cut and
alignment then work on those scaled pixels, where the face is larger than in any window at scale 1.

**In the worker** (``LegacyDeepFaceBackend.window_faces``): D88's steps on each window: the pinned YuNet at the window's input
size, production's 0.9, then for the faces kept DeepFace's own cut, alignment, normalization and Facenet512. A window with no
face at the threshold is never embedded, and no whole-picture stand-in exists on this path.

**Which detections are kept** (``select``), in each window's input pixels, as DeepFace 0.0.99 computes them:
- **Whole cuts only.** DeepFace cuts and aligns a face from its integer box grown by ``int(0.5 * w)`` and ``int(0.5 * h)`` on
  each side (``detection.extract_sub_image``; black beyond the picture). Where that would read past a window edge that is not
  the picture's, the window lacks pixels the picture has, and the detection is dropped (``dropped_at_window_edge``). For a kept
  face at scale 1 the cut is then the same pixels D88's steps would cut from the decoded picture for the same detection:
  DeepFace's cut depends only on that integer box, the eyes' difference and those pixels, and a window is an integer offset.
- **Each face once:** overlapping detections from overlapping windows count once, the highest score kept
  (``head_windows.suppress_duplicates``, ``DUPLICATE_IOU``).
- **An unexplained drop withholds that window's people.** A dropped detection is explained only by a face another window kept
  whole with at least ``EXPLAINED_SHARE`` of it inside (its whole view). A dropped detection that overlaps that much with a
  whole detection of its own window is never explained: YuNet found both in the same pixels and kept them apart, so it is a
  second face, and another window's face over it may be that other face's twin (session 116, part 16 review). An unexplained
  drop is a face this window could not see whole, which might have been a second candidate for someone the window covers.
  Assigning that person's other face without it could turn an AMBIGUOUS pair into an assignment, so every person the window
  covers is **not analysed** this tick.

**Not analysed** are the people the window budget left uncovered (``head_windows.WindowPlan.uncovered``) and the people of a
window with an unexplained drop. They stay in the association, so a face that is also theirs stays AMBIGUOUS for everyone, and
they get ``Ownership.NOT_ANALYSED``: no vote, never NO_FACE ("no face visible") and never a stranger.

**The precondition, stated.** With the margin of half the person box's width, every face that ``associate`` could give a
covered person lies whole inside that person's window if it is at most 0.5 / (1 - min_face_inside) times as wide as the person
box (1.25 at the configured 0.6) and no taller than the box is wide (head_windows; tested). The runtime refuses main_single with
a smaller ``identity.min_face_inside``, for which the bound would shrink (``stream_profile_geometry``). A face the window cuts that YuNet does not detect at all cannot be flagged; such a face
would have to be larger than that. Thresholds, association, ownership, retention and expiry are downstream and unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..contracts import FrameRef, NormalizedBox, TrackKey, TrackObservation
from .head_windows import (
    DEFAULT_MARGIN,
    DEFAULT_MAX_WINDOWS,
    DEFAULT_SIDE,
    DUPLICATE_IOU,
    FaceBox,
    Window,
    detection_to_frame,
    inside_share,
    plan_windows,
    suppress_duplicates,
    to_normalized,
)

STAGE = "head_windows"
CONTEXT = 0.5  # DeepFace 0.0.99's extract_sub_image: the cut reads the box grown by int(0.5 * w) and int(0.5 * h)
EXPLAINED_SHARE = 0.5  # a dropped detection at least this much inside a kept face is part of that face
MIN_FACE_INSIDE_FOR_WINDOWS = 0.6  # the association's inside share for which the stated precondition is 1.25 box widths


class WindowStageError(Exception):
    """A fixed label: the frame, image or people cannot be windowed."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


@dataclass(frozen=True)
class WindowSettings:
    side: int = DEFAULT_SIDE
    margin: float = DEFAULT_MARGIN
    max_windows: int = DEFAULT_MAX_WINDOWS
    head_fraction: float = 0.4  # the configured association geometry's (identity.head_fraction)

    def numbers(self) -> dict[str, Any]:
        return {"kind": STAGE, "side": self.side, "margin": self.margin, "max_windows": self.max_windows,
                "head_fraction": self.head_fraction, "duplicate_iou": DUPLICATE_IOU, "context": CONTEXT,
                "explained_share": EXPLAINED_SHARE}


@dataclass(frozen=True)
class WindowImages:
    """The worker's private part of one frame: the decoded picture's size, the windows, who each covers, who none covers,
    and each window's detector input (a read-only copy)."""

    width: int
    height: int
    windows: tuple[Window, ...]
    owners: tuple[tuple[TrackKey, ...], ...]
    uncovered: tuple[TrackKey, ...]
    inputs: tuple[Any, ...]

    @property
    def nbytes(self) -> int:
        return sum(int(getattr(image, "nbytes", 0)) for image in self.inputs)

    def interior(self, index: int) -> tuple[bool, bool, bool, bool]:
        """Which of the window's edges (left, top, right, bottom) lie inside the picture rather than on its edge."""
        w = self.windows[index]
        return w.x > 0, w.y > 0, w.x + w.w < self.width, w.y + w.h < self.height


def _read_only(image: Any) -> Any:
    setflags = getattr(image, "setflags", None)
    if callable(setflags):
        setflags(write=False)
    return image


def prepare(frame: FrameRef, persons: Sequence[TrackObservation], image: Any, *,
            resize: Callable[[Any, int, int], Any], settings: WindowSettings = WindowSettings()) -> WindowImages:
    """At the offer: plan the windows of ``frame``'s people and copy each at its detector input size from ``image``, the
    decoded picture of that same frame (H x W x channels)."""
    if not frame.whole_picture:
        raise WindowStageError("not_whole_picture")
    width, height = frame.native_width, frame.native_height
    shape = getattr(image, "shape", None)
    if shape is None or tuple(shape[:2]) != (height, width):
        raise WindowStageError("image_size_mismatch")
    if any(p.frame != frame.key for p in persons):
        raise WindowStageError("persons_from_another_frame")
    plan = plan_windows([p.box for p in persons], width, height, side=settings.side,
                        head_fraction=settings.head_fraction, margin=settings.margin, max_windows=settings.max_windows)
    inputs = []
    for window in plan.windows:
        crop = image[window.y:window.y + window.h, window.x:window.x + window.w]
        w_in, h_in = window.input_size
        copied = crop.copy() if (w_in, h_in) == (window.w, window.h) else resize(crop, w_in, h_in)
        copied_shape = getattr(copied, "shape", None)
        if copied_shape is None or tuple(copied_shape[:2]) != (h_in, w_in):
            raise WindowStageError("window_copy_size")
        inputs.append(_read_only(copied))
    return WindowImages(width, height, plan.windows,
                        tuple(tuple(persons[i].key for i in w.owners) for w in plan.windows),
                        tuple(persons[i].key for i in plan.uncovered), tuple(inputs))


def cut_inside(row: Any, input_width: int, input_height: int, interior: tuple[bool, bool, bool, bool]) -> bool:
    """Whether DeepFace's cut of this detection (its integer box grown by half its width and height on each side) reads
    only the window's pixels, or black beyond the picture's own edges; ``row`` in the window's input pixels."""
    x, y, w, h = math.floor(row.x), math.floor(row.y), int(row.w), int(row.h)
    grow_x, grow_y = int(CONTEXT * w), int(CONTEXT * h)
    left, top, right, bottom = interior
    return not ((left and x - grow_x < 0) or (top and y - grow_y < 0)
                or (right and x + w + grow_x > input_width) or (bottom and y + h + grow_y > input_height))


@dataclass(frozen=True)
class Selection:
    kept: tuple[tuple[int, int], ...]  # (window, row) in window then row order
    detected: int  # rows at or above the threshold, before selection
    dropped_at_window_edge: int
    dropped_duplicate: int
    unexplained_windows: tuple[int, ...]
    not_analysed: tuple[TrackKey, ...]  # by track ID


def select(rows: Sequence[Sequence[Any]], images: WindowImages, threshold: float,
           duplicate_iou: float = DUPLICATE_IOU) -> Selection:
    """The detections to embed, from each window's rows (``x``, ``y``, ``w``, ``h``, ``score`` in its input pixels), and the
    people not analysed this tick."""
    if len(rows) != len(images.windows):
        raise WindowStageError("rows_per_window")
    whole: list[FaceBox] = []
    dropped: list[FaceBox] = []
    origin: dict[int, tuple[int, int]] = {}
    for index, (window, window_rows) in enumerate(zip(images.windows, rows, strict=True)):
        w_in, h_in = window.input_size
        interior = images.interior(index)
        for position, row in enumerate(window_rows):
            if row.score < threshold:
                continue
            box = detection_to_frame(window, index, row.x, row.y, row.w, row.h, row.score)
            if cut_inside(row, w_in, h_in, interior):
                whole.append(box)
                origin[id(box)] = (index, position)
            else:
                dropped.append(box)
    kept = suppress_duplicates(whole, duplicate_iou)
    unexplained = sorted({box.window for box in dropped
                          if any(face.window == box.window and inside_share(box, face) >= EXPLAINED_SHARE
                                 for face in whole)
                          or all(inside_share(box, face) < EXPLAINED_SHARE
                                 for face in kept if face.window != box.window)})
    withheld = set(images.uncovered) | {key for index in unexplained for key in images.owners[index]}
    return Selection(tuple(sorted(origin[id(box)] for box in kept)), len(whole) + len(dropped), len(dropped),
                     len(whole) - len(kept), tuple(unexplained), tuple(sorted(withheld, key=lambda k: k.track_id)))


def area_to_frame(window: Window, area: dict[str, Any]) -> dict[str, float]:
    """A reported face area (x, y, w, h in the window's input pixels) in the decoded picture's pixels."""
    x, y = window.to_frame(area["x"], area["y"])
    return {"x": x, "y": y, "w": area["w"] / window.scale_x, "h": area["h"] / window.scale_y}


def box_to_frame(window: Window, row: Any, width: int, height: int) -> NormalizedBox | None:
    """A row's box normalized to the decoded picture and clamped to it; None when nothing of it is inside."""
    return to_normalized(detection_to_frame(window, 0, row.x, row.y, row.w, row.h, row.score), width, height)


@dataclass(frozen=True)
class WindowNumbers:
    """One analysed frame's windows, for records: numbers and track IDs only."""

    windows: int
    inputs: tuple[tuple[int, int], ...]  # each window's detector input (width, height)
    scales: tuple[float, ...]
    people: int
    uncovered: int  # people the window budget left out
    unexplained_windows: int
    not_analysed: tuple[TrackKey, ...]
    detected: int
    dropped_at_window_edge: int
    dropped_duplicate: int
    copy_bytes: int

    def numbers(self) -> dict[str, Any]:
        return {"windows": self.windows, "inputs": [list(size) for size in self.inputs],
                "scales": [round(s, 4) for s in self.scales], "people": self.people, "uncovered": self.uncovered,
                "unexplained_windows": self.unexplained_windows,
                "not_analysed": sorted(k.track_id for k in self.not_analysed), "detected": self.detected,
                "dropped_at_window_edge": self.dropped_at_window_edge, "dropped_duplicate": self.dropped_duplicate,
                "copy_bytes": self.copy_bytes}


def window_numbers(images: WindowImages, selection: Selection) -> WindowNumbers:
    return WindowNumbers(len(images.windows), tuple(w.input_size for w in images.windows),
                         tuple(w.scale for w in images.windows),
                         sum(len(o) for o in images.owners) + len(images.uncovered), len(images.uncovered),
                         len(selection.unexplained_windows), selection.not_analysed, selection.detected,
                         selection.dropped_at_window_edge, selection.dropped_duplicate, images.nbytes)
