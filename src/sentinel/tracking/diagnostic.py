"""The tracking diagnostic's records (opt-in, ``sentinel run --face-validation --tracking-diagnostic``; session 79).

One record per processed frame, numbers and fixed labels only, so that a stationary face occlusion's track breaks can
be read from evidence instead of inferred from the 1 Hz face ticks:

- ``det``: the person detections the production path handed to its tracker on this frame (post-NMS, pre-assignment,
  at the production confidence), each with its input index, score and box; the authoritative detections;
- ``pub``: the rows the tracker published (what Sentinel receives), each with the input index it consumed and how it
  came to be published (``continued``, ``refound`` for a lost track re-activated under its own number, ``confirmed``
  for a track published for the first time);
- ``unpub``: the detections the tracker did not publish, each with why (``new_unconfirmed``: it started a track whose
  number was consumed but not yet published; ``unmatched``);
- ``lost``: the tracker's lost tracks after this frame's update, with how many updates ago each was last detected and
  its predicted box; ``removed``: the numbers it dropped this frame;
- ``assign``: the cost matrices the tracker's own association used on this frame, as it computed them before
  assigning (the ``pool`` of tracked and lost tracks against the detections, then the unconfirmed tracks against what
  was left), each with its limit, so that a failed re-attachment is read from the real cost and its competitors;
- ``shadow``: separately, what the same forward pass shows below the production confidence (a second NMS at a lower
  confidence on a clone of the raw output); never merged with ``det``, and reported with its own limits;
- ``cur``: the tracks Sentinel itself still held current after this frame (its own expiry);
- ``ns``: what the production backend took and what the diagnostic's own parts took, so the overhead is measured in
  the run, never assumed;
- ``problems`` and ``trunc``: evidence that is absent or shortened on this frame, named, so that no absence claim
  rests on omitted evidence.

Lines are bounded in size and the run in count and bytes (DiagnosticBounds); the summary line at the stop carries the
kept, dropped and truncated counts. Boxes are fractions of the frame. Nothing here is an image, a crop or an identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from ..contracts import FrameRef

TRACKING_SCHEMA = 1
PUBLISHED_OUTCOMES = ("continued", "refound", "confirmed")
UNPUBLISHED_OUTCOMES = ("new_unconfirmed", "unmatched")
STAGES = ("pool", "unconfirmed")
TRACK_STATES = ("tracked", "lost")
PROBLEMS = ("capture_missing", "capture_failed", "tracker_replaced", "raw_missing", "clone_failed", "shadow_failed",
            "layout_unknown", "snapshot_failed", "assign_failed", "no_backend_diagnostics")
TIMINGS = ("backend", "capture", "clone", "shadow", "snapshot", "record")
# The lists a line may shorten to fit its byte bound, in the order they are shortened (the shadow first, the
# authoritative detections last); each list is ordered so that its last entry is the least informative.
SHORTENED = ("shadow", "assign", "unpub", "lost", "pub", "det")


@dataclass(frozen=True)
class Detection:
    idx: int
    score: float
    box: tuple[float, float, float, float]  # pixels of the frame the backend was given


@dataclass(frozen=True)
class Published:
    track_id: int
    idx: int | None
    outcome: str  # PUBLISHED_OUTCOMES
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Unpublished:
    idx: int
    outcome: str  # UNPUBLISHED_OUTCOMES
    track_id: int | None


@dataclass(frozen=True)
class Lost:
    track_id: int
    frames_lost: int
    box: tuple[float, float, float, float]  # the tracker's prediction after this frame's update


@dataclass(frozen=True)
class Assignment:
    """One cost matrix as the tracker's association used it, with the limit the assignment applied."""

    stage: str  # STAGES
    thresh: float
    tracks: tuple[tuple[int, str, int], ...]  # (track number, state, updates since its last detection)
    dets: tuple[int, ...]  # detection indices, the matrix's columns
    cost: tuple[tuple[float, ...], ...]  # rows by track
    truncated: bool = False  # rows or columns beyond the bounds were left out


@dataclass(frozen=True)
class ShadowBox:
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackerDiagnostics:
    """What the backend read on one frame (pixels; the backend's own track numbers until the boundary offsets them)."""

    tracker_frame: int | None
    detections: tuple[Detection, ...] = ()
    published: tuple[Published, ...] = ()
    unpublished: tuple[Unpublished, ...] = ()
    lost: tuple[Lost, ...] = ()
    removed: tuple[int, ...] = ()
    assignments: tuple[Assignment, ...] = ()
    second_stage_detections: int = 0  # detections whose score fell in the tracker's low band (none at conf 0.4)
    shadow: tuple[ShadowBox, ...] = ()  # the shadow pass's boxes below the production confidence
    shadow_high: int = 0  # its boxes at or above the production confidence (compared, never used)
    shadow_agree: int = 0  # production detections with a shadow box at or above the confidence at IoU >= 0.9
    shadow_candidates: int | None = None  # anchors above the shadow confidence before its NMS
    shadow_limited: bool | None = None  # the shadow pass hit one of its own caps
    timings_ns: dict[str, int] = field(default_factory=dict)
    problems: tuple[str, ...] = ()

    def with_id_offset(self, offset: int) -> TrackerDiagnostics:
        """The boundary's published numbers: the backend's plus the offset it applies after a reset."""
        if not offset:
            return self
        return replace(
            self,
            published=tuple(replace(p, track_id=p.track_id + offset) for p in self.published),
            unpublished=tuple(replace(u, track_id=None if u.track_id is None else u.track_id + offset)
                              for u in self.unpublished),
            lost=tuple(replace(item, track_id=item.track_id + offset) for item in self.lost),
            removed=tuple(n + offset for n in self.removed),
            assignments=tuple(replace(a, tracks=tuple((n + offset, s, k) for n, s, k in a.tracks))
                              for a in self.assignments),
        )


@dataclass(frozen=True)
class DiagnosticBounds:
    """What a run may write: fixed before the run and recorded at startup."""

    shadow_conf: float = 0.1
    max_records: int = 20_000
    max_bytes: int = 32_000_000
    max_line_bytes: int = 4096
    max_det: int = 16
    max_shadow: int = 12
    max_lost: int = 8
    max_assign_tracks: int = 8
    max_assign_dets: int = 16

    def numbers(self) -> dict[str, float | int]:
        return {"shadow_conf": self.shadow_conf, "max_records": self.max_records, "max_bytes": self.max_bytes,
                "max_line_bytes": self.max_line_bytes, "max_det": self.max_det, "max_shadow": self.max_shadow,
                "max_lost": self.max_lost, "max_assign_tracks": self.max_assign_tracks,
                "max_assign_dets": self.max_assign_dets}


def _fraction(box: tuple[float, float, float, float], width: int, height: int) -> list[float]:
    return [round(box[0] / width, 3), round(box[1] / height, 3), round(box[2] / width, 3), round(box[3] / height, 3)]


def frame_record(frame: FrameRef, diagnostics: TrackerDiagnostics, current: list[int], n: int,
                 backend_ns: int | None, bounds: DiagnosticBounds) -> dict[str, Any]:
    """The frame's record, lists already within the bounds' counts (what they left out is in ``trunc``)."""
    width, height = frame.native_width, frame.native_height
    trunc: dict[str, int] = {}

    def cut(name: str, items: list[Any], limit: int) -> list[Any]:
        if len(items) > limit:
            trunc[name] = len(items) - limit
        return items[:limit]

    d = diagnostics
    det = cut("det", [{"i": x.idx, "s": round(x.score, 3), "b": _fraction(x.box, width, height)}
                      for x in sorted(d.detections, key=lambda x: (-x.score, x.idx))], bounds.max_det)
    pub = cut("pub", [{"id": p.track_id, "i": p.idx, "o": p.outcome, "b": _fraction(p.box, width, height)}
                      for p in sorted(d.published, key=lambda p: p.track_id)], bounds.max_det)
    unpub = cut("unpub", [{"i": u.idx, "o": u.outcome, "id": u.track_id} for u in d.unpublished], bounds.max_det)
    lost = cut("lost", [{"id": x.track_id, "lost": x.frames_lost, "b": _fraction(x.box, width, height)}
                        for x in sorted(d.lost, key=lambda x: x.frames_lost)], bounds.max_lost)
    assign = []
    for a in d.assignments:
        rows = a.cost[:bounds.max_assign_tracks]
        assign.append({"stage": a.stage, "thresh": round(a.thresh, 3),
                       "tracks": [{"id": n, "st": s, "lost": k} for n, s, k in a.tracks[:bounds.max_assign_tracks]],
                       "dets": list(a.dets[:bounds.max_assign_dets]),
                       "rows": [{"id": a.tracks[i][0], "c": [round(c, 3) for c in row[:bounds.max_assign_dets]]}
                                for i, row in enumerate(rows) if i < len(a.tracks)],
                       "cut": a.truncated or len(a.tracks) > bounds.max_assign_tracks
                       or len(a.dets) > bounds.max_assign_dets})
    shadow = cut("shadow", [{"s": round(x.score, 3), "b": _fraction(x.box, width, height)}
                            for x in sorted(d.shadow, key=lambda x: -x.score)], bounds.max_shadow)
    timings = {k: int(v) for k, v in d.timings_ns.items() if k in TIMINGS}
    if backend_ns is not None:
        timings["backend"] = int(backend_ns)
    return {"tracking": "frame", "schema": TRACKING_SCHEMA, "stream_epoch": frame.stream_epoch,
            "frame_seq": frame.frame_seq, "frame_mono_ns": frame.ingest_mono_ns, "n": n, "tf": d.tracker_frame,
            "det": det, "pub": pub, "unpub": unpub, "lost": lost, "removed": sorted(d.removed), "assign": assign,
            "ss": d.second_stage_detections, "shadow": shadow, "sh_hi": d.shadow_high, "sh_agree": d.shadow_agree,
            "sh_cand": d.shadow_candidates, "sh_lim": d.shadow_limited, "cur": sorted(current), "ns": timings,
            "problems": list(d.problems), "trunc": trunc}


def serialize(record: dict[str, Any]) -> str:
    return json.dumps(record, separators=(",", ":"))


def bounded_line(record: dict[str, Any], bounds: DiagnosticBounds) -> tuple[dict[str, Any], str]:
    """The record shortened, if it must be, to at most ``max_line_bytes`` when serialized: the lists in SHORTENED
    order, one entry at a time from the end, each cut counted in ``trunc``; ``assign`` is dropped whole. Returns the
    (possibly shortened) record and its serialization."""
    line = serialize(record)
    for name in SHORTENED:
        while len(line.encode()) > bounds.max_line_bytes and record[name]:
            if name == "assign":
                record["trunc"][name] = record["trunc"].get(name, 0) + len(record[name])
                record[name] = []
            else:
                record[name] = record[name][:-1]
                record["trunc"][name] = record["trunc"].get(name, 0) + 1
            line = serialize(record)
        if len(line.encode()) <= bounds.max_line_bytes:
            break
    return record, line


def summary_record(kept: int, dropped: int, bytes_kept: int, truncated_lines: int, max_line_bytes_seen: int,
                   processed: int, problems: dict[str, int], bounds: DiagnosticBounds) -> dict[str, Any]:
    """The line at the stop: what was kept, dropped and shortened, against the bounds, and the processed frames."""
    return {"tracking": "summary", "schema": TRACKING_SCHEMA, "kept": kept, "dropped": dropped, "bytes": bytes_kept,
            "truncated_lines": truncated_lines, "max_line_bytes_seen": max_line_bytes_seen, "processed": processed,
            "problem_counts": {k: int(v) for k, v in sorted(problems.items()) if k in PROBLEMS},
            "limits": {"max_records": bounds.max_records, "max_bytes": bounds.max_bytes,
                       "max_line_bytes": bounds.max_line_bytes}}
