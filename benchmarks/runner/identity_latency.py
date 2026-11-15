#!/usr/bin/env python3
"""Where the time went before a track became KNOWN: an offline reading of one run's identity records
(session 116, part 28). Measurement only: it decides nothing and changes nothing.

    identity_latency.py RUN.jsonl [--summary]

For every acquisition (a "transition" record to ``known`` from another state) it splits the time into phases taken
from the records' own timestamps:

``tracking``          the track life's first measurement -> the first face tick with the track (needs "life" records,
                      written by --identity-retention runs; ticks are about 1 Hz, so about a second's resolution)
``eligible_face``     that tick -> the frame of the first result on which a face was assigned to the track
``not_matching``      that frame -> the frame of the first vote of the confirming run (the last N consecutive matches
                      that made it KNOWN; N from the transition's own reason)
``confirmation``      first confirming vote's frame -> last confirming vote's frame
``queue_and_apply``   for that last vote: applied - frame - processing. It is the frame's wait before the face worker
                      (the detector's own time on that frame included) plus the finished result's wait for the loop;
                      the records cannot separate those
``inference``         that result's processing_ms (the face worker's time on it)
``apply_to_state``    the result's application -> the transition
``status_publish``    the transition -> the first status the runtime published after it  ("page_read" records only)
``page_read``         that publication -> the first page request that read it (the page asks every 2 s; this is the
                      request reading the status on the device, not a browser drawing it)

The first seven add up to (transition - life start); the reading checks that and reports any row where they do not.
A track that was KNOWN earlier on the same life and lost it is a **reacquisition**: timed from the loss
(``wait_for_match``: the loss -> the first confirming vote's frame; negative when that frame had already been taken),
without the first three phases.

Lives, never track numbers: with "life" records a row belongs to the life the records name. Without them (older runs)
the reading groups a track number's records into **segments**, split where the number was cleared and where its ticks
stop for more than ``SEGMENT_GAP_S``; a segment is not a proven continuous life (it may join lives or people), every
such row says ``segment_number_and_gaps``, and ``tracking`` is unavailable.

What it does not say: a long phase says where the time was spent, never why. ``not_matching`` is no evidence of
distance, pose or occlusion; nothing here is a target or a threshold; one run validates nothing. It prints numbers and
fixed labels only: never an identity ID, a similarity value or anything per face. Identity IDs are compared for
equality inside the reading and never leave it.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

FORMAT = 1
SEGMENT_GAP_S = 5.0  # no "life" records: a track number's ticks further apart than this begin a new segment
SEGMENT_GAP_NS = round(SEGMENT_GAP_S * 1e9)
NS_PER_MS = 1_000_000
KNOWN, CLEARED = "known", "cleared"
NOT_COUNTED = ("before_expiry", "before_break")  # results the raw rule did not count for the track's current life
VOTES = ("match", "unknown")
FIRST, REACQUIRED = "first_on_life", "reacquired"
LIFE_RECORDS, SEGMENTS = "life_records", "segment_number_and_gaps"
CHAIN = ("tracking", "eligible_face", "not_matching", "confirmation", "queue_and_apply", "inference",
         "apply_to_state")
REACQUIRED_CHAIN = ("wait_for_match", "confirmation", "queue_and_apply", "inference", "apply_to_state")
UI = ("status_publish", "page_read")
NOTES = (
    "Descriptive only: where the time before KNOWN was spent according to this run's own records, never why.",
    "One run validates nothing, and no phase is a target, a threshold or a cause.",
    "not_matching is not evidence of distance, pose or occlusion: the records do not say why a face did not match.",
    "Face ticks are about 1 Hz: tracking and eligible_face have about one second of resolution.",
    "queue_and_apply joins the frame's wait before the face worker (the detector's time on it included) and the "
    "finished result's wait for the loop; the records cannot separate them.",
    "page_read is the page's request reading the status on the device, not a browser drawing it.",
    "A segment (runs without life records) is a track number between clears and tick gaps, not a proven continuous "
    "life.",
)


def _ms(ns: float | None) -> float | None:
    return None if ns is None else round(ns / NS_PER_MS, 1)


def distribution(values: Iterable[float | None]) -> dict[str, Any]:
    """n, min, median, p90 and max of what is available (ms), and how many were unavailable."""
    values = list(values)
    have = sorted(v for v in values if v is not None)
    out: dict[str, Any] = {"n": len(have), "n_unavailable": len(values) - len(have)}
    if have:
        def at(q: float) -> float:
            return have[min(len(have) - 1, round(q * (len(have) - 1)))]
        out.update(min=have[0], median=at(0.5), p90=at(0.9), max=have[-1])
    return out


class _Run:
    """One run's identity records by track, in file order (the order the runtime wrote them)."""

    def __init__(self) -> None:
        self.kinds: Counter[str] = Counter()
        self.ticks: dict[tuple[int, int], list[tuple[int, int]]] = {}  # (line, frame_ns) per (epoch, track)
        self.tick_labels: Counter[str] = Counter()
        self.offered: list[int] = []  # offered ticks' frame times
        self.entries: dict[tuple[int, int], list[dict[str, Any]]] = {}  # one track's part of each result
        self.outcomes: Counter[str] = Counter()
        self.result_times: list[tuple[float, float]] = []  # applied results: (queue_and_apply ns, inference ns)
        self.transitions: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self.lives: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self.associations: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self.page_reads: dict[tuple[int, int, str, str, int], tuple[int, int]] = {}
        self.page_read_kinds: Counter[str] = Counter()
        self.status_lines = 0
        self.unreadable = 0

    def add(self, line: int, record: Any, status_utc: str | None) -> None:
        kind = record.get("identity")
        if not isinstance(kind, str):
            return
        self.kinds[kind] += 1
        epoch = record.get("stream_epoch")
        if kind == "tick":
            self.tick_labels[str(record.get("tick"))] += 1
            if record.get("tick") == "offered":
                self.offered.append(record["frame_mono_ns"])
            for track in record.get("tracks") or ():
                self.ticks.setdefault((epoch, track), []).append((line, record["frame_mono_ns"]))
        elif kind == "result":
            self.outcomes[str(record.get("outcome"))] += 1
            applied = record.get("outcome") == "applied"
            processing = record.get("processing_ms")
            processing_ns = None if processing is None else processing * NS_PER_MS
            if applied and processing_ns is not None:
                self.result_times.append((record["applied_mono_ns"] - record["frame_mono_ns"] - processing_ns,
                                          processing_ns))
            if not applied:
                return
            for person in record.get("persons") or ():
                self.entries.setdefault((epoch, person["track_id"]), []).append({
                    "line": line, "frame": record["frame_mono_ns"], "applied": record["applied_mono_ns"],
                    "processing_ns": processing_ns, "ownership": person.get("ownership"), "vote": person.get("vote"),
                    "label": person.get("label"), "who": person.get("identity_id")})  # "who": equality only
        elif kind == "transition":
            self.transitions.setdefault((epoch, record["track_id"]), []).append(
                {"line": line, "from": record.get("from"), "to": record.get("to"), "mono": record["mono_ns"],
                 "reason": record.get("reason") or "", "status_utc": status_utc})
        elif kind == "life":
            self.lives.setdefault((epoch, record["track_id"]), []).append(
                {"line": line, "event": record.get("event"), "since": record["since_mono_ns"]})
        elif kind == "association":
            self.associations.setdefault((epoch, record["track_id"]), []).append(
                {"line": line, "to": record.get("to")})
        elif kind == "page_read":
            self.page_read_kinds[f"{record.get('watched')}:{record.get('to')}"] += 1
            self.page_reads[(epoch, record["track_id"], record.get("watched"), record.get("to"),
                             record["changed_mono_ns"])] = (record["published_mono_ns"], record["read_mono_ns"])


def _confirmations(reason: str) -> int | None:
    """N from a transition's own reason ("N consistent matches"); None when it says something else."""
    head, _, tail = reason.partition(" ")
    return int(head) if head.isdigit() and tail == "consistent matches" else None


def _start(run: _Run, key: tuple[int, int], transition: dict[str, Any]) -> dict[str, Any]:
    """The life or segment the transition belongs to: its source, its start line, the life's first measurement (life
    records only) and its first tick."""
    line, mono = transition["line"], transition["mono"]
    ticks = [(at_line, frame) for at_line, frame in run.ticks.get(key, ()) if at_line < line and frame <= mono]
    if run.kinds["life"]:
        began = [life for life in run.lives.get(key, ()) if life["event"] == "began" and life["line"] < line]
        if not began:
            return {"source": LIFE_RECORDS, "problem": "life_record_missing"}
        life = began[-1]
        first = next((frame for at_line, frame in ticks if frame >= life["since"]), None)
        return {"source": LIFE_RECORDS, "line": life["line"], "since": life["since"], "first_tick": first}
    cleared = [t for t in run.transitions.get(key, ()) if t["to"] == CLEARED and t["line"] < line]
    floor_line = cleared[-1]["line"] if cleared else 0
    chain = [(at_line, frame) for at_line, frame in ticks if at_line > floor_line]
    start = len(chain) - 1
    while start > 0 and chain[start][1] - chain[start - 1][1] <= SEGMENT_GAP_NS:
        start -= 1
    if not chain:
        return {"source": SEGMENTS, "line": floor_line, "since": None, "first_tick": None}
    return {"source": SEGMENTS, "line": max(floor_line, chain[start][0] - 1), "since": None,
            "first_tick": chain[start][1]}


def _labels(entries: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(e["label"]) for e in entries).items()))


def _acquisition(run: _Run, key: tuple[int, int], transition: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"stream_epoch": key[0], "track_id": key[1], "line": transition["line"],
                           "after_status_utc": transition["status_utc"]}
    start = _start(run, key, transition)
    row["life_source"] = start["source"]
    if "problem" in start:
        return {**row, "incomplete": start["problem"]}
    needed = _confirmations(transition["reason"])
    if needed is None:
        return {**row, "incomplete": "reason_without_a_count"}
    since_line = start["line"]
    floor = start["since"] if start["since"] is not None else start["first_tick"]
    entries = [e for e in run.entries.get(key, ())
               if since_line < e["line"] < transition["line"] and e["label"] not in NOT_COUNTED
               and (floor is None or e["frame"] >= floor)]
    votes = [e for e in entries if e["vote"] in VOTES]
    run_votes = votes[-needed:]
    if (len(run_votes) < needed or any(e["vote"] != "match" for e in run_votes)
            or len({e["who"] for e in run_votes}) != 1 or run_votes[0]["who"] is None):
        return {**row, "incomplete": "confirming_votes_not_found"}
    first, last = run_votes[0], run_votes[-1]
    if last["processing_ns"] is None:
        return {**row, "incomplete": "result_without_processing_time"}
    earlier = [t for t in run.transitions.get(key, ()) if since_line < t["line"] < transition["line"]]
    losses = [t for t in earlier if t["from"] == KNOWN and t["to"] != KNOWN]
    tail = {"confirmation": last["frame"] - first["frame"],
            "queue_and_apply": last["applied"] - last["frame"] - last["processing_ns"],
            "inference": last["processing_ns"], "apply_to_state": transition["mono"] - last["applied"]}
    if any(t["to"] == KNOWN for t in earlier) and losses:
        loss = losses[-1]
        phases = {"wait_for_match": first["frame"] - loss["mono"], **tail}
        total, names = transition["mono"] - loss["mono"], REACQUIRED_CHAIN
        row.update(kind=REACQUIRED, since_loss_ms=_ms(total),
                   results_since_loss=_labels(e for e in entries if loss["line"] < e["line"] < first["line"]))
        ordered = (first["frame"], last["frame"], last["applied"], transition["mono"])
    else:
        assigned = next((e for e in entries if e["ownership"] == "assigned"), first)
        tick = start["first_tick"]
        if tick is None:
            return {**row, "incomplete": "no_tick_before_the_first_result"}
        tracking = None if start["since"] is None else tick - start["since"]
        phases = {"tracking": tracking, "eligible_face": assigned["frame"] - tick,
                  "not_matching": first["frame"] - assigned["frame"], **tail}
        total = transition["mono"] - (start["since"] if start["since"] is not None else tick)
        names = CHAIN
        row.update(kind=FIRST, since_start_ms=_ms(total),
                   results_before_a_face=_labels(e for e in entries if e["line"] < assigned["line"]),
                   results_before_confirming=_labels(
                       e for e in entries if assigned["line"] <= e["line"] < first["line"]))
        if tracking is None:
            row["tracking_unavailable"] = "no_life_records"
        ordered = (*(() if start["since"] is None else (start["since"],)), tick, assigned["frame"], first["frame"],
                   last["frame"], last["applied"], transition["mono"])
    row["phases_ms"] = {name: _ms(phases[name]) for name in names}
    have = [phases[name] for name in names if phases[name] is not None]
    # in time order (a reacquisition's loss is left out: it may follow the first confirming vote's frame)
    row["ordered"] = all(a <= b for a, b in zip(ordered, ordered[1:]))
    row["adds_up"] = abs(sum(have) - total) < 1 and row["ordered"]
    read = run.page_reads.get((key[0], key[1], "identity", KNOWN, transition["mono"]))
    if read is None:
        row["ui_ms"] = None
        row["ui_unavailable"] = "not_read" if run.kinds["page_read"] else "no_page_read_records"
    else:
        row["ui_ms"] = {"status_publish": _ms(read[0] - transition["mono"]), "page_read": _ms(read[1] - read[0])}
    if run.kinds["association"]:  # --identity-retention runs: what the page's association said just before
        before = [a for a in run.associations.get(key, ()) if since_line < a["line"] < transition["line"]]
        row["association_before"] = before[-1]["to"] if before else "none"
    return row


def analyse(lines: Iterable[str]) -> dict[str, Any]:
    """The whole reading of one run's lines (see the module docstring): numbers and fixed labels only."""
    run = _Run()
    status_utc: str | None = None
    for number, text in enumerate(lines, 1):
        try:
            record = json.loads(text)
        except ValueError:
            run.unreadable += 1
            continue
        if not isinstance(record, dict):
            continue
        if isinstance(record.get("updated_utc"), str) and "identity" not in record:
            status_utc, run.status_lines = record["updated_utc"], run.status_lines + 1
            continue
        try:
            run.add(number, record, status_utc)
        except (KeyError, TypeError):
            run.unreadable += 1
    rows = []
    for key in sorted(run.transitions):
        for transition in run.transitions[key]:
            if transition["to"] == KNOWN and transition["from"] != KNOWN:
                rows.append(_acquisition(run, key, transition))
    rows.sort(key=lambda row: row["line"])
    complete = [row for row in rows if "incomplete" not in row]
    summary: dict[str, Any] = {}
    for kind, names in ((FIRST, CHAIN), (REACQUIRED, REACQUIRED_CHAIN)):
        of_kind = [row for row in complete if row["kind"] == kind and row["adds_up"]]
        total = "since_start_ms" if kind == FIRST else "since_loss_ms"
        summary[kind] = {
            "rows": len(of_kind), "total_ms": distribution(row[total] for row in of_kind),
            "phases_ms": {name: distribution(row["phases_ms"][name] for row in of_kind) for name in names},
            "ui_ms": {name: distribution(None if row["ui_ms"] is None else row["ui_ms"][name] for row in of_kind)
                      for name in UI}}
    intervals = [b - a for a, b in zip(run.offered, run.offered[1:])]
    return {
        "tool": "identity_latency", "format": FORMAT, "notes": list(NOTES),
        "run": {"records": dict(sorted(run.kinds.items())), "status_lines": run.status_lines,
                "unreadable_lines": run.unreadable,
                "life_source": LIFE_RECORDS if run.kinds["life"] else SEGMENTS, "segment_gap_s": SEGMENT_GAP_S,
                "page_read_records": dict(sorted(run.page_read_kinds.items()))},
        "results": {"outcomes": dict(sorted(run.outcomes.items())), "ticks": dict(sorted(run.tick_labels.items())),
                    "queue_and_apply_ms": distribution(_ms(q) for q, _ in run.result_times),
                    "inference_ms": distribution(_ms(p) for _, p in run.result_times),
                    "offered_tick_interval_ms": distribution(_ms(i) for i in intervals)},
        "acquisitions": {"total": len(rows), "complete": len(complete),
                         "incomplete": dict(sorted(Counter(row["incomplete"] for row in rows
                                                           if "incomplete" in row).items())),
                         "not_adding_up": sum(1 for row in complete if not row["adds_up"])},
        "summary": summary,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Where the time went before a track became KNOWN (one run).")
    parser.add_argument("run", type=Path, help="the run's records (run.jsonl)")
    parser.add_argument("--summary", action="store_true", help="leave the per-acquisition rows out")
    args = parser.parse_args(argv)
    try:
        with args.run.open(encoding="utf-8", errors="replace") as lines:
            document = analyse(lines)
    except OSError as exc:
        print(f"identity_latency: unreadable:{type(exc).__name__}", file=sys.stderr)
        return 2
    if args.summary:
        del document["rows"]
    print(json.dumps(document, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
