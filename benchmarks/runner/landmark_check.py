"""The landmark diagnostic's reading (session 90; ``face_check.py landmarks <session>/lv --f1 ...``).

A diagnostic session runs the normal recognition path (production detection, the original gallery, the plain identity
rules, the scene model loaded) with ``--face-landmark-diagnostic``: D86's native-scale pass keeps YuNet's five
landmarks, and each person on an applied result carries a landmark record (``sentinel.identity.landmarks``: an explicit
association status and, when measured, derived landmark numbers). The enrolled person works at the Mac as usual for
ten minutes, looking at the Mac, turning the head and moving the chair as they normally would; nothing asks them to face
the camera. This reading judges execution, the diagnostic's configuration and records, and privacy. Everything about
the landmark numbers and the votes is DESCRIPTIVE. It is not an F2 reading (the startup marker makes ``face_check f2``
refuse such a run, and this reading refuses a run without it), it validates nothing and it changes nothing.

The analysis, predeclared (session 90; nothing here is chosen after a run):

- **Observations.** Votes (match or unknown) on applied results that list exactly one person, whose frames lie in the
  window W = WINDOW_S after T0 (the runtime's readiness). Results with more than one person are counted apart and not
  analysed. A vote's landmark observation is *usable* when its record is ``measured`` and the native candidate's box
  overlaps the production face it voted with by at least MIN_FACE_IOU (intersection over union); every other vote is
  counted under its reason (each landmark status and invalid reason, and ``face_mismatch`` below MIN_FACE_IOU).
- **Time blocks.** W is cut into BLOCKS blocks of BLOCK_S seconds; the even-numbered blocks (0, 2, ...) are the
  *reference* blocks and the odd-numbered ones the *evaluation* blocks; the first BUFFER_S seconds of every block belong
  to neither, so the votes nearest a block boundary, the most strongly tied to the other set's, are left out.
- **The reference band.** Built from the reference blocks' usable *matching* votes only, on the two PRIMARY features
  (nose_lateral and nose_position): for each, the central interval between its BAND_TAIL and 1 - BAND_TAIL quantiles
  (linear interpolation between order statistics, the type 7 definition); the joint band is the box where both lie
  inside their intervals (inclusive). With two features at 5 % per tail the box holds at least 80 % of those matching
  votes by construction (more when the features are correlated); the realised share is reported. The band is the range
  in which this person's faces matched in the reference blocks, nothing more: it is not called frontal, it is not a
  pose range, and the proxies are not angles.
- **Evaluation.** In the evaluation blocks, each usable vote is inside or outside the band. Reported: the counts, the
  coverage (usable / votes), the miss rate inside and outside, the risk difference (outside - inside) and the risk
  ratio, the miss *episodes* (consecutive unknown votes less than EPISODE_GAP_S apart with no match between them; each
  placed by its first usable vote) inside and outside, and for every feature (primary and SECONDARY) two rank effect
  sizes on the usable votes: the AUC of the distance from the reference matches' median (misses farther than matches:
  above 0.5) and the AUC of the signed value (misses above matches: above 0.5).
- **Temporal clustering.** Adjacent frames are not independent: the intervals come from a cluster bootstrap over the
  evaluation blocks' SEGMENT_S-second segments (five per block after the buffer; BOOTSTRAP_REPLICATES replicates, seed
  BOOTSTRAP_SEED, the central CI_LEVEL percentile interval), and the episode counts stand beside the vote counts.
- **Inconclusive, never forced.** The outcome is *inconclusive* when the account's preconditions are not met
  (``other_faces_in_view`` must be ``no`` and ``stayed_seated`` ``yes``), when the data are insufficient (fewer than
  MIN_REFERENCE_MATCHES usable reference matches, fewer than MIN_EVALUATION usable evaluation votes, fewer than MIN_SIDE
  inside or outside the band, fewer than MIN_EPISODES placed miss episodes, or a coverage under MIN_COVERAGE in either
  set), when the band does not carry over (its share of the evaluation matches differs from its share of the reference
  matches by more than MAX_BAND_SHIFT), when the bootstrap is unstable (more than MAX_UNDEFINED_SHARE of its replicates
  leave one side empty) or imprecise (the risk difference's interval wider than MAX_CI_WIDTH). Collection is never
  extended or repeated to reach these numbers; a short session is reported as it is.
- **Wording.** Otherwise the outcome says only whether non-matching votes were more frequent, less frequent, or not
  detectably different outside the band than inside it in the evaluation blocks, with the interval. An association does
  not establish that head pose caused a miss: the proxies also move with expression, occlusion and landmark error, and
  presentation can share a cause with a miss (movement, light, distance). No association at this precision does not
  show presentation is unrelated. Neither result justifies suppressing, discounting or retaining any vote; any rule
  proposal is a separate decision that needs non-enrolled evidence.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import face_account
import step5_check as s5
from face_check import (
    CUE_TOLERANCE_S,
    F2_FACE,
    LANDMARK_SCHEMA,
    MAX_IDENTITY_RECORDS,
    MISSING,
    Reading,
    _account,
    _counts_items,
    _key_values,
    _startup_keys,
    f1_gate,
    read_records,
)
from tracking_check import CONDITIONS, NOT_IN_THIS_RUN, _workload_items

NS = 1_000_000_000
# The pinned marker (sentinel.identity.landmarks.marker(); test-tied) and the session's shape (the package's l-V).
PINNED_MARKER = {"version": 1, "score": 0.9, "min_eye_spacing_px": 4.0, "persons_max": 4}
DURATION_S = 630.0
CUES = ("t0", "stop")
ACCOUNT_FORMAT = 10
# The predeclared analysis (the module's text)
WINDOW_S = (15.0, 615.0)
BLOCK_S = 60.0
BLOCKS = 10
BUFFER_S = 5.0
SEGMENT_S = 11.0
PRIMARY = ("nose_lateral", "nose_position")
SECONDARY = ("nose_vertical", "mouth_vertical", "mouth_lateral", "eye_line_deg", "eye_spacing_ratio")
BAND_COVERAGE = 0.80
BAND_TAIL = 0.05  # (1 - BAND_COVERAGE) / len(PRIMARY) / 2, exactly: each feature's quantiles 0.05 and 0.95
MIN_FACE_IOU = 0.5
EPISODE_GAP_S = 3.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261019
CI_LEVEL = 0.95
MIN_REFERENCE_MATCHES = 60
MIN_EVALUATION = 100
MIN_SIDE = 20
MIN_EPISODES = 8
MIN_COVERAGE = 0.70
MAX_BAND_SHIFT = 0.15
MAX_UNDEFINED_SHARE = 0.05
MAX_CI_WIDTH = 0.40
SCOPE = ("scope: one ordinary-use session at the Mac (looking at the Mac, head turns and chair movement allowed; no "
         "instruction to face the camera) on the normal recognition path with the landmark diagnostic; the landmark "
         "numbers are proxies (shape descriptors of five regressed points), never head angles; the reference band is "
         "where this person's faces matched in the reference blocks, never called frontal; everything below is "
         "descriptive; an association is not a cause and justifies no change to any vote; never an F2 reading")
CAUSATION = ("an association does not establish that head pose caused a miss (the proxies also move with expression, "
             "occlusion and landmark error, and presentation can share a cause with a miss); no association at this "
             "precision does not show presentation is unrelated; neither justifies suppressing, discounting or "
             "retaining any vote")


# ---------------------------------------------------------------- statistics (stdlib; deterministic)


def quantile(values: list[float], q: float) -> float:
    """The q quantile by linear interpolation between order statistics (type 7; NumPy's default)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def auc(positive: list[float], negative: list[float]) -> float | None:
    """P(a positive value > a negative value), ties counting half; None when either side is empty."""
    if not positive or not negative:
        return None
    ordered = sorted(negative)
    total = 0.0
    for value in positive:
        below = _count_below(ordered, value)
        equal = _count_below(ordered, value, inclusive=True) - below
        total += below + 0.5 * equal
    return total / (len(positive) * len(negative))


def _count_below(ordered: list[float], value: float, inclusive: bool = False) -> int:
    low, high = 0, len(ordered)
    while low < high:
        middle = (low + high) // 2
        if ordered[middle] < value or (inclusive and ordered[middle] == value):
            low = middle + 1
        else:
            high = middle
    return low


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def spread(values: list[float], digits: int = 3) -> dict[str, float] | None:
    """n, p50, p95 and max (type 7 quantiles), rounded to ``digits``; None without values."""
    if not values:
        return None
    return {"n": len(values), "p50": round(quantile(values, 0.5), digits), "p95": round(quantile(values, 0.95), digits),
            "max": round(max(values), digits)}


# ---------------------------------------------------------------- observations


def observations(records: dict[str, list[dict]], t0_ns: int) -> tuple[list[dict], dict[str, int]]:
    """Every person on a one-person applied result in W (votes and no-face results), in frame order, with its set
    (reference, evaluation or buffer), block, segment and landmark record; and the count of the votes left out on
    results with more than one person."""
    out: list[dict] = []
    left_out: Counter[str] = Counter()
    for result in sorted(records.get("result", []), key=lambda x: x["frame_mono_ns"]):
        if result.get("outcome") != "applied":
            continue
        t = (result["frame_mono_ns"] - t0_ns) / NS
        if not WINDOW_S[0] <= t < WINDOW_S[1]:
            continue
        persons = result["persons"]
        if len(persons) != 1:
            left_out["votes_on_results_with_more_than_one_person"] += sum(
                1 for p in persons if p["vote"] in ("match", "unknown"))
            continue
        person = persons[0]
        block = min(int((t - WINDOW_S[0]) // BLOCK_S), BLOCKS - 1)
        offset = t - WINDOW_S[0] - block * BLOCK_S
        part = "buffer" if offset < BUFFER_S else "reference" if block % 2 == 0 else "evaluation"
        segment = None if part == "buffer" else (block, min(int((offset - BUFFER_S) // SEGMENT_S), 4))
        record = person.get("landmarks") or {}
        out.append({"t": t, "track": person["track_id"], "vote": person["vote"], "set": part, "block": block,
                    "segment": segment, "record": record, "category": category(record, person["face"] is not None),
                    "native_only": person["vote"] == "none" and person["face"] is None})
    return out, dict(left_out)


def category(record: dict, production_face: bool) -> str:
    """``usable``, or why not: a landmark status (with an invalid record's reason) or ``face_mismatch``."""
    status = record.get("status")
    if status == "measured":
        if not production_face:
            return "measured_without_production_face"
        overlap = record.get("face_iou")
        return "usable" if isinstance(overlap, (int, float)) and overlap >= MIN_FACE_IOU else "face_mismatch"
    if status == "invalid":
        return f"invalid:{record.get('reason')}"
    return str(status)


def _votes(items: list[dict], part: str) -> list[dict]:
    return [o for o in items if o["set"] == part and o["vote"] in ("match", "unknown")]


def band(reference: list[dict]) -> dict[str, tuple[float, float]] | None:
    """The joint reference band from the reference blocks' usable matching votes (the module's construction)."""
    matches = [o for o in reference if o["vote"] == "match" and o["category"] == "usable"]
    if not matches:
        return None
    return {f: (quantile([o["record"][f] for o in matches], BAND_TAIL),
                quantile([o["record"][f] for o in matches], 1.0 - BAND_TAIL)) for f in PRIMARY}


def inside(observation: dict, limits: dict[str, tuple[float, float]]) -> bool:
    return all(lo <= observation["record"][f] <= hi for f, (lo, hi) in limits.items())


def episodes(votes: list[dict], limits: dict[str, tuple[float, float]] | None) -> Counter[str]:
    """Miss episodes among ``votes`` (time order, one track at a time): consecutive unknown votes less than
    EPISODE_GAP_S apart with no match between them; each placed inside or outside by its first usable vote, or
    ``unplaced`` when none of its votes is usable."""
    placed: Counter[str] = Counter()
    current: list[dict] = []

    def close() -> None:
        if current:
            first = next((o for o in current if o["category"] == "usable"), None)
            placed["unplaced" if first is None or limits is None
                   else "inside" if inside(first, limits) else "outside"] += 1
            current.clear()

    for vote in sorted(votes, key=lambda o: (o["track"], o["t"])):
        if vote["vote"] == "unknown" and current and current[-1]["track"] == vote["track"] \
                and vote["t"] - current[-1]["t"] < EPISODE_GAP_S:
            current.append(vote)
        else:
            close()
            if vote["vote"] == "unknown":
                current.append(vote)
    close()
    return placed


def split(usable: list[dict], limits: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """Counts and miss rates inside and outside the band, the risk difference and ratio."""
    sides = {"inside": [o for o in usable if inside(o, limits)], "outside": [o for o in usable if not inside(o, limits)]}
    out: dict[str, Any] = {}
    for name, items in sides.items():
        misses = sum(1 for o in items if o["vote"] == "unknown")
        out[name] = {"votes": len(items), "misses": misses, "rate": misses / len(items) if items else None}
    rate_in, rate_out = out["inside"]["rate"], out["outside"]["rate"]
    out["risk_difference"] = None if rate_in is None or rate_out is None else rate_out - rate_in
    out["risk_ratio"] = None if not rate_in or rate_out is None else rate_out / rate_in
    return out


def bootstrap(usable: list[dict], limits: dict[str, tuple[float, float]], medians: dict[str, float]) -> dict[str, Any]:
    """The cluster bootstrap over the evaluation segments: the risk difference's and the primary distance AUCs'
    intervals, and the share of replicates with one side empty."""
    units: dict[Any, list[dict]] = {}
    for o in usable:
        units.setdefault(o["segment"], []).append(o)
    keys = sorted(units)
    generator = random.Random(BOOTSTRAP_SEED)
    differences: list[float] = []
    distance: dict[str, list[float]] = {f: [] for f in PRIMARY}
    undefined = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [o for _ in keys for o in units[keys[generator.randrange(len(keys))]]]
        result = split(sample, limits)
        if result["risk_difference"] is None:
            undefined += 1
        else:
            differences.append(result["risk_difference"])
        for f in PRIMARY:
            value = _distance_auc(sample, f, medians[f])
            if value is not None:
                distance[f].append(value)
    tail = (1.0 - CI_LEVEL) / 2

    def interval(values: list[float]) -> list[float] | None:
        return None if not values else [round(quantile(values, tail), 3), round(quantile(values, 1 - tail), 3)]

    return {"units": len(keys), "replicates": BOOTSTRAP_REPLICATES, "undefined_share": undefined / BOOTSTRAP_REPLICATES,
            "risk_difference_interval": interval(differences),
            **{f"{f}_distance_auc_interval": interval(distance[f]) for f in PRIMARY}}


def _distance_auc(items: list[dict], feature: str, median: float) -> float | None:
    misses = [abs(o["record"][feature] - median) for o in items if o["vote"] == "unknown"]
    matches = [abs(o["record"][feature] - median) for o in items if o["vote"] == "match"]
    return auc(misses, matches)


def effect_sizes(usable: list[dict], medians: dict[str, float]) -> dict[str, Any]:
    """Per feature on the evaluation's usable votes: the distance AUC (from the reference matches' median) and the
    signed AUC; the misses' and matches' counts."""
    out = {}
    for f in (*PRIMARY, *SECONDARY):
        misses = [o["record"][f] for o in usable if o["vote"] == "unknown"]
        matches = [o["record"][f] for o in usable if o["vote"] == "match"]
        out[f] = {"distance_auc": _round(_distance_auc(usable, f, medians[f])), "signed_auc": _round(auc(misses, matches))}
    return out


def analyse(items: list[dict], account: dict[str, str] | None, executed: bool) -> dict[str, Any]:
    """The predeclared analysis on ``observations``' output: every figure, and the outcome with its reasons."""
    reference, evaluation = _votes(items, "reference"), _votes(items, "evaluation")
    ref_usable = [o for o in reference if o["category"] == "usable"]
    eval_usable = [o for o in evaluation if o["category"] == "usable"]
    ref_matches = [o for o in ref_usable if o["vote"] == "match"]
    limits = band(reference)
    figures: dict[str, Any] = {
        "votes": {part: {"votes": len(v), "misses": sum(1 for o in v if o["vote"] == "unknown"),
                         "usable": sum(1 for o in v if o["category"] == "usable")}
                  for part, v in (("reference", reference), ("evaluation", evaluation),
                                  ("buffer", _votes(items, "buffer")))},
        "coverage": {part: _round(sum(1 for o in v if o["category"] == "usable") / len(v)) if v else None
                     for part, v in (("reference", reference), ("evaluation", evaluation))},
        "not_usable": {part: dict(sorted(Counter(o["category"] for o in v if o["category"] != "usable").items()))
                       for part, v in (("reference", reference), ("evaluation", evaluation))},
        "band": None if limits is None else {f: [round(lo, 3), round(hi, 3)] for f, (lo, hi) in limits.items()},
    }
    reasons: list[str] = []
    if account is None or account.get("other_faces_in_view") != "no" or account.get("stayed_seated") != "yes":
        reasons.append("precondition: the account must answer other_faces_in_view=no and stayed_seated=yes "
                       f"(it answers {None if account is None else account.get('other_faces_in_view')} and "
                       f"{None if account is None else account.get('stayed_seated')})")
    if len(ref_matches) < MIN_REFERENCE_MATCHES:
        reasons.append(f"insufficient: {len(ref_matches)} usable reference matches (at least {MIN_REFERENCE_MATCHES})")
    if len(eval_usable) < MIN_EVALUATION:
        reasons.append(f"insufficient: {len(eval_usable)} usable evaluation votes (at least {MIN_EVALUATION})")
    for part, coverage in figures["coverage"].items():
        if coverage is None or coverage < MIN_COVERAGE:
            reasons.append(f"insufficient: {part} coverage {coverage} (at least {MIN_COVERAGE})")
    if limits is not None and ref_matches:
        medians = {f: statistics.median(o["record"][f] for o in ref_matches) for f in (*PRIMARY, *SECONDARY)}
        result = split(eval_usable, limits)
        ref_share = sum(1 for o in ref_matches if inside(o, limits)) / len(ref_matches)
        eval_matches = [o for o in eval_usable if o["vote"] == "match"]
        eval_share = sum(1 for o in eval_matches if inside(o, limits)) / len(eval_matches) if eval_matches else None
        placed = episodes(evaluation, limits)
        figures.update({
            "band_share_of_matches": {"reference": _round(ref_share), "evaluation": _round(eval_share)},
            "inside": {k: _round(v) if k == "rate" else v for k, v in result["inside"].items()},
            "outside": {k: _round(v) if k == "rate" else v for k, v in result["outside"].items()},
            "risk_difference": _round(result["risk_difference"]), "risk_ratio": _round(result["risk_ratio"]),
            "episodes": dict(sorted(placed.items())),
            "effect_sizes": effect_sizes(eval_usable, medians),
        })
        for side in ("inside", "outside"):
            if result[side]["votes"] < MIN_SIDE:
                reasons.append(f"insufficient: {result[side]['votes']} usable evaluation votes {side} the band "
                               f"(at least {MIN_SIDE})")
        if placed["inside"] + placed["outside"] < MIN_EPISODES:
            reasons.append(f"insufficient: {placed['inside'] + placed['outside']} placed miss episodes in the "
                           f"evaluation blocks (at least {MIN_EPISODES})")
        if eval_share is None or abs(eval_share - ref_share) > MAX_BAND_SHIFT:
            reasons.append(f"unstable: the band holds {_round(ref_share)} of the reference matches and "
                           f"{_round(eval_share)} of the evaluation matches (at most {MAX_BAND_SHIFT} apart)")
        if not reasons or all(r.startswith(("unstable", "imprecise")) for r in reasons):
            boot = bootstrap(eval_usable, limits, medians)
            figures["bootstrap"] = boot
            if boot["undefined_share"] > MAX_UNDEFINED_SHARE:
                reasons.append(f"unstable: {boot['undefined_share']:.3f} of the bootstrap replicates left a side empty "
                               f"(at most {MAX_UNDEFINED_SHARE})")
            interval = boot["risk_difference_interval"]
            if interval is not None and interval[1] - interval[0] > MAX_CI_WIDTH:
                reasons.append(f"imprecise: the risk difference's interval {interval} is wider than {MAX_CI_WIDTH}")
    elif limits is None:
        reasons.append("insufficient: no reference band (no usable reference match)")
    if not executed:
        outcome = "not interpreted: execution incomplete (the judged items above)"
    elif reasons:
        outcome = "inconclusive: " + "; ".join(reasons)
    else:
        low, high = figures["bootstrap"]["risk_difference_interval"]
        difference = figures["risk_difference"]
        if low > 0:
            outcome = (f"association: in the evaluation blocks non-matching votes were more frequent outside the "
                       f"reference band than inside it (risk difference {difference}, {CI_LEVEL:.0%} interval "
                       f"[{low}, {high}])")
        elif high < 0:
            outcome = (f"association: in the evaluation blocks non-matching votes were less frequent outside the "
                       f"reference band than inside it (risk difference {difference}, {CI_LEVEL:.0%} interval "
                       f"[{low}, {high}])")
        else:
            outcome = (f"no association detected at this precision: risk difference {difference}, {CI_LEVEL:.0%} "
                       f"interval [{low}, {high}] includes 0")
    figures["reasons"] = reasons
    return {"figures": figures, "outcome": outcome}


# ---------------------------------------------------------------- the reading


def _sanity(items: list[dict]) -> tuple[Any, bool | None]:
    """The measured records' medians against the documented landmark order: the nose between the eye and mouth lines,
    below the eyes, and not beside the face. None when nothing was measured (MISSING)."""
    measured = [o["record"] for o in items if o["record"].get("status") == "measured"]
    if not measured:
        return MISSING, None
    value = {"measured": len(measured),
             "median_nose_position": round(statistics.median(r["nose_position"] for r in measured), 3),
             "median_nose_vertical": round(statistics.median(r["nose_vertical"] for r in measured), 3),
             "median_abs_nose_lateral": round(statistics.median(abs(r["nose_lateral"]) for r in measured), 3)}
    return value, (0 < value["median_nose_position"] < 1 and value["median_nose_vertical"] > 0
                   and value["median_abs_nose_lateral"] < 0.5)


def read_landmarks(directory: Path, f1: Path, gate: tuple[Path, Path, Path | None] | None = None) -> Reading:
    r = Reading()
    if gate is not None:
        ok, lines = f1_gate(*gate)
        r.item("F1 basis for this run: the run's own F1 validated, or the maintainer's accepted corrected F1 reading "
               "of it", dict(line.split("=", 1) for line in lines), ok and f1.resolve() == (gate[0] / "f1").resolve())
    result = s5._load(directory / "result.json")
    s5._supervision(r, result, "duration_stop", 0)
    readiness = s5._get(result, "readiness")
    r.item("readiness required before T0: the guard required the landmark diagnostic's readiness on the runtime's "
           "starting line (no detector or face problem; the marker as pinned) and found it met", readiness,
           isinstance(readiness, dict) and readiness.get("required") == "landmark_diagnostic"
           and readiness.get("met") is True and readiness.get("reason") is None)
    start = s5._get(result, "run_output", "starting")
    face = s5._get(start, "face")
    present = sorted(_startup_keys(directory / "run.jsonl") & set(NOT_IN_THIS_RUN))
    r.item("a landmark diagnostic run on the normal recognition path: face admitted for this validation run, one "
           "enrolled identity, the marker as pinned, production detection, no candidate, comparison, experiment, "
           "continuity rule or tracking diagnostic; detector and face admitted; no live view, snapshots, chat or "
           "feature controls", {"face": face, "face_problem": s5._get(start, "face_problem"),
                                "detector_problem": s5._get(start, "detector_problem"), "present": present},
           face == {**F2_FACE, "landmark_diagnostic": PINNED_MARKER} and s5._get(start, "face_problem") is None
           and s5._get(start, "detector_problem") is None and not present)
    _workload_items(r, start, result)
    stopped = s5._get(result, "run_output", "stopped")
    r.item("clean shutdown: all stopped (the face worker included), database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True
           and (stopped.get("stopped") or {}).get("face_worker") is True)
    parameters = s5._get(result, "parameters")
    declared = [c.get("name") for c in s5._list(s5._get(parameters, "cues")) if isinstance(c, dict)]
    printed = {c.get("name"): c.get("printed_mono_s") for c in s5._list(s5._get(result, "cues")) if isinstance(c, dict)}
    ready = s5._get(result, "child", "ready_mono_s")
    r.item(f"the session's shape: {DURATION_S:g} s counted from readiness, only the start and stop cues (no movement "
           "asked), each printed, T0 when the runtime was ready",
           {"duration_s": s5._get(parameters, "duration_s"), "from_launch": s5._get(parameters, "from_launch"),
            "cues": declared, "printed": printed, "ready_mono_s": ready},
           s5._get(parameters, "duration_s") == DURATION_S and s5._get(parameters, "from_launch") is False
           and declared == list(CUES) and sorted(printed) == sorted(CUES)
           and all(isinstance(v, float) for v in printed.values()) and isinstance(ready, float)
           and 0 <= printed.get("t0", -1.0) - ready <= CUE_TOLERANCE_S)
    records, problems, _, _ = read_records(directory / "run.jsonl")
    total = sum(len(v) for v in records.values())
    schemas = sorted({item["schema"] for kind in records.values() for item in kind})
    names = [c.get("name") for c in s5._list(s5._get(result, "captures")) if isinstance(c, dict)]
    captures = [s5._load(directory / f"status-{name}.json") for name in names]
    last = captures[-1] if captures else MISSING
    dropped = s5._get(last, "components", "face", "records", "dropped")
    r.item(f"identity records: schema {LANDMARK_SCHEMA} only, each landmark record fitting its status, complete, none "
           "dropped", {"records": {k: len(v) for k, v in records.items()}, "schemas": schemas,
                       "problems": problems[:5], "dropped": dropped} if total else MISSING,
           total > 0 and not problems and schemas == [LANDMARK_SCHEMA] and dropped == 0
           and total < MAX_IDENTITY_RECORDS)
    t0_ns = round(ready * NS) if isinstance(ready, float) else None
    items, left_out = observations(records, t0_ns) if t0_ns is not None else ([], {})
    value, sane = _sanity(items)
    r.item("the measured landmarks fit the documented landmark order (medians: the nose between the eye and mouth "
           "lines, below the eyes, less than half an eye spacing beside the eye midpoint)", value, sane)
    account, problem, fmt = _account(directory)
    r.item("account saved and locked before any result was shown", problem or "locked", account is not None)
    if account is not None:
        review = (face_account.review_separate(account, face_account.FIELDS_V10) if fmt == ACCOUNT_FORMAT
                  else [f"format {fmt} is not this session's account (format {ACCOUNT_FORMAT})"])
        r.item(f"account in format {ACCOUNT_FORMAT}, every answer yes, no or unknown", review,
               fmt == ACCOUNT_FORMAT and review[-1] == "Answered (yes, no or unknown everywhere): yes")
    _counts_items(r, directory)
    executed = r.ok
    analysis = analyse(items, account if fmt == ACCOUNT_FORMAT else None, executed)
    figures = analysis["figures"]
    r.describe(SCOPE, {"window_s": list(WINDOW_S), "blocks": BLOCKS, "block_s": BLOCK_S, "buffer_s": BUFFER_S,
                       "reference_blocks": "even", "evaluation_blocks": "odd", "primary": list(PRIMARY)})
    r.describe("the run's conditions as its launch recorded them (the framing since the capture and the room light), and "
               "the account's answers (USER-REPORTED)",
               {**{k: _key_values(directory / "provenance.txt").get(k, MISSING) for k in CONDITIONS},
                "account": None if account is None else {k: account.get(k) for k in (
                    "stayed_seated", "worked_as_usual", "looked_at_camera_on_purpose", "other_faces_in_view")}})
    r.describe("votes per set (one-person applied results in the window; misses are unknown votes), usable landmark "
               "observations, coverage, and what was not usable, by reason", {
                   "votes": figures["votes"], "coverage": figures["coverage"], "not_usable": figures["not_usable"],
                   "left_out": left_out})
    statuses = Counter(o["record"].get("status") for o in items)
    native = [o for o in items if o["native_only"]]
    r.describe("landmark statuses over the window's one-person results (every vote and no-face result), and the results "
               "where production found no face but the native pass did (their landmark statuses; band membership for "
               "the usable ones is described, never judged)", {
                   "statuses": dict(sorted(statuses.items())),
                   "native_only": {"results": len(native),
                                   "statuses": dict(sorted(Counter(o["record"].get("status") for o in native).items())),
                                   **({"outside_band": sum(1 for o in native if o["record"].get("status") == "measured"
                                                           and not inside(o, {f: tuple(v) for f, v in
                                                                              figures["band"].items()})),
                                       "measured": sum(1 for o in native if o["record"].get("status") == "measured")}
                                      if figures["band"] is not None else {})}})
    r.describe("the reference band (built from the reference blocks' usable matching votes; per primary feature its "
               f"{BAND_TAIL:g} and {1 - BAND_TAIL:g} quantiles; a joint box) and its share of the matches in each set",
               {"band": figures["band"], "share": figures.get("band_share_of_matches")})
    r.describe("evaluation blocks: usable votes and misses inside and outside the band, the risk difference (outside - "
               "inside) and ratio, and the miss episodes placed by their first usable vote",
               {k: figures.get(k) for k in ("inside", "outside", "risk_difference", "risk_ratio", "episodes")})
    r.describe("effect sizes per feature on the evaluation's usable votes: distance AUC (misses farther from the "
               "reference matches' median than matches: above 0.5) and signed AUC (misses higher: above 0.5)",
               figures.get("effect_sizes"))
    r.describe(f"cluster bootstrap over the evaluation blocks' {SEGMENT_S:g}-second segments ({BOOTSTRAP_REPLICATES} "
               f"replicates, seed {BOOTSTRAP_SEED}; {CI_LEVEL:.0%} percentile intervals); run only when nothing else "
               "made the outcome inconclusive", figures.get("bootstrap"))
    full = [x.get("full_scale") for x in records.get("result", []) if isinstance(x.get("full_scale"), dict)]
    applied = [x for x in records.get("result", []) if x.get("outcome") == "applied"]
    offered = [t for t in records.get("tick", []) if t.get("tick") == "offered"]
    in_window = [x for x in applied if t0_ns is not None and WINDOW_S[0] <= (x["frame_mono_ns"] - t0_ns) / NS < WINDOW_S[1]]
    r.describe("overhead and processed rate: the native pass's time per result and the landmark records' time per "
               "applied result (ms, p50/p95/max), production's processing time, frame to applied, applied results "
               "per second in the window and per offered tick over the run", {
                   "native_pass_ms": spread([f["ms"] for f in full]),
                   "landmark_ms": spread([x["landmark_ms"] for x in applied if x.get("landmark_ms") is not None]),
                   "processing_ms": spread([x["processing_ms"] for x in applied
                                                  if isinstance(x.get("processing_ms"), (int, float))]),
                   "frame_to_applied_ms": spread([(x["applied_mono_ns"] - x["frame_mono_ns"]) / 1e6 for x in applied]),
                   "applied_per_s_in_window": round(len(in_window) / (WINDOW_S[1] - WINDOW_S[0]), 3),
                   "applied_per_offered": round(len(applied) / len(offered), 4) if offered else None})
    r.lines.append(f"DESCRIPTIVE {CAUSATION}")
    r.lines.append(f"OUTCOME (descriptive) {analysis['outcome']}")
    return r
