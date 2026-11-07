"""The multi-pose comparison's reading of one evaluation session (PROPOSED; validated with synthetic data only).

usage: mpe_compare.py SESSION_DIR --config FACE_YAML --frozen EXP_ROOT/candidate.frozen.json

SESSION_DIR holds the guard's result.json, the run's run.jsonl (a route B run with --face-shadow-gallery) and the
person's account, mpe-account.txt (mpe_schedule.read_account). The schedule (mpe_schedule: E_M or E_P) is identified
from the cues; any other session is refused. INCONCLUSIVE, never read as a result: no frozen record, or a run whose
startup shadow digest is not the frozen candidate's prototypes digest (another working gallery was compared); a missing
or invalid account, or one that does not say nobody else was in view. A hold (or E_P window) the account lists as not
done as cued is excluded and reported.

**Decisions, never scores.** The baseline decision is each compared face's vote as the runtime cast it (the original
gallery, full precision). The candidate's is the shadow's ``decision``: the same production function (state.decide)
with the same policy on the same face, made before anything was rounded. No similarity value is read for a decision
or printed.

**Gains and losses** on the same faces, per hold and per condition: both match, the original only (a loss), the
candidate only (a gain), neither.

**Identity history, chronological and whole-session.** The production resolver (state.IdentityResolver) is fed the
session's recorded faces in applied order, once with the baseline decisions and once with the candidate's (the same
inputs: the recorded clears as inputs, the frame times as vote times, its own TTL, fresh window and confirmations from
the session's configuration through runtime.identity_policy), up to the run's last recorded step (the runtime records
every change while it runs, so a later instant is after its end). The baseline replay must reproduce every recorded
transition other than clears (each within REPLAY_TOLERANCE_MS after the instant it predicts, as the runtime notices a
change at its next step); otherwise the reading is INCONCLUSIVE. KNOWN share of hold time, KNOWN at the mark (V3's
rule: KNOWN at the hold bell, or a first KNOWN in the hold built on a vote from before the bell, is not_at_mark; the
acquisitions the candidate loses and gains are both counted) and exits come from these replays.

**Hold-reset analysis**, printed apart and labelled as such: each hold replayed alone from its hold bell (earlier votes
excluded). It is an analysis of the holds, not what the runtime did.

**The participant (E_P):** every compared face of the session, from T0 to the run's end (nothing of theirs left out):
candidate match votes and KNOWN (REJECTED if any), the original's (a finding about the current system, reported
separately), and exposure per window that gave the frozen candidate a prototype, over the builder's own window (from
POSE_SETTLE_S after the pose bell; NOT_EXERCISED where the participant gave no compared face there).

**Not decided here:** the gain bar and the facing-loss limit are the maintainer's (PENDING); this reading reports the
numbers and draws no SUPPORTED or NOT_SUPPORTED verdict. One enrolled and one non-enrolled person, one session each:
k of n, never a rate; zero candidate matches on the participant is not general false-accept safety.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mpe_account
import mpe_schedule
from sentinel.identity.association import Ownership
from sentinel.identity.matching import EnrolledIdentity, Enrollment
from sentinel.identity.state import IdentityPolicy, IdentityResolver, IdentityState
from sentinel.media.clock import MonoInstant

REPLAY_TOLERANCE_MS = 400  # a predicted change must be recorded within this long after it (part 20's)
MIN_COMPARED = 20  # compared faces a hold needs; a hold under it is excluded and reported
MIN_HOLDS = 2  # holds a condition needs after exclusions; otherwise that condition is INCONCLUSIVE
BOOT = "replay"
TOKEN = "idn-000000000000"  # the replay's one identity: any recorded identity maps to it (one enrolled identity)
STUB = Enrollment(model_revision="mpe-replay", dimension=2,
                  identities=(EnrolledIdentity(identity_id=TOKEN, prototypes=((1.0, 0.0),)),))
FACE = {"match": SimpleNamespace(quality=1.0, embedding=(1.0, 0.0)),
        "unknown": SimpleNamespace(quality=1.0, embedding=(0.0, 1.0))}
CONDITION_ORDER = (("s", "facing"), ("4", "facing"), ("4", "natural"), ("5", "facing"), ("5", "natural"))


@dataclass(frozen=True)
class Key:
    """A replay track: the fields IdentityResolver reads from a TrackKey."""

    boot_id: str
    stream_epoch: int
    track_id: int


class Replay:
    """The production resolver driven by recorded decisions ("match", "unknown" or None per compared face) and, in
    the lives mode, by the runtime's own recorded track lives."""

    def __init__(self, policy: IdentityPolicy) -> None:
        self.policy = policy
        self.resolver = IdentityResolver(STUB, policy)
        self.lives: dict[Key, int] = {}
        self.reported: dict[Key, tuple[str, str | None, str | None]] = {}
        # each predicted change: (instant, key, from, to, identity, basis, the newest current vote's frame time or None)
        self.changes: list[tuple[int, Key, str, str, str | None, str | None, int | None]] = []
        self.timeline: dict[Key, list[tuple[int, str]]] = defaultdict(list)
        self.entries: list[tuple[int, Key, list[int]]] = []  # KNOWN entries: (time, key, confirming votes' frame times)

    def begin(self, key: Key, life_ns: int) -> None:
        """Legacy mode only (no life records): a track's life is taken to start when it is first seen."""
        if key not in self.lives:
            self.lives[key] = life_ns
            self.sync()

    def start_life(self, key: Key, since_ns: int) -> None:
        self.lives[key] = since_ns

    def end_life(self, key: Key, at_ns: int) -> None:
        """The life is no longer current at this step: a reported state ends as "cleared", as the runtime records it."""
        self.lives.pop(key, None)
        before = self.reported.pop(key, None)
        if before is not None:
            self.changes.append((at_ns, key, before[0], "cleared", None, None, None))
            self.timeline[key].append((at_ns, "cleared"))

    def sync(self) -> None:
        """The runtime's retain_only at a step: votes of tracks or earlier lives that are not current are dropped."""
        self.resolver.retain_only(self.lives)

    def observe(self, key: Key, frame_ns: int, decision: str | None) -> None:
        person = SimpleNamespace(key=key, observed_mono_ns=frame_ns)
        if decision in FACE:
            association = SimpleNamespace(person_ownership={key: Ownership.ASSIGNED}, face_of={key: 0})
            self.resolver.observe([person], [FACE[decision]], association)
        else:
            self.resolver.observe([person], [], SimpleNamespace(person_ownership={key: Ownership.NO_FACE}, face_of={}))

    def clear(self, key: Key, at_ns: int) -> None:
        """Legacy mode: a recorded clear taken as an input."""
        self.end_life(key, at_ns)
        self.sync()

    def evaluate(self, now_ns: int) -> None:
        now = MonoInstant(BOOT, now_ns)
        for key in list(self.lives):
            resolved = self.resolver.identity(key, now)
            value = (resolved.state.value, resolved.identity_id, None if resolved.basis is None else resolved.basis.value)
            before = self.reported.get(key, ("unresolved", None, None))
            if value != before:
                newest = None if resolved.last_vote_age_ns is None else now_ns - resolved.last_vote_age_ns
                self.changes.append((now_ns, key, before[0], value[0], value[1], value[2], newest))
                self.timeline[key].append((now_ns, value[0]))
                if value[0] == IdentityState.KNOWN.value and before[0] != IdentityState.KNOWN.value:
                    current = self.resolver._current_votes(key, now)  # the very votes the production rule used
                    self.entries.append((now_ns, key, [v.at_ns for v in current][-self.policy.confirmations:]))
            if value[0] == "unresolved":
                self.reported.pop(key, None)
            else:
                self.reported[key] = value

    def known_ns(self, start: int, end: int, keys=None) -> int:
        """Time in [start, end) during which any (chosen) track was KNOWN (the union over tracks)."""
        spans = []
        for key, events in self.timeline.items():
            if keys is not None and key not in keys:
                continue
            state, since = "unresolved", None
            for t, to in events + [(end, "end")]:
                if state == "known" and since is not None:
                    a, b = max(since, start), min(t, end)
                    if b > a:
                        spans.append((a, b))
                state, since = to, t
        spans.sort()
        total, current = 0, None
        for a, b in spans:
            if current is None or a > current[1]:
                if current is not None:
                    total += current[1] - current[0]
                current = [a, b]
            else:
                current[1] = max(current[1], b)
        if current is not None:
            total += current[1] - current[0]
        return total

    def known_at(self, t: int) -> bool:
        for events in self.timeline.values():
            state = "unresolved"
            for at, to in events:
                if at > t:
                    break
                state = to
            if state == "known":
                return True
        return False


def read_session(session: Path) -> dict[str, Any]:
    result = json.loads((session / "result.json").read_text("utf-8"))
    results, transitions, ticks, lives, startup = [], [], [], [], None
    for line in (session / "run.jsonl").read_text("utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        kind = record.get("identity")
        if kind == "result":
            results.append(record)
        elif kind == "transition":
            transitions.append(record)
        elif kind == "tick":
            ticks.append(record)
        elif kind == "life":
            lives.append(record)
        elif record.get("run") == "starting" and startup is None:
            startup = record.get("startup")
    steps = [r["applied_mono_ns"] for r in results] + [t["frame_mono_ns"] for t in ticks] + \
        [t["mono_ns"] for t in transitions] + [life["mono_ns"] for life in lives]
    return {"result": result, "results": results, "transitions": transitions, "ticks": ticks, "lives": lives,
            "startup": startup, "last_ns": max(steps) if steps else 0}


def decisions(results: list[dict]) -> tuple[list[dict], int]:
    """Each compared face (a person whose vote computed a similarity): its frame and applied times, track, the
    baseline vote and the shadow's decision. Also how many compared faces lacked a shadow decision."""
    faces, missing = [], 0
    for record in results:
        if record.get("outcome") != "applied":
            continue
        for person in record.get("persons") or []:
            if person.get("label") not in ("match", "unknown", "margin"):
                continue
            shadow = person.get("shadow")
            decision = shadow.get("decision") if isinstance(shadow, dict) else None
            if decision not in ("match", "unknown", "margin"):
                missing += 1
                continue
            faces.append({"frame": record["frame_mono_ns"], "applied": record["applied_mono_ns"],
                          "epoch": record["stream_epoch"], "track": person["track_id"],
                          "baseline": person["vote"] if person["vote"] in ("match", "unknown") else None,
                          "candidate": decision if decision in ("match", "unknown") else None})
    return faces, missing


def lives_problem(data: dict) -> str | None:
    """None when the session's life records can drive the replay: the run started with them (startup face "lives"),
    each track number's lives alternate began/ended with the same start, a life never starts after its step, and every
    track number a tick or a result names has a life that began at or before that frame. A fixed label otherwise."""
    if not ((data["startup"] or {}).get("face") or {}).get("lives"):
        return "life_records_missing"
    open_lives: dict[tuple[int, int], int] = {}
    began: dict[tuple[int, int], list[int]] = defaultdict(list)
    for record in sorted(data["lives"], key=lambda r: (r["mono_ns"], r["event"] != "ended")):
        key = (record["stream_epoch"], record["track_id"])
        if record["since_mono_ns"] > record["mono_ns"]:
            return "life_records_inconsistent"
        if record["event"] == "began":
            if key in open_lives:
                return "life_records_inconsistent"
            open_lives[key] = record["since_mono_ns"]
            began[key].append(record["since_mono_ns"])
        elif open_lives.pop(key, None) != record["since_mono_ns"]:
            return "life_records_inconsistent"
    seen = [((t["stream_epoch"], track), t["frame_mono_ns"]) for t in data["ticks"] for track in t.get("tracks") or []]
    seen += [((r["stream_epoch"], p["track_id"]), r["frame_mono_ns"]) for r in data["results"]
             if r.get("outcome") == "applied" for p in r.get("persons") or []]
    if any(not any(since <= frame for since in began.get(key, ())) for key, frame in seen):
        return "life_records_incomplete"
    return None


def replay(data: dict, policy: IdentityPolicy, which: str, mode: str = "lives") -> Replay:
    """The whole session in applied order, with ``which`` decisions ("baseline" or "candidate"). ``mode`` "lives": the
    runtime's recorded life boundaries drive the lives, as its own steps did (at a step: the result observed, then the
    lives that step ended and began, then the votes of what is not current dropped, then the decision). "legacy" (a
    session without life records): a life starts when first seen and ends only at a recorded clear, so an unrecorded
    end (a silent expiry) is invisible; it serves the baseline check only and never validates the candidate's."""
    run = Replay(policy)
    events: list[tuple[int, int, int, str, Any]] = []
    order = 0
    faces_by_result = defaultdict(list)
    for face in decisions(data["results"])[0]:
        faces_by_result[(face["applied"], face["epoch"], face["frame"])].append(face)
    for record in data["results"]:
        if record.get("outcome") != "applied":
            continue
        order += 1
        chosen = {f["track"]: f[which] for f in faces_by_result[(record["applied_mono_ns"], record["stream_epoch"],
                                                                 record["frame_mono_ns"])]}
        events.append((record["applied_mono_ns"], 0, order, "result",
                       (record["stream_epoch"], record["frame_mono_ns"],
                        [(p["track_id"], chosen.get(p["track_id"])) for p in record.get("persons") or []])))
    if mode == "lives":
        for record in data["lives"]:
            order += 1
            events.append((record["mono_ns"], 1 if record["event"] == "ended" else 2, order, "life", record))
    else:
        for record in data["ticks"]:
            for track in record.get("tracks") or []:
                order += 1
                events.append((record["frame_mono_ns"], 0, order, "seen",
                               (Key(BOOT, record["stream_epoch"], track), record["frame_mono_ns"])))
        for record in data["transitions"]:
            if record["to"] == "cleared":
                order += 1
                events.append((record["mono_ns"], 1, order, "clear",
                               Key(BOOT, record["stream_epoch"], record["track_id"])))
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    thresholds: list[int] = []  # the instants a vote's basis or currency changes (fresh window + 1 ns; TTL)
    for index, (t, _, _, kind, payload) in enumerate(events):
        for instant in sorted(x for x in thresholds if x < t):  # what changed before this step, first
            run.evaluate(instant)
        thresholds = [x for x in thresholds if x >= t]
        if kind == "seen":
            run.begin(*payload)
        elif kind == "result":
            epoch, frame_ns, persons = payload
            for track, decision in persons:
                key = Key(BOOT, epoch, track)
                if mode == "legacy":
                    run.begin(key, frame_ns)
                run.observe(key, frame_ns, decision)
                if decision is not None:
                    thresholds += [frame_ns + policy.fresh_window_ns + 1, frame_ns + policy.vote_ttl_ns]
        elif kind == "life":
            key = Key(BOOT, payload["stream_epoch"], payload["track_id"])
            if payload["event"] == "ended":
                run.end_life(key, t)
            else:
                run.start_life(key, payload["since_mono_ns"])
        elif kind == "clear":
            run.end_life(payload, t)
        if index + 1 < len(events) and events[index + 1][0] == t:
            continue  # the rest of this step first: the runtime decides once per step, after all of it
        run.sync()
        run.evaluate(t)
    for instant in sorted(x for x in thresholds if x <= data["last_ns"]):  # never past the run's last recorded step:
        run.evaluate(instant)  # the runtime records every change while it runs, so later instants are after its end
    return run


def _newest_agrees(recorded: dict, newest: int | None) -> bool:
    """The recorded transition's newest current vote (its time less last_vote_age_ms, a floor of the age) is the
    replay's: the same vote, not one face tick apart."""
    age = recorded["last_vote_age_ms"]
    if age is None or newest is None:
        return age is None and newest is None
    return 0 <= recorded["mono_ns"] - age * 1_000_000 - newest < 1_000_000


def check_replay(run: Replay, transitions: list[dict], clears: bool = False) -> dict[str, int]:
    """The baseline replay against the recorded history, per track and in order: each recorded transition (clears too
    when ``clears``: the lives mode, where they are outputs) must be matched by the next predicted change with the same
    from and to states, basis, presence of an identity and newest current vote, predicted at most REPLAY_TOLERANCE_MS
    before it was recorded (the runtime notices a change at its next step). A predicted change that nothing recorded
    is extra. In the legacy mode clears are the replay's inputs and not compared."""
    tolerance = REPLAY_TOLERANCE_MS * 1_000_000
    recorded = defaultdict(list)
    for record in transitions:
        if clears or record["to"] != "cleared":
            recorded[(record["stream_epoch"], record["track_id"])].append(record)
    predicted = defaultdict(list)
    for t, key, frm, to, identity, basis, newest in run.changes:
        if clears or to != "cleared":
            predicted[(key.stream_epoch, key.track_id)].append((t, frm, to, identity is not None, basis, newest))
    matched = unmatched = extra = 0
    for track in set(recorded) | set(predicted):
        mine, theirs = predicted.get(track, []), recorded.get(track, [])
        i = 0
        for record in theirs:
            when = record["mono_ns"]
            want = (record["from"], record["to"], record["identity_id"] is not None, record["basis"])
            while i < len(mine) and mine[i][0] < when - tolerance:  # predicted, never recorded in time
                extra += 1
                i += 1
            if i < len(mine) and tuple(mine[i][1:5]) == want and 0 <= when - mine[i][0] <= tolerance \
                    and _newest_agrees(record, mine[i][5]):
                matched += 1
                i += 1
            else:
                unmatched += 1
        extra += len(mine) - i
    return {"recorded": sum(len(v) for v in recorded.values()), "matched": matched, "unmatched": unmatched,
            "extra": extra, "clears": sum(1 for r in transitions if r["to"] == "cleared"), "clears_compared": clears}


def at_mark(run: Replay, bell: int, end: int) -> str:
    """V3's rule on a replay: KNOWN at the bell, or a first KNOWN in [bell, end) built on a vote from before the bell,
    is not_at_mark; a first KNOWN in the hold on votes from it is acquired; never KNOWN is none."""
    if run.known_at(bell):
        return "not_at_mark"
    first = next(((t, votes) for t, _, votes in sorted(run.entries, key=lambda e: e[0]) if bell <= t < end), None)
    if first is None:
        return "none"
    return "acquired" if all(v >= bell for v in first[1]) else "not_at_mark"


def hold_reset(faces: list[dict], policy: IdentityPolicy, which: str, bell: int, end: int) -> float:
    """ANALYSIS ONLY (not what the runtime did): the hold replayed alone from its bell; KNOWN share of the hold."""
    run = Replay(policy)
    chosen = sorted((f for f in faces if bell <= f["frame"] < end), key=lambda f: (f["applied"], f["frame"]))
    thresholds: list[int] = []
    for face in chosen:
        key = Key(BOOT, face["epoch"], face["track"])
        run.begin(key, bell)
        due = sorted(x for x in thresholds if x <= face["applied"])
        thresholds = [x for x in thresholds if x > face["applied"]]
        for instant in due:
            run.evaluate(instant)
        run.observe(key, face["frame"], face[which])
        if face[which] is not None:
            thresholds += [face["frame"] + policy.fresh_window_ns + 1, face["frame"] + policy.vote_ttl_ns]
        run.evaluate(face["applied"])
    for instant in sorted(x for x in thresholds if x < end):
        run.evaluate(instant)
    return run.known_ns(bell, end) / (end - bell)


def pairs(faces: list[dict]) -> dict[str, int]:
    out = {"compared": len(faces), "both": 0, "loss": 0, "gain": 0, "neither": 0}
    for f in faces:
        b, c = f["baseline"] == "match", f["candidate"] == "match"
        out["both" if b and c else "loss" if b else "gain" if c else "neither"] += 1
    return out


def compare(session: Path, policy: IdentityPolicy, frozen: dict | None) -> dict[str, Any]:
    result = json.loads((Path(session) / "result.json").read_text("utf-8"))
    kind = mpe_schedule.schedule_of(result)
    if kind not in ("E_M", "E_P"):
        return {"session": None, "inconclusive": ["not_an_evaluation_session"]}
    try:  # the account step comes first: nothing of the session's records is read or shown before it is locked
        account = mpe_account.require(session, kind)
    except mpe_account.AccountRefused as exc:
        return {"session": None, "stopped": exc.label, "inconclusive": [exc.label]}
    data = read_session(session)
    printed = mpe_schedule.printed_ns(data["result"])
    settle = round(mpe_schedule.POSE_SETTLE_S * 1e9)
    faces, missing = decisions(data["results"])
    inconclusive = []
    # Both galleries are replayed with the runtime's own recorded life boundaries. Without them (or with records that
    # do not hold together) the legacy replay still checks the baseline, but nothing validates the candidate's
    # history: a stale vote from an ended life can change the candidate's states while the baseline's still agree.
    problem = lives_problem(data)
    mode = "lives" if problem is None else "legacy"
    if problem is not None:
        inconclusive.append(problem)
    baseline, candidate = replay(data, policy, "baseline", mode), replay(data, policy, "candidate", mode)
    check = check_replay(baseline, data["transitions"], clears=mode == "lives")
    if check["unmatched"] or check["extra"]:
        inconclusive.append("replay_does_not_reproduce_the_recorded_history")
    replay_valid = mode == "lives" and not (check["unmatched"] or check["extra"])
    if missing:
        inconclusive.append("compared_faces_without_a_shadow_decision")
    # the frozen candidate and this run's shadow gallery: the same prototypes (the digest the run recorded at its start)
    started = ((data["startup"] or {}).get("face") or {}).get("shadow_digest")
    if frozen is None or frozen.get("format") != "mpe-frozen-1":
        inconclusive.append("frozen_record_missing")
    elif started is None or started != frozen.get("prototypes_digest"):
        inconclusive.append("shadow_gallery_not_the_frozen_candidate")
    holds = mpe_schedule.EM_HOLDS if kind == "E_M" else mpe_schedule.EP_HOLDS
    windows = () if kind == "E_M" else mpe_schedule.EP_WINDOWS
    not_as_cued: set[str] = account["not_as_cued"]
    if account["others_in_view"] != "no":
        inconclusive.append("account_others_in_view_not_no")
    rows = []
    for hold in holds:
        bell, end = printed[hold.hold], printed[hold.end]
        inside = [f for f in faces if bell <= f["frame"] < end]
        length = end - bell
        rows.append({"hold": hold.name, "spot": hold.spot, "kind": hold.kind, **pairs(inside),
                     "not_as_cued": hold.name in not_as_cued,
                     "excluded": hold.name in not_as_cued or len(inside) < MIN_COMPARED,
                     # replay-derived: only from a valid replay (otherwise WITHHELD, never the hold-reset's)
                     "known_share": {"baseline": baseline.known_ns(bell, end) / length,
                                     "candidate": candidate.known_ns(bell, end) / length} if replay_valid else None,
                     "at_mark": {"baseline": at_mark(baseline, bell, end),
                                 "candidate": at_mark(candidate, bell, end)} if replay_valid else None,
                     "hold_reset_share": {"baseline": hold_reset(faces, policy, "baseline", bell, end),
                                          "candidate": hold_reset(faces, policy, "candidate", bell, end)}})
    conditions = []
    for spot, kind_ in CONDITION_ORDER:
        mine = [r for r in rows if r["spot"] == spot and r["kind"] == kind_]
        if not mine:
            continue
        kept = [r for r in mine if not r["excluded"]]
        pooled = {k: sum(r[k] for r in kept) for k in ("compared", "both", "loss", "gain", "neither")}
        shares = {w: (sum(r["known_share"][w] for r in kept) / len(kept)) if kept else None
                  for w in ("baseline", "candidate")} if replay_valid else None
        conditions.append({"spot": spot, "kind": kind_, "holds": len(mine), "excluded": len(mine) - len(kept),
                           "not_as_cued": sum(1 for r in mine if r["not_as_cued"]),
                           "inconclusive": len(kept) < MIN_HOLDS, **pooled, "known_share": shares,
                           "lost_acquisitions": sum(1 for r in kept if r["at_mark"]["baseline"] == "acquired"
                                                    and r["at_mark"]["candidate"] != "acquired")
                           if replay_valid else None,
                           "gained_acquisitions": sum(1 for r in kept if r["at_mark"]["candidate"] == "acquired"
                                                      and r["at_mark"]["baseline"] != "acquired")
                           if replay_valid else None})
    out: dict[str, Any] = {"session": kind, "replay": check, "mode": mode, "replay_valid": replay_valid,
                           "inconclusive": inconclusive, "holds": rows, "conditions": conditions,
                           "pending": "the gain bar and the facing-loss limit"}
    if kind == "E_P":
        whole = pairs(faces)  # every compared face of the session, T0 to the run's end: nothing of the participant left out
        out["participant"] = {
            "compared": whole["compared"],
            "candidate_match_votes": whole["both"] + whole["gain"],
            "baseline_match_votes": whole["both"] + whole["loss"],
            "candidate_known_entries": len(candidate.entries) if replay_valid else None,
            "baseline_known_entries": len(baseline.entries) if replay_valid else None,
        }
        # a KNOWN needs match votes: any candidate match vote rejects, with or without a valid replay
        out["rejected"] = out["participant"]["candidate_match_votes"] > 0 or bool(
            out["participant"]["candidate_known_entries"])
        exposure = {}
        by_name = {w.name: w for w in mpe_schedule.EP_WINDOWS}
        for name, _ in (frozen or {}).get("windows", []):
            window = by_name.get("p" + name[1:])
            if window is None or window.name in not_as_cued:
                exposure[name] = None
                continue
            start, end = printed[window.bell] + settle, printed[window.until]  # the builder's window
            exposure[name] = sum(1 for f in faces if start <= f["frame"] < end)
        out["exposure"] = exposure
        out["not_exercised"] = sorted(n for n, v in exposure.items() if not v)
    return out


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def print_report(out: dict[str, Any]) -> None:
    if out.get("stopped"):
        print(f"STOP: enter the account first (mpe_account.py enter): {out['stopped']}; nothing is shown before it")
        return
    if out["session"] is None:
        print(f"mpe-compare: INCONCLUSIVE ({', '.join(out['inconclusive'])})")
        return
    r = out["replay"]
    withheld = "WITHHELD (no valid replay)"
    print(f"mpe-compare: session {out['session']}")
    print(f"VALIDITY replay ({out['mode']} mode) of production identity history: {r['matched']} of {r['recorded']} "
          f"recorded transitions reproduced ({'clears compared' if r['clears_compared'] else 'clears as inputs'}: "
          f"{r['clears']}; unmatched {r['unmatched']}; extra predicted {r['extra']}); both galleries' replays "
          f"{'valid' if out['replay_valid'] else 'NOT valid: every number below that comes from a replay is withheld'}")
    for label in out["inconclusive"]:
        print(f"INCONCLUSIVE {label}")
    print("CHRONOLOGICAL (the whole session replayed in order, with the runtime's own track lives):")
    for c in out["conditions"]:
        shares = c["known_share"]
        if shares is None:
            known = f"KNOWN share {withheld}; acquisitions at the mark {withheld}"
        else:
            diff = (None if shares["baseline"] is None or shares["candidate"] is None
                    else 100 * (shares["candidate"] - shares["baseline"]))
            known = (f"KNOWN share original {_pct(shares['baseline'])}, candidate {_pct(shares['candidate'])}"
                     f"{'' if diff is None else f' ({diff:+.1f} points)'}; acquisitions at the mark lost "
                     f"{c['lost_acquisitions']}, gained {c['gained_acquisitions']}")
        print(f"  {mpe_schedule.POSITIONS[c['spot']]} {c['kind']}: holds {c['holds']} (excluded {c['excluded']}"
              f"{', INCONCLUSIVE' if c['inconclusive'] else ''}); compared {c['compared']}: both {c['both']}, "
              f"loss {c['loss']}, gain {c['gain']}, neither {c['neither']}; {known}; not as cued {c['not_as_cued']}")
    for h in out["holds"]:
        why = " (excluded: not as cued)" if h["not_as_cued"] else " (excluded: under the minimum)" if h["excluded"] else ""
        if h["known_share"] is None:
            known = f"KNOWN {withheld}"
        else:
            known = (f"KNOWN {_pct(h['known_share']['baseline'])} -> {_pct(h['known_share']['candidate'])}; at the mark "
                     f"{h['at_mark']['baseline']} -> {h['at_mark']['candidate']}")
        print(f"    {h['hold']}{why}: compared {h['compared']}, loss {h['loss']}, gain {h['gain']}; {known}")
    print("ANALYSIS hold reset (each hold alone from its bell; NOT runtime behaviour, never a criterion, never a "
          "substitute for a withheld number):")
    for h in out["holds"]:
        print(f"    {h['hold']}: KNOWN {_pct(h['hold_reset_share']['baseline'])} -> "
              f"{_pct(h['hold_reset_share']['candidate'])}")
    if out["session"] == "E_P":
        p = out["participant"]
        entries = withheld if p["candidate_known_entries"] is None else p["candidate_known_entries"]
        print(f"NON-ENROLLED candidate: match votes {p['candidate_match_votes']} of {p['compared']} compared faces, "
              f"KNOWN entries {entries} -> {'REJECTED' if out['rejected'] else 'none observed'}")
        entries = withheld if p["baseline_known_entries"] is None else p["baseline_known_entries"]
        print(f"NON-ENROLLED original gallery (the current system, reported apart): match votes "
              f"{p['baseline_match_votes']}, KNOWN entries {entries}")
        for name, count in out["exposure"].items():
            print(f"  EXPOSURE {name}: {'NOT_EXERCISED' if not count else f'{count} compared faces'}")
    print(f"PENDING {out['pending']} (the maintainer's decision): no SUPPORTED or NOT_SUPPORTED verdict is drawn")
    print("LIMIT one enrolled and one non-enrolled person, one session each: k of n, never a rate; holds seconds "
          "apart are not independent; zero candidate matches is not general false-accept safety")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--f1", type=Path, required=True)
    for flag in ("--run", "--identity-dir", "--recheck-root"):  # the F1 gate, as the launch block ran it
        parser.add_argument(flag, type=Path, default=None)
    args = parser.parse_args(argv)
    from sentinel.config import load_config
    from sentinel.runtime import identity_policy

    import mpe_check

    session = args.session.absolute()
    if not (session.parent / "face.yaml").is_file() or args.config.read_bytes() != (session.parent /
                                                                                     "face.yaml").read_bytes():
        print("STOP: config_not_the_sessions (--config must be the session's own face.yaml)")
        return 1
    frozen = json.loads(args.frozen.read_text("ascii"))
    kind = mpe_schedule.schedule_of(json.loads((session / "result.json").read_text("utf-8")))
    if kind not in ("E_M", "E_P"):
        print("mpe-compare: INCONCLUSIVE (not an evaluation session's schedule)")
        return 1
    gate = (args.run, args.identity_dir, args.recheck_root) if args.run else None
    ok, lines = mpe_check.passed(session, args.f1, kind, frozen, gate)  # the execution check first (and the account)
    if not ok:
        print("\n".join(lines))
        print("STOP: execution_not_passed (nothing of a session that has not passed is read or shown)")
        return 1
    print(lines[-1])  # mpe-execution: passed
    policy = identity_policy(load_config(args.config).identity)
    out = compare(session, policy, frozen)
    print_report(out)
    return 1 if out["inconclusive"] else 0


if __name__ == "__main__":
    sys.exit(main())
