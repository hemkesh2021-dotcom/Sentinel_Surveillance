"""Recorded-fact answers for chat (session 116, part 28): what happened lately, who is in the room, which alerts.

Deterministic: fixed sentences over the system's own records. No model writes or reads any of it, so no record can
instruct a model and no model statement can become a recorded fact. Each answer says what it covers and what it does
not, and keeps three things apart: "nothing was recorded in the covered interval", "no data", and "nothing happened"
(which no answer here ever claims).

Sources, all read only:
- the runtime's bounded in-memory history (``sentinel.history.RecentHistory``): state changes and scene reports;
- the runtime's current view of tracked people (``DemoRuntime.occupants``): tracks and recognition state;
- the incident database through a read-only connection and three fixed, bounded queries (never model-written SQL):
  incidents, their outbox rows, nothing else.

Text that came from outside (an incident's title, a scene summary, a configured display label) is data: cleaned,
bounded, quoted, and shown by the page as text. A person is "an enrolled person" unless the operator configured a
display label for that enrolment (``identity.display_labels``); the scene model is never asked who anyone is.

Chat stays read only: nothing here acknowledges, deletes, changes a rule or runs a command.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .history import EVENTS_MAX, KEEP_S, SCENES_MAX
from .scene.chat import clean_text
from .scene.chat_routing import ALERTS, HISTORY, SCAN, WHO, ZOOM, Route
from .storage.database import connect_reader

EVENT_LINES = 12  # history: the newest events shown; earlier ones are counted
SCENE_LINES = 3  # history: the newest scene summaries quoted
INCIDENT_LINES = 8  # alerts: the newest incidents listed; earlier ones are counted
INCIDENTS_READ = 50  # alerts: the most incidents read for one answer
PEOPLE_LINES = 12
TITLE_MAX = 120
LABEL_MAX = 40
RECORDED_NOTE = ("Recorded facts: from the system's own records, not from the scene model and not a reading of the "
                 "picture.")
NOT_NOTHING = ("This lists what was recorded. It does not show that nothing else happened: what no detector, tracker, "
               "rule or scene check noticed is not recorded.")
HISTORY_BOUNDS = (f"The history is kept in memory only: at most {EVENTS_MAX} events and {SCENES_MAX} scene reports, "
                  f"none older than {KEEP_S / 60:.0f} minutes, and it starts empty each time the runtime starts.")
NO_NAMES = ("People are named only by a display label the operator configured for an enrolment; otherwise an enrolled "
            "person is called that. The scene model is never asked who anyone is.")
EXPERIMENTAL = "Face recognition is experimental and not validated."
ZERO_DELIVERIES = "Zero deliveries does not mean that no incident occurred."


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _clock(stamp: object) -> str:
    """HH:MM:SS UTC of a stored UTC time; the stored text when it cannot be read."""
    if not isinstance(stamp, str):
        return "an unknown time"
    try:
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return clean_text(stamp, 40)
    return when.astimezone(timezone.utc).strftime("%H:%M:%S UTC")


def _ago(ms: object) -> str:
    if not isinstance(ms, int) or ms < 0:
        return "an unknown time"
    seconds = ms / 1000
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def _document(intent: str, answer: str, lines: Sequence[str], limits: Sequence[str], *,
              covered: Mapping[str, Any] | None = None, references: Mapping[str, Sequence[Any]] | None = None
              ) -> dict[str, Any]:
    return {"state": "answered", "kind": "recorded", "intent": intent, "answer": answer, "lines": list(lines),
            "limits": [RECORDED_NOTE, *limits], "covered": dict(covered) if covered is not None else None,
            "references": {name: list(values) for name, values in (references or {}).items()}}


# ---------------------------------------------------------------- who is in the room


def _recognition(person: Mapping[str, Any], face: str) -> str:
    association = person.get("association")
    state = person.get("identity")
    if face != "available":
        return f"recognition is {face}: not recognised"
    if isinstance(association, Mapping) and association.get("state") == "retained":
        return ("Known — retained from an earlier match; last successful match "
                f"{_ago(association.get('last_match_age_ms'))} ago")
    if state == "known":
        if person.get("basis") == "fresh":
            return f"Known · current match; newest face result {_ago(person.get('vote_age_ms'))} ago"
        return (f"Known, kept from earlier face results on this track (no current match; newest result "
                f"{_ago(person.get('vote_age_ms'))} ago)")
    if state == "unknown":
        return "not recognised: the faces seen on this track matched nobody enrolled"
    if state == "uncertain":
        return "uncertain: the face results on this track disagree"
    return "not recognised yet: no usable face result on this track"


def who_answer(view: Mapping[str, Any] | None) -> dict[str, Any]:
    """Who the runtime currently tracks, from its own published state (``DemoRuntime.occupants``)."""
    if view is None:
        return _document(WHO, "No data: the runtime has not published a state yet.", [], [NO_NAMES])
    video = str(view.get("video"))
    when = _clock(view.get("published_utc"))
    people = [p for p in view.get("people") or () if isinstance(p, Mapping)]
    confirmed = [p for p in people if p.get("status") == "confirmed"]
    face = str(view.get("face_recognition"))
    lines = []
    for person in people[:PEOPLE_LINES]:
        label = person.get("label")
        enrolled = person.get("identity") == "known" or isinstance(person.get("association"), Mapping)
        who = (f"“{clean_text(str(label), LABEL_MAX)}” (the operator's label for an enrolment)"
               if label and enrolled else "An enrolled person" if enrolled and face == "available" else "A person")
        status = "" if person.get("status") == "confirmed" else ", not yet confirmed by the tracker"
        lines.append(f"{who} (track {person.get('track_id')}, stream epoch {person.get('stream_epoch')}{status}): "
                     f"{_recognition(person, face)}.")
    omitted = len(people) - len(lines) + int(view.get("omitted") or 0)
    if omitted > 0:
        lines.append(f"{_plural(omitted, 'more track')} not listed.")
    limits = [NO_NAMES, EXPERIMENTAL] if face == "available" else [NO_NAMES]
    if video != "fresh":
        age = view.get("last_frame_age_ms")
        lead = (f"The video is {video} (last frame {_ago(age)} ago), so who is in the room now is not known. "
                f"Last known state, published {when}: {_plural(len(confirmed), 'confirmed person')} tracked.")
        limits.insert(0, "This is the last published state, not the room now: nothing is current while the video is "
                         f"{video}.")
    elif view.get("occupancy") == "empty":
        lead = f"Nobody is detected on fresh video (state published {when})."
        limits.insert(0, "Someone the detector does not see (out of view, hidden, too small) is not listed.")
    elif view.get("occupancy") == "unknown":
        lead = (f"Occupancy is unknown (state published {when}): "
                f"{clean_text(str(view.get('occupancy_reason') or 'no reason given'), 160)}.")
    else:
        lead = (f"{_plural(len(confirmed), 'confirmed person')} tracked on fresh video (state published {when}; "
                "the camera's view only).")
        limits.insert(0, "Someone the detector does not see (out of view, hidden, too small) is not listed.")
    tracks = [{"track_id": p.get("track_id"), "stream_epoch": p.get("stream_epoch")} for p in people[:PEOPLE_LINES]]
    return _document(WHO, lead, lines, limits, references={"tracks": tracks},
                     covered={"from_utc": view.get("published_utc"), "to_utc": view.get("published_utc"),
                              "requested_from_utc": None, "complete": video == "fresh"})


# ---------------------------------------------------------------- alerts: incidents apart from notifications


def read_alerts(db_path: Path, since_utc: datetime) -> dict[str, Any]:
    """Incidents created at or after ``since_utc`` and their notification rows: read only, three fixed queries, at
    most INCIDENTS_READ incidents (the rest counted)."""
    try:
        connection = connect_reader(db_path)
    except sqlite3.Error as exc:
        return {"state": "unavailable", "error": type(exc).__name__}
    try:
        connection.execute("BEGIN")
        try:
            since = _utc(since_utc)
            total = connection.execute("SELECT count(*) FROM incidents WHERE created_utc >= ?", (since,)).fetchone()[0]
            incidents = [dict(zip(("incident_id", "kind", "severity", "status", "title", "first_observed_utc",
                                   "created_utc"), row)) for row in connection.execute(
                """SELECT incident_id, kind, severity, status, title, first_observed_utc, created_utc
                   FROM incidents WHERE created_utc >= ? ORDER BY created_utc DESC, rowid DESC LIMIT ?""",
                (since, INCIDENTS_READ))]
            deliveries: dict[str, list[dict[str, Any]]] = {}
            if incidents:
                marks = ",".join("?" * len(incidents))
                for incident_id, channel, status, attempts, flagged, confirmed in connection.execute(
                        f"""SELECT incident_id, channel, status, attempts, ambiguous, provider_message_id IS NOT NULL
                            FROM outbox WHERE incident_id IN ({marks}) ORDER BY outbox_id""",
                        [item["incident_id"] for item in incidents]):
                    deliveries.setdefault(incident_id, []).append({
                        "channel": channel, "state": delivery_state(status, attempts, bool(flagged), bool(confirmed)),
                        "attempts": attempts})
        finally:
            connection.execute("COMMIT")
    except sqlite3.Error as exc:
        return {"state": "unavailable", "error": type(exc).__name__}
    finally:
        connection.close()
    for item in incidents:
        item["deliveries"] = deliveries.get(item["incident_id"], [])
    return {"state": "available", "total": total, "incidents": incidents, "since_utc": since}


DELIVERY_WORDS = {
    "queued": "queued, not yet attempted", "retrying": "attempted, will be retried", "sending": "being sent now",
    "delivered": "delivered (the provider confirmed it)",
    "sent_unconfirmed": "sent, without a confirmation ID from the provider", "failed": "failed (given up)",
    "ambiguous": "ambiguous: it may or may not have arrived",
}


def delivery_state(status: str, attempts: int, ambiguous: bool, confirmed: bool) -> str:
    """One outbox row's state for an answer: queued, retrying, sending, delivered, sent_unconfirmed, failed or
    ambiguous (an attempt whose outcome is not known, whatever its row says next)."""
    if status == "sent":
        return "delivered" if confirmed else "sent_unconfirmed"
    if ambiguous:
        return "ambiguous"
    if status == "dead":
        return "failed"
    if status == "leased":
        return "sending"
    return "retrying" if attempts > 0 else "queued"


def _incident_line(item: Mapping[str, Any]) -> str:
    rows = item.get("deliveries") or []
    if rows:
        sent = "; ".join(f"{clean_text(str(row['channel']), 24)}: {DELIVERY_WORDS[row['state']]}"
                         + (f" after {_plural(row['attempts'], 'attempt')}" if row["attempts"] else "")
                         for row in rows)
        notifications = f"notifications — {sent}"
    else:
        notifications = "no notification was queued for it"
    return (f"{_clock(item.get('created_utc'))} · {item.get('severity')} · {item.get('status')} · "
            f"“{clean_text(str(item.get('title')), TITLE_MAX)}” [incident {item.get('incident_id')}]: {notifications}.")


def alerts_answer(read: Mapping[str, Any], now_utc: datetime, channels: Sequence[str] | None) -> dict[str, Any]:
    """Incidents (what a rule or the scene check raised) apart from notifications (queued, sent, confirmed, failed,
    ambiguous). ``channels``: the notification channels this run configured (None: not known)."""
    covered = {"from_utc": read.get("since_utc"), "to_utc": _utc(now_utc),
               "requested_from_utc": read.get("since_utc"), "complete": read.get("state") == "available"}
    off = ("External notifications were off in this run (no channel is configured), so nothing could be queued or "
           f"delivered. {ZERO_DELIVERIES}") if channels is not None and not channels else None
    limits = [NOT_NOTHING, "Incidents are read from the incident database, which is kept across restarts."]
    if read.get("state") != "available":
        return _document(ALERTS, "No data: the incident database could not be read, so whether alerts were triggered "
                                 "is not known.", [off] if off else [], limits, covered=covered)
    incidents = read.get("incidents") or []
    total = int(read.get("total") or 0)
    interval = f"between {_clock(read.get('since_utc'))} and {_clock(_utc(now_utc))}"
    if not total:
        lines = [off] if off else ["No notification was queued in that interval either."]
        return _document(ALERTS, f"No incident was recorded {interval}.", lines, limits, covered=covered)
    severities = Counter(str(item.get("severity")) for item in incidents)
    states = Counter(row["state"] for item in incidents for row in item.get("deliveries") or [])
    lead = (f"{_plural(total, 'incident')} recorded {interval}"
            + (f" ({', '.join(f'{count} {name}' for name, count in sorted(severities.items()))})" if incidents else "")
            + ".")
    lines = [_incident_line(item) for item in incidents[:INCIDENT_LINES]]
    if total > len(lines):
        lines.append(f"{_plural(total - len(lines), 'earlier incident')} in that interval not listed.")
    if states:
        lines.append("Notifications for the listed incidents: " + ", ".join(
            f"{count} {DELIVERY_WORDS[name]}" for name, count in sorted(states.items())) + ".")
    elif off is None:
        lines.append(f"No notification was queued for these incidents. {ZERO_DELIVERIES}")
    if off:
        lines.append(off)
    return _document(ALERTS, lead, lines, limits, covered=covered,
                     references={"incidents": [item.get("incident_id") for item in incidents[:INCIDENT_LINES]]})


# ---------------------------------------------------------------- history: what was recorded in a period


def _event_line(item: Mapping[str, Any]) -> str:
    kind, at = item.get("kind"), _clock(item.get("utc"))
    track = f"track {item.get('track_id')} (stream epoch {item.get('stream_epoch')})"
    if kind == "runtime_started":
        text = "the runtime started; nothing earlier is in this history"
    elif kind == "video":
        text = f"the video became {item.get('to')}" + (f" (it was {item.get('from')})" if item.get("from") else "")
    elif kind == "stream_epoch":
        text = f"the camera stream reconnected (stream epoch {item.get('from')} to {item.get('to')})"
    elif kind == "occupancy":
        people = _plural(int(item.get("confirmed_people") or 0), "confirmed person")
        text = f"occupancy became {item.get('to')} ({people})"
    elif kind == "track_appeared":
        text = f"a person began to be tracked: {track}"
    elif kind == "track_gone":
        text = f"{track} is no longer tracked"
    elif kind == "identity":
        text = f"{track}: recognition went from {item.get('from')} to {item.get('to')}"
    elif kind == "association":
        text = (f"{track}: established identity went from {item.get('from')} to {item.get('to')} "
                f"({clean_text(str(item.get('reason')), 40)})")
    else:
        text = clean_text(str(kind), 40)
    return f"{at}: {text}."


def history_answer(window: Mapping[str, Any] | None, read: Mapping[str, Any] | None) -> dict[str, Any]:
    """What the runtime recorded in a period (``RecentHistory.window``), with the incidents the database holds for
    the same interval (``read_alerts``; None: not read)."""
    limits = [NOT_NOTHING, HISTORY_BOUNDS]
    if window is None:
        return _document(HISTORY, "No data: the runtime's history is not available (it is still starting).", [],
                         limits)
    covered = {"from_utc": window.get("covered_from_utc"), "to_utc": window.get("now_utc"),
               "requested_from_utc": window.get("requested_from_utc"), "complete": bool(window.get("complete"))}
    interval = f"{_clock(covered['from_utc'])} to {_clock(covered['to_utc'])}"
    lines = []
    if not covered["complete"]:
        why = ("the runtime started then, and its history is empty before that"
               if window.get("limited_by") == "runtime_start"
               else "older entries were dropped (the history's size and age limits)")
        lines.append(f"You asked from {_clock(covered['requested_from_utc'])}, but the history covers only "
                     f"{interval}: {why}. Before that there is no data.")
    events = [item for item in window.get("events") or () if isinstance(item, Mapping)]
    scenes = [item for item in window.get("scenes") or () if isinstance(item, Mapping)]
    shown = events[-EVENT_LINES:]
    if len(events) > len(shown):
        lines.append(f"{_plural(len(events) - len(shown), 'earlier event')} in the interval not listed.")
    lines += [_event_line(item) for item in shown]
    evidence = []
    if scenes:
        flagged = sum(1 for item in scenes if item.get("fire_or_smoke"))
        threats = Counter(str(item.get("threat")) for item in scenes)
        lines.append(f"{_plural(len(scenes), 'scene report')} in the interval; fire or smoke flagged in {flagged}; "
                     "threat levels reported: "
                     + ", ".join(f"{name} ×{count}" for name, count in sorted(threats.items())) + ".")
        for item in scenes[-SCENE_LINES:]:
            evidence.append(item.get("evidence_id"))
            lines.append(f"{_clock(item.get('utc'))}: the scene model reported (unverified, one frame): "
                         f"“{clean_text(str(item.get('summary')), 200)}”; persons visible "
                         f"{item.get('persons_visible')} [evidence {item.get('evidence_id')}].")
    incidents: list[Any] = []
    if read is not None and read.get("state") == "available":
        found = [item for item in read.get("incidents") or ()]
        incidents = [item.get("incident_id") for item in found[:INCIDENT_LINES]]
        lines.append(f"{_plural(int(read.get('total') or 0), 'incident')} recorded in the incident database since "
                     f"{_clock(read.get('since_utc'))}" + (": " + "; ".join(
                         f"{_clock(item.get('created_utc'))} {item.get('severity')} “"
                         f"{clean_text(str(item.get('title')), TITLE_MAX)}” [incident {item.get('incident_id')}]"
                         for item in found[:INCIDENT_LINES]) if found else "") + ".")
    elif read is not None:
        lines.append("No data on incidents: the incident database could not be read.")
    if not events and not scenes:
        lead = f"No events were recorded in the covered interval, {interval}."
    else:
        lead = (f"Recorded between {interval}: {_plural(len(events), 'event')} and "
                f"{_plural(len(scenes), 'scene report')}.")
    tracks = sorted({(item.get("stream_epoch"), item.get("track_id")) for item in events if "track_id" in item})
    return _document(HISTORY, lead, lines, limits, covered=covered, references={
        "incidents": incidents, "evidence": evidence,
        "tracks": [{"stream_epoch": epoch, "track_id": track} for epoch, track in tracks[:PEOPLE_LINES]]})


# ---------------------------------------------------------------- zoom and scan: a selection, never a guess


def guide_answer(intent: str, available: bool) -> dict[str, Any]:
    """The reply to a typed request to zoom or to scan (chat_routing ZOOM, SCAN). Typed words name no place that can
    be grounded, so nothing is guessed and the model is not asked: the reply points to the page's controls, where the
    viewer selects the region of a held picture (scene.roi)."""
    limits = ["A zoomed answer is about a crop cut from the held picture's own recorded pixels. It can show no more "
              "than was recorded: enlarging cannot recover what is unreadable.",
              "A scan covers what the camera saw in one held picture: not areas out of view, and not a camera "
              "movement."]
    if not available:
        return _document(intent, "Zooming into a region is not available in this run, and typed words cannot say "
                                 "where to look, so nothing was analysed.", [], limits)
    if intent == SCAN:
        return _document(SCAN, "To scan the visible room, press “Scan visible room” below the question box: one "
                               "picture is held and its four overlapping parts are asked about in turn.", [], limits)
    return _document(ZOOM, "To zoom, press “Zoom into a region” below the question box and select the region on the "
                           "held picture (drag, or pick a tracked person's box). Chat does not guess where to look.",
                     [], limits)


# ---------------------------------------------------------------- one entry point for the page


def recorded_answer(route: Route, *, now_utc: datetime, db_path: Path, who: Mapping[str, Any] | None,
                    window: Mapping[str, Any] | None, started_utc: datetime | None,
                    channels: Sequence[str] | None) -> dict[str, Any]:
    """The recorded-fact answer for a routed question (``chat_routing.route``; never the visual intent)."""
    if route.intent == WHO:
        return who_answer(who)
    since = started_utc if route.since_start and started_utc is not None else (
        now_utc - timedelta(seconds=route.period_s or 0))
    if route.intent == ALERTS:
        return alerts_answer(read_alerts(db_path, since), now_utc, channels)
    if route.intent == HISTORY:
        if window is not None and isinstance(window.get("covered_from_utc"), str):
            # the database for the interval the history actually covers, so the two halves describe the same time
            since = datetime.fromisoformat(window["covered_from_utc"].replace("Z", "+00:00"))
        return history_answer(window, read_alerts(db_path, since) if window is not None else None)
    raise ValueError("not a recorded intent")
