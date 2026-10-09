"""Loopback-only, read-only status page for ``sentinel run`` (D-2; demo only).

It stands in for V2-17/V2-18 at the Oct 20 demo and replaces neither: there are
no sessions, roles, API or media. It is reached over an SSH port forward.

- **Loopback only.** It binds 127.0.0.1 and nothing else; any other host is
  refused before a socket is opened. Requests must name 127.0.0.1 or localhost
  with the page's port in their Host header, so a web page on another origin
  cannot read it through DNS rebinding.
- **Read only.** GET ``/`` (HTML, refreshed every 2 s), GET ``/status.json``,
  and one incident's detail at GET ``/incidents/<id>`` (HTML, refreshed every
  10 s) and ``/incidents/<id>.json``. Every other method is refused. The
  database is read through a new read-only SQLite connection per request
  (``connect_reader``), in one short read transaction, so the page never
  writes, migrates or holds the writer's lock.
- **What it shows.** Component readiness, capture and reconnect state, video
  freshness, processing rates, degradation reasons, recent incidents and
  delivery state. Delivery separates *queued* (never attempted), *attempted*
  (sent at least once, not confirmed: retrying or in flight), *delivered* (the
  provider confirmed it) and *failed* (dead-lettered, needs an operator).
  *Ambiguous* marks rows the provider may have accepted although that was not
  confirmed.
- **Incident detail** (guide ch. 15's evidence times, rule reason, VLM summary,
  clip availability, acknowledgment and delivery attempts). It shows the rule
  observations with their source times, track, stream epoch and frames; the
  lifecycle transitions; the worker evidence that annotated the incident, with
  the applicability it was routed with on arrival (only ``current`` could have
  changed live state; anything else is history for this incident only, V2-28);
  and each alert row with its attempts. Every list is bounded (the latest
  rows, with the total). It shows whether, when and by whom the incident was
  acknowledged, but has no acknowledge action: that is ``sentinel incident
  ack``, run while monitoring is stopped (D68). There is no clip: the page says
  so instead of hiding it.
- **What it never shows.** Credentials, tokens, chat IDs, stream URLs, hosts,
  images or exception text. The runtime snapshot holds numbers and fixed
  labels only; stored delivery errors were redacted when written; every value
  is HTML-escaped. Stored payloads are shown only as fields of a contract this
  build validates (zone and hazard observations, scene evidence); the details
  of any other evidence kind, identity evidence included, are not shown.
  Responses are not cached and may not be framed.

A standard-library HTTP server; the stored contracts are validated with the
project's pydantic models. No new dependency; portable.
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from .contracts import Evidence, EvidenceStatus
from .incidents.service import UNRESOLVED, latest_acknowledgment
from .rules.scene_hazard import FIRE_SMOKE_CANDIDATE, HazardCandidate
from .rules.zones import ZONE_KINDS, ZoneObservation
from .scene.report import SceneReport
from .scene.state import SCENE_STATE_KIND
from .storage.database import connect_reader

LOOPBACK_HOST = "127.0.0.1"
MAX_INCIDENTS = 20
MAX_DELIVERIES = 30
REFRESH_S = 2
# Incident detail. IDs with '.' or ':' (allowed by the contracts) get no page; generated IDs never have them.
INCIDENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
INCIDENT_PATH = re.compile(r"/incidents/(?P<id>[A-Za-z0-9][A-Za-z0-9_-]{0,127})(?P<json>\.json)?")
DETAIL_REFRESH_S = 10
MAX_EVIDENCE = 50
MAX_TRANSITIONS = 20
MAX_ANNOTATIONS = 20
MAX_INCIDENT_DELIVERIES = 10
MAX_ATTEMPTS = 10  # per alert row
MAX_LATER_INCIDENTS = 5
ACKNOWLEDGE_HOW = (
    "this page is read only (D45): stop monitoring, run `sentinel incident ack --data-dir DIR ID`, then start it "
    "again (monitoring is interrupted meanwhile; D68)"
)
CLIP_UNAVAILABLE = "none: event clips are not implemented yet (V2-21)"
UTC_TEXT = "%Y-%m-%dT%H:%M:%S.%fZ"  # incidents.service.utc_text
STALE_SNAPSHOT_S = 5.0  # an older runtime snapshot means the loop is not updating
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
}
DELIVERY_STATES = ("queued", "attempted", "delivered", "failed")
# Shown even before the runtime publishes its own wording (D43 limitation).
PENDING_DURABILITY_FALLBACK = (
    "rule observations waiting to be recorded are held in memory only; "
    "they are lost if the process stops abruptly: not a crash-safe spool"
)


def delivery_state(status: str, attempts: int) -> str:
    """queued, attempted, delivered or failed, from an outbox row's status and attempt count."""
    if status == "sent":
        return "delivered"
    if status == "dead":
        return "failed"
    return "attempted" if attempts > 0 else "queued"


def read_store(db_path: Path) -> dict[str, Any]:
    """Recent incidents and deliveries from the database, read only."""
    try:
        connection = connect_reader(db_path)
    except sqlite3.Error as exc:
        return _unavailable(exc)
    try:
        connection.execute("BEGIN")
        try:
            placeholders = ",".join("?" * len(UNRESOLVED))
            unresolved = connection.execute(
                f"SELECT count(*) FROM incidents WHERE status IN ({placeholders})", UNRESOLVED
            ).fetchone()[0]
            incidents = [
                dict(zip(("incident_id", "kind", "zone_id", "severity", "status", "title",
                          "first_observed_utc", "last_observed_utc", "annotations"), row))
                for row in connection.execute(
                    """SELECT i.incident_id, i.kind, i.zone_id, i.severity, i.status, i.title,
                              i.first_observed_utc, i.last_observed_utc,
                              (SELECT count(*) FROM incident_annotations a WHERE a.incident_id = i.incident_id)
                       FROM incidents i ORDER BY i.created_utc DESC, i.rowid DESC LIMIT ?""",
                    (MAX_INCIDENTS,),
                )
            ]
            totals = {state: 0 for state in DELIVERY_STATES}
            ambiguous = 0
            for status, attempted, flagged, count in connection.execute(
                "SELECT status, attempts > 0, ambiguous, count(*) FROM outbox GROUP BY 1, 2, 3"
            ):
                totals[delivery_state(status, attempted)] += count
                ambiguous += count if flagged else 0
            deliveries = [
                _delivery_row(*row) for row in connection.execute(
                    f"SELECT {OUTBOX_COLUMNS} FROM outbox ORDER BY outbox_id DESC LIMIT ?", (MAX_DELIVERIES,)
                )
            ]
        finally:
            connection.execute("COMMIT")
    except sqlite3.Error as exc:
        return _unavailable(exc)
    finally:
        connection.close()
    return {
        "state": "available",
        "incidents": {"unresolved": unresolved, "recent": incidents},
        "delivery": {"totals": {**totals, "ambiguous": ambiguous}, "recent": deliveries},
    }


OUTBOX_COLUMNS = (
    "outbox_id, incident_id, channel, message_kind, status, attempts, ambiguous, last_error, next_attempt_utc, "
    "updated_utc, provider_message_id IS NOT NULL"
)


def _delivery_row(
    outbox_id: int, incident_id: str, channel: str, kind: str, status: str, attempts: int, flagged: int,
    error: str | None, next_utc: str, updated: str, confirmed: int,
) -> dict[str, Any]:
    """One outbox row as shown (``OUTBOX_COLUMNS``); the provider's message ID itself is never shown."""
    state = delivery_state(status, attempts)
    return {
        "outbox_id": outbox_id, "incident_id": incident_id, "channel": channel, "message_kind": kind,
        "state": state, "in_flight": status == "leased", "attempts": attempts,
        "ambiguous": bool(flagged), "provider_confirmed": bool(confirmed), "last_error": error,
        "next_attempt_utc": next_utc if state in ("queued", "attempted") else None,
        "updated_utc": updated,
    }


def _unavailable(exc: Exception) -> dict[str, Any]:
    return {"state": "unavailable", "problem": f"database_unreadable:{type(exc).__name__}",
            "incidents": None, "delivery": None}


# ---------------------------------------------------------------- incident detail


def read_incident(db_path: Path, incident_id: str) -> dict[str, Any]:
    """One incident's evidence chain, read only. ``state`` is available, not_found or unavailable."""
    if not INCIDENT_ID.fullmatch(incident_id):
        return {"state": "not_found"}
    try:
        connection = connect_reader(db_path)
    except sqlite3.Error as exc:
        return {"state": "unavailable", "problem": f"database_unreadable:{type(exc).__name__}"}
    try:
        connection.execute("BEGIN")
        try:
            return _incident_detail(connection, incident_id)
        finally:
            connection.execute("COMMIT")
    except sqlite3.Error as exc:
        return {"state": "unavailable", "problem": f"database_unreadable:{type(exc).__name__}"}
    finally:
        connection.close()


INCIDENT_COLUMNS = (
    "incident_id", "camera_id", "kind", "zone_id", "severity", "status", "revision", "title", "linked_incident_id",
    "first_observed_utc", "last_observed_utc", "created_utc", "updated_utc",
)
EVIDENCE_COLUMNS = (
    "evidence_seq", "observation_id", "episode_id", "kind", "phase", "severity", "observed_utc", "reason", "payload",
)
TRANSITION_COLUMNS = ("transition_seq", "from_status", "to_status", "revision", "actor", "reason", "at_utc")
ANNOTATION_COLUMNS = (
    "annotation_seq", "kind", "status", "producer", "producer_revision", "applicability", "observed_utc",
    "recorded_utc", "payload",
)
ATTEMPT_COLUMNS = ("attempt", "started_utc", "finished_utc", "outcome", "detail")


def _incident_detail(c: sqlite3.Connection, incident_id: str) -> dict[str, Any]:
    row = c.execute(
        f"SELECT {', '.join(INCIDENT_COLUMNS)} FROM incidents WHERE incident_id = ?", (incident_id,)
    ).fetchone()
    if row is None:
        return {"state": "not_found"}
    later = [r[0] for r in c.execute(
        "SELECT incident_id FROM incidents WHERE linked_incident_id = ? ORDER BY created_utc, rowid LIMIT ?",
        (incident_id, MAX_LATER_INCIDENTS),
    )]
    evidence = _latest(c, "incident_evidence", EVIDENCE_COLUMNS, incident_id, MAX_EVIDENCE)
    episodes: dict[str, int] = {}
    for item in evidence["shown"]:
        item["episode"] = episodes.setdefault(item["episode_id"], len(episodes) + 1)  # 1, 2, ... in shown order
        item["source"] = _rule_facts(item["kind"], item.pop("payload"))
    annotations = _latest(c, "incident_annotations", ANNOTATION_COLUMNS, incident_id, MAX_ANNOTATIONS)
    for item in annotations["shown"]:
        item["recorded_after_source_s"] = _seconds_between(item["observed_utc"], item["recorded_utc"])
        item["details"], item["details_problem"] = _annotation_details(item["kind"], item.pop("payload"))
    total = c.execute("SELECT count(*) FROM outbox WHERE incident_id = ?", (incident_id,)).fetchone()[0]
    deliveries = []
    for outbox_row in c.execute(
        f"SELECT {OUTBOX_COLUMNS} FROM outbox WHERE incident_id = ? ORDER BY outbox_id LIMIT ?",
        (incident_id, MAX_INCIDENT_DELIVERIES),
    ).fetchall():
        delivery = _delivery_row(*outbox_row)
        del delivery["incident_id"]
        delivery["attempt_log"] = _latest(
            c, "delivery_attempts", ATTEMPT_COLUMNS, delivery["outbox_id"], MAX_ATTEMPTS, key="outbox_id",
            order="attempt_seq",
        )
        deliveries.append(delivery)
    incident = dict(zip(INCIDENT_COLUMNS, row))
    seen = latest_acknowledgment(c, incident_id)
    acknowledgment = {"state": "not_acknowledged", "at_utc": None, "by": None, "note": None, "how": ACKNOWLEDGE_HOW}
    if seen is not None:
        acknowledgment.update(state="acknowledged", at_utc=seen.at_utc, by=seen.actor, note=seen.reason)
    return {
        "state": "available",
        "incident": incident,
        "linked": {"previous": incident["linked_incident_id"], "later": later},
        "acknowledgment": acknowledgment,
        "clip": CLIP_UNAVAILABLE,
        "evidence": evidence,
        "transitions": _latest(c, "incident_transitions", TRANSITION_COLUMNS, incident_id, MAX_TRANSITIONS),
        "annotations": annotations,
        "deliveries": {"total": total, "shown": deliveries},
    }


def _latest(
    c: sqlite3.Connection, table: str, columns: tuple[str, ...], owner: object, limit: int, *,
    key: str = "incident_id", order: str | None = None,
) -> dict[str, Any]:
    """The latest ``limit`` rows of ``table`` owned by ``owner``, oldest first, and how many there are."""
    order = order or columns[0]
    total = c.execute(f"SELECT count(*) FROM {table} WHERE {key} = ?", (owner,)).fetchone()[0]
    rows = c.execute(
        f"SELECT {', '.join(columns)} FROM {table} WHERE {key} = ? ORDER BY {order} DESC LIMIT ?", (owner, limit)
    ).fetchall()
    return {"total": total, "shown": [dict(zip(columns, r)) for r in reversed(rows)]}


def _rule_facts(kind: str, payload: str) -> dict[str, Any] | None:
    """What a stored rule observation says about its source; None if it is not a contract this build reads."""
    try:
        if kind in ZONE_KINDS.values():
            zone = ZoneObservation.model_validate_json(payload)
            return {
                "track_id": zone.track.track_id, "stream_epoch": zone.track.stream_epoch,
                "first_frame_seq": zone.first_source.frame_seq, "last_frame_seq": zone.last_source.frame_seq,
                "in_zone_s": round(zone.duration_ms / 1000, 1), "zone_revision": zone.zone_revision,
            }
        if kind == FIRE_SMOKE_CANDIDATE:
            hazard = HazardCandidate.model_validate_json(payload)
            return {
                "stream_epoch": hazard.last_source.stream_epoch, "first_frame_seq": hazard.first_source.frame_seq,
                "last_frame_seq": hazard.last_source.frame_seq, "scene_reports": len(hazard.evidence_ids),
                "independent_detector_agreed": hazard.primary_signal, "people_count": hazard.people_count,
            }
    except ValidationError:
        return None
    return None


def _annotation_details(kind: str, payload: str) -> tuple[dict[str, Any] | None, str | None]:
    """Scene evidence's reason, source frame and validated report; nothing for any other kind."""
    if not kind.startswith("scene."):
        return None, "not_scene_evidence"
    try:
        evidence = Evidence.model_validate_json(payload)
    except ValidationError:
        return None, "unreadable"
    report, report_problem = None, None
    if evidence.kind == SCENE_STATE_KIND and evidence.status is EvidenceStatus.OBSERVED:
        try:
            report = SceneReport.model_validate_json(json.dumps(evidence.value)).model_dump(mode="json")
        except ValidationError:
            report_problem = "not_a_valid_scene_report"
    details = {"reason": evidence.reason, "stream_epoch": evidence.source.stream_epoch,
               "frame_seq": evidence.source.frame_seq, "report": report, "report_problem": report_problem}
    return details, None


def _seconds_between(earlier: str, later: str) -> float | None:
    """Wall-clock seconds between two stored UTC texts, for display only."""
    try:
        return round((datetime.strptime(later, UTC_TEXT) - datetime.strptime(earlier, UTC_TEXT)).total_seconds(), 1)
    except (TypeError, ValueError):
        return None


def build_status(snapshot: Mapping[str, Any], store: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    """The /status.json document: the runtime snapshot plus the stored incidents and deliveries."""
    runtime = dict(snapshot.get("runtime") or {"state": "starting"})
    degraded = list(snapshot.get("degraded") or [])
    age = _age_s(runtime.get("updated_utc"), now)
    if age is not None and age > STALE_SNAPSHOT_S and runtime.get("state") == "running":
        degraded.insert(0, f"runtime status not updated for {age:.0f} s")
    if store.get("state") != "available":
        degraded.append(f"incident database unavailable ({store.get('problem')})")
    return {
        "generated_utc": now.isoformat(),
        "runtime": {**runtime, "status_age_s": None if age is None else round(age, 1)},
        "degraded": degraded,
        "components": snapshot.get("components") or {},
        "frames": snapshot.get("frames") or {},
        "rates": snapshot.get("rates") or {},
        "live": snapshot.get("live"),
        "database": {"state": store.get("state"), "problem": store.get("problem")},
        "incidents": store.get("incidents"),
        "delivery": store.get("delivery"),
    }


def _age_s(stamp: object, now: datetime) -> float | None:
    if not isinstance(stamp, str):
        return None
    try:
        return (now - datetime.fromisoformat(stamp)).total_seconds()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------- HTML


class _Markup(str):
    """HTML built here from escaped parts; ``_e`` passes it through."""


def _e(value: object) -> str:
    if isinstance(value, _Markup):
        return value
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return html.escape(str(value), quote=True)


def _incident_link(incident_id: object) -> object:
    """A link to the incident's detail page; an ID with no page is plain text."""
    if isinstance(incident_id, str) and INCIDENT_ID.fullmatch(incident_id):
        return _Markup(f'<a href="/incidents/{_e(incident_id)}">{_e(incident_id)}</a>')
    return incident_id


STYLE = """:root{--bg:#fff;--fg:#1d1f21;--muted:#666;--line:#ddd;--ok:#e6f4ea;--bad:#fdecea;--link:#0b57d0}
@media (prefers-color-scheme:dark){:root{--bg:#16181a;--fg:#e8e8e8;--muted:#9a9a9a;--line:#333;--ok:#173322;--bad:#3b1d1a;--link:#8ab4f8}}
body{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:16px;max-width:1100px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:16px;margin:20px 0 6px} a{color:var(--link)}
table{border-collapse:collapse;margin:4px 0} th,td{border-bottom:1px solid var(--line);padding:3px 10px 3px 0;text-align:left;vertical-align:top}
table.list{width:100%} .muted{color:var(--muted)} .banner{padding:8px 12px;border-radius:6px;margin:10px 0}
.ok{background:var(--ok)} .bad{background:var(--bad)} .grid{display:flex;flex-wrap:wrap;gap:24px}
.wrap{overflow-x:auto} ul{margin:4px 0 0 18px;padding:0}"""


def _document(title: str, refresh_s: int, body: str) -> str:
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<meta http-equiv=refresh content="{refresh_s}"><title>{_e(title)}</title>
<style>
{STYLE}
</style></head><body>
{body}
</body></html>
"""


def _identity_counts(live: dict[str, Any]) -> str | None:
    counts = live.get("identity") if live else None
    if not counts:
        return None
    return (f"known {counts.get('known')} (fresh {counts.get('fresh')}, retained {counts.get('retained')}) · "
            f"unknown {counts.get('unknown')} · uncertain {counts.get('uncertain')} · unresolved {counts.get('unresolved')}")


def _rows(pairs: list[tuple[str, object]]) -> str:
    return "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in pairs)


def _table(headers: list[str], rows: list[list[object]], empty: str) -> str:
    if not rows:
        return f"<p class=muted>{_e(empty)}</p>"
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table class=list><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(status: Mapping[str, Any]) -> str:
    runtime = status["runtime"]
    components = status["components"]
    live = status.get("live") or {}
    rates = status.get("rates") or {}
    frames = status.get("frames") or {}
    degraded = status["degraded"]
    banner = (
        "<div class='banner bad'><strong>Degraded</strong><ul>"
        + "".join(f"<li>{_e(reason)}</li>" for reason in degraded) + "</ul></div>"
        if degraded else "<div class='banner ok'><strong>All components working</strong></div>"
    )
    component_rows = []
    for name in ("capture", "detector", "scene", "face", "incidents"):
        item = components.get(name) or {}
        component_rows.append([name, item.get("state"), item.get("problem")])
    notifications = components.get("notifications") or {}
    for channel, item in (notifications.get("channels") or {}).items():
        component_rows.append([f"notifier: {channel}", item.get("state"), item.get("problem")])
    worker = notifications.get("worker") or {}
    component_rows.append(["delivery worker", worker.get("state"), worker.get("problem")])
    database = status.get("database") or {}
    component_rows.append(["incident database (read)", database.get("state"), database.get("problem")])
    capture = components.get("capture") or {}
    report = live.get("scene_report") or {}
    scene_rows = [("scene", live.get("scene")), ("reason", live.get("scene_reason"))]
    if report:
        scene_rows += [(key.replace("_", " "), report.get(key))
                       for key in ("persons_visible", "fire_or_smoke", "threat", "uncertainty", "summary")]
    incidents = status.get("incidents") or {}
    recording = components.get("incidents") or {}
    durability = recording.get("durability") or PENDING_DURABILITY_FALLBACK
    delivery = status.get("delivery") or {}
    totals = delivery.get("totals") or {}
    incident_rows = [
        [_incident_link(i["incident_id"]), i["title"], i["severity"], i["status"], i["first_observed_utc"],
         i["last_observed_utc"], i["annotations"]]
        for i in incidents.get("recent") or []
    ]
    delivery_rows = [
        [_incident_link(d["incident_id"]), d["channel"], d["message_kind"],
         d["state"] + (" (in flight)" if d["in_flight"] else ""), d["attempts"],
         "may have been delivered" if d["ambiguous"] else "", d["next_attempt_utc"], d["last_error"]]
        for d in delivery.get("recent") or []
    ]
    return _document("Sentinel status", REFRESH_S, f"""<h1>Sentinel status: camera {_e(runtime.get("camera_id"))}</h1>
<p class=muted>Runtime {_e(runtime.get("state"))}; status from {_e(runtime.get("updated_utc"))}
({_e(runtime.get("status_age_s"))} s old). Read-only, loopback only; refreshes every {REFRESH_S} s.</p>
{banner}
<div class=grid>
<section><h2>Components</h2>{_table(["component", "state", "problem"], component_rows, "none")}</section>
<section><h2>Capture</h2><table>{_rows([
        ("state", capture.get("state")), ("stream epoch", capture.get("stream_epoch")),
        ("connects", capture.get("connects")), ("reconnects", capture.get("reconnects")),
        ("open failures", capture.get("open_failures")), ("stream ends", capture.get("stream_ends")),
        ("retry in (s)", capture.get("retry_delay_s")), ("last problem", capture.get("problem"))])}</table></section>
<section><h2>Processing</h2><table>{_rows([
        ("captured fps", rates.get("captured_fps")), ("processed fps", rates.get("processed_fps")),
        ("detector failures /s", rates.get("failed_per_s")), ("window (s)", rates.get("window_s")),
        ("frames captured", frames.get("captured")), ("processed", frames.get("processed")),
        ("failed", frames.get("failed")), ("replaced (skipped by a slow loop)", frames.get("replaced")),
        ("not live (epoch ended)", frames.get("not_live"))])}</table></section>
<section><h2>Live</h2><table>{_rows([
        ("video", live.get("video")), ("last frame age (ms)", live.get("last_frame_age_ms")),
        ("detector", live.get("detector")), ("occupancy", live.get("occupancy")),
        ("why", live.get("occupancy_reason")), ("people (confirmed)",
        None if not live else f"{live.get('people')} ({live.get('confirmed_people')})"),
        ("face recognition", live.get("face_recognition")), ("identities (counts only)", _identity_counts(live)),
        ("scene analysis", live.get("scene_analysis"))])}</table>
<table>{_rows(scene_rows)}</table></section>
</div>
<h2>Incidents</h2><p>{_e(incidents.get("unresolved"))} unresolved; latest {MAX_INCIDENTS} shown.
Waiting to be recorded: {_e(recording.get("pending_signals"))}.</p>
<p class=muted>Limitation: {_e(durability)}.</p>
<div class=wrap>{_table(["incident", "title", "severity", "status", "first seen (UTC)", "latest (UTC)", "annotations"],
                         incident_rows, "No incidents.")}</div>
<h2>Alert delivery</h2>
<p>Queued (never attempted) {_e(totals.get("queued"))} · attempted, not confirmed {_e(totals.get("attempted"))}
· delivered (provider confirmed) {_e(totals.get("delivered"))} · failed (needs operator) {_e(totals.get("failed"))}
· ambiguous {_e(totals.get("ambiguous"))}.
<span class=muted>Delivery is at least once: an ambiguous attempt may have reached the chat, and a repeat carries the same incident ID.</span></p>
<div class=wrap>{_table(["incident", "channel", "message", "state", "attempts", "ambiguous", "next attempt (UTC)", "last error"],
                         delivery_rows, "No alerts.")}</div>""")


APPLICABILITY_NOTE = (
    "Applicability is how the core routed each result when it arrived. Only 'current' could have changed the live "
    "scene; any other value (expired, superseded_epoch, no_live_stream, ...) is history for this incident only and "
    "never changed its status, severity or alerts."
)


def _report_text(details: Mapping[str, Any] | None) -> str | None:
    details = details or {}
    report = details.get("report")
    if details.get("report_problem"):
        return f"not shown ({details['report_problem']})"
    if not report:
        return None
    return (f"threat {report['threat']} · fire or smoke {'yes' if report['fire_or_smoke'] else 'no'} · "
            f"persons {report['persons_visible']} · uncertainty {report['uncertainty']} · {report['summary']}")


def _acknowledgment_text(acknowledgment: Mapping[str, Any]) -> str:
    if acknowledgment["state"] != "acknowledged":
        return f"not acknowledged; {acknowledgment['how']}"
    return f"acknowledged {acknowledgment['at_utc']} by {acknowledgment['by']}: {acknowledgment['note']}"


def _frames(source: Mapping[str, Any] | None) -> str | None:
    if not source:
        return None
    return f"{source['first_frame_seq']}–{source['last_frame_seq']}"


def render_incident_html(detail: Mapping[str, Any]) -> str:
    """One incident's page, from ``read_incident``'s document (state available)."""
    incident = detail["incident"]
    linked = detail["linked"]
    evidence, transitions = detail["evidence"], detail["transitions"]
    annotations, deliveries = detail["annotations"], detail["deliveries"]
    evidence_rows = []
    for item in evidence["shown"]:
        source = item["source"] or {}
        evidence_rows.append([
            item["observed_utc"], item["phase"], item["severity"], item["episode"], source.get("track_id"),
            source.get("stream_epoch"), _frames(item["source"]),
            source.get("in_zone_s") if "in_zone_s" in source else source.get("scene_reports"),
            item["reason"] if item["source"] is not None else f"{item['reason']} (source fields not readable)",
        ])
    transition_rows = [
        [t["at_utc"], t["from_status"], t["to_status"], t["revision"], t["actor"], t["reason"]]
        for t in transitions["shown"]
    ]
    annotation_rows = []
    for item in annotations["shown"]:
        details = item["details"] or {}
        annotation_rows.append([
            item["recorded_utc"], item["observed_utc"], item["recorded_after_source_s"], item["applicability"],
            item["kind"], item["status"], f"{item['producer']} {item['producer_revision']}",
            details.get("stream_epoch"), details.get("frame_seq"),
            details.get("reason") if item["details"] is not None else f"not shown ({item['details_problem']})",
            _report_text(item["details"]),
        ])
    delivery_rows, attempt_rows = [], []
    for d in deliveries["shown"]:
        delivery_rows.append([
            d["channel"], d["message_kind"], d["state"] + (" (in flight)" if d["in_flight"] else ""), d["attempts"],
            "may have been delivered" if d["ambiguous"] else "", d["provider_confirmed"], d["next_attempt_utc"],
            d["last_error"],
        ])
        for a in d["attempt_log"]["shown"]:
            attempt_rows.append([f"{d['channel']} {d['message_kind']}", a["attempt"], a["started_utc"],
                                 a["finished_utc"], a["outcome"], a["detail"]])
    later = _Markup(", ".join(_e(_incident_link(i)) for i in linked["later"])) if linked["later"] else None
    summary = _rows([
        ("title", incident["title"]), ("rule", incident["kind"]), ("zone", incident["zone_id"]),
        ("severity", incident["severity"]), ("status", incident["status"]), ("revision", incident["revision"]),
        ("first seen (UTC)", incident["first_observed_utc"]), ("latest (UTC)", incident["last_observed_utc"]),
        ("recorded (UTC)", incident["created_utc"]), ("updated (UTC)", incident["updated_utc"]),
        ("previous linked incident", _incident_link(linked["previous"])), ("later linked incidents", later),
        ("acknowledgment", _acknowledgment_text(detail["acknowledgment"])), ("clip", detail["clip"]),
    ])

    def shown(part: Mapping[str, Any]) -> str:
        return f"<p class=muted>{len(part['shown'])} of {part['total']} shown (the latest).</p>"

    title = f"Sentinel incident {incident['incident_id']}"
    return _document(title, DETAIL_REFRESH_S, f"""<p><a href="/">Back to status</a></p>
<h1>Incident {_e(incident["incident_id"])}</h1>
<p class=muted>Camera {_e(incident["camera_id"])}. Read-only, loopback only; refreshes every {DETAIL_REFRESH_S} s.
Generated {_e(detail.get("generated_utc"))}.</p>
<table>{summary}</table>
<h2>Rule observations</h2>{shown(evidence)}
<p class=muted>Source times are the ingest times of the frames each observation came from (UTC, for display).
An episode is one continuous presence of one track; ENDED closes it and changes no status.</p>
<div class=wrap>{_table(["source time (UTC)", "phase", "severity", "episode", "track", "stream epoch", "frames",
                         "in zone (s) / scene reports", "reason"], evidence_rows, "No rule observations.")}</div>
<h2>Lifecycle</h2>{shown(transitions)}
<div class=wrap>{_table(["at (UTC)", "from", "to", "revision", "by", "reason"], transition_rows, "No transitions.")}</div>
<h2>Scene evidence on this incident</h2>{shown(annotations)}
<p class=muted>{_e(APPLICABILITY_NOTE)} "Recorded after source" is a wall-clock difference, for display only.</p>
<div class=wrap>{_table(["recorded (UTC)", "source frame (UTC)", "recorded after source (s)", "applicability", "kind",
                         "status", "producer", "stream epoch", "frame", "reason", "report"],
                        annotation_rows, "No scene evidence.")}</div>
<h2>Alert delivery</h2><p class=muted>{len(deliveries["shown"])} of {deliveries["total"]} alert rows shown.
Delivery is at least once: an ambiguous attempt may have reached the chat, and a repeat carries the same incident ID.</p>
<div class=wrap>{_table(["channel", "message", "state", "attempts", "ambiguous", "provider confirmed",
                         "next attempt (UTC)", "last error"], delivery_rows, "No alerts for this incident.")}</div>
<p class=muted>Attempts: the latest {MAX_ATTEMPTS} per alert row, oldest first; details were redacted when stored.</p>
<div class=wrap>{_table(["alert", "attempt", "started (UTC)", "finished (UTC)", "outcome", "detail"], attempt_rows,
                        "No delivery attempts.")}</div>""")


# ---------------------------------------------------------------- server


class StatusServer:
    """Serves the page on 127.0.0.1:``port`` from its own thread; ``port`` 0 picks a free port (tests)."""

    def __init__(
        self,
        port: int,
        snapshot: Callable[[], Mapping[str, Any]],
        db_path: Path,
        *,
        host: str = LOOPBACK_HOST,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if host != LOOPBACK_HOST:
            raise ValueError(f"the status page binds {LOOPBACK_HOST} only")
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        self._snapshot = snapshot
        self._db_path = Path(db_path)
        self._now = now
        handler = type("StatusHandler", (_Handler,), {"page": self})
        self._httpd = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    def status(self) -> dict[str, Any]:
        return build_status(self._snapshot(), read_store(self._db_path), self._now())

    def incident(self, incident_id: str) -> dict[str, Any]:
        return {"generated_utc": self._now().isoformat(), **read_incident(self._db_path, incident_id)}

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the status page can be started once")
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.25}, name="sentinel-status", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float) -> bool:
        """Stop serving and close the socket; False if the serving thread is still running at the timeout."""
        if self._thread is not None:
            stopper = threading.Thread(target=self._httpd.shutdown, daemon=True)
            stopper.start()
            stopper.join(timeout_s)
            self._thread.join(max(0.0, timeout_s))
        self._httpd.server_close()
        return self._thread is None or not self._thread.is_alive()


class _Handler(BaseHTTPRequestHandler):
    page: StatusServer
    server_version = "sentinel-status"
    sys_version = ""

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if not self._host_allowed():
            self._send(421, "text/plain; charset=utf-8", b"misdirected request\n")
            return
        path = urlsplit(self.path).path
        incident = INCIDENT_PATH.fullmatch(path)
        if path not in ("/", "/status.json") and incident is None:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")
            return
        try:
            if incident is not None:
                detail = self.page.incident(incident["id"])
                if detail["state"] == "not_found":  # the requested ID is never echoed
                    self._send(404, "text/plain; charset=utf-8", b"not found\n")
                    return
                if detail["state"] != "available":
                    self._send(503, "text/plain; charset=utf-8", b"incident database unavailable\n")
                    return
                if incident["json"]:
                    body, kind = json.dumps(detail, indent=2).encode(), "application/json"
                else:
                    body, kind = render_incident_html(detail).encode(), "text/html; charset=utf-8"
            elif path == "/status.json":
                body, kind = json.dumps(self.page.status(), indent=2).encode(), "application/json"
            else:
                body, kind = render_html(self.page.status()).encode(), "text/html; charset=utf-8"
        except Exception:  # noqa: BLE001 - never show exception text
            self._send(500, "text/plain; charset=utf-8", b"status unavailable\n")
            return
        self._send(200, kind, body)

    def _refuse(self) -> None:
        self._send(405, "text/plain; charset=utf-8", b"read-only: GET only\n", allow="GET")

    do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _refuse  # noqa: N815

    def _host_allowed(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in (f"{LOOPBACK_HOST}:{port}", f"localhost:{port}")

    def _send(self, code: int, kind: str, body: bytes, *, allow: str | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        if allow is not None:
            self.send_header("Allow", allow)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - no request logging to stderr
        return
