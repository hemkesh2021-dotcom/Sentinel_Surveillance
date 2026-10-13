"""Loopback-only, read-only status page for ``sentinel run`` (D-2; demo only).

It stands in for V2-17/V2-18 at the Oct 20 demo and replaces neither: there are
no sessions, roles, API or media. It is reached over an SSH port forward.

- **Loopback only.** It binds 127.0.0.1 and nothing else; any other host is
  refused before a socket is opened. Requests must name 127.0.0.1 or localhost
  with the page's port in their Host header, so a web page on another origin
  cannot read it through DNS rebinding.
- **Read only.** GET ``/`` (HTML), GET ``/status.json``, and one incident's
  detail at GET ``/incidents/<id>`` (HTML) and ``/incidents/<id>.json``. Every
  other method is refused. The
  database is read through a new read-only SQLite connection per request
  (``connect_reader``), in one short read transaction, so the page never
  writes, migrates or holds the writer's lock.
- **Live view (D69; only with ``sentinel run --live-preview``).** The main
  page shows the newest frame the runtime already reads, with its state
  (live, stale, offline, starting), its age and its ingest time, and a
  picture only while that frame is live, at GET ``/preview/<epoch>-<seq>.jpg``
  (``live_preview``). The picture refreshes with the page. At most
  MAX_PICTURE_RESPONSES pictures are sent at once.
- **A page that stops updating claims nothing current (D71).** Every claim
  that something is current (the page itself, the runtime's healthy state,
  LIVE, the picture) is shown only for PAGE_CURRENT_S, by CSS alone; after
  that the page says it is NOT CURRENT and shows the runtime's report as last
  reported, dimmed. The claims are hidden unless the animation shows them, so
  a browser that runs no CSS animation never shows them.
- **In-place updates (D73).** The pages never reload themselves. Their one
  script (``SCRIPT_PATH``, same origin) polls ``/status.json?page=1`` every
  REFRESH_S (``/incidents/<id>.json?page=1`` every DETAIL_REFRESH_S): the
  status plus the page's regions, rendered here by the same functions the
  page embeds, numbered by server instance and sequence (``marker``). It
  places only changed fragments, keyed, so focus, expanded sections, the
  selected incident, scroll and the theme are kept; it restarts the watchdog
  only after a valid, newer document, and counts its window from when that
  document's request was sent, so a delayed answer is current for less time
  and a late one not at all. The live picture is loaded and decoded before
  it replaces the shown one, never blanks, has its own timer, and when no
  newer one can be shown the last stays, dimmed and marked as an old frame
  with its time. Without script a page is read-only: it shows what was
  generated and turns NOT CURRENT after the window (the window then counts
  from when the page is shown); reloading updates it.
- **Incident snapshot (D70; saved only by ``sentinel run --snapshots``).**
  The incident page shows the frame the opening observation came from, its
  identity, ingest time and time quality, with GET
  ``/incidents/<id>/snapshot.jpg`` (inline) and ``/incidents/<id>/snapshot/download``
  (attachment). A file is served only by incident ID: its name must be the
  one derived from that ID, it is opened inside the data directory's
  ``snapshots`` folder without following links or blocking, and its size and
  SHA-256 must match the row. A missing, pruned or altered snapshot is stated
  as such, and never replaced by another frame.
- **Styling (D72).** Named colour tokens from five reference colours, light
  and dark themes (the browser's preference, or ``?theme=light|dark``, one of
  two exact values, kept in every link), states shown by a text label and an
  icon as well as colour, incidents beside the live view on wide screens and
  below it on narrow ones, technical provenance behind expandable sections.
  Self-hosted (``sentinel.web``): the stylesheet is inlined in each page, so the
  watchdog rules always arrive with their markup, and the CSP allows it by its
  SHA-256 and no other inline style. Green is used for current claims only.
  Camera frames and snapshots are never filtered or recoloured.
- **Advisory chat (D74; only with ``sentinel run --chat``).** POST ``/chat``
  asks the admitted scene model about the newest live frame, and GET
  ``/chat/<job>`` reads its answer; both are this page's only non-GET route
  and its only route that is not plain reading. A POST must come from this
  page: the loopback Host, an Origin of exactly this origin (and
  Sec-Fetch-Site same-origin when sent), a JSON body of at most
  CHAT_MAX_BODY bytes with a Content-Length, and the token this server put in
  its page (both chat routes); OPTIONS stays refused, so no cross-origin
  preflight can pass. Chat is advisory (``sentinel.scene.chat``) and
  unavailable in this build: no chat admission exists.
- **Bounded connections.** At most MAX_CONNECTIONS at once (more get 503 and
  are closed); each connection's reads and writes time out after
  HANDLER_TIMEOUT_S, and its kernel send buffer is fixed, so a slow or
  stalled browser holds one thread for a bounded time and never touches the
  runtime loop. Connection errors are counted, never printed.
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
  exception text, or any image other than the live view's current picture. The runtime snapshot holds numbers and fixed
  labels only; stored delivery errors were redacted when written; every value
  is HTML-escaped. Stored payloads are shown only as fields of a contract this
  build validates (zone and hazard observations, scene evidence); the details
  of any other evidence kind, identity evidence included, are not shown.
  Responses are not cached and may not be framed.

A standard-library HTTP server; the stored contracts are validated with the
project's pydantic models. No new dependency; portable.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import itertools
import json
import os
import re
import secrets
import socket
import sqlite3
import stat
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from .contracts import Evidence, EvidenceStatus, SourceTimeQuality
from .incidents.service import UNRESOLVED, latest_acknowledgment
from .incidents.snapshots import SNAPSHOT_DIR, SnapshotPolicy, file_name as snapshot_file_name
from .live_preview import STREAM_WAIT_S, LivePreview, PreviewPicture, frame_facts, parse_token, time_label
from .rules.scene_hazard import FIRE_SMOKE_CANDIDATE, HazardCandidate
from .rules.zones import ZONE_KINDS, ZoneObservation
from .scene.chat import ADVISORY, CHAT_MAX_BODY, CHAT_MAX_QUESTION, ChatRequestError, parse_chat_request
from .scene.report import SceneReport
from .scene.state import SCENE_STATE_KIND
from .storage.database import connect_reader
from .web import SPRITE, TAB_ICON, csp_hash, head_script, icon, script, stylesheet
from .web.palettes import DEFAULT_PALETTE, PRESETS

LOOPBACK_HOST = "127.0.0.1"
MAX_INCIDENTS = 20
MAX_DELIVERIES = 30
REFRESH_S = 2
# Incident detail. IDs with '.' or ':' (allowed by the contracts) get no page; generated IDs never have them.
INCIDENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
INCIDENT_PATH = re.compile(r"/incidents/(?P<id>[A-Za-z0-9][A-Za-z0-9_-]{0,127})(?P<json>\.json)?")
DETAIL_REFRESH_S = 10
PREVIEW_PATH = re.compile(r"/preview/(?P<token>[0-9]{1,9}-[0-9]{1,12})\.jpg")
LIVE_FRAME_PATH = "/live/frame"  # D75: the player's pictures, one per request (needs this server's token)
MAX_STREAM_REQUESTS = 4  # live picture requests at once, for every viewer together (each holds one connection)
PAGE_CURRENT_S = 5  # current claims are shown only this long after the browser shows the page (D71; CSS only)
PREVIEW_OFF = {"enabled": False, "state": "off",
               "reason": "the live view is off; start sentinel run with --live-preview to show it"}
PREVIEW_STARTING = {"enabled": True, "state": "starting", "reason": "the runtime is starting"}
SNAPSHOT_PATH = re.compile(
    r"/incidents/(?P<id>[A-Za-z0-9][A-Za-z0-9_-]{0,127})/snapshot(?P<kind>\.jpg|/download)")
SNAPSHOT_COLUMNS = (
    "observation_id", "state", "reason", "stream_epoch", "frame_seq", "ingest_utc", "source_time_quality", "width",
    "height", "file_name", "bytes", "sha256", "recorded_utc", "pruned_utc",
)
SNAPSHOT_POLICY = SnapshotPolicy()  # what `sentinel run --snapshots` uses; shown on the page
SNAPSHOT_NONE = (
    "none recorded for this incident: snapshots are saved only by sentinel run --snapshots, for the observation that "
    "opens an incident, and none is left if the runtime stopped before saving it"
)
SNAPSHOT_REASONS = {
    "frame_not_retained": "the frame it came from was no longer held when the incident was recorded",
    "too_many_waiting": "more frames were waiting with unrecorded observations than the limit",
    "queue_full": "the snapshot writer's queue was full",
    "frame_identity_mismatch": "the held frame was not the one the observation came from",
    "incident_id_not_file_safe": "the incident ID cannot name a file",
    "too_large": "the picture was larger than the limit",
    "free_space_reserve": "saving it would have left less free space than the reserve",
    "retention_failed": "older snapshots could not be deleted to make room",
    "unsupported_pixel_format": "the frame's pixel format cannot be encoded",
    "retention_age": "older than the retention period",
    "retention_quota": "the oldest, deleted to stay within the quota",
}
SNAPSHOT_NOTE = (
    f"A snapshot is the frame the observation that opened the incident came from, never a later one. Snapshots are "
    f"kept for {SNAPSHOT_POLICY.retention_s / 86400:g} days; the oldest are deleted first beyond "
    f"{SNAPSHOT_POLICY.quota_bytes / 1e6:g} MB or {SNAPSHOT_POLICY.max_count} snapshots, and none is saved while free "
    f"space would fall below the greater of {SNAPSHOT_POLICY.reserve_min_bytes / 1e9:g} GB and "
    f"{SNAPSHOT_POLICY.reserve_fraction:.0%} of the volume. They are private files (0600) in the data directory's "
    "snapshots folder, never in the repository."
)
MAX_CONNECTIONS = 16
MAX_PICTURE_RESPONSES = 2
HANDLER_TIMEOUT_S = 10.0  # per connection: a request must arrive, and each response be written, within this
SEND_BUFFER_BYTES = 262_144
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
SECURITY_HEADERS = {  # Content-Security-Policy is added per response: it names the inline stylesheet's hash
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
TAB_ICON_PATH = "/assets/icon.svg"
CHAT_PATH = "/chat"
CHAT_JOB_PATH = re.compile(r"/chat/(?P<id>[0-9a-f]{16})")
CHAT_OFF = {"state": "off", "reason": "chat is off; start sentinel run with --chat to offer it"}
CHAT_STARTING = {"state": "unavailable", "reason": "the runtime is starting"}
SCRIPT_PATH = "/assets/dashboard.js"
SCRIPT = script()
HEAD_SCRIPT_PATH = "/assets/appearance.js"  # D96: the saved appearance, before the first paint
HEAD_SCRIPT = head_script()
PAGE_PATHS = ("/", "/dashboard")  # D96: the dashboard at its own name too; "/" and every existing link unchanged
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
            detail = _incident_detail(connection, incident_id)
        finally:
            connection.execute("COMMIT")
    except sqlite3.Error as exc:
        return {"state": "unavailable", "problem": f"database_unreadable:{type(exc).__name__}"}
    finally:
        connection.close()
    snapshot = detail.get("snapshot") or {}
    if "_file" in snapshot:  # checked outside the read transaction
        data = snapshot_bytes(Path(db_path).parent / SNAPSHOT_DIR, incident_id, *snapshot.pop("_file"))
        snapshot["file"] = "verified" if data is not None else "not_verified"
    return detail


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
        "snapshot": _snapshot_detail(c, incident_id),
    }


def _snapshot_detail(c: sqlite3.Connection, incident_id: str) -> dict[str, Any]:
    """The incident's snapshot record (D70); ``_file`` (name, size, SHA-256) is checked after the read."""
    table = c.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'incident_snapshots'").fetchone()
    row = None if table is None else c.execute(
        f"SELECT {', '.join(SNAPSHOT_COLUMNS)} FROM incident_snapshots WHERE incident_id = ?", (incident_id,)
    ).fetchone()
    if row is None:  # also a database from before schema 3
        return {"state": "none", "text": SNAPSHOT_NONE}
    item = dict(zip(SNAPSHOT_COLUMNS, row))
    try:
        label = time_label(SourceTimeQuality(item["source_time_quality"])) if item["source_time_quality"] else None
    except ValueError:
        label = None
    state, reason = item["state"], item["reason"]
    explained = SNAPSHOT_REASONS.get(reason, reason) if reason else None
    text = {
        "saved": "saved: the frame the observation that opened this incident came from",
        "missing": f"missing: {explained}; no other frame was substituted",
        "pruned": f"deleted by retention ({explained}) at {item['pruned_utc']}",
    }.get(state, f"unknown state {state!r}")
    detail = {
        "state": state, "reason": reason, "text": text, "observation_id": item["observation_id"],
        "frame": {"stream_epoch": item["stream_epoch"], "frame_seq": item["frame_seq"]},
        "ingest_utc": item["ingest_utc"], "source_time_quality": item["source_time_quality"], "time_label": label,
        "width": item["width"], "height": item["height"], "bytes": item["bytes"], "sha256": item["sha256"],
        "recorded_utc": item["recorded_utc"], "pruned_utc": item["pruned_utc"],
    }
    if state == "saved":
        detail["file"] = None  # set after the read: verified, or not_verified
        detail["view_url"] = f"/incidents/{incident_id}/snapshot.jpg"
        detail["download_url"] = f"/incidents/{incident_id}/snapshot/download"
        detail["_file"] = (item["file_name"], item["bytes"], item["sha256"])
    return detail


def snapshot_bytes(directory: Path, incident_id: str, name: object, size: object, digest: object) -> bytes | None:
    """The snapshot file's bytes if it is exactly what its row records; None otherwise.

    The name must be the one derived from the incident ID (so no row can point elsewhere); it is opened relative to
    the snapshot directory, which is opened without following a link, and neither open follows a link or blocks
    (a FIFO is refused, not waited on); the file must be a regular file of the recorded size and SHA-256.
    """
    if (name != snapshot_file_name(incident_id) or not isinstance(size, int) or isinstance(size, bool)
            or not 0 < size <= SNAPSHOT_POLICY.max_bytes or not isinstance(digest, str)):
        return None
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        directory_fd = os.open(directory, flags | os.O_DIRECTORY)
        try:
            fd = os.open(name, flags, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            return None
        chunks, left = [], size + 1
        while left > 0:
            chunk = os.read(fd, left)
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
        data = b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(fd)
    return data if len(data) == size and hashlib.sha256(data).hexdigest() == digest else None


def read_snapshot(db_path: Path, incident_id: str) -> tuple[str, bytes | None]:
    """(ok, bytes), or (none | not_verified | unavailable, None): a saved, verified snapshot's file, read only."""
    if not INCIDENT_ID.fullmatch(incident_id):
        return "none", None
    try:
        connection = connect_reader(db_path)
    except sqlite3.Error:
        return "unavailable", None
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'incident_snapshots'").fetchone()
        row = None if table is None else connection.execute(
            "SELECT state, file_name, bytes, sha256 FROM incident_snapshots WHERE incident_id = ?", (incident_id,)
        ).fetchone()
    except sqlite3.Error:
        return "unavailable", None
    finally:
        connection.close()
    if row is None or row[0] != "saved":
        return "none", None
    data = snapshot_bytes(Path(db_path).parent / SNAPSHOT_DIR, incident_id, *row[1:])
    return ("ok", data) if data is not None else ("not_verified", None)


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


def build_status(
    snapshot: Mapping[str, Any], store: Mapping[str, Any], now: datetime, preview: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The /status.json document: the runtime snapshot, the stored incidents and deliveries, and the live view."""
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
        **({"preview": dict(preview)} if preview is not None else {}),  # absent while off: the document is unchanged
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


THEMES = ("light", "dark")
PALETTE_KEYS = tuple(preset.key for preset in PRESETS)  # the default first
DENSITIES = ("compact",)  # comfortable is the default


@dataclass(frozen=True)
class Look:
    """The appearance a page's address chose (D96): exact allowlisted values only, so nothing is ever echoed.

    None means the default: the browser's light or dark preference, Carbon & Cream, comfortable spacing. With script,
    the viewer's saved choice (``appearance.js``) applies wherever the address chose nothing."""

    theme: str | None = None
    palette: str | None = None
    density: str | None = None

    @property
    def query(self) -> str:
        return "&".join(f"{key}={value}" for key, value in (("theme", self.theme), ("palette", self.palette),
                                                           ("density", self.density)) if value)


def look_from_query(query: str) -> Look:
    """The appearance a request asks for: ``theme`` light|dark, ``palette`` one of PALETTE_KEYS, ``density`` compact."""
    theme = palette = density = None
    for part in query.split("&")[:8]:
        key, _, value = part.partition("=")
        if key == "theme" and value in THEMES:
            theme = theme or value
        elif key == "palette" and value in PALETTE_KEYS and value != DEFAULT_PALETTE:
            palette = palette or value
        elif key == "density" and value in DENSITIES:
            density = density or value
    return Look(theme, palette, density)


def theme_from_query(query: str) -> str | None:
    """The theme a request asks for (``?theme=light`` or ``dark``); anything else follows the browser's preference.

    Only these two exact values are used, so nothing from the query is ever echoed.
    """
    return look_from_query(query).theme


def _look(theme: Look | str | None) -> Look:
    """A Look from a Look, a bare theme (``light``/``dark``, as before D96) or None."""
    if isinstance(theme, Look):
        return theme
    return Look(theme if theme in THEMES else None)


def _href(path: str, theme: Look | str | None) -> str:
    query = _look(theme).query
    return f"{path}?{query}" if query else path


def _incident_link(incident_id: object, theme: Look | str | None = None) -> object:
    """A link to the incident's detail page; an ID with no page is plain text."""
    if isinstance(incident_id, str) and INCIDENT_ID.fullmatch(incident_id):
        return _Markup(f'<a class=mono href="{_e(_href(f"/incidents/{incident_id}", theme))}">{_e(incident_id)}</a>')
    return incident_id


STYLE = stylesheet(PAGE_CURRENT_S)
STYLE_SOURCE = csp_hash(STYLE)  # the CSP allows this inline stylesheet and no other inline style
CONTENT_SECURITY_POLICY = (  # the only script is SCRIPT_PATH; it talks to this origin only
    f"default-src 'none'; img-src 'self'; style-src {STYLE_SOURCE}; script-src 'self'; connect-src 'self'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


def _document(title: str, body: str, *, theme: Look | str | None, page: str, source: str, refresh_s: int,
              marker: Mapping[str, Any] | None) -> str:
    """A page: no automatic reload (D73). With script, ``source`` is polled every ``refresh_s`` and the page's regions
    are updated in place; without it, the page is read-only and its D71 watchdog marks it NOT CURRENT. The head script
    applies the viewer's saved appearance before the first paint unless the address chose one (data-pref, D96)."""
    look = _look(theme)
    chosen = "".join(f" data-{key}={value}" for key, value in (("theme", look.theme), ("palette", look.palette),
                                                               ("density", look.density)) if value)
    chosen += " data-pref" if look.query else ""
    numbered = (f" data-instance={_e(marker['instance'])} data-sequence={_e(marker['sequence'])}"
                if marker else "")
    return f"""<!doctype html>
<html lang=en data-page={page}{chosen} data-palettes="{' '.join(PALETTE_KEYS)}" data-source="{_e(source)}" data-refresh-ms={refresh_s * 1000} data-current-ms={PAGE_CURRENT_S * 1000}{numbered}><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><meta name=color-scheme content="light dark">
<title>{_e(title)}</title>
<link rel=icon href="{TAB_ICON_PATH}" type="image/svg+xml">
<style>{STYLE}</style>
<script src="{HEAD_SCRIPT_PATH}"></script></head><body>
{SPRITE}
{body}
<script src="{SCRIPT_PATH}" defer></script>
</body></html>
"""


def _source(path: str, theme: Look | str | None) -> str:
    """The JSON a page polls for its regions (same origin), with the appearance its address chose."""
    query = _look(theme).query
    return f"{path}?page=1" + (f"&{query}" if query else "")


NOSCRIPT = ("<noscript><p class='note noscript'>Automatic updates need JavaScript, which is off in this browser: this "
            "page shows the state when it was generated. Reload to update it.</p></noscript>")


def _appearance(path: str, theme: Look | str | None) -> str:
    """Theme, palette and density in one compact panel (D96). Each choice is a link to this page with that choice, so
    it works without script; with script the choice applies in place (nothing reloads: the conversation, the draft,
    the selected incident and the reader's place stay) and is saved in this browser. States keep their colours,
    labels and icons in every palette, and pictures are never recoloured."""
    look = _look(theme)

    def choice(cls: str, kind: str, value: str, selected: bool, href_look: Look, inner: str) -> str:
        current = ' aria-current="true"' if selected else ""
        return (f'<a class={cls} href="{_e(_href(path, href_look))}" data-set={kind} data-value={value}{current}>'
                f"{inner}</a>")

    themes = "".join(choice("seg", "theme", value or "auto", value == look.theme,
                            Look(value, look.palette, look.density), f"{icon(glyph)}{label}")
                     for value, label, glyph in ((None, "Auto", "auto"), ("light", "Light", "sun"),
                                                 ("dark", "Dark", "moon")))
    swatches = "".join(choice("swatch", "palette", preset.key, preset.key == (look.palette or DEFAULT_PALETTE),
                              Look(look.theme, None if preset.key == DEFAULT_PALETTE else preset.key, look.density),
                              f"<span class='dots sw-{preset.key}' aria-hidden=true>"
                              + "".join(f"<i class=d{i}></i>" for i in range(1, len(preset.colours) + 1))
                              + f"</span><span>{_e(preset.name)}</span>{icon('check', 'ic tick')}")
                       for preset in PRESETS)
    densities = "".join(choice("seg", "density", value or "comfortable", value == look.density,
                               Look(look.theme, look.palette, value), label)
                        for value, label in ((None, "Comfortable"), ("compact", "Compact")))
    return (f"<div class=prefs><div class=pref><span class=pref-label id=pref-theme>Theme</span>"
            f"<div class=segs role=group aria-labelledby=pref-theme>{themes}</div></div>"
            f"<div class=pref><span class=pref-label id=pref-palette>Palette</span>"
            f"<div class=swatches role=group aria-labelledby=pref-palette>{swatches}</div></div>"
            f"<div class=pref><span class=pref-label id=pref-density>Density</span>"
            f"<div class=segs role=group aria-labelledby=pref-density>{densities}</div></div></div>"
            "<p class=note>Saved in this browser (with JavaScript); without it, the choice is kept in this page's "
            "address. States keep the same meaning, label and icon in every palette; camera pictures and snapshots are "
            "never recoloured.</p>")


def _clock(stamp: object, *, date: bool = False) -> str:
    """A stored UTC time for display, to the millisecond, in a <time> element that keeps the stored text."""
    if not isinstance(stamp, str) or not stamp:
        return _e(None)
    try:
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return f"<span class=mono>{_e(stamp)}</span>"
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc)
    text = when.strftime("%Y-%m-%d %H:%M:%S" if date else "%H:%M:%S") + f".{when.microsecond // 1000:03d} UTC"
    return f'<time datetime="{_e(stamp)}">{text}</time>'


KIND_ICONS = {"ok": "check", "warn": "warn", "crit": "crit", "off": "off", "done": "check", "stale": "clock"}


def _badge(kind: str, text: object, *, glyph: str | None = None, big: bool = False) -> _Markup:
    """A state label with its icon; the colour (k-*) is never the only cue."""
    size = " big" if big else ""
    return _Markup(f'<span class="badge k-{kind}{size}">{icon(glyph or KIND_ICONS[kind])}{_e(text)}</span>')


def _claim(current: str, replacement: str, *, block: bool = False) -> _Markup:
    """D71: a claim that something is current, stacked on what replaces it once the page is not current."""
    tag, cls = ("div", "fs") if block else ("span", "fs inl")
    return _Markup(f"<{tag} class='{cls}'><{tag} class=cur>{current}</{tag}><{tag} class=old>{replacement}</{tag}></{tag}>")


OK_STATES = frozenset({"available", "streaming", "ok", "running", "enabled", "live"})
WARN_STATES = frozenset({"degraded", "connecting", "waiting", "starting", "stale"})
CRIT_STATES = frozenset({"unavailable", "failed", "offline"})
STATE_LABELS = {
    "available": "Available", "unavailable": "Unavailable", "disabled": "Off", "streaming": "Streaming",
    "connecting": "Connecting", "waiting": "Waiting to reconnect", "starting": "Starting", "stopped": "Stopped",
    "failed": "Failed", "ok": "Working", "degraded": "Degraded", "running": "Running", "not_started": "Not started",
    "not_configured": "Not configured", "enabled": "On",
}


def _state_kind(state: object, problem: object = None, *, running: bool = True) -> str:
    if state in OK_STATES:
        return "warn" if problem else "ok"
    if state in WARN_STATES:
        return "warn"
    if state in CRIT_STATES or (state == "stopped" and running):
        return "crit"
    return "off"


def _state(state: object, problem: object = None, *, running: bool = True) -> _Markup:
    """A component's state; a healthy one is a current claim (D71)."""
    if state is None:
        return _badge("off", "Not reported", glyph="info")
    kind = _state_kind(state, problem, running=running)
    label = STATE_LABELS.get(str(state), str(state).replace("_", " "))
    if kind == "ok":
        return _claim(_badge("ok", label), _badge("off", f"Was {label.lower()}", glyph="clock"))
    return _badge(kind, label)


def _rows(pairs: list[tuple[str, object]]) -> str:
    return "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in pairs)


def _table(headers: list[str], rows: list[list[object]], empty: str) -> str:
    if not rows:
        return f"<p class=muted>{_e(empty)}</p>"
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<div class=wrap><table class=list><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _details(summary: str, body: str, *, key: str) -> str:
    """Expandable technical detail (closed by default)."""
    return (f"<details class=tech id=d-{key}><summary>{icon('chevron', 'ic chev')}{_e(summary)}</summary>"
            f"<div class=body>{body}</div></details>")


FACE_EXPERIMENTAL = "experimental: a validation run, not validated"


def _face_experimental(components: Mapping[str, Any]) -> bool:
    """D92: face recognition in a --face-validation run (its admission is PENDING_VALIDATION): what it shows is an
    experiment, and the page says so wherever it shows face recognition."""
    return (components.get("face") or {}).get("validation_run") is True


def _identity_counts(live: dict[str, Any]) -> str | None:
    counts = live.get("identity") if live else None
    if not counts:
        return None
    return (f"known {counts.get('known')} (fresh {counts.get('fresh')}, retained {counts.get('retained')}) · "
            f"unknown {counts.get('unknown')} · uncertain {counts.get('uncertain')} · unresolved {counts.get('unresolved')}")


# ---------------------------------------------------------------- recognition (D95)

FACE_FLAG = "Experimental · pending validation"
FACE_CONTEXT = ("Context only, never an access decision. Unknown means recent faces matched nobody enrolled, not that "
                "the person is a stranger.")
UNRESOLVED_TEXT = {  # the core's reasons (identity.state), in plain words
    "no face visible": "No face is visible on this track.",
    "no_face": "No face is visible on this track.",
    "face ownership ambiguous": "A face could belong to more than one person, so it is given to none.",
    "ambiguous": "A face could belong to more than one person, so it is given to none.",
    "no face observed yet": "No face has been seen on this track yet.",
    "face quality too low": "The face was too small or unclear to compare.",
    "low_quality": "The face was too small or unclear to compare.",
    "face matches more than one identity": "The face is too close to more than one enrolled identity to choose.",
    "embedding from an incompatible model": "The face could not be compared (a different model made the enrollment).",
    "no identities enrolled": "Nobody is enrolled, so no face can be recognized.",
    "face matched": "One face matched so far; recognition needs consistent matches.",
    "face matched nobody": "One face matched nobody so far; this needs consistent results.",
    "face recognition disabled": "Face recognition is off.",
    "face recognition unavailable": "Face recognition is unavailable.",
}


def _seconds_text(ms: object) -> str | None:
    return f"{ms / 1000:.1f} s" if isinstance(ms, int) and not isinstance(ms, bool) and ms >= 0 else None


def _vote_time(published: object, vote_age_ms: object) -> str:
    """When the newest face result on a track was observed: its age at the report, and the clock time that implies."""
    age = _seconds_text(vote_age_ms)
    if age is None:
        return "No face result on this track yet."
    when = ""
    if isinstance(published, str):  # the report's clock time less a monotonic age: approximate, hence "about"
        try:
            observed = datetime.fromisoformat(published.replace("Z", "+00:00")) - timedelta(milliseconds=vote_age_ms)
            when = f" (about {_clock(observed.isoformat())})"
        except (ValueError, TypeError, OverflowError):
            when = ""
    return f"Newest face result {age} before this report{when}."


def _person(entry: Mapping[str, Any], published: object, policy: Mapping[str, Any]) -> str:
    """One tracked person's recognition state: the track it lives on, the state with its basis, and when its newest
    face result was observed. A current match is a current claim (D71); a retained identity says it is retained."""
    track, epoch = entry.get("track_id"), entry.get("stream_epoch")
    state, basis = entry.get("state"), entry.get("basis")
    ttl = _number(policy.get("vote_ttl_s"))
    fresh_s = _number(policy.get("fresh_window_s"))
    needed = _number(policy.get("confirmations"))
    who = "an enrolled identity"  # the status carries no identity ID (D61, D95)
    if state == "known" and basis == "fresh":
        verdict = _claim(_badge("ok", "Known · current match", glyph="check"),
                         _badge("off", "Was a current match", glyph="clock"))
        what = f"Matches {who} ({_e(entry.get('reason'))}; the newest within {_e(fresh_s or '–')} s)."
    elif state == "known":
        verdict = _badge("done", "Known · retained", glyph="clock")
        what = (f"Retained from an earlier match with {who}; no usable face on this track since. Kept on this track "
                f"only, until its confirming matches are {_e(ttl or '–')} s old or a different result arrives.")
    elif state == "unknown":
        verdict = _badge("off", "Unknown" + (" · retained" if basis == "retained" else ""), glyph="help")
        what = "Recent faces on this track matched nobody enrolled. Not a stranger verdict."
    elif state == "uncertain":
        verdict = _badge("warn", "Uncertain", glyph="warn")
        what = "Recent faces on this track disagree; no identity is shown."
    else:
        verdict = _badge("off", "Not recognized", glyph="off")
        reason = entry.get("reason")
        what = UNRESOLVED_TEXT.get(str(reason), f"Not recognized: {_e(reason)}.")
    status = entry.get("track_status")
    track_text = "confirmed" if status == "confirmed" else "not yet confirmed"
    if entry.get("predicted"):
        measured = _seconds_text(entry.get("last_measured_age_ms"))
        track_text += f"; position predicted, last detected {measured or '–'} before"
    rule = f" Recognition needs {_e(needed)} consistent face results." if needed and state == "unresolved" else ""
    return (f'<li class=person data-key="p-{_e(epoch)}-{_e(track)}" data-state="{_e(state)}" '
            f'data-basis="{_e(basis or "none")}"><div class=who><strong>{icon("user")}Track {_e(track)}</strong>'
            f"<span class=muted>stream epoch {_e(epoch)} · {_e(track_text)}</span></div>"
            f"<div class=verdict>{verdict}</div><p class=what>{what}{rule}</p>"
            f"<p class=when>{_vote_time(published, entry.get('vote_age_ms'))}</p></li>")


def _faces(status: Mapping[str, Any]) -> str:
    """The recognition panel's region (D95): what the runtime derived for each current person, with its basis and
    observation time, apart from connection health; experimental and pending validation wherever it is shown."""
    components = status.get("components") or {}
    live = status.get("live") or {}
    face = components.get("face") or {}
    state = face.get("state")
    # D92: shown wherever face recognition runs before its validation (a --face-validation run, the only way it can run
    # while its admission is PENDING_VALIDATION); an admitted run would carry no flag.
    flag = f"<span class=flag>{icon('flask')}{FACE_FLAG}</span>" if _face_experimental(components) else ""
    if state is None:
        return (f"<p class=empty>{icon('clock')}<span><strong>Not reported yet.</strong> The runtime has not reported "
                "face recognition yet.</span></p>")
    if state != "available" or "faces" not in live:
        if state == "disabled":
            title, text, glyph = "Face recognition is off.", "This run does not use face recognition.", "off"
        else:
            title, glyph = "Face recognition is unavailable:", "crit"
            text = f"{face.get('problem') or 'not working'}."
        return (f"{flag}<p class=empty>{icon(glyph)}<span><strong>{_e(title)}</strong> {_e(text)}</span></p>")
    validation = (" Validation run: recognition is not validated." if face.get("validation_run") else "")
    lede = f"<div class=lede>{flag}<p class=muted>{_e(FACE_CONTEXT)}{_e(validation)}</p></div>"
    video = live.get("video")
    published = live.get("published_utc")
    # D71: the panel says so itself once the page is not current (the report below is then dimmed as well); both layers
    # are one line, so switching between them moves nothing.
    context = _claim(f"<div class=context>{icon('clock')}<p>Derived by the runtime at {_clock(published)}.</p></div>",
                     f"<div class='context k-stale'>{icon('clock')}<p>NOT CURRENT: last derived at {_clock(published)}."
                     "</p></div>", block=True)
    if video != "fresh":
        context += (f"<div class='context k-warn'>{icon('warn')}<p>Video is {_e(video or 'not reported')}: what is "
                    "listed is as the runtime last derived it, not the scene now.</p></div>")
    policy = face.get("policy") or {}
    rows = [_person(entry, published, policy) for entry in live.get("faces") or []]
    if rows:
        people = f"<ul class=people>{''.join(rows)}</ul>"
    else:
        reason = live.get("occupancy_reason")
        people = (f"<p class=empty>{icon('user')}<span>Nobody is tracked right now"
                  + (f" ({_e(reason)})" if reason and live.get("occupancy") == "unknown" else "") + ".</span></p>")
    more = live.get("faces_not_shown")
    extra = (f"<p class=note>{_e(more)} more tracked people are not listed (the first {len(rows)} by track are).</p>"
             if isinstance(more, int) and more > 0 else "")
    report = f"<p class=note>Enrolled identities: {_e(face.get('identities_enrolled'))}.</p>"
    rules = _details("How recognition is decided", (
        f"<p>A track becomes known after {_e(_number(policy.get('confirmations')) or '–')} consistent face results "
        "for the same enrolled identity, and unknown after as many that match nobody. \"Current match\" means the "
        f"newest of them is at most {_e(_number(policy.get('fresh_window_s')) or '–')} s old at the report; otherwise "
        "the result is retained, on that track only, until its confirming results are "
        f"{_e(_number(policy.get('vote_ttl_s')) or '–')} s old. A track that ends takes its result with it; a new "
        "track starts unresolved. Contradicting results make it uncertain at once.</p>"
        "<p class=muted>Thresholds, tracking and expiry are the runtime's own and are not changed by this page. Face "
        "data, similarities, names and identity IDs are never shown here.</p>"),
        key="faces")
    return f"{lede}{context}{people}{extra}{report}{rules}"


PREVIEW_LABELS = {"live": "LIVE", "stale": "STALE: not current", "offline": "OFFLINE", "starting": "STARTING"}
PREVIEW_KINDS = {"live": "ok", "stale": "warn", "offline": "crit", "starting": "off"}
SEVERITY_KINDS = {"info": "off", "warning": "warn", "critical": "crit"}
SEVERITY_ICONS = {"info": "info", "warning": "warn", "critical": "crit"}
STATUS_KINDS = {"candidate": "off", "open": "warn", "acknowledged": "done", "resolved": "done", "dismissed": "off"}
STATUS_ICONS = {"candidate": "clock", "open": "bell", "acknowledged": "check", "resolved": "check", "dismissed": "off"}


def _number(value: object) -> str | None:
    """A whole or decimal number for display (``15``, ``7.5``); None for anything else."""
    return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _seconds(ms: object) -> str | None:
    return None if not isinstance(ms, int) else f"{ms / 1000:.1f} s"


def _severity(severity: object) -> _Markup:
    text = str(severity)
    return _badge(SEVERITY_KINDS.get(text, "off"), text.capitalize(), glyph=SEVERITY_ICONS.get(text, "info"))


def _status(status: object) -> _Markup:
    text = str(status)
    return _badge(STATUS_KINDS.get(text, "off"), text.capitalize(), glyph=STATUS_ICONS.get(text, "info"))


def _empty_frame(glyph: str, title: str, text: object) -> str:
    return (f"<div class=frame><div class=frame-empty>{icon(glyph)}<strong>{_e(title)}</strong>"
            f"<p>{_e(text)}</p></div></div>")


def _live_head(view: Mapping[str, Any] | None) -> str:
    """The live view's state line: LIVE (a current claim, D71), or why there is no current picture."""
    view = view or PREVIEW_OFF
    if not view.get("enabled"):
        return f"<div class=row>{_badge('off', 'Off', big=True)}<span class=muted>{_e(view.get('reason'))}.</span></div>"
    state = view.get("state")
    label = PREVIEW_LABELS.get(str(state), str(state).upper())
    reason = f"<span class=muted>{_e(view.get('reason'))}</span>"
    if state == "live":  # a claim that the video is current: only while the page is (D71); a missing picture is the
        # frame area's to say (D75: pictures come from the player's own requests, not from this document)
        return _claim(f"<div class=row>{_badge('ok', label, glyph='video', big=True)}{reason}</div>",
                      f"<div class=row>{_badge('stale', 'NOT CURRENT: last reported LIVE', big=True)}<span>when this "
                      f"page was last updated; it has not refreshed for {PAGE_CURRENT_S} s, so what it shows is "
                      "old.</span></div>", block=True)
    return f"<div class=row>{_badge(PREVIEW_KINDS.get(str(state), 'off'), label, big=True)}{reason}</div>"


def _frame_empty(view: Mapping[str, Any] | None) -> str:
    """The frame area when there is no picture to show (and none to keep)."""
    view = view or PREVIEW_OFF
    if not view.get("enabled"):
        return _empty_frame("video", "The live view is off", f"{view.get('reason')}.")
    state = view.get("state")
    if state == "live":
        return _empty_frame("image", "No picture", f"{view.get('problem') or 'none encoded yet'}.")
    label = PREVIEW_LABELS.get(str(state), str(state).upper())
    shown = view.get("picture") or view.get("frame")
    return _empty_frame(KIND_ICONS[PREVIEW_KINDS.get(str(state), "off")], label,
                        "No picture is shown: the last frame is not current." if shown else view.get("reason"))


def _live_frame(view: Mapping[str, Any] | None) -> str:
    """The frame area as first served: the live picture, with its old-picture captions, or the empty frame."""
    picture = (view or {}).get("picture")
    if not picture:
        return _frame_empty(view)
    return (f'<figure class=frame data-frame="{_e(picture["stream_epoch"])}-{_e(picture["frame_seq"])}" '
            f'data-ingest="{_e(picture["ingest_utc"])}" data-age-ms="{_e(picture["age_ms"])}"><img class=pic '
            f'src="{_e(picture["url"])}" width="{_e(picture["width"])}" height="{_e(picture["height"])}" alt="Camera '
            f'picture, stream epoch {_e(picture["stream_epoch"])}, frame {_e(picture["frame_seq"])}">'
            f"<figcaption class=pic-old>{icon('clock')}<span>NOT CURRENT: no newer picture has arrived in time; this one "
            f"was received at {_clock(picture['ingest_utc'])}.</span></figcaption><figcaption class=kept>"
            f"{icon('clock')}<span></span></figcaption></figure>")


def _live_facts(view: Mapping[str, Any] | None) -> str:
    view = view or PREVIEW_OFF
    if not view.get("enabled"):
        return ""
    picture = view.get("picture")
    shown = picture or view.get("frame")
    facts = ""
    if shown:
        age = picture["age_ms"] if picture else view.get("age_ms")
        facts = ("<dl class=facts>"
                 f"<div><dt>Frame age</dt><dd>{_e(_seconds(age))} <span class=muted>when this update was generated</span></dd></div>"
                 f"<div><dt>Ingest time</dt><dd>{_clock(shown['ingest_utc'])}</dd></div>"
                 f"<div><dt>Frame</dt><dd>stream epoch {_e(shown['stream_epoch'])}, frame {_e(shown['frame_seq'])}</dd></div>"
                 "</dl>")
    bounds = view.get("bounds") or {}
    note = (f"With JavaScript, this page asks the device for one picture at a time, the newest, as soon as it has "
            f"shown the last; a picture is shown as live for at most {_e(_seconds(bounds.get('stale_after_ms')))} after "
            "its frame was received by the device (its age when sent, plus the time since this browser asked for it). "
            f"Pictures are at most {bounds.get('max_width')}×{bounds.get('max_height')}, at most "
            f"{_e(_number(bounds.get('max_encodes_per_s')))} per second for all viewers (fewer while monitoring work "
            "waits), "
            "encoded only while someone watches, from the frames the runtime already reads; nothing is saved. When no "
            "newer picture can be shown, the last one stays, dimmed and marked as an old frame with its time. Without "
            "JavaScript the picture is the one made with the page.")
    about = _details("About this picture", (f"<p>{_e(view.get('time_label'))}</p>" if view.get("time_label") else "")
                     + f"<p class=muted>{_e(note)}</p>", key="picture")
    return f"{facts}{about}"


def _capabilities(live: Mapping[str, Any], components: Mapping[str, Any], running: bool) -> str:
    """What the runtime can do right now: detector, scene analysis, face recognition, alerts."""
    if not live:
        return ""
    channels = (components.get("notifications") or {}).get("channels") or {}
    alerts = ("disabled" if not channels else
              "available" if all(c.get("state") == "available" for c in channels.values()) else "unavailable")
    chips = []
    experimental = _face_experimental(components)
    for name, state in (("Detector", live.get("detector")), ("Scene analysis", live.get("scene_analysis")),
                        ("Face recognition", live.get("face_recognition")), ("Alerts", alerts)):
        kind = _state_kind(state, running=running)
        word = STATE_LABELS.get(str(state), str(state)).lower()
        if name == "Face recognition" and experimental:  # D92: a face validation run is never shown as plain
            word += f" ({FACE_EXPERIMENTAL})"
        label = f"{name}: {word}"
        chips.append(_claim(_badge("ok", label), _badge("off", f"{name}: was {word}", glyph="clock"))
                     if kind == "ok" else _badge(kind, label))
    return f"<div class=caps aria-label=Capabilities>{''.join(chips)}</div>"


def _monitoring(runtime: Mapping[str, Any], degraded: list[str]) -> _Markup:
    state = runtime.get("state")
    if state == "running":
        current = (_badge("ok", "Monitoring", glyph="shield") if not degraded
                   else _badge("warn", "Monitoring, degraded"))
        return _claim(current, _badge("off", "Last reported: monitoring" if not degraded
                                      else "Last reported: degraded", glyph="clock"))
    return _badge("off", STATE_LABELS.get(str(state), str(state).capitalize()), glyph="clock")


def _chat_state(view: Mapping[str, Any] | None) -> str:
    """Whether chat can be asked now (D74), with the reason; available is a current claim (D71)."""
    view = view or CHAT_OFF
    state = str(view.get("state"))
    if state == "available":
        badge = _claim(_badge("ok", "Available", glyph="chat"), _badge("off", "Was available", glyph="clock"))
    elif state == "paused":  # D77: memory pressure or missed scene deadlines; monitoring comes first
        badge = _badge("warn", "Paused", glyph="clock")
    else:
        badge = _badge("off", {"off": "Off", "unavailable": "Unavailable"}.get(state, state.capitalize()), glyph="off")
    return f'<div class=row data-chat-state="{_e(state)}">{badge}<span class=muted>{_e(view.get("reason"))}.</span></div>'


def status_regions(status: Mapping[str, Any], *, theme: Look | str | None = None) -> dict[str, str]:
    """The main page's updatable regions, by name: the full page embeds them, and ``/status.json?page=1`` carries them
    for in-place updates (D73). One renderer for both, so the two can never disagree."""
    runtime = status["runtime"]
    components = status["components"]
    live = status.get("live") or {}
    rates = status.get("rates") or {}
    frames = status.get("frames") or {}
    degraded = status["degraded"]
    generated = status.get("generated_utc")
    running = runtime.get("state") == "running"
    fresh = (  # the page's own freshness, apart from what the runtime last reported (D71)
        f"<div class=fs>"
        f"<div class='cur now'>{icon('clock')}<p>This page is current: updated less than {PAGE_CURRENT_S} s ago, "
        f"generated at {_clock(generated)}; it updates every {REFRESH_S} s without reloading. If updates stop, this "
        f"bar turns NOT CURRENT within {PAGE_CURRENT_S} s and the report below is marked as last reported.</p></div>"
        f"<div class='old gone'>{icon('clock')}<p><strong>NOT CURRENT: this page has not refreshed for "
        f"{PAGE_CURRENT_S} s</strong>. Everything below is what the runtime had last reported when this page was "
        f"generated at {_clock(generated)}, and may have changed since. Only a valid, newer update makes it current "
        "again; it is retried automatically (without JavaScript, reload).</p></div></div>"
    )
    pills = (f"<span class=updated>Last update {_clock(runtime.get('updated_utc'))}</span>{_monitoring(runtime, degraded)}"
             f"{_claim(_badge('ok', 'Page current'), _badge('stale', 'NOT CURRENT'))}")
    if degraded:
        summary = _badge("warn", "Degraded", big=True)
        problems = ("<div class='alertbox k-warn'>" + icon("warn") + "<div><strong>Degraded</strong><ul>"
                    + "".join(f"<li>{_e(reason)}</li>" for reason in degraded) + "</ul></div></div>")
    else:
        summary = _claim(_badge("ok", "All components working", big=True),
                         _badge("off", "Last reported: all components working. Whether they still are is not known: "
                                "this page is not current.", glyph="clock", big=True))
        problems = ""
    notifications = components.get("notifications") or {}
    parts: list[tuple[str, Mapping[str, Any]]] = [
        (name, components.get(name) or {}) for name in ("capture", "detector", "scene", "face", "incidents")]
    parts += [(f"notifier: {channel}", item) for channel, item in (notifications.get("channels") or {}).items()]
    parts.append(("delivery worker", notifications.get("worker") or {}))
    parts += [(label, components[name]) for name, label in (("preview", "live view"), ("snapshots", "incident snapshots"))
              if components.get(name) is not None]  # listed only when enabled
    database = status.get("database") or {}
    parts.append(("incident database (read)", database))
    component_rows = "".join(
        f"<li data-key=\"c-{_e(name)}\"><span class=name>{_e(name)}</span>"
        f"{_state(item.get('state'), item.get('problem'), running=running)}"
        + (f"<span class=why>{_e(item.get('problem'))}</span>" if item.get("problem") else "") + "</li>"
        for name, item in parts)
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
    recent = incidents.get("recent") or []
    if recent:
        items = []
        for i in recent:
            incident_id = i["incident_id"]
            meta = (f"<span class=inc-meta><span>first {_clock(i['first_observed_utc'])}</span>"
                    f"<span>latest {_clock(i['last_observed_utc'])}</span>"
                    + (f"<span>{_e(i['annotations'])} scene annotation(s)</span>" if i["annotations"] else "")
                    + "</span>")
            body = (f"<span class=inc-meta>{_severity(i['severity'])}{_status(i['status'])}</span>"
                    f"<span class=inc-title>{_e(i['title'])}</span>"
                    f"<span class=inc-meta><span class=mono>{_e(incident_id)}</span></span>{meta}")
            fields = f'data-status="{_e(i["status"])}" data-severity="{_e(i["severity"])}"'  # D96: the filters' fields
            if isinstance(incident_id, str) and INCIDENT_ID.fullmatch(incident_id):
                href = _href(f"/incidents/{incident_id}", theme)
                items.append(f'<li data-key="i-{_e(incident_id)}" {fields}><a class=inc href="{_e(href)}">{body}</a>'
                             f'<button type=button class=peek data-select="{_e(incident_id)}" aria-pressed=false>'
                             f'{icon("info")}<span>Preview</span></button></li>')
            else:  # an ID with no page is plain text
                items.append(f"<li {fields}><div class=inc>{body}</div></li>")
        incident_list = f"<ol class=inc-list>{''.join(items)}</ol>"
    else:
        incident_list = f"<p class=empty>{icon('check')}No incidents recorded.</p>"
    delivery_rows = [
        [_incident_link(d["incident_id"], theme), d["channel"], d["message_kind"],
         d["state"] + (" (in flight)" if d["in_flight"] else ""), d["attempts"],
         "may have been delivered" if d["ambiguous"] else "", d["next_attempt_utc"], d["last_error"]]
        for d in delivery.get("recent") or []
    ]
    totals_html = "".join(
        _badge(kind, f"{label} {totals.get(key) if totals.get(key) is not None else '–'}", glyph=glyph)
        for key, label, kind, glyph in (
            ("queued", "Queued (never attempted)", "off", "clock"),
            ("attempted", "attempted, not confirmed", "warn", "send"),
            ("delivered", "delivered (provider confirmed)", "done", "check"),
            ("failed", "failed (needs operator)", "crit", "crit"),
            ("ambiguous", "ambiguous", "warn", "warn")))
    technical = _details("Capture, processing and live state", f"""<div class=wrap><table class=kv>{_rows([
        ("capture state", capture.get("state")), ("stream epoch", capture.get("stream_epoch")),
        ("connects", capture.get("connects")), ("reconnects", capture.get("reconnects")),
        ("open failures", capture.get("open_failures")), ("stream ends", capture.get("stream_ends")),
        ("retry in (s)", capture.get("retry_delay_s")), ("last problem", capture.get("problem")),
        ("captured fps", rates.get("captured_fps")), ("processed fps", rates.get("processed_fps")),
        ("detector failures /s", rates.get("failed_per_s")), ("window (s)", rates.get("window_s")),
        ("frames captured", frames.get("captured")), ("processed", frames.get("processed")),
        ("failed", frames.get("failed")), ("replaced (skipped by a slow loop)", frames.get("replaced")),
        ("not live (epoch ended)", frames.get("not_live")),
        ("video", live.get("video")), ("last frame age (ms)", live.get("last_frame_age_ms")),
        ("detector", live.get("detector")), ("occupancy", live.get("occupancy")),
        ("why", live.get("occupancy_reason")), ("people (confirmed)",
        None if not live else f"{live.get('people')} ({live.get('confirmed_people')})"),
        ("face recognition", live.get("face_recognition")),
        ("identities (counts only; " + FACE_EXPERIMENTAL + ")" if _face_experimental(components)
         else "identities (counts only)", _identity_counts(live)),
        ("scene analysis", live.get("scene_analysis")), *scene_rows])}</table></div>""", key="technical")
    view = status.get("preview")
    return {
        "pills": pills,
        "fresh": fresh,
        "live-head": _live_head(view),
        "live-facts": _live_facts(view),
        "caps": _capabilities(live, components, running),
        "incidents-meta": (f"<span class=muted data-loaded={len(recent)}>{_e(incidents.get('unresolved'))} unresolved "
                           f"in the database; the latest {len(recent)} of at most {MAX_INCIDENTS} are loaded here"
                           "</span>"),
        "incidents": (f"{incident_list}<p class=note>Waiting to be recorded: {_e(recording.get('pending_signals'))}. "
                      f"Limitation: {_e(durability)}.</p>"),
        "faces": _faces(status),
        "health": (f"<div class=phead><h2>{icon('shield')}System health</h2>{summary}</div>{problems}"
                   f"<ul class=comp>{component_rows}</ul><p class=note>Last report from the runtime: "
                   f"{_e(runtime.get('state'))}, updated {_clock(runtime.get('updated_utc'))} "
                   f"({_e(runtime.get('status_age_s'))} s before this page was generated). Read-only, loopback only.</p>"
                   f"{technical}"),
        "delivery": (f"<div class=phead><h2>{icon('send')}Alert delivery</h2></div><div class=totals>{totals_html}</div>"
                     "<p class=note>Delivery is at least once: an ambiguous attempt may have reached the chat, and a "
                     "repeat carries the same incident ID.</p>"
                     + _details("Alert rows", _table(["incident", "channel", "message", "state", "attempts", "ambiguous",
                                                      "next attempt (UTC)", "last error"], delivery_rows, "No alerts."),
                                key="alerts")),
        "chat-state": _chat_state(status.get("chat")),
        "foot": f"Read-only, loopback only (D45). Generated at {_clock(generated)}.",
    }


def page_document(status: Mapping[str, Any], marker: Mapping[str, Any], *,
                  theme: Look | str | None = None) -> dict[str, Any]:
    """``/status.json?page=1``: the status, plus what the page needs to update itself in place (D73)."""
    return {**status, "page": {**marker, "regions": status_regions(status, theme=theme),
                               "frame_empty": _frame_empty(status.get("preview"))}}


# Navigation (D96, after D82's section links): the dashboard's sections in page order, as plain links that work without
# script. With script the section in view is marked as current, the sidebar collapses to a rail on wide screens and
# becomes a drawer on narrow ones. Every destination is a real section of the page.
NAV_GROUPS = (
    ("Monitor", (("live", "Live view", "video"), ("faces", "Recognition", "scan"), ("incidents", "Incidents", "bell"),
                 ("chat", "Ask", "chat"))),
    ("System", (("health", "Health", "shield"), ("delivery", "Alerts", "send"))),
    ("Preferences", (("settings", "Appearance", "sliders"),)),
)
SECTIONS = tuple((target, label) for _, items in NAV_GROUPS for target, label, _ in items)
STATUS_FILTERS = (("open", "Open"), ("acknowledged", "Acknowledged"), ("resolved", "Resolved"),
                  ("candidate", "Candidate"), ("dismissed", "Dismissed"))
SEVERITY_FILTERS = (("critical", "Critical"), ("warning", "Warning"), ("info", "Info"))


def _sidebar(camera: object, look: Look, *, base: str = "", current: str | None = None) -> str:
    """The sidebar: brand, sections and the camera. ``base`` is "" on the dashboard (links to its own sections) or the
    dashboard's address from another page; ``current`` marks that page's section."""
    groups = []
    for group, items in NAV_GROUPS:
        links = "".join(
            f'<li><a href="{_e(base)}#{target}" data-nav={target}'
            + (' aria-current="page"' if target == current else "")
            + f">{icon(glyph)}<span class=label>{label}</span><span class=tip aria-hidden=true>{label}</span></a></li>"
            for target, label, glyph in items)
        groups.append(f"<p class=group>{group}</p><ul>{links}</ul>")
    return (f'<aside class=sidebar id=sidebar aria-label="Dashboard navigation"><div class=side-head>'
            f'<a class=brand href="{_e(_href("/dashboard", look))}"><span class=brand-tile>{icon("logo")}</span>'
            f"<span class=wordmark>Sentinel</span></a>"
            f'<button type=button class="icon-btn rail-toggle js-only" id=rail-toggle aria-controls=sidebar '
            f'aria-expanded=true><span class=vh>Collapse the sidebar</span>{icon("sidebar")}</button>'
            f'<button type=button class="icon-btn drawer-close js-only" id=drawer-close><span class=vh>Close the menu'
            f'</span>{icon("close")}</button></div><nav class=nav aria-label="Dashboard sections">{"".join(groups)}</nav>'
            f'<div class=side-foot><div class=cam-chip>{icon("video")}<span class=cam-text><span>Camera</span>'
            f"<strong class=mono>{_e(camera)}</strong></span></div></div></aside>"
            "<div class=scrim id=scrim hidden></div>")


def _menu_button() -> str:
    return (f'<button type=button class="icon-btn menu js-only" id=menu aria-controls=sidebar aria-expanded=false>'
            f'<span class=vh>Open the menu</span>{icon("menu")}</button>')


def _options(choices: tuple[tuple[str, str], ...], everything: str) -> str:
    return f'<option value="">{everything}</option>' + "".join(f"<option value={v}>{label}</option>" for v, label in choices)


def render_html(status: Mapping[str, Any], *, theme: Look | str | None = None,
                marker: Mapping[str, Any] | None = None, chat_token: str = "", path: str = "/") -> str:
    """The dashboard (served at "/" and "/dashboard"): the live picture first, recognition beside it, incidents below."""
    look = _look(theme)
    regions = status_regions(status, theme=look)
    camera = status["runtime"].get("camera_id")
    view = status.get("preview")
    path = path if path in PAGE_PATHS else "/"
    return _document(f"Sentinel · {camera}", f"""<a class=skip href="#live">Skip to the live view</a>
<div class=app>
{_sidebar(camera, look)}
<div class=shell>
<header class=topbar>
{_menu_button()}
<div class=heading><h1>Dashboard</h1><p class=sub>Camera <strong class=mono>{_e(camera)}</strong> · read-only, loopback only</p></div>
<div class=pills data-region=pills>{regions["pills"]}</div>
</header>
<div class=freshbar data-region=fresh>{regions["fresh"]}</div>
<p class=sync id=sync role=status></p>
{NOSCRIPT}
<main class=board id=board>
<section id=live class="panel live" aria-labelledby=h-live>
<div class=phead><h2 id=h-live>{icon("video")}Live view</h2><button type=button class="ghost js-only" id=focus-toggle aria-pressed=false>{icon("expand")}<span>Focus</span></button></div>
<div id=player-state class=player-state role=status></div>
<div class="report live-report" data-region=live-head>{regions["live-head"]}</div>
<div id=frame data-live={"on" if (view or {}).get("enabled") else "off"} data-token="{_e(chat_token)}" data-stale-after-ms="{_e(((view or {}).get("bounds") or {}).get("stale_after_ms") or 2000)}">{_live_frame(view)}</div>
<details class="tech js-only" id=d-delivery><summary>{icon("chevron", "ic chev")}Picture delivery in this browser</summary><div class=body><dl id=player-facts class="facts player-facts" hidden></dl><p class=note>Measured here, from the pictures this page has shown; never sent anywhere.</p></div></details>
<div class="report live-report" data-region=live-facts>{regions["live-facts"]}</div>
<div class=report data-region=caps>{regions["caps"]}</div>
</section>
<section id=faces class="panel faces" aria-labelledby=h-faces>
<div class=phead><h2 id=h-faces>{icon("scan")}Recognition</h2></div>
<div class=report data-region=faces>{regions["faces"]}</div>
</section>
<section id=incidents class="panel incidents" aria-labelledby=h-incidents>
<div class=phead><h2 id=h-incidents>{icon("bell")}Incidents</h2><div data-region=incidents-meta>{regions["incidents-meta"]}</div></div>
<div class="filters js-only" id=inc-filters role=group aria-label="Filter the loaded incidents">
<label>Status<select id=f-status>{_options(STATUS_FILTERS, "Any status")}</select></label>
<label>Severity<select id=f-severity>{_options(SEVERITY_FILTERS, "Any severity")}</select></label>
<p class=count id=f-count aria-live=polite></p>
</div>
<p class=empty id=f-empty hidden>{icon("filter")}<span>No loaded incident matches these filters. Only the latest incidents are loaded on this page; older ones are not searched.</span></p>
<section id=inc-preview class=preview aria-label="Selected incident" aria-live=polite hidden></section>
<div class=report data-region=incidents>{regions["incidents"]}</div>
</section>
<section id=chat class="panel chat" aria-labelledby=h-chat><div class=report>
<div class=phead><h2 id=h-chat>{icon("chat")}Ask about the live picture</h2></div>
<div data-region=chat-state>{regions["chat-state"]}</div>
<p class=note>{_e(ADVISORY)}</p>
<ol id=chat-log class=chat-log aria-live=polite></ol>
<form id=chat-form class=chat-form autocomplete=off data-token="{_e(chat_token)}">
<label for=chat-q>Your question <span class=muted>(at most {CHAT_MAX_QUESTION} characters)</span></label>
<textarea id=chat-q name=question maxlength={CHAT_MAX_QUESTION} rows=2 placeholder="For example: is the door open?"></textarea>
<div class=row><button type=submit class=button id=chat-ask disabled>{icon("send")}Ask</button><span id=chat-note class=muted role=status></span></div>
</form>
<noscript><p class=note>Chat needs JavaScript.</p></noscript>
</div></section>
<section id=health class="panel health" aria-label="System health"><div class=report data-region=health>{regions["health"]}</div></section>
<section id=delivery class="panel delivery" aria-label="Alert delivery"><div class=report data-region=delivery>{regions["delivery"]}</div></section>
<section id=settings class="panel settings" aria-labelledby=h-settings>
<div class=phead><h2 id=h-settings>{icon("sliders")}Appearance</h2></div>
{_appearance(path, look)}
</section>
</main>
<footer class=foot data-region=foot>{regions["foot"]}</footer>
</div>
</div>""", theme=look, page="status", source=_source("/status.json", look), refresh_s=REFRESH_S, marker=marker)


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


def _snapshot_html(snapshot: Mapping[str, Any] | None) -> str:
    """The snapshot section: the verified picture with a download link, or why there is none."""
    snapshot = snapshot or {"state": "none", "text": SNAPSHOT_NONE}
    state = snapshot["state"]
    if state == "saved" and snapshot.get("file") == "verified":
        frame = snapshot["frame"]
        body = (f'<figure class=frame><img src="{_e(snapshot["view_url"])}" width="{_e(snapshot["width"])}" '
                f'height="{_e(snapshot["height"])}" alt="Snapshot: stream epoch {_e(frame["stream_epoch"])}, frame '
                f'{_e(frame["frame_seq"])}"></figure><div class=row><a class=button href="{_e(snapshot["download_url"])}" '
                f'download>{icon("download")}Download this snapshot</a><span class=muted>JPEG, {_e(snapshot["bytes"])} '
                f'bytes, {_e(snapshot["width"])}×{_e(snapshot["height"])}</span></div>')
    elif state == "saved":
        body = (f"<div class='alertbox k-crit'>{icon('crit')}<p>Saved, but its file is missing or does not match its "
                "record (size or SHA-256), so it is not shown.</p></div>")
    else:
        glyph = "off" if state in ("none", "pruned") else "warn"
        body = _empty_frame(glyph, {"none": "No snapshot", "missing": "Snapshot missing",
                                    "pruned": "Snapshot deleted"}.get(state, "No snapshot"), f"{snapshot['text']}.")
    facts = ""
    if "frame" in snapshot:
        facts = ("<dl class=facts>"
                 f"<div><dt>Frame</dt><dd>stream epoch {_e(snapshot['frame']['stream_epoch'])}, frame "
                 f"{_e(snapshot['frame']['frame_seq'])}</dd></div>"
                 f"<div><dt>Ingest time</dt><dd>{_clock(snapshot['ingest_utc'], date=True) if snapshot['ingest_utc'] else _e('not known: the frame’s record was no longer held')}</dd></div>"
                 "</dl>")
        rows: list[tuple[str, object]] = [
            ("frame", f"stream epoch {snapshot['frame']['stream_epoch']}, frame {snapshot['frame']['frame_seq']}"),
            ("observation", snapshot["observation_id"]),
            ("time (UTC)", snapshot["ingest_utc"] or "not known: the frame's record was no longer held"),
            ("what the time means", snapshot["time_label"]),
            ("recorded (UTC)", snapshot["recorded_utc"]),
        ]
        if snapshot["sha256"]:
            rows += [("size", f"{snapshot['width']}×{snapshot['height']}, {snapshot['bytes']} bytes"),
                     ("SHA-256", _Markup(f"<span class=mono>{_e(snapshot['sha256'])}</span>"))]
        facts += _details("Snapshot provenance", f"<div class=wrap><table class=kv>{_rows(rows)}</table></div>",
                          key="snapshot")
    return f"{body}{facts}<p class=note>{_e(SNAPSHOT_NOTE)}</p>"


def _ack_tag(status: object, acknowledgment: Mapping[str, Any]) -> str:
    """The acknowledgment beside the status, unless the status already says it."""
    if acknowledgment["state"] == "acknowledged":
        return "" if status == "acknowledged" else _badge("done", "Acknowledged")
    return _badge("off", "Not acknowledged", glyph="bell")


def incident_regions(detail: Mapping[str, Any], *, theme: Look | str | None = None) -> dict[str, str]:
    """One incident page's updatable regions (D73), from ``read_incident``'s document (state available)."""
    incident = detail["incident"]
    linked = detail["linked"]
    evidence, transitions = detail["evidence"], detail["transitions"]
    annotations, deliveries = detail["annotations"], detail["deliveries"]
    acknowledgment = detail["acknowledgment"]
    evidence_rows = []
    for item in evidence["shown"]:
        source = item["source"] or {}
        evidence_rows.append([
            item["observed_utc"], item["phase"], item["severity"], item["episode"], source.get("track_id"),
            source.get("stream_epoch"), _frames(item["source"]),
            source.get("in_zone_s") if "in_zone_s" in source else source.get("scene_reports"),
            item["reason"] if item["source"] is not None else f"{item['reason']} (source fields not readable)",
        ])
    opening = next((item for item in evidence["shown"] if item["phase"] == "entered"), None)
    if opening is not None:
        why = (f"<p class=reason>{_e(opening['reason'])}</p><p class=note>The observation that opened this incident, "
               f"at {_clock(opening['observed_utc'], date=True)} (the ingest time of its source frame).</p>")
    elif evidence["total"]:
        why = "<p class=reason>The opening observation is older than the rows shown below.</p>"
    else:
        why = "<p class=reason>No rule observation is recorded.</p>"
    timeline = "".join(
        f"<li><strong>{_e(t['from_status'] or 'new')} → {_e(t['to_status'])}</strong>"
        f"<div class=when>{_clock(t['at_utc'], date=True)} · by {_e(t['actor'])} · revision {_e(t['revision'])}</div>"
        f"<div>{_e(t['reason'])}</div></li>"
        for t in transitions["shown"])
    annotation_rows, scene_cards = [], []
    for item in annotations["shown"]:
        details = item["details"] or {}
        annotation_rows.append([
            item["recorded_utc"], item["observed_utc"], item["recorded_after_source_s"], item["applicability"],
            item["kind"], item["status"], f"{item['producer']} {item['producer_revision']}",
            details.get("stream_epoch"), details.get("frame_seq"),
            details.get("reason") if item["details"] is not None else f"not shown ({item['details_problem']})",
            _report_text(item["details"]),
        ])
        text = _report_text(item["details"]) if item["details"] is not None else f"not shown ({item['details_problem']})"
        scene_cards.append(
            f"<div class=card><h3>{_badge('done' if item['applicability'] == 'current' else 'off', item['applicability'], glyph='info')}"
            f"<span>{_e(item['status'])}</span></h3><p>{_e(text or details.get('reason'))}</p>"
            f"<p class=note>Source frame {_clock(item['observed_utc'], date=True)}; recorded "
            f"{_e(item['recorded_after_source_s'])} s later.</p></div>")
    delivery_rows, attempt_rows, delivery_cards = [], [], []
    for d in deliveries["shown"]:
        state = d["state"] + (" (in flight)" if d["in_flight"] else "")
        delivery_rows.append([
            d["channel"], d["message_kind"], state, d["attempts"],
            "may have been delivered" if d["ambiguous"] else "", d["provider_confirmed"], d["next_attempt_utc"],
            d["last_error"],
        ])
        for a in d["attempt_log"]["shown"]:
            attempt_rows.append([f"{d['channel']} {d['message_kind']}", a["attempt"], a["started_utc"],
                                 a["finished_utc"], a["outcome"], a["detail"]])
        kind = {"delivered": "done", "failed": "crit", "attempted": "warn"}.get(d["state"], "off")
        delivery_cards.append(
            f"<div class=card><h3>{_badge(kind, state, glyph={'delivered': 'check', 'failed': 'crit'}.get(d['state'], 'send'))}"
            f"<span>{_e(d['channel'])} · {_e(d['message_kind'])}</span></h3>"
            f"<p>{_e(d['attempts'])} attempt(s); provider confirmed: {_e(d['provider_confirmed'])}"
            + ("; may have been delivered" if d["ambiguous"] else "")
            + (f"; next attempt {_clock(d['next_attempt_utc'], date=True)}" if d["next_attempt_utc"] else "")
            + "</p>" + (f"<p class=note>Last error: {_e(d['last_error'])}</p>" if d["last_error"] else "") + "</div>")
    later = _Markup(", ".join(_e(_incident_link(i, theme)) for i in linked["later"])) if linked["later"] else None
    summary = _rows([
        ("rule", incident["kind"]), ("zone", incident["zone_id"]), ("severity", incident["severity"]),
        ("status", incident["status"]), ("revision", incident["revision"]),
        ("first seen (UTC)", _Markup(_clock(incident["first_observed_utc"], date=True))),
        ("latest (UTC)", _Markup(_clock(incident["last_observed_utc"], date=True))),
        ("recorded (UTC)", _Markup(_clock(incident["created_utc"], date=True))),
        ("updated (UTC)", _Markup(_clock(incident["updated_utc"], date=True))),
        ("previous linked incident", _incident_link(linked["previous"], theme)), ("later linked incidents", later),
        ("clip", detail["clip"]),
    ])
    if acknowledgment["state"] == "acknowledged":
        ack = (f"{_badge('done', 'Acknowledged', big=True)}<p class=reason>{_e(_acknowledgment_text(acknowledgment))}</p>")
    else:
        ack = (f"{_badge('off', 'Not acknowledged', glyph='bell', big=True)}"
               f"<p class=reason>{_e(_acknowledgment_text(acknowledgment))}</p>")

    def shown(part: Mapping[str, Any]) -> str:
        return f"<p class=note>{len(part['shown'])} of {part['total']} shown (the latest).</p>"

    technical = (
        f"<h3>Rule observations</h3>{shown(evidence)}"
        "<p class=note>Source times are the ingest times of the frames each observation came from (UTC, for display). "
        "An episode is one continuous presence of one track; ENDED closes it and changes no status.</p>"
        + _table(["source time (UTC)", "phase", "severity", "episode", "track", "stream epoch", "frames",
                  "in zone (s) / scene reports", "reason"], evidence_rows, "No rule observations.")
        + f"<h3>Lifecycle rows</h3>{shown(transitions)}"
        + _table(["at (UTC)", "from", "to", "revision", "by", "reason"],
                 [[t["at_utc"], t["from_status"], t["to_status"], t["revision"], t["actor"], t["reason"]]
                  for t in transitions["shown"]], "No transitions.")
        + f"<h3>Scene evidence rows</h3>{shown(annotations)}<p class=note>{_e(APPLICABILITY_NOTE)} \"Recorded after "
          "source\" is a wall-clock difference, for display only.</p>"
        + _table(["recorded (UTC)", "source frame (UTC)", "recorded after source (s)", "applicability", "kind",
                  "status", "producer", "stream epoch", "frame", "reason", "report"], annotation_rows,
                 "No scene evidence.")
        + f"<h3>Delivery attempts</h3><p class=note>Attempts: the latest {MAX_ATTEMPTS} per alert row, oldest first; "
          "details were redacted when stored.</p>"
        + _table(["alert", "attempt", "started (UTC)", "finished (UTC)", "outcome", "detail"], attempt_rows,
                 "No delivery attempts.")
        + _table(["channel", "message", "state", "attempts", "ambiguous", "provider confirmed", "next attempt (UTC)",
                  "last error"], delivery_rows, "No alert rows.")
    )
    return {
        "title": (f"<div class=tags>{_severity(incident['severity'])}{_status(incident['status'])}"
                  f"{_ack_tag(incident['status'], acknowledgment)}</div><p class=muted>Read-only, loopback only; "
                  f"updated every {DETAIL_REFRESH_S} s. Generated {_clock(detail.get('generated_utc'), date=True)}.</p>"),
        "snap": _snapshot_html(detail.get("snapshot")),
        "summary": f"{why}<div class=wrap><table class=kv>{summary}</table></div>",
        "ack": ack,
        "delivery": (f"<p class=note>{len(deliveries['shown'])} of {deliveries['total']} alert rows shown. Delivery is at "
                     "least once: an ambiguous attempt may have reached the chat, and a repeat carries the same incident "
                     "ID.</p><div class=cards>" + ("".join(delivery_cards) or "<p class=empty>" + icon("off")
                                                     + "No alerts for this incident.</p>") + "</div>"),
        "lifecycle": shown(transitions) + (f"<ol class=timeline>{timeline}</ol>" if timeline else
                                           "<p class=empty>" + icon("off") + "No transitions.</p>"),
        "scene": shown(annotations) + "<div class=cards>" + ("".join(scene_cards) or "<p class=empty>" + icon("off")
                                                              + "No scene evidence.</p>") + "</div>",
        "tech": _details("Technical provenance: every stored row", technical, key="provenance"),
    }


def incident_document(detail: Mapping[str, Any], marker: Mapping[str, Any], *,
                      theme: Look | str | None = None) -> dict[str, Any]:
    """``/incidents/<id>.json?page=1``: the incident, plus its page's regions for in-place updates (D73)."""
    return {**detail, "page": {**marker, "regions": incident_regions(detail, theme=theme)}}


def render_incident_html(detail: Mapping[str, Any], *, theme: Look | str | None = None,
                         marker: Mapping[str, Any] | None = None) -> str:
    """One incident's page, from ``read_incident``'s document (state available), in the dashboard's shell."""
    look = _look(theme)
    incident = detail["incident"]
    regions = incident_regions(detail, theme=look)
    path = f"/incidents/{incident['incident_id']}"
    dashboard = _href("/dashboard", look)
    return _document(f"Sentinel incident {incident['incident_id']}", f"""<a class=skip href="#snapshot">Skip to the snapshot</a>
<div class=app>
{_sidebar(incident["camera_id"], look, base=dashboard, current="incidents")}
<div class=shell>
<header class=topbar>
{_menu_button()}
<div class=heading><a class=back href="{_e(dashboard)}">{icon("back")}Back to the dashboard</a><h1>Incident</h1><p class=sub>Camera <strong class=mono>{_e(incident["camera_id"])}</strong> · read-only, loopback only</p></div>
</header>
<div class=title-block>
<p class=muted>Incident <span class=mono>{_e(incident["incident_id"])}</span></p>
<h2>{_e(incident["title"])}</h2>
<div data-region=title>{regions["title"]}</div>
</div>
<p class=sync id=sync role=status></p>
{NOSCRIPT}
<main class=detail>
<section id=snapshot class="panel p-snap" aria-label=Snapshot><div class=phead><h2>{icon("image")}Snapshot</h2></div><div data-region=snap>{regions["snap"]}</div></section>
<section class="panel p-summary" aria-label="What happened"><div class=phead><h2>{icon("info")}What happened</h2></div><div data-region=summary>{regions["summary"]}</div></section>
<section class="panel p-ack" aria-label=Acknowledgment><div class=phead><h2>{icon("check")}Acknowledgment</h2></div><div data-region=ack>{regions["ack"]}</div></section>
<section class="panel p-delivery" aria-label="Alert delivery"><div class=phead><h2>{icon("send")}Alert delivery</h2></div><div data-region=delivery>{regions["delivery"]}</div></section>
<section class="panel p-lifecycle" aria-label=Lifecycle><div class=phead><h2>{icon("list")}Lifecycle</h2></div><div data-region=lifecycle>{regions["lifecycle"]}</div></section>
<section class="panel p-scene" aria-label="Scene evidence"><div class=phead><h2>{icon("chat")}Scene evidence on this incident</h2></div><div data-region=scene>{regions["scene"]}</div></section>
<section class="panel p-tech" aria-label="Technical provenance"><div data-region=tech>{regions["tech"]}</div></section>
</main>
<footer class=foot>Read-only, loopback only (D45). Appearance: <a href="{_e(dashboard)}#settings">the dashboard's settings</a>.</footer>
</div>
</div>""", theme=look, page="incident", source=_source(f"{path}.json", look),
                     refresh_s=DETAIL_REFRESH_S, marker=marker)


# ---------------------------------------------------------------- server


class _Server(ThreadingHTTPServer):
    """At most ``max_connections`` connections at once; errors are counted, never printed."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], handler: type, max_connections: int) -> None:
        super().__init__(address, handler)
        self._slots = threading.BoundedSemaphore(max_connections)
        self._counts_lock = threading.Lock()
        self.counts = {"refused_busy": 0, "write_timeouts": 0, "connection_errors": 0, "stream_busy": 0,
                       "stream_refused": 0}

    def count(self, name: str) -> None:
        with self._counts_lock:
            self.counts[name] += 1

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            self.count("refused_busy")
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)  # starts the connection's thread
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request: Any, client_address: Any) -> None:  # a stalled or vanished browser, not a fault
        self.count("connection_errors")


class StatusServer:
    """Serves the page on 127.0.0.1:``port`` from its own thread; ``port`` 0 picks a free port (tests).

    ``preview`` (D69): None while the live view is off; otherwise it returns the runtime's LivePreview, or None
    while the runtime is still starting.
    """

    def __init__(
        self,
        port: int,
        snapshot: Callable[[], Mapping[str, Any]],
        db_path: Path,
        *,
        host: str = LOOPBACK_HOST,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        preview: Callable[[], LivePreview | None] | None = None,
        chat: Callable[[], Any] | None = None,
        max_connections: int = MAX_CONNECTIONS,
        handler_timeout_s: float = HANDLER_TIMEOUT_S,
    ) -> None:
        if host != LOOPBACK_HOST:
            raise ValueError(f"the status page binds {LOOPBACK_HOST} only")
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if not 1 <= max_connections <= MAX_CONNECTIONS:
            raise ValueError(f"max_connections must be between 1 and {MAX_CONNECTIONS}")
        if not 0 < handler_timeout_s <= HANDLER_TIMEOUT_S:
            raise ValueError(f"handler_timeout_s must be above 0 and at most {HANDLER_TIMEOUT_S:g}")
        self._snapshot = snapshot
        self._db_path = Path(db_path)
        self._now = now
        self._preview = preview
        self._chat = chat  # D74: None while chat is off; else returns the runtime's ChatService, or None while starting
        self.chat_token = secrets.token_hex(16)  # in this server's pages only; required by the chat and live routes
        self._picture_slots = threading.BoundedSemaphore(MAX_PICTURE_RESPONSES)
        self._stream_slots = threading.BoundedSemaphore(MAX_STREAM_REQUESTS)
        self._instance = secrets.token_hex(8)  # a page's updates are ordered within one server instance (D73)
        self._sequence = itertools.count(1)
        self._sequence_lock = threading.Lock()
        handler = type("StatusHandler", (_Handler,), {"page": self, "timeout": handler_timeout_s})
        self._httpd = _Server((LOOPBACK_HOST, port), handler, max_connections)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    def status(self, *, with_picture: bool = False, for_page: bool = False) -> dict[str, Any]:
        """``with_picture`` (pages only): let the live view encode a picture if one is due. ``for_page`` adds the chat's
        availability, which the plain /status.json does not carry."""
        status = build_status(self._snapshot(), read_store(self._db_path), self._now(),
                              self.preview_view(with_picture=with_picture))
        return {**status, "chat": self.chat_view()} if for_page else status

    def chat_service(self) -> Any | None:
        return None if self._chat is None else self._chat()

    def chat_view(self) -> dict[str, str]:
        if self._chat is None:
            return dict(CHAT_OFF)
        service = self._chat()
        return dict(CHAT_STARTING) if service is None else service.availability()

    def marker(self) -> dict[str, Any]:
        """The next document's place in this server's order: a page applies only newer ones (D73)."""
        with self._sequence_lock:
            return {"instance": self._instance, "sequence": next(self._sequence)}

    def preview_view(self, *, with_picture: bool = False) -> dict[str, Any] | None:
        """The live view's state; None while it is off."""
        if self._preview is None:
            return None
        preview = self._preview()
        return dict(PREVIEW_STARTING) if preview is None else preview.view(with_picture=with_picture)

    def snapshot(self, incident_id: str) -> tuple[str, bytes | None]:
        return read_snapshot(self._db_path, incident_id)

    def preview_picture(self, token: str) -> PreviewPicture | None:
        preview = None if self._preview is None else self._preview()
        return None if preview is None else preview.picture(token)

    def connection_counts(self) -> dict[str, int]:
        with self._httpd._counts_lock:
            return dict(self._httpd.counts)

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

    def setup(self) -> None:
        super().setup()  # applies ``timeout`` to the connection's reads and writes
        try:
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUFFER_BYTES)
        except OSError:
            pass

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if not self._host_allowed():
            self._send(421, "text/plain; charset=utf-8", b"misdirected request\n")
            return
        parts = urlsplit(self.path)
        path, theme = parts.path, look_from_query(parts.query)
        for_page = "page=1" in parts.query.split("&")[:8]  # a page updating itself (D73); exact, never echoed
        if path == TAB_ICON_PATH:
            self._send(200, "image/svg+xml", TAB_ICON)
            return
        if path == SCRIPT_PATH:
            self._send(200, "text/javascript; charset=utf-8", SCRIPT)
            return
        if path == HEAD_SCRIPT_PATH:
            self._send(200, "text/javascript; charset=utf-8", HEAD_SCRIPT)
            return
        if path == LIVE_FRAME_PATH:
            self._send_live_frame(parts.query)
            return
        job = CHAT_JOB_PATH.fullmatch(path)
        if job is not None:
            self._chat_poll(job["id"])
            return
        preview = PREVIEW_PATH.fullmatch(path)
        if preview is not None:
            self._send_picture(preview["token"])
            return
        snapshot = SNAPSHOT_PATH.fullmatch(path)
        if snapshot is not None:
            self._send_snapshot(snapshot["id"], download=snapshot["kind"] == "/download")
            return
        incident = INCIDENT_PATH.fullmatch(path)
        if path not in (*PAGE_PATHS, "/status.json") and incident is None:
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
                if incident["json"] and for_page:
                    document = incident_document(detail, self.page.marker(), theme=theme)
                    body, kind = json.dumps(document).encode(), "application/json"
                elif incident["json"]:
                    body, kind = json.dumps(detail, indent=2).encode(), "application/json"
                else:
                    body = render_incident_html(detail, theme=theme, marker=self.page.marker()).encode()
                    kind = "text/html; charset=utf-8"
            elif path == "/status.json" and for_page:  # the live page's update; pictures come from /live/frame (D75)
                document = page_document(self.page.status(for_page=True),
                                         {**self.page.marker(), "token": self.page.chat_token}, theme=theme)
                body, kind = json.dumps(document).encode(), "application/json"
            elif path == "/status.json":  # unchanged: never encodes, no page fields
                body, kind = json.dumps(self.page.status(), indent=2).encode(), "application/json"
            else:
                body = render_html(self.page.status(with_picture=True, for_page=True), theme=theme,
                                   marker=self.page.marker(), chat_token=self.page.chat_token, path=path).encode()
                kind = "text/html; charset=utf-8"
        except Exception:  # noqa: BLE001 - never show exception text
            self._send(500, "text/plain; charset=utf-8", b"status unavailable\n")
            return
        self._send(200, kind, body)

    def _send_snapshot(self, incident_id: str, *, download: bool) -> None:
        """A saved snapshot, only after its row, name, size and SHA-256 check; the ID is never echoed."""
        if not self.page._picture_slots.acquire(blocking=False):
            self._send(503, "text/plain; charset=utf-8", b"busy: too many picture requests\n")
            return
        try:
            try:
                state, data = self.page.snapshot(incident_id)
            except Exception:  # noqa: BLE001 - never show exception text
                self._send(500, "text/plain; charset=utf-8", b"snapshot unavailable\n")
                return
            if state == "unavailable":
                self._send(503, "text/plain; charset=utf-8", b"incident database unavailable\n")
            elif state == "not_verified":
                self._send(409, "text/plain; charset=utf-8", b"snapshot file missing or not matching its record\n")
            elif state != "ok" or data is None:
                self._send(404, "text/plain; charset=utf-8", b"no snapshot\n")
            else:
                disposition = "attachment" if download else "inline"
                self._send(200, "image/jpeg", data,
                           extra={"Content-Disposition": f'{disposition}; filename="sentinel-{incident_id}.jpg"'})
        finally:
            self.page._picture_slots.release()

    def _send_live_frame(self, query: str) -> None:
        """The player's next picture (D75): the newest live one newer than ``after``, or, after waiting at most
        STREAM_WAIT_S, why there is none (JSON). Only for this server's pages (its token, same origin); at most
        MAX_STREAM_REQUESTS at once. A picture carries its frame, ingest time and age when sent in headers."""
        if self.page._preview is None:
            self._json(404, {"state": "off", "reason": PREVIEW_OFF["reason"]})
            return
        if not self._token_ok() or self.headers.get("Sec-Fetch-Site", "same-origin") != "same-origin":
            self.server.count("stream_refused")  # type: ignore[attr-defined]
            self._json(403, {"state": "refused", "reason": "this page was served by an earlier runtime: reload it"})
            return
        after = None
        for part in query.split("&")[:8]:
            key, _, value = part.partition("=")
            if key == "after":
                after = parse_token(value)
                if after is None:
                    self._json(400, {"state": "invalid", "reason": "after must be <stream epoch>-<frame sequence>"})
                    return
        if not self.page._stream_slots.acquire(blocking=False):
            self.server.count("stream_busy")  # type: ignore[attr-defined]
            self._json(503, {"state": "busy", "reason": "too many live picture requests at once"})
            return
        try:
            try:
                preview = self.page._preview()
                answer = None if preview is None else preview.next_frame(after, STREAM_WAIT_S)
                stale_after_ms = None if preview is None else preview.bounds()["stale_after_ms"]
            except Exception:  # noqa: BLE001 - never show exception text
                self._json(500, {"state": "error", "reason": "the live picture is unavailable"})
                return
            if answer is None:
                self._json(200, {**PREVIEW_STARTING, "frame": None, "age_ms": None, "stale_after_ms": None})
            elif answer.picture is None:
                view = answer.view or {}
                self._json(200, {**{key: view.get(key) for key in ("state", "reason", "age_ms", "frame")},
                                 "stale_after_ms": stale_after_ms})
            else:
                picture = answer.picture
                self._send(200, "image/jpeg", picture.jpeg, extra={
                    "X-Frame": picture.token, "X-Frame-Ingest": frame_facts(picture.frame)["ingest_utc"],
                    "X-Frame-Age-Ms": str(answer.age_ms), "X-Frame-Size": f"{picture.width}x{picture.height}",
                    "X-Time-Quality": picture.frame.source_time_quality.value,
                    "X-Stale-After-Ms": str(stale_after_ms)})
        finally:
            self.page._stream_slots.release()

    def _send_picture(self, token: str) -> None:
        """The live view's picture named ``token``, only while its frame is live; the token is never echoed."""
        if not self.page._picture_slots.acquire(blocking=False):
            self._send(503, "text/plain; charset=utf-8", b"busy: too many picture requests\n")
            return
        try:
            try:
                picture = self.page.preview_picture(token)
            except Exception:  # noqa: BLE001 - never show exception text
                self._send(500, "text/plain; charset=utf-8", b"picture unavailable\n")
                return
            if picture is None:
                self._send(404, "text/plain; charset=utf-8", b"no current picture\n")
                return
            self._send(200, "image/jpeg", picture.jpeg)
        finally:
            self.page._picture_slots.release()

    def _refuse(self) -> None:
        self._send(405, "text/plain; charset=utf-8", b"read-only: GET only\n", allow="GET")

    do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _refuse  # noqa: N815

    def do_POST(self) -> None:  # noqa: N802 - only the chat route takes a POST (D74)
        if urlsplit(self.path).path != CHAT_PATH:
            self._refuse()
            return
        if not self._host_allowed():
            self._send(421, "text/plain; charset=utf-8", b"misdirected request\n")
            return
        problem = self._chat_request_problem()
        if problem is not None:
            self._json(*problem)
            return
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            self._json(411, {"state": "refused", "reason": "a Content-Length is required"})
            return
        if int(length) > CHAT_MAX_BODY:
            self._json(413, {"state": "invalid", "reason": "request too large"})
            return
        body = self.rfile.read(int(length))
        if self.page._chat is None:
            self._json(503, dict(CHAT_OFF))
            return
        service = self.page.chat_service()
        if service is None:
            self._json(503, dict(CHAT_STARTING))
            return
        try:
            question = parse_chat_request(body)
        except ChatRequestError as exc:
            self._json(400, {"state": "invalid", "reason": exc.label})
            return
        try:
            self._json(*service.ask(question))
        except Exception:  # noqa: BLE001 - never show exception text
            self._json(500, {"state": "error", "reason": "the question could not be taken"})

    def _chat_request_problem(self) -> tuple[int, dict[str, str]] | None:
        """Why a POST /chat is not from this page, or None. The token check runs last."""
        port = self.server.server_address[1]
        if self.headers.get("Origin") not in (f"http://{LOOPBACK_HOST}:{port}", f"http://localhost:{port}"):
            return 403, {"state": "refused", "reason": "only this page may ask"}
        if self.headers.get("Sec-Fetch-Site", "same-origin") != "same-origin":
            return 403, {"state": "refused", "reason": "only this page may ask"}
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            return 415, {"state": "refused", "reason": "a JSON body is required"}
        if self.headers.get("Transfer-Encoding") is not None:
            return 411, {"state": "refused", "reason": "a Content-Length is required"}
        if not self._token_ok():
            return 403, {"state": "refused", "reason": "only this page may ask"}
        return None

    def _token_ok(self) -> bool:
        """The token this server put in its page, compared in constant time (as bytes: any header text compares)."""
        given = self.headers.get("X-Sentinel-Token", "").encode("latin-1", "replace")
        return hmac.compare_digest(given, self.page.chat_token.encode("ascii"))

    def _chat_poll(self, job_id: str) -> None:
        if not self._token_ok():
            self._json(403, {"state": "refused", "reason": "only this page may ask"})
            return
        service = self.page.chat_service()
        if service is None:
            self._json(404, {"state": "unknown"})
            return
        self._json(*service.poll(job_id))

    def _json(self, code: int, document: Mapping[str, Any]) -> None:
        self._send(code, "application/json", json.dumps(document).encode())

    def _host_allowed(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in (f"{LOOPBACK_HOST}:{port}", f"localhost:{port}")

    def _send(self, code: int, kind: str, body: bytes, *, allow: str | None = None,
              extra: Mapping[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        if allow is not None:
            self.send_header("Allow", allow)
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        try:
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except TimeoutError:  # a browser that stopped reading; http.server then drops the connection silently
            self.server.count("write_timeouts")  # type: ignore[attr-defined]
            raise

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - no request logging to stderr
        return
