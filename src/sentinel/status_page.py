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
- **Live view (D69; only with ``sentinel run --live-preview``).** The main
  page shows the newest frame the runtime already reads, with its state
  (live, stale, offline, starting), its age and its ingest time, and a
  picture only while that frame is live, at GET ``/preview/<epoch>-<seq>.jpg``
  (``live_preview``). The picture refreshes with the page. If the page stops
  refreshing (a stalled tunnel or a stopped runtime), CSS alone dims it and
  marks it not current after PREVIEW_EXPIRY_S, so a frozen frame is never
  shown as current. At most MAX_PICTURE_RESPONSES pictures are sent at once.
- **Incident snapshot (D70; saved only by ``sentinel run --snapshots``).**
  The incident page shows the frame the opening observation came from, its
  identity, ingest time and time quality, with GET
  ``/incidents/<id>/snapshot.jpg`` (inline) and ``/incidents/<id>/snapshot/download``
  (attachment). A file is served only by incident ID: its name must be the
  one derived from that ID, it is opened inside the data directory's
  ``snapshots`` folder without following links or blocking, and its size and
  SHA-256 must match the row. A missing, pruned or altered snapshot is stated
  as such, and never replaced by another frame.
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
import html
import json
import os
import re
import socket
import sqlite3
import stat
import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from .contracts import Evidence, EvidenceStatus, SourceTimeQuality
from .incidents.service import UNRESOLVED, latest_acknowledgment
from .incidents.snapshots import SNAPSHOT_DIR, SnapshotPolicy, file_name as snapshot_file_name
from .live_preview import LivePreview, PreviewPicture, time_label
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
PREVIEW_PATH = re.compile(r"/preview/(?P<token>[0-9]{1,9}-[0-9]{1,12})\.jpg")
PREVIEW_EXPIRY_S = 5  # the shown picture is dimmed and marked not current once the page is this old
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
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
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
.wrap{overflow-x:auto} ul{margin:4px 0 0 18px;padding:0}
.pv{position:relative;display:inline-block;margin:4px 0;max-width:100%}
.snap{margin:4px 0} .snap img{display:block;max-width:100%;height:auto}
.pv img{display:block;max-width:100%;height:auto;animation:pv-old .1s linear EXPIRYs forwards}
.pv figcaption{visibility:hidden;position:absolute;left:0;right:0;top:0;padding:8px;background:var(--bad);
font-weight:bold;animation:pv-show .1s linear EXPIRYs forwards}
@keyframes pv-old{to{opacity:.12}} @keyframes pv-show{to{visibility:visible}}""".replace("EXPIRY", str(PREVIEW_EXPIRY_S))


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


PREVIEW_LABELS = {"live": "LIVE", "stale": "STALE: not current", "offline": "OFFLINE", "starting": "STARTING"}


def _seconds(ms: object) -> str | None:
    return None if not isinstance(ms, int) else f"{ms / 1000:.1f} s"


def _preview_html(view: Mapping[str, Any] | None) -> str:
    """The live view: state, age, ingest time, and a picture only while its frame is live."""
    view = view or PREVIEW_OFF
    if not view.get("enabled"):
        return f"<section><h2>Live view</h2><p class=muted>{_e(view.get('reason'))}.</p></section>"
    state = view.get("state")
    picture = view.get("picture")
    shown = picture or view.get("frame")
    banner = (f"<div class='banner {'ok' if state == 'live' and picture else 'bad'}'>"
              f"<strong>{_e(PREVIEW_LABELS.get(str(state), str(state).upper()))}</strong>: {_e(view.get('reason'))}</div>")
    if picture:
        body = (f'<figure class=pv><img src="{_e(picture["url"])}" width="{_e(picture["width"])}" '
                f'height="{_e(picture["height"])}" alt="camera picture, stream epoch {_e(picture["stream_epoch"])} '
                f'frame {_e(picture["frame_seq"])}"><figcaption>NOT CURRENT: this page has not refreshed for '
                f"{PREVIEW_EXPIRY_S} s, so this picture is old.</figcaption></figure>")
    elif state == "live":
        body = f"<p>No picture: {_e(view.get('problem') or 'none encoded yet')}.</p>"
    else:
        body = "<p>No picture is shown: the last frame is not current.</p>" if shown else ""
    facts = ""
    if shown:
        age = picture["age_ms"] if picture else view.get("age_ms")
        facts = "<table>" + _rows([
            ("frame", f"stream epoch {shown['stream_epoch']}, frame {shown['frame_seq']}"),
            ("frame age when this page was generated", _seconds(age)),
            ("time (UTC)", shown["ingest_utc"]),
            ("what the time means", view.get("time_label")),
        ]) + "</table>"
    bounds = view.get("bounds") or {}
    note = (f"Refreshes with the page every {REFRESH_S} s. Pictures are at most {bounds.get('max_width')}×"
            f"{bounds.get('max_height')}, at most {bounds.get('max_encodes_per_s')} encoded per second for all "
            "viewers, from the frames the runtime already reads; nothing is saved.")
    return f"<section><h2>Live view</h2>{banner}{body}{facts}<p class=muted>{_e(note)}</p></section>"


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
    for name, label in (("preview", "live view"), ("snapshots", "incident snapshots")):  # listed only when enabled
        item = components.get(name)
        if item is not None:
            component_rows.append([label, item.get("state"), item.get("problem")])
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
{_preview_html(status.get("preview"))}
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


def _snapshot_html(snapshot: Mapping[str, Any] | None) -> str:
    """The snapshot section: the verified picture with a download link, or why there is none."""
    snapshot = snapshot or {"state": "none", "text": SNAPSHOT_NONE}
    state = snapshot["state"]
    if state == "saved" and snapshot.get("file") == "verified":
        frame = snapshot["frame"]
        body = (f'<figure class=snap><img src="{_e(snapshot["view_url"])}" width="{_e(snapshot["width"])}" '
                f'height="{_e(snapshot["height"])}" alt="snapshot: stream epoch {_e(frame["stream_epoch"])} frame '
                f'{_e(frame["frame_seq"])}"></figure><p><a href="{_e(snapshot["download_url"])}" download>Download this '
                f'snapshot (JPEG, {_e(snapshot["bytes"])} bytes)</a></p>')
    elif state == "saved":
        body = ("<div class='banner bad'>Saved, but its file is missing or does not match its record (size or "
                "SHA-256), so it is not shown.</div>")
    else:
        body = f"<p>{_e(snapshot['text'])}.</p>"
    facts = ""
    if "frame" in snapshot:
        rows: list[tuple[str, object]] = [
            ("frame", f"stream epoch {snapshot['frame']['stream_epoch']}, frame {snapshot['frame']['frame_seq']}"),
            ("observation", snapshot["observation_id"]),
            ("time (UTC)", snapshot["ingest_utc"] or "not known: the frame's record was no longer held"),
            ("what the time means", snapshot["time_label"]),
            ("recorded (UTC)", snapshot["recorded_utc"]),
        ]
        if snapshot["sha256"]:
            rows += [("size", f"{snapshot['width']}×{snapshot['height']}, {snapshot['bytes']} bytes"),
                     ("SHA-256", snapshot["sha256"])]
        facts = f"<table>{_rows(rows)}</table>"
    return f"{body}{facts}<p class=muted>{_e(SNAPSHOT_NOTE)}</p>"


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
<h2>Snapshot</h2>{_snapshot_html(detail.get("snapshot"))}
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


class _Server(ThreadingHTTPServer):
    """At most ``max_connections`` connections at once; errors are counted, never printed."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], handler: type, max_connections: int) -> None:
        super().__init__(address, handler)
        self._slots = threading.BoundedSemaphore(max_connections)
        self._counts_lock = threading.Lock()
        self.counts = {"refused_busy": 0, "write_timeouts": 0, "connection_errors": 0}

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
        self._picture_slots = threading.BoundedSemaphore(MAX_PICTURE_RESPONSES)
        handler = type("StatusHandler", (_Handler,), {"page": self, "timeout": handler_timeout_s})
        self._httpd = _Server((LOOPBACK_HOST, port), handler, max_connections)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    def status(self, *, with_picture: bool = False) -> dict[str, Any]:
        """``with_picture`` (the HTML page only): let the live view encode a picture if one is due."""
        return build_status(self._snapshot(), read_store(self._db_path), self._now(),
                            self.preview_view(with_picture=with_picture))

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
        path = urlsplit(self.path).path
        preview = PREVIEW_PATH.fullmatch(path)
        if preview is not None:
            self._send_picture(preview["token"])
            return
        snapshot = SNAPSHOT_PATH.fullmatch(path)
        if snapshot is not None:
            self._send_snapshot(snapshot["id"], download=snapshot["kind"] == "/download")
            return
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
                body, kind = render_html(self.page.status(with_picture=True)).encode(), "text/html; charset=utf-8"
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

    do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _refuse  # noqa: N815

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
