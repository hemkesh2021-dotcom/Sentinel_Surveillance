"""Loopback-only, read-only status page for ``sentinel run`` (D-2; demo only).

It stands in for V2-17/V2-18 at the Oct 20 demo and replaces neither: there are
no sessions, roles, API or media. It is reached over an SSH port forward.

- **Loopback only.** It binds 127.0.0.1 and nothing else; any other host is
  refused before a socket is opened. Requests must name 127.0.0.1 or localhost
  with the page's port in their Host header, so a web page on another origin
  cannot read it through DNS rebinding.
- **Read only.** GET ``/`` (HTML, refreshed every 2 s) and GET ``/status.json``.
  Every other method is refused. The database is read through a new read-only
  SQLite connection per request (``connect_reader``), in one short read
  transaction, so the page never writes, migrates or holds the writer's lock.
- **What it shows.** Component readiness, capture and reconnect state, video
  freshness, processing rates, degradation reasons, recent incidents and
  delivery state. Delivery separates *queued* (never attempted), *attempted*
  (sent at least once, not confirmed: retrying or in flight), *delivered* (the
  provider confirmed it) and *failed* (dead-lettered, needs an operator).
  *Ambiguous* marks rows the provider may have accepted although that was not
  confirmed.
- **What it never shows.** Credentials, tokens, chat IDs, stream URLs, hosts,
  images or exception text. The runtime snapshot holds numbers and fixed
  labels only; stored delivery errors were redacted when written; every value
  is HTML-escaped. Responses are not cached and may not be framed.

Standard library only; portable.
"""

from __future__ import annotations

import html
import json
import sqlite3
import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .incidents.service import UNRESOLVED
from .storage.database import connect_reader

LOOPBACK_HOST = "127.0.0.1"
MAX_INCIDENTS = 20
MAX_DELIVERIES = 30
REFRESH_S = 2
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
            deliveries = []
            for row in connection.execute(
                """SELECT outbox_id, incident_id, channel, message_kind, status, attempts, ambiguous, last_error,
                          next_attempt_utc, updated_utc FROM outbox ORDER BY outbox_id DESC LIMIT ?""",
                (MAX_DELIVERIES,),
            ):
                (outbox_id, incident_id, channel, kind, status, attempts, flagged, error, next_utc, updated) = row
                state = delivery_state(status, attempts)
                deliveries.append({
                    "outbox_id": outbox_id, "incident_id": incident_id, "channel": channel, "message_kind": kind,
                    "state": state, "in_flight": status == "leased", "attempts": attempts,
                    "ambiguous": bool(flagged), "last_error": error,
                    "next_attempt_utc": next_utc if state in ("queued", "attempted") else None,
                    "updated_utc": updated,
                })
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


def _unavailable(exc: Exception) -> dict[str, Any]:
    return {"state": "unavailable", "problem": f"database_unreadable:{type(exc).__name__}",
            "incidents": None, "delivery": None}


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


def _e(value: object) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return html.escape(str(value), quote=True)


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
        [i["incident_id"], i["title"], i["severity"], i["status"], i["first_observed_utc"], i["last_observed_utc"],
         i["annotations"]]
        for i in incidents.get("recent") or []
    ]
    delivery_rows = [
        [d["incident_id"], d["channel"], d["message_kind"],
         d["state"] + (" (in flight)" if d["in_flight"] else ""), d["attempts"],
         "may have been delivered" if d["ambiguous"] else "", d["next_attempt_utc"], d["last_error"]]
        for d in delivery.get("recent") or []
    ]
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<meta http-equiv=refresh content="{REFRESH_S}"><title>Sentinel status</title>
<style>
:root{{--bg:#fff;--fg:#1d1f21;--muted:#666;--line:#ddd;--ok:#e6f4ea;--bad:#fdecea}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181a;--fg:#e8e8e8;--muted:#9a9a9a;--line:#333;--ok:#173322;--bad:#3b1d1a}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:16px;max-width:1100px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:20px 0 6px}}
table{{border-collapse:collapse;margin:4px 0}} th,td{{border-bottom:1px solid var(--line);padding:3px 10px 3px 0;text-align:left;vertical-align:top}}
table.list{{width:100%}} .muted{{color:var(--muted)}} .banner{{padding:8px 12px;border-radius:6px;margin:10px 0}}
.ok{{background:var(--ok)}} .bad{{background:var(--bad)}} .grid{{display:flex;flex-wrap:wrap;gap:24px}}
.wrap{{overflow-x:auto}} ul{{margin:4px 0 0 18px;padding:0}}
</style></head><body>
<h1>Sentinel status: camera {_e(runtime.get("camera_id"))}</h1>
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
                         delivery_rows, "No alerts.")}</div>
</body></html>
"""


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
        if path not in ("/", "/status.json"):
            self._send(404, "text/plain; charset=utf-8", b"not found\n")
            return
        try:
            status = self.page.status()
            if path == "/status.json":
                body, kind = json.dumps(status, indent=2).encode(), "application/json"
            else:
                body, kind = render_html(status).encode(), "text/html; charset=utf-8"
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
