"""D-2: the loopback-only, read-only status page (stdlib HTTP on 127.0.0.1; fake runtime data, real SQLite)."""

from __future__ import annotations

import http.client
import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from sentinel.alerts.outbox import DeliveryResult, DeliveryStatus, OutboxWorker
from sentinel.config import IncidentsConfig, NotificationsConfig
from sentinel.incidents.service import IncidentService
from sentinel.incidents.signals import IncidentSignal, SignalPhase
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.rules.scene_hazard import Severity
from sentinel.status_page import StatusServer, build_status, delivery_state, read_store, render_html
from sentinel.storage.database import Database

T0 = datetime(2026, 10, 4, 6, 0, tzinfo=timezone.utc)
FAKE_TOKEN = "123456789:FAKE-test-token-not-real"
SETTINGS = NotificationsConfig(channels=["telegram"])


def signal(zone: str) -> IncidentSignal:
    return IncidentSignal(
        observation_id=f"zone-{zone}.entered", episode_id=f"zone-{zone}", camera_id="cam-1",
        kind="zone.restricted_entry", phase=SignalPhase.ENTERED, zone_id=zone, rule_revision="rev1",
        severity=Severity.WARNING, title=f"Person in restricted zone {zone!r}", observed_utc=T0, boot_id="boot-1",
        observed_mono_ns=NS_PER_SECOND, reason="test", payload="{}",
    )


class Scripted:
    channel = "telegram"

    def __init__(self, *results: DeliveryResult) -> None:
        self.results = list(results)

    def send(self, message) -> DeliveryResult:
        return self.results.pop(0)


def store_with_every_delivery_state(tmp_path: Path) -> Path:
    """Four incidents whose alerts are delivered, attempted (ambiguous), failed and queued."""
    clock = FakeClock(utc=T0)
    db = Database.open(tmp_path / "sentinel.db")
    service = IncidentService(db, clock, incidents=IncidentsConfig(), notifications=SETTINGS)
    notifier = Scripted(
        DeliveryResult(DeliveryStatus.SENT, "sent", provider_message_id="7"),
        DeliveryResult(DeliveryStatus.RETRY, "timeout after sending", ambiguous=True),
        DeliveryResult(DeliveryStatus.PERMANENT, f"HTTP 401 for https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage"),
    )
    worker = OutboxWorker(db, clock, {"telegram": notifier}, SETTINGS)
    for zone in ("delivered", "attempted", "failed"):
        service.record(signal(zone))
        worker.run_once(limit=1)
    service.record(signal("queued"))
    db.close()
    return tmp_path / "sentinel.db"


SNAPSHOT: dict[str, Any] = {
    "runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": T0.isoformat(), "shutdown": None},
    "components": {
        "capture": {"state": "streaming", "ready": True, "stream_epoch": 3, "connects": 3, "reconnects": 2,
                    "open_failures": 1, "stream_ends": 2, "retry_delay_s": None, "problem": "no_frame"},
        "detector": {"state": "available", "problem": None},
        "scene": {"state": "disabled", "problem": None},
        "incidents": {"state": "ok", "pending_signals": 0, "signals_dropped": 0, "problem": None},
        "notifications": {"channels": {"telegram": {"state": "available", "problem": None}},
                          "worker": {"state": "running", "problem": None}},
    },
    "frames": {"captured": 900, "processed": 880, "failed": 0, "replaced": 20, "not_live": 1},
    "rates": {"window_s": 10.0, "captured_fps": 15.0, "processed_fps": 14.7, "failed_per_s": 0.0},
    "live": {"video": "fresh", "last_frame_age_ms": 40, "detector": "available", "occupancy": "occupied",
             "occupancy_reason": "1 confirmed person(s) on fresh video", "people": 1, "confirmed_people": 1,
             "face_recognition": "disabled", "scene_analysis": "disabled", "scene": "no_current_result",
             "scene_reason": "scene analysis disabled", "scene_report": None},
    "degraded": [],
}


def get(server: StatusServer, path: str, *, host: str | None = None, method: str = "GET"):
    address, port = server.address
    conn = http.client.HTTPConnection(address, port, timeout=5)
    try:
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", host if host is not None else f"127.0.0.1:{port}")
        conn.endheaders()
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


@pytest.fixture
def page(tmp_path):
    db_path = store_with_every_delivery_state(tmp_path)
    state = {"snapshot": SNAPSHOT}
    server = StatusServer(0, lambda: state["snapshot"], db_path, now=lambda: T0 + timedelta(seconds=1))
    server.start()
    server.state = state  # type: ignore[attr-defined]
    yield server
    assert server.stop(2.0)


# ---------------------------------------------------------------- delivery states


def test_delivery_states_separate_queued_attempted_delivered_and_failed() -> None:
    assert delivery_state("pending", 0) == "queued"
    assert delivery_state("pending", 2) == "attempted"
    assert delivery_state("leased", 1) == "attempted"
    assert delivery_state("sent", 1) == "delivered"
    assert delivery_state("dead", 8) == "failed"


def test_the_store_reports_every_delivery_state_with_ambiguity_and_redacted_errors(tmp_path) -> None:
    store = read_store(store_with_every_delivery_state(tmp_path))
    assert store["state"] == "available"
    assert store["delivery"]["totals"] == {"queued": 1, "attempted": 1, "delivered": 1, "failed": 1, "ambiguous": 1}
    by_zone = {row["incident_id"]: row for row in store["delivery"]["recent"]}
    titles = {i["incident_id"]: i["zone_id"] for i in store["incidents"]["recent"]}
    states = {titles[incident]: (row["state"], row["attempts"], row["ambiguous"]) for incident, row in by_zone.items()}
    assert states == {"delivered": ("delivered", 1, False), "attempted": ("attempted", 1, True),
                      "failed": ("failed", 1, False), "queued": ("queued", 0, False)}
    assert store["incidents"]["unresolved"] == 4
    assert FAKE_TOKEN not in json.dumps(store) and "bot<redacted>" in json.dumps(store)


def test_the_store_reads_without_writing_while_the_runtime_owns_the_database(tmp_path) -> None:
    db_path = store_with_every_delivery_state(tmp_path)
    writer = Database.open(db_path)  # holds the single-writer lock, as the runtime does
    try:
        before = db_path.read_bytes()
        assert read_store(db_path)["state"] == "available"
        assert db_path.read_bytes() == before
    finally:
        writer.close()


def test_a_missing_or_unreadable_database_is_reported_by_label(tmp_path) -> None:
    store = read_store(tmp_path / "absent.db")
    assert not (tmp_path / "absent.db").exists()  # the page never creates a database
    assert store == {"state": "unavailable", "problem": "database_unreadable:OperationalError",
                     "incidents": None, "delivery": None}
    status = build_status(SNAPSHOT, store, T0)
    assert "incident database unavailable (database_unreadable:OperationalError)" in status["degraded"]


def test_a_runtime_snapshot_that_stopped_updating_is_called_out() -> None:
    status = build_status(SNAPSHOT, {"state": "available"}, T0 + timedelta(seconds=30))
    assert status["degraded"][0] == "runtime status not updated for 30 s"
    starting = build_status({"runtime": {"state": "starting", "camera_id": "cam-1"}}, {"state": "available"}, T0)
    assert starting["runtime"]["state"] == "starting" and starting["degraded"] == []
    assert "starting" in render_html(starting)


# ---------------------------------------------------------------- loopback, read only


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "localhost", "192.0.2.10", ""])
def test_any_bind_other_than_127_0_0_1_is_refused_before_a_socket_opens(tmp_path, host: str) -> None:
    with pytest.raises(ValueError, match="127.0.0.1 only"):
        StatusServer(0, lambda: SNAPSHOT, tmp_path / "sentinel.db", host=host)


def test_the_page_listens_on_loopback_and_serves_html_and_json(page) -> None:
    assert page.address[0] == "127.0.0.1"
    status, headers, body = get(page, "/")
    assert status == 200 and headers["Content-Type"] == "text/html; charset=utf-8"
    assert headers["Cache-Control"] == "no-store" and headers["X-Frame-Options"] == "DENY"
    assert headers["X-Content-Type-Options"] == "nosniff" and "default-src 'none'" in headers["Content-Security-Policy"]
    assert headers["Server"].startswith("sentinel-status") and "Python" not in headers["Server"]
    text = body.decode()
    for phrase in ("All components working", "stream epoch", "reconnects", "processed fps", "Queued (never attempted) 1",
                   "attempted, not confirmed 1", "delivered (provider confirmed) 1", "failed (needs operator) 1",
                   "may have been delivered"):
        assert phrase in text, phrase
    status, headers, body = get(page, "/status.json")
    document = json.loads(body)
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert document["components"]["capture"]["reconnects"] == 2
    assert document["delivery"]["totals"]["delivered"] == 1
    assert document["rates"]["processed_fps"] == 14.7


def test_the_page_is_read_only(page) -> None:
    for method in ("POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
        status, headers, _ = get(page, "/", method=method)
        assert (status, headers.get("Allow")) == (405, "GET"), method
    assert get(page, "/incidents/1")[0] == 404
    assert get(page, "/../sentinel.db")[0] == 404


@pytest.mark.parametrize("host", ["evil.example", "evil.example:18090", "127.0.0.1", "127.0.0.1:1", ""])
def test_requests_naming_another_host_are_refused(page, host: str) -> None:
    status, _, body = get(page, "/status.json", host=host)
    assert status == 421 and b"cam-1" not in body


def test_localhost_with_the_page_port_is_accepted(page) -> None:
    assert get(page, "/", host=f"localhost:{page.address[1]}")[0] == 200


def test_degradation_is_shown_and_model_text_is_escaped(page) -> None:
    page.state["snapshot"] = {
        **SNAPSHOT,
        "degraded": ["capture waiting (open_failed)", "detector unavailable (libcuda_not_l4t)"],
        "live": {**SNAPSHOT["live"], "scene": "reported", "scene_report": {
            "persons_visible": 1, "fire_or_smoke": False, "threat": "low", "uncertainty": "low",
            "summary": "<script>alert(1)</script>"}},
    }
    text = get(page, "/")[2].decode()
    assert "Degraded" in text and "detector unavailable (libcuda_not_l4t)" in text
    assert "<script>" not in text and "&lt;script&gt;alert(1)&lt;/script&gt;" in text


def test_a_failure_while_building_the_page_shows_no_exception_text(page) -> None:
    page.state["snapshot"] = None  # the runtime handed over something unusable
    page._snapshot = lambda: (_ for _ in ()).throw(RuntimeError("rtsp://admin:hunter2@192.0.2.10/stream"))
    status, _, body = get(page, "/")
    assert status == 500 and body == b"status unavailable\n"


def test_stop_closes_the_socket(tmp_path) -> None:
    server = StatusServer(0, lambda: SNAPSHOT, tmp_path / "sentinel.db")
    server.start()
    port = server.address[1]
    assert server.stop(2.0)
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0
