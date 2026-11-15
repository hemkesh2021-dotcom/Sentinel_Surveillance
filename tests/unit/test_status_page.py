"""D-2: the loopback-only, read-only status page and incident detail (fake runtime data, real SQLite)."""

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
from sentinel.contracts import Applicability, Evidence, EvidenceStatus, FrameKey, FrameRef, PixelFormat, TrackKey
from sentinel.incidents.service import IncidentService, IncidentStatus
from sentinel.incidents.signals import IncidentSignal, SignalPhase, signal_from_zone
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.rules.scene_hazard import Severity
from sentinel.rules.zones import ZoneObservation, ZonePhase
from sentinel.status_page import (
    MAX_ANNOTATIONS,
    MAX_EVIDENCE,
    StatusServer,
    build_status,
    delivery_state,
    read_incident,
    read_store,
    render_html,
    render_incident_html,
)
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


def test_the_in_memory_buffer_limitation_is_always_shown(page) -> None:
    from sentinel.demo_runtime import PENDING_DURABILITY

    page.state["snapshot"] = {**SNAPSHOT, "components": {**SNAPSHOT["components"], "incidents": {
        "state": "degraded", "pending_signals": 3, "pending_limit": 256, "durability": PENDING_DURABILITY,
        "signals_dropped": 0, "problem": "record_failed:OperationalError"}}}
    text = get(page, "/")[2].decode()
    assert "Waiting to be recorded: 3." in text and "not a crash-safe spool" in text
    assert "at most 256" in text
    document = json.loads(get(page, "/status.json")[2])
    assert document["components"]["incidents"]["durability"] == PENDING_DURABILITY
    page.state["snapshot"] = {"runtime": {"state": "starting", "camera_id": "cam-1"}}  # before the runtime exists
    assert "not a crash-safe spool" in get(page, "/")[2].decode()


def test_face_recognition_and_identity_counts_are_shown_without_ids(page) -> None:
    page.state["snapshot"] = {
        **SNAPSHOT,
        "components": {**SNAPSHOT["components"], "face": {"state": "available", "problem": None}},
        "live": {**SNAPSHOT["live"], "face_recognition": "available", "identity": {
            "unresolved": 1, "unknown": 0, "known": 1, "uncertain": 0, "fresh": 1, "retained": 0}},
    }
    text = get(page, "/")[2].decode()
    assert "<span class=name>face</span>" in text and "Face recognition: available" in text
    assert "known 1 (fresh 1, retained 0) · unknown 0 · uncertain 0 · unresolved 1" in text
    assert "idn-" not in text and "experimental" not in text


def test_face_recognition_in_a_validation_run_is_shown_as_experimental(page) -> None:
    """D92: in a --face-validation run (the only way face can run before its validation) the chip and the identity
    counts say "experimental: a validation run, not validated"; the counts themselves are unchanged."""
    page.state["snapshot"] = {
        **SNAPSHOT,
        "components": {**SNAPSHOT["components"], "face": {"state": "available", "problem": None,
                                                          "validation_run": True, "detection": "full_scale"}},
        "live": {**SNAPSHOT["live"], "face_recognition": "available", "identity": {
            "unresolved": 0, "unknown": 0, "known": 1, "uncertain": 0, "fresh": 1, "retained": 0}},
    }
    text = get(page, "/")[2].decode()
    assert "Face recognition: available (experimental: a validation run, not validated)" in text
    assert "identities (counts only; experimental: a validation run, not validated)" in text
    assert "known 1 (fresh 1, retained 0)" in text and "idn-" not in text
    regions = json.loads(get(page, "/status.json?page=1")[2])["page"]["regions"]  # the in-place update too
    assert "experimental: a validation run, not validated" in regions["caps"]


# ---------------------------------------------------------------- incident detail (guide ch. 15, V2-28 display)

REPORT = {"persons_visible": 1, "fire_or_smoke": False, "threat": "low", "observations": ["one person"],
          "uncertainty": "low", "summary": "One person stands by the door."}


def zone_observation(phase: ZonePhase, *, episode: str = "zone-door-ep1", at_s: float = 2.0, track_id: int = 4,
                     epoch: int = 2, first_seq: int = 10, last_seq: int = 40) -> ZoneObservation:
    def key(seq: int) -> FrameKey:
        return FrameKey(camera_id="cam-1", boot_id="boot-1", run_id="run-1", stream_epoch=epoch, frame_seq=seq)

    return ZoneObservation(
        observation_id=f"{episode}.{phase.value}", episode_id=episode, kind="zone.restricted_entry", phase=phase,
        zone_id="door", zone_revision="door-r1", severity=Severity.WARNING,
        track=TrackKey(camera_id="cam-1", boot_id="boot-1", run_id="run-1", stream_epoch=epoch, track_id=track_id),
        first_source=key(first_seq), last_source=key(last_seq), first_mono_ns=NS_PER_SECOND,
        last_mono_ns=round(at_s * NS_PER_SECOND), observed_utc=T0 + timedelta(seconds=at_s),
        duration_ms=round((at_s - 1.0) * 1000), reason=f"confirmed track {track_id} in restricted zone 'door'",
    )


def source_frame() -> FrameRef:
    stamper = FrameStamper("cam-1", FakeClock(utc=T0))
    stamper.connect()
    return stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)


def scene(incident_id: str, job: str, *, status: EvidenceStatus = EvidenceStatus.OBSERVED, value: object = REPORT,
          kind: str = "scene.report", late: bool = True) -> Evidence:
    observed = status is EvidenceStatus.OBSERVED
    return Evidence.observed_on(
        source_frame(), evidence_id=f"{job}.{status.value}" + (".late" if late else ""), kind=kind, status=status,
        value=value if observed else None, producer="scene.stub", producer_revision="1+stub",
        ttl_ns=10 * NS_PER_SECOND, correlation_group="scene:cam-1",
        reason="stub result" if observed else "job timed out", incident_id=incident_id,
    )


def one_incident_with_its_evidence_chain(tmp_path: Path) -> tuple[Path, str]:
    """A zone incident (ENTERED, ENDED), a timeout and a late report, and an alert sent on its second attempt."""
    clock = FakeClock(utc=T0)
    db = Database.open(tmp_path / "sentinel.db")
    service = IncidentService(db, clock, incidents=IncidentsConfig(), notifications=SETTINGS)
    incident_id = service.record(signal_from_zone(zone_observation(ZonePhase.ENTERED))).incident_id
    assert incident_id is not None
    clock.advance(1.0)
    assert service.annotate(scene(incident_id, "job-1", status=EvidenceStatus.TIMEOUT, late=False),
                            Applicability.CURRENT).outcome.value == "annotated"
    clock.advance(11.5)
    assert service.annotate(scene(incident_id, "job-1"), Applicability.EXPIRED).outcome.value == "annotated"
    service.record(signal_from_zone(zone_observation(ZonePhase.ENDED, at_s=6.0, last_seq=95)))
    worker = OutboxWorker(db, clock, {"telegram": Scripted(
        DeliveryResult(DeliveryStatus.RETRY, f"HTTP 502 for https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage"),
        DeliveryResult(DeliveryStatus.SENT, "sent", provider_message_id="7"),
    )}, SETTINGS)
    worker.run_once()
    clock.advance(3600)
    worker.run_once()
    db.close()
    return tmp_path / "sentinel.db", incident_id


def test_an_incident_page_shows_rule_evidence_lifecycle_scene_evidence_and_alert_attempts(tmp_path) -> None:
    db_path, incident_id = one_incident_with_its_evidence_chain(tmp_path)
    detail = read_incident(db_path, incident_id)
    assert detail["state"] == "available"
    assert detail["incident"]["status"] == "open" and detail["incident"]["severity"] == "warning"
    entered, ended = detail["evidence"]["shown"]
    assert detail["evidence"]["total"] == 2
    assert (entered["phase"], ended["phase"], entered["episode"], ended["episode"]) == ("entered", "ended", 1, 1)
    assert entered["source"] == {"track_id": 4, "stream_epoch": 2, "first_frame_seq": 10, "last_frame_seq": 40,
                                 "in_zone_s": 1.0, "zone_revision": "door-r1"}
    assert ended["source"]["last_frame_seq"] == 95 and entered["observed_utc"] == "2026-10-04T06:00:02.000000Z"
    assert "payload" not in entered  # never the stored payload itself
    (opened,) = detail["transitions"]["shown"]
    assert (opened["from_status"], opened["to_status"], opened["revision"]) == (None, "open", 1)
    timeout, late = detail["annotations"]["shown"]
    assert (timeout["status"], timeout["applicability"], timeout["details"]["report"]) == ("timeout", "current", None)
    assert (late["status"], late["applicability"]) == ("observed", "expired")
    assert late["details"]["report"]["summary"] == "One person stands by the door."
    assert late["recorded_after_source_s"] == 12.5 and late["details"]["frame_seq"] == 0
    (alert,) = detail["deliveries"]["shown"]
    assert (alert["state"], alert["attempts"], alert["provider_confirmed"]) == ("delivered", 2, True)
    first, second = alert["attempt_log"]["shown"]
    assert (first["attempt"], second["attempt"], second["outcome"]) == (1, 2, "sent")
    assert FAKE_TOKEN not in json.dumps(detail) and "bot<redacted>" in first["detail"]
    assert "provider_message_id" not in alert  # whether the provider confirmed it, never its message ID
    assert detail["acknowledgment"]["state"] == "not_acknowledged" and detail["acknowledgment"]["at_utc"] is None
    text = render_incident_html({"generated_utc": T0.isoformat(), **detail})
    for phrase in ("Person in restricted zone &#x27;door&#x27;", "<td>entered</td>", "<td>ended</td>",
                   "<td>10–40</td>", "<td>expired</td>", "One person stands by the door.", "threat low",
                   "not acknowledged; this page is read only (D45)", "none: event clips are not implemented yet",
                   "Only &#x27;current&#x27; could have changed the live scene", "<td>delivered</td>",
                   'href="/dashboard"'):
        assert phrase in text, phrase
    assert FAKE_TOKEN not in text


def test_the_status_page_links_incidents_to_pages_served_as_html_and_json(tmp_path) -> None:
    db_path, incident_id = one_incident_with_its_evidence_chain(tmp_path)
    server = StatusServer(0, lambda: SNAPSHOT, db_path, now=lambda: T0 + timedelta(seconds=1))
    server.start()
    try:
        main_page = get(server, "/")[2].decode()
        assert f'<a class=inc href="/incidents/{incident_id}">' in main_page and incident_id in main_page
        status, headers, body = get(server, f"/incidents/{incident_id}")
        assert status == 200 and headers["Content-Type"] == "text/html; charset=utf-8"
        assert headers["Cache-Control"] == "no-store" and "default-src 'none'" in headers["Content-Security-Policy"]
        assert f"Incident <span class=mono>{incident_id}</span>" in body.decode()
        status, headers, body = get(server, f"/incidents/{incident_id}.json")
        document = json.loads(body)
        assert status == 200 and headers["Content-Type"] == "application/json"
        assert document["incident"]["incident_id"] == incident_id and document["generated_utc"].startswith("2026")
        for method in ("POST", "PUT", "DELETE"):
            assert get(server, f"/incidents/{incident_id}", method=method)[0] == 405
        assert get(server, f"/incidents/{incident_id}", host="evil.example")[0] == 421
    finally:
        assert server.stop(2.0)


@pytest.mark.parametrize("path", [
    "/incidents/inc-gone", "/incidents/inc-gone.json", "/incidents/", "/incidents/a.b", "/incidents/a:b",
    "/incidents/<script>", "/incidents/%2E%2E", "/incidents/x/y", "/incidents/x.json.json", "/incidents/" + "a" * 129,
    "/incidents/..%2Fsentinel.db",
])
def test_unknown_or_malformed_incident_paths_are_not_found_and_never_echoed(page, path: str) -> None:
    status, _, body = get(page, path)
    assert (status, body) == (404, b"not found\n")


def test_an_unreadable_database_gives_a_fixed_answer_and_is_never_created(tmp_path) -> None:
    missing = tmp_path / "absent.db"
    assert read_incident(missing, "inc-1") == {
        "state": "unavailable", "problem": "database_unreadable:OperationalError"}
    server = StatusServer(0, lambda: SNAPSHOT, missing)
    server.start()
    try:
        for path in ("/incidents/inc-1", "/incidents/inc-1.json"):
            status, _, body = get(server, path)
            assert (status, body) == (503, b"incident database unavailable\n")
    finally:
        assert server.stop(2.0)
    assert not missing.exists()


def test_the_incident_page_reads_without_writing_while_the_runtime_owns_the_database(tmp_path) -> None:
    db_path, incident_id = one_incident_with_its_evidence_chain(tmp_path)
    writer = Database.open(db_path)  # holds the single-writer lock, as the runtime does
    try:
        before = db_path.read_bytes()
        assert read_incident(db_path, incident_id)["state"] == "available"
        assert db_path.read_bytes() == before
    finally:
        writer.close()


def test_only_validated_scene_evidence_is_detailed_and_other_kinds_show_nothing_of_their_value(tmp_path) -> None:
    clock = FakeClock(utc=T0)
    db = Database.open(tmp_path / "sentinel.db")
    service = IncidentService(db, clock, incidents=IncidentsConfig(), notifications=NotificationsConfig())
    incident_id = service.record(signal("door")).incident_id  # payload "{}": not a zone observation
    hostile = {**REPORT, "summary": "<script>alert(1)</script>"}
    service.annotate(scene(incident_id, "job-1", value=hostile), Applicability.SUPERSEDED_EPOCH)
    service.annotate(scene(incident_id, "job-2", kind="identity.face", value={"identity_id": "idn-0123456789ab"}),
                     Applicability.CURRENT)
    service.annotate(scene(incident_id, "job-3", value={"not": "a scene report"}), Applicability.EXPIRED)
    db.close()
    detail = read_incident(tmp_path / "sentinel.db", incident_id)
    (rule,) = detail["evidence"]["shown"]
    assert rule["source"] is None
    report, face, malformed = detail["annotations"]["shown"]
    assert report["applicability"] == "superseded_epoch" and report["details"]["report"]["summary"].startswith("<")
    assert (face["details"], face["details_problem"]) == (None, "not_scene_evidence")
    assert malformed["details"]["report"] is None  # not a valid SceneReport: no report shown, the rest is
    assert (malformed["details"]["report_problem"], malformed["details"]["reason"]) == (
        "not_a_valid_scene_report", "stub result")
    text = render_incident_html(detail)
    assert "idn-0123456789ab" not in text and "idn-0123456789ab" not in json.dumps(detail)
    assert "not shown (not_scene_evidence)" in text and "(source fields not readable)" in text
    assert "not shown (not_a_valid_scene_report)" in text and "a scene report" not in text
    assert "<script>" not in text and "&lt;script&gt;alert(1)&lt;/script&gt;" in text


def test_incident_lists_are_bounded_to_the_latest_rows_with_their_totals(tmp_path) -> None:
    clock = FakeClock(utc=T0)
    db = Database.open(tmp_path / "sentinel.db")
    service = IncidentService(db, clock, incidents=IncidentsConfig(), notifications=NotificationsConfig())
    incident_id = None
    for n in range(MAX_EVIDENCE + 10):  # one episode per tracker ID switch: each joins the incident
        observation = zone_observation(ZonePhase.ENTERED, episode=f"zone-door-ep{n}", at_s=2.0 + n, track_id=n)
        incident_id = service.record(signal_from_zone(observation)).incident_id
    for n in range(MAX_ANNOTATIONS + 5):
        service.annotate(scene(incident_id, f"job-{n}"), Applicability.EXPIRED)
    revision = service.incident(incident_id).revision
    service.transition(incident_id, IncidentStatus.ACKNOWLEDGED, expected_revision=revision, actor="test", reason="seen")
    db.close()
    detail = read_incident(tmp_path / "sentinel.db", incident_id)
    evidence, annotations = detail["evidence"], detail["annotations"]
    assert (evidence["total"], len(evidence["shown"])) == (MAX_EVIDENCE + 10, MAX_EVIDENCE)
    assert [e["source"]["track_id"] for e in evidence["shown"]] == list(range(10, MAX_EVIDENCE + 10))
    assert [e["episode"] for e in evidence["shown"]][:3] == [1, 2, 3]
    assert (annotations["total"], len(annotations["shown"])) == (MAX_ANNOTATIONS + 5, MAX_ANNOTATIONS)
    assert annotations["shown"][-1]["details"]["reason"] == "stub result"
    assert [t["to_status"] for t in detail["transitions"]["shown"]] == ["open", "acknowledged"]
    text = render_incident_html(detail)
    assert f"{MAX_EVIDENCE} of {MAX_EVIDENCE + 10} shown (the latest)" in text
    assert "<td>acknowledged</td>" in text and "<td>test</td>" in text


def test_linked_incidents_point_to_each_other_and_ids_without_a_page_are_not_linked(tmp_path) -> None:
    clock = FakeClock(utc=T0)
    db = Database.open(tmp_path / "sentinel.db")
    ids = iter(["inc-first", "inc-second", "inc.dotted"])
    service = IncidentService(db, clock, incidents=IncidentsConfig(merge_window_s=10.0),
                              notifications=NotificationsConfig(), new_id=lambda now: next(ids))
    service.record(signal_from_zone(zone_observation(ZonePhase.ENTERED, episode="ep-a", at_s=2.0)))
    service.record(signal_from_zone(zone_observation(ZonePhase.ENTERED, episode="ep-b", at_s=30.0)))
    service.record(signal("garage"))  # another zone: its own incident, with an ID that has no page
    db.close()
    first = read_incident(tmp_path / "sentinel.db", "inc-first")
    second = read_incident(tmp_path / "sentinel.db", "inc-second")
    assert (first["linked"], second["linked"]) == ({"previous": None, "later": ["inc-second"]},
                                                   {"previous": "inc-first", "later": []})
    assert '<a class=mono href="/incidents/inc-second">inc-second</a>' in render_incident_html(first)
    assert '<a class=mono href="/incidents/inc-first">inc-first</a>' in render_incident_html(second)
    assert read_incident(tmp_path / "sentinel.db", "inc.dotted") == {"state": "not_found"}
    main_page = render_html(build_status(SNAPSHOT, read_store(tmp_path / "sentinel.db"), T0))
    assert "<span class=mono>inc.dotted</span>" in main_page and 'href="/incidents/inc.dotted"' not in main_page
    assert '<a class=inc href="/incidents/inc-first">' in main_page
