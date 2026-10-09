"""D-1 `sentinel run`: the demo runtime wiring, with fake capture, detector, scene worker and notifier.

No camera, model, llama-server or Telegram is used. A real SQLite database,
IncidentService, OutboxWorker, EdgeCore, PersonTracker and ThreadedSceneAnalyzer
are wired as `sentinel run` wires them.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import signal
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sentinel.alerts.outbox import DeliveryResult, DeliveryStatus, OutboxWorker
from sentinel.cli import main
from sentinel.config import CaptureConfig, parse_config
from sentinel.contracts import PixelFormat
from sentinel.demo_runtime import (
    MAX_PENDING_SIGNALS,
    DemoRuntime,
    Devices,
    OutboxLoop,
    RunOptions,
    SceneOptions,
    SceneRuntime,
    StartupRefused,
    assemble,
    environment_notifiers,
)
from sentinel.incidents.service import IncidentService
from sentinel.incidents.signals import IncidentSignal, SignalPhase
from sentinel.jobs import OutcomeKind, WorkerOutcome
from sentinel.live_state import Capability, Occupancy, SceneStatus
from sentinel.media.capture import CapturedFrame, CaptureState, CaptureStatus, DecodedFrame, LatestFrame, SourceError
from sentinel.media.clock import FakeClock, SystemClock
from sentinel.media.frames import FrameStamper
from sentinel.media.jpeg import EncodedJpeg
from sentinel.rules.scene_hazard import Severity
from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer
from sentinel.status_page import read_incident, render_incident_html
from sentinel.storage.database import Database
from sentinel.tracking.tracker import PersonTracker, RawTrack, TrackerError

T0 = datetime(2026, 10, 4, 6, 0, tzinfo=timezone.utc)
FRAME_S = 1 / 15
IMAGE = SimpleNamespace(shape=(480, 640, 3))
PERSON = [RawTrack(1, 100.0, 100.0, 200.0, 400.0, 0.9)]
FAKE_TOKEN = "123456789:FAKE-test-token-not-real"
FAKE_URL = "rtsp://admin:hunter2@192.0.2.10:554/cam/realmonitor?channel=1&subtype=1"
REPORT = json.dumps({"persons_visible": 1, "fire_or_smoke": False, "threat": "low", "observations": ["one person"],
                     "uncertainty": "low", "summary": "One person stands in the room."})
ZONE = {"zone_id": "yard", "rule": "restricted", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "min_duration_s": 0.2, "gap_tolerance_s": 1.0, "severity": "warning"}


def make_config(**sections: Any):
    data: dict[str, Any] = {"config_version": 1, "camera": {"id": "cam-1"}, "zones": [ZONE],
                            "notifications": {"channels": ["telegram"], "request_timeout_s": 0.2}}
    data.update(sections)
    return parse_config(data)


class FakeCapture:
    """The Capture protocol over a real FrameStamper and LatestFrame, driven by the test."""

    def __init__(self, clock: FakeClock, slot: LatestFrame) -> None:
        self.clock = clock
        self.slot = slot
        self.stamper = FrameStamper("cam-1", clock)
        self.connects = self.ends = 0
        self.started = False
        self.stop_timeouts: list[float] = []
        self.problem: str | None = None

    @property
    def connected(self):
        return self.stamper.current_stream

    def status(self) -> CaptureStatus:
        stream = self.connected
        return CaptureStatus(CaptureState.STREAMING if stream else CaptureState.WAITING, stream, 0,
                             self.connects, 0, self.ends, None if stream else 1.0, self.problem)

    def connect(self) -> None:
        self.stamper.connect()
        self.connects += 1

    def disconnect(self) -> None:
        self.stamper.disconnect()
        self.slot.discard()
        self.ends += 1
        self.problem = "no_frame"

    def publish(self, dt: float = FRAME_S, image: Any = None):
        self.clock.advance(dt)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        self.slot.publish(CapturedFrame(frame, IMAGE if image is None else image))
        return frame

    def start(self) -> None:
        self.started = True

    def stop(self, timeout_s: float) -> bool:
        self.stop_timeouts.append(timeout_s)
        return True


class FakeBackend:
    def __init__(self) -> None:
        self.script: deque[object] = deque()
        self.calls = self.resets = self.loads = 0
        self.load_error: Exception | None = None

    def load(self) -> None:
        self.loads += 1
        if self.load_error is not None:
            raise self.load_error

    def track(self, image: object) -> list[RawTrack]:
        self.calls += 1
        item = self.script.popleft() if self.script else []
        if isinstance(item, Exception):
            raise item
        return list(item)  # type: ignore[call-overload]

    def reset(self) -> None:
        self.resets += 1


class FakeNotifier:
    channel = "telegram"

    def __init__(self, results: list[object] | None = None, *, block: threading.Event | None = None) -> None:
        self.results = deque(results or [])
        self.sent: list[Any] = []
        self.block = block

    def send(self, message):
        self.sent.append(message)
        if self.block is not None:
            self.block.wait(10)
        result = self.results.popleft() if self.results else DeliveryResult(
            DeliveryStatus.SENT, "sent", provider_message_id=str(len(self.sent)))
        if isinstance(result, Exception):
            raise result
        return result


class FlakyIncidents:
    """IncidentService that fails before (nothing stored) or after (stored, return lost) record()."""

    def __init__(self, inner: IncidentService, *, before: int = 0, after: int = 0) -> None:
        self.inner, self.before, self.after = inner, before, after

    def record(self, signal):
        if self.before:
            self.before -= 1
            raise sqlite3.OperationalError("database is locked")
        result = self.inner.record(signal)
        if self.after:
            self.after -= 1
            raise sqlite3.OperationalError("disk I/O error after commit")
        return result

    def annotate(self, evidence, applicability):
        return self.inner.annotate(evidence, applicability)


class Harness:
    def __init__(self, tmp_path: Path, *, config=None, notifier: FakeNotifier | None = None, tracker: bool = True,
                 detector_problem: str | None = None, scene: SceneRuntime | None = None,
                 scene_problem: str | None = None, notifier_problems: dict[str, str] | None = None,
                 flaky: dict[str, int] | None = None, live_preview: bool = False, picture_bytes: int = 0) -> None:
        self.clock = FakeClock(utc=T0)
        self.config = config or make_config()
        self.db = Database.open(tmp_path / "sentinel.db")
        self.slot = LatestFrame()
        self.capture = FakeCapture(self.clock, self.slot)
        self.backend = FakeBackend()
        self.notifier = notifier if notifier is not None else FakeNotifier()
        service = IncidentService(self.db, self.clock, incidents=self.config.incidents,
                                  notifications=self.config.notifications)
        self.incidents = FlakyIncidents(service, **flaky) if flaky else service
        notifiers = {} if notifier_problems else {"telegram": self.notifier}
        self.worker = OutboxWorker(self.db, self.clock, notifiers, self.config.notifications, rng=random.Random(7))
        self.outbox = OutboxLoop(self.worker, interval_s=0.05)
        self.runtime = DemoRuntime(
            self.config, self.clock, capture=self.capture, slot=self.slot, incidents=self.incidents,  # type: ignore[arg-type]
            outbox=self.outbox, tracker=PersonTracker(self.backend) if tracker else None,
            detector_problem=detector_problem, scene=scene, scene_problem=scene_problem,
            notifier_problems=notifier_problems, status_refresh_s=0.0, live_preview=live_preview,
            encode=lambda image, fmt, **kw: label_encode(image, fmt, picture_bytes=picture_bytes, **kw),
        )
        self.capture.connect()

    def frame(self, persons: list[RawTrack] | Exception = PERSON, dt: float = FRAME_S, image: Any = None):
        self.backend.script.append(persons)
        frame = self.capture.publish(dt, image)
        self.runtime.step(timeout_s=0)
        return frame

    def idle(self, seconds: float) -> None:
        self.clock.advance(seconds)
        self.runtime.step(timeout_s=0)

    def until_incident(self, limit: int = 30) -> None:
        for _ in range(limit):
            self.frame()
            if self.outbox_rows():
                return
        raise AssertionError("no incident was opened")

    def rows(self, sql: str) -> list[tuple]:
        with self.db.read() as c:
            return c.execute(sql).fetchall()

    def outbox_rows(self) -> list[tuple]:
        return self.rows("SELECT incident_id, status, attempts, ambiguous FROM outbox ORDER BY outbox_id")

    def live(self) -> dict[str, Any]:
        return self.runtime.snapshot()["live"]


def label_encode(image: Any, pixel_format: PixelFormat, *, max_width: int, max_height: int, quality: int,
                 picture_bytes: int = 0) -> EncodedJpeg:
    """A fake encoder: the JPEG's bytes name the image it was given (cv2 is never needed)."""
    label = getattr(image, "label", "image").encode()
    return EncodedJpeg(b"\xff\xd8" + label.ljust(max(0, picture_bytes - 4), b"\0") + b"\xff\xd9", 640, 480)


def wait_for(predicate, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.005)


# ---------------------------------------------------------------- end to end


def test_a_person_in_a_zone_opens_one_incident_and_one_alert_is_delivered(tmp_path) -> None:
    h = Harness(tmp_path)
    h.until_incident()
    for _ in range(20):  # the person stays: same episode, no new incident or alert
        h.frame()
    rows = h.outbox_rows()
    assert len(rows) == 1 and rows[0][1:] == ("pending", 0, 0)
    assert h.live()["occupancy"] == "occupied"
    assert h.worker.run_once() == [(1, "sent")]
    assert h.worker.run_once() == []  # nothing is sent twice
    assert h.outbox_rows()[0][1:] == ("sent", 1, 0)
    (message,) = h.notifier.sent
    assert rows[0][0] in message.text
    assert h.rows("SELECT count(*) FROM incidents") == [(1,)]
    records = h.runtime.snapshot()["records"]
    assert records["record_created"] == 1


def test_nobody_on_fresh_video_reads_empty_and_creates_nothing(tmp_path) -> None:
    h = Harness(tmp_path)
    for _ in range(10):
        h.frame([])
    assert h.live()["occupancy"] == Occupancy.EMPTY.value
    assert h.outbox_rows() == []
    assert h.runtime.snapshot()["degraded"] == []


# ---------------------------------------------------------------- reconnect and epochs


def test_a_reconnect_starts_a_new_epoch_resets_tracking_and_ends_the_zone_episode(tmp_path) -> None:
    h = Harness(tmp_path)
    h.until_incident()
    resets = h.backend.resets
    h.capture.disconnect()
    h.idle(0.5)
    assert h.runtime.snapshot()["components"]["capture"]["ready"] is False
    assert any(reason.startswith("capture waiting") for reason in h.runtime.snapshot()["degraded"])
    h.capture.connect()
    h.frame()
    assert h.backend.resets == resets + 1  # one tracker state per epoch
    snapshot = h.runtime.snapshot()
    assert snapshot["components"]["capture"]["stream_epoch"] == 2
    assert snapshot["components"]["capture"]["reconnects"] == 1
    phases = [r[0] for r in h.rows("SELECT phase FROM incident_evidence ORDER BY evidence_seq")]
    assert phases[0] == "entered" and "ended" in phases  # the outage ended the old episode


def test_a_frame_from_an_epoch_that_already_ended_gets_no_inference(tmp_path) -> None:
    h = Harness(tmp_path)
    h.frame([])
    h.backend.script.append(PERSON)
    h.capture.publish()  # stamped in epoch 1 ...
    h.capture.stamper.connect()  # ... but epoch 2 started before the loop took it
    calls = h.backend.calls
    h.runtime.step(timeout_s=0)
    assert h.backend.calls == calls
    assert h.runtime.snapshot()["frames"]["not_live"] == 1
    assert h.live()["people"] == 0


# ---------------------------------------------------------------- stale evidence


def test_a_stall_withdraws_people_and_occupancy_instead_of_freezing_them(tmp_path) -> None:
    h = Harness(tmp_path)
    for _ in range(5):
        h.frame()
    assert h.live()["people"] == 1
    h.idle(2.5)  # past stale_after_s without frames; the stream is still "connected"
    live = h.live()
    assert (live["video"], live["occupancy"], live["people"]) == ("stale", "unknown", 0)
    assert "video stale" in h.runtime.snapshot()["degraded"]


def test_a_late_scene_result_only_annotates_its_incident(tmp_path) -> None:
    images = RecentImages()
    analyzer = ThreadedSceneAnalyzer(lambda job, image: WorkerOutcome.completed(job.job_id, REPORT), images,
                                     revision="1+lfm2")
    h = Harness(tmp_path, scene=SceneRuntime(analyzer, images))
    h.runtime.start()
    try:
        h.frame()
        wait_for(lambda: analyzer.counters["delivered"] >= 1)
        h.frame()  # the periodic report arrives on time and becomes current
        assert h.live()["scene"] == SceneStatus.REPORTED.value
        h.until_incident()  # the new incident requests enrichment of its own frame
        wait_for(lambda: analyzer.counters["delivered"] >= 2)
        h.idle(9.0)  # the enrichment job's deadline passes before its outcome is collected
        h.runtime.step(timeout_s=0)
    finally:
        h.runtime.shutdown()
    annotations = h.rows("SELECT evidence_id, status, applicability FROM incident_annotations ORDER BY annotation_seq")
    assert [a[1] for a in annotations] == ["timeout", "observed"]
    late = annotations[1]
    assert late[0].endswith(".observed.late") and late[2] != "current"
    live = h.live()
    assert live["scene"] != SceneStatus.REPORTED.value  # a late report is history, never current
    assert len(h.outbox_rows()) == 1  # enrichment never notifies
    assert h.rows("SELECT status, severity FROM incidents") == [("open", "warning")]


def test_the_incident_page_shows_the_runtime_s_own_evidence_chain(tmp_path) -> None:
    """D-2's incident detail on what the real wiring stored: rule source, late scene result, alert attempt."""
    images = RecentImages()
    analyzer = ThreadedSceneAnalyzer(lambda job, image: WorkerOutcome.completed(job.job_id, REPORT), images,
                                     revision="1+lfm2")
    h = Harness(tmp_path, scene=SceneRuntime(analyzer, images))
    h.runtime.start()
    try:
        first = h.frame()
        wait_for(lambda: analyzer.counters["delivered"] >= 1)
        h.until_incident()
        wait_for(lambda: analyzer.counters["delivered"] >= 2)
        h.idle(9.0)  # the enrichment result is collected after its deadline: history only
        h.runtime.step(timeout_s=0)
        wait_for(lambda: h.outbox_rows()[0][1] == "sent")
        (incident_id,) = [row[0] for row in h.rows("SELECT incident_id FROM incidents")]
        detail = read_incident(tmp_path / "sentinel.db", incident_id)  # read only, while the runtime owns writes
    finally:
        h.runtime.shutdown()
    assert detail["state"] == "available"
    entered = detail["evidence"]["shown"][0]
    assert entered["phase"] == "entered" and entered["kind"] == "zone.restricted_entry"
    assert entered["source"]["track_id"] is not None and entered["source"]["in_zone_s"] >= 0.2
    assert entered["source"]["first_frame_seq"] <= entered["source"]["last_frame_seq"]
    assert entered["source"]["stream_epoch"] == first.stream_epoch  # no reconnect in this run
    # Collected after 9 s without frames: the video is stale, so neither could be current; both are history here.
    timeout, late = detail["annotations"]["shown"]
    assert [(a["status"], a["applicability"], a["recorded_after_source_s"]) for a in (timeout, late)] == [
        ("timeout", "no_live_stream", 9.0), ("observed", "no_live_stream", 9.0)]
    assert late["details"]["reason"] == "scene report arrived after the job deadline"
    assert late["details"]["report"]["summary"] == "One person stands in the room."
    (alert,) = detail["deliveries"]["shown"]
    assert (alert["state"], alert["attempts"], alert["provider_confirmed"]) == ("delivered", 1, True)
    assert [a["outcome"] for a in alert["attempt_log"]["shown"]] == ["sent"]
    assert [t["to_status"] for t in detail["transitions"]["shown"]] == ["open"]
    text = render_incident_html(detail)
    assert "One person stands in the room." in text and FAKE_URL not in text and FAKE_TOKEN not in text


# ---------------------------------------------------------------- dependency failures


def test_a_failing_detector_makes_occupancy_unknown_and_recovers(tmp_path) -> None:
    h = Harness(tmp_path)
    h.frame([])
    h.frame(RuntimeError("CUDA error: an illegal memory access"))
    snapshot = h.runtime.snapshot()
    assert snapshot["live"]["occupancy"] == "unknown"  # never "empty" without a working detector
    assert snapshot["components"]["detector"] == {
        **snapshot["components"]["detector"], "state": "unavailable", "problem": "backend_error (RuntimeError)"}
    assert "detector unavailable (backend_error (RuntimeError))" in snapshot["degraded"]
    assert "illegal memory" not in json.dumps(snapshot)
    h.frame([])
    h.frame([])
    assert h.runtime.snapshot()["components"]["detector"]["state"] == "available"
    assert h.live()["occupancy"] == "empty"


def test_a_detector_that_never_loaded_is_unavailable_with_its_reason(tmp_path) -> None:
    h = Harness(tmp_path, tracker=False, detector_problem="libcuda_not_l4t")
    for _ in range(5):
        h.frame()
    snapshot = h.runtime.snapshot()
    assert snapshot["components"]["detector"] == {"state": "unavailable", "problem": "libcuda_not_l4t"}
    assert snapshot["live"]["occupancy"] == "unknown" and snapshot["live"]["detector"] == "unavailable"
    assert h.outbox_rows() == []


def test_unavailable_scene_analysis_is_shown_as_such_not_as_disabled(tmp_path) -> None:
    h = Harness(tmp_path, scene_problem="server_failed:not_fully_offloaded")
    h.frame([])
    snapshot = h.runtime.snapshot()
    assert snapshot["live"]["scene_analysis"] == Capability.UNAVAILABLE.value
    assert snapshot["live"]["scene_reason"] == "scene analysis unavailable: server_failed:not_fully_offloaded"
    assert "scene analysis unavailable (server_failed:not_fully_offloaded)" in snapshot["degraded"]
    default = Harness(tmp_path / "default")
    default.frame([])
    assert default.live()["scene_analysis"] == Capability.DISABLED.value
    assert default.runtime.snapshot()["components"]["scene"] == {"state": "disabled", "problem": None}


def test_a_failed_record_keeps_the_observation_and_retries_it_once_a_second(tmp_path) -> None:
    h = Harness(tmp_path, flaky={"before": 2})
    for _ in range(6):
        h.frame()
    snapshot = h.runtime.snapshot()
    assert snapshot["components"]["incidents"]["pending_signals"] == 1
    assert snapshot["components"]["incidents"]["problem"] == "record_failed:OperationalError"
    assert h.outbox_rows() == []
    h.frame(dt=1.0)  # second failure, after the retry wait
    assert h.outbox_rows() == []
    h.frame(dt=1.0)
    assert len(h.outbox_rows()) == 1
    incidents = h.runtime.snapshot()["components"]["incidents"]
    assert incidents == {**incidents, "state": "ok", "pending_signals": 0, "signals_dropped": 0, "problem": None,
                         "annotation_problem": None, "pending_limit": 256}
    assert incidents["durability"].endswith("not a crash-safe spool")


def test_a_retry_after_a_commit_whose_return_was_lost_is_a_duplicate(tmp_path) -> None:
    h = Harness(tmp_path, flaky={"after": 1})
    for _ in range(6):
        h.frame()
    assert len(h.outbox_rows()) == 1  # committed, but not acknowledged
    assert h.runtime.snapshot()["components"]["incidents"]["pending_signals"] == 1
    h.frame(dt=1.0)
    assert len(h.outbox_rows()) == 1 and h.rows("SELECT count(*) FROM incidents") == [(1,)]
    records = h.runtime.snapshot()["records"]
    assert records["record_duplicate"] == 1 and records.get("record_created") is None


def test_observations_beyond_the_bound_are_counted_as_lost_not_hidden(tmp_path) -> None:
    h = Harness(tmp_path, flaky={"before": 10**6})
    for _ in range(MAX_PENDING_SIGNALS + 5):
        h.runtime._enqueue(object(), None)  # type: ignore[arg-type]
    snapshot_dropped = h.runtime.counters["signals_dropped"]
    assert snapshot_dropped == 5
    h.idle(0.0)
    assert "5 rule observation(s) lost: not recorded" in h.runtime.snapshot()["degraded"]


def test_a_channel_without_credentials_keeps_its_alert_queued(tmp_path) -> None:
    h = Harness(tmp_path, notifier_problems={"telegram": "credentials_missing"})
    h.until_incident()
    assert h.worker.run_once() == []
    assert h.outbox_rows()[0][1:] == ("pending", 0, 0)
    assert "telegram unavailable (credentials_missing); its alerts stay queued" in h.runtime.snapshot()["degraded"]
    notifiers, problems = environment_notifiers(h.config.notifications, {})
    assert (notifiers, problems) == ({}, {"telegram": "credentials_missing"})
    notifiers, problems = environment_notifiers(
        h.config.notifications, {"SENTINEL_TELEGRAM_BOT_TOKEN": FAKE_TOKEN, "SENTINEL_TELEGRAM_CHAT_ID": "42"})
    assert list(notifiers) == ["telegram"] and problems == {} and FAKE_TOKEN not in repr(notifiers)


# ---------------------------------------------------------------- duplicate delivery attempts


def test_an_ambiguous_timeout_is_retried_flagged_and_never_counted_as_delivered(tmp_path) -> None:
    timeout = DeliveryResult(DeliveryStatus.RETRY, "timeout after sending; Telegram may have delivered the message",
                             ambiguous=True)
    h = Harness(tmp_path, notifier=FakeNotifier([timeout, RuntimeError("socket text with a token")]))
    h.until_incident()
    assert h.worker.run_once() == [(1, "retry")]
    assert h.outbox_rows()[0][1:] == ("pending", 1, 1)
    h.idle(30.0)
    assert h.worker.run_once() == [(1, "retry")]  # a notifier bug is a retry, never a crash
    h.idle(60.0)
    assert h.worker.run_once() == [(1, "sent")]
    assert h.outbox_rows()[0][1:] == ("sent", 3, 1)
    incident = h.outbox_rows()[0][0]
    assert [incident in m.text for m in h.notifier.sent] == [True, True, True]  # a repeat is recognisable
    outcomes = [r[0] for r in h.rows("SELECT outcome FROM delivery_attempts ORDER BY attempt_seq")]
    assert outcomes == ["retry (ambiguous)", "retry (ambiguous)", "sent"]
    assert "token" not in json.dumps(h.rows("SELECT detail FROM delivery_attempts"))


def test_a_lease_left_by_a_dead_worker_is_retried_as_ambiguous(tmp_path) -> None:
    h = Harness(tmp_path)
    h.until_incident()
    dead = OutboxWorker(h.db, h.clock, {"telegram": h.notifier}, h.config.notifications, worker_id="dead")
    assert len(dead.lease(10)) == 1  # leased, then the worker "dies" without an outcome
    assert h.worker.run_once() == []  # still leased
    h.idle(61.0)
    assert h.worker.run_once() == [(1, "sent")]
    assert h.outbox_rows()[0][1:] == ("sent", 2, 1)
    assert [r[0] for r in h.rows("SELECT outcome FROM delivery_attempts ORDER BY attempt_seq")] == ["abandoned", "sent"]


# ---------------------------------------------------------------- shutdown


def test_shutdown_stops_every_owned_component_within_its_bound(tmp_path) -> None:
    images = RecentImages()
    analyzer = ThreadedSceneAnalyzer(lambda job, image: WorkerOutcome.failed(job.job_id, OutcomeKind.ERROR, "x"),
                                     images, revision="1")
    server = SimpleNamespace(stops=[], status=lambda: SimpleNamespace(
        state=SimpleNamespace(value="ready"), problem=None, layers="17/17", vision_on_gpu=True))
    server.stop = lambda grace: server.stops.append(grace) or True
    h = Harness(tmp_path, scene=SceneRuntime(analyzer, images, server))
    h.runtime.start()
    assert h.capture.started and h.outbox.status()["state"] == "running"
    h.until_incident()
    h.runtime.request_stop()
    assert h.runtime.stop_requested
    started = time.monotonic()
    result = h.runtime.shutdown()
    assert time.monotonic() - started < 3.0
    assert result["all_stopped"] and set(result["stopped"]) == {"capture", "scene_server", "scene_worker", "outbox"}
    assert h.capture.stop_timeouts == [16.0] and server.stops == [10.0]
    assert h.outbox.status()["state"] == "stopped"
    assert h.runtime.snapshot()["runtime"]["state"] == "stopped"


def test_shutdown_reports_a_send_that_outlasts_its_bound_and_keeps_the_database_open(tmp_path) -> None:
    release = threading.Event()
    h = Harness(tmp_path, notifier=FakeNotifier(block=release))
    h.until_incident()
    h.runtime.start()
    wait_for(lambda: h.notifier.sent)
    result = h.runtime.shutdown()
    try:
        assert result["stopped"]["outbox"] is False and result["all_stopped"] is False
        from sentinel.demo_runtime import Assembly

        assert Assembly(h.runtime, h.db).close(result) is False  # the sender still uses the connection
    finally:
        release.set()
        wait_for(lambda: h.outbox.status()["state"] == "stopped")
        h.db.close()


def test_shutdown_records_observations_that_were_still_waiting(tmp_path) -> None:
    h = Harness(tmp_path, flaky={"before": 1})
    for _ in range(6):
        h.frame()
    assert h.outbox_rows() == []
    result = h.runtime.shutdown()  # no wait for the retry timer
    assert result["signals_not_recorded"] == 0
    assert len(h.outbox_rows()) == 1


class ThreadedSource:
    """A VideoSource for the real CaptureWorker: a few frames per connection, then the stream ends."""

    def __init__(self, frames_per_connection: int = 6, *, on_read=None) -> None:
        self.per_connection = frames_per_connection
        self.left = 0
        self.opens = self.closes = self.reads = 0
        self.on_read = on_read

    def open(self) -> None:
        self.opens += 1
        self.left = self.per_connection

    def read(self):
        self.reads += 1
        if self.on_read is not None:
            self.on_read(self.reads)
        time.sleep(1 / 120)
        if self.left == 0:
            return None
        self.left -= 1
        return DecodedFrame(IMAGE, 640, 480, PixelFormat.BGR)

    def close(self) -> None:
        self.closes += 1


def fake_devices(source: Any, backend: FakeBackend | None = None, *, server: Any = None, memfree: float = 7e9,
                 calls: list[str] | None = None) -> Devices:
    calls = calls if calls is not None else []
    backend = backend or FakeBackend()

    def tracker_backend(engine: Path) -> FakeBackend:
        calls.append("tracker")
        return backend

    def scene_server(options: SceneOptions, port: int) -> Any:
        calls.append("scene_server")
        return server

    def capture_source(config: CaptureConfig) -> Any:
        if isinstance(source, Exception):
            raise source
        return source

    return Devices(
        capture_source=capture_source,
        tracker_backend=tracker_backend,
        scene_server=scene_server,
        scene_request=lambda port, timeout: (lambda job, image: WorkerOutcome.completed(job.job_id, REPORT)),
        meminfo=lambda: {"MemFree": int(memfree), "MemAvailable": 7_000_000_000},
        notifiers=lambda config: ({}, {}),
    )


def test_the_real_capture_worker_reconnects_and_the_runtime_stops_cleanly(tmp_path) -> None:
    config = make_config(capture={"reconnect_initial_s": 0.05, "reconnect_max_s": 0.1, "read_timeout_s": 0.5,
                                  "open_timeout_s": 0.5})
    source = ThreadedSource()
    backend = FakeBackend()
    assembly = assemble(config, RunOptions(tmp_path, tmp_path / "x.engine"), fake_devices(source, backend), SystemClock())
    runtime = assembly.runtime
    runtime.start()
    loop = threading.Thread(target=runtime.run)
    loop.start()
    try:
        wait_for(lambda: runtime.snapshot()["components"]["capture"]["connects"] >= 3, timeout_s=5.0)
    finally:
        runtime.request_stop()
        loop.join(2.0)
        started = time.monotonic()
        result = runtime.shutdown()
        elapsed = time.monotonic() - started
        assert assembly.close(result) is True
    assert not loop.is_alive() and elapsed < 3.0 and result["all_stopped"]
    assert source.opens == source.closes  # the one reader closed every connection it opened
    assert backend.resets >= 2  # tracker reset for each new epoch it processed
    assert runtime.snapshot()["frames"]["processed"] >= 1


# ---------------------------------------------------------------- startup


def test_a_missing_stream_url_refuses_before_any_database_or_model(tmp_path) -> None:
    calls: list[str] = []
    with pytest.raises(StartupRefused) as refused:
        assemble(make_config(), RunOptions(tmp_path / "data", tmp_path / "x.engine"),
                 fake_devices(SourceError("rtsp_url_missing"), calls=calls), FakeClock())
    assert refused.value.label == "rtsp_url_missing"
    assert calls == [] and not (tmp_path / "data").exists()


def test_scene_analysis_is_off_by_default_and_refused_unless_admitted(tmp_path) -> None:
    calls: list[str] = []
    scene = SceneOptions(Path("llama-server"), Path("m"), Path("p"))
    with pytest.raises(StartupRefused) as refused:
        assemble(make_config(), RunOptions(tmp_path, tmp_path / "x.engine", scene=scene),
                 fake_devices(ThreadedSource(), calls=calls), FakeClock())
    assert refused.value.label == "scene_not_admitted: no llama-lfm2-vl-scene adapter in the configuration"
    assert calls == []
    unprofiled = make_config(adapters=[scene_manifest(resource_profile_id=None)])
    with pytest.raises(StartupRefused, match="names no resource_profile_id"):
        assemble(unprofiled, RunOptions(tmp_path, tmp_path / "x.engine", scene=scene),
                 fake_devices(ThreadedSource(), calls=calls), FakeClock())
    assembly = assemble(make_config(), RunOptions(tmp_path, tmp_path / "x.engine"),
                        fake_devices(ThreadedSource(), calls=calls), FakeClock())
    assert calls == ["tracker"]
    assert assembly.runtime.snapshot()["components"]["scene"]["state"] == "disabled"
    assembly.database.close()


def scene_manifest(**overrides: Any) -> dict[str, Any]:
    manifest = {"adapter_id": "llama-lfm2-vl-scene", "contract_version": 1, "implementation_revision": "1",
                "enabled": True, "input_kinds": ["frame"], "output_kinds": ["scene.report"],
                "model_revision": "lfm2-vl-1.6b-q4_0", "resource_profile_id": "provisional-demo-20261003T085010Z",
                "timeout_ms": 8000}
    manifest.update(overrides)
    return manifest


def ready_server(state: str = "ready", problem: str | None = None) -> Any:
    status = SimpleNamespace(state=SimpleNamespace(value=state), problem=problem, layers="17/17", vision_on_gpu=True)
    server = SimpleNamespace(started=[], stopped=[], status=lambda: status)
    server.start = lambda timeout: server.started.append(timeout) or status
    server.stop = lambda grace: server.stopped.append(grace) or True
    return server


def test_an_admitted_scene_server_that_is_ready_enables_scene_analysis(tmp_path, accepted_scene) -> None:
    server = ready_server()
    fixture = accepted_scene()
    config = make_config(adapters=[fixture.manifest])
    scene = dataclasses.replace(fixture.options, ready_timeout_s=99.0)
    assembly = assemble(config, RunOptions(tmp_path, tmp_path / "x.engine", scene=scene),
                        fake_devices(ThreadedSource(), server=server), FakeClock(), profiles=fixture.profiles)
    assert server.started == [99.0]
    assert assembly.runtime.core.lane is not None
    assert assembly.runtime.snapshot()["components"]["scene"]["state"] == "available"
    assembly.database.close()


def test_model_failures_leave_that_component_unavailable_and_core_monitoring_runs(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()
    server = ready_server("failed", "not_fully_offloaded")
    backend = FakeBackend()
    backend.load_error = TrackerError("libcuda_not_l4t")
    config = make_config(adapters=[fixture.manifest])
    assembly = assemble(config, RunOptions(tmp_path, tmp_path / "x.engine", scene=fixture.options),
                        fake_devices(ThreadedSource(), backend, server=server), FakeClock(), profiles=fixture.profiles)
    assert assembly.startup["scene_problem"] == "server_failed:not_fully_offloaded"
    assert assembly.startup["detector_problem"] == "libcuda_not_l4t"
    components = assembly.runtime.snapshot()["components"]
    assert components["scene"]["state"] == components["detector"]["state"] == "unavailable"
    assembly.database.close()


def test_low_memfree_skips_the_gpu_loads(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()
    calls: list[str] = []
    backend = FakeBackend()
    config = make_config(adapters=[fixture.manifest])
    assembly = assemble(config, RunOptions(tmp_path, tmp_path / "x.engine", scene=fixture.options),
                        fake_devices(ThreadedSource(), backend, memfree=1e9, calls=calls), FakeClock(),
                        profiles=fixture.profiles)
    assert calls == [] and backend.loads == 0
    assert assembly.startup["detector_problem"] == assembly.startup["scene_problem"] == "memfree_below_minimum"
    assembly.database.close()


def test_an_unexpected_startup_error_stops_the_scene_server_and_releases_the_database(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()
    server = ready_server()
    backend = FakeBackend()
    backend.load_error = MemoryError()
    config = make_config(adapters=[fixture.manifest])
    with pytest.raises(MemoryError):
        assemble(config, RunOptions(tmp_path, tmp_path / "x.engine", scene=fixture.options),
                 fake_devices(ThreadedSource(), backend, server=server), FakeClock(), profiles=fixture.profiles)
    assert server.stopped == [10.0]
    Database.open(tmp_path / "sentinel.db").close()  # the writer lock was released


def test_a_second_runtime_on_the_same_database_is_refused(tmp_path) -> None:
    first = assemble(make_config(), RunOptions(tmp_path, tmp_path / "x.engine"),
                     fake_devices(ThreadedSource()), FakeClock())
    with pytest.raises(StartupRefused, match="only one Sentinel runtime"):
        assemble(make_config(), RunOptions(tmp_path, tmp_path / "x.engine"), fake_devices(ThreadedSource()), FakeClock())
    first.database.close()


def stored_incident(directory: Path) -> str:
    """A stopped runtime's database with one open incident."""
    db = Database.open(directory / "sentinel.db")
    service = IncidentService(db, FakeClock(utc=T0), incidents=make_config().incidents,
                              notifications=make_config().notifications)
    incident_id = service.record(IncidentSignal(
        observation_id="ep-1.entered", episode_id="ep-1", camera_id="cam-1", kind="zone.restricted_entry",
        phase=SignalPhase.ENTERED, zone_id="yard", rule_revision="rev1", severity=Severity.WARNING,
        title="Person in restricted zone 'yard'", observed_utc=T0, boot_id="boot-1", observed_mono_ns=1,
        reason="test", payload="{}")).incident_id
    db.close()
    return incident_id


def test_a_runtime_started_during_an_acknowledgment_is_refused_and_starts_after_it(tmp_path, capsys, monkeypatch):
    """D68's startup race, acknowledgment first: the runtime's startup finds the writer lock taken."""
    incident_id = stored_incident(tmp_path)
    labels: list[str] = []
    acknowledge = IncidentService.acknowledge

    def runtime_starts_meanwhile(self, *args, **kwargs):
        try:  # inside the command, between taking the lock and acknowledging
            started = assemble(make_config(), RunOptions(tmp_path, tmp_path / "x.engine"),
                               fake_devices(ThreadedSource()), FakeClock())
        except StartupRefused as exc:
            labels.append(exc.label)
        else:
            started.database.close()
            labels.append("started")
        return acknowledge(self, *args, **kwargs)

    monkeypatch.setattr(IncidentService, "acknowledge", runtime_starts_meanwhile)
    assert main(["incident", "ack", "--data-dir", str(tmp_path), incident_id]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "acknowledged"
    assert labels == ["database: another process is writing to sentinel.db; only one Sentinel runtime or operator "
                      "command may own it"]
    later = assemble(make_config(), RunOptions(tmp_path, tmp_path / "x.engine"), fake_devices(ThreadedSource()),
                     FakeClock())  # the command has exited: the runtime owns the database again
    try:
        with later.database.read() as c:
            assert c.execute("SELECT status FROM incidents").fetchall() == [("acknowledged",)]
    finally:
        later.database.close()


def test_an_acknowledgment_while_the_runtime_owns_the_database_is_refused_without_changes(tmp_path, capsys) -> None:
    """D68's startup race, runtime first."""
    incident_id = stored_incident(tmp_path)
    running = assemble(make_config(), RunOptions(tmp_path, tmp_path / "x.engine"), fake_devices(ThreadedSource()),
                       FakeClock())
    try:
        assert main(["incident", "ack", "--data-dir", str(tmp_path), incident_id]) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err.startswith("incident ack: database_busy")
        with running.database.read() as c:
            assert c.execute("SELECT status, revision FROM incidents").fetchall() == [("open", 1)]
            assert c.execute("SELECT count(*) FROM incident_transitions").fetchone() == (1,)
    finally:
        running.database.close()
    assert main(["incident", "ack", "--data-dir", str(tmp_path), incident_id]) == 0  # monitoring stopped: accepted


# ---------------------------------------------------------------- command line


def test_sentinel_run_starts_and_stops_on_sigterm_without_leaking_secrets(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("SENTINEL_RTSP_URL", FAKE_URL)
    monkeypatch.setenv("SENTINEL_TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 1\ncamera: {id: cam-1}\n")

    def stop_soon(reads: int) -> None:
        if reads == 20:
            os.kill(os.getpid(), signal.SIGTERM)

    devices = fake_devices(ThreadedSource(on_read=stop_soon))
    previous = signal.getsignal(signal.SIGTERM)
    code = main(["run", str(config), "--data-dir", str(tmp_path / "data"), "--engine", str(tmp_path / "x.engine"),
                 "--status-interval-s", "1", "--status-port", "0"], devices=devices)
    out = capsys.readouterr()
    assert code == 0, out.err
    lines = [json.loads(line) for line in out.out.splitlines()]
    assert lines[0]["run"] == "starting" and lines[-1]["run"] == "stopped"
    assert lines[-1]["shutdown"]["all_stopped"] is True and lines[-1]["database_closed"] is True
    for secret in ("hunter2", "192.0.2.10", "rtsp://", FAKE_TOKEN):
        assert secret not in out.out + out.err
    assert signal.getsignal(signal.SIGTERM) is previous  # handlers restored


def test_sentinel_run_reports_a_refusal_by_label(tmp_path, capsys) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 1\ncamera: {id: cam-1}\n")
    code = main(["run", str(config), "--data-dir", str(tmp_path), "--engine", "x", "--status-port", "0"],
                devices=fake_devices(SourceError("rtsp_url_missing")))
    assert code == 1 and capsys.readouterr().err.strip() == "run: rtsp_url_missing"


def test_a_scene_server_that_exits_while_running_is_shown_unavailable(tmp_path) -> None:
    images = RecentImages()
    analyzer = ThreadedSceneAnalyzer(lambda job, image: WorkerOutcome.completed(job.job_id, REPORT), images,
                                     revision="1")
    server = ready_server()
    h = Harness(tmp_path, scene=SceneRuntime(analyzer, images, server))
    h.frame([])
    assert h.runtime.snapshot()["components"]["scene"]["state"] == "available"
    server.status = lambda: SimpleNamespace(state=SimpleNamespace(value="exited"), problem="exited:134",
                                            layers="17/17", vision_on_gpu=True)
    h.frame([])
    snapshot = h.runtime.snapshot()
    assert snapshot["components"]["scene"]["server"]["problem"] == "exited:134"
    assert "scene analysis unavailable (server_exited)" in snapshot["degraded"]


def test_a_delivery_pass_that_raises_is_shown_and_the_loop_keeps_going(tmp_path) -> None:
    h = Harness(tmp_path)
    passes = {"n": 0}

    def broken(limit: int = 10):
        passes["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    h.worker.run_once = broken  # type: ignore[method-assign]
    h.outbox.start()
    try:
        wait_for(lambda: passes["n"] >= 2)
        h.idle(0.0)
        assert h.outbox.status()["problem"] == "outbox_error:OperationalError"
        assert "delivery worker problem (outbox_error:OperationalError)" in h.runtime.snapshot()["degraded"]
    finally:
        assert h.outbox.stop(1.0)


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_sentinel_run_serves_the_status_page_while_running(tmp_path, capsys, monkeypatch) -> None:
    import http.client

    monkeypatch.setenv("SENTINEL_RTSP_URL", FAKE_URL)
    monkeypatch.setenv("SENTINEL_TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.delenv("SENTINEL_TELEGRAM_CHAT_ID", raising=False)
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 1\ncamera: {id: cam-1}\nnotifications: {channels: [telegram]}\n")
    port = free_port()
    pages: list[tuple[int, str]] = []

    def fetch(reads: int) -> None:
        if reads in (15, 16):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/" if reads == 15 else "/status.json", headers={"Host": f"127.0.0.1:{port}"})
            response = conn.getresponse()
            pages.append((response.status, response.read().decode()))
            conn.close()
        if reads == 25:
            os.kill(os.getpid(), signal.SIGTERM)

    devices = fake_devices(ThreadedSource(on_read=fetch))
    devices.notifiers = environment_notifiers  # the real lookup: the chat ID is missing, so the channel is unavailable
    code = main(["run", str(config), "--data-dir", str(tmp_path / "data"), "--engine", str(tmp_path / "x.engine"),
                 "--status-port", str(port)], devices=devices)
    out = capsys.readouterr()
    assert code == 0, out.err
    assert [status for status, _ in pages] == [200, 200]
    html_page, document = pages[0][1], json.loads(pages[1][1])
    assert "telegram unavailable (credentials_missing)" in html_page
    assert document["components"]["capture"]["connects"] >= 1
    assert document["runtime"]["state"] == "running"
    lines = [json.loads(line) for line in out.out.splitlines()]
    assert lines[0]["startup"]["status_page"] == f"http://127.0.0.1:{port}/"
    assert lines[-1]["shutdown"]["stopped"]["status_page"] is True
    for secret in ("hunter2", "192.0.2.10", "rtsp://", FAKE_TOKEN):
        assert secret not in html_page + pages[1][1] + out.out + out.err
    with __import__("socket").socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0  # the page stopped with the runtime


def test_a_busy_status_port_refuses_startup_before_any_model_loads(tmp_path, capsys) -> None:
    import socket

    config = tmp_path / "config.yaml"
    config.write_text("config_version: 1\ncamera: {id: cam-1}\n")
    calls: list[str] = []
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        code = main(["run", str(config), "--data-dir", str(tmp_path), "--engine", "x", "--status-port", str(port)],
                    devices=fake_devices(ThreadedSource(), calls=calls))
    assert code == 1 and capsys.readouterr().err.strip() == "run: status_port_unavailable:OSError"
    assert calls == []


# ---------------------------------------------------------------- the live view (D69)

VIDEO_TO_PREVIEW = {"fresh": "live", "stale": "stale", "offline": "offline", "starting": "starting"}


def test_the_live_view_shows_the_frames_the_loop_takes_and_follows_video_freshness(tmp_path) -> None:
    h = Harness(tmp_path, live_preview=True)
    preview = h.runtime.preview
    assert preview is not None
    seen: list[tuple[str, str]] = []

    def check() -> dict[str, Any]:
        view = preview.view(with_picture=True)
        seen.append((h.live()["video"], view["state"]))
        return view

    first = h.frame(image=SimpleNamespace(shape=(480, 640, 3), label="first"))
    view = check()
    assert view["picture"]["frame_seq"] == first.frame_seq and preview.picture(f"1-{first.frame_seq}").jpeg[2:7] == b"first"
    h.idle(2.5)  # a stall: no frame, the stream still connected
    assert check()["picture"] is None
    h.capture.disconnect()
    h.idle(0.0)
    assert check()["reason"].startswith("no new frame for 2.5 s")
    h.idle(10.0)
    assert check()["state"] == "offline"
    h.capture.connect()
    h.frame(image=SimpleNamespace(shape=(480, 640, 3), label="after"))
    view = check()
    assert view["picture"]["stream_epoch"] == 2 and view["picture"]["url"].startswith("/preview/2-")
    assert [VIDEO_TO_PREVIEW[video] for video, _ in seen] == [state for _, state in seen]  # one freshness rule
    assert [state for _, state in seen] == ["live", "stale", "stale", "offline", "live"]
    assert h.capture.connects == 2  # the preview opened no connection of its own
    assert preview.summary()["counters"]["encodes"] == 2
    h.idle(0.0)  # the runtime's status, taken at its next step, carries the same numbers
    assert h.runtime.snapshot()["components"]["preview"] == preview.summary()


def test_the_live_view_is_off_by_default_and_keeps_no_frames(tmp_path) -> None:
    h = Harness(tmp_path)
    h.frame()
    assert h.runtime.preview is None and h.runtime._recent is None  # type: ignore[attr-defined]
    assert "preview" not in h.runtime.snapshot()["components"]  # a run without it reports exactly as before


def test_a_stalled_browser_never_slows_the_runtime_loop(tmp_path, monkeypatch) -> None:
    import socket

    from sentinel import status_page
    from sentinel.live_preview import PREVIEW_MAX_JPEG_BYTES
    from sentinel.status_page import StatusServer

    monkeypatch.setattr(status_page, "SEND_BUFFER_BYTES", 4096)
    h = Harness(tmp_path, live_preview=True, picture_bytes=PREVIEW_MAX_JPEG_BYTES)
    h.frame()
    page = StatusServer(0, h.runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: h.runtime.preview,
                        handler_timeout_s=0.5)
    page.start()
    stalled = []
    try:
        url = h.runtime.preview.view(with_picture=True)["picture"]["url"]  # type: ignore[union-attr]
        for _ in range(2):  # both picture slots, each request sent and its response never read
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            sock.connect(page.address)
            sock.sendall(f"GET {url} HTTP/1.1\r\nHost: 127.0.0.1:{page.address[1]}\r\n\r\n".encode())
            stalled.append(sock)
        time.sleep(0.1)
        durations = []
        for _ in range(30):
            started = time.perf_counter()
            h.frame()
            durations.append(time.perf_counter() - started)
        wait_for(lambda: page.connection_counts()["write_timeouts"] == 2)
    finally:
        for sock in stalled:
            sock.close()
        assert page.stop(2.0)
    assert max(durations) < 0.05 and h.runtime.snapshot()["frames"]["processed"] == 31


def test_sentinel_run_serves_the_live_view_only_with_its_flag(tmp_path, capsys, monkeypatch) -> None:
    import http.client
    import re

    monkeypatch.setenv("SENTINEL_RTSP_URL", FAKE_URL)
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 1\ncamera: {id: cam-1}\n")
    assert main(["run", str(config), "--data-dir", str(tmp_path / "data"), "--engine", "x", "--status-port", "0",
                 "--live-preview"], devices=fake_devices(ThreadedSource())) == 1
    assert capsys.readouterr().err.strip() == "run: --live-preview needs the status page (--status-port other than 0)"

    port = free_port()
    fetched: list[tuple[int, str, bytes]] = []

    def fetch(path: str) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
        response = conn.getresponse()
        fetched.append((response.status, response.getheader("Content-Type"), response.read()))
        conn.close()

    def on_read(reads: int) -> None:
        if reads == 15:  # in the capture thread: no new frame arrives meanwhile
            fetch("/")
            match = re.search(r'<img src="(/preview/1-\d+\.jpg)"', fetched[0][2].decode())
            fetch(match.group(1) if match else "/preview/none.jpg")
        if reads == 25:
            os.kill(os.getpid(), signal.SIGTERM)

    devices = fake_devices(ThreadedSource(1000, on_read=on_read))
    devices.encode = label_encode
    code = main(["run", str(config), "--data-dir", str(tmp_path / "data"), "--engine", str(tmp_path / "x.engine"),
                 "--status-port", str(port), "--live-preview"], devices=devices)
    out = capsys.readouterr()
    assert code == 0, out.err
    assert [(status, kind) for status, kind, _ in fetched] == [(200, "text/html; charset=utf-8"), (200, "image/jpeg")]
    assert fetched[1][2] == b"\xff\xd8image\xff\xd9" and "<strong>LIVE</strong>" in fetched[0][2].decode()
    lines = [json.loads(line) for line in out.out.splitlines()]
    assert lines[0]["startup"]["live_preview"]["max_width"] == 640
    assert lines[-1]["status"]["state"] == "stopped" and lines[-1]["shutdown"]["all_stopped"] is True
    for secret in ("hunter2", "192.0.2.10", "rtsp://"):
        assert secret not in out.out + out.err + fetched[0][2].decode()
