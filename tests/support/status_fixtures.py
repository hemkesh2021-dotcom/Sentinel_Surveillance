"""Stand-ins for the status-page tests: a runtime whose report the test sets, a live view on a FakeClock, and an
incident database with deliveries and a snapshot. Synthetic data only: the pictures are small generated JPEGs, never
camera frames."""

from __future__ import annotations

import base64
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sentinel.alerts.outbox import DeliveryResult, DeliveryStatus, OutboxWorker
from sentinel.config import FreshnessConfig, IncidentsConfig, NotificationsConfig
from sentinel.contracts import PixelFormat
from sentinel.incidents.service import IncidentService
from sentinel.incidents.signals import IncidentSignal, SignalPhase
from sentinel.incidents.snapshots import SNAPSHOT_DIR, SnapshotJob, SnapshotStore
from sentinel.live_preview import LivePreview
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.media.jpeg import EncodedJpeg
from sentinel.media.recent import RecentFrames
from sentinel.rules.scene_hazard import Severity
from sentinel.storage.database import Database

# An 8×8 grey JPEG (OpenCV, quality 50), so the browser has a real picture to decode and show.
PICTURE = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBk"
    "eFxlZ2P/wAALCAAIAAgBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEG"
    "E1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJ"
    "ipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/ACv/2Q=="
)


class Runtime:
    """The runtime stand-in: how old its last update is, whether the page's requests are answered, and its state."""

    def __init__(self, *, degraded: list[str] | None = None) -> None:
        self.answering = threading.Event()
        self.answering.set()
        self.lag_s = 0.0
        self.degraded = list(degraded or [])
        self.unanswered = 0  # page requests that arrived while the runtime was not answering
        self.served = 0

    def snapshot(self) -> dict[str, Any]:
        if not self.answering.is_set():
            self.unanswered += 1
            self.answering.wait(60.0)
        self.served += 1
        updated = datetime.now(timezone.utc) - timedelta(seconds=self.lag_s)
        return {"runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": updated.isoformat()},
                "degraded": list(self.degraded),
                "components": {"capture": {"state": "streaming", "stream_epoch": 1},
                               "detector": {"state": "available", "problem": None}},
                "live": {"video": "fresh", "detector": "available", "face_recognition": "disabled",
                         "scene_analysis": "disabled", "occupancy": "occupied", "people": 1, "confirmed_people": 1}}


class Camera:
    """A live view over a real FrameStamper and RecentFrames on a FakeClock; the encoder returns ``picture``."""

    def __init__(self, picture: bytes = PICTURE, size: tuple[int, int] = (8, 8)) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.frames = RecentFrames()
        self.picture, self.size = picture, size
        self.preview = LivePreview(self.frames, self.clock, FreshnessConfig(), lambda: self.stamper.current_stream,
                                   encode=lambda *_, **__: EncodedJpeg(self.picture, *self.size))
        self.stamper.connect()
        self.frame()

    def frame(self) -> None:
        self.clock.advance(1 / 15)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        self.frames.put(frame, object())


class _Notifier:
    channel = "telegram"

    def __init__(self, results: list[DeliveryResult]) -> None:
        self.results = results

    def send(self, message: Any) -> DeliveryResult:
        return self.results.pop(0) if self.results else DeliveryResult(DeliveryStatus.SENT, "sent", provider_message_id="1")


def incident_store(path: Path, *, count: int = 3, long_titles: bool = False, snapshot: bytes | None = PICTURE) -> list[str]:
    """A database at ``path`` (sentinel.db) with ``count`` incidents, one delivered and one failed alert, the first
    acknowledged, and (unless ``snapshot`` is None) a saved snapshot of the first. Returns the incident IDs, oldest first."""
    clock = FakeClock(utc=datetime.now(timezone.utc) - timedelta(minutes=5))
    db = Database.open(path)
    settings = NotificationsConfig(channels=["telegram"])
    service = IncidentService(db, clock, incidents=IncidentsConfig(), notifications=settings)
    worker = OutboxWorker(db, clock, {"telegram": _Notifier([
        DeliveryResult(DeliveryStatus.SENT, "sent", provider_message_id="7"),
        DeliveryResult(DeliveryStatus.PERMANENT, "HTTP 400 for https://api.telegram.org/bot<redacted>/sendMessage" + "x" * 80),
    ])}, settings)
    ids = []
    for n in range(count):
        clock.advance(11.0)
        zone = f"zone-{n}"
        title = ("Unbroken" + "x" * 112) if long_titles and n else f"Person in restricted zone {zone!r}"
        signal = IncidentSignal(
            observation_id=f"zone-{zone}.entered", episode_id=f"zone-{zone}", camera_id="cam-1",
            kind="zone.restricted_entry", phase=SignalPhase.ENTERED, zone_id=zone, rule_revision="rev1",
            severity=Severity.CRITICAL if n == 1 else Severity.WARNING, title=title, observed_utc=clock.utc_now(),
            boot_id="boot-1", observed_mono_ns=clock.monotonic_ns(),
            reason="track 3 stayed in the zone for 1.0 s (at least 1.0 s)" + (" " + "y" * 150 if long_titles else ""),
            payload="{}")
        ids.append(service.record(signal).incident_id)
        worker.run_once(limit=1)
    service.acknowledge(ids[0], actor="operator:maintainer", reason="seen on the dashboard")
    if snapshot is not None:
        stamper = FrameStamper("cam-1", clock)
        stamper.connect()
        frame = stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        store = SnapshotStore(db, path.parent / SNAPSHOT_DIR, clock, None,
                              encode=lambda *_, **__: EncodedJpeg(snapshot, 8, 8),
                              disk_usage=lambda _: SimpleNamespace(total=10**11, free=5 * 10**10))
        store.open()
        store.save(SnapshotJob(incident_id=ids[0], observation_id="zone-zone-0.entered", key=frame.key, frame=frame,
                               image=object(), missing=None))
        store.close()
    db.close()
    return ids
