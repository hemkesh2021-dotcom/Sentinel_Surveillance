"""``sentinel run``: the demo runtime loop (D-1; demo only, not a backlog package).

It wires the existing components through their own contracts:

    CaptureWorker --LatestFrame--> loop thread: PersonTracker -> EdgeCore (zone and hazard rules)
        -> IncidentService.record() -> outbox rows -> OutboxLoop thread -> notifiers (Telegram)
    EdgeCore's scene lane <-> ThreadedSceneAnalyzer -> llama-server on 127.0.0.1 (optional)

What the loop keeps:

- **One capture owner, bounded hand-off, explicit frame ownership.** Only the
  CaptureWorker reads the camera. The loop takes each frame from the one-slot
  LatestFrame, so a slow loop skips frames instead of queueing them. The loop
  owns the taken image: the tracker uses it synchronously, RecentImages keeps
  a reference to the last IMAGES_KEPT frames only while scene analysis runs,
  and the loop drops its own reference before the next frame.
- **Epochs and staleness.** The loop passes ``capture.connected`` to EdgeCore,
  which decides freshness and what is current. A frame from an epoch that has
  already ended skips inference. PersonTracker resets per epoch and processes
  each frame once, in order. Scene outcomes are delivered on the loop thread,
  and late ones only annotate their own incident (D15, D35).
- **Durable incidents (D31).** Every zone observation and hazard candidate
  becomes an IncidentSignal and is acknowledged only after ``record()``
  returns. A failed ``record()`` keeps the signal, in order, and retries it at
  most once a second. ``record()`` deduplicates by observation ID, so a retry
  after a commit whose return was lost changes nothing. At most
  MAX_PENDING_SIGNALS wait. Beyond that, new signals are dropped, counted and
  shown as lost durability: this is not a spool, and nothing pretends it is.
- **Enrichment (D35).** Evidence that names an incident is passed to
  ``annotate()``. A newly created zone incident asks the scene lane to enrich
  its own source frame while scene analysis runs.
- **Delivery (D32)** stays the OutboxWorker's: leased, at least once, with
  retries and ambiguity recorded in SQLite. OutboxLoop only runs it in its own
  thread.
- **Visible state.** ``snapshot()`` returns numbers and fixed labels, including
  every degradation reason. It never includes URLs, credentials, images or
  exception text.
- **Bounded shutdown.** Each component that the runtime started gets a bounded
  stop, and the result says which ones did not stop in time.

Not V2-29: there is no memory-pressure admission or degradation controller.
GPU components are loaded once at startup, behind the D27 guard and a
provisional MemFree precheck (U18 stays open). A component that fails is shown
unavailable; nothing is restarted or re-admitted at runtime (V2-19, V2-29).
Face recognition is not wired: it stays disabled.

Not thread-safe apart from ``snapshot()`` and ``request_stop()``: one loop thread calls ``step()``.
"""

from __future__ import annotations

import hashlib
import threading
from collections import Counter, OrderedDict, deque
from datetime import datetime, timezone
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .adapters import (
    RESOURCE_PROFILES,
    STARTUP_HASH_LIMIT_BYTES,
    AdapterState,
    FileFacts,
    ResourceProfile,
    resolve,
)
from .alerts.outbox import Notifier, OutboxWorker
from .alerts.telegram import NotifierUnavailable, TelegramNotifier
from .config import CaptureConfig, NotificationsConfig, SentinelConfig
from .contracts import FrameKey, FrameRef, StreamIdentity, TrackStatus
from .incidents.service import IncidentService, RecordOutcome
from .inference.legacy_ultralytics import LEGACY_ENGINE_SHA256
from .incidents.signals import IncidentSignal, signal_from_hazard, signal_from_zone
from .jobs import AnalysisJob, WorkerOutcome
from .live_state import Capability, LiveState
from .media.capture import CapturedFrame, CaptureState, CaptureStatus, LatestFrame, SourceError, VideoSource
from .media.clock import NS_PER_SECOND, Clock
from .rules.zones import ZonePhase
from .runtime import CoreOutput, EdgeCore
from .scene.analyzer import IMAGES_KEPT, RecentImages, ThreadedSceneAnalyzer
from .scene.llama_server import PROFILED_FLAGS, PROMPT_CACHE_FLAGS, SCENE_REQUEST_SHA256
from .storage.database import Database, DatabaseError
from .tracking.tracker import FrameOutcome, PersonTracker, TrackerError

SCENE_ADAPTER_ID = "llama-lfm2-vl-scene"
DATABASE_NAME = "sentinel.db"
MAX_PENDING_SIGNALS = 256
PENDING_DURABILITY = (
    f"rule observations waiting to be recorded are held in memory only (at most {MAX_PENDING_SIGNALS}); "
    "they are lost if the process stops abruptly: not a crash-safe spool"
)
RECORD_RETRY_NS = NS_PER_SECOND
RATE_WINDOW_NS = 10 * NS_PER_SECOND


class Capture(Protocol):
    """The camera reader; CaptureWorker satisfies it."""

    @property
    def connected(self) -> StreamIdentity | None: ...

    def status(self) -> CaptureStatus: ...

    def start(self) -> None: ...

    def stop(self, timeout_s: float) -> bool: ...


class ServerHandle(Protocol):
    """A started scene server; LlamaServerProcess satisfies it."""

    def status(self) -> Any: ...

    def stop(self, grace_s: float) -> bool: ...


class RuntimeState(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class SceneRuntime:
    analyzer: ThreadedSceneAnalyzer
    images: RecentImages
    server: ServerHandle | None = None


class OutboxLoop:
    """Runs ``OutboxWorker.run_once()`` in its own thread until stopped."""

    def __init__(self, worker: OutboxWorker, *, interval_s: float = 1.0, batch: int = 10) -> None:
        self.worker = worker
        self._interval_s = interval_s
        self._batch = batch
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._passes = 0
        self._problem: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the outbox loop can be started once")
        self._thread = threading.Thread(target=self._run, name="sentinel-outbox", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float) -> bool:
        """False if a send is still running after ``timeout_s``; its lease then expires and it is retried."""
        self._stop.set()
        if self._thread is None:
            return True
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        thread = self._thread
        if thread is None:
            state = "not_started"
        elif thread.is_alive():
            state = "running"
        else:
            state = "stopped"
        with self._lock:
            return {"state": state, "passes": self._passes, "problem": self._problem,
                    "counters": dict(sorted(self.worker.counters.items()))}

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.worker.run_once(self._batch)
                problem = None
            except Exception as exc:  # noqa: BLE001 - a database error must not end delivery for good
                problem = f"outbox_error:{type(exc).__name__}"
            with self._lock:
                self._passes += 1
                self._problem = problem
            self._stop.wait(self._interval_s)


class DemoRuntime:
    def __init__(
        self,
        config: SentinelConfig,
        clock: Clock,
        *,
        capture: Capture,
        slot: LatestFrame,
        incidents: IncidentService,
        outbox: OutboxLoop | None = None,
        tracker: PersonTracker | None = None,
        detector_problem: str | None = None,
        scene: SceneRuntime | None = None,
        scene_problem: str | None = None,
        notifier_problems: Mapping[str, str] | None = None,
        tick_s: float = 0.25,
        status_refresh_s: float = 0.5,
    ) -> None:
        if tracker is None and detector_problem is None:
            detector_problem = "detector_not_configured"
        self._config = config
        self._clock = clock
        self._capture = capture
        self._slot = slot
        self._incidents = incidents
        self._outbox = outbox
        self._tracker = tracker
        self._detector_problem = detector_problem  # set at startup; never cleared
        self._scene = scene
        self._scene_problem = None if scene is not None else scene_problem
        self._notifier_problems = dict(notifier_problems or {})
        self._tick_s = tick_s
        self._status_refresh_ns = round(status_refresh_s * NS_PER_SECOND)
        self._core = EdgeCore(
            config,
            clock,
            scene.analyzer if scene is not None else None,
            detector=Capability.AVAILABLE if tracker is not None else Capability.UNAVAILABLE,
            scene_problem=self._scene_problem,
        )
        self._frames: OrderedDict[FrameKey, FrameRef] = OrderedDict()
        self._pending: deque[tuple[IncidentSignal, FrameKey | None]] = deque()
        self._retry_at_ns = 0
        self._record_problem: str | None = None
        self._annotate_problem: str | None = None
        self._last_frame_problem: str | None = None  # the latest tracker failure, until a frame succeeds
        self._live: LiveState | None = None
        self.counters: Counter[str] = Counter()
        self._samples: deque[tuple[int, int, int, int]] = deque()
        self._next_status_ns = 0
        self._state = RuntimeState.STARTING
        self._stop = threading.Event()
        self._status_lock = threading.Lock()
        self._snapshot: dict[str, Any] = {}
        self._shutdown: dict[str, Any] | None = None
        self._refresh_status(force=True)

    @property
    def core(self) -> EdgeCore:
        return self._core

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start the threads this runtime owns: capture, scene worker, outbox."""
        if self._scene is not None:
            self._scene.analyzer.start()
        if self._outbox is not None:
            self._outbox.start()
        self._capture.start()
        self._state = RuntimeState.RUNNING
        self._refresh_status(force=True)

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def request_stop(self) -> None:
        """Safe from a signal handler or another thread."""
        self._stop.set()
        self._slot.close()

    def run(self) -> None:
        """Run the loop on this thread until request_stop()."""
        while not self._stop.is_set():
            self.step()

    def shutdown(self) -> dict[str, Any]:
        """Stop what this runtime started, each within its own bound; returns what stopped."""
        self._state = RuntimeState.STOPPING
        self._refresh_status(force=True)
        capture = self._config.capture
        stopped: dict[str, bool] = {"capture": self._capture.stop(capture.open_timeout_s + capture.read_timeout_s + 1.0)}
        if self._scene is not None:
            self._scene.analyzer.stop(0.0)  # drop a waiting job; a running request ends when the server goes
            if self._scene.server is not None:
                stopped["scene_server"] = self._scene.server.stop(10.0)
            stopped["scene_worker"] = self._scene.analyzer.stop(self._config.scene_server.request_timeout_s + 1.0)
        self._flush(force=True)  # last chance for observations that are still waiting
        if self._outbox is not None:
            stopped["outbox"] = self._outbox.stop(self._config.notifications.request_timeout_s + 2.0)
        self._shutdown = {
            "stopped": stopped,
            "all_stopped": all(stopped.values()),
            "signals_not_recorded": len(self._pending) + self.counters["signals_dropped"],
        }
        self._state = RuntimeState.STOPPED
        self._refresh_status(force=True)
        return dict(self._shutdown)

    # ------------------------------------------------------------ one step

    def step(self, timeout_s: float | None = None) -> None:
        captured = self._slot.take(self._tick_s if timeout_s is None else timeout_s)
        connected = self._capture.connected
        if captured is None:
            output = self._core.tick(connected)
        else:
            output = self._on_frame(captured, connected)
            del captured  # the loop's reference to the image ends here
        self._handle(output)
        if self._scene is not None:
            for job_id, outcome in self._scene.analyzer.drain():
                self._handle(self._core.on_scene_outcome(job_id, outcome, self._capture.connected))
        if self._pending:
            self._handle(None)
        self._refresh_status()

    def _on_frame(self, captured: CapturedFrame, connected: StreamIdentity | None) -> CoreOutput:
        frame = captured.frame
        self.counters["frames_taken"] += 1
        self._frames[frame.key] = frame
        while len(self._frames) > IMAGES_KEPT:
            self._frames.popitem(last=False)
        if self._scene is not None:
            self._scene.images.put(frame.key, captured.image)
        if self._tracker is None:
            return self._core.on_frame(frame, (), connected)
        if connected is None or frame.stream != connected:
            self.counters["frames_not_live"] += 1  # its epoch already ended: no inference
            return self._core.on_frame(frame, (), connected)
        result = self._tracker.process(frame, captured.image)
        if result.outcome is FrameOutcome.PROCESSED:
            self.counters["frames_processed"] += 1
            self._last_frame_problem = None
            self._core.set_detector(Capability.AVAILABLE)
            return self._core.on_frame(frame, result.persons, connected)
        if result.outcome is FrameOutcome.FAILED:
            self.counters["frames_failed"] += 1
            detail = f" ({result.error_type})" if result.error_type else ""
            self._last_frame_problem = f"{result.problem}{detail}"
            self._core.set_detector(Capability.UNAVAILABLE)  # occupancy is unknown, never empty (D18)
            return self._core.on_frame(frame, (), connected)
        self.counters["frames_skipped"] += 1
        return self._core.tick(connected)

    def _handle(self, output: CoreOutput | None) -> None:
        outputs = deque([output] if output is not None else [None])
        while outputs:
            current = outputs.popleft()
            if current is not None:
                self._live = current.state
                self._annotate(current)
                for zone in current.zones:
                    enrich = zone.last_source if zone.phase is ZonePhase.ENTERED else None
                    self._enqueue(signal_from_zone(zone), enrich)
                for candidate in current.candidates:
                    self._enqueue(signal_from_hazard(candidate), None)
            for frame, incident_id in self._flush():
                outputs.append(self._core.request_enrichment(frame, incident_id, self._capture.connected))

    def _annotate(self, output: CoreOutput) -> None:
        for routed in output.evidence:
            if routed.routing.annotates is None:
                continue
            try:
                result = self._incidents.annotate(routed.evidence, routed.routing.applicability)
            except Exception as exc:  # noqa: BLE001 - enrichment is history; core monitoring continues
                self.counters["annotations_failed"] += 1
                self._annotate_problem = f"annotate_failed:{type(exc).__name__}"
                continue
            self.counters[f"annotation_{result.outcome.value}"] += 1

    def _enqueue(self, signal: IncidentSignal, enrich: FrameKey | None) -> None:
        if len(self._pending) >= MAX_PENDING_SIGNALS:
            self.counters["signals_dropped"] += 1
            return
        self._pending.append((signal, enrich))

    def _flush(self, *, force: bool = False) -> list[tuple[FrameRef, str]]:
        """Record waiting signals in order; returns enrichment requests for created incidents.

        After a failure, the next attempt waits RECORD_RETRY_NS unless ``force``.
        """
        enrich: list[tuple[FrameRef, str]] = []
        if not force and self._clock.monotonic_ns() < self._retry_at_ns:
            return enrich
        while self._pending:
            signal, source = self._pending[0]
            try:
                result = self._incidents.record(signal)
            except Exception as exc:  # noqa: BLE001 - kept and retried; record() deduplicates
                self.counters["record_failures"] += 1
                self._record_problem = f"record_failed:{type(exc).__name__}"
                self._retry_at_ns = self._clock.monotonic_ns() + RECORD_RETRY_NS
                break
            self._pending.popleft()  # acknowledged only now (D31)
            self._record_problem = None
            self.counters[f"record_{result.outcome.value}"] += 1
            if result.outcome is RecordOutcome.CREATED and source is not None and self._scene is not None:
                frame = self._frames.get(source)
                if frame is None:
                    self.counters["enrichment_frame_gone"] += 1
                else:
                    enrich.append((frame, result.incident_id))  # type: ignore[arg-type]
        return enrich

    # ------------------------------------------------------------ status

    def snapshot(self) -> dict[str, Any]:
        """The latest status: numbers and fixed labels only. Thread-safe."""
        with self._status_lock:
            return self._snapshot

    def _refresh_status(self, *, force: bool = False) -> None:
        now = self._clock.mono()
        if not force and now.ns < self._next_status_ns:
            return
        self._next_status_ns = now.ns + self._status_refresh_ns
        slot = self._slot.counts()
        self._samples.append((now.ns, slot["published"], self.counters["frames_processed"], self.counters["frames_failed"]))
        while len(self._samples) > 2 and now.ns - self._samples[1][0] >= RATE_WINDOW_NS:
            self._samples.popleft()
        capture = self._capture.status()
        components = {
            "capture": self._capture_status(capture),
            "detector": self._detector_status(),
            "scene": self._scene_status(),
            "incidents": {
                "state": "degraded" if self._pending or self.counters["signals_dropped"] else "ok",
                "pending_signals": len(self._pending),
                "pending_limit": MAX_PENDING_SIGNALS,
                "durability": PENDING_DURABILITY,
                "signals_dropped": self.counters["signals_dropped"],
                "problem": self._record_problem,
                "annotation_problem": self._annotate_problem,
            },
            "notifications": self._notification_status(),
        }
        snapshot = {
            "runtime": {
                "state": self._state.value,
                "camera_id": self._config.camera.id,
                "updated_utc": self._clock.utc_now().isoformat(),
                "shutdown": self._shutdown,
            },
            "components": components,
            "frames": {
                "captured": slot["published"],
                "replaced": slot["replaced"],
                "discarded": slot["discarded"],
                **{name: self.counters[f"frames_{name}"] for name in ("taken", "processed", "failed", "skipped", "not_live")},
            },
            "rates": self._rates(),
            "live": self._live_summary(),
            "records": {key: value for key, value in sorted(self.counters.items())
                        if key.startswith(("record_", "annotation_", "enrichment_"))},
        }
        snapshot["degraded"] = degradation(snapshot)
        with self._status_lock:
            self._snapshot = snapshot

    def _capture_status(self, status: CaptureStatus) -> dict[str, Any]:
        return {
            "state": status.state.value,
            "ready": status.state is CaptureState.STREAMING,
            "stream_epoch": status.stream.stream_epoch if status.stream is not None else None,
            "connects": status.connects,
            "reconnects": max(0, status.connects - 1),
            "open_failures": status.open_failures,
            "stream_ends": status.stream_ends,
            "retry_delay_s": status.retry_delay_s,
            "problem": status.problem,
        }

    def _detector_status(self) -> dict[str, Any]:
        if self._tracker is None:
            return {"state": Capability.UNAVAILABLE.value, "problem": self._detector_problem}
        state = Capability.UNAVAILABLE if self._last_frame_problem else Capability.AVAILABLE
        return {"state": state.value, "problem": self._last_frame_problem,
                "counters": dict(sorted(self._tracker.counters.items()))}

    def _scene_status(self) -> dict[str, Any]:
        if self._scene is None:
            state = Capability.DISABLED if self._scene_problem is None else Capability.UNAVAILABLE
            return {"state": state.value, "problem": self._scene_problem}
        server = None
        problem = None
        if self._scene.server is not None:
            status = self._scene.server.status()
            server = {"state": status.state.value, "problem": status.problem, "layers": status.layers,
                      "vision_on_gpu": status.vision_on_gpu}
            if status.state.value != "ready":
                problem = f"server_{status.state.value}"
        lane = self._core.lane
        return {
            "state": (Capability.UNAVAILABLE if problem else Capability.AVAILABLE).value,
            "problem": problem,
            "server": server,
            "worker": dict(sorted(self._scene.analyzer.counters.items())),
            "lane": dict(sorted(lane.counters.items())) if lane is not None else {},
        }

    def _notification_status(self) -> dict[str, Any]:
        channels = {}
        for channel in self._config.notifications.channels:
            problem = self._notifier_problems.get(channel)
            channels[channel] = {"state": "unavailable" if problem else "available", "problem": problem}
        return {
            "channels": channels,
            "min_severity": self._config.notifications.min_severity,
            "worker": self._outbox.status() if self._outbox is not None else {"state": "not_configured"},
        }

    def _rates(self) -> dict[str, Any]:
        first, last = self._samples[0], self._samples[-1]
        span_s = (last[0] - first[0]) / NS_PER_SECOND
        if span_s <= 0:
            return {"window_s": 0.0, "captured_fps": None, "processed_fps": None, "failed_per_s": None}
        return {
            "window_s": round(span_s, 1),
            "captured_fps": round((last[1] - first[1]) / span_s, 2),
            "processed_fps": round((last[2] - first[2]) / span_s, 2),
            "failed_per_s": round((last[3] - first[3]) / span_s, 2),
        }

    def _live_summary(self) -> dict[str, Any] | None:
        live = self._live
        if live is None:
            return None
        report = live.scene_report
        return {
            "sequence": live.sequence,
            "video": live.video.value,
            "last_frame_age_ms": live.last_frame_age_ms,
            "detector": live.detector.value,
            "face_recognition": live.face_recognition.value,
            "scene_analysis": live.scene_analysis.value,
            "occupancy": live.occupancy.value,
            "occupancy_reason": live.occupancy_reason,
            "people": len(live.people),
            "confirmed_people": sum(1 for p in live.people if p.status is TrackStatus.CONFIRMED),
            "scene": live.scene.value,
            "scene_reason": live.scene_reason,
            "scene_report": None if report is None else {
                "persons_visible": report.persons_visible,
                "fire_or_smoke": report.fire_or_smoke,
                "threat": report.threat.value,
                "uncertainty": report.uncertainty.value,
                "summary": report.summary,
            },
        }


def degradation(snapshot: Mapping[str, Any]) -> list[str]:
    """Why the runtime is not fully working, as short fixed phrases (empty when it is)."""
    reasons = []
    components = snapshot["components"]
    capture = components["capture"]
    if not capture["ready"]:
        reasons.append(f"capture {capture['state']}" + (f" ({capture['problem']})" if capture["problem"] else ""))
    live = snapshot.get("live")
    if live is not None and live["video"] != "fresh":
        reasons.append(f"video {live['video']}")
    detector = components["detector"]
    if detector["state"] != "available":
        reasons.append(f"detector unavailable ({detector['problem']})")
    scene = components["scene"]
    if scene["state"] == "unavailable":
        reasons.append(f"scene analysis unavailable ({scene['problem']})")
    incidents = components["incidents"]
    if incidents["pending_signals"]:
        reasons.append(f"{incidents['pending_signals']} rule observation(s) not yet recorded ({incidents['problem']})")
    if incidents["signals_dropped"]:
        reasons.append(f"{incidents['signals_dropped']} rule observation(s) lost: not recorded")
    notifications = components["notifications"]
    for channel, state in notifications["channels"].items():
        if state["state"] != "available":
            reasons.append(f"{channel} unavailable ({state['problem']}); its alerts stay queued")
    worker = notifications["worker"]
    if notifications["channels"] and worker.get("problem"):
        reasons.append(f"delivery worker problem ({worker['problem']})")
    if snapshot["runtime"]["state"] == "running" and notifications["channels"] and worker.get("state") == "stopped":
        reasons.append("delivery worker stopped")
    return reasons


# ---------------------------------------------------------------- assembly (device side)


class StartupRefused(Exception):
    """``sentinel run`` cannot start; the label says why. Nothing it started is left running."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


@dataclass(frozen=True)
class SceneOptions:
    binary: Path
    model: Path
    mmproj: Path
    ready_timeout_s: float = 180.0
    min_free_bytes: int = 3_000_000_000  # demo_profile's precheck before llama-server (V2-01)


@dataclass(frozen=True)
class RunOptions:
    data_dir: Path
    engine: Path
    min_free_bytes: int = 1_500_000_000  # the track probe's provisional guard (U18 open)
    scene: SceneOptions | None = None  # None: scene analysis disabled (the default)


def environment_notifiers(
    config: NotificationsConfig, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, Notifier], dict[str, str]]:
    """Notifiers for the listed channels; a channel without credentials is unavailable and its rows wait."""
    notifiers: dict[str, Notifier] = {}
    problems: dict[str, str] = {}
    if "telegram" in config.channels:
        try:
            notifiers["telegram"] = TelegramNotifier.from_environment(
                timeout_s=config.request_timeout_s, environ=None if environ is None else dict(environ)
            )
        except NotifierUnavailable:
            problems["telegram"] = "credentials_missing"
    return notifiers, problems


@dataclass
class Devices:
    """How the device-side parts are made; tests pass fakes."""

    capture_source: Callable[[CaptureConfig], VideoSource]
    tracker_backend: Callable[[Path], Any]  # has load(), track(), reset()
    scene_server: Callable[[SceneOptions, int], Any]  # has start(ready_timeout_s), status(), stop(grace_s)
    scene_request: Callable[[int, float], Callable[[AnalysisJob, Any], WorkerOutcome]]
    meminfo: Callable[[], dict[str, int] | None]
    notifiers: Callable[[NotificationsConfig], tuple[dict[str, Notifier], dict[str, str]]] = environment_notifiers


@dataclass
class Assembly:
    runtime: DemoRuntime
    database: Database
    startup: dict[str, Any] = field(default_factory=dict)

    def close(self, shutdown: Mapping[str, Any]) -> bool:
        """Close the database unless a thread that uses it is still running."""
        if not shutdown["stopped"].get("outbox", True):
            return False
        self.database.close()
        return True


def memfree_problem(meminfo: Callable[[], dict[str, int] | None], min_free_bytes: int) -> tuple[str | None, Any]:
    before = meminfo()
    if before is not None and before["MemFree"] < min_free_bytes:
        return "memfree_below_minimum", before
    return None, before


def file_facts(path: Path) -> FileFacts | None:
    """Name, size and modification time, as demo_profile.py records them; metadata only, nothing is read."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds")
    return FileFacts(Path(path).name, stat.st_size, mtime)


def _sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _identity_problem(label: str, path: Path, recorded: FileFacts) -> str | None:
    """Hash files up to STARTUP_HASH_LIMIT_BYTES; compare larger ones by name, size and mtime only."""
    actual = file_facts(path)
    if actual is None:
        return f"{label} {Path(path).name} is missing"
    if actual.name != recorded.name:
        return f"{label} {actual.name} is not the profiled {recorded.name}"
    if recorded.bytes <= STARTUP_HASH_LIMIT_BYTES and recorded.sha256:
        if actual.bytes != recorded.bytes or _sha256(path) != recorded.sha256:
            return f"{label} {actual.name} does not have the profiled SHA-256"
        return None
    if (actual.bytes, actual.mtime_utc) != (recorded.bytes, recorded.mtime_utc):
        return (f"{label} {actual.name} ({actual.bytes} B, {actual.mtime_utc}) is not the profiled "
                f"{recorded.name} ({recorded.bytes} B, {recorded.mtime_utc})")
    return None


def profile_mismatch(
    profile: ResourceProfile, config: SentinelConfig, scene: SceneOptions, *, engine_sha256: str = LEGACY_ENGINE_SHA256
) -> str | None:
    """None if the selected runtime configuration is the one ``profile`` measured, else the first difference.

    Startup checks, in order: the server flags; the scene interval; the request
    fingerprint; the llama-server binary and every build library up to
    STARTUP_HASH_LIMIT_BYTES by SHA-256; larger libraries (libggml-cuda) and the
    model files by name, size and modification time only; the detector engine's
    pin. Large files are not hashed here: that would fill the page cache just
    before the MemFree precheck (U18). Metadata checks do not detect a same-size
    replacement that keeps its modification time (STARTUP_IDENTITY_LIMITATION);
    the step-4 identity snapshot and acceptance rehashing cover the measured run.
    """
    expected = PROFILED_FLAGS + PROMPT_CACHE_FLAGS
    if tuple(profile.llama_flags) != expected:
        return (f"resource profile {profile.profile_id} measured llama-server flags {' '.join(profile.llama_flags)}, "
                f"not the runtime's {' '.join(expected)}")
    if profile.scene_interval_s != config.scene.interval_s:
        return (f"resource profile {profile.profile_id} measured a {profile.scene_interval_s:g} s scene interval, "
                f"not the configured {config.scene.interval_s:g} s")
    if profile.scene_request_sha256 != SCENE_REQUEST_SHA256:
        return f"resource profile {profile.profile_id} measured another scene request than the runtime sends"
    if profile.llama_server is None:
        return f"resource profile {profile.profile_id} lacks the llama-server identity"
    problem = _identity_problem("llama-server", scene.binary, profile.llama_server)
    if problem is not None:
        return problem
    for library in profile.llama_libraries:
        problem = _identity_problem("llama.cpp library", Path(scene.binary).parent / library.name, library)
        if problem is not None:
            return problem
    for label, path, recorded in (("scene model", scene.model, profile.llm),
                                  ("scene projector", scene.mmproj, profile.mmproj)):
        actual = file_facts(path)
        if actual is None:
            return f"{label} {Path(path).name} is missing"
        if (actual.name, actual.bytes, actual.mtime_utc) != (recorded.name, recorded.bytes, recorded.mtime_utc):
            return (f"{label} {actual.name} ({actual.bytes} B, {actual.mtime_utc}) is not the profiled "
                    f"{recorded.name} ({recorded.bytes} B, {recorded.mtime_utc})")
    if profile.engine_sha256 != engine_sha256:
        return f"resource profile {profile.profile_id} measured another detector engine than the pinned one"
    return None


def scene_admission(
    config: SentinelConfig,
    scene: SceneOptions,
    *,
    profiles: Mapping[str, ResourceProfile] = RESOURCE_PROFILES,
) -> tuple[str | None, str | None]:
    """(problem, producer revision) for ``--scene``: an enabled scene adapter whose resource
    profile is ACCEPTED (step-4 evidence, D46) and matches the selected configuration."""
    statuses = [s for s in resolve(config.adapters, profiles=profiles) if s.manifest.adapter_id == SCENE_ADAPTER_ID]
    if not statuses:
        return f"no {SCENE_ADAPTER_ID} adapter in the configuration", None
    status = statuses[0]
    if status.state is not AdapterState.ENABLED:
        return f"{SCENE_ADAPTER_ID} {status.state.value}: {status.reason}", None
    profile = profiles[status.manifest.resource_profile_id]  # type: ignore[index]  # resolve() checked it
    problem = profile_mismatch(profile, config, scene)
    if problem is not None:
        return f"{SCENE_ADAPTER_ID}: {problem}", None
    return None, status.manifest.producer_revision


def assemble(
    config: SentinelConfig,
    options: RunOptions,
    devices: Devices,
    clock: Clock,
    *,
    profiles: Mapping[str, ResourceProfile] = RESOURCE_PROFILES,
) -> Assembly:
    """Check, open and load everything ``sentinel run`` needs, then build the runtime (not started).

    Refusals (StartupRefused) happen before any model is loaded: a missing or
    invalid stream URL, a scene request without an accepted, matching combined
    profile (D46; checked before the database opens or llama-server starts), a
    database another process owns. ``profiles`` is the static registry; tests
    pass synthetic fixtures, and ``sentinel run`` has no option to change it. Model failures do not refuse: that component
    is unavailable, with its reason, and core monitoring runs without it.
    """
    from .media.capture import CaptureWorker
    from .media.frames import FrameStamper

    try:
        source = devices.capture_source(config.capture)  # the URL is checked before any model loads
    except SourceError as exc:
        raise StartupRefused(exc.reason) from None
    revision = None
    if options.scene is not None:
        problem, revision = scene_admission(config, options.scene, profiles=profiles)
        if problem is not None:
            raise StartupRefused(f"scene_not_admitted: {problem}")
    try:
        database = Database.open(options.data_dir / DATABASE_NAME)
    except DatabaseError as exc:
        raise StartupRefused(f"database: {exc}") from None
    startup: dict[str, Any] = {}
    server = None
    try:
        notifiers, notifier_problems = devices.notifiers(config.notifications)
        incidents = IncidentService(database, clock, incidents=config.incidents, notifications=config.notifications)
        outbox = OutboxLoop(OutboxWorker(database, clock, notifiers, config.notifications))

        scene = None
        scene_problem = None
        if options.scene is not None:
            scene_problem, memory = memfree_problem(devices.meminfo, options.scene.min_free_bytes)
            startup["scene_memory_before"] = memory
            if scene_problem is None:
                server = devices.scene_server(options.scene, config.scene_server.port)
                status = server.start(options.scene.ready_timeout_s)
                startup["scene_server"] = {"state": status.state.value, "problem": status.problem,
                                           "layers": status.layers, "vision_on_gpu": status.vision_on_gpu}
                if status.state.value == "ready":
                    images = RecentImages()
                    run_job = devices.scene_request(config.scene_server.port, config.scene_server.request_timeout_s)
                    analyzer = ThreadedSceneAnalyzer(run_job, images, revision=revision or "unknown")
                    scene = SceneRuntime(analyzer, images, server)
                else:
                    scene_problem = f"server_{status.state.value}:{status.problem}"
                    server = None

        tracker = None
        detector_problem, memory = memfree_problem(devices.meminfo, options.min_free_bytes)
        startup["detector_memory_before"] = memory
        if detector_problem is None:
            backend = devices.tracker_backend(options.engine)
            try:
                backend.load()  # the D27 guard and the engine pin run first
            except TrackerError as exc:
                detector_problem = exc.label + (f" ({exc.error_type})" if exc.error_type else "")
            else:
                tracker = PersonTracker(backend)
        startup["detector_problem"] = detector_problem
        startup["scene_problem"] = scene_problem
        startup["notifier_problems"] = notifier_problems

        slot = LatestFrame()
        capture = CaptureWorker(source, FrameStamper(config.camera.id, clock), slot, config.capture)
        runtime = DemoRuntime(
            config, clock, capture=capture, slot=slot, incidents=incidents, outbox=outbox,
            tracker=tracker, detector_problem=detector_problem, scene=scene, scene_problem=scene_problem,
            notifier_problems=notifier_problems,
        )
    except BaseException:
        if server is not None:
            server.stop(10.0)
        database.close()
        raise
    return Assembly(runtime, database, startup)
