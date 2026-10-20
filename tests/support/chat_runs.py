"""The alpha chat check's run, made by the real runtime (D80), and the face + chat integration run's (session 103).

``build_run`` drives DemoRuntime with a threaded scene worker and chat in its validation run (and, with ``face``, a
threaded face worker in its validation run too) over simulated minutes, writing the guard's files as step5_guard writes
them and the chat load as alpha_load records it. Used by test_alpha_check and test_integration_check.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentinel.cli import status_line
from sentinel.config import parse_config
from sentinel.contracts import NormalizedBox, PixelFormat, StreamIdentity
from sentinel.demo_runtime import DemoRuntime, FaceRuntime, SceneRuntime
from sentinel.identity.association import FaceObservation
from sentinel.identity.legacy_deepface import FaceRun
from sentinel.identity.matching import EnrolledIdentity, Enrollment
from sentinel.identity.worker import FaceWorker
from sentinel.incidents.service import IncidentService
from sentinel.jobs import WorkerOutcome
from sentinel.media.capture import CapturedFrame, CaptureState, CaptureStatus, LatestFrame
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.media.jpeg import EncodedJpeg
from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer
from sentinel.scene.chat import CHAT_ADMISSION, ChatCancelled, ChatQuestion
from sentinel.status_page import build_status, read_store
from sentinel.storage.database import Database
from sentinel.tracking.tracker import PersonTracker, RawTrack

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)
REPORT = json.dumps({"persons_visible": 1, "fire_or_smoke": False, "threat": "low", "observations": ["a person"],
                     "uncertainty": "low", "summary": "A synthetic report."})
CAPTURES = ((60.0, "e1"), (180.0, "k1"), (300.0, "a"), (420.0, "e2"), (540.0, "k2"), (590.0, "end"))
HELD_AT_S = 42.0  # this question is held until the next periodic scene job (every 4 s) ends it after its grace
LOOPBACK = [{"family": "tcp", "port": 18081, "scope": "loopback"}, {"family": "tcp", "port": 18090, "scope": "loopback"}]



class Capture:
    def __init__(self, clock: FakeClock) -> None:
        self.stamper = FrameStamper("cam-1", clock)
        self.stamper.connect()

    @property
    def connected(self) -> StreamIdentity | None:
        return self.stamper.current_stream

    def status(self) -> CaptureStatus:
        return CaptureStatus(CaptureState.STREAMING, self.connected, 0, 1, 0, 0, None, None)

    def start(self) -> None:
        pass

    def stop(self, timeout_s: float) -> bool:
        return True


class Tracker:
    def __init__(self, box: tuple[float, float, float, float] = (0.3, 0.2, 0.6, 0.9)) -> None:
        self.box = box

    def load(self) -> None:
        pass

    def track(self, image: Any) -> list[RawTrack]:
        return [RawTrack(1, *self.box, 0.9)]

    def reset(self) -> None:
        pass


class Image:
    shape = (480, 640, 3)

    def copy(self) -> "Image":  # the face worker keeps a private copy of an offered frame
        return self


class Model:
    """The scene model's stand-in for chat: answers at once, except a question it is told to hold, which runs until a
    scene job's arrival aborts it (the real transport ends the request the same way)."""

    def __init__(self) -> None:
        self.hold_next = False
        self.holding = threading.Event()

    def __call__(self, question: ChatQuestion, image: Any, abort: Any = None) -> tuple[str, bool]:
        if not self.hold_next:
            return "A chair stands by the window.", False
        self.hold_next = False
        self.holding.set()
        try:
            deadline = time.monotonic() + 10
            while abort is not None and abort.reason is None and time.monotonic() < deadline:
                time.sleep(0.002)
            raise ChatCancelled((abort.reason if abort is not None else None) or "cancelled")
        finally:
            self.holding.clear()


class SimTimers:
    """The scene worker's grace timers on the simulated clock: the run fires each when its time has come."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.pending: list[dict[str, Any]] = []

    def __call__(self, delay_s: float, call: Any) -> Any:
        entry = {"due": self.clock.monotonic_ns() + round(delay_s * 1e9), "call": call, "cancelled": False}
        self.pending.append(entry)
        return lambda: entry.update(cancelled=True)

    def fire_due(self) -> int:
        now, fired = self.clock.monotonic_ns(), 0
        for entry in [e for e in self.pending if e["due"] <= now]:
            self.pending.remove(entry)
            if not entry["cancelled"]:
                entry["call"]()
                fired += 1
        return fired


STARTUP = {
    "memory_policy": {"thp": "workload_disabled", "model_file_release": "post_load"},
    "thp_scope": {"before_launch": {"runtime": 1}, "llama_ready": {"runtime": 1, "scene_server": 1},
                  "workload_verified": {"runtime": 0, "scene_server": 1}},
    "thp_disable": {"verified": True, "reason": None, "set_rc": 0, "thp_enabled": 0},
    "releases": {"scene": {"settle_s": 15.0, "files": {"llm": {"result": "returned_0"},
                                                       "mmproj": {"result": "returned_0"}}},
                 "detector": {"settle_s": 15.0, "files": {"engine": {"result": "returned_0"}}}},
    "scene_server": {"state": "ready", "problem": None, "layers": "17/17", "vision_on_gpu": True},
    "scene_problem": None, "detector_problem": None, "notifier_problems": {}, "status_page": "http://127.0.0.1:18090/",
}


# Session 103, the face + chat integration run: one seated person (pixels on the 640x480 frame) whose face, inside the
# head region, matches the one enrolled identity (synthetic 4-D embeddings).
PERSON_PX = (100.0, 100.0, 200.0, 400.0)
FACE_BOX = NormalizedBox(x1=0.19, y1=0.22, x2=0.28, y2=0.35)
ENROLLMENT = Enrollment(model_revision="synthetic", dimension=4,
                        identities=(EnrolledIdentity(identity_id="idn-00000000000a",
                                                     prototypes=((1.0, 0.0, 0.0, 0.0),)),))
FACE_RELEASES = {"files": {"facenet512_weights.h5": {"result": "returned_0"},
                           "face_detection_yunet_2023mar.onnx": {"result": "returned_0"}}}
FCI_CAPTURES = ((60.0, "c1"), (240.0, "c2"), (470.0, "end"))


def _matching_face(frame, image) -> FaceRun:
    return FaceRun((FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.9, embedding=(0.97, 0.1, 0.0, 0.05)),),
                   1, 0)


def _settle(worker: FaceWorker | None) -> None:
    deadline = time.monotonic() + 5
    while worker is not None and (worker.status()["busy"] or worker.status()["waiting"]):
        assert time.monotonic() < deadline
        time.sleep(0.0005)


def _follow(runtime: DemoRuntime, status: int, document: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Follow an accepted question to its end, as alpha_load does (through the service directly)."""
    job = document.get("job")
    deadline = time.monotonic() + 10
    while job and document.get("state") in ("sending", "waiting", "answering") and time.monotonic() < deadline:
        time.sleep(0.002)
        status, document = runtime.chat.poll(job)
    return status, document


def build_run(mods, root: Path, *, fps: float = 4.0, duration: float = 600.0, pressure: Any = None,
              face: bool = False, live: bool = True, captures_at: tuple = CAPTURES,
              analyze: Any = _matching_face) -> Path:
    """``face`` (session 103): face recognition in its validation run as well, one person in view, its identity records
    on stdout, and the guard's face_chat readiness recorded; ``live``: the live view and its viewer."""
    run = root / "run"
    run.mkdir()
    clock = FakeClock(utc=NOW)
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
    slot, capture = LatestFrame(), Capture(clock)
    images = RecentImages()
    timers = SimTimers(clock)
    scene = SceneRuntime(ThreadedSceneAnalyzer(lambda job, image: WorkerOutcome.completed(job.job_id, REPORT), images,
                                               revision="1+lfm2-vl-1.6b-q4_0", now=clock.monotonic_ns,
                                               schedule=timers), images)
    db = Database.open(root / "sentinel.db")
    model = Model()
    worker = FaceWorker(analyze, clock) if face else None
    runtime = DemoRuntime(config, clock, capture=capture, slot=slot,
                          incidents=IncidentService(db, clock, incidents=config.incidents,
                                                    notifications=config.notifications),
                          tracker=PersonTracker(Tracker(PERSON_PX) if face else Tracker()), scene=scene,
                          status_refresh_s=0.0, live_preview=live,
                          encode=lambda *a, **k: EncodedJpeg(b"\xff\xd8p\xff\xd9", 8, 8), chat=True, chat_run=model,
                          chat_admission=CHAT_ADMISSION, chat_validation=True, pressure=lambda: 3_000_000_000,
                          **({"face": FaceRuntime(worker, 1, True), "enrollment": ENROLLMENT} if face else {}))
    runtime.start()
    ready = 1.0  # FakeClock's start, the runtime's readiness on the guard's clock
    t0 = clock.monotonic_ns()
    startup = {**STARTUP, "chat": runtime.chat.availability(),
               **({"live_preview": runtime.preview.bounds()} if live else {}),
               **({"face": {"validation_run": True, "identities_enrolled": 1}, "face_problem": None,
                   "releases": {**STARTUP["releases"], "face": FACE_RELEASES}} if face else {})}
    lines = [json.dumps({"run": "starting", "startup": startup})]
    chat_lines = [json.dumps({"event": "started", "mono_s": ready, "questions": mods.load.QUESTIONS,
                              "every_s": 6.0})]
    viewer_lines, pending, captures, next_status = [], list(captures_at), [], 0.0
    held = None  # (asked at, job) of the question held until the next scene job ends it
    for n in range(1, int(duration * fps) + 1):
        t = n / fps
        clock.advance(ns=t0 + round(n * 1e9 / fps) - clock.monotonic_ns())
        frame = capture.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        slot.publish(CapturedFrame(frame, Image()))
        runtime.step(timeout_s=0)
        _settle(worker)  # the face worker's analysis (its own thread) ends before the loop goes on
        deadline = time.monotonic() + 5
        if timers.fire_due():  # the grace is over: the question ends before the simulated clock moves on
            while model.holding.is_set() and time.monotonic() < deadline:
                time.sleep(0.0005)
        while scene.analyzer.busy() and not model.holding.is_set() and time.monotonic() < deadline:
            time.sleep(0.0005)  # scene jobs finish at once here; only the held question stays running
        runtime.step(timeout_s=0)
        lines.extend(json.dumps(record) for record in runtime.drain_identity_records())
        if held is not None and not model.holding.is_set():
            status, document = _follow(runtime, *runtime.chat.poll(held[1]))
            chat_lines.append(json.dumps(mods.load._record(held[0], "long", document, status)))
            held = None
        if t % 6 == 0 and not scene.analyzer.busy():  # alpha_load's cadence: a question every 6 s, alternating
            kind = "short" if (t // 6) % 2 else "long"
            model.hold_next = t == HELD_AT_S
            status, document = runtime.chat.ask(ChatQuestion(mods.load.QUESTIONS[kind]))
            if t == HELD_AT_S:
                deadline = time.monotonic() + 5
                while not model.holding.is_set() and time.monotonic() < deadline:
                    time.sleep(0.0005)
                held = (ready + t, document["job"])
            else:
                status, document = _follow(runtime, status, document)
                chat_lines.append(json.dumps(mods.load._record(ready + t, kind, document, status)))
        if live and n % int(fps) == 0:  # the viewer's tallies are written here; alpha_load's own on a real page
            viewer_lines.append(json.dumps({"second_mono_s": ready + t, "pictures": 14, "bytes": 14 * 30_000,
                                            "max_age_ms_when_sent": 180, "states": {}}))
        if t >= next_status:
            lines.append(json.dumps(status_line(runtime.snapshot())))
            next_status += 30.0
        while pending and t >= pending[0][0]:
            at, name = pending.pop(0)
            status = build_status(runtime.snapshot(), read_store(root / "sentinel.db"), clock.utc_now(),
                                  runtime.preview.view() if live else None)
            (run / f"status-{name}.json").write_text(json.dumps(status))
            captures.append({"name": name, "at_s": at, "status": "saved", "listeners": LOOPBACK,
                             "taken_after_launch_s": 70.0 + at})
    shutdown = runtime.shutdown()
    lines.extend(json.dumps(record) for record in runtime.drain_identity_records())
    lines.append(json.dumps({"run": "stopped", "shutdown": shutdown, "database_closed": True,
                             "status": status_line(runtime.snapshot())}))
    db.close()
    (run / "run.jsonl").write_text("\n".join(lines) + "\n")
    (run / "chat-load.jsonl").write_text("\n".join(chat_lines) + "\n")
    if live:
        (run / "viewer.jsonl").write_text("\n".join(viewer_lines) + "\n")
    samples = (pressure or (lambda s: 3_900_000_000 if s > 30 else 4_400_000_000))
    (run / "guard.jsonl").write_text("".join(
        json.dumps({"t_mono": ready - 60 + k * 0.2, "pressure_bytes": samples(-60 + k * 0.2)}) + "\n"
        for k in range(int((duration + 60) / 0.2))))
    result = {
        "schema_version": 1, "mode": "step5_guard", "status": "duration_stop",
        "parameters": {"duration_s": duration, **({"require_ready": "face_chat"} if face else {})},
        "child": {"returncode": 0, "ready_after_s": 60.0, "ready_mono_s": ready},
        "guard": {"trigger": None, "peak_pressure_bytes": 4_400_000_000, "min_mem_free_bytes": 1_400_000_000,
                  "min_mem_available_bytes": 3_500_000_000},
        "stop": {"forced": False}, "cleanup_clear": True,
        "leftovers": {"llama_server_after_exit": 0, "llama_server_left": 0, "runtime_group_left": False,
                      "listeners_after": []},
        "captures": captures, "cues": [],
        "run_output": mods.guard.read_run_output(run / "run.jsonl"),
        **({"readiness": {"required": "face_chat", "met": True, "reason": None, "starting_after_s": 60.0}}
           if face else {}),
    }
    (run / "result.json").write_text(json.dumps(result))
    return run
