"""The alpha chat check's run, made by the real runtime (D80), and the face + chat integration run's (session 103).

``build_run`` drives DemoRuntime with a threaded scene worker and chat in its validation run (and, with ``face``, a
threaded face worker in its validation run too) over simulated minutes, writing the guard's files as step5_guard writes
them and the chat load as alpha_load records it. Used by test_alpha_check and test_integration_check. With
``grounding`` (session 106) the grounding check's four questions are asked at alpha_load's G slots and recorded with
alpha_load's own G record function, their synthetic answers (never real ones) in grounding.jsonl.
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

    def __init__(self, answers: dict[str, tuple[str, bool]] | None = None) -> None:
        self.hold_next = False
        self.holding = threading.Event()
        self.answers = answers or {}  # session 106: synthetic answers by question (text, cut short)

    def __call__(self, question: ChatQuestion, image: Any, abort: Any = None) -> tuple[str, bool]:
        if not self.hold_next:
            return self.answers.get(question.question, ("A chair stands by the window.", False))
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
              analyze: Any = _matching_face, grounding: dict[str, dict[str, Any]] | None = None,
              viewer: bool = False) -> Path:
    """``face`` (session 103): face recognition in its validation run as well, one person in view, its identity records
    on stdout, and the guard's face_chat readiness recorded; ``live``: the live view and its viewer. ``grounding``
    (session 106): G's questions at alpha_load's slots, by ID: {"answer": synthetic text, "cut": bool} or
    {"hold": True} (held until the next scene job cancels it); grounding.jsonl written as alpha_load writes it, with the
    slots the run never reached as not asked. ``viewer`` (session 109, with ``live``): a viewer pulls the newest picture
    once a simulated second through the live view's own next_frame, as the dashboard's player does, so that the status
    page's live-view counters count it."""
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
    g_slots = {g: (g, offset, question) for g, offset, question in mods.load.GROUNDING} if grounding is not None else {}
    model = Model({g_slots[g][2]: (spec["answer"], spec.get("cut", False)) for g, spec in (grounding or {}).items()
                   if "answer" in spec})
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
    held = None  # (asked at, job, kind, G slot) of the question held until the next scene job ends it
    a0, g_pending, g_lines = None, sorted(g_slots.values(), key=lambda item: item[1]), []
    seen = None  # the newest picture the viewer was sent (stream epoch, frame)
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
            record = mods.load._record(held[0], mods.load.GROUNDING_KIND if held[3] else held[2], document, status)
            if held[3]:
                record["g"] = held[3][0]
                g_lines.append(json.dumps(mods.load._grounding_record(held[3], a0, held[0], held[2], document, status)))
            chat_lines.append(json.dumps(record))
            held = None
        if t % 6 == 0 and not scene.analyzer.busy():  # alpha_load's cadence: a question every 6 s, alternating
            kind = "short" if (t // 6) % 2 else "long"
            g_slot = (g_pending.pop(0) if g_pending and a0 is not None and ready + t >= a0 + g_pending[0][1]
                      else None)
            hold = t == HELD_AT_S or bool(g_slot and grounding[g_slot[0]].get("hold"))
            model.hold_next = hold
            status, document = runtime.chat.ask(ChatQuestion(g_slot[2] if g_slot else mods.load.QUESTIONS[kind]))
            if status == 202 and a0 is None:
                a0 = ready + t  # A0: the first acceptance (the simulated clock does not move during the request)
            if hold:
                deadline = time.monotonic() + 5
                while not model.holding.is_set() and time.monotonic() < deadline:
                    time.sleep(0.0005)
                held = (ready + t, document["job"], kind if g_slot else "long", g_slot)  # as recorded before G
            else:
                status, document = _follow(runtime, status, document)
                record = mods.load._record(ready + t, mods.load.GROUNDING_KIND if g_slot else kind, document, status)
                if g_slot:
                    record["g"] = g_slot[0]
                    g_lines.append(json.dumps(mods.load._grounding_record(g_slot, a0, ready + t, kind, document,
                                                                          status)))
                chat_lines.append(json.dumps(record))
        if live and viewer and n % int(fps) == 0:  # the dashboard's player: the newest picture newer than its last
            runtime.preview.encode_newest()
            answer = runtime.preview.next_frame(seen, wait_s=0.0)
            seen = answer.picture.order if answer.picture is not None else seen
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
    if grounding is not None:  # as alpha_load writes it when it ends: the slots never reached, not asked
        why = mods.load.NOT_ASKED["never_accepted"] if a0 is None else mods.load.NOT_ASKED["page_gone"]
        g_lines += [json.dumps(mods.load._grounding_record(g_slot, a0, None, None, {}, None, state="not_asked",
                                                           reason=why)) for g_slot in g_pending]
        path = run / "grounding.jsonl"
        path.write_text("".join(line + "\n" for line in g_lines))
        path.chmod(0o600)
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
