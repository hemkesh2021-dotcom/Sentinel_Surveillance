"""D80: the alpha chat check's reading (alpha_check.py) and its load tool's records (alpha_load.py).

The run directory is made by the real runtime (DemoRuntime, a threaded scene worker, the live view and chat in a
validation run) over ten simulated minutes, with the guard's files written as step5_guard writes them and the chat
load recorded with alpha_load's own record function; one question is held until a scene job arrives, so the
preemption really happens. The viewer's per-second tallies are written by the test. Each judged item is then shown
failing (or MISSING) when its evidence says otherwise. alpha_load itself asks and watches a real StatusServer.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from sentinel.cli import status_line
from sentinel.config import parse_config
from sentinel.contracts import PixelFormat, StreamIdentity
from sentinel.demo_runtime import DemoRuntime, SceneRuntime
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

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
NOW = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)
REPORT = json.dumps({"persons_visible": 1, "fire_or_smoke": False, "threat": "low", "observations": ["a person"],
                     "uncertainty": "low", "summary": "A synthetic report."})
CAPTURES = ((60.0, "e1"), (180.0, "k1"), (300.0, "a"), (420.0, "e2"), (540.0, "k2"), (590.0, "end"))
HELD_AT_S = 42.0  # this question is held until the next scene job (every 4 s) ends it
LOOPBACK = [{"family": "tcp", "port": 18081, "scope": "loopback"}, {"family": "tcp", "port": 18090, "scope": "loopback"}]


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(RUNNERS))
    try:
        yield type("Mods", (), {"check": importlib.import_module("alpha_check"),
                                "load": importlib.import_module("alpha_load"),
                                "guard": importlib.import_module("step5_guard"),
                                "evidence": importlib.import_module("face_evidence")})
    finally:
        sys.path.remove(str(RUNNERS))


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
    def load(self) -> None:
        pass

    def track(self, image: Any) -> list[RawTrack]:
        return [RawTrack(1, 0.3, 0.2, 0.6, 0.9, 0.9)]

    def reset(self) -> None:
        pass


class Image:
    shape = (480, 640, 3)


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


def _follow(runtime: DemoRuntime, status: int, document: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Follow an accepted question to its end, as alpha_load does (through the service directly)."""
    job = document.get("job")
    deadline = time.monotonic() + 10
    while job and document.get("state") in ("sending", "waiting", "answering") and time.monotonic() < deadline:
        time.sleep(0.002)
        status, document = runtime.chat.poll(job)
    return status, document


def build_run(mods, root: Path, *, fps: float = 4.0, duration: float = 600.0, pressure: Any = None) -> Path:
    run = root / "run"
    run.mkdir()
    clock = FakeClock(utc=NOW)
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
    slot, capture = LatestFrame(), Capture(clock)
    images = RecentImages()
    scene = SceneRuntime(ThreadedSceneAnalyzer(lambda job, image: WorkerOutcome.completed(job.job_id, REPORT), images,
                                               revision="1+lfm2-vl-1.6b-q4_0"), images)
    db = Database.open(root / "sentinel.db")
    model = Model()
    runtime = DemoRuntime(config, clock, capture=capture, slot=slot,
                          incidents=IncidentService(db, clock, incidents=config.incidents,
                                                    notifications=config.notifications),
                          tracker=PersonTracker(Tracker()), scene=scene, status_refresh_s=0.0, live_preview=True,
                          encode=lambda *a, **k: EncodedJpeg(b"\xff\xd8p\xff\xd9", 8, 8), chat=True, chat_run=model,
                          chat_admission=CHAT_ADMISSION, chat_validation=True, pressure=lambda: 3_000_000_000)
    runtime.start()
    ready = 1.0  # FakeClock's start, the runtime's readiness on the guard's clock
    t0 = clock.monotonic_ns()
    lines = [json.dumps({"run": "starting", "startup": {**STARTUP, "live_preview": runtime.preview.bounds(),
                                                         "chat": runtime.chat.availability()}})]
    chat_lines = [json.dumps({"event": "started", "mono_s": ready, "question": mods.load.QUESTION,
                              "every_s": 6.0})]
    viewer_lines, pending, captures, next_status = [], list(CAPTURES), [], 0.0
    held = None  # (asked at, job) of the question held until the next scene job ends it
    for n in range(1, int(duration * fps) + 1):
        t = n / fps
        clock.advance(ns=t0 + round(n * 1e9 / fps) - clock.monotonic_ns())
        frame = capture.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        slot.publish(CapturedFrame(frame, Image()))
        runtime.step(timeout_s=0)
        deadline = time.monotonic() + 5
        while scene.analyzer.busy() and not model.holding.is_set() and time.monotonic() < deadline:
            time.sleep(0.0005)  # scene jobs finish at once here; only the held question stays running
        runtime.step(timeout_s=0)
        if held is not None and not model.holding.is_set():
            status, document = _follow(runtime, *runtime.chat.poll(held[1]))
            chat_lines.append(json.dumps(mods.load._record(held[0], document, status)))
            held = None
        if t % 6 == 0 and not scene.analyzer.busy():  # alpha_load's cadence: a question every 6 s
            model.hold_next = t == HELD_AT_S
            status, document = runtime.chat.ask(ChatQuestion(mods.load.QUESTION))
            if model.hold_next or t == HELD_AT_S:
                deadline = time.monotonic() + 5
                while not model.holding.is_set() and time.monotonic() < deadline:
                    time.sleep(0.0005)
                held = (ready + t, document["job"])
            else:
                chat_lines.append(json.dumps(mods.load._record(ready + t, *reversed(_follow(runtime, status,
                                                                                             document)))))
        if n % int(fps) == 0:  # the viewer's tallies are written here; alpha_load's own are tested on a real page
            viewer_lines.append(json.dumps({"second_mono_s": ready + t, "pictures": 14, "bytes": 14 * 30_000,
                                            "max_age_ms_when_sent": 180, "states": {}}))
        if t >= next_status:
            lines.append(json.dumps(status_line(runtime.snapshot())))
            next_status += 30.0
        while pending and t >= pending[0][0]:
            at, name = pending.pop(0)
            status = build_status(runtime.snapshot(), read_store(root / "sentinel.db"), clock.utc_now(),
                                  runtime.preview.view())
            (run / f"status-{name}.json").write_text(json.dumps(status))
            captures.append({"name": name, "at_s": at, "status": "saved", "listeners": LOOPBACK,
                             "taken_after_launch_s": 70.0 + at})
    shutdown = runtime.shutdown()
    lines.append(json.dumps({"run": "stopped", "shutdown": shutdown, "database_closed": True,
                             "status": status_line(runtime.snapshot())}))
    db.close()
    (run / "run.jsonl").write_text("\n".join(lines) + "\n")
    (run / "chat-load.jsonl").write_text("\n".join(chat_lines) + "\n")
    (run / "viewer.jsonl").write_text("\n".join(viewer_lines) + "\n")
    samples = (pressure or (lambda s: 3_900_000_000 if s > 30 else 4_400_000_000))
    (run / "guard.jsonl").write_text("".join(
        json.dumps({"t_mono": ready - 60 + k * 0.2, "pressure_bytes": samples(-60 + k * 0.2)}) + "\n"
        for k in range(int((duration + 60) / 0.2))))
    result = {
        "schema_version": 1, "mode": "step5_guard", "status": "duration_stop",
        "parameters": {"duration_s": duration},
        "child": {"returncode": 0, "ready_after_s": 60.0, "ready_mono_s": ready},
        "guard": {"trigger": None, "peak_pressure_bytes": 4_400_000_000, "min_mem_free_bytes": 1_400_000_000,
                  "min_mem_available_bytes": 3_500_000_000},
        "stop": {"forced": False}, "cleanup_clear": True,
        "leftovers": {"llama_server_after_exit": 0, "llama_server_left": 0, "runtime_group_left": False,
                      "listeners_after": []},
        "captures": captures, "cues": [],
        "run_output": mods.guard.read_run_output(run / "run.jsonl"),
    }
    (run / "result.json").write_text(json.dumps(result))
    return run


@pytest.fixture(scope="module")
def made(mods, tmp_path_factory) -> Path:
    return build_run(mods, tmp_path_factory.mktemp("alpha"))


def text_of(reading) -> str:
    return "\n".join(reading.lines)


def line_for(text: str, start: str) -> str:
    return next(line for line in text.splitlines() if line.split(" ", 1)[1].startswith(start))


def fresh_copy(made: Path, tmp_path: Path) -> Path:
    import shutil
    target = tmp_path / "copy"
    shutil.copytree(made, target)
    return target


def test_a_run_from_the_real_runtime_passes_every_item(mods, made) -> None:
    reading = mods.check.read_chat(made)
    text = text_of(reading)
    assert reading.ok, text
    assert "MISSING" not in text and "FAIL" not in text
    cancel = line_for(text, "cancellation verified")
    assert '"side_preempted": 1' in cancel and '"questions_cancelled_for_a_scene_job": 1' in cancel
    assert line_for(text, "chat-induced scene delay").startswith("PASS")
    assert "DESCRIPTIVE chat load: questions by final state" in text and '"answered"' in text


def test_the_load_records_never_hold_an_answers_text_and_fit_the_privacy_profiles(mods, made) -> None:
    text = (made / "chat-load.jsonl").read_text()
    assert "A chair stands by the window." not in text and '"answer_chars": 29' in text
    for name in ("chat-load.jsonl", "viewer.jsonl", "result.json", "run.jsonl", *(f"status-{n}.json" for _, n in CAPTURES)):
        read = mods.evidence.read(made / name, "f2")
        assert read.nonconforming == [], (name, read.nonconforming[:5])


def edit_last_status(run: Path, change) -> None:
    path = run / "status-end.json"
    status = json.loads(path.read_text())
    change(status)
    path.write_text(json.dumps(status))


@pytest.mark.parametrize(("change", "item", "label"), [
    (lambda s: s["components"]["scene"]["lane"].update(deadline_missed=1), "scene deadlines", "FAIL"),
    (lambda s: s["components"]["chat"]["scene_waits_behind_chat"].update(max_ever_ms=2500.0),
     "chat-induced scene delay", "FAIL"),
    (lambda s: s["components"]["chat"]["scene_waits_behind_chat"].update(n=0), "chat-induced scene delay", "MISSING"),
    (lambda s: s["components"]["scene"]["worker"].update(side_preempted=0), "cancellation verified", "MISSING"),
    (lambda s: s["components"]["incidents"].update(signals_dropped=1), "monitoring kept first", "FAIL"),
    (lambda s: s["live"].update(video="stale"), "every capture", "FAIL"),
])
def test_each_status_item_fails_or_is_missing_when_its_evidence_says_so(mods, made, tmp_path, change, item,
                                                                        label) -> None:
    run = fresh_copy(made, tmp_path)
    edit_last_status(run, change)
    text = text_of(mods.check.read_chat(run))
    assert line_for(text, item).startswith(label), line_for(text, item)


@pytest.mark.parametrize(("pressure", "label"), [
    (lambda s: 5_000_000_001 if s > 130 else 3_000_000_000, "FAIL"),  # steady above 5.0 GB
    (lambda s: 5_400_000_001 if -10 < s < -9 else 3_000_000_000, "FAIL"),  # a loading peak above 5.4 GB
    (lambda s: 4_999_999_999 if s > 110 else 5_399_999_999, "PASS"),  # at both limits
])
def test_memory_is_judged_against_the_existing_decimal_limits(mods, tmp_path, pressure, label) -> None:
    run = build_run(mods, tmp_path, duration=200.0, pressure=pressure)
    text = text_of(mods.check.read_chat(run))
    assert line_for(text, "memory:").startswith(label), line_for(text, "memory:")


def test_answers_must_name_their_frame_and_be_historical_exactly_when_old(mods, made, tmp_path) -> None:
    run = fresh_copy(made, tmp_path)
    lines = (run / "chat-load.jsonl").read_text().splitlines()
    for n, line in enumerate(lines):
        record = json.loads(line)
        if record.get("state") == "answered":
            record["historical"] = not record["historical"]
            lines[n] = json.dumps(record)
            break
    (run / "chat-load.jsonl").write_text("\n".join(lines) + "\n")
    text = text_of(mods.check.read_chat(run))
    assert line_for(text, "real answers").startswith("FAIL")


def test_missing_load_files_and_a_userinfo_url_never_pass(mods, made, tmp_path) -> None:
    run = fresh_copy(made, tmp_path)
    (run / "viewer.jsonl").unlink()
    (run / "chat-load.jsonl").unlink()
    (run / "run.err").write_text("opened rtsp://user:secret@camera.invalid/stream\n")
    text = text_of(mods.check.read_chat(run))
    assert line_for(text, "the live view delivered pictures").startswith("MISSING")
    assert line_for(text, "real answers").startswith("MISSING")
    assert line_for(text, "cancellation verified").startswith("MISSING")
    assert line_for(text, "no URL with userinfo").startswith("FAIL")


def test_the_load_tool_asks_and_watches_a_real_page_and_keeps_numbers_and_labels_only(mods, tmp_path,
                                                                                     monkeypatch) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
    from status_fixtures import Camera, incident_store

    from sentinel.config import FreshnessConfig
    from sentinel.media.health import assess_frame
    from sentinel.scene.chat import ChatAdmission, ChatService
    from sentinel.status_page import StatusServer

    monkeypatch.setattr(mods.load, "GONE_S", 0.5)
    incident_store(tmp_path / "sentinel.db", count=1)
    camera = Camera()
    analyzer = ThreadedSceneAnalyzer(lambda job, image: None, RecentImages(), revision="test")
    analyzer.start()
    started = camera.clock.mono()
    service = ChatService(enabled=True, admission=ChatAdmission(True, "admitted for this test", "a fixture"),
                          scene=analyzer, latest=camera.frames.latest,
                          assess=lambda f: assess_frame(f, camera.stamper.current_stream, camera.clock.mono(), started,
                                                        FreshnessConfig()),
                          clock=camera.clock, run=lambda q, image, abort=None: ("A secret-free answer.", False),
                          stale_after_s=2.0)
    server = StatusServer(0, lambda: {"runtime": {"state": "running", "camera_id": "cam-1",
                                                  "updated_utc": NOW.isoformat()}, "degraded": [], "components": {}},
                          tmp_path / "sentinel.db", preview=lambda: camera.preview, chat=lambda: service)
    server.start()
    port = server.address[1]
    stop = threading.Event()

    def video() -> None:  # the fake clock runs ten times faster than the wall: one question per 0.4 s of wall time
        while not stop.wait(0.05):
            camera.frame(0.5)

    threading.Thread(target=video, daemon=True).start()
    exits: dict[str, int] = {}
    clients = [threading.Thread(target=lambda: exits.update(chat=mods.load.main(
                   ["chat", str(tmp_path / "chat-load.jsonl"), "--port", str(port), "--every-s", "0.5",
                    "--wait-s", "5"]))),
               threading.Thread(target=lambda: exits.update(view=mods.load.main(
                   ["view", str(tmp_path / "viewer.jsonl"), "--port", str(port), "--wait-s", "5"])))]
    for client in clients:
        client.start()
    time.sleep(3.0)
    token = server.chat_token
    assert server.stop(3.0)
    for client in clients:
        client.join(10)
    stop.set()
    camera.preview.close(2.0)
    assert analyzer.stop(3.0)
    assert exits == {"chat": 0, "view": 0}
    chat_text, view_text = (tmp_path / "chat-load.jsonl").read_text(), (tmp_path / "viewer.jsonl").read_text()
    assert "secret-free" not in chat_text and token not in chat_text and token not in view_text
    chat_lines = [json.loads(line) for line in chat_text.splitlines()]
    assert chat_lines[0]["event"] == "started" and chat_lines[-1]["event"] == "page_gone"
    answered = [x for x in chat_lines if x.get("state") == "answered"]
    assert answered and all(isinstance(x["stream_epoch"], int) and isinstance(x["frame_seq"], int)
                            and x["answer_chars"] == len("A secret-free answer.") and x["reason"] is None
                            for x in answered)
    assert all(x["state"] in mods.load.FINAL for x in chat_lines if "state" in x)
    view_lines = [json.loads(line) for line in view_text.splitlines()]
    assert view_lines[0]["event"] == "started" and view_lines[-1]["event"] == "page_gone"
    seconds = [x for x in view_lines if "pictures" in x]
    assert sum(x["pictures"] for x in seconds) >= 3
    assert all(isinstance(x["max_age_ms_when_sent"], int) for x in seconds if x["pictures"])
    for name in ("chat-load.jsonl", "viewer.jsonl"):
        assert mods.evidence.read(tmp_path / name, "f2").nonconforming == []
