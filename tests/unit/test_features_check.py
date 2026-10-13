"""The feature-controls check's readings (features_check.py, D99 with D97's continuity rule).

Both sessions' evidence comes from the real runtime, as in test_face_check: DemoRuntime with its feature controls runs
the package's nine minutes on a fake clock while the operator's switches are requested through FeatureControls (as the
page's POST does), the control lines and identity records come from the runtime's own drains, the captures from the
status page's build_status() at the package's times, the startup's face fields from assemble() (with the continuity
rule) and the guard summary from step5_guard.read_run_output(). Session face uses a real FaceWorker with synthetic
embeddings; session scene a real ThreadedSceneAnalyzer, ChatService, snapshot store and a whole-view zone, with only the
model's answers faked. The privacy counts come from face_counts.sh, as the package's blocks run them.
"""

from __future__ import annotations

import copy
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from face_runs import (
    BOX,
    FACE,
    FACE_BOX,
    GALLERY,
    GATE_LINES,
    NOW,
    REPORT,
    RUNNERS,
    Capture,
    F2_PROVENANCE,
    Image,
    Tracker,
    bash_counts,
    real_face_startup,
    segment,
    startup_line,
)

from sentinel import features
from sentinel.cli import status_line
from sentinel.config import IdentityConfig, parse_config
from sentinel.contracts import PixelFormat
from sentinel.demo_runtime import DemoRuntime, FaceRuntime, SceneRuntime
from sentinel.identity.association import FaceObservation
from sentinel.identity.legacy_deepface import FaceRun
from sentinel.identity.worker import FaceWorker
from sentinel.incidents.service import IncidentService
from sentinel.incidents.snapshots import SnapshotStore, SnapshotWriter
from sentinel.jobs import WorkerOutcome
from sentinel.live_state import Capability
from sentinel.media.capture import CapturedFrame, LatestFrame
from sentinel.media.clock import FakeClock
from sentinel.media.jpeg import EncodedJpeg
from sentinel.runtime import break_record
from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer
from sentinel.scene.chat import CHAT_ADMISSION, OPERATOR_OFF, ChatQuestion
from sentinel.status_page import build_status, read_store
from sentinel.storage.database import Database
from sentinel.tracking.tracker import PersonTracker, RawTrack

READY_AFTER_S = 70.0  # the guard's ready_after_s in these fixtures
FPS = 4.0
STATUS_ONLY = [{"family": "tcp", "port": 18090, "scope": "loopback"}]
WITH_SCENE = [{"family": "tcp", "port": 18081, "scope": "loopback"}, *STATUS_ONLY]
# Session face: seated facing, turned away seated (no face) from 1:30, facing again, out of view 3:00-4:00, back as a
# new track; the operator turns face recognition off at 5:02 and on at 5:47, the live view off at 6:32 and on at 7:17.
FACE_SCRIPT = ((0, 90, "facing", 1), (90, 150, "away", 1), (150, 180, "facing", 1), (180, 240, "absent", None),
               (240, 541, "facing", 2))
FACE_CHANGES = ((302.0, "face", "off"), (347.0, "face", "on"), (392.0, "live_view", "off"),
                (437.0, "live_view", "on"))
# Session scene: seated in the whole-view zone throughout; questions at 1:05, 2:40 (scene analysis off) and 4:25
# (chat off, refused); scene analysis off 2:02-3:32, chat off 4:17-5:17, snapshots off 5:47-6:32.
SCENE_CHANGES = ((122.0, "scene", "off"), (212.0, "scene", "on"), (257.0, "chat", "off"), (317.0, "chat", "on"),
                 (347.0, "snapshots", "off"), (392.0, "snapshots", "on"))
SCENE_QUESTIONS = (65.0, 160.0, 265.0)
ZONE = {"zone_id": "whole-view", "rule": "restricted", "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "min_duration_s": 1.0, "severity": "warning"}


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return SimpleNamespace(features=importlib.import_module("features_check"),
                           check=importlib.import_module("face_check"), guard=importlib.import_module("step5_guard"),
                           evidence=importlib.import_module("face_evidence"))


@pytest.fixture
def f1_list(tmp_path) -> Path:
    f1 = tmp_path / "f1"
    f1.mkdir()
    (f1 / "list.json").write_text(json.dumps({"identity": "list", "identities": [
        {"identity_id": "idn-00000000000a", "prototypes": 3, "consent_date": "2026-10-08",
         "enrolled_utc": NOW.isoformat()}]}))
    return f1


class ReadyServer:
    """The scene server's handle as the runtime reads it (LlamaServerProcess's status() and stop())."""

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(state=SimpleNamespace(value="ready"), problem=None, layers="17/17", vision_on_gpu=True)

    def stop(self, grace_s: float) -> bool:
        return True


def encode(*args: Any, **kwargs: Any) -> EncodedJpeg:
    return EncodedJpeg(b"\xff\xd8p\xff\xd9", 8, 8)


def wait(predicate, what: str) -> None:
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, what
        time.sleep(0.0005)


def run_session(root: Path, session: str, startup: dict, *, changes, captures, script=FACE_SCRIPT,
                questions=SCENE_QUESTIONS, duration: float = 540.0) -> Path:
    """The real runtime through the package's nine minutes, the operator's switches requested as the page's POST
    requests them, written as f-L leaves the run directory (before the counts)."""
    out = root / "fl"
    out.mkdir()
    clock = FakeClock(utc=NOW)
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"},
                           **({"zones": [ZONE]} if session == "scene" else {})})
    slot, capture, tracker = LatestFrame(), Capture(clock), Tracker()
    t0 = clock.monotonic_ns()
    db = Database.open(root / "sentinel.db")
    worker = analyzer = store = None
    extra: dict[str, Any] = {}
    if session == "face":
        def analyze(frame, image):
            state, _ = segment(script, (frame.ingest_mono_ns - t0) / 1e9)
            if state != "facing":
                return FaceRun((), 0, 1)  # turned away or absent: no face
            return FaceRun((FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.9, embedding=FACE),), 1, 0)

        worker = FaceWorker(analyze, clock)
        extra = dict(face=FaceRuntime(worker, 1, True, continuity=config.identity.continuity_policy()),
                     enrollment=GALLERY.enrollment())
    else:
        images = RecentImages()
        analyzer = ThreadedSceneAnalyzer(lambda job, image: WorkerOutcome.completed(job.job_id, REPORT), images,
                                         revision="1+lfm2-vl-1.6b-q4_0")
        store = SnapshotStore(db, root / "snapshots", clock, encode=lambda *a, **k: EncodedJpeg(b"\xff\xd8s\xff\xd9", 8, 8),
                              disk_usage=lambda path: SimpleNamespace(total=10**11, free=5 * 10**10))
        opened = store.open()
        extra = dict(scene=SceneRuntime(analyzer, images, ReadyServer()), snapshots=SnapshotWriter(store, clock),
                     chat=True, chat_admission=CHAT_ADMISSION, chat_validation=True,
                     chat_run=lambda question, image, abort=None: ("A chair by the window.", False),
                     pressure=lambda: 2_500_000_000)
    runtime = DemoRuntime(config, clock, capture=capture, slot=slot,
                          incidents=IncidentService(db, clock, incidents=config.incidents,
                                                    notifications=config.notifications),
                          tracker=PersonTracker(tracker), status_refresh_s=0.0, live_preview=True, encode=encode,
                          feature_controls=True, **extra)
    runtime.start()
    first = {**startup["startup"], "live_preview": runtime.preview.bounds(),
             "feature_controls": {name: runtime.controls.switchable(name) for name in features.FEATURES}}
    if session == "scene":
        first |= {"snapshots": {**opened, "policy": store.policy.describe()}, "chat": runtime.chat.availability()}
    lines = [json.dumps({**startup, "startup": first})]
    pending, todo, taken, jobs = list(captures), sorted(changes), [], []
    asks = list(questions) if session == "scene" else []  # chat is never part of session face
    next_status = 0.0

    def idle() -> None:
        if worker is not None:
            wait(lambda: not (worker.status()["busy"] or worker.status()["waiting"]), "the face worker")
        if analyzer is not None:
            wait(lambda: not analyzer.busy(), "the scene worker")

    for n in range(1, int(duration * FPS) + 1):
        t = n / FPS
        while todo and t >= todo[0][0]:  # the page's POST: accepted now, applied by the loop at its next step
            _, feature, state = todo.pop(0)
            assert runtime.controls.request(feature, state)[0] == 202
        state, track = segment(script, t) if session == "face" else ("facing", 1)
        tracker.next = [] if state == "absent" else [RawTrack(track, *BOX, 0.9)]
        clock.advance(ns=t0 + round(n * 1e9 / FPS) - clock.monotonic_ns())
        frame = capture.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        slot.publish(CapturedFrame(frame, Image()))
        runtime.step(timeout_s=0)
        idle()
        runtime.step(timeout_s=0)
        idle()
        lines += [json.dumps(record) for record in runtime.drain_identity_records()]
        lines += [json.dumps(record) for record in runtime.drain_control_records()]
        if t >= next_status:
            lines.append(json.dumps(status_line(runtime.snapshot())))
            next_status += 30.0
        if n % int(FPS * 5) == 0:  # an open dashboard: a live picture asked for every 5 s
            runtime.preview.view(with_picture=True)
        while asks and t >= asks[0]:
            asks.pop(0)
            code, body = runtime.chat.ask(ChatQuestion("What is visible?"))
            jobs.append((code, body))
        while pending and t >= pending[0][0]:
            at, name = pending.pop(0)
            status = build_status(runtime.snapshot(), read_store(root / "sentinel.db"), clock.utc_now(),
                                  runtime.preview.view())
            (out / f"status-{name}.json").write_text(json.dumps(status))
            taken.append({"name": name, "at_s": at, "taken_after_launch_s": READY_AFTER_S + t, "status": "saved",
                          "bytes": 1, "listeners": STATUS_ONLY if session == "face" else WITH_SCENE})
    shutdown = runtime.shutdown()
    shutdown["stopped"]["status_page"] = True  # what the CLI adds after the runtime's own shutdown
    shutdown["stopped"]["control_key_removed"] = True
    shutdown["all_stopped"] = all(shutdown["stopped"].values())
    lines += [json.dumps(record) for record in runtime.drain_identity_records()]
    lines += [json.dumps(record) for record in runtime.drain_control_records()]
    lines.append(json.dumps({"run": "stopped", "shutdown": shutdown, "database_closed": True,
                             "status": status_line(runtime.snapshot())}))
    db.close()
    (out / "run.jsonl").write_text("\n".join(lines) + "\n")
    (root / "captures.json").write_text(json.dumps(taken))  # outside the run directory: the guard keeps it in result
    (root / "jobs.json").write_text(json.dumps(jobs))
    return out


def finish(mods, out: Path, session: str, *, cue_delay_s: float = 0.1, counts: bool = True,
           cues: dict[str, float] | None = None) -> None:
    """result.json as the guard writes it, the launch block's provenance, then the counts as f-E makes them."""
    ready = 1.0  # FakeClock's start (seconds): the runtime's T0
    plan = mods.features.PLAN[session][0] if cues is None else cues
    result = {
        "schema_version": 1, "mode": "step5_guard", "status": "duration_stop",
        "parameters": {"duration_s": 540.0, "cues": [{"at_s": at, "name": name} for name, at in plan.items()]},
        "child": {"returncode": 0, "ready_after_s": READY_AFTER_S, "ready_mono_s": ready},
        "guard": {"trigger": None, "peak_pressure_bytes": 1, "min_mem_free_bytes": 2, "min_mem_available_bytes": 3},
        "stop": {"forced": False}, "cleanup_clear": True,
        "leftovers": {"llama_server_after_exit": 0, "llama_server_left": 0, "runtime_group_left": False,
                      "listeners_after": []},
        "captures": json.loads((out.parent / "captures.json").read_text()),
        "cues": [{"name": name, "at_s": float(at), "printed_after_launch_s": READY_AFTER_S + at + cue_delay_s,
                  "printed_mono_s": ready + at + cue_delay_s} for name, at in plan.items()],
        "run_output": mods.guard.read_run_output(out / "run.jsonl"),
    }
    (out / "result.json").write_text(json.dumps(result))
    (out / "provenance.txt").write_text(F2_PROVENANCE + (GATE_LINES if session == "face" else ""))
    if not counts:
        return
    for function, name in (("face_counts_camera", "secret-counts-camera.txt"),
                           *((("face_counts_f2", "secret-counts-identity.txt"),) if session == "face" else ())):
        done = bash_counts(function, str(out))
        assert done.returncode == 0, done.stderr
        (out / name).write_text(done.stdout)


def face_startup(root: Path, accepted_scene) -> dict:
    """The starting line of a face-only --face-continuity run: no scene server, so no scene release or llama_ready
    checkpoint (as the sealed dashboard run's startup shows)."""
    (root / "startup").mkdir(parents=True)
    startup = startup_line(real_face_startup(root / "startup", accepted_scene, continuity=True))
    first = startup["startup"]
    del first["scene_server"], first["releases"]["scene"]
    first["thp_scope"] = {"before_launch": {"runtime": 1}, "workload_verified": {"runtime": 0}}
    return startup


def scene_startup() -> dict:
    """The starting line of a scene run's assemble() (scene server, detector, no face)."""
    release = {"settle_s": 15.0, "files": {}}
    return {"run": "starting", "startup": {
        "memory_policy": {"thp": "workload_disabled", "model_file_release": "post_load"},
        "thp_scope": {"before_launch": {"runtime": 1}, "llama_ready": {"runtime": 1, "scene_server": 1},
                      "workload_verified": {"runtime": 0, "scene_server": 1}},
        "thp_disable": {"verified": True, "reason": None, "set_rc": 0, "thp_enabled": 0},
        "releases": {"scene": {**release, "files": {"llm": {"result": "returned_0"}, "mmproj": {"result": "returned_0"}}},
                     "detector": {**release, "files": {"engine": {"result": "returned_0"}}}},
        "scene_server": {"state": "ready", "problem": None, "layers": "17/17", "vision_on_gpu": True},
        "scene_problem": None, "detector_problem": None, "notifier_problems": {},
        "status_page": "http://127.0.0.1:18090/"}}


def face_run(mods, tmp_path, accepted_scene, *, name: str = "face", changes=FACE_CHANGES, script=FACE_SCRIPT,
             counts: bool = True) -> Path:
    root = tmp_path / name
    out = run_session(root, "face", face_startup(root, accepted_scene), changes=changes, script=script,
                      captures=mods.features.FACE_CAPTURES)
    finish(mods, out, "face", counts=counts)
    return out


def scene_run(mods, tmp_path, *, name: str = "scene", changes=SCENE_CHANGES, questions=SCENE_QUESTIONS,
              counts: bool = True) -> Path:
    root = tmp_path / name
    root.mkdir()
    out = run_session(root, "scene", scene_startup(), changes=changes, captures=mods.features.SCENE_CAPTURES,
                      questions=questions)
    finish(mods, out, "scene", counts=counts)
    return out


def read(mods, out: Path, session: str, f1: Path | None = None):
    reading = mods.features.read_face(out, f1, None) if session == "face" else mods.features.read_scene(out)
    return reading, "\n".join(reading.lines)


def line_for(text: str, start: str) -> str:
    return next(line for line in text.splitlines() if line.split(" ", 1)[1].startswith(start))


def value_of(line: str) -> Any:
    """The JSON value at the end of a reading's line (item names may contain ': ' themselves)."""
    at = line.find(": ")
    while at >= 0:
        try:
            return json.loads(line[at + 2:])
        except ValueError:
            at = line.find(": ", at + 1)
    raise AssertionError(f"no value in {line!r}")


def plain(text: str, start: str) -> Any:
    return value_of(next(line for line in text.splitlines() if line.startswith(f"DESCRIPTIVE {start}")))


def edit_jsonl(path: Path, change) -> None:
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("\n".join(json.dumps(item) for item in change(lines)) + "\n")


def edit_json(path: Path, change) -> None:
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc))


# ---------------------------------------------------------------- session face


def test_the_face_session_is_read_from_the_runtimes_own_output(mods, tmp_path, accepted_scene, f1_list) -> None:
    out = face_run(mods, tmp_path, accepted_scene)
    for path in sorted(out.iterdir()):  # every evidence file fits its profile: the counts read it strictly
        if path.is_file() and path.name not in mods.check.NOT_COUNTED:
            assert mods.evidence.read(path, "f2").nonconforming == [], path.name
    reading, text = read(mods, out, "face", f1_list)
    assert reading.ok, text
    assert not [line for line in reading.lines if line.startswith(("FAIL", "MISSING", "NOT_EXERCISED"))], text
    assert line_for(text, "while face recognition was off").startswith("PASS")
    off = value_of(line_for(text, "while face recognition was off"))
    assert off["known_when_turned_off"][0]["ok"] and off["known_when_turned_off"][0]["track"] == [1, 2]
    on = value_of(line_for(text, "after face recognition was turned on again"))
    assert [f["track"] for f in on] == [[1, 2]] and all(v[2] >= f["turned_on_s"] for f in on for v in f["votes"])
    placed = plain(text, "each capture: when the guard took it")
    assert placed["faceoff1"]["off"] == ["face"] and placed["liveoff2"]["off"] == ["live_view"]
    assert placed["seated"]["after_lines"] == 0 and placed["end"]["off"] == []
    segments = plain(text, "per cue interval")
    assert segments["turned away, seated"]["known_s"] > 50  # retained while followed (D97), not cut at 30 s
    assert segments["face recognition turned off"]["entries_into_known"] == 0
    assert plain(text, "the longest stretch KNOWN with basis retained") >= 55.0
    rows = plain(text, "each capture's recognition rows")
    assert rows["faceoff1"] is None and rows["turned"][0][:3] == [1, "known", "retained"] and rows["turned"][0][4] > 0
    changes = plain(text, "the changes the loop recorded")
    assert [c[1:4] for c in changes] == [["face", "off", "applied"], ["face", "on", "applied"],
                                        ["live_view", "off", "applied"], ["live_view", "on", "applied"]]
    assert mods.features.main(["face", str(out), "--f1", str(f1_list), "--run", str(tmp_path / "missing"),
                               "--identity-dir", str(tmp_path)]) == 1  # the F1 gate needs the real run
    assert "similarity" not in text and "idn-" not in text


def test_a_switch_not_made_or_made_too_late_is_not_exercised(mods, tmp_path, accepted_scene, f1_list) -> None:
    out = face_run(mods, tmp_path, accepted_scene, changes=FACE_CHANGES[2:], counts=False)  # face never turned off
    reading, text = read(mods, out, "face", f1_list)
    assert not reading.ok
    for start in ("while face recognition was off", "every capture while face recognition was off",
                  "after face recognition was turned on again"):
        assert line_for(text, start).startswith("NOT_EXERCISED"), start
    assert line_for(text, "every capture while the live view was off").startswith("PASS")
    late = ((340.0, "face", "off"), (347.0, "face", "on"), *FACE_CHANGES[2:])  # after both captures, for 7 s
    out = face_run(mods, tmp_path, accepted_scene, name="late", changes=late, counts=False)
    reading, text = read(mods, out, "face", f1_list)
    assert line_for(text, "while face recognition was off").startswith("NOT_EXERCISED")  # under 20 s
    assert line_for(text, "every capture while face recognition was off").startswith("NOT_EXERCISED")
    assert not reading.ok


@pytest.fixture
def face_out(mods, tmp_path, accepted_scene) -> Path:
    return face_run(mods, tmp_path, accepted_scene, counts=False)


def _first(lines: list[dict], kind: str, after_s: float, **match: Any) -> dict:
    t0 = 1.1e9  # the reading's T0 (the t0 cue, 0.1 s after the runtime's start)
    return next(x for x in lines if x.get("identity") == kind and (x.get("frame_mono_ns", x.get("mono_ns", 0)) - t0)
                / 1e9 > after_s and all(x.get(k) == v for k, v in match.items()))


def test_work_done_while_face_recognition_was_off_fails(mods, face_out, f1_list) -> None:
    name = "while face recognition was off"

    def offered(lines):  # a tick offered while it was off
        tick = copy.deepcopy(_first(lines, "tick", 280.0, tick="offered"))
        tick["frame_mono_ns"] = round(1.1e9 + 320e9)
        return [*lines, tick]

    def applied(lines):  # a face result applied while it was off
        result = copy.deepcopy(_first(lines, "result", 280.0, outcome="applied"))
        result["applied_mono_ns"] = round(1.1e9 + 320e9)
        return [*lines, result]

    def reached(lines):  # an identity reached while it was off
        transition = copy.deepcopy(_first(lines, "transition", 340.0, to="known"))
        transition["mono_ns"] = round(1.1e9 + 320e9)
        return [*lines, transition]

    for change in (offered, applied, reached):
        out = face_out.parent / change.__name__
        out.mkdir()
        for path in face_out.iterdir():
            (out / path.name).write_bytes(path.read_bytes())
        edit_jsonl(out / "run.jsonl", change)
        reading, text = read(mods, out, "face", f1_list)
        assert line_for(text, name).startswith("FAIL"), (change.__name__, line_for(text, name))


def test_an_identity_held_across_the_switch_or_shown_while_off_fails(mods, face_out, f1_list) -> None:
    def carried(lines):  # KNOWN again at once on the votes from before it was turned off
        on = next(x for x in lines if x.get("control") and x["feature"] == "face" and x["requested"] == "on")
        known = _first(lines, "transition", (on["mono_ns"] - 1.1e9) / 1e9, to="known")
        known["mono_ns"] = on["mono_ns"] + 1_000_000
        return lines

    edit_jsonl(face_out / "run.jsonl", carried)
    reading, text = read(mods, face_out, "face", f1_list)
    assert line_for(text, "after face recognition was turned on again").startswith("FAIL")
    edit_json(face_out / "status-faceoff2.json", lambda doc: doc["live"].update(faces=[]))
    reading, text = read(mods, face_out, "face", f1_list)
    assert line_for(text, "every capture while face recognition was off").startswith("FAIL")


def test_the_live_view_encoding_while_off_or_not_back_fails(mods, face_out, f1_list) -> None:
    edit_json(face_out / "status-liveoff2.json",
              lambda doc: doc["components"]["preview"]["counters"].update(encodes=999))
    reading, text = read(mods, face_out, "face", f1_list)
    assert line_for(text, "no picture encoded while the live view was off").startswith("FAIL")
    assert line_for(text, "every capture while the live view was off").startswith("PASS")
    edit_json(face_out / "status-liveon.json", lambda doc: doc["preview"].update(state="paused"))
    reading, text = read(mods, face_out, "face", f1_list)
    assert line_for(text, "after each feature was turned on again").startswith("FAIL")


def test_a_feature_not_running_before_its_first_switch_fails_unless_its_change_was_waiting(mods, face_out,
                                                                                        f1_list) -> None:
    name = "before each feature was first turned off"
    edit_json(face_out / "status-returned.json", lambda doc: doc["components"]["face"].update(paused=True))
    reading, text = read(mods, face_out, "face", f1_list)
    assert line_for(text, name).startswith("FAIL")
    # the page's request accepted there but not yet applied by the loop: that capture is left out, never judged
    edit_json(face_out / "status-returned.json", lambda doc: doc["features"]["log"].append(
        {"change": 1, "at_utc": NOW.isoformat(), "feature": "face", "requested": "off", "result": "accepted"}))
    reading, text = read(mods, face_out, "face", f1_list)
    assert line_for(text, name).startswith("PASS")
    assert "returned" not in [n for n, _ in value_of(line_for(text, name))["face"]]
    assert "returned" in [n for n, _ in value_of(line_for(text, name))["live_view"]]


def test_captures_are_placed_by_their_own_log_never_by_their_clock(mods, face_out, f1_list) -> None:
    result = json.loads((face_out / "result.json").read_text())  # the guard's clock says faceoff1 came first
    for capture in result["captures"]:
        if capture["name"] == "faceoff1":
            capture["taken_after_launch_s"] = READY_AFTER_S + 290.0
    (face_out / "result.json").write_text(json.dumps(result))
    reading, text = read(mods, face_out, "face", f1_list)
    assert plain(text, "each capture: when the guard took it")["faceoff1"]["off"] == ["face"]
    assert line_for(text, "every capture while face recognition was off").startswith("PASS")
    assert line_for(text, "control lines:").startswith("PASS")
    # a log that does not match the control lines (the first applied change missing, later ones there) is a failure
    edit_json(face_out / "status-liveoff1.json", lambda doc: doc["features"].update(
        log=[e for e in doc["features"]["log"] if not (e["change"] == 1 and e["result"] == "applied")]))
    reading, text = read(mods, face_out, "face", f1_list)
    assert line_for(text, "control lines:").startswith("FAIL") and "liveoff1" in line_for(text, "control lines:")
    # a full log cannot be placed: that capture is left out, never guessed
    edit_json(face_out / "status-faceoff2.json", lambda doc: doc["features"].update(
        log=[{"change": 99, "at_utc": NOW.isoformat(), "feature": "face", "requested": "off", "result": "replaced"}]
        * mods.features.LOG_KEPT))
    reading, text = read(mods, face_out, "face", f1_list)
    assert plain(text, "each capture: when the guard took it")["faceoff2"]["after_lines"] is None
    assert line_for(text, "every capture while face recognition was off").startswith("PASS")  # faceoff1 alone


def test_execution_problems_fail_the_face_session(mods, face_out, f1_list) -> None:
    edit_jsonl(face_out / "run.jsonl", lambda lines: [
        {**x, "result": "refused"} if x.get("control") and x["requested"] == "on" and x["feature"] == "face" else x
        for x in lines])
    reading, text = read(mods, face_out, "face", f1_list)
    assert line_for(text, "control lines:").startswith("FAIL")

    def unswitchable(lines):
        lines[0]["startup"]["feature_controls"]["scene"] = True
        return lines

    edit_jsonl(face_out / "run.jsonl", unswitchable)
    reading, text = read(mods, face_out, "face", f1_list)
    assert line_for(text, "feature controls on, with exactly").startswith("FAIL")
    edit_json(face_out / "result.json",
              lambda doc: doc["run_output"]["stopped"]["stopped"].update(control_key_removed=False))
    reading, text = read(mods, face_out, "face", f1_list)
    assert line_for(text, "clean shutdown:").startswith("FAIL")
    edit_json(face_out / "result.json", lambda doc: doc["run_output"]["starting"]["face"].pop("continuity"))
    reading, text = read(mods, face_out, "face", f1_list)
    assert line_for(text, "face admitted for this validation run").startswith("FAIL")


def test_the_face_sessions_privacy_counts_are_judged(mods, tmp_path, accepted_scene, f1_list) -> None:
    out = face_run(mods, tmp_path, accepted_scene, counts=False)
    (out / "status-end.json").write_text((out / "status-end.json").read_text().replace(
        '"camera_id": "cam-1"', '"camera_id": "cam-1", "note": "Jane"'))
    finish(mods, out, "face")
    reading, text = read(mods, out, "face", f1_list)
    assert line_for(text, "passphrase, name and nonconforming counts").startswith("FAIL")
    assert line_for(text, "camera userinfo counts").startswith("PASS")


# ---------------------------------------------------------------- session scene


def test_the_scene_session_is_read_from_the_runtimes_own_output(mods, tmp_path) -> None:
    out = scene_run(mods, tmp_path)
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name not in mods.check.NOT_COUNTED:
            assert mods.evidence.read(path, "f2").nonconforming == [], path.name
    jobs = json.loads((out.parent / "jobs.json").read_text())
    assert [code for code, _ in jobs] == [202, 202, 503]  # the third, while chat was off, refused
    reading, text = read(mods, out, "scene")
    assert reading.ok, text
    assert not [line for line in reading.lines if line.startswith(("FAIL", "MISSING", "NOT_EXERCISED"))], text
    placed = plain(text, "each capture: when the guard took it")
    assert placed["sceneoff1"]["off"] == ["scene"] and placed["chatoff2"]["off"] == ["chat"]
    assert placed["snapoff"]["off"] == ["snapshots"] and placed["end"]["off"] == []
    chat = plain(text, "chat's counts at each capture")
    assert chat["sceneoff2"]["answered"] == chat["sceneoff1"]["answered"] + 1  # answered with scene analysis off
    assert chat["chaton"]["refused"] == 1
    periodic = plain(text, "periodic scene checks submitted")
    assert periodic["sceneoff1"] == periodic["sceneoff2"] < periodic["sceneon"]
    assert periodic["chatoff1"] < periodic["chatoff2"]
    store = plain(text, "the snapshot store's counters")
    assert store["saved"] == 1  # the first incident's; nothing pruned
    assert mods.features.main(["scene", str(out)]) == 0
    assert "What is visible" not in text and "chair" not in text  # no question or answer is ever shown


@pytest.fixture
def scene_out(mods, tmp_path) -> Path:
    return scene_run(mods, tmp_path, counts=False)


def test_scene_analysis_working_while_off_or_chat_failing_fails(mods, scene_out) -> None:
    edit_json(scene_out / "status-sceneoff2.json",
              lambda doc: doc["components"]["scene"]["lane"].update(submitted_periodic=999))
    reading, text = read(mods, scene_out, "scene")
    assert line_for(text, "no periodic scene check submitted while scene analysis was off").startswith("FAIL")
    edit_json(scene_out / "status-sceneoff2.json", lambda doc: doc["components"]["chat"]["counters"].update(
        answered=doc["components"]["chat"]["counters"]["answered"] - 1, timeouts=1))
    reading, text = read(mods, scene_out, "scene")
    assert line_for(text, "while scene analysis was off (chat on)").startswith("FAIL")
    edit_json(scene_out / "status-sceneoff1.json",
              lambda doc: doc["components"]["scene"]["server"].update(state="stopped"))
    reading, text = read(mods, scene_out, "scene")
    assert line_for(text, "every capture while scene analysis was off").startswith("FAIL")


def test_no_question_while_scene_analysis_was_off_is_not_exercised(mods, tmp_path) -> None:
    out = scene_run(mods, tmp_path, questions=(65.0, 265.0), counts=False)
    reading, text = read(mods, out, "scene")
    assert line_for(text, "while scene analysis was off (chat on)").startswith("NOT_EXERCISED")
    assert not reading.ok


def test_chat_off_stopping_scene_analysis_or_snapshots_deleted_fails(mods, scene_out) -> None:
    first = json.loads((scene_out / "status-chatoff1.json").read_text())["components"]["scene"]["lane"]
    edit_json(scene_out / "status-chatoff2.json", lambda doc: doc["components"]["scene"]["lane"].update(
        submitted_periodic=first["submitted_periodic"]))
    reading, text = read(mods, scene_out, "scene")
    assert line_for(text, "periodic scene checks still submitted while chat was off").startswith("FAIL")
    edit_json(scene_out / "status-chatoff1.json",
              lambda doc: doc["components"]["chat"].update(state="available", reason="monitoring comes first"))
    reading, text = read(mods, scene_out, "scene")
    assert line_for(text, "every capture while chat was off").startswith("FAIL")
    edit_json(scene_out / "status-end.json",
              lambda doc: doc["components"]["snapshots"]["store"].update({"pruned:retention_age": 1}))
    reading, text = read(mods, scene_out, "scene")
    assert line_for(text, "no incident snapshot deleted over the run").startswith("FAIL")


def test_the_scene_session_needs_its_own_startup_and_counts(mods, tmp_path) -> None:
    out = scene_run(mods, tmp_path, counts=False)
    finish(mods, out, "scene")  # the camera counts only: this session opens no gallery
    assert not (out / "secret-counts-identity.txt").exists()
    reading, text = read(mods, out, "scene")
    assert reading.ok, text
    edit_json(out / "result.json", lambda doc: doc["run_output"]["starting"].update(face={"validation_run": True}))
    reading, text = read(mods, out, "scene")
    assert line_for(text, "scene server ready").startswith("FAIL")
    (out / "secret-counts-camera.txt").write_text((out / "secret-counts-camera.txt").read_text().replace(
        "userinfo=0", "userinfo=1", 1))
    reading, text = read(mods, out, "scene")
    assert line_for(text, "camera userinfo counts").startswith("FAIL")


# ---------------------------------------------------------------- the reading's constants are the runtime's


def test_the_readings_constants_match_the_runtime(mods) -> None:
    f = mods.features
    assert f.LOG_KEPT == features.LOG_KEPT and f.FEATURES == features.FEATURES
    assert set(f.LOOP_RESULTS) == {features.APPLIED, features.UNCHANGED, features.REFUSED}
    assert f.CHAT_OFF_REASON == OPERATOR_OFF
    assert f.FACE_OFF_REASON == f"face recognition {Capability.DISABLED.value}"
    assert f.CONTINUITY == IdentityConfig().continuity_policy().numbers()
    from sentinel.contracts import TrackKey
    from sentinel.identity.continuity import Break, Cause
    key = TrackKey(camera_id="cam-1", boot_id="boot-1", run_id="run-1", stream_epoch=1, track_id=2)
    assert f.BREAK_KEYS == set(break_record(Break(key, 5, Cause.GAP), 3))
    clock = FakeClock()
    controls = features.FeatureControls(clock, {name: True for name in features.FEATURES})
    controls.request("face", "off")
    (change,) = controls.take()
    controls.done(change, features.APPLIED)
    (record,) = controls.drain_records()
    assert set(record) == f.CONTROL_KEYS
    for cues, captures in f.PLAN.values():  # the guard's evidence names are declared for the privacy counts
        assert set(cues) <= set(mods.evidence.CUE_NAMES)
        assert {name for _, name in captures} <= set(mods.evidence.STATUS_NAMES)
        assert all(0 < at < f.DURATION_S for _, at in [(None, at) for at, _ in captures])
        assert sorted(cues.values()) == list(cues.values()) and max(cues.values()) < f.DURATION_S
