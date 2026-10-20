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
from pathlib import Path
from types import SimpleNamespace

import pytest
from chat_runs import CAPTURES, NOW, build_run

from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"


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
    assert '"after_grace": 1' in cancel  # the periodic job let the question run for its grace, then ended it
    delay = line_for(text, "chat-induced scene delay")
    assert delay.startswith("PASS") and '"grace_ms": 1500' in delay and '"max_ever_ms": 1500.0' in delay
    assert 'DESCRIPTIVE chat load: questions by kind and final state: {"long": {"answered": 50, "cancelled": 1}, ' \
           '"short": {"answered": 49}}' in text


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
    (lambda s: s["components"]["chat"]["scene_waits_behind_chat"].update(grace_ms=3000), "chat-induced scene delay",
     "FAIL"),  # a run with another grace than the declared one
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
    (tmp_path / "limited").mkdir()
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
               threading.Thread(target=lambda: exits.update(chat_for=mods.load.main(
                   ["chat", str(tmp_path / "limited" / "chat-load.jsonl"), "--port", str(port), "--every-s", "0.5",
                    "--wait-s", "5", "--for-s", "1.2"]))),
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
    assert exits == {"chat": 0, "chat_for": 0, "view": 0}
    limited = [json.loads(line) for line in (tmp_path / "limited" / "chat-load.jsonl").read_text().splitlines()]
    asked = [x["asked_mono_s"] for x in limited if "asked_mono_s" in x]
    # --for-s counts from the first ACCEPTED question (202, then followed to its end: 200), not the first asked one,
    # which the page may refuse (busy, or no fresh frame yet) depending on thread scheduling. Against the tool's own
    # recorded times (rounded to 1 ms): nothing is asked 1.2 s or more after that question, and it stops at or after it.
    first = next(x["asked_mono_s"] for x in limited if x.get("http_status") in (200, 202))
    assert limited[-1]["event"] == "stopped_asking" and limited[-1]["mono_s"] - first >= 1.2 - 0.001
    assert all(t - first < 1.2 + 0.001 for t in asked)
    assert mods.evidence.read(tmp_path / "limited" / "chat-load.jsonl", "f2").nonconforming == []
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


def test_the_load_tools_for_s_window_counts_from_the_first_accepted_question(mods, tmp_path, monkeypatch) -> None:
    """--for-s on a fake clock with a scripted page whose first question is refused: no wall time, no scheduling."""
    now = [0.0]
    monkeypatch.setattr(mods.load, "time", SimpleNamespace(monotonic=lambda: now[0],
                                                           sleep=lambda s: now.__setitem__(0, now[0] + s)))

    class ScriptedPage:
        def __init__(self, port: int) -> None:
            self.posts = 0

        def refresh_token(self) -> bool:
            return True

        def json(self, method: str, path: str, document: dict | None = None) -> tuple[int, dict]:
            if method == "GET":
                return 200, {"state": "answered", "answer": "An answer."}
            self.posts += 1
            if self.posts == 1:
                return 409, {"state": "no_fresh_frame", "reason": "no fresh frame yet"}
            if self.posts > 8:  # a tool that never stops would see the page go away, not stop by itself
                raise OSError("the page has gone")
            return 202, {"job": f"job-{self.posts}", "state": "sending"}

    monkeypatch.setattr(mods.load, "Page", ScriptedPage)
    monkeypatch.setattr(mods.load, "_wait_for", lambda page, wait_s: True)
    out = tmp_path / "chat-load.jsonl"
    assert mods.load.chat(out, 18090, every_s=0.5, wait_s=5.0, for_s=1.2) == 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    asked = [(x["asked_mono_s"], x["state"]) for x in lines if "asked_mono_s" in x]
    # refused at 0.0 (not counted); accepted at 0.5, so asking continues while less than 1.2 s has passed since then
    assert asked == [(0.0, "no_fresh_frame"), (0.5, "answered"), (1.0, "answered"), (1.5, "answered")]
    assert lines[-1] == {"event": "stopped_asking", "mono_s": 2.0}


def test_the_combined_reading_needs_an_incident_with_its_snapshot_and_reads_f2_with_its_gate(mods, made,
                                                                                             tmp_path) -> None:
    run = fresh_copy(made, tmp_path)
    face_run, identity = tmp_path / "face-run", tmp_path / "identity"
    reading, f2 = mods.check.read_combined(run, face_run, identity, None)
    text = text_of(reading)
    assert line_for(text, "the alpha flow ran alongside").startswith("MISSING")  # no zone, no snapshots: not shown
    assert not f2.ok and text_of(f2).splitlines()[0].startswith("FAIL F1 basis for this F2")  # the gate comes first
    edit_last_status(run, lambda s: (s["incidents"].update(recent=[{"incident_id": "inc-1"}]),
                                     s["components"].setdefault("snapshots", {}).setdefault("writer", {}).update(
                                         saved=1)))
    reading, _ = mods.check.read_combined(run, face_run, identity, None)
    assert line_for(text_of(reading), "the alpha flow ran alongside").startswith("PASS")
