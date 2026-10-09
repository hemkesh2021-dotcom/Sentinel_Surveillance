"""D74: the chat panel in a real browser, with a scripted model (no camera, no model).

Headless Chromium (``browser_harness``) loads the real StatusServer, with a real ThreadedSceneAnalyzer as the scene
worker and a ChatService admitted for the test only; the model is a function the test holds and releases. The page
is driven as a viewer would: typing into the question box, pressing Ask.

Covered: sending, waiting/answering and answered, with the frame the answer describes, its time and its age (marked
historical once that frame stopped being live); busy (the scene worker comes first), no fresh frame and unavailable,
each said, with the draft kept; the draft, the conversation and keyboard focus kept across page updates; an answer
never making a stale page current; a restarted runtime (a new token) asking for a reload; another origin unable to
ask; and the panel on a phone.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from browser_harness import Browser, DevToolsError, Page, headless_browser, wait_until
from status_fixtures import Camera, Runtime, incident_store

from sentinel.config import FreshnessConfig
from sentinel.jobs import AnalysisJob, JobPurpose, WorkerOutcome
from sentinel.media.clock import NS_PER_SECOND
from sentinel.media.health import assess_frame
from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer
from sentinel.scene.chat import CHAT_ADMISSION, CHAT_MIN_INTERVAL_S, ChatAdmission, ChatService
from sentinel.status_page import PAGE_CURRENT_S, REFRESH_S, StatusServer

ADMITTED = ChatAdmission(True, "admitted for this test", "a synthetic fixture")
PAGE_CURRENT = "This page is current"
PAGE_NOT_CURRENT = f"NOT CURRENT: this page has not refreshed for {PAGE_CURRENT_S} s"
CHAT = """(() => {
  const q = document.getElementById('chat-q');
  return {draft: q.value, focused: document.activeElement === q, ask_disabled: document.getElementById('chat-ask').disabled,
          note: document.getElementById('chat-note').textContent, text: document.body.innerText,
          sequence: Number(document.documentElement.dataset.sequence || '-1'), document: performance.timeOrigin,
          state: (document.querySelector('[data-chat-state]') || {dataset: {}}).dataset.chatState || null,
          log: [...document.querySelectorAll('#chat-log > li')].map((li) => ({state: li.dataset.state,
                text: li.innerText, kept: li.__testMark || null}))};
})()"""


class Model:
    """The scene model's stand-in: answers on the scene worker's thread when the test lets it."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.release.set()
        self.questions: list[str] = []

    def __call__(self, question: Any, image: Any) -> tuple[str, bool]:
        self.questions.append(question.question)
        self.release.wait(20.0)
        return "One chair stands by the window; nobody is in the picture.", False


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with headless_browser() as connection:
        yield connection


@pytest.fixture
def served(tmp_path: Path) -> Iterator[dict[str, Any]]:
    incident_store(tmp_path / "sentinel.db", count=2)
    runtime, camera, model = Runtime(), Camera(), Model()
    stop = camera.run()
    scene_hold = threading.Event()
    scene_hold.set()

    def scene_job(job: AnalysisJob, image: Any) -> WorkerOutcome:
        scene_hold.wait(20.0)
        return WorkerOutcome.completed(job.job_id, "{}")

    analyzer = ThreadedSceneAnalyzer(scene_job, RecentImages(), revision="test")
    analyzer.start()
    started = camera.clock.mono()

    def service(admission: ChatAdmission) -> ChatService:
        return ChatService(enabled=True, admission=admission, scene=analyzer, latest=camera.frames.latest,
                           assess=lambda f: assess_frame(f, camera.stamper.current_stream, camera.clock.mono(), started,
                                                         FreshnessConfig()),
                           clock=camera.clock, run=model, stale_after_s=2.0)

    chat = {"service": service(ADMITTED)}
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview,
                          chat=lambda: chat["service"])
    server.start()
    try:
        yield {"url": f"http://127.0.0.1:{server.address[1]}", "runtime": runtime, "camera": camera, "model": model,
               "stop": stop, "analyzer": analyzer, "scene_hold": scene_hold, "chat": chat, "service": service,
               "server": server, "db": tmp_path / "sentinel.db"}
    finally:
        stop.set()
        runtime.answering.set()
        model.release.set()
        scene_hold.set()
        assert server.stop(3.0) and analyzer.stop(5.0)


def read(page: Page) -> dict[str, Any]:
    return page.evaluate(CHAT)


def wait_chat(page: Page, predicate: Any, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    seen = None
    while time.monotonic() < deadline:
        with contextlib.suppress(DevToolsError):
            seen = read(page)
            if predicate(seen):
                return seen
        time.sleep(0.05)
    raise AssertionError(f"the chat never reached the expected state; last seen: {seen!r}")


def opened(browser: Browser, url: str, **options: Any) -> Page:
    page = browser.page(**options)
    page.navigate(url)
    wait_chat(page, lambda s: PAGE_CURRENT in s["text"] and s["state"] == "available" and s["sequence"] >= 0, 30.0)
    return page


def type_question(page: Page, text: str) -> None:
    page.evaluate("document.getElementById('chat-q').focus()")
    page.call("Input.insertText", {"text": text})


def press_ask(page: Page) -> None:
    page.evaluate("document.getElementById('chat-ask').focus()")
    page.key("Enter")


def last(seen: dict[str, Any]) -> dict[str, Any]:
    return seen["log"][-1]


def wait_for_interval(served: dict[str, Any]) -> None:
    """One question per CHAT_MIN_INTERVAL_S on the runtime's clock, which the running fake camera advances."""
    service, clock = served["chat"]["service"], served["camera"].clock
    assert wait_until(lambda: service._last_accepted_ns is None
                      or clock.monotonic_ns() - service._last_accepted_ns >= CHAT_MIN_INTERVAL_S * NS_PER_SECOND, 10.0)


def test_a_question_is_sent_answered_and_bound_to_the_frame_it_describes(browser: Browser, served: dict[str, Any]) -> None:
    page = opened(browser, f"{served['url']}/")
    model = served["model"]
    try:
        seen = read(page)
        assert seen["ask_disabled"] and seen["log"] == []  # nothing to send yet
        type_question(page, "Is anyone in the room?")
        assert not read(page)["ask_disabled"]
        model.release.clear()  # the model takes its time
        press_ask(page)
        seen = wait_chat(page, lambda s: s["log"] and last(s)["state"] == "answering", 5.0)
        assert "The scene model is answering…" in last(seen)["text"] and "Is anyone in the room?" in last(seen)["text"]
        assert seen["draft"] == "" and seen["ask_disabled"] and seen["note"] == "One question at a time."
        time.sleep(2.5)  # the frame stops being live while the model answers
        model.release.set()
        seen = wait_chat(page, lambda s: last(s)["state"] == "answered", 5.0)
        text = last(seen)["text"]
        assert "One chair stands by the window; nobody is in the picture." in text and "Answered." in text
        assert "Historical: about the frame received at " in text and " UTC (stream epoch 1, frame " in text
        assert " old when asked and " in text and "It describes that frame only, not the scene now." in text
        assert model.questions == ["Is anyone in the room?"]
        # A second question, answered at once, describes a frame that was still live: it is not marked historical.
        wait_for_interval(served)
        type_question(page, "And the door?")
        press_ask(page)
        seen = wait_chat(page, lambda s: len(s["log"]) == 2 and last(s)["state"] == "answered", 8.0)
        assert last(seen)["text"].count("About the frame received at ") == 1 and "Historical" not in last(seen)["text"]
        assert seen["log"][0]["state"] == "answered"  # the conversation is kept
    finally:
        page.close()


def test_busy_no_fresh_frame_and_unavailable_are_said_and_the_draft_is_kept(
        browser: Browser, served: dict[str, Any]) -> None:
    page = opened(browser, f"{served['url']}/")
    camera, analyzer = served["camera"], served["analyzer"]
    try:
        # Busy: the scene worker is analysing the scene; the question is refused, never queued.
        served["scene_hold"].clear()
        frame, image = camera.frames.latest()
        analyzer._images.put(frame.key, image)
        analyzer.submit(AnalysisJob(job_id="scene-1", purpose=JobPurpose.PERIODIC, frame=frame,
                                    deadline_mono_ns=frame.ingest_mono_ns + 60 * NS_PER_SECOND))
        assert wait_until(analyzer.busy, 5.0)
        type_question(page, "What is on the desk?")
        press_ask(page)
        seen = wait_chat(page, lambda s: s["log"] and last(s)["state"] == "busy", 5.0)
        assert "Busy: the scene model is busy; monitoring comes first, so ask again shortly." in last(seen)["text"]
        assert seen["draft"] == "What is on the desk?" and not seen["ask_disabled"]  # kept, to ask again
        served["scene_hold"].set()
        assert wait_until(lambda: not analyzer.busy(), 5.0)
        # No fresh frame: the camera stops; the newest frame is old, and the model is not asked.
        served["stop"].set()
        time.sleep(0.5)  # the fake camera's last tick is over
        camera.clock.advance(3.0)
        press_ask(page)
        seen = wait_chat(page, lambda s: len(s["log"]) == 2 and last(s)["state"] == "no_fresh_frame", 5.0)
        assert "No fresh frame: the newest frame is 3." in last(seen)["text"]
        assert "ask again when the camera is live" in last(seen)["text"] and " old when asked." in last(seen)["text"]
        assert seen["draft"] == "What is on the desk?" and served["model"].questions == []
        # Unavailable: chat loses its admission; the next update disables Ask and says why.
        served["chat"]["service"] = served["service"](CHAT_ADMISSION)
        seen = wait_chat(page, lambda s: s["state"] == "unavailable", REFRESH_S + 6.0)
        assert seen["ask_disabled"] and seen["note"] == "Chat is not available now: see the reason above."
        assert "operator-run resource and interference checks" in seen["text"]
        assert seen["draft"] == "What is on the desk?" and len(seen["log"]) == 2
    finally:
        page.close()


def test_updates_keep_the_draft_conversation_and_focus_and_an_answer_never_makes_a_stale_page_current(
        browser: Browser, served: dict[str, Any]) -> None:
    page = opened(browser, f"{served['url']}/")
    runtime = served["runtime"]
    try:
        type_question(page, "What colour is the chair?")
        press_ask(page)
        wait_chat(page, lambda s: s["log"] and last(s)["state"] == "answered", 8.0)
        page.evaluate("document.querySelector('#chat-log > li').__testMark = 'first'")
        type_question(page, "A draft, half writ")
        before = read(page)
        assert before["focused"]
        time.sleep(2 * REFRESH_S + 1.0)
        after = read(page)
        assert after["sequence"] >= before["sequence"] + 2 and after["document"] == before["document"]  # updated in place
        assert after["draft"] == "A draft, half writ" and after["focused"]
        assert after["log"] == before["log"] and after["log"][0]["kept"] == "first"  # the same exchange, untouched
        # The page stalls: status updates hang, the chat still answers. An answer is not camera data: the page stays
        # NOT CURRENT, before and after it.
        runtime.answering.clear()
        wait_chat(page, lambda s: PAGE_NOT_CURRENT in s["text"], PAGE_CURRENT_S + REFRESH_S + 1.0)
        wait_for_interval(served)
        page.evaluate("document.getElementById('chat-q').value = ''")
        type_question(page, "Still there?")
        press_ask(page)
        seen = wait_chat(page, lambda s: len(s["log"]) == 2 and last(s)["state"] == "answered", 8.0)
        for _ in range(10):
            seen = read(page)
            assert PAGE_NOT_CURRENT in seen["text"] and PAGE_CURRENT not in seen["text"], seen["text"][:200]
            time.sleep(0.2)
        runtime.answering.set()
        wait_chat(page, lambda s: PAGE_CURRENT in s["text"], 16.0)
    finally:
        page.close()


def test_after_the_runtime_restarts_the_page_asks_for_a_reload_before_asking(
        browser: Browser, served: dict[str, Any]) -> None:
    page = opened(browser, f"{served['url']}/")
    port = served["server"].address[1]
    again = None
    try:
        type_question(page, "Is the door open?")
        assert not read(page)["ask_disabled"]
        assert served["server"].stop(3.0)  # the runtime restarts: a new server, a new token
        again = StatusServer(port, served["runtime"].snapshot, served["db"], preview=lambda: served["camera"].preview,
                             chat=lambda: served["chat"]["service"])
        again.start()
        seen = wait_chat(page, lambda s: "The runtime restarted" in s["note"], 20.0)
        assert seen["note"] == "The runtime restarted after this page was loaded: reload the page to ask."
        assert seen["ask_disabled"] and seen["draft"] == "Is the door open?" and PAGE_CURRENT in seen["text"]
        assert served["model"].questions == []
    finally:
        page.close()
        if again is not None:
            assert again.stop(3.0)


def test_another_origin_cannot_ask(browser: Browser, served: dict[str, Any], tmp_path: Path) -> None:
    other_db = tmp_path / "other"
    other_db.mkdir()
    incident_store(other_db / "sentinel.db", count=1)
    other = StatusServer(0, Runtime().snapshot, other_db / "sentinel.db")
    other.start()
    page = browser.page()
    try:
        page.navigate(f"http://127.0.0.1:{other.address[1]}/")
        target = json.dumps(f"{served['url']}/chat")
        results = page.evaluate(f"""Promise.all([
          fetch({target}, {{method: 'POST', headers: {{'Content-Type': 'application/json', 'X-Sentinel-Token': 'x'}},
                body: JSON.stringify({{question: 'preflighted'}})}}).then((r) => r.status, (e) => 'blocked'),
          fetch({target}, {{method: 'POST', mode: 'no-cors', headers: {{'Content-Type': 'text/plain'}},
                body: JSON.stringify({{question: 'simple'}})}}).then((r) => r.type, (e) => 'blocked'),
        ])""")
        # The first needs a preflight, which this server refuses; the second is sent, refused, and unreadable.
        assert results[0] == "blocked" and results[1] in ("opaque", "blocked")
        assert served["model"].questions == [] and served["chat"]["service"].summary()["counters"] == {}
    finally:
        page.close()
        assert other.stop(3.0)


def test_the_chat_panel_works_on_a_phone(browser: Browser, served: dict[str, Any]) -> None:
    page = opened(browser, f"{served['url']}/", width=390, height=844, mobile=True)
    try:
        order = page.evaluate("[...document.querySelectorAll('main > section, main > aside')].map((e) => e.id)")
        assert order == ["live", "incidents", "chat", "health", "delivery"]
        tops = page.evaluate("['incidents', 'chat'].map((id) => document.getElementById(id).getBoundingClientRect().top)")
        assert tops[0] < tops[1]  # below the incidents, in one column
        type_question(page, "x" * 300)  # the longest question, one unbroken word
        press_ask(page)
        seen = wait_chat(page, lambda s: s["log"] and last(s)["state"] == "answered", 8.0)
        width = page.evaluate("({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth})")
        assert width["sw"] <= width["cw"], width  # nothing scrolls sideways
        page.evaluate("document.body.focus(); window.scrollTo(0, 0)")
        reached = False
        for _ in range(60):  # the question box is reachable from the keyboard
            page.key("Tab")
            if page.evaluate("document.activeElement && document.activeElement.id") == "chat-q":
                reached = True
                break
        assert reached and seen["draft"] == ""
    finally:
        page.close()
