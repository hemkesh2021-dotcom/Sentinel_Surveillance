"""D74: the chat panel in a real browser, with a scripted model (no camera, no model).

Headless Chromium (``browser_harness``) loads the real StatusServer, with a real ThreadedSceneAnalyzer as the scene
worker and a ChatService admitted for the test only; the model is a function the test holds and releases. The page
is driven as a viewer would: typing into the question box, pressing Ask.

Covered: sending, waiting/answering and answered, with the frame the answer describes, its time and its age (marked
historical once that frame stopped being live); busy (the scene worker comes first), no fresh frame and unavailable,
each said, with the draft kept; the draft, the conversation and keyboard focus kept across page updates; an answer
never making a stale page current; a restarted runtime (a new token) asking for a reload; another origin unable to
ask; and the panel on a phone. Session 114, from scripted server documents: a question that ends unanswered
(cancelled, incomplete) comes back into an empty box; a shortened answer says so and shows its unfinished rest apart;
an answer about a newer frame sent in place of the one asked about says so, with that frame's own facts.
Session 116, part 34: the conversation scrolls inside its own bounded list (desktop and phone, light and dark), which
never lengthens the page or moves the question box; the newest exchange is shown from its start; a reader who
scrolled up, or read on to the end, is left there through a reply and page updates, with the draft and focus kept; a
card asked again is shown from its start; the list scrolls from the keyboard, and on a phone a swipe at its end
moves the page. Part 35: the panel's explanations are in a card that an information key beside the heading opens on
hover, keyboard focus, a click or a tap (desktop and phone, light and dark), worded as before and fitting the panel;
it stays open under the pointer and through page updates, and Escape, a second press or a press elsewhere closes
it; a validation run's flag and every reason but "available" stay in view.
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
from sentinel.history import RecentHistory
from sentinel.jobs import AnalysisJob, JobPurpose, WorkerOutcome
from sentinel.media.clock import NS_PER_SECOND
from sentinel.media.health import assess_frame
from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer
from sentinel import status_page
from sentinel.scene.chat import ADVISORY, CHAT_ADMISSION, CHAT_MIN_INTERVAL_S, ChatAdmission, ChatService
from sentinel.status_page import PAGE_CURRENT_S, REFRESH_S, StatusServer
from sentinel.web import palettes

ADMITTED = ChatAdmission(True, "admitted for this test", "a synthetic fixture")
PAGE_CURRENT = "This page is current"
PAGE_NOT_CURRENT = f"NOT CURRENT: this page has not refreshed for {PAGE_CURRENT_S} s"
CHAT = """(() => {
  const q = document.getElementById('chat-q');
  return {draft: q.value, focused: document.activeElement === q, ask_disabled: document.getElementById('chat-ask').disabled,
          note: document.getElementById('chat-note').textContent, text: document.body.innerText,
          sequence: Number(document.documentElement.dataset.sequence || '-1'), document: performance.timeOrigin,
          // the page's own script has run: it runs in one go, and marks the Settings window it set up on the way
          ready: document.documentElement.classList.contains('sd'),
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

    def __call__(self, question: Any, image: Any, abort: Any = None,
                 max_tokens: Any = None) -> tuple[str, bool]:
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

    def service(admission: ChatAdmission, **options: Any) -> ChatService:
        return ChatService(enabled=True, admission=admission, scene=analyzer, latest=camera.frames.latest,
                           assess=lambda f: assess_frame(f, camera.stamper.current_stream, camera.clock.mono(), started,
                                                         FreshnessConfig()),
                           clock=camera.clock, run=model, stale_after_s=2.0, **options)

    chat = {"service": service(ADMITTED)}

    class Source:  # part 28: what recorded-fact answers read; a label with markup in it, to show it stays text
        history = RecentHistory(camera.clock)

        @staticmethod
        def occupants() -> dict[str, Any]:
            return {"published_utc": "2026-11-10T12:00:00+00:00", "video": "fresh", "last_frame_age_ms": 40,
                    "occupancy": "occupied", "occupancy_reason": "1 confirmed person(s) on fresh video",
                    "face_recognition": "available", "omitted": 0, "people": [
                        {"track_id": 61, "stream_epoch": 1, "status": "confirmed", "identity": "known",
                         "basis": "fresh", "vote_age_ms": 500, "association": None,
                         "label": "<b id=inj>Res</b>ident"}]}

    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview,
                          chat=lambda: chat["service"], evidence=lambda: Source)
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


def opened(browser: Browser, url: str, *, touch: bool = False, **options: Any) -> Page:
    page = browser.page(**options)
    if touch:  # part 35: a touch screen from the start, as on a phone (not switched on under a loaded page)
        page.call("Emulation.setTouchEmulationEnabled", {"enabled": True, "maxTouchPoints": 1})
    page.navigate(url)
    wait_chat(page, lambda s: s["ready"] and PAGE_CURRENT in s["text"] and s["state"] == "available"
              and s["sequence"] >= 0, 30.0)
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
        # a question about the picture ("is anyone in the room?" is answered from the tracker's records since part 28)
        type_question(page, "Is there a chair by the window?")
        assert not read(page)["ask_disabled"]
        model.release.clear()  # the model takes its time
        press_ask(page)
        seen = wait_chat(page, lambda s: s["log"] and last(s)["state"] == "answering", 5.0)
        assert "The scene model is answering…" in last(seen)["text"] and "Is there a chair by the window?" in last(seen)["text"]
        assert seen["draft"] == "" and seen["ask_disabled"] and seen["note"] == "One question at a time."
        time.sleep(2.5)  # the frame stops being live while the model answers
        model.release.set()
        seen = wait_chat(page, lambda s: last(s)["state"] == "answered", 5.0)
        text = last(seen)["text"]
        assert "One chair stands by the window; nobody is in the picture." in text and "Answered." in text
        assert "Historical: about the frame received at " in text and " UTC (stream epoch 1, frame " in text
        assert " old when asked and " in text and "It describes that frame only, not the scene now." in text
        assert model.questions == ["Is there a chair by the window?"]
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
        # Busy: the scene worker is analysing the scene; the question waits for it (D81) and, when the scene job
        # outlasts CHAT_START_S, gives up as busy: never queued, never sent to the model.
        served["scene_hold"].clear()
        frame, image = camera.frames.latest()
        analyzer._images.put(frame.key, image)
        analyzer.submit(AnalysisJob(job_id="scene-1", purpose=JobPurpose.PERIODIC, frame=frame,
                                    deadline_mono_ns=frame.ingest_mono_ns + 60 * NS_PER_SECOND))
        assert wait_until(analyzer.busy, 5.0)
        type_question(page, "What is on the desk?")
        press_ask(page)
        seen = wait_chat(page, lambda s: s["log"] and last(s)["state"] == "waiting", 5.0)
        assert "Waiting for the scene model" in last(seen)["text"]
        camera.clock.advance(5.5)  # longer than CHAT_START_S on the runtime's clock
        seen = wait_chat(page, lambda s: s["log"] and last(s)["state"] == "busy", 8.0)
        assert "Busy: the scene model stayed busy; monitoring comes first." in last(seen)["text"]
        # Handed over, then given up: it comes back into the empty box, to send again without retyping (session 114).
        assert seen["draft"] == "What is on the desk?" and served["model"].questions == []
        served["scene_hold"].set()
        assert wait_until(lambda: not analyzer.busy(), 5.0)
        # No fresh frame: the camera stops; the newest frame is old, and the model is not asked.
        served["stop"].set()
        time.sleep(0.5)  # the fake camera's last tick is over
        camera.clock.advance(3.0)
        press_ask(page)  # the same question, as it came back
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
        # part 28: with script, Features and Appearance are the Settings dialog's two sections, not page sections
        assert order == ["live", "faces", "incidents", "chat", "health", "delivery"]
        assert page.evaluate("[...document.querySelectorAll('#sd-panel > section')].map((e) => e.id)") == [
            "settings", "features"]
        tops = page.evaluate("['incidents', 'chat'].map((id) => document.getElementById(id).getBoundingClientRect().top)")
        assert tops[0] < tops[1]  # below the incidents, in one column
        type_question(page, "x" * 300)  # the longest question, one unbroken word
        press_ask(page)
        seen = wait_chat(page, lambda s: s["log"] and last(s)["state"] == "answered", 8.0)
        width = page.evaluate("({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth})")
        assert width["sw"] <= width["cw"], width  # nothing scrolls sideways
        # nor is the card cut off inside the list, which hides what does not fit (part 34)
        assert page.evaluate("(() => { const log = document.getElementById('chat-log');"
                             " return log.scrollWidth <= log.clientWidth; })()")
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


class Scripted:
    """A chat service that gives each question the next scripted server document (what the page shows of it)."""

    def __init__(self, *documents: dict[str, Any]) -> None:
        self.documents = list(documents)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.asked: list[str] = []

    def availability(self) -> dict[str, str]:
        return {"state": "available", "reason": "ask about the picture the camera is showing now"}

    def ask(self, question: Any) -> tuple[int, dict[str, Any]]:
        self.asked.append(question.question)
        job = f"{len(self.asked):016x}"
        self.jobs[job] = {"job": job, "advisory": ADVISORY, **self.documents.pop(0)}
        return 202, {"job": job, "state": "waiting", "frame": self.jobs[job]["frame"], "advisory": ADVISORY}

    def poll(self, job_id: str) -> tuple[int, dict[str, Any]]:
        return (200, self.jobs[job_id]) if job_id in self.jobs else (404, {"state": "unknown"})


ASKED = {"stream_epoch": 1, "frame_seq": 3, "ingest_utc": "2026-01-01T00:00:00.200000Z", "age_ms_when_asked": 200}
SENT = {"stream_epoch": 2, "frame_seq": 7, "ingest_utc": "2026-01-01T00:00:04.900000Z", "age_ms_when_asked": None,
        "age_ms_when_sent": 300}


def test_unanswered_questions_come_back_and_shortened_or_replaced_frame_answers_say_so(
        browser: Browser, served: dict[str, Any]) -> None:
    scripted = Scripted(
        {"state": "cancelled", "reason": "a scene job is waiting: monitoring comes first", "frame": ASKED},
        {"state": "answered", "answer": "A person stands by the door.", "cut_short": False, "shortened": True,
         "unfinished": "It is hard to tell whether the door", "frame": SENT, "frame_replaced": True,
         "asked_frame": ASKED, "age_ms_when_answered": 1500, "historical": False},
        {"state": "incomplete", "reason": "the answer had to stop before the next scene check, before its first "
                                          "sentence was complete; ask again", "frame": ASKED},
    )
    served["chat"]["service"] = scripted
    page = opened(browser, f"{served['url']}/")
    try:
        type_question(page, "Who is at the door?")
        press_ask(page)
        seen = wait_chat(page, lambda s: s["log"] and last(s)["state"] == "cancelled", 8.0)
        assert "Cancelled: a scene job is waiting: monitoring comes first." in last(seen)["text"]
        assert seen["draft"] == "Who is at the door?" and not seen["ask_disabled"]  # back in the box, to send again
        press_ask(page)  # sent again as it is: no retyping
        seen = wait_chat(page, lambda s: len(s["log"]) == 2 and last(s)["state"] == "answered", 8.0)
        text = last(seen)["text"]
        assert "Answered, shortened: it had to stop before the next scene check and may have said more." in text
        assert "A person stands by the door." in text
        assert "Left unfinished when it stopped: “It is hard to tell whether the door…”" in text  # shown, not dropped
        assert "(stream epoch 2, frame 7), sent in place of the frame asked about, which was no longer live: " in text
        assert "0.3 s old when sent and 1.5 s old when answered." in text and "Historical" not in text
        assert seen["draft"] == ""  # answered: the box stays clear
        assert scripted.asked == ["Who is at the door?", "Who is at the door?"]
        type_question(page, "Is the door open?")
        press_ask(page)
        wait_chat(page, lambda s: len(s["log"]) == 3 and last(s)["state"] == "waiting", 5.0)
        type_question(page, "Something else?")  # typed while it waits: never replaced by the question coming back
        seen = wait_chat(page, lambda s: len(s["log"]) == 3 and last(s)["state"] == "incomplete", 8.0)
        assert "Incomplete: the answer had to stop before the next scene check" in last(seen)["text"]
        assert "A person stands" not in last(seen)["text"] and seen["draft"] == "Something else?"
    finally:
        page.close()


def test_a_question_about_records_is_answered_as_recorded_facts_shown_as_text_and_kept_from_the_model(
        browser: Browser, served: dict[str, Any]) -> None:
    """Part 28: "who is in the room?" is answered from the runtime's records at once, labelled as recorded facts, with
    its lines and limits as text (markup in a configured label stays text); the box is cleared; the model is not
    asked; and the exchange never joins the context sent with a later question about the picture."""
    page = opened(browser, f"{served['url']}/", width=1280, height=900)
    try:
        type_question(page, "Who is in the room?")
        press_ask(page)
        seen = wait_chat(page, lambda s: s["log"] and last(s)["state"] == "answered", 10.0)
        entry = last(seen)["text"]
        assert "Answered from recorded facts, not from the picture." in entry
        assert "1 confirmed person tracked on fresh video" in entry
        assert "“<b id=inj>Res</b>ident” (the operator's label for an enrolment) (track 61, stream epoch 1)" in entry
        assert "Recorded facts: from the system's own records" in entry  # what it is, with its limits
        assert page.evaluate("document.getElementById('inj')") is None  # the label's markup never became an element
        assert page.evaluate("document.querySelector('#chat-log > li:last-child').dataset.kind") == "recorded"
        assert page.evaluate("document.querySelectorAll('#chat-log > li:last-child ul.lines > li').length") == 1
        assert seen["draft"] == "" and served["model"].questions == []  # taken, and never sent to the model
        type_question(page, "Were any alerts triggered?")
        press_ask(page)
        seen = wait_chat(page, lambda s: len(s["log"]) == 2 and last(s)["state"] == "answered", 10.0)
        assert "recorded between" in last(seen)["text"] and "[incident " in last(seen)["text"]
        asked: list[Any] = []
        original = served["chat"]["service"]._run
        served["chat"]["service"]._run = lambda question, *a, **k: asked.append(question) or original(question, *a, **k)
        type_question(page, "Is the door open?")
        press_ask(page)
        seen = wait_chat(page, lambda s: len(s["log"]) == 3 and last(s)["state"] == "answered", 20.0)
        assert served["model"].questions == ["Is the door open?"]
        assert asked[0].history == ()  # neither recorded exchange was sent as context
    finally:
        page.close()


# Session 116, part 34: the conversation scrolls inside its own bounded list. Longer answers, more questions and
# status polling never lengthen the dashboard or move the question box. The newest exchange is shown from its start
# (an answered exchange is taller than the list); a reader who scrolled up to an earlier answer, or read on to the
# end of the newest one, is left there; a card asked again is shown from its start.
SCROLL = """(() => {
  const log = document.getElementById('chat-log');
  const box = log.getBoundingClientRect();
  const rect = (id) => { const r = document.getElementById(id).getBoundingClientRect();
                        return {top: r.top, bottom: r.bottom, height: r.height}; };
  // in view: the node's first line lies inside the list's own box
  const shown = (node) => { if (!node) return null; const r = node.getBoundingClientRect();
                           return r.top >= box.top - 1 && r.top + Math.min(r.height, 18) <= box.bottom + 1; };
  const exchange = (li) => li && {offset: li.getBoundingClientRect().top - box.top, height: li.offsetHeight,
                                  question: shown(li.querySelector('.question')),
                                  status: shown(li.querySelector('.status')),
                                  answer: shown(li.querySelector('.answer')), end: shown(li.lastElementChild)};
  return {top: log.scrollTop, height: log.clientHeight, content: log.scrollHeight,
          bottom: log.scrollHeight - log.clientHeight - log.scrollTop,
          newest: exchange(log.lastElementChild), first: exchange(log.firstElementChild),
          panel: rect('chat'), header: rect('h-chat'), composer: rect('chat-form'), tools: rect('zoom-tools'),
          page: document.documentElement.scrollHeight, y: window.scrollY,
          horizontal: document.documentElement.scrollWidth > document.documentElement.clientWidth,
          sideways: log.scrollWidth > log.clientWidth};  // cut off inside the list, which hides it
})()"""
NEWEST = "document.querySelector('#chat-log > li:last-child')"
TO_END = "(() => { const log = document.getElementById('chat-log'); log.scrollTop = log.scrollHeight; })()"


def long_reply(sentences: int = 14) -> dict[str, Any]:
    """An answered document as the server sends it, with its held picture: the card has every one of its rows."""
    return {"state": "answered", "answer": "A synthetic chair stands beside the table. " * sentences,
            "cut_short": False, "frame": ASKED, "age_ms_when_answered": 400, "historical": False,
            "picture": "00000000000000ab"}


def ask_scripted(page: Page, number: int) -> None:
    type_question(page, f"Describe synthetic picture {number}.")
    press_ask(page)
    wait_chat(page, lambda s: len(s["log"]) == number and last(s)["state"] == "answered", 8.0)


def scroll(page: Page) -> dict[str, Any]:
    return page.evaluate(SCROLL)


def frames(page: Page, count: int = 4) -> None:
    """Lets the page render ``count`` frames. A scroll event, a toggle event and a resize observation are each
    delivered with a frame after their cause, never at once: the page has seen them when this returns."""
    assert page.evaluate("new Promise((done) => { let left = %d; const step = () => (--left > 0"
                         " ? requestAnimationFrame(step) : done(true)); requestAnimationFrame(step); })" % count)


def from_its_start(seen: dict[str, Any], which: str = "newest") -> bool:
    """That exchange begins at the top of the list: its question, state and the answer's first line in view."""
    exchange = seen[which]
    return bool(exchange and 0 <= exchange["offset"] <= 12 and exchange["question"] and exchange["status"]
                and exchange["answer"])


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width,height", [(1280, 900), (390, 844)])
def test_chat_history_scrolls_without_growing_the_page_or_moving_the_composer(
        browser: Browser, served: dict[str, Any], theme: str, width: int, height: int) -> None:
    served["chat"]["service"] = Scripted(*(long_reply() for _ in range(4)))
    page = opened(browser, f"{served['url']}/?theme={theme}", width=width, height=height, mobile=width < 500)
    try:
        ask_scripted(page, 1)
        first = scroll(page)
        for n in range(2, 5):
            ask_scripted(page, n)
        # Each reply was followed from the list's end, and the newest exchange is shown from its start: what was
        # asked and the beginning of the answer, not the rows at its end.
        assert wait_until(lambda: from_its_start(scroll(page)), 3.0), scroll(page)
        filled = scroll(page)
        assert filled["newest"]["height"] > filled["height"] and not filled["newest"]["end"]  # its end: scrolled to
        assert filled["content"] > filled["height"] * 2  # actual overflow, not just an empty fixed box
        assert abs(filled["panel"]["height"] - first["panel"]["height"]) <= 1
        assert abs(filled["page"] - first["page"]) <= 1
        assert not filled["horizontal"] and not filled["sideways"]
        page.evaluate("document.getElementById('chat').scrollIntoView({block: 'start', behavior: 'instant'})")
        page.save_screenshot(f"ask-scroll-{theme}-{width}", view=True)
        # The named history is a keyboard stop of its own, just before the controls inside it, and scrolls from
        # the keyboard without scrolling the page itself.
        page.evaluate("document.querySelector('#chat-log button').focus()")
        page.key("Tab", shift=True)
        assert page.evaluate("document.activeElement && document.activeElement.id") == "chat-log"
        page.evaluate("document.getElementById('chat-log').scrollTop = 0")
        assert wait_until(lambda: scroll(page)["top"] == 0, 3.0)
        frames(page)  # the browser delivers its scroll event before the next input
        before = scroll(page)
        page.key("ArrowDown")
        assert wait_until(lambda: scroll(page)["top"] > before["top"] + 10, 3.0)
        after = scroll(page)
        assert abs(after["y"] - before["y"]) <= 1
        for name in ("header", "composer", "tools"):
            assert abs(after[name]["top"] - before[name]["top"]) <= 1
        assert page.evaluate("document.getElementById('chat-log').getAttribute('aria-label')") == "Conversation history"
    finally:
        page.close()


def test_reading_earlier_chat_keeps_scroll_and_draft_through_a_reply_and_page_updates(
        browser: Browser, served: dict[str, Any]) -> None:
    pending = {"state": "answering", "frame": ASKED}
    scripted = Scripted(long_reply(), long_reply(), pending, long_reply())
    served["chat"]["service"] = scripted
    page = opened(browser, f"{served['url']}/", width=1280, height=900)
    try:
        for n in (1, 2):
            ask_scripted(page, n)
        type_question(page, "A question whose answer takes longer.")
        press_ask(page)
        wait_chat(page, lambda s: len(s["log"]) == 3 and last(s)["state"] == "answering", 5.0)
        # A short newest exchange (still being answered) fits: the list is at its end, with that exchange whole.
        assert wait_until(lambda: scroll(page)["bottom"] <= 2, 3.0), scroll(page)
        waiting = scroll(page)["newest"]
        assert waiting["question"] and waiting["status"] and waiting["end"] and waiting["offset"] > 12
        type_question(page, "Keep this unfinished draft")
        page.evaluate("document.getElementById('chat-log').scrollTop = 60")
        assert wait_until(lambda: scroll(page)["top"] == 60, 3.0)
        frames(page)
        before, sequence = scroll(page), read(page)["sequence"]
        job = next(reversed(scripted.jobs))
        scripted.jobs[job].update(long_reply())
        wait_chat(page, lambda s: last(s)["state"] == "answered" and s["sequence"] >= sequence + 2, 20.0)
        after, seen = scroll(page), read(page)
        assert abs(after["top"] - before["top"]) <= 1  # the answer did not drag us to the newest exchange
        assert abs(after["y"] - before["y"]) <= 1
        assert after["bottom"] > 100 and not after["newest"]["question"]
        assert seen["draft"] == "Keep this unfinished draft" and seen["focused"]
        # Asking deliberately returns to the new exchange; it still preserves the same DOM and tab-only history.
        press_ask(page)
        wait_chat(page, lambda s: len(s["log"]) == 4 and last(s)["state"] == "answered", 8.0)
        assert wait_until(lambda: from_its_start(scroll(page)), 3.0), scroll(page)
        assert len(scripted.asked) == 4 and served["model"].questions == []
    finally:
        page.close()


def test_reading_on_to_the_end_of_the_newest_answer_stays_there_and_a_card_asked_again_is_shown_from_its_start(
        browser: Browser, served: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in (("AUTO_REFRESH_MIN_S", 1), ("AUTO_REFRESH_DEFAULT_S", 1), ("AUTO_REFRESH_MAX_S", 4)):
        monkeypatch.setattr(status_page, name, value)  # an automatic refresh a second after it is started
    scripted = Scripted(*(long_reply() for _ in range(14)))
    served["chat"]["service"] = scripted
    page = opened(browser, f"{served['url']}/", width=1280, height=900)
    try:
        for n in (1, 2):
            ask_scripted(page, n)
        assert wait_until(lambda: from_its_start(scroll(page)), 3.0), scroll(page)
        # The reader reads on, to the end of the newest exchange: its controls are there, its question is above.
        page.evaluate(TO_END)
        assert wait_until(lambda: scroll(page)["bottom"] <= 1, 3.0)
        frames(page)
        at_end = scroll(page)
        assert at_end["newest"]["end"] and not at_end["newest"]["question"]
        # The card changes under the reader: its automatic refresh, started from there, asks again and files the
        # earlier result in a new line at the card's end. The view stays at the end and shows that line; it is
        # never taken back to the card's start.
        page.evaluate(f"{NEWEST}.querySelector('button[data-auto=start]').click()")
        wait_chat(page, lambda s: "Earlier results (" in last(s)["text"] and last(s)["state"] == "answered", 15.0)
        page.evaluate(f"{NEWEST}.querySelector('button[data-auto=stop]').click()")
        wait_chat(page, lambda s: "Auto-refresh stopped." in last(s)["text"] and last(s)["state"] == "answered"
                  and s["note"] == "", 10.0)  # a refresh that was on its way has ended too
        frames(page)
        grown = scroll(page)
        assert grown["content"] >= at_end["content"] + 10  # the card grew at its end, under a reader who was there
        assert grown["bottom"] <= 2 and grown["newest"]["end"] and not grown["newest"]["question"], grown
        # Asked again by the reader from those controls, the card is shown from its start, where its new state
        # and its new answer appear; the list did not lengthen the page.
        sent = len(scripted.asked)
        page.evaluate(f"{NEWEST}.querySelector('button[data-again=latest]').click()")
        seen = wait_chat(page, lambda s: len(scripted.asked) == sent + 1 and last(s)["state"] == "answered"
                         and s["note"] == "", 8.0)
        assert len(seen["log"]) == 2  # the same card, in place
        again = scroll(page)
        assert from_its_start(again), again
        assert abs(again["page"] - at_end["page"]) <= 1 and abs(again["y"] - at_end["y"]) <= 1
        # Back at the end, the reader opens "Earlier results": the card grows below them with no scroll and no
        # change to the conversation. A change that follows (the automatic refresh's line) leaves them there.
        page.evaluate(TO_END)
        assert wait_until(lambda: scroll(page)["bottom"] <= 1, 3.0)
        frames(page)
        page.evaluate(f"{NEWEST}.querySelector('details.earlier summary').click()")
        assert wait_until(lambda: scroll(page)["bottom"] > 20, 3.0), scroll(page)  # its list opened, below the view
        frames(page)  # the browser delivers the toggle event
        opened_at = scroll(page)
        page.evaluate(f"{NEWEST}.querySelector('button[data-auto=start]').click()")
        assert wait_until(lambda: page.evaluate(f"!!{NEWEST}.querySelector('button[data-auto=stop]')"), 3.0)
        assert abs(scroll(page)["top"] - opened_at["top"]) <= 1, (opened_at, scroll(page))
        page.evaluate(f"{NEWEST}.querySelector('button[data-auto=stop]').click()")
    finally:
        page.close()


def test_a_card_asked_again_while_the_newest_is_also_in_view_is_followed_itself_not_the_newest(
        browser: Browser, served: dict[str, Any]) -> None:
    """Two questions that ended unanswered, both in view (a large window): "Try again" on the first one shows its
    answer from the card's start; the list is not taken to the newest card, which did not change."""
    ended = {"state": "cancelled", "reason": "monitoring comes first", "frame": ASKED}
    scripted = Scripted(ended, dict(ended), long_reply())
    served["chat"]["service"] = scripted
    page = opened(browser, f"{served['url']}/", width=1920, height=1080)
    try:
        type_question(page, "Is the door open?")
        press_ask(page)
        wait_chat(page, lambda s: len(s["log"]) == 1 and last(s)["state"] == "cancelled", 8.0)
        press_ask(page)  # the question came back into the box: sent again as it is, a second card
        wait_chat(page, lambda s: len(s["log"]) == 2 and last(s)["state"] == "cancelled" and s["note"] == "", 8.0)
        both = scroll(page)
        assert both["content"] <= both["height"] and both["first"]["question"] and both["newest"]["end"], both
        page.evaluate("document.querySelector('#chat-log > li:first-child button[data-again=latest]').click()")
        assert wait_until(lambda: page.evaluate(
            "document.querySelector('#chat-log > li:first-child').dataset.state") == "answered", 8.0)
        frames(page)
        seen = scroll(page)
        assert seen["first"]["height"] > seen["height"]  # its answer made it taller than the list
        assert from_its_start(seen, "first"), seen  # shown from its start: the question and the new answer
        assert len(scripted.asked) == 3 and len(read(page)["log"]) == 2
    finally:
        page.close()


@pytest.mark.parametrize("anchoring", [True, False])
def test_a_reader_who_opened_an_earlier_cards_results_is_not_taken_to_the_newest_reply(
        browser: Browser, served: dict[str, Any], anchoring: bool) -> None:
    """At the list's end, with a short newest exchange still being answered below an earlier card: the reader opens
    that card's "Earlier results". The reply to the newest question then arrives; they stay at what they opened.
    Also where the browser does not move the view itself when the results open (its scroll anchoring off): then
    no scroll tells the page that the reader's place changed, only the opening itself."""
    pending = {"state": "answering", "frame": ASKED}
    scripted = Scripted(long_reply(3), long_reply(3), pending)
    served["chat"]["service"] = scripted
    page = opened(browser, f"{served['url']}/", width=1280, height=900)
    earlier = "document.querySelector('#chat-log > li:first-child details.earlier')"
    try:
        if not anchoring:
            page.evaluate("document.getElementById('chat-log').style.overflowAnchor = 'none'")
        ask_scripted(page, 1)
        page.evaluate("document.querySelector('#chat-log > li:first-child button[data-again=latest]').click()")
        wait_chat(page, lambda s: len(scripted.asked) == 2 and last(s)["state"] == "answered" and s["note"] == "", 8.0)
        type_question(page, "A question whose answer takes longer.")
        press_ask(page)
        wait_chat(page, lambda s: len(s["log"]) == 2 and last(s)["state"] == "answering", 5.0)
        assert wait_until(lambda: scroll(page)["bottom"] <= 2, 3.0), scroll(page)  # at the end, the new card whole
        frames(page)
        closed = scroll(page)
        page.evaluate(f"{earlier}.querySelector('summary').click()")
        assert wait_until(lambda: scroll(page)["content"] > closed["content"] + 8, 3.0), scroll(page)
        frames(page)  # the browser delivers the toggle event
        before = scroll(page)
        job = next(reversed(scripted.jobs))
        scripted.jobs[job].update(long_reply())
        wait_chat(page, lambda s: last(s)["state"] == "answered", 8.0)
        frames(page)
        after = scroll(page)
        assert abs(after["top"] - before["top"]) <= 1 and not after["newest"]["question"], (before, after)
        assert page.evaluate(f"(() => {{ const r = {earlier}.querySelector('li').getBoundingClientRect(),"
                             " b = document.getElementById('chat-log').getBoundingClientRect();"
                             " return r.top >= b.top - 1 && r.top < b.bottom; })()")  # what they opened, still in view
    finally:
        page.close()


def test_hiding_and_showing_the_panel_keeps_the_readers_place_inside_the_newest_answer(
        browser: Browser, served: dict[str, Any]) -> None:
    """The focus view hides the Ask panel and gives it back: a reader part-way through the newest answer is where
    they were, not at the card's end."""
    served["chat"]["service"] = Scripted(long_reply(), long_reply())
    page = opened(browser, f"{served['url']}/", width=1280, height=900)
    try:
        for n in (1, 2):
            ask_scripted(page, n)
        assert wait_until(lambda: from_its_start(scroll(page)), 3.0), scroll(page)
        page.evaluate("document.getElementById('chat-log').scrollTop += 80")  # reads on, not to the end
        frames(page)
        before = scroll(page)
        assert before["newest"]["offset"] < -60 and before["bottom"] > 60, before
        page.evaluate("document.getElementById('focus-toggle').click()")
        assert wait_until(lambda: scroll(page)["height"] == 0, 3.0)  # the panel, and the list in it, are hidden
        frames(page)
        page.evaluate("document.getElementById('focus-toggle').click()")
        assert wait_until(lambda: scroll(page)["height"] > 0, 3.0)
        frames(page)
        after = scroll(page)
        assert abs(after["top"] - before["top"]) <= 1 and after["bottom"] > 60, (before, after)
    finally:
        page.close()


def test_a_reader_near_the_end_of_the_newest_answer_is_not_thrown_to_its_end_when_the_window_narrows(
        browser: Browser, served: dict[str, Any]) -> None:
    """Narrowing the window makes every card above the reader taller, and the browser moves the view itself to keep
    in place what is being read; that is not the reader having gone to the end."""
    served["chat"]["service"] = Scripted(long_reply(), long_reply())
    page = opened(browser, f"{served['url']}/", width=1440, height=900)
    try:
        for n in (1, 2):
            ask_scripted(page, n)
        assert wait_until(lambda: from_its_start(scroll(page)), 3.0), scroll(page)
        page.evaluate("(() => { const log = document.getElementById('chat-log');"
                      " log.scrollTop = log.scrollHeight - log.clientHeight - 60; })()")  # near the end, not at it
        frames(page)
        before = scroll(page)
        assert 50 <= before["bottom"] <= 70 and before["newest"]["offset"] < 0, before
        page.viewport(1280, 900)
        assert wait_until(lambda: scroll(page)["content"] > before["content"] + 100, 5.0), scroll(page)  # reflowed
        frames(page)
        after = scroll(page)
        assert after["bottom"] > 40 and after["newest"]["offset"] < 0, (before, after)  # still inside it, not at its end
    finally:
        page.close()


def test_without_the_browsers_own_scroll_anchoring_the_newest_exchange_is_still_shown_from_its_start(
        browser: Browser, served: dict[str, Any]) -> None:
    """Some browsers do not keep the view in place when what is above it changes size (here: switched off for the
    list). When the cards above the newest get shorter, its question must not end up hidden above the view."""
    served["chat"]["service"] = Scripted(long_reply(), long_reply())
    page = opened(browser, f"{served['url']}/", width=1280, height=900)
    try:
        page.evaluate("document.getElementById('chat-log').style.overflowAnchor = 'none'")
        for n in (1, 2):
            ask_scripted(page, n)
        assert wait_until(lambda: from_its_start(scroll(page)), 3.0), scroll(page)
        before = scroll(page)
        page.viewport(1440, 900)  # wider: every card is shorter, and the newest one's start moves up
        assert wait_until(lambda: scroll(page)["content"] < before["content"] - 100, 5.0), scroll(page)
        assert wait_until(lambda: from_its_start(scroll(page)), 3.0), (before, scroll(page))
    finally:
        page.close()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_conversations_scrollbar_is_visible_on_its_track_in_every_palette(
        browser: Browser, served: dict[str, Any], theme: str) -> None:
    """Its thumb is the palette's marker, which every palette keeps at 3:1 or more on every surface; an accent is a
    fill and is not (in three palette modes it is close to the track's own colour)."""
    page = opened(browser, f"{served['url']}/?theme={theme}", width=1280, height=900)

    def colour(value: str) -> str:
        r, g, b = (int(float(part)) for part in value[value.index("(") + 1:value.index(")")].split(",")[:3])
        return f"#{r:02x}{g:02x}{b:02x}"

    try:
        for preset in palettes.PRESETS:
            page.evaluate(f"document.documentElement.dataset.palette = {preset.key!r}")
            thumb, track = page.evaluate(
                "getComputedStyle(document.getElementById('chat-log')).scrollbarColor.match(/rgba?\\([^)]*\\)/g)")
            ratio = palettes.contrast(colour(thumb), colour(track))
            assert ratio >= palettes.UI_FLOOR, (preset.key, theme, thumb, track, round(ratio, 2))
    finally:
        page.close()


class Recorded(Scripted):
    """Answers every question at once, as a recorded-fact answer is: no waiting, no poll."""

    def ask(self, question: Any) -> tuple[int, dict[str, Any]]:
        self.asked.append(question.question)
        return 200, {"state": "answered", "kind": "recorded", "answer": f"Recorded answer {len(self.asked)}.",
                     "lines": [], "limits": [], "advisory": ADVISORY}


def test_the_list_keeps_twelve_exchanges_and_shows_the_newest_when_the_oldest_is_removed(
        browser: Browser, served: dict[str, Any]) -> None:
    scripted = Recorded()
    served["chat"]["service"] = scripted
    page = opened(browser, f"{served['url']}/", width=1280, height=900)
    try:
        for n in range(1, 14):
            type_question(page, f"Recorded question {n}?")
            press_ask(page)
            wait_chat(page, lambda s, n=n: len(scripted.asked) == n and last(s)["state"] == "answered"
                      and f"Recorded question {n}?" in last(s)["text"] and s["note"] == "", 8.0)
            if n == 12:
                frames(page)
                twelve = scroll(page)
        frames(page)
        seen, after = read(page), scroll(page)
        assert len(seen["log"]) == 12  # the thirteenth took the first one's place
        assert "Recorded question 2?" in seen["log"][0]["text"] and "Recorded question 13?" in last(seen)["text"]
        # the newest, whole: a short card, so the list is at its end with the card's question and answer in view
        assert after["bottom"] <= 2 and after["newest"]["height"] < after["height"], after
        assert after["newest"]["question"] and after["newest"]["answer"], after
        assert abs(after["content"] - twelve["content"]) <= 1 and abs(after["page"] - twelve["page"]) <= 1
    finally:
        page.close()


def swipe(page: Page, x: int, y: int, dy: int) -> None:
    """One finger on (x, y), moved by dy and lifted: a swipe as a phone delivers it."""
    page.call("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
    for step in range(1, 13):
        page.call("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x, "y": y + dy * step // 12}]})
        time.sleep(0.02)
    page.call("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})


def test_on_a_phone_a_swipe_inside_the_conversation_scrolls_it_and_at_its_end_moves_the_page(
        browser: Browser, served: dict[str, Any]) -> None:
    """On a phone the list is as wide as the page: a list that kept every swipe to itself would hold the page still
    whenever a finger landed on it."""
    served["chat"]["service"] = Scripted(long_reply(), long_reply())
    page = opened(browser, f"{served['url']}/", width=390, height=844, mobile=True)
    page.call("Emulation.setTouchEmulationEnabled", {"enabled": True, "maxTouchPoints": 1})
    middle = ("(() => { const r = document.getElementById('chat-log').getBoundingClientRect();"
              " return [Math.round((r.left + r.right) / 2), Math.round((r.top + r.bottom) / 2)]; })()")
    try:
        for n in (1, 2):
            ask_scripted(page, n)
        page.evaluate("document.getElementById('chat-log').scrollIntoView({block: 'center', behavior: 'instant'})")
        page.evaluate("document.getElementById('chat-log').scrollTop = 0")
        frames(page)
        before = scroll(page)
        swipe(page, *page.evaluate(middle), -120)  # inside its content: the list scrolls, the page does not
        assert wait_until(lambda: scroll(page)["top"] > before["top"] + 40, 3.0), scroll(page)
        assert abs(scroll(page)["y"] - before["y"]) <= 1
        page.evaluate(TO_END)
        assert wait_until(lambda: scroll(page)["bottom"] <= 1, 3.0)
        frames(page)
        before = scroll(page)
        swipe(page, *page.evaluate(middle), -200)  # already at its end: the same swipe moves the page on
        assert wait_until(lambda: scroll(page)["y"] > before["y"] + 50, 3.0), (before["y"], scroll(page)["y"])
        assert scroll(page)["bottom"] <= 1
    finally:
        page.close()


# ---------------------------------------------------------------- part 35: the panel's information key

AVAILABLE = "ask about the picture the camera is showing now"  # the runtime's own reason (ChatService.availability)
ABOUT = """(() => {
  const at = (e) => { const r = e.getBoundingClientRect();
                      return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, x: r.left + r.width / 2,
                              y: r.top + r.height / 2}; };
  const key = document.getElementById('chat-info'), card = document.getElementById('chat-about-card');
  const rgb = (c) => (c.match(/[\\d.]+/g) || []).slice(0, 3).map(Number);
  const lum = (c) => { const v = rgb(c).map((x) => (x /= 255) <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4);
                       return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]; };
  const ratio = (a, b) => { const [hi, lo] = [lum(a), lum(b)].sort((p, q) => q - p); return (hi + 0.05) / (lo + 0.05); };
  const box = at(card), notes = [...card.querySelectorAll('.note')];
  const hit = document.elementFromPoint(box.x, Math.min(box.top + 14, innerHeight - 1));
  return {open: getComputedStyle(card).display !== 'none', expanded: key.getAttribute('aria-expanded'),
          name: key.getAttribute('aria-label'), controls: key.getAttribute('aria-controls'),
          key: at(key), heading: at(document.getElementById('h-chat')), card: box, panel: at(document.getElementById('chat')),
          composer: at(document.getElementById('chat-form')), tools: at(document.getElementById('zoom-tools')),
          paragraphs: notes.map((p) => p.textContent), shown: card.innerText,
          row: document.querySelector('[data-chat-state]').innerText, body: document.body.innerText,
          contrast: notes.length ? Math.min(...notes.map((p) => ratio(getComputedStyle(p).color,
                                                                   getComputedStyle(card).backgroundColor))) : 0,
          on_top: !!(hit && hit.closest('#chat-about-card')), width: innerWidth,
          scroll_width: document.documentElement.scrollWidth, focused: document.activeElement && document.activeElement.id,
          visible_focus: document.activeElement === key && key.matches(':focus-visible'),
          sequence: Number(document.documentElement.dataset.sequence || '-1'),
          draft: document.getElementById('chat-q').value};
})()"""


def about(page: Page) -> dict[str, Any]:
    return page.evaluate(ABOUT)


def to_panel(page: Page) -> None:
    page.evaluate("document.getElementById('chat').scrollIntoView({block: 'start', behavior: 'instant'})")
    frames(page)


def centre_of(page: Page, selector: str) -> tuple[float, float]:
    box = page.evaluate(f"(() => {{ const r = document.querySelector({selector!r}).getBoundingClientRect();"
                        " return [r.left + r.width / 2, r.top + r.height / 2]; })()")
    return box[0], box[1]


def pointer(page: Page, kind: str, x: float, y: float) -> None:
    extra = {"button": "left", "clickCount": 1} if kind in ("mousePressed", "mouseReleased") else {}
    page.call("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y, **extra})


def click(page: Page, x: float, y: float) -> None:
    for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
        pointer(page, kind, x, y)


def tap(page: Page, x: float, y: float) -> None:
    """One finger down and up on (x, y): a tap, followed by its click (the headless browser's own tap gesture
    is delivered without one)."""
    page.call("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
    page.call("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})


def is_open(page: Page) -> bool:
    return about(page)["open"]


def key_point(page: Page) -> tuple[float, float]:
    """The key's centre now, checked to be what a press there reaches: updates above the panel can move the page
    between two presses, and a press that misses the key is a press elsewhere."""
    for _ in range(40):
        x, y = centre_of(page, "#chat-info")
        if page.evaluate(f"(() => {{ const e = document.elementFromPoint({x}, {y});"
                         " return !!(e && e.closest('#chat-info')); })()"):
            return x, y
        time.sleep(0.05)
    raise AssertionError("the information key is not under its own centre")


KEY_CLICKS = ("(() => { const key = document.getElementById('chat-info');"
              " if (!key.__clicks) { key.__clicks = {n: 0}; key.addEventListener('click', () => { key.__clicks.n += 1; }); }"
              " return key.__clicks.n; })()")


def tap_key(page: Page) -> int:
    """A tap on the key, tapped again only when the browser made no click of it at all (a headless tap can be taken
    for the end of a scroll); a click that reached the key is never repeated, so a page that ignores it still fails.
    Returns how many taps it took."""
    for attempt in range(1, 4):
        before = page.evaluate(KEY_CLICKS)
        tap(page, *key_point(page))
        if wait_until(lambda: page.evaluate(KEY_CLICKS) > before, 1.0):
            return attempt
    raise AssertionError("three taps on the information key made no click")


def click_key(page: Page) -> int:
    """A press on the key, pressed again only when no click reached the key at all: just after the page opens, its
    first update can add a line above the panel between the measurement and the press, and the press then lands
    beside the key. A click that reached the key is never repeated, so a page that ignores it still fails.
    Returns how many presses it took."""
    for attempt in range(1, 4):
        before = page.evaluate(KEY_CLICKS)
        click(page, *key_point(page))
        if wait_until(lambda: page.evaluate(KEY_CLICKS) > before, 1.0):
            return attempt
    raise AssertionError("three presses on the information key made no click on it")


def wording(reason: str | None) -> list[str]:
    """The card's paragraphs as the server writes them: the runtime's reason (only while chat is available), then
    the scene model's advisory and the recorded-questions hint, each worded as before."""
    return ([f"{reason}."] if reason else []) + [ADVISORY, status_page.RECORDED_HINT]


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width,height", [(1280, 900), (390, 844)])
def test_the_information_key_beside_the_heading_shows_the_explanations_over_the_panel_and_they_fit(
        browser: Browser, served: dict[str, Any], theme: str, width: int, height: int) -> None:
    page = opened(browser, f"{served['url']}/?theme={theme}", width=width, height=height, mobile=width < 500)
    try:
        type_question(page, "Is the door open?")
        to_panel(page)
        closed = about(page)
        # A named key beside the heading, on its line; closed, the explanations are not shown and the availability is
        assert closed["name"] == status_page.CHAT_ABOUT_NAME and closed["controls"] == "chat-about-card"
        assert 0 <= closed["key"]["left"] - closed["heading"]["right"] <= 24
        assert abs(closed["key"]["y"] - closed["heading"]["y"]) <= 4
        assert not closed["open"] and closed["expanded"] == "false"
        assert closed["row"].startswith("Available") and AVAILABLE not in closed["row"]
        for text in wording(AVAILABLE):
            assert text not in closed["body"]
        assert "Experimental" not in closed["row"]  # an admitted chat (this fixture), not a validation run's
        click_key(page)  # the first press after the page opened: its first update may still be moving the panel
        assert wait_until(lambda: is_open(page), 3.0)
        shown = about(page)
        assert shown["expanded"] == "true" and shown["paragraphs"] == wording(AVAILABLE)
        assert shown["shown"].startswith("Ask about the picture the camera is showing now.")  # only shown capitalised
        # Over the panel, under the heading, inside the panel and the window, readable in this palette; nothing moved
        assert shown["on_top"] and shown["card"]["top"] >= shown["key"]["bottom"]
        assert shown["panel"]["left"] - 0.5 <= shown["card"]["left"] and shown["card"]["right"] <= shown["panel"]["right"] + 0.5
        assert shown["card"]["right"] <= width and shown["scroll_width"] <= width
        assert shown["contrast"] >= 4.5, shown["contrast"]
        for name in ("composer", "tools"):  # where they are in the panel (the page itself may scroll meanwhile)
            assert abs((shown[name]["top"] - shown["panel"]["top"]) - (closed[name]["top"] - closed["panel"]["top"])) <= 0.5
        page.save_screenshot(f"ask-about-{theme}-{width}", view=True)
        # Page updates leave it open, with the draft kept
        assert wait_until(lambda: about(page)["sequence"] >= shown["sequence"] + 2, 3 * REFRESH_S + 3.0)
        kept = about(page)
        assert kept["open"] and kept["paragraphs"] == wording(AVAILABLE) and kept["draft"] == "Is the door open?"
        page.key("Escape")
        assert wait_until(lambda: not is_open(page), 2.0) and about(page)["expanded"] == "false"
    finally:
        page.close()


def test_the_information_opens_on_hover_focus_and_press_stays_under_the_pointer_and_closes_when_asked(
        browser: Browser, served: dict[str, Any]) -> None:
    page = opened(browser, f"{served['url']}/", width=1280, height=900)
    try:
        to_panel(page)
        away_x, away_y = 1100, 860  # below the Ask panel, outside the key and the card (checked below)
        # Hover: open while the pointer is over the key, and while it crosses to the card and stays on it
        x, y = key_point(page)
        pointer(page, "mouseMoved", x, y)
        assert wait_until(lambda: is_open(page), 2.0)
        card = about(page)["card"]
        assert away_y > card["bottom"] + 40
        while y < card["top"] + 30:
            y += 6
            pointer(page, "mouseMoved", x, y)
            time.sleep(0.01)
        time.sleep(0.6)  # past the close delay
        assert is_open(page)
        pointer(page, "mouseMoved", away_x, away_y)
        assert wait_until(lambda: not is_open(page), 2.0)
        # A press keeps it open once the pointer has gone; a second press closes it, the pointer still over the key
        click(page, *key_point(page))
        assert wait_until(lambda: is_open(page), 2.0)
        pointer(page, "mouseMoved", away_x, away_y)
        time.sleep(0.6)
        assert is_open(page)
        click(page, *key_point(page))
        assert wait_until(lambda: not is_open(page), 2.0)
        time.sleep(0.6)
        assert not is_open(page)  # still under the pointer: closed as asked
        pointer(page, "mouseMoved", away_x, away_y)
        pointer(page, "mouseMoved", *key_point(page))  # back again: hovering opens it again
        assert wait_until(lambda: is_open(page), 2.0)
        # A press elsewhere closes a card that a press opened
        click(page, *key_point(page))
        elsewhere = centre_of(page, "#h-incidents")
        click(page, *elsewhere)
        assert wait_until(lambda: not is_open(page), 2.0)
        pointer(page, "mouseMoved", 5, 450)
        page.evaluate("document.activeElement && document.activeElement.blur()")
        time.sleep(0.4)
        # Keyboard: focus opens it; Escape closes it and focus stays; Enter keeps it open, Enter again closes it;
        # focus moving on closes it, coming back opens it
        page.key("Tab")
        page.evaluate("document.getElementById('chat-info').focus({preventScroll: true})")
        seen = about(page)
        assert seen["visible_focus"] and wait_until(lambda: is_open(page), 2.0)
        page.key("Escape")
        assert wait_until(lambda: not is_open(page), 2.0) and about(page)["focused"] == "chat-info"
        assert about(page)["expanded"] == "false"
        page.key("Enter")
        assert wait_until(lambda: is_open(page), 2.0)
        page.key("Enter")
        assert wait_until(lambda: not is_open(page), 2.0)
        page.key("Enter")
        assert wait_until(lambda: is_open(page), 2.0)
        page.key("Tab")
        assert wait_until(lambda: not is_open(page), 2.0) and about(page)["focused"] != "chat-info"
        page.key("Tab", shift=True)
        assert about(page)["visible_focus"] and wait_until(lambda: is_open(page), 2.0)
    finally:
        page.close()


def test_on_a_phone_a_tap_opens_the_information_a_tap_closes_it_and_it_fits_the_screen(
        browser: Browser, served: dict[str, Any]) -> None:
    page = opened(browser, f"{served['url']}/", width=390, height=844, mobile=True, touch=True)
    try:
        to_panel(page)
        tap_key(page)
        assert wait_until(lambda: is_open(page), 2.0)
        time.sleep(0.6)  # no pointer to wait for: a tap keeps it open
        seen = about(page)
        assert seen["open"] and seen["on_top"] and seen["paragraphs"] == wording(AVAILABLE)
        assert 0 <= seen["card"]["left"] and seen["card"]["right"] <= 390 and seen["scroll_width"] <= 390
        assert seen["panel"]["left"] - 0.5 <= seen["card"]["left"] and seen["card"]["right"] <= seen["panel"]["right"] + 0.5
        card = about(page)["card"]
        tap(page, card["x"], card["top"] + 20)  # reading it: a tap on its text keeps it
        time.sleep(0.4)
        assert is_open(page)
        tap_key(page)
        assert wait_until(lambda: not is_open(page), 2.0)
        time.sleep(0.4)  # a moment later, not a double tap
        tap_key(page)
        assert wait_until(lambda: is_open(page), 2.0)
        below = about(page)["card"]["bottom"] + 24
        assert below < 844
        tap(page, 30, below)  # a tap elsewhere closes it
        assert wait_until(lambda: not is_open(page), 2.0)
    finally:
        page.close()


def test_a_validation_runs_flag_and_reason_and_every_unavailable_or_stale_state_stay_where_they_belong(
        browser: Browser, served: dict[str, Any]) -> None:
    """The flag and the reasons come from the runtime: a validation run's chat (D77) says so beside its badge and in
    the card; an unavailable chat's reason, and a page that is no longer current, are said outside the card."""
    from sentinel.scene.chat import VALIDATION_NOTE

    server, plain = served["server"], served["server"]._snapshot

    def snapshot() -> dict[str, Any]:  # the runtime's own chat status, as the real runtime adds it
        seen = plain()
        return {**seen, "components": {**seen["components"], "chat": served["chat"]["service"].summary()}}

    server._snapshot = snapshot
    served["chat"]["service"] = served["service"](CHAT_ADMISSION, validation_run=True)
    page = opened(browser, f"{served['url']}/", width=1280, height=900)
    try:
        to_panel(page)
        seen = about(page)
        assert seen["row"].startswith("Available") and "Experimental" in seen["row"]
        assert VALIDATION_NOTE not in seen["body"]
        page.save_screenshot("ask-about-closed-validation-run", view=True)
        page.key("Tab")
        page.evaluate("document.getElementById('chat-info').focus({preventScroll: true})")
        assert wait_until(lambda: is_open(page), 2.0)
        assert about(page)["paragraphs"] == wording(f"{AVAILABLE} ({VALIDATION_NOTE})")
        page.save_screenshot("ask-about-validation-run", view=True)
        page.key("Escape")
        # Unavailable: the reason is said in view, the card has only what does not change, and the flag is gone
        served["chat"]["service"] = served["service"](CHAT_ADMISSION)
        wait_chat(page, lambda s: s["state"] == "unavailable", REFRESH_S + 6.0)
        seen = about(page)
        assert not seen["open"] and seen["row"].startswith("Unavailable") and "Experimental" not in seen["row"]
        assert CHAT_ADMISSION.reason in seen["row"] and CHAT_ADMISSION.reason in seen["body"]
        assert seen["paragraphs"] == wording(None)
        page.save_screenshot("ask-about-unavailable", view=True)
        # Not current: the badge says what was last reported, in view
        served["chat"]["service"] = served["service"](CHAT_ADMISSION, validation_run=True)
        wait_chat(page, lambda s: s["state"] == "available", REFRESH_S + 6.0)
        served["runtime"].answering.clear()
        wait_chat(page, lambda s: PAGE_NOT_CURRENT in s["text"], PAGE_CURRENT_S + REFRESH_S + 1.0)
        seen = about(page)
        assert not seen["open"] and seen["row"].startswith("Was available") and "Experimental" in seen["row"]
        served["runtime"].answering.set()
    finally:
        page.close()
