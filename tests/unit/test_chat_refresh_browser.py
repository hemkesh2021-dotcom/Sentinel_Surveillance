"""Follow-ups, "Analyse again", "Try again" and automatic refresh in a real browser (session 116, part 28; headless
Chromium through ``browser_harness``), with a stand-in model and synthetic frames only.

One card per question: asked again it is updated in place, with a bounded list of its earlier results, and says which
picture each result is about (the same held one, or a new one). Automatic refresh is off until started on a card, shows
running or paused with the reason and a Stop control, sends nothing while the tab is hidden, the camera is not fresh
or chat is paused, never overlaps a manual question, and backs off when it is not answered. The intervals are shortened
here (the page's and the server's own constants), so that cycles can be watched.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from browser_harness import Browser, Page, headless_browser, wait_until
from status_fixtures import Camera, Runtime, incident_store

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib mode: the tests' own modules, by name

from sentinel import status_page  # noqa: E402
from sentinel.config import FreshnessConfig  # noqa: E402
from sentinel.media.health import assess_frame  # noqa: E402
from sentinel.scene import chat as chat_module  # noqa: E402
from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer  # noqa: E402
from sentinel.scene.chat import ChatAdmission, ChatRequestError, ChatService  # noqa: E402
from sentinel.status_page import StatusServer  # noqa: E402
from test_status_page_chat_browser import last, opened, press_ask, read, type_question, wait_chat  # noqa: E402

ADMITTED = ChatAdmission(True, "admitted for this test", "a synthetic fixture")
CARD = "document.querySelector('#chat-log > li:last-child')"
AUTO_STATE = f"({CARD}.querySelector('.auto-state') || {{textContent: ''}}).textContent"


class Model:
    """Counts what it is asked, how many at once, and fails on request."""

    def __init__(self) -> None:
        self.questions: list[tuple[str, str | None, bool]] = []
        self.images: list[Any] = []
        self.at_once = 0
        self.most_at_once = 0
        self.fail = False
        self.hold = threading.Event()
        self.hold.set()
        self._lock = threading.Lock()

    def __call__(self, question: Any, image: Any, abort: Any = None, max_tokens: Any = None) -> tuple[str, bool]:
        with self._lock:
            self.questions.append((question.question, question.read_as, question.auto))
            self.images.append(image)
            self.at_once += 1
            self.most_at_once = max(self.most_at_once, self.at_once)
        try:
            self.hold.wait(10.0)
            if self.fail:
                raise ChatRequestError("the model could not be reached (test)")
            return f"Answer number {len(self.questions)}.", False
        finally:
            with self._lock:
                self.at_once -= 1


class Gate:
    def __init__(self) -> None:
        self.open = True

    def allows_new(self) -> tuple[bool, str | None]:
        return self.open, None if self.open else "memory pressure: monitoring comes first"

    def summary(self) -> dict[str, Any]:
        return {}


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with headless_browser() as connection:
        yield connection


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.setattr(status_page, "REFRESH_S", 1)
    monkeypatch.setattr(status_page, "AUTO_REFRESH_MIN_S", 1)
    monkeypatch.setattr(status_page, "AUTO_REFRESH_DEFAULT_S", 1)
    monkeypatch.setattr(status_page, "AUTO_REFRESH_MAX_S", 4)
    monkeypatch.setattr(chat_module, "CHAT_MIN_INTERVAL_S", 0.2)
    monkeypatch.setattr(chat_module, "AUTO_MIN_INTERVAL_S", 0.3)
    incident_store(tmp_path / "sentinel.db", count=1)
    runtime, camera, model, gate = Runtime(), Camera(), Model(), Gate()
    stop = camera.run()
    analyzer = ThreadedSceneAnalyzer(lambda job, image: None, RecentImages(), revision="test")
    analyzer.start()
    started = camera.clock.mono()
    service = ChatService(enabled=True, admission=ADMITTED, scene=analyzer, latest=camera.frames.latest,
                          assess=lambda f: assess_frame(f, camera.stamper.current_stream, camera.clock.mono(), started,
                                                        FreshnessConfig()),
                          clock=camera.clock, run=model, stale_after_s=2.0, gate=gate)  # type: ignore[arg-type]
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview,
                          chat=lambda: service)
    server.start()
    try:
        yield {"url": f"http://127.0.0.1:{server.address[1]}", "runtime": runtime, "camera": camera, "model": model,
               "service": service, "gate": gate, "stop": stop}
    finally:
        stop.set()
        runtime.answering.set()
        model.hold.set()
        assert server.stop(3.0) and analyzer.stop(5.0)


def asked(browser: Browser, served: dict[str, Any], question: str = "Is there a phone on the desk?") -> Page:
    page = opened(browser, f"{served['url']}/", width=1280, height=900)
    type_question(page, question)
    press_ask(page)
    wait_chat(page, lambda s: s["log"] and last(s)["state"] == "answered", 15.0)
    return page


def click(page: Page, selector: str) -> None:
    page.evaluate(f"{CARD}.querySelector({selector!r}).click()")


def card(page: Page) -> dict[str, Any]:
    return page.evaluate(f"""(() => {{ const c = {CARD}; return {{state: c.dataset.state, auto: c.dataset.auto || null,
      text: c.innerText, answer: (c.querySelector('p.answer') || {{}}).textContent || '',
      frame: (c.querySelector('p.frame') || {{}}).textContent || '',
      earlier: [...c.querySelectorAll('details.earlier li')].map((li) => li.textContent),
      buttons: [...c.querySelectorAll('button')].map((b) => b.textContent + (b.disabled ? ' (disabled)' : '')),
      cards: document.querySelectorAll('#chat-log > li').length}}; }})()""")


# ---------------------------------------------------------------- follow-ups and Analyse again


def test_a_follow_up_is_read_against_the_earlier_question_and_the_card_says_how(browser: Browser,
                                                                              served: dict[str, Any]) -> None:
    page = asked(browser, served)
    try:
        time.sleep(0.4)
        type_question(page, "near the water bottle?")
        press_ask(page)
        seen = wait_chat(page, lambda s: len(s["log"]) == 2 and last(s)["state"] == "answered", 15.0)
        assert ("Read as a follow-up to your earlier question: “Is there a phone near the water bottle?”"
                in last(seen)["text"])
        assert served["model"].questions[-1] == ("near the water bottle?", "Is there a phone near the water bottle?",
                                                 False)
    finally:
        page.close()


def test_analyse_again_offers_the_same_picture_or_the_latest_and_updates_the_card_in_place(
        browser: Browser, served: dict[str, Any]) -> None:
    page = asked(browser, served)
    try:
        first = card(page)
        assert first["buttons"][:2] == ["Same picture", "Latest picture"] and "Analyse again:" in first["text"]
        assert first["cards"] == 1 and first["earlier"] == [] and first["auto"] is None
        time.sleep(0.4)
        click(page, "button[data-again=same]")
        assert wait_until(lambda: card(page)["answer"] == "Answer number 2.", 10.0), card(page)
        same = card(page)
        assert same["frame"].startswith("The same picture again (the held one, not a new picture). ")
        assert same["cards"] == 1 and len(same["earlier"]) == 1 and same["earlier"][0].startswith("Answer number 1.")
        assert served["model"].images[1] is served["model"].images[0]  # exactly the same picture was sent
        time.sleep(0.4)
        click(page, "button[data-again=latest]")
        assert wait_until(lambda: card(page)["answer"] == "Answer number 3.", 10.0), card(page)
        latest = card(page)
        assert latest["frame"].startswith("A new picture (the latest), not the one before. ")
        assert served["model"].images[2] is not served["model"].images[0]
        assert latest["cards"] == 1 and [e[:16] for e in latest["earlier"]] == ["Answer number 2.", "Answer number 1."]
        assert read(page)["draft"] == ""  # nothing was typed, and nothing was put in the box
        assert [q[0] for q in served["model"].questions] == ["Is there a phone on the desk?"] * 3  # no retyping
    finally:
        page.close()


def test_a_second_click_while_it_is_being_answered_makes_no_second_question(browser: Browser,
                                                                           served: dict[str, Any]) -> None:
    page = asked(browser, served)
    try:
        time.sleep(0.4)
        served["model"].hold.clear()
        click(page, "button[data-again=latest]")
        assert wait_until(lambda: len(served["model"].questions) == 2, 5.0)
        assert all("(disabled)" in b for b in card(page)["buttons"] if b.startswith(("Same", "Latest", "Start")))
        page.evaluate(f"[...{CARD}.querySelectorAll('button[data-again]')].forEach((b) => "
                      "{ b.disabled = false; b.click(); b.click(); })")  # even forced clicks do nothing
        time.sleep(0.5)
        served["model"].hold.set()
        assert wait_until(lambda: card(page)["answer"] == "Answer number 2.", 10.0)
        time.sleep(0.5)
        assert len(served["model"].questions) == 2 and served["service"].counters["accepted"] == 2
    finally:
        page.close()


def test_a_picture_that_is_no_longer_held_says_so_and_offers_only_the_latest(browser: Browser,
                                                                            served: dict[str, Any]) -> None:
    page = asked(browser, served)
    try:
        time.sleep(0.4)
        served["service"].release_pictures()  # as when it has expired
        click(page, "button[data-again=same]")
        assert wait_until(lambda: card(page)["state"] == "picture_expired", 10.0), card(page)
        gone = card(page)
        assert "Picture no longer held: the picture this question used is no longer held" in gone["text"]
        assert "nothing was analysed" in gone["text"] and "it will be a different picture" in gone["text"]
        assert gone["buttons"] == ["Use the latest picture"] and len(served["model"].questions) == 1
        assert gone["earlier"] == ["Answer number 1. — " + gone["earlier"][0].split(" — ", 1)[1]]  # still shown
        time.sleep(0.4)
        click(page, "button[data-again=latest]")
        assert wait_until(lambda: card(page)["answer"] == "Answer number 2.", 10.0)
        assert card(page)["frame"].startswith("A new picture (the latest), not the one before. ")
    finally:
        page.close()


def test_a_failed_answer_offers_try_again_without_retyping_and_never_touches_a_draft(browser: Browser,
                                                                                   served: dict[str, Any]) -> None:
    served["model"].fail = True
    page = opened(browser, f"{served['url']}/", width=1280, height=900)
    try:
        type_question(page, "Is the door open?")
        press_ask(page)
        seen = wait_chat(page, lambda s: s["log"] and last(s)["state"] == "error", 15.0)
        assert seen["draft"] == "Is the door open?"  # it came back into the box, as before
        assert card(page)["buttons"] == ["Try again"]
        page.evaluate("document.getElementById('chat-q').value = ''")
        type_question(page, "A draft, half writ")
        served["model"].fail = False
        time.sleep(0.4)
        click(page, "button[data-again=latest]")
        assert wait_until(lambda: card(page)["state"] == "answered", 10.0), card(page)
        assert read(page)["draft"] == "A draft, half writ" and card(page)["cards"] == 1  # the draft is untouched
        assert [q[0] for q in served["model"].questions] == ["Is the door open?", "Is the door open?"]
    finally:
        page.close()


# ---------------------------------------------------------------- automatic refresh


def test_auto_refresh_is_off_until_started_updates_one_card_and_stops(browser: Browser, served: dict[str, Any]) -> None:
    page = asked(browser, served)
    try:
        time.sleep(2.5)
        assert len(served["model"].questions) == 1  # off by default: nothing is sent by itself
        assert "Auto-refresh (off): every" in card(page)["text"] and "Start" in card(page)["buttons"]
        click(page, "button[data-auto=start]")
        assert wait_until(lambda: "Auto-refresh running: every 1 s" in page.evaluate(AUTO_STATE), 3.0)
        assert card(page)["auto"] == "running" and "Stop" in card(page)["buttons"]
        assert wait_until(lambda: len(served["model"].questions) >= 8, 30.0)
        seen = card(page)
        assert seen["cards"] == 1  # the same card, never a growing list of replies
        assert len(seen["earlier"]) == 4  # and a bounded history of its earlier results
        assert all(q == ("Is there a phone on the desk?", None, True) for q in served["model"].questions[1:])
        assert served["model"].most_at_once == 1
        click(page, "button[data-auto=stop]")
        assert page.evaluate(AUTO_STATE) == "Auto-refresh stopped." and card(page)["auto"] == "off"
        time.sleep(0.6)
        count = len(served["model"].questions)
        time.sleep(2.5)
        assert len(served["model"].questions) == count and "Start" in card(page)["buttons"]
    finally:
        page.close()


def test_auto_refresh_sends_nothing_while_hidden_stale_or_paused_and_says_why(browser: Browser,
                                                                            served: dict[str, Any]) -> None:
    page = asked(browser, served)
    try:
        click(page, "button[data-auto=start]")
        assert wait_until(lambda: len(served["model"].questions) >= 2, 10.0)
        page.evaluate("Object.defineProperty(document, 'hidden', {configurable: true, get: () => true})")
        assert wait_until(lambda: "paused: this tab is hidden. Nothing is sent meanwhile." in page.evaluate(AUTO_STATE),
                          6.0), page.evaluate(AUTO_STATE)
        assert card(page)["auto"] == "paused"
        time.sleep(0.6)
        count = len(served["model"].questions)
        time.sleep(3.0)
        assert len(served["model"].questions) == count  # no request while hidden
        page.evaluate("Object.defineProperty(document, 'hidden', {configurable: true, get: () => false})")
        assert wait_until(lambda: len(served["model"].questions) > count, 10.0)  # and it resumes by itself
        served["gate"].open = False  # the memory gate holds chat
        assert wait_until(lambda: "paused: chat is paused by the runtime" in page.evaluate(AUTO_STATE), 10.0)
        time.sleep(0.6)
        count = len(served["model"].questions)
        time.sleep(3.0)
        assert len(served["model"].questions) == count
        served["gate"].open = True
        assert wait_until(lambda: len(served["model"].questions) > count, 10.0)
        served["runtime"].video = "stale"  # the camera stops being fresh
        assert wait_until(lambda: "paused: the camera is stale" in page.evaluate(AUTO_STATE), 10.0)
        time.sleep(0.6)
        count = len(served["model"].questions)
        time.sleep(3.0)
        assert len(served["model"].questions) == count
    finally:
        page.close()


def test_a_manual_question_and_an_automatic_one_are_never_in_flight_together(browser: Browser,
                                                                            served: dict[str, Any]) -> None:
    page = asked(browser, served)
    try:
        click(page, "button[data-auto=start]")
        assert wait_until(lambda: len(served["model"].questions) >= 2, 10.0)
        served["model"].hold.clear()  # the next answer takes a while
        time.sleep(0.4)
        type_question(page, "Is the door open?")
        assert wait_until(lambda: not read(page)["ask_disabled"] or served["model"].at_once == 1, 5.0)
        served["model"].hold.set()
        wait_chat(page, lambda s: not s["ask_disabled"], 10.0)
        served["model"].hold.clear()
        press_ask(page)  # a manual question, held at the model
        assert wait_until(lambda: served["model"].questions[-1][0] == "Is the door open?", 5.0)
        time.sleep(2.5)  # automatic cycles are due meanwhile
        assert served["model"].questions[-1][0] == "Is the door open?" and served["model"].at_once == 1
        assert "paused: a question is being answered" in page.evaluate(
            "document.querySelector('#chat-log > li:first-child .auto-state').textContent")
        served["model"].hold.set()
        wait_chat(page, lambda s: len(s["log"]) == 2 and last(s)["state"] == "answered", 10.0)
        assert served["model"].most_at_once == 1
        assert wait_until(lambda: served["model"].questions[-1][2] is True, 10.0)  # then the cycles go on
    finally:
        page.close()


def test_auto_refresh_backs_off_when_it_is_not_answered_up_to_its_limit(browser: Browser, served: dict[str, Any],
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    page = asked(browser, served)
    try:
        monkeypatch.setattr(chat_module, "AUTO_MIN_INTERVAL_S", 3600.0)  # every later automatic question is busy
        served["service"]._last_auto_ns = served["camera"].clock.monotonic_ns()
        click(page, "button[data-auto=start]")
        waits = []
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline and len(waits) < 3:
            text = page.evaluate(AUTO_STATE)
            if "next try is in" in text:
                seconds = int(text.split("next try is in ")[1].split(" s")[0])
                if not waits or waits[-1] != seconds:
                    waits.append(seconds)
            time.sleep(0.1)
        assert waits == [2, 4, 4][:len(waits)] and len(waits) >= 2, waits  # doubled, and bounded by the maximum (4 s)
        assert card(page)["state"] == "busy" and "Try again" in card(page)["buttons"]
        assert card(page)["auto"] == "running" and len(served["model"].questions) == 1  # refused: never at the model
        assert served["service"].counters["busy_auto_interval"] >= 2
    finally:
        page.close()
