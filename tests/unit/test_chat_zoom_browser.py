"""Zooming into a selected region and scanning the visible room, in a real browser (session 116, part 28; headless
Chromium through ``browser_harness``), with a stand-in model and stand-in pictures only.

The viewer selects the region on the held picture (a drag, the arrow keys, the numbers, or a tracked person's own
box); the server cuts it from the held frame's recorded pixels; the card says which pixels of which picture were sent.
A typed request to zoom or scan is answered by pointing to these controls: nothing guesses where to look. The scan is
one card for one held picture, its four parts asked about in turn, and it says what was not analysed.
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
from status_fixtures import PICTURE, Runtime, incident_store

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib mode: the tests' own modules, by name

from sentinel import status_page  # noqa: E402
from sentinel.config import FreshnessConfig  # noqa: E402
from sentinel.contracts import PixelFormat  # noqa: E402
from sentinel.media.clock import FakeClock  # noqa: E402
from sentinel.media.frames import FrameStamper  # noqa: E402
from sentinel.media.health import assess_frame  # noqa: E402
from sentinel.media.jpeg import EncodedJpeg, fit_within  # noqa: E402
from sentinel.scene import chat as chat_module  # noqa: E402
from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer  # noqa: E402
from sentinel.scene.chat import ChatAdmission, ChatService  # noqa: E402
from sentinel.scene.roi import PinnedFrames, PreparedCrop  # noqa: E402
from sentinel.status_page import StatusServer  # noqa: E402
from test_chat_roi import NATIVE, WORK, Person, Picture  # noqa: E402
from test_status_page_chat_browser import last, opened, press_ask, read, type_question, wait_chat  # noqa: E402

ADMITTED = ChatAdmission(True, "admitted for this test", "a synthetic fixture")
CARD = "document.querySelector('#chat-log > li:last-child')"
PANEL_OPEN = "!document.getElementById('zoom-panel').hidden"
NOTE = "document.getElementById('zoom-note').textContent"
SIZE = "document.getElementById('zoom-size').textContent"


class Site:
    def __init__(self, tmp_path: Path) -> None:
        incident_store(tmp_path / "sentinel.db", count=1)
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stamper.connect()
        self.pins = PinnedFrames()
        self.sent: list[tuple[str, Any]] = []
        self.at_once = 0
        self.most_at_once = 0
        self.hold = threading.Event()
        self.hold.set()
        self.gate_open = True
        self.people: tuple[Person, ...] = (Person(61, (0.4, 0.2, 0.6, 0.9)),)
        self.n = 0
        self.analyzer = ThreadedSceneAnalyzer(lambda job, image: None, RecentImages(), revision="test")
        self.analyzer.start()
        started = self.clock.mono()
        site = self

        class Gate:
            def allows_new(self) -> tuple[bool, str | None]:
                return site.gate_open, None if site.gate_open else "memory pressure: monitoring comes first"

            def summary(self) -> dict:
                return {}

        self.service = ChatService(
            enabled=True, admission=ADMITTED, scene=self.analyzer, latest=lambda: None,
            assess=lambda f: assess_frame(f, self.stamper.current_stream, self.clock.mono(), started,
                                          FreshnessConfig()),
            clock=self.clock, run=self.model, stale_after_s=2.0, gate=Gate(),  # type: ignore[arg-type]
            pins=self.pins, encode=self.encode)
        self.runtime = Runtime()
        self.server = StatusServer(0, self.runtime.snapshot, tmp_path / "sentinel.db", chat=lambda: self.service)
        self.server.start()
        self.url = f"http://127.0.0.1:{self.server.address[1]}"
        self.stop = threading.Event()
        threading.Thread(target=self.video, daemon=True).start()

    def video(self) -> None:
        while not self.stop.wait(0.1):  # live video: a new frame every 0.1 s, the fake clock keeping pace
            self.clock.advance(0.1)
            self.n += 1
            frame = self.stamper.stamp(native_width=NATIVE[0], native_height=NATIVE[1], pixel_format=PixelFormat.BGR)
            self.pins.latest(frame, Picture(f"work{self.n}", *WORK), Picture(f"main{self.n}", *NATIVE), self.people)

    def encode(self, image: Any, pixel_format: Any, *, max_width: int, max_height: int, quality: int) -> EncodedJpeg:
        height, width = image.shape[:2]
        size = fit_within(width, height, max_width, max_height)
        if image.name.startswith("work"):  # the held picture shown for selecting: a real (small) JPEG
            return EncodedJpeg(PICTURE, 8, 8)
        return EncodedJpeg(f"crop {image.name} at {size[0]}x{size[1]}".encode(), *size)

    def model(self, question: Any, image: Any, abort: Any = None, max_tokens: Any = None) -> tuple[str, bool]:
        self.at_once += 1
        self.most_at_once = max(self.most_at_once, self.at_once)
        try:
            self.sent.append((question.question, image))
            self.hold.wait(10.0)
            return f"Part answer {len(self.sent)}.", False
        finally:
            self.at_once -= 1

    def close(self) -> None:
        self.stop.set()
        self.hold.set()
        assert self.server.stop(3.0) and self.analyzer.stop(5.0)


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with headless_browser() as connection:
        yield connection


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    monkeypatch.setattr(status_page, "REFRESH_S", 1)
    monkeypatch.setattr(chat_module, "CHAT_MIN_INTERVAL_S", 0.3)
    monkeypatch.setattr(chat_module, "PIN_MIN_INTERVAL_S", 0.3)
    made = Site(tmp_path)
    try:
        yield made
    finally:
        made.close()


def zoom_opened(browser: Browser, site: Site) -> Page:
    page = opened(browser, f"{site.url}/", width=1280, height=900)
    assert wait_until(lambda: page.evaluate("!document.getElementById('zoom-open').disabled"), 5.0)
    page.evaluate("document.getElementById('zoom-open').click()")
    assert wait_until(lambda: page.evaluate(PANEL_OPEN), 8.0), page.evaluate(NOTE)
    return page


def drag(page: Page, start: tuple[float, float], end: tuple[float, float]) -> None:
    """A real pointer drag between two points given in fractions of the held picture as shown."""
    page.evaluate("document.getElementById('zoom-surface').scrollIntoView({block: 'center', behavior: 'instant'})")
    time.sleep(0.2)
    rect = page.evaluate("(() => { const r = document.getElementById('zoom-surface').getBoundingClientRect(); "
                         "return [r.left, r.top, r.width, r.height]; })()")

    def point(fraction: tuple[float, float]) -> dict[str, float]:
        return {"x": rect[0] + fraction[0] * rect[2], "y": rect[1] + fraction[1] * rect[3]}

    page.call("Input.dispatchMouseEvent", {"type": "mouseMoved", **point(start)})
    page.call("Input.dispatchMouseEvent", {"type": "mousePressed", "button": "left", "clickCount": 1, **point(start)})
    page.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "button": "left", "buttons": 1, **point(end)})
    page.call("Input.dispatchMouseEvent", {"type": "mouseReleased", "button": "left", "clickCount": 1, **point(end)})


def card(page: Page) -> dict[str, Any]:
    return page.evaluate(f"""(() => {{ const c = {CARD}; return {{state: c.dataset.state, kind: c.dataset.kind || null,
      text: c.innerText, region: (c.querySelector('p.region') || {{}}).textContent || '',
      tiles: [...c.querySelectorAll('ul.tiles > li')].map((li) => [li.dataset.state, li.textContent]),
      buttons: [...c.querySelectorAll('button')].map((b) => b.textContent),
      cards: document.querySelectorAll('#chat-log > li').length}}; }})()""")


# ---------------------------------------------------------------- zoom: the viewer selects, the server cuts


def test_a_dragged_region_of_the_held_picture_is_asked_about_and_the_card_says_which_pixels(browser: Browser,
                                                                                         site: Site) -> None:
    page = zoom_opened(browser, site)
    try:
        facts = page.evaluate("document.getElementById('zoom-facts').textContent")
        assert "Held picture: the frame received at" in facts and "kept for at most 120 s" in facts
        assert "cut from the main stream's picture (2304 × 1296 recorded pixels)" in facts
        assert page.evaluate("document.getElementById('zoom-ask').disabled") is True  # nothing selected yet
        assert page.evaluate(SIZE) == "Drag on the picture to select a region."
        drag(page, (0.25, 0.25), (0.5, 0.5))
        assert wait_until(lambda: "recorded pixels (main stream)" in page.evaluate(SIZE), 3.0), page.evaluate(SIZE)
        chosen = page.evaluate("['x1','y1','x2','y2'].map((n) => Number(document.getElementById('zoom-' + n).value))")
        assert all(abs(got - want) <= 1 for got, want in zip(chosen, (25, 25, 50, 50))), chosen
        assert page.evaluate("document.getElementById('zoom-ask').disabled") is False
        type_question(page, "What is on the shelf?")
        page.evaluate("document.getElementById('zoom-ask').click()")
        assert wait_until(lambda: card(page)["state"] == "answered", 10.0), card(page)
        seen = card(page)
        (question, image), = site.sent
        assert question == "What is on the shelf?" and isinstance(image, PreparedCrop)
        name = image.jpeg.decode()
        assert name.startswith("crop main") and "[" in name  # cut from a held frame's main picture
        x1, y1, x2, y2 = (int(v) for v in name.split("[")[1].split("]")[0].split(","))
        assert abs(x1 - 576) <= 8 and abs(y1 - 324) <= 8 and abs(x2 - 1152) <= 8 and abs(y2 - 648) <= 8
        assert seen["region"].startswith(f"Region: recorded pixels {x1},{y1} to {x2},{y2} of the main stream's picture "
                                         "(2304 × 1296), cut from that picture and sent as ")
        assert "It shows no more than was recorded." in seen["region"]
        assert "What is on the shelf? (a selected region of the held picture)" in seen["text"]
        assert "Historical: about the frame received at" in seen["text"]
        assert seen["buttons"] == [] and seen["cards"] == 1  # no "same or latest picture" on a region card
        assert page.evaluate(PANEL_OPEN) is False and read(page)["draft"] == ""
        assert page.evaluate("document.getElementById('zoom-canvas').width") == 0  # the held picture is not kept here
    finally:
        page.close()


def test_a_selection_too_small_cannot_be_asked_and_says_why_and_cancel_asks_nothing(browser: Browser,
                                                                                  site: Site) -> None:
    page = zoom_opened(browser, site)
    try:
        drag(page, (0.50, 0.50), (0.515, 0.9))
        assert wait_until(lambda: "too small to analyse (at least 64 on a side)" in page.evaluate(SIZE), 3.0)
        assert page.evaluate("document.getElementById('zoom-ask').disabled") is True
        page.evaluate("document.getElementById('zoom-cancel').click()")
        assert page.evaluate(PANEL_OPEN) is False and page.evaluate(NOTE) == "Cancelled: nothing was asked."
        time.sleep(0.5)
        assert site.sent == [] and page.evaluate("document.querySelectorAll('#chat-log > li').length") == 0
    finally:
        page.close()


def test_the_region_can_be_set_from_the_keyboard_the_numbers_and_a_tracked_persons_own_box(browser: Browser,
                                                                                          site: Site) -> None:
    page = zoom_opened(browser, site)
    try:
        assert page.evaluate("document.activeElement.id") == "zoom-surface"  # focus moves to the picture
        page.key("ArrowRight")  # the arrow keys place and move a region
        page.key("ArrowDown")
        first = page.evaluate("['x1','y1','x2','y2'].map((n) => Number(document.getElementById('zoom-' + n).value))")
        assert first == [37, 37, 67, 67]
        page.key("ArrowRight", shift=True)  # Shift+arrows resize it
        assert page.evaluate("Number(document.getElementById('zoom-x2').value)") == 69
        page.evaluate("(() => { for (const [n, v] of [['x1', 10], ['y1', 20], ['x2', 60], ['y2', 70]]) { const i = "
                      "document.getElementById('zoom-' + n); i.value = v; i.dispatchEvent(new Event('input')); } })()")
        assert "1153 × 649 recorded pixels (main stream)." == page.evaluate(SIZE)  # outward to whole pixels
        buttons = page.evaluate("[...document.querySelectorAll('#zoom-people button')].map((b) => b.textContent)")
        assert buttons == ["Track 61 (the tracker's box)"]
        page.evaluate("document.querySelector('#zoom-people button').click()")
        page.evaluate("document.getElementById('zoom-panel').scrollIntoView({block: 'center', behavior: 'instant'})")
        page.save_screenshot("zoom-panel", full=True)
        assert page.evaluate("['x1','y1','x2','y2'].map((n) => Number(document.getElementById('zoom-' + n).value))") \
            == [40, 20, 60, 90]
        page.evaluate("document.getElementById('zoom-ask').click()")  # no typed question: the default one
        assert wait_until(lambda: card(page)["state"] == "answered", 10.0)
        question, image = site.sent[0]
        assert question == "Describe this area." and "[921,259,1383,1167]" in image.jpeg.decode()  # the tracker's box
    finally:
        page.close()


def test_a_typed_request_to_zoom_or_scan_points_to_the_controls_and_never_reaches_the_model(browser: Browser,
                                                                                         site: Site) -> None:
    page = opened(browser, f"{site.url}/", width=1280, height=900)
    try:
        type_question(page, "can you zoom in the pic?")
        press_ask(page)
        seen = wait_chat(page, lambda s: s["log"] and last(s)["state"] == "answered", 10.0)
        assert "To zoom, press “Zoom into a region” below the question box and select the region" in last(seen)["text"]
        assert "Chat does not guess where to look." in last(seen)["text"]
        assert "enlarging cannot recover what is unreadable" in last(seen)["text"]
        time.sleep(0.5)
        type_question(page, "full room scan")
        press_ask(page)
        seen = wait_chat(page, lambda s: len(s["log"]) == 2 and last(s)["state"] == "answered", 10.0)
        assert "To scan the visible room, press “Scan visible room”" in last(seen)["text"]
        assert "not areas out of view, and not a camera movement" in last(seen)["text"]
        assert site.sent == [] and site.service.counters["recorded_zoom"] == 1
    finally:
        page.close()


# ---------------------------------------------------------------- the scan: one held picture, four parts in turn


def test_a_scan_asks_about_the_four_parts_of_one_held_picture_in_turn_on_one_card(browser: Browser,
                                                                                site: Site) -> None:
    page = opened(browser, f"{site.url}/", width=1280, height=900)
    try:
        assert wait_until(lambda: page.evaluate("!document.getElementById('scan-open').disabled"), 5.0)
        page.evaluate("document.getElementById('scan-open').click()")
        assert wait_until(lambda: page.evaluate("document.querySelectorAll('#chat-log > li').length") == 1, 8.0)
        assert card(page)["kind"] == "scan" and "Stop the scan" in card(page)["buttons"]
        assert read(page)["ask_disabled"] is True  # nothing else is asked while it runs
        assert wait_until(lambda: "Scan ended: 4 of 4 parts analysed." in card(page)["text"], 40.0), card(page)
        seen = card(page)
        page.evaluate(f"{CARD}.scrollIntoView({{block: 'center', behavior: 'instant'}})")
        page.save_screenshot("scan-card", full=True)
        assert [state for state, _ in seen["tiles"]] == ["answered"] * 4 and seen["cards"] == 1
        assert seen["tiles"][0][1] == "top left (recorded pixels 0,0 to 1213,683): Part answer 1."
        assert seen["tiles"][3][1] == "bottom right (recorded pixels 1091,613 to 2304,1296): Part answer 4."
        assert "one held picture" in seen["text"] and "Each part was asked about by itself" in seen["text"]
        assert "not areas out of view, and not a camera movement" in seen["text"]
        assert "Stop the scan" not in seen["buttons"]
        crops = [image.jpeg.decode() for _, image in site.sent]
        frames = {name.split("[")[0] for name in crops}
        assert len(crops) == 4 and len(frames) == 1  # four crops, all cut from the same held frame
        assert [name.split("[")[1].split("]")[0] for name in crops] == [
            "0,0,1213,683", "1091,0,2304,683", "0,613,1213,1296", "1091,613,2304,1296"]
        assert all(name.endswith(" at 480x270") for name in crops)  # each within the model's image size
        assert site.most_at_once == 1 and site.service.counters["regions"] == 4 and site.service.counters["held"] == 1
        assert wait_until(lambda: read(page)["ask_disabled"] is True and read(page)["note"] == "", 3.0)
    finally:
        page.close()


def test_a_scan_that_is_stopped_or_loses_its_picture_says_which_parts_were_not_analysed(browser: Browser,
                                                                                      site: Site) -> None:
    page = opened(browser, f"{site.url}/", width=1280, height=900)
    try:
        assert wait_until(lambda: page.evaluate("!document.getElementById('scan-open').disabled"), 5.0)
        site.hold.clear()  # the first part's answer takes a while
        page.evaluate("document.getElementById('scan-open').click()")
        assert wait_until(lambda: len(site.sent) == 1, 10.0)
        page.evaluate(f"{CARD}.querySelector('button[data-scan=stop]').click()")
        site.hold.set()
        assert wait_until(lambda: "Scan ended: 1 of 4 parts analysed (stopped by you)." in card(page)["text"], 15.0), \
            card(page)
        tiles = card(page)["tiles"]
        assert tiles[0][0] == "answered" and [state for state, _ in tiles[1:]] == ["skipped"] * 3
        assert tiles[1][1] == "top right: not analysed (stopped by you)."
        assert len(site.sent) == 1
        assert wait_until(lambda: page.evaluate("!document.getElementById('scan-open').disabled"), 5.0)
        time.sleep(0.4)
        site.hold.clear()
        page.evaluate("document.getElementById('scan-open').click()")  # a second scan, whose picture then goes away
        assert wait_until(lambda: len(site.sent) == 2, 10.0)
        site.service.release_pictures()
        site.hold.set()
        assert wait_until(lambda: "Scan ended: 1 of 4 parts analysed (the held picture is gone)." in card(page)["text"],
                          20.0), card(page)
        tiles = card(page)["tiles"]
        assert tiles[1][0] == "failed" and "not analysed (the held picture is gone" in tiles[1][1]
        assert [state for state, _ in tiles[2:]] == ["skipped", "skipped"] and len(site.sent) == 2
    finally:
        page.close()


def test_zoom_and_scan_follow_chats_availability(browser: Browser, site: Site) -> None:
    page = opened(browser, f"{site.url}/", width=1280, height=900)
    try:
        assert wait_until(lambda: page.evaluate("!document.getElementById('zoom-open').disabled"), 5.0)
        site.gate_open = False  # the memory gate holds chat: nothing new may be held
        assert wait_until(lambda: page.evaluate("document.getElementById('zoom-open').disabled && "
                                                "document.getElementById('scan-open').disabled"), 8.0)
        site.gate_open = True
        assert wait_until(lambda: page.evaluate("!document.getElementById('zoom-open').disabled"), 8.0)
        site.stop.set()  # the camera stops: within two seconds the newest frame is stale, and it is not pinned
        time.sleep(0.3)
        site.clock.advance(5.0)
        page.evaluate("document.getElementById('zoom-open').click()")
        assert wait_until(lambda: "No picture was held: the camera is not live" in page.evaluate(NOTE), 8.0)
        assert page.evaluate(PANEL_OPEN) is False and site.service.counters["held"] == 0
    finally:
        page.close()
