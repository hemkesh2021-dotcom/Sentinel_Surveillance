"""D75: the live player in a real browser (headless Chromium; synthetic pictures, no camera).

The real StatusServer serves a camera stand-in on a FakeClock that keeps pace with real time (``status_fixtures``);
the DevTools Fetch and Network domains hold, fail or slow requests without changing the server. These are portable
regressions of the player's behaviour on synthetic input; they measure nothing about the camera, the device, the
SSH tunnel or Safari.

Covered: pictures arrive continuously at the source's rate, apart from the status updates; a frozen stream stops
claiming LIVE within its own window while status updates (and other successful requests) keep arriving; a slow viewer
gets fewer pictures, never older ones; one request at a time; a runtime restart's new token is taken from the page's
own update; delivered rate and age are measured and shown apart; pictures are not asked for out of sight; reduced
motion keeps the picture's watchdog.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from browser_harness import Browser, DevToolsError, Page, headless_browser, wait_until
from status_fixtures import PICTURE, Camera, Runtime

from sentinel.status_page import REFRESH_S, StatusServer
from sentinel.storage.database import Database

STALE_AFTER_S = 2.0
PROBE = """(() => {
  const slot = document.getElementById('frame');
  const figure = slot && slot.querySelector('figure.frame');
  const pic = figure && figure.querySelector('.pic');
  let opacity = null;
  if (pic) { opacity = 1; for (let node = pic; node; node = node.parentElement) opacity *= parseFloat(getComputedStyle(node).opacity); }
  const player = document.getElementById('player-state');
  const facts = document.getElementById('player-facts');
  return {player: player ? player.innerText.trim() : '', text: document.body.innerText,
          frame: figure ? figure.dataset.frame || null : null, state: figure ? figure.dataset.state || null : null,
          drawn: !!(pic && pic.tagName === 'CANVAS' && pic.dataset.drawn === '1'), opacity,
          fps: Number(slot.dataset.fps || 0), ageSent: Number(slot.dataset.ageSentMs || -1),
          ageBound: Number(slot.dataset.ageBoundMs || -1), ageBoundMax: Number(slot.dataset.ageBoundMaxMs || -1),
          facts: facts && !facts.hidden ? facts.textContent : ''};  // behind 'Picture delivery' (D96), closed
})()"""


def padded(jpeg: bytes, size: int) -> bytes:
    """``jpeg`` grown to about ``size`` bytes with comment segments, which decoders skip: still a valid picture."""
    segments, left = b"", size - len(jpeg)
    while left > 0:
        chunk = min(left, 65000)
        segments += b"\xff\xfe" + (chunk + 2).to_bytes(2, "big") + b"x" * chunk
        left -= chunk + 4
    return jpeg[:2] + segments + jpeg[2:]


class Counting:
    """The preview, with the number of picture requests under way at once recorded, and each picture it hands out
    (when, on the test's monotonic clock, and its frame sequence)."""

    def __init__(self, camera: Camera) -> None:
        self.camera, self.now, self.most, self.lock = camera, 0, 0, threading.Lock()
        self.served: list[tuple[float, int]] = []
        original = camera.preview.next_frame

        def next_frame(*args: Any, **kwargs: Any) -> Any:
            with self.lock:
                self.now += 1
                self.most = max(self.most, self.now)
            try:
                answer = original(*args, **kwargs)
                if answer.picture is not None:
                    with self.lock:
                        self.served.append((time.monotonic(), answer.picture.order[1]))
                return answer
            finally:
                with self.lock:
                    self.now -= 1

        camera.preview.next_frame = next_frame  # type: ignore[method-assign]

    def served_by(self, moment: float) -> list[int]:
        with self.lock:
            return [seq for at, seq in self.served if at <= moment]


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with headless_browser() as connection:
        yield connection


@contextlib.contextmanager
def serving(tmp_path: Path, *, fps: float = 15.0, picture: bytes = PICTURE) -> Iterator[dict[str, Any]]:
    Database.open(tmp_path / "sentinel.db").close()
    runtime, camera = Runtime(), Camera(picture=picture)
    counting = Counting(camera)
    stop = camera.run(1 / fps)
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview)
    server.start()
    try:
        yield {"url": f"http://127.0.0.1:{server.address[1]}/", "runtime": runtime, "camera": camera, "stop": stop,
               "server": server, "counting": counting}
    finally:
        stop.set()
        runtime.answering.set()
        assert server.stop(3.0)
        camera.preview.close(2.0)


def read(page: Page) -> dict[str, Any]:
    return page.evaluate(PROBE)


def wait_for(page: Page, predicate: Callable[[dict[str, Any]], bool], timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    seen = None
    while time.monotonic() < deadline:
        with contextlib.suppress(DevToolsError):
            seen = read(page)
            if predicate(seen):
                return seen
        time.sleep(0.03)
    raise AssertionError(f"the player never reached the expected state; last seen: {seen!r}")


def live(seen: dict[str, Any]) -> bool:
    return seen["player"] == "LIVE" and seen["drawn"] and seen["state"] == "live" and seen["opacity"] == 1


def not_live(seen: dict[str, Any]) -> bool:
    return seen["player"] != "LIVE" and seen["drawn"] and (seen["opacity"] or 1) < 0.5


def opened(browser: Browser, url: str, **options: Any) -> Page:
    page = browser.page(**options)
    page.navigate(url)
    wait_for(page, live, 30.0)
    return page


def test_pictures_arrive_at_the_sources_rate_apart_from_the_status_updates(browser: Browser, tmp_path: Path) -> None:
    with serving(tmp_path, fps=15.0) as served:
        page = opened(browser, served["url"])
        try:
            statuses = served["runtime"].served
            shown, started = set(), time.monotonic()
            while time.monotonic() - started < 3.0:
                seen = read(page)
                assert seen["drawn"] and live(seen), seen
                shown.add(seen["frame"])
                time.sleep(0.02)
            assert len(shown) >= 3.0 * 15 * 0.6  # synthetic 15 fps: most frames shown (portable, not a device figure)
            assert served["runtime"].served - statuses <= 3.0 / REFRESH_S + 1  # status updates keep their own pace
        finally:
            page.close()


def test_a_frozen_stream_stops_claiming_live_within_its_window_while_other_answers_keep_arriving(
        browser: Browser, tmp_path: Path) -> None:
    with serving(tmp_path) as served:
        page = opened(browser, served["url"])
        try:
            page.call("Fetch.enable", {"patterns": [{"urlPattern": "*/live/frame*", "requestStage": "Request"}]})
            frozen = time.monotonic()
            statuses = served["runtime"].served
            # Other requests keep succeeding meanwhile: a status update, and the page's own JSON.
            page.evaluate("fetch('/status.json').then(r => r.status)")
            seen = wait_for(page, not_live, STALE_AFTER_S + 1.0)
            assert time.monotonic() - frozen <= STALE_AFTER_S + 1.0
            assert "NOT CURRENT: no newer picture has arrived in time" in seen["text"]
            assert seen["player"] == "NOT CURRENT: no newer picture in time"
            time.sleep(REFRESH_S + 0.5)
            seen = read(page)
            assert served["runtime"].served > statuses and "This page is current" in seen["text"]  # status updates
            assert not_live(seen)  # ... restarted nothing of the picture's
            page.call("Fetch.disable")
            assert live(wait_for(page, live, 10.0))
        finally:
            page.close()


def test_a_slow_viewer_gets_fewer_pictures_never_older_ones(browser: Browser, tmp_path: Path) -> None:
    with serving(tmp_path, picture=padded(PICTURE, 60_000)) as served:
        page = opened(browser, served["url"])
        try:
            page.call("Network.enable")
            page.call("Network.emulateNetworkConditions", {"offline": False, "latency": 40,
                                                           "downloadThroughput": 150_000, "uploadThroughput": 150_000})
            time.sleep(1.5)  # what was under way when the link slowed has arrived
            counting, frames, shown = served["counting"], set(), []
            started = time.monotonic()
            while time.monotonic() - started < 6.0:
                before = counting.served_by(time.monotonic())  # what the server had handed out before this look
                seen = read(page)
                if seen["frame"]:
                    frames.add(seen["frame"])
                    shown.append((int(seen["frame"].split("-")[1]), before))
                time.sleep(0.05)
            # About 2.5 pictures a second fit through this link (60 kB each at 150 kB/s): fewer pictures than the
            # camera's 15 a second, never an older one, and never falling behind what the link delivers. The player
            # asks for the next picture once the last one's bytes have arrived and decodes the newest (D75), so when
            # the server has handed out pictures ..., p[k-1], p[k], p[k-1] has arrived: the page shows p[k-2] or
            # newer unless decoding this small picture took longer than a whole transfer of the next one. Counted in
            # the player's own pictures, so neither the camera's rate nor the link's speed sets the bound. That each
            # picture handed out is the newest, never a backlog, is the server's (test_live_stream.py).
            assert 6 <= len(frames) <= 6 * 6
            sequence = [seq for seq, _ in shown]
            assert sequence == sorted(sequence)  # never an older picture than one already shown
            served_all = set(counting.served_by(time.monotonic()))
            assert all(seq in served_all for seq in sequence)  # only pictures the server handed out
            behind = [(seq, before[-3]) for seq, before in shown if len(before) >= 3 and seq < before[-3]]
            assert not behind, behind  # (shown, the picture two before the newest handed out) when it fell behind
        finally:
            with contextlib.suppress(DevToolsError):
                page.call("Network.emulateNetworkConditions", {"offline": False, "latency": 0,
                                                               "downloadThroughput": -1, "uploadThroughput": -1})
            page.close()


def test_the_player_asks_for_one_picture_at_a_time(browser: Browser, tmp_path: Path) -> None:
    with serving(tmp_path) as served:
        page = opened(browser, served["url"])
        try:
            time.sleep(3.0)
            assert served["counting"].most == 1
        finally:
            page.close()


def test_after_a_restart_the_player_takes_the_new_token_from_the_pages_own_update(
        browser: Browser, tmp_path: Path) -> None:
    with serving(tmp_path) as served:
        page = opened(browser, served["url"])
        try:
            served["server"].chat_token = "f" * 32  # what a restarted runtime's server would hand out
            seen = wait_for(page, lambda s: not live(s), 6.0)  # refused with the old token
            assert seen["frame"]
            assert live(wait_for(page, live, 3 * REFRESH_S + 3))  # the next status update carried the new token
        finally:
            page.close()


def test_delivered_rate_and_age_are_measured_and_shown_apart(browser: Browser, tmp_path: Path) -> None:
    with serving(tmp_path) as served:
        page = opened(browser, served["url"])
        try:
            seen = wait_for(page, lambda s: s["fps"] > 0 and s["facts"], 4.0)
            assert 5 <= seen["fps"] <= 30  # synthetic 15 fps
            assert 0 <= seen["ageSent"] <= seen["ageBound"] <= seen["ageBoundMax"]
            for words in ("Delivered", "pictures/s", "Age when sent", "on the device's clock", "Age when shown",
                          "Asking to arrival", "Decoding", "the camera's capture time is not known"):
                assert words.lower() in seen["facts"].lower(), words  # the labels are styled in capitals
        finally:
            page.close()


def test_pictures_are_not_asked_for_out_of_sight(browser: Browser, tmp_path: Path) -> None:
    with serving(tmp_path) as served:
        page = opened(browser, served["url"], width=900, height=500)
        preview = served["camera"].preview
        try:
            page.evaluate("document.body.style.minHeight = '6000px'; window.scrollTo(0, 5000)")
            time.sleep(0.6)
            asked = preview.summary()["counters"]["stream_requests"]
            time.sleep(2.0)
            assert preview.summary()["counters"]["stream_requests"] <= asked + 1
            assert read(page)["player"].startswith("PAUSED")
            page.evaluate("window.scrollTo(0, 0)")
            assert live(wait_for(page, live, 3.0))
        finally:
            page.close()


def test_reduced_motion_keeps_the_pictures_watchdog(browser: Browser, tmp_path: Path) -> None:
    with serving(tmp_path) as served:
        page = opened(browser, served["url"], media={"prefers-reduced-motion": "reduce"})
        try:
            page.call("Fetch.enable", {"patterns": [{"urlPattern": "*/live/frame*", "requestStage": "Request"}]})
            frozen = time.monotonic()
            wait_for(page, not_live, STALE_AFTER_S + 1.0)
            assert time.monotonic() - frozen <= STALE_AFTER_S + 1.0
            page.call("Fetch.disable")
        finally:
            page.close()
