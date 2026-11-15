"""Session 114, part 4: less explanatory text on a phone, so the live picture and the actions come sooner; in a real
browser (headless Chromium; ``browser_harness``), with synthetic data only.

On a phone (at most 599 px wide) the freshness bar is a short line with its time in either state, NOT CURRENT at once
and with its words; how the page stays current is a closed, labelled disclosure beside it. An incident page keeps its
title, severity, status, the event, the acknowledgment's state and every failure in view; the snapshot note, why there
is no snapshot and the command-line acknowledgment are labelled disclosures, each holding the full text. Disclosures
open from the keyboard and stay as the reader left them through updates. Wider screens keep the full texts as before;
nothing scrolls sideways at 320, 360 or 390 px; recognition's experimental label stays in view.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from browser_harness import Browser, Page, headless_browser, wait_until
from status_fixtures import Camera, Runtime, face_row, incident_store

from sentinel import status_page
from sentinel.status_page import PAGE_CURRENT_S, SNAPSHOT_NONE, SNAPSHOT_NOTE, StatusServer

PHONES = {"390": (390, 844), "360": (360, 780), "320": (320, 640)}
VISIBLE = "(selector => (document.querySelector(selector) || {innerText: ''}).innerText)"
SIDEWAYS = "document.documentElement.scrollWidth > document.documentElement.clientWidth"
DISCLOSURES = """(() => [...document.querySelectorAll('details.narrow-only')].map(d => ({id: d.id, open: d.open,
  shown: getComputedStyle(d).display !== 'none', summary: d.querySelector('summary').innerText.trim(),
  body: d.querySelector('.body').textContent.replace(/\\s+/g, ' ').trim()})))()"""
WIDE_TEXTS = """(() => [...document.querySelectorAll('p.wide-only')].map(p => ({text: p.textContent.replace(/\\s+/g, ' ')
  .trim(), shown: getComputedStyle(p).display !== 'none'})))()"""


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with headless_browser() as connection:
        yield connection


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.setattr(status_page, "REFRESH_S", 1)  # updates every second, so a test sees several
    monkeypatch.setattr(status_page, "DETAIL_REFRESH_S", 1)
    ids = incident_store(tmp_path / "sentinel.db", count=3)  # 0: a snapshot, acknowledged; 1: neither, a failed alert
    runtime = Runtime(faces=[face_row(3, "known", basis="fresh", vote_age_ms=700)])
    camera = Camera()
    stop = camera.run(0.1)
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview)
    server.start()
    try:
        yield {"url": f"http://127.0.0.1:{server.address[1]}", "ids": ids, "runtime": runtime}
    finally:
        stop.set()
        runtime.answering.set()
        assert server.stop(3.0)
        camera.preview.close(2.0)


def opened(browser: Browser, url: str, width: int, height: int, **options: Any) -> Page:
    page = browser.page(width=width, height=height, mobile=width < 600, **options)
    page.navigate(url)
    assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 15.0), "the page did not load"
    assert wait_until(lambda: not page.evaluate("document.documentElement.classList.contains('enter')"), 5.0)
    return page


def text(page: Page, selector: str = "body") -> str:
    return page.evaluate(f"{VISIBLE}({selector!r})")


def flat(value: str) -> str:
    return " ".join(value.split())


# ---------------------------------------------------------------- the dashboard on a phone


@pytest.mark.parametrize("phone", ["390", "320"])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_on_a_phone_the_freshness_bar_is_a_line_and_the_picture_comes_sooner(
        browser: Browser, site: dict[str, Any], phone: str, theme: str) -> None:
    width, height = PHONES[phone]
    page = opened(browser, f"{site['url']}/dashboard?theme={theme}", width, height)
    try:
        bar = flat(text(page, ".freshbar"))
        assert bar.startswith("This page is current · generated ") and bar.endswith("How this page stays current")
        assert "without reloading" not in bar and "less than" not in bar, bar  # the mechanics are behind the disclosure
        seen = page.evaluate(f"""(() => {{ const d = document.getElementById('d-freshness');
          return {{open: d.open, shown: getComputedStyle(d).display, bar: document.querySelector('.freshbar')
          .getBoundingClientRect().height, frame: document.getElementById('frame').getBoundingClientRect().top + scrollY,
          sideways: {SIDEWAYS}}}; }})()""")
        assert seen["open"] is False and seen["shown"] == "block" and not seen["sideways"], seen
        # measured before (684e095's text at d2114ed): the bar 169 / 234 px; the picture's frame from 561 / 736 px
        assert seen["bar"] <= (120 if phone == "390" else 140), seen
        assert seen["frame"] < height - 30, seen  # the picture starts within the first screen, even at 320 x 640
        assert "Experimental · pending validation" in text(page, "#faces")  # recognition says what it is, in view
    finally:
        page.close()


def test_the_disclosure_opens_from_the_keyboard_and_stays_as_left_through_updates(
        browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard?theme=dark", *PHONES["390"])
    try:
        page.key("Tab")  # keyboard modality, then the disclosure's own control
        page.evaluate("document.querySelector('#d-freshness > summary').focus()")
        page.key("Enter")
        assert page.evaluate("document.getElementById('d-freshness').open") is True
        assert "It is shown as current only within" in text(page, ".freshbar")
        before = int(page.evaluate("document.documentElement.dataset.sequence"))
        assert wait_until(lambda: int(page.evaluate("document.documentElement.dataset.sequence")) >= before + 2, 8.0)
        kept = page.evaluate("""({open: document.getElementById('d-freshness').open, focus: document.activeElement ===
          document.querySelector('#d-freshness > summary')})""")
        assert kept == {"open": True, "focus": True}, kept  # updated in place, the reader's choice and focus kept
        page.key(" ")
        assert page.evaluate("document.getElementById('d-freshness').open") is False
    finally:
        page.close()


def test_on_a_phone_not_current_shows_at_once_with_its_words(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard?theme=light", *PHONES["320"])
    try:
        assert "This page is current" in text(page, ".freshbar")
        site["runtime"].answering.clear()  # the runtime stops answering: no update arrives
        assert wait_until(lambda: f"NOT CURRENT: this page has not refreshed for {PAGE_CURRENT_S} s"
                          in text(page, ".freshbar"), PAGE_CURRENT_S + 4.0)
        bar = flat(text(page, ".freshbar"))
        assert "This page is current" not in bar and " · last report " in bar, bar
        assert "NOT CURRENT" in text(page, ".pills") and not page.evaluate(SIDEWAYS)
    finally:
        site["runtime"].answering.set()
        page.close()


def test_wider_screens_keep_every_full_text(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard?theme=dark", 1280, 900)
    try:
        bar = flat(text(page, ".freshbar"))
        assert bar.startswith("This page is current: updated less than ") and "without reloading" in bar, bar
        assert all(not d["shown"] for d in page.evaluate(DISCLOSURES))
        page.navigate(f"{site['url']}/incidents/{site['ids'][1]}?theme=dark")
        assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 15.0)
        shown = flat(text(page))
        for full in (SNAPSHOT_NOTE, SNAPSHOT_NONE, "run `sentinel incident ack --data-dir DIR ID`"):
            assert full in shown, full
        assert all(not d["shown"] for d in page.evaluate(DISCLOSURES))
    finally:
        page.close()


# ---------------------------------------------------------------- incident pages on a phone


@pytest.mark.parametrize("phone", ["390", "360", "320"])
def test_an_incident_on_a_phone_keeps_what_matters_in_view_and_the_rest_one_tap_away(
        browser: Browser, site: dict[str, Any], phone: str) -> None:
    page = opened(browser, f"{site['url']}/incidents/{site['ids'][1]}?theme=dark", *PHONES[phone])
    try:
        shown = flat(text(page))
        for needed in ("Person in restricted zone 'zone-1'", "Critical", "Open", "Not acknowledged",
                       "track 3 stayed in the zone for 1.0 s", "No snapshot", "None was saved for this incident.",
                       "failed", "Last error: needs operator action"):
            assert needed in shown, needed
        for behind in ("sentinel run --snapshots", "sentinel incident ack", "kept for 7 days"):
            assert behind not in shown, behind  # in a closed disclosure, not gone (below)
        disclosures = {d["summary"]: d for d in page.evaluate(DISCLOSURES)}
        assert set(disclosures) == {"Why there is no snapshot", "How to acknowledge it (this page is read only)",
                                    "About snapshots: what is kept, and for how long"}, disclosures
        assert all(d["shown"] and not d["open"] for d in disclosures.values())
        assert SNAPSHOT_NONE in disclosures["Why there is no snapshot"]["body"]
        assert "sentinel incident ack --data-dir DIR ID" in disclosures["How to acknowledge it (this page is read only)"][
            "body"]
        assert disclosures["About snapshots: what is kept, and for how long"]["body"] == SNAPSHOT_NOTE
        assert "Technical provenance: every stored row" in shown and not page.evaluate(SIDEWAYS)
        page.navigate(f"{site['url']}/incidents/{site['ids'][0]}?theme=light")
        assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 15.0)
        shown = flat(text(page))
        assert "Download this snapshot" in shown and "Acknowledged" in shown and "kept for 7 days" not in shown
        assert page.evaluate("document.querySelector('#snapshot img').getBoundingClientRect().width") > 0
        assert not page.evaluate(SIDEWAYS)
    finally:
        page.close()


def test_every_phone_disclosure_holds_the_full_wide_text_and_survives_updates(
        browser: Browser, site: dict[str, Any]) -> None:
    """Nothing is lost on a phone: each paragraph a wide screen shows in full is the body of a disclosure (the
    freshness bar's mechanics, reworded without a current claim, are checked above); an open disclosure stays open
    through the incident page's updates."""
    page = opened(browser, f"{site['url']}/incidents/{site['ids'][1]}?theme=dark", *PHONES["390"])
    try:
        bodies = [d["body"] for d in page.evaluate(DISCLOSURES)]
        wide = page.evaluate(WIDE_TEXTS)
        assert wide and all(not p["shown"] for p in wide)
        assert all(any(p["text"] == body or p["text"] in body for body in bodies) for p in wide), (wide, bodies)
        page.key("Tab")
        page.evaluate("document.querySelector('#d-ack-how > summary').focus()")
        page.key("Enter")
        before = int(page.evaluate("document.documentElement.dataset.sequence"))
        assert wait_until(lambda: int(page.evaluate("document.documentElement.dataset.sequence")) >= before + 2, 8.0)
        assert page.evaluate("document.getElementById('d-ack-how').open") is True
        assert "stop monitoring, run" in text(page, "#d-ack-how")
    finally:
        page.close()


@pytest.mark.parametrize("phone", ["390", "360", "320"])
def test_nothing_scrolls_sideways_on_a_narrow_phone(browser: Browser, site: dict[str, Any], phone: str) -> None:
    page = opened(browser, f"{site['url']}/dashboard?theme=dark", *PHONES[phone])
    try:
        assert not page.evaluate(SIDEWAYS)
        for incident in site["ids"][:2]:
            page.navigate(f"{site['url']}/incidents/{incident}?theme=light")
            assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 15.0)
            time.sleep(0.3)
            assert not page.evaluate(SIDEWAYS), incident  # the no-snapshot frame once stuck out at 320 px
    finally:
        page.close()
