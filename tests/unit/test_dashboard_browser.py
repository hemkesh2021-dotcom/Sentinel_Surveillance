"""D95 and D96 in a real browser (headless Chromium; ``browser_harness``), with synthetic data only.

D95, recognition on the dashboard: each tracked person's state, keyed by its track through in-place updates (a row
never moves to another track and goes when its track goes), a current match told apart from a retained one, unknown
never shown as a stranger verdict, and the whole panel saying NOT CURRENT when the page stops updating.

D96, the dashboard's shell: the focus view (the picture and recognition only, freshness and the validation label in
view, nothing reloaded, the reader's place and draft kept); the sidebar's rail (saved) and phone drawer (keyboard,
Escape, scrim); theme, palette and density applied in place and saved, an address's own choice kept; filters over the
loaded incidents only, with an honest empty state; every palette's contrast as the browser computes it; reduced motion;
a single entrance; and the selected incident's snapshot never fetched again by a refresh.

Screenshots can be saved for review with SENTINEL_TEST_SCREENSHOTS; they never decide a result.
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
from sentinel.status_page import PAGE_CURRENT_S, StatusServer
from sentinel.web import palettes

TEXT = "document.body.innerText"


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with headless_browser() as connection:
        yield connection


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.setattr(status_page, "REFRESH_S", 1)  # updates every second, so a test sees several
    ids = incident_store(tmp_path / "sentinel.db", count=15)
    runtime = Runtime(faces=[face_row(3, "known", basis="fresh", vote_age_ms=700)])
    camera = Camera()
    stop = camera.run(0.1)
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview)
    server.start()
    try:
        yield {"url": f"http://127.0.0.1:{server.address[1]}", "ids": ids, "runtime": runtime, "camera": camera}
    finally:
        stop.set()
        runtime.answering.set()
        assert server.stop(3.0)
        camera.preview.close(2.0)


def opened(browser: Browser, url: str, **options: Any) -> Page:
    page = browser.page(**options)
    page.navigate(url)
    assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 15.0), "the page did not load"
    return page


def visible_text(page: Page, selector: str) -> str:
    return page.evaluate(f"(document.querySelector({selector!r}) || {{innerText: ''}}).innerText")


# ---------------------------------------------------------------- D95: recognition


def test_recognition_rows_follow_their_tracks_through_in_place_updates(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        assert wait_until(lambda: "Known · current match" in visible_text(page, "#faces"), 5.0)
        panel = visible_text(page, "#faces")
        assert "Track 3" in panel and "an enrolled identity" in panel and "Experimental · pending validation" in panel
        assert "idn-" not in page.evaluate("document.documentElement.outerHTML")  # no identity ID anywhere (D61)
        assert "Newest face result 0.7 s before this report" in panel
        page.evaluate("document.querySelector('#faces li[data-key=\"p-1-3\"]').__testMark = 'track 3'")
        # The match ages into retention on the same track, and another person appears whose faces matched nobody.
        site["runtime"].faces = [face_row(3, "known", basis="retained", vote_age_ms=9400),
                                 face_row(9, "unknown", basis="fresh", vote_age_ms=900)]
        assert wait_until(lambda: "Known · retained" in visible_text(page, "#faces"), 5.0)
        panel = visible_text(page, "#faces")
        assert page.evaluate("document.querySelector('#faces li[data-key=\"p-1-3\"]').__testMark") == "track 3"
        assert "Known · current match" not in panel and "Retained from an earlier match" in panel
        row9 = visible_text(page, "#faces li[data-key='p-1-9']")
        assert "Unknown" in row9 and "Not a stranger verdict" in row9 and "enrolled identity" not in row9
        # Track 3 ends: its row goes with it; the identity never moves to the other track.
        site["runtime"].faces = [face_row(9, "unknown", basis="fresh", vote_age_ms=900)]
        assert wait_until(lambda: page.evaluate("!document.querySelector('#faces li[data-key=\"p-1-3\"]')"), 5.0)
        assert "Known" not in visible_text(page, "#faces")  # nothing known carries over to the other track
        badges = page.evaluate("[...document.querySelectorAll('#faces .badge')].map(b => b.innerText.trim())")
        assert not [b for b in badges if "stranger" in b.lower()]
        site["runtime"].faces = []  # nobody tracked: said so, not left blank
        assert wait_until(lambda: "Nobody is tracked right now" in visible_text(page, "#faces"), 5.0)
    finally:
        page.close()


def test_recognition_says_not_current_when_updates_stop_and_recovers(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        assert wait_until(lambda: "Known · current match" in visible_text(page, "#faces"), 5.0)
        assert "Derived by the runtime at" in visible_text(page, "#faces")
        site["runtime"].answering.clear()  # the runtime stops answering, as over a broken tunnel
        assert wait_until(lambda: "NOT CURRENT: last derived at" in visible_text(page, "#faces"), PAGE_CURRENT_S + 3)
        panel = visible_text(page, "#faces")
        assert "Known · current match" not in panel and "Was a current match" in panel
        assert page.evaluate("parseFloat(getComputedStyle(document.querySelector('#faces .report')).opacity)") < 1
        assert wait_until(lambda: "Reconnecting: no current update" in page.evaluate(
            "document.getElementById('sync').textContent"), 4.0)  # once the stalled request is abandoned
        site["runtime"].answering.set()
        assert wait_until(lambda: "Known · current match" in visible_text(page, "#faces"), 20.0)
        assert "NOT CURRENT" not in visible_text(page, "#faces")
    finally:
        site["runtime"].answering.set()
        page.close()


# ---------------------------------------------------------------- D96: the focus view


def test_the_focus_view_keeps_freshness_the_label_the_picture_and_the_readers_state(
        browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        assert wait_until(lambda: page.evaluate("document.getElementById('player-state').innerText.trim()") == "LIVE", 8)
        page.evaluate("const q = document.getElementById('chat-q'); q.value = 'is the door open?';"
                      "document.getElementById('d-technical').open = true; window.scrollTo({top: 600, behavior: 'instant'})")
        origin = page.evaluate("performance.timeOrigin")
        first = page.evaluate("document.querySelector('#frame figure').dataset.frame")
        page.evaluate("document.getElementById('focus-toggle').click()")
        assert wait_until(lambda: page.evaluate("scrollY") == 0, 3.0)  # a short, smooth scroll to the top
        seen = page.evaluate("""(() => { const shown = id => getComputedStyle(document.getElementById(id)).display !== 'none';
          const inView = el => { const r = el.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight; };
          const canvas = document.querySelector('#frame canvas');
          return {focus: document.documentElement.dataset.focus, pressed: document.getElementById('focus-toggle')
                    .getAttribute('aria-pressed'), shown: ['live', 'faces', 'incidents', 'chat', 'health', 'delivery',
                    'settings'].filter(shown), fresh: inView(document.querySelector('.freshbar')),
                  flag: inView(document.querySelector('#faces .flag')), fit: canvas ? getComputedStyle(canvas).objectFit
                    : null, y: scrollY}; })()""")
        assert seen["focus"] == "live" and seen["pressed"] == "true" and seen["shown"] == ["live", "faces"], seen
        assert seen["fresh"] and seen["flag"] and seen["fit"] == "contain" and seen["y"] == 0, seen
        assert wait_until(lambda: page.evaluate("document.querySelector('#frame figure').dataset.frame") != first, 5.0)
        page.key("Escape")  # leaves the focus view where the reader was, with nothing reset
        assert wait_until(lambda: page.evaluate("scrollY") == 600, 3.0)
        after = page.evaluate("""({focus: document.documentElement.dataset.focus || null, y: scrollY,
          draft: document.getElementById('chat-q').value, open: document.getElementById('d-technical').open,
          origin: performance.timeOrigin, active: document.activeElement.id})""")
        assert after == {"focus": None, "y": 600, "draft": "is the door open?", "open": True, "origin": origin,
                         "active": "focus-toggle"}, after
    finally:
        page.close()


# ---------------------------------------------------------------- D96: the sidebar


def test_the_rail_is_saved_and_its_names_stay_for_assistive_technology(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1280, height=900)
    try:
        page.evaluate("localStorage.clear()")
        page.evaluate("document.getElementById('rail-toggle').click()")
        assert wait_until(lambda: page.evaluate(
            "document.getElementById('sidebar').getBoundingClientRect().width") < 100, 2.0)  # after its transition
        seen = page.evaluate("""({rail: document.documentElement.dataset.rail, expanded: document.getElementById(
          'rail-toggle').getAttribute('aria-expanded'), width: document.getElementById('sidebar').getBoundingClientRect()
          .width, saved: JSON.parse(localStorage.getItem('sentinel.appearance')).rail,
          names: [...document.querySelectorAll('.nav a .label')].map(l => l.textContent)})""")
        assert seen["rail"] == "compact" and seen["expanded"] == "false" and seen["saved"] == "compact", seen
        assert seen["width"] < 100 and seen["names"][:3] == ["Live view", "Recognition", "Incidents"], seen
        page.navigate(f"{site['url']}/dashboard")  # a new page view: the saved rail applies before the first paint
        assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 10.0)
        assert page.evaluate("document.documentElement.dataset.rail") == "compact"
    finally:
        page.evaluate("localStorage.clear()")
        page.close()


def test_the_phone_drawer_is_keyboard_operable_and_gives_focus_back(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=390, height=844, mobile=True)
    try:
        page.evaluate("document.getElementById('menu').focus()")
        page.key("Enter")
        assert wait_until(lambda: page.evaluate("document.documentElement.classList.contains('drawer-open')"), 3.0)
        assert page.evaluate("document.getElementById('menu').getAttribute('aria-expanded')") == "true"
        assert page.evaluate("document.activeElement.dataset.nav") == "live"
        inside = []
        for _ in range(14):  # Tab stays inside the open drawer
            page.key("Tab")
            inside.append(page.evaluate("document.getElementById('sidebar').contains(document.activeElement)"))
        assert all(inside), inside
        page.key("Escape")
        seen = page.evaluate("""({open: document.documentElement.classList.contains('drawer-open'),
          active: document.activeElement.id, scrim: document.getElementById('scrim').hidden,
          visibility: getComputedStyle(document.getElementById('sidebar')).visibility})""")
        assert seen["open"] is False and seen["active"] == "menu" and seen["scrim"] is True, seen
        assert wait_until(lambda: page.evaluate(
            "getComputedStyle(document.getElementById('sidebar')).visibility") == "hidden", 2.0)  # not tabbable
        page.evaluate("document.getElementById('menu').click()")
        assert wait_until(lambda: not page.evaluate("document.getElementById('scrim').hidden"), 2.0)
        page.evaluate("document.getElementById('scrim').click()")
        assert page.evaluate("document.documentElement.classList.contains('drawer-open')") is False
    finally:
        page.close()


# ---------------------------------------------------------------- D96: appearance


def test_appearance_applies_in_place_is_saved_and_an_address_keeps_its_own_choice(
        browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1280, height=900,
                  media={"prefers-color-scheme": "light"})
    try:
        page.evaluate("localStorage.clear()")
        page.evaluate("document.getElementById('chat-q').value = 'kept draft'; "
                      f"document.querySelector('.peek[data-select=\"{site['ids'][2]}\"]').click()")
        origin = page.evaluate("performance.timeOrigin")
        for kind, value in (("theme", "dark"), ("palette", "teal-coconut"), ("density", "compact")):
            page.evaluate(f"document.querySelector('[data-set={kind}][data-value={value}]').click()")
        dark = palettes.tokens("teal-coconut", "dark")["canvas"].lower()
        seen = page.evaluate("""(() => { const root = document.documentElement; return {theme: root.dataset.theme,
          palette: root.dataset.palette, density: root.dataset.density, pref: root.hasAttribute('data-pref'),
          canvas: getComputedStyle(root).getPropertyValue('--canvas').trim().toLowerCase(), url: location.pathname +
          location.search, saved: JSON.parse(localStorage.getItem('sentinel.appearance')), origin: performance.timeOrigin,
          draft: document.getElementById('chat-q').value, selected: [...document.querySelectorAll('.inc-list > li.selected')]
          .length, marked: [...document.querySelectorAll('[data-set][aria-current]')].map(a => a.dataset.value),
          links: [...document.querySelectorAll('a.inc')].map(a => a.getAttribute('href')).filter(h => h.includes('?'))};
          })()""")
        assert (seen["theme"], seen["palette"], seen["density"], seen["pref"]) == ("dark", "teal-coconut", "compact", False)
        assert seen["canvas"] == dark and seen["url"] == "/dashboard" and seen["origin"] == origin, seen
        assert seen["saved"] == {"theme": "dark", "palette": "teal-coconut", "density": "compact"}, seen
        assert seen["draft"] == "kept draft" and seen["selected"] == 1 and seen["links"] == [], seen
        assert sorted(seen["marked"]) == ["compact", "dark", "teal-coconut"]
        time.sleep(2.5)  # updates arrive meanwhile: nothing they bring restores the old appearance
        assert page.evaluate("[...document.querySelectorAll('a.inc')].some(a => a.getAttribute('href').includes('?'))") \
            is False
        page.navigate(f"{site['url']}/incidents/{site['ids'][0]}")  # another page applies the saved choice itself
        assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 10.0)
        assert page.evaluate("[document.documentElement.dataset.theme, document.documentElement.dataset.palette]") == \
            ["dark", "teal-coconut"]
        page.navigate(f"{site['url']}/dashboard?theme=light")  # an address's own choice is kept for that view
        assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 10.0)
        assert page.evaluate("[document.documentElement.dataset.theme, document.documentElement.dataset.palette || null,"
                             " document.documentElement.hasAttribute('data-pref')]") == ["light", None, True]
    finally:
        page.evaluate("localStorage.clear()")
        page.close()


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_every_palette_keeps_its_contrast_and_a_visible_focus_ring_in_the_browser(
        browser: Browser, site: dict[str, Any], mode: str) -> None:
    """What the browser computes, for each palette: body text on the canvas and on a panel at least 7:1, muted text at
    least 4.5:1, and a focus ring of at least 2 px whose colour has at least 3:1 against the panel."""
    contrast = palettes.contrast

    def rgb(text: str) -> str:
        r, g, b = (int(part) for part in text[text.index("(") + 1:text.index(")")].split(",")[:3])
        return f"#{r:02x}{g:02x}{b:02x}"

    page = opened(browser, f"{site['url']}/dashboard?theme={mode}", width=1280, height=900)
    try:
        for preset in palettes.PRESETS:
            page.evaluate(f"document.documentElement.dataset.palette = {preset.key!r}")
            page.evaluate("document.querySelector('[data-set=density][data-value=compact]').focus()")
            seen = page.evaluate("""(() => { const style = e => getComputedStyle(e); const panel = document.querySelector(
              '#settings'); const focused = document.activeElement; return {canvas: style(document.body).backgroundColor,
              ink: style(document.body).color, panel: style(panel).backgroundColor, muted: style(document.querySelector(
              '#settings .note')).color, ring: style(focused).outlineColor, width: parseFloat(style(focused).outlineWidth),
              outline: style(focused).outlineStyle}; })()""")
            ink, canvas, panel = rgb(seen["ink"]), rgb(seen["canvas"]), rgb(seen["panel"])
            assert contrast(ink, canvas) >= 7 and contrast(ink, panel) >= 7, (preset.key, seen)
            assert contrast(rgb(seen["muted"]), panel) >= 4.5, (preset.key, seen)
            assert seen["outline"] != "none" and seen["width"] >= 2 and contrast(rgb(seen["ring"]), panel) >= 3, seen
            page.save_screenshot(f"palette-{preset.key}-{mode}")
    finally:
        page.close()


# ---------------------------------------------------------------- D96: incidents


def test_filters_cover_only_the_loaded_incidents_and_say_so(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1280, height=900)
    try:
        def choose(status: str, severity: str) -> dict[str, Any]:
            page.evaluate(f"""(() => {{ const s = document.getElementById('f-status'), v = document.getElementById(
              'f-severity'); s.value = {status!r}; v.value = {severity!r}; s.dispatchEvent(new Event('change')); }})()""")
            return page.evaluate("""({shown: [...document.querySelectorAll('.inc-list > li')].filter(li => !li.hidden)
              .length, count: document.getElementById('f-count').textContent, empty: !document.getElementById(
              'f-empty').hidden, meta: document.querySelector('[data-region=incidents-meta]').innerText})""")

        everything = choose("", "")
        assert everything["shown"] == 15 and everything["count"] == "15 loaded incidents (the latest only)."
        assert "15 unresolved in the database; the latest 15 of at most 20 are loaded here" in everything["meta"]
        assert choose("acknowledged", "")["shown"] == 1 and choose("", "critical")["shown"] == 1
        none = choose("acknowledged", "critical")
        assert none["shown"] == 0 and none["empty"], none
        assert none["count"] == "Showing 0 of the 15 loaded incidents (the latest only; older ones are not searched)."
        time.sleep(2.5)  # updates re-render the list: the choice stays applied to what they bring
        kept = page.evaluate("""({status: document.getElementById('f-status').value, shown: [...document.querySelectorAll(
          '.inc-list > li')].filter(li => !li.hidden).length})""")
        assert kept == {"status": "acknowledged", "shown": 0}
        assert choose("open", "warning")["shown"] == 13
    finally:
        page.close()


def test_the_selected_incidents_snapshot_is_not_fetched_again_by_a_refresh(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1280, height=900)
    try:
        page.evaluate(f"document.querySelector('.peek[data-select=\"{site['ids'][0]}\"]').click()")
        assert wait_until(lambda: page.evaluate("!!document.querySelector('#inc-preview .frame.thumb img')"), 5.0)
        page.evaluate("document.querySelector('#inc-preview img').__testMark = 'kept'")
        time.sleep(11)  # one refresh of the selected incident (every 10 s)
        seen = page.evaluate("""({mark: (document.querySelector('#inc-preview img') || {}).__testMark || null,
          fetched: performance.getEntriesByType('resource').filter(e => e.name.endsWith('/snapshot.jpg')).length})""")
        assert seen == {"mark": "kept", "fetched": 1}, seen
    finally:
        page.close()


# ---------------------------------------------------------------- D96: motion


def test_one_entrance_only_and_reduced_motion_removes_motion_not_the_watchdog(
        browser: Browser, site: dict[str, Any]) -> None:
    for preference, entrance, smooth in (("no-preference", "enter", "smooth"), ("reduce", "none", "auto")):
        page = browser.page(width=1280, height=900, media={"prefers-reduced-motion": preference})
        try:
            page.navigate(f"{site['url']}/dashboard")
            assert wait_until(lambda: page.evaluate("document.readyState") != "loading", 10.0)
            seen = page.evaluate("""({entering: document.documentElement.classList.contains('enter'),
              name: getComputedStyle(document.getElementById('faces')).animationName,
              scroll: getComputedStyle(document.documentElement).scrollBehavior,
              watchdog: getComputedStyle(document.querySelector('#faces .report')).animationName})""")
            assert seen["entering"] and seen["name"] == entrance and seen["scroll"] == smooth, (preference, seen)
            assert seen["watchdog"].startswith("full"), seen  # the freshness timer runs either way
            assert wait_until(lambda: not page.evaluate("document.documentElement.classList.contains('enter')"), 3.0)
            time.sleep(1.5)  # updates arrive: no panel enters again
            assert page.evaluate("getComputedStyle(document.getElementById('faces')).animationName") == "none"
        finally:
            page.close()
