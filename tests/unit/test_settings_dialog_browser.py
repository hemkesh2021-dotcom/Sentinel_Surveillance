"""The Settings dialog in a real browser (session 116, part 28; headless Chromium through ``browser_harness``), with a
stand-in runtime and synthetic data only.

With script, the sidebar's Appearance and Features open one compact modal dialog with that section chosen, instead of
scrolling to a section down the page; the two sections are the page's own elements, moved (one set of controls).
Keyboard: focus starts on the chosen section, stays inside, Escape and the close button close it and focus returns to
what opened it. Updates go on while it is open and never touch a typed key, the unlocked state or its scroll position.
Appearance stays this browser's and calls no feature route; closing changes no feature. NOT CURRENT and a degraded
runtime stay in view inside it. Without script the dialog never opens and both sections stay on the page.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from browser_harness import Browser, Page, headless_browser, wait_until
from status_fixtures import Runtime, incident_store

from sentinel import status_page
from sentinel.features import FeatureControls
from sentinel.media.clock import FakeClock
from sentinel.status_page import PAGE_CURRENT_S, StatusServer, build_status, new_control_key, render_html

OPEN = "document.getElementById('settings-dialog').open"
SECTION = "document.getElementById('settings-dialog').dataset.section"
ACTIVE = ("(() => { const a = document.activeElement; return a ? (a.id || a.dataset.sd || a.dataset.nav || a.tagName) "
          ": null; })()")
INSIDE = "document.getElementById('settings-dialog').contains(document.activeElement)"


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with headless_browser() as connection:
        yield connection


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.setattr(status_page, "REFRESH_S", 1)
    incident_store(tmp_path / "sentinel.db", count=2)
    controls = FeatureControls(FakeClock(), {"face": True, "scene": True, "live_view": True})
    runtime = Runtime(controls=controls)
    key = new_control_key()
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db")
    server.enable_controls(lambda: controls, key)
    server.start()
    try:
        yield {"url": f"http://127.0.0.1:{server.address[1]}", "runtime": runtime, "controls": controls, "key": key,
               "server": server}
    finally:
        runtime.answering.set()
        assert server.stop(3.0)


def opened(browser: Browser, url: str, **options: Any) -> Page:
    page = browser.page(**options)
    page.navigate(url)
    assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 15.0), "the page did not load"
    assert wait_until(lambda: page.evaluate("document.documentElement.classList.contains('sd')"), 5.0)
    return page


def click_nav(page: Page, target: str) -> None:
    page.evaluate(f"(() => {{ const a = document.querySelector('#sidebar .nav a[data-nav={target}]'); a.focus(); "
                  "a.click(); })()")


def visible(page: Page, selector: str) -> bool:
    return page.evaluate(f"(() => {{ const n = document.querySelector({selector!r}); if (!n) return false; "
                         "const r = n.getBoundingClientRect(); return n.offsetParent !== null && r.width > 0 && "
                         "r.height > 0; })()")


def requests_seen(page: Page) -> list[str]:
    return page.evaluate("performance.getEntriesByType('resource').map(e => new URL(e.name).pathname)")


# ---------------------------------------------------------------- both entry points, and the section each selects


def test_appearance_and_features_each_open_the_dialog_on_their_own_section_without_scrolling(
        browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        assert page.evaluate(OPEN) is False
        assert not visible(page, "#settings") and not visible(page, "#features")  # no large sections down the page
        page.evaluate("window.scrollTo(0, 300)")
        before = page.evaluate("window.scrollY")
        click_nav(page, "settings")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        assert page.evaluate(SECTION) == "settings"
        assert visible(page, "#settings [data-set=theme]") and not visible(page, "#features")
        assert page.evaluate("window.scrollY") == before and page.evaluate("location.hash") == ""
        assert page.evaluate("document.querySelector('#settings-dialog a[data-sd=settings]')"
                             ".getAttribute('aria-current')") == "true"
        assert page.evaluate("document.querySelector('#sidebar a[data-nav=settings]').getAttribute('aria-current')") \
            == "true"
        page.evaluate("document.getElementById('sd-close').click()")
        assert wait_until(lambda: page.evaluate(OPEN) is False, 3.0)
        click_nav(page, "features")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        assert page.evaluate(SECTION) == "features"
        assert visible(page, "#features button.switch[data-feature=face]") and not visible(page, "#settings")
        assert page.evaluate("document.getElementById('sd-panel').getAttribute('aria-labelledby')") == "h-features"
        box = page.evaluate("(() => { const r = document.getElementById('settings-dialog').getBoundingClientRect(); "
                            "return [r.width, r.height, r.left, r.top]; })()")
        assert box[0] <= 760 and box[1] <= 640 and box[2] > 100 and box[3] > 50  # compact, contained, centred
        # one set of controls: the page's own elements, not copies
        assert page.evaluate("document.querySelectorAll('#features').length") == 1
        assert page.evaluate("document.querySelectorAll('[data-region=features]').length") == 1
        assert page.evaluate("document.querySelectorAll('#control-key').length") == 1
        assert page.evaluate("document.getElementById('settings-dialog').contains(document.getElementById('features'))")
    finally:
        page.close()


def test_the_sections_switch_inside_the_dialog_by_click_and_by_arrow_keys(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        click_nav(page, "settings")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        assert page.evaluate(ACTIVE) == "settings"  # initial focus: the chosen section's entry
        page.key("ArrowDown")
        assert page.evaluate(SECTION) == "features" and page.evaluate(ACTIVE) == "features"
        assert visible(page, "#features") and not visible(page, "#settings")
        page.key("ArrowUp")
        assert page.evaluate(SECTION) == "settings"
        page.evaluate("document.querySelector('#settings-dialog a[data-sd=features]').click()")
        assert page.evaluate(SECTION) == "features" and page.evaluate(OPEN) is True
        assert page.evaluate("location.hash") == ""  # the entry is a link for assistive technology; it never navigates
    finally:
        page.close()


def test_an_address_ending_in_the_section_opens_the_dialog_and_closing_clears_it(browser: Browser,
                                                                              site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard#features", width=1440, height=900)
    try:
        assert wait_until(lambda: page.evaluate(OPEN) is True, 10.0)
        assert page.evaluate(SECTION) == "features"
        page.key("Escape")
        assert wait_until(lambda: page.evaluate(OPEN) is False, 3.0)
        assert page.evaluate("location.hash") == ""
        page.evaluate("location.hash = '#settings'")
        assert wait_until(lambda: page.evaluate(OPEN) is True, 3.0) and page.evaluate(SECTION) == "settings"
    finally:
        page.close()


# ---------------------------------------------------------------- keyboard: containment, Escape, close, focus return


def test_focus_stays_inside_and_returns_to_the_opener_on_escape_and_on_close(browser: Browser,
                                                                          site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        click_nav(page, "features")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        seen = set()
        for _ in range(40):  # forward through every control and around
            page.key("Tab")
            assert page.evaluate(INSIDE), page.evaluate(ACTIVE)
            seen.add(page.evaluate(ACTIVE))
        assert {"sd-close", "control-key", "features"} <= seen  # the close button, the key field, the section entry
        for _ in range(40):
            page.key("Tab", shift=True)
            assert page.evaluate(INSIDE), page.evaluate(ACTIVE)
        page.evaluate("document.querySelector('#settings-dialog a[data-sd=features]').focus()")
        page.key("Tab", shift=True)  # backwards from the first control: wraps to the last, still inside
        assert page.evaluate(INSIDE)
        page.key("Escape")
        assert wait_until(lambda: page.evaluate(OPEN) is False, 3.0)
        assert page.evaluate(ACTIVE) == "features"  # the sidebar's Features link, which opened it
        assert page.evaluate("document.activeElement.closest('#sidebar') !== null")
        click_nav(page, "settings")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        page.evaluate("document.getElementById('sd-close').focus()")
        page.key("Enter")
        assert wait_until(lambda: page.evaluate(OPEN) is False, 3.0)
        assert page.evaluate(ACTIVE) == "settings" and page.evaluate("document.activeElement.closest('#sidebar') !== null")
    finally:
        page.close()


def test_escape_closes_the_dialog_only_and_leaves_the_focus_view_as_it_was(browser: Browser,
                                                                        site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        page.evaluate("document.getElementById('focus-toggle').click()")
        assert page.evaluate("document.documentElement.dataset.focus") == "live"
        click_nav(page, "settings")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        page.key("Escape")
        assert wait_until(lambda: page.evaluate(OPEN) is False, 3.0)
        assert page.evaluate("document.documentElement.dataset.focus") == "live"
    finally:
        page.close()


# ---------------------------------------------------------------- updates while editing


def test_updates_while_the_key_is_being_typed_keep_the_text_the_focus_the_scroll_and_the_unlocked_state(
        browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1000, height=520)
    try:
        click_nav(page, "features")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        page.evaluate("(() => { const k = document.getElementById('control-key'); k.focus(); k.value = 'ABCD-EF'; })()")
        page.evaluate("document.getElementById('sd-panel').scrollTop = 120")
        scroll = page.evaluate("document.getElementById('sd-panel').scrollTop")
        assert scroll > 0
        sequence = page.evaluate("document.documentElement.dataset.sequence || ''")
        site["runtime"].degraded = ["capture stale (no new frame)"]  # the next update differs
        assert wait_until(lambda: "degraded" in page.evaluate(
            "document.querySelector('[data-region=settings-status]').innerText"), 6.0), sequence
        assert page.evaluate("document.getElementById('control-key').value") == "ABCD-EF"
        assert page.evaluate(ACTIVE) == "control-key"
        assert page.evaluate("document.getElementById('sd-panel').scrollTop") == scroll
        assert page.evaluate(OPEN) is True and page.evaluate(SECTION) == "features"
        # unlocked, then several more updates, a switch to Appearance and back, and a close and reopen: still unlocked
        page.evaluate(f"document.getElementById('control-key').value = {site['key']!r}")
        page.evaluate("document.getElementById('controls-unlock').requestSubmit()")
        assert wait_until(lambda: page.evaluate(
            "!document.querySelector('#features button.switch[data-feature=face]').disabled"), 5.0)
        site["runtime"].degraded = []
        assert wait_until(lambda: "degraded" not in page.evaluate(
            "document.querySelector('[data-region=settings-status]').innerText"), 6.0)
        page.evaluate("document.querySelector('#settings-dialog a[data-sd=settings]').click()")
        page.evaluate("document.querySelector('#settings-dialog a[data-sd=features]').click()")
        page.key("Escape")
        assert wait_until(lambda: page.evaluate(OPEN) is False, 3.0)
        click_nav(page, "features")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        assert page.evaluate("!document.querySelector('#features button.switch[data-feature=face]').disabled")
        assert page.evaluate("document.getElementById('controls-unlock').hidden") is True
    finally:
        page.close()


def test_a_wrong_key_is_refused_in_the_dialog_with_the_existing_message(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        click_nav(page, "features")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        page.evaluate("document.getElementById('control-key').value = 'AAAA-BBBB-CCCC-DDDD-EEEE-FFFF'")
        page.evaluate("document.getElementById('controls-unlock').requestSubmit()")
        assert wait_until(lambda: "Not unlocked: the control key is not this run's" in page.evaluate(
            "document.getElementById('controls-note').textContent"), 5.0)
        assert visible(page, "#controls-note")
        assert page.evaluate("[...document.querySelectorAll('#features button.switch')].every(b => b.disabled)")
        assert site["controls"].view()["log"] == []
    finally:
        page.close()


# ---------------------------------------------------------------- appearance is this browser's; closing changes nothing


def test_appearance_changes_call_no_feature_route_and_closing_changes_no_feature(browser: Browser,
                                                                              site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        before = site["controls"].view()
        items = site["runtime"].snapshot()["features"]["items"]
        click_nav(page, "settings")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        page.evaluate("document.querySelector('#settings [data-set=theme][data-value=dark]').click()")
        assert wait_until(lambda: page.evaluate("document.documentElement.dataset.theme") == "dark", 3.0)
        page.evaluate("document.querySelector('#settings [data-set=density][data-value=compact]').click()")
        assert wait_until(lambda: page.evaluate("document.documentElement.dataset.density") == "compact", 3.0)
        assert page.evaluate(OPEN) is True  # nothing reloads and the dialog stays
        page.evaluate("document.querySelector('#settings-dialog a[data-sd=features]').click()")
        page.evaluate("document.getElementById('sd-close').click()")
        assert wait_until(lambda: page.evaluate(OPEN) is False, 3.0)
        paths = requests_seen(page)
        assert not [path for path in paths if path.startswith("/features") or path.startswith("/chat")], paths
        after = site["controls"].view()
        assert after["log"] == [] == before["log"] and after["counters"] == before["counters"]
        assert site["runtime"].snapshot()["features"]["items"] == items  # every feature as it was
    finally:
        page.close()


# ---------------------------------------------------------------- warnings stay in view


def test_not_current_and_a_degraded_runtime_show_inside_the_open_dialog(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        click_nav(page, "features")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        status = "document.querySelector('#settings-dialog [data-region=settings-status]')"
        shown = (f"[...{status}.querySelectorAll('.badge, .updated')].filter(n => getComputedStyle(n).visibility "
                 "!== 'hidden' && n.offsetParent !== null).map(n => n.innerText.trim()).join(' | ')")
        assert wait_until(lambda: "Page current" in page.evaluate(shown), 5.0), page.evaluate(shown)
        site["runtime"].degraded = ["capture stale (no new frame)"]
        assert wait_until(lambda: "Monitoring, degraded" in page.evaluate(shown), 6.0), page.evaluate(shown)
        site["runtime"].answering.clear()  # updates stop: within PAGE_CURRENT_S the dialog says NOT CURRENT by itself
        assert wait_until(lambda: "NOT CURRENT" in page.evaluate(shown), PAGE_CURRENT_S + 8.0), page.evaluate(shown)
        assert "Page current" not in page.evaluate(shown)
        assert page.evaluate("[...document.querySelectorAll('#features button.switch')].every(b => b.disabled)")
        site["runtime"].answering.set()
    finally:
        page.close()


def test_a_warning_the_features_section_holds_stays_on_the_page_with_a_button_that_opens_it(
        browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(status_page, "REFRESH_S", 1)
    incident_store(tmp_path / "sentinel.db", count=1)
    controls = FeatureControls(FakeClock(), {"face": True, "scene": True, "live_view": True})
    runtime = Runtime(controls=controls)
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db")
    server.enable_controls(lambda: controls, new_control_key())
    server.start()
    page = opened(browser, f"http://127.0.0.1:{server.address[1]}/dashboard", width=1440, height=900)
    try:
        assert not visible(page, ".hw[data-panel=features]")
        for _ in range(6):  # wrong keys until the controls lock
            page.evaluate("document.getElementById('control-key').value = 'AAAA-BBBB-CCCC-DDDD-EEEE-FFFF'")
            page.evaluate("document.getElementById('controls-unlock').requestSubmit()")
            page.evaluate("new Promise(r => setTimeout(r, 250))")
        assert wait_until(lambda: visible(page, ".hw[data-panel=features]"), 8.0)
        text = page.evaluate("document.querySelector('.hw[data-panel=features]').innerText")
        assert "Feature controls are locked" in text and "Open Features" in text
        page.evaluate("document.querySelector('.hw[data-panel=features] button[data-open-settings]').click()")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0) and page.evaluate(SECTION) == "features"
        assert "every control is locked" in page.evaluate("document.querySelector('[data-region=features]').innerText")
    finally:
        page.close()
        assert server.stop(3.0)


# ---------------------------------------------------------------- narrow screens, light and dark


def test_on_a_phone_the_dialog_is_a_fitted_sheet_with_internal_scrolling_and_reachable_controls(
        browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=390, height=700, mobile=True)
    try:
        page.evaluate("document.getElementById('menu').click()")  # the drawer, then Features in it
        assert wait_until(lambda: page.evaluate("document.documentElement.classList.contains('drawer-open')"), 3.0)
        page.evaluate("document.querySelector('#sidebar .nav a[data-nav=features]').click()")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        assert page.evaluate("document.documentElement.classList.contains('drawer-open')") is False
        box = page.evaluate("(() => { const r = document.getElementById('settings-dialog').getBoundingClientRect(); "
                            "return [r.left, r.top, r.right, r.bottom, innerWidth, innerHeight]; })()")
        assert box[0] >= 0 and box[1] >= 0 and box[2] <= box[4] and box[3] <= box[5]  # fitted inside the screen
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")  # no sideways page scroll
        panel = page.evaluate("(() => { const p = document.getElementById('sd-panel'); return [p.scrollHeight, "
                              "p.clientHeight, getComputedStyle(p).overflowY]; })()")
        assert panel[0] > panel[1] and panel[2] == "auto"  # the section scrolls inside the sheet
        for selector in ("#control-key", "#features button.switch[data-feature=live_view]", "#sd-close"):
            page.evaluate(f"document.querySelector({selector!r}).scrollIntoView({{block: 'center'}})")
            reach = page.evaluate(
                f"(() => {{ const n = document.querySelector({selector!r}); const r = n.getBoundingClientRect(); "
                "const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2); "
                "return r.width > 0 && r.top >= 0 && r.bottom <= innerHeight && (hit === n || n.contains(hit) || "
                "hit.contains(n)); })()")
            assert reach, selector
        for selector in ("#settings-dialog a[data-sd=settings]", "#settings-dialog a[data-sd=features]"):
            assert visible(page, selector)  # both sections' entries, side by side above the panel
        page.save_screenshot("settings-dialog-phone")
        page.key("Escape")
        assert wait_until(lambda: page.evaluate(OPEN) is False, 3.0)
        assert page.evaluate(ACTIVE) == "menu"  # the drawer is closed: focus returns to the menu button
    finally:
        page.close()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_dialog_uses_the_pages_own_surfaces_in_light_and_dark(browser: Browser, site: dict[str, Any],
                                                                   theme: str) -> None:
    page = opened(browser, f"{site['url']}/dashboard?theme={theme}", width=1440, height=900)
    try:
        click_nav(page, "settings")
        assert wait_until(lambda: page.evaluate(OPEN), 3.0)
        style = page.evaluate(
            "(() => { const s = (n) => getComputedStyle(n); const d = document.getElementById('settings-dialog'); "
            "return {dialog: s(d).backgroundColor, body: s(document.body).backgroundColor, ink: s(d).color, "
            "bodyInk: s(document.body).color, radius: s(d).borderTopLeftRadius, "
            "panelRadius: s(document.querySelector('.panel.live')).borderTopLeftRadius, "
            "tab: s(document.querySelector('#settings-dialog a[data-sd=settings]')).outlineStyle}; })()")
        assert style["dialog"] == style["body"] and style["ink"] == style["bodyInk"]  # the palette's own canvas and ink
        assert style["radius"] == style["panelRadius"]
        assert style["tab"] == "none"  # nav items never get a ring; the selected one keeps its glow
        assert page.evaluate("document.documentElement.dataset.theme") == theme
        page.save_screenshot(f"settings-dialog-{theme}")
    finally:
        page.close()


# ---------------------------------------------------------------- without script


def test_without_script_both_sections_stay_on_the_page_and_the_dialog_never_shows(tmp_path: Path) -> None:
    incident_store(tmp_path / "sentinel.db", count=1)
    server = StatusServer(0, Runtime().snapshot, tmp_path / "sentinel.db")
    try:
        html = render_html(server.status(with_picture=True, for_page=True), marker=server.marker(),
                           chat_token=server.chat_token, path="/dashboard")
    finally:
        server.stop(1.0)
    main, _, rest = html.partition("</main>")
    assert "<section id=features" in main and "<section id=settings" in main  # in the page flow, as before
    assert 'href="#features" data-nav=features' in html and 'href="#settings" data-nav=settings' in html
    dialog = rest[rest.index("<dialog"):rest.index("</dialog>")]
    assert 'class="sdialog js-only"' in dialog and " open" not in dialog.split(">", 1)[0]
    assert "data-region=settings-status" in dialog and "Close settings" in dialog
    assert "<section" not in dialog and "control-key" not in dialog  # no copy of any control
    assert build_status is not None
