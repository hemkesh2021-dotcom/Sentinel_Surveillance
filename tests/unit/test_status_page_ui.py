"""D72: the status pages' styling and structure, without a browser.

Contrast is computed from the stylesheet's own tokens (WCAG 2.x relative luminance): text at least 4.5:1, focus
rings and control borders at least 3:1, in both themes. States carry a text label and an icon, never colour alone.
Images are never filtered or recoloured. Reduced motion turns transitions off and keeps the D71 watchdog.
"""

from __future__ import annotations

import http.client
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from status_fixtures import incident_store

from sentinel import status_page
from sentinel.status_page import (
    STYLE,
    STYLE_SOURCE,
    StatusServer,
    build_status,
    read_incident,
    read_store,
    render_html,
    render_incident_html,
    theme_from_query,
)
from sentinel.web import ICONS, csp_hash, stylesheet

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
SURFACES = ("canvas", "sidebar", "panel", "raised", "sunken")
TEXT_PAIRS = [(fg, s) for fg in ("ink", "muted", "link") for s in SURFACES] + \
    [("accent-ink", "accent"), ("ok-ink", "ok-bg"), ("warn-ink", "warn-bg"), ("crit-ink", "crit-bg"),
     ("off-ink", "off-bg"), ("done-ink", "done-bg"), ("stale-ink", "stale-bg")]
UI_PAIRS = [(fg, s) for fg in ("focus", "border", "marker") for s in SURFACES]


def tokens(block: str) -> dict[str, str]:
    return {name: value.strip() for name, value in re.findall(r"--([a-z0-9-]+):\s*([^;]+);", block)}


def luminance(colour: str) -> float:
    channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def blocks(key: str) -> dict[str, dict[str, str]]:
    """The stylesheet's own token blocks for palette ``key``: light, automatic dark and chosen dark (D96)."""
    from sentinel.web import palettes

    css = stylesheet(5)
    selector = f":root[data-palette={key}]"
    if key == palettes.DEFAULT_PALETTE:
        selector = f":root, {selector}"
    found = {}
    light = css.index(selector + " {")
    found["light"] = tokens(css[light:css.index("}", light)])
    auto_selector = ", ".join(f"{part.strip()}:not([data-theme=light])" for part in selector.split(","))
    auto = css.index("@media (prefers-color-scheme: dark) { " + auto_selector + " {")
    found["auto"] = tokens(css[auto:css.index("}", auto)])
    dark_selector = ", ".join(f"{part.strip()}[data-theme=dark]" for part in selector.split(","))
    dark = css.index(dark_selector + " {")
    found["dark"] = tokens(css[dark:css.index("}", dark)])
    return found


@pytest.mark.parametrize("mode", ["light", "dark", "auto"])
def test_text_and_controls_meet_their_contrast_in_every_palette_and_mode(mode: str) -> None:
    from sentinel.web import palettes

    for preset in palettes.PRESETS:
        values = blocks(preset.key)[mode]
        low = {f"{fg} on {bg}": round(contrast(values[fg], values[bg]), 2) for fg, bg in TEXT_PAIRS
               if contrast(values[fg], values[bg]) < 4.5}
        assert not low, (preset.key, low)
        primary = {f"ink on {bg}": round(contrast(values["ink"], values[bg]), 2) for bg in SURFACES
                   if contrast(values["ink"], values[bg]) < 7.0}
        assert not primary, (preset.key, primary)
        weak = {f"{fg} on {bg}": round(contrast(values[fg], values[bg]), 2) for fg, bg in UI_PAIRS
                if contrast(values[fg], values[bg]) < 3.0}
        assert not weak, (preset.key, weak)


def test_carbon_and_cream_is_the_default_with_the_maintainers_colours() -> None:
    """The maintainer's Carbon & Cream: carbon #2E302F and cream #EDDCC9; dark slightly deeper, canvas #1C1D1C and
    sidebar #181918, panels #232423, raised #2E302F, sunken #191A19 (D82, D96). Both dark blocks agree."""
    found = {mode: {k: v.lower() for k, v in block.items()} for mode, block in blocks("carbon-cream").items()}
    light, auto, dark = found["light"], found["auto"], found["dark"]
    assert (light["ink"], light["accent"], light["accent-ink"], light["brand"]) == ("#2e302f", "#2e302f", "#eddcc9",
                                                                                   "#2e302f")
    assert {k: dark[k] for k in SURFACES} == {"canvas": "#1c1d1c", "sidebar": "#181918", "panel": "#232423",
                                              "raised": "#2e302f", "sunken": "#191a19"}
    assert (dark["ink"], dark["accent"]) == ("#eddcc9", "#eddcc9")
    assert auto == dark  # the browser's dark preference and the chosen dark theme set the same values
    assert stylesheet(5).index(":root, :root[data-palette=carbon-cream] {") < stylesheet(5).index(
        ":root[data-palette=sesame-mango] {")  # the default first, so a chosen palette overrides it


def test_the_stylesheet_carries_every_palette_exactly_as_derived_and_states_are_fixed() -> None:
    from sentinel.web import palettes

    assert [p.key for p in palettes.PRESETS] == ["carbon-cream", "sesame-mango", "matcha-cream", "teal-coconut",
                                                 "burgundy-rose", "port-fern-mauve", "charcoal-sheen"]
    for preset in palettes.PRESETS:
        found = blocks(preset.key)
        for mode, block in (("light", "light"), ("dark", "dark"), ("dark", "auto")):
            assert found[block] == palettes.tokens(preset.key, mode), (preset.key, block)
        for mode in ("light", "dark"):  # states keep one meaning in every palette
            assert {k: v for k, v in palettes.tokens(preset.key, mode).items() if k in palettes.STATES[mode]} == \
                palettes.STATES[mode]
        for i, colour in enumerate(preset.colours, 1):  # each swatch shows the palette's own colours
            assert f".sw-{preset.key} .d{i} {{ background: {colour.lower()}; }}" in stylesheet(5)


def test_every_palette_meets_its_contrast_floors_in_both_modes() -> None:
    from sentinel.web import palettes

    for preset in palettes.PRESETS:
        for mode in ("light", "dark"):
            low = [row for row in palettes.contrast_report(preset.key, mode) if row[2] < row[3]]
            assert not low, (preset.key, mode, low)


def test_the_sidebar_links_each_section_of_the_page_in_page_order() -> None:
    """D96: every destination is a real section of the page (no placeholder), in page order, with an icon and a name."""
    page = render_html(build_status({"runtime": {"state": "running", "camera_id": "cam-1"}},
                                    {"state": "available"}, NOW))
    nav = re.search(r'<nav class=nav aria-label="Dashboard sections">(.*?)</nav>', page).group(1)
    targets = re.findall(r'<a href="#([a-z-]+)" data-nav=\1><svg[^>]*><use href="#i-[a-z]+"/></svg>'
                         r'<span class=label>([^<]+)</span>', nav)
    assert targets == [("live", "Live view"), ("faces", "Recognition"), ("incidents", "Incidents"), ("chat", "Ask"),
                       ("health", "Health"), ("delivery", "Alerts"), ("settings", "Appearance")]
    positions = [page.index(f"<section id={target} ") for target, _ in targets]
    assert positions == sorted(positions) and page.index("<nav class=nav") < positions[0]
    assert re.findall(r"<p class=group>([^<]+)</p>", nav) == ["Monitor", "System", "Preferences"]
    assert 'id=menu aria-controls=sidebar aria-expanded=false' in page and 'id=rail-toggle aria-controls=sidebar' in page


def test_images_are_never_recoloured_and_reduced_motion_keeps_the_watchdog() -> None:
    assert not re.search(r"\b(filter|backdrop-filter|mix-blend-mode|mask-image)\s*:", STYLE)
    reduced = STYLE[STYLE.index("@media (prefers-reduced-motion: reduce)"):]
    assert "transition-duration: 0s" in reduced and "animation" not in reduced
    assert ".cur { visibility: hidden; animation: cur 5s linear var(--fresh-delay, 0s); }" in STYLE  # untouched
    assert STYLE == stylesheet(status_page.PAGE_CURRENT_S) and "CURRENT" not in STYLE.replace("NOT CURRENT", "")


def test_every_state_has_a_label_and_an_icon_and_kinds_are_distinct() -> None:
    snapshot = {"runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": NOW.isoformat()},
                "degraded": ["detector unavailable (libcuda_not_l4t)"],
                "components": {"capture": {"state": "waiting", "problem": "open_failed"},
                               "detector": {"state": "unavailable", "problem": "libcuda_not_l4t"},
                               "scene": {"state": "disabled"}, "face": {"state": "available"},
                               "incidents": {"state": "ok"}}}
    page = render_html(build_status(snapshot, {"state": "available"}, NOW))
    badges = re.findall(r'<span class="badge k-(\w+)[^"]*"><svg[^>]*><use href="#i-(\w+)"/></svg>([^<]+)</span>', page)
    assert len(badges) == page.count('<span class="badge ')  # every badge: icon, then a text label
    kinds = {kind: icon for kind, icon, _ in badges}
    assert {"ok", "warn", "crit", "off"} <= set(kinds) and len({kinds[k] for k in ("ok", "warn", "crit", "off")}) == 4
    assert all(text.strip() for _, _, text in badges)
    assert set(re.findall(r'<use href="#i-(\w+)"/>', page)) <= set(ICONS)
    assert 'aria-hidden="true" focusable="false"' in page  # decorative: the label carries the meaning


def test_the_csp_allows_exactly_the_inline_stylesheet_and_no_other_inline_code(tmp_path: Path) -> None:
    incident_store(tmp_path / "sentinel.db", count=1)
    server = StatusServer(0, lambda: {"runtime": {"state": "running", "camera_id": "cam-1"}}, tmp_path / "sentinel.db")
    server.start()
    try:
        for path in ("/", "/assets/icon.svg", "/status.json"):
            status, headers, body = get(server, path)
            assert status == 200, path
            policy = headers["Content-Security-Policy"]
            assert f"style-src {STYLE_SOURCE};" in policy and "unsafe" not in policy
            assert "script-src 'self';" in policy and "connect-src 'self';" in policy  # D73: one same-origin script
            assert "default-src 'none'" in policy and "frame-ancestors 'none'" in policy
        page = get(server, "/")[2].decode()
        (style,) = re.findall(r"<style>(.*?)</style>", page, flags=re.S)
        assert csp_hash(style) == STYLE_SOURCE and " style=" not in page
        assert re.findall(r"<script[^>]*>(.*?)</script>", page, flags=re.S) == ["", ""]  # no inline script
        assert re.findall(r"<script ([^>]*)>", page) == ['src="/assets/appearance.js"', 'src="/assets/dashboard.js" defer']
        assert "http-equiv=refresh" not in page
    finally:
        assert server.stop(3.0)


def get(server: StatusServer, path: str, *, host: str | None = None, method: str = "GET") -> tuple[int, Any, bytes]:
    address, port = server.address
    conn = http.client.HTTPConnection(address, port, timeout=5)
    try:
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", host if host is not None else f"127.0.0.1:{port}")
        conn.endheaders()
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


@pytest.mark.parametrize(("query", "theme"), [
    ("theme=light", "light"), ("theme=dark", "dark"), ("", None), ("theme=", None), ("theme=Dark", None),
    ("theme=dark%22%3E%3Cscript%3E", None), ("x=1&theme=dark", "dark"), ("theme=sepia", None),
])
def test_the_theme_choice_is_one_of_two_exact_values_and_never_echoed(query: str, theme: str | None) -> None:
    assert theme_from_query(query) == theme


def test_the_chosen_theme_is_applied_and_kept_in_every_link(tmp_path: Path) -> None:
    ids = incident_store(tmp_path / "sentinel.db", count=2)
    server = StatusServer(0, lambda: {"runtime": {"state": "running", "camera_id": "cam-1"}}, tmp_path / "sentinel.db")
    server.start()
    try:
        page = get(server, "/?theme=dark")[2].decode()
        assert ('<html lang=en data-page=status data-theme=dark data-pref data-palettes="carbon-cream sesame-mango '
                'matcha-cream teal-coconut burgundy-rose port-fern-mauve charcoal-sheen" '
                'data-source="/status.json?page=1&amp;theme=dark"') in page
        assert f'href="/incidents/{ids[0]}?theme=dark"' in page
        assert '<a class=seg href="/?theme=dark" data-set=theme data-value=dark aria-current="true">' in page
        assert '<a class=seg href="/" data-set=theme data-value=auto>' in page  # the other choices stay reachable
        assert '<a class=seg href="/?theme=light" data-set=theme data-value=light>' in page
        assert '<a class=swatch href="/?theme=dark&amp;palette=teal-coconut" data-set=palette data-value=teal-coconut>' in page
        page = get(server, "/?theme=%22%3E%3Cscript%3Ealert(1)%3C/script%3E")[2].decode()
        assert '<html lang=en data-page=status data-palettes=' in page and "alert(1)" not in page
        assert "data-pref" not in page.split(">", 2)[1]  # nothing was chosen by that address
        detail = get(server, f"/incidents/{ids[0]}?theme=light")[2].decode()
        assert "data-theme=light" in detail and '<a class=back href="/dashboard?theme=light">' in detail
        assert '<a href="/dashboard?theme=light#incidents" data-nav=incidents aria-current="page">' in detail
    finally:
        assert server.stop(3.0)


@pytest.mark.parametrize(("query", "look"), [
    ("palette=teal-coconut", (None, "teal-coconut", None)), ("palette=carbon-cream", (None, None, None)),
    ("palette=Teal-Coconut", (None, None, None)), ("palette=teal-coconut%22", (None, None, None)),
    ("density=compact&theme=dark", ("dark", None, "compact")), ("density=comfortable", (None, None, None)),
    ("theme=dark&theme=light", ("dark", None, None)), ("palette=sepia", (None, None, None)),
])
def test_the_appearance_choices_are_exact_values_and_never_echoed(query: str, look: tuple) -> None:
    from sentinel.status_page import look_from_query

    chosen = look_from_query(query)
    assert (chosen.theme, chosen.palette, chosen.density) == look


def test_the_dashboard_is_served_at_its_own_name_and_at_the_root(tmp_path: Path) -> None:
    """D96: /dashboard is the same page as /, which (with every existing link) is unchanged; its appearance links stay
    on the address the reader used."""
    server = StatusServer(0, lambda: {"runtime": {"state": "running", "camera_id": "cam-1"}}, tmp_path / "sentinel.db")
    server.start()
    try:
        root, dashboard = get(server, "/"), get(server, "/dashboard?theme=dark")
        assert root[0] == dashboard[0] == 200 and dashboard[1]["Content-Type"] == "text/html; charset=utf-8"
        assert '<a class=seg href="/dashboard" data-set=theme data-value=auto>' in dashboard[2].decode()
        assert '<a class=seg href="/" data-set=theme data-value=auto aria-current="true">' in root[2].decode()
        assert get(server, "/dashboard/")[0] == 404 and get(server, "/Dashboard")[0] == 404
    finally:
        assert server.stop(3.0)


def test_the_tab_icon_is_served_read_only_on_the_loopback_host(tmp_path: Path) -> None:
    server = StatusServer(0, lambda: {"runtime": {"state": "running"}}, tmp_path / "sentinel.db")
    server.start()
    try:
        status, headers, body = get(server, "/assets/icon.svg")
        assert (status, headers["Content-Type"], headers["X-Content-Type-Options"]) == (200, "image/svg+xml", "nosniff")
        assert body.startswith(b"<svg") and b"<script" not in body and headers["Cache-Control"] == "no-store"
        assert get(server, "/assets/icon.svg", method="POST")[0] == 405
        assert get(server, "/assets/icon.svg", host="evil.example")[0] == 421
        assert get(server, "/assets/other.svg")[0] == 404
    finally:
        assert server.stop(3.0)


def test_the_incident_page_exposes_snapshot_download_reason_times_lifecycle_acknowledgment_and_delivery(
        tmp_path: Path) -> None:
    ids = incident_store(tmp_path / "sentinel.db", count=2)
    page = render_incident_html({"generated_utc": NOW.isoformat(), **read_incident(tmp_path / "sentinel.db", ids[0])})
    sections = re.findall(r'<section (?:id=\w+ )?class="panel p-(\w+)" aria-label="?([^">]+)"?>', page)
    assert [s for s, _ in sections] == ["snap", "summary", "ack", "delivery", "lifecycle", "scene", "tech"]
    assert f'<img src="/incidents/{ids[0]}/snapshot.jpg"' in page
    assert f'<a class=button href="/incidents/{ids[0]}/snapshot/download" download>' in page
    assert '<p class=reason>track 3 stayed in the zone for 1.0 s (at least 1.0 s)</p>' in page  # why it opened
    for label in ("first seen (UTC)", "latest (UTC)", "recorded (UTC)", "updated (UTC)"):
        assert f"<th>{label}</th><td><time datetime=" in page, label
    assert "<ol class=timeline>" in page and "open → acknowledged" in page
    assert "acknowledged" in page and "by operator:maintainer: seen on the dashboard" in page
    assert "delivered" in page and "provider confirmed: yes" in page
    # Technical provenance is kept, behind expandable sections.
    assert page.count("<details class=tech") == 2 and "Snapshot provenance" in page and "every stored row" in page
    failed = render_incident_html({"generated_utc": NOW.isoformat(), **read_incident(tmp_path / "sentinel.db", ids[1])})
    assert "failed" in failed and "provider confirmed: no" in failed and "Not acknowledged" in failed


def test_the_main_page_puts_recognition_beside_the_camera_and_incidents_below_with_details_behind_expanders(
        tmp_path: Path) -> None:
    ids = incident_store(tmp_path / "sentinel.db", count=3)
    page = render_html(build_status({"runtime": {"state": "running", "camera_id": "cam-1"}},
                                    read_store(tmp_path / "sentinel.db"), NOW))
    order = re.findall(r'<section id=([\w-]+) class="panel', page)
    assert order == ["live", "faces", "incidents", "chat", "health", "delivery", "settings"]  # the phone's order
    assert '<a class=skip href="#live">' in page
    assert all(f'href="/incidents/{i}"' in page for i in ids)
    assert page.count("<details class=tech") == 2  # the runtime's counters, the alert rows (the live view adds one)
    assert re.search(r'\.board \{[^}]*grid-template-areas: "live" "faces" "incidents" "chat" "health" "delivery" '
                     r'"settings"', STYLE)
    assert '"live faces" "live chat" "incidents health" "incidents delivery" "settings settings"' in STYLE
    # A panel never carries the watchdog's .report and the entrance animation together (D96).
    assert not re.search(r'class="panel [^"]*\breport\b', page)
