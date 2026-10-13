"""D98: the Sentinel mark (an S in a faint lens ring) in the sidebar's sprite and as the browser tab's icon.

The sprite's mark draws only in the current colour, so it follows every theme and palette; the tab icon is a
self-contained SVG on the Carbon tile whose S has strong contrast against it (it must read at 16 px in light and dark
browser chrome). Rendering at 16-96 px on light and dark tiles was inspected by eye when the mark was chosen; these tests
pin what can be checked without a browser.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from sentinel.web import ICONS, SPRITE, TAB_ICON
from sentinel.web.palettes import contrast

SVG = "{http://www.w3.org/2000/svg}"


def test_the_sprite_mark_is_an_s_and_a_lens_ring_in_the_current_colour_only() -> None:
    mark = ICONS["logo"]
    shapes = ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{mark}</svg>')
    assert [element.tag for element in shapes] == [f"{SVG}path", f"{SVG}circle"]
    assert not re.search(r"(fill|stroke)=\"#", mark) and "style=" not in mark  # no colour of its own, no inline style
    s_path = shapes[0].get("d")
    assert s_path.startswith("M16.2 7.3A") and s_path.count("a") + s_path.count("A") == 2  # two tangent arcs: an S
    ring = shapes[1]
    assert (ring.get("cx"), ring.get("cy"), ring.get("r")) == ("12", "12", "10.1") and float(ring.get("opacity")) < 1
    assert '<symbol id="i-logo" viewBox="0 0 24 24">' in SPRITE


def test_the_tab_icon_is_the_same_mark_on_the_carbon_tile_with_strong_contrast() -> None:
    root = ET.fromstring(TAB_ICON.decode("ascii"))
    assert root.tag == f"{SVG}svg" and root.get("viewBox") == "0 0 32 32"
    tile = root.find(f"{SVG}rect")
    group = root.find(f"{SVG}g")
    assert tile is not None and group is not None and group.get("stroke") == "#eddcc9"
    assert group.find(f"{SVG}path").get("d") == ET.fromstring(
        f'<svg xmlns="http://www.w3.org/2000/svg">{ICONS["logo"]}</svg>')[0].get("d")
    assert contrast(tile.get("fill"), group.get("stroke")) >= 7.0  # the S on its tile, in any browser chrome
    assert b"<script" not in TAB_ICON and b"href" not in TAB_ICON  # nothing active or external
