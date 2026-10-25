"""Colour palettes for the status pages (D76, offered in D96): shared design tokens, derived per preset and mode.

Each preset names two or three base colours. Every token the stylesheet uses (surfaces, text, borders, focus, the
accent and its text) is derived from them for a light and a dark mode, in OKLab, and then moved in lightness only as
far as needed to meet its contrast floor (WCAG 2.x relative luminance) against every surface it appears on: text 4.5:1
(primary text 7:1), focus rings, control borders and selection markers 3:1. ``contrast_report`` lists every pairing,
and the tests check all of them for all presets in both modes.

States keep one meaning in every preset: their colours (healthy and live green, warning amber, error red, the
not-current banner) are fixed per mode and never derived from a palette, a palette's accent never marks a state, and
every state is also given by its text and icon. Camera pictures and snapshots are never recoloured by any of this.
The one exception (session 114): Carbon & Cream's dark mode shows warnings and the not-current banner in orange, so
that no warning looks like its gold highlight.

Carbon & Cream is the default. Its dark mode is the maintainer's approved deep black (session 114, preview v5's
[data-black="deep"]): page #060605, recessed panels #0E0D0B, charcoal keys #2E2C28 / #23221F / #1B1A17 on a #000000
edge, ivory text and a warm gold highlight; ``DEEP`` holds that look's shadows and grain, set only there. Every light
mode and the six other presets' dark modes keep the colours they had before (their tokens are pinned in the tests).

Every preset also carries the highlight of the controls' states (session 114): the hover label, the trace that runs
once around a hovered or focused control, its rings and glows (``hl-*``, ``glow-selected``). In the light modes it is
v5's ivory gold, in every preset (the maintainer's choice); in the dark modes, each preset's own marker (Carbon &
Cream's gold). Focus rings and selected lines keep each preset's own colours (``focus``, ``marker-line``).
Portable: stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

DEFAULT_PALETTE = "carbon-cream"
SURFACES = ("canvas", "sidebar", "panel", "raised", "sunken")
TEXT_FLOOR = 4.5
INK_FLOOR = 7.0
UI_FLOOR = 3.0

# States: fixed per mode, the same in every preset (text on its own background; contrast checked in the tests).
STATES = {
    "light": {"ok-bg": "#dcebda", "ok-ink": "#1d6a3b", "warn-bg": "#f6e3bf", "warn-ink": "#734400",
              "crit-bg": "#f8dad5", "crit-ink": "#a11d1d", "stale-bg": "#7a2e12", "stale-ink": "#fdf6e8"},
    "dark": {"ok-bg": "#1e3326", "ok-ink": "#8fd3a0", "warn-bg": "#3a2e19", "warn-ink": "#f2c76e",
             "crit-bg": "#3f2320", "crit-ink": "#ff9e94", "stale-bg": "#f2a65a", "stale-ink": "#1c1d1c"},
}

# Session 114: the controls' highlight in every light mode, v5's ivory gold (the hover label and the trace's middle,
# the trace's light and deep ends, the glows' warmer tone); in the dark modes, each preset's marker (below).
LIGHT_HIGHLIGHT = {"hl": "#8a5d0f", "hl-hi": "#b07a18", "hl-lo": "#6e4a0b", "hl-warm": "#b0781e"}
HL_STEPS = (0.081, -0.160)  # a dark highlight's trace ends, in OKLab lightness (v5's gold: #fde58a, #c98a2e)

# Session 114: v5's [data-black="deep"] look, set only where a mode's seeds ask for it (Carbon & Cream's dark mode):
# its shadows and grain, the keys' gradient on a #000000 edge, and what the stylesheet's rules take in place of their
# earlier look there. Everywhere else each is ``initial``, so every rule's own fallback (the earlier look) applies.
# The look's own colours (a key's gradient ends, a disabled label) are given exactly by that mode's seeds.
DEEP_COLOURS = ("key-top", "key-bottom", "faint")
DEEP = {
    "deep-grain": 'url("/assets/grain.svg")',  # status_page.GRAIN_PATH: a 160 px noise tile at alpha .03
    "deep-key": "var(--deep-grain), linear-gradient(170deg, var(--key-top), var(--raised) 55%, var(--key-bottom))",
    "deep-raised": "inset 0 1px 0 rgba(255, 244, 222, .08), inset 0 -1px 0 rgba(0, 0, 0, .6), 0 1px 0 0 #000000, "
                   "0 0 0 1px #000000, 0 8px 18px rgba(0, 0, 0, .85)",
    "deep-pressed": "inset 0 2px 7px rgba(0, 0, 0, .9), inset 0 -1px 0 rgba(255, 244, 222, .035), 0 0 0 1px #000000",
    "deep-well": "inset 0 2px 7px rgba(0, 0, 0, .85), inset 0 -1px 0 rgba(255, 244, 222, .035), "
                 "0 0 0 1px rgba(255, 244, 222, .03)",
    "deboss": "0 1px 0 rgba(0, 0, 0, .6)",  # under a key's label
    "deep-primary": "inset 0 0 0 1px var(--hl-edge)",  # a primary key's understated edge
    "deep-clear": "transparent",  # borders: the shadows draw the edges
    "deep-round": "12px",  # a key's corners (pills become keys)
    "deep-stack": "8px",  # between stacked keys
    "deep-ink": "var(--ink)",  # a key's label where the earlier control was filled with the accent
    "deep-hl": "var(--hl)",  # its hover label
    "deep-card": "var(--sunken)",  # cards are recessed, not raised
    "deep-flat": "var(--panel)",  # a disabled key: flat and recessed, with a faint label
    "deep-faint": "var(--faint)",
    "deep-dim": "1",  # (the earlier look dims a disabled control instead)
}


# ---------------------------------------------------------------- colour arithmetic


def _rgb(hex_colour: str) -> tuple[float, float, float]:
    text = hex_colour.lstrip("#")
    return tuple(int(text[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(min(1.0, max(0.0, c)) * 255):02x}" for c in rgb)


def _to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _from_linear(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def luminance(hex_colour: str) -> float:
    r, g, b = (_to_linear(c) for c in _rgb(hex_colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _oklab(hex_colour: str) -> tuple[float, float, float]:
    r, g, b = (_to_linear(c) for c in _rgb(hex_colour))
    l_ = math.copysign(abs(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3), 1)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def _from_oklab(lightness: float, a: float, b: float) -> tuple[float, float, float] | None:
    """Gamma-encoded sRGB, or None outside the sRGB gamut."""
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    linear = (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
              -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
              -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)
    if any(c < -1e-4 or c > 1 + 1e-4 for c in linear):
        return None
    return tuple(_from_linear(min(1.0, max(0.0, c))) for c in linear)  # type: ignore[return-value]


def _lch(lightness: float, chroma: float, hue: float) -> str:
    """OKLab lightness, chroma and hue as hex, the chroma reduced (hue kept) until it is inside the sRGB gamut."""
    lightness = min(1.0, max(0.0, lightness))
    while True:
        rgb = _from_oklab(lightness, chroma * math.cos(hue), chroma * math.sin(hue))
        if rgb is not None:
            return _hex(rgb)
        if chroma <= 0:
            return _hex((lightness, lightness, lightness))
        chroma = max(0.0, chroma - 0.002)


def tone(seed: str, lightness: float, max_chroma: float | None = None) -> str:
    """``seed``'s hue at OKLab ``lightness``, with its chroma (at most ``max_chroma``), reduced into the sRGB gamut."""
    _, a, b = _oklab(seed)
    chroma = math.hypot(a, b)
    if max_chroma is not None:
        chroma = min(chroma, max_chroma)
    return _lch(lightness, chroma, math.atan2(b, a))


def mix(a: str, b: str, weight: float) -> str:
    """``weight`` of ``b`` mixed into ``a``, in OKLab."""
    la, lb = _oklab(a), _oklab(b)
    lightness, ca, cb = (x + (y - x) * weight for x, y in zip(la, lb))
    return _lch(lightness, math.hypot(ca, cb), math.atan2(cb, ca))


def at_least(colour: str, against: tuple[str, ...], floor: float, *, lighter: bool) -> str:
    """``colour``, moved lighter or darker in OKLab lightness only as far as needed to reach ``floor`` against all of
    ``against``. Raises if no lightness reaches it."""
    if all(contrast(colour, other) >= floor + 0.02 for other in against):
        return colour.lower()
    lightness, a, b = _oklab(colour)
    chroma, hue = math.hypot(a, b), math.atan2(b, a)
    step = 0.004 if lighter else -0.004
    while 0.0 <= lightness <= 1.0:
        candidate = _lch(lightness, chroma, hue)
        if all(contrast(candidate, other) >= floor + 0.02 for other in against):
            return candidate
        lightness += step
    raise ValueError(f"no lightness of {colour} reaches {floor}:1 against {against}")


def best_text(fill: str, choices: tuple[str, ...]) -> str:
    """The choice with the most contrast on ``fill``."""
    return max(choices, key=lambda choice: contrast(fill, choice))


def rgba(hex_colour: str, alpha: float) -> str:
    r, g, b = (round(c * 255) for c in _rgb(hex_colour))
    return f"rgba({r}, {g}, {b}, {alpha:g})"


# ---------------------------------------------------------------- presets


@dataclass(frozen=True)
class Seeds:
    """One mode's starting colours. ``tint``: the surfaces' hue; ``accent``: fills of selected and primary controls;
    ``marker``: selection bars and focus rings; ``link``: link text (default: the marker); ``lift``: the accent's
    OKLab lightness in this mode, when the base colour itself is too dark or too light for it; ``surfaces``: exact
    surfaces that replace the derived ones; ``exact``: any other token given exactly (a state's colours included);
    ``deep``: this mode takes the DEEP look."""

    tint: str
    ink: str
    accent: str
    marker: str
    link: str | None = None
    lift: float | None = None
    surfaces: dict[str, str] = field(default_factory=dict)
    accent_ink: str | None = None  # text on the accent, used when it reaches the floor (else the best of the rest)
    exact: dict[str, str] = field(default_factory=dict)  # other tokens given exactly, in place of the derived ones
    deep: bool = False  # session 114: this mode takes the DEEP look


@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    colours: tuple[str, ...]  # as named by the maintainer; shown as the preset's swatches
    brand: str  # the monogram's fill: the palette's character, never a state
    light: Seeds
    dark: Seeds
    brand_ink: str | None = None  # the monogram's letter, used when it reaches 3:1 on the brand


PRESETS: tuple[Preset, ...] = (  # in the maintainer's order; Carbon & Cream first (the default, emitted first)
    Preset("carbon-cream", "Carbon & Cream", ("#2E302F", "#EDDCC9"), "#2E302F", brand_ink="#EDDCC9",
           light=Seeds(tint="#EDDCC9", ink="#2E302F", accent="#2E302F", marker="#2E302F", accent_ink="#EDDCC9"),
           # session 114: the maintainer's approved deep black (preview v5) with ivory text and a gold highlight;
           # warnings and the not-current banner in orange, so that none looks like the gold
           dark=Seeds(tint="#2E302F", ink="#efe7d6", accent="#f2c45a", marker="#f2c45a", link="#efe7d6",
                      accent_ink="#060605", deep=True,
                      surfaces={"canvas": "#060605", "sidebar": "#060605", "panel": "#0e0d0b", "raised": "#23221f",
                                "sunken": "#060605"},
                      exact={"muted": "#b9af9d", "faint": "#8a8170", "key-top": "#2e2c28", "key-bottom": "#1b1a17",
                             "hl-hi": "#fde58a", "hl-lo": "#c98a2e", "hl-warm": "#d68c2e",
                             "marker-line": "rgba(242, 196, 90, 0.9)",
                             "warn-ink": "#ff8a3d", "warn-bg": "#261a10", "stale-bg": "#ff8a3d"})),
    Preset("sesame-mango", "Sesame & Mango", ("#1F1D1A", "#F4B54A"), "#F4B54A",
           light=Seeds(tint="#F4B54A", ink="#1F1D1A", accent="#F4B54A", marker="#1F1D1A", link="#F4B54A"),
           dark=Seeds(tint="#1F1D1A", ink="#F7EEDF", accent="#F4B54A", marker="#F4B54A")),
    Preset("matcha-cream", "Matcha & Cream", ("#4E5529", "#F0E2C6"), "#4E5529", brand_ink="#F0E2C6",
           light=Seeds(tint="#F0E2C6", ink="#4E5529", accent="#4E5529", marker="#4E5529", accent_ink="#F0E2C6"),
           dark=Seeds(tint="#4E5529", ink="#F0E2C6", accent="#4E5529", marker="#4E5529", lift=0.76)),
    Preset("teal-coconut", "Teal & Coconut", ("#174F4A", "#FDFAF5"), "#174F4A", brand_ink="#FDFAF5",
           light=Seeds(tint="#FDFAF5", ink="#174F4A", accent="#174F4A", marker="#174F4A", accent_ink="#FDFAF5"),
           dark=Seeds(tint="#174F4A", ink="#FDFAF5", accent="#174F4A", marker="#174F4A", lift=0.74)),
    Preset("burgundy-rose", "Burgundy & Rose", ("#501627", "#AF144B", "#FACCD7"), "#AF144B", brand_ink="#FDEFF2",
           light=Seeds(tint="#FACCD7", ink="#501627", accent="#AF144B", marker="#AF144B", accent_ink="#FDEFF2"),
           dark=Seeds(tint="#501627", ink="#FBE8EC", accent="#AF144B", marker="#FACCD7", accent_ink="#FDEFF2")),
    Preset("port-fern-mauve", "Port, Fern & Mauve", ("#5C2C35", "#9AA067", "#E7DCD9"), "#9AA067",
           light=Seeds(tint="#E7DCD9", ink="#5C2C35", accent="#5C2C35", marker="#5C2C35", accent_ink="#F3ECEA"),
           dark=Seeds(tint="#5C2C35", ink="#F1E9E7", accent="#5C2C35", marker="#5C2C35", lift=0.74)),
    Preset("charcoal-sheen", "Charcoal & Sheen", ("#2B2C30", "#D9CE52", "#F0EDE5"), "#D9CE52",
           light=Seeds(tint="#F0EDE5", ink="#2B2C30", accent="#D9CE52", marker="#2B2C30", link="#D9CE52"),
           dark=Seeds(tint="#2B2C30", ink="#F0EDE5", accent="#D9CE52", marker="#D9CE52")),
)
PALETTES = {preset.key: preset for preset in PRESETS}

# Lightness and chroma ceilings of the derived surfaces (OKLab), per mode: shallow steps, so panels separate by a fine
# border and a soft shadow rather than by strong colour.
_LIGHT_SURFACES = {"canvas": (0.955, 0.016), "sidebar": (0.93, 0.024), "panel": (0.982, 0.008),
                   "raised": (0.995, 0.004), "sunken": (0.925, 0.02)}
_DARK_SURFACES = {"canvas": (0.215, 0.012), "sidebar": (0.197, 0.012), "panel": (0.245, 0.012),
                  "raised": (0.285, 0.012), "sunken": (0.19, 0.012)}


@lru_cache(maxsize=None)
def tokens(key: str, mode: str) -> dict[str, str]:
    """Every token of preset ``key`` in ``mode`` (light or dark): its colours, and the DEEP look's values (or
    ``initial`` where a mode does not take it)."""
    preset = PALETTES[key]
    seeds = preset.light if mode == "light" else preset.dark
    lighter = mode == "dark"
    plan = _DARK_SURFACES if lighter else _LIGHT_SURFACES
    out = {name: seeds.surfaces.get(name) or tone(seeds.tint, lightness, chroma)
           for name, (lightness, chroma) in plan.items()}
    surfaces = tuple(out[name] for name in SURFACES)
    exact = seeds.exact
    out["ink"] = at_least(seeds.ink, surfaces, INK_FLOOR, lighter=lighter)
    out["muted"] = exact.get("muted") or at_least(mix(out["ink"], out["panel"], 0.42), surfaces, TEXT_FLOOR,
                                                  lighter=lighter)
    out["border"] = at_least(mix(out["ink"], out["panel"], 0.62), surfaces, UI_FLOOR, lighter=lighter)
    accent = tone(seeds.accent, seeds.lift, 0.12) if seeds.lift is not None else seeds.accent
    out["accent"] = accent
    preferred = seeds.accent_ink
    out["accent-ink"] = (preferred.lower() if preferred and contrast(accent, preferred) >= TEXT_FLOOR + 0.2 else
                         best_text(accent, (out["raised"], out["ink"], out["canvas"], "#ffffff", "#1c1d1c")))
    if contrast(accent, out["accent-ink"]) < TEXT_FLOOR:
        raise ValueError(f"{key} {mode}: no text reaches {TEXT_FLOOR}:1 on its accent")
    marker = tone(seeds.marker, seeds.lift, 0.12) if seeds.lift is not None and seeds.marker == seeds.accent \
        else seeds.marker
    out["marker"] = at_least(marker, surfaces, UI_FLOOR, lighter=lighter)
    out["focus"] = out["marker"]
    out["link"] = at_least(seeds.link or out["marker"], surfaces, TEXT_FLOOR, lighter=lighter)
    out["brand"] = preset.brand
    out["brand-ink"] = (preset.brand_ink.lower() if preset.brand_ink and contrast(preset.brand, preset.brand_ink) >= 4.5
                        else best_text(preset.brand, ("#ffffff", "#1c1d1c", out["ink"], out["raised"])))
    out["line"] = rgba(out["ink"], 0.11 if lighter else 0.13)
    out["line-strong"] = rgba(out["ink"], 0.2 if lighter else 0.24)
    out["shadow-lo"] = rgba("#000000", 0.5) if lighter else rgba(mix(out["ink"], out["canvas"], 0.35), 0.16)
    out["shadow-hi"] = rgba("#ffffff", 0.03) if lighter else rgba("#ffffff", 0.85)
    out["off-bg"] = out["sunken"]
    out["off-ink"] = out["muted"]
    out["done-bg"] = out["raised"]
    out["done-ink"] = out["ink"]
    out.update(STATES[mode])
    out.update({name: value for name, value in exact.items() if name in STATES[mode]})  # Carbon & Cream's dark only
    # Session 114: the controls' highlight: in dark, the preset's marker; in light, v5's ivory gold
    if lighter:
        hl = at_least(out["marker"], surfaces, TEXT_FLOOR, lighter=True)
        level = _oklab(hl)[0]
        hi = exact.get("hl-hi") or tone(hl, level + HL_STEPS[0])
        lo = exact.get("hl-lo") or tone(hl, level + HL_STEPS[1])
        warm = exact.get("hl-warm") or mix(hl, lo, 0.6)
        ring, glow = rgba(hl, 0.34), rgba(warm, 0.3)
        selected = f"0 0 12px {rgba(hl, 0.34)}, 0 0 28px {rgba(warm, 0.24)}"
    else:
        hl = at_least(LIGHT_HIGHLIGHT["hl"], surfaces, TEXT_FLOOR, lighter=False)
        hi, lo, warm = LIGHT_HIGHLIGHT["hl-hi"], LIGHT_HIGHLIGHT["hl-lo"], LIGHT_HIGHLIGHT["hl-warm"]
        ring, glow = rgba(hl, 0.32), rgba(warm, 0.24)
        selected = f"0 0 12px {rgba(warm, 0.36)}, 0 0 26px {rgba(warm, 0.22)}"
    out.update({"hl": hl, "hl-hi": hi, "hl-lo": lo, "hl-warm": warm, "hl-ring": ring, "hl-glow": glow,
                "hl-edge": rgba(hl, 0.16), "glow-selected": selected,
                "marker-line": exact.get("marker-line") or out["marker"]})
    if seeds.deep:
        out.update({name: exact[name] for name in DEEP_COLOURS})
        out.update(DEEP)
    else:
        out.update(dict.fromkeys((*DEEP_COLOURS, *DEEP), "initial"))
    return out


def contrast_report(key: str, mode: str) -> list[tuple[str, str, float, float]]:
    """(foreground, background, ratio, floor) for every pairing the stylesheet relies on."""
    t = tokens(key, mode)
    pairs: list[tuple[str, str, float]] = []
    for surface in SURFACES:
        pairs += [("ink", surface, INK_FLOOR), ("muted", surface, TEXT_FLOOR), ("link", surface, TEXT_FLOOR),
                  ("border", surface, UI_FLOOR), ("focus", surface, UI_FLOOR), ("marker", surface, UI_FLOOR)]
    pairs += [("accent-ink", "accent", TEXT_FLOOR), ("brand-ink", "brand", UI_FLOOR), ("off-ink", "off-bg", TEXT_FLOOR),
              ("done-ink", "done-bg", TEXT_FLOOR)]
    pairs += [(f"{state}-ink", f"{state}-bg", TEXT_FLOOR) for state in ("ok", "warn", "crit", "stale")]
    # session 114: the hover label on every surface; where keys are drawn (the DEEP look), a key's label, hover label
    # and selected line on its gradient
    pairs += [("hl", surface, TEXT_FLOOR) for surface in SURFACES]
    for shade in ("key-top", "key-bottom") if t["key-top"] != "initial" else ():
        pairs += [("ink", shade, INK_FLOOR), ("muted", shade, TEXT_FLOOR), ("hl", shade, TEXT_FLOOR),
                  ("marker", shade, UI_FLOOR)]
    return [(fg, bg, round(contrast(t[fg], t[bg]), 2), floor) for fg, bg, floor in pairs]


def _block(selector: str, values: dict[str, str]) -> str:
    body = " ".join(f"--{name}: {value};" for name, value in values.items())
    return f"{selector} {{ {body} }}"


@lru_cache(maxsize=None)
def palette_css() -> str:
    """The tokens of every preset, light and dark (the browser's preference, or data-theme), as CSS."""
    lines = ["/* D76: generated by sentinel.web.palettes; every pairing's contrast is checked in the tests. */"]
    for preset in PRESETS:
        light, dark = tokens(preset.key, "light"), tokens(preset.key, "dark")
        selector = f":root[data-palette={preset.key}]"
        if preset.key == DEFAULT_PALETTE:
            selector = f":root, {selector}"
        lines.append(_block(selector, {"scheme": "light", **light}).replace("--scheme: light;", "color-scheme: light;"))
        lines.append("@media (prefers-color-scheme: dark) { "
                     + _block(", ".join(f"{part.strip()}:not([data-theme=light])" for part in selector.split(",")),
                              {"scheme": "dark", **dark}).replace("--scheme: dark;", "color-scheme: dark;") + " }")
        lines.append(_block(", ".join(f"{part.strip()}[data-theme=dark]" for part in selector.split(",")),
                            {"scheme": "dark", **dark}).replace("--scheme: dark;", "color-scheme: dark;"))
    return "\n".join(lines) + "\n"


def swatch_css() -> str:
    """Each preset's named colours as the appearance panel's swatch dots (D96); classes, so no inline style is needed.
    The dots show the palette's own colours; they are never a state."""
    return "".join(f".sw-{preset.key} .d{i} {{ background: {colour.lower()}; }}\n"
                   for preset in PRESETS for i, colour in enumerate(preset.colours, 1))
