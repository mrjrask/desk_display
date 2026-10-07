"""Draw the LAN's Raspberry Pi CPU temperatures (the ``pi remote temp`` screen).

The screen copies the MMM-RemoteTempMonitor MagicMirror module's look: black
background, an uppercase MagicMirror module header with its grey underline,
then the module's table, "Device" / "°C" / "°F" headers over a grey rule and
one row per device, hottest first.  The device name (with its Pi model and
RAM, as the module shows them) is in MagicMirror's grey; the temperatures are
bold, right-aligned and coloured with the module's scale (green, yellow-green,
orange, red, purple) and soft glow.

The layout scales from the panel size, so the same code draws every profile
from 128x64 to 1920x1080: rows grow to fill the panel, narrow panels drop
the °C column, and devices that do not fit at a readable size are counted
in a "+N more" line.  Data comes from :mod:`services.remote_temps`; nothing
here makes a network request.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import config
from services import remote_temps
from utils import ScreenImage, log_call

TITLE = "Pi Remote Temp"
# MagicMirror's palette (main.css) and the module's stylesheet.
BACKGROUND = (0, 0, 0)
TEXT = (153, 153, 153)  # --color-text #999
DIMMED = (102, 102, 102)  # --color-text-dimmed #666, the header and table rules
STALE_COLOR = (255, 176, 46)
LEVEL_COLORS = {
    remote_temps.NORMAL: (0, 255, 0),  # #00ff00
    remote_temps.WARM: (154, 205, 50),  # #9acd32
    remote_temps.HOT: (255, 140, 0),  # #ff8c00
    remote_temps.VERY_HOT: (255, 0, 0),  # #ff0000
    remote_temps.CRITICAL: (148, 0, 211),  # #9400d3
}
# text-shadow opacity and blur (px at MagicMirror's 20 px font) per level.
GLOW = {
    remote_temps.NORMAL: (0.5, 5),
    remote_temps.WARM: (0.5, 5),
    remote_temps.HOT: (0.5, 5),
    remote_temps.VERY_HOT: (0.6, 8),
    remote_temps.CRITICAL: (0.8, 10),
}
# MagicMirror's 20 px font on a 320x240 panel, scaled with the panel.
BASE_FONT = 20
LINE_HEIGHT = 1.25  # MagicMirror's .small: 20 px font, 25 px line
MIN_FONT = 8


@lru_cache(maxsize=128)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(config.FONTS_DIR, name), max(6, int(size)))


def _width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    return int(draw.textlength(text, font=font))


def _cap(font) -> int:
    """Height of a capital letter above the baseline."""

    return -font.getbbox("H", anchor="ls")[1]


def _fit(draw: ImageDraw.ImageDraw, text: str, size: int, max_width: int, *, bold: bool = False,
         minimum: int = 7):
    """The largest font up to *size* that fits *text* in *max_width* (and the text, cut if needed)."""

    size = int(size)
    while size > minimum and _width(draw, text, _font(size, bold)) > max_width:
        size -= 1
    font = _font(size, bold)
    if _width(draw, text, font) <= max_width:
        return font, text
    while len(text) > 1 and _width(draw, text + "…", font) > max_width:
        text = text[:-1]
    return font, text.rstrip() + "…"


def _message(image: Image.Image, top: int, text: str, scale: float, margin: int) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font, text = _fit(draw, text, round(BASE_FONT * 0.8 * scale), image.width - 2 * margin)
    draw.text((image.width // 2, (top + image.height) // 2), text, font=font, fill=DIMMED, anchor="mm")
    return image


def freshness_text(report: dict[str, Any]) -> str:
    age = report.get("age_minutes") or 0
    return f"Cached · {age} min old" if age >= 1 else "Cached"


def compose_remote_temps_image(payload: Any, *, width: Optional[int] = None, height: Optional[int] = None,
                               now: Optional[float] = None) -> Image.Image:
    """The pi remote temp screen from a ``remote_temps`` feed payload."""

    width = int(width or config.WIDTH)
    height = int(height or config.HEIGHT)
    report = remote_temps.select(payload, now=time.time() if now is None else now)
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    scale = min(width / 320, height / 240)
    tiny = height < 100
    margin = 1 if tiny else max(3, round(8 * scale))
    rule = max(1, round(scale))

    # The MagicMirror module header: uppercase, small, grey underline.  It is
    # drawn once the table's width is known, so the two line up.
    top = margin
    header_font = None
    if not tiny:
        header_font = _font(round(BASE_FONT * 0.75 * scale))
        header_y = top + _cap(header_font)
        top = header_y + max(2, round(5 * scale))
        header_rule_y = top
        top += rule + max(2, round(8 * scale))

    def draw_header(left: int, right: int) -> None:
        if header_font is None:
            return
        font, text = _fit(draw, TITLE.upper(), header_font.size, right - left)
        draw.text((left, header_y), text, font=font, fill=TEXT, anchor="ls")
        draw.rectangle((left, header_rule_y, right - 1, header_rule_y + rule - 1), fill=DIMMED)

    if report is None or not report["rows"]:
        draw_header(margin, width - margin)
        text = "Temperature monitors unavailable" if report is None else "No temperature monitors found"
        return _message(image, top, text, scale, margin)
    rows = report["rows"]

    bottom = height - margin
    if report.get("stale"):
        stale_font, stale_text = _fit(draw, freshness_text(report), max(7, round(10 * scale)), width - 2 * margin)
        draw.text((margin, bottom), stale_text, font=stale_font, fill=STALE_COLOR, anchor="ls")
        bottom -= _cap(stale_font) + max(2, round(4 * scale))

    # Row font: as large as the panel allows, from MagicMirror's 20 px up to
    # 1.5x, shrinking (to MIN_FONT) as devices are added.
    gap_after_rule = max(1, round(3 * scale))
    available = bottom - top - rule - gap_after_rule
    max_font = round(BASE_FONT * 1.5 * scale) if not tiny else height // 5
    min_font = max(MIN_FONT if not tiny else 7, round(MIN_FONT * scale))
    shown = len(rows)
    while True:
        lines = shown + 1 + (1 if shown < len(rows) else 0)
        font_size = min(max_font, int(available / (lines * LINE_HEIGHT * 1.1)))
        if font_size >= min_font or shown <= 1:
            break
        shown -= 1
    font_size = max(font_size, 6)
    # Columns: °C and °F right-aligned like the module; narrow panels show °F only.
    columns = ["celsius", "fahrenheit"] if width >= 200 else ["fahrenheit"]
    column_gap = max(4, round(14 * scale)) if not tiny else 3
    left, right = margin, width - margin
    names = [row["name"] for row in rows[:shown]] + ["Device"]

    def needed(size: int) -> int:
        column = max(_width(draw, "188.8", _font(round(size * 1.1), bold=True)), _width(draw, "°F", _font(size, True)))
        return max(_width(draw, name, _font(size)) for name in names) + len(columns) * (column_gap + column)

    # Shrink the whole table (not just the names) until the longest name fits.
    while font_size > min_font and needed(font_size) > right - left:
        font_size -= 1
    line_h = max(font_size + 1, round(font_size * LINE_HEIGHT * 1.1))
    # A table narrowed by a long name spreads its rows (up to 1.6x) over the panel.
    lines = shown + 1 + (1 if shown < len(rows) else 0)
    line_h = max(line_h, min(available // lines, round(line_h * 1.6)))
    head_font = _font(font_size, bold=True)
    temp_font = _font(round(font_size * 1.1), bold=True)
    column_w = max(_width(draw, "188.8", temp_font), _width(draw, "°F", head_font))
    name_size = font_size
    name_w_needed = min(needed(font_size), right - left) - len(columns) * (column_gap + column_w)
    table_w = name_w_needed + len(columns) * (column_gap + column_w)
    if table_w < right - left and width > height * 1.4:
        # Wide panels: keep the table compact like the module, centered.
        table_w = max(table_w, round((right - left) * 0.6))
        left = (width - table_w) // 2
        right = left + table_w
    column_right = [right - (len(columns) - 1 - index) * (column_w + column_gap) for index in range(len(columns))]
    name_max = column_right[0] - column_w - column_gap - left
    draw_header(left, right)

    # Table header and its rule.
    y = top + line_h // 2 + _cap(head_font) // 2
    draw.text((left, y), "Device", font=head_font, fill=TEXT, anchor="ls")
    for key, x in zip(columns, column_right):
        draw.text((x, y), "°C" if key == "celsius" else "°F", font=head_font, fill=TEXT, anchor="rs")
    y = top + line_h
    draw.rectangle((left, y, right - 1, y + rule - 1), fill=DIMMED)
    y += rule + gap_after_rule

    glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    strong = Image.new("RGBA", image.size, (0, 0, 0, 0))
    strong_draw = ImageDraw.Draw(strong)
    texts = []
    for row in rows[:shown]:
        baseline = y + line_h // 2 + _cap(temp_font) // 2
        font, name = _fit(draw, row["name"], name_size, name_max, minimum=name_size)
        draw.text((left, baseline), name, font=font, fill=TEXT, anchor="ls")
        color = LEVEL_COLORS[row["level"]]
        alpha, _blur = GLOW[row["level"]]
        target = strong_draw if row["level"] in (remote_temps.VERY_HOT, remote_temps.CRITICAL) else glow_draw
        for key, x in zip(columns, column_right):
            value = f"{row[key]:.1f}"
            target.text((x, baseline), value, font=temp_font, fill=(*color, round(255 * alpha)), anchor="rs")
            texts.append(((x, baseline), value, color))
        y += line_h
    if not tiny:
        for layer, level in ((glow, remote_temps.NORMAL), (strong, remote_temps.CRITICAL)):
            radius = GLOW[level][1] / 2 * font_size / BASE_FONT
            blurred = layer.filter(ImageFilter.GaussianBlur(max(1.0, radius)))
            image.paste(blurred, (0, 0), blurred)
        draw = ImageDraw.Draw(image)
    for xy, value, color in texts:
        draw.text(xy, value, font=temp_font, fill=color, anchor="rs")

    if shown < len(rows):
        more_font = _font(max(6, round(font_size * 0.8)))
        draw.text((left, y + line_h // 2 + _cap(more_font) // 2), f"+{len(rows) - shown} more",
                  font=more_font, fill=DIMMED, anchor="ls")
    return image


@log_call
def draw_remote_temps(display, payload: Any = None, transition: bool = False) -> ScreenImage:
    """Draw the pi remote temp screen.

    *payload* is the ``remote_temps`` feed value; ``None`` (standalone) reads
    the shared one-minute cache, which falls back to the last good snapshot.
    """

    if payload is None:
        payload = remote_temps.get_snapshot()
    return ScreenImage(compose_remote_temps_image(payload), displayed=False)


__all__ = ["compose_remote_temps_image", "draw_remote_temps", "freshness_text"]
