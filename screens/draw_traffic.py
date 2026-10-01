"""Draw the Chicagoland traffic report (the ``traffic`` screen).

A title with the direction under it, then the Edens and Kennedy segments as
rounded rows: travel time large, speed beside it, and a coloured bar for
Travel Midwest's status (normal, heavy when ``over``, N/A when there is no
reading).  A small footer gives the source and the report's age, and says
when cached data is being shown.

The layout scales from the panel size, so the same code draws every profile
from 128x64 to 1920x1080; large panels add the road's route number and a
status line under each segment.  Data comes from :mod:`services.traffic`;
nothing here makes a network request.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

import config
from services import traffic
from utils import ScreenImage, log_call

BACKGROUND = (0, 0, 0)
TILE_FILL = (16, 21, 28)
TILE_OUTLINE = (44, 54, 66)
TEXT = (236, 239, 242)
DIM = (140, 150, 162)
FAINT = (98, 108, 120)
TITLE = (255, 255, 255)
DIRECTION = (90, 190, 255)
ROAD = (255, 206, 84)
STATUS_COLORS = {
    traffic.NORMAL: (64, 200, 112),
    traffic.ELEVATED: (255, 176, 46),
    traffic.HEAVY: (255, 76, 76),
    traffic.UNAVAILABLE: (96, 104, 116),
}
TIME_COLORS = {
    traffic.NORMAL: TEXT,
    traffic.ELEVATED: STATUS_COLORS[traffic.ELEVATED],
    traffic.HEAVY: STATUS_COLORS[traffic.HEAVY],
    traffic.UNAVAILABLE: DIM,
}
STATUS_WORDS = {
    traffic.NORMAL: "Normal",
    traffic.ELEVATED: "Slower than usual",
    traffic.HEAVY: "Heavy · well over normal",
    traffic.UNAVAILABLE: "No current reading",
}
STALE_COLOR = (255, 176, 46)
# Labels for panels too narrow for the full ones (the 128x64 OLED).
TINY_LABELS = {
    "edens_lakecook_jane_byrne": "E LkCook-Dtwn",
    "edens_lakecook_montrose": "E LkCook-Mont",
    "kennedy_montrose_jane_byrne": "K Mont-Dtwn",
    "kennedy_reversible_inbound": "K Rev Mont-Ohio",
    "kennedy_jane_byrne_montrose": "K Dtwn-Mont",
    "kennedy_reversible_outbound": "K Rev Ohio-Mont",
    "edens_jane_byrne_lakecook": "E Dtwn-LkCook",
    "edens_montrose_lakecook": "E Mont-LkCook",
}
ATTRIBUTION_SHORT = "Travel Midwest"
ATTRIBUTION_LONG = "Travel Midwest · IDOT, Illinois Tollway"


@lru_cache(maxsize=128)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(config.FONTS_DIR, name), max(6, int(size)))


def _width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    return int(draw.textlength(text, font=font))


def _height(font) -> int:
    top, bottom = font.getbbox("Hg")[1], font.getbbox("Hg")[3]
    return bottom - top


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


def _text(draw, xy, text, font, fill, anchor="ls"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def freshness_text(report: dict[str, Any], *, long: bool = False) -> str:
    age = report.get("age_minutes")
    if age is None:
        age_text = "age unknown"
    elif age < 1:
        age_text = "just now" if not long else "updated just now"
    else:
        age_text = f"{age} min old"
    source = ATTRIBUTION_LONG if long else ATTRIBUTION_SHORT
    if report.get("stale"):
        return f"Cached · {source} · {age_text}"
    return f"{source} · {age_text}"


def _label(row: dict[str, Any]) -> str:
    return f"Reversible · {row['label']}" if row.get("reversible") else row["label"]


def _secondary(row: dict[str, Any]) -> str:
    if row["status"] == traffic.UNAVAILABLE and row.get("reversible"):
        return "Reversible lanes not open this way"
    return STATUS_WORDS[row["status"]]


# ── Layouts ─────────────────────────────────────────────────────────────────


def _draw_header(draw, width: int, scale: float, margin: int, direction: Optional[str]) -> int:
    """TRAFFIC with the direction under it; returns the y below the header."""

    title_font = _font(round(22 * scale), bold=True)
    title_h = _height(title_font)
    y = margin + title_h
    _text(draw, (margin, y), "TRAFFIC", title_font, TITLE)
    if direction:
        dir_font = _font(round(13 * scale), bold=True)
        y += round(4 * scale) + _height(dir_font)
        _text(draw, (margin, y), direction.upper(), dir_font, DIRECTION)
    return y + round(5 * scale)


def _draw_footer(draw, width: int, height: int, scale: float, margin: int, report: dict[str, Any]) -> int:
    """The source/freshness line; returns the y above it."""

    long = width >= 700
    font, text = _fit(draw, freshness_text(report, long=long), round(10 * scale), width - 2 * margin)
    baseline = height - margin
    _text(draw, (margin, baseline), text, font, STALE_COLOR if report.get("stale") else FAINT)
    return baseline - _height(font) - round(4 * scale)


def _draw_row(draw, box, row: dict[str, Any], scale: float, *, detailed: bool) -> None:
    x0, y0, x1, y1 = box
    radius = max(3, round(7 * scale))
    draw.rounded_rectangle(box, radius=radius, fill=TILE_FILL, outline=TILE_OUTLINE, width=max(1, round(scale)))
    status = row["status"]
    bar = max(3, round(5 * scale))
    inset = max(2, round(4 * scale))
    draw.rounded_rectangle((x0 + inset, y0 + inset, x0 + inset + bar, y1 - inset),
                           radius=max(1, bar // 2), fill=STATUS_COLORS[status])
    tile_h = y1 - y0
    pad = inset * 2 + bar + max(3, round(6 * scale))

    # Right side: travel time (large), then speed (secondary).
    time_size = min(round(24 * scale), int(tile_h * (0.56 if detailed else 0.66)))
    unit_size = max(7, round(time_size * 0.5))
    time_font, unit_font = _font(time_size, bold=True), _font(unit_size)
    speed_font = _font(max(7, round(time_size * 0.55)))
    right = x1 - max(4, round(8 * scale))
    mid = (y0 + y1) // 2
    baseline = mid + _height(time_font) // 2
    speed_text = f"{row['speed']} mph" if row.get("speed") is not None else ""
    speed_col = _width(draw, "88 mph", speed_font)
    show_speed = (x1 - x0) >= 260 * min(scale, 1.5) or scale >= 1.5
    if show_speed:
        if speed_text:
            _text(draw, (right, baseline), speed_text, speed_font, DIM, anchor="rs")
        right -= speed_col + max(6, round(10 * scale))
    if row.get("travel_time") is None:
        _text(draw, (right, baseline), "N/A", time_font, TIME_COLORS[status], anchor="rs")
        time_left = right - _width(draw, "N/A", time_font)
    else:
        _text(draw, (right, baseline), "min", unit_font, DIM, anchor="rs")
        unit_left = right - _width(draw, "min", unit_font) - max(2, round(3 * scale))
        _text(draw, (unit_left, baseline), str(row["travel_time"]), time_font, TIME_COLORS[status], anchor="rs")
        time_left = unit_left - _width(draw, "888", time_font)

    # Left side: the segment, and on large panels its status under it.
    label_max = max(20, time_left - (x0 + pad) - max(4, round(6 * scale)))
    label_size = min(round(15 * scale), int(tile_h * (0.36 if detailed else 0.44)))
    label_font, label = _fit(draw, _label(row), label_size, label_max)
    if detailed:
        sub_font, sub = _fit(draw, _secondary(row), max(7, round(label_size * 0.7)), label_max)
        gap = max(2, round(3 * scale))
        block = _height(label_font) + gap + _height(sub_font)
        top = mid - block // 2
        _text(draw, (x0 + pad, top + _height(label_font)), label, label_font, TEXT)
        sub_color = STATUS_COLORS[status] if status != traffic.NORMAL else DIM
        _text(draw, (x0 + pad, top + block), sub, sub_font, sub_color)
    else:
        _text(draw, (x0 + pad, mid + _height(label_font) // 2), label, label_font, TEXT)


def _compose_standard(width: int, height: int, report: Optional[dict[str, Any]],
                      direction: str) -> Image.Image:
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    scale = min(width / 320, height / 240)
    margin = max(4, round(7 * scale))
    top = _draw_header(draw, width, scale, margin, direction)
    if report is None:
        font, text = _fit(draw, "Traffic data unavailable", round(16 * scale), width - 2 * margin, bold=True)
        _text(draw, (width // 2, (top + height) // 2), text, font, DIM, anchor="mm")
        return image
    bottom = _draw_footer(draw, width, height, scale, margin, report)
    detailed = scale >= 1.9
    road_font = _font(round(12 * scale), bold=True)
    route_font = _font(round(10 * scale))
    road_h = _height(road_font) + round(5 * scale)
    groups = report["groups"]
    rows = sum(len(group["rows"]) for group in groups)
    row_gap = max(2, round(4 * scale))
    group_gap = max(2, round(5 * scale))
    available = bottom - top - len(groups) * road_h - (rows - len(groups)) * row_gap \
        - (len(groups) - 1) * group_gap
    row_h = max(12, available // max(1, rows))
    y = top
    for index, group in enumerate(groups):
        if index:
            y += group_gap
        baseline = y + _height(road_font)
        _text(draw, (margin, baseline), group["road"].upper(), road_font, ROAD)
        if detailed and group.get("route"):
            route_x = margin + _width(draw, group["road"].upper(), road_font) + round(8 * scale)
            _text(draw, (route_x, baseline), group["route"], route_font, DIM)
        y += road_h
        for row_index, row in enumerate(group["rows"]):
            if row_index:
                y += row_gap
            _draw_row(draw, (margin, y, width - margin, y + row_h), row, scale, detailed=detailed)
            y += row_h
    return image


def _compose_compact(width: int, height: int, report: Optional[dict[str, Any]],
                     direction: str) -> Image.Image:
    """Short panels (240x135, 128x64): one header line, then one line per segment."""

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    tiny = width < 200
    margin = 1 if tiny else 4
    header_size = max(8, height // 9)
    header = f"TRAFFIC · {direction.upper()}" if not tiny else f"TRAFFIC {direction[:3].upper()}"
    header_font, header = _fit(draw, header, header_size, width - 2 * margin, bold=True)
    _text(draw, (margin, margin + _height(header_font)), header, header_font, DIRECTION if not tiny else TITLE)
    top = margin + _height(header_font) + (2 if tiny else 5)
    if report is None:
        font, text = _fit(draw, "Traffic data unavailable", header_size, width - 2 * margin)
        _text(draw, (width // 2, (top + height) // 2), text, font, DIM, anchor="mm")
        return image
    footer_h = 0
    if not tiny:
        font, text = _fit(draw, freshness_text(report), max(7, height // 14), width - 2 * margin)
        _text(draw, (margin, height - margin), text, font, STALE_COLOR if report.get("stale") else FAINT)
        footer_h = _height(font) + 3
    rows = [row for group in report["groups"] for row in group["rows"]]
    gap = 0 if tiny else 2
    row_h = max(8, (height - top - footer_h - margin - gap * (len(rows) - 1)) // len(rows))
    y = top
    for row in rows:
        status = row["status"]
        if not tiny:
            draw.rounded_rectangle((margin, y, width - margin, y + row_h - 1), radius=3,
                                   fill=TILE_FILL, outline=TILE_OUTLINE)
            draw.rectangle((margin + 2, y + 2, margin + 4, y + row_h - 3), fill=STATUS_COLORS[status])
        value = "N/A" if row.get("travel_time") is None else f"{row['travel_time']}m"
        if row.get("over") and tiny and row.get("travel_time") is not None:
            value += "!"
        font_size = max(7, row_h - (1 if tiny else 4))
        value_font = _font(font_size, bold=True)
        right = width - margin - (0 if tiny else 4)
        baseline = y + (row_h + _height(value_font)) // 2 - (0 if tiny else 1)
        _text(draw, (right, baseline), value, value_font, TIME_COLORS[status] if not tiny else TEXT, anchor="rs")
        left = margin + (0 if tiny else 8)
        road = row["key"].split("_", 1)[0].capitalize()
        text = TINY_LABELS[row["key"]] if tiny else f"{road} {'Rev ' if row.get('reversible') else ''}{row['label']}"
        label_font, text = _fit(draw, text, font_size - (0 if tiny else 1),
                                right - _width(draw, value, value_font) - left - 3, minimum=6)
        _text(draw, (left, baseline), text, label_font, TEXT)
        y += row_h + gap
    return image


def compose_traffic_image(payload: Any, direction: str, *, width: Optional[int] = None,
                          height: Optional[int] = None, now: Optional[float] = None) -> Image.Image:
    """The traffic screen for *direction* from a ``traffic`` feed payload."""

    width = int(width or config.WIDTH)
    height = int(height or config.HEIGHT)
    direction = direction if direction in traffic.DIRECTIONS else traffic.INBOUND
    report = traffic.select(payload, direction, now=time.time() if now is None else now)
    if height < 200 or width < 200:
        return _compose_compact(width, height, report, direction)
    return _compose_standard(width, height, report, direction)


@log_call
def draw_traffic(display, payload: Any = None, direction: Optional[str] = None,
                 transition: bool = False) -> ScreenImage:
    """Draw the traffic screen.

    *payload* is the ``traffic`` feed value; ``None`` (standalone) reads the
    shared five-minute cache, which falls back to the last good report.
    *direction* defaults to this display's own (:func:`traffic.local_direction`).
    """

    if payload is None:
        payload = traffic.get_report()
    image = compose_traffic_image(payload, direction or traffic.local_direction())
    return ScreenImage(image, displayed=False)


__all__ = ["compose_traffic_image", "draw_traffic", "freshness_text"]
