"""Render the ADS-B receiver stats screen.

Summarizes daily aircraft-tracking stats aggregated from the local SQLite
database that the standalone collector (``scripts/adsb_collector.py``)
writes to. This module never talks to a receiver directly — see
``services/adsb.py`` for the collector/storage layer.

Visually this follows the same "colored badge + chart, then a row/tile of
dark stat cards" language as the air-quality and inside-sensor screens
(``draw_air_quality.py``, ``draw_inside.py``): a big headline number and a
small history-style chart up top, supporting numbers below as individually
sized cards. Every piece of text is sized with ``fit_font`` so it shrinks to
fit its box instead of wrapping; ``_fit_text`` ellipsis-truncates only as a
last resort if even the smallest allowed size would overflow.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Any, Optional

from PIL import Image, ImageDraw

import config
from config import (
    FONT_WEATHER_DETAILS_SMALL,
    FONT_WEATHER_DETAILS_SMALL_BOLD,
    FONT_WEATHER_DETAILS_TINY,
    HEIGHT,
    WIDTH,
    get_screen_background_color,
)
from services.adsb import AdsbStore, DailyStats, FurthestCatch, today_key
from utils import ScreenImage, clone_font, fit_font, log_call, measure_text

_store: Optional[AdsbStore] = None
_store_failed = False

_CARD_BG = (12, 28, 42)
_TEXT_COLOR = (235, 242, 248)
_LABEL_COLOR = (165, 185, 205)

# One accent hue per stat tile so the dashboard reads as more than a plain
# list, while staying within the dark-navy-card palette AQI/Inside already
# use (accents are mixed into _CARD_BG, never used at full saturation).
_ACCENT_FURTHEST = (90, 205, 225)
_ACCENT_RECEIVER = (110, 150, 230)
_ACCENT_ALL_TIME = (230, 185, 70)
_ACCENT_LIVE = (95, 220, 150)
_ACCENT_MESSAGES = (230, 150, 95)

_DOT_ONLINE = (95, 220, 150)
_DOT_OFFLINE = (225, 90, 90)

# Below this row height, a grid cell can no longer legibly fit its text (or,
# for the airline tile, its icon + count) -- the caption should be dropped
# first to free up room rather than let rows render illegibly small.
_MIN_GRID_ROW_H = 16

# Below this icon size, a logo or a text abbreviation is too small to read
# (a 2-3 letter airline code shrinks to an unreadable sliver) -- the airline
# tile falls back to plain "CODE: count" text instead of icon + count.
_MIN_ICON_DIM = 16

# The ADS-B dashboard is three separate registered screens (see
# screens/registry.py) rather than one screen that rotates through every
# tile combination: Furthest and By Receiver always lead, and each variant
# supplies a fixed pair of tiles for the remaining two slots.
_VARIANT_BEST = "best"
_VARIANT_LIVE = "live"
_VARIANT_LIVE_AIRLINES = "live airlines"

_AIRLINE_LOGO_EXTENSIONS = (".png", ".jpg", ".jpeg")
_AIRLINE_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}


def _get_store() -> Optional[AdsbStore]:
    global _store, _store_failed
    if not config.ENABLE_ADSB:
        return None
    if _store is None and not _store_failed:
        try:
            _store = AdsbStore()
        except Exception:
            logging.exception("ADS-B: failed to open stats database")
            _store_failed = True
    return _store


def _mix_color(
    color: tuple[int, int, int], target: tuple[int, int, int], factor: float
) -> tuple[int, int, int]:
    factor = max(0.0, min(1.0, factor))
    return tuple(round(color[i] * (1 - factor) + target[i] * factor) for i in range(3))


def _text_extent(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    """Return (top_offset, visual_height) for *text* rendered in *font*.

    ``measure_text``/``textsize`` report a tight ``bottom - top`` height that
    does not include the leading gap PIL leaves above a glyph's ink (often
    several pixels at bold/large sizes). Stacking lines by drawing the next
    one at ``y + measured_height`` therefore lands *inside* the previous
    line's true bottom and visibly overlaps it, especially once gaps shrink
    at small display sizes. Callers should draw at ``y - top_offset`` to
    make the ink start exactly at the intended ``y``, then advance by
    ``visual_height`` for the next line.
    """

    _left, top, _right, bottom = draw.textbbox((0, 0), text, font=font)
    return top, bottom - top


def _draw_line(draw: ImageDraw.ImageDraw, x: int, y_top: int, text: str, font, fill) -> int:
    """Draw *text* so its ink starts at ``y_top``; return the next ``y_top``."""

    top_offset, height = _text_extent(draw, text, font)
    draw.text((x, y_top - top_offset), text, font=font, fill=fill)
    return y_top + height


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    """Ellipsis-truncate as an absolute last resort, once fit_font has
    already shrunk to its minimum allowed size and it still overflows."""

    if measure_text(draw, text, font)[0] <= max_width:
        return text
    while text and measure_text(draw, text + "…", font)[0] > max_width:
        text = text[:-1]
    return text + "…" if text else "…"


def _activity_color(total: int, *, quiet: int = 8, busy: int = 150) -> tuple[int, int, int]:
    """Calm blue for a quiet day, warming toward gold as traffic picks up."""

    t = max(0.0, min(1.0, (total - quiet) / max(1, busy - quiet)))
    cool = (55, 130, 220)
    warm = (235, 165, 55)
    return _mix_color(cool, warm, t)


def _hour_range_label(hour: int) -> str:
    start_h = hour % 12 or 12
    start_period = "AM" if hour < 12 else "PM"
    end_hour_24 = (hour + 1) % 24
    end_h = end_hour_24 % 12 or 12
    end_period = "AM" if end_hour_24 < 12 else "PM"
    if start_period == end_period:
        return f"{start_h}–{end_h} {end_period}"
    return f"{start_h} {start_period}–{end_h} {end_period}"


def _time_text(ts: Optional[float]) -> str:
    if not ts:
        return ""
    local = dt.datetime.fromtimestamp(ts, tz=config.CENTRAL_TIME)
    return local.strftime("%I:%M %p").lstrip("0")


def _date_text(ts: Optional[float]) -> str:
    if not ts:
        return ""
    local = dt.datetime.fromtimestamp(ts, tz=config.CENTRAL_TIME)
    return f"{local:%b} {local.day}"


def _distance_text(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.1f} {config.ADSB_DISTANCE_UNIT}"


def _format_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return str(value)


def _catch_detail(catch: FurthestCatch, *, when: str) -> Optional[str]:
    """Two-line caption: date/time on the first line, flight number and
    receiver (whichever are known) on the second."""

    id_bits = [bit for bit in (catch.callsign, catch.device) if bit]
    lines = [line for line in (when, " · ".join(id_bits) if id_bits else None) if line]
    return "\n".join(lines) if lines else None


def _by_receiver_lines(stats: DailyStats) -> list[tuple[str, bool]]:
    """One ``"label: count"`` line per receiver, paired with whether that
    receiver is currently online (for the status dot drawn beside it)."""

    labels = [device["label"] for device in config.ADSB_DEVICES] or sorted(stats.total_by_device)
    return [
        (f"{label}: {stats.total_by_device.get(label, 0)}", bool(stats.device_online.get(label)))
        for label in labels
    ]


_OTHER_LABEL = "Other"


def _top_breakdown_items(counts: dict[str, int], *, max_lines: int = 4) -> list[tuple[str, int]]:
    """Top entries by current count, folding any overflow into "Other" so
    the entries always add up to the same total as the headline count.

    *counts* may already carry its own "Other" bucket (e.g. unknown
    aircraft types or unparseable callsigns) — that's merged into the
    single folded-overflow "Other" line rather than kept as a separate
    entry, so "Other" never appears twice.
    """

    if not counts:
        return []
    other_total = counts.get(_OTHER_LABEL, 0)
    named = sorted(
        ((label, count) for label, count in counts.items() if label != _OTHER_LABEL),
        key=lambda kv: (-kv[1], kv[0]),
    )
    if len(named) + (1 if other_total else 0) > max_lines:
        keep = max(0, max_lines - 1)
        other_total += sum(count for _, count in named[keep:])
        named = named[:keep]
    return named + [(_OTHER_LABEL, other_total)] if other_total else named


def _live_now_breakdown_lines(model_counts: dict[str, int], *, max_lines: int = 4) -> list[str]:
    """Top models by current count, formatted as ``"model: count"`` lines."""

    return [
        f"{model}: {count}"
        for model, count in _top_breakdown_items(model_counts, max_lines=max_lines)
    ]


def _airline_logo_path(code: str) -> Optional[str]:
    """Path to ``images/air/<code>.png``/``.jpg`` (case-insensitive), if any."""

    cleaned = (code or "").strip().lower()
    if not cleaned:
        return None
    try:
        entries = os.listdir(config.AIR_IMAGES_DIR)
    except OSError:
        return None
    for name in entries:
        stem, ext = os.path.splitext(name)
        if stem.lower() == cleaned and ext.lower() in _AIRLINE_LOGO_EXTENSIONS:
            return os.path.join(config.AIR_IMAGES_DIR, name)
    return None


def _airline_logo(code: str, height: int) -> Optional[Image.Image]:
    """Load and cache the logo for an airline code, resized to *height*.

    Returns ``None`` (no drawing performed) when no matching file exists,
    so the caller falls back to printing the airline code as text.
    """

    height = max(1, height)
    cache_key = ((code or "").strip().lower(), height)
    if cache_key in _AIRLINE_LOGO_CACHE:
        cached = _AIRLINE_LOGO_CACHE[cache_key]
        return cached.copy() if cached is not None else None

    logo = None
    path = _airline_logo_path(code)
    if path is not None:
        try:
            opened = Image.open(path).convert("RGBA")
            ratio = height / opened.height
            logo = opened.resize((max(1, round(opened.width * ratio)), height), Image.ANTIALIAS)
        except Exception:
            logging.exception("ADS-B: failed to load airline logo '%s'", path)
            logo = None

    _AIRLINE_LOGO_CACHE[cache_key] = logo
    return logo.copy() if logo is not None else None


def _status_text(stats: Optional[DailyStats]) -> str:
    device_online = stats.device_online if stats else {}
    total_devices = len(device_online) or len(config.ADSB_DEVICES)
    if not total_devices:
        return ""
    online = sum(1 for is_online in device_online.values() if is_online)
    if total_devices == 1:
        return "ONLINE" if online else "OFFLINE"
    if online == total_devices:
        return f"{total_devices} ONLINE"
    return f"{online} OF {total_devices} ONLINE"


def _draw_hour_bars(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    hourly_counts: dict[int, int],
    busiest_hour: Optional[int],
) -> None:
    """Draw a 24-bar aircraft-per-hour histogram, peak hour highlighted."""

    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    if width < 30 or height < 8 or not hourly_counts:
        return
    max_count = max(hourly_counts.values())
    if max_count <= 0:
        return

    bar_gap = 1
    bar_w = max(1, (width - 23 * bar_gap) // 24)
    used_width = bar_w * 24 + bar_gap * 23
    start_x = x0 + max(0, (width - used_width) // 2)
    baseline_y = y1
    empty_color = (255, 255, 255)

    for hour in range(24):
        count = hourly_counts.get(hour, 0)
        if count <= 0:
            continue
        bar_h = max(1, round((count / max_count) * max(1, height - 1)))
        bx0 = start_x + hour * (bar_w + bar_gap)
        is_peak = hour == busiest_hour
        color = (255, 255, 255) if is_peak else _mix_color(empty_color, (0, 0, 0), 0.45)
        draw.rectangle((bx0, baseline_y - bar_h, bx0 + bar_w - 1, baseline_y - 1), fill=color)

    draw.line((x0, baseline_y, x1, baseline_y), fill=_mix_color(empty_color, (0, 0, 0), 0.6))


def _fit_font_for_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    base_font,
    max_width: int,
    max_height: int,
    *,
    min_pt: int = 8,
    max_pt: Optional[int] = None,
):
    """Like ``fit_font``, but sizes for a set of independently-positioned
    lines (e.g. grid cells) rather than one stacked block of text: finds
    the largest size at which every line individually fits *max_width* x
    *max_height*."""

    base_size = getattr(base_font, "size", 16)
    hi = max_pt if max_pt else base_size
    lo = min_pt
    best = clone_font(base_font, lo)
    while lo <= hi:
        mid = (lo + hi) // 2
        test_font = clone_font(base_font, mid)
        if all(
            w <= max_width and h <= max_height
            for w, h in (measure_text(draw, line, test_font) for line in lines)
        ):
            best = test_font
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _draw_grid_lines(
    draw: ImageDraw.ImageDraw,
    x: int,
    y_top: int,
    max_width: int,
    max_height: int,
    lines: list[str],
    base_font,
    color: tuple[int, int, int],
    *,
    columns: int = 2,
    min_pt: int = 8,
    max_pt: Optional[int] = None,
) -> None:
    """Lay *lines* out left-to-right, top-to-bottom in *columns* columns
    instead of one line per row, so more entries fit in the same vertical
    space (e.g. the "by aircraft" tile)."""

    if not lines or max_width <= 0 or max_height <= 0:
        return
    columns = max(1, min(columns, len(lines)))
    rows = -(-len(lines) // columns)  # ceil division
    col_gap = max(6, max_width // 20) if columns > 1 else 0
    col_width = max(1, (max_width - col_gap * (columns - 1)) // columns)
    row_h = max(1, max_height // rows)

    font = _fit_font_for_lines(
        draw, lines, base_font, col_width, row_h, min_pt=min_pt, max_pt=max_pt
    )

    for index, text in enumerate(lines):
        row, col = divmod(index, columns)
        text = _fit_text(draw, text, font, col_width)
        top_offset, line_h = _text_extent(draw, text, font)
        cell_x = x + col * (col_width + col_gap)
        cell_y = y_top + row * row_h + max(0, (row_h - line_h) // 2)
        draw.text((cell_x, cell_y - top_offset), text, font=font, fill=color)


def _tile_grid_cells(
    rect: tuple[int, int, int, int], count: int
) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = rect
    width, height = x1 - x0, y1 - y0
    if count <= 0 or width <= 0 or height <= 0:
        return []

    columns = count if count <= 2 else 2
    rows = -(-count // columns)  # ceil division
    gap_x = max(6, width // 40) if columns > 1 else 0
    gap_y = max(6, height // 40) if rows > 1 else 0
    cell_w = max(1, (width - gap_x * (columns - 1)) // columns)
    cell_h = max(1, (height - gap_y * (rows - 1)) // rows)

    cells = []
    for index in range(count):
        row, col = divmod(index, columns)
        left = x0 + col * (cell_w + gap_x)
        top = y0 + row * (cell_h + gap_y)
        right = x1 if col == columns - 1 else left + cell_w
        bottom = y1 if row == rows - 1 else top + cell_h
        cells.append((left, top, right, bottom))
    return cells


def _draw_stat_tile(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    label: str,
    value: str,
    caption: Optional[str],
    accent: tuple[int, int, int],
    line_online: Optional[list[bool]] = None,
    columns: int = 1,
) -> None:
    x0, y0, x1, y1 = rect
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return

    radius = max(6, min(16, min(width, height) // 5))
    draw.rounded_rectangle(
        rect,
        radius=radius,
        fill=_mix_color(accent, _CARD_BG, 0.72),
        outline=_mix_color(accent, _CARD_BG, 0.4),
        width=1,
    )

    pad_x = max(8, width // 12)
    pad_y = max(6, height // 10)
    gap = max(2, height // 16)
    max_text_width = max(1, width - 2 * pad_x)
    content_top = y0 + pad_y
    content_bottom = y1 - pad_y

    _raw_lines = value.split("\n")
    use_grid = columns >= 2 and len(_raw_lines) > 1 and not line_online

    label_text = label.upper()
    label_font = fit_font(
        draw,
        label_text,
        FONT_WEATHER_DETAILS_TINY,
        max_width=max_text_width,
        max_height=max(9, int(height * 0.22)),
        min_pt=7,
        max_pt=FONT_WEATHER_DETAILS_TINY.size,
    )
    _, label_h = _text_extent(draw, label_text, label_font)

    caption_h = 0
    caption_font = None
    if caption:
        caption_lines = caption.count("\n") + 1
        caption_font = fit_font(
            draw,
            caption,
            FONT_WEATHER_DETAILS_TINY,
            max_width=max_text_width,
            max_height=max(9, int(height * 0.2)) * caption_lines,
            min_pt=6,
            max_pt=FONT_WEATHER_DETAILS_TINY.size,
        )
        caption = _fit_text(draw, caption, caption_font, max_text_width)
        _, caption_h = _text_extent(draw, caption, caption_font)

    value_top = content_top + label_h + gap
    value_bottom = content_bottom - (caption_h + gap if caption_h else 0)
    value_max_h = value_bottom - value_top

    min_value_h = 12
    if use_grid:
        grid_rows = -(-len(_raw_lines) // columns)  # ceil division
        min_value_h = max(min_value_h, grid_rows * _MIN_GRID_ROW_H)
    if caption_h and value_max_h < min_value_h:
        # Too tight for label + value + caption on this tile size — drop
        # the caption rather than let it overlap the value.
        caption, caption_font, caption_h = None, None, 0
        value_bottom = content_bottom
        value_max_h = value_bottom - value_top
    value_max_h = max(min_value_h, value_max_h)

    # Reserve room for a status dot to the left of each value line when the
    # caller wants receiver-online indicators (e.g. the "By Receiver" tile).
    dot_reserve = max(9, width // 14) + max(3, width // 40) if line_online else 0
    value_max_width = max(1, max_text_width - dot_reserve)

    label_color = _mix_color(accent, _TEXT_COLOR, 0.25)
    _draw_line(draw, x0 + pad_x, content_top, label_text, label_font, label_color)

    if use_grid:
        _draw_grid_lines(
            draw,
            x0 + pad_x,
            value_top,
            value_max_width,
            value_max_h,
            _raw_lines,
            FONT_WEATHER_DETAILS_SMALL_BOLD,
            _TEXT_COLOR,
            columns=columns,
            min_pt=9,
            max_pt=max(FONT_WEATHER_DETAILS_SMALL_BOLD.size, int(height * 0.28)),
        )
        if caption and caption_font is not None:
            _draw_line(
                draw,
                x0 + pad_x,
                content_bottom - caption_h,
                caption,
                caption_font,
                _mix_color(accent, _TEXT_COLOR, 0.45),
            )
        return

    value_font = fit_font(
        draw,
        value,
        FONT_WEATHER_DETAILS_SMALL_BOLD,
        max_width=value_max_width,
        max_height=value_max_h,
        min_pt=9,
        max_pt=max(FONT_WEATHER_DETAILS_SMALL_BOLD.size, int(height * 0.4)),
    )
    value = _fit_text(draw, value, value_font, value_max_width)
    _, value_h = _text_extent(draw, value, value_font)

    if caption_h and value_h > value_max_h:
        # Even fit_font's smallest allowed size doesn't fit the value above
        # the caption (e.g. a multi-line breakdown in a small tile) — drop
        # the caption rather than let the two overlap.
        caption, caption_font, caption_h = None, None, 0
        value_max_h = max(min_value_h, content_bottom - value_top)
        value_font = fit_font(
            draw,
            value,
            FONT_WEATHER_DETAILS_SMALL_BOLD,
            max_width=value_max_width,
            max_height=value_max_h,
            min_pt=9,
            max_pt=max(FONT_WEATHER_DETAILS_SMALL_BOLD.size, int(height * 0.4)),
        )
        value = _fit_text(draw, value, value_font, value_max_width)
        _, value_h = _text_extent(draw, value, value_font)

    value_draw_top = value_top + max(0, (value_max_h - value_h) // 2)

    if line_online:
        lines = value.split("\n")
        flags = list(line_online[: len(lines)]) + [False] * max(0, len(lines) - len(line_online))
        text_x = x0 + pad_x + dot_reserve
        y = value_draw_top
        for line_text, online in zip(lines, flags):
            top_offset, line_h = _text_extent(draw, line_text, value_font)
            dot_d = max(5, min(dot_reserve - 4, line_h - 2))
            dot_cy = y + line_h // 2
            dot_color = _DOT_ONLINE if online else _DOT_OFFLINE
            draw.ellipse(
                (
                    x0 + pad_x,
                    dot_cy - dot_d // 2,
                    x0 + pad_x + dot_d,
                    dot_cy - dot_d // 2 + dot_d,
                ),
                fill=dot_color,
            )
            draw.text((text_x, y - top_offset), line_text, font=value_font, fill=_TEXT_COLOR)
            y += line_h + 2
    else:
        _draw_line(draw, x0 + pad_x, value_draw_top, value, value_font, _TEXT_COLOR)

    if caption and caption_font is not None:
        _draw_line(
            draw,
            x0 + pad_x,
            content_bottom - caption_h,
            caption,
            caption_font,
            _mix_color(accent, _TEXT_COLOR, 0.45),
        )


def _draw_airline_tile(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    label: str,
    rows: list[tuple[str, int]],
    caption: Optional[str],
    accent: tuple[int, int, int],
    columns: int = 1,
) -> None:
    """Like ``_draw_stat_tile``, but each value line shows an airline logo
    (when ``images/air/<code>.png``/``.jpg`` exists) or the airline code as
    text, followed by its current aircraft count. When *columns* is 2+,
    rows are laid out left-to-right, top-to-bottom in that many columns so
    more airlines fit in the same vertical space."""

    x0, y0, x1, y1 = rect
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0 or not rows:
        return

    radius = max(6, min(16, min(width, height) // 5))
    draw.rounded_rectangle(
        rect,
        radius=radius,
        fill=_mix_color(accent, _CARD_BG, 0.72),
        outline=_mix_color(accent, _CARD_BG, 0.4),
        width=1,
    )

    pad_x = max(8, width // 12)
    pad_y = max(6, height // 10)
    gap = max(2, height // 16)
    max_text_width = max(1, width - 2 * pad_x)
    content_top = y0 + pad_y
    content_bottom = y1 - pad_y

    label_text = label.upper()
    label_font = fit_font(
        draw,
        label_text,
        FONT_WEATHER_DETAILS_TINY,
        max_width=max_text_width,
        max_height=max(9, int(height * 0.22)),
        min_pt=7,
        max_pt=FONT_WEATHER_DETAILS_TINY.size,
    )
    _, label_h = _text_extent(draw, label_text, label_font)

    caption_h = 0
    caption_font = None
    if caption:
        caption_font = fit_font(
            draw,
            caption,
            FONT_WEATHER_DETAILS_TINY,
            max_width=max_text_width,
            max_height=max(9, int(height * 0.2)),
            min_pt=6,
            max_pt=FONT_WEATHER_DETAILS_TINY.size,
        )
        caption = _fit_text(draw, caption, caption_font, max_text_width)
        _, caption_h = _text_extent(draw, caption, caption_font)

    label_color = _mix_color(accent, _TEXT_COLOR, 0.25)
    _draw_line(draw, x0 + pad_x, content_top, label_text, label_font, label_color)

    columns = max(1, min(columns, len(rows)))
    grid_rows = -(-len(rows) // columns)  # ceil division

    rows_top = content_top + label_h + gap
    rows_bottom = content_bottom - (caption_h + gap if caption_h else 0)
    rows_height = max(10, rows_bottom - rows_top)
    row_h = max(10, rows_height // grid_rows)

    if caption_h and (row_h < _MIN_GRID_ROW_H or row_h * grid_rows > rows_height):
        # Rows would render too small to be legible (or wouldn't fit at
        # all) above the caption in this tile size — drop the caption
        # rather than let the rows spill into it or shrink to illegible
        # icon/text sizes.
        caption, caption_font, caption_h = None, None, 0
        rows_bottom = content_bottom
        rows_height = max(10, rows_bottom - rows_top)
        row_h = max(10, rows_height // grid_rows)

    col_gap = max(6, max_text_width // 20) if columns > 1 else 0
    col_width = max(1, (max_text_width - col_gap * (columns - 1)) // columns)
    icon_dim = max(10, min(row_h - 4, col_width // 5))

    if icon_dim < _MIN_ICON_DIM:
        # Too little room left for a legible icon or code abbreviation --
        # fall back to plain "CODE: count" text, like the by-aircraft tile.
        _draw_grid_lines(
            draw,
            x0 + pad_x,
            rows_top,
            max_text_width,
            rows_height,
            [f"{code}: {count}" for code, count in rows],
            FONT_WEATHER_DETAILS_SMALL_BOLD,
            _TEXT_COLOR,
            columns=columns,
            min_pt=7,
            max_pt=max(FONT_WEATHER_DETAILS_SMALL_BOLD.size, int(height * 0.26)),
        )
        if caption and caption_font is not None:
            _draw_line(
                draw,
                x0 + pad_x,
                content_bottom - caption_h,
                caption,
                caption_font,
                _mix_color(accent, _TEXT_COLOR, 0.45),
            )
        return

    icon_col = icon_dim + max(4, col_width // 12)

    count_font = _fit_font_for_lines(
        draw,
        [str(count) for _, count in rows],
        FONT_WEATHER_DETAILS_SMALL_BOLD,
        max(1, col_width - icon_col),
        row_h - 2,
        min_pt=8,
        max_pt=max(FONT_WEATHER_DETAILS_SMALL_BOLD.size, int(height * 0.24)),
    )

    for index, (code, count) in enumerate(rows):
        row, col = divmod(index, columns)
        cell_x0 = x0 + pad_x + col * (col_width + col_gap)
        row_center = rows_top + row * row_h + row_h // 2
        icon_x = cell_x0
        logo = _airline_logo(code, icon_dim)
        if logo is not None:
            # Airline logos vary widely in color and many have transparent
            # backgrounds, so they can vanish against the tile's dark tinted
            # background -- a light plate behind the logo keeps it legible
            # regardless of the logo's own colors.
            plate_pad = max(1, icon_dim // 8)
            draw.rounded_rectangle(
                (
                    icon_x - plate_pad,
                    row_center - icon_dim // 2 - plate_pad,
                    icon_x + icon_dim + plate_pad,
                    row_center + icon_dim // 2 + plate_pad,
                ),
                radius=max(2, plate_pad + 1),
                fill=(245, 247, 250),
            )
            paste_x = icon_x + max(0, (icon_dim - logo.width) // 2)
            paste_y = row_center - logo.height // 2
            img.paste(logo, (paste_x, paste_y), logo)
        else:
            abbr_font = fit_font(
                draw,
                code,
                FONT_WEATHER_DETAILS_TINY,
                max_width=icon_dim,
                max_height=icon_dim,
                min_pt=6,
                max_pt=FONT_WEATHER_DETAILS_TINY.size,
            )
            abbr_text = _fit_text(draw, code, abbr_font, icon_dim)
            abbr_top_offset, abbr_h = _text_extent(draw, abbr_text, abbr_font)
            draw.text(
                (icon_x, row_center - abbr_h // 2 - abbr_top_offset),
                abbr_text,
                font=abbr_font,
                fill=_TEXT_COLOR,
            )

        count_text = _fit_text(draw, str(count), count_font, max(1, col_width - icon_col))
        top_offset, text_h = _text_extent(draw, count_text, count_font)
        draw.text(
            (cell_x0 + icon_col, row_center - text_h // 2 - top_offset),
            count_text,
            font=count_font,
            fill=_TEXT_COLOR,
        )

    if caption and caption_font is not None:
        _draw_line(
            draw,
            x0 + pad_x,
            content_bottom - caption_h,
            caption,
            caption_font,
            _mix_color(accent, _TEXT_COLOR, 0.45),
        )


def _tile_furthest(stats: DailyStats) -> dict[str, Any]:
    if stats.furthest is not None:
        return {
            "label": "Furthest",
            "value": _distance_text(stats.furthest.distance_nm),
            "caption": _catch_detail(stats.furthest, when=_time_text(stats.furthest.seen_at)),
            "accent": _ACCENT_FURTHEST,
        }
    return {
        "label": "Furthest",
        "value": "--",
        "caption": "No position data yet",
        "accent": _ACCENT_FURTHEST,
    }


def _tile_by_receiver(stats: DailyStats) -> dict[str, Any]:
    receiver_lines = _by_receiver_lines(stats)
    return {
        "label": "By Receiver",
        "value": "\n".join(text for text, _ in receiver_lines) if receiver_lines else "--",
        "caption": None,
        "accent": _ACCENT_RECEIVER,
        "line_online": [online for _, online in receiver_lines] if receiver_lines else None,
    }


def _tile_all_time_best(stats: DailyStats) -> Optional[dict[str, Any]]:
    show_all_time = stats.all_time_furthest is not None and (
        stats.furthest is None
        or stats.all_time_furthest.hex != stats.furthest.hex
        or stats.all_time_furthest.distance_nm != stats.furthest.distance_nm
    )
    if not show_all_time:
        return None
    at = stats.all_time_furthest
    return {
        "label": "All-Time Best",
        "value": _distance_text(at.distance_nm),
        "caption": _catch_detail(at, when=_date_text(at.seen_at)),
        "accent": _ACCENT_ALL_TIME,
    }


def _tile_messages(stats: DailyStats) -> Optional[dict[str, Any]]:
    total_messages = sum(stats.messages_today_by_device.values())
    if not total_messages:
        return None
    return {
        "label": "Messages",
        "value": _format_count(total_messages),
        "caption": "today",
        "accent": _ACCENT_MESSAGES,
    }


def _tile_live_total(stats: DailyStats) -> Optional[dict[str, Any]]:
    if not stats.currently_tracked_combined:
        return None
    return {
        "label": "Live Now",
        "value": str(stats.currently_tracked_combined),
        "caption": "aircraft in range",
        "accent": _ACCENT_LIVE,
    }


def _tile_live_by_aircraft(stats: DailyStats) -> Optional[dict[str, Any]]:
    if not stats.currently_tracked_combined:
        return None
    lines = _live_now_breakdown_lines(stats.currently_tracked_by_model, max_lines=6)
    if not lines:
        return None
    return {
        "label": "Live Now",
        "value": "\n".join(lines),
        "caption": "by aircraft",
        "accent": _ACCENT_LIVE,
        "columns": 2,
    }


def _tile_live_by_airline(stats: DailyStats) -> Optional[dict[str, Any]]:
    if not stats.currently_tracked_combined:
        return None
    rows = _top_breakdown_items(stats.currently_tracked_by_airline, max_lines=6)
    if not rows:
        return None
    return {
        "label": "Live Now",
        "caption": "by airline",
        "accent": _ACCENT_LIVE,
        "airline_rows": rows,
        "columns": 2,
    }


def _build_tiles(stats: DailyStats, variant: str) -> list[dict[str, Any]]:
    """Furthest and By Receiver always lead; the other two slots are a
    fixed pair of tiles for *variant* (one of ``_VARIANT_BEST``,
    ``_VARIANT_LIVE``, ``_VARIANT_LIVE_AIRLINES``) rather than a rotating
    selection, since each variant is its own registered screen."""

    tiles: list[dict[str, Any]] = [_tile_furthest(stats), _tile_by_receiver(stats)]

    if variant == _VARIANT_LIVE:
        extras = [_tile_live_total(stats), _tile_live_by_aircraft(stats)]
    elif variant == _VARIANT_LIVE_AIRLINES:
        extras = [_tile_live_by_aircraft(stats), _tile_live_by_airline(stats)]
    else:
        extras = [_tile_all_time_best(stats), _tile_messages(stats)]

    tiles.extend(tile for tile in extras if tile is not None)
    return tiles


def _render_no_data(stats: Optional[DailyStats]) -> Image.Image:
    background = get_screen_background_color("adsb stats", (0, 0, 0))
    img = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)
    margin = max(4, WIDTH // 32)
    max_width = WIDTH - margin * 2

    title = "ADS-B"
    title_bottom = _draw_line(
        draw, margin, margin, title, FONT_WEATHER_DETAILS_SMALL_BOLD, (235, 235, 235)
    )

    if not config.ENABLE_ADSB:
        message = "No receivers configured."
        detail = "Set ADSB_DEVICE_1_HOST in .env."
    elif stats is not None and stats.device_errors:
        message = "Receiver offline."
        device, error = next(iter(stats.device_errors.items()))
        detail = f"{device}: {error}"
    else:
        message = "No aircraft tracked yet today."
        detail = "Waiting for the collector service…"

    text_y = title_bottom + max(8, HEIGHT // 16)
    message_font = fit_font(
        draw,
        message,
        FONT_WEATHER_DETAILS_SMALL,
        max_width=max_width,
        max_height=max(14, HEIGHT // 8),
        min_pt=9,
        max_pt=FONT_WEATHER_DETAILS_SMALL.size,
    )
    message = _fit_text(draw, message, message_font, max_width)
    text_y = _draw_line(draw, margin, text_y, message, message_font, (220, 220, 220))
    text_y += max(4, HEIGHT // 40)

    detail_font = fit_font(
        draw,
        detail,
        FONT_WEATHER_DETAILS_TINY,
        max_width=max_width,
        max_height=max(10, HEIGHT // 10),
        min_pt=7,
        max_pt=FONT_WEATHER_DETAILS_TINY.size,
    )
    detail = _fit_text(draw, detail, detail_font, max_width)
    _draw_line(draw, margin, text_y, detail, detail_font, (150, 165, 180))

    if stats is not None and stats.all_time_furthest is not None:
        at = stats.all_time_furthest
        tile_h = max(40, HEIGHT // 4)
        tile_rect = (margin, HEIGHT - margin - tile_h, WIDTH - margin, HEIGHT - margin)
        _draw_stat_tile(
            draw,
            tile_rect,
            "All-Time Best",
            _distance_text(at.distance_nm),
            _catch_detail(at, when=_date_text(at.seen_at)),
            _ACCENT_ALL_TIME,
        )

    return img


def _render_stats(stats: DailyStats, variant: str) -> Image.Image:
    background = get_screen_background_color("adsb stats", (0, 0, 0))
    img = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)
    margin = max(4, WIDTH // 32)

    title = "ADS-B"
    title_w = measure_text(draw, title, FONT_WEATHER_DETAILS_SMALL_BOLD)[0]
    title_bottom = _draw_line(
        draw, margin, margin, title, FONT_WEATHER_DETAILS_SMALL_BOLD, (235, 235, 235)
    )
    title_h = title_bottom - margin

    status_text = _status_text(stats)
    if status_text:
        status_max_width = max(1, WIDTH - margin * 2 - title_w - 8)
        status_font = fit_font(
            draw,
            status_text,
            FONT_WEATHER_DETAILS_TINY,
            max_width=status_max_width,
            max_height=title_h,
            min_pt=6,
            max_pt=FONT_WEATHER_DETAILS_TINY.size,
        )
        status_w, status_h = measure_text(draw, status_text, status_font)
        _draw_line(
            draw,
            WIDTH - margin - status_w,
            margin + max(0, (title_h - status_h) // 2),
            status_text,
            status_font,
            _LABEL_COLOR,
        )

    # Hero badge: headline aircraft-today count, colored by how busy the day
    # has been, with a 24-hour activity histogram alongside it when there's
    # room (mirrors the AQI/Inside badge+chart layout).
    badge_top = margin + title_h + max(4, HEIGHT // 40)
    badge_h = max(40, HEIGHT // 4)
    accent = _activity_color(stats.total_combined)
    draw.rounded_rectangle(
        (margin, badge_top, WIDTH - margin, badge_top + badge_h),
        radius=8,
        fill=_mix_color(accent, background, 0.72),
    )

    badge_pad = max(8, WIDTH // 32)
    interior_left = margin + badge_pad
    interior_right = WIDTH - margin - badge_pad
    interior_width = max(1, interior_right - interior_left)

    chart_width = 0
    if stats.hourly_counts_combined and interior_width >= 90:
        chart_width = max(60, int(interior_width * 0.42))
    chart_gap = max(8, WIDTH // 40) if chart_width else 0
    number_right = interior_right - chart_width - chart_gap if chart_width else interior_right
    number_max_width = max(20, number_right - interior_left)

    total_text = str(stats.total_combined)
    total_font = fit_font(
        draw,
        total_text,
        FONT_WEATHER_DETAILS_SMALL_BOLD,
        max_width=number_max_width,
        max_height=max(20, int(badge_h * 0.6)),
        min_pt=14,
        max_pt=max(36, HEIGHT // 5),
    )
    _, total_h = _text_extent(draw, total_text, total_font)

    caption_text = "AIRCRAFT TODAY"
    caption_font = fit_font(
        draw,
        caption_text,
        FONT_WEATHER_DETAILS_TINY,
        max_width=number_max_width,
        max_height=max(9, int(badge_h * 0.2)),
        min_pt=6,
        max_pt=FONT_WEATHER_DETAILS_TINY.size,
    )
    _, caption_h = _text_extent(draw, caption_text, caption_font)

    number_gap = max(2, int(badge_h * 0.05))
    block_h = total_h + number_gap + caption_h
    block_top = badge_top + (badge_h - block_h) // 2
    next_y = _draw_line(draw, interior_left, block_top, total_text, total_font, (255, 255, 255))
    caption_color = (225, 235, 245)
    _draw_line(draw, interior_left, next_y + number_gap, caption_text, caption_font, caption_color)

    if chart_width:
        chart_left = number_right + chart_gap
        peak_reserve = max(12, int(badge_h * 0.26))
        chart_box = (
            chart_left,
            badge_top + max(6, badge_h // 6),
            interior_right,
            badge_top + badge_h - peak_reserve,
        )
        peak_hour = (stats.busiest_hour_combined or (None,))[0]
        _draw_hour_bars(draw, chart_box, stats.hourly_counts_combined, peak_hour)

        if stats.busiest_hour_combined is not None:
            peak_text = f"PEAK {_hour_range_label(stats.busiest_hour_combined[0])}"
            peak_max_width = max(1, interior_right - chart_left)
            peak_font = fit_font(
                draw,
                peak_text,
                FONT_WEATHER_DETAILS_TINY,
                max_width=peak_max_width,
                max_height=peak_reserve,
                min_pt=6,
                max_pt=FONT_WEATHER_DETAILS_TINY.size,
            )
            peak_text = _fit_text(draw, peak_text, peak_font, peak_max_width)
            peak_w, _peak_h = measure_text(draw, peak_text, peak_font)
            _draw_line(
                draw,
                chart_left + max(0, (peak_max_width - peak_w) // 2),
                chart_box[3] + max(2, badge_h // 30),
                peak_text,
                peak_font,
                (225, 235, 245),
            )

    # Stat tile row/grid below the badge.
    card_top = badge_top + badge_h + max(4, HEIGHT // 40)
    card_bottom = HEIGHT - margin
    tiles = _build_tiles(stats, variant)
    cells = _tile_grid_cells((margin, card_top, WIDTH - margin, card_bottom), len(tiles))
    for tile, cell in zip(tiles, cells, strict=False):
        if "airline_rows" in tile:
            _draw_airline_tile(
                img,
                draw,
                cell,
                tile["label"],
                tile["airline_rows"],
                tile["caption"],
                tile["accent"],
                columns=tile.get("columns", 1),
            )
        else:
            _draw_stat_tile(
                draw,
                cell,
                tile["label"],
                tile["value"],
                tile["caption"],
                tile["accent"],
                tile.get("line_online"),
                columns=tile.get("columns", 1),
            )

    return img


@log_call
def draw_adsb_stats_screen(
    display,
    stats: Optional[DailyStats] = None,
    transition: bool = False,
    variant: str = _VARIANT_BEST,
):
    """Draw today's ADS-B receive stats, or a graceful "no data yet" state.

    *variant* selects the fixed extra-tile pair (see ``_build_tiles``);
    each variant is registered as its own screen in ``screens/registry.py``.
    """

    if stats is None:
        store = _get_store()
        if store is not None:
            try:
                day = today_key(config.CENTRAL_TIME)
                stats = store.compute_daily_stats(day=day, tz=config.CENTRAL_TIME)
            except Exception:
                logging.exception("ADS-B: failed to compute daily stats")
                stats = None

    if stats is None or stats.total_combined == 0:
        img = _render_no_data(stats)
    else:
        img = _render_stats(stats, variant)

    return ScreenImage(img, displayed=False)
