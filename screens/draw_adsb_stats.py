"""Render the ADS-B receiver stats screen.

Summarizes daily aircraft-tracking stats aggregated from the local SQLite
database that the standalone collector (``scripts/adsb_collector.py``)
writes to. This module never talks to a receiver directly — see
``services/adsb.py`` for the collector/storage layer.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

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
from utils import ScreenImage, log_call, wrap_text

_store: Optional[AdsbStore] = None
_store_failed = False


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


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if draw.textsize(text, font=font)[0] <= max_width:
        return text
    while text and draw.textsize(text + "…", font=font)[0] > max_width:
        text = text[:-1]
    return text + "…" if text else "…"


def _wrap_row_lines(
    draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int
) -> list[str]:
    lines = wrap_text(text, font, max_width) or [text]
    if len(lines) <= max_lines:
        return lines
    kept = lines[: max_lines - 1]
    remainder = " ".join(lines[max_lines - 1 :])
    kept.append(_fit_text(draw, remainder, font, max_width))
    return kept


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


def _distance_text(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.1f} {config.ADSB_DISTANCE_UNIT}"


def _format_count(value: int) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return str(value)


def _furthest_text(catch: FurthestCatch) -> str:
    parts = [_distance_text(catch.distance_nm)]
    detail_bits = [bit for bit in (catch.callsign, catch.device, _time_text(catch.seen_at)) if bit]
    if detail_bits:
        parts.append(" · ".join(detail_bits))
    return "  ".join(parts)


def _by_receiver_text(stats: DailyStats) -> str:
    labels = [device["label"] for device in config.ADSB_DEVICES] or sorted(stats.total_by_device)
    parts = [f"{label}: {stats.total_by_device.get(label, 0)}" for label in labels]
    return " · ".join(parts) if parts else "--"


def _extras_text(stats: DailyStats) -> str:
    bits: list[str] = []
    if stats.highest_altitude_ft is not None:
        bits.append(f"Alt {stats.highest_altitude_ft:,} ft")
    if stats.currently_tracked_combined:
        bits.append(f"Live {stats.currently_tracked_combined}")
    total_messages = sum(stats.messages_today_by_device.values())
    if total_messages:
        bits.append(f"Msgs {_format_count(total_messages)}")
    return " · ".join(bits)


_SIMPLE_ROW_WEIGHT = 3
_WRAPPABLE_ROW_WEIGHT = 5


def _build_rows(stats: DailyStats) -> list[tuple[str, str, int]]:
    """Return (label, value, row_weight) tuples; wrap-prone rows get more height."""

    rows: list[tuple[str, str, int]] = [
        ("By Receiver", _by_receiver_text(stats), _SIMPLE_ROW_WEIGHT)
    ]

    if stats.furthest is not None:
        rows.append(("Furthest", _furthest_text(stats.furthest), _WRAPPABLE_ROW_WEIGHT))
    else:
        rows.append(("Furthest", "No position data yet", _SIMPLE_ROW_WEIGHT))

    if stats.busiest_hour_combined is not None:
        hour, count = stats.busiest_hour_combined
        rows.append(
            ("Busiest Hour", f"{_hour_range_label(hour)}  ·  {count} aircraft", _SIMPLE_ROW_WEIGHT)
        )

    extras = _extras_text(stats)
    if extras:
        rows.append(("More", extras, _WRAPPABLE_ROW_WEIGHT))

    if stats.all_time_furthest is not None and (
        stats.furthest is None
        or stats.all_time_furthest.hex != stats.furthest.hex
        or stats.all_time_furthest.distance_nm != stats.furthest.distance_nm
    ):
        all_time_text = _furthest_text(stats.all_time_furthest)
        rows.append(("All-Time Best", all_time_text, _WRAPPABLE_ROW_WEIGHT))

    return rows


def _render_no_data(stats: Optional[DailyStats]) -> Image.Image:
    background = get_screen_background_color("adsb stats", (0, 0, 0))
    img = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)
    margin = max(4, WIDTH // 32)

    title = "ADS-B"
    draw.text((margin, margin), title, font=FONT_WEATHER_DETAILS_SMALL_BOLD, fill=(235, 235, 235))
    title_h = draw.textsize(title, font=FONT_WEATHER_DETAILS_SMALL_BOLD)[1]

    if not config.ENABLE_ADSB:
        message = "No receivers configured."
        detail = "Set ADSB_DEVICE_1_HOST in .env."
    elif stats is not None and stats.device_errors:
        message = "Receiver(s) offline."
        detail = "; ".join(f"{d}: {e}" for d, e in stats.device_errors.items())
    else:
        message = "No aircraft tracked yet today."
        detail = "Waiting for the collector service…"

    text_y = margin + title_h + max(8, HEIGHT // 20)
    max_width = WIDTH - margin * 2
    for line in _wrap_row_lines(draw, message, FONT_WEATHER_DETAILS_SMALL, max_width, 2):
        draw.text((margin, text_y), line, font=FONT_WEATHER_DETAILS_SMALL, fill=(220, 220, 220))
        text_y += draw.textsize(line, font=FONT_WEATHER_DETAILS_SMALL)[1] + 2
    text_y += 4
    for line in _wrap_row_lines(draw, detail, FONT_WEATHER_DETAILS_TINY, max_width, 3):
        draw.text((margin, text_y), line, font=FONT_WEATHER_DETAILS_TINY, fill=(150, 165, 180))
        text_y += draw.textsize(line, font=FONT_WEATHER_DETAILS_TINY)[1] + 2

    if stats is not None and stats.all_time_furthest is not None:
        footer = f"All-time best: {_furthest_text(stats.all_time_furthest)}"
        footer = _fit_text(draw, footer, FONT_WEATHER_DETAILS_TINY, max_width)
        footer_h = draw.textsize(footer, font=FONT_WEATHER_DETAILS_TINY)[1]
        draw.text(
            (margin, HEIGHT - margin - footer_h),
            footer,
            font=FONT_WEATHER_DETAILS_TINY,
            fill=(150, 165, 180),
        )

    return img


def _render_stats(stats: DailyStats) -> Image.Image:
    background = get_screen_background_color("adsb stats", (0, 0, 0))
    img = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)
    margin = max(4, WIDTH // 32)

    header_font = FONT_WEATHER_DETAILS_SMALL_BOLD
    value_font = FONT_WEATHER_DETAILS_SMALL
    label_font = FONT_WEATHER_DETAILS_TINY
    label_color = (165, 185, 205)
    value_color = (235, 242, 248)
    card_fill = (12, 28, 42)
    card_outline = (34, 70, 98)
    badge_fill = (0, 120, 170)
    badge_text = (255, 255, 255)

    title = "ADS-B"
    draw.text((margin, margin), title, font=header_font, fill=(235, 235, 235))
    title_h = draw.textsize(title, font=header_font)[1]

    badge_top = margin + title_h + max(4, HEIGHT // 40)
    badge_h = max(26, HEIGHT // 5)
    badge_box = (margin, badge_top, WIDTH - margin, badge_top + badge_h)
    draw.rounded_rectangle(badge_box, radius=8, fill=badge_fill)

    total_text = str(stats.total_combined)
    caption_text = "AIRCRAFT TODAY"
    total_w, total_h = draw.textsize(total_text, font=header_font)
    caption_w, _caption_h = draw.textsize(caption_text, font=label_font)
    center_y = badge_top + badge_h // 2
    total_xy = ((WIDTH - total_w) // 2, center_y - total_h - 1)
    draw.text(total_xy, total_text, font=header_font, fill=badge_text)
    caption_xy = ((WIDTH - caption_w) // 2, center_y + 2)
    draw.text(caption_xy, caption_text, font=label_font, fill=badge_text)

    card_top = badge_top + badge_h + max(4, HEIGHT // 40)
    card_bottom = HEIGHT - margin
    card_box = (margin, card_top, WIDTH - margin, card_bottom)
    draw.rounded_rectangle(card_box, radius=8, fill=card_fill, outline=card_outline)

    rows = _build_rows(stats)
    content_x = margin + 6
    right_edge = WIDTH - margin - 6
    label_w = (
        max(draw.textsize(label.upper(), font=label_font)[0] for label, _value, _weight in rows)
        + 10
    )
    value_x = content_x + label_w

    row_space = card_bottom - card_top - 8
    row_weights = [weight for _label, _value, weight in rows]
    total_weight = sum(row_weights)
    row_tops = [card_top + 4]
    consumed = 0
    for weight in row_weights[:-1]:
        consumed += weight
        row_tops.append(card_top + 4 + round(row_space * consumed / total_weight))
    row_bottoms = [*row_tops[1:], card_top + 4 + row_space]

    for index, (label, value, _weight) in enumerate(rows):
        y0, y1 = row_tops[index], row_bottoms[index]
        row_h = y1 - y0
        if index > 0:
            draw.line((margin + 6, y0, right_edge, y0), fill=(26, 54, 76))
        label_h = draw.textsize(label, font=label_font)[1]
        label_y = y0 + (row_h - label_h) // 2
        draw.text((content_x, label_y), label.upper(), font=label_font, fill=label_color)

        available_width = right_edge - value_x
        line_gap = 2
        value_h = draw.textsize(value, font=value_font)[1]
        max_lines = max(1, min(2, row_h // (value_h + line_gap)))
        lines = _wrap_row_lines(draw, value, value_font, available_width, max_lines)
        block_h = len(lines) * value_h + (len(lines) - 1) * line_gap
        line_y = y0 + max(0, (row_h - block_h) // 2)
        for line in lines:
            draw.text((value_x, line_y), line, font=value_font, fill=value_color)
            line_y += value_h + line_gap

    return img


@log_call
def draw_adsb_stats_screen(display, stats: Optional[DailyStats] = None, transition: bool = False):
    """Draw today's ADS-B receive stats, or a graceful "no data yet" state."""

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
        img = _render_stats(stats)

    return ScreenImage(img, displayed=False)
