#!/usr/bin/env python3
"""Travel time screen helpers."""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from config import (
    CENTRAL_TIME,
    FONT_TITLE_SPORTS,
    FONT_TRAVEL_HEADER,
    FONT_TRAVEL_VALUE,
    GOOGLE_MAPS_API_KEY,
    HEIGHT,
    IMAGES_DIR,
    TRAVEL_ACTIVE_WINDOW,
    TRAVEL_DESTINATION,
    TRAVEL_DIRECTIONS_URL,
    TRAVEL_ORIGIN,
    TRAVEL_TITLE,
    WIDTH,
)
from utils import (
    ScreenImage,
    choose_route_by_any,
    fastest_route,
    fetch_directions_routes,
    format_duration_text,
    log_call,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: Google Directions fetching/parsing
# ──────────────────────────────────────────────────────────────────────────────

def _api_key() -> str:
    return os.environ.get("GOOGLE_MAPS_API_KEY") or GOOGLE_MAPS_API_KEY

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: Google Directions fetching/parsing
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_routes(avoid_highways: bool = False) -> List[Dict[str, Any]]:
    return fetch_directions_routes(
        TRAVEL_ORIGIN,
        TRAVEL_DESTINATION,
        _api_key(),
        avoid_highways=avoid_highways,
        url=TRAVEL_DIRECTIONS_URL,
    )


def _pop_route(pool: List[dict], tokens: Sequence[str]) -> Optional[dict]:
    match = choose_route_by_any(pool, list(tokens))
    if match:
        pool.remove(match)
        return match

    fallback = fastest_route(pool)
    if fallback and fallback in pool:
        pool.remove(fallback)
    return fallback

@dataclass
class TravelTimeResult:
    """Container for travel time results."""

    raw_text: str
    seconds: Optional[int] = None

    @classmethod
    def from_route(cls, route: Optional[dict]) -> "TravelTimeResult":
        if not route:
            return cls("N/A")

        seconds_raw = route.get("_duration_sec")
        seconds: Optional[int]
        if isinstance(seconds_raw, (int, float)):
            seconds = int(seconds_raw)
        else:
            seconds = None

        return cls(format_duration_text(route), seconds)

    def normalized(self) -> str:
        text = (self.raw_text or "").strip()

        if self.seconds is not None:
            minutes = max(1, math.ceil(self.seconds / 60))
            return f"{minutes} min"

        hours_match = re.search(r"(\d+)\s*hour", text)
        minutes_match = re.search(r"(\d+)\s*min", text)

        total_minutes = 0
        if hours_match:
            total_minutes += int(hours_match.group(1)) * 60
        if minutes_match:
            total_minutes += int(minutes_match.group(1))

        if total_minutes:
            return f"{total_minutes} min"

        if text:
            return text.replace("mins", "min")

        return "N/A"


SCROLL_STEP = 2
SCROLL_DELAY = 0.035
SCROLL_PAUSE_TOP = 1.0
SCROLL_PAUSE_BOTTOM = 1.0


def _coerce_time(value: Any) -> Optional[dt.time]:
    """Best-effort conversion of ``value`` into a ``datetime.time`` instance."""

    if isinstance(value, dt.time):
        return value

    if isinstance(value, dt.datetime):
        return value.timetz() if value.tzinfo else value.time()

    if isinstance(value, str):
        text = value.strip()
        formats = ["%H:%M", "%I:%M%p", "%I:%M %p", "%H%M", "%I%p"]
        for fmt in formats:
            try:
                return dt.datetime.strptime(text, fmt).time()
            except ValueError:
                continue
        logging.warning("Travel screen: could not parse time string '%s'.", value)

    return None


def get_travel_active_window() -> Optional[Tuple[dt.time, dt.time]]:
    """Return the configured travel active window as ``(start, end)`` times.

    The configuration can provide native ``datetime.time`` objects or strings in
    a handful of common formats (e.g. ``"14:30"`` or ``"2:30 PM"``). If the
    window cannot be interpreted, ``None`` is returned which callers may treat
    as "always active".
    """

    window = TRAVEL_ACTIVE_WINDOW

    if not window:
        return None

    if not isinstance(window, (tuple, list)) or len(window) != 2:
        logging.warning(
            "Travel screen: invalid active window %r (expected 2-item tuple).", window
        )
        return None

    start_raw, end_raw = window
    start = _coerce_time(start_raw)
    end = _coerce_time(end_raw)

    if not start or not end:
        logging.warning(
            "Travel screen: active window contains invalid times (%r, %r).",
            start_raw,
            end_raw,
        )
        return None

    return start, end


def get_travel_times() -> Dict[str, TravelTimeResult]:
    """Return formatted travel times keyed by route identifier."""

    try:
        routes_all = list(_fetch_routes(avoid_highways=False))
        remaining = list(routes_all)

        lake_shore_tokens = [
            "lake shore",
            "lake shore dr",
            "lake shore drive",
            "us-41",
            "us 41",
            "lsd",
            "sheridan",
            "sheridan rd",
            "sheridan road",
            "dundee",
            "dundee rd",
            "dundee road",
        ]
        kennedy_edens_tokens = [
            "edens",
            "edens expressway",
            "i-94",
            "i 94",
            "i94",
            "90/94",
            "kennedy",
            "dan ryan",
        ]
        kennedy_294_tokens = [
            "i-294",
            "i 294",
            "i294",
            "294",
            "294 tollway",
            "tri-state",
            "willow",
            "willow rd",
            "willow road",
        ]

        lake_shore = _pop_route(remaining, lake_shore_tokens)
        kennedy_edens = _pop_route(remaining, kennedy_edens_tokens)
        kennedy_294 = _pop_route(remaining, kennedy_294_tokens)

        return {
            "lake_shore": TravelTimeResult.from_route(lake_shore),
            "kennedy_edens": TravelTimeResult.from_route(kennedy_edens),
            "kennedy_294": TravelTimeResult.from_route(kennedy_294),
        }
    except Exception as exc:  # pragma: no cover - defensive guard for runtime issues
        logging.warning("Travel time parse failed: %s", exc)
        return {
            "lake_shore": TravelTimeResult("N/A"),
            "kennedy_edens": TravelTimeResult("N/A"),
            "kennedy_294": TravelTimeResult("N/A"),
        }

# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers for travel route icons
# ──────────────────────────────────────────────────────────────────────────────

def _load_icon(path: str, height: int = 52) -> Image.Image:
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        logging.warning("Travel screen: could not load image %s", path)
        return Image.new("RGBA", (height, height), (0, 0, 0, 0))

    if img.height != height and img.height > 0:
        ratio = height / float(img.height)
        width = max(1, int(img.width * ratio))
        img = img.resize((width, height), Image.LANCZOS)

    return img


def _compose_icons(paths: Sequence[str], height: int = 52, gap: int = 2) -> Image.Image:
    icons = [_load_icon(path, height=height) for path in paths]
    valid_icons = [icon for icon in icons if icon.width > 0 and icon.height > 0]

    if not valid_icons:
        return Image.new("RGBA", (height, height), (0, 0, 0, 0))

    width = sum(icon.width for icon in valid_icons) + gap * (len(valid_icons) - 1)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    x = 0
    for icon in valid_icons:
        canvas.paste(icon, (x, 0), icon)
        x += icon.width + gap

    return canvas


TRAVEL_ICON_LSD = os.path.join(IMAGES_DIR, "travel", "lsd.png")
TRAVEL_ICON_90 = os.path.join(IMAGES_DIR, "travel", "90.png")
TRAVEL_ICON_94 = os.path.join(IMAGES_DIR, "travel", "94.png")
TRAVEL_ICON_294 = os.path.join(IMAGES_DIR, "travel", "294.png")
ROUTE_ICON_HEIGHT = 33

# ──────────────────────────────────────────────────────────────────────────────
# Composition
# ──────────────────────────────────────────────────────────────────────────────

def _compose_travel_image(times: Dict[str, TravelTimeResult]) -> Image.Image:
    """Return a rendered travel time image for the provided timings."""

    measurement_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def _measure(text: str, font) -> Tuple[int, int]:
        if hasattr(measurement_draw, "textbbox"):
            left, top, right, bottom = measurement_draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
        return measurement_draw.textsize(text, font=font)

    lane_definitions: List[
        Tuple[str, str, Callable[[], Image.Image], Tuple[int, int, int]]
    ] = [
        (
            "lake_shore",
            "Lake Shore → Sheridan → Dundee",
            lambda: _compose_icons([TRAVEL_ICON_LSD], height=ROUTE_ICON_HEIGHT),
            (120, 200, 255),
        ),
        (
            "kennedy_edens",
            "Kennedy → Edens → Dundee",
            lambda: _compose_icons([TRAVEL_ICON_90, TRAVEL_ICON_94], height=ROUTE_ICON_HEIGHT),
            (200, 170, 255),
        ),
        (
            "kennedy_294",
            "Kennedy → 294 → Willow",
            lambda: _compose_icons([TRAVEL_ICON_90, TRAVEL_ICON_294], height=ROUTE_ICON_HEIGHT),
            (255, 200, 160),
        ),
    ]

    rows: List[Dict[str, Any]] = []
    time_font = FONT_TRAVEL_VALUE

    max_sign_height = 0
    max_time_height = 0

    for key, _label, factory, color in lane_definitions:
        time_result = times.get(key, TravelTimeResult("N/A"))
        normalized = time_result.normalized()
        sign_image = factory()
        time_width, time_height = _measure(normalized, time_font)

        max_sign_height = max(max_sign_height, sign_image.height)
        max_time_height = max(max_time_height, time_height)

        rows.append(
            {
                "sign": sign_image,
                "normalized": normalized,
                "color": color,
                "time_width": time_width,
                "time_height": time_height,
            }
        )

    title_width, title_height = _measure(TRAVEL_TITLE, FONT_TITLE_SPORTS)

    outer_margin = 4
    row_padding_x = 10
    row_padding_y = 4
    row_gap = 4
    header_gap = 4
    row_height = max(max_sign_height, max_time_height) + 2 * row_padding_y

    all_na = all(row["normalized"].upper() == "N/A" for row in rows)
    warning_text = "Travel data unavailable · Check Google Directions API"
    warning_width, warning_height = _measure(warning_text, FONT_TRAVEL_HEADER)

    row_count = len(rows)
    content_height = (
        title_height
        + header_gap
        + row_count * row_height
        + max(0, row_count - 1) * row_gap
        + outer_margin
    )
    if all_na:
        content_height += warning_height + 6

    canvas_height = max(content_height, HEIGHT)
    img = Image.new("RGB", (WIDTH, canvas_height), "black")
    draw = ImageDraw.Draw(img)

    y = title_height + header_gap
    row_left = outer_margin
    row_right = WIDTH - outer_margin

    # When the travel screen has vertical space to spare, expand the route tiles so
    # that they fill the region below the title instead of floating towards the
    # top. This keeps the layout balanced on taller canvases (e.g. the earlier 128px panels).
    row_box_heights: List[int]
    if rows:
        available_height = HEIGHT - outer_margin - y
        base_total_height = row_count * row_height + (row_count - 1) * row_gap
        extra_space = max(0, available_height - base_total_height)

        if extra_space and row_count:
            even_extra, remainder = divmod(extra_space, row_count)
            row_box_heights = [row_height + even_extra] * row_count
            for index in range(remainder):
                row_box_heights[index] += 1
        else:
            row_box_heights = [row_height] * row_count
    else:
        row_box_heights = []

    for row, box_height in zip(rows, row_box_heights):
        sign_image = row["sign"]
        normalized = row["normalized"]
        color = row["color"]
        display_color = color if normalized.upper() != "N/A" else (230, 230, 230)

        row_top = y
        row_bottom = y + box_height

        draw.rounded_rectangle(
            (row_left, row_top, row_right, row_bottom),
            radius=10,
            fill=(28, 28, 28),
            outline=(80, 80, 80),
        )

        sign_x = row_left + row_padding_x
        sign_y = row_top + (box_height - sign_image.height) // 2
        img.paste(sign_image, (sign_x, sign_y), sign_image)

        time_width = row["time_width"]
        time_height = row["time_height"]
        time_x = row_right - row_padding_x - time_width
        time_y = row_top + (box_height - time_height) // 2
        draw.text((time_x, time_y), normalized, font=time_font, fill=display_color)

        y = row_bottom + row_gap

    if all_na:
        warning_y = min(canvas_height - warning_height - 4, y)
        draw.text(
            ((WIDTH - warning_width) // 2, warning_y),
            warning_text,
            font=FONT_TRAVEL_HEADER,
            fill=(200, 200, 200),
        )

    draw.text(
        ((WIDTH - title_width) // 2, 0),
        TRAVEL_TITLE,
        font=FONT_TITLE_SPORTS,
        fill=(255, 255, 255),
    )

    return img


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scroll_travel_display(display, full_img: Image.Image) -> None:
    if display is None:
        return

    if full_img.height <= HEIGHT:
        display.image(full_img)
        display.show()
        time.sleep(SCROLL_PAUSE_BOTTOM)
        return

    max_offset = full_img.height - HEIGHT
    frame = full_img.crop((0, 0, WIDTH, HEIGHT))
    display.image(frame)
    display.show()
    time.sleep(SCROLL_PAUSE_TOP)

    for offset in range(SCROLL_STEP, max_offset + 1, SCROLL_STEP):
        frame = full_img.crop((0, offset, WIDTH, offset + HEIGHT))
        display.image(frame)
        display.show()
        time.sleep(SCROLL_DELAY)

    time.sleep(SCROLL_PAUSE_BOTTOM)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────────

def is_travel_screen_active(now: Optional[dt.time] = None) -> bool:
    window = get_travel_active_window()
    if not window:
        return True

    start, end = window

    if start == end:
        return True

    if isinstance(now, dt.datetime):
        now = now.timetz() if now.tzinfo else now.time()
    elif now is None:
        now = dt.datetime.now(CENTRAL_TIME).time()

    if not isinstance(now, dt.time):
        logging.warning("Travel screen: could not interpret current time %r.", now)
        return True

    if start <= end:
        active = start <= now < end
    else:
        active = now >= start or now < end

    if not active:
        logging.debug("Travel screen skipped—outside active window.")

    return active


@log_call
def draw_travel_time_screen(
    display,
    transition: bool = False,
) -> Optional[Image.Image | ScreenImage]:
    if not is_travel_screen_active():
        return None

    times = get_travel_times()
    img = _compose_travel_image(times)

    displayed = display is not None

    if transition:
        _scroll_travel_display(display, img)
        return ScreenImage(img, displayed=displayed)

    _scroll_travel_display(display, img)
    return ScreenImage(img, displayed=displayed)


__all__ = [
    "draw_travel_time_screen",
    "get_travel_times",
    "is_travel_screen_active",
    "get_travel_active_window",
]
