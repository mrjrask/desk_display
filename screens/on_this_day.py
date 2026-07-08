"""On This Day screen.

Renders a colorful, vertically scrolling digest of notable events for today's
month/day using Wikimedia's On This Day feed when available, with curated local
fallbacks for offline use.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from io import BytesIO
from typing import Callable
from urllib.parse import unquote

from PIL import Image, ImageDraw, ImageOps

import config
from services.http_client import http_get
from utils import (
    ScreenImage,
    clear_display,
    clone_font,
    measure_text,
    scroll_vertical_content,
    wrap_text,
)

W, H = config.WIDTH, config.HEIGHT
_BG = (8, 10, 24)
_CARD = (22, 27, 50)
_TEXT = (245, 247, 255)
_MUTED = (178, 190, 220)
_ACCENTS = [
    (255, 111, 145),
    (255, 198, 93),
    (93, 214, 255),
    (119, 255, 175),
    (196, 132, 255),
    (255, 149, 94),
]

# Edit these per-display defaults to tune the On This Day typography.
# Keys are display profile IDs returned by config.get_display_profile_id().
ON_THIS_DAY_FONT_SIZES_BY_PROFILE = {
    "hdmi_1080p": {"title": 84, "section": 48, "body": 36, "year": 32},
    "fallback_hd": {"title": 56, "section": 32, "body": 26, "year": 24},
    "fallback_default": {"title": 32, "section": 20, "body": 15, "year": 13},
    "display_hat_mini": {"title": 32, "section": 20, "body": 15, "year": 13},
    "adafruit_minipitft_114": {"title": 24, "section": 15, "body": 12, "year": 11},
    "hyperpixel4_square": {"title": 58, "section": 34, "body": 28, "year": 24},
    "hyperpixel4": {"title": 50, "section": 30, "body": 24, "year": 22},
}

_DEFAULT_FONT_SIZES = {"title": 32, "section": 20, "body": 15, "year": 13}


def _font_sizes_for_profile(profile_id: str | None = None) -> dict[str, int]:
    """Return editable On This Day font sizes for the active display profile."""

    profile_id = profile_id or config.get_display_profile_id()
    sizes = ON_THIS_DAY_FONT_SIZES_BY_PROFILE.get(profile_id, _DEFAULT_FONT_SIZES)
    return {**_DEFAULT_FONT_SIZES, **sizes}


_FONT_SIZES = _font_sizes_for_profile()
TITLE_FONT = clone_font(config.FONT_DAY_DATE, _FONT_SIZES["title"])
SECTION_FONT = clone_font(config.FONT_WEATHER_DETAILS_BOLD, _FONT_SIZES["section"])
BODY_FONT = clone_font(config.FONT_WEATHER_DETAILS_SMALL, _FONT_SIZES["body"])
YEAR_FONT = clone_font(config.FONT_WEATHER_DETAILS_SMALL_BOLD, _FONT_SIZES["year"])

_SCROLL_FRAME_SECONDS = 0.030
_SCROLL_START_PAUSE_SECONDS = 2.0
_SCROLL_END_PAUSE_SECONDS = 3.0

# Keep live Wikimedia content off the display hot path.  The Raspberry Pi Zero 2 W
# can spend seconds fetching multiple feeds and thumbnails; caching by date means
# the screen checks for new On This Day content only when the calendar day changes.
_SECTIONS_CACHE_LOCK = threading.Lock()
_SECTIONS_CACHE_DATE: dt.date | None = None
_SECTIONS_CACHE_VALUE: dict[str, list["DayItem"]] | None = None
_RENDER_CACHE_LOCK = threading.Lock()
_RENDER_CACHE_DATE: dt.date | None = None
_RENDER_CACHE_IMAGE: Image.Image | None = None
_THUMBNAIL_CACHE_LOCK = threading.Lock()
_THUMBNAIL_CACHE: dict[tuple[str, int], Image.Image | None] = {}

_HEBCAL_JEWISH_HOLIDAYS_ICS_URL = (
    "https://download.hebcal.com/ical/jewish-holidays-all-v2.ics"
)
_JEWISH_HOLIDAY_LIMIT = 3
_FEED_BUILD_TIMEOUT_SECONDS = max(
    0.5, float(os.environ.get("ON_THIS_DAY_FEED_BUILD_TIMEOUT_SECONDS", "3.5"))
)
_LIVE_THUMBNAILS_ENABLED = os.environ.get(
    "ON_THIS_DAY_LIVE_THUMBNAILS", "0"
).strip().lower() in {"1", "true", "yes", "on"}

_EMOJI_PREFIX_RE = re.compile(r"^[^\w#]+\s*")

_SECTION_LABELS = {
    "General History": "General History",
    "American History": "American History",
    "Chicago History": "Chicago History",
    "Sports History": "Sports History",
    "Famous Birthdays": "Famous Birthdays",
    "Tech & Science": "Tech & Science",
    "Notable Lives": "Notable Lives",
    "Holidays & Culture": "Holidays & Culture",
}


def _display_label(label: str) -> str:
    """Return text that is safe for non-emoji fonts on small displays."""

    cleaned = _EMOJI_PREFIX_RE.sub("", label).strip()
    return _SECTION_LABELS.get(cleaned, cleaned or label)


@dataclass(frozen=True)
class DayItem:
    year: int | None
    text: str
    thumbnail_url: str | None = None


_FALLBACK_BY_DATE: dict[tuple[int, int], dict[str, list[DayItem]]] = {
    (7, 6): {
        "🌎 General History": [
            DayItem(
                1415,
                "Jan Hus was burned at the stake in Constance, helping spark the Hussite movement.",
            ),
            DayItem(
                1885,
                "Louis Pasteur successfully tested his rabies vaccine on Joseph Meister.",
            ),
            DayItem(
                1957,
                "John Lennon and Paul McCartney met at a church fete in Liverpool.",
            ),
        ],
        "🇺🇸 American History": [
            DayItem(
                1777,
                "Fort Ticonderoga was captured by British forces during the American Revolutionary War.",
            ),
            DayItem(
                1944,
                "The Hartford circus fire became one of the deadliest fire disasters in U.S. history.",
            ),
        ],
        "🏙️ Chicago History": [
            DayItem(
                1933,
                "The first Major League Baseball All-Star Game was played at Comiskey Park in Chicago.",
            ),
            DayItem(
                1957,
                "Chicago music history got a Beatles footnote when Lennon met McCartney on this date.",
            ),
        ],
        "🏟️ Sports History": [
            DayItem(
                1933,
                "Babe Ruth hit the first home run in MLB All-Star Game history at Comiskey Park.",
            ),
            DayItem(
                2013,
                "Andy Murray won Wimbledon, becoming the first British men's singles champion there since 1936.",
            ),
        ],
        "🎂 Famous Birthdays": [
            DayItem(
                1907,
                "Frida Kahlo, Mexican painter known for vivid self-portraits, was born.",
            ),
            DayItem(
                1946, "George W. Bush, 43rd president of the United States, was born."
            ),
            DayItem(1975, "50 Cent, rapper, actor, and entrepreneur, was born."),
        ],
        "💾 Tech & Science": [
            DayItem(
                1885,
                "Pasteur's rabies vaccine milestone helped define modern immunology.",
            ),
            DayItem(
                1997,
                "NASA's Sojourner rover began rolling on Mars during the Pathfinder mission weekend.",
            ),
        ],
    }
}


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").split())


def _wiki_items(feed_type: str, month: int, day: int, limit: int = 3) -> list[DayItem]:
    url = f"https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday/{feed_type}/{month}/{day}"
    try:
        response = http_get(url, timeout=3.0)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logging.debug("on_this_day: Wikimedia feed failed for %s: %s", feed_type, exc)
        return []

    items: list[DayItem] = []
    for raw in payload.get(feed_type, []) if isinstance(payload, dict) else []:
        if not isinstance(raw, dict):
            continue
        text = _clean_text(raw.get("text", ""))
        if not text:
            continue
        thumb = None
        pages = raw.get("pages")
        if isinstance(pages, list):
            for page in pages:
                thumbnail = page.get("thumbnail") if isinstance(page, dict) else None
                source = (
                    thumbnail.get("source") if isinstance(thumbnail, dict) else None
                )
                if source:
                    thumb = str(source)
                    break
        year = raw.get("year")
        items.append(
            DayItem(
                int(year) if isinstance(year, int) else None,
                text=text,
                thumbnail_url=thumb,
            )
        )
        if len(items) >= limit:
            break
    return items


def _unfold_ics_lines(text: str) -> list[str]:
    """Return RFC 5545-unfolded calendar lines."""

    lines: list[str] = []
    for raw_line in (
        str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ):
        if raw_line.startswith((" ", "\t")) and lines:
            lines[-1] += raw_line[1:]
        else:
            lines.append(raw_line)
    return lines


def _ics_value(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    name, value = line.split(":", 1)
    return name.split(";", 1)[0].upper(), unquote(
        value.replace("\\,", ",").replace("\\;", ";").replace("\\n", " ").strip()
    )


def _parse_ics_date(value: str) -> dt.date | None:
    value = str(value or "").strip()
    if not value:
        return None
    date_part = value[:8]
    try:
        return dt.datetime.strptime(date_part, "%Y%m%d").date()
    except ValueError:
        return None


def _parse_jewish_holidays_ics(text: str, today: dt.date) -> list[DayItem]:
    """Extract Hebcal all-holidays events for today's Gregorian date."""

    items: list[DayItem] = []
    in_event = False
    event: dict[str, str] = {}
    for line in _unfold_ics_lines(text):
        if line == "BEGIN:VEVENT":
            in_event = True
            event = {}
            continue
        if line == "END:VEVENT":
            start = _parse_ics_date(event.get("DTSTART", ""))
            summary = _clean_text(event.get("SUMMARY", ""))
            if start and start == today and summary:
                items.append(DayItem(None, f"Jewish holiday: {summary}."))
            in_event = False
            event = {}
            continue
        if not in_event:
            continue
        parsed = _ics_value(line)
        if parsed is None:
            continue
        name, value = parsed
        if name in {"DTSTART", "SUMMARY"}:
            event[name] = value
    return items


def _jewish_holiday_items(
    today: dt.date, limit: int = _JEWISH_HOLIDAY_LIMIT
) -> list[DayItem]:
    try:
        response = http_get(_HEBCAL_JEWISH_HOLIDAYS_ICS_URL, timeout=3.0)
        response.raise_for_status()
        items = _parse_jewish_holidays_ics(response.text, today)
    except Exception as exc:
        logging.debug("on_this_day: Hebcal Jewish holiday feed failed: %s", exc)
        return []
    return items[:limit]


def _copy_sections(sections: dict[str, list[DayItem]]) -> dict[str, list[DayItem]]:
    return {title: list(items) for title, items in sections.items()}


def _build_sections_uncached(today: dt.date) -> dict[str, list[DayItem]]:
    month, day = today.month, today.day
    fallback = _FALLBACK_BY_DATE.get((month, day), {})
    if fallback:
        # Curated local categories add the requested Chicago/American/Sports/Tech
        # flavor and should render promptly when the live feed is unavailable.
        # Prefer them without probing Wikimedia first; the shared HTTP client can
        # spend several retry cycles per feed while offline or under DNS failure.
        return {title: list(items) for title, items in fallback.items() if items}

    # Keep On This Day off the rotation hot path.  Fetching four Wikimedia feeds
    # plus Hebcal serially can hold the previous screen for 10+ seconds on a Pi
    # Zero 2 W when DNS, Wi-Fi, or the upstream API is slow.  Start the feeds in
    # parallel, collect only the calls that finish within a small overall budget,
    # and let the screen render with whichever sections are ready.
    task_specs: dict[str, tuple[str, Callable[[], list[DayItem]]]] = {
        "🌎 General History": ("events", lambda: _wiki_items("events", month, day, 4)),
        "🎂 Famous Birthdays": ("births", lambda: _wiki_items("births", month, day, 3)),
        "🕯️ Notable Lives": ("deaths", lambda: _wiki_items("deaths", month, day, 2)),
        "jewish_holidays": ("jewish_holidays", lambda: _jewish_holiday_items(today)),
        "wiki_holidays": ("holidays", lambda: _wiki_items("holidays", month, day, 2)),
    }

    completed: dict[str, list[DayItem]] = {}
    executor = ThreadPoolExecutor(max_workers=len(task_specs))
    future_to_title = {
        executor.submit(fetcher): title
        for title, (_feed_name, fetcher) in task_specs.items()
    }
    done, pending = wait(
        future_to_title,
        timeout=_FEED_BUILD_TIMEOUT_SECONDS,
    )
    for future in done:
        title = future_to_title[future]
        try:
            completed[title] = future.result() or []
        except Exception as exc:
            feed_name = task_specs[title][0]
            logging.debug("on_this_day: feed failed for %s: %s", feed_name, exc)
    if pending:
        logging.warning(
            "on_this_day: %d feed(s) exceeded %.1fs budget; rendering partial content.",
            len(pending),
            _FEED_BUILD_TIMEOUT_SECONDS,
        )
        for future in pending:
            future.cancel()
    executor.shutdown(wait=False, cancel_futures=True)

    holiday_items = (completed.get("jewish_holidays") or []) + (
        completed.get("wiki_holidays") or []
    )
    sections = {
        "🌎 General History": completed.get("🌎 General History") or [],
        "🎂 Famous Birthdays": completed.get("🎂 Famous Birthdays") or [],
        "🕯️ Notable Lives": completed.get("🕯️ Notable Lives") or [],
        "🎉 Holidays & Culture": holiday_items,
    }

    return {title: items for title, items in sections.items() if items}


def _build_sections(today: dt.date) -> dict[str, list[DayItem]]:
    global _SECTIONS_CACHE_DATE, _SECTIONS_CACHE_VALUE

    with _SECTIONS_CACHE_LOCK:
        if _SECTIONS_CACHE_DATE == today and _SECTIONS_CACHE_VALUE is not None:
            return _copy_sections(_SECTIONS_CACHE_VALUE)

    sections = _build_sections_uncached(today)

    with _SECTIONS_CACHE_LOCK:
        _SECTIONS_CACHE_DATE = today
        _SECTIONS_CACHE_VALUE = _copy_sections(sections)
        return _copy_sections(_SECTIONS_CACHE_VALUE)


def _download_thumbnail(url: str | None, size: int) -> Image.Image | None:
    if not url or size <= 0:
        return None

    cache_key = (url, size)
    with _THUMBNAIL_CACHE_LOCK:
        if cache_key in _THUMBNAIL_CACHE:
            cached = _THUMBNAIL_CACHE[cache_key]
            return cached.copy() if cached is not None else None

    # Thumbnail downloads are nice-to-have, but they happen during image layout.
    # Default them off so a slow image host cannot freeze the previous screen;
    # operators can opt in once their network/display is known to be fast enough.
    if not _LIVE_THUMBNAILS_ENABLED:
        return None

    try:
        response = http_get(url, timeout=2.5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)
    except Exception:
        img = None

    with _THUMBNAIL_CACHE_LOCK:
        _THUMBNAIL_CACHE[cache_key] = img.copy() if img is not None else None

    return img.copy() if img is not None else None


def _draw_gradient(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img)
    for y in range(img.height):
        t = y / max(1, img.height - 1)
        r = int(_BG[0] + 18 * t)
        g = int(_BG[1] + 10 * t)
        b = int(_BG[2] + 42 * t)
        draw.line([(0, y), (img.width, y)], fill=(r, g, b))


def _rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
) -> None:
    radius = max(6, min(W, H) // 28)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)


def _item_text_layout(
    draw: ImageDraw.ImageDraw,
    item: DayItem,
    card_left: int,
    card_right: int,
    thumb_size: int,
) -> tuple[int, int]:
    """Return the rendered x-coordinate and wrap width for an item body."""

    text_x = card_left + 8
    if thumb_size:
        text_x += thumb_size + 8
    if item.year is not None:
        text_x += measure_text(draw, str(item.year), YEAR_FONT)[0] + 7
    return text_x, max(20, card_right - 8 - text_x)


def _estimate_height(
    draw: ImageDraw.ImageDraw,
    sections: dict[str, list[DayItem]],
    pad: int,
    thumb_size: int,
    subtitle: str,
) -> int:
    title = "On This Day"
    title_h = measure_text(draw, title, TITLE_FONT)[1]
    subtitle_h = measure_text(draw, subtitle, BODY_FONT)[1]
    y = 20 + title_h + subtitle_h + 10
    card_left = pad
    card_right = W - pad
    line_h = measure_text(draw, "Ag", BODY_FONT)[1] + 3

    for title, items in sections.items():
        y += 7
        label = _display_label(title)
        y += measure_text(draw, label, SECTION_FONT)[1] + 6
        for item in items:
            _, text_width = _item_text_layout(
                draw, item, card_left, card_right, thumb_size
            )
            lines = wrap_text(item.text, BODY_FONT, text_width)
            card_h = max(42, len(lines) * line_h + 18)
            y += card_h + 7
        y += 6
    return max(H, y + 18)


def _render_full_image_uncached(today: dt.date) -> Image.Image:
    probe = Image.new("RGB", (W, H), _BG)
    probe_draw = ImageDraw.Draw(probe)
    pad = max(8, W // 32)
    thumb_size = 34 if W >= 300 else 0
    sections = _build_sections(today)
    title = "On This Day"
    subtitle = (
        today.strftime("%B %-d")
        if hasattr(today, "strftime")
        else f"{today.month}/{today.day}"
    )
    height = _estimate_height(probe_draw, sections, pad, thumb_size, subtitle)
    img = Image.new("RGB", (W, height), _BG)
    _draw_gradient(img)
    draw = ImageDraw.Draw(img)

    tw, th = measure_text(draw, title, TITLE_FONT)
    draw.text(((W - tw) // 2, 10), title, font=TITLE_FONT, fill=_TEXT)
    sw, sh = measure_text(draw, subtitle, BODY_FONT)
    draw.text(((W - sw) // 2, 14 + th), subtitle, font=BODY_FONT, fill=_MUTED)
    y = 20 + th + sh + 10

    for idx, (section, items) in enumerate(sections.items()):
        accent = _ACCENTS[idx % len(_ACCENTS)]
        draw.rounded_rectangle((pad, y, W - pad, y + 3), radius=2, fill=accent)
        y += 7
        section_label = _display_label(section)
        dot_r = max(3, min(6, W // 64))
        dot_y = y + max(
            0, (measure_text(draw, section_label, SECTION_FONT)[1] - dot_r * 2) // 2
        )
        draw.ellipse((pad, dot_y, pad + dot_r * 2, dot_y + dot_r * 2), fill=accent)
        draw.text(
            (pad + dot_r * 2 + 6, y), section_label, font=SECTION_FONT, fill=accent
        )
        y += measure_text(draw, section_label, SECTION_FONT)[1] + 6
        for item in items:
            card_top = y
            text_x, text_width = _item_text_layout(draw, item, pad, W - pad, thumb_size)
            lines = wrap_text(item.text, BODY_FONT, text_width)
            line_h = measure_text(draw, "Ag", BODY_FONT)[1] + 3
            card_h = max(42, len(lines) * line_h + 18)
            _rounded(
                draw, (pad, card_top, W - pad, card_top + card_h), _CARD, (44, 54, 88)
            )
            x = pad + 8
            if thumb_size:
                thumb = _download_thumbnail(item.thumbnail_url, thumb_size)
                if thumb is not None:
                    img.paste(thumb, (x, card_top + 7))
                else:
                    draw.ellipse(
                        (x, card_top + 8, x + thumb_size, card_top + 8 + thumb_size),
                        fill=accent,
                    )
                    inner_pad = max(8, thumb_size // 3)
                    draw.ellipse(
                        (
                            x + inner_pad,
                            card_top + 8 + inner_pad,
                            x + thumb_size - inner_pad,
                            card_top + 8 + thumb_size - inner_pad,
                        ),
                        fill=_BG,
                    )
                x += thumb_size + 8
            if item.year is not None:
                draw.text(
                    (x, card_top + 7), str(item.year), font=YEAR_FONT, fill=accent
                )
            text_y = card_top + 7
            for line in lines:
                draw.text((text_x, text_y), line, font=BODY_FONT, fill=_TEXT)
                text_y += line_h
            y += card_h + 7
        y += 6

    return img


def _render_full_image(today: dt.date) -> Image.Image:
    global _RENDER_CACHE_DATE, _RENDER_CACHE_IMAGE

    with _RENDER_CACHE_LOCK:
        if _RENDER_CACHE_DATE == today and _RENDER_CACHE_IMAGE is not None:
            return _RENDER_CACHE_IMAGE.copy()

    img = _render_full_image_uncached(today)

    with _RENDER_CACHE_LOCK:
        _RENDER_CACHE_DATE = today
        _RENDER_CACHE_IMAGE = img.copy()
        return _RENDER_CACHE_IMAGE.copy()


def _clear_caches_for_tests() -> None:
    global _SECTIONS_CACHE_DATE, _SECTIONS_CACHE_VALUE, _RENDER_CACHE_DATE, _RENDER_CACHE_IMAGE

    with _SECTIONS_CACHE_LOCK:
        _SECTIONS_CACHE_DATE = None
        _SECTIONS_CACHE_VALUE = None
    with _RENDER_CACHE_LOCK:
        _RENDER_CACHE_DATE = None
        _RENDER_CACHE_IMAGE = None
    with _THUMBNAIL_CACHE_LOCK:
        _THUMBNAIL_CACHE.clear()


def draw_on_this_day(
    display, transition: bool = True, today: dt.date | None = None
) -> ScreenImage:
    today = today or dt.datetime.now(config.CENTRAL_TIME).date()
    clear_display(display)
    full_img = _render_full_image(today)

    def _show(offset: int) -> None:
        display.image(full_img.crop((0, offset, W, offset + H)))

    if transition and full_img.height > H:
        scroll_vertical_content(
            display=display,
            content_height=full_img.height,
            viewport_width=W,
            viewport_height=H,
            render_at_offset=_show,
            base_step=1,
            pause_start=_SCROLL_START_PAUSE_SECONDS,
            pause_end=_SCROLL_END_PAUSE_SECONDS,
            page_jump_mode=False,
            min_frame_time=_SCROLL_FRAME_SECONDS,
        )
        bottom_offset = max(0, full_img.height - H)
        bottom_frame = full_img.crop((0, bottom_offset, W, bottom_offset + H))
        return ScreenImage(
            bottom_frame,
            displayed=True,
            consumed_delay=True,
            screenshot_image=full_img,
        )

    first = full_img.crop((0, 0, W, H))
    display.image(first)
    return ScreenImage(first, displayed=True, screenshot_image=full_img)
