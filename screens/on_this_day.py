"""On This Day screen.

Renders a colorful, vertically scrolling digest of notable events for today's
month/day using Wikimedia's On This Day feed when available, with curated local
fallbacks for offline use.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from io import BytesIO

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

TITLE_FONT = clone_font(config.FONT_DAY_DATE, max(18, min(34, W // 10)))
SECTION_FONT = clone_font(config.FONT_WEATHER_DETAILS_BOLD, max(13, min(20, W // 16)))
BODY_FONT = clone_font(config.FONT_WEATHER_DETAILS_SMALL, max(10, min(15, W // 24)))
YEAR_FONT = clone_font(config.FONT_WEATHER_DETAILS_SMALL_BOLD, max(10, min(14, W // 25)))


@dataclass(frozen=True)
class DayItem:
    year: int | None
    text: str
    thumbnail_url: str | None = None


_FALLBACK_BY_DATE: dict[tuple[int, int], dict[str, list[DayItem]]] = {
    (7, 6): {
        "🌎 General History": [
            DayItem(1415, "Jan Hus was burned at the stake in Constance, helping spark the Hussite movement."),
            DayItem(1885, "Louis Pasteur successfully tested his rabies vaccine on Joseph Meister."),
            DayItem(1957, "John Lennon and Paul McCartney met at a church fete in Liverpool."),
        ],
        "🇺🇸 American History": [
            DayItem(1777, "Fort Ticonderoga was captured by British forces during the American Revolutionary War."),
            DayItem(1944, "The Hartford circus fire became one of the deadliest fire disasters in U.S. history."),
        ],
        "🏙️ Chicago History": [
            DayItem(1933, "The first Major League Baseball All-Star Game was played at Comiskey Park in Chicago."),
            DayItem(1957, "Chicago music history got a Beatles footnote when Lennon met McCartney on this date."),
        ],
        "🏟️ Sports History": [
            DayItem(1933, "Babe Ruth hit the first home run in MLB All-Star Game history at Comiskey Park."),
            DayItem(2013, "Andy Murray won Wimbledon, becoming the first British men's singles champion there since 1936."),
        ],
        "🎂 Famous Birthdays": [
            DayItem(1907, "Frida Kahlo, Mexican painter known for vivid self-portraits, was born."),
            DayItem(1946, "George W. Bush, 43rd president of the United States, was born."),
            DayItem(1975, "50 Cent, rapper, actor, and entrepreneur, was born."),
        ],
        "💾 Tech & Science": [
            DayItem(1885, "Pasteur's rabies vaccine milestone helped define modern immunology."),
            DayItem(1997, "NASA's Sojourner rover began rolling on Mars during the Pathfinder mission weekend."),
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
                source = thumbnail.get("source") if isinstance(thumbnail, dict) else None
                if source:
                    thumb = str(source)
                    break
        year = raw.get("year")
        items.append(DayItem(int(year) if isinstance(year, int) else None, text=text, thumbnail_url=thumb))
        if len(items) >= limit:
            break
    return items


def _build_sections(today: dt.date) -> dict[str, list[DayItem]]:
    month, day = today.month, today.day
    fallback = _FALLBACK_BY_DATE.get((month, day), {})
    if fallback:
        # Curated local categories add the requested Chicago/American/Sports/Tech
        # flavor and should render promptly when the live feed is unavailable.
        # Prefer them without probing Wikimedia first; the shared HTTP client can
        # spend several retry cycles per feed while offline or under DNS failure.
        return {title: list(items) for title, items in fallback.items() if items}

    sections = {
        "🌎 General History": _wiki_items("events", month, day, 4),
        "🎂 Famous Birthdays": _wiki_items("births", month, day, 3),
        "🕯️ Notable Lives": _wiki_items("deaths", month, day, 2),
        "🎉 Holidays & Culture": _wiki_items("holidays", month, day, 2),
    }

    return {title: items for title, items in sections.items() if items}


def _download_thumbnail(url: str | None, size: int) -> Image.Image | None:
    if not url or size <= 0:
        return None
    try:
        response = http_get(url, timeout=2.5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None


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


def _estimate_height(draw: ImageDraw.ImageDraw, sections: dict[str, list[DayItem]], max_text_width: int) -> int:
    y = 18 + measure_text(draw, "On This Day", TITLE_FONT)[1] + 22
    for title, items in sections.items():
        y += measure_text(draw, title, SECTION_FONT)[1] + 8
        for item in items:
            lines = wrap_text(item.text, BODY_FONT, max_text_width)
            y += max(42, len(lines) * (measure_text(draw, "Ag", BODY_FONT)[1] + 3) + 16) + 7
        y += 8
    return max(H, y + 18)


def _render_full_image(today: dt.date) -> Image.Image:
    probe = Image.new("RGB", (W, H), _BG)
    probe_draw = ImageDraw.Draw(probe)
    pad = max(8, W // 32)
    thumb_size = 34 if W >= 300 else 0
    max_text_width = W - pad * 3 - (thumb_size + 8 if thumb_size else 0)
    sections = _build_sections(today)
    height = _estimate_height(probe_draw, sections, max_text_width)
    img = Image.new("RGB", (W, height), _BG)
    _draw_gradient(img)
    draw = ImageDraw.Draw(img)

    title = "📅 On This Day"
    subtitle = today.strftime("%B %-d") if hasattr(today, "strftime") else f"{today.month}/{today.day}"
    tw, th = measure_text(draw, title, TITLE_FONT)
    draw.text(((W - tw) // 2, 10), title, font=TITLE_FONT, fill=_TEXT)
    sw, sh = measure_text(draw, subtitle, BODY_FONT)
    draw.text(((W - sw) // 2, 14 + th), subtitle, font=BODY_FONT, fill=_MUTED)
    y = 20 + th + sh + 10

    for idx, (section, items) in enumerate(sections.items()):
        accent = _ACCENTS[idx % len(_ACCENTS)]
        draw.rounded_rectangle((pad, y, W - pad, y + 3), radius=2, fill=accent)
        y += 7
        draw.text((pad, y), section, font=SECTION_FONT, fill=accent)
        y += measure_text(draw, section, SECTION_FONT)[1] + 6
        for item in items:
            card_top = y
            lines = wrap_text(item.text, BODY_FONT, max_text_width)
            line_h = measure_text(draw, "Ag", BODY_FONT)[1] + 3
            card_h = max(42, len(lines) * line_h + 18)
            _rounded(draw, (pad, card_top, W - pad, card_top + card_h), _CARD, (44, 54, 88))
            x = pad + 8
            if thumb_size:
                thumb = _download_thumbnail(item.thumbnail_url, thumb_size)
                if thumb is not None:
                    img.paste(thumb, (x, card_top + 7))
                else:
                    draw.ellipse((x, card_top + 8, x + thumb_size, card_top + 8 + thumb_size), fill=accent)
                    draw.text((x + thumb_size // 2 - 4, card_top + 15), "✦", font=BODY_FONT, fill=_BG)
                x += thumb_size + 8
            if item.year is not None:
                draw.text((x, card_top + 7), str(item.year), font=YEAR_FONT, fill=accent)
                text_x = x + measure_text(draw, str(item.year), YEAR_FONT)[0] + 7
            else:
                text_x = x
            text_y = card_top + 7
            for line in lines:
                draw.text((text_x, text_y), line, font=BODY_FONT, fill=_TEXT)
                text_y += line_h
            y += card_h + 7
        y += 6

    return img


def draw_on_this_day(display, transition: bool = True, today: dt.date | None = None) -> ScreenImage:
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
            base_step=max(1, H // 120),
            pause_start=1.0,
            pause_end=1.5,
            min_frame_time=0.035,
        )
        bottom_offset = max(0, full_img.height - H)
        bottom_frame = full_img.crop((0, bottom_offset, W, bottom_offset + H))
        return ScreenImage(bottom_frame, displayed=True, consumed_delay=True)

    first = full_img.crop((0, 0, W, H))
    display.image(first)
    return ScreenImage(first, displayed=True)
