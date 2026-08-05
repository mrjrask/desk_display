"""News headlines screen: one horizontally scrolling ticker row per topic.

Each configured topic (see news_feeds.json) renders as its own colored,
labeled ticker lane so several categories are visible on screen at once,
each scrolling at a slightly different speed. Headline thumbnails are shown
inline when a feed provides one. On a touch-capable display, tapping a
headline opens a full-screen "Reader"-style overlay with the article text.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from PIL import Image, ImageDraw, ImageOps

import config
from config import HEIGHT, NEWS_HEADLINES_DISPLAY_SECONDS, WIDTH
from services.http_client import http_get
from services.news_feeds import (
    ArticleContent,
    NewsHeadline,
    NewsTopic,
    _strip_html,
    fetch_all_headlines,
    fetch_article_text,
    load_news_feed_config,
)
from services.stock_quotes import StockQuote, default_symbol_order, fetch_stock_quotes
from utils import (
    ScreenImage,
    _pygame_module_for_display,
    _scroll_tap_candidates,
    clear_display,
    clone_font,
    log_call,
    measure_text,
    wrap_text,
)

# Edit these per-display defaults to tune ticker typography. Keys are display
# profile IDs returned by config.get_display_profile_id().
NEWS_TICKER_FONT_SIZES_BY_PROFILE = {
    "hdmi_1080p": {"label": 30, "headline": 34},
    "fallback_hd": {"label": 20, "headline": 22},
    "fallback_default": {"label": 12, "headline": 13},
    "display_hat_mini": {"label": 12, "headline": 13},
    "adafruit_minipitft_114": {"label": 10, "headline": 11},
    "hyperpixel4_square": {"label": 22, "headline": 24},
    "hyperpixel4": {"label": 19, "headline": 21},
}
_DEFAULT_TICKER_FONT_SIZES = {"label": 12, "headline": 13}


def _font_sizes_for_profile(profile_id: str | None = None) -> dict[str, int]:
    profile_id = profile_id or config.get_display_profile_id()
    sizes = NEWS_TICKER_FONT_SIZES_BY_PROFILE.get(profile_id, _DEFAULT_TICKER_FONT_SIZES)
    return {**_DEFAULT_TICKER_FONT_SIZES, **sizes}


_FONT_SIZES = _font_sizes_for_profile()
LABEL_FONT = clone_font(config.FONT_WEATHER_DETAILS_SMALL_BOLD, _FONT_SIZES["label"])
HEADLINE_FONT = clone_font(config.FONT_WEATHER_DETAILS_SMALL, _FONT_SIZES["headline"])

# Distinct background/label/text theme per topic so tickers read at a glance.
# Topic ids not listed here (e.g. a custom feed added by an operator) get
# _FALLBACK_THEME. Edit freely to restyle a lane.
_ROW_THEMES: dict[str, dict[str, tuple[int, int, int]]] = {
    "local": {"bg": (14, 56, 39), "label_bg": (8, 110, 66), "text": (232, 255, 240)},
    "national": {"bg": (19, 28, 74), "label_bg": (36, 58, 168), "text": (232, 238, 255)},
    "world": {"bg": (61, 18, 66), "label_bg": (140, 40, 150), "text": (250, 235, 255)},
    "technology": {"bg": (10, 50, 68), "label_bg": (8, 120, 150), "text": (225, 250, 255)},
    "sports": {"bg": (74, 30, 12), "label_bg": (178, 68, 14), "text": (255, 240, 225)},
    "business": {"bg": (64, 56, 8), "label_bg": (160, 132, 8), "text": (255, 250, 225)},
}
_FALLBACK_THEME = {"bg": (30, 30, 38), "label_bg": (62, 62, 78), "text": (232, 232, 238)}

# Synthetic topic + theme for the stock ticker row appended at the bottom of
# the screen (see _build_stock_row). Not a real entry in news_feeds.json.
_STOCK_TOPIC = NewsTopic(id="markets", label="Markets", name="markets", url="")
_STOCK_THEME = {"bg": (8, 8, 14), "label_bg": (28, 28, 40), "text": (235, 235, 245)}
_STOCK_UP_COLOR = (60, 220, 100)
_STOCK_DOWN_COLOR = (235, 70, 70)
_STOCK_FLAT_COLOR = (235, 235, 245)

_ENTRY_SEPARATOR = "     •     "
_MIN_ROW_HEIGHT = 20
_FRAME_INTERVAL_SECONDS = 0.045
_OVERLAY_MAX_SECONDS = 45.0
_OVERLAY_FRAME_INTERVAL_SECONDS = 0.03
_OVERLAY_SCROLL_STEP = 1


def _theme_for_topic(topic_id: str) -> dict[str, tuple[int, int, int]]:
    return _ROW_THEMES.get(topic_id, _FALLBACK_THEME)


def _speed_multiplier(topic_id: str) -> float:
    """Deterministic per-topic speed variety so lanes visibly desync."""

    digest = sum(ord(ch) for ch in topic_id) if topic_id else 0
    return 0.82 + (digest % 55) / 100.0


@dataclass
class _TickerEntry:
    headline: Optional[NewsHeadline]
    text: str
    width: int
    thumb: Optional[Image.Image]
    thumb_size: int
    text_color: Optional[tuple[int, int, int]] = None


@dataclass
class _TickerRow:
    topic: NewsTopic
    theme: dict[str, tuple[int, int, int]]
    entries: list[_TickerEntry]
    speed: float
    offset: float = 0.0


# ─── Thumbnail/hero image fetching (opt out via NEWS_HEADLINES_SHOW_IMAGES) ────

_THUMB_CACHE_LOCK = threading.Lock()
_THUMB_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}
_HERO_CACHE_LOCK = threading.Lock()
_HERO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}


def _download_thumbnail(url: Optional[str], size: int) -> Optional[Image.Image]:
    """Return a *size*x*size* cropped thumbnail for inline ticker display."""

    if not url or size <= 0 or not config.NEWS_HEADLINES_SHOW_IMAGES:
        return None
    key = (url, size)
    with _THUMB_CACHE_LOCK:
        if key in _THUMB_CACHE:
            cached = _THUMB_CACHE[key]
            return cached.copy() if cached is not None else None
    try:
        response = http_get(url, timeout=3.0)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)
    except Exception as exc:
        logging.debug("news_headlines: thumbnail download failed for %s: %s", url, exc)
        img = None
    with _THUMB_CACHE_LOCK:
        _THUMB_CACHE[key] = img.copy() if img is not None else None
    return img.copy() if img is not None else None


def _download_hero_image(url: Optional[str], max_width: int) -> Optional[Image.Image]:
    """Return an aspect-preserving hero image for the reader overlay."""

    if not url or max_width <= 0 or not config.NEWS_HEADLINES_SHOW_IMAGES:
        return None
    key = (url, max_width)
    with _HERO_CACHE_LOCK:
        if key in _HERO_CACHE:
            cached = _HERO_CACHE[key]
            return cached.copy() if cached is not None else None
    try:
        response = http_get(url, timeout=4.0)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        if img.width > max_width:
            ratio = max_width / float(img.width)
            img = img.resize((max_width, max(1, int(img.height * ratio))), Image.Resampling.LANCZOS)
        if img.height > HEIGHT:
            img = img.crop((0, 0, img.width, HEIGHT))
    except Exception as exc:
        logging.debug("news_headlines: hero image download failed for %s: %s", url, exc)
        img = None
    with _HERO_CACHE_LOCK:
        _HERO_CACHE[key] = img.copy() if img is not None else None
    return img.copy() if img is not None else None


# ─── Row construction ───────────────────────────────────────────────────────


def _compute_row_layout(num_rows: int) -> tuple[int, list[int]]:
    if num_rows <= 0:
        return 0, []
    row_height = max(_MIN_ROW_HEIGHT, HEIGHT // num_rows)
    visible_rows = min(num_rows, max(1, HEIGHT // row_height))
    return row_height, [index * row_height for index in range(visible_rows)]


def _build_rows(
    topics: list[NewsTopic],
    headlines_by_topic: dict[str, list[NewsHeadline]],
    row_height: int,
) -> list[_TickerRow]:
    thumb_size = max(0, row_height - 10) if config.NEWS_HEADLINES_SHOW_IMAGES else 0
    probe_draw = ImageDraw.Draw(Image.new("RGB", (4, 4)))

    rows: list[_TickerRow] = []
    for topic in topics:
        items = headlines_by_topic.get(topic.id) or []
        if not items:
            continue
        entries: list[_TickerEntry] = []
        for headline in items:
            thumb = _download_thumbnail(headline.image_url, thumb_size) if thumb_size else None
            text = headline.title + _ENTRY_SEPARATOR
            text_width = measure_text(probe_draw, text, HEADLINE_FONT)[0]
            width = text_width + (thumb.size[0] + 8 if thumb is not None else 0)
            entries.append(
                _TickerEntry(
                    headline=headline,
                    text=text,
                    width=max(1, width),
                    thumb=thumb,
                    thumb_size=thumb_size,
                )
            )
        if entries:
            rows.append(
                _TickerRow(
                    topic=topic,
                    theme=_theme_for_topic(topic.id),
                    entries=entries,
                    speed=config.NEWS_TICKER_BASE_SPEED * _speed_multiplier(topic.id),
                )
            )
    return rows


def _format_stock_entry_text(quote: StockQuote) -> tuple[str, tuple[int, int, int]]:
    price_str = f"{quote.price:,.2f}"
    if quote.change is not None and quote.change_pct is not None:
        if quote.change > 0:
            arrow, color = "▲", _STOCK_UP_COLOR
        elif quote.change < 0:
            arrow, color = "▼", _STOCK_DOWN_COLOR
        else:
            arrow, color = "◆", _STOCK_FLAT_COLOR
        change_str = f"{arrow} {quote.change:+.2f} ({quote.change_pct:+.2f}%)"
    else:
        color = _STOCK_FLAT_COLOR
        change_str = "N/A"
    return f"{quote.label} {price_str}  {change_str}" + _ENTRY_SEPARATOR, color


def _build_stock_row(quotes: list[StockQuote]) -> Optional[_TickerRow]:
    """Build the bottom ticker row from fetched quotes; None if none priced."""

    probe_draw = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    entries: list[_TickerEntry] = []
    for quote in quotes:
        if quote.price is None:
            continue
        text, color = _format_stock_entry_text(quote)
        width = measure_text(probe_draw, text, HEADLINE_FONT)[0]
        entries.append(
            _TickerEntry(
                headline=None,
                text=text,
                width=max(1, width),
                thumb=None,
                thumb_size=0,
                text_color=color,
            )
        )
    if not entries:
        return None
    return _TickerRow(
        topic=_STOCK_TOPIC,
        theme=_STOCK_THEME,
        entries=entries,
        speed=config.NEWS_TICKER_BASE_SPEED * _speed_multiplier(_STOCK_TOPIC.id),
    )


# ─── Pure marquee layout math (no PIL/pygame; easy to unit test) ──────────────


def layout_row_entries(
    entry_widths: list[float], offset: float, lane_width: float
) -> list[tuple[float, float, int]]:
    """Return (x0, x1, entry_index) tuples visible within [0, lane_width).

    Entries repeat/loop indefinitely as *offset* grows. This is the core
    marquee math shared by rendering and touch hit-testing.
    """

    entry_count = len(entry_widths)
    if entry_count == 0 or lane_width <= 0:
        return []
    total_width = sum(entry_widths)
    if total_width <= 0:
        return []

    offset = offset % total_width
    cursor = 0.0
    start_index = 0
    local_offset = offset
    for index, width in enumerate(entry_widths):
        if cursor + width > offset:
            start_index = index
            local_offset = offset - cursor
            break
        cursor += width
    else:
        start_index, local_offset = 0, 0.0

    visible: list[tuple[float, float, int]] = []
    x = -local_offset
    index = start_index
    safety = 0
    max_iterations = entry_count * 3 + 6
    while x < lane_width and safety < max_iterations:
        width = entry_widths[index]
        x0, x1 = x, x + width
        if x1 > 0:
            visible.append((x0, x1, index))
        x += width
        index = (index + 1) % entry_count
        safety += 1
    return visible


# ─── Frame rendering ────────────────────────────────────────────────────────


def _render_frame(
    rows: list[_TickerRow], row_height: int, row_tops: list[int]
) -> tuple[Image.Image, list[tuple[int, int, int, int, NewsHeadline]]]:
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    probe_draw = ImageDraw.Draw(img)
    hit_rects: list[tuple[int, int, int, int, NewsHeadline]] = []
    label_pad_x = 10

    for row, top in zip(rows, row_tops, strict=True):
        theme = row.theme
        label_text = row.topic.label.upper()
        label_w, label_h = measure_text(probe_draw, label_text, LABEL_FONT)
        label_box_w = max(1, min(WIDTH - 20, label_w + label_pad_x * 2))
        img.paste(Image.new("RGB", (label_box_w, row_height), theme["label_bg"]), (0, top))
        label_draw = ImageDraw.Draw(img)
        label_draw.text(
            (label_pad_x, top + (row_height - label_h) // 2),
            label_text,
            font=LABEL_FONT,
            fill=theme["text"],
        )

        lane_x0 = label_box_w + 4
        lane_width = max(1, WIDTH - lane_x0)
        lane_img = Image.new("RGB", (lane_width, row_height), theme["bg"])
        lane_draw = ImageDraw.Draw(lane_img)

        widths = [entry.width for entry in row.entries]
        for x0, x1, index in layout_row_entries(widths, row.offset, lane_width):
            entry = row.entries[index]
            content_x = x0
            if entry.thumb is not None:
                thumb_y = (row_height - entry.thumb_size) // 2
                lane_img.paste(entry.thumb, (int(content_x), int(thumb_y)))
                content_x += entry.thumb_size + 8
            text_h = measure_text(lane_draw, entry.text, HEADLINE_FONT)[1]
            text_y = (row_height - text_h) // 2
            fill = entry.text_color or theme["text"]
            lane_draw.text((content_x, text_y), entry.text, font=HEADLINE_FONT, fill=fill)

            if entry.headline is not None:
                hit_x0 = lane_x0 + max(0, int(x0))
                hit_x1 = lane_x0 + min(lane_width, int(x1))
                if hit_x1 > hit_x0:
                    hit_rects.append((hit_x0, top, hit_x1, top + row_height, entry.headline))

        img.paste(lane_img, (lane_x0, top))

    return img, hit_rects


def _render_empty_state(display) -> ScreenImage:
    img = Image.new("RGB", (WIDTH, HEIGHT), (18, 18, 24))
    draw = ImageDraw.Draw(img)
    font = clone_font(config.FONT_WEATHER_DETAILS_BOLD, _FONT_SIZES["headline"] + 2)
    text = "Headlines unavailable"
    tw, th = measure_text(draw, text, font)
    draw.text(((WIDTH - tw) // 2, (HEIGHT - th) // 2), text, font=font, fill=(220, 220, 230))
    clear_display(display)
    display.image(img)
    return ScreenImage(img, displayed=True)


# ─── Touch handling (headline tap -> Reader-style overlay) ─────────────────


def _is_touch_capable(pygame_module) -> bool:
    if pygame_module is None:
        return False
    has_finger = getattr(pygame_module, "FINGERDOWN", None)
    has_mouse = getattr(pygame_module, "MOUSEBUTTONDOWN", None)
    return bool(has_finger or has_mouse)


def _map_touch_to_render_coords(
    x_pos: float,
    y_pos: float,
    *,
    display_width: float,
    display_height: float,
    rotation_degrees: int,
) -> tuple[float, float]:
    """Map raw touch/mouse pixel coordinates into unrotated render space.

    Mirrors main._touch_to_render_coords so ticker hit-testing agrees with
    the rest of the app's touch handling on rotated displays.
    """

    if display_width <= 0 or display_height <= 0:
        return x_pos, y_pos

    x_norm = max(0.0, min(1.0, x_pos / display_width))
    y_norm = max(0.0, min(1.0, y_pos / display_height))
    rotation = rotation_degrees % 360

    if rotation == 90:
        mapped_x_norm, mapped_y_norm = 1.0 - y_norm, x_norm
    elif rotation == 180:
        mapped_x_norm, mapped_y_norm = 1.0 - x_norm, 1.0 - y_norm
    elif rotation == 270:
        mapped_x_norm, mapped_y_norm = y_norm, 1.0 - x_norm
    else:
        mapped_x_norm, mapped_y_norm = x_norm, y_norm

    return mapped_x_norm * display_width, mapped_y_norm * display_height


def _tap_pixel_position(
    event, display_width: float, display_height: float
) -> Optional[tuple[float, float]]:
    x_norm = getattr(event, "x", None)
    y_norm = getattr(event, "y", None)
    pos = getattr(event, "pos", None)
    if isinstance(pos, tuple) and len(pos) >= 2:
        return (
            max(0.0, min(display_width, float(pos[0]))),
            max(0.0, min(display_height, float(pos[1]))),
        )
    if x_norm is not None and y_norm is not None:
        return (
            max(0.0, min(display_width, float(x_norm) * display_width)),
            max(0.0, min(display_height, float(y_norm) * display_height)),
        )
    return None


def _poll_taps(display, pygame_module) -> list[tuple[float, float]]:
    """Drain touch/mouse events for this frame, returning resolved taps.

    A "tap" is a press+release without a drag in between (see
    utils._scroll_tap_candidates), consistent with how drag-scroll gestures
    elsewhere in the app are distinguished from taps.
    """

    if pygame_module is None:
        return []

    event_type_names = (
        "FINGERDOWN", "FINGERMOTION", "FINGERUP",
        "MOUSEBUTTONDOWN", "MOUSEMOTION", "MOUSEBUTTONUP",
    )
    event_types = [
        getattr(pygame_module, name, None)
        for name in event_type_names
        if getattr(pygame_module, name, None) is not None
    ]
    if not event_types:
        return []

    try:
        raw_events = pygame_module.event.get(event_types)
    except Exception:
        return []
    if not raw_events:
        return []

    tap_events = _scroll_tap_candidates(raw_events, pygame_module)
    if not tap_events:
        return []

    display_width = float(getattr(display, "width", WIDTH) or WIDTH)
    display_height = float(getattr(display, "height", HEIGHT) or HEIGHT)
    rotation = int(getattr(display, "rotation", 0) or 0)

    taps: list[tuple[float, float]] = []
    for event in tap_events:
        pixel_pos = _tap_pixel_position(event, display_width, display_height)
        if pixel_pos is None:
            continue
        taps.append(
            _map_touch_to_render_coords(
                pixel_pos[0],
                pixel_pos[1],
                display_width=display_width,
                display_height=display_height,
                rotation_degrees=rotation,
            )
        )
    return taps


def _hit_test(
    hit_rects: list[tuple[int, int, int, int, NewsHeadline]], x: float, y: float
) -> Optional[NewsHeadline]:
    for x0, y0, x1, y1, headline in reversed(hit_rects):
        if x0 <= x <= x1 and y0 <= y <= y1:
            return headline
    return None


# ─── Reader overlay ─────────────────────────────────────────────────────────


def _build_overlay_image(headline: NewsHeadline, article: Optional[ArticleContent]) -> Image.Image:
    margin = max(10, WIDTH // 24)
    content_width = max(20, WIDTH - margin * 2)
    title_font = clone_font(config.FONT_WEATHER_DETAILS_BOLD, _FONT_SIZES["headline"] + 6)
    body_font = clone_font(config.FONT_WEATHER_DETAILS_SMALL, max(12, _FONT_SIZES["headline"]))
    meta_font = clone_font(config.FONT_WEATHER_DETAILS_TINY, max(10, _FONT_SIZES["label"]))

    title_text = (article.title if article and article.title else headline.title) or headline.title
    paragraphs = list(article.paragraphs) if article and article.paragraphs else []
    if not paragraphs:
        fallback_text = headline.summary
        if not fallback_text and headline.content_html:
            fallback_text = _strip_html(headline.content_html)
        paragraphs = [fallback_text] if fallback_text else [
            "Full article text is unavailable right now. Tap anywhere to close."
        ]

    hero_url = (article.image_url if article and article.image_url else headline.image_url)
    hero_img = _download_hero_image(hero_url, content_width)

    probe_draw = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    title_lines = wrap_text(title_text, title_font, content_width)
    line_h_title = measure_text(probe_draw, "Ag", title_font)[1] + 4
    line_h_body = measure_text(probe_draw, "Ag", body_font)[1] + 6
    meta_h = measure_text(probe_draw, "Ag", meta_font)[1]

    body_lines: list[str] = []
    for paragraph in paragraphs:
        body_lines.extend(wrap_text(paragraph, body_font, content_width))
        body_lines.append("")

    total_height = margin + meta_h + 8
    total_height += len(title_lines) * line_h_title + 10
    if hero_img is not None:
        total_height += hero_img.height + 12
    total_height += len(body_lines) * line_h_body + margin * 2

    img = Image.new("RGB", (WIDTH, max(HEIGHT, total_height)), (10, 12, 20))
    draw = ImageDraw.Draw(img)
    y = margin

    topic_label = (headline.topic_id or "").upper() or "ARTICLE"
    draw.text((margin, y), topic_label, font=meta_font, fill=(150, 200, 255))
    y += meta_h + 8

    for line in title_lines:
        draw.text((margin, y), line, font=title_font, fill=(255, 255, 255))
        y += line_h_title
    y += 6

    if hero_img is not None:
        img.paste(hero_img, (margin, y))
        y += hero_img.height + 12

    for line in body_lines:
        if line:
            draw.text((margin, y), line, font=body_font, fill=(222, 226, 234))
        y += line_h_body

    return img


def _show_reader_overlay(display, headline: NewsHeadline, pygame_module) -> None:
    """Block until the reader overlay is dismissed (tap, skip, or timeout)."""

    loading_font = clone_font(config.FONT_WEATHER_DETAILS_BOLD, _FONT_SIZES["headline"] + 4)
    loading_img = Image.new("RGB", (WIDTH, HEIGHT), (10, 12, 20))
    loading_draw = ImageDraw.Draw(loading_img)
    loading_text = "Loading article..."
    tw, th = measure_text(loading_draw, loading_text, loading_font)
    loading_pos = ((WIDTH - tw) // 2, (HEIGHT - th) // 2)
    loading_draw.text(loading_pos, loading_text, font=loading_font, fill=(230, 230, 240))
    display.image(loading_img)
    if hasattr(display, "show"):
        with contextlib.suppress(Exception):
            display.show()

    article = fetch_article_text(headline.link, timeout=config.NEWS_ARTICLE_FETCH_TIMEOUT_SECONDS)
    content_img = _build_overlay_image(headline, article)

    skip_requested = getattr(display, "skip_requested", None)
    has_wait_for_skip = callable(getattr(display, "wait_for_skip", None))

    def _should_stop() -> bool:
        return bool(callable(skip_requested) and skip_requested())

    max_offset = max(0, content_img.height - HEIGHT)
    offset = 0.0
    display.image(content_img.crop((0, 0, WIDTH, min(HEIGHT, content_img.height))))

    end_time = time.monotonic() + _OVERLAY_MAX_SECONDS
    while time.monotonic() < end_time and not _should_stop():
        frame_start = time.monotonic()

        if pygame_module is not None and _poll_taps(display, pygame_module):
            return

        if max_offset > 0:
            offset = min(max_offset, offset + _OVERLAY_SCROLL_STEP)
            display.image(content_img.crop((0, int(offset), WIDTH, int(offset) + HEIGHT)))

        elapsed = time.monotonic() - frame_start
        sleep_for = max(0.0, _OVERLAY_FRAME_INTERVAL_SECONDS - elapsed)
        if has_wait_for_skip:
            if display.wait_for_skip(sleep_for):
                return
        else:
            time.sleep(sleep_for)


# ─── Main ticker loop ───────────────────────────────────────────────────────


def _run_ticker(display, rows: list[_TickerRow]) -> ScreenImage:
    clear_display(display)
    pygame_module = _pygame_module_for_display(display)
    touch_capable = _is_touch_capable(pygame_module)

    row_height, row_tops = _compute_row_layout(len(rows))
    rows = rows[: len(row_tops)]

    skip_requested = getattr(display, "skip_requested", None)
    has_wait_for_skip = callable(getattr(display, "wait_for_skip", None))

    def _should_stop() -> bool:
        return bool(callable(skip_requested) and skip_requested())

    last_frame: Optional[Image.Image] = None
    hit_rects: list[tuple[int, int, int, int, NewsHeadline]] = []
    end_time = time.monotonic() + float(NEWS_HEADLINES_DISPLAY_SECONDS)

    while time.monotonic() < end_time and not _should_stop():
        frame_start = time.monotonic()

        if touch_capable:
            for tap_x, tap_y in _poll_taps(display, pygame_module):
                headline = _hit_test(hit_rects, tap_x, tap_y)
                if headline is not None:
                    _show_reader_overlay(display, headline, pygame_module)
                    end_time = time.monotonic() + float(NEWS_HEADLINES_DISPLAY_SECONDS)
                    frame_start = time.monotonic()

        last_frame, hit_rects = _render_frame(rows, row_height, row_tops)
        display.image(last_frame)
        for row in rows:
            row.offset += row.speed

        elapsed = time.monotonic() - frame_start
        sleep_for = max(0.0, _FRAME_INTERVAL_SECONDS - elapsed)
        if has_wait_for_skip:
            if display.wait_for_skip(sleep_for):
                break
        else:
            time.sleep(sleep_for)

    if last_frame is None:
        last_frame, _hit_rects = _render_frame(rows, row_height, row_tops)
        display.image(last_frame)

    return ScreenImage(last_frame, displayed=True, consumed_delay=True)


@log_call
def draw_news_headlines(display, transition: bool = True) -> ScreenImage:
    """Render the multi-topic news ticker screen.

    Topics/feeds come from news_feeds.json (see
    paths.resolve_news_feeds_config_path). Each topic becomes one ticker
    lane; lanes with no fetched headlines are skipped for this pass rather
    than shown empty.
    """

    if not config.ENABLE_NEWS_HEADLINES:
        return _render_empty_state(display)

    topics, _headline_count, _refresh_minutes = load_news_feed_config()
    if not topics:
        logging.warning("news_headlines: no topics configured in news_feeds.json")
        return _render_empty_state(display)

    headlines_by_topic = fetch_all_headlines()
    row_count_estimate = len(topics) + (1 if config.ENABLE_STOCK_TICKER else 0)
    row_height_estimate, _row_tops_estimate = _compute_row_layout(row_count_estimate)
    rows = _build_rows(topics, headlines_by_topic, row_height_estimate)

    if config.ENABLE_STOCK_TICKER:
        quotes = fetch_stock_quotes(default_symbol_order())
        stock_row = _build_stock_row(quotes)
        if stock_row is not None:
            rows.append(stock_row)

    if not rows:
        return _render_empty_state(display)

    return _run_ticker(display, rows)
