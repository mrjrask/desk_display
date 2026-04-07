#!/usr/bin/env python3
"""Render the Wrigley Field webcam feed."""

from __future__ import annotations

import io
import logging
import re
import time
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin

import requests
from PIL import Image, ImageDraw, ImageOps

from config import HEIGHT, WIDTH, FONT_STOCK_TEXT, FONT_STOCK_TITLE, get_screen_background_color
from utils import clear_display, log_call

WRIGLEY_PAGE_URL = "https://www.earthcam.com/usa/illinois/chicago/wrigleyfield/?cam=wrigleyfield_hd"
_REQUEST_TIMEOUT = 8
_SNAPSHOT_TTL_SECONDS = 20
_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; desk-display/1.0; +https://www.earthcam.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
    "Referer": WRIGLEY_PAGE_URL,
}

_SESSION = requests.Session()

_snapshot_cache: dict[str, object] = {"image": None, "ts": 0.0, "source": None}
_DISALLOWED_URL_TOKENS = (
    "square",
    "logo",
    "icon",
    "avatar",
    "sprite",
    "thumb",
    "thumbnail",
    "adservice",
    "placeholder",
    "generic",
    "default",
    "offline",
    "unavailable",
    "noimage",
    "comingsoon",
    "preview",
    "poster",
)


class _MediaCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.candidates: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {k.lower(): (v or "") for k, v in attrs}
        tag_lower = tag.lower()
        if tag_lower == "img":
            for key in ("src", "data-src", "data-lazy-src", "srcset"):
                value = attrs_map.get(key, "")
                if value:
                    self._add(value)
        if tag_lower in {"iframe", "video", "source"}:
            src = attrs_map.get("src", "")
            if src:
                self._add(src)

    def _add(self, value: str) -> None:
        raw = value.split(",")[0].strip().split()[0].strip()
        if raw:
            self.candidates.append(raw)


def _looks_like_image_url(url: str) -> bool:
    lowered = url.lower()
    return any(
        token in lowered
        for token in (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            "snapshot",
            "jpg?",
            "jpeg?",
            "image",
            "camera",
            "cam",
            "mjpg",
        )
    )


def _should_skip_candidate_url(url: str) -> bool:
    lowered = url.lower()
    return any(token in lowered for token in _DISALLOWED_URL_TOKENS)


def _candidate_priority(url: str) -> int:
    lowered = url.lower()
    score = 0
    for token, weight in (
        ("wrigley", 8),
        ("snapshot", 6),
        ("live", 4),
        ("camera", 3),
        ("cam", 3),
        ("stream", 2),
        ("hd", 2),
    ):
        if token in lowered:
            score += weight
    return score


def _candidate_urls_from_html(html: str, base_url: str) -> list[str]:
    collector = _MediaCollector()
    collector.feed(html)

    regex_hits = re.findall(r"https?://[^\"'\s<>]+", html, flags=re.IGNORECASE)
    regex_hits.extend(re.findall(r"(?:src|href)=['\"]([^'\"]+)['\"]", html, flags=re.IGNORECASE))

    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in [*collector.candidates, *regex_hits]:
        absolute = urljoin(base_url, candidate.strip())
        if not absolute.startswith(("http://", "https://")):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        ordered.append(absolute)

    image_like = [url for url in ordered if _looks_like_image_url(url)]
    image_like.sort(key=_candidate_priority, reverse=True)
    remaining = [url for url in ordered if url not in image_like]
    remaining.sort(key=_candidate_priority, reverse=True)
    return image_like + remaining


def _fetch_image(url: str) -> Optional[Image.Image]:
    try:
        response = _SESSION.get(url, timeout=_REQUEST_TIMEOUT, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            return None
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except Exception:
        return None


def _is_plausible_webcam_frame(image: Image.Image) -> bool:
    width, height = image.size
    if width < 240 or height < 140:
        return False
    ratio = width / max(height, 1)
    return ratio >= 1.2


def _download_wrigley_frame() -> tuple[Optional[Image.Image], Optional[str]]:
    response = _SESSION.get(WRIGLEY_PAGE_URL, timeout=_REQUEST_TIMEOUT, headers=_REQUEST_HEADERS)
    response.raise_for_status()
    candidates = _candidate_urls_from_html(response.text, WRIGLEY_PAGE_URL)

    for candidate in candidates[:30]:
        if _should_skip_candidate_url(candidate):
            continue
        image = _fetch_image(candidate)
        if image is not None and _is_plausible_webcam_frame(image):
            return image, candidate
    return None, None


def _fallback_image(message: str) -> Image.Image:
    bg = get_screen_background_color("Wrigley", (0, 0, 0))
    img = Image.new("RGB", (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(img)
    title = "Wrigley"
    subtitle = "Camera unavailable"

    tw, th = draw.textsize(title, font=FONT_STOCK_TITLE)
    draw.text(((WIDTH - tw) // 2, 12), title, fill=(255, 255, 255), font=FONT_STOCK_TITLE)

    sw, sh = draw.textsize(subtitle, font=FONT_STOCK_TEXT)
    draw.text(((WIDTH - sw) // 2, max(40, HEIGHT // 2 - 20)), subtitle, fill=(220, 220, 220), font=FONT_STOCK_TEXT)

    for idx, line in enumerate((message, "earthcam.com")):
        lw, lh = draw.textsize(line, font=FONT_STOCK_TEXT)
        draw.text(((WIDTH - lw) // 2, HEIGHT - (2 - idx) * (lh + 6) - 8), line, fill=(180, 180, 180), font=FONT_STOCK_TEXT)

    return img


def _wrigley_image() -> Image.Image:
    now = time.time()
    cached = _snapshot_cache.get("image")
    if isinstance(cached, Image.Image) and (now - float(_snapshot_cache.get("ts", 0.0))) < _SNAPSHOT_TTL_SECONDS:
        return cached

    try:
        image, source = _download_wrigley_frame()
    except Exception as exc:
        logging.warning("Wrigley cam: failed to fetch webcam frame: %s", exc)
        image, source = None, None

    if image is None:
        fallback = _fallback_image("Live feed did not return an image")
        _snapshot_cache.update({"image": fallback, "ts": now, "source": None})
        return fallback

    fitted = ImageOps.fit(image, (WIDTH, HEIGHT), method=Image.LANCZOS)
    _snapshot_cache.update({"image": fitted, "ts": now, "source": source})
    return fitted


@log_call
def draw_wrigley_cam(display, transition: bool = False) -> Image.Image:
    img = _wrigley_image()
    clear_display(display)
    return img
