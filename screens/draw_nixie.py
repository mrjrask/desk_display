#!/usr/bin/env python3
"""Animated Nixie tube clock that prefers rich image assets when available."""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from config import (
    HEIGHT,
    WIDTH,
    SCREEN_DELAY,
    TIMES_SQUARE_FONT_PATH,
    get_screen_background_color,
)
from utils import ScreenImage, clear_display, log_call

BACKGROUND_COLOR = get_screen_background_color("nixie", (0, 0, 0))
TUBE_HIGHLIGHT = (255, 214, 170, 90)
DIGIT_COLOR = (255, 232, 179, 255)
DIGIT_EDGE_COLOR = (255, 180, 90, 255)
GLOW_COLOR = (255, 118, 44, 220)

H_MARGIN = 28
V_MARGIN = 24
SPACING_RATIO = 0.12
COLON_RATIO = 0.28
COLON_SIZE_RATIO = 0.35

LOGGER = logging.getLogger(__name__)


def _get_time_format() -> str:
    """Return the configured time format ('12' or '24'), defaulting to '12'."""

    value = os.environ.get("NIXIE_TIME_FORMAT", "12").strip()
    if value in ("12", "24"):
        return value
    return "12"


def _candidate_asset_directories() -> Iterable[Path]:
    """Yield likely directories containing downloaded Nixie digit artwork."""

    repo_root = Path(__file__).resolve().parents[1]
    images_dir = repo_root / "images"
    candidates = [
        images_dir / "nixie-digits",
        images_dir / "nixie_digits",
        images_dir / "nixie",
        images_dir / "nixie_clock",
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            yield path


def _directories_with_digits(base_directories: Iterable[Path]) -> Sequence[Path]:
    """Return all directories under ``base_directories`` containing 0-9 assets."""

    found: list[Path] = []
    seen: set[Path] = set()

    def _maybe_add(directory: Path) -> None:
        if directory in seen:
            return
        if all((directory / f"{digit}.png").exists() for digit in "0123456789"):
            found.append(directory)
            seen.add(directory)

    for root in base_directories:
        _maybe_add(root)
        for child in root.iterdir():
            if child.is_dir():
                _maybe_add(child)

    return tuple(found)


ASSET_DIGIT_DIRECTORIES = _directories_with_digits(_candidate_asset_directories())


def _sample_digit_height(directory: Path) -> Optional[int]:
    try:
        with Image.open(directory / "0.png") as img:
            return img.height
    except Exception:  # pragma: no cover - defensive guard for malformed assets
        LOGGER.exception("Failed to read sample digit height from %s", directory)
        return None


@lru_cache(maxsize=32)
def _preferred_digit_directory(height: int) -> Optional[Path]:
    """Choose the asset directory whose digits best match ``height``."""

    choices: list[tuple[Path, int]] = []
    for directory in ASSET_DIGIT_DIRECTORIES:
        sample_height = _sample_digit_height(directory)
        if sample_height:
            choices.append((directory, sample_height))

    if not choices:
        return None

    larger_or_equal = [item for item in choices if item[1] >= height]
    if larger_or_equal:
        directory, _ = min(larger_or_equal, key=lambda item: (item[1], abs(item[1] - height)))
        return directory

    # No large-enough asset; pick the highest resolution available.
    directory, _ = max(choices, key=lambda item: item[1])
    return directory


def _detect_colon_asset(preferred: Optional[Path] = None) -> Optional[Path]:
    def _search(directory: Path) -> Optional[Path]:
        for name in ("colon.png", "dot.png", "colon.gif"):
            candidate = directory / name
            if candidate.exists():
                return candidate
        parent = directory.parent
        if parent != directory:
            for name in ("colon.png", "dot.png", "colon.gif"):
                candidate = parent / name
                if candidate.exists():
                    return candidate
        return None

    if preferred:
        path = _search(preferred)
        if path:
            return path

    for directory in ASSET_DIGIT_DIRECTORIES:
        path = _search(directory)
        if path:
            return path
    return None


def _load_and_scale(path: Path, height: int) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if image.height == height:
        return image

    target_height = max(1, height)
    target_width = max(1, int(round(image.width * (target_height / image.height))))
    return image.resize((target_width, target_height), Image.LANCZOS)


@lru_cache(maxsize=32)
def _font_for_height(height: int) -> ImageFont.FreeTypeFont:
    target_height = max(20, height)
    size = int(round(target_height * 0.95))
    size = max(size, 24)

    while size > 8:
        font = ImageFont.truetype(TIMES_SQUARE_FONT_PATH, size=size)
        bbox = font.getbbox("0")
        text_height = bbox[3] - bbox[1]
        if text_height <= target_height * 0.92:
            return font
        size -= 1

    return ImageFont.truetype(TIMES_SQUARE_FONT_PATH, size=10)


@lru_cache(maxsize=128)
def _generate_digit_image(height: int, value: str) -> Image.Image:
    font = _font_for_height(height)
    padding_x = max(6, int(round(height * 0.1)))
    bbox = font.getbbox(value)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    canvas_height = max(height, text_height + padding_x)
    img = Image.new("RGBA", (text_width + padding_x * 2, canvas_height), (0, 0, 0, 0))

    # Vertical alignment keeps the digits visually centered within the tube.
    offset_y = (canvas_height - text_height) // 2 - bbox[1]
    text_position = (padding_x - bbox[0], offset_y)

    glow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_draw.text(text_position, value, fill=GLOW_COLOR, font=font)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=max(3, int(round(height * 0.12)))))

    digit_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    digit_draw = ImageDraw.Draw(digit_layer)
    digit_draw.text(text_position, value, fill=DIGIT_COLOR, font=font)
    stroke_width = max(1, int(round(height * 0.04)))
    digit_draw.text(
        text_position,
        value,
        fill=DIGIT_COLOR,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=DIGIT_EDGE_COLOR,
    )

    combined = Image.new("RGBA", img.size, (0, 0, 0, 0))
    combined = Image.alpha_composite(combined, glow_layer)
    combined = Image.alpha_composite(combined, digit_layer)

    # Subtle glass reflection
    highlight = Image.new("RGBA", img.size, (0, 0, 0, 0))
    highlight_draw = ImageDraw.Draw(highlight)
    highlight_height = max(2, int(round(canvas_height * 0.18)))
    highlight_draw.rectangle(
        (
            int(padding_x * 0.3),
            int(canvas_height * 0.12),
            int(img.width * 0.35),
            int(canvas_height * 0.12) + highlight_height,
        ),
        fill=TUBE_HIGHLIGHT,
    )
    highlight = highlight.filter(ImageFilter.GaussianBlur(radius=max(2, int(round(height * 0.05)))))
    combined = Image.alpha_composite(combined, highlight)

    return combined


@lru_cache(maxsize=128)
def _asset_digit_image(height: int, value: str) -> Optional[Image.Image]:
    asset_dir = _preferred_digit_directory(height)
    if not asset_dir:
        return None

    path = asset_dir / f"{value}.png"
    if not path.exists():
        LOGGER.warning("Nixie asset missing for digit %s at %s", value, path)
        return None

    try:
        return _load_and_scale(path, height)
    except Exception:  # pragma: no cover - asset could not be opened or resized
        LOGGER.exception("Failed to load Nixie digit asset: %s", path)
        return None


@lru_cache(maxsize=128)
def _digit_image(height: int, value: str) -> Image.Image:
    asset = _asset_digit_image(height, value)
    if asset is not None:
        return asset
    return _generate_digit_image(height, value)


@lru_cache(maxsize=32)
def _digits_for_height(height: int) -> Dict[str, Image.Image]:
    return {value: _digit_image(height, value) for value in "0123456789"}


@lru_cache(maxsize=32)
def _colon_image(height: int) -> Image.Image:
    asset_colon_path = _detect_colon_asset(_preferred_digit_directory(height))
    if asset_colon_path:
        if asset_colon_path.parent in ASSET_DIGIT_DIRECTORIES:
            asset = _asset_digit_image(height, asset_colon_path.stem)
            if asset is not None:
                return asset
        try:
            return _load_and_scale(asset_colon_path, max(12, height))
        except Exception:  # pragma: no cover - fallback to procedural colon
            LOGGER.exception("Failed to load Nixie colon asset: %s", asset_colon_path)

    height = max(12, height)
    width = max(12, int(round(height * COLON_RATIO)))
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    dots = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    dot_draw = ImageDraw.Draw(dots)

    top_cy = int(round(height * 0.32))
    bottom_cy = int(round(height * 0.68))
    radius = max(2, int(round(height * 0.08)))
    glow_radius = max(radius + 2, int(round(height * 0.18)))
    center_x = width // 2

    for cy in (top_cy, bottom_cy):
        glow_draw.ellipse(
            (
                center_x - glow_radius,
                cy - glow_radius,
                center_x + glow_radius,
                cy + glow_radius,
            ),
            fill=(252, 121, 44, 110),
        )
        dot_draw.ellipse(
            (
                center_x - radius,
                cy - radius,
                center_x + radius,
                cy + radius,
            ),
            fill=(255, 248, 138, 255),
        )

    combined = Image.alpha_composite(glow, dots)
    return combined


def _compose_frame(now: dt.datetime | None = None) -> Image.Image:
    now = now or dt.datetime.now()

    # Format time according to user preference (12 or 24 hour)
    time_format = _get_time_format()
    if time_format == "12":
        time_digits = now.strftime("%I%M%S")
    else:
        time_digits = now.strftime("%H%M%S")

    elements = [time_digits[0], time_digits[1], ":", time_digits[2], time_digits[3], ":", time_digits[4], time_digits[5]]

    available_height = max(1, HEIGHT - V_MARGIN * 2)
    available_width = max(1, WIDTH - H_MARGIN * 2)
    gap_count = len(elements) - 1

    digits_scaled = _digits_for_height(available_height)
    sample = digits_scaled["0"]
    digit_ratio = sample.width / float(sample.height)

    total_ratio = digit_ratio * 6 + COLON_RATIO * 2 + SPACING_RATIO * gap_count
    height_by_width = int(round(available_width / total_ratio))
    target_height = max(48, min(available_height, height_by_width))
    spacing = max(8, int(round(target_height * SPACING_RATIO)))

    digits_scaled = _digits_for_height(target_height)
    colon_img = _colon_image(int(target_height * COLON_SIZE_RATIO))

    element_widths = []
    for item in elements:
        if item == ":":
            element_widths.append(colon_img.width)
        else:
            element_widths.append(digits_scaled[item].width)

    total_width = sum(element_widths) + spacing * gap_count
    x = max(H_MARGIN, (WIDTH - total_width) // 2)
    y = max(V_MARGIN, (HEIGHT - target_height) // 2)

    frame = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)

    for item, width_px in zip(elements, element_widths):
        if item == ":":
            # Center colons vertically with respect to digit height
            colon_y_offset = (target_height - colon_img.height) // 2
            frame.paste(colon_img, (x, y + colon_y_offset), colon_img)
        else:
            digit_img = digits_scaled[item]
            frame.paste(digit_img, (x, y), digit_img)
        x += width_px + spacing

    return frame


def nixie_frame(now: dt.datetime | None = None) -> Image.Image:
    """Compose a Nixie clock frame for the provided time (or now)."""

    return _compose_frame(now)


def _play_flicker(display, base: Image.Image) -> None:
    enhancer = ImageEnhance.Brightness(base)
    for factor in (0.94, 1.06, 1.0):
        frame = enhancer.enhance(factor)
        try:
            display.image(frame)
            if hasattr(display, "show"):
                display.show()
        except Exception:  # pragma: no cover - defensive refresh guard
            break
        time.sleep(0.08)


@log_call
def draw_nixie(display, transition: bool = False):
    global BACKGROUND_COLOR
    BACKGROUND_COLOR = get_screen_background_color("nixie", (0, 0, 0))
    frame = _compose_frame()

    if transition and not hasattr(display, "image"):
        return frame

    clear_display(display)

    last_frame = frame
    end_time = time.monotonic() + max(1, SCREEN_DELAY)
    last_second = None

    while True:
        now = dt.datetime.now()
        if now.second != last_second:
            last_second = now.second
            last_frame = _compose_frame(now)
            try:
                display.image(last_frame)
                if hasattr(display, "show"):
                    display.show()
            except Exception:  # pragma: no cover - defensive refresh guard
                logging.exception("Failed to render Nixie clock")
                return ScreenImage(last_frame, displayed=False)

            if not transition:
                _play_flicker(display, last_frame)

        remaining = end_time - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.25, remaining))

    return ScreenImage(last_frame, displayed=True)
