#!/usr/bin/env python3
"""Render simple status content on the Waveshare OLED/LCD HAT (A) OLED displays.

Default behavior:
- Left OLED (0x3c): current date in M/D/YY format
- Right OLED (0x3d): local time in 12-hour format, no leading zero (small AM/PM)
"""

from __future__ import annotations

import os
import random
import re
import signal
import subprocess
import time
import logging
import json
import importlib.util
import importlib
import sys
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from threading import Event

from PIL import Image, ImageDraw, ImageFont

try:
    from smbus import SMBus
except ImportError:  # pragma: no cover - environment specific fallback
    from smbus2 import SMBus


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw, 0)
    except ValueError:
        return default


I2C_BUS = _env_int("WAVESHARE_OLED_I2C_BUS", 1)
TEMP_ADDR = _env_int("WAVESHARE_OLED_TEMP_ADDR", 0x3C)
TIME_ADDR = _env_int("WAVESHARE_OLED_TIME_ADDR", 0x3D)
OLED_WIDTH = _env_int("WAVESHARE_OLED_WIDTH", 128)
OLED_HEIGHT = _env_int("WAVESHARE_OLED_HEIGHT", 64)
TEMP_SOURCE = os.getenv("WAVESHARE_OLED_TEMP_SOURCE", "weather1").strip().lower()
TEMP_COMMAND = os.getenv("WAVESHARE_OLED_TEMP_COMMAND", "")
TEMP_UNIT = os.getenv("WAVESHARE_OLED_TEMP_UNIT", "C").strip().upper()
REFRESH_SECONDS = max(1, _env_int("WAVESHARE_OLED_REFRESH_SECONDS", 5))
SWAP_INTERVAL_MIN_SECONDS = max(1, _env_int("WAVESHARE_OLED_SWAP_INTERVAL_MIN_SECONDS", 60))
SWAP_INTERVAL_MAX_SECONDS = max(
    SWAP_INTERVAL_MIN_SECONDS,
    _env_int("WAVESHARE_OLED_SWAP_INTERVAL_MAX_SECONDS", 240),
)
FADE_STEPS = max(1, _env_int("WAVESHARE_OLED_FADE_STEPS", 8))
FADE_STEP_MS = max(5, _env_int("WAVESHARE_OLED_FADE_STEP_MS", 35))
WAIT_FOR_WEATHER2 = os.getenv("WAVESHARE_OLED_WAIT_FOR_WEATHER2", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
MIN_VALUE_FONT_SIZE = max(6, _env_int("WAVESHARE_OLED_MIN_VALUE_FONT_SIZE", 8))
MAX_VALUE_FONT_SIZE = max(
    MIN_VALUE_FONT_SIZE,
    _env_int("WAVESHARE_OLED_MAX_VALUE_FONT_SIZE", 26),
)
MIN_TIME_FONT_SIZE = max(6, _env_int("WAVESHARE_OLED_MIN_TIME_FONT_SIZE", MIN_VALUE_FONT_SIZE))
MAX_TIME_FONT_SIZE = max(
    MIN_TIME_FONT_SIZE,
    _env_int("WAVESHARE_OLED_MAX_TIME_FONT_SIZE", MAX_VALUE_FONT_SIZE),
)


LOGGER = logging.getLogger("waveshare_oled_status")
_STOP_EVENT = Event()
_LAST_WEATHER_TEMP_F: float | None = None
_WEATHER2_RENDERED = False


class SSD1306Display:
    def __init__(self, bus: SMBus, address: int, width: int = 128, height: int = 64) -> None:
        self.bus = bus
        self.address = address
        self.width = width
        self.height = height
        self.pages = height // 8

    def _cmd(self, value: int) -> None:
        self.bus.write_byte_data(self.address, 0x00, value)

    def _data(self, data: bytes) -> None:
        chunk_size = 16
        for idx in range(0, len(data), chunk_size):
            chunk = data[idx : idx + chunk_size]
            self.bus.write_i2c_block_data(self.address, 0x40, list(chunk))

    def initialize(self) -> None:
        init_sequence = [
            0xAE,
            0x20,
            0x00,
            0xB0,
            0xC8,
            0x00,
            0x10,
            0x40,
            0x81,
            0x7F,
            0xA1,
            0xA6,
            0xA8,
            self.height - 1,
            0xA4,
            0xD3,
            0x00,
            0xD5,
            0x80,
            0xD9,
            0xF1,
            0xDA,
            0x12 if self.height == 64 else 0x02,
            0xDB,
            0x40,
            0x8D,
            0x14,
            0xAF,
        ]
        for cmd in init_sequence:
            self._cmd(cmd)

    def set_contrast(self, value: int) -> None:
        contrast = max(0, min(255, int(value)))
        self._cmd(0x81)
        self._cmd(contrast)

    def clear(self) -> None:
        self.display_image(Image.new("1", (self.width, self.height), 0))

    def display_image(self, image: Image.Image) -> None:
        if image.mode != "1":
            image = image.convert("1")
        image = image.resize((self.width, self.height))
        pixels = image.load()

        for page in range(self.pages):
            self._cmd(0xB0 + page)
            self._cmd(0x00)
            self._cmd(0x10)
            buf = bytearray(self.width)
            for x in range(self.width):
                value = 0
                for bit in range(8):
                    y = page * 8 + bit
                    if y < self.height and pixels[x, y] != 0:
                        value |= 1 << bit
                buf[x] = value
            self._data(bytes(buf))


def _parse_temperature_value(text: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _read_cpu_temp_c() -> float | None:
    candidates = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            raw = open(path, "r", encoding="utf-8").read().strip()
            value = float(raw)
            if value > 1000:
                value = value / 1000.0
            return value
        except Exception:
            continue

    try:
        output = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, timeout=2)
        return _parse_temperature_value(output)
    except Exception:
        return None


def _read_weather1_temp_f() -> float | None:
    global _LAST_WEATHER_TEMP_F

    def _resolve_fetch_weather():
        try:
            module = importlib.import_module("data_fetch")
            fetch_weather = getattr(module, "fetch_weather", None)
            if callable(fetch_weather):
                return fetch_weather
            legacy = getattr(module, "get_weather_data", None)
            if callable(legacy):
                return legacy
        except Exception:
            pass

        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "data_fetch.py"
        if not module_path.exists():
            return None

        spec = importlib.util.spec_from_file_location("data_fetch", module_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("data_fetch", module)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None

        fetch_weather = getattr(module, "fetch_weather", None)
        if callable(fetch_weather):
            return fetch_weather
        legacy = getattr(module, "get_weather_data", None)
        return legacy if callable(legacy) else None

    fetch_weather = _resolve_fetch_weather()
    if fetch_weather is None:
        return _LAST_WEATHER_TEMP_F

    def _fetch_weather_payload(force_refresh: bool):
        try:
            return fetch_weather(force_refresh=force_refresh)
        except TypeError:
            # Backward compatibility for older data_fetch modules.
            if force_refresh:
                return None
            return fetch_weather()

    try:
        # Prefer cached weather to avoid API rate limits and transient misses on
        # the OLED loop cadence.
        weather = _fetch_weather_payload(force_refresh=False)
        if weather is None:
            weather = _fetch_weather_payload(force_refresh=True)
    except Exception:
        return _LAST_WEATHER_TEMP_F

    if not isinstance(weather, dict):
        return _LAST_WEATHER_TEMP_F

    current = weather.get("current")
    if not isinstance(current, dict):
        return _LAST_WEATHER_TEMP_F

    temp_f = (
        current.get("temp")
        or current.get("temp_f")
        or current.get("temperature")
    )
    try:
        _LAST_WEATHER_TEMP_F = float(temp_f)
        return _LAST_WEATHER_TEMP_F
    except (TypeError, ValueError):
        return _LAST_WEATHER_TEMP_F


def _display_status_path() -> Path:
    override = os.getenv("WAVESHARE_OLED_DISPLAY_STATUS_PATH")
    if override:
        return Path(override).expanduser()

    repo_root = Path(__file__).resolve().parents[1]
    screenshot_dir = os.getenv("SCREENSHOT_DIR", str(repo_root / "screenshots"))
    return Path(screenshot_dir).expanduser() / "current" / "display_status.json"


def _weather2_screen_has_rendered() -> bool:
    global _WEATHER2_RENDERED
    if _WEATHER2_RENDERED:
        return True
    if TEMP_SOURCE not in {"weather", "weather1"}:
        return True
    if not WAIT_FOR_WEATHER2:
        return True

    status_path = _display_status_path()
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if str(payload.get("screen_id", "")).strip().lower() == "weather2":
        _WEATHER2_RENDERED = True
        return True
    return False


def read_temperature() -> str:
    value_c: float | None = None
    value_f: float | None = None

    if TEMP_SOURCE == "command" and TEMP_COMMAND:
        try:
            output = subprocess.check_output(TEMP_COMMAND, shell=True, text=True, timeout=4)
            value_c = _parse_temperature_value(output)
        except Exception:
            value_c = None
    elif TEMP_SOURCE in {"weather", "weather1"}:
        value_f = _read_weather1_temp_f()
    else:
        value_c = _read_cpu_temp_c()

    if value_f is None and value_c is None:
        return ""

    if value_f is not None:
        return f"{round(value_f)}°F"

    if TEMP_UNIT == "F":
        return f"{(value_c * 9 / 5) + 32:.1f}°F"

    return f"{value_c:.1f}°C"


def current_time_12h() -> str:
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def current_date_mdy() -> str:
    now = datetime.now()
    return f"{now.month}/{now.day}/{now.strftime('%y')}"


def random_swap_interval_seconds() -> int:
    return random.randint(SWAP_INTERVAL_MIN_SECONDS, SWAP_INTERVAL_MAX_SECONDS)


@lru_cache(maxsize=96)
def _load_value_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate_paths = [
        os.getenv("WAVESHARE_OLED_FONT_PATH"),
        "/workspace/desk_display/fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidate_paths:
        if not path:
            continue
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _best_value_font_size(width: int, height: int, text: str, top_margin: int) -> int:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    max_height = height - top_margin - 2
    best_size = MIN_VALUE_FONT_SIZE
    for size in range(MIN_VALUE_FONT_SIZE, MAX_VALUE_FONT_SIZE + 1):
        font = _load_value_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= width - 4 and text_h <= max_height:
            best_size = size
        else:
            break
    return best_size


def _best_time_font_size(width: int, height: int, time_text: str, top_margin: int) -> int:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    max_height = height - top_margin - 2
    best_size = MIN_TIME_FONT_SIZE
    time_match = re.match(r"^(.*?)(?:\s+([AP]M))?$", time_text.strip(), re.IGNORECASE)
    base_time = time_match.group(1) if time_match else time_text
    meridiem = (time_match.group(2) or "").upper() if time_match else ""

    for size in range(MIN_TIME_FONT_SIZE, MAX_TIME_FONT_SIZE + 1):
        main_font = _load_value_font(size)
        meridiem_font = _load_value_font(max(8, size // 2))

        main_bbox = draw.textbbox((0, 0), base_time, font=main_font)
        main_w = main_bbox[2] - main_bbox[0]
        main_h = main_bbox[3] - main_bbox[1]

        gap = 3 if meridiem else 0
        meridiem_bbox = (
            draw.textbbox((0, 0), meridiem, font=meridiem_font)
            if meridiem
            else (0, 0, 0, 0)
        )
        meridiem_w = meridiem_bbox[2] - meridiem_bbox[0]
        meridiem_h = meridiem_bbox[3] - meridiem_bbox[1]

        total_w = main_w + gap + meridiem_w
        total_h = max(main_h, meridiem_h)
        if total_w <= width - 4 and total_h <= max_height:
            best_size = size
        else:
            break
    return best_size


def _title_top_margin(width: int, title: str | None) -> int:
    if not title:
        return 2
    image = Image.new("1", (width, OLED_HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_h = title_bbox[3] - title_bbox[1]
    return max(2, title_h + 6)


def render_centered_text(
    width: int,
    height: int,
    text: str,
    *,
    title: str | None = None,
    value_font_size: int | None = None,
) -> Image.Image:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()

    y_offset = 0
    if title:
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_w) // 2, 2), title, font=title_font, fill=255)
        y_offset = title_h + 6

    top_margin = max(2, y_offset)
    if value_font_size is None:
        value_font_size = _best_value_font_size(width, height, text, top_margin)

    best_font = _load_value_font(value_font_size)
    best_bbox = draw.textbbox((0, 0), text, font=best_font)

    if best_bbox is None:
        best_bbox = draw.textbbox((0, 0), text, font=best_font)

    value_w = best_bbox[2] - best_bbox[0]
    value_h = best_bbox[3] - best_bbox[1]
    max_height = height - top_margin - 2
    value_x = (width - value_w) // 2
    value_y = top_margin + max(0, (max_height - value_h) // 2)
    draw.text((value_x, value_y), text, font=best_font, fill=255)
    return image


def render_centered_time_text(
    width: int,
    height: int,
    time_text: str,
    *,
    title: str | None = None,
    value_font_size: int | None = None,
) -> Image.Image:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()

    y_offset = 0
    if title:
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_w) // 2, 2), title, font=title_font, fill=255)
        y_offset = title_h + 6

    top_margin = max(2, y_offset)
    time_match = re.match(r"^(.*?)(?:\s+([AP]M))?$", time_text.strip(), re.IGNORECASE)
    base_time = time_match.group(1) if time_match else time_text
    meridiem = (time_match.group(2) or "").upper() if time_match else ""

    if value_font_size is None:
        value_font_size = _best_value_font_size(width, height, base_time, top_margin)

    main_font = _load_value_font(value_font_size)
    meridiem_font = _load_value_font(max(8, value_font_size // 2))

    main_bbox = draw.textbbox((0, 0), base_time, font=main_font)
    main_w = main_bbox[2] - main_bbox[0]
    main_h = main_bbox[3] - main_bbox[1]

    gap = 3 if meridiem else 0
    meridiem_bbox = draw.textbbox((0, 0), meridiem, font=meridiem_font) if meridiem else (0, 0, 0, 0)
    meridiem_w = meridiem_bbox[2] - meridiem_bbox[0]
    meridiem_h = meridiem_bbox[3] - meridiem_bbox[1]

    total_w = main_w + gap + meridiem_w
    total_h = max(main_h, meridiem_h)
    max_height = height - top_margin - 2
    start_x = (width - total_w) // 2
    start_y = top_margin + max(0, (max_height - total_h) // 2)

    draw.text((start_x, start_y), base_time, font=main_font, fill=255)
    if meridiem:
        meridiem_y = start_y + max(0, main_h - meridiem_h)
        draw.text((start_x + main_w + gap, meridiem_y), meridiem, font=meridiem_font, fill=255)

    return image


def fade_transition(display: SSD1306Display, new_image: Image.Image) -> None:
    for step in range(FADE_STEPS, -1, -1):
        display.set_contrast(int(255 * step / FADE_STEPS))
        time.sleep(FADE_STEP_MS / 1000)

    display.display_image(new_image)

    for step in range(0, FADE_STEPS + 1):
        display.set_contrast(int(255 * step / FADE_STEPS))
        time.sleep(FADE_STEP_MS / 1000)


def _request_stop(signum: int, _frame: object) -> None:
    LOGGER.info("Received signal %s; stopping OLED helper loop.", signum)
    _STOP_EVENT.set()


def _safe_clear_display(display: SSD1306Display, name: str) -> None:
    try:
        display.clear()
    except Exception as exc:
        LOGGER.warning("Failed to clear %s OLED during shutdown: %s", name, exc)


def _safe_render(display: SSD1306Display, image: Image.Image, name: str) -> bool:
    try:
        fade_transition(display, image)
        return True
    except Exception as exc:
        LOGGER.warning("Failed to render frame on %s OLED: %s", name, exc)
        return False


def main() -> int:
    logging.basicConfig(
        level=os.getenv("WAVESHARE_OLED_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    bus = SMBus(I2C_BUS)
    temp_display = SSD1306Display(bus, TEMP_ADDR, OLED_WIDTH, OLED_HEIGHT)
    time_display = SSD1306Display(bus, TIME_ADDR, OLED_WIDTH, OLED_HEIGHT)

    temp_display.initialize()
    time_display.initialize()
    temp_display.clear()
    time_display.clear()

    show_date_on_left = True
    next_swap_at = time.monotonic() + random_swap_interval_seconds()
    try:
        while not _STOP_EVENT.is_set():
            time_text = current_time_12h()
            date_text = current_date_mdy()
            time_top_margin = _title_top_margin(OLED_WIDTH, "Time")
            date_top_margin = _title_top_margin(OLED_WIDTH, "Date")
            time_value_font_size = _best_time_font_size(
                OLED_WIDTH,
                OLED_HEIGHT,
                time_text,
                time_top_margin,
            )
            date_value_font_size = _best_value_font_size(
                OLED_WIDTH,
                OLED_HEIGHT,
                date_text,
                date_top_margin,
            )

            date_image = render_centered_text(
                OLED_WIDTH,
                OLED_HEIGHT,
                date_text,
                title="Date",
                value_font_size=date_value_font_size,
            )
            time_image = render_centered_time_text(
                OLED_WIDTH,
                OLED_HEIGHT,
                time_text,
                title="Time",
                value_font_size=time_value_font_size,
            )

            left_image, right_image = (
                (date_image, time_image) if show_date_on_left else (time_image, date_image)
            )
            left_ok = _safe_render(temp_display, left_image, "left")
            right_ok = _safe_render(time_display, right_image, "right")

            if not left_ok or not right_ok:
                LOGGER.info("Reinitializing OLED displays after render failure.")
                try:
                    temp_display.initialize()
                    time_display.initialize()
                except Exception as exc:
                    LOGGER.warning("OLED reinitialization failed: %s", exc)

            if time.monotonic() >= next_swap_at:
                show_date_on_left = not show_date_on_left
                next_swap_at = time.monotonic() + random_swap_interval_seconds()

            _STOP_EVENT.wait(REFRESH_SECONDS)
    finally:
        _safe_clear_display(temp_display, "left")
        _safe_clear_display(time_display, "right")
        try:
            bus.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
