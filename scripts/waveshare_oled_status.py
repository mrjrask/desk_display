#!/usr/bin/env python3
"""Render simple status content on the Waveshare OLED/LCD HAT (A) OLED displays.

Default behavior:
- Left OLED (0x3c): current temperature (CPU temperature by default)
- Right OLED (0x3d): local time in 12-hour format, no leading zero
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

from smbus import SMBus


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
TEMP_SOURCE = os.getenv("WAVESHARE_OLED_TEMP_SOURCE", "cpu").strip().lower()
TEMP_COMMAND = os.getenv("WAVESHARE_OLED_TEMP_COMMAND", "")
TEMP_UNIT = os.getenv("WAVESHARE_OLED_TEMP_UNIT", "C").strip().upper()
REFRESH_SECONDS = max(1, _env_int("WAVESHARE_OLED_REFRESH_SECONDS", 5))


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


def read_temperature() -> str:
    value_c: float | None = None

    if TEMP_SOURCE == "command" and TEMP_COMMAND:
        try:
            output = subprocess.check_output(TEMP_COMMAND, shell=True, text=True, timeout=4)
            value_c = _parse_temperature_value(output)
        except Exception:
            value_c = None
    else:
        value_c = _read_cpu_temp_c()

    if value_c is None:
        return "--°"

    if TEMP_UNIT == "F":
        return f"{(value_c * 9 / 5) + 32:.1f}°F"

    return f"{value_c:.1f}°C"


def current_time_12h() -> str:
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def render_centered_text(width: int, height: int, text: str, *, title: str | None = None) -> Image.Image:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    value_font = ImageFont.load_default()

    y_offset = 0
    if title:
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_w) // 2, 2), title, font=title_font, fill=255)
        y_offset = title_h + 6

    value_bbox = draw.textbbox((0, 0), text, font=value_font)
    value_w = value_bbox[2] - value_bbox[0]
    value_h = value_bbox[3] - value_bbox[1]
    draw.text(((width - value_w) // 2, max(y_offset, (height - value_h) // 2)), text, font=value_font, fill=255)
    return image


def main() -> int:
    bus = SMBus(I2C_BUS)
    temp_display = SSD1306Display(bus, TEMP_ADDR, OLED_WIDTH, OLED_HEIGHT)
    time_display = SSD1306Display(bus, TIME_ADDR, OLED_WIDTH, OLED_HEIGHT)

    temp_display.initialize()
    time_display.initialize()
    temp_display.clear()
    time_display.clear()

    while True:
        temp_text = read_temperature()
        time_text = current_time_12h()

        temp_image = render_centered_text(OLED_WIDTH, OLED_HEIGHT, temp_text, title="Temp")
        time_image = render_centered_text(OLED_WIDTH, OLED_HEIGHT, time_text, title="Time")

        temp_display.display_image(temp_image)
        time_display.display_image(time_image)

        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
