#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
draw_sensors.py — clean 2×2 sensor dashboard for Pimoroni Multi-Sensor Stick.
- Optimized for 320×240 canvases (Display HAT Mini).
- Shows: Ambient light (lux), Proximity, Motion force (|accel| g), Rotation (gyro Z °/s).
- Holds the screen for 12 seconds and updates ~4×/sec.
- If ctx.present_frame(img) exists, pushes live frames during the hold.
- Otherwise returns the last frame and requests a 12s duration if supported.
"""

from __future__ import annotations

import math
import time
import logging
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from config import get_screen_background_color
from utils import ScreenImage

# Optional sensor modules: only used if available.
try:
    import ltr559  # Ambient light + proximity
except Exception:
    ltr559 = None  # type: ignore

# IMU: try lsm6dsox first (Pimoroni’s library), fall back to lsm6ds3 if present.
_IMU = None
try:
    import lsm6dsox as _IMU  # type: ignore
except Exception:
    try:
        import lsm6ds3 as _IMU  # type: ignore
    except Exception:
        _IMU = None  # type: ignore


# ---------- Layout ----------
W, H = 320, 240
PADDING = 8
GAP = 8
CARD_RADIUS = 18

# Colors
BG = (10, 16, 24)
FG = (230, 236, 244)
SUB = (172, 182, 196)
STAMP = (140, 160, 180)

CARD_OUTLINES = [
    (210, 180, 110, 255),  # amber
    (110, 150, 210, 255),  # blue
    (160, 120, 200, 255),  # purple
    (100, 170, 160, 255),  # teal
]

TITLE = "Sensor Stick"
SUBTITLE = "Pimoroni Multi-Sensor (LTR559 • LSM6DS3)"

# ---------- Fonts ----------
def _try_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()

FONT_TITLE = _try_font("DejaVuSans.ttf", 22)
FONT_SUBTITLE = _try_font("DejaVuSans.ttf", 12)
FONT_CARD_LABEL = _try_font("DejaVuSans.ttf", 13)
FONT_CARD_VALUE = _try_font("DejaVuSansMono.ttf", 22)
FONT_STAMP = _try_font("DejaVuSans.ttf", 11)

# ---------- Sensor wrappers ----------
class LTR559Reader:
    def __init__(self):
        self.ok = False
        self.dev = None
        if ltr559:
            try:
                self.dev = ltr559.LTR559()
                _ = self.dev.get_lux()  # wake some units
                self.ok = True
            except Exception as e:
                logging.warning(f"draw_sensors: LTR559 init failed: {e}")

    def sample(self) -> Tuple[Optional[float], Optional[int]]:
        if not self.ok:
            return None, None
        lux = prox = None
        try:
            lux = float(self.dev.get_lux())
        except Exception:
            lux = None
        try:
            prox = int(self.dev.get_proximity())
        except Exception:
            prox = None
        return lux, prox


class IMUReader:
    """Read accel magnitude (g) and gyro Z (deg/s) if an LSM6D* is available."""
    def __init__(self):
        self.ok = False
        self.dev = None
        if _IMU:
            try:
                ctor = getattr(_IMU, "LSM6DSOX", None) or getattr(_IMU, "LSM6DS3", None) or getattr(_IMU, "IMU", None)
                self.dev = ctor() if callable(ctor) else _IMU
                self.ok = True
            except Exception as e:
                logging.warning(f"draw_sensors: IMU init failed: {e}")

    def sample(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.ok or self.dev is None:
            return None, None

        ax = ay = az = gz = None
        try:
            if hasattr(self.dev, "read_accelerometer"):
                ax, ay, az = self.dev.read_accelerometer()
            elif hasattr(self.dev, "acceleration"):
                ax, ay, az = self.dev.acceleration  # type: ignore
        except Exception:
            ax = ay = az = None

        try:
            if hasattr(self.dev, "read_gyroscope"):
                gx, gy, gz = self.dev.read_gyroscope()
            elif hasattr(self.dev, "gyroscope"):
                gx, gy, gz = self.dev.gyroscope  # type: ignore
        except Exception:
            gz = None

        accel_mag = None
        if ax is not None and ay is not None and az is not None:
            try:
                accel_mag = math.sqrt(ax * ax + ay * ay + az * az)
            except Exception:
                accel_mag = None

        return accel_mag, gz


# ---------- Drawing helpers ----------
def round_rect(draw: ImageDraw.ImageDraw, box, radius, outline=None, width=2, fill=None):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=width)

def label(draw, xy, text, font, fill):
    draw.text(xy, text, font=font, fill=fill)

def center_text(draw, box, text, font, fill, dy=0):
    x0, y0, x1, y1 = box
    w, h = draw.textbbox((0, 0), text, font=font)[2:]
    cx = (x0 + x1 - w) // 2
    cy = (y0 + y1 - h) // 2 + dy
    draw.text((cx, cy), text, font=font, fill=fill)

def layout_cards():
    inner_w = W - 2 * PADDING
    inner_h = H - 2 * PADDING - 48  # room for titles
    col = (inner_w - GAP) // 2
    row = (inner_h - GAP) // 2
    x0 = PADDING
    y0 = PADDING + 48
    c1 = (x0,             y0,             x0 + col,       y0 + row)
    c2 = (x0 + col + GAP, y0,             x0 + 2*col+GAP, y0 + row)
    c3 = (x0,             y0 + row + GAP, x0 + col,       y0 + 2*row+GAP)
    c4 = (x0 + col + GAP, y0 + row + GAP, x0 + 2*col+GAP, y0 + 2*row+GAP)
    return c1, c2, c3, c4

def _fmt(value: Optional[float], fmt: str = "{:.1f}") -> str:
    return "—" if value is None else fmt.format(value)

def render_frame(light_lux: Optional[float],
                 prox: Optional[int],
                 accel_g: Optional[float],
                 rot_z: Optional[float],
                 now_ts: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Title + subtitle
    label(d, (PADDING, 8), TITLE, FONT_TITLE, FG)
    label(d, (PADDING, 8 + 24), SUBTITLE, FONT_SUBTITLE, SUB)

    c1, c2, c3, c4 = layout_cards()

    # Card 1: Ambient light
    round_rect(d, c1, CARD_RADIUS, outline=CARD_OUTLINES[0], width=3)
    label(d, (c1[0]+12, c1[1]+10), "Ambient light", FONT_CARD_LABEL, SUB)
    center_text(d, c1, _fmt(light_lux, "{:.1f}") + " lx", FONT_CARD_VALUE, FG, dy=8)
    # Guidance text removed for a cleaner presentation

    # Card 2: Proximity
    round_rect(d, c2, CARD_RADIUS, outline=CARD_OUTLINES[1], width=3)
    label(d, (c2[0]+12, c2[1]+10), "Proximity", FONT_CARD_LABEL, SUB)
    prox_str = "—" if prox is None else str(prox)
    center_text(d, c2, prox_str, FONT_CARD_VALUE, FG, dy=8)

    # Card 3: Motion force (accel magnitude)
    round_rect(d, c3, CARD_RADIUS, outline=CARD_OUTLINES[2], width=3)
    label(d, (c3[0]+12, c3[1]+10), "Motion force", FONT_CARD_LABEL, SUB)
    center_text(d, c3, _fmt(accel_g, "{:.2f}") + " g", FONT_CARD_VALUE, FG, dy=8)

    # Card 4: Rotation (gyro Z)
    round_rect(d, c4, CARD_RADIUS, outline=CARD_OUTLINES[3], width=3)
    label(d, (c4[0]+12, c4[1]+10), "Rotation", FONT_CARD_LABEL, SUB)
    center_text(d, c4, _fmt(rot_z, "{:.1f}") + " °/s", FONT_CARD_VALUE, FG, dy=8)

    # Stamp
    stamp = time.strftime("Updated %I:%M:%S %p", time.localtime(now_ts)).lstrip("0")
    tw, th = d.textbbox((0, 0), stamp, font=FONT_STAMP)[2:]
    d.text((W - PADDING - tw, H - PADDING - th), stamp, font=FONT_STAMP, fill=STAMP)

    return img


# ---------- Public API for screen runner ----------
def _extract_display(context):
    display = getattr(context, "display", None)
    if display is None and hasattr(context, "image"):
        candidate = getattr(context, "image")
        if callable(candidate):
            display = context
    return display


def draw(context, **kwargs) -> Optional[Image.Image]:
    """
    Main entrypoint called by the screen registry.
    Keeps this screen up for 12 seconds, updating ~4×/sec.
    Accepts and ignores extra kwargs (e.g., transition, duration) for compatibility.
    If `context.present_frame(img)` exists, pushes live frames.
    Otherwise returns the last frame and requests a 12s duration if supported.
    """
    global BG
    BG = get_screen_background_color("sensors", BG)
    duration_s = 12.0
    interval_s = 0.25  # ~4 Hz

    set_duration = getattr(context, "set_duration", None)
    if callable(set_duration):
        try:
            set_duration(int(duration_s))
        except Exception:
            pass

    light = LTR559Reader()
    imu = IMUReader()

    t_end = time.time() + duration_s
    last_img: Optional[Image.Image] = None

    display = _extract_display(context)
    present_frame = getattr(context, "present_frame", None)

    while True:
        now = time.time()
        if now >= t_end:
            break

        lux, prox = light.sample() if light.ok else (None, None)
        accel_g, rot_z = imu.sample() if imu.ok else (None, None)

        img = render_frame(lux, prox, accel_g, rot_z, now)
        last_img = img

        if callable(present_frame):
            try:
                present_frame(img)
            except Exception:
                pass
        elif display is not None:
            try:
                display.image(img)
                if hasattr(display, "show"):
                    display.show()
            except Exception:
                pass

        remaining = t_end - time.time()
        if remaining <= 0:
            break
        time.sleep(min(interval_s, max(remaining, 0)))

    if callable(present_frame):
        return None
    if display is not None and last_img is not None:
        return ScreenImage(last_img, displayed=True)
    return last_img


# Back-compat for registries importing `draw_sensors`
def draw_sensors(context, **kwargs) -> Optional[Image.Image]:
    return draw(context, **kwargs)


__all__ = ["draw", "draw_sensors"]
