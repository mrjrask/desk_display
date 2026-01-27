#!/usr/bin/env python3
"""
utils.py

Core utilities for the desk display project:
- Display wrapper
- Drawing helpers
- Animations
- Text wrapping/centering
- Team/MLB helpers
- GitHub update checker
"""
import datetime
import html
import os
import random
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import functools
import logging
import math
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

# ─── Pillow compatibility shim ─────────────────────────────────────────────
# Re-add ImageDraw.textsize if missing (Pillow ≥10 compatibility)
import PIL.ImageDraw as _ID
if not hasattr(_ID.ImageDraw, "textsize"):
    def _textsize(self, text, font=None, *args, **kwargs):
        bbox = self.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    _ID.ImageDraw.textsize = _textsize
# Compatibility for ANTIALIAS (Pillow ≥11)
try:
    Image.ANTIALIAS
except AttributeError:
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# Display HAT Mini driver (optional at import time)
try:  # pragma: no cover - hardware import
    from displayhatmini import DisplayHATMini  # type: ignore
except (ImportError, RuntimeError) as _displayhat_exc:  # pragma: no cover - hardware import
    DisplayHATMini = None  # type: ignore
    _DISPLAY_HAT_ERROR = _displayhat_exc
else:  # pragma: no cover - hardware import
    _DISPLAY_HAT_ERROR = None

_FORCE_HEADLESS = os.environ.get("DESK_DISPLAY_FORCE_HEADLESS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_ACTIVE_DISPLAY: Optional["Display"] = None
_LED_INDICATOR_ANIMATOR: Optional["_LedAnimator"] = None


@dataclass
class _UpdateStatus:
    github: bool = False
    apt: bool = False


_UPDATE_STATUS = _UpdateStatus()

_DISPLAY_UPDATE_GATE = threading.Event()
_DISPLAY_UPDATE_GATE.set()
_DEFER_CLEAR_DISPLAY = threading.Event()


def get_update_status() -> _UpdateStatus:
    """Return the last known update status for GitHub and apt."""

    return _UPDATE_STATUS


def suspend_display_updates() -> None:
    """Prevent subsequent display updates from reaching the hardware."""

    _DISPLAY_UPDATE_GATE.clear()


def resume_display_updates() -> None:
    """Allow display updates to be pushed to the hardware again."""

    _DISPLAY_UPDATE_GATE.set()


def display_updates_enabled() -> bool:
    """Return True when display updates are currently allowed."""

    return _DISPLAY_UPDATE_GATE.is_set()


@contextmanager
def defer_clear_display() -> Iterable[None]:
    """Temporarily suppress immediate clears to avoid black flashes."""

    _DEFER_CLEAR_DISPLAY.set()
    try:
        yield
    finally:
        _DEFER_CLEAR_DISPLAY.clear()

# Use the dimmest still-visible LED brightness.
LED_INDICATOR_LEVEL = 1 / 1024.0

# Project config
from config import WIDTH, HEIGHT, CENTRAL_TIME, DISPLAY_ROTATION
# Color utilities
from screens.color_palettes import random_color
# Colored logging
from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

# ─── Logging decorator ──────────────────────────────────────────────────────
def log_call(func):
    """
    Decorator that logs entry & exit at DEBUG level only.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(f"→ {func.__name__}()")
        result = func(*args, **kwargs)
        logging.debug(f"← {func.__name__}()")
        return result
    return wrapper

# ─── Display wrapper ────────────────────────────────────────────────────────
class Display:
    """Wrapper around the Pimoroni Display HAT Mini (320×240 LCD)."""

    _BUTTON_NAMES = ("A", "B", "X", "Y")

    def __init__(self):
        global _ACTIVE_DISPLAY

        self.width = WIDTH
        self.height = HEIGHT
        self.rotation = DISPLAY_ROTATION % 360
        if self.rotation not in (0, 90, 180, 270):
            logging.warning(
                "Unsupported display rotation %d°; falling back to 0°.",
                self.rotation,
            )
            self.rotation = 0
        self._buffer = Image.new("RGB", (self.width, self.height), "black")
        self._display = None
        self._button_pins: Dict[str, Optional[int]] = {name: None for name in self._BUTTON_NAMES}
        self._button_callback: Optional[Callable[[str], None]] = None
        self._backlight_level = 1.0
        self._backlight_lock = threading.Lock()
        self._skip_event: Optional[threading.Event] = None
        self._frame_id = 0
        self._frame_lock = threading.Lock()

        if _FORCE_HEADLESS:
            logging.info(
                "Display initialization skipped; running headless via DESK_DISPLAY_FORCE_HEADLESS."
            )
        elif DisplayHATMini is None:  # pragma: no cover - hardware import
            if _DISPLAY_HAT_ERROR:
                logging.warning(
                    "Display HAT Mini driver unavailable; running headless (%s)",
                    _DISPLAY_HAT_ERROR,
                )
            else:
                logging.warning(
                    "Display HAT Mini driver unavailable; running headless."
                )
        else:
            try:  # pragma: no cover - hardware import
                self._display = DisplayHATMini(self._buffer)
                for name in self._BUTTON_NAMES:
                    pin_name = f"BUTTON_{name}"
                    self._button_pins[name] = getattr(self._display, pin_name, None)
                if hasattr(self._display, "on_button_pressed"):
                    try:
                        self._display.on_button_pressed(self._handle_hw_button_event)
                    except Exception as exc:  # pragma: no cover - hardware import
                        logging.debug("Failed to register hardware button callback: %s", exc)
            except Exception as exc:  # pragma: no cover - hardware import
                logging.warning(
                    "Failed to initialize Display HAT Mini hardware; running headless (%s)",
                    exc,
                )
                self._display = None
            else:  # pragma: no cover - hardware import
                logging.info(
                    "🖼️  Display HAT Mini initialized (%dx%d, rotation %d°).",
                    self.width,
                    self.height,
                    self.rotation,
                )

        _ACTIVE_DISPLAY = self

    def register_skip_event(self, event: Optional[threading.Event]) -> None:
        """Associate a skip event so long-running screens can bail out early."""

        self._skip_event = event

    def skip_requested(self) -> bool:
        """Return True when a registered skip event is active."""

        return bool(self._skip_event and self._skip_event.is_set())

    def wait_for_skip(self, timeout: float, *, poll_interval: float = 0.05) -> bool:
        """Sleep up to *timeout* seconds, returning True if a skip is requested."""

        if not self._skip_event:
            time.sleep(timeout)
            return False

        end = time.monotonic() + timeout
        while True:
            if self._skip_event.is_set():
                return True

            remaining = end - time.monotonic()
            if remaining <= 0:
                break

            time.sleep(min(poll_interval, remaining))

        return False

    def _update_display(self):
        if not display_updates_enabled():
            return
        if self._display is None:  # pragma: no cover - hardware import
            return
        try:
            buffer_to_display = self._buffer
            if self.rotation:
                buffer_to_display = self._buffer.rotate(self.rotation, expand=False)
            self._display.buffer = buffer_to_display
            self._display.display()
        except Exception as exc:  # pragma: no cover - hardware import
            logging.warning("Display refresh failed: %s", exc)

    def _bump_frame_id(self) -> None:
        with self._frame_lock:
            self._frame_id += 1

    def clear(self):
        self._buffer = Image.new("RGB", (self.width, self.height), "black")
        self._bump_frame_id()
        self._update_display()

    def image(self, pil_img: Image.Image):
        if pil_img.size != (self.width, self.height):
            pil_img = pil_img.resize((self.width, self.height), Image.ANTIALIAS)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        self._buffer = pil_img.copy()
        self._bump_frame_id()
        self._update_display()

    def show(self):
        # No additional action required; display() is triggered during image()
        self._update_display()

    def capture(self) -> Image.Image:
        """Return a copy of the currently buffered frame."""

        return self._buffer.copy()

    def frame_id(self) -> int:
        """Return the current frame identifier."""

        with self._frame_lock:
            return self._frame_id

    # ----- Hardware helpers -------------------------------------------------
    def set_backlight(self, level: float) -> float:
        """Set the LCD backlight brightness (0.0 – 1.0)."""

        # Clamp the requested level to keep the screen visible but not blinding.
        # Allow 0.0 for explicit "off" requests (used by display toggle).
        level = max(0.0, min(1.0, level))

        with self._backlight_lock:
            self._backlight_level = level

            if self._display is None:  # pragma: no cover - hardware import
                return self._backlight_level

            try:  # pragma: no cover - hardware import
                self._display.set_backlight(self._backlight_level)
            except Exception as exc:  # pragma: no cover - hardware import
                logging.debug("Failed to set backlight level: %s", exc)

        return self._backlight_level

    def adjust_backlight(self, delta: float) -> float:
        """Adjust the backlight brightness by *delta* (0.0 – 1.0)."""

        with self._backlight_lock:
            new_level = self._backlight_level + delta

        return self.set_backlight(new_level)

    def backlight_level(self) -> float:
        """Return the current backlight level."""

        with self._backlight_lock:
            return self._backlight_level

    def set_led(self, r: float = 0.0, g: float = 0.0, b: float = 0.0) -> None:
        """Set the onboard RGB LED, if hardware is available."""

        if self._display is None:  # pragma: no cover - hardware import
            return
        try:  # pragma: no cover - hardware import
            self._display.set_led(r=r, g=g, b=b)
        except Exception as exc:  # pragma: no cover - hardware import
            logging.debug("Display LED update failed: %s", exc)

    def is_button_pressed(self, name: str) -> bool:
        """Return True if the named button is currently pressed."""

        if self._display is None:  # pragma: no cover - hardware import
            return False

        pin = self._button_pins.get(name.upper())
        if pin is None:  # pragma: no cover - hardware import
            return False

        try:  # pragma: no cover - hardware import
            raw_state = self._display.read_button(pin)
        except Exception as exc:  # pragma: no cover - hardware import
            logging.debug("Display button read failed (%s): %s", name, exc)
            return False

        if isinstance(raw_state, bool):  # pragma: no cover - hardware import
            return raw_state

        if isinstance(raw_state, (int, float)):  # pragma: no cover - hardware import
            # Buttons are wired active-low; a ``0`` reading means the button is
            # being held down.  ``read_button`` previously returned ``True``
            # when pressed but newer firmware returns the raw ``0/1`` GPIO
            # value.  Treat both styles uniformly so the skip button works
            # regardless of driver version.
            return raw_state == 0

        return bool(raw_state)

    def set_button_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Register a callable invoked when a hardware button is pressed."""

        self._button_callback = callback

    def _handle_hw_button_event(self, pin) -> None:  # pragma: no cover - hardware import
        name = None
        for button_name, button_pin in self._button_pins.items():
            if button_pin == pin:
                name = button_name
                break

        if not name or self._display is None:
            return

        try:
            state = self._display.read_button(pin)
        except Exception as exc:
            logging.debug("Hardware button callback read failed: %s", exc)
            return

        if isinstance(state, bool):
            pressed = state
        elif isinstance(state, (int, float)):
            pressed = state == 0
        else:
            pressed = bool(state)

        if not pressed:
            return

        callback = self._button_callback
        if callback is None:
            return

        try:
            callback(name)
        except Exception as exc:
            logging.debug("Button callback raised %s", exc)


def get_active_display() -> Optional["Display"]:
    """Return the most recently constructed :class:`Display` instance, if any."""

    return _ACTIVE_DISPLAY


@dataclass
class ScreenImage:
    """Container for a rendered screen image.

    Attributes
    ----------
    image:
        The full PIL image representing the screen.
    displayed:
        Whether the image has already been pushed to the display by the
        originating function. This allows callers to skip redundant redraws
        while still accessing the image data (e.g., for screenshots).
    led_override:
        Optional RGB tuple describing an LED color override that should remain
        active while the image is shown.
    """

    image: Image.Image
    displayed: bool = False
    led_override: Optional[Tuple[float, float, float]] = None

# ─── Basic utilities ────────────────────────────────────────────────────────
@log_call
def clear_display(display):
    """
    Clear the connected display, falling back to a blank frame.
    """
    if _DEFER_CLEAR_DISPLAY.is_set():
        try:
            if hasattr(display, "_buffer"):
                display._buffer = Image.new(
                    "RGB",
                    (getattr(display, "width", WIDTH), getattr(display, "height", HEIGHT)),
                    "black",
                )
            return
        except Exception:
            return
    try:
        display.clear()
    except Exception:
        try:
            blank = Image.new("RGB", (getattr(display, "width", WIDTH), getattr(display, "height", HEIGHT)), "black")
            display.image(blank)
            display.show()
        except Exception:
            pass

@log_call
def draw_text_centered(
    draw: ImageDraw.Draw,
    text: str,
    font: ImageFont.FreeTypeFont,
    y_offset: int = 0,
    width: int = WIDTH,
    height: int = HEIGHT,
    *,
    fill=(255,255,255)
):
    """
    Draw `text` centered horizontally at vertical center + y_offset.
    """
    w, h = draw.textsize(text, font=font)
    x = (width - w) // 2
    y = (height - h) // 2 + y_offset
    draw.text((x, y), text, font=font, fill=fill)

@log_call
def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int):
    """
    Break `text` into lines so each line fits within max_width.
    """
    words = text.split()
    if not words:
        return []
    dummy = Image.new("RGB", (max_width, 1))
    draw = ImageDraw.Draw(dummy)
    lines = [words[0]]
    for w in words[1:]:
        test = f"{lines[-1]} {w}"
        if draw.textsize(test, font=font)[0] <= max_width:
            lines[-1] = test
        else:
            lines.append(w)
    return lines


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    try:
        return draw.textsize(text, font=font)
    except Exception:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top


def clone_font(font: ImageFont.FreeTypeFont, size: int) -> ImageFont.FreeTypeFont:
    path = getattr(font, "path", None)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return font


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    base_font: ImageFont.FreeTypeFont,
    max_width: int,
    max_height: int,
    *,
    min_pt: int = 8,
    max_pt: int | None = None,
) -> ImageFont.FreeTypeFont:
    base_size = getattr(base_font, "size", 16)
    hi = max_pt if max_pt else base_size
    lo = min_pt
    best = clone_font(base_font, lo)
    while lo <= hi:
        mid = (lo + hi) // 2
        test_font = clone_font(base_font, mid)
        width, height = measure_text(draw, text, test_font)
        if width <= max_width and height <= max_height:
            best = test_font
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def format_voc_ohms(value) -> str:
    if value is None:
        return "N/A"
    try:
        val = float(value)
    except Exception:
        return "N/A"
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f} MΩ"
    if val >= 1_000:
        return f"{val / 1_000:.1f} kΩ"
    return f"{val:.0f} Ω"


def temperature_color(temp_f: float, lo: float = 50.0, hi: float = 80.0) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, (temp_f - lo) / (hi - lo + 1e-6)))
    if t < 0.5:
        alpha = t / 0.5
        r = int(0 + (80 - 0) * alpha)
        g = int(150 + (220 - 150) * alpha)
        b = int(255 + (180 - 255) * alpha)
    else:
        alpha = (t - 0.5) / 0.5
        r = int(80 + (255 - 80) * alpha)
        g = int(220 + (120 - 220) * alpha)
        b = int(180 + (0 - 180) * alpha)
    return (r, g, b)

@log_call
def animate_fade_in(
    display: Display,
    new_image: Image.Image,
    steps: int = 10,
    delay: float = 0.02,
    *,
    from_image: Image.Image | None = None,
):
    """
    Fade from the current display buffer (or ``from_image``) into ``new_image``.
    """

    if steps <= 0:
        display.image(new_image)
        return

    if from_image is None:
        try:
            base = display.capture()
        except AttributeError:
            base = None
        if base is None:
            base = Image.new("RGB", new_image.size, (0, 0, 0))
    else:
        base = from_image

    base = base.convert("RGB")
    if base.size != new_image.size:
        base = base.resize(new_image.size, Image.ANTIALIAS)

    target = new_image.convert("RGB")

    for i in range(steps + 1):
        frame_start = time.time()

        alpha = i / steps
        frame = Image.blend(base, target, alpha)
        display.image(frame)

        # Account for rendering time to maintain consistent frame rate
        elapsed = time.time() - frame_start
        sleep_time = max(0, delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

@log_call
def animate_scroll(display: Display, image: Image.Image, speed=3, y_offset=None):
    """
    Scroll an image across the display.
    """
    if image is None:
        return

    bands = image.getbands() if hasattr(image, "getbands") else ()
    has_alpha = "A" in bands
    image = image.convert("RGBA" if has_alpha else "RGB")

    w, h = display.width, display.height
    img_w, img_h = image.size
    y = y_offset if y_offset is not None else (h - img_h) // 2
    direction = random.choice(("ltr", "rtl"))
    start, end, step = ((-img_w, w, speed) if direction == "ltr" else (w, -img_w, -speed))

    background_color = (0, 0, 0, 0) if has_alpha else (0, 0, 0)
    frame_mode = "RGBA" if has_alpha else "RGB"

    target_frame_time = 0.016  # ~60 FPS for smoother animation
    for x in range(start, end + step, step):
        frame_start = time.time()

        frame = Image.new(frame_mode, (w, h), background_color)
        if has_alpha:
            frame.paste(image, (x, y), image)
            frame_to_show = frame.convert("RGB")
        else:
            frame.paste(image, (x, y))
            frame_to_show = frame
        display.image(frame_to_show)

        # Account for rendering time to maintain consistent frame rate
        elapsed = time.time() - frame_start
        sleep_time = max(0, target_frame_time - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Ensure the display is clear once the image has fully scrolled off-screen.
    final_frame = Image.new(frame_mode, (w, h), background_color)
    display.image(final_frame.convert("RGB") if has_alpha else final_frame)

# ─── Date & Time Helpers ─────────────────────────────────────────────────────
def parse_game_date(iso_date_str: str, time_str: str = "TBD") -> str:
    try:
        d = datetime.datetime.strptime(iso_date_str, "%Y-%m-%d").date()
    except Exception:
        return time_str
    today = datetime.datetime.now(CENTRAL_TIME).date()
    if d == today:
        day = "Today"
    elif d == today + datetime.timedelta(days=1):
        day = "Tomorrow"
    else:
        day = d.strftime("%a %-m/%-d")
    return f"{day} {time_str}" if time_str.upper() != "TBD" else f"{day} TBD"

def format_date_no_leading(dt_date: datetime.date) -> str:
    return f"{dt_date.month}/{dt_date.day}"

def format_time_no_leading(dt_time: datetime.time) -> str:
    return dt_time.strftime("%I:%M %p").lstrip("0")

def split_time_period(dt_time: datetime.time) -> tuple[str,str]:
    full = dt_time.strftime("%I:%M %p").lstrip("0")
    parts = full.rsplit(" ", 1)
    return (parts[0], parts[1]) if len(parts)==2 else (full, "")

# ─── Team & Standings Helpers ────────────────────────────────────────────────
def get_team_display_name(team) -> str:
    if not isinstance(team, dict):
        return str(team)
    t = team.get("team", team)
    for key in ("commonName","name","teamName","fullName","city"): 
        val = t.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return "UNK"

def get_opponent_last_game(team) -> str:
    if not isinstance(team, dict):
        return str(team)
    city = team.get("placeName", {}).get("default", "").strip()
    return city or get_team_display_name(team)

def extract_split_record(split_records: list, record_type: str) -> str:
    for sp in split_records:
        if sp.get("type", "").lower() == record_type.lower():
            w = sp.get("wins", "N/A")
            l = sp.get("losses", "N/A")
            p = sp.get("pct", "N/A")
            return f"{w}-{l} ({p})"
    return "N/A"

def wind_direction(degrees: float) -> str:
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    try:
        idx = int((degrees / 22.5) + 0.5) % 16
        return dirs[idx]
    except Exception:
        return ""

wind_deg_to_compass = wind_direction

def center_coords(
    img_size: tuple[int,int],
    content_size: tuple[int,int],
    y_offset: int = 0
) -> tuple[int,int]:
    w, h = img_size
    cw, ch = content_size
    return ((w - cw)//2, (h - ch)//2 + y_offset)

def _week_sort_value(week_label: str) -> float:
    """Return a numeric sort key for week labels.

    Supports values like ``"Week 16"`` or preseason fractions such as ``"0.2"``.
    Unknown or missing labels sort to the end of the schedule.
    """

    label = (week_label or "").strip().lower()
    if label.startswith("week"):
        try:
            return float(label.split()[1])
        except Exception:
            return float("inf")
    try:
        return float(label)
    except Exception:
        return float("inf")


def _game_sort_value(entry: Dict[str, Any]) -> float:
    """Return the numeric sort order for a schedule entry.

    Prefers ``game_no`` (to support decimal preseason numbering) and falls back to
    ``week`` labels.
    """

    try:
        game_no = entry.get("game_no")
    except AttributeError:  # pragma: no cover - defensive
        game_no = None
    if game_no is not None:
        try:
            return float(str(game_no))
        except Exception:
            pass

    return _week_sort_value(str(entry.get("week", "")))


def _parse_game_date(
    date_text: str, *, default_year: int
) -> Optional[datetime.date]:
    if not date_text:
        return None
    date_text = str(date_text).strip()
    if not date_text or date_text.upper() in {"TBD", "BYE"}:
        return None
    for fmt in ("%a, %b %d %Y", "%a, %b %d, %Y"):
        try:
            parsed = datetime.datetime.strptime(date_text, fmt)
            return parsed.date()
        except Exception:
            continue
    try:
        parsed = datetime.datetime.strptime(date_text, "%a, %b %d")
        return datetime.date(default_year, parsed.month, parsed.day)
    except Exception:
        return None


def next_game_from_schedule(
    schedule: List[Dict[str, Any]], today: Optional[datetime.date] = None
) -> Optional[Dict[str, Any]]:
    today = today or datetime.date.today()
    year = today.year

    candidates: List[tuple[Optional[datetime.date], float, int, Dict[str, Any]]] = []
    for idx, entry in enumerate(schedule):
        if entry.get("opponent") == "—":
            continue

        parsed_date = _parse_game_date(entry.get("date", ""), default_year=year)

        sort_value = _game_sort_value(entry)
        candidates.append((parsed_date, sort_value, idx, entry))

    if not candidates:
        return None

    future_dated = [
        (parsed_date, sort_value, idx, entry)
        for parsed_date, sort_value, idx, entry in candidates
        if parsed_date is not None and parsed_date >= today
    ]

    if future_dated:
        parsed_date, _, _, entry = min(
            future_dated, key=lambda item: (item[0], item[1], item[2])
        )
        return entry

    dated_past = [
        (parsed_date, sort_value, idx, entry)
        for parsed_date, sort_value, idx, entry in candidates
        if parsed_date is not None and parsed_date <= today
    ]
    if dated_past:
        _, last_sort_value, _, _ = max(
            dated_past, key=lambda item: (item[0], item[1], item[2])
        )
        higher_games = [
            (parsed_date, sort_value, idx, entry)
            for parsed_date, sort_value, idx, entry in candidates
            if sort_value > last_sort_value
        ]
        if higher_games:
            _, _, _, entry = min(
                higher_games,
                key=lambda item: (item[1], item[0] or datetime.date.max, item[2]),
            )
            return entry

    _, _, _, entry = min(
        candidates, key=lambda item: (item[1], item[0] or datetime.date.max, item[2])
    )
    return entry


_LOGO_BRIGHTNESS_OVERRIDES: dict[tuple[str, str], float] = {
    ("nhl", "WAS"): 1.35,
    ("nhl", "TBL"): 1.35,
    ("nhl", "TB"): 1.35,
    ("nfl", "NYJ"): 1.4,
    ("mlb", "SD"): 1.35,
    ("mlb", "DET"): 1.35,
    ("mlb", "NYY"): 1.35,
}


def _adjust_logo_brightness(logo: Image.Image, base_dir: str, abbr: str) -> Image.Image:
    sport = os.path.basename(os.path.normpath(base_dir or ""))
    key = (sport.lower(), (abbr or "").upper())
    factor = _LOGO_BRIGHTNESS_OVERRIDES.get(key)
    if not factor:
        return logo
    return ImageEnhance.Brightness(logo).enhance(factor)


def standard_next_game_logo_height(panel_height: int) -> int:
    """Return the shared next-game logo height used across team screens."""
    if panel_height >= 128:
        return 150
    if panel_height >= 96:
        return 109
    return 89


def standard_next_game_logo_height_for_space(
    panel_height: int,
    available_height: int,
    *,
    scale: float = 1.0,
) -> int:
    """Return a shared next-game logo height constrained by available space."""
    clamped_scale = max(0.5, min(float(scale or 1.0), 1.2))
    desired = max(1, int(round(standard_next_game_logo_height(panel_height) * clamped_scale)))
    return max(1, min(desired, max(1, available_height)))


def standard_scoreboard_team_logo_height(panel_height: int, *, compact: bool = False) -> int:
    """Return the shared scoreboard team logo height for the given panel size."""
    if compact:
        return 26
    return 52


def standard_scoreboard_league_logo_height(team_logo_height: int) -> int:
    """Return the shared scoreboard league logo height for the given team logo height."""
    return int(round(max(1, team_logo_height) * 1.25))


def standard_next_game_logo_frame_width(
    logo_height: int, logos: Iterable[Image.Image | None] = ()
) -> int:
    """Width to reserve for each logo on next-game screens.

    The returned width ensures that both logos share the same frame size, avoiding
    "crowding" around the centered "@" regardless of each logo's aspect ratio.
    """

    # Slightly wider than tall to give horizontally oriented marks breathing room.
    min_width = int(round(max(1, logo_height) * 1.1))
    max_logo_width = max((logo.width for logo in logos if logo), default=0)
    return max(min_width, max_logo_width)


def fit_logo_to_box(logo: Image.Image | None, box_size: int) -> Image.Image | None:
    """Resize a logo to fit within a square box, preserving aspect ratio."""
    if logo is None or box_size <= 0:
        return logo
    width, height = logo.size
    if width <= 0 or height <= 0:
        return logo
    scale = min(box_size / float(width), box_size / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if new_width == width and new_height == height:
        return logo
    return logo.resize((new_width, new_height), Image.LANCZOS)


def load_team_logo(
    base_dir: str,
    abbr: str,
    height: int = 36,
    *,
    box_size: int | None = None,
    trim: bool = False,
) -> Image.Image | None:
    cleaned = (abbr or "").strip()
    if not cleaned:
        return None
    candidates: list[str] = []
    for candidate in (cleaned, cleaned.upper(), cleaned.lower()):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    last_error: Optional[Exception] = None
    for candidate in candidates:
        filename = f"{candidate}.png"
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            logo = Image.open(path).convert("RGBA")
            logo = _adjust_logo_brightness(logo, base_dir, candidate)
            if trim:
                alpha = logo.split()[-1]
                bbox = alpha.getbbox()
                if bbox:
                    logo = logo.crop(bbox)
            if box_size and box_size > 0:
                return fit_logo_to_box(logo, box_size)
            ratio = height / logo.height
            return logo.resize((int(logo.width * ratio), height), Image.ANTIALIAS)
        except Exception as exc:  # pragma: no cover - rare file corruption
            last_error = exc
            continue
    if last_error:
        logging.warning("Could not load logo '%s': %s", abbr, last_error)
    return None

@log_call
def colored_image(mono_img: Image.Image, screen_key: str) -> Image.Image:
    rgb = Image.new("RGB", mono_img.size, (0,0,0))
    pix = mono_img.load()
    draw = ImageDraw.Draw(rgb)
    col = random_color(screen_key)
    for y in range(mono_img.height):
        for x in range(mono_img.width):
            if pix[x, y]:
                draw.point((x, y), fill=col)
    return rgb

@log_call
def load_svg(key, url) -> Image.Image | None:
    cache_dir = os.path.join(os.path.dirname(__file__), "images", "nhl")
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, f"{key}.svg")
    if not os.path.exists(local):
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            with open(local, "wb") as f:
                f.write(r.content)
        except Exception as e:
            logging.warning(f"Failed to download NHL logo: {e}")
            return None
    try:
        from cairosvg import svg2png
        png = svg2png(url=local)
        return Image.open(BytesIO(png))
    except Exception:
        return None

# ─── Update Indicator LED ───────────────────────────────────────────────────
class _LedAnimator:
    """Animate the onboard LED using a cycle of colors."""

    def __init__(
        self,
        display: "Display",
        pattern: Tuple[Tuple[float, float, float], ...],
        interval: float = 0.6,
    ) -> None:
        self._display = display
        self._pattern = pattern
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def is_running_for(
        self,
        display: "Display",
        pattern: Tuple[Tuple[float, float, float], ...],
        interval: float,
    ) -> bool:
        return (
            self._display is display
            and self._thread.is_alive()
            and self._pattern == pattern
            and abs(self._interval - interval) < 1e-6
        )

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._display.set_led(r=0.0, g=0.0, b=0.0)

    def _run(self) -> None:
        idx = 0
        while not self._stop.is_set():
            r, g, b = self._pattern[idx]
            self._display.set_led(r=r, g=g, b=b)
            idx = (idx + 1) % len(self._pattern)
            if self._stop.wait(self._interval):
                break
        self._display.set_led(r=0.0, g=0.0, b=0.0)


def _led_pattern(status: _UpdateStatus) -> Tuple[Tuple[Tuple[float, float, float], ...], float] | tuple[None, None]:
    blue = (0.0, 0.0, LED_INDICATOR_LEVEL)
    yellow = (LED_INDICATOR_LEVEL, LED_INDICATOR_LEVEL, 0.0)

    if status.apt:
        # Alternate blue/yellow when apt updates are pending.
        return ((blue, yellow), 0.6)
    if status.github:
        return ((blue,), 0.8)
    return (None, None)


def _refresh_led_indicator(display: Optional["Display"] = None) -> None:
    """Reflect update status on the Display HAT Mini LED."""

    global _LED_INDICATOR_ANIMATOR

    display = display or get_active_display()
    status = _UPDATE_STATUS

    if display is None:
        if _LED_INDICATOR_ANIMATOR is not None and not (status.github or status.apt):
            try:  # pragma: no cover - hardware import
                _LED_INDICATOR_ANIMATOR.stop()
            except Exception as exc:
                logging.debug("Failed to stop LED animator without display: %s", exc)
            finally:
                _LED_INDICATOR_ANIMATOR = None
        return

    pattern, interval = _led_pattern(status)

    if pattern is None:
        if _LED_INDICATOR_ANIMATOR is not None:
            try:  # pragma: no cover - hardware import
                _LED_INDICATOR_ANIMATOR.stop()
            except Exception as exc:
                logging.debug("Failed to stop LED animator: %s", exc)
            finally:
                _LED_INDICATOR_ANIMATOR = None
        else:
            try:
                display.set_led(r=0.0, g=0.0, b=0.0)
            except Exception as exc:  # pragma: no cover - hardware import
                logging.debug("Failed to clear LED: %s", exc)
        return

    if _LED_INDICATOR_ANIMATOR is not None:
        if _LED_INDICATOR_ANIMATOR.is_running_for(display, pattern, interval):
            return
        try:  # pragma: no cover - hardware import
            _LED_INDICATOR_ANIMATOR.stop()
        except Exception as exc:
            logging.debug("Failed to stop existing LED animator: %s", exc)
    animator = _LedAnimator(display, pattern, interval)
    _LED_INDICATOR_ANIMATOR = animator
    try:  # pragma: no cover - hardware import
        animator.start()
    except Exception as exc:
        logging.debug("Failed to start LED animator: %s", exc)
        _LED_INDICATOR_ANIMATOR = None


def _set_update_status(github: Optional[bool] = None, apt: Optional[bool] = None) -> None:
    global _UPDATE_STATUS

    if github is None and apt is None:
        return

    status = _UPDATE_STATUS
    _UPDATE_STATUS = _UpdateStatus(
        github=status.github if github is None else github,
        apt=status.apt if apt is None else apt,
    )
    _refresh_led_indicator()


def clear_update_indicator(display: Optional["Display"] = None) -> None:
    """Stop the update LED animation and turn the LED off."""

    global _LED_INDICATOR_ANIMATOR, _UPDATE_STATUS

    _UPDATE_STATUS = _UpdateStatus()
    display = display or get_active_display()

    if _LED_INDICATOR_ANIMATOR is not None:
        try:  # pragma: no cover - hardware import
            _LED_INDICATOR_ANIMATOR.stop()
        except Exception as exc:
            logging.debug("Failed to stop LED animator during cleanup: %s", exc)
        finally:
            _LED_INDICATOR_ANIMATOR = None

    if display is None:
        return

    try:
        display.set_led(r=0.0, g=0.0, b=0.0)
    except Exception as exc:  # pragma: no cover - hardware import
        logging.debug("Failed to clear LED during cleanup: %s", exc)


@contextmanager
def temporary_display_led(r: float, g: float, b: float):
    """Temporarily override the display LED, restoring update status after."""

    global _LED_INDICATOR_ANIMATOR

    display = get_active_display()
    if display is None:
        yield
        return

    pattern, interval = _led_pattern(_UPDATE_STATUS)

    animator = _LED_INDICATOR_ANIMATOR
    if animator is not None and (pattern is None or interval is None or not animator.is_running_for(display, pattern, interval)):
        animator = None

    update_led_active = bool(_UPDATE_STATUS.github or _UPDATE_STATUS.apt)
    if animator is not None:
        update_led_active = True

    if animator is not None:
        try:  # pragma: no cover - hardware import
            animator.stop()
        except Exception as exc:
            logging.debug("Failed to stop LED animator before override: %s", exc)
        finally:
            _LED_INDICATOR_ANIMATOR = None

    try:
        display.set_led(r=r, g=g, b=b)
        yield
    finally:
        if update_led_active or _UPDATE_STATUS.github or _UPDATE_STATUS.apt:
            _refresh_led_indicator(display)
        else:
            try:
                display.set_led(r=0.0, g=0.0, b=0.0)
            except Exception as exc:
                logging.debug("Failed to reset LED after override: %s", exc)

_GIT_COMMAND_TIMEOUT = 10
_APT_COMMAND_TIMEOUT = 20


def check_github_updates() -> bool:
    """
    Return True if the local branch differs from its upstream tracking branch.
    Also logs the list of files that have changed on the remote.

    Safe fallbacks:
      - Handles non-git directories gracefully.
      - Skips detached HEADs or branches without an upstream.
      - Silently returns False if remote can't be fetched.
    """
    repo_dir = os.path.dirname(__file__)

    # Is this a git repo?
    try:
        subprocess.check_call(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        logging.warning("check_github_updates: git probe timed out")
        return False
    except Exception:
        logging.info("check_github_updates: not a git repository, skipping check")
        return False

    # Local branch name (skip detached HEADs)
    try:
        local_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_COMMAND_TIMEOUT,
        ).decode().strip()
    except subprocess.TimeoutExpired:
        logging.warning("check_github_updates: git branch lookup timed out")
        return False
    except Exception:
        logging.exception("check_github_updates: failed to determine local branch")
        return False

    if local_branch in {"HEAD", ""}:
        logging.info("check_github_updates: detached HEAD, skipping check")
        return False

    # Local SHA
    try:
        local_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_COMMAND_TIMEOUT,
        ).decode().strip()
    except subprocess.TimeoutExpired:
        logging.warning("check_github_updates: git HEAD lookup timed out")
        return False
    except Exception:
        logging.exception("check_github_updates: failed to read local HEAD")
        return False

    # Upstream branch for the current branch
    try:
        upstream_ref = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_COMMAND_TIMEOUT,
        ).decode().strip()
    except subprocess.TimeoutExpired:
        logging.warning("check_github_updates: git upstream lookup timed out")
        return False
    except Exception:
        logging.info(
            "check_github_updates: no upstream tracking branch for %s, skipping check",
            local_branch,
        )
        return False

    # Fetch remote so we can diff against it
    try:
        subprocess.check_call(
            ["git", "fetch", "--quiet", "origin"],
            cwd=repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        logging.warning("check_github_updates: git fetch timed out")
        return False
    except Exception:
        logging.warning("check_github_updates: failed to fetch from origin")
        return False

    # Remote SHA for the upstream branch
    try:
        remote_sha = subprocess.check_output(
            ["git", "rev-parse", upstream_ref],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_COMMAND_TIMEOUT,
        ).decode().strip()
    except subprocess.TimeoutExpired:
        logging.warning("check_github_updates: git upstream SHA lookup timed out")
        return False
    except Exception:
        logging.warning(
            "check_github_updates: failed to resolve upstream %s for %s",
            upstream_ref,
            local_branch,
        )
        return False

    updated = (local_sha != remote_sha)
    logging.info(f"check_github_updates: updates available = {updated}")
    _set_update_status(github=updated)

    # If updated, log which files changed
    if updated:
        try:
            changed = subprocess.check_output(
                ["git", "diff", "--name-only", f"{local_sha}..{remote_sha}"],
                cwd=repo_dir,
                timeout=_GIT_COMMAND_TIMEOUT,
            ).decode().splitlines()
        except subprocess.TimeoutExpired:
            logging.warning("check_github_updates: git diff timed out")
            return updated
        except Exception:
            logging.exception("check_github_updates: failed to list changed files")
        else:
            if not changed:
                logging.info("check_github_updates: no file list available (empty diff?)")
            else:
                # Keep the log readable if there are many files
                MAX_LIST = 100
                shown = changed[:MAX_LIST]
                logging.info(
                    f"check_github_updates: {len(changed)} file(s) differ from {upstream_ref}:"
                )
                for p in shown:
                    logging.info(f"  • {p}")
                if len(changed) > MAX_LIST:
                    logging.info(f"  …and {len(changed) - MAX_LIST} more")

    return updated


_APT_CACHE_TTL_SECONDS = 4 * 60 * 60
_APT_CACHE_RESULT: Optional[bool] = None
_APT_CACHE_AT: float = 0.0


def check_apt_updates() -> bool:
    """Return True if `apt` has upgradeable packages (cached for four hours)."""

    global _APT_CACHE_RESULT, _APT_CACHE_AT

    now = time.time()
    if _APT_CACHE_RESULT is not None and (now - _APT_CACHE_AT) < _APT_CACHE_TTL_SECONDS:
        logging.info(
            "check_apt_updates: using cached result (%s)",
            "updates" if _APT_CACHE_RESULT else "no updates",
        )
        _set_update_status(apt=_APT_CACHE_RESULT)
        return _APT_CACHE_RESULT

    try:
        proc = subprocess.run(
            ["apt-get", "-s", "-o", "Debug::NoLocking=1", "upgrade"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_APT_COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        logging.warning("check_apt_updates: apt-get simulation timed out")
        _set_update_status(apt=False)
        return False
    except Exception:
        logging.exception("check_apt_updates: failed to run apt-get simulation")
        _set_update_status(apt=False)
        return False

    if proc.returncode != 0:
        logging.warning("check_apt_updates: apt-get exited with %s", proc.returncode)
        _set_update_status(apt=False)
        return False

    updates_available = any(line.startswith("Inst ") for line in proc.stdout.splitlines())
    logging.info("check_apt_updates: updates available = %s", updates_available)

    _APT_CACHE_RESULT = updates_available
    _APT_CACHE_AT = now

    _set_update_status(apt=updates_available)
    return updates_available

MLB_ABBREVIATIONS = {
    # National League
    "Arizona Diamondbacks": "ARI",
    "Diamondbacks": "ARI",
    "D-backs": "ARI",
    "Atlanta Braves": "ATL",
    "Braves": "ATL",
    "Chicago Cubs": "CUBS",
    "Cubs": "CUBS",
    "Cincinnati Reds": "CIN",
    "Reds": "CIN",
    "Colorado Rockies": "COL",
    "Rockies": "COL",
    "Los Angeles Dodgers": "LAD",
    "Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Brewers": "MIL",
    "New York Mets": "NYM",
    "Mets": "NYM",
    "Philadelphia Phillies": "PHI",
    "Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "Pirates": "PIT",
    "San Diego Padres": "SD",
    "Padres": "SD",
    "San Francisco Giants": "SF",
    "Giants": "SF",
    "St. Louis Cardinals": "STL",
    "Cardinals": "STL",
    "Washington Nationals": "WSH",
    "Nationals": "WSH",

    # American League
    "Baltimore Orioles": "BAL",
    "Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Red Sox": "BOS",
    "Chicago White Sox": "SOX",
    "White Sox": "SOX",
    "Cleveland Guardians": "CLE",
    "Guardians": "CLE",
    "Detroit Tigers": "DET",
    "Tigers": "DET",
    "Houston Astros": "HOU",
    "Astros": "HOU",
    "Kansas City Royals": "KC",
    "Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Angels": "LAA",
    "Minnesota Twins": "MIN",
    "Twins": "MIN",
    "New York Yankees": "NYY",
    "Yankees": "NYY",
    "Oakland Athletics": "ATH",
    "Seattle Mariners": "SEA",
    "Mariners": "SEA",
    "Tampa Bay Rays": "TB",
    "Rays": "TB",
    "Texas Rangers": "TEX",
    "Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Blue Jays": "TOR",
    "Las Vegas Athletics": "ATH",
    "Athletics": "ATH",
}

MLB_LOGO_OVERRIDES = {
    "CHC": "CUBS",
    "CWS": "SOX",
    "OAK": "ATH",
    "ATHLETICS": "ATH",
    "RED SOX": "BOS",
    "REDSOX": "BOS",
    "WHITE SOX": "SOX",
    "WHITESOX": "SOX",
}


def _normalize_mlb_tricode(abbr: str | None) -> str:
    if not isinstance(abbr, str):
        return ""
    normalized = abbr.strip().upper()
    return MLB_LOGO_OVERRIDES.get(normalized, normalized)


def get_mlb_abbreviation(team_name: str) -> str:
    abbr = MLB_ABBREVIATIONS.get(team_name)
    if isinstance(abbr, str):
        return _normalize_mlb_tricode(abbr)
    return str(team_name)


def get_mlb_tricode(team: dict | str | None) -> str:
    if isinstance(team, dict):
        for key in (
            "triCode",
            "tricode",
            "teamTricode",
            "abbreviation",
            "teamCode",
            "fileCode",
        ):
            val = team.get(key)
            normalized = _normalize_mlb_tricode(val)
            if normalized:
                return normalized
        team_name = team.get("name")
        if isinstance(team_name, str):
            return get_mlb_abbreviation(team_name).upper()
        return ""
    if isinstance(team, str):
        return get_mlb_abbreviation(team).upper()
    return ""

# ─── Weather helpers ──────────────────────────────────────────────────────────
def _draw_cloud(draw: ImageDraw.ImageDraw, center: tuple[float, float], radius: float, color: tuple[int, int, int]):
    cx, cy = center
    for dx, dy, scale in [(-radius * 0.8, 0, 1), (0, -radius * 0.5, 1.1), (radius * 0.8, 0, 1)]:
        r = radius * scale
        draw.ellipse(
            (cx + dx - r, cy + dy - r, cx + dx + r, cy + dy + r),
            fill=color,
        )
    draw.rectangle((cx - radius * 1.6, cy, cx + radius * 1.6, cy + radius * 1.1), fill=color)


def _render_sun(size: int) -> Image.Image:
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    center = size / 2
    radius = size * 0.24
    draw.ellipse((center - radius, center - radius, center + radius, center + radius), fill=(255, 204, 0, 255))
    for i in range(12):
        angle = math.radians(i * 30)
        x0 = center + math.cos(angle) * radius * 1.5
        y0 = center + math.sin(angle) * radius * 1.5
        x1 = center + math.cos(angle) * radius * 2.1
        y1 = center + math.sin(angle) * radius * 2.1
        draw.line((x0, y0, x1, y1), fill=(255, 215, 0, 255), width=max(2, size // 20))
    return icon


def _render_cloudy(size: int, with_sun: bool = False) -> Image.Image:
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    center = (size * 0.5, size * 0.55)
    radius = size * 0.18
    if with_sun:
        sun = _render_sun(size)
        icon.alpha_composite(sun, (int(size * 0.05), int(size * 0.05)))
    _draw_cloud(draw, center, radius, (220, 220, 220, 255))
    return icon


def _render_precip(size: int, kind: str) -> Image.Image:
    icon = _render_cloudy(size)
    draw = ImageDraw.Draw(icon)
    base_y = int(size * 0.65)
    spacing = size * 0.12
    start_x = size * 0.32
    color = (100, 170, 255, 255)
    for idx in range(3):
        x = start_x + idx * spacing
        if kind == "snow":
            arm = size * 0.05
            draw.line((x, base_y, x, base_y + size * 0.18), fill=color, width=max(1, size // 30))
            draw.line((x - arm, base_y + size * 0.08, x + arm, base_y + size * 0.1), fill=color, width=max(1, size // 30))
            draw.line((x - arm, base_y + size * 0.12, x + arm, base_y + size * 0.14), fill=color, width=max(1, size // 30))
        elif kind == "sleet":
            draw.line((x, base_y, x, base_y + size * 0.18), fill=color, width=max(2, size // 18))
            draw.text((x - size * 0.04, base_y + size * 0.16), "•", font=ImageFont.load_default(), fill=color)
        else:
            draw.line((x, base_y, x, base_y + size * 0.2), fill=color, width=max(2, size // 18))
    if kind == "storm":
        bolt = [
            (size * 0.62, base_y - size * 0.05),
            (size * 0.55, base_y + size * 0.15),
            (size * 0.66, base_y + size * 0.12),
            (size * 0.6, base_y + size * 0.35),
            (size * 0.74, base_y + size * 0.12),
            (size * 0.64, base_y + size * 0.12),
        ]
        draw.polygon(bolt, fill=(255, 204, 0, 255))
    return icon


def _render_fog(size: int) -> Image.Image:
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    y = size * 0.35
    for _ in range(4):
        draw.rounded_rectangle((size * 0.18, y, size * 0.82, y + size * 0.08), radius=size * 0.04, fill=(200, 200, 200, 200))
        y += size * 0.12
    return icon


def _render_wind(size: int) -> Image.Image:
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    y = size * 0.35
    for idx in range(3):
        draw.arc((size * 0.18, y - size * 0.05, size * 0.9, y + size * 0.25), start=200, end=350, fill=(180, 220, 255, 255), width=max(2, size // 24))
        y += size * 0.18
    return icon


ICON_RENDERERS = {
    "sunny": _render_sun,
    "partly-cloudy": lambda size: _render_cloudy(size, with_sun=True),
    "cloudy": _render_cloudy,
    "rain": lambda size: _render_precip(size, "rain"),
    "snow": lambda size: _render_precip(size, "snow"),
    "sleet": lambda size: _render_precip(size, "sleet"),
    "storm": lambda size: _render_precip(size, "storm"),
    "fog": _render_fog,
    "wind": _render_wind,
}


@log_call
def fetch_weather_icon(icon_code: str, size: int) -> Image.Image | None:
    if not icon_code:
        return None

    icon_lookup = str(icon_code).strip()
    if not icon_lookup:
        return None

    alias_map = {
        "sunny": "Clear",
        "partly-cloudy": "PartlyCloudy",
        "cloudy": "Cloudy",
        "rain": "Rain",
        "snow": "Snow",
        "sleet": "Sleet",
        "storm": "Thunderstorms",
        "fog": "Fog",
        "wind": "Windy",
    }
    icon_name = alias_map.get(icon_lookup.lower(), icon_lookup)

    icon_dir = Path(__file__).resolve().parent / "images" / "WeatherKit"
    candidates = [icon_dir / f"{icon_name}.png"]
    if icon_name != "Cloudy":
        candidates.append(icon_dir / "Cloudy.png")

    for candidate in candidates:
        try:
            if candidate.is_file():
                icon = Image.open(candidate).convert("RGBA")
                if icon.size != (size, size):
                    icon = icon.resize((size, size), Image.ANTIALIAS)
                return icon
        except Exception as exc:  # pragma: no cover - drawing failures are non-fatal
            logging.warning("Weather icon load failed for %s: %s", candidate, exc)

    logging.warning("Weather icon %s not found; returning None", icon_name)
    return None


def uv_index_color(uvi: int) -> tuple[int, int, int]:
    if uvi <= 1:
        return (0, 255, 0)
    if uvi == 2:
        return (200, 120, 255)
    if 3 <= uvi <= 5:
        return (255, 255, 0)
    if 6 <= uvi <= 7:
        return (255, 165, 0)
    if 8 <= uvi <= 10:
        return (255, 0, 0)
    return (128, 0, 128)


def timestamp_to_datetime(value, tz) -> datetime.datetime | None:
    try:
        return datetime.datetime.fromtimestamp(value, tz)
    except Exception:
        return None


def bright_color(min_luma: int = 160) -> tuple[int, int, int]:
    for _ in range(20):
        r = random.randint(80, 255)
        g = random.randint(80, 255)
        b = random.randint(80, 255)
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if luma >= min_luma:
            return (r, g, b)
    return (255, 255, 255)


_GH_ICON_CACHE: dict[tuple[int, bool, tuple[str, ...]], Image.Image | None] = {}


def load_github_icon(size: int, invert: bool, paths: list[str]) -> Image.Image | None:
    key = (size, bool(invert), tuple(paths))
    if key in _GH_ICON_CACHE:
        return _GH_ICON_CACHE[key]

    path = next((p for p in paths if os.path.exists(p)), None)
    if not path:
        _GH_ICON_CACHE[key] = None
        return None

    try:
        icon = Image.open(path).convert("RGBA")
        if icon.height != size:
            ratio = size / float(icon.height)
            icon = icon.resize((max(1, int(round(icon.width * ratio))), size), Image.ANTIALIAS)

        if invert:
            r, g, b, a = icon.split()
            rgb_inv = ImageOps.invert(Image.merge("RGB", (r, g, b)))
            icon = Image.merge("RGBA", (*rgb_inv.split(), a))

        _GH_ICON_CACHE[key] = icon
        return icon
    except Exception:
        _GH_ICON_CACHE[key] = None
        return None


def time_strings(now: datetime.datetime) -> tuple[str, str]:
    time_str = now.strftime("%-I:%M")
    am_pm = now.strftime("%p")
    if time_str.startswith("0"):
        time_str = time_str[1:]
    return time_str, am_pm


def date_strings(now: datetime.datetime) -> tuple[str, str]:
    weekday = now.strftime("%A")
    return weekday, f"{now.strftime('%B')} {now.day}, {now.year}"


def decode_html(text: str) -> str:
    try:
        return html.unescape(text)
    except Exception:
        return text


def fetch_directions_routes(
    origin: str,
    destination: str,
    api_key: str,
    *,
    avoid_highways: bool = False,
    avoid_tolls: bool = False,
    url: str,
) -> List[Dict[str, Any]]:
    if not api_key:
        logging.warning("Travel: no GOOGLE_MAPS_API_KEY configured.")
        return []

    params = {
        "origin": origin,
        "destination": destination,
        "alternatives": "true",
        "departure_time": "now",
        "traffic_model": "best_guess",
        "region": "us",
        "key": api_key,
    }
    avoid = []
    if avoid_highways:
        avoid.append("highways")
    if avoid_tolls:
        avoid.append("tolls")
    if avoid:
        params["avoid"] = "|".join(avoid)

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logging.warning("Directions request failed: %s", exc)
        return []

    if payload.get("status") != "OK":
        logging.warning(
            "Directions status=%s, error_message=%s",
            payload.get("status"),
            payload.get("error_message"),
        )
        return []

    routes = payload.get("routes", []) or []
    for route in routes:
        leg = (route.get("legs") or [{}])[0]
        route["_summary"] = decode_html(route.get("summary", "")).lower()
        duration = leg.get("duration_in_traffic") or leg.get("duration") or {}
        route["_duration_text"] = duration.get("text", "")
        route["_duration_sec"] = duration.get("value", 0)
        steps = leg.get("steps", []) or []
        fragments = []
        for step in steps:
            instruction = decode_html(step.get("html_instructions", "")).lower()
            fragments.append(instruction)
        route["_steps_text"] = " ".join(fragments)
    return routes


def route_contains(route: Dict[str, Any], token: str) -> bool:
    token = token.lower()
    return token in route.get("_summary", "") or token in route.get("_steps_text", "")


def choose_route_by_token(routes: List[Dict[str, Any]], token: str) -> Optional[Dict[str, Any]]:
    for route in routes:
        if route_contains(route, token):
            return route
    return None


def choose_route_by_any(routes: List[Dict[str, Any]], tokens: List[str]) -> Optional[Dict[str, Any]]:
    for token in tokens:
        match = choose_route_by_token(routes, token)
        if match:
            return match
    return None


def format_duration_text(route: Optional[Dict[str, Any]]) -> str:
    if not route:
        return "N/A"
    text = route.get("_duration_text") or ""
    return text if text else "N/A"


def fastest_route(routes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not routes:
        return None
    return min(routes, key=lambda r: r.get("_duration_sec", math.inf))
