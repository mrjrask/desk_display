#!/usr/bin/env python3
"""
utils.py

Core utilities for OLED display project:
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
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import functools
import logging
import math
import requests
import spidev
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

# OLED driver
from waveshare_OLED.OLED_1in5_rgb import OLED_1in5_rgb
# Project config
from config import WIDTH, HEIGHT, CENTRAL_TIME, SPI_FREQUENCY
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
    """
    Wrapper around Waveshare SSD1351 for 1.5" RGB OLED.
    """
    def __init__(self):
        spi = spidev.SpiDev(0, 0)
        self.disp = OLED_1in5_rgb(spi=spi, spi_freq=SPI_FREQUENCY)
        if self.disp.Init() == -1:
            raise RuntimeError("Failed to initialize SSD1351")
        self.width, self.height = self.disp.width, self.disp.height

    def clear(self):
        self.disp.clear()

    def image(self, pil_img: Image.Image):
        buf = self.disp.getbuffer(pil_img)
        self.disp.ShowImage(buf)

    def show(self):
        # No-op: ShowImage pushes the buffer
        pass


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
    """

    image: Image.Image
    displayed: bool = False

# ─── Basic utilities ────────────────────────────────────────────────────────
@log_call
def clear_display(display):
    """
    Clear the OLED display.
    """
    try:
        display.clear()
    except Exception:
        try:
            display.fill(0)
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
def animate_fade_in(display: Display, new_image: Image.Image, steps=10, delay=0.02):
    """
    Fade an image into view over `steps` frames.
    """
    base = Image.new("RGB", new_image.size, (0,0,0))
    target = new_image.convert("RGB")
    for i in range(steps + 1):
        alpha = i / steps
        frame = Image.blend(base, target, alpha)
        display.image(frame)
        time.sleep(delay)

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

    for x in range(start, end, step):
        frame = Image.new(frame_mode, (w, h), background_color)
        if has_alpha:
            frame.paste(image, (x, y), image)
            frame_to_show = frame.convert("RGB")
        else:
            frame.paste(image, (x, y))
            frame_to_show = frame
        display.image(frame_to_show)
        time.sleep(0.01)

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

# ─── MLB Abbreviations ────────────────────────────────────────────────────────
MLB_ABBREVIATIONS = {
    "Chicago Cubs": "CUBS", "Atlanta Braves": "ATL", "Miami Marlins": "MIA",
    # ... other teams ...
}

def get_mlb_abbreviation(team_name: str) -> str:
    return MLB_ABBREVIATIONS.get(team_name, team_name)


def next_game_from_schedule(schedule: List[Dict[str, Any]], today: Optional[datetime.date] = None) -> Optional[Dict[str, Any]]:
    today = today or datetime.date.today()
    year = today.year
    upcoming: List[tuple[datetime.date, Dict[str, Any]]] = []
    for entry in schedule:
        if entry.get("opponent") == "—" or str(entry.get("time", "")).upper() == "TBD":
            continue
        try:
            parsed = datetime.datetime.strptime(entry.get("date", ""), "%a, %b %d")
            game_date = datetime.date(year, parsed.month, parsed.day)
        except Exception:
            continue
        if game_date >= today:
            upcoming.append((game_date, entry))
    if not upcoming:
        return None
    return sorted(upcoming, key=lambda item: item[0])[0][1]


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


def load_team_logo(base_dir: str, abbr: str, height: int = 36) -> Image.Image | None:
    filename = f"{abbr}.png"
    path = os.path.join(base_dir, filename)
    try:
        logo = Image.open(path).convert("RGBA")
        logo = _adjust_logo_brightness(logo, base_dir, abbr)
        ratio = height / logo.height
        return logo.resize((int(logo.width * ratio), height), Image.ANTIALIAS)
    except Exception as exc:
        logging.warning("Could not load logo '%s': %s", filename, exc)
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

# ─── GitHub Update Checker ─────────────────────────────────────────────────
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
        )
    except Exception:
        logging.info("check_github_updates: not a git repository, skipping check")
        return False

    # Local branch name (skip detached HEADs)
    try:
        local_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
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
        ).decode().strip()
    except Exception:
        logging.exception("check_github_updates: failed to read local HEAD")
        return False

    # Upstream branch for the current branch
    try:
        upstream_ref = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
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
        )
    except Exception:
        logging.warning("check_github_updates: failed to fetch from origin")
        return False

    # Remote SHA for the upstream branch
    try:
        remote_sha = subprocess.check_output(
            ["git", "rev-parse", upstream_ref],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        logging.warning(
            "check_github_updates: failed to resolve upstream %s for %s",
            upstream_ref,
            local_branch,
        )
        return False

    updated = (local_sha != remote_sha)
    logging.info(f"check_github_updates: updates available = {updated}")

    # If updated, log which files changed
    if updated:
        try:
            changed = subprocess.check_output(
                ["git", "diff", "--name-only", f"{local_sha}..{remote_sha}"],
                cwd=repo_dir,
            ).decode().splitlines()

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
        except Exception:
            logging.exception("check_github_updates: failed to list changed files")

    return updated

MLB_ABBREVIATIONS = {
    "Chicago Cubs": "CUBS", "Atlanta Braves": "ATL",  "Miami Marlins": "MIA",
    "New York Mets": "NYM", "Philadelphia Phillies": "PHI","Washington Nationals": "WAS",
    "Cincinnati Reds": "CIN","Milwaukee Brewers": "MIL", "Pittsburgh Pirates": "PIT",
    "St. Louis Cardinals": "STL","Arizona Diamondbacks": "ARI","Colorado Rockies": "COL",
    "Los Angeles Dodgers": "LAD","San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Baltimore Orioles": "BAL","Boston Red Sox": "BOS",  "New York Yankees": "NYY",
    "Tampa Bay Rays": "TB", "Toronto Blue Jays": "TOR", "Chicago White Sox": "SOX",
    "Cleveland Guardians": "CLE","Detroit Tigers": "DET", "Kansas City Royals": "KC",
    "Minnesota Twins": "MIN","Houston Astros": "HOU",    "Los Angeles Angels": "LAA",
    "Athletics": "ATH",     "Seattle Mariners": "SEA","Texas Rangers": "TEX",
}

def get_mlb_abbreviation(team_name: str) -> str:
    return MLB_ABBREVIATIONS.get(team_name, team_name)

# ─── Weather helpers ──────────────────────────────────────────────────────────
@log_call
def fetch_weather_icon(icon_code: str, size: int) -> Image.Image | None:
    if not icon_code:
        return None
    try:
        url = f"https://openweathermap.org/img/wn/{icon_code}@2x.png"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        icon = Image.open(BytesIO(response.content)).convert("RGBA")
        return icon.resize((size, size), Image.ANTIALIAS)
    except Exception as exc:  # pragma: no cover - network failures are non-fatal
        logging.warning("Weather icon fetch failed: %s", exc)
        return None


def uv_index_color(uvi: int) -> tuple[int, int, int]:
    if uvi < 2:
        return (0, 255, 0)
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
    if avoid_highways:
        params["avoid"] = "highways"

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
