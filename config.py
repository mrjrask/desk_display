# config.py

#!/usr/bin/env python3
import datetime
import glob
import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ─── Environment helpers ───────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_env_file(path: str) -> None:
    """Load simple KEY=VALUE pairs from *path* without overriding existing vars."""

    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return
    except OSError:
        logging.debug("Could not read .env file at %s", path)
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def _initialise_env() -> None:
    """Load environment variables from `.env` if present."""

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        load_dotenv = None

    candidate_paths = []

    project_root = Path(SCRIPT_DIR)
    candidate_paths.append(project_root / ".env")

    cwd_path = Path.cwd() / ".env"
    if cwd_path != candidate_paths[0]:
        candidate_paths.append(cwd_path)

    for path in candidate_paths:
        if not path.is_file():
            continue
        if load_dotenv is not None:
            load_dotenv(path, override=False)
        else:
            _load_env_file(str(path))


_ENV_INITIALISED = False


def initialise_env_if_requested(force: bool = False) -> None:
    """Conditionally load `.env` files based on CONFIG_LOAD_DOTENV flag."""

    global _ENV_INITIALISED

    if _ENV_INITIALISED and not force:
        return

    raw_flag = os.environ.get("CONFIG_LOAD_DOTENV", "0").strip().lower()
    should_load = raw_flag in {"1", "true", "yes", "on"}

    if should_load:
        _initialise_env()

    _ENV_INITIALISED = True


initialise_env_if_requested()


def _get_first_env_var(*names: str):
    """Return the first populated environment variable from *names.*"""

    for name in names:
        value = os.environ.get(name)
        if value:
            return value

    return None


def _get_bool_env(name: str, default: bool) -> bool:
    """Parse boolean feature flags from environment variables."""

    raw = os.environ.get(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _get_required_env_var(*names: str) -> str:
    value = _get_first_env_var(*names)
    if value:
        return value

    joined = ", ".join(names)
    raise RuntimeError(
        "Missing required environment variable. Set one of: "
        f"{joined}"
    )

import pytz
from PIL import Image, ImageDraw, ImageFont

from config_store import ConfigStore

try:
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # Pillow >= 9.1
except AttributeError:  # pragma: no cover - fallback for older Pillow
    _RESAMPLE_LANCZOS = Image.LANCZOS

# ─── Project paths ────────────────────────────────────────────────────────────
IMAGES_DIR  = os.path.join(SCRIPT_DIR, "images")

STYLE_CONFIG_PATH = os.environ.get(
    "SCREENS_STYLE_PATH", os.path.join(SCRIPT_DIR, "screens_style.json")
)
_STYLE_CONFIG_STORE = ConfigStore(STYLE_CONFIG_PATH)
_STYLE_CONFIG_CACHE: Dict[str, Any] = {"screens": {}}
_STYLE_CONFIG_MTIME: Optional[float] = None
_STYLE_CONFIG_LOCK = threading.Lock()

# ─── Feature flags ────────────────────────────────────────────────────────────
ENABLE_SCREENSHOTS   = _get_bool_env("ENABLE_SCREENSHOTS", True)
ENABLE_VIDEO         = _get_bool_env("ENABLE_VIDEO", False)
VIDEO_FPS            = 30
ENABLE_WIFI_MONITOR  = _get_bool_env("ENABLE_WIFI_MONITOR", True)
ENABLE_WIFI_RECOVERY = _get_bool_env("ENABLE_WIFI_RECOVERY", True)
WIFI_TCP_PROBE_URLS  = os.environ.get("WIFI_TCP_PROBE_URLS", "")
WIFI_TCP_PROBE_HOSTS = os.environ.get("WIFI_TCP_PROBE_HOSTS", "")
WIFI_TCP_PROBE_PORT  = os.environ.get("WIFI_TCP_PROBE_PORT", "443")
RPI_CONNECT_CONTROL_HOST = os.environ.get("RPI_CONNECT_CONTROL_HOST")

WIFI_RETRY_DURATION  = 180
WIFI_CHECK_INTERVAL  = 60
WIFI_OFF_DURATION    = 180

VRNO_CACHE_TTL       = 1800

def get_current_ssid():
    try:
        return subprocess.check_output(
            ["iwgetid", "-r"],
            timeout=3,
        ).decode("utf-8").strip()
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

CURRENT_SSID = get_current_ssid()

if CURRENT_SSID == "Verano":
    ENABLE_WEATHER = True
    LATITUDE       = 41.9103
    LONGITUDE      = -87.6340
    TRAVEL_MODE    = "to_home"
elif CURRENT_SSID == "wiffy":
    ENABLE_WEATHER = True
    LATITUDE       = 42.13444
    LONGITUDE      = -87.876389
    TRAVEL_MODE    = "to_work"
else:
    ENABLE_WEATHER = True
    LATITUDE       = 41.9103
    LONGITUDE      = -87.6340
    TRAVEL_MODE    = "to_home"

WEATHERKIT_TEAM_ID     = os.environ.get("WEATHERKIT_TEAM_ID")
WEATHERKIT_KEY_ID      = os.environ.get("WEATHERKIT_KEY_ID")
WEATHERKIT_SERVICE_ID  = os.environ.get("WEATHERKIT_SERVICE_ID")
WEATHERKIT_KEY_PATH    = os.environ.get("WEATHERKIT_KEY_PATH")
WEATHERKIT_PRIVATE_KEY = os.environ.get("WEATHERKIT_PRIVATE_KEY")
WEATHERKIT_LANGUAGE    = os.environ.get("WEATHERKIT_LANGUAGE", "en")
WEATHERKIT_TIMEZONE    = os.environ.get("WEATHERKIT_TIMEZONE", "America/Chicago")

def _get_owm_api_key(ssid: Optional[str]) -> Optional[str]:
    """Select the OpenWeatherMap API key based on the connected SSID."""

    if ssid in {"wiffy", "wiffyToo"}:
        return _get_first_env_var("OWM_API_KEY_WIFFY", "OWM_API_KEY")

    if ssid == "Verano":
        return _get_first_env_var("OWM_API_KEY_VERANO", "OWM_API_KEY_DEFAULT")

    return _get_first_env_var("OWM_API_KEY_DEFAULT", "OWM_API_KEY")


OWM_API_KEY = _get_owm_api_key(CURRENT_SSID)
OWM_API_URL   = "https://api.openweathermap.org/data/3.0/onecall"
OWM_UNITS     = os.environ.get("OWM_UNITS", "imperial")
OWM_LANGUAGE  = os.environ.get("OWM_LANGUAGE", "en")

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
MAPKIT_TOKEN = os.environ.get("MAPKIT_TOKEN")
APPLE_MAPS_API_KEY = os.environ.get("APPLE_MAPS_API_KEY") or MAPKIT_TOKEN
APPLE_MAPS_TEAM_ID = os.environ.get("APPLE_MAPS_TEAM_ID")
APPLE_MAPS_KEY_ID = os.environ.get("APPLE_MAPS_KEY_ID")
APPLE_MAPS_KEY_PATH = os.environ.get("APPLE_MAPS_KEY_PATH")
APPLE_MAPS_PRIVATE_KEY = os.environ.get("APPLE_MAPS_PRIVATE_KEY")
APPLE_MAPS_DIRECTIONS_URL = os.environ.get(
    "APPLE_MAPS_DIRECTIONS_URL",
    "https://maps-api.apple.com/v1/directions",
)
APPLE_MAPS_SNAPSHOT_URL = os.environ.get(
    "APPLE_MAPS_SNAPSHOT_URL",
    "https://maps-api.apple.com/v1/snapshot",
)

# ─── Display configuration ─────────────────────────────────────────────────────
WIDTH                    = 320
HEIGHT                   = 240
SCREEN_DELAY             = 4
try:
    HOURLY_FORECAST_HOURS = int(os.environ.get("HOURLY_FORECAST_HOURS", "5"))
    if HOURLY_FORECAST_HOURS < 1:
        HOURLY_FORECAST_HOURS = 1
except (TypeError, ValueError):
    logging.warning(
        "Invalid HOURLY_FORECAST_HOURS value; defaulting to 5 hours."
    )
    HOURLY_FORECAST_HOURS = 5
if HOURLY_FORECAST_HOURS > 12:
    HOURLY_FORECAST_HOURS = 12

try:
    WEATHER_REFRESH_SECONDS = int(os.environ.get("WEATHER_REFRESH_SECONDS", "1800"))
    if WEATHER_REFRESH_SECONDS < 600:
        logging.warning(
            "WEATHER_REFRESH_SECONDS too low; clamping to 600 seconds to limit API usage."
        )
        WEATHER_REFRESH_SECONDS = 600
except (TypeError, ValueError):
    logging.warning(
        "Invalid WEATHER_REFRESH_SECONDS value; defaulting to 1800 seconds."
    )
    WEATHER_REFRESH_SECONDS = 1800
try:
    TEAM_STANDINGS_DISPLAY_SECONDS = int(
        os.environ.get("TEAM_STANDINGS_DISPLAY_SECONDS", "5")
    )
except (TypeError, ValueError):
    logging.warning(
        "Invalid TEAM_STANDINGS_DISPLAY_SECONDS value; defaulting to 5 seconds."
    )
    TEAM_STANDINGS_DISPLAY_SECONDS = 5
SCHEDULE_UPDATE_INTERVAL = 600

try:
    DISPLAY_ROTATION = int(os.environ.get("DISPLAY_ROTATION", "180"))
except (TypeError, ValueError):
    logging.warning(
        "Invalid DISPLAY_ROTATION value; defaulting to 180 degrees."
    )
    DISPLAY_ROTATION = 180

# ─── Dark hours configuration ─────────────────────────────────────────────────

MINUTES_PER_DAY = 24 * 60

_DAY_NAME_TO_INDEX = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "weds": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


def _parse_time_token(token: str) -> int:
    cleaned = token.strip()
    if not cleaned:
        raise ValueError("Empty time token")

    lowered = cleaned.lower()
    if lowered in {"midnight"}:
        return 0
    if lowered in {"noon"}:
        return 12 * 60
    if lowered in {"24:00", "24", "24h", "24hr", "24hrs"}:
        return MINUTES_PER_DAY

    for fmt in ("%H:%M", "%H", "%I:%M%p", "%I%p", "%I:%M %p", "%I %p"):
        try:
            parsed = datetime.datetime.strptime(cleaned.upper(), fmt)
        except ValueError:
            continue
        return parsed.hour * 60 + parsed.minute

    raise ValueError(f"Unrecognized time token '{token}'")


def _expand_day_spec(spec: str) -> list[int]:
    days: list[int] = []
    seen = set()
    for part in spec.split(","):
        piece = part.strip()
        if not piece:
            continue

        if "-" in piece:
            start_text, end_text = piece.split("-", 1)
            start_name = start_text.strip().lower()
            end_name = end_text.strip().lower()
            if start_name not in _DAY_NAME_TO_INDEX or end_name not in _DAY_NAME_TO_INDEX:
                raise ValueError(f"Unknown day name in range '{piece}'")
            start_idx = _DAY_NAME_TO_INDEX[start_name]
            end_idx = _DAY_NAME_TO_INDEX[end_name]
            idx = start_idx
            while True:
                if idx not in seen:
                    days.append(idx)
                    seen.add(idx)
                if idx == end_idx:
                    break
                idx = (idx + 1) % 7
        else:
            name = piece.lower()
            if name not in _DAY_NAME_TO_INDEX:
                raise ValueError(f"Unknown day name '{piece}'")
            idx = _DAY_NAME_TO_INDEX[name]
            if idx not in seen:
                days.append(idx)
                seen.add(idx)
    return days


@dataclass(frozen=True)
class DarkHoursSegment:
    weekday: int
    start_minute: int
    end_minute: int


def _parse_dark_hours_spec(raw_value: Optional[str]) -> tuple[DarkHoursSegment, ...]:
    if not raw_value:
        return ()

    entries = []
    for chunk in re.split(r"[;\n]+", raw_value):
        if not chunk:
            continue

        normalized = re.sub(r"\s*-\s*", "-", chunk.strip())
        if not normalized:
            continue

        parts = normalized.split(None, 1)
        if len(parts) != 2:
            logging.warning("Ignoring dark-hours entry '%s' (missing time range)", chunk)
            continue

        day_spec, time_spec = parts[0], parts[1].strip()
        if not day_spec:
            logging.warning("Ignoring dark-hours entry '%s' (missing day spec)", chunk)
            continue

        if not time_spec:
            logging.warning("Ignoring dark-hours entry '%s' (missing time spec)", chunk)
            continue

        try:
            days = _expand_day_spec(day_spec)
        except ValueError as exc:
            logging.warning("Ignoring dark-hours entry '%s': %s", chunk, exc)
            continue

        if not days:
            logging.warning("Ignoring dark-hours entry '%s' (no valid days)", chunk)
            continue

        normalized_time = time_spec.lower().replace(" ", "")
        if normalized_time in {"off", "allday", "all-day", "alldaylong"}:
            start_minutes = 0
            end_minutes = MINUTES_PER_DAY
        else:
            if "-" not in time_spec:
                logging.warning(
                    "Ignoring dark-hours entry '%s' (missing start/end times)", chunk
                )
                continue
            start_text, end_text = time_spec.split("-", 1)
            try:
                start_minutes = _parse_time_token(start_text)
                end_minutes = _parse_time_token(end_text)
            except ValueError as exc:
                logging.warning("Ignoring dark-hours entry '%s': %s", chunk, exc)
                continue

        for day in days:
            if start_minutes == end_minutes:
                entries.append(
                    DarkHoursSegment(day, 0, MINUTES_PER_DAY)
                )
                continue
            if start_minutes < end_minutes:
                entries.append(DarkHoursSegment(day, start_minutes, end_minutes))
            else:
                entries.append(DarkHoursSegment(day, start_minutes, MINUTES_PER_DAY))
                next_day = (day + 1) % 7
                entries.append(DarkHoursSegment(next_day, 0, end_minutes))

    return tuple(entries)


DARK_HOURS_RAW = os.environ.get("DARK_HOURS")
DARK_HOURS_SEGMENTS = _parse_dark_hours_spec(DARK_HOURS_RAW)
DARK_HOURS_ENABLED = bool(DARK_HOURS_SEGMENTS)


def is_within_dark_hours(moment: Optional[datetime.datetime] = None) -> bool:
    if not DARK_HOURS_SEGMENTS:
        return False

    current = moment or datetime.datetime.now(CENTRAL_TIME)
    if current.tzinfo is None:
        current = CENTRAL_TIME.localize(current)  # type: ignore[attr-defined]
    else:
        current = current.astimezone(CENTRAL_TIME)

    weekday = current.weekday()
    minute_of_day = current.hour * 60 + current.minute

    for segment in DARK_HOURS_SEGMENTS:
        if segment.weekday != weekday:
            continue
        if segment.start_minute <= minute_of_day < segment.end_minute:
            return True
    return False

# ─── Scoreboard appearance ────────────────────────────────────────────────────


def _coerce_color_component(env_name: str, default: int) -> int:
    """Return a color channel value from 0-255 with logging on invalid input."""

    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid %s value %r; using default %d", env_name, raw_value, default
        )
        return default

    if not 0 <= value <= 255:
        logging.warning(
            "%s must be between 0 and 255; clamping %d to valid range", env_name, value
        )
        return max(0, min(255, value))

    return value


# Default background color for scoreboards and standings screens. Use an RGB
# tuple so callers can request either RGB or RGBA colors as needed.
SCOREBOARD_BACKGROUND_COLOR = (
    _coerce_color_component("SCOREBOARD_BACKGROUND_R", 125),
    _coerce_color_component("SCOREBOARD_BACKGROUND_G", 125),
    _coerce_color_component("SCOREBOARD_BACKGROUND_B", 125),
)

# Score colors shared across scoreboard implementations.
SCOREBOARD_IN_PROGRESS_SCORE_COLOR = (255, 210, 66)
SCOREBOARD_FINAL_WINNING_SCORE_COLOR = (255, 255, 255)
SCOREBOARD_FINAL_LOSING_SCORE_COLOR = (200, 200, 200)

# ─── Scoreboard scrolling configuration ───────────────────────────────────────
SCOREBOARD_SCROLL_STEP         = 1
SCOREBOARD_SCROLL_DELAY        = 0.001
SCOREBOARD_SCROLL_PAUSE_TOP    = 0.75
SCOREBOARD_SCROLL_PAUSE_BOTTOM = 0.5

# ─── API endpoints ────────────────────────────────────────────────────────────
WEATHERKIT_URL_TEMPLATE = (
    "https://weatherkit.apple.com/api/v1/weather/{language}/{lat}/{lon}"
)
NHL_API_URL        = "https://api-web.nhle.com/v1/club-schedule-season/CHI/20252026"
MLB_API_URL        = "https://statsapi.mlb.com/api/v1/schedule"
MLB_CUBS_TEAM_ID   = "112"
MLB_SOX_TEAM_ID    = "145"

NBA_TEAM_ID        = "1610612741"
NBA_TEAM_TRICODE   = "CHI"
NBA_IMAGES_DIR     = os.path.join(IMAGES_DIR, "nba")
NBA_FALLBACK_LOGO  = os.path.join(NBA_IMAGES_DIR, "NBA.png")

CENTRAL_TIME = pytz.timezone("America/Chicago")

# ─── Fonts ────────────────────────────────────────────────────────────────────
# Drop your TimesSquare-m105.ttf, DejaVuSans.ttf, and DejaVuSans-Bold.ttf into
# a folder named `fonts` alongside this file. Emoji glyphs are provided by the
# system Noto Color Emoji font (installed via package managers) or another
# system emoji font fallback.
FONTS_DIR = os.path.join(SCRIPT_DIR, "fonts")

def _load_font(name, size):
    path = os.path.join(FONTS_DIR, name)
    return ImageFont.truetype(path, size)


def _try_load_font(name: str, size: int):
    path = os.path.join(FONTS_DIR, name)
    if not os.path.isfile(path):
        return None

    try:
        return ImageFont.truetype(path, size)
    except OSError as exc:
        message = str(exc).lower()
        log = logging.debug if "invalid pixel size" in message else logging.warning
        log("Unable to load font %s: %s", path, exc)
        return None


class _BitmapEmojiFont(ImageFont.ImageFont):
    """Scale bitmap-only emoji fonts to an arbitrary size."""

    def __init__(self, path: str, native_size: int, size: int):
        super().__init__()
        self._native_size = native_size
        self.size = size
        self._scale = size / native_size
        self._font = ImageFont.truetype(path, native_size)

    def getbbox(self, text, *args, **kwargs):  # type: ignore[override]
        bbox = self._font.getbbox(text, *args, **kwargs)
        if bbox is None:
            return None
        left, top, right, bottom = bbox
        scale = self._scale
        return (
            int(round(left * scale)),
            int(round(top * scale)),
            int(round(right * scale)),
            int(round(bottom * scale)),
        )

    def getmetrics(self):  # type: ignore[override]
        ascent, descent = self._font.getmetrics()
        scale = self._scale
        return int(round(ascent * scale)), int(round(descent * scale))

    def getsize(self, text, *args, **kwargs):  # type: ignore[override]
        bbox = self.getbbox(text, *args, **kwargs)
        if bbox:
            left, top, right, bottom = bbox
            return right - left, bottom - top
        width, height = self._font.getsize(text, *args, **kwargs)
        scale = self._scale
        return int(round(width * scale)), int(round(height * scale))

    def getlength(self, text, *args, **kwargs):  # type: ignore[override]
        width, _ = self.getsize(text, *args, **kwargs)
        return width

    def _render_native(self, text, *args, **kwargs):
        bbox = self._font.getbbox(text, *args, **kwargs)
        if bbox:
            left, top, right, bottom = bbox
            width = max(1, right - left)
            height = max(1, bottom - top)
        else:
            left = top = 0
            width, height = self._font.getsize(text, *args, **kwargs)

        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        draw.text((-left, -top), text, font=self._font, fill=255)
        return image

    def getmask(self, text, mode="L", *args, **kwargs):  # type: ignore[override]
        base = self._render_native(text, *args, **kwargs)
        scaled = base.resize(
            (
                max(1, int(round(base.width * self._scale))),
                max(1, int(round(base.height * self._scale))),
            ),
            resample=_RESAMPLE_LANCZOS,
        )

        if mode == "1":
            return scaled.convert("1").im
        if mode == "L":
            return scaled.im
        if mode == "RGBA":
            rgba = Image.new("RGBA", scaled.size, (255, 255, 255, 0))
            rgba.putalpha(scaled)
            return rgba.im
        return scaled.im

FONT_DAY_DATE           = _load_font("DejaVuSans-Bold.ttf", 39)
FONT_DATE               = _load_font("DejaVuSans.ttf",      22)
FONT_TIME               = _load_font("DejaVuSans-Bold.ttf", 59)
FONT_AM_PM              = _load_font("DejaVuSans.ttf",      20)

FONT_TEMP               = _load_font("DejaVuSans-Bold.ttf", 44)
FONT_CONDITION          = _load_font("DejaVuSans-Bold.ttf", 20)
FONT_WEATHER_DETAILS    = _load_font("DejaVuSans.ttf",      22)
FONT_WEATHER_DETAILS_BOLD = _load_font("DejaVuSans-Bold.ttf", 18)
FONT_WEATHER_DETAILS_SMALL = _load_font("DejaVuSans.ttf",      14)
FONT_WEATHER_DETAILS_SMALL_BOLD = _load_font("DejaVuSans-Bold.ttf", 14)
FONT_WEATHER_DETAILS_TINY = _load_font("DejaVuSans.ttf",      12)
FONT_WEATHER_DETAILS_TINY_LARGE = _load_font("DejaVuSans.ttf",      13)
FONT_WEATHER_DETAILS_TINY_MICRO = _load_font("DejaVuSans.ttf",      10)
FONT_WEATHER_LABEL      = _load_font("DejaVuSans.ttf",      18)

FONT_TITLE_SPORTS       = _load_font("TimesSquare-m105.ttf", 30)
FONT_TEAM_SPORTS        = _load_font("TimesSquare-m105.ttf", 37)
FONT_DATE_SPORTS        = _load_font("TimesSquare-m105.ttf", 30)
FONT_TEAM_SPORTS_SMALL  = _load_font("TimesSquare-m105.ttf", 33)
FONT_SCORE              = _load_font("TimesSquare-m105.ttf", 41)
FONT_STATUS             = _load_font("TimesSquare-m105.ttf", 30)

FONT_INSIDE_LABEL       = _load_font("DejaVuSans-Bold.ttf", 18)
FONT_INSIDE_VALUE       = _load_font("DejaVuSans.ttf", 17)
FONT_TITLE_INSIDE       = _load_font("DejaVuSans-Bold.ttf", 17)

FONT_TRAVEL_TITLE       = _load_font("TimesSquare-m105.ttf", 17)
FONT_TRAVEL_HEADER      = _load_font("TimesSquare-m105.ttf", 17)
FONT_TRAVEL_VALUE       = _load_font("HWYGNRRW.TTF", 26)

FONT_IP_LABEL           = FONT_INSIDE_LABEL
FONT_IP_VALUE           = FONT_INSIDE_VALUE

FONT_STOCK_TITLE        = _load_font("DejaVuSans-Bold.ttf", 18)
FONT_STOCK_PRICE        = _load_font("DejaVuSans-Bold.ttf", 44)
FONT_STOCK_CHANGE       = _load_font("DejaVuSans.ttf",      22)
FONT_STOCK_TEXT         = _load_font("DejaVuSans.ttf",      17)

# Standings fonts...
FONT_STAND1_WL          = _load_font("DejaVuSans-Bold.ttf", 26)
FONT_STAND1_WL_LARGE    = _load_font("DejaVuSans-Bold.ttf", 65)
FONT_STAND1_RANK        = _load_font("DejaVuSans.ttf",      22)
FONT_STAND1_GB_LABEL    = _load_font("DejaVuSans.ttf",      17)
FONT_STAND1_WCGB_LABEL  = _load_font("DejaVuSans.ttf",      17)
FONT_STAND1_GB_VALUE    = _load_font("DejaVuSans.ttf",      17)
FONT_STAND1_WCGB_VALUE  = _load_font("DejaVuSans.ttf",      17)

FONT_STAND2_RECORD      = _load_font("DejaVuSans.ttf",      26)
FONT_STAND2_LABEL       = _load_font("DejaVuSans.ttf",      22)
FONT_STAND2_VALUE       = _load_font("DejaVuSans.ttf",      22)

FONT_DIV_HEADER         = _load_font("DejaVuSans-Bold.ttf", 20)
FONT_DIV_RECORD         = _load_font("DejaVuSans.ttf",      22)
FONT_DIV_GB             = _load_font("DejaVuSans.ttf",      18)
FONT_GB_VALUE           = _load_font("DejaVuSans.ttf",      18)
FONT_GB_LABEL           = _load_font("DejaVuSans.ttf",      15)

def _load_emoji_font(size: int) -> ImageFont.ImageFont:
    noto = _try_load_font("NotoColorEmoji.ttf", size)
    if noto:
        return noto

    noto_path = os.path.join(FONTS_DIR, "NotoColorEmoji.ttf")
    if os.path.isfile(noto_path):
        for native_size in (109, 128, 160):
            try:
                return _BitmapEmojiFont(noto_path, native_size, size)
            except OSError as exc:
                logging.debug(
                    "Unable to load bitmap emoji font %s at native size %s: %s",
                    noto_path,
                    native_size,
                    exc,
                )

    noto_system_paths = glob.glob(
        "/usr/share/fonts/**/NotoColorEmoji.ttf", recursive=True
    )
    for path in noto_system_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError as exc:
            message = str(exc).lower()
            logging.debug("Unable to load system emoji font %s: %s", path, exc)
            if "invalid pixel size" in message:
                for native_size in (109, 128, 160):
                    try:
                        return _BitmapEmojiFont(path, native_size, size)
                    except OSError as inner_exc:
                        logging.debug(
                            "Unable to load system bitmap emoji font %s at native size %s: %s",
                            path,
                            native_size,
                            inner_exc,
                        )

    symbola_paths = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    for path in symbola_paths:
        if "symbola" not in path.lower():
            continue
        try:
            return ImageFont.truetype(path, size)
        except OSError as exc:
            logging.debug("Unable to load fallback emoji font %s: %s", path, exc)

    logging.warning("Emoji font not found; falling back to PIL default font")
    return ImageFont.load_default()


FONT_EMOJI = _load_emoji_font(30)
FONT_EMOJI_SMALL = _load_emoji_font(18)


def _normalise_hex_color(value: str) -> Optional[str]:
    cleaned = value.strip()
    if not cleaned:
        return None
    if not re.fullmatch(r"#?[0-9a-fA-F]{6}", cleaned):
        return None
    return cleaned.upper() if cleaned.startswith("#") else f"#{cleaned.upper()}"


def _parse_hex_color(value: str) -> Optional[Tuple[int, int, int]]:
    normalised = _normalise_hex_color(value)
    if not normalised:
        return None
    return tuple(int(normalised[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def _normalise_style_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalised: Dict[str, Any] = {"screens": {}}
    if not isinstance(payload, dict):
        return normalised

    screens = payload.get("screens")
    if not isinstance(screens, dict):
        return normalised

    for screen_id, spec in screens.items():
        if not isinstance(screen_id, str) or not isinstance(spec, dict):
            continue

        fonts: Dict[str, Dict[str, Any]] = {}
        images: Dict[str, Dict[str, Any]] = {}

        font_specs = spec.get("fonts")
        if isinstance(font_specs, dict):
            for font_slot, font_spec in font_specs.items():
                if not isinstance(font_slot, str) or not isinstance(font_spec, dict):
                    continue
                entry: Dict[str, Any] = {}
                family = font_spec.get("family")
                if isinstance(family, str) and family.strip():
                    entry["family"] = family.strip()
                size = font_spec.get("size")
                if isinstance(size, int) and size > 0:
                    entry["size"] = size
                if entry:
                    fonts[font_slot] = entry

        image_specs = spec.get("images")
        if isinstance(image_specs, dict):
            for image_slot, image_spec in image_specs.items():
                if not isinstance(image_slot, str) or not isinstance(image_spec, dict):
                    continue
                scale = image_spec.get("scale")
                try:
                    scale_value = float(scale)
                except (TypeError, ValueError):
                    continue
                if scale_value <= 0:
                    continue
                images[image_slot] = {"scale": scale_value}

        entry: Dict[str, Any] = {}
        background = spec.get("background")
        if isinstance(background, str):
            normalised_background = _normalise_hex_color(background)
            if normalised_background:
                entry["background"] = normalised_background
        if fonts:
            entry["fonts"] = fonts
        if images:
            entry["images"] = images
        if entry:
            normalised["screens"][screen_id] = entry

    return normalised


def _load_style_config(*, force: bool = False) -> Dict[str, Any]:
    global _STYLE_CONFIG_CACHE, _STYLE_CONFIG_MTIME

    try:
        mtime = os.path.getmtime(STYLE_CONFIG_PATH)
    except OSError:
        mtime = None

    with _STYLE_CONFIG_LOCK:
        if not force and _STYLE_CONFIG_CACHE is not None and _STYLE_CONFIG_MTIME == mtime:
            return _STYLE_CONFIG_CACHE

        try:
            raw = _STYLE_CONFIG_STORE.load()
        except Exception as exc:  # pragma: no cover - unexpected read failure
            logging.debug("Unable to load style configuration: %s", exc)
            raw = {}

        normalised = _normalise_style_config(raw)
        _STYLE_CONFIG_CACHE = normalised
        _STYLE_CONFIG_MTIME = mtime
        return normalised


def reload_style_config() -> Dict[str, Any]:
    """Force a reload of the style configuration."""

    return _load_style_config(force=True)


def get_style_config() -> Dict[str, Any]:
    """Return the cached style configuration."""

    return _load_style_config()


def get_screen_style(screen_id: str) -> Dict[str, Any]:
    """Return the style overrides for *screen_id*."""

    config = get_style_config()
    screens = config.get("screens") or {}
    if not isinstance(screens, dict):
        return {}
    entry = screens.get(screen_id)
    return entry if isinstance(entry, dict) else {}


def get_screen_background_color(
    screen_id: str,
    default: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Return the background color override for *screen_id* if configured."""

    style = get_screen_style(screen_id)
    background = style.get("background")
    if isinstance(background, str):
        parsed = _parse_hex_color(background)
        if parsed is not None:
            return parsed
    return default


def _clone_font_instance(font: ImageFont.FreeTypeFont, size: int) -> ImageFont.FreeTypeFont:
    path = getattr(font, "path", None)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            logging.debug("Unable to clone font %s at size %s", path, size)
    return font


def _load_font_from_family(family: str, size: int) -> Optional[ImageFont.FreeTypeFont]:
    candidates = [family]
    if not os.path.isabs(family):
        candidates.insert(0, os.path.join(FONTS_DIR, family))

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    logging.debug("Unable to load override font '%s'", family)
    return None


def get_screen_font(
    screen_id: str,
    font_slot: str,
    *,
    base_font: ImageFont.FreeTypeFont,
    default_size: Optional[int] = None,
) -> ImageFont.FreeTypeFont:
    """Return a font for *screen_id*/*font_slot* applying style overrides."""

    style = get_screen_style(screen_id)
    fonts = style.get("fonts") if isinstance(style.get("fonts"), dict) else {}
    spec = fonts.get(font_slot) if isinstance(fonts, dict) else None

    target_size = default_size or getattr(base_font, "size", None)
    if isinstance(spec, dict):
        size_override = spec.get("size")
        if isinstance(size_override, int) and size_override > 0:
            target_size = size_override
        family = spec.get("family")
        if isinstance(family, str) and family.strip():
            override_font = _load_font_from_family(family.strip(), target_size or getattr(base_font, "size", 12))
            if override_font is not None:
                return override_font

    if target_size and getattr(base_font, "size", None) != target_size:
        return _clone_font_instance(base_font, target_size)
    return base_font


def get_screen_image_scale(screen_id: str, image_slot: str, default: float = 1.0) -> float:
    """Return an image scaling factor for *screen_id*/*image_slot*."""

    style = get_screen_style(screen_id)
    images = style.get("images") if isinstance(style.get("images"), dict) else {}
    spec = images.get(image_slot) if isinstance(images, dict) else None
    if isinstance(spec, dict):
        scale = spec.get("scale")
        try:
            value = float(scale)
        except (TypeError, ValueError):
            value = default
        else:
            if value > 0:
                return value
    return default

# ─── Screen-specific configuration ─────────────────────────────────────────────

# Weather screen
WEATHER_ICON_SIZE = 218
WEATHER_DESC_GAP  = 8

# Date/time screen
DATE_TIME_GH_ICON_INVERT = True
DATE_TIME_GH_ICON_SIZE   = 33
DATE_TIME_GH_ICON_PATHS  = [
    os.path.join(IMAGES_DIR, "gh.png"),
    os.path.join(SCRIPT_DIR, "image", "gh.png"),
]

# Indoor sensor screen colors
INSIDE_COL_BG     = (0, 0, 0)
INSIDE_COL_TITLE  = (240, 240, 240)
INSIDE_CHIP_BLUE  = (34, 124, 236)
INSIDE_CHIP_AMBER = (233, 165, 36)
INSIDE_CHIP_PURPLE = (150, 70, 200)
INSIDE_COL_TEXT   = (255, 255, 255)
INSIDE_COL_STROKE = (230, 230, 230)

# Travel time screen
DEFAULT_WORK_ADDRESS = "224 W Hill St, Chicago, IL"
DEFAULT_HOME_ADDRESS = "3912 Rutgers Ln, Northbrook, IL"

TRAVEL_TO_HOME_ORIGIN = os.environ.get("TRAVEL_TO_HOME_ORIGIN", DEFAULT_WORK_ADDRESS)
TRAVEL_TO_HOME_DESTINATION = os.environ.get(
    "TRAVEL_TO_HOME_DESTINATION", DEFAULT_HOME_ADDRESS
)
TRAVEL_TO_WORK_ORIGIN = os.environ.get(
    "TRAVEL_TO_WORK_ORIGIN", TRAVEL_TO_HOME_DESTINATION
)
TRAVEL_TO_WORK_DESTINATION = os.environ.get(
    "TRAVEL_TO_WORK_DESTINATION", TRAVEL_TO_HOME_ORIGIN
)

TRAVEL_PROFILES = {
    "to_home": {
        "origin": TRAVEL_TO_HOME_ORIGIN,
        "destination": TRAVEL_TO_HOME_DESTINATION,
        "title": "To home:",
        "active_window": (datetime.time(14, 30), datetime.time(19, 0)),
    },
    "to_work": {
        "origin": TRAVEL_TO_WORK_ORIGIN,
        "destination": TRAVEL_TO_WORK_DESTINATION,
        "title": "To work:",
        "active_window": (datetime.time(6, 0), datetime.time(11, 0)),
    },
    "default": {
        "origin": TRAVEL_TO_HOME_ORIGIN,
        "destination": TRAVEL_TO_HOME_DESTINATION,
        "title": "Travel time:",
        "active_window": (datetime.time(6, 0), datetime.time(19, 0)),
    },
}

_travel_profile = TRAVEL_PROFILES.get(TRAVEL_MODE, TRAVEL_PROFILES["default"])
TRAVEL_ORIGIN        = _travel_profile["origin"]
TRAVEL_DESTINATION   = _travel_profile["destination"]
TRAVEL_TITLE         = _travel_profile["title"]
TRAVEL_ACTIVE_WINDOW = _travel_profile["active_window"]
TRAVEL_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

# Bears schedule screen
BEARS_BOTTOM_MARGIN = 6
BEARS_SCHEDULE = [
    {"game_no": "0.1","week":"Preseason 1","date":"Sat, Aug 9, 2025",  "opponent":"Miami Dolphins",       "home_away":"Home","time":"Noon"},
    {"game_no": "0.2","week":"Preseason 2","date":"Sun, Aug 17, 2025", "opponent":"Buffalo Bills",        "home_away":"Home","time":"7PM"},
    {"game_no": "0.3","week":"Preseason 3","date":"Fri, Aug 22, 2025", "opponent":"Kansas City Chiefs",   "home_away":"Away","time":"7:20PM"},
    {"game_no": "1","week":"Week 1",  "date":"Mon, Sep 8, 2025",  "opponent":"Minnesota Vikings",    "home_away":"Home","time":"7:15PM"},
    {"game_no": "2","week":"Week 2",  "date":"Sun, Sep 14, 2025", "opponent":"Detroit Lions",        "home_away":"Away","time":"Noon"},
    {"game_no": "3","week":"Week 3",  "date":"Sun, Sep 21, 2025", "opponent":"Dallas Cowboys",       "home_away":"Home","time":"3:25PM"},
    {"game_no": "4","week":"Week 4",  "date":"Sun, Sep 28, 2025", "opponent":"Las Vegas Raiders",    "home_away":"Away","time":"3:25PM"},
    {"game_no": "5","week":"Week 5",  "date":"BYE",         "opponent":"—",                    "home_away":"—",   "time":"—"},
    {"game_no": "6","week":"Week 6",  "date":"Mon, Oct 13, 2025","opponent":"Washington Commanders", "home_away":"Away","time":"7:15PM"},
    {"game_no": "7","week":"Week 7",  "date":"Sun, Oct 19, 2025","opponent":"New Orleans Saints",    "home_away":"Home","time":"Noon"},
    {"game_no": "8","week":"Week 8",  "date":"Sun, Oct 26, 2025","opponent":"Baltimore Ravens",      "home_away":"Away","time":"Noon"},
    {"game_no": "9","week":"Week 9",  "date":"Sun, Nov 2, 2025", "opponent":"Cincinnati Bengals",    "home_away":"Away","time":"Noon"},
    {"game_no": "10","week":"Week 10", "date":"Sun, Nov 9, 2025", "opponent":"New York Giants",       "home_away":"Home","time":"Noon"},
    {"game_no": "11","week":"Week 11", "date":"Sun, Nov 16, 2025","opponent":"Minnesota Vikings",     "home_away":"Away","time":"Noon"},
    {"game_no": "12","week":"Week 12", "date":"Sun, Nov 23, 2025","opponent":"Pittsburgh Steelers",   "home_away":"Home","time":"Noon"},
    {"game_no": "13","week":"Week 13", "date":"Fri, Nov 28, 2025","opponent":"Philadelphia Eagles",   "home_away":"Away","time":"2PM"},
    {"game_no": "14","week":"Week 14", "date":"Sun, Dec 7, 2025", "opponent":"Green Bay Packers",     "home_away":"Away","time":"3:25PM"},
    {"game_no": "15","week":"Week 15", "date":"Sun, Dec 14, 2025","opponent":"Cleveland Browns",      "home_away":"Home","time":"Noon"},
    {"game_no": "16","week":"Week 16", "date":"Sat, Dec 20, 2025","opponent":"Green Bay Packers",     "home_away":"Home","time":"7:20PM"},
    {"game_no": "17","week":"Week 17", "date":"Sun, Dec 28, 2025","opponent":"San Francisco 49ers",   "home_away":"Away","time":"7:20PM"},
    {"game_no": "18","week":"Week 18", "date":"Sun, Jan 4, 2026","opponent":"Detroit Lions",         "home_away":"Home","time":"3:25PM"},
    {"game_no": "19","week":"Wild Card", "date":"Sat, Jan 10, 2026","opponent":"Green Bay Packers",   "home_away":"Home","time":"7PM"},
    {"game_no": "20","week":"Divisional Playoff ", "date":"Sun, Jan 18, 2026","opponent":"Los Angeles Rams",  "home_away":"Home","time":"5:30PM"},
]
NFL_TEAM_ABBREVIATIONS = {
    "dolphins": "mia",   "bills": "buf",   "chiefs": "kc",
    "vikings": "min",    "lions": "det",   "cowboys": "dal",
    "raiders": "lv",     "commanders": "was","saints": "no",
    "ravens": "bal",     "bengals": "cin",  "giants": "nyg",
    "steelers": "pit",   "eagles": "phi",   "packers": "gb",
    "browns": "cle",     "49ers": "sf",     "rams": "lar",
}

# VRNO screen
VRNO_FRESHNESS_LIMIT = 10 * 60
VRNO_LOTS = [
    {"shares": 125, "cost": 3.39},
    {"shares": 230, "cost": 0.74},
    {"shares": 230, "cost": 1.34},
    {"shares": 555, "cost": 0.75},
    {"shares": 107, "cost": 0.64},
    {"shares": 157, "cost": 0.60},
]

# Hockey assets
NHL_IMAGES_DIR = os.path.join(IMAGES_DIR, "nhl")
AHL_IMAGES_DIR = os.path.join(IMAGES_DIR, "ahl")
TIMES_SQUARE_FONT_PATH = os.path.join(FONTS_DIR, "TimesSquare-m105.ttf")
os.makedirs(NHL_IMAGES_DIR, exist_ok=True)
os.makedirs(AHL_IMAGES_DIR, exist_ok=True)

NHL_API_ENDPOINTS = {
    "team_month_now": "https://api-web.nhle.com/v1/club-schedule/{tric}/month/now",
    "team_season_now": "https://api-web.nhle.com/v1/club-schedule-season/{tric}/now",
    "game_landing": "https://api-web.nhle.com/v1/gamecenter/{gid}/landing",
    "game_boxscore": "https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore",
    "stats_schedule": "https://statsapi.web.nhl.com/api/v1/schedule",
    "stats_feed": "https://statsapi.web.nhl.com/api/v1/game/{gamePk}/feed/live",
}

NHL_TEAM_ID      = 16
NHL_TEAM_TRICODE = "CHI"
NHL_FALLBACK_LOGO = os.path.join(NHL_IMAGES_DIR, "NHL.jpg")

AHL_API_BASE_URL   = os.environ.get("AHL_API_BASE_URL", "https://lscluster.hockeytech.com/feed/")
AHL_API_KEY        = os.environ.get("AHL_API_KEY", "50c4cd9b5df2e390")
AHL_CLIENT_CODE    = os.environ.get("AHL_CLIENT_CODE", "ahl")
AHL_LEAGUE_ID      = os.environ.get("AHL_LEAGUE_ID", "4")
AHL_SITE_ID        = os.environ.get("AHL_SITE_ID", "1")
AHL_SEASON_ID      = os.environ.get("AHL_SEASON_ID")
AHL_SCHEDULE_ICS_URL = os.environ.get(
    "AHL_SCHEDULE_ICS_URL",
    "https://app.stanzacal.com/api/calendar/webcal/ahl-chicagowolves/55db9bc32a0c4b9e35d487c5/67191f4120dfd9eadf697a35.ics",
)
try:
    AHL_TEAM_ID = int(os.environ.get("AHL_TEAM_ID", "624"))
except (TypeError, ValueError):
    logging.warning("Invalid AHL_TEAM_ID value; defaulting to 624")
    AHL_TEAM_ID = 624
AHL_TEAM_TRICODE   = os.environ.get("AHL_TEAM_TRICODE", "CHI")
AHL_FALLBACK_LOGO  = os.path.join(AHL_IMAGES_DIR, "AHL.png")
AHL_TEAM_NAME      = os.environ.get("AHL_TEAM_NAME", "Chicago Wolves")
