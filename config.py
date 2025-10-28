# config.py

#!/usr/bin/env python3
import datetime
import glob
import logging
import os
import subprocess
from pathlib import Path

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


_initialise_env()


def _get_first_env_var(*names: str):
    """Return the first populated environment variable from *names.*"""

    for name in names:
        value = os.environ.get(name)
        if value:
            return value

    return None


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

try:
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # Pillow >= 9.1
except AttributeError:  # pragma: no cover - fallback for older Pillow
    _RESAMPLE_LANCZOS = Image.LANCZOS

# ─── Project paths ────────────────────────────────────────────────────────────
IMAGES_DIR  = os.path.join(SCRIPT_DIR, "images")

# ─── Feature flags ────────────────────────────────────────────────────────────
ENABLE_SCREENSHOTS   = True
ENABLE_VIDEO         = False
VIDEO_FPS            = 30
ENABLE_WIFI_MONITOR  = True

WIFI_RETRY_DURATION  = 180
WIFI_CHECK_INTERVAL  = 60
WIFI_OFF_DURATION    = 180

VRNOF_CACHE_TTL      = 1800

def get_current_ssid():
    try:
        return subprocess.check_output(["iwgetid", "-r"]).decode("utf-8").strip()
    except Exception:
        return None

CURRENT_SSID = get_current_ssid()

if CURRENT_SSID == "Verano":
    ENABLE_WEATHER = True
    OWM_API_KEY    = _get_first_env_var("OWM_API_KEY_VERANO", "OWM_API_KEY")
    LATITUDE       = 41.9103
    LONGITUDE      = -87.6340
    TRAVEL_MODE    = "to_home"
elif CURRENT_SSID == "wiffy":
    ENABLE_WEATHER = True
    OWM_API_KEY    = _get_first_env_var("OWM_API_KEY_WIFFY", "OWM_API_KEY")
    LATITUDE       = 42.13444
    LONGITUDE      = -87.876389
    TRAVEL_MODE    = "to_work"
else:
    ENABLE_WEATHER = True
    OWM_API_KEY    = _get_first_env_var("OWM_API_KEY_DEFAULT", "OWM_API_KEY")
    LATITUDE       = 41.9103
    LONGITUDE      = -87.6340
    TRAVEL_MODE    = "to_home"

if not OWM_API_KEY:
    logging.warning(
        "OpenWeatherMap API key not configured; the app will use fallback weather data only."
    )

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

# ─── Display configuration ─────────────────────────────────────────────────────
WIDTH                    = 320
HEIGHT                   = 240
SCREEN_DELAY             = 4
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
SCOREBOARD_SCROLL_DELAY        = 0.005
SCOREBOARD_SCROLL_PAUSE_TOP    = 0.75
SCOREBOARD_SCROLL_PAUSE_BOTTOM = 0.5

# ─── API endpoints ────────────────────────────────────────────────────────────
ONE_CALL_URL      = "https://api.openweathermap.org/data/3.0/onecall"
OPEN_METEO_URL    = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_PARAMS = {
    "latitude":        LATITUDE,
    "longitude":       LONGITUDE,
    "current_weather": True,
    "timezone":        "America/Chicago",
    "temperature_unit":"fahrenheit",
    "windspeed_unit":  "mph",
    "daily":           "temperature_2m_max,temperature_2m_min,sunrise,sunset"
}

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
# Drop your TimesSquare-m105.ttf, DejaVuSans.ttf, DejaVuSans-Bold.ttf and
# NotoColorEmoji.ttf into a new folder named `fonts` alongside this file.
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
BEARS_BOTTOM_MARGIN = 4
BEARS_SCHEDULE = [
    {"week":"0.1","date":"Sat, Aug 9",  "opponent":"Miami Dolphins",       "home_away":"Home","time":"Noon"},
    {"week":"0.2","date":"Sun, Aug 17", "opponent":"Buffalo Bills",        "home_away":"Home","time":"7PM"},
    {"week":"0.3","date":"Fri, Aug 22", "opponent":"Kansas City Chiefs",   "home_away":"Away","time":"7:20PM"},
    {"week":"Wk. 1",  "date":"Mon, Sep 8",  "opponent":"Minnesota Vikings",    "home_away":"Home","time":"7:15PM"},
    {"week":"Wk. 2",  "date":"Sun, Sep 14", "opponent":"Detroit Lions",        "home_away":"Away","time":"Noon"},
    {"week":"Wk. 3",  "date":"Sun, Sep 21", "opponent":"Dallas Cowboys",       "home_away":"Home","time":"3:25PM"},
    {"week":"Wk. 4",  "date":"Sun, Sep 28", "opponent":"Las Vegas Raiders",    "home_away":"Away","time":"3:25PM"},
    {"week":"Wk. 5",  "date":"BYE",         "opponent":"—",                    "home_away":"—",   "time":"—"},
    {"week":"Wk. 6",  "date":"Mon, Oct 13","opponent":"Washington Commanders", "home_away":"Away","time":"7:15PM"},
    {"week":"Wk. 7",  "date":"Sun, Oct 19","opponent":"New Orleans Saints",    "home_away":"Home","time":"Noon"},
    {"week":"Wk. 8",  "date":"Sun, Oct 26","opponent":"Baltimore Ravens",      "home_away":"Away","time":"Noon"},
    {"week":"Wk. 9",  "date":"Sun, Nov 2", "opponent":"Cincinnati Bengals",    "home_away":"Away","time":"Noon"},
    {"week":"Wk. 10", "date":"Sun, Nov 9", "opponent":"New York Giants",       "home_away":"Home","time":"Noon"},
    {"week":"Wk. 11", "date":"Sun, Nov 16","opponent":"Minnesota Vikings",     "home_away":"Away","time":"Noon"},
    {"week":"Wk. 12", "date":"Sun, Nov 23","opponent":"Pittsburgh Steelers",   "home_away":"Home","time":"Noon"},
    {"week":"Wk. 13", "date":"Fri, Nov 28","opponent":"Philadelphia Eagles",   "home_away":"Away","time":"2PM"},
    {"week":"Wk. 14", "date":"Sun, Dec 7", "opponent":"Green Bay Packers",     "home_away":"Away","time":"Noon"},
    {"week":"Wk. 15", "date":"Sun, Dec 14","opponent":"Cleveland Browns",      "home_away":"Home","time":"Noon"},
    {"week":"Wk. 16", "date":"Sat, Dec 20","opponent":"Green Bay Packers",     "home_away":"Home","time":"TBD"},
    {"week":"Wk. 17", "date":"Sun, Dec 28","opponent":"San Francisco 49ers",   "home_away":"Away","time":"7:20PM"},
    {"week":"Wk. 18", "date":"TBD",        "opponent":"Detroit Lions",         "home_away":"Home","time":"TBD"},
]

NFL_TEAM_ABBREVIATIONS = {
    "dolphins": "mia",   "bills": "buf",   "chiefs": "kc",
    "vikings": "min",    "lions": "det",   "cowboys": "dal",
    "raiders": "lv",     "commanders": "was","saints": "no",
    "ravens": "bal",     "bengals": "cin",  "giants": "nyg",
    "steelers": "pit",   "eagles": "phi",   "packers": "gb",
    "browns": "cle",     "49ers": "sf",
}

# VRNOF screen
VRNOF_FRESHNESS_LIMIT = 10 * 60
VRNOF_LOTS = [
    {"shares": 125, "cost": 3.39},
    {"shares": 230, "cost": 0.74},
    {"shares": 230, "cost": 1.34},
    {"shares": 555, "cost": 0.75},
    {"shares": 107, "cost": 0.64},
    {"shares": 157, "cost": 0.60},
]

# Hockey assets
NHL_IMAGES_DIR = os.path.join(IMAGES_DIR, "nhl")
TIMES_SQUARE_FONT_PATH = os.path.join(FONTS_DIR, "TimesSquare-m105.ttf")
os.makedirs(NHL_IMAGES_DIR, exist_ok=True)

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
