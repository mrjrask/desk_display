# config.py

#!/usr/bin/env python3
import datetime
import glob
import inspect
import logging
import os
import re
import re
import subprocess
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

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
            try:
                load_dotenv(path, override=False)
            except OSError:
                logging.debug("Could not read .env file at %s", path)
        else:
            _load_env_file(str(path))


_ENV_INITIALISED = False


def initialise_env_if_requested(force: bool = False) -> None:
    """Conditionally load `.env` files based on CONFIG_LOAD_DOTENV flag."""

    global _ENV_INITIALISED

    if _ENV_INITIALISED and not force:
        return

    # Default to loading `.env` so direct module/script execution behaves the
    # same as the main app service. Set CONFIG_LOAD_DOTENV=0 to opt out.
    raw_flag = os.environ.get("CONFIG_LOAD_DOTENV", "1").strip().lower()
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

from PIL import Image, ImageDraw, ImageFont


def _supports_embedded_color() -> bool:
    try:
        parameters = inspect.signature(ImageDraw.ImageDraw.text).parameters
    except (TypeError, ValueError):
        return False
    return "embedded_color" in parameters


EMOJI_EMBEDDED_COLOR = _supports_embedded_color()

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
WEATHER_USE_EMOJI_ICONS = _get_bool_env("WEATHER_USE_EMOJI_ICONS", False)

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
BASE_WIDTH = 320
BASE_HEIGHT = 240

def _parse_mode_size(raw: Optional[str]) -> Optional[Tuple[int, int]]:
    if not raw:
        return None
    match = re.search(r"(\d+)\s*x\s*(\d+)", raw)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _read_framebuffer_mode_size(device_path: str) -> Optional[Tuple[int, int]]:
    fb_name = Path(device_path).name
    sysfs_base = Path("/sys/class/graphics") / fb_name
    try:
        raw = (sysfs_base / "mode").read_text(encoding="utf-8").strip()
    except OSError:
        raw = ""
    mode_size = _parse_mode_size(raw)
    if mode_size:
        return mode_size
    try:
        modes_raw = (sysfs_base / "modes").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    first_line = modes_raw.splitlines()[0] if modes_raw else ""
    return _parse_mode_size(first_line)


def _read_framebuffer_fbset_size(device_path: str) -> Optional[Tuple[int, int]]:
    try:
        result = subprocess.run(
            ["fbset", "-fb", device_path, "-s"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    for line in output.splitlines():
        if "geometry" not in line:
            continue
        match = re.search(r"geometry\s+(\d+)\s+(\d+)", line)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _read_framebuffer_virtual_size(device_path: str) -> Optional[Tuple[int, int]]:
    fb_name = Path(device_path).name
    sysfs_base = Path("/sys/class/graphics") / fb_name / "virtual_size"
    try:
        raw = sysfs_base.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        width_str, height_str = raw.split(",", 1)
        return int(width_str), int(height_str)
    except ValueError:
        return None


def _read_drm_mode_size() -> Optional[Tuple[int, int]]:
    for status_path in Path("/sys/class/drm").glob("card*-*/status"):
        try:
            status = status_path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if status != "connected":
            continue
        modes_path = status_path.parent / "modes"
        try:
            modes = modes_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if not modes:
            continue
        mode = modes[0]
        if "x" not in mode:
            continue
        try:
            width_str, height_str = mode.split("x", 1)
            return int(width_str), int(height_str)
        except ValueError:
            continue
    return None


def _read_kernel_overlay_rotation() -> Optional[int]:
    """Read display rotation from Raspberry Pi dtoverlay configuration."""

    config_paths = (
        Path("/boot/firmware/config.txt"),
        Path("/boot/config.txt"),
    )
    overlays_of_interest = {"vc4-kms-dpi-hyperpixel4", "vc4-kms-dpi-hyperpixel4sq"}

    for config_path in config_paths:
        try:
            lines = config_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if not line or not line.lower().startswith("dtoverlay="):
                continue

            overlay_config = line.split("=", 1)[1].strip()
            if not overlay_config:
                continue

            parts = [part.strip() for part in overlay_config.split(",") if part.strip()]
            if not parts:
                continue

            overlay_name = parts[0].lower()
            should_check = (
                overlay_name in overlays_of_interest or "rotate=" in overlay_config.lower()
            )
            if not should_check:
                continue

            for part in parts[1:]:
                if not part.lower().startswith("rotate="):
                    continue
                raw_value = part.split("=", 1)[1].strip()
                try:
                    parsed = int(raw_value)
                except ValueError:
                    logging.warning(
                        "Ignoring invalid rotate value '%s' in %s.",
                        raw_value,
                        config_path,
                    )
                    break
                if parsed in (0, 1, 2, 3):
                    return parsed * 90
                return parsed
                break

    return None


_display_width_set = "DISPLAY_WIDTH" in os.environ
_display_height_set = "DISPLAY_HEIGHT" in os.environ

try:
    WIDTH = int(os.environ.get("DISPLAY_WIDTH", str(BASE_WIDTH)))
except (TypeError, ValueError):
    logging.warning("Invalid DISPLAY_WIDTH value; defaulting to %s.", BASE_WIDTH)
    WIDTH = BASE_WIDTH

try:
    HEIGHT = int(os.environ.get("DISPLAY_HEIGHT", str(BASE_HEIGHT)))
except (TypeError, ValueError):
    logging.warning("Invalid DISPLAY_HEIGHT value; defaulting to %s.", BASE_HEIGHT)
    HEIGHT = BASE_HEIGHT

_display_output = os.environ.get("DESK_DISPLAY_OUTPUT", "auto").strip().lower()
_hyperpixel_panel = os.environ.get("HYPERPIXEL_PANEL", "").strip().lower()
if (_display_width_set, _display_height_set) != (True, True):
    if _display_output in {"framebuffer", "fb", "framebuffer-device"}:
        fb_device = os.environ.get("DISPLAY_FB_DEVICE", "/dev/fb0")
        fb_size = (
            _read_framebuffer_mode_size(fb_device)
            or _read_framebuffer_fbset_size(fb_device)
            or _read_framebuffer_virtual_size(fb_device)
        )
        if fb_size:
            fb_width, fb_height = fb_size
            if not _display_width_set:
                WIDTH = fb_width
            if not _display_height_set:
                HEIGHT = fb_height
    elif _display_output in {"kernel", "kms", "drm", "sdl"}:
        drm_size = _read_drm_mode_size()
        if drm_size:
            drm_width, drm_height = drm_size
            if not _display_width_set:
                WIDTH = drm_width
            if not _display_height_set:
                HEIGHT = drm_height

if (WIDTH, HEIGHT) == (480, 800) and (
    _hyperpixel_panel == "hyperpixel4" or _display_output in {"kernel", "kms", "drm", "sdl"}
):
    logging.info(
        "Detected HyperPixel 4 portrait mode dimensions (480x800); normalizing to 800x480."
    )
    WIDTH, HEIGHT = 800, 480

HYPERPIXEL_LED_INDICATOR_BORDER_ENABLED = _get_bool_env(
    "HYPERPIXEL_LED_INDICATOR_BORDER_ENABLED",
    True,
)

DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED = _get_bool_env(
    "DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED",
    True,
)

DISPLAY_HAT_MINI_LED_ENABLED = _get_bool_env(
    "DISPLAY_HAT_MINI_LED_ENABLED",
    True,
)

try:
    HYPERPIXEL_LED_INDICATOR_BORDER_WIDTH = int(
        os.environ.get("HYPERPIXEL_LED_INDICATOR_BORDER_WIDTH", "2")
    )
except (TypeError, ValueError):
    logging.warning(
        "Invalid HYPERPIXEL_LED_INDICATOR_BORDER_WIDTH value; defaulting to 2."
    )
    HYPERPIXEL_LED_INDICATOR_BORDER_WIDTH = 2

if HYPERPIXEL_LED_INDICATOR_BORDER_WIDTH < 1:
    logging.warning(
        "HYPERPIXEL_LED_INDICATOR_BORDER_WIDTH must be >= 1; clamping to 1."
    )
    HYPERPIXEL_LED_INDICATOR_BORDER_WIDTH = 1

def _compute_display_scale(
    base_width: int,
    base_height: int,
    target_width: int,
    target_height: int,
) -> float:
    """Return the proportional scale based on percentage resolution differences."""
    if base_width <= 0 or base_height <= 0:
        return 1.0
    width_ratio = target_width / base_width
    height_ratio = target_height / base_height
    scale = (width_ratio + height_ratio) / 2.0
    return max(0.1, scale)


DISPLAY_SCALE = _compute_display_scale(BASE_WIDTH, BASE_HEIGHT, WIDTH, HEIGHT)
DISPLAY_SCALE_WIDTH = max(0.1, WIDTH / BASE_WIDTH) if BASE_WIDTH > 0 else 1.0


def scale_value(value: float) -> int:
    return max(1, int(round(value * DISPLAY_SCALE)))


def scale_value_width(value: float) -> int:
    return max(1, int(round(value * DISPLAY_SCALE_WIDTH)))
HYPERPIXEL_NEXT_LAYOUT_SIZES = {(800, 480), (480, 800), (720, 720)}


def is_hyperpixel_next_layout(width: int | None = None, height: int | None = None) -> bool:
    if width is None:
        width = WIDTH
    if height is None:
        height = HEIGHT
    return (width, height) in HYPERPIXEL_NEXT_LAYOUT_SIZES


def is_hyperpixel_4_square_layout(width: int | None = None, height: int | None = None) -> bool:
    if width is None:
        width = WIDTH
    if height is None:
        height = HEIGHT
    return (width, height) == (720, 720)
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

_use_kernel_rotation_source = (
    _hyperpixel_panel.startswith("hyperpixel")
    or _display_output in {"kernel", "kms", "drm", "sdl"}
)
_kernel_overlay_rotation = _read_kernel_overlay_rotation() if _use_kernel_rotation_source else None
_display_rotation_raw = os.environ.get("DISPLAY_ROTATION")

if _display_rotation_raw is not None:
    try:
        _parsed_display_rotation = int(_display_rotation_raw)
        if _parsed_display_rotation in (0, 1, 2, 3):
            _parsed_display_rotation *= 90
        DISPLAY_ROTATION = _parsed_display_rotation
    except (TypeError, ValueError):
        logging.warning("Invalid DISPLAY_ROTATION value; defaulting to 0°.")
        DISPLAY_ROTATION = 0
else:
    DISPLAY_ROTATION = 0

if _kernel_overlay_rotation is not None:
    if _display_rotation_raw is not None:
        logging.info(
            "Kernel overlay rotate=%d° detected and DISPLAY_ROTATION=%d° is set; "
            "both transforms will apply.",
            _kernel_overlay_rotation,
            DISPLAY_ROTATION,
        )
    else:
        logging.info(
            "Kernel overlay rotate=%d° detected; DISPLAY_ROTATION defaults to 0° "
            "to avoid double-rotation.",
            _kernel_overlay_rotation,
        )


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
        current = current.replace(tzinfo=CENTRAL_TIME)
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

CENTRAL_TIME = ZoneInfo("America/Chicago")

# ─── Fonts ────────────────────────────────────────────────────────────────────
# Drop your TimesSquare-m105.ttf, DejaVuSans.ttf, and DejaVuSans-Bold.ttf into
# a folder named `fonts` alongside this file. Emoji glyphs are provided by the
# system Noto Color Emoji font (installed via package managers) or another
# system emoji font fallback.
FONTS_DIR = os.path.join(SCRIPT_DIR, "fonts")

def _load_font(name, size):
    path = os.path.join(FONTS_DIR, name)
    return ImageFont.truetype(path, scale_value(size))


def _try_load_font(name: str, size: int):
    path = os.path.join(FONTS_DIR, name)
    if not os.path.isfile(path):
        return None

    try:
        return ImageFont.truetype(path, scale_value(size))
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

    def _render_native(self, text, mode="L", *args, **kwargs):
        bbox = self._font.getbbox(text, *args, **kwargs)
        if bbox:
            left, top, right, bottom = bbox
            width = max(1, right - left)
            height = max(1, bottom - top)
        else:
            left = top = 0
            width, height = self._font.getsize(text, *args, **kwargs)

        if mode == "RGBA":
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            if EMOJI_EMBEDDED_COLOR:
                draw.text((-left, -top), text, font=self._font, embedded_color=True)
            else:
                draw.text((-left, -top), text, font=self._font, fill=(255, 255, 255, 255))
            return image

        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        draw.text((-left, -top), text, font=self._font, fill=255)
        return image

    def getmask(self, text, mode="L", *args, **kwargs):  # type: ignore[override]
        base = self._render_native(text, mode, *args, **kwargs)
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
            return scaled.im
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
    scaled_size = scale_value(size)
    noto = _try_load_font("NotoColorEmoji.ttf", size)
    if noto:
        return noto

    noto_path = os.path.join(FONTS_DIR, "NotoColorEmoji.ttf")
    if os.path.isfile(noto_path):
        for native_size in (109, 128, 160):
            try:
                return _BitmapEmojiFont(noto_path, native_size, scaled_size)
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
            return ImageFont.truetype(path, scaled_size)
        except OSError as exc:
            message = str(exc).lower()
            logging.debug("Unable to load system emoji font %s: %s", path, exc)
            if "invalid pixel size" in message:
                for native_size in (109, 128, 160):
                    try:
                        return _BitmapEmojiFont(path, native_size, scaled_size)
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
            return ImageFont.truetype(path, scaled_size)
        except OSError as exc:
            logging.debug("Unable to load fallback emoji font %s: %s", path, exc)

    logging.warning("Emoji font not found; falling back to PIL default font")
    return ImageFont.load_default()


FONT_EMOJI = _load_emoji_font(30)
FONT_EMOJI_SMALL = _load_emoji_font(18)

_EMOJI_FONT_CACHE: Dict[int, ImageFont.ImageFont] = {}


def get_emoji_font(size: int) -> ImageFont.ImageFont:
    size = max(8, int(size))
    cached = _EMOJI_FONT_CACHE.get(size)
    if cached is not None:
        return cached
    font = _load_emoji_font(size)
    _EMOJI_FONT_CACHE[size] = font
    return font


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

    target_size = getattr(base_font, "size", None)
    if default_size is not None:
        target_size = scale_value(default_size)
    if isinstance(spec, dict):
        size_override = spec.get("size")
        if isinstance(size_override, int) and size_override > 0:
            target_size = scale_value(size_override)
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
                return value * DISPLAY_SCALE
    return default * DISPLAY_SCALE

# ─── Screen-specific configuration ─────────────────────────────────────────────

# Weather screen
WEATHER_ICON_SIZE = scale_value(218)
WEATHER_DESC_GAP  = scale_value(8)

# Date/time screen
DATE_TIME_GH_ICON_INVERT = True
DATE_TIME_GH_ICON_SIZE   = scale_value(33)
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
