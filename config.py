# config.py

#!/usr/bin/env python3
import copy
import datetime
import glob
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

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

try:  # Optional YAML support for profile files
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

import pytz
from PIL import Image, ImageDraw, ImageFont

try:
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # Pillow >= 9.1
except AttributeError:  # pragma: no cover - fallback for older Pillow
    _RESAMPLE_LANCZOS = Image.LANCZOS


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep merge of *base* with *overrides* without mutating inputs."""

    result: Dict[str, Any] = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        elif isinstance(value, list):
            result[key] = [copy.deepcopy(item) for item in value]
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_profile_file(path: str) -> Dict[str, Any]:
    """Load an optional JSON/YAML display profile file."""

    if not path:
        return {}
    if not os.path.isfile(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as fh:
            if path.endswith((".yaml", ".yml")):
                if yaml is None:
                    logging.warning("YAML profile file requested but PyYAML is not installed: %s", path)
                    return {}
                data = yaml.safe_load(fh)
            else:
                data = json.load(fh)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.warning("Failed to load display profile configuration %s: %s", path, exc)
        return {}

    if not isinstance(data, Mapping):
        logging.warning("Display profile file %s did not contain an object", path)
        return {}

    return dict(data)

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

# ─── Display profiles ──────────────────────────────────────────────────────────
_FONT_LIBRARY: Dict[str, Dict[str, Any]] = {
    "day_date": {"file": "DejaVuSans-Bold.ttf", "size": 39},
    "date": {"file": "DejaVuSans.ttf", "size": 22},
    "time": {"file": "DejaVuSans-Bold.ttf", "size": 59},
    "am_pm": {"file": "DejaVuSans.ttf", "size": 20},
    "temp": {"file": "DejaVuSans-Bold.ttf", "size": 44},
    "condition": {"file": "DejaVuSans-Bold.ttf", "size": 20},
    "weather_details": {"file": "DejaVuSans.ttf", "size": 22},
    "weather_details_bold": {"file": "DejaVuSans-Bold.ttf", "size": 18},
    "weather_label": {"file": "DejaVuSans.ttf", "size": 18},
    "title_sports": {"file": "TimesSquare-m105.ttf", "size": 30},
    "team_sports": {"file": "TimesSquare-m105.ttf", "size": 37},
    "date_sports": {"file": "TimesSquare-m105.ttf", "size": 30},
    "team_sports_small": {"file": "TimesSquare-m105.ttf", "size": 33},
    "score": {"file": "TimesSquare-m105.ttf", "size": 41},
    "status": {"file": "TimesSquare-m105.ttf", "size": 30},
    "inside_label": {"file": "DejaVuSans-Bold.ttf", "size": 18},
    "inside_value": {"file": "DejaVuSans.ttf", "size": 17},
    "inside_subtitle": {"file": "DejaVuSans.ttf", "size": 18},
    "title_inside": {"file": "DejaVuSans-Bold.ttf", "size": 17},
    "travel_title": {"file": "TimesSquare-m105.ttf", "size": 17},
    "travel_header": {"file": "TimesSquare-m105.ttf", "size": 17},
    "travel_value": {"file": "HWYGNRRW.TTF", "size": 26},
    "stock_title": {"file": "DejaVuSans-Bold.ttf", "size": 18},
    "stock_price": {"file": "DejaVuSans-Bold.ttf", "size": 44},
    "stock_change": {"file": "DejaVuSans.ttf", "size": 22},
    "stock_text": {"file": "DejaVuSans.ttf", "size": 17},
    "stand1_wl": {"file": "DejaVuSans-Bold.ttf", "size": 26},
    "stand1_rank": {"file": "DejaVuSans.ttf", "size": 22},
    "stand1_gb_label": {"file": "DejaVuSans.ttf", "size": 17},
    "stand1_wcgb_label": {"file": "DejaVuSans.ttf", "size": 17},
    "stand1_gb_value": {"file": "DejaVuSans.ttf", "size": 17},
    "stand1_wcgb_value": {"file": "DejaVuSans.ttf", "size": 17},
    "stand2_record": {"file": "DejaVuSans.ttf", "size": 26},
    "stand2_label": {"file": "DejaVuSans.ttf", "size": 22},
    "stand2_value": {"file": "DejaVuSans.ttf", "size": 22},
    "div_header": {"file": "DejaVuSans-Bold.ttf", "size": 20},
    "div_record": {"file": "DejaVuSans.ttf", "size": 22},
    "div_gb": {"file": "DejaVuSans.ttf", "size": 18},
    "gb_value": {"file": "DejaVuSans.ttf", "size": 18},
    "gb_label": {"file": "DejaVuSans.ttf", "size": 15},
    "emoji": {"size": 30},
}


def _base_profile_template() -> Dict[str, Any]:
    fonts = {key: dict(spec) for key, spec in _FONT_LIBRARY.items()}
    return {
        "description": "Pimoroni Display HAT Mini (320×240)",
        "display_backend": "displayhatmini",
        "canvas": {"width": 320, "height": 240},
        "baseline": {"width": 320, "height": 240},
        "font_scale": 1.0,
        "icon_scale": 1.0,
        "fonts": fonts,
        "icons": {
            "github": {
                "size": 33,
                "padding_x": 2,
                "padding_y": 2,
                "baseline_offset": 4,
                "invert": True,
            },
            "weather": {"size": 218},
        },
        "animation": {
            "screen_delay": 4.0,
            "scoreboard": {
                "intro_delay": 0.06,
                "intro_hold": 0.4,
            },
            "inside": {"hold": 5.0},
        },
        "scoreboard": {
            "column_widths": [70, 60, 60, 60, 70],
            "title_gap": 8,
            "block_spacing": 10,
            "score_row_height": 56,
            "status_row_height": 18,
            "logo_height": 52,
            "league_logo_gap": 4,
            "intro_max_height": 100,
            "scroll": {
                "step": 1,
                "delay": 0.005,
                "pause_top": 0.75,
                "pause_bottom": 0.5,
            },
        },
        "inside": {
            "title_padding": 8,
            "subtitle_gap": 6,
            "content_gap": 12,
            "bottom_margin": 12,
            "side_padding": 12,
            "metric_block_gap": 12,
            "metric_row_height": 44,
            "metric_row_gap": 10,
            "min_temp_floor": 54,
        },
        "travel": {
            "outer_margin": 4,
            "row_padding_x": 10,
            "row_padding_y": 4,
            "row_gap": 4,
            "header_gap": 4,
            "warning_gap": 6,
            "warning_bottom_margin": 4,
            "row_corner_radius": 10,
        },
    }


def _build_default_display_profiles() -> Dict[str, Dict[str, Any]]:
    base = _base_profile_template()
    return {
        "display_hat_mini": base,
        "hyperpixel_4_0": _deep_merge(
            base,
            {
                "description": "Pimoroni HyperPixel 4.0 (800×480)",
                "display_backend": "framebuffer",
                "canvas": {"width": 800, "height": 480},
                "font_scale": 2.1,
                "icon_scale": 2.0,
                "icons": {
                    "github": {
                        "padding_x": 6,
                        "padding_y": 6,
                        "baseline_offset": 8,
                    }
                },
                "animation": {
                    "screen_delay": 6.0,
                    "scoreboard": {"intro_hold": 0.6, "intro_delay": 0.05},
                },
                "scoreboard": {"intro_max_height": 160},
                "inside": {
                    "content_gap": 18,
                    "bottom_margin": 18,
                    "metric_block_gap": 18,
                    "metric_row_height": 60,
                    "metric_row_gap": 14,
                    "min_temp_floor": 72,
                },
            },
        ),
        "hyperpixel_4_0_square": _deep_merge(
            base,
            {
                "description": "Pimoroni HyperPixel 4.0 Square (720×720)",
                "display_backend": "framebuffer",
                "canvas": {"width": 720, "height": 720},
                "baseline": {"width": 320, "height": 320},
                "font_scale": 2.25,
                "icon_scale": 2.15,
                "icons": {
                    "github": {
                        "padding_x": 8,
                        "padding_y": 8,
                        "baseline_offset": 10,
                    },
                    "weather": {"size": 480},
                },
                "animation": {
                    "screen_delay": 6.5,
                    "scoreboard": {"intro_hold": 0.65, "intro_delay": 0.05},
                },
                "scoreboard": {"intro_max_height": 220},
                "inside": {
                    "content_gap": 24,
                    "bottom_margin": 24,
                    "side_padding": 24,
                    "metric_block_gap": 24,
                    "metric_row_height": 90,
                    "metric_row_gap": 18,
                    "min_temp_floor": 72,
                },
                "travel": {
                    "outer_margin": {"absolute": True, "value": 24},
                    "row_padding_x": {"absolute": True, "value": 36},
                    "row_padding_y": {"absolute": True, "value": 16},
                    "row_gap": {"absolute": True, "value": 24},
                    "header_gap": {"absolute": True, "value": 20},
                    "warning_gap": {"absolute": True, "value": 28},
                    "warning_bottom_margin": {"absolute": True, "value": 30},
                    "row_corner_radius": {"absolute": True, "value": 32},
                },
            },
        ),
        "xpt2046_3_5": _deep_merge(
            base,
            {
                "description": "3.5″ SPI (XPT2046, 480×320)",
                "canvas": {"width": 480, "height": 320},
                "font_scale": 1.5,
                "icon_scale": 1.4,
                "icons": {
                    "github": {
                        "padding_x": 4,
                        "padding_y": 4,
                        "baseline_offset": 6,
                    }
                },
                "animation": {"screen_delay": 5.0},
                "scoreboard": {"intro_max_height": 130},
                "inside": {
                    "content_gap": 16,
                    "bottom_margin": 16,
                    "metric_block_gap": 14,
                    "metric_row_height": 52,
                    "metric_row_gap": 12,
                    "min_temp_floor": 64,
                },
            },
        ),
    }


DISPLAY_PROFILES_PATH = os.environ.get(
    "DISPLAY_PROFILES_PATH", os.path.join(SCRIPT_DIR, "display_profiles.json")
)
DEFAULT_DISPLAY_PROFILES = _build_default_display_profiles()
_profile_file_payload = _load_profile_file(DISPLAY_PROFILES_PATH)
if isinstance(_profile_file_payload, Mapping) and "profiles" in _profile_file_payload:
    raw_overrides = _profile_file_payload.get("profiles")
else:
    raw_overrides = _profile_file_payload

PROFILE_OVERRIDES: Dict[str, Any]
if isinstance(raw_overrides, Mapping):
    PROFILE_OVERRIDES = dict(raw_overrides)
else:
    PROFILE_OVERRIDES = {}

DISPLAY_PROFILES: Dict[str, Dict[str, Any]] = {}
for profile_id, base_profile in DEFAULT_DISPLAY_PROFILES.items():
    overrides = PROFILE_OVERRIDES.get(profile_id)
    if isinstance(overrides, Mapping):
        DISPLAY_PROFILES[profile_id] = _deep_merge(base_profile, overrides)
    else:
        DISPLAY_PROFILES[profile_id] = copy.deepcopy(base_profile)

for profile_id, profile_data in PROFILE_OVERRIDES.items():
    if profile_id in DISPLAY_PROFILES:
        continue
    if isinstance(profile_data, Mapping):
        DISPLAY_PROFILES[profile_id] = _deep_merge(
            _base_profile_template(), profile_data
        )

if "display_hat_mini" not in DISPLAY_PROFILES:
    DISPLAY_PROFILES["display_hat_mini"] = _base_profile_template()

DISPLAY_PROFILE_ID = os.environ.get("DISPLAY_PROFILE", "display_hat_mini")
if DISPLAY_PROFILE_ID not in DISPLAY_PROFILES:
    logging.warning(
        "Unknown display profile %s; defaulting to display_hat_mini",
        DISPLAY_PROFILE_ID,
    )
    DISPLAY_PROFILE_ID = "display_hat_mini"

ACTIVE_DISPLAY_PROFILE = DISPLAY_PROFILES[DISPLAY_PROFILE_ID]


def get_available_display_profiles() -> Sequence[str]:
    return tuple(sorted(DISPLAY_PROFILES.keys()))


def get_display_profile_id() -> str:
    return DISPLAY_PROFILE_ID


def get_display_profile() -> Dict[str, Any]:
    return copy.deepcopy(ACTIVE_DISPLAY_PROFILE)


def _resolve_default_display_backend() -> str:
    backend = ACTIVE_DISPLAY_PROFILE.get("display_backend")
    if isinstance(backend, str) and backend.strip():
        return backend.strip()
    return "displayhatmini"


DEFAULT_DISPLAY_BACKEND = _resolve_default_display_backend()


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return int(default)


def profile_value(path: str, default: Any = None) -> Any:
    if not path:
        return copy.deepcopy(ACTIVE_DISPLAY_PROFILE)

    parts = [part for part in path.split(".") if part]
    node: Any = ACTIVE_DISPLAY_PROFILE
    for part in parts:
        if not isinstance(node, Mapping):
            return default
        node = node.get(part)
        if node is None:
            return default

    if isinstance(node, Mapping):
        return copy.deepcopy(node)
    if isinstance(node, list):
        return list(node)
    return node


def _resolve_canvas_dimension(key: str, default: int) -> int:
    canvas = ACTIVE_DISPLAY_PROFILE.get("canvas")
    value = (canvas or {}).get(key, default)
    return max(1, _coerce_int(value, default))


WIDTH = _resolve_canvas_dimension("width", 320)
HEIGHT = _resolve_canvas_dimension("height", 240)

baseline = ACTIVE_DISPLAY_PROFILE.get("baseline") or {}
_BASELINE_WIDTH = max(1, _coerce_int(baseline.get("width", WIDTH), WIDTH))
_BASELINE_HEIGHT = max(1, _coerce_int(baseline.get("height", HEIGHT), HEIGHT))

_WIDTH_SCALE = WIDTH / _BASELINE_WIDTH if _BASELINE_WIDTH else 1.0
_HEIGHT_SCALE = HEIGHT / _BASELINE_HEIGHT if _BASELINE_HEIGHT else 1.0

_FONT_SCALE = _coerce_float(
    ACTIVE_DISPLAY_PROFILE.get("font_scale"), min(_WIDTH_SCALE, _HEIGHT_SCALE)
)
if _FONT_SCALE <= 0:
    _FONT_SCALE = 1.0
_ICON_SCALE = _coerce_float(ACTIVE_DISPLAY_PROFILE.get("icon_scale"), _FONT_SCALE)
if _ICON_SCALE <= 0:
    _ICON_SCALE = _FONT_SCALE


def scale_width(value: Any, *, minimum: int = 0) -> int:
    return max(minimum, int(round(_coerce_float(value, 0.0) * _WIDTH_SCALE)))


def scale_height(value: Any, *, minimum: int = 0) -> int:
    return max(minimum, int(round(_coerce_float(value, 0.0) * _HEIGHT_SCALE)))


def scale_font_size(value: Any, *, minimum: int = 1) -> int:
    return max(minimum, int(round(_coerce_float(value, 0.0) * _FONT_SCALE)))


def scale_icon_size(value: Any, *, minimum: int = 1) -> int:
    return max(minimum, int(round(_coerce_float(value, 0.0) * _ICON_SCALE)))


def scale_to_width(values: Sequence[Any], *, total: Optional[int] = None) -> list[int]:
    if not values:
        return []

    target_total = int(total if total is not None else WIDTH)
    floats = [max(0.0, _coerce_float(v, 0.0)) for v in values]
    base_total = sum(floats)
    if base_total <= 0:
        width = max(1, target_total // len(floats))
        result = [width] * len(floats)
        diff = target_total - sum(result)
        if diff:
            result[-1] = max(1, result[-1] + diff)
        return result

    scale = target_total / base_total
    result = [max(1, int(round(v * scale))) for v in floats]
    diff = target_total - sum(result)
    if diff:
        result[-1] = max(1, result[-1] + diff)
    return result


def get_canvas_size() -> tuple[int, int]:
    return WIDTH, HEIGHT


def resolve_icon_size(key: str, default: int) -> int:
    entry = profile_value(f"icons.{key}", None)
    if isinstance(entry, Mapping):
        if entry.get("size") is not None:
            return max(1, _coerce_int(entry.get("size"), default))
        if entry.get("scale") is not None:
            scale = _coerce_float(entry.get("scale"), 1.0)
            return max(1, int(round(default * scale)))
    elif isinstance(entry, (int, float)):
        return max(1, int(round(_coerce_float(entry, default))))
    return max(1, scale_icon_size(default))


def resolve_dimension(path: str, default: int, *, axis: str) -> int:
    raw = profile_value(path, None)
    absolute = False
    if isinstance(raw, Mapping):
        if raw.get("absolute") is True:
            absolute = True
        if raw.get("value") is not None:
            raw_value = _coerce_float(raw.get("value"), default)
        elif raw.get("amount") is not None:
            raw_value = _coerce_float(raw.get("amount"), default)
        else:
            raw_value = _coerce_float(raw.get("size", default), default)
    elif isinstance(raw, (int, float)):
        raw_value = _coerce_float(raw, default)
    else:
        raw_value = float(default)

    if absolute:
        return max(0, int(round(raw_value)))

    if axis.lower() == "width":
        return max(0, scale_width(raw_value))
    if axis.lower() == "height":
        return max(0, scale_height(raw_value))
    return max(0, int(round(raw_value)))


# ─── Display configuration ─────────────────────────────────────────────────────
SCREEN_DELAY = _coerce_float(profile_value("animation.screen_delay", 4.0), 4.0)
INSIDE_SCREEN_HOLD = _coerce_float(profile_value("animation.inside.hold", 5.0), 5.0)
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
SCOREBOARD_SCROLL_STEP = max(
    1, _coerce_int(profile_value("scoreboard.scroll.step", 1), 1)
)
SCOREBOARD_SCROLL_DELAY = max(
    0.0, _coerce_float(profile_value("scoreboard.scroll.delay", 0.005), 0.005)
)
SCOREBOARD_SCROLL_PAUSE_TOP = max(
    0.0, _coerce_float(profile_value("scoreboard.scroll.pause_top", 0.75), 0.75)
)
SCOREBOARD_SCROLL_PAUSE_BOTTOM = max(
    0.0, _coerce_float(profile_value("scoreboard.scroll.pause_bottom", 0.5), 0.5)
)

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


def _load_profile_font(key: str) -> ImageFont.ImageFont:
    spec = _FONT_LIBRARY.get(key, {})
    default_file = spec.get("path") or spec.get("file")
    default_size = spec.get("size", 16)

    override = profile_value(f"fonts.{key}", None)
    font_path = default_file
    explicit_size: Optional[float] = None

    if isinstance(override, Mapping):
        override_path = override.get("path") or override.get("file")
        if isinstance(override_path, str) and override_path.strip():
            font_path = override_path.strip()
        if override.get("size") is not None:
            explicit_size = _coerce_float(override.get("size"), default_size)
        elif override.get("scale") is not None:
            explicit_size = default_size * _coerce_float(override.get("scale"), 1.0)
    elif isinstance(override, (int, float)):
        explicit_size = float(override)
    elif isinstance(override, str) and override.strip():
        font_path = override.strip()

    if explicit_size is None or explicit_size <= 0:
        computed_size = scale_font_size(default_size)
    else:
        computed_size = max(1, int(round(explicit_size)))

    if not font_path:
        raise ValueError(f"Font definition '{key}' is missing a file path")

    if os.path.isabs(font_path):
        return ImageFont.truetype(font_path, computed_size)

    return _load_font(font_path, computed_size)


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

FONT_DAY_DATE = _load_profile_font("day_date")
FONT_DATE = _load_profile_font("date")
FONT_TIME = _load_profile_font("time")
FONT_AM_PM = _load_profile_font("am_pm")

FONT_TEMP = _load_profile_font("temp")
FONT_CONDITION = _load_profile_font("condition")
FONT_WEATHER_DETAILS = _load_profile_font("weather_details")
FONT_WEATHER_DETAILS_BOLD = _load_profile_font("weather_details_bold")
FONT_WEATHER_LABEL = _load_profile_font("weather_label")

FONT_TITLE_SPORTS = _load_profile_font("title_sports")
FONT_TEAM_SPORTS = _load_profile_font("team_sports")
FONT_DATE_SPORTS = _load_profile_font("date_sports")
FONT_TEAM_SPORTS_SMALL = _load_profile_font("team_sports_small")
FONT_SCORE = _load_profile_font("score")
FONT_STATUS = _load_profile_font("status")

FONT_INSIDE_LABEL = _load_profile_font("inside_label")
FONT_INSIDE_VALUE = _load_profile_font("inside_value")
FONT_INSIDE_SUBTITLE = _load_profile_font("inside_subtitle")
FONT_TITLE_INSIDE = _load_profile_font("title_inside")

FONT_TRAVEL_TITLE = _load_profile_font("travel_title")
FONT_TRAVEL_HEADER = _load_profile_font("travel_header")
FONT_TRAVEL_VALUE = _load_profile_font("travel_value")

FONT_IP_LABEL = FONT_INSIDE_LABEL
FONT_IP_VALUE = FONT_INSIDE_VALUE

FONT_STOCK_TITLE = _load_profile_font("stock_title")
FONT_STOCK_PRICE = _load_profile_font("stock_price")
FONT_STOCK_CHANGE = _load_profile_font("stock_change")
FONT_STOCK_TEXT = _load_profile_font("stock_text")

# Standings fonts...
FONT_STAND1_WL = _load_profile_font("stand1_wl")
FONT_STAND1_RANK = _load_profile_font("stand1_rank")
FONT_STAND1_GB_LABEL = _load_profile_font("stand1_gb_label")
FONT_STAND1_WCGB_LABEL = _load_profile_font("stand1_wcgb_label")
FONT_STAND1_GB_VALUE = _load_profile_font("stand1_gb_value")
FONT_STAND1_WCGB_VALUE = _load_profile_font("stand1_wcgb_value")

FONT_STAND2_RECORD = _load_profile_font("stand2_record")
FONT_STAND2_LABEL = _load_profile_font("stand2_label")
FONT_STAND2_VALUE = _load_profile_font("stand2_value")

FONT_DIV_HEADER = _load_profile_font("div_header")
FONT_DIV_RECORD = _load_profile_font("div_record")
FONT_DIV_GB = _load_profile_font("div_gb")
FONT_GB_VALUE = _load_profile_font("gb_value")
FONT_GB_LABEL = _load_profile_font("gb_label")

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


_EMOJI_OVERRIDE = profile_value("fonts.emoji", None)
if isinstance(_EMOJI_OVERRIDE, Mapping):
    _emoji_size = _EMOJI_OVERRIDE.get("size")
elif isinstance(_EMOJI_OVERRIDE, (int, float)):
    _emoji_size = _EMOJI_OVERRIDE
else:
    _emoji_size = None

if _emoji_size is None or _coerce_float(_emoji_size, 30) <= 0:
    emoji_point_size = scale_font_size(_FONT_LIBRARY.get("emoji", {}).get("size", 30))
else:
    emoji_point_size = max(1, int(round(_coerce_float(_emoji_size, 30))))

FONT_EMOJI = _load_emoji_font(int(emoji_point_size))

# ─── Screen-specific configuration ─────────────────────────────────────────────

# Weather screen
WEATHER_ICON_SIZE = resolve_icon_size("weather", 218)
WEATHER_DESC_GAP = resolve_dimension("weather.description_gap", 8, axis="height")

# Date/time screen
DATE_TIME_GH_ICON_INVERT = bool(profile_value("icons.github.invert", True))
DATE_TIME_GH_ICON_SIZE = resolve_icon_size("github", 33)
DATE_TIME_GH_ICON_PADDING_X = resolve_dimension(
    "icons.github.padding_x", 2, axis="width"
)
DATE_TIME_GH_ICON_PADDING_Y = resolve_dimension(
    "icons.github.padding_y", 2, axis="height"
)
DATE_TIME_GH_ICON_BASELINE_OFFSET = resolve_dimension(
    "icons.github.baseline_offset", 4, axis="height"
)
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
TRAVEL_OUTER_MARGIN = resolve_dimension("travel.outer_margin", 4, axis="width")
TRAVEL_ROW_PADDING_X = resolve_dimension("travel.row_padding_x", 10, axis="width")
TRAVEL_ROW_PADDING_Y = resolve_dimension("travel.row_padding_y", 4, axis="height")
TRAVEL_ROW_GAP = resolve_dimension("travel.row_gap", 4, axis="height")
TRAVEL_HEADER_GAP = resolve_dimension("travel.header_gap", 4, axis="height")
TRAVEL_WARNING_EXTRA_GAP = resolve_dimension("travel.warning_gap", 6, axis="height")
TRAVEL_WARNING_BOTTOM_MARGIN = resolve_dimension(
    "travel.warning_bottom_margin", 4, axis="height"
)
TRAVEL_ROW_CORNER_RADIUS = resolve_dimension(
    "travel.row_corner_radius", 10, axis="width"
)
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
