#!/usr/bin/env python3
"""
data_fetch.py

All remote data fetchers for weather, Blackhawks, MLB, etc.,
with resilient retries via a shared requests.Session.
"""

import csv
import datetime
import io
import json
import logging
import os
import re
import socket
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import pytz
import requests
import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from services.http_client import NHL_HEADERS, get_session
from screens.nba_scoreboard import _fetch_games_for_date as _nba_fetch_games_for_date

from config import (
    LATITUDE,
    LONGITUDE,
    NHL_API_URL,
    NHL_TEAM_ID,
    MLB_API_URL,
    MLB_CUBS_TEAM_ID,
    MLB_SOX_TEAM_ID,
    CENTRAL_TIME,
    NBA_TEAM_ID,
    NBA_TEAM_TRICODE,
    AHL_API_BASE_URL,
    AHL_API_KEY,
    AHL_CLIENT_CODE,
    AHL_LEAGUE_ID,
    AHL_SCHEDULE_ICS_URL,
    AHL_SEASON_ID,
    AHL_SITE_ID,
    AHL_TEAM_ID,
    AHL_TEAM_NAME,
    AHL_TEAM_TRICODE,
    WEATHERKIT_TEAM_ID,
    WEATHERKIT_KEY_ID,
    WEATHERKIT_SERVICE_ID,
    WEATHERKIT_KEY_PATH,
    WEATHERKIT_PRIVATE_KEY,
    WEATHERKIT_LANGUAGE,
    WEATHERKIT_TIMEZONE,
    WEATHERKIT_URL_TEMPLATE,
    WEATHER_REFRESH_SECONDS,
    OWM_API_KEY,
    OWM_API_URL,
    OWM_LANGUAGE,
    OWM_UNITS,
)

# ─── Shared HTTP session ─────────────────────────────────────────────────────
_session = get_session()

# Weather caching to limit API usage across devices
_weather_cache: Optional[dict[str, Any]] = None
_weather_cache_fetched_at: Optional[datetime.datetime] = None
_weatherkit_token: Optional[str] = None
_weatherkit_token_exp: Optional[datetime.datetime] = None
_weatherkit_key_cache: Optional[Any] = None
_owm_backoff_until: Optional[datetime.datetime] = None
_PRESSURE_TREND_WINDOW_SECONDS = 3 * 60 * 60
_PRESSURE_TREND_PRUNE_SECONDS = 6 * 60 * 60
_PRESSURE_TREND_THRESHOLD_HPA = 0.5
_PRESSURE_HISTORY: deque[tuple[float, float]] = deque()
_PRESSURE_HISTORY_LOADED = False
_PRESSURE_HISTORY_LAST_SAVE = 0.0
_PRESSURE_HISTORY_SAVE_INTERVAL_SECONDS = 60.0
_PRESSURE_HISTORY_PATH = os.environ.get("PRESSURE_HISTORY_PATH", "pressure_history.json")
# Cache statsapi DNS availability to avoid repeated slow lookups
_statsapi_dns_available: Optional[bool] = None
_statsapi_dns_checked_at: Optional[float] = None
_STATSAPI_DNS_RECHECK_SECONDS = 600
# Cache team standings to avoid repeated slow API calls.
_TEAM_STANDINGS_CACHE: dict[str, dict[str, Any]] = {}
_TEAM_STANDINGS_CACHE_SUCCESS_SECONDS = 300
_TEAM_STANDINGS_CACHE_FAILURE_SECONDS = 60
_TEAM_STANDINGS_TIMEOUT = (3.05, 5.0)


def _get_cached_team_standings(cache_key: str) -> tuple[Optional[dict], bool]:
    entry = _TEAM_STANDINGS_CACHE.get(cache_key)
    if not entry:
        return None, False

    ttl = (
        _TEAM_STANDINGS_CACHE_SUCCESS_SECONDS
        if entry.get("success")
        else _TEAM_STANDINGS_CACHE_FAILURE_SECONDS
    )
    if time.time() - entry.get("fetched_at", 0) < ttl:
        return entry.get("data"), True
    return None, False


def _store_team_standings_cache(cache_key: str, data: Optional[dict], *, success: bool) -> None:
    _TEAM_STANDINGS_CACHE[cache_key] = {
        "data": data,
        "success": success,
        "fetched_at": time.time(),
    }


def _update_pressure_trend(timestamp: Optional[float], pressure_hpa: Optional[float]) -> Optional[str]:
    _load_pressure_history()
    if pressure_hpa is None:
        return None

    try:
        pressure_val = float(pressure_hpa)
    except (TypeError, ValueError):
        return None

    now_ts = float(timestamp) if timestamp is not None else time.time()
    if _PRESSURE_HISTORY and now_ts < _PRESSURE_HISTORY[-1][0]:
        now_ts = _PRESSURE_HISTORY[-1][0] + 0.001

    history_changed = False
    if not _PRESSURE_HISTORY or _PRESSURE_HISTORY[-1] != (now_ts, pressure_val):
        _PRESSURE_HISTORY.append((now_ts, pressure_val))
        history_changed = True

    while _PRESSURE_HISTORY and now_ts - _PRESSURE_HISTORY[0][0] > _PRESSURE_TREND_PRUNE_SECONDS:
        _PRESSURE_HISTORY.popleft()
        history_changed = True

    if history_changed:
        _save_pressure_history(now_ts)

    target_time = now_ts - _PRESSURE_TREND_WINDOW_SECONDS
    baseline = None
    for ts, val in _PRESSURE_HISTORY:
        if ts <= target_time:
            baseline = val
        else:
            break

    if baseline is None:
        return None

    delta = pressure_val - baseline
    if delta > _PRESSURE_TREND_THRESHOLD_HPA:
        return "rising"
    if delta < -_PRESSURE_TREND_THRESHOLD_HPA:
        return "falling"
    return "steady"


def _load_pressure_history() -> None:
    global _PRESSURE_HISTORY_LOADED
    if _PRESSURE_HISTORY_LOADED:
        return

    _PRESSURE_HISTORY_LOADED = True
    path = _expand_path(_PRESSURE_HISTORY_PATH)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        history = payload.get("history", payload)
        if not isinstance(history, list):
            raise ValueError("Invalid pressure history payload")

        for entry in history:
            if (
                isinstance(entry, (list, tuple))
                and len(entry) == 2
                and isinstance(entry[0], (int, float))
                and isinstance(entry[1], (int, float))
            ):
                _PRESSURE_HISTORY.append((float(entry[0]), float(entry[1])))
        if _PRESSURE_HISTORY:
            now_ts = time.time()
            while _PRESSURE_HISTORY and now_ts - _PRESSURE_HISTORY[0][0] > _PRESSURE_TREND_PRUNE_SECONDS:
                _PRESSURE_HISTORY.popleft()
    except FileNotFoundError:
        return
    except Exception as exc:
        logging.warning("Unable to load pressure history from %s: %s", path, exc)


def _save_pressure_history(now_ts: float) -> None:
    global _PRESSURE_HISTORY_LAST_SAVE
    if now_ts - _PRESSURE_HISTORY_LAST_SAVE < _PRESSURE_HISTORY_SAVE_INTERVAL_SECONDS:
        return

    path = _expand_path(_PRESSURE_HISTORY_PATH)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"history": list(_PRESSURE_HISTORY)}
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_path, path)
        _PRESSURE_HISTORY_LAST_SAVE = now_ts
    except Exception as exc:
        logging.warning("Unable to save pressure history to %s: %s", path, exc)

# -----------------------------------------------------------------------------
# WEATHER — Apple WeatherKit primary, OpenWeatherMap secondary
# -----------------------------------------------------------------------------
def _expand_path(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def load_weatherkit_private_key(key_path: str):
    key_path = _expand_path(key_path)

    with open(key_path, "rb") as f:
        pem_bytes = f.read()

    # normalize line endings; ensure trailing newline
    pem_bytes = pem_bytes.replace(b"\r\n", b"\n").strip() + b"\n"

    return serialization.load_pem_private_key(
        pem_bytes,
        password=None,
        backend=default_backend(),
    )


def _load_weatherkit_private_key() -> Optional[Any]:
    global _weatherkit_key_cache
    if _weatherkit_key_cache:
        return _weatherkit_key_cache

    if WEATHERKIT_PRIVATE_KEY:
        raw_key = WEATHERKIT_PRIVATE_KEY.strip()
        # Remove accidental surrounding quotes that can be introduced by shell
        # quoting or dotenv files.
        if (raw_key.startswith("'") and raw_key.endswith("'")) or (
            raw_key.startswith('"') and raw_key.endswith('"')
        ):
            raw_key = raw_key[1:-1].strip()

        if raw_key:
            # Handle keys provided as literal "\n" sequences (common in env vars)
            # and normalize CRLF line endings.
            normalized_key = raw_key.replace("\\n", "\n").replace("\r\n", "\n")

            # If the key looks like bare base64 (missing PEM framing), wrap it so
            # cryptography can parse it correctly.
            if "-----BEGIN" not in normalized_key and re.fullmatch(
                r"[A-Za-z0-9+/=\s]+", normalized_key
            ):
                normalized_key = (
                    "-----BEGIN PRIVATE KEY-----\n"
                    + "\n".join(line.strip() for line in normalized_key.splitlines() if line.strip())
                    + "\n-----END PRIVATE KEY-----"
                )

            # If the env var accidentally contains a file path instead of the
            # actual key, attempt to read the file.
            normalized_path = _expand_path(normalized_key)
            if (
                "-----BEGIN" not in normalized_key
                and os.path.exists(normalized_path)
                and os.path.isfile(normalized_path)
            ):
                try:
                    _weatherkit_key_cache = load_weatherkit_private_key(normalized_path)
                    return _weatherkit_key_cache
                except Exception as exc:
                    logging.error(
                        "Unable to read WEATHERKIT_PRIVATE_KEY path %s: %s",
                        normalized_path,
                        exc,
                    )
                    return None

            try:
                pem_bytes = normalized_key.replace("\r\n", "\n").strip().encode("utf-8")
                if not pem_bytes.endswith(b"\n"):
                    pem_bytes += b"\n"

                _weatherkit_key_cache = serialization.load_pem_private_key(
                    pem_bytes,
                    password=None,
                    backend=default_backend(),
                )
            except Exception as exc:
                logging.error("Unable to parse WEATHERKIT_PRIVATE_KEY: %s", exc)
                return None

            return _weatherkit_key_cache

    if WEATHERKIT_KEY_PATH:
        try:
            _weatherkit_key_cache = load_weatherkit_private_key(WEATHERKIT_KEY_PATH)
            if _weatherkit_key_cache:
                return _weatherkit_key_cache
        except Exception as exc:
            logging.error("Unable to read WEATHERKIT_KEY_PATH %s: %s", WEATHERKIT_KEY_PATH, exc)

    logging.error("WeatherKit private key not configured. Set WEATHERKIT_PRIVATE_KEY or WEATHERKIT_KEY_PATH.")
    return None


def _build_weatherkit_token(now: datetime.datetime) -> Optional[str]:
    global _weatherkit_token, _weatherkit_token_exp

    missing = [
        name
        for name, value in (
            ("WEATHERKIT_TEAM_ID", WEATHERKIT_TEAM_ID),
            ("WEATHERKIT_KEY_ID", WEATHERKIT_KEY_ID),
            ("WEATHERKIT_SERVICE_ID", WEATHERKIT_SERVICE_ID),
        )
        if not value
    ]
    if missing:
        logging.error("Missing WeatherKit configuration: %s", ", ".join(missing))
        return None

    if _weatherkit_token and _weatherkit_token_exp:
        if (_weatherkit_token_exp - now).total_seconds() > 300:
            return _weatherkit_token

    key = _load_weatherkit_private_key()
    if not key:
        return None

    iat = int(now.timestamp())
    exp = int((now + datetime.timedelta(minutes=30)).timestamp())
    try:
        _weatherkit_token = jwt.encode(
            {
                "iss": WEATHERKIT_TEAM_ID,
                "iat": iat,
                "exp": exp,
                "sub": WEATHERKIT_SERVICE_ID,
            },
            key,
            algorithm="ES256",
            headers={"kid": WEATHERKIT_KEY_ID},
        )
        _weatherkit_token_exp = datetime.datetime.fromtimestamp(exp, datetime.timezone.utc)
        return _weatherkit_token
    except Exception as exc:
        logging.error("Unable to sign WeatherKit token: %s", exc)
        return None


def _camel_to_words(text: str) -> str:
    if not text:
        return ""
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", text)
    return words.strip().title()


def _night_icon_name(icon: str) -> str:
    mapping = {
        "Clear": "Clear_night",
        "MostlyClear": "MostlyClear_night",
        "PartlyCloudy": "PartlyCloudy_night",
    }
    return mapping.get(icon, icon)


def _is_night_time(ts: Any, sunrise: Any, sunset: Any) -> bool:
    try:
        ts_val = int(ts)
        sunrise_val = int(sunrise)
        sunset_val = int(sunset)
    except (TypeError, ValueError, OverflowError):
        return False
    return ts_val >= sunset_val or ts_val < sunrise_val


def _sun_windows(daily: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    windows: list[tuple[int, int, int]] = []
    for day in daily:
        sunrise = day.get("sunrise")
        sunset = day.get("sunset")
        anchor = day.get("dt") or sunrise or sunset
        try:
            anchor_val = int(anchor)
            sunrise_val = int(sunrise)
            sunset_val = int(sunset)
        except (TypeError, ValueError, OverflowError):
            continue
        windows.append((anchor_val, sunrise_val, sunset_val))
    return sorted(windows, key=lambda w: w[0])


def _sun_times_for(ts: Any, windows: list[tuple[int, int, int]]) -> tuple[Optional[int], Optional[int]]:
    try:
        ts_val = int(ts)
    except (TypeError, ValueError, OverflowError):
        return None, None

    if not windows:
        return None, None

    best: tuple[int, int, int] | None = None
    best_diff: int | None = None
    for anchor, sunrise, sunset in windows:
        diff = abs(ts_val - anchor)
        if best is None or best_diff is None or diff < best_diff:
            best = (anchor, sunrise, sunset)
            best_diff = diff
    if best is None:
        return None, None
    return best[1], best[2]


def _apply_nighttime_icons(weather: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(weather, dict):
        return weather

    def _apply(entry: dict[str, Any], sunrise_val: Any, sunset_val: Any) -> None:
        if not isinstance(entry, dict):
            return
        weather_list = entry.get("weather") if isinstance(entry.get("weather"), list) else []
        if not weather_list:
            return
        icon = weather_list[0].get("icon")
        if not icon:
            return
        if _is_night_time(entry.get("dt"), sunrise_val, sunset_val):
            night_icon = _night_icon_name(icon)
            if night_icon != icon:
                weather_list[0]["icon"] = night_icon
                entry["weather"] = weather_list

    daily_entries = weather.get("daily") if isinstance(weather.get("daily"), list) else []
    sun_windows = _sun_windows(daily_entries)

    current = weather.get("current")
    if isinstance(current, dict):
        sunrise = current.get("sunrise")
        sunset = current.get("sunset")
        if sunrise is None or sunset is None:
            sunrise, sunset = _sun_times_for(current.get("dt"), sun_windows)
        if sunrise is not None and sunset is not None:
            _apply(current, sunrise, sunset)

    hourly_entries = weather.get("hourly") if isinstance(weather.get("hourly"), list) else []
    for hour in hourly_entries:
        sunrise = hour.get("sunrise")
        sunset = hour.get("sunset")
        if sunrise is None or sunset is None:
            sunrise, sunset = _sun_times_for(hour.get("dt"), sun_windows)
        if sunrise is not None and sunset is not None:
            _apply(hour, sunrise, sunset)

    return weather


def _condition_mapping(condition_code: str) -> tuple[str, str]:
    mapping = {
        "Blizzard": ("Blizzard", "Blizzard"),
        "BlowingSnow": ("Blowing Snow", "HeavySnow"),
        "Breezy": ("Breezy", "Windy"),
        "Clear": ("Clear", "Clear"),
        "Cloudy": ("Cloudy", "Cloudy"),
        "Drizzle": ("Drizzle", "Rain"),
        "Flurries": ("Flurries", "Snow"),
        "Fog": ("Fog", "Fog"),
        "FreezingDrizzle": ("Freezing Drizzle", "Sleet"),
        "FreezingRain": ("Freezing Rain", "Sleet"),
        "Frigid": ("Frigid", "Haze"),
        "Haze": ("Haze", "Haze"),
        "Hazy": ("Hazy", "Haze"),
        "HeavyRain": ("Heavy Rain", "HeavyRain"),
        "HeavySnow": ("Heavy Snow", "HeavySnow"),
        "Hot": ("Hot", "Clear"),
        "Hurricane": ("Hurricane", "Thunderstorms"),
        "IsolatedThunderstorms": ("Isolated Thunderstorms", "Thunderstorms"),
        "MostlyClear": ("Mostly Clear", "MostlyClear"),
        "MostlyCloudy": ("Mostly Cloudy", "Cloudy"),
        "MostlySunny": ("Mostly Sunny", "MostlyClear"),
        "PartlyCloudy": ("Partly Cloudy", "PartlyCloudy"),
        "Rain": ("Rain", "Rain"),
        "ScatteredThunderstorms": ("Scattered Thunderstorms", "Thunderstorms"),
        "Sleet": ("Sleet", "Sleet"),
        "Smoky": ("Smoky", "Haze"),
        "Snow": ("Snow", "Snow"),
        "StrongStorms": ("Strong Storms", "Thunderstorms"),
        "SunShowers": ("Sun Showers", "Rain"),
        "Tornado": ("Tornado", "Thunderstorms"),
        "TropicalStorm": ("Tropical Storm", "Thunderstorms"),
        "Windy": ("Windy", "Windy"),
    }
    default_desc = _camel_to_words(condition_code or "") or "Unknown"
    desc, icon = mapping.get(condition_code, (default_desc, "cloudy"))
    return desc, icon


def _parse_iso_timestamp(value: Any) -> Optional[int]:
    if not isinstance(value, str):
        return None
    try:
        cleaned = value.replace("Z", "+00:00")
        dt_value = datetime.datetime.fromisoformat(cleaned)
        if dt_value.tzinfo is None:
            dt_value = dt_value.replace(tzinfo=datetime.timezone.utc)
        return int(dt_value.timestamp())
    except Exception:
        return None


def _normalise_weatherkit_response(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(data, dict):
        return None

    def _measurement_value(value: Any) -> Optional[float]:
        """Extract a numeric value from WeatherKit measurements.

        Measurements can be plain numbers or objects like {"value": 3.5}.
        Return None if the value cannot be interpreted as a float.
        """

        if isinstance(value, dict):
            value = value.get("value")

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    current_raw = data.get("currentWeather") or {}
    daily_raw = (data.get("forecastDaily") or {}).get("days") or []
    hourly_raw = (data.get("forecastHourly") or {}).get("hours") or []
    alerts_raw = (data.get("weatherAlerts") or {}).get("alerts") or []

    first_day = daily_raw[0] if daily_raw else {}
    sunrise = _parse_iso_timestamp(first_day.get("sunrise"))
    sunset = _parse_iso_timestamp(first_day.get("sunset"))

    desc, icon = _condition_mapping(current_raw.get("conditionCode", ""))
    humidity_raw = current_raw.get("humidity")
    try:
        humidity_pct = int(round(float(humidity_raw) * 100)) if humidity_raw is not None else None
    except Exception:
        humidity_pct = None

    def _to_fahrenheit(value: Optional[float]) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric * 9 / 5 + 32

    current_temp_c = current_raw.get("temperature")
    current_feels_c = current_raw.get("temperatureApparent") or current_temp_c

    current: dict[str, Any] = {
        "temp": _to_fahrenheit(current_temp_c),
        "feels_like": _to_fahrenheit(current_feels_c),
        "weather": [
            {
                "description": desc,
                "icon": icon,
            }
        ],
        "wind_speed": _measurement_value(current_raw.get("windSpeed")),
        "wind_gust": _measurement_value(current_raw.get("windGust")),
        "wind_deg": current_raw.get("windDirection"),
        "humidity": humidity_pct,
        "pressure": _measurement_value(current_raw.get("pressure")),
        "uvi": current_raw.get("uvIndex"),
        "sunrise": sunrise,
        "sunset": sunset,
        "dt": _parse_iso_timestamp(current_raw.get("asOf")),
        "clouds": int(round(float(current_raw.get("cloudCover")) * 100)) if current_raw.get("cloudCover") is not None else None,
    }
    current["pressure_trend"] = _update_pressure_trend(current.get("dt"), current.get("pressure"))

    daily: list[dict[str, Any]] = []
    for day in daily_raw:
        day_desc, day_icon = _condition_mapping(day.get("conditionCode", ""))
        day_dt = _parse_iso_timestamp(day.get("forecastStart"))
        daily.append(
            {
                "dt": day_dt,
                "temp": {
                    "max": _to_fahrenheit(day.get("temperatureMax")),
                    "min": _to_fahrenheit(day.get("temperatureMin")),
                },
                "sunrise": _parse_iso_timestamp(day.get("sunrise")),
                "sunset": _parse_iso_timestamp(day.get("sunset")),
                "pop": day.get("precipitationChance"),
                "weather": [
                    {
                        "description": day_desc,
                        "icon": day_icon,
                    }
                ],
            }
        )

    hourly: list[dict[str, Any]] = []
    for hour in hourly_raw:
        hour_desc, hour_icon = _condition_mapping(hour.get("conditionCode", ""))
        hourly.append(
            {
                "dt": _parse_iso_timestamp(hour.get("forecastStart")),
                "temp": _to_fahrenheit(hour.get("temperature")),
                "feels_like": _to_fahrenheit(hour.get("temperatureApparent") or hour.get("temperature")),
                "pop": hour.get("precipitationChance"),
                "wind_speed": _measurement_value(hour.get("windSpeed")),
                "wind_gust": _measurement_value(hour.get("windGust")),
                "wind_deg": hour.get("windDirection"),
                "uvi": hour.get("uvIndex"),
                "weather": [
                    {
                        "description": hour_desc,
                        "icon": hour_icon,
                    }
                ],
            }
        )

    mapped = {
        "current": current,
        "daily": daily,
        "hourly": hourly,
        "alerts": alerts_raw,
    }
    return _apply_nighttime_icons(mapped)



def _owm_condition_mapping(weather: dict[str, Any]) -> tuple[str, str]:
    weather_id = weather.get("id") if isinstance(weather, dict) else None
    main = (weather.get("main") or "").lower() if isinstance(weather, dict) else ""
    description = (weather.get("description") or "").title() if isinstance(weather, dict) else ""

    icon = "Cloudy"
    if isinstance(weather_id, int):
        if 200 <= weather_id < 300:
            icon = "Thunderstorms"
        elif 300 <= weather_id < 400:
            icon = "Rain"
        elif weather_id == 511:
            icon = "Sleet"
        elif 500 <= weather_id < 600:
            icon = "Rain"
        elif 600 <= weather_id < 700:
            icon = "Snow"
        elif weather_id in {701, 711, 721, 731, 741, 751, 761}:
            icon = "Fog"
        elif 762 <= weather_id <= 781:
            icon = "Thunderstorms"
        elif weather_id == 800:
            icon = "Clear"
        elif weather_id in {801, 802}:
            icon = "PartlyCloudy"
        elif weather_id in {803, 804}:
            icon = "Cloudy"

    if icon == "Cloudy" and main:
        if "thunder" in main:
            icon = "Thunderstorms"
        elif "drizzle" in main or "rain" in main:
            icon = "Rain"
        elif "snow" in main or "sleet" in main:
            icon = "Snow"
        elif main in {"mist", "fog", "haze"}:
            icon = "Fog"
        elif "clear" in main:
            icon = "Clear"
        elif "cloud" in main:
            icon = "Cloudy"

    return description or "Unknown", icon


def _normalise_openweathermap_response(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(data, dict):
        return None

    current_raw = data.get("current") or {}
    daily_raw = data.get("daily") or []
    hourly_raw = data.get("hourly") or []
    alerts_raw = data.get("alerts") or []

    current_weather_list = current_raw.get("weather") if isinstance(current_raw.get("weather"), list) else []
    current_weather = (current_weather_list or [{}])[0]
    current_desc, current_icon = _owm_condition_mapping(current_weather)

    current: dict[str, Any] = {
        "temp": current_raw.get("temp"),
        "feels_like": current_raw.get("feels_like") or current_raw.get("temp"),
        "weather": [
            {
                "description": current_desc,
                "icon": current_icon,
            }
        ],
        "wind_speed": current_raw.get("wind_speed"),
        "wind_gust": current_raw.get("wind_gust"),
        "wind_deg": current_raw.get("wind_deg"),
        "humidity": current_raw.get("humidity"),
        "pressure": current_raw.get("pressure"),
        "uvi": current_raw.get("uvi"),
        "sunrise": current_raw.get("sunrise"),
        "sunset": current_raw.get("sunset"),
        "dt": current_raw.get("dt"),
        "clouds": current_raw.get("clouds"),
    }
    current["pressure_trend"] = _update_pressure_trend(current.get("dt"), current.get("pressure"))

    daily: list[dict[str, Any]] = []
    for day in daily_raw:
        weather_list = day.get("weather") if isinstance(day.get("weather"), list) else []
        weather_entry = (weather_list or [{}])[0]
        desc, icon = _owm_condition_mapping(weather_entry)
        daily.append(
            {
                "dt": day.get("dt"),
                "temp": {"max": (day.get("temp") or {}).get("max"), "min": (day.get("temp") or {}).get("min")},
                "sunrise": day.get("sunrise"),
                "sunset": day.get("sunset"),
                "pop": day.get("pop"),
                "weather": [
                    {
                        "description": desc,
                        "icon": icon,
                    }
                ],
            }
        )

    hourly: list[dict[str, Any]] = []
    for hour in hourly_raw:
        weather_list = hour.get("weather") if isinstance(hour.get("weather"), list) else []
        weather_entry = (weather_list or [{}])[0]
        desc, icon = _owm_condition_mapping(weather_entry)
        hourly.append(
            {
                "dt": hour.get("dt"),
                "temp": hour.get("temp"),
                "feels_like": hour.get("feels_like") or hour.get("temp"),
                "pop": hour.get("pop"),
                "wind_speed": hour.get("wind_speed"),
                "wind_gust": hour.get("wind_gust"),
                "wind_deg": hour.get("wind_deg"),
                "uvi": hour.get("uvi"),
                "weather": [
                    {
                        "description": desc,
                        "icon": icon,
                    }
                ],
            }
        )

    mapped = {
        "current": current,
        "daily": daily,
        "hourly": hourly,
        "alerts": alerts_raw,
    }
    return _apply_nighttime_icons(mapped)


def _fetch_weatherkit(now: datetime.datetime) -> Optional[dict[str, Any]]:
    token = _build_weatherkit_token(now)
    if not token:
        return None

    url = WEATHERKIT_URL_TEMPLATE.format(
        language=WEATHERKIT_LANGUAGE,
        lat=LATITUDE,
        lon=LONGITUDE,
    )

    params = {
        "dataSets": "currentWeather,forecastDaily,forecastHourly,weatherAlerts",
        "timezone": WEATHERKIT_TIMEZONE,
    }

    headers = {
        "Authorization": f"Bearer {token}",
    }

    try:
        r = _session.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        normalized = _normalise_weatherkit_response(r.json())
        return normalized
    except Exception as exc:
        logging.error("Error fetching WeatherKit data: %s", exc)
        return None


def _fetch_openweathermap(now: datetime.datetime) -> Optional[dict[str, Any]]:
    global _owm_backoff_until

    if not OWM_API_KEY:
        logging.debug("OpenWeatherMap API key not configured; skipping secondary source")
        return None

    if _owm_backoff_until and now < _owm_backoff_until:
        remaining = int((_owm_backoff_until - now).total_seconds())
        logging.info(
            "Skipping OpenWeatherMap fetch; still backing off for %ds after rate limit",
            remaining,
        )
        return None

    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "appid": OWM_API_KEY,
        "units": OWM_UNITS,
        "lang": OWM_LANGUAGE,
        "exclude": "minutely",
    }

    try:
        r = _session.get(OWM_API_URL, params=params, timeout=10)
        r.raise_for_status()
        normalized = _normalise_openweathermap_response(r.json())
        return normalized
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            _owm_backoff_until = now + datetime.timedelta(minutes=15)
            logging.warning(
                "OpenWeatherMap rate limited (429); backing off until %s",
                _owm_backoff_until.isoformat(),
            )
        else:
            logging.error("Error fetching OpenWeatherMap data: %s", exc)
        return None
    except Exception as exc:
        logging.error("Error fetching OpenWeatherMap data: %s", exc)
        return None


def fetch_weather(force_refresh: bool = False):
    """Fetch weather from WeatherKit with OpenWeatherMap as a fallback.

    Args:
        force_refresh: If True, bypass the local cache TTL and fetch new data.
    """

    global _weather_cache, _weather_cache_fetched_at
    now = datetime.datetime.now(datetime.timezone.utc)

    cache_age = None
    if _weather_cache and _weather_cache_fetched_at:
        cache_age = (now - _weather_cache_fetched_at).total_seconds()
        if not force_refresh and cache_age < WEATHER_REFRESH_SECONDS:
            logging.debug(
                "Returning cached weather data (age: %.0fs, TTL: %ds)", cache_age, WEATHER_REFRESH_SECONDS
            )
            return _weather_cache

    normalized = _fetch_weatherkit(now)
    if normalized is None:
        normalized = _fetch_openweathermap(now)

    if normalized is not None:
        _weather_cache = normalized
        _weather_cache_fetched_at = now
        return _weather_cache

    # If both sources fail, fall back to stale cached data instead of returning
    # nothing (and hammering the APIs on the next attempt).
    if _weather_cache and _weather_cache_fetched_at:
        cache_age = cache_age if cache_age is not None else (now - _weather_cache_fetched_at).total_seconds()
        logging.info(
            "Returning stale weather cache after fetch errors (age: %.0fs)", cache_age
        )
        return _weather_cache

    return None


def get_weather_cache_timestamp() -> Optional[datetime.datetime]:
    """Return the UTC timestamp for the most recent successful weather fetch."""

    return _weather_cache_fetched_at


# -----------------------------------------------------------------------------
# NHL — Blackhawks
# -----------------------------------------------------------------------------
def fetch_blackhawks_next_game():
    try:
        r = _session.get(NHL_API_URL, timeout=10, headers=NHL_HEADERS)
        r.raise_for_status()
        games = r.json().get("games", [])
        fut   = [g for g in games if g.get("gameState") == "FUT"]

        for g in fut:
            if not g.get("startTimeCentral"):
                utc = g.get("startTimeUTC")
                if utc:
                    dt = datetime.datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ")
                    dt = dt.replace(tzinfo=pytz.utc).astimezone(CENTRAL_TIME)
                    g["startTimeCentral"] = dt.strftime("%I:%M %p").lstrip("0")
                else:
                    g["startTimeCentral"] = "TBD"

        fut.sort(key=lambda g: g.get("gameDate", ""))
        return fut[0] if fut else None

    except Exception as e:
        logging.error("Error fetching next Blackhawks game: %s", e)
        return None


def _extract_team_value(team, *keys):
    """Return the first string value found for the provided keys."""
    if not isinstance(team, dict):
        return ""
    for key in keys:
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for subkey in ("default", "en", "fullName"):
                subval = value.get(subkey)
                if isinstance(subval, str) and subval.strip():
                    return subval.strip()
    return ""


def _is_blackhawks_team(team):
    if not isinstance(team, dict):
        return False
    if isinstance(team.get("team"), dict):
        team = team["team"]
    team_id = team.get("id") or team.get("teamId")
    if team_id == NHL_TEAM_ID:
        return True
    name = _extract_team_value(team, "commonName", "name", "teamName", "clubName")
    return "blackhawks" in name.lower() if name else False


def _team_id(team):
    if not isinstance(team, dict):
        return None
    if isinstance(team.get("team"), dict):
        return _team_id(team["team"])
    return team.get("id") or team.get("teamId")


def _same_game(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    for key in ("id", "gamePk", "gameId", "gameUUID"):
        av = a.get(key)
        bv = b.get(key)
        if av and bv and av == bv:
            return True
    a_date = a.get("gameDate")
    b_date = b.get("gameDate")
    if a_date and b_date and a_date == b_date:
        a_home = _team_id(a.get("homeTeam") or a.get("home_team") or {})
        b_home = _team_id(b.get("homeTeam") or b.get("home_team") or {})
        a_away = _team_id(a.get("awayTeam") or a.get("away_team") or {})
        b_away = _team_id(b.get("awayTeam") or b.get("away_team") or {})
        return a_home == b_home and a_away == b_away
    return False


# -----------------------------------------------------------------------------
# NBA — Chicago Bulls
# -----------------------------------------------------------------------------
_BULLS_TEAM_ID = str(NBA_TEAM_ID)
_BULLS_TRICODE = (NBA_TEAM_TRICODE or "CHI").upper()
_NBA_LOOKBACK_DAYS = 7
# ESPN's NBA scoreboard only surfaces games within roughly the next few months.
# A 30-day window is too short when the Bulls' next home date follows a long
# road swing, so extend the lookahead to capture those games as well.
_NBA_LOOKAHEAD_DAYS = 120


def _parse_nba_datetime(value):
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            parsed = datetime.datetime.strptime(text, fmt)
        except Exception:
            continue
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        return parsed.astimezone(CENTRAL_TIME)
    try:
        parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(CENTRAL_TIME)


def _copy_nba_team(entry):
    if not isinstance(entry, dict):
        return {}
    cloned = dict(entry)
    team_info = cloned.get("team")
    if isinstance(team_info, dict):
        cloned["team"] = dict(team_info)
    return cloned


def _augment_nba_game(game):
    if not isinstance(game, dict):
        return None
    cloned = dict(game)
    teams = cloned.get("teams")
    if isinstance(teams, dict):
        cloned_teams = {}
        for side in ("home", "away"):
            cloned_teams[side] = _copy_nba_team(teams.get(side) or {})
        cloned["teams"] = cloned_teams
    start_local = cloned.get("_start_local")
    if not isinstance(start_local, datetime.datetime):
        start_local = _parse_nba_datetime(cloned.get("gameDate"))
    if isinstance(start_local, datetime.datetime):
        cloned["_start_local"] = start_local
        cloned["officialDate"] = start_local.date().isoformat()
    else:
        date_text = (cloned.get("officialDate") or cloned.get("gameDate") or "").strip()
        cloned["officialDate"] = date_text[:10]
    return cloned


def _is_bulls_team(entry):
    if not isinstance(entry, dict):
        return False
    team_info = entry.get("team") if isinstance(entry.get("team"), dict) else entry
    team_id = str(team_info.get("id") or team_info.get("teamId") or "")
    if team_id and team_id == _BULLS_TEAM_ID:
        return True
    tri = str(team_info.get("triCode") or team_info.get("abbreviation") or "").upper()
    return tri == _BULLS_TRICODE if tri else False


def _is_bulls_game(game):
    if not isinstance(game, dict):
        return False
    teams = game.get("teams") or {}
    return _is_bulls_team(teams.get("home")) or _is_bulls_team(teams.get("away"))


def _nba_game_state(game):
    status = game.get("status") or {}
    abstract = str(status.get("abstractGameState") or "").lower()
    if abstract:
        return abstract
    detailed = str(status.get("detailedState") or "").lower()
    if "final" in detailed:
        return "final"
    if "live" in detailed or "progress" in detailed:
        return "live"
    if "preview" in detailed or "schedule" in detailed or "pregame" in detailed:
        return "preview"
    code = str(status.get("statusCode") or "")
    if code == "3":
        return "final"
    if code == "2":
        return "live"
    if code == "1":
        return "preview"
    return detailed


def _get_bulls_games_for_day(day):
    try:
        games = _nba_fetch_games_for_date(day)
    except Exception as exc:
        logging.error("Failed to fetch NBA scoreboard for %s: %s", day, exc)
        return []
    results = []
    for game in games or []:
        if not _is_bulls_game(game):
            continue
        augmented = _augment_nba_game(game)
        if augmented:
            results.append(augmented)
    return results


def _future_bulls_games(days_forward):
    today = datetime.datetime.now(CENTRAL_TIME).date()
    for delta in range(0, days_forward + 1):
        day = today + datetime.timedelta(days=delta)
        for game in _get_bulls_games_for_day(day):
            yield game


def _past_bulls_games(days_back):
    today = datetime.datetime.now(CENTRAL_TIME).date()
    for delta in range(0, days_back + 1):
        day = today - datetime.timedelta(days=delta)
        games = _get_bulls_games_for_day(day)
        for game in reversed(games):
            yield game


def fetch_bulls_next_game():
    try:
        for game in _future_bulls_games(_NBA_LOOKAHEAD_DAYS):
            if _nba_game_state(game) in {"preview", "scheduled", "pregame"}:
                return game
    except Exception as exc:
        logging.error("Error fetching next Bulls game: %s", exc)
    return None


def fetch_bulls_next_home_game():
    try:
        next_game = fetch_bulls_next_game()
        fallback_game = None
        skipped_next_home = False
        for game in _future_bulls_games(_NBA_LOOKAHEAD_DAYS):
            teams = game.get("teams") or {}
            if not _is_bulls_team(teams.get("home")):
                continue

            if next_game and _same_game(game, next_game):
                skipped_next_home = True
                continue

            state = _nba_game_state(game)
            if state in {"preview", "scheduled", "pregame"}:
                return game
            if fallback_game is None and state not in {"final", "postponed"}:
                fallback_game = game
        return fallback_game if fallback_game or not skipped_next_home else None
    except Exception as exc:
        logging.error("Error fetching next Bulls home game: %s", exc)
    return None


def fetch_bulls_last_game():
    try:
        for game in _past_bulls_games(_NBA_LOOKBACK_DAYS):
            if _nba_game_state(game) == "final":
                return game
    except Exception as exc:
        logging.error("Error fetching last Bulls game: %s", exc)
    return None


def fetch_bulls_live_game():
    try:
        for game in _future_bulls_games(0):
            if _nba_game_state(game) == "live":
                return game
        for game in _past_bulls_games(1):
            if _nba_game_state(game) == "live":
                return game
    except Exception as exc:
        logging.error("Error fetching live Bulls game: %s", exc)
    return None


# -----------------------------------------------------------------------------
# Team standings — NFL / NHL / NBA
# -----------------------------------------------------------------------------
def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return value


def _extract_rank_value(entry: dict, *keys) -> Optional[int]:
    """Return the first positive integer value found for any key in entry."""

    for key in keys:
        if not isinstance(entry, dict) or key not in entry:
            continue

        value = _safe_int(entry.get(key))
        if isinstance(value, (int, float)) and value > 0:
            return int(value)

    return None


def _format_streak_code(prefix, count):
    try:
        c = int(count)
    except Exception:
        return "-"
    if c <= 0:
        return "-"
    return f"{prefix}{c}"


def _format_streak_from_dict(streak_blob):
    if not isinstance(streak_blob, dict):
        return "-"
    prefix = streak_blob.get("type") or streak_blob.get("streakType")
    count = streak_blob.get("count") or streak_blob.get("streakNumber")
    if isinstance(prefix, str):
        prefix = prefix[:1].upper()
    return _format_streak_code(prefix or "-", count)


def _build_split_record(split_type, wins, losses):
    return {"type": split_type, "wins": wins, "losses": losses}


def _extract_split_records(**kwargs):
    splits = []
    for key, value in kwargs.items():
        if not value:
            continue
        wins = value.get("wins")
        losses = value.get("losses")
        if wins is None and losses is None:
            continue
        splits.append(_build_split_record(key, wins, losses))
    return splits


def _empty_standings_record(team_abbr: str) -> dict:
    """Return a placeholder standings structure so screens can still render."""

    return {
        "leagueRecord": {"wins": "-", "losses": "-", "pct": "-"},
        "divisionRank": "-",
        "divisionGamesBack": "-",
        "wildCardGamesBack": None,
        "streak": {"streakCode": "-"},
        "records": {"splitRecords": []},
        "points": None,
        "team": team_abbr,
    }


def _statsapi_available() -> bool:
    """Lightweight DNS check so we avoid slow statsapi fallbacks when DNS fails."""

    global _statsapi_dns_available, _statsapi_dns_checked_at

    now = time.time()
    if _statsapi_dns_checked_at and (now - _statsapi_dns_checked_at) < _STATSAPI_DNS_RECHECK_SECONDS:
        return bool(_statsapi_dns_available)

    try:
        socket.getaddrinfo("statsapi.web.nhl.com", 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        logging.debug("NHL statsapi DNS lookup failed: %s", exc)
        _statsapi_dns_available = False
    except Exception as exc:
        logging.debug("Unexpected error checking NHL statsapi DNS: %s", exc)
        _statsapi_dns_available = False
    else:
        _statsapi_dns_available = True

    _statsapi_dns_checked_at = now
    return bool(_statsapi_dns_available)


def _fetch_nfl_team_standings(team_abbr: str):
    try:
        url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/standings.csv"
        resp = _session.get(url, timeout=10)
        resp.raise_for_status()
        entries = [row for row in csv.DictReader(io.StringIO(resp.text)) if row.get("team") == team_abbr]
        if not entries:
            logging.warning("Team %s not found in NFL standings", team_abbr)
            return None

        latest = max(entries, key=lambda r: r.get("season", "0"))
        wins = _safe_int(latest.get("wins"))
        losses = _safe_int(latest.get("losses"))
        ties = _safe_int(latest.get("ties"))

        record = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "pct": latest.get("pct"),
        }

        return {
            "leagueRecord": record,
            "divisionRank": latest.get("div_rank") or latest.get("divRank") or "-",
            "division": latest.get("division"),
            "streak": {"streakCode": "-"},
            "records": {"splitRecords": []},
        }
    except Exception as exc:
        logging.error("Error fetching NFL standings for %s: %s", team_abbr, exc)
        return None


def fetch_bears_standings():
    return _fetch_nfl_team_standings("CHI")


def _fetch_nhl_team_standings(team_abbr: str):
    cache_key = f"nhl:{team_abbr}"
    cached, hit = _get_cached_team_standings(cache_key)
    if hit:
        logging.debug("Using cached NHL standings for %s", team_abbr)
        return cached

    try:
        url = "https://api-web.nhle.com/v1/standings/now"
        resp = _session.get(url, timeout=_TEAM_STANDINGS_TIMEOUT, headers=NHL_HEADERS)
        resp.raise_for_status()
        payload = resp.json() or {}
        standings = payload.get("standings", []) or []
        entry = None
        for row in standings:
            abbr = row.get("teamAbbrev")
            if isinstance(abbr, dict):
                abbr = abbr.get("default") or abbr.get("alternate")
            if abbr == team_abbr:
                entry = row
                break
        if entry:
            record = {
                "wins": _safe_int(entry.get("wins")),
                "losses": _safe_int(entry.get("losses")),
                "ot": _safe_int(entry.get("otLosses")),
                "pct": entry.get("pointsPctg"),
            }

            home = {"wins": entry.get("homeWins"), "losses": entry.get("homeLosses")}
            away = {"wins": entry.get("roadWins"), "losses": entry.get("roadLosses")}
            l10 = {"wins": entry.get("l10Wins"), "losses": entry.get("l10Losses")}
            division = {
                "wins": entry.get("divisionWins"),
                "losses": entry.get("divisionLosses"),
            }
            conference = {
                "wins": entry.get("conferenceWins"),
                "losses": entry.get("conferenceLosses"),
            }

            splits = _extract_split_records(
                home=home, away=away, lastTen=l10, division=division, conference=conference
            )

            streak_code = entry.get("streakCode") or _format_streak_code(entry.get("streakType"), entry.get("streakNumber"))

            # Properly handle division and conference rankings
            # Include newer sequence fields in addition to the older Rank values.
            div_rank = _extract_rank_value(
                entry,
                "divisionSeq",
                "divisionSequence",
                "divisionSequenceNumber",
                "divisionRank",
                "divRank",
            )

            conf_rank = _extract_rank_value(
                entry,
                "conferenceSeq",
                "conferenceSequence",
                "conferenceSequenceNumber",
                "conferenceRank",
                "confRank",
            )

            payload = {
                "leagueRecord": record,
                "divisionRank": div_rank,
                "divisionGamesBack": None,
                "wildCardGamesBack": None,
                "streak": {"streakCode": streak_code or "-"},
                "records": {"splitRecords": splits},
                "points": entry.get("points"),
                "conferenceRank": conf_rank,
                "conferenceName": entry.get("conferenceName")
                or entry.get("conferenceAbbrev"),
                "divisionName": entry.get("divisionName")
                or entry.get("divisionAbbrev"),
            }
            _store_team_standings_cache(cache_key, payload, success=True)
            return payload
        logging.warning("Team %s not found in NHL standings; trying fallback", team_abbr)
    except Exception as exc:
        logging.error("Error fetching NHL standings for %s: %s", team_abbr, exc)
    fallback = _fetch_nhl_team_standings_espn(team_abbr)
    if fallback:
        _store_team_standings_cache(cache_key, fallback, success=True)
        return fallback
    if not _statsapi_available():
        logging.info("Skipping statsapi NHL standings fallback due to DNS failure")
        _store_team_standings_cache(cache_key, None, success=False)
        return None
    statsapi = _fetch_nhl_team_standings_statsapi(team_abbr)
    _store_team_standings_cache(cache_key, statsapi, success=bool(statsapi))
    return statsapi


def fetch_blackhawks_standings():
    return _fetch_nhl_team_standings("CHI")


def _fetch_nhl_team_standings_espn(team_abbr: str):
    """Fallback to ESPN standings when NHL endpoints are unavailable."""

    def _iter_entries(node):
        if not isinstance(node, dict):
            return
        standings = node.get("standings", {})
        for entry in standings.get("entries", []) or []:
            yield entry
        for child in node.get("children", []) or []:
            yield from _iter_entries(child)

    def _stat(stats, name, default=None):
        for stat in stats or []:
            if stat.get("name") == name:
                if stat.get("value") is not None:
                    return stat.get("value")
                return stat.get("displayValue") or stat.get("summary")
        return default

    try:
        url = "https://site.web.api.espn.com/apis/v2/sports/hockey/nhl/standings"
        resp = _session.get(url, timeout=_TEAM_STANDINGS_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or {}

        for entry in _iter_entries(data):
            team = entry.get("team", {}) or {}
            if (team.get("abbreviation") or team.get("shortDisplayName")) != team_abbr:
                continue

            stats = entry.get("stats") or []
            streak_code = _stat(stats, "streak", "-")
            pct = _stat(stats, "winPercent")
            try:
                pct = float(pct)
            except Exception:
                pass

            logging.info("Using ESPN NHL standings fallback for %s", team_abbr)

            division_rank = _stat(stats, "divisionStanding")
            if division_rank in (None, ""):
                division_rank = _stat(stats, "divisionRank")
            if division_rank in (None, ""):
                division_rank = _stat(stats, "divisionPlace")
            if division_rank in (None, ""):
                division_rank = _stat(stats, "playoffSeed")

            conference_rank = _stat(stats, "conferenceStanding")
            if conference_rank in (None, ""):
                conference_rank = _stat(stats, "conferenceRank")
            if conference_rank in (None, ""):
                conference_rank = _stat(stats, "playoffSeed")

            division_name = (
                _stat(stats, "divisionAbbreviation")
                or _stat(stats, "divisionName")
                or _stat(stats, "divisionDisplayName")
            )

            conference_name = (
                _stat(stats, "conferenceAbbreviation")
                or _stat(stats, "conferenceName")
                or _stat(stats, "conferenceDisplayName")
            )

            return {
                "leagueRecord": {
                    "wins": _safe_int(_stat(stats, "wins")),
                    "losses": _safe_int(_stat(stats, "losses")),
                    "ot": _safe_int(_stat(stats, "otLosses")),
                    "pct": pct,
                },
                "divisionRank": division_rank,
                "divisionGamesBack": _stat(stats, "divisionGamesBehind"),
                "wildCardGamesBack": None,
                "streak": {"streakCode": streak_code or "-"},
                "records": {"splitRecords": []},
                "points": _stat(stats, "points"),
                "conferenceRank": conference_rank,
                "conferenceName": conference_name,
                "divisionName": division_name,
            }
    except Exception as exc:
        logging.error("Error fetching NHL standings (ESPN fallback) for %s: %s", team_abbr, exc)
    return None


def _fetch_nhl_team_standings_statsapi(team_abbr: str):
    try:
        url = "https://statsapi.web.nhl.com/api/v1/standings"
        resp = _session.get(url, timeout=_TEAM_STANDINGS_TIMEOUT, headers=NHL_HEADERS)
        resp.raise_for_status()
        payload = resp.json() or {}
        for record in payload.get("records", []) or []:
            for team in record.get("teamRecords", []) or []:
                info = team.get("team", {}) or {}
                abbr = info.get("abbreviation") or info.get("teamName")
                if abbr != team_abbr:
                    continue

                league_record = team.get("leagueRecord", {}) or {}
                streak = team.get("streak", {}) or {}
                streak_code = streak.get("streakCode") or _format_streak_code(
                    streak.get("streakType"), streak.get("streakNumber")
                )

                split_records = []
                for split in (team.get("records") or {}).get("splitRecords", []) or []:
                    wins = split.get("wins")
                    losses = split.get("losses")
                    if wins is None and losses is None:
                        continue
                    split_records.append(
                        {"type": split.get("type"), "wins": wins, "losses": losses}
                    )

            logging.info("Using statsapi NHL standings fallback for %s", team_abbr)
            return {
                "leagueRecord": {
                    "wins": _safe_int(league_record.get("wins")),
                    "losses": _safe_int(league_record.get("losses")),
                    "ot": _safe_int(league_record.get("ot")),
                    "pct": league_record.get("pct") or league_record.get("pointsPercentage"),
                },
                "divisionRank": team.get("divisionRank"),
                "divisionGamesBack": team.get("divisionGamesBack"),
                "wildCardGamesBack": team.get("wildCardRank"),
                "streak": {"streakCode": streak_code or "-"},
                "records": {"splitRecords": split_records},
                "points": team.get("points"),
                "conferenceRank": team.get("conferenceRank"),
                "conferenceName": (team.get("conference") or {}).get("name")
                or (team.get("conference") or {}).get("abbreviation"),
                "divisionName": (team.get("division") or {}).get("name")
                or (team.get("division") or {}).get("abbreviation"),
            }
        logging.error("Team %s not found in NHL standings (statsapi fallback)", team_abbr)
    except Exception as exc:
        logging.error("Error fetching NHL standings (statsapi) for %s: %s", team_abbr, exc)
    return None


def _fetch_nba_team_standings(team_tricode: str):
    cache_key = f"nba:{team_tricode}"
    cached, hit = _get_cached_team_standings(cache_key)
    if hit:
        logging.debug("Using cached NBA standings for %s", team_tricode)
        return cached

    def _load_json() -> Optional[dict]:
        endpoints = {
            "cdn": "https://cdn.nba.com/static/json/liveData/standings/league.json",
            "espn": "https://site.web.api.espn.com/apis/v2/sports/basketball/nba/standings",
        }

        headers = {
            "Origin": "https://www.nba.com",
            "Referer": "https://www.nba.com/",
            "User-Agent": "Mozilla/5.0 (compatible; DeskDisplay/1.0)",
        }

        # Try the NBA CDN first because it matches the data shape the rest of the
        # parsing code expects. If it is blocked (403) or missing (404), fall back
        # to ESPN without emitting warnings on every launch.
        try:
            resp = _session.get(
                endpoints["cdn"],
                timeout=_TEAM_STANDINGS_TIMEOUT,
                headers=headers,
            )
            if resp.status_code == 403:
                logging.debug(
                    "NBA CDN standings blocked (HTTP 403); using ESPN fallback instead"
                )
            else:
                resp.raise_for_status()
                return resp.json() or {}
        except Exception as exc:
            logging.debug("NBA CDN standings unavailable: %s", exc)

        logging.debug("Using ESPN NBA standings fallback")
        return _fetch_nba_team_standings_espn()

    payload = _load_json() or {}
    teams = payload.get("league", {}).get("standard", {}).get("teams", [])

    try:
        entry = next((row for row in teams if row.get("teamTricode") == team_tricode), None)
        if entry:
            record = {
                "wins": _safe_int(entry.get("wins") or entry.get("win")),
                "losses": _safe_int(entry.get("losses") or entry.get("loss")),
                "pct": entry.get("winPct"),
            }

            streak_blob = entry.get("streak") or {}
            streak_code = entry.get("streakText") or entry.get("streakCode")
            if not streak_code:
                streak_code = _format_streak_from_dict(streak_blob)

            splits = _extract_split_records(
                lastTen=entry.get("lastTen"),
                home=entry.get("home"),
                away=entry.get("away"),
            )

            conference = entry.get("conference") or {}
            conference_name = conference.get("name") or conference.get("displayName")
            conference_rank = conference.get("rank")

            payload = {
                "leagueRecord": record,
                "divisionRank": entry.get("divisionRank")
                or (entry.get("teamDivision") or {}).get("rank"),
                "divisionGamesBack": entry.get("gamesBehind")
                or entry.get("gamesBehindDivision"),
                "wildCardGamesBack": None,
                "streak": {"streakCode": streak_code},
                "records": {"splitRecords": splits},
                "conferenceRank": conference_rank,
                "conferenceName": conference_name,
            }
            _store_team_standings_cache(cache_key, payload, success=True)
            return payload

        if teams:
            logging.warning("Team %s not found in NBA standings", team_tricode)
    except Exception as exc:
        logging.error("Error fetching NBA standings for %s: %s", team_tricode, exc)

    fallback = _fetch_nba_team_standings_espn()
    if fallback:
        _store_team_standings_cache(cache_key, fallback, success=True)
        return fallback

    logging.warning("Using placeholder NBA standings for %s due to fetch errors", team_tricode)
    placeholder = _empty_standings_record(team_tricode)
    _store_team_standings_cache(cache_key, placeholder, success=False)
    return placeholder


def fetch_bulls_standings():
    return _fetch_nba_team_standings(NBA_TEAM_TRICODE)


def _fetch_nba_team_standings_espn() -> Optional[dict]:
    """Fallback for NBA standings using ESPN when NBA CDN blocks access."""

    def _iter_entries(node):
        if not isinstance(node, dict):
            return
        standings = node.get("standings", {})
        for entry in standings.get("entries", []) or []:
            yield entry
        for child in node.get("children", []) or []:
            yield from _iter_entries(child)

    def _stat(stats, name, default=None):
        for stat in stats or []:
            if stat.get("name") == name:
                if stat.get("value") is not None:
                    return stat.get("value")
                return stat.get("displayValue") or stat.get("summary")
        return default

    try:
        url = "https://site.web.api.espn.com/apis/v2/sports/basketball/nba/standings"
        resp = _session.get(url, timeout=_TEAM_STANDINGS_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or {}

        for entry in _iter_entries(data):
            team = entry.get("team", {}) or {}
            if (team.get("abbreviation") or team.get("shortDisplayName")) != NBA_TEAM_TRICODE:
                continue

            stats = entry.get("stats") or []
            streak_code = _stat(stats, "streak", "-")
            pct = _stat(stats, "winPercent")
            try:
                pct = float(pct)
            except Exception:
                pass

            logging.debug("Using ESPN NBA standings fallback for %s", NBA_TEAM_TRICODE)

            division_rank = _stat(stats, "divisionStanding")
            if division_rank in (None, ""):
                division_rank = _stat(stats, "divisionRank")
            if division_rank in (None, ""):
                division_rank = _stat(stats, "divisionPlace")
            if division_rank in (None, ""):
                division_rank = _stat(stats, "playoffSeed")

            return {
                "leagueRecord": {
                    "wins": _safe_int(_stat(stats, "wins")),
                    "losses": _safe_int(_stat(stats, "losses")),
                    "pct": pct,
                },
                "divisionRank": division_rank,
                "divisionGamesBack": _stat(stats, "gamesBehind"),
                "wildCardGamesBack": None,
                "streak": {"streakCode": streak_code or "-"},
                "records": {"splitRecords": []},
                "conferenceRank": _stat(stats, "playoffSeed"),
            }
    except Exception as exc:
        logging.error("Error fetching NBA standings (ESPN fallback) for %s: %s", NBA_TEAM_TRICODE, exc)
    return None


def fetch_blackhawks_next_home_game():
    try:
        next_game = fetch_blackhawks_next_game()
        r = _session.get(NHL_API_URL, timeout=10, headers=NHL_HEADERS)
        r.raise_for_status()
        games = r.json().get("games", [])
        home  = []
        skipped_duplicate = False

        for g in games:
            if g.get("gameState") != "FUT":
                continue
            team = g.get("homeTeam", {}) or g.get("home_team", {})
            if _is_blackhawks_team(team):
                if next_game and _same_game(next_game, g):
                    skipped_duplicate = True
                    continue
                utc = g.get("startTimeUTC")
                if utc:
                    dt = datetime.datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ")
                    dt = dt.replace(tzinfo=pytz.utc).astimezone(CENTRAL_TIME)
                    g["startTimeCentral"] = dt.strftime("%I:%M %p").lstrip("0")
                else:
                    g["startTimeCentral"] = "TBD"
                home.append(g)

        home.sort(key=lambda g: g.get("gameDate", ""))
        if not home:
            if skipped_duplicate:
                logging.info(
                    "Next home Blackhawks game matches the next scheduled game; suppressing duplicate screen."
                )
            else:
                logging.info("No upcoming additional Blackhawks home games were found.")
        return home[0] if home else None

    except Exception as e:
        logging.error("Error fetching next home Blackhawks game: %s", e)
        return None


def fetch_blackhawks_last_game():
    try:
        r = _session.get(NHL_API_URL, timeout=10, headers=NHL_HEADERS)
        r.raise_for_status()
        data  = r.json()
        games = []

        if "dates" in data:
            for di in data["dates"]:
                games.extend(di.get("games", []))
        else:
            games = data.get("games", [])

        offs = [g for g in games if g.get("gameState") == "OFF"]
        if offs:
            offs.sort(key=lambda g: g.get("gameDate", ""))
            return offs[-1]
        return None

    except Exception as e:
        logging.error("Error fetching last Blackhawks game: %s", e)
        return None


def fetch_blackhawks_live_game():
    try:
        r = _session.get(NHL_API_URL, timeout=10, headers=NHL_HEADERS)
        r.raise_for_status()
        games = r.json().get("games", [])
        for g in games:
            state = g.get("gameState", "").lower()
            if state in ("live", "in progress"):
                if not g.get("startTimeCentral"):
                    utc = g.get("startTimeUTC")
                    if utc:
                        dt = datetime.datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ")
                        dt = dt.replace(tzinfo=pytz.utc).astimezone(CENTRAL_TIME)
                        g["startTimeCentral"] = dt.strftime("%I:%M %p").lstrip("0")
                    else:
                        g["startTimeCentral"] = "TBD"
                return g
        return None

    except Exception as e:
        logging.error("Error fetching live Blackhawks game: %s", e)
        return None


# -----------------------------------------------------------------------------
# MLB — schedule helper + Cubs/Sox wrappers
# -----------------------------------------------------------------------------
def _fetch_mlb_schedule(team_id):
    try:
        today = datetime.datetime.now(CENTRAL_TIME).date()
        start = today - datetime.timedelta(days=3)
        end   = today + datetime.timedelta(days=30)

        url = (
            f"{MLB_API_URL}"
            f"?sportId=1&teamId={team_id}"
            f"&startDate={start}&endDate={end}&hydrate=team,linescore"
        )
        r = _session.get(url, timeout=10)
        r.raise_for_status()
        data   = r.json()
        result = {
            "next_game": None,
            "next_home_game": None,
            "live_game": None,
            "last_game": None,
        }
        finished = []
        home_candidates = []
        skipped_home_duplicate = False
        team_id_int = int(team_id)

        for di in data.get("dates", []):
            day = datetime.datetime.strptime(di["date"], "%Y-%m-%d").date()
            for g in di.get("games", []):
                # Convert UTC to Central
                utc = g.get("gameDate")
                local_dt = None
                if utc:
                    dt = datetime.datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ")
                    dt = dt.replace(tzinfo=pytz.utc).astimezone(CENTRAL_TIME)
                    g["startTimeCentral"] = dt.strftime("%I:%M %p").lstrip("0")
                    local_dt = dt
                else:
                    g["startTimeCentral"] = "TBD"
                    try:
                        local_dt = CENTRAL_TIME.localize(
                            datetime.datetime.combine(day, datetime.time(12, 0))
                        )
                    except Exception:
                        local_dt = None

                # Determine game state
                status      = g.get("status", {})
                code        = status.get("statusCode", "").upper()
                abstract    = status.get("abstractGameState", "").lower()
                detailed    = status.get("detailedState", "").lower()

                # Track upcoming home games for dedicated screen
                home_team_id = (
                    ((g.get("teams") or {}).get("home") or {}).get("team", {})
                ).get("id")
                is_home_game = False
                try:
                    is_home_game = int(home_team_id) == team_id_int
                except Exception:
                    is_home_game = False

                if is_home_game and local_dt and local_dt.date() >= today:
                    is_scheduled = code in {"S", "I"} or abstract in {
                        "preview",
                        "scheduled",
                        "live",
                    } or "progress" in detailed
                    is_postponed = any(
                        kw in detailed for kw in ("postponed", "suspended")
                    )
                    if is_scheduled and not is_postponed:
                        home_candidates.append((local_dt, g))

                # Live game
                if code == "I" or abstract == "live" or "progress" in detailed:
                    result["live_game"] = g

                # Next game (today scheduled)
                if day == today and (code == "S" or abstract in ("preview","scheduled")):
                    result["next_game"] = g

                # Finished up to today
                if day <= today and code not in ("S","I") and abstract not in ("preview","scheduled","live"):
                    finished.append(g)

        # Fallback next future
        if not result["next_game"]:
            for di in data.get("dates", []):
                day = datetime.datetime.strptime(di["date"], "%Y-%m-%d").date()
                if day > today:
                    for g in di.get("games", []):
                        status   = g.get("status", {})
                        code2    = status.get("statusCode", "").upper()
                        abs2     = status.get("abstractGameState", "").lower()
                        if code2 == "S" or abs2 in ("preview","scheduled"):
                            result["next_game"] = g
                            break
                    if result["next_game"]:
                        break

        # Pick earliest upcoming home game
        if home_candidates:
            home_candidates.sort(key=lambda item: item[0])

            next_game_pk = None
            if result["next_game"]:
                next_game_pk = result["next_game"].get("gamePk")

            for _, home_game in home_candidates:
                # If the upcoming game is already a home game, skip duplicating it
                if next_game_pk and home_game.get("gamePk") == next_game_pk:
                    skipped_home_duplicate = True
                    continue
                if (
                    result["next_game"]
                    and home_game.get("gameDate") == result["next_game"].get("gameDate")
                ):
                    skipped_home_duplicate = True
                    continue

                result["next_home_game"] = home_game
                break

        if not result["next_home_game"] and skipped_home_duplicate:
            logging.info(
                "Next MLB home game for team %s matches the upcoming game; suppressing duplicate home screen.",
                team_id,
            )

        # Pick last finished
        if finished:
            finished.sort(key=lambda x: x.get("officialDate",""))
            result["last_game"] = finished[-1]

        return result

    except Exception as e:
        logging.error("Error fetching MLB schedule for %s: %s", team_id, e)
        return {
            "next_game": None,
            "next_home_game": None,
            "live_game": None,
            "last_game": None,
        }


def fetch_cubs_games():
    return _fetch_mlb_schedule(MLB_CUBS_TEAM_ID)


def fetch_sox_games():
    return _fetch_mlb_schedule(MLB_SOX_TEAM_ID)


# -----------------------------------------------------------------------------
# MLB — standings helper + Cubs/Sox wrappers
# -----------------------------------------------------------------------------
def _fetch_mlb_standings(league_id, division_id, team_id):
    try:
        url = (
            "https://statsapi.mlb.com/api/v1/standings"
            f"?season=2025&leagueId={league_id}&divisionId={division_id}"
        )
        r = _session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        for rec in data.get("records", []):
            for tr in rec.get("teamRecords", []):
                if tr.get("team", {}).get("id") == int(team_id):
                    return tr

        logging.warning("Team %s not found in standings (L%d/D%d)", team_id, league_id, division_id)
        return None

    except Exception as e:
        logging.error("Error fetching standings for team %s: %s", team_id, e)
        return None


def fetch_cubs_standings():
    return _fetch_mlb_standings(104, 205, MLB_CUBS_TEAM_ID)


def fetch_sox_standings():
    return _fetch_mlb_standings(103, 202, MLB_SOX_TEAM_ID)


# -----------------------------------------------------------------------------
# AHL — Chicago Wolves schedule + scores (HockeyTech / AHL stats feed)
# -----------------------------------------------------------------------------
_AHL_SEASON_CACHE: Optional[str] = None
_AHL_DEFAULT_BASE = "https://lscluster.hockeytech.com/feed/index.php"


def _ahl_endpoint() -> str:
    base = (AHL_API_BASE_URL or "").strip()
    if not base:
        return _AHL_DEFAULT_BASE

    if not base.startswith(("http://", "https://")):
        base = f"https://{base.lstrip('/')}"

    if "?" in base or base.endswith(".php"):
        return base

    trimmed = base.rstrip("/")
    if trimmed.endswith("/feed"):
        return f"{trimmed}/index.php"
    return f"{trimmed}/index.php"


def _sanitize_ahl_payload(raw_text: str) -> str:
    """Remove common HockeyTech JSON padding and whitespace guards."""
    if not isinstance(raw_text, str):
        return ""

    cleaned = raw_text.lstrip("\ufeff\r\n\t ")

    # Some HockeyTech responses start with "while(1);" or similar guards.
    guards = ("while(1);", "while(1){", "while(1){;", "while(1) {", "while(1) { }")
    for guard in guards:
        if cleaned.startswith(guard):
            cleaned = cleaned[len(guard) :].lstrip(";\r\n\t ")
            cleaned = cleaned.lstrip("\ufeff")
            break

    if cleaned.startswith("/*"):
        end = cleaned.find("*/")
        if end != -1:
            cleaned = cleaned[end + 2 :].lstrip()

    return cleaned


def _ahl_request(view: str, *, feed: str = "statviewfeed", **extra_params):
    params: Dict[str, object] = {
        "feed": feed,
        "view": view,
        "client_code": AHL_CLIENT_CODE,
        "site_id": AHL_SITE_ID,
        "league_id": AHL_LEAGUE_ID,
        "format": "json",
        "lang": "en",
    }
    for key, value in extra_params.items():
        if value is not None:
            params[key] = value

    use_key = bool(AHL_API_KEY)
    no_key_attempted = False
    if not use_key:
        logging.warning("AHL API key missing; attempting request without authentication")

    endpoint = _ahl_endpoint()

    headers = {
        "User-Agent": "desk-display/1.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://theahl.com/stats/",
        "X-Requested-With": "XMLHttpRequest",
    }
    while True:
        attempt_params = dict(params)
        if use_key:
            attempt_params["key"] = AHL_API_KEY
        else:
            if not no_key_attempted:
                no_key_attempted = True

        try:
            resp = _session.get(endpoint, params=attempt_params, headers=headers, timeout=10)
            resp.raise_for_status()
            payload = _sanitize_ahl_payload(resp.text)
            if not payload:
                logging.error(
                    "Empty response when fetching AHL %s data (status %s)",
                    view,
                    resp.status_code,
                )
                return None
            try:
                return json.loads(payload)
            except json.JSONDecodeError:  # pragma: no cover - depends on upstream
                snippet = payload[:200].replace("\n", " ").replace("\r", " ")
                snippet_lower = snippet.lower()
                if "invalid key" in snippet_lower:
                    if use_key and not no_key_attempted:
                        logging.info(
                            "AHL API rejected the configured key for %s; retrying without a key",
                            view,
                        )
                        use_key = False
                        continue
                    logging.debug(
                        "AHL feed for %s responded with 'Invalid key' (feed=%s); skipping",
                        view,
                        feed,
                    )
                    return None
                logging.error(
                    "Error parsing AHL %s data (status %s, content-type %s): %s",  # noqa: B950
                    view,
                    resp.status_code,
                    resp.headers.get("content-type"),
                    snippet,
                )
                logging.debug("Full AHL %s payload: %s", view, payload)
                return None
        except Exception as exc:
            logging.error("Error fetching AHL %s data: %s", view, exc)
            return None


def _dict_rows(value) -> List[Dict]:
    rows: List[Dict] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                rows.append(item)
    elif isinstance(value, dict):
        inner = value.get("rows") or value.get("data") or value.get("schedule")
        if isinstance(inner, list):
            for item in inner:
                if isinstance(item, dict):
                    rows.append(item)
        elif isinstance(inner, dict):
            maybe_rows = inner.get("rows")
            if isinstance(maybe_rows, list):
                for item in maybe_rows:
                    if isinstance(item, dict):
                        rows.append(item)
    return rows


def _rows_from_table(table: Dict) -> List[Dict]:
    columns: List[str] = []
    for col in table.get("columns", []):
        if isinstance(col, dict):
            name = col.get("name") or col.get("field") or col.get("key")
            if name:
                columns.append(str(name))
        elif isinstance(col, str):
            columns.append(col)

    rows: List[Dict] = []
    for raw_row in table.get("rows", []):
        if isinstance(raw_row, dict):
            rows.append(raw_row)
            continue
        if not isinstance(raw_row, list):
            continue
        entry: Dict[str, object] = {}
        for idx, value in enumerate(raw_row):
            if idx < len(columns):
                entry[columns[idx]] = value
        if entry:
            rows.append(entry)
    return rows


def _extract_rows(payload: Optional[Dict], *keys: str) -> List[Dict]:
    rows: List[Dict] = []
    if not isinstance(payload, dict):
        return rows

    site = payload.get("SiteKit") or payload
    for key in keys:
        block = site.get(key)
        rows.extend(_dict_rows(block))
        if isinstance(block, dict):
            table = block.get("table")
            if isinstance(table, dict):
                rows.extend(_rows_from_table(table))

    if not rows:
        rows.extend(_dict_rows(site))
        table = site.get("table") if isinstance(site, dict) else None
        if isinstance(table, dict):
            rows.extend(_rows_from_table(table))

    return rows


def _current_ahl_season_id() -> Optional[str]:
    if AHL_SEASON_ID:
        return str(AHL_SEASON_ID)

    global _AHL_SEASON_CACHE
    if _AHL_SEASON_CACHE:
        return _AHL_SEASON_CACHE

    data = _ahl_request("season")
    rows = _extract_rows(data, "Seasons", "Season")
    if not rows:
        alt = _ahl_request("season", feed="modulekit")
        rows = _extract_rows(alt, "Seasons", "Season")
    for row in rows:
        flag = row.get("is_current") or row.get("isCurrent") or row.get("current")
        if str(flag).lower() in {"1", "true", "yes"}:
            season_id = row.get("season_id") or row.get("seasonId") or row.get("id")
            if season_id:
                _AHL_SEASON_CACHE = str(season_id)
                return _AHL_SEASON_CACHE

    if rows:
        season_id = rows[0].get("season_id") or rows[0].get("seasonId") or rows[0].get("id")
        if season_id:
            _AHL_SEASON_CACHE = str(season_id)
            return _AHL_SEASON_CACHE

    logging.warning("Unable to determine current AHL season id from feed")
    return None


def _first_value(entry: Dict, *keys: str):
    for key in keys:
        if not key:
            continue
        if key in entry and entry[key] not in (None, ""):
            return entry[key]
    return None


def _extract_team_info(row: Dict, prefix: str) -> Optional[Dict]:
    base = (
        row.get(f"{prefix}_team")
        or row.get(f"{prefix}Team")
        or row.get(prefix)
        or {}
    )
    if not isinstance(base, dict):
        base = {}

    def _get(*candidates):
        for cand in candidates:
            val = base.get(cand)
            if val not in (None, ""):
                return val
        return None

    team_id = _first_value(
        {**row, **base},
        f"{prefix}_team_id",
        f"{prefix}_teamid",
        f"{prefix}TeamID",
        "team_id",
        "teamId",
        "id",
    )
    name = _get("name", "fullName") or row.get(f"{prefix}_team_name")
    city = row.get(f"{prefix}_city") or base.get("city")
    nickname = (
        row.get(f"{prefix}_nickname")
        or row.get(f"{prefix}_short_name")
        or base.get("nickname")
        or base.get("shortname")
    )
    if not name:
        if city and nickname:
            name = f"{city} {nickname}".strip()
        elif nickname:
            name = nickname
        elif city:
            name = city

    abbr = (
        row.get(f"{prefix}_code")
        or row.get(f"{prefix}_abbr")
        or row.get(f"{prefix}_abbrev")
        or row.get(f"{prefix}_short")
        or base.get("abbrev")
        or base.get("code")
    )
    if isinstance(abbr, str):
        abbr = abbr.upper()

    score = _first_value({**row, **base}, f"{prefix}_score", "score")
    shots = _first_value(
        {**row, **base},
        f"{prefix}_shots",
        f"{prefix}_sog",
        f"{prefix}Shots",
        "shots",
    )

    def _as_int(value):
        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    if not (name or abbr):
        return None

    return {
        "id": team_id,
        "name": name,
        "abbr": (abbr or "").upper() or None,
        "nickname": nickname,
        "score": _as_int(score),
        "shots": _as_int(shots),
    }


def _parse_datetime_candidates(row: Dict) -> Optional[datetime.datetime]:
    candidates = [
        row.get("game_date_time"),
        row.get("game_date_time_local"),
        row.get("game_date_time_utc"),
        row.get("game_date_utc"),
        row.get("game_time_utc"),
        row.get("gameTimeUTC"),
        row.get("gameDate"),
    ]
    for text in candidates:
        if not isinstance(text, str):
            continue
        cleaned = text.strip()
        if not cleaned:
            continue
        try:
            if cleaned.endswith("Z"):
                cleaned = cleaned[:-1] + "+00:00"
            if len(cleaned) > 5 and cleaned[-3] != ":" and cleaned[-5] in "+-":
                cleaned = f"{cleaned[:-2]}:{cleaned[-2:]}"
            dt_obj = datetime.datetime.fromisoformat(cleaned)
            if dt_obj.tzinfo is None:
                return pytz.UTC.localize(dt_obj)
            return dt_obj
        except Exception:
            pass

    date_only = row.get("game_date") or row.get("date")
    time_only = row.get("game_time") or row.get("time")
    if isinstance(date_only, str) and isinstance(time_only, str):
        for fmt in ("%Y-%m-%d %I:%M %p", "%Y-%m-%d %H:%M"):
            try:
                naive = datetime.datetime.strptime(f"{date_only} {time_only}", fmt)
                return CENTRAL_TIME.localize(naive)
            except Exception:
                continue
    if isinstance(date_only, str):
        try:
            naive = datetime.datetime.strptime(date_only, "%Y-%m-%d")
            return CENTRAL_TIME.localize(naive)
        except Exception:
            return None
    return None


def _normalize_status(raw_status: Optional[str]) -> str:
    if not raw_status:
        return "FUT"
    text = str(raw_status).strip().upper()
    if not text:
        return "FUT"
    if "FINAL" in text or text.startswith("F"):
        return "FINAL"
    if any(token in text for token in ("LIVE", "IN PROGRESS", "INPROGRESS", "CRIT")):
        return "LIVE"
    if any(token in text for token in ("PREGAME", "SCHEDULED", "FUT", "PRE")):
        return "FUT"
    return text


def _format_local_time(dt_obj: datetime.datetime) -> str:
    fmt = "%-I:%M %p" if os.name != "nt" else "%#I:%M %p"
    try:
        return dt_obj.strftime(fmt).lstrip("0")
    except Exception:
        return dt_obj.strftime("%I:%M %p").lstrip("0")


def _normalize_ahl_game(row: Dict) -> Optional[Dict]:
    home = _extract_team_info(row, "home")
    away = _extract_team_info(row, "away")
    start = _parse_datetime_candidates(row)
    if not home or not away or not start:
        return None

    state = _normalize_status(row.get("game_status") or row.get("status"))
    detail = str(row.get("game_status") or row.get("status_detail") or row.get("result") or state).strip()
    period = row.get("period_desc") or row.get("period") or row.get("periodName")
    clock = row.get("period_time") or row.get("time_remaining") or row.get("clock")
    note = row.get("status_note") or row.get("note")
    venue = row.get("venue") or row.get("venue_name")
    game_id = (
        row.get("game_id")
        or row.get("gameId")
        or row.get("id")
        or row.get("gameUuid")
    )

    start_utc = start.astimezone(pytz.UTC)
    central = start.astimezone(CENTRAL_TIME)
    official_date = start.date().isoformat()

    return {
        "game_id": str(game_id) if game_id else None,
        "home": home,
        "away": away,
        "start": start,
        "start_utc": start_utc,
        "start_iso": start_utc.isoformat(),
        "start_time_central": _format_local_time(central),
        "official_date": official_date,
        "status": {
            "state": state,
            "detail": detail,
            "period": period,
            "clock": clock,
            "note": note,
        },
        "venue": venue,
        "is_home": str(home.get("id")) == str(AHL_TEAM_ID),
    }


def _fetch_ahl_schedule() -> List[Dict]:
    def _fetch_rows(season_id: Optional[str]) -> List[Dict]:
        params: Dict[str, Optional[str]] = {"team_id": AHL_TEAM_ID}
        if season_id:
            params["season_id"] = season_id
        data = _ahl_request("schedule", **params)
        rows = _extract_rows(data, "Schedule", "TeamSchedule")
        if not rows:
            alt = _ahl_request("schedule", feed="modulekit", **params)
            rows = _extract_rows(alt, "Schedule", "TeamSchedule")
        return rows

    rows: List[Dict] = []

    if AHL_SEASON_ID:
        rows = _fetch_rows(str(AHL_SEASON_ID))

    if not rows:
        rows = _fetch_rows(None)

    if not rows:
        season_id = _current_ahl_season_id()
        if season_id:
            rows = _fetch_rows(season_id)

    if not rows:
        return []

    games: List[Dict] = []
    for row in rows:
        normalized = _normalize_ahl_game(row)
        if normalized:
            games.append(normalized)
    games.sort(key=lambda g: g.get("start_utc"))
    return games


def _classify_wolves_games(games: List[Dict]) -> Dict[str, Optional[Dict]]:
    now = datetime.datetime.now(pytz.UTC)
    last_final: Optional[Dict] = None
    live_game: Optional[Dict] = None
    upcoming: List[Dict] = []

    for game in games:
        start = game.get("start_utc") or now
        state = (game.get("status") or {}).get("state", "").upper()
        if state.startswith("FIN"):
            if not last_final or start > last_final.get("start_utc", start):
                last_final = game
            continue
        if state in {"LIVE", "CRIT", "IN PROGRESS", "INPROGRESS"}:
            if live_game is None or start < live_game.get("start_utc", start):
                live_game = game
            continue
        if start >= now:
            upcoming.append(game)
        elif not last_final or start > last_final.get("start_utc", start):
            last_final = game

    upcoming.sort(key=lambda g: g.get("start_utc"))
    next_game = upcoming[0] if upcoming else None
    next_home = None
    for game in upcoming:
        if game.get("is_home"):
            next_home = game
            break

    return {
        "last_game": last_final,
        "live_game": live_game,
        "next_game": next_game,
        "next_home_game": next_home,
    }


def _wolves_schedule_url() -> Optional[str]:
    url = (AHL_SCHEDULE_ICS_URL or "").strip()
    if not url:
        return None
    if url.startswith("webcal://"):
        return "https://" + url[len("webcal://") :].lstrip("/")
    return url


def _unfold_ics_lines(text: str) -> List[str]:
    lines: List[str] = []
    buffer: List[str] = []
    for raw_line in text.splitlines():
        if raw_line.startswith((" ", "\t")) and buffer:
            buffer[-1] += raw_line[1:]
        else:
            buffer.append(raw_line.rstrip())
    lines.extend(buffer)
    return lines


def _parse_ics_events(text: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    params: Dict[str, Dict[str, str]] = {}
    for line in _unfold_ics_lines(text):
        if not line:
            continue
        upper = line.upper()
        if upper == "BEGIN:VEVENT":
            current = {}
            params = {}
            continue
        if upper == "END:VEVENT":
            if current:
                if params:
                    current["__params__"] = params
                events.append(current)
            current = {}
            params = {}
            continue
        if ":" not in line:
            continue
        key_part, value = line.split(":", 1)
        key_bits = key_part.split(";")
        key = key_bits[0].upper()
        param_map: Dict[str, str] = {}
        for bit in key_bits[1:]:
            if "=" in bit:
                p_key, p_val = bit.split("=", 1)
                param_map[p_key.upper()] = p_val
        if param_map:
            params[key] = param_map
        current[key] = value
    return events


def _parse_ics_datetime(value: Optional[str], meta: Dict[str, str]) -> Optional[datetime.datetime]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    tzid = meta.get("TZID") if isinstance(meta, dict) else None
    val_type = (meta.get("VALUE") or "").upper() if isinstance(meta, dict) else ""
    fmts = ["%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S", "%Y%m%dT%H%M"]
    if val_type == "DATE" or ("T" not in text and len(text) == 8):
        fmts = ["%Y%m%d"]
    last_error: Optional[Exception] = None
    for fmt in fmts:
        try:
            dt = datetime.datetime.strptime(text, fmt)
            if fmt.endswith("Z") or text.endswith("Z"):
                return pytz.UTC.localize(dt)
            if tzid:
                try:
                    tz = pytz.timezone(tzid)
                except Exception:
                    tz = CENTRAL_TIME
                return tz.localize(dt)
            return CENTRAL_TIME.localize(dt)
        except Exception as exc:
            last_error = exc
            continue
    logging.debug("Unable to parse ICS datetime '%s': %s", text, last_error)
    return None


_AHL_TEAM_ABBR_OVERRIDES = {
    "abbotsford canucks": "ABB",
    "ahl": "AHL",
    "ahl all star": "AHL",
    "ahl all star challenge": "AHL",
    "ahl all star classic": "AHL",
    "ahl all star game": "AHL",
    "ahl all star skills competition": "AHL",
    "ahl skills competition": "AHL",
    "american hockey league": "AHL",
    "bakersfield condors": "BAK",
    "belleville senators": "BEL",
    "bridgeport islanders": "BRI",
    "calgary wranglers": "CGY",
    "charlotte checkers": "CLT",
    "chicago wolves": "CHI",
    "cleveland monsters": "CLE",
    "coachella valley firebirds": "CV",
    "colorado eagles": "COL",
    "grand rapids griffins": "GR",
    "hartford wolf pack": "HFD",
    "henderson silver knights": "HSK",
    "hershey bears": "HER",
    "iowa wild": "IA",
    "laval rocket": "LAV",
    "lehigh valley phantoms": "LV",
    "manitoba moose": "MB",
    "milwaukee admirals": "MIL",
    "ontario reign": "ONT",
    "providence bruins": "PRO",
    "rockford icehogs": "RFD",
    "rochester americans": "ROC",
    "san diego gulls": "SD",
    "san jose barracuda": "SJ",
    "springfield thunderbirds": "SPR",
    "syracuse crunch": "SYR",
    "texas stars": "TEX",
    "toronto marlies": "TOR",
    "tucson roadrunners": "TUC",
    "utica comets": "UTC",
    "wilkes barre scranton penguins": "WBS",
}


def _derive_team_abbr(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", name or "").strip()
    if not cleaned:
        return "TBD"
    parts = cleaned.split()
    if len(parts) == 1:
        return parts[0][:3].upper()
    abbr = "".join(part[0] for part in parts[:3])
    return (abbr or cleaned[:3]).upper()


def _nickname_from_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name or "").strip()
    if not cleaned:
        return ""
    parts = cleaned.split(" ")
    if len(parts) == 1:
        return cleaned
    return parts[-1]


def _ics_team_payload(name: str, is_wolves: bool) -> Dict[str, Any]:
    if is_wolves:
        abbr = (AHL_TEAM_TRICODE or _derive_team_abbr(name)).upper()
        team_id: Optional[int] = AHL_TEAM_ID
    else:
        key = re.sub(r"[^a-z0-9]+", " ", name or "").strip()
        abbr = _AHL_TEAM_ABBR_OVERRIDES.get(key.lower()) or _derive_team_abbr(name)
        team_id = None
    return {
        "id": team_id,
        "abbr": abbr,
        "name": name,
        "nickname": _nickname_from_name(name),
        "score": None,
        "shots": None,
    }


_RESULT_SUFFIX_RE = re.compile(r"\s*\(([WL]|WIN|LOSS)\)\s*$", re.IGNORECASE)
_SCORE_PAIR_RE = re.compile(r"([^()]+?)\s*\((\d+)\)")


def _clean_ics_team_fragment(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = _RESULT_SUFFIX_RE.sub("", cleaned)
    return cleaned.strip(" \t-–—✔✓✖✕•")


def _extract_ics_score_line(description: Optional[str]) -> Optional[str]:
    if not description:
        return None
    for raw_line in description.replace("\r", "").splitlines():
        line = raw_line.strip()
        if not line or "(" not in line or ")" not in line:
            continue
        if any(ch.isdigit() for ch in line):
            return line.lstrip("✔✓✖✕•-–— ").strip()
    return None


def _parse_ics_note_scores(description: Optional[str]) -> Optional[Dict[str, Any]]:
    line = _extract_ics_score_line(description)
    if not line:
        return None
    entries: List[Tuple[str, int]] = []
    for raw_name, score_text in _SCORE_PAIR_RE.findall(line):
        name = _clean_ics_team_fragment(raw_name)
        if not name:
            continue
        try:
            score = int(score_text)
        except ValueError:
            continue
        entries.append((name, score))
    if len(entries) < 2:
        return None
    return {"line": line, "entries": entries}


def _match_score_for_team(name: str, entries: List[Tuple[str, int]]) -> Optional[int]:
    if not name:
        return None
    name_lower = name.lower()
    for entry_name, score in entries:
        entry_lower = entry_name.lower()
        if name_lower in entry_lower or entry_lower in name_lower:
            return score
    return None


def _split_wolves_summary(summary: str) -> Optional[Dict[str, Any]]:
    if not summary:
        return None
    clean = summary.strip()
    if not clean:
        return None
    wolves = (AHL_TEAM_NAME or "Chicago Wolves").lower()
    patterns = [
        (r"(.+?)\s+at\s+(.+)", False),
        (r"(.+?)\s+@\s+(.+)", False),
        (r"(.+?)\s+vs\.?\s+(.+)", True),
        (r"(.+?)\s+v\.?\s+(.+)", True),
    ]
    for pattern, first_is_home in patterns:
        match = re.match(pattern, clean, re.IGNORECASE)
        if not match:
            continue
        first = match.group(1).strip()
        second = match.group(2).strip()
        if first_is_home:
            home_name, away_name = first, second
        else:
            away_name, home_name = first, second
        home_name = _clean_ics_team_fragment(home_name)
        away_name = _clean_ics_team_fragment(away_name)
        home_lower = home_name.lower()
        away_lower = away_name.lower()
        wolves_home = wolves in home_lower
        wolves_away = wolves in away_lower
        if not (wolves_home or wolves_away):
            continue
        return {
            "home_name": home_name,
            "away_name": away_name,
            "wolves_is_home": wolves_home,
            "wolves_is_away": wolves_away,
        }
    return None


def _normalize_wolves_ics_game(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if event.get("STATUS", "CONFIRMED").upper() == "CANCELLED":
        return None
    params = (event.get("__params__") or {}).get("DTSTART") or {}
    start = _parse_ics_datetime(event.get("DTSTART"), params)
    if not start:
        return None
    summary = event.get("SUMMARY")
    parsed = _split_wolves_summary(summary or "")
    if not parsed:
        return None
    home_name = parsed["home_name"]
    away_name = parsed["away_name"]
    wolves_is_home = bool(parsed.get("wolves_is_home"))
    wolves_is_away = bool(parsed.get("wolves_is_away"))
    if not (wolves_is_home or wolves_is_away):
        return None
    home_team = _ics_team_payload(home_name, wolves_is_home)
    away_team = _ics_team_payload(away_name, wolves_is_away)
    description = event.get("DESCRIPTION") or event.get("X-ALT-DESC")
    note_scores = _parse_ics_note_scores(description)
    status_state = "FUT"
    status_detail = "Scheduled"
    status_note: Optional[str] = None
    if note_scores:
        entries = note_scores["entries"]
        home_score = _match_score_for_team(home_name, entries)
        away_score = _match_score_for_team(away_name, entries)
        if home_score is not None:
            home_team["score"] = home_score
        if away_score is not None:
            away_team["score"] = away_score
        if home_score is not None and away_score is not None:
            status_state = "FINAL"
            status_detail = "Final"
            status_note = note_scores.get("line")
    start_utc = start.astimezone(pytz.UTC)
    start_central = start.astimezone(CENTRAL_TIME)
    return {
        "game_id": event.get("UID"),
        "home": home_team,
        "away": away_team,
        "start": start,
        "start_utc": start_utc,
        "start_iso": start_utc.isoformat(),
        "start_time_central": _format_local_time(start_central),
        "official_date": start.date().isoformat(),
        "status": {
            "state": status_state,
            "detail": status_detail,
            "period": None,
            "clock": None,
            "note": status_note,
        },
        "venue": event.get("LOCATION"),
        "is_home": wolves_is_home,
    }


def _fetch_wolves_ics_games() -> List[Dict[str, Any]]:
    url = _wolves_schedule_url()
    if not url:
        logging.warning("AHL schedule ICS URL not configured")
        return []
    headers = {
        "User-Agent": "desk-display/1.0",
        "Accept": "text/calendar, text/plain;q=0.9, */*;q=0.8",
        "Referer": "https://stanzacal.com/",
    }
    try:
        resp = _session.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        events = _parse_ics_events(resp.text)
    except Exception as exc:
        logging.error("Error fetching Wolves ICS schedule: %s", exc)
        return []
    games: List[Dict[str, Any]] = []
    for event in events:
        normalized = _normalize_wolves_ics_game(event)
        if normalized:
            games.append(normalized)
    games.sort(key=lambda g: g.get("start_utc"))
    return games


def _ics_game_has_scores(game: Dict[str, Any]) -> bool:
    home = game.get("home") or {}
    away = game.get("away") or {}
    return isinstance(home.get("score"), int) and isinstance(away.get("score"), int)


def _classify_wolves_ics_games(games: List[Dict[str, Any]]) -> Dict[str, Optional[Dict]]:
    now = datetime.datetime.now(pytz.UTC)
    upcoming: List[Dict[str, Any]] = []
    last_final: Optional[Dict[str, Any]] = None
    live_game: Optional[Dict[str, Any]] = None
    live_window = datetime.timedelta(hours=4)

    for game in games:
        start = game.get("start_utc")
        if not isinstance(start, datetime.datetime):
            continue
        status_state = ((game.get("status") or {}).get("state") or "").upper()
        has_scores = _ics_game_has_scores(game)

        if has_scores and (status_state.startswith("FIN") or start <= now):
            if not last_final or start > last_final.get("start_utc", start):
                last_final = game
            continue

        if start <= now:
            if not has_scores and now - start <= live_window:
                if (live_game is None) or start < live_game.get("start_utc", start):
                    live_game = game
                continue
            if not last_final or start > last_final.get("start_utc", start):
                last_final = game
            continue

        upcoming.append(game)

    upcoming.sort(key=lambda g: g.get("start_utc"))
    next_game = upcoming[0] if upcoming else None
    next_home = None
    for game in upcoming:
        if game.get("is_home"):
            next_home = game
            break

    if live_game is not None:
        status = live_game.setdefault("status", {})
        status.setdefault("state", "LIVE")
        status.setdefault("detail", "In Progress")

    return {
        "last_game": last_final,
        "live_game": live_game,
        "next_game": next_game,
        "next_home_game": next_home,
    }


_WOLVES_CACHE_TTL = 15 * 60  # seconds
_wolves_cache: Dict[str, Any] = {"expires": 0.0, "data": None}


def fetch_wolves_games(force_refresh: bool = False) -> Dict[str, Optional[Dict]]:
    now = time.time()
    cached = _wolves_cache.get("data")
    expires = _wolves_cache.get("expires", 0.0)
    if (
        not force_refresh
        and isinstance(expires, (int, float))
        and isinstance(cached, dict)
        and now < float(expires)
    ):
        return cached

    last_game: Optional[Dict] = None
    live_game: Optional[Dict] = None
    next_game: Optional[Dict] = None
    next_home: Optional[Dict] = None

    try:
        ics_games = _fetch_wolves_ics_games()
        if ics_games:
            classified_ics = _classify_wolves_ics_games(ics_games)
            last_game = classified_ics.get("last_game")
            live_game = classified_ics.get("live_game")
            next_game = classified_ics.get("next_game")
            next_home = classified_ics.get("next_home_game")
    except Exception as exc:
        logging.error("Error parsing Wolves ICS schedule: %s", exc)

    payload = {
        "last_game": last_game,
        "live_game": live_game,
        "next_game": next_game,
        "next_home_game": next_home,
    }
    _wolves_cache["data"] = payload
    _wolves_cache["expires"] = now + _WOLVES_CACHE_TTL
    return payload
