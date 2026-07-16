"""Air quality provider helpers and normalization."""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from services.http_client import http_get, request_json

OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
IQAIR_AIR_QUALITY_URLS: tuple[tuple[float, float, str], ...] = (
    (42.1373, -87.8446, "https://www.iqair.com/air-quality/usa/illinois/northbrook"),
    (41.9037, -87.6357, "https://www.iqair.com/air-quality/usa/illinois/chicago"),
)
POLLEN_KEYS = (
    "alder_pollen",
    "birch_pollen",
    "grass_pollen",
    "mugwort_pollen",
    "olive_pollen",
    "ragweed_pollen",
)
POLLUTANT_KEYS = (
    "pm2_5",
    "pm10",
    "ozone",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
)
POLLUTANT_LABELS = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "ozone": "Ozone",
    "nitrogen_dioxide": "NO₂",
    "sulphur_dioxide": "SO₂",
    "carbon_monoxide": "CO",
}


@dataclass(frozen=True)
class AirQualityReport:
    """Normalized air-quality data for display screens."""

    aqi_value: Optional[int]
    aqi_category: str
    primary_pollutant: Optional[str]
    pollen_level: Optional[str] = None
    advisory_text: Optional[str] = None
    pollutant_breakdown: tuple[tuple[str, float], ...] = ()
    pm2_5_value: Optional[float] = None
    trend_text: Optional[str] = None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def aqi_category(aqi: Optional[float]) -> str:
    """Return the U.S. EPA AQI category for *aqi*."""

    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def pollen_level_from_value(value: Optional[float]) -> Optional[str]:
    """Normalize Open-Meteo pollen concentration to a simple level."""

    if value is None:
        return None
    if value <= 0:
        return "None"
    if value < 10:
        return "Low"
    if value < 50:
        return "Moderate"
    if value < 100:
        return "High"
    return "Very High"


def advisory_for(category: str, pollen_level: Optional[str]) -> str:
    """Return a short, plain-language outdoor-activity recommendation."""

    if category in {"Hazardous", "Very Unhealthy"}:
        return "Avoid outdoor activity."
    if category == "Unhealthy":
        return "Keep activity short."
    if category == "Unhealthy for Sensitive Groups":
        return "Limit exertion if sensitive."
    if pollen_level in {"High", "Very High"}:
        return "Okay outside; pollen high."
    if category == "Moderate":
        return "Okay for most outdoor plans."
    if category == "Good":
        return "Good for outdoor activity."
    return "Check local conditions."


def _hourly_values(payload: dict[str, Any], key: str) -> list[Optional[float]]:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        return []
    values = hourly.get(key)
    if not isinstance(values, list):
        return []
    return [_to_float(value) for value in values]


def trend_for(payload: dict[str, Any], key: str = "us_aqi", hours: int = 6) -> Optional[str]:
    """Summarize the expected trend for the next several hours."""

    values = [value for value in _hourly_values(payload, key)[: max(2, hours)] if value is not None]
    if len(values) < 2:
        return None

    start = values[0]
    end = values[-1]
    delta = end - start
    if abs(delta) < 5:
        direction = "steady"
    elif delta > 0:
        direction = "rising"
    else:
        direction = "improving"
    return f"{direction.title()} next {len(values)}h"


def _current_value(payload: dict[str, Any], key: str) -> Optional[float]:
    current = payload.get("current")
    if isinstance(current, dict):
        return _to_float(current.get(key))

    values = _hourly_values(payload, key)
    if not values:
        return None
    return values[0]


def _primary_pollutant(payload: dict[str, Any]) -> tuple[Optional[str], tuple[tuple[str, float], ...]]:
    readings: list[tuple[str, float]] = []
    for key in POLLUTANT_KEYS:
        value = _current_value(payload, key)
        if value is not None:
            readings.append((POLLUTANT_LABELS.get(key, key), value))
    if not readings:
        return None, ()
    # Prefer particulate matter when available because it commonly drives window guidance.
    primary = readings[0][0]
    return primary, tuple(readings[:4])


def normalize_open_meteo(payload: dict[str, Any]) -> AirQualityReport:
    """Normalize an Open-Meteo air-quality response."""

    raw_aqi = _current_value(payload, "us_aqi")
    category = aqi_category(raw_aqi)
    primary, breakdown = _primary_pollutant(payload)

    pollen_values = [_current_value(payload, key) for key in POLLEN_KEYS]
    max_pollen = max((value for value in pollen_values if value is not None), default=None)
    pollen = pollen_level_from_value(max_pollen)

    return AirQualityReport(
        aqi_value=int(round(raw_aqi)) if raw_aqi is not None else None,
        aqi_category=category,
        primary_pollutant=primary,
        pollen_level=pollen,
        advisory_text=advisory_for(category, pollen),
        pollutant_breakdown=breakdown,
        pm2_5_value=_current_value(payload, "pm2_5"),
        trend_text=trend_for(payload),
    )


def _clean_iqair_html(text: str) -> list[str]:
    """Return visible text lines from an IQAir location page."""

    without_scripts = re.sub(r"<script\b[^>]*>.*?</script>", "\n", text, flags=re.IGNORECASE | re.DOTALL)
    without_styles = re.sub(r"<style\b[^>]*>.*?</style>", "\n", without_scripts, flags=re.IGNORECASE | re.DOTALL)
    with_breaks = re.sub(r"<(?:br|/p|/div|/h[1-6]|/li|/span)\b[^>]*>", "\n", without_styles, flags=re.IGNORECASE)
    visible = re.sub(r"<[^>]+>", "\n", with_breaks)
    visible = html.unescape(visible).replace("\xa0", " ")
    return [line.strip() for line in visible.splitlines() if line.strip()]


def _find_iqair_line_index(lines: list[str], value: str) -> Optional[int]:
    needle = value.casefold()
    for index, line in enumerate(lines):
        if line.casefold() == needle:
            return index
    return None


def _parse_iqair_hourly_values(lines: list[str], current_aqi: Optional[int]) -> list[int]:
    start = _find_iqair_line_index(lines, "Hourly forecast")
    if start is None:
        return [current_aqi] if current_aqi is not None else []

    values: list[int] = []
    for index, line in enumerate(lines[start + 1 :], start=start + 1):
        if index > start + 90:
            break
        if line in {"Now"} or re.fullmatch(r"\d{1,2}:\d{2}", line):
            for candidate in lines[index + 1 : index + 5]:
                if re.fullmatch(r"\d{1,3}", candidate):
                    values.append(int(candidate))
                    break
        if len(values) >= 6:
            break
    if not values and current_aqi is not None:
        values.append(current_aqi)
    return values


def _trend_from_values(values: list[int], hours: int = 6) -> Optional[str]:
    limited = values[: max(2, hours)]
    if len(limited) < 2:
        return None
    delta = limited[-1] - limited[0]
    if abs(delta) < 5:
        direction = "steady"
    elif delta > 0:
        direction = "rising"
    else:
        direction = "improving"
    return f"{direction.title()} next {len(limited)}h"


def iqair_url_for_coordinates(latitude: float, longitude: float, *, tolerance: float = 0.0001) -> Optional[str]:
    """Return the configured IQAir URL for supported display coordinates."""

    for target_lat, target_lon, url in IQAIR_AIR_QUALITY_URLS:
        if abs(latitude - target_lat) <= tolerance and abs(longitude - target_lon) <= tolerance:
            return url
    return None


def normalize_iqair_page(html_text: str) -> Optional[AirQualityReport]:
    """Normalize the visible data on an IQAir location page."""

    lines = _clean_iqair_html(html_text)
    aqi_index = _find_iqair_line_index(lines, "US AQI+")
    if aqi_index is None:
        aqi_index = _find_iqair_line_index(lines, "US AQI⁺")
    if aqi_index is None or aqi_index == 0:
        return None

    aqi_raw = _to_float(lines[aqi_index - 1])
    if aqi_raw is None:
        return None
    aqi_value = int(round(aqi_raw))
    category = lines[aqi_index + 1] if aqi_index + 1 < len(lines) else aqi_category(aqi_value)
    primary = None
    pm2_5_value = None
    pollutant_breakdown: tuple[tuple[str, float], ...] = ()
    main_pollutant_index = _find_iqair_line_index(lines, "Main pollutant:")
    if main_pollutant_index is not None and main_pollutant_index + 1 < len(lines):
        primary = lines[main_pollutant_index + 1]
        if primary.upper().replace(" ", "") == "PM2.5" and main_pollutant_index + 2 < len(lines):
            match = re.search(r"[-+]?\d+(?:\.\d+)?", lines[main_pollutant_index + 2])
            if match:
                pm2_5_value = float(match.group(0))
                pollutant_breakdown = (("PM2.5", pm2_5_value),)

    hourly_values = _parse_iqair_hourly_values(lines, aqi_value)
    return AirQualityReport(
        aqi_value=aqi_value,
        aqi_category=category or aqi_category(aqi_value),
        primary_pollutant=primary,
        advisory_text=advisory_for(aqi_category(aqi_value), None),
        pollutant_breakdown=pollutant_breakdown,
        pm2_5_value=pm2_5_value,
        trend_text=_trend_from_values(hourly_values),
    )


def fetch_iqair_air_quality(url: str, *, timeout: float = 8.0) -> Optional[AirQualityReport]:
    """Fetch and normalize an IQAir location page."""

    try:
        response = http_get(url, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - defensive network layer
        logging.warning("IQAir air quality request failed: %s (%s)", url, exc)
        return None
    report = normalize_iqair_page(response.text)
    if report is None:
        logging.warning("IQAir air quality page could not be parsed: %s", url)
    return report


def fetch_air_quality(latitude: float, longitude: float, *, include_pollen: bool = True, timeout: float = 8.0) -> Optional[AirQualityReport]:
    """Fetch and normalize current air quality for coordinates."""

    iqair_url = iqair_url_for_coordinates(latitude, longitude)
    if iqair_url:
        report = fetch_iqair_air_quality(iqair_url, timeout=timeout)
        if report is not None:
            return report
        logging.warning("Falling back to Open-Meteo AQI for %s, %s", latitude, longitude)

    variables = ["us_aqi", *POLLUTANT_KEYS]
    if include_pollen:
        variables.extend(POLLEN_KEYS)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(variables),
        "hourly": ",".join(variables),
        "timezone": "auto",
        "forecast_days": 1,
    }
    payload = request_json(OPEN_METEO_AIR_QUALITY_URL, params=params, timeout=timeout, quiet=True)
    if not payload:
        logging.warning("Air quality request returned no data.")
        return None
    return normalize_open_meteo(payload)
