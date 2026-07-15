"""Air quality provider helpers and normalization."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any, Optional

from services.http_client import request_json

OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
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
    """Return a short, plain-language recommendation."""

    if category in {"Hazardous", "Very Unhealthy", "Unhealthy"}:
        return "Keep windows closed."
    if category == "Unhealthy for Sensitive Groups":
        return "Limit outdoor time if sensitive."
    if pollen_level in {"High", "Very High"}:
        return "Air is OK, but pollen is high."
    if category == "Moderate":
        return "Okay to air out briefly."
    if category == "Good":
        return "Good day to open windows."
    return "Check local conditions."


def _current_value(payload: dict[str, Any], key: str) -> Optional[float]:
    current = payload.get("current")
    if isinstance(current, dict):
        return _to_float(current.get(key))

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        return None
    values = hourly.get(key)
    if not isinstance(values, list) or not values:
        return None
    return _to_float(values[0])


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
    )


def fetch_air_quality(latitude: float, longitude: float, *, include_pollen: bool = True, timeout: float = 8.0) -> Optional[AirQualityReport]:
    """Fetch and normalize current air quality for coordinates."""

    variables = ["us_aqi", *POLLUTANT_KEYS]
    if include_pollen:
        variables.extend(POLLEN_KEYS)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(variables),
        "timezone": "auto",
        "forecast_days": 1,
    }
    payload = request_json(OPEN_METEO_AIR_QUALITY_URL, params=params, timeout=timeout, quiet=True)
    if not payload:
        logging.warning("Air quality request returned no data.")
        return None
    return normalize_open_meteo(payload)
