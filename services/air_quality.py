"""AirNow air-quality provider helpers and normalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Optional

from services.http_client import request_json

AIRNOW_CURRENT_OBSERVATION_URL = "https://www.airnowapi.org/aq/observation/latLong/current/"
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
_AIRNOW_PARAMETER_LABELS = {
    "PM2.5": "PM2.5",
    "PM10": "PM10",
    "OZONE": "Ozone",
}
_OPEN_METEO_COMPONENT_KEYS = (
    "us_aqi_pm2_5",
    "us_aqi_pm10",
    "us_aqi_ozone",
)


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
    us_aqi_pm2_5: Optional[int] = None
    us_aqi_pm10: Optional[int] = None
    us_aqi_ozone: Optional[int] = None
    trend_text: Optional[str] = None
    # Recent component readings are retained by the refresh loop for the
    # compact history charts on the air-quality screen.
    component_history: tuple[tuple[float, Optional[int], Optional[int], Optional[int]], ...] = ()


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _open_meteo_component_aqis(payload: Any) -> dict[str, int]:
    """Extract available US AQI component values from Open-Meteo data."""

    if not isinstance(payload, dict):
        return {}
    current = payload.get("current")
    if not isinstance(current, dict):
        return {}
    return {
        key: value
        for key in _OPEN_METEO_COMPONENT_KEYS
        if (value := _to_int(current.get(key))) is not None
    }


def _fill_missing_component_aqis(report: AirQualityReport, payload: Any) -> AirQualityReport:
    """Supplement AirNow observations with available modeled component AQIs."""

    components = _open_meteo_component_aqis(payload)
    return replace(
        report,
        us_aqi_pm2_5=(
            report.us_aqi_pm2_5
            if report.us_aqi_pm2_5 is not None
            else components.get("us_aqi_pm2_5")
        ),
        us_aqi_pm10=(
            report.us_aqi_pm10 if report.us_aqi_pm10 is not None else components.get("us_aqi_pm10")
        ),
        us_aqi_ozone=(
            report.us_aqi_ozone
            if report.us_aqi_ozone is not None
            else components.get("us_aqi_ozone")
        ),
    )


def _fetch_open_meteo_component_aqis(latitude: float, longitude: float, *, timeout: float) -> Any:
    """Fetch modeled component AQIs when AirNow has no observation for a pollutant."""

    return request_json(
        OPEN_METEO_AIR_QUALITY_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join(_OPEN_METEO_COMPONENT_KEYS),
        },
        timeout=timeout,
        quiet=True,
    )


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


def advisory_for(category: str, pollen_level: Optional[str]) -> str:
    """Return a short, plain-language outdoor-activity recommendation."""

    if category in {"Hazardous", "Very Unhealthy"}:
        return "Avoid outdoor activity."
    if category == "Unhealthy":
        return "Keep activity short."
    if category == "Unhealthy for Sensitive Groups":
        return "Limit exertion if sensitive."
    if category == "Moderate":
        return "Okay for most outdoor plans."
    if category == "Good":
        return "Good for outdoor activity."
    return "Check local conditions."


def normalize_airnow(payload: Any) -> Optional[AirQualityReport]:
    """Normalize AirNow current-observation records into one display report."""

    if not isinstance(payload, list):
        return None

    readings: list[tuple[str, int, str]] = []
    for record in payload:
        if not isinstance(record, dict):
            continue
        parameter = record.get("ParameterName")
        label = _AIRNOW_PARAMETER_LABELS.get(parameter)
        value = _to_int(record.get("AQI"))
        if label is not None and value is not None:
            category = record.get("Category")
            category_name = category.get("Name") if isinstance(category, dict) else None
            resolved_category = (
                category_name if isinstance(category_name, str) else aqi_category(value)
            )
            readings.append((label, value, resolved_category))

    if not readings:
        return None

    # AirNow reports one AQI per pollutant. The highest is the official overall AQI.
    primary, aqi_value, category = max(readings, key=lambda reading: reading[1])
    components = {label: value for label, value, _ in readings}
    breakdown = tuple((label, float(value)) for label, value, _ in readings)
    return AirQualityReport(
        aqi_value=aqi_value,
        aqi_category=category,
        primary_pollutant=primary,
        advisory_text=advisory_for(category, None),
        pollutant_breakdown=breakdown,
        us_aqi_pm2_5=components.get("PM2.5"),
        us_aqi_pm10=components.get("PM10"),
        us_aqi_ozone=components.get("Ozone"),
    )


def fetch_air_quality(
    latitude: float,
    longitude: float,
    *,
    api_key: str,
    include_pollen: bool = True,
    timeout: float = 8.0,
) -> Optional[AirQualityReport]:
    """Fetch current AirNow observations for a U.S. location.

    ``include_pollen`` remains accepted for the caller API but AirNow does not
    provide pollen observations.
    """

    del include_pollen
    if not api_key:
        logging.warning("AirNow API key is not configured.")
        return None
    payload = request_json(
        AIRNOW_CURRENT_OBSERVATION_URL,
        params={
            "format": "application/json",
            "latitude": latitude,
            "longitude": longitude,
            "distance": 25,
            "API_KEY": api_key,
        },
        timeout=timeout,
        quiet=True,
    )
    report = normalize_airnow(payload)
    if report is None:
        logging.warning("AirNow returned no current AQI observations.")
    elif None in (report.us_aqi_pm2_5, report.us_aqi_pm10, report.us_aqi_ozone):
        # AirNow only returns pollutants observed at nearby monitoring stations.
        # Fill any gaps with Open-Meteo's modeled components without replacing
        # AirNow's official AQI, category, or observed component values.
        report = _fill_missing_component_aqis(
            report,
            _fetch_open_meteo_component_aqis(latitude, longitude, timeout=timeout),
        )
    return report
