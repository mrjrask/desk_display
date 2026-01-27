"""Helpers for Apple Maps web services."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import jwt
import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from config import (
    APPLE_MAPS_KEY_ID,
    APPLE_MAPS_KEY_PATH,
    APPLE_MAPS_PRIVATE_KEY,
    APPLE_MAPS_TEAM_ID,
    WEATHERKIT_KEY_ID,
    WEATHERKIT_KEY_PATH,
    WEATHERKIT_PRIVATE_KEY,
    WEATHERKIT_TEAM_ID,
)

APPLE_MAPS_USER_AGENT = "desk-display/apple-maps"
APPLE_MAPS_TOKEN_TTL_MINUTES = 30

_apple_maps_token: Optional[str] = None
_apple_maps_token_exp: Optional[datetime.datetime] = None
_apple_maps_private_key_cache = None


def _load_private_key(
    inline_key: Optional[str],
    key_path: Optional[str],
    label: str,
):
    global _apple_maps_private_key_cache

    if _apple_maps_private_key_cache is not None:
        return _apple_maps_private_key_cache

    if inline_key:
        try:
            normalized_key = inline_key.replace("\r\n", "\n").strip()
            pem_bytes = normalized_key.encode("utf-8")
            if not pem_bytes.endswith(b"\n"):
                pem_bytes += b"\n"
            _apple_maps_private_key_cache = serialization.load_pem_private_key(
                pem_bytes,
                password=None,
                backend=default_backend(),
            )
            return _apple_maps_private_key_cache
        except Exception as exc:
            logging.warning("Apple Maps: unable to parse %s private key: %s", label, exc)
            return None

    if key_path:
        try:
            with open(key_path, "rb") as fh:
                key_bytes = fh.read()
            _apple_maps_private_key_cache = serialization.load_pem_private_key(
                key_bytes,
                password=None,
                backend=default_backend(),
            )
            return _apple_maps_private_key_cache
        except Exception as exc:
            logging.warning("Apple Maps: unable to read %s private key: %s", label, exc)
            return None

    logging.warning(
        "Apple Maps: no %s private key configured; set %s_PRIVATE_KEY or %s_KEY_PATH.",
        label,
        label,
        label,
    )
    return None


def _build_apple_maps_token() -> Optional[str]:
    global _apple_maps_token, _apple_maps_token_exp

    team_id = APPLE_MAPS_TEAM_ID or WEATHERKIT_TEAM_ID
    key_id = APPLE_MAPS_KEY_ID or WEATHERKIT_KEY_ID
    if not team_id or not key_id:
        logging.warning(
            "Apple Maps: missing TEAM_ID or KEY_ID; set APPLE_MAPS_TEAM_ID/APPLE_MAPS_KEY_ID "
            "or reuse WEATHERKIT_TEAM_ID/WEATHERKIT_KEY_ID."
        )
        return None

    now = datetime.datetime.now(datetime.timezone.utc)
    if _apple_maps_token and _apple_maps_token_exp:
        if (_apple_maps_token_exp - now).total_seconds() > 300:
            return _apple_maps_token

    key = _load_private_key(
        APPLE_MAPS_PRIVATE_KEY or WEATHERKIT_PRIVATE_KEY,
        APPLE_MAPS_KEY_PATH or WEATHERKIT_KEY_PATH,
        "APPLE_MAPS",
    )
    if not key:
        return None

    iat = int(now.timestamp())
    exp = int((now + datetime.timedelta(minutes=APPLE_MAPS_TOKEN_TTL_MINUTES)).timestamp())
    try:
        _apple_maps_token = jwt.encode(
            {"iss": team_id, "iat": iat, "exp": exp},
            key,
            algorithm="ES256",
            headers={"kid": key_id},
        )
        _apple_maps_token_exp = datetime.datetime.fromtimestamp(exp, datetime.timezone.utc)
        return _apple_maps_token
    except Exception as exc:
        logging.warning("Apple Maps: unable to sign token: %s", exc)
        return None


def _resolve_api_key(api_key: str) -> Optional[str]:
    if api_key:
        return api_key
    return _build_apple_maps_token()


def _coerce_seconds(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _format_duration(seconds: Optional[int]) -> str:
    if not seconds:
        return "N/A"
    minutes_total = int(round(seconds / 60))
    if minutes_total < 60:
        return f"{max(minutes_total, 1)} min"
    hours = minutes_total // 60
    minutes = minutes_total % 60
    if minutes:
        return f"{hours} hr {minutes} min"
    return f"{hours} hr"


def _extract_points(value: Any) -> Tuple[Optional[List[Tuple[float, float]]], Optional[str]]:
    if isinstance(value, str):
        return None, value
    if isinstance(value, dict):
        points_value = value.get("points")
        if isinstance(points_value, str):
            return None, points_value
        if isinstance(points_value, list):
            value = points_value
        else:
            return None, None
    if isinstance(value, list):
        points: List[Tuple[float, float]] = []
        for item in value:
            if isinstance(item, dict):
                lat = item.get("latitude")
                lng = item.get("longitude")
                if lat is None or lng is None:
                    lat = item.get("lat")
                    lng = item.get("lng")
                if lat is None or lng is None:
                    continue
                try:
                    points.append((float(lat), float(lng)))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    first = float(item[0])
                    second = float(item[1])
                except (TypeError, ValueError):
                    continue
                if abs(first) <= 90 and abs(second) <= 180:
                    points.append((first, second))
                elif abs(second) <= 90 and abs(first) <= 180:
                    points.append((second, first))
        return (points or None), None
    return None, None


def _extract_instruction(step: dict) -> Optional[str]:
    for key in ("instruction", "instructions", "guidance", "maneuver", "description"):
        value = step.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _normalize_route(route: dict) -> dict:
    duration_sec = _coerce_seconds(route.get("expectedTravelTime"))
    baseline_sec = _coerce_seconds(route.get("staticTravelTime"))
    if baseline_sec is None:
        baseline_sec = _coerce_seconds(route.get("typicalTravelTime"))
    if baseline_sec is None:
        baseline_sec = _coerce_seconds(route.get("travelTime"))
    if duration_sec is None:
        duration_sec = baseline_sec

    route["_duration_sec"] = duration_sec
    route["_duration_base_sec"] = baseline_sec
    route["_duration_text"] = _format_duration(duration_sec)

    summary = route.get("name") or route.get("summary") or route.get("routeName") or ""
    route["_summary"] = str(summary).lower()

    steps = route.get("steps") or (route.get("legs") or [{}])[0].get("steps") or []
    instructions: List[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        instruction = _extract_instruction(step)
        if instruction:
            instructions.append(instruction)
        points, polyline = _extract_points(
            step.get("polyline")
            or step.get("path")
            or step.get("shape")
            or step.get("points")
        )
        if points:
            step["_path_points"] = points
        if polyline:
            step["_path_polyline"] = polyline

    route["_steps_text"] = " ".join(instructions).lower()

    overview_points, overview_polyline = _extract_points(
        route.get("polyline") or route.get("path") or route.get("shape") or route.get("points")
    )
    if overview_points:
        route["_overview_points"] = overview_points
    if overview_polyline:
        route["_overview_polyline"] = overview_polyline

    return route


def fetch_apple_maps_routes(
    origin: str,
    destination: str,
    api_key: str,
    *,
    avoid_highways: bool = False,
    avoid_tolls: bool = False,
    url: str,
) -> List[Dict[str, Any]]:
    resolved_key = _resolve_api_key(api_key)
    if not resolved_key:
        logging.warning("Travel: no Apple Maps token configured.")
        return []

    params = {
        "origin": origin,
        "destination": destination,
        "transportType": "automobile",
        "alternateRoutes": "true",
        "token": resolved_key,
    }
    avoid_values: List[str] = []
    if avoid_highways:
        avoid_values.append("highways")
    if avoid_tolls:
        avoid_values.append("tolls")
    if avoid_values:
        params["avoid"] = ",".join(avoid_values)

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
            headers={"User-Agent": APPLE_MAPS_USER_AGENT},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logging.warning("Apple Maps directions request failed: %s", exc)
        return []

    routes = payload.get("routes", []) if isinstance(payload, dict) else []
    if not routes:
        logging.warning("Apple Maps directions returned no routes.")
        return []

    return [_normalize_route(route) for route in routes if isinstance(route, dict)]


def fetch_apple_maps_snapshot(
    center: Tuple[float, float],
    zoom: int,
    size: Tuple[int, int],
    api_key: str,
    *,
    map_type: str = "mutedStandard",
    url: str,
) -> Optional[bytes]:
    resolved_key = _resolve_api_key(api_key)
    if not resolved_key:
        logging.warning("Apple Maps snapshot: no Apple Maps token configured.")
        return None

    width, height = size
    params = {
        "center": f"{center[0]},{center[1]}",
        "zoom": str(zoom),
        "size": f"{width}x{height}",
        "scale": "2",
        "mapType": map_type,
        "token": resolved_key,
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
            headers={"User-Agent": APPLE_MAPS_USER_AGENT},
        )
        response.raise_for_status()
        return response.content
    except Exception as exc:
        logging.warning("Apple Maps snapshot request failed: %s", exc)
        return None
