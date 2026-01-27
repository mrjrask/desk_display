#!/usr/bin/env python3
"""Apple Maps-based travel time screen helpers (v2)."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from PIL import Image

from config import (
    APPLE_MAPS_API_KEY,
    APPLE_MAPS_DIRECTIONS_URL,
    TRAVEL_DESTINATION,
    TRAVEL_ORIGIN,
)
from screens.draw_travel_time import (
    TravelTimeResult,
    _compose_travel_image,
    _select_travel_routes,
    is_travel_screen_active,
)
from services.apple_maps import fetch_apple_maps_routes
from utils import ScreenImage, log_call


def _fetch_routes(*, avoid_highways: bool = False, avoid_tolls: bool = False):
    return fetch_apple_maps_routes(
        TRAVEL_ORIGIN,
        TRAVEL_DESTINATION,
        APPLE_MAPS_API_KEY or "",
        avoid_highways=avoid_highways,
        avoid_tolls=avoid_tolls,
        url=APPLE_MAPS_DIRECTIONS_URL,
    )


def get_travel_routes_v2() -> Dict[str, Optional[dict]]:
    """Return Apple Maps route objects keyed by route identifier."""

    try:
        routes_all = list(_fetch_routes(avoid_highways=False))
        lake_shore_routes = list(_fetch_routes(avoid_highways=True, avoid_tolls=True))
        kennedy_edens_routes = list(_fetch_routes(avoid_tolls=True))
        return _select_travel_routes(
            routes_all,
            lake_shore_routes=lake_shore_routes,
            kennedy_edens_routes=kennedy_edens_routes,
        )
    except Exception as exc:  # pragma: no cover - defensive guard for runtime issues
        logging.warning("Travel v2 route fetch failed: %s", exc)
        return {
            "lake_shore": None,
            "kennedy_edens": None,
            "kennedy_294": None,
        }


def get_travel_times_v2() -> Dict[str, TravelTimeResult]:
    """Return formatted travel times keyed by route identifier."""

    routes = get_travel_routes_v2()
    return {key: TravelTimeResult.from_route(route) for key, route in routes.items()}


@log_call
def draw_travel_time_v2_screen(
    display, transition: bool = False
) -> Optional[Image.Image | ScreenImage]:
    if not is_travel_screen_active():
        return None

    times = get_travel_times_v2()
    img = _compose_travel_image(times)

    if display is not None:
        display.image(img)
        display.show()

    return ScreenImage(img, displayed=display is not None)


__all__ = ["draw_travel_time_v2_screen", "get_travel_routes_v2", "get_travel_times_v2"]
