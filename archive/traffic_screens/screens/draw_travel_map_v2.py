#!/usr/bin/env python3
"""Render a traffic map using Apple Maps for the configured commute routes."""

from __future__ import annotations

import logging
import math
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageEnhance

from config import (
    APPLE_MAPS_API_KEY,
    APPLE_MAPS_SNAPSHOT_URL,
    FONT_TRAVEL_VALUE,
    FONT_WEATHER_DETAILS_SMALL_BOLD,
    HEIGHT,
    LATITUDE,
    LONGITUDE,
    WIDTH,
    get_screen_background_color,
)
from screens.draw_travel_time import (
    TRAVEL_ICON_294,
    TRAVEL_ICON_90,
    TRAVEL_ICON_94,
    TRAVEL_ICON_LSD,
    TravelTimeResult,
    _compose_icons,
    is_travel_screen_active,
)
from screens.draw_travel_time_v2 import get_travel_routes_v2
from services.apple_maps import fetch_apple_maps_snapshot
from utils import ScreenImage, log_call

ROUTE_ICON_HEIGHT = 26
MAP_MARGIN = 6
LEGEND_GAP = 4
LEGEND_PADDING = 4
LEGEND_ROW_GAP = 2
LEGEND_ICON_HEIGHT = 18
LEGEND_VALUE_FONT = FONT_WEATHER_DETAILS_SMALL_BOLD
BACKGROUND_COLOR = get_screen_background_color("travel map v2", (18, 18, 18))
MAP_COLOR = (36, 36, 36)
MAP_NIGHT_BRIGHTNESS = 0.9
MAP_ZOOM_LEVELS = range(18, 6, -1)
ROUTE_METADATA = {
    "lake_shore": {
        "label": "Lake Shore",
        "short_label": "LSD",
        "icons": [TRAVEL_ICON_LSD],
        "color": (90, 170, 255),
    },
    "kennedy_edens": {
        "label": "I-90 / I-94",
        "short_label": "90/94",
        "icons": [TRAVEL_ICON_90, TRAVEL_ICON_94],
        "color": (186, 140, 255),
    },
    "kennedy_294": {
        "label": "I-90 / I-294",
        "short_label": "90/294",
        "icons": [TRAVEL_ICON_90, TRAVEL_ICON_294],
        "color": (255, 160, 100),
    },
}
ROUTE_ORDER = ["lake_shore", "kennedy_edens", "kennedy_294"]


def _decode_polyline(polyline: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    index = 0
    lat = lng = 0

    while index < len(polyline):
        shift = result = 0
        while True:
            b = ord(polyline[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if result & 1 else result >> 1
        lat += dlat

        shift = result = 0
        while True:
            b = ord(polyline[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if result & 1 else result >> 1
        lng += dlng

        points.append((lat / 1e5, lng / 1e5))

    return points


def _flatten(points_by_route: Iterable[Sequence[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    flattened: List[Tuple[float, float]] = []
    for points in points_by_route:
        flattened.extend(points)
    return flattened


def _bounds(points: Sequence[Tuple[float, float]]):
    lats = [p[0] for p in points]
    lngs = [p[1] for p in points]
    return (min(lats), min(lngs)), (max(lats), max(lngs))


def _latlng_to_world_xy(lat: float, lng: float, zoom: int) -> Tuple[float, float]:
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    scale = 256 * (2 ** zoom)
    x = (lng + 180.0) / 360.0 * scale
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * scale
    return x, y


def _project(point: Tuple[float, float], top_left, bottom_right, width: int, height: int) -> Tuple[int, int]:
    (min_lat, min_lng) = top_left
    (max_lat, max_lng) = bottom_right
    lat, lng = point

    if max_lat == min_lat or max_lng == min_lng:
        return width // 2, height // 2

    x = (lng - min_lng) / (max_lng - min_lng)
    y = 1 - (lat - min_lat) / (max_lat - min_lat)

    return int(MAP_MARGIN + x * (width - 2 * MAP_MARGIN)), int(
        MAP_MARGIN + y * (height - 2 * MAP_MARGIN)
    )


def _project_to_map(
    point: Tuple[float, float],
    center: Tuple[float, float],
    zoom: int,
    width: int,
    height: int,
) -> Tuple[int, int]:
    lat, lng = point
    center_lat, center_lng = center
    center_x, center_y = _latlng_to_world_xy(center_lat, center_lng, zoom)
    x, y = _latlng_to_world_xy(lat, lng, zoom)
    return int((x - center_x) + width / 2), int((y - center_y) + height / 2)


def _select_map_view(
    polylines: Iterable[Sequence[Tuple[float, float]]],
    canvas_size: Tuple[int, int],
    fallback_center: Tuple[float, float],
) -> Tuple[Tuple[float, float], int]:
    all_points = _flatten(polylines)
    if not all_points:
        return fallback_center, 12

    (min_lat, min_lng), (max_lat, max_lng) = _bounds(all_points)
    midpoint_candidates: List[Tuple[float, float]] = []
    for points in polylines:
        if len(points) < 2:
            continue
        start_lat, start_lng = points[0]
        end_lat, end_lng = points[-1]
        midpoint_candidates.append(((start_lat + end_lat) / 2, (start_lng + end_lng) / 2))
    if midpoint_candidates:
        center_lat = sum(lat for lat, _ in midpoint_candidates) / len(midpoint_candidates)
        center_lng = sum(lng for _, lng in midpoint_candidates) / len(midpoint_candidates)
        center = (center_lat, center_lng)
    else:
        center = ((min_lat + max_lat) / 2, (min_lng + max_lng) / 2)
    available_w = max(1, canvas_size[0] - 2 * MAP_MARGIN)
    available_h = max(1, canvas_size[1] - 2 * MAP_MARGIN)

    for zoom in MAP_ZOOM_LEVELS:
        xs: List[float] = []
        ys: List[float] = []
        for lat, lng in ((min_lat, min_lng), (min_lat, max_lng), (max_lat, min_lng), (max_lat, max_lng)):
            x, y = _latlng_to_world_xy(lat, lng, zoom)
            xs.append(x)
            ys.append(y)
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        if span_x <= available_w and span_y <= available_h:
            return center, zoom

    return center, 8


def _route_ratio(route: Optional[dict]) -> Optional[float]:
    if not route:
        return None
    traffic = route.get("_duration_sec")
    baseline = route.get("_duration_base_sec")
    if traffic and baseline:
        return traffic / baseline if baseline else None
    return None


def _traffic_color_for_ratio(ratio: Optional[float]) -> Tuple[int, int, int]:
    if ratio is None:
        return (160, 160, 160)

    if ratio <= 1.1:
        return (40, 200, 120)
    if ratio <= 1.35:
        return (255, 195, 60)
    return (240, 80, 80)


def _step_ratio(step: dict, fallback: Optional[float]) -> Optional[float]:
    duration_value = step.get("duration") or step.get("expectedTravelTime")
    traffic_value = step.get("duration_in_traffic") or step.get("expectedTravelTimeWithTraffic")

    def _coerce(value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict) and isinstance(value.get("value"), (int, float)):
            return float(value["value"])
        return None

    duration = _coerce(duration_value)
    traffic = _coerce(traffic_value)

    if traffic is not None and duration is not None and duration:
        return traffic / duration
    if fallback is not None:
        return fallback
    if duration:
        return 1
    return None


def _step_points(step: dict) -> List[Tuple[float, float]]:
    if not isinstance(step, dict):
        return []

    if step.get("_path_points"):
        return step["_path_points"]

    polyline = step.get("_path_polyline") or step.get("polyline") or step.get("path")
    if isinstance(polyline, dict):
        encoded_step = polyline.get("points")
    elif isinstance(polyline, str):
        encoded_step = polyline
    else:
        encoded_step = None
    if not encoded_step or not isinstance(encoded_step, str):
        return []
    try:
        return _decode_polyline(encoded_step)
    except Exception:
        logging.warning("Travel map v2: failed to decode step polyline")
    return []


def _extract_route_segments(
    routes: Dict[str, Optional[dict]],
) -> Dict[str, List[Tuple[List[Tuple[float, float]], Tuple[int, int, int]]]]:
    """Return decoded route segments with per-step traffic colors."""

    segments: Dict[str, List[Tuple[List[Tuple[float, float]], Tuple[int, int, int]]]] = {}
    for key, route in routes.items():
        if not route:
            continue

        route_ratio = _route_ratio(route)
        steps = route.get("steps") or (route.get("legs") or [{}])[0].get("steps") or []
        for step in steps:
            points = _step_points(step)
            if len(points) < 2:
                continue
            ratio = _step_ratio(step, route_ratio)
            color = _traffic_color_for_ratio(ratio)
            segments.setdefault(key, []).append((points, color))

        if segments.get(key):
            continue

        if route.get("_overview_points"):
            segments[key] = [(route["_overview_points"], _traffic_color_for_ratio(route_ratio))]
            continue

        encoded = route.get("_overview_polyline")
        if encoded and isinstance(encoded, str):
            try:
                decoded = _decode_polyline(encoded)
                color = _traffic_color_for_ratio(route_ratio)
                segments[key] = [(decoded, color)]
            except Exception:
                logging.warning("Travel map v2: failed to decode overview polyline for %s", key)

    return segments


def _fetch_base_map(
    center: Tuple[float, float],
    zoom: int,
    size: Tuple[int, int],
) -> Optional[Image.Image]:
    content = fetch_apple_maps_snapshot(
        center,
        zoom,
        size,
        APPLE_MAPS_API_KEY or "",
        map_type="mutedStandard",
        url=APPLE_MAPS_SNAPSHOT_URL,
    )
    if not content:
        return None
    try:
        return Image.open(BytesIO(content)).convert("RGB")
    except Exception as exc:
        logging.warning("Travel map v2: failed to load Apple Maps snapshot: %s", exc)
        return None


def _draw_routes(
    draw: ImageDraw.ImageDraw,
    route_segments: Dict[str, List[Tuple[List[Tuple[float, float]], Tuple[int, int, int]]]],
    canvas_size: Tuple[int, int],
    map_view: Optional[Tuple[Tuple[float, float], int]] = None,
    route_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    route_order: Optional[Sequence[str]] = None,
) -> None:
    if not route_segments:
        return

    route_colors = route_colors or {}
    route_order = list(route_order) if route_order else list(route_segments.keys())

    if map_view:
        center, zoom = map_view
        projector = lambda pt: _project_to_map(pt, center, zoom, *canvas_size)
    else:
        all_points = _flatten(points for segments in route_segments.values() for points, _ in segments)
        top_left, bottom_right = _bounds(all_points)
        projector = lambda pt: _project(pt, top_left, bottom_right, *canvas_size)

    for key in route_order:
        segments = route_segments.get(key, [])
        if not segments:
            continue

        outline_color = route_colors.get(key, (200, 200, 200))
        for points, _ in segments:
            if len(points) < 2:
                continue
            projected = [projector(pt) for pt in points]
            draw.line(projected, fill=outline_color, width=8, joint="curve")

        for points, color in segments:
            if len(points) < 2:
                continue
            projected = [projector(pt) for pt in points]
            draw.line(projected, fill=color, width=4, joint="curve")


def _compose_legend_entry(
    value: str,
    icon_paths: Sequence[str],
    swatch_color: Tuple[int, int, int],
    value_color: Tuple[int, int, int],
    *,
    value_font=FONT_TRAVEL_VALUE,
    icon_height: int = ROUTE_ICON_HEIGHT,
) -> Image.Image:
    icon = _compose_icons(icon_paths, height=icon_height)
    swatch = Image.new("RGBA", (icon.width, icon.height), swatch_color + (255,))
    swatch.putalpha(128)
    swatch = swatch.convert("RGB")

    entry_height = max(icon.height, 20)
    padding = 4
    measurement = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    value_w, value_h = measurement.textsize(value, font=value_font)

    width = icon.width + value_w + padding * 3
    canvas = Image.new("RGB", (width, entry_height), (0, 0, 0))

    canvas.paste(swatch, (padding, (entry_height - swatch.height) // 2))
    canvas.paste(icon, (padding, (entry_height - icon.height) // 2), icon)

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (icon.width + padding * 2, (entry_height - value_h) // 2),
        value,
        font=value_font,
        fill=value_color,
    )

    return canvas


def _compose_legend(routes: Dict[str, Optional[dict]]) -> Optional[Image.Image]:
    entries: List[Image.Image] = []
    for key in ROUTE_ORDER:
        route = routes.get(key)
        meta = ROUTE_METADATA.get(key)
        if not meta:
            continue
        time_result = TravelTimeResult.from_route(route)
        value = time_result.normalized()
        value_color = time_result.color or (200, 200, 200)
        entry = _compose_legend_entry(
            value,
            meta["icons"],
            meta["color"],
            value_color,
            value_font=LEGEND_VALUE_FONT,
            icon_height=LEGEND_ICON_HEIGHT,
        )
        entries.append(entry)

    if not entries:
        return None

    legend_width = min(max(entry.width for entry in entries) + LEGEND_PADDING * 2, WIDTH)
    legend_height = (
        sum(entry.height for entry in entries)
        + LEGEND_ROW_GAP * (len(entries) - 1)
        + LEGEND_PADDING * 2
    )
    legend = Image.new("RGB", (legend_width, legend_height), BACKGROUND_COLOR)

    y = LEGEND_PADDING
    for entry in entries:
        x = LEGEND_PADDING + max(0, (legend_width - 2 * LEGEND_PADDING - entry.width) // 2)
        legend.paste(entry, (x, y))
        y += entry.height + LEGEND_ROW_GAP

    return legend


def _compose_travel_map(routes: Dict[str, Optional[dict]]) -> Image.Image:
    route_segments = _extract_route_segments(routes)
    brightness = MAP_NIGHT_BRIGHTNESS

    legend = _compose_legend(routes)

    polylines = [points for segments in route_segments.values() for points, _ in segments]
    map_view = _select_map_view(polylines, (WIDTH, HEIGHT), (LATITUDE, LONGITUDE))
    base_map = _fetch_base_map(map_view[0], map_view[1], (WIDTH, HEIGHT))
    if base_map is None:
        map_canvas = Image.new("RGB", (WIDTH, HEIGHT), MAP_COLOR)
    else:
        map_canvas = ImageEnhance.Brightness(base_map).enhance(brightness)

    draw = ImageDraw.Draw(map_canvas)
    _draw_routes(
        draw,
        route_segments,
        (WIDTH, HEIGHT),
        map_view=map_view,
        route_colors={key: meta["color"] for key, meta in ROUTE_METADATA.items()},
        route_order=ROUTE_ORDER,
    )

    if legend:
        legend_x = max(0, WIDTH - legend.width - MAP_MARGIN)
        legend_y = MAP_MARGIN
        map_canvas.paste(legend, (legend_x, legend_y))

    return map_canvas


@log_call
def draw_travel_map_v2_screen(display, transition: bool = False) -> Optional[Image.Image | ScreenImage]:
    global BACKGROUND_COLOR
    BACKGROUND_COLOR = get_screen_background_color("travel map v2", (18, 18, 18))
    if not is_travel_screen_active():
        return None

    routes = get_travel_routes_v2()
    img = _compose_travel_map(routes)

    if display is not None:
        display.image(img)
        display.show()

    return ScreenImage(img, displayed=display is not None)


__all__ = ["draw_travel_map_v2_screen"]
