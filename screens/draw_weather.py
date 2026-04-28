#!/usr/bin/env python3
"""
draw_weather.py

Two weather screens (basic + detailed) in RGB.

Screen 1:
  • Temp & description at top
  • 64×64 weather icon
  • Two-line Feels/Hi/Lo: labels on the line above values, each centered.

Screen 2:
  • Detailed info: Sunrise/Sunset, Wind, Gust, Humidity, Pressure, UV Index
  • Each label/value pair vertically centered within its row.
"""

import datetime
import logging
import math
import time
from io import BytesIO
from typing import NamedTuple, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

import config
from config import (
    WIDTH,
    HEIGHT,
    CENTRAL_TIME,
    FONT_TEMP,
    FONT_CONDITION,
    FONT_WEATHER_LABEL,
    FONT_WEATHER_DETAILS,
    FONT_WEATHER_DETAILS_BOLD,
    FONT_WEATHER_DETAILS_SMALL,
    FONT_WEATHER_DETAILS_TINY,
    FONT_WEATHER_DETAILS_TINY_LARGE,
    FONT_WEATHER_DETAILS_TINY_MICRO,
    FONT_WEATHER_DETAILS_SMALL_BOLD,
    FONT_EMOJI,
    FONT_EMOJI_SMALL,
    EMOJI_EMBEDDED_COLOR,
    WEATHER_ICON_SIZE,
    WEATHER_DESC_GAP,
    HOURLY_FORECAST_HOURS,
    LATITUDE,
    LONGITUDE,
    WEATHER_USE_EMOJI_ICONS,
    get_screen_background_color,
    is_hyperpixel_4_square_layout,
    is_hyperpixel_next_layout,
)
from utils import (
    LED_INDICATOR_LEVEL,
    ScreenImage,
    fetch_weather_icon,
    log_call,
    timestamp_to_datetime,
    uv_index_color,
    wind_direction,
)

ALERT_SYMBOL = "⚠️"
ALERT_PRIORITY = {"warning": 3, "watch": 2, "hazard": 1}
ALERT_LED_COLORS = {
    "warning": (LED_INDICATOR_LEVEL, 0.0, 0.0),
    "watch": (LED_INDICATOR_LEVEL, LED_INDICATOR_LEVEL * 0.5, 0.0),
    "hazard": (LED_INDICATOR_LEVEL, LED_INDICATOR_LEVEL, 0.0),
}
ALERT_ICON_COLORS = {
    "warning": (255, 64, 64),
    "watch": (255, 165, 0),
    "hazard": (255, 215, 0),
}
SUN_EVENT_GRACE = datetime.timedelta(minutes=20)
PRESSURE_TREND_SYMBOLS = {
    "rising": ("↑", (0, 255, 0)),
    "falling": ("↓", (255, 0, 0)),
    "steady": ("↔", (255, 255, 255)),
}
EMOJI_DRAW_KWARGS = {"embedded_color": True} if EMOJI_EMBEDDED_COLOR else {}
TEMPERATURE_COLOR_STOPS_F: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (-10.0, (211, 46, 179)),
    (0.0, (172, 45, 176)),
    (10.0, (134, 67, 186)),
    (20.0, (76, 86, 232)),
    (30.0, (30, 160, 236)),
    (40.0, (0, 184, 105)),
    (50.0, (155, 206, 96)),
    (60.0, (255, 214, 0)),
    (70.0, (243, 171, 90)),
    (80.0, (255, 58, 20)),
    (90.0, (229, 30, 0)),
    (100.0, (204, 0, 0)),
)

# Keep weather radar tiles centered on downtown Chicago regardless of the
# weather location used for forecast details.
RADAR_CENTER_LATITUDE = 41.8781
RADAR_CENTER_LONGITUDE = -87.6298
RADAR_MAX_FRAME_AGE = datetime.timedelta(hours=2)
RADAR_ANIMATION_FRAME_DELAY_SECONDS = 0.2
RADAR_ANIMATION_LOOPS = 3
RAINVIEWER_METADATA_URLS = (
    "https://api.rainviewer.com/public/weather-maps.json",
    # RainViewer's free metadata endpoint has moved at times; keep a fallback.
    "https://api.rainviewer.com/public/maps.json",
)


_IS_1080P_LAYOUT = config.is_hdmi_1080p_layout()

def _safe_textbbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
) -> tuple[int, int, int, int]:
    """Return a safe text bounding box for emoji/text rendering."""

    try:
        return draw.textbbox((0, 0), text, font=font, **EMOJI_DRAW_KWARGS)
    except Exception:
        try:
            return draw.textbbox((0, 0), text, font=font)
        except Exception:
            width, height = draw.textsize(text, font=font)
            return (0, 0, width, height)


def _draw_text_with_fallback(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int] | Tuple[int, int, int, int],
) -> None:
    """Draw text, retrying without embedded color if needed."""

    try:
        draw.text(position, text, font=font, fill=fill, **EMOJI_DRAW_KWARGS)
    except Exception as exc:
        if EMOJI_DRAW_KWARGS:
            logging.debug("Emoji text draw failed with embedded color: %s", exc)
            draw.text(position, text, font=font, fill=fill)
        else:
            raise


def _render_emoji_glyph(
    emoji: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int] | Tuple[int, int, int, int],
) -> Image.Image:
    scratch = Image.new("RGB", (1, 1))
    scratch_draw = ImageDraw.Draw(scratch)
    bbox = _safe_textbbox(scratch_draw, emoji, font)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    icon = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    icon_draw = ImageDraw.Draw(icon)
    _draw_text_with_fallback(
        icon_draw,
        (-bbox[0], -bbox[1]),
        emoji,
        font,
        fill,
    )
    return icon


def _ensure_rgba_icon(icon: Image.Image) -> Image.Image:
    if icon.mode != "RGBA" or "A" not in icon.getbands():
        return icon.convert("RGBA")
    return icon


def _render_stat_text(parts):
    """Render a left-to-right text image from ``(text, font, color)`` parts."""

    scratch = Image.new("RGB", (1, 1))
    scratch_draw = ImageDraw.Draw(scratch)

    widths = []
    min_y = 0
    max_y = 0
    for text, font, _ in parts:
        bbox = _safe_textbbox(scratch_draw, text, font)
        w = bbox[2] - bbox[0]
        widths.append(w)
        min_y = min(min_y, bbox[1])
        max_y = max(max_y, bbox[3])

    # Add a cushion to avoid clipping descenders/antialiasing for tall glyphs,
    # especially on narrow hourly stat rows.
    padding_x = 1
    padding_y = 4
    content_h = max(0, max_y - min_y)
    total_w = sum(widths) + padding_x * 2
    total_h = content_h + padding_y * 2
    result = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(result)

    x = padding_x
    for (text, font, color), w in zip(parts, widths):
        y = padding_y - min_y
        draw.text((x, y), text, font=font, fill=color)
        x += w

    return result


def _temperature_chart_color(temp_f: float | int | None) -> tuple[int, int, int]:
    """Map Fahrenheit values to the rainbow weather-chart palette."""

    try:
        value = float(temp_f)
    except (TypeError, ValueError):
        return (255, 255, 255)

    stops = TEMPERATURE_COLOR_STOPS_F
    if value <= stops[0][0]:
        return stops[0][1]
    if value >= stops[-1][0]:
        return stops[-1][1]

    for idx in range(1, len(stops)):
        low_temp, low_color = stops[idx - 1]
        high_temp, high_color = stops[idx]
        if value <= high_temp:
            span = high_temp - low_temp or 1.0
            alpha = (value - low_temp) / span
            return tuple(
                int(round(low + (high - low) * alpha))
                for low, high in zip(low_color, high_color)
            )

    return stops[-1][1]


def _pop_pct_from(entry):
    if not isinstance(entry, dict):
        return None
    pop_raw = entry.get("pop")
    if pop_raw is None:
        pop_raw = entry.get("probabilityOfPrecipitation")
    if pop_raw is None:
        return None
    try:
        pop_val = float(pop_raw)
    except Exception:
        return None
    if 0 <= pop_val <= 1:
        pop_val *= 100
    return int(round(pop_val))


def _is_snow_condition(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False

    weather_list = entry.get("weather") if isinstance(entry.get("weather"), list) else []
    weather = (weather_list or [{}])[0]
    weather_id = weather.get("id")
    weather_main = (weather.get("main") or "").strip().lower()

    if weather_main == "snow":
        return True
    if isinstance(weather_id, int) and 600 <= weather_id < 700:
        return True
    if isinstance(weather_id, int) and weather_id in {511}:
        return True
    frozen_tokens = ("sleet", "freez", "ice", "wintry", "mix")
    if any(token in weather_main for token in frozen_tokens):
        return True
    description = (weather.get("description") or "").strip().lower()
    if any(token in description for token in frozen_tokens):
        return True
    if entry.get("snow"):
        return True

    return False


def _normalise_alerts(weather: object) -> list:
    alerts = []
    if isinstance(weather, dict):
        raw_alerts = weather.get("alerts")
    else:
        raw_alerts = None

    if isinstance(raw_alerts, list):
        alerts = [alert for alert in raw_alerts if isinstance(alert, dict)]
    elif isinstance(raw_alerts, dict):
        inner = raw_alerts.get("alerts")
        if isinstance(inner, list):
            alerts = [alert for alert in inner if isinstance(alert, dict)]
        else:
            alerts = [raw_alerts]
    return alerts


def _classify_alert(alert: dict) -> Optional[str]:
    texts = []
    for key in ("event", "title", "headline"):
        value = alert.get(key)
        if isinstance(value, str):
            texts.append(value.lower())
    tags = alert.get("tags")
    if isinstance(tags, (list, tuple, set)):
        texts.extend(str(tag).lower() for tag in tags if tag)
    description = alert.get("description")
    if isinstance(description, str):
        texts.append(description.lower())

    for text in texts:
        if "warning" in text:
            return "warning"
    for text in texts:
        if "watch" in text:
            return "watch"
    for text in texts:
        if any(token in text for token in ("hazard", "alert", "advisory")):
            return "hazard"
    return None


def _render_precip_icon(is_snow: bool, size: int, color: Tuple[int, int, int]) -> Image.Image:
    """Return a simple precipitation marker that doesn't rely on emoji fonts.

    Some systems don't ship an emoji font Pillow can render, which results in
    an empty box for the precipitation glyph. Drawing a small vector icon keeps
    the UI legible regardless of available fonts.
    """

    size = max(8, size)
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon_draw = ImageDraw.Draw(icon)

    if is_snow:
        center = size / 2
        radius = size * 0.42
        arm_width = max(1, int(round(size * 0.09)))
        branch = radius * 0.4
        for idx in range(6):
            angle = math.radians(idx * 60)
            end_x = center + radius * math.cos(angle)
            end_y = center + radius * math.sin(angle)
            icon_draw.line((center, center, end_x, end_y), fill=color, width=arm_width)

            branch_dx = branch * math.sin(angle)
            branch_dy = branch * math.cos(angle)
            icon_draw.line(
                (end_x, end_y, end_x - branch_dx, end_y + branch_dy),
                fill=color,
                width=max(1, arm_width - 1),
            )
            icon_draw.line(
                (end_x, end_y, end_x + branch_dx, end_y - branch_dy),
                fill=color,
                width=max(1, arm_width - 1),
            )
    else:
        center_x = size / 2
        base_radius = size * 0.34
        base_center_y = size * 0.64
        tip_y = size * 0.08

        drop_mask = Image.new("L", (size, size), 0)
        mask_draw = ImageDraw.Draw(drop_mask)

        mask_draw.ellipse(
            (
                center_x - base_radius,
                base_center_y - base_radius,
                center_x + base_radius,
                base_center_y + base_radius,
            ),
            fill=255,
        )

        shoulder_offset = base_radius * 0.9
        shoulder_height = base_radius * 0.75
        mask_draw.polygon(
            [
                (center_x, tip_y),
                (center_x - shoulder_offset, base_center_y - shoulder_height),
                (center_x + shoulder_offset, base_center_y - shoulder_height),
            ],
            fill=255,
        )

        body = Image.new("RGBA", (size, size), color + (255,))
        icon.paste(body, mask=drop_mask)

        highlight = Image.new("L", (size, size), 0)
        highlight_draw = ImageDraw.Draw(highlight)
        highlight_draw.ellipse(
            (
                center_x - base_radius * 0.35,
                base_center_y - base_radius * 0.9,
                center_x - base_radius * 0.05,
                base_center_y - base_radius * 0.25,
            ),
            fill=80,
        )
        highlight_color = Image.new("RGBA", (size, size), (255, 255, 255, 140))
        icon.paste(highlight_color, mask=highlight)

    return icon


def _detect_weather_alert(weather: object) -> Tuple[Optional[str], Optional[Tuple[float, float, float]]]:
    alerts = _normalise_alerts(weather)
    severity: Optional[str] = None
    for alert in alerts:
        level = _classify_alert(alert)
        if level is None:
            continue
        if severity is None or ALERT_PRIORITY[level] > ALERT_PRIORITY[severity]:
            severity = level
            if severity == "warning":
                break
    return severity, ALERT_LED_COLORS.get(severity)


def _draw_alert_indicator(
    img: Image.Image, draw: ImageDraw.ImageDraw, severity: Optional[str]
) -> None:
    if not severity:
        return
    icon_color = ALERT_ICON_COLORS.get(severity, (255, 215, 0))
    icon_img = _ensure_rgba_icon(_render_emoji_glyph(ALERT_SYMBOL, FONT_EMOJI_SMALL, icon_color))
    w_icon, h_icon = icon_img.size
    x_icon = WIDTH - w_icon - 2
    y_icon = HEIGHT - h_icon - 2
    img.paste(icon_img, (x_icon, y_icon), icon_img)

# ─── Screen 1: Basic weather + two-line Feels/Hi/Lo ────────────────────────────
@log_call
def draw_weather_screen_1(display, weather, transition=False):
    if not weather:
        return None

    background = get_screen_background_color("weather1", (0, 0, 0))
    severity, led_color = _detect_weather_alert(weather)

    current = weather.get("current", {})
    daily   = weather.get("daily", [{}])[0]
    hourly  = weather.get("hourly") if isinstance(weather.get("hourly"), list) else None

    temp  = round(current.get("temp", 0))
    desc  = current.get("weather", [{}])[0].get("description", "").title()

    feels = round(current.get("feels_like", 0))
    hi    = round(daily.get("temp", {}).get("max", 0))
    lo    = round(daily.get("temp", {}).get("min", 0))

    img  = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)

    # Temperature
    temp_str = f"{temp}°F"
    w_temp, h_temp = draw.textsize(temp_str, font=FONT_TEMP)
    draw.text(
        ((WIDTH - w_temp) // 2, 0),
        temp_str,
        font=FONT_TEMP,
        fill=_temperature_chart_color(temp),
    )

    font_desc = FONT_CONDITION
    w_desc, h_desc = draw.textsize(desc, font=font_desc)
    if w_desc > WIDTH:
        font_desc = FONT_WEATHER_DETAILS_BOLD
        w_desc, h_desc = draw.textsize(desc, font=font_desc)
    draw.text(
        ((WIDTH - w_desc)//2, h_temp + WEATHER_DESC_GAP),
        desc,
        font=font_desc,
        fill=(255,255,255)
    )

    cloud_cover = current.get("clouds")
    try:
        cloud_cover = int(round(float(cloud_cover)))
    except Exception:
        cloud_cover = None

    pop_pct = None
    next_hour = None
    if hourly:
        current_dt = current.get("dt")
        if isinstance(current_dt, (int, float)):
            for hour in hourly:
                if not isinstance(hour, dict):
                    continue
                hour_dt = hour.get("dt")
                if isinstance(hour_dt, (int, float)) and hour_dt > current_dt:
                    next_hour = hour
                    break
        if next_hour is None:
            if len(hourly) > 1 and isinstance(hourly[1], dict):
                next_hour = hourly[1]
            elif hourly and isinstance(hourly[0], dict):
                next_hour = hourly[0]
        pop_pct = _pop_pct_from(next_hour)

    if pop_pct is None:
        pop_pct = _pop_pct_from(daily)

    daily_weather_list = daily.get("weather") if isinstance(daily.get("weather"), list) else []
    daily_weather = (daily_weather_list or [{}])[0]
    is_snow = _is_snow_condition(daily) or _is_snow_condition(current)
    if not is_snow and next_hour:
        is_snow = _is_snow_condition(next_hour)

    precip_percent = None
    if pop_pct is not None:
        precip_percent = f"{max(0, min(pop_pct, 100))}%"

    precip_intensity_text = _format_precip_intensity_inches_per_hour(current.get("precipitation_intensity"))

    cloud_percent = None
    if cloud_cover is not None:
        cloud_percent = f"{max(0, min(cloud_cover, 100))}%"

    visibility_text = _format_visibility_miles(current.get("visibility"))

    # Feels/Hi/Lo groups
    labels    = ["Feels", "Hi", "Lo"]
    values    = [f"{feels}°", f"{hi}°", f"{lo}°"]
    # dynamic colors
    if feels > hi:
        feels_col = (255,165,0)
    elif feels < lo:
        feels_col = uv_index_color(2)
    else:
        feels_col = (255,255,255)
    val_colors = [feels_col, (255,0,0), (0,0,255)]

    groups = []
    for lbl, val in zip(labels, values):
        lw, lh = draw.textsize(lbl, font=FONT_WEATHER_LABEL)
        vw, vh = draw.textsize(val, font=FONT_WEATHER_DETAILS)
        gw = max(lw, vw)
        groups.append((lbl, lw, lh, val, vw, vh, gw))

    # horizontal layout
    SPACING_X = 12
    total_w   = sum(g[6] for g in groups) + SPACING_X * (len(groups)-1)
    x0        = (WIDTH - total_w)//2

    # vertical positions
    max_val_h = max(g[5] for g in groups)
    max_lbl_h = max(g[2] for g in groups)
    if _IS_1080P_LAYOUT:
        bottom_safe_margin = 36
    elif is_hyperpixel_4_square_layout():
        bottom_safe_margin = 18
    elif is_hyperpixel_next_layout():
        bottom_safe_margin = 15
    else:
        bottom_safe_margin = 9
    y_val = HEIGHT - max_val_h - bottom_safe_margin
    LABEL_GAP = 2
    y_lbl     = y_val - max_lbl_h - LABEL_GAP

    # paste icon between desc and labels
    top_of_icons = h_temp + h_desc + WEATHER_DESC_GAP * 2
    current_weather = current.get("weather", [{}])[0]
    icon_code = current_weather.get("icon")
    condition_code = current_weather.get("condition_code")
    is_daylight = current.get("is_daylight")
    # Fit the weather icon into the available gap between the description and
    # the Feels/Hi/Lo labels so it doesn't overlap other content.
    available_icon_height = y_lbl - top_of_icons
    if available_icon_height > 0:
        weather_icon_size = max(1, min(WEATHER_ICON_SIZE, available_icon_height))
    else:
        weather_icon_size = min(WEATHER_ICON_SIZE, HEIGHT // 2)
    icon_img = fetch_weather_icon(
        icon_code,
        weather_icon_size,
        condition_code=condition_code,
        is_daylight=is_daylight,
    )
    y_icon = top_of_icons + ((y_lbl - top_of_icons - weather_icon_size)//2)
    icon_x = (WIDTH - weather_icon_size) // 2
    icon_center_y = top_of_icons + max(0, (y_lbl - top_of_icons) // 2)

    if icon_img:
        icon_img = _ensure_rgba_icon(icon_img)
        img.paste(icon_img, (icon_x, y_icon), icon_img)

    side_font = FONT_WEATHER_DETAILS
    sub_font = FONT_WEATHER_DETAILS_TINY
    stack_gap = 2
    edge_margin = 4
    if precip_percent:
        precip_emoji = "❄️" if is_snow else "💧"
        precip_color = (173, 216, 230) if is_snow else (135, 206, 250)
        precip_icon = _ensure_rgba_icon(_render_emoji_glyph(precip_emoji, FONT_EMOJI, precip_color))
        emoji_w, emoji_h = precip_icon.size
        pct_w, pct_h = draw.textsize(precip_percent, font=side_font)
        intensity_w, intensity_h = (0, 0)
        if precip_intensity_text:
            intensity_w, intensity_h = draw.textsize(precip_intensity_text, font=sub_font)
        block_w = max(emoji_w, pct_w, intensity_w)
        block_h = emoji_h + stack_gap + pct_h
        if precip_intensity_text:
            block_h += stack_gap + intensity_h
        left_available = max(0, icon_x - edge_margin)
        precip_x = edge_margin + max(0, (left_available - block_w) // 2)
        precip_x = min(precip_x, max(edge_margin, icon_x - block_w))
        block_y = icon_center_y - block_h // 2
        emoji_x = precip_x + (block_w - emoji_w) // 2
        pct_x = precip_x + (block_w - pct_w) // 2
        img.paste(precip_icon, (emoji_x, block_y), precip_icon)
        next_y = block_y + emoji_h + stack_gap
        draw.text((pct_x, next_y), precip_percent, font=side_font, fill=precip_color)
        if precip_intensity_text:
            intensity_x = precip_x + (block_w - intensity_w) // 2
            draw.text((intensity_x, next_y + pct_h + stack_gap), precip_intensity_text, font=sub_font, fill=precip_color)

    if cloud_percent:
        cloud_emoji = "☁️"
        cloud_icon = _ensure_rgba_icon(_render_emoji_glyph(cloud_emoji, FONT_EMOJI, (211, 211, 211)))
        emoji_w, emoji_h = cloud_icon.size
        pct_w, pct_h = draw.textsize(cloud_percent, font=side_font)
        visibility_w, visibility_h = (0, 0)
        if visibility_text:
            visibility_w, visibility_h = draw.textsize(visibility_text, font=sub_font)
        block_w = max(emoji_w, pct_w, visibility_w)
        block_h = emoji_h + stack_gap + pct_h
        if visibility_text:
            block_h += stack_gap + visibility_h
        right_start = icon_x + weather_icon_size
        right_available = max(0, WIDTH - edge_margin - right_start)
        cloud_x = right_start + max(0, (right_available - block_w) // 2)
        cloud_x = min(cloud_x, max(edge_margin, WIDTH - edge_margin - block_w))
        block_y = icon_center_y - block_h // 2
        emoji_x = cloud_x + (block_w - emoji_w) // 2
        pct_x = cloud_x + (block_w - pct_w) // 2
        img.paste(cloud_icon, (emoji_x, block_y), cloud_icon)
        next_y = block_y + emoji_h + stack_gap
        draw.text((pct_x, next_y), cloud_percent, font=side_font, fill=(211, 211, 211))
        if visibility_text:
            visibility_x = cloud_x + (block_w - visibility_w) // 2
            draw.text((visibility_x, next_y + pct_h + stack_gap), visibility_text, font=sub_font, fill=(211, 211, 211))

    # draw groups
    x = x0
    for idx, (lbl, lw, lh, val, vw, vh, gw) in enumerate(groups):
        cx = x + gw//2
        draw.text((cx - lw//2, y_lbl), lbl, font=FONT_WEATHER_LABEL,      fill=(255,255,255))
        draw.text((cx - vw//2, y_val), val, font=FONT_WEATHER_DETAILS,     fill=val_colors[idx])
        x += gw + SPACING_X

    _draw_alert_indicator(img, draw, severity)


    return ScreenImage(img, displayed=False, led_override=led_color)


def _format_hour_label(timestamp: Optional[int], *, index: int) -> str:
    dt = timestamp_to_datetime(timestamp, CENTRAL_TIME)
    if dt:
        return dt.strftime("%-I%p").lower()
    return f"+{index}h"


def _normalise_condition(hour: dict) -> str:
    if not isinstance(hour, dict):
        return ""
    weather_list = hour.get("weather") if isinstance(hour.get("weather"), list) else []
    if weather_list:
        main_val = weather_list[0].get("main") or weather_list[0].get("description")
        if isinstance(main_val, str) and main_val.strip():
            return main_val.title()
    return ""


def _format_day_label(timestamp: Optional[int], *, index: int) -> str:
    dt = timestamp_to_datetime(timestamp, CENTRAL_TIME)
    if dt:
        return dt.strftime("%a")
    return f"+{index}d"


def _format_precip_intensity_inches_per_hour(intensity: object) -> Optional[str]:
    try:
        intensity_val = float(intensity)
    except (TypeError, ValueError):
        return None
    if intensity_val <= 0:
        return None
    intensity_in_hr = intensity_val * 0.0393701
    return f"{intensity_in_hr:.2f} in/hr"


def _format_visibility_miles(visibility: object) -> Optional[str]:
    try:
        visibility_val = float(visibility)
    except (TypeError, ValueError):
        return None
    if visibility_val < 0:
        return None
    visibility_mi = visibility_val * 0.000621371
    if visibility_mi >= 10:
        return f"{visibility_mi:.0f} mi"
    return f"{visibility_mi:.1f} mi"


def _wind_arrow(degrees: Optional[float]) -> str:
    try:
        deg_val = float(degrees)
    except (TypeError, ValueError):
        return ""

    arrows = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]
    idx = int((deg_val % 360) / 45.0 + 0.5) % len(arrows)
    return arrows[idx]


def _gather_hourly_forecast(
    weather: object, hours: int, *, now: Optional[datetime.datetime] = None
) -> list[dict]:
    if not isinstance(weather, dict):
        return []
    hourly = weather.get("hourly") if isinstance(weather.get("hourly"), list) else []
    reference_time = (now or datetime.datetime.now(CENTRAL_TIME)) - datetime.timedelta(minutes=5)

    future_hours = []
    for hour in hourly:
        ts = hour.get("dt") if isinstance(hour, dict) else None
        dt_val = timestamp_to_datetime(ts, CENTRAL_TIME)
        if dt_val and dt_val < reference_time:
            continue
        future_hours.append(hour)

    future_hours.sort(
        key=lambda h: h.get("dt") if isinstance(h, dict) and h.get("dt") is not None else float("inf")
    )

    # Sample the forecast every two hours so each column represents a two-hour block when there
    # is enough data. If we only have a handful of entries, show them all to avoid dropping
    # recent hours.
    if len(future_hours) > hours:
        two_hourly_forecast = future_hours[::2]
    else:
        two_hourly_forecast = future_hours

    forecast = []
    for idx, hour in enumerate(two_hourly_forecast[:hours]):
        if not isinstance(hour, dict):
            continue
        wind_speed = None
        try:
            wind_speed = int(round(float(hour.get("wind_speed", 0))))
        except Exception:
            wind_speed = None
        wind_dir = ""
        if hour.get("wind_deg") is not None:
            wind_dir = _wind_arrow(hour.get("wind_deg")) or wind_direction(hour.get("wind_deg"))
        uvi_val = None
        try:
            uvi_val = int(round(float(hour.get("uvi", 0))))
        except Exception:
            uvi_val = None

        # Detect if precipitation is snow or rain
        weather_list = hour.get("weather") if isinstance(hour.get("weather"), list) else []
        hourly_weather = (weather_list or [{}])[0]
        is_snow = _is_snow_condition(hour)

        feels_like_val = None
        try:
            feels_like_val = round(float(hour.get("feels_like", 0)))
        except Exception:
            feels_like_val = None

        entry = {
            "temp": round(hour.get("temp", 0)),
            "time": _format_hour_label(hour.get("dt"), index=(idx + 1) * 2),
            "condition": _normalise_condition(hour),
            "icon": None,
            "condition_code": None,
            "pop": _pop_pct_from(hour),
            "wind_speed": wind_speed,
            "wind_dir": wind_dir,
            "uvi": uvi_val,
            "is_snow": is_snow,
            "feels_like": feels_like_val,
            "is_daylight": hour.get("is_daylight"),
        }
        if weather_list:
            entry["icon"] = weather_list[0].get("icon")
            entry["condition_code"] = weather_list[0].get("condition_code")
        forecast.append(entry)
    return forecast


def _gather_daily_forecast(weather: object, days: int) -> list[dict]:
    if not isinstance(weather, dict):
        return []
    daily = weather.get("daily") if isinstance(weather.get("daily"), list) else []
    if not daily:
        return []

    start_idx = 1 if len(daily) > 1 else 0
    entries = daily[start_idx : start_idx + days]
    forecast = []

    for idx, day in enumerate(entries):
        if not isinstance(day, dict):
            continue
        weather_list = day.get("weather") if isinstance(day.get("weather"), list) else []
        daily_weather = (weather_list or [{}])[0]
        temp_data = day.get("temp") if isinstance(day.get("temp"), dict) else {}
        try:
            hi_val = round(float(temp_data.get("max", 0)))
        except Exception:
            hi_val = None
        try:
            lo_val = round(float(temp_data.get("min", 0)))
        except Exception:
            lo_val = None

        entry = {
            "day": _format_day_label(day.get("dt"), index=idx + 1),
            "hi": hi_val,
            "lo": lo_val,
            "pop": _pop_pct_from(day),
            "is_snow": _is_snow_condition(day),
            "condition": _normalise_condition(day),
            "icon": daily_weather.get("icon"),
            "condition_code": daily_weather.get("condition_code"),
        }
        forecast.append(entry)
    return forecast


@log_call
def draw_weather_hourly(display, weather, transition: bool = False, hours: int = HOURLY_FORECAST_HOURS):
    background = get_screen_background_color("weather hourly", (0, 0, 0))
    forecast = _gather_hourly_forecast(weather, hours)
    if not forecast:
        img = Image.new("RGB", (WIDTH, HEIGHT), background)
        draw = ImageDraw.Draw(img)
        msg = "No hourly data"
        w, h = draw.textsize(msg, font=FONT_WEATHER_DETAILS_BOLD)
        draw.text(((WIDTH - w) // 2, (HEIGHT - h) // 2), msg, font=FONT_WEATHER_DETAILS_BOLD, fill=(255, 255, 255))
        return ScreenImage(img, displayed=False)

    img = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)

    hours_to_show = len(forecast)
    title = "Next 10 hours..."
    title_w, title_h = draw.textsize(title, font=FONT_WEATHER_LABEL)
    title_x = (WIDTH - title_w) // 2
    title_y = 2
    if is_hyperpixel_next_layout():
        title_y = 4
    draw.text((title_x, title_y), title, font=FONT_WEATHER_LABEL, fill=(200, 200, 200))

    gap = 4
    available_width = WIDTH - gap * (hours_to_show + 1)
    col_w = max(1, available_width // hours_to_show)
    icon_cache: dict[tuple[Optional[str], Optional[str], Optional[bool], bool], Optional[Image.Image]] = {}
    if WEATHER_USE_EMOJI_ICONS:
        icon_size = max(18, min(WEATHER_ICON_SIZE, col_w - 12))
    else:
        icon_size = max(32, min(WEATHER_ICON_SIZE, col_w - 10))
    time_font = FONT_WEATHER_DETAILS_SMALL_BOLD

    card_top = title_h + 6
    card_bottom = HEIGHT - 6
    card_height = card_bottom - card_top
    x_start = (WIDTH - (hours_to_show * col_w + gap * (hours_to_show - 1))) // 2

    card_layouts = []
    temps = []

    for idx, hour in enumerate(forecast):
        x0 = x_start + idx * (col_w + gap)
        x1 = x0 + col_w
        cx = (x0 + x1) // 2

        draw.rounded_rectangle(
            (x0, card_top, x1, card_bottom),
            radius=6,
            fill=(18, 18, 28),
            outline=(40, 40, 60),
        )

        time_label = hour.get("time", "")
        time_w, time_h = draw.textsize(time_label, font=time_font)

        trend_area_top = card_top + 6 + time_h + 6
        trend_area_bottom = card_top + int(card_height * 0.36)
        if trend_area_bottom - trend_area_top < 16:
            trend_area_bottom = trend_area_top + 16

        icon_area_top = trend_area_bottom + 6
        icon_area_bottom = card_top + int(card_height * 0.68)

        stat_area_top = icon_area_bottom + 8
        stat_area_bottom = card_bottom - (10 if is_hyperpixel_4_square_layout() else 8)

        card_layouts.append(
            {
                "hour": hour,
                "x0": x0,
                "x1": x1,
                "cx": cx,
                "time_label": time_label,
                "time_size": (time_w, time_h),
                "trend_area": (trend_area_top, trend_area_bottom),
                "icon_area": (icon_area_top, icon_area_bottom),
                "stat_area": (stat_area_top, stat_area_bottom),
            }
        )
        temps.append(hour.get("temp", 0))

    if temps:
        min_temp = min(temps)
        max_temp = max(temps)
    else:
        min_temp = max_temp = 0

    temp_range = max(1, max_temp - min_temp)

    for layout in card_layouts:
        hour = layout["hour"]
        x0, x1 = layout["x0"], layout["x1"]
        cx = layout["cx"]
        time_label = layout["time_label"]
        time_w, time_h = layout["time_size"]
        trend_top, trend_bottom = layout["trend_area"]
        icon_area_top, icon_area_bottom = layout["icon_area"]
        stat_area_top, stat_area_bottom = layout["stat_area"]
        stat_area_height = max(1, stat_area_bottom - stat_area_top)

        temp_val = hour.get("temp", 0)
        temp_frac = (temp_val - min_temp) / temp_range
        temp_y = int(trend_bottom - temp_frac * (trend_bottom - trend_top))
        layout["temp_y"] = temp_y

        draw.text((cx - time_w // 2, card_top + 6), time_label, font=time_font, fill=(235, 235, 235))

    for layout in card_layouts:
        hour = layout["hour"]
        x0, x1 = layout["x0"], layout["x1"]
        cx = layout["cx"]
        trend_top, trend_bottom = layout["trend_area"]
        icon_area_top, icon_area_bottom = layout["icon_area"]
        stat_area_top, stat_area_bottom = layout["stat_area"]
        stat_area_height = max(1, stat_area_bottom - stat_area_top)
        temp_y = layout.get("temp_y", trend_bottom)

        temp_val = hour.get("temp", 0)
        temp_str = f"{temp_val}°"
        temp_w, temp_h = draw.textsize(temp_str, font=FONT_CONDITION)
        temp_text_y = max(trend_top, min(trend_bottom - temp_h, temp_y - temp_h // 2))
        draw.text(
            (cx - temp_w // 2, temp_text_y),
            temp_str,
            font=FONT_CONDITION,
            fill=_temperature_chart_color(temp_val),
        )

        icon_code = hour.get("icon")
        condition_code = hour.get("condition_code")
        is_daylight = hour.get("is_daylight")
        icon_img = None
        icon_key = (icon_code, condition_code, is_daylight, True)
        if icon_code:
            if icon_key not in icon_cache:
                icon_cache[icon_key] = fetch_weather_icon(
                    icon_code,
                    icon_size,
                    condition_code=condition_code,
                    is_daylight=is_daylight,
                    stack_emojis=True,
                )
            icon_img = icon_cache[icon_key]

        if icon_img:
            icon_y = icon_area_top + max(0, (icon_area_bottom - icon_area_top - icon_size) // 2)
            img.paste(icon_img, (cx - icon_size // 2, icon_y), icon_img)
        else:
            condition = hour.get("condition", "")
            if condition:
                display_text = condition
                cond_w, cond_h = draw.textsize(display_text, font=FONT_WEATHER_DETAILS)
                while cond_w > col_w - 10 and len(display_text) > 3:
                    display_text = display_text[:-1]
                    cond_w, cond_h = draw.textsize(display_text + "…", font=FONT_WEATHER_DETAILS)
                if display_text != condition:
                    display_text = display_text + "…"
                    cond_w, cond_h = draw.textsize(display_text, font=FONT_WEATHER_DETAILS)
                cond_y = icon_area_top + max(0, (icon_area_bottom - icon_area_top - cond_h) // 2)
                draw.text((cx - cond_w // 2, cond_y), display_text, font=FONT_WEATHER_DETAILS, fill=(170, 180, 240))

        draw.line((x0 + 6, stat_area_top, x1 - 6, stat_area_top), fill=(50, 50, 80), width=1)

        stat_items = []
        wind_overlay = None

        wind_speed = hour.get("wind_speed")
        wind_dir = hour.get("wind_dir", "") or ""
        if wind_speed is not None:
            wind_parts = [
                (f"{wind_speed}", FONT_WEATHER_DETAILS_TINY_LARGE, (180, 225, 255)),
                (" mph", FONT_WEATHER_DETAILS_TINY_MICRO, (180, 225, 255)),
            ]
            if wind_dir:
                wind_parts.append((f" {wind_dir}", FONT_WEATHER_DETAILS_TINY_LARGE, (180, 225, 255)))
            wind_image = _render_stat_text(wind_parts)
            max_wind_width = max(1, col_w - 8)
            if wind_image.width > max_wind_width and wind_dir:
                wind_parts = [
                    (f"{wind_speed}", FONT_WEATHER_DETAILS_TINY_LARGE, (180, 225, 255)),
                    (" mph", FONT_WEATHER_DETAILS_TINY_MICRO, (180, 225, 255)),
                ]
                wind_image = _render_stat_text(wind_parts)
            if wind_image.width > max_wind_width:
                wind_parts = [
                    (f"{wind_speed}", FONT_WEATHER_DETAILS_TINY_MICRO, (180, 225, 255)),
                    ("mph", FONT_WEATHER_DETAILS_TINY_MICRO, (180, 225, 255)),
                ]
                wind_image = _render_stat_text(wind_parts)
            if wind_image.width > max_wind_width:
                wind_parts = [
                    (f"{wind_speed}", FONT_WEATHER_DETAILS_TINY_MICRO, (180, 225, 255)),
                ]
                wind_image = _render_stat_text(wind_parts)
            # Keep wind text as a top overlay pass so it remains readable.
            wind_overlay = {"image": wind_image}
            stat_items.append(wind_overlay)

        pop = hour.get("pop")
        if pop is not None:
            clamped_pop = max(0, min(pop, 100))
            is_snow = hour.get("is_snow", False)
            precip_color = (173, 216, 230) if is_snow else (135, 206, 250)
            precip_emoji = "❄️" if is_snow else "💧"
            pop_text = f"{clamped_pop}%"
            _, pop_text_h = draw.textsize(pop_text, font=FONT_WEATHER_DETAILS_TINY_LARGE)
            target_icon_size = max(8, pop_text_h)
            precip_icon = _ensure_rgba_icon(_render_emoji_glyph(precip_emoji, FONT_EMOJI, precip_color))
            if precip_icon.height != target_icon_size and precip_icon.height > 0:
                scale = target_icon_size / precip_icon.height
                resized_w = max(1, int(round(precip_icon.width * scale)))
                precip_icon = precip_icon.resize((resized_w, target_icon_size), Image.Resampling.LANCZOS)
            stat_items.append((pop_text, FONT_WEATHER_DETAILS_TINY_LARGE, precip_color, precip_icon))

        uvi_val = hour.get("uvi")
        if uvi_val is not None:
            uv_color = uv_index_color(uvi_val)
            uv_text = f"UV {uvi_val}"
            stat_items.append((uv_text, FONT_WEATHER_DETAILS_TINY_LARGE, uv_color))

        if stat_items:
            num_items = len(stat_items)
            segment_height = stat_area_height / num_items if num_items else stat_area_height

            for idx, item in enumerate(stat_items):
                # Support both (text, font, color), (text, font, color, icon), and pre-rendered image items
                icon = None
                text_image = None
                if isinstance(item, dict):
                    text_image = item.get("image")
                    text = ""
                    font = FONT_WEATHER_DETAILS_TINY_LARGE
                    color = (255, 255, 255)
                elif len(item) == 4:
                    text, font, color, icon = item
                else:
                    text, font, color = item

                if text_image is not None:
                    text_w, text_h = text_image.size
                else:
                    text_w, text_h = draw.textsize(text, font=font)

                center_y = stat_area_top + segment_height * (idx + 0.5)
                text_y = int(center_y - text_h / 2)
                bottom_clamp = stat_area_bottom - text_h - (1 if is_hyperpixel_4_square_layout() else 0)
                text_y = max(stat_area_top, min(text_y, bottom_clamp))

                if icon:
                    # Render icon + text side by side
                    icon_w, icon_h = icon.size
                    gap = 2
                    total_w = icon_w + gap + text_w
                    icon_x = cx - total_w // 2
                    text_x = icon_x + icon_w + gap
                    icon_y = text_y + (text_h - icon_h) // 2
                    img.paste(icon, (icon_x, icon_y), icon)
                    draw.text((text_x, text_y), text, font=font, fill=color)
                elif text_image is not None:
                    text_x = max(x0 + 2, min(cx - text_w // 2, x1 - 2 - text_w))
                    text_y = max(stat_area_top, min(text_y, stat_area_bottom - text_h))
                    img.paste(text_image, (text_x, text_y), text_image)
                    if wind_overlay is not None and text_image is wind_overlay.get("image"):
                        wind_overlay["pos"] = (text_x, text_y)
                else:
                    # Just render text centered
                    draw.text((cx - text_w // 2, text_y), text, font=font, fill=color)

            # Re-paste wind on top in case other stat rows are tight and overlap.
            if wind_overlay and wind_overlay.get("image") is not None and wind_overlay.get("pos"):
                wx, wy = wind_overlay["pos"]
                img.paste(wind_overlay["image"], (wx, wy), wind_overlay["image"])


    # Re-draw title with a small background strip so it always stays readable above cards/content.
    if is_hyperpixel_next_layout():
        draw.rectangle((0, 0, WIDTH, title_y + title_h + 2), fill=background)
    draw.text((title_x, title_y), title, font=FONT_WEATHER_LABEL, fill=(200, 200, 200))

    return ScreenImage(img, displayed=False)


@log_call
def draw_weather_daily(display, weather, transition: bool = False, days: int = 5):
    background = get_screen_background_color("weather daily", (0, 0, 0))
    forecast = _gather_daily_forecast(weather, days)
    if not forecast:
        img = Image.new("RGB", (WIDTH, HEIGHT), background)
        draw = ImageDraw.Draw(img)
        msg = "No daily data"
        w, h = draw.textsize(msg, font=FONT_WEATHER_DETAILS_BOLD)
        draw.text(((WIDTH - w) // 2, (HEIGHT - h) // 2), msg, font=FONT_WEATHER_DETAILS_BOLD, fill=(255, 255, 255))
        return ScreenImage(img, displayed=False)

    img = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)

    title = "Next 5 days"
    title_w, title_h = draw.textsize(title, font=FONT_WEATHER_LABEL)
    title_x = (WIDTH - title_w) // 2
    title_y = 2
    if is_hyperpixel_next_layout():
        title_y = 4
    draw.text((title_x, title_y), title, font=FONT_WEATHER_LABEL, fill=(200, 200, 200))

    days_to_show = len(forecast)
    gap = 4
    available_width = WIDTH - gap * (days_to_show + 1)
    col_w = max(1, available_width // days_to_show)
    x_start = (WIDTH - (days_to_show * col_w + gap * (days_to_show - 1))) // 2

    card_top = title_h + 6
    card_bottom = HEIGHT - 6
    card_height = card_bottom - card_top
    icon_cache: dict[tuple[Optional[str], Optional[str], bool], Optional[Image.Image]] = {}
    icon_size = max(16, min(WEATHER_ICON_SIZE, col_w - 10))
    day_font = FONT_WEATHER_DETAILS_SMALL_BOLD
    stat_font = FONT_WEATHER_DETAILS_SMALL
    pop_font = FONT_WEATHER_DETAILS_TINY_LARGE

    for idx, day in enumerate(forecast):
        x0 = x_start + idx * (col_w + gap)
        x1 = x0 + col_w
        cx = (x0 + x1) // 2

        draw.rounded_rectangle(
            (x0, card_top, x1, card_bottom),
            radius=6,
            fill=(18, 18, 28),
            outline=(40, 40, 60),
        )

        day_label = day.get("day", "")
        hi_val = day.get("hi")
        lo_val = day.get("lo")
        pop_val = day.get("pop")
        is_snow = day.get("is_snow", False)
        icon_code = day.get("icon")
        condition_code = day.get("condition_code")
        condition_label = day.get("condition") or ""

        day_w, day_h = draw.textsize(day_label, font=day_font)
        day_y = card_top + 6
        draw.text((cx - day_w // 2, day_y), day_label, font=day_font, fill=(235, 235, 235))

        icon_area_top = day_y + day_h + 4
        icon_area_bottom = card_top + int(card_height * 0.64)
        stat_area_top = icon_area_bottom + 6
        stat_area_bottom = card_bottom - 6

        icon_img = None
        icon_key = (icon_code, condition_code, True)
        if icon_code:
            if icon_key not in icon_cache:
                icon_cache[icon_key] = fetch_weather_icon(
                    icon_code,
                    icon_size,
                    condition_code=condition_code,
                    is_daylight=True,
                    stack_emojis=True,
                )
            icon_img = icon_cache[icon_key]

        if icon_img:
            icon_y = icon_area_top + max(0, (icon_area_bottom - icon_area_top - icon_size) // 2)
            img.paste(icon_img, (cx - icon_size // 2, icon_y), icon_img)
        elif condition_label:
            display_text = condition_label
            cond_w, cond_h = draw.textsize(display_text, font=FONT_WEATHER_DETAILS_TINY)
            while cond_w > col_w - 8 and len(display_text) > 3:
                display_text = display_text[:-1]
                cond_w, cond_h = draw.textsize(display_text + "…", font=FONT_WEATHER_DETAILS_TINY)
            if display_text != condition_label:
                display_text = f"{display_text}…"
            cond_y = icon_area_top + max(0, (icon_area_bottom - icon_area_top - cond_h) // 2)
            draw.text((cx - cond_w // 2, cond_y), display_text, font=FONT_WEATHER_DETAILS_TINY, fill=(190, 190, 190))

        hi_value = f"{hi_val}°" if hi_val is not None else "—"
        lo_value = f"{lo_val}°" if lo_val is not None else "—"

        pop_text = "—"
        precip_icon = None
        precip_color = (135, 206, 250)
        if pop_val is not None:
            clamped_pop = max(0, min(pop_val, 100))
            precip_color = (173, 216, 230) if is_snow else (135, 206, 250)
            pop_text = f"{clamped_pop}%"
            precip_emoji = "❄️" if is_snow else "💧"
            _, pop_text_h = draw.textsize(pop_text, font=pop_font)
            target_icon_size = max(8, pop_text_h)
            precip_icon = _ensure_rgba_icon(_render_emoji_glyph(precip_emoji, FONT_EMOJI, precip_color))
            if precip_icon.height != target_icon_size and precip_icon.height > 0:
                scale = target_icon_size / precip_icon.height
                resized_w = max(1, int(round(precip_icon.width * scale)))
                precip_icon = precip_icon.resize((resized_w, target_icon_size), Image.Resampling.LANCZOS)

        stats = [
            (
                _render_stat_text(
                    [
                        ("Hi ", stat_font, (235, 235, 235)),
                        (hi_value, stat_font, _temperature_chart_color(hi_val)),
                    ]
                ),
                None,
                None,
                None,
            ),
            (
                _render_stat_text(
                    [
                        ("Lo ", stat_font, (235, 235, 235)),
                        (lo_value, stat_font, _temperature_chart_color(lo_val)),
                    ]
                ),
                None,
                None,
                None,
            ),
            (pop_text, pop_font, precip_color, precip_icon),
        ]
        segment_h = max(1, (stat_area_bottom - stat_area_top) / max(1, len(stats)))
        for stat_idx, (text, font, color, icon) in enumerate(stats):
            if isinstance(text, Image.Image):
                text_w, text_h = text.size
            else:
                text_w, text_h = draw.textsize(text, font=font)
            center_y = stat_area_top + segment_h * (stat_idx + 0.5)
            text_y = int(center_y - text_h / 2)
            text_y = max(stat_area_top, min(text_y, stat_area_bottom - text_h))
            if icon:
                icon_w, icon_h = icon.size
                gap_icon = 2
                total_w = icon_w + gap_icon + text_w
                icon_x = cx - total_w // 2
                icon_y = text_y + (text_h - icon_h) // 2
                text_x = icon_x + icon_w + gap_icon
                img.paste(icon, (icon_x, icon_y), icon)
                draw.text((text_x, text_y), text, font=font, fill=color)
            elif isinstance(text, Image.Image):
                text_x = cx - text_w // 2
                img.paste(text, (text_x, text_y), text)
            else:
                text_x = cx - text_w // 2
                draw.text((text_x, text_y), text, font=font, fill=color)

    if is_hyperpixel_next_layout():
        draw.rectangle((0, 0, WIDTH, title_y + title_h + 2), fill=background)
    draw.text((title_x, title_y), title, font=FONT_WEATHER_LABEL, fill=(200, 200, 200))

    return ScreenImage(img, displayed=False)


# ─── Screen 2: Detailed (with UV index) ───────────────────────────────────────
def draw_weather_screen_2(display, weather, transition=False):
    if not weather:
        return None

    background = get_screen_background_color("weather2", (0, 0, 0))
    severity, led_color = _detect_weather_alert(weather)

    current = weather.get("current", {})
    daily   = weather.get("daily", [{}])[0]

    now = datetime.datetime.now(CENTRAL_TIME)
    next_label, next_time = _next_sun_event(weather.get("daily"), now=now)
    if next_label and next_time:
        items = [(f"{next_label}:", next_time.strftime("%-I:%M %p"))]
    else:
        items = []

    # Other details
    wind_speed = round(current.get('wind_speed', 0))
    wind_dir = wind_direction(current.get('wind_deg'))
    wind_value = f"{wind_speed} mph"
    if wind_dir:
        wind_value = f"{wind_value} {wind_dir}"

    pressure_raw = current.get("pressure")
    pressure_inhg = None
    if pressure_raw is not None:
        try:
            pressure_inhg = float(pressure_raw) * 0.0338639
        except (TypeError, ValueError):
            pressure_inhg = None
    pressure_text = f"{pressure_inhg:.2f} inHg" if pressure_inhg is not None else "—"
    pressure_trend = current.get("pressure_trend")
    pressure_value = pressure_text
    if pressure_trend in PRESSURE_TREND_SYMBOLS:
        symbol, symbol_color = PRESSURE_TREND_SYMBOLS[pressure_trend]
        pressure_value = _render_stat_text(
            [
                (pressure_text, FONT_WEATHER_DETAILS, (255, 255, 255)),
                (" ", FONT_WEATHER_DETAILS, (255, 255, 255)),
                (symbol, FONT_WEATHER_DETAILS, symbol_color),
            ]
        )

    items += [
        ("Wind:",     wind_value),
        ("Gust:",     f"{round(current.get('wind_gust',0))} mph"),
        ("Humidity:", f"{current.get('humidity',0)}%"),
        ("Pressure:", pressure_value),
    ]

    uvi = round(current.get("uvi", 0))
    uv_col = uv_index_color(uvi)
    items.append(("UV Index:", str(uvi), uv_col))

    img  = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)

    # compute per-row heights
    row_metrics = []
    total_h = 0
    for it in items:
        lbl, val = it[0], it[1]
        lbl_bbox = _safe_textbbox(draw, lbl, FONT_WEATHER_DETAILS_BOLD)
        lbl_w = lbl_bbox[2] - lbl_bbox[0]
        lbl_h = lbl_bbox[3] - lbl_bbox[1]
        if isinstance(val, Image.Image):
            val_w, val_h = val.size
            val_bbox = None
        else:
            val_bbox = _safe_textbbox(draw, val, FONT_WEATHER_DETAILS)
            val_w = val_bbox[2] - val_bbox[0]
            val_h = val_bbox[3] - val_bbox[1]
        row_h = max(lbl_h, val_h)
        row_metrics.append(
            (
                lbl,
                val,
                row_h,
                lbl_h,
                val_h,
                lbl_w,
                val_w,
                lbl_bbox,
                val_bbox,
                it[2] if len(it) == 3 else (255, 255, 255),
            )
        )
        total_h += row_h

    # vertical spacing
    space = (HEIGHT - total_h) // (len(items) + 1)
    y = space

    # render each row, vertically centering label & value
    for lbl, val, row_h, h_lbl, h_val, lw, v_w, lbl_bbox, val_bbox, color in row_metrics:
        row_w = lw + 4 + v_w
        x0    = (WIDTH - row_w)//2

        y_lbl = y + (row_h - h_lbl)//2
        y_val = y + (row_h - h_val)//2

        lbl_y_draw = y_lbl - (lbl_bbox[1] if lbl_bbox else 0)

        draw.text((x0, lbl_y_draw), lbl, font=FONT_WEATHER_DETAILS_BOLD, fill=(255,255,255))
        if isinstance(val, Image.Image):
            img.paste(val, (x0 + lw + 4, y_val), val)
        else:
            val_y_draw = y_val - (val_bbox[1] if val_bbox else 0)
            draw.text((x0 + lw + 4, val_y_draw), val, font=FONT_WEATHER_DETAILS, fill=color)
        y += row_h + space

    _draw_alert_indicator(img, draw, severity)

    return ScreenImage(img, displayed=False, led_override=led_color)


def _latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int, float, float]:
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x_float = (lon + 180.0) / 360.0 * n
    y_float = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    x_tile = int(x_float)
    y_tile = int(y_float)
    return x_tile, y_tile, x_float - x_tile, y_float - y_tile


class RadarFrame(NamedTuple):
    image: Image.Image
    timestamp: Optional[int]


def _normalise_radar_timestamp(value: object) -> Optional[int]:
    try:
        ts_int = int(value)  # type: ignore[arg-type]
    except Exception:
        return None
    # RainViewer typically returns seconds, but guard against millisecond inputs.
    if ts_int > 1_000_000_000_000:
        ts_int = ts_int // 1000
    return ts_int


def _format_radar_timestamp(timestamp: Optional[int]) -> str:
    dt = timestamp_to_datetime(timestamp, CENTRAL_TIME)
    if dt is None:
        return ""
    return dt.strftime("%-I:%M %p")


def _fetch_radar_frames(zoom: int = 7, max_frames: int = 6) -> list[RadarFrame]:
    frames = _fetch_rainviewer_frames(zoom=zoom, max_frames=max_frames)
    if not frames:
        frames = _fetch_iem_radar_fallback_frames(zoom=zoom)
    if not frames:
        return []

    now_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    fresh_frames = [
        frame
        for frame in frames
        if frame.timestamp is not None and now_ts - frame.timestamp <= int(RADAR_MAX_FRAME_AGE.total_seconds())
    ]
    if fresh_frames:
        return fresh_frames
    return frames


def _fetch_rainviewer_frames(zoom: int = 7, max_frames: int = 6) -> list[RadarFrame]:
    metadata = None
    for metadata_url in RAINVIEWER_METADATA_URLS:
        try:
            meta_resp = requests.get(metadata_url, timeout=6)
            meta_resp.raise_for_status()
            metadata = meta_resp.json()
            break
        except Exception as exc:
            logging.warning("Radar metadata fetch failed from %s: %s", metadata_url, exc)

    if not isinstance(metadata, dict):
        return []

    host = metadata.get("host", "https://tilecache.rainviewer.com")
    radar_info = metadata.get("radar") or {}
    frames = (radar_info.get("past") or []) + (radar_info.get("nowcast") or [])
    frames = sorted(
        frames,
        key=lambda frame: _normalise_radar_timestamp(
            frame.get("time") if isinstance(frame, dict) else None
        )
        or 0,
    )
    frames = frames[-max_frames:]

    x_tile, y_tile, x_offset, y_offset = _latlon_to_tile(
        RADAR_CENTER_LATITUDE,
        RADAR_CENTER_LONGITUDE,
        zoom,
    )
    images: list[RadarFrame] = []

    for frame in frames:
        path = frame.get("path") if isinstance(frame, dict) else None
        timestamp = _normalise_radar_timestamp(frame.get("time") if isinstance(frame, dict) else None)
        if not path:
            continue
        url = (
            f"{host.rstrip('/')}/{path.strip('/')}/256/{zoom}/{x_tile}/{y_tile}/2/1_1.png"
        )
        try:
            tile_resp = requests.get(url, timeout=6)
            tile_resp.raise_for_status()
            tile = Image.open(BytesIO(tile_resp.content)).convert("RGBA")
        except Exception as exc:  # pragma: no cover - network failures are non-fatal
            logging.warning("Radar tile fetch failed: %s", exc)
            continue

        frame_img = Image.new("RGBA", tile.size, (0, 0, 0, 255))
        frame_img.alpha_composite(tile)
        final_frame = frame_img.resize((WIDTH, HEIGHT), Image.LANCZOS).convert("RGBA")
        images.append(RadarFrame(final_frame, timestamp))

    return images


def _fetch_iem_radar_fallback_frames(zoom: int = 7) -> list[RadarFrame]:
    """Fetch a free, no-key radar tile from Iowa State Mesonet as a fallback."""
    x_tile, y_tile, _, _ = _latlon_to_tile(
        RADAR_CENTER_LATITUDE,
        RADAR_CENTER_LONGITUDE,
        zoom,
    )
    url = f"https://mesonet.agron.iastate.edu/cache/tile.py/1.0.0/q2-hsr-900913/{zoom}/{x_tile}/{y_tile}.png"
    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "desk-display/weather-radar"})
        resp.raise_for_status()
        tile = Image.open(BytesIO(resp.content)).convert("RGBA")
    except Exception as exc:  # pragma: no cover - network failures are non-fatal
        logging.warning("IEM radar fallback fetch failed: %s", exc)
        return []

    final_frame = tile.resize((WIDTH, HEIGHT), Image.LANCZOS).convert("RGBA")
    return [RadarFrame(final_frame, int(datetime.datetime.now(datetime.timezone.utc).timestamp()))]


def _fetch_base_map(zoom: int = 7) -> Optional[Image.Image]:
    x_tile, y_tile, _, _ = _latlon_to_tile(
        RADAR_CENTER_LATITUDE,
        RADAR_CENTER_LONGITUDE,
        zoom,
    )
    headers = {
        "User-Agent": "desk-display/weather-radar",
    }
    urls = [
        f"https://tile.openstreetmap.org/{zoom}/{x_tile}/{y_tile}.png",
        f"https://basemaps.cartocdn.com/light_all/{zoom}/{x_tile}/{y_tile}.png",
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=6, headers=headers)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as exc:  # pragma: no cover - network failures are non-fatal
            logging.warning("Radar base map fetch failed from %s: %s", url, exc)

    return None


@log_call
def draw_weather_radar(display, weather=None, transition: bool = False):
    background = get_screen_background_color("weather radar", (0, 0, 0))
    zoom_level = 7
    frames = _fetch_radar_frames(zoom=zoom_level)
    base_map = _fetch_base_map(zoom=zoom_level)
    if not frames:
        img = Image.new("RGB", (WIDTH, HEIGHT), background)
        draw = ImageDraw.Draw(img)
        msg = "Radar unavailable"
        w, h = draw.textsize(msg, font=FONT_WEATHER_DETAILS_BOLD)
        draw.text(((WIDTH - w) // 2, (HEIGHT - h) // 2), msg, font=FONT_WEATHER_DETAILS_BOLD, fill=(255, 255, 255))
        return ScreenImage(img, displayed=False)

    map_section = None
    if base_map:
        map_section = base_map.resize((WIDTH, HEIGHT), Image.LANCZOS).convert("RGBA")
    else:
        map_section = Image.new("RGBA", (WIDTH, HEIGHT), background + (255,))

    def _compose_frame(frame: RadarFrame) -> Image.Image:
        radar_resized = frame.image.resize((WIDTH, HEIGHT), Image.LANCZOS).convert("RGBA")
        radar_opacity = 0.6
        if radar_opacity < 1.0:
            alpha = radar_resized.getchannel("A")
            alpha = alpha.point(lambda p: int(p * radar_opacity))
            radar_resized.putalpha(alpha)
        combined = map_section.copy()
        combined.alpha_composite(radar_resized)
        result = combined.convert("RGB")

        label = _format_radar_timestamp(frame.timestamp)
        if label:
            draw = ImageDraw.Draw(result)
            bbox = draw.textbbox((0, 0), label, font=FONT_WEATHER_DETAILS_TINY, stroke_width=1)
            text_w = bbox[2] - bbox[0]
            x = WIDTH - text_w - 6
            y = 6
            draw.text(
                (x, y),
                label,
                font=FONT_WEATHER_DETAILS_TINY,
                fill=(255, 255, 255),
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )

        return result

    composed_frames = [_compose_frame(frame) for frame in frames]

    def _display_frame(frame_image: Image.Image) -> None:
        if hasattr(display, "image"):
            display.image(frame_image)
            return
        if hasattr(display, "display"):
            display.display(frame_image)

    if transition and len(composed_frames) > 1:
        for _ in range(RADAR_ANIMATION_LOOPS):
            for frame_image in composed_frames:
                _display_frame(frame_image)
                time.sleep(RADAR_ANIMATION_FRAME_DELAY_SECONDS)
        return ScreenImage(composed_frames[-1], displayed=True)

    last_frame = composed_frames[-1]
    return ScreenImage(last_frame, displayed=False)


def _next_sun_event(daily_entries, now: datetime.datetime | None = None) -> tuple[str | None, datetime.datetime | None]:
    """Return the next sunrise/sunset event, allowing a post-event grace window."""

    if now is None:
        now = datetime.datetime.now(CENTRAL_TIME)

    events: list[tuple[str, datetime.datetime]] = []
    for day in list(daily_entries or [])[:2]:
        if not isinstance(day, dict):
            continue

        sunrise = timestamp_to_datetime(day.get("sunrise"), CENTRAL_TIME)
        if sunrise:
            events.append(("Sunrise", sunrise))

        sunset = timestamp_to_datetime(day.get("sunset"), CENTRAL_TIME)
        if sunset:
            events.append(("Sunset", sunset))

    events.sort(key=lambda entry: entry[1])

    for label, event_time in events:
        if now <= event_time + SUN_EVENT_GRACE:
            return label, event_time

    if events:
        return events[-1]

    return None, None
