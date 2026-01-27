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
from PIL import Image, ImageDraw

from config import (
    GOOGLE_MAPS_API_KEY,
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
    WEATHER_ICON_SIZE,
    WEATHER_DESC_GAP,
    HOURLY_FORECAST_HOURS,
    LATITUDE,
    LONGITUDE,
    get_screen_background_color,
)
from utils import (
    LED_INDICATOR_LEVEL,
    ScreenImage,
    clear_display,
    fetch_weather_icon,
    log_call,
    temporary_display_led,
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


def _render_stat_text(parts):
    """Render a left-to-right text image from ``(text, font, color)`` parts."""

    scratch = Image.new("RGB", (1, 1))
    scratch_draw = ImageDraw.Draw(scratch)

    widths = []
    heights = []
    offsets = []
    extents = []
    for text, font, _ in parts:
        bbox = scratch_draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        widths.append(w)
        heights.append(h)
        offset_y = -bbox[1]
        offsets.append(offset_y)
        extents.append(offset_y + h)

    # Add a small cushion to avoid clipping wide glyphs (e.g., arrows) and give
    # slightly more vertical room for taller fonts such as the wind speed value.
    padding_x = 1
    padding_y = 2
    content_h = max(extents) if extents else 0
    total_w = sum(widths) + padding_x * 2
    total_h = content_h + padding_y * 2
    result = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(result)

    x = padding_x
    for (text, font, color), w, h, offset_y, extent in zip(parts, widths, heights, offsets, extents):
        y = padding_y + offset_y + (content_h - extent) // 2
        draw.text((x, y), text, font=font, fill=color)
        x += w

    return result


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


def _draw_alert_indicator(draw: ImageDraw.ImageDraw, severity: Optional[str]) -> None:
    if not severity:
        return
    icon_color = ALERT_ICON_COLORS.get(severity, (255, 215, 0))
    w_icon, h_icon = draw.textsize(ALERT_SYMBOL, font=FONT_EMOJI_SMALL)
    x_icon = WIDTH - w_icon - 2
    y_icon = HEIGHT - h_icon - 2
    draw.text((x_icon, y_icon), ALERT_SYMBOL, font=FONT_EMOJI_SMALL, fill=icon_color)

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

    clear_display(display)
    img  = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)

    # Temperature
    temp_str = f"{temp}°F"
    w_temp, h_temp = draw.textsize(temp_str, font=FONT_TEMP)
    draw.text(((WIDTH - w_temp)//2, 0), temp_str, font=FONT_TEMP, fill=(255,255,255))

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

    cloud_percent = None
    if cloud_cover is not None:
        cloud_percent = f"{max(0, min(cloud_cover, 100))}%"

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
    y_val     = HEIGHT - max_val_h - 9
    LABEL_GAP = 2
    y_lbl     = y_val - max_lbl_h - LABEL_GAP

    # paste icon between desc and labels
    top_of_icons = h_temp + h_desc + WEATHER_DESC_GAP * 2
    icon_code = current.get("weather", [{}])[0].get("icon")
    # Fit the weather icon into the available gap between the description and
    # the Feels/Hi/Lo labels so it doesn't overlap other content.
    available_icon_height = y_lbl - top_of_icons
    if available_icon_height > 0:
        weather_icon_size = max(1, min(WEATHER_ICON_SIZE, available_icon_height))
    else:
        weather_icon_size = min(WEATHER_ICON_SIZE, HEIGHT // 2)
    icon_img = fetch_weather_icon(icon_code, weather_icon_size)
    y_icon = top_of_icons + ((y_lbl - top_of_icons - weather_icon_size)//2)
    icon_x = (WIDTH - weather_icon_size) // 2
    icon_center_y = top_of_icons + max(0, (y_lbl - top_of_icons) // 2)

    if icon_img:
        img.paste(icon_img, (icon_x, y_icon), icon_img)

    side_font = FONT_WEATHER_DETAILS
    stack_gap = 2
    edge_margin = 4
    if precip_percent:
        precip_color = (173, 216, 230) if is_snow else (135, 206, 250)
        precip_icon_size = FONT_EMOJI.size if hasattr(FONT_EMOJI, "size") else 26
        precip_icon = _render_precip_icon(is_snow, precip_icon_size, precip_color)
        emoji_w, emoji_h = precip_icon.size
        pct_w, pct_h = draw.textsize(precip_percent, font=side_font)
        block_w = max(emoji_w, pct_w)
        block_h = emoji_h + stack_gap + pct_h
        left_available = max(0, icon_x - edge_margin)
        precip_x = edge_margin + max(0, (left_available - block_w) // 2)
        precip_x = min(precip_x, max(edge_margin, icon_x - block_w))
        block_y = icon_center_y - block_h // 2
        emoji_x = precip_x + (block_w - emoji_w) // 2
        pct_x = precip_x + (block_w - pct_w) // 2
        img.paste(precip_icon, (emoji_x, block_y), precip_icon)
        draw.text((pct_x, block_y + emoji_h + stack_gap), precip_percent, font=side_font, fill=precip_color)

    if cloud_percent:
        cloud_emoji = "☁"
        emoji_w, emoji_h = draw.textsize(cloud_emoji, font=FONT_EMOJI)
        pct_w, pct_h = draw.textsize(cloud_percent, font=side_font)
        block_w = max(emoji_w, pct_w)
        block_h = emoji_h + stack_gap + pct_h
        right_start = icon_x + weather_icon_size
        right_available = max(0, WIDTH - edge_margin - right_start)
        cloud_x = right_start + max(0, (right_available - block_w) // 2)
        cloud_x = min(cloud_x, max(edge_margin, WIDTH - edge_margin - block_w))
        block_y = icon_center_y - block_h // 2
        emoji_x = cloud_x + (block_w - emoji_w) // 2
        pct_x = cloud_x + (block_w - pct_w) // 2
        draw.text((emoji_x, block_y), cloud_emoji, font=FONT_EMOJI, fill=(211, 211, 211))
        draw.text((pct_x, block_y + emoji_h + stack_gap), cloud_percent, font=side_font, fill=(211, 211, 211))

    # draw groups
    x = x0
    for idx, (lbl, lw, lh, val, vw, vh, gw) in enumerate(groups):
        cx = x + gw//2
        draw.text((cx - lw//2, y_lbl), lbl, font=FONT_WEATHER_LABEL,      fill=(255,255,255))
        draw.text((cx - vw//2, y_val), val, font=FONT_WEATHER_DETAILS,     fill=val_colors[idx])
        x += gw + SPACING_X

    _draw_alert_indicator(draw, severity)

    if transition:
        return ScreenImage(img, displayed=False, led_override=led_color)

    def _render_screen() -> None:
        display.image(img)
        display.show()

    if led_color is not None:
        with temporary_display_led(*led_color):
            _render_screen()
    else:
        _render_screen()
    return None


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
            "pop": _pop_pct_from(hour),
            "wind_speed": wind_speed,
            "wind_dir": wind_dir,
            "uvi": uvi_val,
            "is_snow": is_snow,
            "feels_like": feels_like_val,
        }
        if weather_list:
            entry["icon"] = weather_list[0].get("icon")
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

    clear_display(display)
    img = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)

    hours_to_show = len(forecast)
    title = "Next 10 hours..."
    title_w, title_h = draw.textsize(title, font=FONT_WEATHER_LABEL)
    draw.text(((WIDTH - title_w) // 2, 2), title, font=FONT_WEATHER_LABEL, fill=(200, 200, 200))

    gap = 4
    available_width = WIDTH - gap * (hours_to_show + 1)
    col_w = max(1, available_width // hours_to_show)
    icon_cache: dict[str, Optional[Image.Image]] = {}
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
        stat_area_bottom = card_bottom - 8

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
        draw.text((cx - temp_w // 2, temp_text_y), temp_str, font=FONT_CONDITION, fill=(255, 255, 255))

        icon_code = hour.get("icon")
        icon_img = None
        if icon_code:
            if icon_code not in icon_cache:
                icon_cache[icon_code] = fetch_weather_icon(icon_code, icon_size)
            icon_img = icon_cache[icon_code]

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
            stat_items.append({"image": wind_image})

        pop = hour.get("pop")
        if pop is not None:
            clamped_pop = max(0, min(pop, 100))
            is_snow = hour.get("is_snow", False)
            precip_color = (173, 216, 230) if is_snow else (135, 206, 250)
            pop_text = f"{clamped_pop}%"
            # Render small precipitation icon
            precip_icon_size = 10
            precip_icon = _render_precip_icon(is_snow, precip_icon_size, precip_color)
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
                text_y = max(stat_area_top, min(text_y, stat_area_bottom - text_h))

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
                    img.paste(text_image, (cx - text_w // 2, text_y), text_image)
                else:
                    # Just render text centered
                    draw.text((cx - text_w // 2, text_y), text, font=font, fill=color)


    if transition:
        return ScreenImage(img, displayed=False)

    display.image(img)
    display.show()
    return None


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

    clear_display(display)
    img = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)

    title = "Next 5 days"
    title_w, title_h = draw.textsize(title, font=FONT_WEATHER_LABEL)
    draw.text(((WIDTH - title_w) // 2, 2), title, font=FONT_WEATHER_LABEL, fill=(200, 200, 200))

    days_to_show = len(forecast)
    gap = 4
    available_width = WIDTH - gap * (days_to_show + 1)
    col_w = max(1, available_width // days_to_show)
    x_start = (WIDTH - (days_to_show * col_w + gap * (days_to_show - 1))) // 2

    card_top = title_h + 6
    card_bottom = HEIGHT - 6

    line_gap = 2

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

        items = []
        items.append(
            {
                "text": day_label,
                "font": FONT_WEATHER_DETAILS_SMALL_BOLD,
                "color": (235, 235, 235),
            }
        )

        hi_text = f"Hi {hi_val}°" if hi_val is not None else "Hi —"
        lo_text = f"Lo {lo_val}°" if lo_val is not None else "Lo —"
        items.append(
            {
                "text": hi_text,
                "font": FONT_WEATHER_DETAILS_SMALL,
                "color": (255, 120, 120),
            }
        )
        items.append(
            {
                "text": lo_text,
                "font": FONT_WEATHER_DETAILS_SMALL,
                "color": (120, 170, 255),
            }
        )

        pop_text = "PoP —"
        precip_icon = None
        precip_color = (135, 206, 250)
        if pop_val is not None:
            clamped_pop = max(0, min(pop_val, 100))
            precip_color = (173, 216, 230) if is_snow else (135, 206, 250)
            pop_text = f"PoP {clamped_pop}%"
            precip_icon = _render_precip_icon(is_snow, 10, precip_color)

        items.append(
            {
                "text": pop_text,
                "font": FONT_WEATHER_DETAILS_TINY_LARGE,
                "color": precip_color,
                "icon": precip_icon,
            }
        )

        item_heights = []
        for item in items:
            text_w, text_h = draw.textsize(item["text"], font=item["font"])
            icon = item.get("icon")
            icon_h = icon.size[1] if icon else 0
            item_heights.append(max(text_h, icon_h))

        total_h = sum(item_heights) + line_gap * (len(items) - 1)
        content_top = card_top + 6
        content_bottom = card_bottom - 6
        start_y = content_top + max(0, (content_bottom - content_top - total_h) // 2)

        y = start_y
        for item, item_h in zip(items, item_heights):
            text = item["text"]
            font = item["font"]
            color = item["color"]
            text_w, text_h = draw.textsize(text, font=font)
            icon = item.get("icon")
            if icon:
                icon_w, icon_h = icon.size
                gap_icon = 2
                total_w = icon_w + gap_icon + text_w
                icon_x = cx - total_w // 2
                icon_y = y + (item_h - icon_h) // 2
                text_x = icon_x + icon_w + gap_icon
                text_y = y + (item_h - text_h) // 2
                img.paste(icon, (icon_x, icon_y), icon)
                draw.text((text_x, text_y), text, font=font, fill=color)
            else:
                text_x = cx - text_w // 2
                text_y = y + (item_h - text_h) // 2
                draw.text((text_x, text_y), text, font=font, fill=color)
            y += item_h + line_gap

    if transition:
        return ScreenImage(img, displayed=False)

    display.image(img)
    display.show()
    return None


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

    clear_display(display)
    img  = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)

    # compute per-row heights
    row_metrics = []
    total_h = 0
    for it in items:
        lbl, val = it[0], it[1]
        h1 = draw.textsize(lbl, font=FONT_WEATHER_DETAILS_BOLD)[1]
        if isinstance(val, Image.Image):
            val_w, val_h = val.size
        else:
            val_w, val_h = draw.textsize(val, font=FONT_WEATHER_DETAILS)
        row_h = max(h1, val_h)
        row_metrics.append((lbl, val, row_h, h1, val_h, val_w, it[2] if len(it)==3 else (255,255,255)))
        total_h += row_h

    # vertical spacing
    space = (HEIGHT - total_h) // (len(items) + 1)
    y = space

    # render each row, vertically centering label & value
    for lbl, val, row_h, h_lbl, h_val, v_w, color in row_metrics:
        lw, _ = draw.textsize(lbl, font=FONT_WEATHER_DETAILS_BOLD)
        row_w = lw + 4 + v_w
        x0    = (WIDTH - row_w)//2

        y_lbl = y + (row_h - h_lbl)//2
        y_val = y + (row_h - h_val)//2

        draw.text((x0,          y_lbl), lbl, font=FONT_WEATHER_DETAILS_BOLD, fill=(255,255,255))
        if isinstance(val, Image.Image):
            img.paste(val, (x0 + lw + 4, y_val), val)
        else:
            draw.text((x0 + lw + 4, y_val), val, font=FONT_WEATHER_DETAILS,      fill=color)
        y += row_h + space

    _draw_alert_indicator(draw, severity)

    if transition:
        return ScreenImage(img, displayed=False, led_override=led_color)

    def _render_screen() -> None:
        display.image(img)
        display.show()

    if led_color is not None:
        with temporary_display_led(*led_color):
            _render_screen()
    else:
        _render_screen()
    return None


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
    return frames


def _fetch_rainviewer_frames(zoom: int = 7, max_frames: int = 6) -> list[RadarFrame]:
    try:
        meta_resp = requests.get(
            "https://api.rainviewer.com/public/weather-maps.json", timeout=6
        )
        meta_resp.raise_for_status()
        metadata = meta_resp.json()
    except Exception as exc:
        logging.warning("Radar metadata fetch failed: %s", exc)
        return []

    host = metadata.get("host", "https://tilecache.rainviewer.com")
    radar_info = metadata.get("radar") or {}
    frames = (radar_info.get("past") or []) + (radar_info.get("nowcast") or [])
    frames = frames[-max_frames:]

    x_tile, y_tile, x_offset, y_offset = _latlon_to_tile(LATITUDE, LONGITUDE, zoom)
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


def _fetch_base_map(zoom: int = 7) -> Optional[Image.Image]:
    if not GOOGLE_MAPS_API_KEY:
        logging.warning("Radar base map: GOOGLE_MAPS_API_KEY not set; skipping base map fetch")
        return None

    lat = LATITUDE
    lon = LONGITUDE
    url = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={zoom}&size={WIDTH}x{HEIGHT}&maptype=roadmap"
        f"&key={GOOGLE_MAPS_API_KEY}"
    )
    headers = {
        "User-Agent": "desk-display/weather-radar",
    }

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

    clear_display(display)
    loops = 2
    delay = 0.5
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

    for _ in range(loops):
        for frame in composed_frames:
            display.image(frame)
            display.show()
            time.sleep(delay)

    last_frame = composed_frames[-1]
    if transition:
        return ScreenImage(last_frame, displayed=True)

    display.image(last_frame)
    display.show()
    return None


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
