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
from PIL import Image, ImageDraw

from config import (
    WIDTH,
    HEIGHT,
    CENTRAL_TIME,
    FONT_TEMP,
    FONT_CONDITION,
    FONT_WEATHER_LABEL,
    FONT_WEATHER_DETAILS,
    FONT_WEATHER_DETAILS_BOLD,
    FONT_EMOJI,
    WEATHER_ICON_SIZE,
    WEATHER_DESC_GAP,
)
from utils import (
    clear_display,
    fetch_weather_icon,
    log_call,
    timestamp_to_datetime,
    uv_index_color,
    wind_direction,
)

# ─── Screen 1: Basic weather + two-line Feels/Hi/Lo ────────────────────────────
@log_call
def draw_weather_screen_1(display, weather, transition=False):
    if not weather:
        return None

    current = weather.get("current", {})
    daily   = weather.get("daily", [{}])[0]
    hourly  = weather.get("hourly") if isinstance(weather.get("hourly"), list) else None

    temp  = round(current.get("temp", 0))
    desc  = current.get("weather", [{}])[0].get("description", "").title()

    feels = round(current.get("feels_like", 0))
    hi    = round(daily.get("temp", {}).get("max", 0))
    lo    = round(daily.get("temp", {}).get("min", 0))

    clear_display(display)
    img  = Image.new("RGB", (WIDTH, HEIGHT), "black")
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

    icon_code = current.get("weather", [{}])[0].get("icon")
    icon_img = fetch_weather_icon(icon_code, WEATHER_ICON_SIZE)

    cloud_cover = current.get("clouds")
    try:
        cloud_cover = int(round(float(cloud_cover)))
    except Exception:
        cloud_cover = None

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
    weather_id = daily_weather.get("id")
    weather_main = (daily_weather.get("main") or "").strip().lower()
    is_snow = False
    if weather_main == "snow":
        is_snow = True
    elif isinstance(weather_id, int) and 600 <= weather_id < 700:
        is_snow = True
    elif daily.get("snow") or current.get("snow"):
        is_snow = True

    precip_emoji = "❄" if is_snow else "💧"
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
        feels_col = (128,0,128)
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
    y_icon = top_of_icons + ((y_lbl - top_of_icons - WEATHER_ICON_SIZE)//2)
    icon_x = (WIDTH - WEATHER_ICON_SIZE) // 2
    icon_center_y = top_of_icons + max(0, (y_lbl - top_of_icons) // 2)

    if icon_img:
        img.paste(icon_img, (icon_x, y_icon), icon_img)

    side_font = FONT_WEATHER_DETAILS
    stack_gap = 2
    if precip_percent:
        emoji_color = (173, 216, 230) if precip_emoji == "❄" else (135, 206, 250)
        emoji_w, emoji_h = draw.textsize(precip_emoji, font=FONT_EMOJI)
        pct_w, pct_h = draw.textsize(precip_percent, font=side_font)
        block_w = max(emoji_w, pct_w)
        block_h = emoji_h + stack_gap + pct_h
        precip_x = icon_x - 6 - block_w
        if precip_x < 0:
            precip_x = 0
        block_y = icon_center_y - block_h // 2
        emoji_x = precip_x + (block_w - emoji_w) // 2
        pct_x = precip_x + (block_w - pct_w) // 2
        draw.text((emoji_x, block_y), precip_emoji, font=FONT_EMOJI, fill=emoji_color)
        draw.text((pct_x, block_y + emoji_h + stack_gap), precip_percent, font=side_font, fill=emoji_color)

    if cloud_percent:
        cloud_emoji = "☁"
        emoji_w, emoji_h = draw.textsize(cloud_emoji, font=FONT_EMOJI)
        pct_w, pct_h = draw.textsize(cloud_percent, font=side_font)
        block_w = max(emoji_w, pct_w)
        block_h = emoji_h + stack_gap + pct_h
        cloud_x = icon_x + WEATHER_ICON_SIZE + 6
        if cloud_x + block_w > WIDTH:
            cloud_x = WIDTH - block_w
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

    if transition:
        return img

    display.image(img)
    display.show()
    return None


# ─── Screen 2: Detailed (with UV index) ───────────────────────────────────────
def draw_weather_screen_2(display, weather, transition=False):
    if not weather:
        return None

    current = weather.get("current", {})
    daily   = weather.get("daily", [{}])[0]

    now = datetime.datetime.now(CENTRAL_TIME)
    s_r = timestamp_to_datetime(daily.get("sunrise"), CENTRAL_TIME)
    s_s = timestamp_to_datetime(daily.get("sunset"), CENTRAL_TIME)

    # Sunrise or Sunset first
    if s_r and now < s_r:
        items = [("Sunrise:", s_r.strftime("%-I:%M %p"))]
    elif s_s:
        items = [("Sunset:",  s_s.strftime("%-I:%M %p"))]
    else:
        items = []

    # Other details
    wind_speed = round(current.get('wind_speed', 0))
    wind_dir = wind_direction(current.get('wind_deg'))
    wind_value = f"{wind_speed} mph"
    if wind_dir:
        wind_value = f"{wind_value} {wind_dir}"

    items += [
        ("Wind:",     wind_value),
        ("Gust:",     f"{round(current.get('wind_gust',0))} mph"),
        ("Humidity:", f"{current.get('humidity',0)}%"),
        ("Pressure:", f"{round(current.get('pressure',0)*0.0338639,2)} inHg"),
    ]

    uvi = round(current.get("uvi", 0))
    uv_col = uv_index_color(uvi)
    items.append(("UV Index:", str(uvi), uv_col))

    clear_display(display)
    img  = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    # compute per-row heights
    row_metrics = []
    total_h = 0
    for it in items:
        lbl, val = it[0], it[1]
        h1 = draw.textsize(lbl, font=FONT_WEATHER_DETAILS_BOLD)[1]
        h2 = draw.textsize(val, font=FONT_WEATHER_DETAILS)[1]
        row_h = max(h1, h2)
        row_metrics.append((lbl, val, row_h, h1, h2, it[2] if len(it)==3 else (255,255,255)))
        total_h += row_h

    # vertical spacing
    space = (HEIGHT - total_h) // (len(items) + 1)
    y = space

    # render each row, vertically centering label & value
    for lbl, val, row_h, h_lbl, h_val, color in row_metrics:
        lw, _ = draw.textsize(lbl, font=FONT_WEATHER_DETAILS_BOLD)
        vw, _ = draw.textsize(val, font=FONT_WEATHER_DETAILS)
        row_w = lw + 4 + vw
        x0    = (WIDTH - row_w)//2

        y_lbl = y + (row_h - h_lbl)//2
        y_val = y + (row_h - h_val)//2

        draw.text((x0,          y_lbl), lbl, font=FONT_WEATHER_DETAILS_BOLD, fill=(255,255,255))
        draw.text((x0 + lw + 4, y_val), val, font=FONT_WEATHER_DETAILS,      fill=color)
        y += row_h + space

    if transition:
        return img

    display.image(img)
    display.show()
    return None
