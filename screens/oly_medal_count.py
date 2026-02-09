#!/usr/bin/env python3
"""Olympic medal table screen (compact/small table) with vertical scroll."""

from __future__ import annotations

import os
import time

from PIL import Image, ImageDraw

from config import (
    FONT_STATUS,
    FONT_TEAM_SPORTS,
    FONT_TITLE_SPORTS,
    HEIGHT,
    IMAGES_DIR,
    SCOREBOARD_BACKGROUND_COLOR,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_STEP,
    WIDTH,
    get_screen_background_color,
    get_screen_font,
    get_screen_image_scale,
    is_hyperpixel_next_layout,
    scale_value,
    scale_value_width,
)
from screens.data_sources.olympic_medals import fetch_olympic_medal_table
from utils import ScreenImage, clear_display, load_team_logo, log_call, standard_scoreboard_league_logo_height

HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()


def _scale_y(value: int) -> int:
    return scale_value(value) if HYPERPIXEL_LAYOUT else scale_value_width(value)


TITLE = "Olympic Medals"
SCREEN_ID = "Olympic Medal Count"
TITLE_FONT = FONT_TITLE_SPORTS
LOGO_DIR = os.path.join(IMAGES_DIR, "oly")
LEAGUE_LOGO_KEYS = ("olympics", "OLY", "iihf")

ROW_H = _scale_y(21)
HEADER_H = _scale_y(18)
TITLE_GAP = _scale_y(8)
TABLE_MARGIN_X = scale_value_width(6)

COL_WIDTHS = [
    scale_value_width(26),  # rank
    scale_value_width(34),  # flag/logo
    scale_value_width(56),  # country code
    scale_value_width(34),  # G
    scale_value_width(34),  # S
    scale_value_width(34),  # B
    scale_value_width(40),  # T
]
_TOTAL_COL_WIDTH = sum(COL_WIDTHS)
_COL_LEFT = max(TABLE_MARGIN_X, (WIDTH - _TOTAL_COL_WIDTH) // 2)
COL_X = [_COL_LEFT]
for w in COL_WIDTHS:
    COL_X.append(COL_X[-1] + w)

BASE_ROW_FONT = 16
BASE_HEADER_FONT = 14
TEAM_LOGO_BASE_HEIGHT = scale_value_width(18)
LEAGUE_LOGO_BASE_HEIGHT = standard_scoreboard_league_logo_height(TEAM_LOGO_BASE_HEIGHT)
LEAGUE_LOGO_GAP = _scale_y(4)

_LOGO_CACHE: dict[tuple[str, int], Image.Image | None] = {}
_LEAGUE_LOGO_CACHE: dict[int, Image.Image | None] = {}


def _style():
    row_font = get_screen_font(SCREEN_ID, "row", base_font=FONT_TEAM_SPORTS, default_size=BASE_ROW_FONT)
    header_font = get_screen_font(SCREEN_ID, "header", base_font=FONT_STATUS, default_size=BASE_HEADER_FONT)
    background = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)
    logo_scale = get_screen_image_scale(SCREEN_ID, "team_logo", 1.0)
    league_scale = get_screen_image_scale(SCREEN_ID, "league_logo", logo_scale)
    logo_h = max(10, int(round(TEAM_LOGO_BASE_HEIGHT * logo_scale)))
    league_logo_h = max(10, int(round(LEAGUE_LOGO_BASE_HEIGHT * league_scale)))
    return row_font, header_font, background, logo_h, league_logo_h


def _center_text(draw, text, font, x, width, y, height, fill=(255, 255, 255)):
    if not text:
        return
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text((x + (width - (r - l)) // 2 - l, y + (height - (b - t)) // 2 - t), text, font=font, fill=fill)


def _load_logo(code: str, logo_h: int):
    code = (code or "").upper()
    if not code:
        return None
    token = (code, logo_h)
    if token in _LOGO_CACHE:
        return _LOGO_CACHE[token]
    for candidate in (code, code.lower()):
        logo = load_team_logo(LOGO_DIR, candidate, height=logo_h, box_size=logo_h, trim=True)
        if logo is not None:
            _LOGO_CACHE[token] = logo
            return logo
    _LOGO_CACHE[token] = None
    return None


def _league_logo(league_logo_h: int):
    if league_logo_h in _LEAGUE_LOGO_CACHE:
        return _LEAGUE_LOGO_CACHE[league_logo_h]
    for key in LEAGUE_LOGO_KEYS:
        logo = load_team_logo(LOGO_DIR, key, height=league_logo_h, box_size=league_logo_h)
        if logo is not None:
            _LEAGUE_LOGO_CACHE[league_logo_h] = logo
            return logo
    _LEAGUE_LOGO_CACHE[league_logo_h] = None
    return None


def _render_table(rows, row_font, header_font, background, logo_h):
    table_h = HEADER_H + len(rows) * ROW_H
    canvas = Image.new("RGB", (WIDTH, table_h), background)
    draw = ImageDraw.Draw(canvas)

    headers = ("#", "", "CTY", "G", "S", "B", "T")
    for idx, label in enumerate(headers):
        _center_text(draw, label, header_font, COL_X[idx], COL_WIDTHS[idx], 0, HEADER_H, fill=(220, 220, 220))
    draw.line((COL_X[0], HEADER_H, COL_X[-1], HEADER_H), fill=(70, 70, 70))

    for i, row in enumerate(rows):
        y = HEADER_H + i * ROW_H
        if i % 2 == 1:
            draw.rectangle((COL_X[0], y, COL_X[-1], y + ROW_H), fill=(20, 20, 20))

        _center_text(draw, str(row["rank"]), row_font, COL_X[0], COL_WIDTHS[0], y, ROW_H)

        code = str(row["country"])
        logo = _load_logo(code, logo_h)
        if logo:
            canvas.paste(logo, (COL_X[1] + (COL_WIDTHS[1] - logo.width) // 2, y + (ROW_H - logo.height) // 2), logo)

        _center_text(draw, code, row_font, COL_X[2], COL_WIDTHS[2], y, ROW_H)
        _center_text(draw, str(row["gold"]), row_font, COL_X[3], COL_WIDTHS[3], y, ROW_H, fill=(255, 225, 80))
        _center_text(draw, str(row["silver"]), row_font, COL_X[4], COL_WIDTHS[4], y, ROW_H, fill=(220, 220, 220))
        _center_text(draw, str(row["bronze"]), row_font, COL_X[5], COL_WIDTHS[5], y, ROW_H, fill=(205, 127, 50))
        _center_text(draw, str(row["total"]), row_font, COL_X[6], COL_WIDTHS[6], y, ROW_H)

        draw.line((COL_X[0], y + ROW_H, COL_X[-1], y + ROW_H), fill=(40, 40, 40))

    return canvas


@log_call
def draw_olympic_medal_count(display, transition: bool = False) -> ScreenImage:
    row_font, header_font, background, logo_h, league_logo_h = _style()
    rows = fetch_olympic_medal_table(top_n=20)
    table = _render_table(rows, row_font, header_font, background, logo_h)

    dummy = Image.new("RGB", (WIDTH, 10), background)
    title_h = ImageDraw.Draw(dummy).textbbox((0, 0), TITLE, font=TITLE_FONT)[3]
    logo = _league_logo(league_logo_h)
    logo_h_title = logo.height if logo else 0
    logo_gap = LEAGUE_LOGO_GAP if logo else 0
    content_top = logo_h_title + logo_gap + title_h + TITLE_GAP
    full_img = Image.new("RGB", (WIDTH, max(HEIGHT, content_top + table.height)), background)
    draw = ImageDraw.Draw(full_img)

    if logo:
        full_img.paste(logo, ((WIDTH - logo.width) // 2, 0), logo)
    draw.text(((WIDTH - draw.textbbox((0, 0), TITLE, font=TITLE_FONT)[2]) // 2, logo_h_title + logo_gap), TITLE, font=TITLE_FONT, fill=(255, 255, 255))
    full_img.paste(table, (0, content_top))

    if transition:
        clear_display(display)

    if full_img.height <= HEIGHT:
        display.image(full_img)
        return ScreenImage(full_img, displayed=True)

    max_offset = full_img.height - HEIGHT
    display.image(full_img.crop((0, 0, WIDTH, HEIGHT)))
    time.sleep(SCOREBOARD_SCROLL_PAUSE_TOP)
    for offset in range(SCOREBOARD_SCROLL_STEP, max_offset + 1, SCOREBOARD_SCROLL_STEP):
        display.image(full_img.crop((0, offset, WIDTH, offset + HEIGHT)))
        time.sleep(SCOREBOARD_SCROLL_DELAY)
    time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    return ScreenImage(full_img, displayed=True)
