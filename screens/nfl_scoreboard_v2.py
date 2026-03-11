#!/usr/bin/env python3
"""
nfl_scoreboard_v2.py

Dual-game NFL scoreboard layout - displays 2 games per line.
Compact layout with smaller fonts and logos for a denser presentation.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import time
from typing import Optional

from PIL import Image, ImageDraw

from config import (
    WIDTH,
    HEIGHT,
    FONT_TITLE_SPORTS,
    FONT_TEAM_SPORTS,
    FONT_STATUS,
    CENTRAL_TIME,
    IMAGES_DIR,
    SCOREBOARD_SCROLL_STEP,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
    SCOREBOARD_STANDINGS_BOTTOM_PADDING,
    SCOREBOARD_BACKGROUND_COLOR,
    SCOREBOARD_IN_PROGRESS_SCORE_COLOR,
    SCOREBOARD_FINAL_WINNING_SCORE_COLOR,
    SCOREBOARD_FINAL_LOSING_SCORE_COLOR,
    get_screen_background_color,
    get_screen_font,
    get_screen_image_scale,
    is_kernel_driven_display,
    is_hdmi_1080p_layout,
    is_hyperpixel_next_layout,
    is_display_profile,
    scale_value,
    scale_value_width,
)
from utils import (
    ScreenImage,
    clear_display,
    load_team_logo,
    log_missing_team_logo,
    log_call,
    scroll_vertical_content,
    clone_font,
)

# Import shared NFL data fetching logic
from screens.nfl_scoreboard import (
    _fetch_games_for_week,
    _is_game_in_progress,
    _is_game_final,
    _is_super_bowl_game,
    _should_display_scores,
    _score_text,
    _score_value,
    _team_result,
    _final_results,
    _format_status,
    _team_logo_abbr,
    _get_league_logo,
    render_nfl_scoreboard as render_nfl_scoreboard_v1,
)

# ─── Constants ────────────────────────────────────────────────────────────────
HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()
_IS_1080P_LAYOUT = is_hdmi_1080p_layout()
_HD_LAYOUT_TEXT_BOOST = 1.25 if _IS_1080P_LAYOUT else 1.0
_IS_HYPERPIXEL_4_PROFILE = is_display_profile("hyperpixel4") or is_display_profile("hyperpixel4_square")


def _scale_y(value: int) -> int:
    return scale_value(value) if HYPERPIXEL_LAYOUT else scale_value_width(value)


TITLE = "NFL Scoreboard"
TITLE_GAP = _scale_y(8)
BLOCK_SPACING = _scale_y(8)
PAIR_SPACING = scale_value_width(4)
SCORE_ROW_H = max(1, int(round(_scale_y(30) * _HD_LAYOUT_TEXT_BOOST)))
STATUS_ROW_H = max(1, int(round(_scale_y(14) * _HD_LAYOUT_TEXT_BOOST)))
SUPER_BOWL_LOGO_GAP = _scale_y(6)

# Dual-game column layout (per game, 160px wide)
# [Score 40][Logo 30][@ 20][Logo 30][Score 40] = 160
GAME_COL_WIDTHS = [
    scale_value_width(40),
    scale_value_width(30),
    scale_value_width(20),
    scale_value_width(30),
    scale_value_width(40),
]
GAME_WIDTH = sum(GAME_COL_WIDTHS)
GAME_COL_X = [0]
for w in GAME_COL_WIDTHS:
    GAME_COL_X.append(GAME_COL_X[-1] + w)

SCREEN_ID = "NFL Scoreboard v2"
MIN_GAMES_FOR_V2_LAYOUT = 6
TITLE_FONT = FONT_TITLE_SPORTS
TEAM_LOGO_BASE_HEIGHT = scale_value_width(26)
LEAGUE_LOGO_BASE_HEIGHT = TEAM_LOGO_BASE_HEIGHT
LOGO_HEIGHT = TEAM_LOGO_BASE_HEIGHT
LEAGUE_LOGO_HEIGHT = LEAGUE_LOGO_BASE_HEIGHT
SCORE_FONT = get_screen_font(
    SCREEN_ID,
    "score",
    base_font=FONT_TEAM_SPORTS,
    default_size=20,
)
if _IS_HYPERPIXEL_4_PROFILE:
    SCORE_FONT = clone_font(SCORE_FONT, getattr(SCORE_FONT, "size", 20) + 3)
STATUS_FONT = get_screen_font(
    SCREEN_ID,
    "status",
    base_font=FONT_STATUS,
    default_size=18,
)
CENTER_FONT = get_screen_font(
    SCREEN_ID,
    "center",
    base_font=FONT_STATUS,
    default_size=18,
)
LOGO_DIR = os.path.join(IMAGES_DIR, "nfl")
LEAGUE_LOGO_KEYS = ("NFL", "nfl")
LEAGUE_LOGO_GAP = _scale_y(4)
LEAGUE_LOGO_HEIGHT = LEAGUE_LOGO_BASE_HEIGHT

IN_PROGRESS_SCORE_COLOR = SCOREBOARD_IN_PROGRESS_SCORE_COLOR
IN_PROGRESS_STATUS_COLOR = IN_PROGRESS_SCORE_COLOR
FINAL_WINNING_SCORE_COLOR = SCOREBOARD_FINAL_WINNING_SCORE_COLOR
FINAL_LOSING_SCORE_COLOR = SCOREBOARD_FINAL_LOSING_SCORE_COLOR
BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)

_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}
_SUPER_BOWL_LOGO_CACHE: dict[int, Optional[Image.Image]] = {}


def _apply_style_overrides() -> None:
    global SCORE_FONT, STATUS_FONT, CENTER_FONT, LOGO_HEIGHT, LEAGUE_LOGO_HEIGHT, BACKGROUND_COLOR

    SCORE_FONT = get_screen_font(
        SCREEN_ID,
        "score",
        base_font=FONT_TEAM_SPORTS,
        default_size=20,
    )
    if _IS_HYPERPIXEL_4_PROFILE:
        SCORE_FONT = clone_font(SCORE_FONT, getattr(SCORE_FONT, "size", 20) + 3)
    STATUS_FONT = get_screen_font(
        SCREEN_ID,
        "status",
        base_font=FONT_STATUS,
        default_size=18,
    )
    CENTER_FONT = get_screen_font(
        SCREEN_ID,
        "center",
        base_font=FONT_STATUS,
        default_size=18,
    )
    if _IS_1080P_LAYOUT:
        SCORE_FONT = clone_font(SCORE_FONT, max(1, int(round(getattr(SCORE_FONT, "size", 20) * _HD_LAYOUT_TEXT_BOOST))))
        STATUS_FONT = clone_font(STATUS_FONT, max(1, int(round(getattr(STATUS_FONT, "size", 18) * _HD_LAYOUT_TEXT_BOOST))))
        CENTER_FONT = clone_font(CENTER_FONT, max(1, int(round(getattr(CENTER_FONT, "size", 18) * _HD_LAYOUT_TEXT_BOOST))))
    BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)
    team_scale = get_screen_image_scale(SCREEN_ID, "team_logo", 1.0)
    if _IS_1080P_LAYOUT:
        team_scale *= 1.2
    LOGO_HEIGHT = max(1, int(round(TEAM_LOGO_BASE_HEIGHT * team_scale)))
    if is_kernel_driven_display():
        LEAGUE_LOGO_HEIGHT = LOGO_HEIGHT
    else:
        league_scale = get_screen_image_scale(SCREEN_ID, "league_logo", team_scale)
        LEAGUE_LOGO_HEIGHT = max(1, int(round(LEAGUE_LOGO_BASE_HEIGHT * league_scale)))


def _load_logo_cached(abbr: str) -> Optional[Image.Image]:
    key = (abbr or "").strip()
    if not key:
        return None
    cache_key = key.upper()
    height = LOGO_HEIGHT
    cache_token = (cache_key, height)
    if cache_token in _LOGO_CACHE:
        return _LOGO_CACHE[cache_token]

    candidates = [cache_key, cache_key.lower(), cache_key.title()]
    for candidate in candidates:
        path = os.path.join(LOGO_DIR, f"{candidate}.png")
        if os.path.exists(path):
            logo = load_team_logo(LOGO_DIR, candidate, height=height, box_size=height)
            _LOGO_CACHE[cache_token] = logo
            return logo

    _LOGO_CACHE[cache_token] = None
    return None


def _get_super_bowl_logo() -> Optional[Image.Image]:
    height = LOGO_HEIGHT
    if height in _SUPER_BOWL_LOGO_CACHE:
        return _SUPER_BOWL_LOGO_CACHE[height]
    logo = load_team_logo(LOGO_DIR, "SB", height=height, box_size=height)
    _SUPER_BOWL_LOGO_CACHE[height] = logo
    return logo


def _center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    x: int,
    width: int,
    y: int,
    height: int,
    *,
    fill=(255, 255, 255),
):
    if not text:
        return
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
        tx = x + (width - tw) // 2 - l
        ty = y + (height - th) // 2 - t
    except Exception:
        tw, th = draw.textsize(text, font=font)
        tx = x + (width - tw) // 2
        ty = y + (height - th) // 2
    draw.text((tx, ty), text, font=font, fill=fill)


def _score_fill(
    team_key: str, *, in_progress: bool, final: bool, results: dict
) -> tuple[int, int, int]:
    if in_progress:
        return IN_PROGRESS_SCORE_COLOR
    if final:
        result = results.get(team_key)
        if result == "loss":
            return FINAL_LOSING_SCORE_COLOR
        if result == "win":
            return FINAL_WINNING_SCORE_COLOR
    return (255, 255, 255)


def _draw_single_game(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, game: dict, x_offset: int, top: int
):
    """Draw a single game within the dual-game layout."""
    teams = (game or {}).get("teams", {})
    away = teams.get("away", {})
    home = teams.get("home", {})

    show_scores = _should_display_scores(game)
    away_text = _score_text(away, show=show_scores)
    home_text = _score_text(home, show=show_scores)
    in_progress = _is_game_in_progress(game)
    final = _is_game_final(game)
    results = _final_results(away, home) if final else {"away": None, "home": None}

    score_top = top

    # Draw scores and @ symbol
    for idx, text in ((0, away_text), (2, "@"), (4, home_text)):
        font = SCORE_FONT if idx != 2 else CENTER_FONT
        if idx == 0:
            fill = _score_fill("away", in_progress=in_progress, final=final, results=results)
        elif idx == 4:
            fill = _score_fill("home", in_progress=in_progress, final=final, results=results)
        else:
            fill = (255, 255, 255)
        _center_text(
            draw,
            text,
            font,
            x_offset + GAME_COL_X[idx],
            GAME_COL_WIDTHS[idx],
            score_top,
            SCORE_ROW_H,
            fill=fill,
        )

    # Draw team logos
    for idx, team_side, team_key in ((1, away, "away"), (3, home, "home")):
        team_obj = (team_side or {}).get("team", {})
        abbr = _team_logo_abbr(team_obj)
        logo = _load_logo_cached(abbr)
        if not logo:
            team_name = (
                (team_obj or {}).get("displayName")
                or (team_obj or {}).get("name")
                or (team_obj or {}).get("shortDisplayName")
                or "Unknown Team"
            )
            log_missing_team_logo(SCREEN_ID, team_name, abbr)
            continue
        x0 = x_offset + GAME_COL_X[idx] + (GAME_COL_WIDTHS[idx] - logo.width) // 2
        y0 = score_top + (SCORE_ROW_H - logo.height) // 2
        canvas.paste(logo, (x0, y0), logo)

    # Draw status
    status_top = score_top + SCORE_ROW_H
    status_text = _format_status(game)
    status_fill = IN_PROGRESS_STATUS_COLOR if in_progress else (255, 255, 255)
    _center_text(
        draw,
        status_text,
        STATUS_FONT,
        x_offset + GAME_COL_X[0],
        GAME_WIDTH,
        status_top,
        STATUS_ROW_H,
        fill=status_fill,
    )


def _draw_game_pair(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, game1: dict, game2: Optional[dict], top: int
):
    """Draw a pair of games side by side."""
    _draw_single_game(canvas, draw, game1, 0, top)
    if game2:
        _draw_single_game(canvas, draw, game2, GAME_WIDTH, top)


def _compose_canvas(games: list[dict], *, show_super_bowl_logo: bool) -> Image.Image:
    if not games:
        return Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)

    # Calculate pairs
    pairs = []
    for i in range(0, len(games), 2):
        game1 = games[i]
        game2 = games[i + 1] if i + 1 < len(games) else None
        pairs.append((game1, game2))

    # Calculate canvas height
    pair_height = SCORE_ROW_H + STATUS_ROW_H
    total_height = pair_height * len(pairs)
    if len(pairs) > 1:
        total_height += BLOCK_SPACING * (len(pairs) - 1)
    super_bowl_logo = _get_super_bowl_logo() if show_super_bowl_logo else None
    if super_bowl_logo:
        total_height += SUPER_BOWL_LOGO_GAP + super_bowl_logo.height

    canvas = Image.new("RGB", (WIDTH, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    y = 0
    for idx, (game1, game2) in enumerate(pairs):
        _draw_game_pair(canvas, draw, game1, game2, y)
        y += pair_height
        if idx < len(pairs) - 1:
            sep_y = y + BLOCK_SPACING // 2
            draw.line((10, sep_y, WIDTH - 10, sep_y), fill=(45, 45, 45))
            y += BLOCK_SPACING
    if super_bowl_logo:
        y += SUPER_BOWL_LOGO_GAP
        logo_x = (WIDTH - super_bowl_logo.width) // 2
        canvas.paste(super_bowl_logo, (logo_x, y), super_bowl_logo)

    return canvas


def _render_scoreboard(games: list[dict], *, show_super_bowl_logo: bool) -> Image.Image:
    canvas = _compose_canvas(games, show_super_bowl_logo=show_super_bowl_logo)

    dummy = Image.new("RGB", (WIDTH, 10), BACKGROUND_COLOR)
    dd = ImageDraw.Draw(dummy)
    try:
        l, t, r, b = dd.textbbox((0, 0), TITLE, font=TITLE_FONT)
        title_h = b - t
    except Exception:
        _, title_h = dd.textsize(TITLE, font=TITLE_FONT)

    league_logo = _get_league_logo()
    logo_height = league_logo.height if league_logo else 0
    logo_gap = LEAGUE_LOGO_GAP if league_logo else 0

    content_top = logo_height + logo_gap + title_h + TITLE_GAP
    img_height = max(
        HEIGHT,
        content_top + canvas.height + SCOREBOARD_STANDINGS_BOTTOM_PADDING,
    )
    img = Image.new("RGB", (WIDTH, img_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    if league_logo:
        logo_x = (WIDTH - league_logo.width) // 2
        img.paste(league_logo, (logo_x, 0), league_logo)
    title_top = logo_height + logo_gap

    try:
        l, t, r, b = draw.textbbox((0, 0), TITLE, font=TITLE_FONT)
        tw, th = r - l, b - t
        tx = (WIDTH - tw) // 2 - l
        ty = title_top - t
    except Exception:
        tw, th = draw.textsize(TITLE, font=TITLE_FONT)
        tx = (WIDTH - tw) // 2
        ty = title_top
    draw.text((tx, ty), TITLE, font=TITLE_FONT, fill=(255, 255, 255))

    img.paste(canvas, (0, content_top))
    return img


def _scroll_display(display, full_img: Image.Image):
    scroll_vertical_content(
        display=display,
        content_height=full_img.height,
        viewport_width=WIDTH,
        viewport_height=HEIGHT,
        render_at_offset=lambda offset: display.image(
            full_img.crop((0, offset, WIDTH, offset + HEIGHT))
        ),
        base_step=SCOREBOARD_SCROLL_STEP,
        pause_start=SCOREBOARD_SCROLL_PAUSE_TOP,
        pause_end=SCOREBOARD_SCROLL_PAUSE_BOTTOM,
        min_frame_time=SCOREBOARD_SCROLL_DELAY,
    )


def render_nfl_scoreboard_v2(display, games: list[dict], transition: bool = False) -> ScreenImage:
    if len(games) < MIN_GAMES_FOR_V2_LAYOUT:
        return render_nfl_scoreboard_v1(display, games, transition=transition)

    _apply_style_overrides()
    show_super_bowl_logo = len(games) == 1 and _is_super_bowl_game(games[0])

    if not games:
        clear_display(display)
        img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        league_logo = _get_league_logo()
        title_top = 0
        if league_logo:
            logo_x = (WIDTH - league_logo.width) // 2
            img.paste(league_logo, (logo_x, 0), league_logo)
            title_top = league_logo.height + LEAGUE_LOGO_GAP
        try:
            l, t, r, b = draw.textbbox((0, 0), TITLE, font=TITLE_FONT)
            tw, th = r - l, b - t
            tx = (WIDTH - tw) // 2 - l
            ty = title_top - t
        except Exception:
            tw, th = draw.textsize(TITLE, font=TITLE_FONT)
            tx = (WIDTH - tw) // 2
            ty = title_top
        draw.text((tx, ty), TITLE, font=TITLE_FONT, fill=(255, 255, 255))
        _center_text(
            draw, "No games", STATUS_FONT, 0, WIDTH, HEIGHT // 2 - STATUS_ROW_H // 2, STATUS_ROW_H
        )
        if transition:
            return ScreenImage(img, displayed=False)
        display.image(img)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
        return ScreenImage(img, displayed=True)

    full_img = _render_scoreboard(games, show_super_bowl_logo=show_super_bowl_logo)
    if transition:
        _scroll_display(display, full_img)
        return ScreenImage(full_img, displayed=True)

    if full_img.height <= HEIGHT:
        display.image(full_img)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    else:
        _scroll_display(display, full_img)
    return ScreenImage(full_img, displayed=True)


@log_call
def draw_nfl_scoreboard_v2(display, transition: bool = False) -> ScreenImage:
    now = datetime.datetime.now(CENTRAL_TIME)
    games = _fetch_games_for_week(now)
    return render_nfl_scoreboard_v2(display, games, transition=transition)


if __name__ == "__main__":
    from utils import Display

    disp = Display()
    try:
        draw_nfl_scoreboard_v2(disp)
    finally:
        clear_display(disp)
