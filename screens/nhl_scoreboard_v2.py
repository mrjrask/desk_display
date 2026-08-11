#!/usr/bin/env python3
"""
nhl_scoreboard_v2.py

Dual-game NHL scoreboard layout - displays 2 games per line.
Compact layout with smaller fonts and logos for a denser presentation.
"""

from __future__ import annotations

import os
from typing import Optional

from PIL import Image, ImageDraw

from config import (
    FONT_STATUS,
    FONT_TEAM_SPORTS,
    FONT_TITLE_SPORTS,
    HEIGHT,
    IMAGES_DIR,
    SCOREBOARD_BACKGROUND_COLOR,
    SCOREBOARD_FINAL_LOSING_SCORE_COLOR,
    SCOREBOARD_FINAL_WINNING_SCORE_COLOR,
    SCOREBOARD_IN_PROGRESS_SCORE_COLOR,
    WIDTH,
    get_screen_background_color,
    get_screen_font,
    get_screen_image_scale,
    is_display_profile,
    is_hdmi_1080p_layout,
    is_hyperpixel_next_layout,
    is_kernel_driven_display,
    scale_value,
    scale_value_width,
)

# Import shared NHL render helpers from v1 module.
from screens.nhl_scoreboard import (
    _final_results,
    _format_status,
    _get_league_logo,
    _is_game_final,
    _is_game_in_progress,
    _score_text,
    _should_display_scores,
    _team_logo_abbr,
    render_nhl_scoreboard as render_nhl_scoreboard_v1,
)
from screens.scoreboard_components import (
    display_no_games,
    display_or_scroll_scoreboard,
    draw_score_game_row,
    render_headered_scoreboard,
    render_no_games_image,
    score_fill as shared_score_fill,
    scroll_display,
)
from services.sports.nhl import fetch_scoreboard
from utils import (
    ScreenImage,
    clear_display,
    clone_font,
    load_team_logo,
    log_call,
    standard_scoreboard_league_logo_height,
)

# ─── Constants ────────────────────────────────────────────────────────────────
HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()
_IS_1080P_LAYOUT = is_hdmi_1080p_layout()
_HD_LAYOUT_TEXT_BOOST = 1.25 if _IS_1080P_LAYOUT else 1.0
_IS_HYPERPIXEL_4_PROFILE = is_display_profile("hyperpixel4") or is_display_profile("hyperpixel4_square")


def _scale_y(value: int) -> int:
    return scale_value(value) if HYPERPIXEL_LAYOUT else scale_value_width(value)


TITLE = "NHL Scoreboard"
TITLE_GAP = _scale_y(8)
BLOCK_SPACING = _scale_y(8)
PAIR_SPACING = scale_value_width(4)
SCORE_ROW_H = max(1, int(round(_scale_y(30) * _HD_LAYOUT_TEXT_BOOST)))
STATUS_ROW_H = max(1, int(round(_scale_y(14) * _HD_LAYOUT_TEXT_BOOST)))
# Dual-game column layout (per game, 160px wide)
# [Score 44][Logo 30][@ 12][Logo 30][Score 44] = 160
GAME_COL_WIDTHS = [
    scale_value_width(44),
    scale_value_width(30),
    scale_value_width(12),
    scale_value_width(30),
    scale_value_width(44),
]
GAME_WIDTH = sum(GAME_COL_WIDTHS)
GAME_COL_X = [0]
for w in GAME_COL_WIDTHS:
    GAME_COL_X.append(GAME_COL_X[-1] + w)

SCREEN_ID = "NHL Scoreboard v2"
MIN_GAMES_FOR_V2_LAYOUT = 6
V2_DISABLED_RESOLUTIONS = {(320, 240), (240, 320), (135, 240)}
TITLE_FONT = FONT_TITLE_SPORTS
TEAM_LOGO_BASE_HEIGHT = scale_value_width(26)
LEAGUE_LOGO_BASE_HEIGHT = TEAM_LOGO_BASE_HEIGHT if is_kernel_driven_display() else standard_scoreboard_league_logo_height(TEAM_LOGO_BASE_HEIGHT)
LOGO_HEIGHT = TEAM_LOGO_BASE_HEIGHT
LEAGUE_LOGO_HEIGHT = LEAGUE_LOGO_BASE_HEIGHT

_SCOREBOARD_BASE_FONT_SIZES = {
    "score": 24,
    "status": 18,
    "center": 18,
}


def _scoreboard_font_sizes() -> dict[str, int]:
    return dict(_SCOREBOARD_BASE_FONT_SIZES)


def _scoreboard_fonts() -> tuple:
    sizes = _scoreboard_font_sizes()
    score = get_screen_font(
        SCREEN_ID,
        "score",
        base_font=FONT_TEAM_SPORTS,
        default_size=sizes["score"],
    )
    if _IS_HYPERPIXEL_4_PROFILE:
        score = clone_font(score, getattr(score, "size", 20) + 3)
    status = get_screen_font(
        SCREEN_ID,
        "status",
        base_font=FONT_STATUS,
        default_size=sizes["status"],
    )
    center = get_screen_font(
        SCREEN_ID,
        "center",
        base_font=FONT_STATUS,
        default_size=sizes["center"],
    )
    if _IS_1080P_LAYOUT:
        score = clone_font(score, max(1, int(round(getattr(score, "size", 20) * _HD_LAYOUT_TEXT_BOOST))))
        status = clone_font(status, max(1, int(round(getattr(status, "size", 18) * _HD_LAYOUT_TEXT_BOOST))))
        center = clone_font(center, max(1, int(round(getattr(center, "size", 18) * _HD_LAYOUT_TEXT_BOOST))))
    return score, status, center


SCORE_FONT, STATUS_FONT, CENTER_FONT = _scoreboard_fonts()
LOGO_DIR = os.path.join(IMAGES_DIR, "nhl")
LEAGUE_LOGO_KEYS = ("NHL", "nhl")
LEAGUE_LOGO_GAP = _scale_y(4)
LEAGUE_LOGO_HEIGHT = LEAGUE_LOGO_BASE_HEIGHT

IN_PROGRESS_SCORE_COLOR = SCOREBOARD_IN_PROGRESS_SCORE_COLOR
IN_PROGRESS_STATUS_COLOR = IN_PROGRESS_SCORE_COLOR
FINAL_WINNING_SCORE_COLOR = SCOREBOARD_FINAL_WINNING_SCORE_COLOR
FINAL_LOSING_SCORE_COLOR = SCOREBOARD_FINAL_LOSING_SCORE_COLOR
BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)

_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}


def _apply_style_overrides() -> None:
    global SCORE_FONT, STATUS_FONT, CENTER_FONT, LOGO_HEIGHT, LEAGUE_LOGO_HEIGHT, BACKGROUND_COLOR

    SCORE_FONT, STATUS_FONT, CENTER_FONT = _scoreboard_fonts()
    BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)
    team_scale = get_screen_image_scale(SCREEN_ID, "team_logo", 1.0)
    if _IS_1080P_LAYOUT:
        team_scale *= 1.2
    target_logo_height = max(1, int(round(TEAM_LOGO_BASE_HEIGHT * team_scale)))
    max_row_fit = max(1, SCORE_ROW_H - scale_value(4))
    LOGO_HEIGHT = min(target_logo_height, max_row_fit)
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
            logo = load_team_logo(LOGO_DIR, candidate, height=height, box_size=height, trim=True)
            _LOGO_CACHE[cache_token] = logo
            return logo

    _LOGO_CACHE[cache_token] = None
    return None


def _score_fill(
    team_key: str, *, in_progress: bool, final: bool, results: dict
) -> tuple[int, int, int]:
    return shared_score_fill(
        team_key,
        in_progress=in_progress,
        final=final,
        results=results,
        in_progress_color=IN_PROGRESS_SCORE_COLOR,
        winning_color=FINAL_WINNING_SCORE_COLOR,
        losing_color=FINAL_LOSING_SCORE_COLOR,
    )


def _draw_single_game(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, game: dict, x_offset: int, top: int
):
    """Draw a single game within the dual-game layout."""
    draw_score_game_row(
        canvas=canvas,
        draw=draw,
        game=game,
        x_offset=x_offset,
        top=top,
        col_x=GAME_COL_X,
        col_widths=GAME_COL_WIDTHS,
        score_row_h=SCORE_ROW_H,
        status_row_h=STATUS_ROW_H,
        score_font=SCORE_FONT,
        center_font=CENTER_FONT,
        status_font=STATUS_FONT,
        score_text=lambda team, show: _score_text(team, show=show),
        should_display_scores=_should_display_scores,
        is_game_in_progress=_is_game_in_progress,
        is_game_final=_is_game_final,
        final_results=_final_results,
        format_status=_format_status,
        team_logo_abbr=_team_logo_abbr,
        load_logo=_load_logo_cached,
        screen_id=SCREEN_ID,
        in_progress_score_color=IN_PROGRESS_SCORE_COLOR,
        in_progress_status_color=IN_PROGRESS_STATUS_COLOR,
        final_winning_score_color=FINAL_WINNING_SCORE_COLOR,
        final_losing_score_color=FINAL_LOSING_SCORE_COLOR,
        status_x_index=0,
        status_width=GAME_WIDTH,
    )


def _draw_game_pair(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, game1: dict, game2: Optional[dict], top: int
):
    """Draw a pair of games side by side."""
    _draw_single_game(canvas, draw, game1, 0, top)
    if game2:
        _draw_single_game(canvas, draw, game2, GAME_WIDTH, top)


def _compose_canvas(games: list[dict]) -> Image.Image:
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

    return canvas


def _render_scoreboard(games: list[dict]) -> Image.Image:
    canvas = _compose_canvas(games)
    return render_headered_scoreboard(
        canvas=canvas,
        title=TITLE,
        title_font=TITLE_FONT,
        league_logo=_get_league_logo(),
        league_logo_gap=LEAGUE_LOGO_GAP,
        title_gap=TITLE_GAP,
        background_color=BACKGROUND_COLOR,
    )

def _scroll_display(display, full_img: Image.Image):
    return scroll_display(display, full_img)

def render_nhl_scoreboard_v2(display, games: list[dict], transition: bool = False) -> ScreenImage:
    if (WIDTH, HEIGHT) in V2_DISABLED_RESOLUTIONS:
        return render_nhl_scoreboard_v1(display, games, transition=transition)

    if len(games) < MIN_GAMES_FOR_V2_LAYOUT:
        return render_nhl_scoreboard_v1(display, games, transition=transition)

    _apply_style_overrides()

    if not games:
        img = render_no_games_image(
            title=TITLE,
            title_font=TITLE_FONT,
            status_font=STATUS_FONT,
            status_row_h=STATUS_ROW_H,
            league_logo=_get_league_logo(),
            league_logo_gap=LEAGUE_LOGO_GAP,
            background_color=BACKGROUND_COLOR,
            message="No games",
        )
        return display_no_games(display, img, transition=transition)


    full_img = _render_scoreboard(games)
    return display_or_scroll_scoreboard(display, full_img, transition=transition)


@log_call
def draw_nhl_scoreboard_v2(display, transition: bool = False) -> ScreenImage:
    games = fetch_scoreboard()
    return render_nhl_scoreboard_v2(display, games, transition=transition)


if __name__ == "__main__":
    from utils import Display

    disp = Display()
    try:
        draw_nhl_scoreboard_v2(disp)
    finally:
        clear_display(disp)
