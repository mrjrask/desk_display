#!/usr/bin/env python3
"""
draw_bears_schedule.py

Shows the next Chicago Bears game with:
  - Title at y=0
  - Opponent wrapped in up to two lines, prefixed by '@' if the Bears are away,
    or 'vs.' if the Bears are home.
  - Between those and the bottom line, a row of logos: AWAY @ HOME, each logo
    auto-sized similarly to the Hawks schedule screen.
  - Bottom lines with week above date/time.
"""

import datetime
import logging
import os
import re
import time
from functools import lru_cache
from typing import Any
from PIL import Image, ImageDraw, ImageFont
import config
from config import (
    BEARS_BOTTOM_MARGIN,
    BEARS_SCHEDULE,
    NFL_TEAM_ABBREVIATIONS,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_STEP,
    get_screen_background_color,
)
from utils import (
    ScreenImage,
    load_team_logo,
    scroll_vertical_content,
    standard_next_game_logo_frame_width,
    standard_next_game_logo_height,
    standard_next_game_logo_height_for_space,
    wrap_text,
)


def _text_size(draw, text, *, font):
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    except Exception:
        return draw.textsize(text, font=font)


def _parse_schedule_date(date_text: str) -> datetime.datetime | None:
    if not date_text:
        return None
    date_text = str(date_text).strip()
    if not date_text or date_text.upper() == "BYE":
        return None
    for fmt in ("%a, %b %d %Y", "%a, %b %d, %Y", "%a, %b %d"):
        try:
            return datetime.datetime.strptime(date_text, fmt)
        except Exception:
            continue
    return None


def _format_game_date(date_text: str) -> str:
    if not date_text:
        return ""
    date_text = str(date_text).strip()
    if not date_text:
        return ""
    dt0 = _parse_schedule_date(date_text)
    if dt0 is None:
        return date_text
    date_part = f"{dt0.month}/{dt0.day}"
    if dt0.weekday() == 6:
        return date_part
    return f"{dt0.strftime('%a')} {date_part}"


NFL_LOGO_DIR = os.path.join(config.IMAGES_DIR, "nfl")
DROP_STEPS = 18
DROP_STAGGER = 0.25
DROP_FRAME_DELAY = 0.01
BEARS_TEAM_ABBR = "CHI"
BEARS_SCOREBOARD_CACHE_TTL_SECONDS = 10 * 60
_BEAR_SCORE_CACHE: dict[datetime.date, tuple[float, list[dict[str, Any]]]] = {}

DEFAULT_BEARS_NEXT_SEASON_HOME_OPPONENTS = ("det", "gb", "jax", "min", "ne", "no", "nyj", "phi", "tb")
DEFAULT_BEARS_NEXT_SEASON_AWAY_OPPONENTS = ("atl", "buf", "car", "det", "gb", "mia", "min", "sea")


@lru_cache(maxsize=96)
def _cached_team_logo(abbr: str, logo_size: int) -> Image.Image | None:
    return load_team_logo(NFL_LOGO_DIR, abbr, height=logo_size, box_size=logo_size)


def _bears_week_sort_value(week_label: str) -> float:
    week_txt = str(week_label or "").strip().lower()
    if week_txt.startswith("preseason"):
        match = re.search(r"(\d+)", week_txt)
        return float(f"0.{match.group(1)}") if match else 0.0
    if week_txt.startswith("week"):
        match = re.search(r"(\d+)", week_txt)
        return float(match.group(1)) if match else float("inf")
    try:
        return float(week_txt)
    except ValueError:
        return float("inf")


def _bears_game_sort_value(game: dict) -> float:
    game_no = game.get("game_no")
    if game_no is not None:
        try:
            return float(str(game_no))
        except ValueError:
            pass
    return _bears_week_sort_value(str(game.get("week") or ""))


def _has_completed_bears_game(game: dict, today: datetime.date) -> bool:
    parsed_date = _parse_schedule_date(str(game.get("date") or ""))
    if parsed_date is not None:
        if parsed_date.date() < today:
            return True
        if parsed_date.date() == today and _bears_schedule_score_text(game):
            return True
    return False


def _next_bears_game_from_schedule(
    schedule: list[dict],
    today: datetime.date | None = None,
) -> dict | None:
    today = today or datetime.datetime.now(config.CENTRAL_TIME).date()
    candidates = [
        game
        for game in schedule
        if _should_show_bears_schedule_game(game) and not _is_bye_week(game)
    ]
    if not candidates:
        return None

    last_completed_sort = max(
        (_bears_game_sort_value(game) for game in candidates if _has_completed_bears_game(game, today)),
        default=float("-inf"),
    )
    upcoming = [
        game
        for game in candidates
        if _bears_game_sort_value(game) > last_completed_sort
    ]
    if upcoming:
        return min(upcoming, key=_bears_game_sort_value)
    return min(candidates, key=_bears_game_sort_value)


def _ease_out_cubic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    inv = 1.0 - t
    return 1.0 - inv * inv * inv


def _animate_logo_drop(display, base: Image.Image, row_positions):
    has_logos = any(row for row in row_positions)
    if not has_logos:
        return

    steps = max(2, DROP_STEPS)
    stagger = max(1, int(round(steps * DROP_STAGGER)))

    schedule = []
    start_step = 0
    for row_idx in range(len(row_positions) - 1, -1, -1):
        drops = row_positions[row_idx]
        if not drops:
            continue
        schedule.append((start_step, drops))
        start_step += stagger

    if not schedule:
        return

    total_duration = schedule[-1][0] + steps + 1
    placed = []
    completed = [False] * len(schedule)

    for current_step in range(total_duration):
        frame_start = time.time()

        for idx, (start, drops) in enumerate(schedule):
            if current_step >= start + steps and not completed[idx]:
                placed.extend(drops)
                completed[idx] = True

        frame = base.copy()
        for logo, x0, y0 in placed:
            frame.paste(logo, (x0, y0), logo)

        for idx, (start, drops) in enumerate(schedule):
            progress = current_step - start
            if progress < 0 or progress >= steps:
                continue

            frac = progress / (steps - 1) if steps > 1 else 1.0
            eased = _ease_out_cubic(frac)
            for logo, x0, y_target in drops:
                start_y = -logo.height
                y_pos = int(start_y + (y_target - start_y) * eased)
                if y_pos > y_target:
                    y_pos = y_target
                frame.paste(logo, (x0, y_pos), logo)

        display.image(frame)
        if hasattr(display, "show"):
            display.show()

        elapsed = time.time() - frame_start
        sleep_time = max(0, DROP_FRAME_DELAY - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


def show_bears_next_game(display, transition=False):
    game = _next_bears_game_from_schedule(BEARS_SCHEDULE)
    title = "Next for Da Bears:"
    background = get_screen_background_color("bears next", (0, 0, 0))
    img   = Image.new("RGB", (config.WIDTH, config.HEIGHT), background)
    draw  = ImageDraw.Draw(img)

    hyperpixel_layout = config.is_hyperpixel_next_layout()
    edge_pad = max(2, config.scale_value(2)) if hyperpixel_layout else 2
    line_gap = max(2, config.scale_value(2)) if hyperpixel_layout else 2

    # Title
    tw, th = draw.textsize(title, font=config.FONT_TITLE_SPORTS)
    draw.text(
        ((config.WIDTH - tw) // 2, edge_pad),
        title,
        font=config.FONT_TITLE_SPORTS,
        fill=(255, 255, 255),
    )

    if game:
        opp = game["opponent"]
        ha  = game["home_away"].lower()
        prefix = "@" if ha == "away" else "vs."

        # Opponent text (up to 2 lines)
        lines  = wrap_text(f"{prefix} {opp}", config.FONT_TEAM_SPORTS, config.WIDTH)[:2]
        y_txt = th + (config.scale_value(4) if hyperpixel_layout else 4)
        for ln in lines:
            w_ln, h_ln = draw.textsize(ln, font=config.FONT_TEAM_SPORTS)
            draw.text(((config.WIDTH - w_ln)//2, y_txt),
                      ln, font=config.FONT_TEAM_SPORTS, fill=(255,255,255))
            y_txt += h_ln + line_gap

        # Logos row: AWAY @ HOME
        bears_ab = "chi"
        opp_key = opp.split()[-1].lower()
        opp_ab = NFL_TEAM_ABBREVIATIONS.get(opp_key, opp_key[:3])
        week_label = str(game.get("week", "") or "")
        if opp.strip().upper() == "TBD":
            if "super bowl" in week_label.lower():
                opp_ab = "afc"
            else:
                opp_ab = "nfc"
        if opp_ab == "was":
            opp_ab = "wsh"
        if ha == "away":
            away_ab, home_ab = bears_ab, opp_ab
        else:
            away_ab, home_ab = opp_ab, bears_ab

        # Bottom lines text — week above date/time
        wk = (game.get("week") or "").strip()
        if not wk:
            game_no = str(game.get("game_no", "")).strip()
            wk = f"Game {game_no}" if game_no else ""
        date_time = _format_schedule_line_time(
            str(game.get("date") or ""),
            str(game.get("time") or ""),
        )
        bottom_lines = [line for line in (wk, date_time) if line]
        bottom_line_gap = line_gap
        if bottom_lines:
            heights = [
                _text_size(draw, line, font=config.FONT_DATE_SPORTS)[1]
                for line in bottom_lines
            ]
            bottom_h = sum(heights) + (bottom_line_gap * (len(bottom_lines) - 1))
        else:
            bottom_h = 0
        bottom_margin = (
            config.scale_value(BEARS_BOTTOM_MARGIN) if hyperpixel_layout else BEARS_BOTTOM_MARGIN
        )
        bottom_y = config.HEIGHT - bottom_h - bottom_margin  # keep on-screen

        available_h = max(10, bottom_y - (y_txt + 2))
        if hyperpixel_layout:
            desired_logo_h = max(
                1,
                int(round(standard_next_game_logo_height(config.HEIGHT) * config.DISPLAY_SCALE)),
            )
            logo_h = min(desired_logo_h, available_h)
        else:
            logo_h = standard_next_game_logo_height_for_space(config.HEIGHT, available_h)

        logo_away = load_team_logo(NFL_LOGO_DIR, away_ab, height=logo_h, box_size=logo_h)
        logo_home = load_team_logo(NFL_LOGO_DIR, home_ab, height=logo_h, box_size=logo_h)

        if hyperpixel_layout:
            gap = max(config.scale_value(6), min(config.scale_value(10), config.WIDTH // 30))
        else:
            gap = max(6, min(10, config.WIDTH // 30))
        frame_w = standard_next_game_logo_frame_width(logo_h, (logo_away, logo_home))
        at_symbol = "@"
        try:
            l, t, r, b = draw.textbbox((0, 0), at_symbol, font=config.FONT_TEAM_SPORTS)
            at_w, at_h, at_t = r - l, b - t, t
        except Exception:
            at_w, at_h = draw.textsize(at_symbol, font=config.FONT_TEAM_SPORTS)
            at_t = 0

        block_h = logo_h if (logo_away or logo_home) else at_h
        total_w = (frame_w * 2) + (gap * 2) + at_w

        if total_w > config.WIDTH:
            gap = max(4, int(round(gap * (config.WIDTH / max(total_w, 1)))))
            total_w = (frame_w * 2) + (gap * 2) + at_w

        if total_w > config.WIDTH:
            max_frame = max(1, (config.WIDTH - at_w - (gap * 2)) // 2)
            if max_frame < frame_w:
                scale = max_frame / frame_w if frame_w else 1.0
                logo_h = max(1, int(round(logo_h * scale)))
                logo_away = load_team_logo(NFL_LOGO_DIR, away_ab, height=logo_h, box_size=logo_h)
                logo_home = load_team_logo(NFL_LOGO_DIR, home_ab, height=logo_h, box_size=logo_h)
                frame_w = min(standard_next_game_logo_frame_width(logo_h, (logo_away, logo_home)), max_frame)

            def _fit_logo(logo):
                if logo and logo.width > frame_w:
                    ratio = frame_w / logo.width
                    new_h = max(1, int(round(logo.height * ratio)))
                    return logo.resize((frame_w, new_h), Image.ANTIALIAS)
                return logo

            logo_away = _fit_logo(logo_away)
            logo_home = _fit_logo(logo_home)
            block_h = max((logo.height for logo in (logo_away, logo_home) if logo), default=at_h if not (logo_away or logo_home) else logo_h)
            total_w = (frame_w * 2) + (gap * 2) + at_w

        x0 = max(0, (config.WIDTH - total_w) // 2)

        # Vertical center of logos/text block between opponent text and bottom label
        y_logo = y_txt + ((bottom_y - y_txt) - block_h)//2

        left_x = x0
        at_x = left_x + frame_w + gap
        right_x = at_x + at_w + gap

        def _paste_logo(logo, frame_x):
            if not logo:
                return
            lx = frame_x + (frame_w - logo.width)//2
            ly = y_logo + (logo_h - logo.height)//2
            img.paste(logo, (lx, ly), logo)

        _paste_logo(logo_away, left_x)
        at_y = y_logo + (block_h - at_h)//2 - at_t
        draw.text((at_x, at_y), at_symbol, font=config.FONT_TEAM_SPORTS, fill=(255,255,255))
        _paste_logo(logo_home, right_x)

        # Draw bottom text
        if bottom_lines:
            y_bottom_text = bottom_y
            for line in bottom_lines:
                w_line, h_line = _text_size(draw, line, font=config.FONT_DATE_SPORTS)
                draw.text(
                    ((config.WIDTH - w_line) // 2, y_bottom_text),
                    line,
                    font=config.FONT_DATE_SPORTS,
                    fill=(255, 255, 255),
                )
                y_bottom_text += h_line + bottom_line_gap

    if transition:
        return img

    display.image(img)
    display.show()
    return None


@lru_cache(maxsize=8)
def _cached_bears_next_season_image(
    width: int,
    height: int,
    background: tuple[int, int, int],
    home_opponents: tuple[str, ...],
    away_opponents: tuple[str, ...],
) -> Image.Image:
    # Legacy dynamic Bears-next-season generator (kept intentionally deactivated).
    # The active implementation now uses pre-rendered static PNG assets.
    title = "2026 Bears Opponents"
    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)

    title_w, title_h = _text_size(draw, title, font=config.FONT_TITLE_SPORTS)
    draw.text(
        ((width - title_w) // 2, 0),
        title,
        font=config.FONT_TITLE_SPORTS,
        fill=(255, 255, 255),
    )

    column_width = width // 2
    header_y = title_h + 4
    header_font = config.FONT_DATE_SPORTS
    home_label = "Home"
    away_label = "Away"
    home_w, home_h = _text_size(draw, home_label, font=header_font)
    away_w, away_h = _text_size(draw, away_label, font=header_font)

    draw.text(
        ((column_width - home_w) // 2, header_y),
        home_label,
        font=header_font,
        fill=(255, 255, 255),
    )
    draw.text(
        (column_width + (column_width - away_w) // 2, header_y),
        away_label,
        font=header_font,
        fill=(255, 255, 255),
    )

    logos_top = header_y + max(home_h, away_h) + 4
    row_gap = 2
    col_gap = 4
    columns_per_side = 2
    home_rows = max(1, (len(home_opponents) + columns_per_side - 1) // columns_per_side)
    away_rows = max(1, (len(away_opponents) + columns_per_side - 1) // columns_per_side)
    rows = max(home_rows, away_rows)
    available_h = height - logos_top - 2
    subcolumn_width = max(1, (column_width - col_gap) // columns_per_side)
    logo_size = max(
        1,
        min(subcolumn_width, (available_h - row_gap * (rows - 1)) // rows),
    )

    def _logo_position(logo, x, y):
        lx = x + (logo_size - logo.width) // 2
        ly = y + (logo_size - logo.height) // 2
        return lx, ly

    placements = []

    for idx, abbr in enumerate(home_opponents):
        row = idx // columns_per_side
        col = idx % columns_per_side
        y = logos_top + row * (logo_size + row_gap)
        x = col * (subcolumn_width + col_gap) + (subcolumn_width - logo_size) // 2
        logo = _cached_team_logo(abbr, logo_size)
        if logo:
            lx, ly = _logo_position(logo, x, y)
            placements.append((logo, lx, ly))

    for idx, abbr in enumerate(away_opponents):
        row = idx // columns_per_side
        col = idx % columns_per_side
        y = logos_top + row * (logo_size + row_gap)
        x = (
            column_width
            + col * (subcolumn_width + col_gap)
            + (subcolumn_width - logo_size) // 2
        )
        logo = _cached_team_logo(abbr, logo_size)
        if logo:
            lx, ly = _logo_position(logo, x, y)
            placements.append((logo, lx, ly))

    for logo, lx, ly in placements:
        img.paste(logo, (lx, ly), logo)

    return img


def render_bears_next_season_image(
    width: int,
    height: int,
    background: tuple[int, int, int],
    home_opponents: list[str] | tuple[str, ...] | None = None,
    away_opponents: list[str] | tuple[str, ...] | None = None,
) -> Image.Image:
    home = tuple(home_opponents or DEFAULT_BEARS_NEXT_SEASON_HOME_OPPONENTS)
    away = tuple(away_opponents or DEFAULT_BEARS_NEXT_SEASON_AWAY_OPPONENTS)
    return _cached_bears_next_season_image(width, height, tuple(background), home, away).copy()


@lru_cache(maxsize=12)
def _cached_bears_next_season_static_image(width: int, height: int) -> Image.Image | None:
    if config.is_hyperpixel_4_square_layout(width, height):
        candidates = ["bears_next_season_h4sq.png", "bears_next_season.png"]
    elif config.is_hyperpixel_next_layout(width, height):
        candidates = ["bears_next_season_h4.png", "bears_next_season.png"]
    else:
        candidates = ["bears_next_season_dhm.png", "bears_next_season.png"]

    for filename in candidates:
        image_path = os.path.join(config.IMAGES_DIR, filename)
        if not os.path.exists(image_path):
            continue
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            continue
    return None



def _text_vcenter_y(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    row_y: int,
    row_h: int,
) -> int:
    try:
        _left, top, _right, bottom = draw.textbbox((0, 0), text, font=font)
        text_h = bottom - top
        return int(row_y + (row_h - text_h) / 2 - top)
    except Exception:
        text_h = draw.textsize(text, font=font)[1]
        return int(row_y + (row_h - text_h) / 2)


def _opponent_team_name(opponent: str) -> str:
    opponent = str(opponent or "").strip()
    if not opponent:
        return ""
    if opponent in {"—", "TBD"}:
        return opponent
    return opponent.split()[-1]


def _format_schedule_line_time(date_text: str, time_text: str) -> str:
    date_part = _format_game_date(date_text)
    time_part = str(time_text or "").strip()
    if not date_part:
        return time_part
    if not time_part or time_part in {"—", "TBD"}:
        return date_part
    return f"{date_part} {time_part}"


def _normalize_nfl_abbr(abbr: str) -> str:
    normalized = str(abbr or "").strip().upper()
    if normalized == "WSH":
        return "WAS"
    return normalized


def _schedule_opponent_abbr(game: dict) -> str:
    opponent = str(game.get("opponent") or "").strip()
    if not opponent or opponent in {"—", "TBD"}:
        return ""
    team_key = opponent.split()[-1].lower()
    return _normalize_nfl_abbr(NFL_TEAM_ABBREVIATIONS.get(team_key, team_key[:3]))


def _scoreboard_competitor_abbr(competitor: dict) -> str:
    team = (competitor or {}).get("team") or {}
    for key in ("abbreviation", "abbrev"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_nfl_abbr(value)
    return ""


def _scoreboard_score_value(competitor: dict) -> int | None:
    score = (competitor or {}).get("score")
    if isinstance(score, bool):
        return None
    if isinstance(score, int):
        return score
    if isinstance(score, float):
        return int(score)
    if isinstance(score, str):
        cleaned = score.strip()
        if not cleaned:
            return None
        try:
            return int(float(cleaned))
        except ValueError:
            return None
    return None


def _scoreboard_game_is_final(game: dict) -> bool:
    status = (game or {}).get("status") or {}
    type_info = status.get("type") or {}
    state = str(type_info.get("state") or "").strip().lower()
    if state == "post":
        return True
    if type_info.get("completed") is True:
        return True
    description = str(type_info.get("description") or "").strip().lower()
    return "final" in description


def _fetch_bears_scoreboard_games_for_date(day: datetime.date) -> list[dict[str, Any]]:
    now = time.monotonic()
    cached = _BEAR_SCORE_CACHE.get(day)
    if cached and (now - cached[0]) < BEARS_SCOREBOARD_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        from screens.nfl_scoreboard import _fetch_games_for_date

        games = _fetch_games_for_date(day)
    except Exception as exc:
        logging.error("Failed to fetch Bears scores from NFL scoreboard feed for %s: %s", day, exc)
        games = []

    _BEAR_SCORE_CACHE[day] = (now, games)
    return games


def _scoreboard_scores_for_bears_game(game: dict) -> tuple[int | None, int | None] | None:
    if _is_bye_week(game):
        return None

    parsed_date = _parse_schedule_date(str(game.get("date") or ""))
    if parsed_date is None:
        return None
    today = datetime.datetime.now(config.CENTRAL_TIME).date()
    if parsed_date.date() > today:
        return None

    expected_opponent = _schedule_opponent_abbr(game)
    if not expected_opponent:
        return None

    expected_home_away = str(game.get("home_away") or "").strip().lower()
    for scoreboard_game in _fetch_bears_scoreboard_games_for_date(parsed_date.date()):
        if not _scoreboard_game_is_final(scoreboard_game):
            continue
        competitors = (scoreboard_game or {}).get("competitors") or []
        bears = next((c for c in competitors if _scoreboard_competitor_abbr(c) == BEARS_TEAM_ABBR), None)
        opponent = next((c for c in competitors if _scoreboard_competitor_abbr(c) == expected_opponent), None)
        if not bears or not opponent:
            continue
        bears_home_away = str(bears.get("homeAway") or "").strip().lower()
        if expected_home_away in {"home", "away"} and bears_home_away != expected_home_away:
            continue
        bears_score = _scoreboard_score_value(bears)
        opponent_score = _scoreboard_score_value(opponent)
        if bears_score is None or opponent_score is None:
            continue
        if expected_home_away == "away":
            return opponent_score, bears_score
        return bears_score, opponent_score
    return None


def _bears_schedule_score_text(game: dict) -> str:
    configured_score = str(game.get("final_score") or "").strip()
    if configured_score:
        return configured_score

    scores = _scoreboard_scores_for_bears_game(game)
    if scores is not None:
        home_score, away_score = scores
    else:
        home_score = game.get("home_score")
        away_score = game.get("away_score")
    if home_score is None or away_score is None:
        return ""

    ha = str(game.get("home_away") or "").strip().lower()
    # Keep score perspective consistent with row labeling: each row is keyed by
    # opponent ("vs." or "@" + opponent), so render opponent score first.
    if ha == "away":
        opp_score, bears_score = home_score, away_score
    else:
        opp_score, bears_score = away_score, home_score
    return f"F {opp_score}-{bears_score}"


def _is_postseason_week(week: str) -> bool:
    week_lower = str(week or "").strip().lower()
    return any(token in week_lower for token in ("wild", "divisional", "conference", "championship", "super bowl", "playoff"))


def _is_bye_week(game: dict) -> bool:
    date_text = str(game.get("date") or "").strip().upper()
    opponent = str(game.get("opponent") or "").strip().upper()
    return date_text == "BYE" or opponent == "BYE"


def _should_show_bears_schedule_game(game: dict, today: datetime.date | None = None) -> bool:
    opponent = str(game.get("opponent") or "").strip()
    week = str(game.get("week") or game.get("game_no") or "").strip()
    week_lower = week.lower()

    if _is_bye_week(game):
        return bool(re.match(r"^week\s*\d+\b", week_lower))

    if not opponent or opponent in {"—", "TBD"}:
        return False

    if week_lower.startswith("preseason") or re.match(r"^week\s*\d+\b", week_lower):
        return True

    if _is_postseason_week(week_lower):
        game_date = _parse_schedule_date(str(game.get("date") or ""))
        if game_date is None:
            return False
        if today is None:
            today = datetime.date.today()
        return game_date.date() >= today

    return False


def _format_bears_schedule_week_label(week: str) -> str:
    week_txt = str(week or "").strip()
    preseason = re.search(r"(?i)preseason\s*(\d+)?", week_txt)
    if preseason:
        return f"P{preseason.group(1)}" if preseason.group(1) else "Pre"

    regular = re.search(r"(?i)^week\s*(\d+)\b", week_txt)
    if regular:
        return f"W{regular.group(1)}"

    if week_txt and not week_txt.lower().startswith("w"):
        return week_txt
    return week_txt


def _clone_font_at_size(font: ImageFont.ImageFont, size: int) -> ImageFont.ImageFont:
    path = getattr(font, "path", None)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return font


def _fit_font_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    base_font: ImageFont.ImageFont,
    max_width: int,
    *,
    min_size: int = 12,
) -> ImageFont.ImageFont:
    size = int(getattr(base_font, "size", min_size) or min_size)
    while size > min_size and draw.textsize(text, font=_clone_font_at_size(base_font, size))[0] > max_width:
        size -= 1
    return _clone_font_at_size(base_font, max(min_size, size))


def show_bears_next_season_sched(display, transition=False):
    background = get_screen_background_color("bears next season sched", (0, 0, 0))
    img = Image.new("RGB", (config.WIDTH, config.HEIGHT), background)
    draw = ImageDraw.Draw(img)

    title = "Bears Schedule"
    title_font = config.FONT_TITLE_SPORTS
    row_font = config.FONT_DATE_SPORTS
    date_font = config.FONT_WEATHER_DETAILS_TINY
    header_h = draw.textsize(title, font=title_font)[1]
    draw.text(((config.WIDTH - draw.textsize(title, font=title_font)[0]) // 2, 2), title, font=title_font, fill=(255, 255, 255))

    rows_top = header_h + 12
    base_row_h = max(14, draw.textsize("Ag", font=row_font)[1] + 6)
    row_h = base_row_h

    schedule_rows = []
    for game in BEARS_SCHEDULE:
        if not _should_show_bears_schedule_game(game):
            continue
        opponent = str(game.get("opponent") or "").strip()
        ha = str(game.get("home_away") or "").strip().lower()
        is_bye = _is_bye_week(game)
        prefix = "" if is_bye else ("@" if ha == "away" else "vs.")
        team_key = opponent.split()[-1].lower() if opponent else ""
        abbr = "" if is_bye else NFL_TEAM_ABBREVIATIONS.get(team_key, team_key[:3])
        if abbr == "was":
            abbr = "wsh"
        when = (
            ""
            if is_bye
            else _format_schedule_line_time(str(game.get("date") or ""), str(game.get("time") or ""))
        )
        score = _bears_schedule_score_text(game)
        week = str(game.get("week") or game.get("game_no") or "").strip()
        team_name = "BYE" if is_bye else _opponent_team_name(opponent)
        schedule_rows.append(
            {
                "week": week,
                "prefix": prefix,
                "abbr": abbr,
                "opponent": team_name,
                "when": when,
                "score": score,
            }
        )

    if not schedule_rows:
        draw.text((4, rows_top), "No scheduled games.", font=row_font, fill=(255, 255, 255))
    else:
        logo_size = max(14, int((base_row_h - 3) * 1.55))
        row_h = max(base_row_h, logo_size + 4)
        content_h = rows_top + (row_h * len(schedule_rows)) + 2
        full_h = max(config.HEIGHT, content_h)
        full_img = Image.new("RGB", (config.WIDTH, full_h), background)
        full_draw = ImageDraw.Draw(full_img)
        full_draw.text(((config.WIDTH - draw.textsize(title, font=title_font)[0]) // 2, 2), title, font=title_font, fill=(255, 255, 255))
        week_w = max(draw.textsize(sample, font=date_font)[0] for sample in ("W18", "P3"))
        logo_col_w = max(logo_size, *(
            (
                _cached_team_logo(row["abbr"], logo_size).width
                if row["abbr"] and _cached_team_logo(row["abbr"], logo_size)
                else logo_size
            )
            for row in schedule_rows
        ))
        x_week = 2
        x_prefix = x_week + week_w + 3
        x_logo = x_prefix + full_draw.textsize("vs.", font=date_font)[0] + 2
        x_name = x_logo + logo_col_w + 3
        for idx, row in enumerate(schedule_rows):
            y = rows_top + idx * row_h
            week_disp = _format_bears_schedule_week_label(row["week"])
            if week_disp:
                full_draw.text(
                    (x_week, _text_vcenter_y(full_draw, week_disp, date_font, y, row_h)),
                    week_disp,
                    font=date_font,
                    fill=(160, 180, 220),
                )

            full_draw.text(
                (x_prefix, _text_vcenter_y(full_draw, row["prefix"], date_font, y, row_h)),
                row["prefix"],
                font=date_font,
                fill=(255, 255, 255),
            )
            logo = _cached_team_logo(row["abbr"], logo_size) if row["abbr"] else None
            if logo:
                lx = x_logo + max(0, (logo_col_w - logo.width) // 2)
                ly = y + (row_h - logo.height) // 2
                full_img.paste(logo, (lx, ly), logo)
            right_text = row["when"]
            if row["score"]:
                right_text = f"{right_text} {row['score']}".strip()
            right_w = full_draw.textsize(right_text, font=date_font)[0]
            max_name_w = max(20, config.WIDTH - x_name - right_w - 3)
            name = row["opponent"]
            name_font = _fit_font_to_width(full_draw, name, row_font, max_name_w)
            full_draw.text(
                (x_name, _text_vcenter_y(full_draw, name, name_font, y, row_h)),
                name,
                font=name_font,
                fill=(255, 255, 255),
            )
            rw = full_draw.textsize(right_text, font=date_font)[0]
            full_draw.text(
                (config.WIDTH - rw - 2, _text_vcenter_y(full_draw, right_text, date_font, y, row_h)),
                right_text,
                font=date_font,
                fill=(180, 220, 255),
            )
        img = full_img

    if transition:
        scroll_vertical_content(
            display=display,
            content_height=img.height,
            viewport_width=config.WIDTH,
            viewport_height=config.HEIGHT,
            render_at_offset=lambda offset: display.image(img.crop((0, offset, config.WIDTH, offset + config.HEIGHT))),
            base_step=SCOREBOARD_SCROLL_STEP,
            pause_start=SCOREBOARD_SCROLL_PAUSE_TOP,
            pause_end=SCOREBOARD_SCROLL_PAUSE_BOTTOM,
            min_frame_time=SCOREBOARD_SCROLL_DELAY,
        )
        return ScreenImage(img, displayed=True)
    scroll_vertical_content(
        display=display,
        content_height=img.height,
        viewport_width=config.WIDTH,
        viewport_height=config.HEIGHT,
        render_at_offset=lambda offset: display.image(img.crop((0, offset, config.WIDTH, offset + config.HEIGHT))),
        base_step=SCOREBOARD_SCROLL_STEP,
        pause_start=SCOREBOARD_SCROLL_PAUSE_TOP,
        pause_end=SCOREBOARD_SCROLL_PAUSE_BOTTOM,
        min_frame_time=SCOREBOARD_SCROLL_DELAY,
    )
    return ScreenImage(img, displayed=True)


def show_bears_next_season(display, transition=False):
    background = get_screen_background_color("bears next season", (0, 0, 0))
    static_img = _cached_bears_next_season_static_image(config.WIDTH, config.HEIGHT)

    if static_img is not None:
        final_img = static_img.copy().resize((config.WIDTH, config.HEIGHT), Image.LANCZOS)
    else:
        # Fallback to dynamic rendering if no static image assets are available.
        background_key = tuple(background)
        final_img = render_bears_next_season_image(
            config.WIDTH,
            config.HEIGHT,
            background_key,
        )

    if transition:
        return final_img

    display.image(final_img)
    display.show()
    return None
