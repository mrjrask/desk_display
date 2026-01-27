#!/usr/bin/env python3
"""
nfl_scoreboard.py

Render a scrolling NFL scoreboard mirroring the layout of the MLB board.
Shows all games in the active NFL week (Thursday through Monday), including
playoff weeks that can span January and February. During January and February,
final scores persist until the playoff-aware cutoff time before advancing to
the next week.
"""

from __future__ import annotations

import datetime
import logging
import os
import time
from typing import Iterable, Optional

import requests
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
    SCOREBOARD_BACKGROUND_COLOR,
    SCOREBOARD_IN_PROGRESS_SCORE_COLOR,
    SCOREBOARD_FINAL_WINNING_SCORE_COLOR,
    SCOREBOARD_FINAL_LOSING_SCORE_COLOR,
    get_screen_background_color,
    get_screen_font,
    get_screen_image_scale,
)
from utils import (
    ScreenImage,
    clear_display,
    load_team_logo,
    log_call,
    standard_scoreboard_league_logo_height,
    standard_scoreboard_team_logo_height,
)

# ─── Constants ────────────────────────────────────────────────────────────────
TITLE               = "NFL Scoreboard"
TITLE_GAP           = 8
BLOCK_SPACING       = 10
SCORE_ROW_H         = 56
STATUS_ROW_H        = 18
REQUEST_TIMEOUT     = 10
SUPER_BOWL_LOGO_GAP = 6
SUPER_BOWL_DATE     = (2, 8)  # Feb 8

COL_WIDTHS = [70, 60, 60, 60, 70]
_TOTAL_COL_WIDTH = sum(COL_WIDTHS)
_COL_LEFT = max(0, (WIDTH - _TOTAL_COL_WIDTH) // 2)
COL_X = [_COL_LEFT]
for w in COL_WIDTHS:
    COL_X.append(COL_X[-1] + w)

SCREEN_ID = "NFL Scoreboard"
TITLE_FONT = FONT_TITLE_SPORTS
LOGO_DIR = os.path.join(IMAGES_DIR, "nfl")
LEAGUE_LOGO_KEYS = ("NFL", "nfl")
LEAGUE_LOGO_GAP = 4
TEAM_LOGO_BASE_HEIGHT = standard_scoreboard_team_logo_height(HEIGHT)
LEAGUE_LOGO_BASE_HEIGHT = standard_scoreboard_league_logo_height(TEAM_LOGO_BASE_HEIGHT)
LOGO_HEIGHT = TEAM_LOGO_BASE_HEIGHT
LEAGUE_LOGO_HEIGHT = LEAGUE_LOGO_BASE_HEIGHT
SCORE_FONT = get_screen_font(
    SCREEN_ID,
    "score",
    base_font=FONT_TEAM_SPORTS,
    default_size=39,
)
STATUS_FONT = get_screen_font(
    SCREEN_ID,
    "status",
    base_font=FONT_STATUS,
    default_size=28,
)
CENTER_FONT = get_screen_font(
    SCREEN_ID,
    "center",
    base_font=FONT_STATUS,
    default_size=28,
)
IN_PROGRESS_SCORE_COLOR = SCOREBOARD_IN_PROGRESS_SCORE_COLOR
IN_PROGRESS_STATUS_COLOR = IN_PROGRESS_SCORE_COLOR
FINAL_WINNING_SCORE_COLOR = SCOREBOARD_FINAL_WINNING_SCORE_COLOR
FINAL_LOSING_SCORE_COLOR = SCOREBOARD_FINAL_LOSING_SCORE_COLOR
BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)

IN_GAME_STATUS_OVERRIDES = {
    "end of the 1st": "End of the 1st",
    "end of 1st": "End of the 1st",
    "halftime": "Halftime",
    "end of the 3rd": "End of the 3rd",
    "end of 3rd": "End of the 3rd",
}

_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}
_LEAGUE_LOGO_CACHE: dict[int, Optional[Image.Image]] = {}
_SUPER_BOWL_LOGO_CACHE: dict[int, Optional[Image.Image]] = {}


def _apply_style_overrides() -> None:
    global SCORE_FONT, STATUS_FONT, CENTER_FONT, LOGO_HEIGHT, LEAGUE_LOGO_HEIGHT, BACKGROUND_COLOR

    SCORE_FONT = get_screen_font(
        SCREEN_ID,
        "score",
        base_font=FONT_TEAM_SPORTS,
        default_size=39,
    )
    STATUS_FONT = get_screen_font(
        SCREEN_ID,
        "status",
        base_font=FONT_STATUS,
        default_size=28,
    )
    CENTER_FONT = get_screen_font(
        SCREEN_ID,
        "center",
        base_font=FONT_STATUS,
        default_size=28,
    )
    BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)
    team_scale = get_screen_image_scale(SCREEN_ID, "team_logo", 1.0)
    LOGO_HEIGHT = max(1, int(round(TEAM_LOGO_BASE_HEIGHT * team_scale)))
    league_scale = get_screen_image_scale(SCREEN_ID, "league_logo", team_scale)
    LEAGUE_LOGO_HEIGHT = max(1, int(round(LEAGUE_LOGO_BASE_HEIGHT * league_scale)))


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _week_start_for_date(day: datetime.date) -> datetime.date:
    days_since_thursday = (day.weekday() - 3) % 7
    return day - datetime.timedelta(days=days_since_thursday)


def _week_dates_from_start(start: datetime.date) -> list[datetime.date]:
    return [start + datetime.timedelta(days=offset) for offset in range(5)]


def _playoff_rules_active(now: datetime.datetime) -> bool:
    return now.month in {1, 2}


def _regular_week_start(now: datetime.datetime) -> datetime.date:
    if now.weekday() == 2:  # Wednesday
        cutoff = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if now >= cutoff:
            ref_date = now.date() + datetime.timedelta(days=1)
        else:
            ref_date = now.date()
    else:
        ref_date = now.date()
    return _week_start_for_date(ref_date)


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


def _get_league_logo() -> Optional[Image.Image]:
    height = LEAGUE_LOGO_HEIGHT
    if height in _LEAGUE_LOGO_CACHE:
        return _LEAGUE_LOGO_CACHE[height]
    for key in LEAGUE_LOGO_KEYS:
        logo = load_team_logo(LOGO_DIR, key, height=height, box_size=height)
        if logo is not None:
            _LEAGUE_LOGO_CACHE[height] = logo
            return logo
    _LEAGUE_LOGO_CACHE[height] = None
    return None


def _get_super_bowl_logo() -> Optional[Image.Image]:
    height = LOGO_HEIGHT
    if height in _SUPER_BOWL_LOGO_CACHE:
        return _SUPER_BOWL_LOGO_CACHE[height]
    logo = load_team_logo(LOGO_DIR, "SB", height=height, box_size=height)
    _SUPER_BOWL_LOGO_CACHE[height] = logo
    return logo


def _team_logo_abbr(team: dict) -> str:
    if not isinstance(team, dict):
        return ""
    for key in ("abbreviation", "abbrev", "shortDisplayName", "displayName"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip().upper()
            for suffix in (candidate, candidate.lower()):
                if os.path.exists(os.path.join(LOGO_DIR, f"{suffix}.png")):
                    return candidate
    nickname = (team.get("nickname") or team.get("name") or "").strip()
    return nickname[:3].upper() if nickname else ""


def _should_display_scores(game: dict) -> bool:
    status = (game or {}).get("status", {}) or {}
    type_info = status.get("type") or {}
    state = (type_info.get("state") or "").lower()
    if state in {"in", "post"}:
        return True
    if (type_info.get("completed") or False) is True:
        return True
    return False


def _is_game_in_progress(game: dict) -> bool:
    status = (game or {}).get("status", {}) or {}
    type_info = status.get("type") or {}
    state = (type_info.get("state") or "").lower()
    return state == "in"


def _is_game_final(game: dict) -> bool:
    status = (game or {}).get("status", {}) or {}
    type_info = status.get("type") or {}
    state = (type_info.get("state") or "").lower()
    completed = type_info.get("completed")
    if state == "post":
        return True
    if isinstance(completed, bool) and completed:
        return True
    description = (type_info.get("description") or "").lower()
    if "final" in description:
        return True
    return False


def _score_text(side: dict, *, show: bool) -> str:
    if not show:
        return "—"
    score = (side or {}).get("score")
    return "—" if score is None else str(score)


def _score_value(side: dict) -> Optional[int]:
    score = (side or {}).get("score")
    if isinstance(score, (int, float)):
        return int(score)
    if isinstance(score, str):
        cleaned = score.strip()
        if cleaned.isdigit():
            try:
                return int(cleaned)
            except Exception:
                return None
        try:
            return int(float(cleaned))
        except Exception:
            return None
    return None


def _team_result(side: dict, opponent: dict) -> Optional[str]:
    for key in ("isWinner", "winner", "won"):
        value = (side or {}).get(key)
        if isinstance(value, bool):
            return "win" if value else "loss"

    side_score = _score_value(side)
    opp_score = _score_value(opponent)
    if side_score is not None and opp_score is not None:
        if side_score > opp_score:
            return "win"
        if side_score < opp_score:
            return "loss"
    return None


def _final_results(away: dict, home: dict) -> dict:
    away_result = _team_result(away, home)
    home_result = _team_result(home, away)

    if away_result == "win":
        home_result = "loss"
    elif away_result == "loss":
        home_result = "win"
    elif home_result == "win":
        away_result = "loss"
    elif home_result == "loss":
        away_result = "win"

    return {"away": away_result, "home": home_result}
def _score_fill(team_key: str, *, in_progress: bool, final: bool, results: dict) -> tuple[int, int, int]:
    if in_progress:
        return IN_PROGRESS_SCORE_COLOR
    if final:
        result = results.get(team_key)
        if result == "loss":
            return FINAL_LOSING_SCORE_COLOR
        if result == "win":
            return FINAL_WINNING_SCORE_COLOR
    return (255, 255, 255)


def _format_status(game: dict) -> str:
    status = (game or {}).get("status", {}) or {}
    type_info = status.get("type") or {}
    short_detail = (type_info.get("shortDetail") or "").strip()
    detail = (type_info.get("detail") or "").strip()
    state = (type_info.get("state") or "").lower()
    short_detail_lower = short_detail.lower()
    detail_lower = detail.lower()

    def _override_in_game_status() -> Optional[str]:
        for candidate in (short_detail, detail):
            normalized = (candidate or "").strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in IN_GAME_STATUS_OVERRIDES:
                return IN_GAME_STATUS_OVERRIDES[key]
        return None

    if state == "postponed" or "postponed" in short_detail_lower or "postponed" in detail_lower:
        return "Postponed"
    if state == "post":
        return short_detail or detail or "Final"
    if state == "in":
        override = _override_in_game_status()
        if override:
            return override
        clock = status.get("displayClock") or ""
        period = status.get("period")
        if clock and period:
            return f"{clock} Q{period}"
        return short_detail or detail or "In Progress"

    if state == "pre":
        start_local = game.get("_start_local")
        if isinstance(start_local, datetime.datetime):
            time_text = start_local.strftime("%I:%M %p").lstrip("0")
            if start_local.weekday() != 6:  # Not Sunday
                day_text = start_local.strftime("%a")
                return f"{day_text} {time_text}"
            return time_text
        return short_detail or detail or "TBD"

    return short_detail or detail or "TBD"


def _center_text(draw: ImageDraw.ImageDraw, text: str, font, x: int, width: int,
                 y: int, height: int, *, fill=(255, 255, 255)):
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


def _draw_game_block(canvas: Image.Image, draw: ImageDraw.ImageDraw, game: dict, top: int):
    competitors = (game or {}).get("competitors", [])
    away = next((c for c in competitors if c.get("homeAway") == "away"), {})
    home = next((c for c in competitors if c.get("homeAway") == "home"), {})

    show_scores = _should_display_scores(game)
    away_text = _score_text(away, show=show_scores)
    home_text = _score_text(home, show=show_scores)
    in_progress = _is_game_in_progress(game)
    final = _is_game_final(game)
    results = _final_results(away, home) if final else {"away": None, "home": None}

    score_top = top
    for idx, text in ((0, away_text), (2, "@"), (4, home_text)):
        font = SCORE_FONT if idx != 2 else CENTER_FONT
        if idx == 0:
            fill = _score_fill("away", in_progress=in_progress, final=final, results=results)
        elif idx == 4:
            fill = _score_fill("home", in_progress=in_progress, final=final, results=results)
        else:
            fill = (255, 255, 255)
        _center_text(draw, text, font, COL_X[idx], COL_WIDTHS[idx], score_top, SCORE_ROW_H, fill=fill)

    for idx, team_side, team_key in ((1, away, "away"), (3, home, "home")):
        team_obj = (team_side or {}).get("team", {})
        abbr = _team_logo_abbr(team_obj)
        logo = _load_logo_cached(abbr)
        if not logo:
            continue
        x0 = COL_X[idx] + (COL_WIDTHS[idx] - logo.width) // 2
        y0 = score_top + (SCORE_ROW_H - logo.height) // 2
        canvas.paste(logo, (x0, y0), logo)

    status_top = score_top + SCORE_ROW_H
    status_text = _format_status(game)
    status_fill = IN_PROGRESS_STATUS_COLOR if in_progress else (255, 255, 255)
    _center_text(draw, status_text, STATUS_FONT, COL_X[2], COL_WIDTHS[2], status_top, STATUS_ROW_H, fill=status_fill)


def _compose_canvas(games: list[dict], *, show_super_bowl_logo: bool) -> Image.Image:
    if not games:
        return Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    block_height = SCORE_ROW_H + STATUS_ROW_H
    total_height = block_height * len(games)
    if len(games) > 1:
        total_height += BLOCK_SPACING * (len(games) - 1)
    super_bowl_logo = _get_super_bowl_logo() if show_super_bowl_logo else None
    if super_bowl_logo:
        total_height += SUPER_BOWL_LOGO_GAP + super_bowl_logo.height
    canvas = Image.new("RGB", (WIDTH, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    y = 0
    for idx, game in enumerate(games):
        _draw_game_block(canvas, draw, game, y)
        y += SCORE_ROW_H + STATUS_ROW_H
        if idx < len(games) - 1:
            sep_y = y + BLOCK_SPACING // 2
            draw.line((10, sep_y, WIDTH - 10, sep_y), fill=(45, 45, 45))
            y += BLOCK_SPACING
    if super_bowl_logo:
        y += SUPER_BOWL_LOGO_GAP
        logo_x = (WIDTH - super_bowl_logo.width) // 2
        canvas.paste(super_bowl_logo, (logo_x, y), super_bowl_logo)
    return canvas


def _timestamp_to_local(ts: str) -> Optional[datetime.datetime]:
    if not ts:
        return None
    try:
        dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%MZ")
    except ValueError:
        try:
            dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None
    dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(CENTRAL_TIME)


def _game_sort_key(game: dict):
    return (
        game.get("_start_sort", float("inf")),
        str(game.get("id") or game.get("uid") or ""),
    )


def _hydrate_games(raw_games: Iterable[dict]) -> list[dict]:
    games: list[dict] = []
    for game in raw_games:
        game = game or {}
        start_local = _timestamp_to_local(game.get("_event_date"))
        if start_local:
            game["_start_local"] = start_local
            game["_start_sort"] = start_local.timestamp()
        else:
            game["_start_sort"] = float("inf")
        games.append(game)
    games.sort(key=_game_sort_key)
    return games


def _is_super_bowl_game(game: dict) -> bool:
    if not isinstance(game, dict):
        return False
    for key in ("_event_name", "_event_short_name", "name", "shortName"):
        value = game.get(key)
        if isinstance(value, str) and "super bowl" in value.lower():
            return True
    start_local = game.get("_start_local")
    if isinstance(start_local, datetime.datetime):
        return (start_local.month, start_local.day) == SUPER_BOWL_DATE
    return False


def _fetch_games_for_date(day: datetime.date) -> list[dict]:
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        f"?dates={day.strftime('%Y%m%d')}"
    )
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logging.error("Failed to fetch NFL scoreboard: %s", exc)
        return []

    raw_games: list[dict] = []
    for event in data.get("events", []) or []:
        event_date = event.get("date")
        local_start = _timestamp_to_local(event_date)
        if local_start and local_start.date() != day:
            continue
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0] or {}
        comp = dict(comp)
        comp["_event_date"] = event_date
        comp["_event_name"] = event.get("name")
        comp["_event_short_name"] = event.get("shortName")
        raw_games.append(comp)
    return _hydrate_games(raw_games)


def _week_cutoff_datetime(week_start: datetime.date, game_count: int) -> datetime.datetime:
    def _localize(day: datetime.date, hour: int, minute: int) -> datetime.datetime:
        naive = datetime.datetime.combine(day, datetime.time(hour=hour, minute=minute))
        try:
            return CENTRAL_TIME.localize(naive)  # type: ignore[attr-defined]
        except Exception:
            return naive.replace(tzinfo=CENTRAL_TIME)

    if game_count == 1:
        super_bowl_date = datetime.date(week_start.year, *SUPER_BOWL_DATE)
        week_end = week_start + datetime.timedelta(days=6)
        if week_start <= super_bowl_date <= week_end:
            return _localize(super_bowl_date, 23, 59)
    if game_count == 6:
        return _localize(week_start + datetime.timedelta(days=5), 15, 0)
    if game_count in {2, 4}:
        return _localize(week_start + datetime.timedelta(days=4), 15, 15)
    return _localize(week_start + datetime.timedelta(days=6), 9, 0)


def _fetch_games_for_week(now: Optional[datetime.datetime] = None) -> list[dict]:
    now = now or datetime.datetime.now(CENTRAL_TIME)
    if not _playoff_rules_active(now):
        week_start = _regular_week_start(now)
        games = []
        for day in _week_dates_from_start(week_start):
            games.extend(_fetch_games_for_date(day))
        games.sort(key=_game_sort_key)
        return games

    week_start = _week_start_for_date(now.date())
    games = []
    for day in _week_dates_from_start(week_start):
        games.extend(_fetch_games_for_date(day))
    games.sort(key=_game_sort_key)

    cutoff = _week_cutoff_datetime(week_start, len(games))
    if now >= cutoff:
        week_start = week_start + datetime.timedelta(days=7)
        games = []
        for day in _week_dates_from_start(week_start):
            games.extend(_fetch_games_for_date(day))
        games.sort(key=_game_sort_key)
    return games


def _fetch_next_games(
    start_date: datetime.date,
    *,
    max_days: int = 370,
) -> list[dict]:
    for offset in range(max_days + 1):
        day = start_date + datetime.timedelta(days=offset)
        games = _fetch_games_for_date(day)
        if games:
            return games
    logging.warning("NFL scoreboard could not find upcoming games after %s", start_date)
    return []


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
    img_height = max(HEIGHT, content_top + canvas.height)
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
    if full_img.height <= HEIGHT:
        display.image(full_img)
        return

    wait_for_skip = getattr(display, "wait_for_skip", None)
    skip_requested = getattr(display, "skip_requested", None)

    def _should_skip() -> bool:
        return bool(skip_requested and skip_requested())

    def _sleep(duration: float) -> bool:
        if callable(wait_for_skip):
            return bool(wait_for_skip(duration))
        time.sleep(duration)
        return False

    max_offset = full_img.height - HEIGHT
    frame = full_img.crop((0, 0, WIDTH, HEIGHT))
    display.image(frame)
    if _sleep(SCOREBOARD_SCROLL_PAUSE_TOP):
        return

    target_frame_time = 0.016  # ~60 FPS for smoother scrolling
    for offset in range(
        SCOREBOARD_SCROLL_STEP, max_offset + 1, SCOREBOARD_SCROLL_STEP
    ):
        if _should_skip():
            return

        frame_start = time.time()

        frame = full_img.crop((0, offset, WIDTH, offset + HEIGHT))
        display.image(frame)

        # Account for rendering time to maintain consistent frame rate
        elapsed = time.time() - frame_start
        sleep_time = max(0, target_frame_time - elapsed)
        if sleep_time > 0 and _sleep(sleep_time):
            return

    _sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)


# ─── Public API ───────────────────────────────────────────────────────────────
@log_call
def draw_nfl_scoreboard(display, transition: bool = False) -> ScreenImage:
    _apply_style_overrides()
    now = datetime.datetime.now(CENTRAL_TIME)
    games = _fetch_games_for_week(now)
    if not games:
        games = _fetch_next_games(now.date())
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
        _center_text(draw, "No games", STATUS_FONT, 0, WIDTH, HEIGHT // 2 - STATUS_ROW_H // 2, STATUS_ROW_H)
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


if __name__ == "__main__":  # pragma: no cover
    from utils import Display

    disp = Display()
    try:
        draw_nfl_scoreboard(disp)
    finally:
        clear_display(disp)
