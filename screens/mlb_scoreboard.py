#!/usr/bin/env python3
"""
mlb_scoreboard.py

Render a scrolling MLB scoreboard showing that day's games.
For temporary testing, the scoreboard can be pinned to a fixed date. Outside
that override window, the previous day's scores are retained until 10:10 AM
Central before refreshing to the current date.
Layout:
    • Title "MLB Scoreboard" centered at the top.
    • Each game occupies two rows arranged in five conceptual columns:
        [Away Score] [Away Logo] [ @ ] [Home Logo] [Home Score]
      The second row centers the status/time text in the middle column.
    • When the combined height exceeds the display area, the list scrolls top → bottom.
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import time
from typing import Iterable, Optional

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
    is_hyperpixel_next_layout,
    scale_value,
    scale_value_width,
)
from services.http_client import get_session
from utils import (
    ScreenImage,
    clear_display,
    get_mlb_abbreviation,
    get_mlb_tricode,
    load_team_logo,
    log_call,
    log_missing_team_logo,
    scroll_vertical_content,
    standard_scoreboard_team_logo_height,
)

# ─── Constants ────────────────────────────────────────────────────────────────
HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()


def _scale_width(value: int) -> int:
    return scale_value_width(value) if HYPERPIXEL_LAYOUT else scale_value(value)


TITLE                 = "MLB Scoreboard"
TITLE_GAP             = scale_value(8)
BLOCK_SPACING         = scale_value(10)
SCORE_ROW_H           = scale_value(56)
STATUS_ROW_H          = scale_value(18)
REQUEST_TIMEOUT       = 10
FETCH_CACHE_TTL_SECONDS = 60
MLB_SPORT_IDS = "1,51"  # MLB + World Baseball Classic
COL_WIDTHS = [
    _scale_width(80),
    _scale_width(60),
    _scale_width(40),
    _scale_width(60),
    _scale_width(80),
]  # total = 320 (WIDTH)
_TOTAL_COL_WIDTH = sum(COL_WIDTHS)
_COL_LEFT = max(0, (WIDTH - _TOTAL_COL_WIDTH) // 2)
COL_X = [_COL_LEFT]
for w in COL_WIDTHS:
    COL_X.append(COL_X[-1] + w)

SCREEN_ID = "MLB Scoreboard"
TITLE_FONT = FONT_TITLE_SPORTS
LOGO_DIR = os.path.join(IMAGES_DIR, "mlb")
LEAGUE_LOGO_KEYS = ("MLB", "mlb")
LEAGUE_LOGO_GAP = scale_value(4)
TEAM_LOGO_BASE_HEIGHT = standard_scoreboard_team_logo_height(HEIGHT)
IN_PROGRESS_SCORE_COLOR = SCOREBOARD_IN_PROGRESS_SCORE_COLOR
IN_PROGRESS_STATUS_COLOR = IN_PROGRESS_SCORE_COLOR
FINAL_WINNING_SCORE_COLOR = SCOREBOARD_FINAL_WINNING_SCORE_COLOR
FINAL_LOSING_SCORE_COLOR = SCOREBOARD_FINAL_LOSING_SCORE_COLOR
TEMP_SCHEDULE_OVERRIDE_DATE = datetime.date(2026, 2, 20)
TEMP_SCHEDULE_OVERRIDE_UNTIL = datetime.datetime(2026, 2, 20, 23, 59, 59, tzinfo=CENTRAL_TIME)


def _scoreboard_fonts():
    score = get_screen_font(
        SCREEN_ID,
        "score",
        base_font=FONT_TEAM_SPORTS,
        default_size=43,
    )
    status = get_screen_font(
        SCREEN_ID,
        "status",
        base_font=FONT_STATUS,
        default_size=28,
    )
    center = get_screen_font(
        SCREEN_ID,
        "center",
        base_font=FONT_STATUS,
        default_size=28,
    )
    return score, status, center


def _team_logo_height() -> int:
    scale = get_screen_image_scale(SCREEN_ID, "team_logo", 1.0)
    height = max(1, int(round(TEAM_LOGO_BASE_HEIGHT * scale)))
    # Keep logos constrained to the score row so tall marks never overlap
    # the status line or neighboring game blocks.
    max_row_fit = max(1, SCORE_ROW_H - scale_value(8))
    return min(height, max_row_fit)


def _league_logo_height() -> int:
    # Keep the league mark visually consistent with the scoreboard team logos.
    return _team_logo_height()
BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)

# Cache for resized logos { (abbr, height): Image }
_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}
_LEAGUE_LOGO_CACHE: dict[int, Optional[Image.Image]] = {}
_GAMES_CACHE: dict[tuple[datetime.date, str], tuple[float, list[dict]]] = {}
_SESSION = get_session()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _load_logo_cached(abbr: str) -> Optional[Image.Image]:
    key = (abbr or "").strip()
    if not key:
        return None
    cache_key_abbr = key.upper()
    height = _team_logo_height()
    cache_key = (cache_key_abbr, height)
    if cache_key in _LOGO_CACHE:
        return _LOGO_CACHE[cache_key]
    logo = load_team_logo(
        LOGO_DIR,
        cache_key_abbr,
        height=height,
        box_size=height,
        trim=True,
    )
    _LOGO_CACHE[cache_key] = logo
    return logo


def _get_league_logo() -> Optional[Image.Image]:
    height = _league_logo_height()
    if height in _LEAGUE_LOGO_CACHE:
        return _LEAGUE_LOGO_CACHE[height]
    for key in LEAGUE_LOGO_KEYS:
        logo = load_team_logo(LOGO_DIR, key, height=height)
        if logo is not None:
            _LEAGUE_LOGO_CACHE[height] = logo
            return logo
    _LEAGUE_LOGO_CACHE[height] = None
    return None


def _team_logo_abbr(team: dict) -> str:
    tricode = get_mlb_tricode(team)
    if not tricode:
        return ""
    return tricode.upper()


def _should_display_scores(game: dict) -> bool:
    """Return True when the game's status indicates the scores are real."""

    status = (game or {}).get("status", {}) or {}
    abstract = (status.get("abstractGameState") or "").lower()
    detailed = (status.get("detailedState") or "").lower()
    code = (status.get("statusCode") or "").upper()

    if abstract in {"final", "completed", "live"}:
        return True
    if code in {"F", "O", "I"}:  # Final, Over, In-progress
        return True
    if "progress" in detailed or "final" in detailed:
        return True
    return False


def _is_game_in_progress(game: dict) -> bool:
    status = (game or {}).get("status", {}) or {}
    abstract = (status.get("abstractGameState") or "").lower()
    if abstract == "live":
        return True
    code = (status.get("statusCode") or "").upper()
    if code == "I":
        return True
    detailed = (status.get("detailedState") or "").lower()
    if "progress" in detailed:
        return True
    return False


def _is_game_final(game: dict) -> bool:
    status = (game or {}).get("status", {}) or {}
    abstract = (status.get("abstractGameState") or "").lower()
    detailed = (status.get("detailedState") or "").lower()
    code = (status.get("statusCode") or "").upper()

    if abstract in {"final", "completed"}:
        return True
    if code in {"F", "O"}:
        return True
    if "final" in detailed:
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


def _final_inning(linescore: dict) -> Optional[int]:
    if not isinstance(linescore, dict):
        return None
    try:
        cur = int(linescore.get("currentInning"))
        if cur:
            return cur
    except Exception:
        pass
    innings = linescore.get("innings")
    if isinstance(innings, Iterable):
        for inning in reversed(list(innings)):
            try:
                num = int(inning.get("num"))
                if num:
                    return num
            except Exception:
                continue
    return None




def _strip_leading_zero_minutes(clock_text: str) -> str:
    if not isinstance(clock_text, str):
        return ""
    return re.sub(r"\b0(\d):([0-5]\d)\b", r"\1:\2", clock_text)


def _format_status(game: dict) -> str:
    status = (game or {}).get("status", {}) or {}
    linescore = (game or {}).get("linescore", {}) or {}
    abstract = (status.get("abstractGameState") or "").lower()
    detailed = status.get("detailedState") or ""
    code = (status.get("statusCode") or "").upper()

    detailed_lower = detailed.lower()

    if "postponed" in detailed_lower:
        return "Postponed"
    if "canceled" in detailed_lower or "cancelled" in detailed_lower:
        return "Canceled"
    normalized_detail = detailed_lower.replace("-", "")

    if "warmup" in normalized_detail:
        return "Warmup"
    if detailed_lower == "delayed":
        return "Delayed"
    if "suspended" in detailed_lower:
        return detailed

    if abstract in ("final", "completed") or code in {"F", "O"} or "final" in detailed_lower:
        innings = _final_inning(linescore)
        scheduled = linescore.get("scheduledInnings")
        if isinstance(innings, int):
            if isinstance(scheduled, int) and innings != scheduled:
                return f"Final/{innings}"
            if innings > 9:
                return f"Final/{innings}"
        return "Final"

    if abstract == "live" or code == "I" or "progress" in detailed_lower:
        inning_state = (linescore.get("inningState") or "").strip()
        inning_ord = (linescore.get("currentInningOrdinal") or "").strip()
        if inning_state and inning_ord:
            return f"{inning_state} {inning_ord}"
        if detailed:
            return _strip_leading_zero_minutes(detailed)
        return "In Progress"

    if "delay" in detailed_lower:
        return detailed

    start_local = game.get("_start_local")
    if isinstance(start_local, datetime.datetime):
        return start_local.strftime("%I:%M %p").lstrip("0")

    if detailed:
        return detailed
    return (status.get("status") or "TBD")


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


def _draw_game_block(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    game: dict,
    top: int,
    *,
    score_font,
    center_font,
    status_font,
):
    teams = (game or {}).get("teams", {})
    away = teams.get("away", {})
    home = teams.get("home", {})

    show_scores = _should_display_scores(game)
    away_text = _score_text(away, show=show_scores)
    home_text = _score_text(home, show=show_scores)
    in_progress = _is_game_in_progress(game)
    final = _is_game_final(game)
    results = _final_results(away, home) if final else {"away": None, "home": None}

    # Score row (5 columns)
    score_top = top
    for idx, text in ((0, away_text), (2, "@"), (4, home_text)):
        font = score_font if idx != 2 else center_font
        if idx == 0:
            fill = _score_fill("away", in_progress=in_progress, final=final, results=results)
        elif idx == 4:
            fill = _score_fill("home", in_progress=in_progress, final=final, results=results)
        else:
            fill = (255, 255, 255)
        _center_text(draw, text, font, COL_X[idx], COL_WIDTHS[idx], score_top, SCORE_ROW_H, fill=fill)

    # Logos
    for idx, team_side, team_key in ((1, away, "away"), (3, home, "home")):
        team_obj = (team_side or {}).get("team", {})
        abbr = _team_logo_abbr(team_obj)
        logo = _load_logo_cached(abbr) if abbr else None
        if not logo:
            team_name = (
                (team_obj or {}).get("name")
                or (team_obj or {}).get("teamName")
                or (team_obj or {}).get("displayName")
                or "Unknown Team"
            )
            log_missing_team_logo(SCREEN_ID, team_name, abbr)
            continue
        x0 = COL_X[idx] + (COL_WIDTHS[idx] - logo.width) // 2
        y0 = score_top + (SCORE_ROW_H - logo.height) // 2
        canvas.paste(logo, (x0, y0), logo)

    # Status row (center column text)
    status_top = score_top + SCORE_ROW_H
    status_text = _format_status(game)
    status_fill = IN_PROGRESS_STATUS_COLOR if in_progress else (255, 255, 255)
    _center_text(draw, status_text, status_font, COL_X[2], COL_WIDTHS[2], status_top, STATUS_ROW_H, fill=status_fill)


def _compose_canvas(
    games: list[dict], *, score_font, center_font, status_font
) -> Image.Image:
    if not games:
        return Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    block_height = SCORE_ROW_H + STATUS_ROW_H
    total_height = block_height * len(games)
    if len(games) > 1:
        total_height += BLOCK_SPACING * (len(games) - 1)
    canvas = Image.new("RGB", (WIDTH, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    y = 0
    for idx, game in enumerate(games):
        _draw_game_block(
            canvas,
            draw,
            game,
            y,
            score_font=score_font,
            center_font=center_font,
            status_font=status_font,
        )
        y += SCORE_ROW_H + STATUS_ROW_H
        if idx < len(games) - 1:
            # separator line and spacing
            sep_y = y + BLOCK_SPACING // 2
            draw.line((10, sep_y, WIDTH - 10, sep_y), fill=(45, 45, 45))
            y += BLOCK_SPACING
    return canvas


def _timestamp_to_local(ts: str) -> Optional[datetime.datetime]:
    if not ts:
        return None
    try:
        dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(CENTRAL_TIME)
    except Exception:
        return None


def _hydrate_games(raw_games: Iterable[dict]) -> list[dict]:
    games: list[dict] = []
    for game in raw_games:
        game = game or {}
        start_local = _timestamp_to_local(game.get("gameDate"))
        if start_local:
            game["_start_local"] = start_local
            game["_start_sort"] = start_local.timestamp()
        else:
            game["_start_sort"] = float("inf")
        games.append(game)
    games.sort(
        key=lambda g: (
            g.get("_start_sort", float("inf")),
            g.get("gamePk", 0),
        )
    )
    return games


def _scoreboard_date(now: Optional[datetime.datetime] = None) -> datetime.date:
    """Return the date whose games should be shown on the scoreboard."""

    now = now or datetime.datetime.now(CENTRAL_TIME)
    if now <= TEMP_SCHEDULE_OVERRIDE_UNTIL:
        return TEMP_SCHEDULE_OVERRIDE_DATE

    cutoff = now.replace(hour=10, minute=10, second=0, microsecond=0)
    if now < cutoff:
        return (now - datetime.timedelta(days=1)).date()
    return now.date()


def _fetch_games_for_date(day: datetime.date) -> list[dict]:
    cache_key = (day, "mlb_statsapi")
    now = time.monotonic()
    cached = _GAMES_CACHE.get(cache_key)
    if cached and (now - cached[0]) < FETCH_CACHE_TTL_SECONDS:
        return cached[1]

    # Explicitly request spring training + regular season + postseason game
    # types so the scoreboard still works during preseason and playoffs.
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId={MLB_SPORT_IDS}&date={day.isoformat()}&hydrate=team,linescore"
        "&gameTypes=S,E,R,F,D,L,W"
    )
    try:
        response = _SESSION.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logging.error("Failed to fetch MLB schedule: %s", exc)
        return []

    raw_games: list[dict] = []
    for day in data.get("dates", []):
        raw_games.extend(day.get("games", []) or [])
    games = _hydrate_games(raw_games)
    _GAMES_CACHE[cache_key] = (now, games)
    return games


def _render_scoreboard(
    games: list[dict], *, score_font, center_font, status_font
) -> Image.Image:
    canvas = _compose_canvas(
        games,
        score_font=score_font,
        center_font=center_font,
        status_font=status_font,
    )

    # Measure title height on a throwaway canvas to size the final image precisely.
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

    # Title (recompute placement on the real canvas)
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


# ─── Public API ───────────────────────────────────────────────────────────────
def render_mlb_scoreboard(display, games: list[dict] | None, transition: bool = False) -> ScreenImage:
    from screens.mlb_scoreboard_v2 import (
        MIN_GAMES_FOR_V2_LAYOUT,
        V2_DISABLED_RESOLUTIONS,
        render_mlb_scoreboard_v2,
    )

    if (
        games is not None
        and (WIDTH, HEIGHT) not in V2_DISABLED_RESOLUTIONS
        and len(games) >= MIN_GAMES_FOR_V2_LAYOUT
    ):
        return render_mlb_scoreboard_v2(display, games, transition=transition)

    global BACKGROUND_COLOR
    BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)
    score_font, status_font, center_font = _scoreboard_fonts()

    if games is None:
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
            draw,
            "Loading games...",
            status_font,
            0,
            WIDTH,
            HEIGHT // 2 - STATUS_ROW_H // 2,
            STATUS_ROW_H,
        )
        if transition:
            return ScreenImage(img, displayed=False)
        display.image(img)
        return ScreenImage(img, displayed=True)

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
            draw,
            "No games today",
            status_font,
            0,
            WIDTH,
            HEIGHT // 2 - STATUS_ROW_H // 2,
            STATUS_ROW_H,
        )
        if transition:
            return ScreenImage(img, displayed=False)
        display.image(img)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
        return ScreenImage(img, displayed=True)

    full_img = _render_scoreboard(
        games,
        score_font=score_font,
        center_font=center_font,
        status_font=status_font,
    )
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
def draw_mlb_scoreboard(display, transition: bool = False) -> ScreenImage:
    from services.sports.mlb import fetch_scoreboard

    games = fetch_scoreboard()
    return render_mlb_scoreboard(display, games, transition=transition)


if __name__ == "__main__":  # pragma: no cover
    from utils import Display

    disp = Display()
    try:
        draw_mlb_scoreboard(disp)
    finally:
        clear_display(disp)
