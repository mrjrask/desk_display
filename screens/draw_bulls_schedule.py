#!/usr/bin/env python3
"""draw_bulls_schedule.py

Chicago Bulls schedule screens mirroring the Blackhawks layout: last game,
live game, next game, and next home game cards with NBA logos.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from config import (
    FONT_DATE_SPORTS,
    FONT_TEAM_SPORTS,
    FONT_TITLE_SPORTS,
    NBA_IMAGES_DIR,
    NBA_TEAM_ID,
    NBA_TEAM_TRICODE,
    TIMES_SQUARE_FONT_PATH,
    WIDTH,
    HEIGHT,
    CENTRAL_TIME,
)

from utils import clear_display, load_team_logo

TS_PATH = TIMES_SQUARE_FONT_PATH
NBA_DIR = NBA_IMAGES_DIR
BULLS_TEAM_ID = str(NBA_TEAM_ID)
BULLS_TRICODE = (NBA_TEAM_TRICODE or "CHI").upper()


def _ts(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(TS_PATH, size)
    except Exception:
        logging.warning("TimesSquare font missing at %s; using default.", TS_PATH)
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


FONT_ABBR = _ts(33 if HEIGHT > 64 else 30)
FONT_SCORE = _ts(48 if HEIGHT > 64 else 37)
FONT_SMALL = _ts(22 if HEIGHT > 64 else 19)

TITLE_GAP = 6
ROW_HEIGHT = 74 if HEIGHT >= 96 else 59
ROW_GAP = 4
STATUS_GAP = 6
FOOTER_MARGIN = 4
LOGO_HEIGHT = ROW_HEIGHT - 11
NEXT_LOGO_HEIGHT = 96 if HEIGHT >= 128 else 74
MATCHUP_GAP = 6
BACKGROUND_COLOR = (0, 0, 0)
HIGHLIGHT_COLOR = (55, 14, 18)
TEXT_COLOR = (255, 255, 255)
BULLS_RED = (200, 32, 45)

_LOGO_CACHE: Dict[Tuple[str, int], Optional[Image.Image]] = {}


def _load_logo_cached(abbr: str, height: int) -> Optional[Image.Image]:
    key = ((abbr or "").upper(), height)
    if key in _LOGO_CACHE:
        logo = _LOGO_CACHE[key]
        return logo.copy() if logo else None

    logo = load_team_logo(NBA_DIR, key[0], height=height)
    if logo is None and key[0] != "NBA":
        logo = load_team_logo(NBA_DIR, "NBA", height=height)
    _LOGO_CACHE[key] = logo
    return logo.copy() if logo else None


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int, int, int]:
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top, left, top
    except Exception:
        width, height = draw.textsize(text, font=font)
        return width, height, 0, 0


def _draw_center(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, y: int, *, fill=TEXT_COLOR) -> int:
    if not text:
        return y
    width, height, left, top = _measure(draw, text, font)
    x = (WIDTH - width) // 2 - left
    draw.text((x, y - top), text, font=font, fill=fill)
    return y + height


def _draw_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, x: int, top: int, height: int, *, align: str = "left", fill=TEXT_COLOR) -> None:
    width, text_height, left, asc = _measure(draw, text, font)
    y = top + (height - text_height) // 2 - asc
    if align == "right":
        tx = x - width - left
    elif align == "center":
        tx = x - width // 2 - left
    else:
        tx = x - left
    draw.text((tx, y), text, font=font, fill=fill)


def _draw_title(draw: ImageDraw.ImageDraw, text: str, y: int = 0) -> int:
    width, height, left, top = _measure(draw, text, FONT_TITLE_SPORTS)
    x = (WIDTH - width) // 2 - left
    draw.text((x, y - top), text, font=FONT_TITLE_SPORTS, fill=TEXT_COLOR)
    return y + height


def _parse_datetime(value: str) -> Optional[dt.datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            parsed = dt.datetime.strptime(text, fmt)
        except Exception:
            continue
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(CENTRAL_TIME)
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(CENTRAL_TIME)


def _get_local_start(game: Dict) -> Optional[dt.datetime]:
    start = game.get("_start_local")
    if isinstance(start, dt.datetime):
        return start.astimezone(CENTRAL_TIME) if start.tzinfo else CENTRAL_TIME.localize(start)
    return _parse_datetime(game.get("gameDate"))


def _get_official_date(game: Dict) -> Optional[dt.date]:
    official = game.get("officialDate")
    if isinstance(official, str) and official:
        try:
            return dt.date.fromisoformat(official[:10])
        except ValueError:
            pass
    start = _get_local_start(game)
    return start.date() if isinstance(start, dt.datetime) else None


def _relative_label(date_obj: Optional[dt.date]) -> str:
    if not isinstance(date_obj, dt.date):
        return ""
    today = dt.datetime.now(CENTRAL_TIME).date()
    if date_obj == today:
        return "Today"
    if date_obj == today + dt.timedelta(days=1):
        return "Tomorrow"
    if date_obj == today - dt.timedelta(days=1):
        return "Yesterday"
    fmt = "%a %b %-d" if os.name != "nt" else "%a %b %#d"
    return date_obj.strftime(fmt)


def _format_time(start: Optional[dt.datetime]) -> str:
    if not isinstance(start, dt.datetime):
        return ""
    fmt = "%-I:%M %p" if os.name != "nt" else "%#I:%M %p"
    return start.strftime(fmt).replace(" 0", " ").lstrip("0")


def _team_entry(game: Dict, side: str) -> Dict[str, Optional[str]]:
    teams = game.get("teams") or {}
    entry = teams.get(side) or {}
    team_info = entry.get("team") if isinstance(entry.get("team"), dict) else {}
    tri = (team_info.get("triCode") or team_info.get("abbreviation") or "").upper()
    name = team_info.get("name") or ""
    team_id = str(team_info.get("id") or "")
    score_raw = entry.get("score")
    try:
        score = int(score_raw)
    except (TypeError, ValueError):
        score = None
    return {
        "tri": tri,
        "name": name,
        "id": team_id,
        "score": score,
    }


def _is_bulls_side(entry: Dict[str, Optional[str]]) -> bool:
    return (entry.get("id") and entry["id"] == BULLS_TEAM_ID) or (entry.get("tri") and entry["tri"].upper() == BULLS_TRICODE)


def _game_state(game: Dict) -> str:
    status = game.get("status") or {}
    abstract = str(status.get("abstractGameState") or "").lower()
    if abstract:
        return abstract
    detailed = str(status.get("detailedState") or "").lower()
    if "final" in detailed:
        return "final"
    if "live" in detailed or "progress" in detailed:
        return "live"
    if "preview" in detailed or "schedule" in detailed or "pregame" in detailed:
        return "preview"
    code = str(status.get("statusCode") or "")
    if code == "3":
        return "final"
    if code == "2":
        return "live"
    if code == "1":
        return "preview"
    return detailed


def _status_text(game: Dict) -> str:
    status = game.get("status") or {}
    return str(status.get("detailedState") or status.get("abstractGameState") or "").strip()


def _live_status(game: Dict) -> str:
    linescore = game.get("linescore") or {}
    clock = (linescore.get("currentPeriodTimeRemaining") or "").strip()
    period = (linescore.get("currentPeriodOrdinal") or "").strip()
    pieces = [piece for piece in (clock, period) if piece]
    if not pieces:
        return _status_text(game) or "Live"
    return " • ".join(pieces)


def _render_message(title: str, message: str) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    bottom = _draw_title(draw, title)
    bottom += TITLE_GAP
    _draw_center(draw, message, FONT_DATE_SPORTS, bottom)
    return img


def _render_scoreboard(game: Dict, *, title: str, footer: str, status_line: str) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    y = _draw_title(draw, title)
    y += TITLE_GAP

    away = _team_entry(game, "away")
    home = _team_entry(game, "home")
    rows = [away, home]

    for idx, info in enumerate(rows):
        top = y + idx * (ROW_HEIGHT + ROW_GAP)
        bottom = top + ROW_HEIGHT
        highlight = _is_bulls_side(info)
        if highlight:
            draw.rectangle((4, top, WIDTH - 5, bottom), fill=HIGHLIGHT_COLOR)
        logo = _load_logo_cached(info["tri"], LOGO_HEIGHT)
        text_left = 10
        if logo is not None:
            lx = 8
            ly = top + (ROW_HEIGHT - logo.height) // 2
            img.paste(logo, (lx, ly), logo)
            text_left = lx + logo.width + 6
        abbr = info["tri"] or "—"
        _draw_text(draw, abbr, FONT_ABBR, text_left, top, ROW_HEIGHT, align="left")
        score_txt = "" if info["score"] is None else str(info["score"])
        score_color = BULLS_RED if highlight else TEXT_COLOR
        _draw_text(draw, score_txt, FONT_SCORE, WIDTH - 8, top, ROW_HEIGHT, align="right", fill=score_color)

    y += 2 * ROW_HEIGHT + ROW_GAP
    y += STATUS_GAP
    if status_line:
        _draw_center(draw, status_line, FONT_SMALL, y)
        y += FONT_SMALL.size if hasattr(FONT_SMALL, "size") else 12

    footer_text = footer.strip()
    if footer_text:
        footer_y = HEIGHT - FOOTER_MARGIN - FONT_DATE_SPORTS.size
        footer_y = max(footer_y, y + STATUS_GAP)
        _draw_center(draw, footer_text, FONT_DATE_SPORTS, footer_y)

    return img


def _format_footer_last(game: Dict) -> str:
    label = _relative_label(_get_official_date(game))
    away = _team_entry(game, "away")
    home = _team_entry(game, "home")
    bulls_home = _is_bulls_side(home)
    opponent = away if bulls_home else home
    opponent_name = opponent.get("name") or opponent.get("tri") or ""
    if label and opponent_name:
        return f"{label} vs {opponent_name}" if bulls_home else f"{label} @ {opponent_name}"
    return label or opponent_name


def _format_footer_next(game: Dict) -> str:
    start = _get_local_start(game)
    date_label = _relative_label(_get_official_date(game))
    time_label = _format_time(start)
    pieces = [piece for piece in (date_label, time_label) if piece]
    return " ".join(pieces)


def _format_matchup_line(game: Dict) -> str:
    away = _team_entry(game, "away")
    home = _team_entry(game, "home")
    bulls_home = _is_bulls_side(home)
    opponent = away if bulls_home else home
    prefix = "vs." if bulls_home else "@"
    return f"{prefix} {opponent.get('name') or opponent.get('tri') or ''}".strip()


def _render_next_game(game: Dict, *, title: str) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    y = _draw_title(draw, title)
    y += TITLE_GAP

    matchup = _format_matchup_line(game)
    y = _draw_center(draw, matchup, FONT_TEAM_SPORTS, y)
    y += MATCHUP_GAP

    away = _team_entry(game, "away")
    home = _team_entry(game, "home")
    away_logo = _load_logo_cached(away["tri"], NEXT_LOGO_HEIGHT)
    home_logo = _load_logo_cached(home["tri"], NEXT_LOGO_HEIGHT)
    at_text = "@"
    at_width, at_height, at_left, at_top = _measure(draw, at_text, FONT_TITLE_SPORTS)
    logo_width = (away_logo.width if away_logo else 0) + (home_logo.width if home_logo else 0)
    total_width = logo_width + at_width + 24
    start_x = max(0, (WIDTH - total_width) // 2)
    logo_y = y
    if away_logo:
        ay = logo_y + (NEXT_LOGO_HEIGHT - away_logo.height) // 2
        img.paste(away_logo, (start_x, ay), away_logo)
        start_x += away_logo.width + 12
    else:
        _draw_text(draw, away.get("tri") or "AWY", FONT_TEAM_SPORTS, start_x, logo_y, NEXT_LOGO_HEIGHT, align="left")
        start_x += FONT_TEAM_SPORTS.size + 12

    draw.text((start_x - at_left, logo_y + (NEXT_LOGO_HEIGHT - at_height) // 2 - at_top), at_text, font=FONT_TITLE_SPORTS, fill=TEXT_COLOR)
    start_x += at_width + 12

    if home_logo:
        hy = logo_y + (NEXT_LOGO_HEIGHT - home_logo.height) // 2
        img.paste(home_logo, (start_x, hy), home_logo)
    else:
        _draw_text(draw, home.get("tri") or "HOME", FONT_TEAM_SPORTS, start_x, logo_y, NEXT_LOGO_HEIGHT, align="left")

    footer = _format_footer_next(game)
    if footer:
        footer_y = HEIGHT - FOOTER_MARGIN - FONT_DATE_SPORTS.size
        _draw_center(draw, footer, FONT_DATE_SPORTS, footer_y)

    return img


def _push(display, img: Optional[Image.Image], *, transition: bool = False) -> Optional[Image.Image]:
    if img is None or display is None:
        return None
    if transition:
        return img
    try:
        clear_display(display)
    except Exception:
        pass
    try:
        if hasattr(display, "image"):
            display.image(img)
        elif hasattr(display, "ShowImage"):
            buf = display.getbuffer(img) if hasattr(display, "getbuffer") else img
            display.ShowImage(buf)
        elif hasattr(display, "display"):
            display.display(img)
    except Exception as exc:
        logging.exception("Failed to push Bulls screen: %s", exc)
    return None


def draw_last_bulls_game(display, game: Optional[Dict], transition: bool = False):
    if not game:
        logging.warning("bulls last: no data")
        img = _render_message("Last Bulls game:", "No results available")
        return _push(display, img, transition=transition)

    footer = _format_footer_last(game)
    status_line = _status_text(game) or "Final"
    img = _render_scoreboard(game, title="Last Bulls game:", footer=footer, status_line=status_line)
    return _push(display, img, transition=transition)


def draw_live_bulls_game(display, game: Optional[Dict], transition: bool = False):
    if not game:
        logging.info("bulls live: no live game")
        img = _render_message("Bulls Live:", "Not in progress")
        return _push(display, img, transition=transition)

    if _game_state(game) != "live":
        logging.info("bulls live: game not live (state=%s)", _game_state(game))
        img = _render_message("Bulls Live:", "Not in progress")
        return _push(display, img, transition=transition)

    footer = _relative_label(_get_official_date(game))
    img = _render_scoreboard(game, title="Bulls Live:", footer=footer, status_line=_live_status(game))
    return _push(display, img, transition=transition)


def draw_sports_screen_bulls(display, game: Optional[Dict], transition: bool = False):
    if not game:
        logging.warning("bulls next: no upcoming game")
        img = _render_message("Next Bulls game:", "No upcoming game found")
        return _push(display, img, transition=transition)

    img = _render_next_game(game, title="Next Bulls game:")
    return _push(display, img, transition=transition)


def draw_bulls_next_home_game(display, game: Optional[Dict], transition: bool = False):
    if not game:
        logging.info("bulls next home: no upcoming home game")
        img = _render_message("Next at home...", "No United Center games scheduled")
        return _push(display, img, transition=transition)

    img = _render_next_game(game, title="Next at home...")
    return _push(display, img, transition=transition)
