#!/usr/bin/env python3
"""
draw_bulls_schedule.py

Bulls screens styled to match the Blackhawks cards:
- Last Bulls game: compact 2-row scoreboard (logo+abbr | score) with title strip and relative-date footer.
- Bulls Live: same scoreboard with live status line.
- Next Bulls game / Next at home: centered matchup + two big logos with an '@' between them,
  footer with relative date + local time.

Changes requested:
- Removed the colored background behind the Bulls score row.
- Removed the "PTS" label above the score column.
- Added '@' between the two team logos on both Next-game screens.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from collections.abc import Sequence
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

import config
from config import (
    CENTRAL_TIME,
    FONT_DATE_SPORTS,
    FONT_TEAM_SPORTS,
    FONT_TITLE_SPORTS,
    HEIGHT,
    NBA_IMAGES_DIR,  # images/nba/
    NBA_TEAM_TRICODE,  # e.g., "CHI"
    TIMES_SQUARE_FONT_PATH,  # TimesSquare-m105.ttf
    WIDTH,
    get_screen_background_color,
    is_hyperpixel_4_square_layout,
)
from utils import (
    LED_INDICATOR_LEVEL,
    ScreenImage,
    fit_font,
    fit_logo_to_box,
    standard_next_game_logo_frame_width,
    standard_next_game_logo_height,
    standard_next_game_logo_height_for_space,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fonts & layout

TS_PATH = TIMES_SQUARE_FONT_PATH
NBA_DIR = NBA_IMAGES_DIR
TEAM_TRICODE = (NBA_TEAM_TRICODE or "CHI").upper()

# Map API abbreviations to logo filenames when they differ
LOGO_ABBREVIATION_OVERRIDES = {
    "BKN": "BRK",  # Brooklyn Nets
    "NOP": "NO",   # New Orleans Pelicans
    "WAS": "WSH",  # Washington Wizards
    "GSW": "GS",   # Golden State Warriors
    "NYK": "NY",   # New York Knicks
    "SAS": "SA",   # San Antonio Spurs
    "PHO": "PHX",  # Phoenix Suns (some feeds use PHO)
}

NBA_TEAM_NICKNAMES = {
    "ATL": "Hawks",
    "BOS": "Celtics",
    "BKN": "Nets",
    "BRK": "Nets",
    "CHA": "Hornets",
    "CHO": "Hornets",
    "CHI": "Bulls",
    "CLE": "Cavaliers",
    "DAL": "Mavericks",
    "DEN": "Nuggets",
    "DET": "Pistons",
    "GSW": "Warriors",
    "GS": "Warriors",
    "HOU": "Rockets",
    "IND": "Pacers",
    "LAC": "Clippers",
    "LAL": "Lakers",
    "MEM": "Grizzlies",
    "MIA": "Heat",
    "MIL": "Bucks",
    "MIN": "Timberwolves",
    "NOP": "Pelicans",
    "NO": "Pelicans",
    "NYK": "Knicks",
    "NY": "Knicks",
    "OKC": "Thunder",
    "ORL": "Magic",
    "PHI": "76ers",
    "PHL": "76ers",
    "PHX": "Suns",
    "PHO": "Suns",
    "POR": "Trail Blazers",
    "SAC": "Kings",
    "SAS": "Spurs",
    "SA": "Spurs",
    "TOR": "Raptors",
    "UTA": "Jazz",
    "UTAH": "Jazz",
    "WAS": "Wizards",
    "WSH": "Wizards",
}

def _ts(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(TS_PATH, size)
    except Exception:
        logging.warning("TimesSquare font missing at %s; using fallback.", TS_PATH)
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

# Tuned for 320x240; scales acceptably for larger canvases
_IS_HYPERPIXEL_4_SQUARE = is_hyperpixel_4_square_layout()
_IS_HYPERPIXEL_4 = config.is_hyperpixel_next_layout() and not _IS_HYPERPIXEL_4_SQUARE
_IS_ADAFRUIT_MINIPITFT = sorted((WIDTH, HEIGHT)) == [135, 240]
_MINIPITFT_FONT_SCALE = 0.56 if _IS_ADAFRUIT_MINIPITFT else 1.0
_FONT_ABBR_SIZE = 40 if _IS_HYPERPIXEL_4_SQUARE else (33 if HEIGHT >= 240 else 28)
_FONT_SCORE_SIZE = 60 if _IS_HYPERPIXEL_4_SQUARE else (48 if HEIGHT >= 240 else 38)
_H4SQ_SCORE_FONT_SIZE = round(60 * 3.0)
_SCORE_X_SHIFT_PX = 10
if _IS_HYPERPIXEL_4_SQUARE:
    # HyperPixel 4 Square request:
    # - team names on Last/Live scoreboard are 2× bigger
    # - scores on Last/Live scoreboard are 3× bigger
    _FONT_ABBR_SIZE = int(round(_FONT_ABBR_SIZE * 2.0))
    _FONT_SCORE_SIZE = int(round(_FONT_SCORE_SIZE * 3.0))
elif _IS_HYPERPIXEL_4:
    # HyperPixel 4 request: match H4 Square score sizing on Last/Live cards.
    _FONT_SCORE_SIZE = _H4SQ_SCORE_FONT_SIZE
if _IS_ADAFRUIT_MINIPITFT:
    _FONT_ABBR_SIZE = max(14, int(round(_FONT_ABBR_SIZE * _MINIPITFT_FONT_SCALE)))
    _FONT_SCORE_SIZE = max(22, int(round(_FONT_SCORE_SIZE * _MINIPITFT_FONT_SCALE)))
FONT_ABBR   = _ts(_FONT_ABBR_SIZE)  # team abbr in table
FONT_SCORE  = _ts(_FONT_SCORE_SIZE)  # score digits
_font_small_size = 26 if _IS_HYPERPIXEL_4_SQUARE else (22 if HEIGHT >= 240 else 18)
if _IS_ADAFRUIT_MINIPITFT:
    _font_small_size = max(12, int(round(_font_small_size * _MINIPITFT_FONT_SCALE)))
FONT_SMALL  = _ts(_font_small_size)  # status / small lines

# Shared sports fonts from config (keeps Hawks look)
FONT_TITLE    = FONT_TITLE_SPORTS                # title strip
FONT_BOTTOM   = FONT_DATE_SPORTS                 # footer (date/time)
FONT_NEXT_OPP = FONT_TEAM_SPORTS                 # opponent line in "next" cards

BOTTOM_LINE_MARGIN = 6
_IS_1080P_LAYOUT = config.is_hdmi_1080p_layout()
_LOGO_SCALE_1080 = config.DISPLAY_PROFILE_LOGO_SCALE_CAP


def _bottom_line_margin(*, hyperpixel_layout: bool = False, extra: int = 0) -> int:
    if _IS_HYPERPIXEL_4_SQUARE:
        margin = 18
    elif _IS_HYPERPIXEL_4:
        margin = 15
    elif hyperpixel_layout:
        margin = 10
    else:
        margin = BOTTOM_LINE_MARGIN
    if _IS_1080P_LAYOUT:
        margin += 30
    return margin + extra

# Colors
BACKGROUND_COLOR = (0, 0, 0)
TEXT_COLOR       = (255, 255, 255)

# ─────────────────────────────────────────────────────────────────────────────
# Text helpers

def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t, l, t
    except Exception:
        w, h = draw.textsize(text, font=font)
        return w, h, 0, 0

def _text_w(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    return _measure(draw, text, font)[0]

def _text_h(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    return _measure(draw, "Ag", font)[1]

def _center_text(draw: ImageDraw.ImageDraw, y: int, text: str, font: ImageFont.ImageFont, *, fill=TEXT_COLOR) -> int:
    if not text:
        return 0
    w = _text_w(draw, text, font)
    x = max(0, (WIDTH - w) // 2)
    draw.text((x, y), text, font=font, fill=fill)
    return _text_h(draw, font)

def _center_wrapped_text(
    draw: ImageDraw.ImageDraw,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    *,
    max_width: int,
    line_gap: int = 2,
    fill=TEXT_COLOR,
) -> int:
    if not text:
        return 0
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w]) if cur else w
        if _text_w(draw, trial, font) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    total = 0
    for line in lines:
        total += _center_text(draw, y + total, line, font, fill=fill)
        total += line_gap
    return max(0, total - line_gap)

# ─────────────────────────────────────────────────────────────────────────────
# Logos

def _load_logo_png(abbr: str, height: int) -> Optional[Image.Image]:
    abbr = (abbr or "NBA").upper()
    # Apply abbreviation overrides to match actual filenames
    abbr = LOGO_ABBREVIATION_OVERRIDES.get(abbr, abbr)
    path = os.path.join(NBA_DIR, f"{abbr}.png")
    try:
        if os.path.exists(path):
            img = Image.open(path).convert("RGBA")
            return fit_logo_to_box(img, height)
    except Exception:
        pass
    # fallback
    try:
        generic = os.path.join(NBA_DIR, "NBA.png")
        if os.path.exists(generic):
            img = Image.open(generic).convert("RGBA")
            return fit_logo_to_box(img, height)
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers

def _get_s(game: dict, path: Sequence[str], default="") -> str:
    d: object = game
    for key in path:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
    if d is None:
        return default
    if isinstance(d, (int, float)):
        return str(d)
    if isinstance(d, str):
        return d
    return default

def _str_or_blank(value: object) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, str):
            value = value.strip()
        if value:
            return str(value)
    except Exception:
        pass
    return ""

def _strip_location_prefix(name: str, locations: Sequence[str]) -> str:
    if not name:
        return ""
    candidate = name.strip()
    if not candidate:
        return ""
    locs = [loc.strip() for loc in locations if isinstance(loc, str) and loc.strip()]
    for loc in sorted(set(locs), key=len, reverse=True):
        loc_lower = loc.lower()
        cand_lower = candidate.lower()
        if cand_lower.startswith(loc_lower):
            idx = len(loc)
            if idx < len(candidate) and candidate[idx] not in " -–—,:":
                continue
            remainder = candidate[len(loc):].lstrip(" -–—,:")
            if remainder:
                return remainder
    return candidate

def _team_entry(game: dict, side: str) -> dict[str, Optional[str]]:
    teams = game.get("teams") or {}
    entry = teams.get(side) or {}
    team_info = entry.get("team") if isinstance(entry.get("team"), dict) else None

    tri_candidates = (
        "tri",
        "triCode",
        "tricode",
        "teamTricode",
        "teamTriCode",
        "abbreviation",
        "abbr",
        "teamAbbreviation",
        "teamAbbrev",
    )
    name_candidates = (
        "nickname",
        "teamNickname",
        "shortName",
        "teamShortName",
        "teamName",
        "name",
        "displayName",
        "fullName",
        "clubName",
        "clubNickname",
    )
    location_candidates = (
        "city",
        "teamCity",
        "teamLocation",
        "cityName",
        "market",
        "location",
        "homeCity",
    )

    score = entry.get("score")
    try:
        score = int(score) if score is not None and str(score).strip() != "" else None
    except Exception:
        score = None

    tri = ""
    if team_info and isinstance(team_info, dict):
        for k in tri_candidates:
            if team_info.get(k):
                tri = str(team_info[k])
                break
    if not tri:
        for k in tri_candidates:
            if entry.get(k):
                tri = str(entry[k])
                break

    names: list[str] = []
    locations: list[str] = []

    if team_info and isinstance(team_info, dict):
        for key in name_candidates:
            value = team_info.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
        for key in location_candidates:
            value = team_info.get(key)
            if isinstance(value, str) and value.strip():
                locations.append(value.strip())

    for key in name_candidates:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            names.append(value.strip())

    for key in location_candidates:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            locations.append(value.strip())

    tri_upper = (tri or "").upper()
    nickname = NBA_TEAM_NICKNAMES.get(tri_upper)

    cleaned_name = ""
    if not nickname and names:
        for candidate in names:
            stripped = _strip_location_prefix(candidate, locations)
            if stripped and stripped.lower() != candidate.lower():
                cleaned_name = stripped
                break
        if not cleaned_name:
            for candidate in names:
                stripped = _strip_location_prefix(candidate, locations)
                if stripped:
                    cleaned_name = stripped
                    break

    label = (nickname or cleaned_name or (names[0] if names else "") or tri or "").strip() or "NBA"

    if not nickname and tri_upper:
        nickname = NBA_TEAM_NICKNAMES.get(tri_upper)

    name_value = (nickname or cleaned_name or (names[0] if names else label) or label).strip()

    location_value = ""
    if locations:
        location_value = locations[0]
    elif names and cleaned_name:
        # Try to derive the location from the first full name entry.
        for candidate in names:
            candidate_clean = candidate.strip()
            idx = candidate_clean.lower().find(cleaned_name.lower())
            if idx > 0:
                prefix = candidate_clean[:idx].strip(" -–—,:")
                if prefix:
                    location_value = prefix
                    break

    full_name = ""
    if location_value and cleaned_name:
        full_name = f"{location_value} {cleaned_name}".strip()
    elif names:
        full_name = names[0].strip()
    else:
        full_name = label

    return {
        "tri": tri or label,
        "name": name_value,
        "label": label,
        "score": score,
        "location": location_value,
        "full_name": full_name,
    }

def _is_bulls_side(entry: dict[str, Optional[str]]) -> bool:
    tri = (entry.get("tri") or "").upper()
    return tri == TEAM_TRICODE or tri == "CHI"  # ensure CHI is always considered Bulls

def _game_state(game: dict) -> str:
    state = _str_or_blank(game.get("gameStatusText") or game.get("gameStatus") or game.get("status"))
    s = state.lower()
    if "final" in s or s == "finished":
        return "final"
    if "live" in s or "q" in s or "1st" in s or "2nd" in s or "3rd" in s or "4th" in s or "ot" in s:
        return "live"
    return "pre"

def _official_date_from_str(official: str) -> Optional[dt.date]:
    try:
        y, m, d = [int(x) for x in official.split("-")]
        return dt.date(y, m, d)
    except Exception:
        return None

def _official_date(game: dict) -> Optional[dt.date]:
    for k in ("officialDate", "official_date", "gameDate", "date", "game_date"):
        d = game.get(k)
        if isinstance(d, str):
            d = _official_date_from_str(d)
        if d:
            return d
    start = _get_local_start(game)
    return start.date() if isinstance(start, dt.datetime) else None

def _get_local_start(game: dict) -> Optional[dt.datetime]:
    iso = (game.get("dateTime") or game.get("startTime") or game.get("gameDate") or "")
    if not iso:
        return None
    try:
        t = iso.replace("Z", "+00:00")
        dt_obj = dt.datetime.fromisoformat(t)
        if dt_obj.tzinfo:
            return dt_obj.astimezone(CENTRAL_TIME)
        return dt_obj.replace(tzinfo=CENTRAL_TIME)
    except Exception:
        return None

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
    delta = (date_obj - today).days
    if -6 <= delta <= 6:
        return date_obj.strftime("%A")
    return date_obj.strftime("%b %-d")

def _status_text(game: dict) -> str:
    raw_status = game.get("gameStatusText") or game.get("statusText") or game.get("status") or game.get("gameStatus")

    def _from_mapping(status_obj: dict) -> str:
        for key in ("detailedState", "shortDetail", "detail", "description", "state", "name", "text"):
            value = status_obj.get(key)
            if isinstance(value, dict):
                nested = _from_mapping(value)
                if nested:
                    return nested
            elif isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    return candidate
        type_obj = status_obj.get("type")
        if isinstance(type_obj, dict):
            return _from_mapping(type_obj)
        return ""

    if isinstance(raw_status, dict):
        extracted = _from_mapping(raw_status)
        if extracted:
            return extracted
        # Fall back to generic string conversion if nothing useful found.
        return ""

    return _str_or_blank(raw_status)

def _format_footer_last(game: dict) -> str:
    status = _status_text(game).strip()
    if not status:
        status = "Final"
    date_label = _relative_label(_official_date(game))
    parts = [part for part in (status, date_label) if part]
    return " • ".join(parts)

def _format_footer_next(game: dict) -> str:
    # Date + local time with relative-day labels when applicable.
    start = _get_local_start(game)
    if not isinstance(start, dt.datetime):
        return _relative_label(_official_date(game))

    today = dt.datetime.now(CENTRAL_TIME).date()
    start_date = start.date()
    if start_date == today:
        day_label = "Tonight" if start.hour >= 18 else "Today"
    elif start_date == today + dt.timedelta(days=1):
        day_label = "Tomorrow"
    else:
        day_label = start.strftime("%a %b %-d")

    return f"{day_label} • {start.strftime('%-I:%M %p')}"


def _format_footer_live(game: dict) -> str:
    status = _status_text(game).strip()
    if not status:
        status = "Live"
    return status

def _format_matchup_line(game: dict) -> str:
    away = _team_entry(game, "away")
    home = _team_entry(game, "home")
    bulls_home = _is_bulls_side(home)
    opponent = away if bulls_home else home
    if not opponent:
        return ""
    opponent_city = _str_or_blank(opponent.get("location"))
    opponent_name = _str_or_blank(opponent.get("name"))
    opponent_full = _str_or_blank(opponent.get("full_name"))

    if opponent_city and opponent_name:
        opponent_display = f"{opponent_city} {opponent_name}".strip()
    elif opponent_full:
        opponent_display = opponent_full
    else:
        opponent_display = _str_or_blank(opponent.get("label") or opponent.get("tri"))

    if not opponent_display:
        return ""
    prefix = "vs." if bulls_home else "@"
    return f"{prefix} {opponent_display}".strip()

# ─────────────────────────────────────────────────────────────────────────────
# Drawing primitives

def _draw_title_line(draw: ImageDraw.ImageDraw, y: int, text: str) -> int:
    return _center_text(draw, y, text, FONT_TITLE)

def _draw_scoreboard_table(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    top_y: int,
    rows: tuple[dict[str, object], ...],
    *,
    bottom_reserved_px: int = 0,
    hyperpixel_layout: bool = False,
    center_vertically: bool = False,
) -> int:
    """
    2-row compact table: team cell at left, score column at right.
    NOTE: No header label (PTS) by request.
    """
    if not rows:
        return top_y

    # HyperPixel layouts render larger overall; trim team/score fonts 15%
    # on Last/Live cards for better balance.
    if _IS_HYPERPIXEL_4:
        team_font = _ts(getattr(FONT_TITLE, "size", _FONT_ABBR_SIZE))
    else:
        team_font = _ts(int(round(_FONT_ABBR_SIZE * 0.85))) if hyperpixel_layout else FONT_ABBR
    score_font = _ts(int(round(_FONT_SCORE_SIZE * 0.85))) if hyperpixel_layout else FONT_SCORE

    row_count = len(rows)
    col1_w = min(WIDTH - 24, max(84, int(WIDTH * 0.72)))
    col2_w = max(20, WIDTH - col1_w)
    x0, x1, x2 = 0, col1_w, WIDTH

    header_h = 0  # removed
    table_top = top_y

    # Reserve space: rows + bottom_reserved_px
    min_row_h = config.scale_value(40) if hyperpixel_layout else 40
    row_h = max(min_row_h, int((HEIGHT - top_y - header_h - bottom_reserved_px) / row_count))
    table_h = header_h + row_h * row_count
    if center_vertically:
        free_space = max(0, HEIGHT - bottom_reserved_px - table_h - top_y)
        table_top = top_y + free_space // 2
    y = table_top + header_h  # header_h = 0

    # Rows
    for i, r in enumerate(rows):
        top = y
        h = row_h
        label = _str_or_blank(r.get("label") or "")
        tri = _str_or_blank(r.get("tri") or "")
        score = r.get("score")

        # Background highlight behind Bulls row was removed per request.

        # Logo
        pad_logo = config.scale_value(6) if hyperpixel_layout else 6
        logo_min = config.scale_value(24) if hyperpixel_layout else 24
        logo_max = config.scale_value(64) if hyperpixel_layout else 64
        base_h = max(1, h - pad_logo)
        logo_h = min(logo_max, max(logo_min, int(round(base_h * _LOGO_SCALE_1080))))
        logo   = _load_logo_png(tri, logo_h)
        px = config.scale_value(6) if hyperpixel_layout else 6
        if logo:
            ly = top + (h - logo.height) // 2
            img.paste(logo, (px, ly), logo)
            px += logo.width + (config.scale_value(6) if hyperpixel_layout else 6)

        # Team label
        max_text_w = max(1, x1 - (config.scale_value(6) if hyperpixel_layout else 6) - px)
        use_font = team_font if _text_w(draw, label, team_font) <= max_text_w else FONT_SMALL
        draw.text((px, top + (h - _text_h(draw, use_font)) // 2), label, font=use_font, fill=TEXT_COLOR)

        # Score column (right aligned)
        if score is not None:
            s = str(score)
            sw = _text_w(draw, s, score_font)
            score_shift = _SCORE_X_SHIFT_PX + (20 if _IS_HYPERPIXEL_4_SQUARE else 0)
            sx = x1 + (col2_w - sw) // 2 - score_shift
            sy = top + (h - _text_h(draw, score_font)) // 2
            draw.text((sx, sy), s, font=score_font, fill=TEXT_COLOR)

        y += h

    return y

def _render_message(title: str, message: str, *, hyperpixel_layout: bool = False) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    edge_pad = config.scale_value(2) if hyperpixel_layout else 2
    line_gap = config.scale_value(4) if hyperpixel_layout else 4
    y = edge_pad
    y += _draw_title_line(draw, y, title)
    y += line_gap
    _center_wrapped_text(
        draw,
        y,
        message,
        FONT_TEAM_SPORTS,
        max_width=WIDTH - (config.scale_value(12) if hyperpixel_layout else 12),
    )
    return img

def _render_scoreboard(
    game: dict,
    *,
    title: str,
    footer: Optional[str] = "",
    status_line: Optional[str] = "",
    hyperpixel_layout: bool = False,
) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    edge_pad = config.scale_value(2) if hyperpixel_layout else 2
    line_gap = config.scale_value(2) if hyperpixel_layout else 2
    y = edge_pad
    y += _draw_title_line(draw, y, title)
    if status_line:
        y += line_gap + _center_text(draw, y, status_line, FONT_SMALL)
    y += line_gap

    away = _team_entry(game, "away")
    home = _team_entry(game, "home")

    bottom_line = footer or ""
    bottom_reserved = (
        _text_h(draw, FONT_BOTTOM)
        + _bottom_line_margin(hyperpixel_layout=hyperpixel_layout)
        if bottom_line
        else 0
    )

    rows = (
        {"tri": away["tri"], "label": away["label"], "score": away["score"]},
        {"tri": home["tri"], "label": home["label"], "score": home["score"]},
    )
    _draw_scoreboard_table(
        img,
        draw,
        y,
        rows,
        bottom_reserved_px=bottom_reserved,
        hyperpixel_layout=hyperpixel_layout,
        center_vertically=_IS_HYPERPIXEL_4_SQUARE and title in {"Last Bulls game:", "Bulls Live:"},
    )

    if bottom_line:
        bottom_margin = _bottom_line_margin(hyperpixel_layout=hyperpixel_layout)
        by = HEIGHT - _text_h(draw, FONT_BOTTOM) - bottom_margin
        _center_text(draw, by, bottom_line, FONT_BOTTOM, fill=TEXT_COLOR)

    return img

def _render_next_game(game: dict, *, title: str, logo_scale: float = 1.0) -> Image.Image:
    """
    Two large logos with an '@' centered between them, plus matchup text and footer.
    """
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    hyperpixel_layout = config.is_hyperpixel_next_layout()
    edge_pad = max(2, config.scale_value(2)) if hyperpixel_layout else 2
    line_gap = max(2, config.scale_value(2)) if hyperpixel_layout else 2

    y = edge_pad
    y += _draw_title_line(draw, y, title)
    y += line_gap

    matchup = _format_matchup_line(game)
    if matchup:
        line_width = max(1, WIDTH - (edge_pad * 2))
        matchup_font = fit_font(
            draw,
            matchup,
            FONT_NEXT_OPP,
            max_width=line_width,
            max_height=max(1, _text_h(draw, FONT_NEXT_OPP) * 2),
            min_pt=max(8, int(round(getattr(FONT_NEXT_OPP, "size", 20) * 0.6))),
        )
        y += _center_text(draw, y, matchup, matchup_font) + line_gap

    away = _team_entry(game, "away")
    home = _team_entry(game, "home")

    footer = _format_footer_next(game)

    # Two large logos with '@' between them
    bottom_margin = _bottom_line_margin(hyperpixel_layout=hyperpixel_layout)
    bottom_reserved = _text_h(draw, FONT_BOTTOM) + bottom_margin if footer else 0
    bottom_y = HEIGHT - bottom_reserved
    logo_top_min = y + (config.scale_value(6) if hyperpixel_layout else 6)
    available_h = max(10, bottom_y - logo_top_min)
    if hyperpixel_layout:
        desired_logo_h = max(
            1,
            int(round(standard_next_game_logo_height(HEIGHT) * config.DISPLAY_SCALE)),
        )
        logo_h = min(int(round(desired_logo_h * logo_scale)), available_h)
    else:
        logo_h = standard_next_game_logo_height_for_space(HEIGHT, available_h, scale=logo_scale)
    logo_left  = _load_logo_png(away["tri"], logo_h) if away else None
    logo_right = _load_logo_png(home["tri"], logo_h) if home else None

    frame_w = standard_next_game_logo_frame_width(logo_h, (logo_left, logo_right))
    gap = config.scale_value(10) if hyperpixel_layout else 10
    at_symbol = "@"
    # Match Hawks Next/Hawks Next Home "@" sizing.
    at_font = FONT_NEXT_OPP

    at_w, at_h, at_l, at_t = _measure(draw, at_symbol, at_font)
    block_h = logo_h if (logo_left or logo_right) else at_h
    total_w = (frame_w * 2) + (gap * 2) + at_w

    if total_w > WIDTH:
        gap = max(4, int(round(gap * (WIDTH / max(total_w, 1)))))
        total_w = (frame_w * 2) + (gap * 2) + at_w

    if total_w > WIDTH:
        max_frame = max(1, (WIDTH - at_w - (gap * 2)) // 2)
        if max_frame < frame_w:
            scale = max_frame / frame_w if frame_w else 1.0
            logo_h = max(1, int(round(logo_h * scale)))
            logo_left = _load_logo_png(away["tri"], logo_h) if away else None
            logo_right = _load_logo_png(home["tri"], logo_h) if home else None
            frame_w = min(
                standard_next_game_logo_frame_width(logo_h, (logo_left, logo_right)),
                max_frame,
            )

        def _fit_logo(logo):
            if logo and logo.width > frame_w:
                ratio = frame_w / logo.width
                new_h = max(1, int(round(logo.height * ratio)))
                return logo.resize((frame_w, new_h), Image.ANTIALIAS)
            return logo

        logo_left = _fit_logo(logo_left)
        logo_right = _fit_logo(logo_right)
        block_h = max(
            (logo.height for logo in (logo_left, logo_right) if logo),
            default=at_h if not (logo_left or logo_right) else logo_h,
        )
        total_w = (frame_w * 2) + (gap * 2) + at_w

    if _IS_HYPERPIXEL_4_SQUARE and bottom_y > logo_top_min:
        y2 = logo_top_min + max(0, (bottom_y - logo_top_min - block_h) // 2)
    else:
        y2 = logo_top_min

    x = max(0, (WIDTH - total_w) // 2)
    baseline_y = y2 + (block_h - at_h) // 2 - at_t

    left_x = x
    at_x = left_x + frame_w + gap
    right_x = at_x + at_w + gap

    def _paste_logo(logo, frame_x):
        if not logo:
            return
        lx = frame_x + (frame_w - logo.width) // 2
        ly = y2 + (logo_h - logo.height) // 2
        img.paste(logo, (lx, ly), logo)

    _paste_logo(logo_left, left_x)
    draw.text((at_x, baseline_y), at_symbol, font=at_font, fill=TEXT_COLOR)
    _paste_logo(logo_right, right_x)

    if footer:
        by = HEIGHT - _text_h(draw, FONT_BOTTOM) - bottom_margin
        _center_text(draw, by, footer, FONT_BOTTOM, fill=TEXT_COLOR)

    return img

# ─────────────────────────────────────────────────────────────────────────────
# Display push

def _push(display, img: Optional[Image.Image], *, transition: bool = False, led_override: Optional[tuple[float, float, float]] = None):
    if img is None or display is None:
        return None
    return ScreenImage(img, displayed=False, led_override=led_override)

# ─────────────────────────────────────────────────────────────────────────────
# Public entry points (used by screens/registry.py)

def draw_last_bulls_game(display, game: Optional[dict], transition: bool = False):
    global BACKGROUND_COLOR
    BACKGROUND_COLOR = get_screen_background_color("bulls last", (0, 0, 0))
    hyperpixel_layout = config.is_hyperpixel_next_layout()
    if not game:
        img = _render_message("Last Bulls game:", "No results", hyperpixel_layout=hyperpixel_layout)
        return _push(display, img, transition=transition)

    footer = _format_footer_last(game)
    img = _render_scoreboard(
        game,
        title="Last Bulls game:",
        footer=footer,
        hyperpixel_layout=hyperpixel_layout,
    )

    # LED: green win, red loss (if both scores present)
    led_override: Optional[tuple[float, float, float]] = None
    away = _team_entry(game, "away")
    home = _team_entry(game, "home")
    bulls = away if _is_bulls_side(away) else home
    opp   = home if bulls is away else away
    if bulls and opp and bulls.get("score") is not None and opp.get("score") is not None:
        try:
            b, o = int(bulls["score"]), int(opp["score"])
            if b > o: led_override = (0.0, 1.0, 0.0)
            elif b < o: led_override = (1.0, 0.0, 0.0)
        except Exception:
            pass

    if led_override is not None and LED_INDICATOR_LEVEL and LED_INDICATOR_LEVEL > 0:
        led_override = (
            led_override[0] * LED_INDICATOR_LEVEL,
            led_override[1] * LED_INDICATOR_LEVEL,
            led_override[2] * LED_INDICATOR_LEVEL,
        )
    return _push(display, img, transition=transition, led_override=led_override)

def draw_live_bulls_game(display, game: Optional[dict], transition: bool = False):
    global BACKGROUND_COLOR
    BACKGROUND_COLOR = get_screen_background_color("bulls live", (0, 0, 0))
    hyperpixel_layout = config.is_hyperpixel_next_layout()
    if not game or _game_state(game) != "live":
        img = _render_message("Bulls Live:", "Not in progress", hyperpixel_layout=hyperpixel_layout)
        return _push(display, img, transition=transition)

    footer = _format_footer_live(game)
    img = _render_scoreboard(
        game,
        title="Bulls Live:",
        footer=footer,
        hyperpixel_layout=hyperpixel_layout,
    )
    return _push(display, img, transition=transition)

def draw_sports_screen_bulls(display, game: Optional[dict], transition: bool = False):
    global BACKGROUND_COLOR
    BACKGROUND_COLOR = get_screen_background_color("bulls next", (0, 0, 0))
    if not game:
        img = _render_message("Next Bulls game:", "No upcoming games scheduled")
        return _push(display, img, transition=transition)
    img = _render_next_game(game, title="Next Bulls game:", logo_scale=_LOGO_SCALE_1080)
    return _push(display, img, transition=transition)

def draw_bulls_next_home_game(display, game: Optional[dict], transition: bool = False):
    global BACKGROUND_COLOR
    BACKGROUND_COLOR = get_screen_background_color("bulls next home", (0, 0, 0))
    if not game:
        img = _render_message("Following at home...", "No United Center games scheduled")
        return _push(display, img, transition=transition)
    # Uses the same '@' treatment between logos
    img = _render_next_game(game, title="Following at home...", logo_scale=_LOGO_SCALE_1080)
    return _push(display, img, transition=transition)
