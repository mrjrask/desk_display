#!/usr/bin/env python3
"""
draw_hawks_schedule.py

Blackhawks screens:

- Last Hawks game: compact 2×3 scoreboard (logo+abbr | score | SOG)
  * Title: "Last Hawks game:" (uses same title font as mlb_schedule if available)
  * SOG label sits right above the table
  * Bottom date: "Yesterday" or "Wed Sep 24" (no year) using the same footer/small font as mlb_schedule if available

- Hawks Live: compact scoreboard (same), optional live clock line.

- Next Hawks game:
  * Title: "Next Hawks game:" (mlb title font)
  * Opponent line: "@ FULL TEAM NAME" (if CHI is away) or "vs. FULL TEAM NAME" (if CHI is home)
  * Logos row: AWAY logo  @  HOME logo from local PNGs: images/nhl/{ABBR}.png
    - Logos are centered vertically on the screen and auto-sized larger (up to ~44px on 128px tall panels)
  * Bottom: Always includes time ("Today 7:30 PM", "Tomorrow 6:00 PM", or "Wed Sep 24 7:30 PM")

- Next Hawks home game:
  * Title: "Next at home..."
  * Layout matches the standard next-game card

Function signatures (match main.py):
  - draw_last_hawks_game(display, game, transition=False)
  - draw_live_hawks_game(display, game, transition=False)
  - draw_sports_screen_hawks(display, game, transition=False)
  - draw_hawks_next_home_game(display, game, transition=False)
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

import config
from config import (
    FONT_DATE_SPORTS,
    FONT_TEAM_SPORTS,
    FONT_TITLE_SPORTS,
    NHL_API_ENDPOINTS,
    NHL_FALLBACK_LOGO,
    NHL_IMAGES_DIR,
    NHL_TEAM_ID,
    NHL_TEAM_TRICODE,
    TIMES_SQUARE_FONT_PATH,
    WIDTH,
    HEIGHT,
)
from services.http_client import NHL_HEADERS, get_session, request_json
from utils import standard_next_game_logo_height

TS_PATH = TIMES_SQUARE_FONT_PATH
NHL_DIR = NHL_IMAGES_DIR

def _ts(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(TS_PATH, size)
    except Exception:
        logging.warning("TimesSquare font missing at %s; using default.", TS_PATH)
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

# Try to reuse MLB's helper functions for title layout and date labels.
_MLB = None
try:
    import screens.mlb_schedule as _MLB  # noqa: N816
except Exception:
    _MLB = None

_MLB_DRAW_TITLE = getattr(_MLB, "_draw_title_with_bold_result", None) if _MLB else None
_MLB_REL_DATE_ONLY = getattr(_MLB, "_rel_date_only", None) if _MLB else None
_MLB_FORMAT_GAME_LABEL = getattr(_MLB, "_format_game_label", None) if _MLB else None

# Title and footer fonts mirror the MLB screens via config definitions.
FONT_TITLE  = FONT_TITLE_SPORTS
FONT_BOTTOM = FONT_DATE_SPORTS

# Opponent line on "Next" screens should mirror MLB's 20 pt team font.
FONT_NEXT_OPP = FONT_TEAM_SPORTS

# Scoreboard fonts (TimesSquare family as requested for numeric/abbr)
_ABBR_BASE = 33 if HEIGHT > 64 else 30
_SOG_BASE = 30 if HEIGHT > 64 else 26

_ABBR_FONT_SIZE = int(round(_ABBR_BASE * 1.3))
_SOG_FONT_SIZE = _SOG_BASE

FONT_ABBR  = _ts(_ABBR_FONT_SIZE)
FONT_SOG   = _ts(_SOG_FONT_SIZE)
FONT_SCORE = _ts(int(round(_SOG_FONT_SIZE * 1.45)))    # make goals column stand out more
FONT_SMALL = _ts(22 if HEIGHT > 64 else 19)    # for SOG label / live clock

# NHL endpoints (prefer api-web; quiet legacy fallback)
NHL_WEB_TEAM_MONTH_NOW   = NHL_API_ENDPOINTS["team_month_now"]
NHL_WEB_TEAM_SEASON_NOW  = NHL_API_ENDPOINTS["team_season_now"]
NHL_WEB_GAME_LANDING     = NHL_API_ENDPOINTS["game_landing"]
NHL_WEB_GAME_BOXSCORE    = NHL_API_ENDPOINTS["game_boxscore"]

NHL_STATS_SCHEDULE = NHL_API_ENDPOINTS["stats_schedule"]
NHL_STATS_FEED     = NHL_API_ENDPOINTS["stats_feed"]

TEAM_ID      = NHL_TEAM_ID
TEAM_TRICODE = NHL_TEAM_TRICODE

# ─────────────────────────────────────────────────────────────────────────────
# Display helpers

def _clear_display(display):
    try:
        from utils import clear_display  # in your repo
        clear_display(display)
    except Exception:
        pass

def _push(display, img: Optional[Image.Image], *, transition: bool=False):
    if img is None or display is None:
        return
    if transition:
        return img
    try:
        _clear_display(display)
        if hasattr(display, "image"):
            display.image(img)
        elif hasattr(display, "ShowImage"):
            buf = display.getbuffer(img) if hasattr(display, "getbuffer") else img
            display.ShowImage(buf)
        elif hasattr(display, "display"):
            display.display(img)
    except Exception as e:
        logging.exception("Failed to push image to display: %s", e)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Net helpers

_SESSION = get_session()


def _req_json(url: str, **kwargs) -> Optional[Dict]:
    """GET → JSON with optional quiet logging (quiet=True)."""
    headers = kwargs.pop("headers", None)
    if headers is None and "api-web.nhle.com" in url:
        headers = NHL_HEADERS
    return request_json(url, headers=headers, session=_SESSION, **kwargs)

def _map_apiweb_game(g: Dict) -> Dict:
    """Map api-web game into a minimal StatsAPI-like shape."""
    gid = g.get("id") or g.get("gameId") or g.get("gamePk")
    game_date = (
        g.get("gameDate") or g.get("startTimeUTC") or g.get("startTime")
        or g.get("gameDateTime") or ""
    )
    home = g.get("homeTeam", {}) or g.get("home", {}) or {}
    away = g.get("awayTeam", {}) or g.get("away", {}) or {}

    def _tri(team: Dict, default: str) -> str:
        return team.get("abbrev") or team.get("triCode") or team.get("abbreviation") or default

    home_tri = _tri(home, "HOME")
    away_tri = _tri(away, "AWAY")
    home_id  = home.get("id") or home.get("teamId")
    away_id  = away.get("id") or away.get("teamId")

    st = (g.get("gameState") or g.get("gameStatus") or "").upper()
    if st in ("LIVE", "CRIT"):
        ds = "In Progress"
    elif st in ("FINAL", "OFF"):
        ds = "Final"
    elif st in ("PRE", "FUT", "SCHEDULED", "PREGAME"):
        ds = "Scheduled"
    else:
        ds = st or "Scheduled"

    return {
        "gamePk": gid,
        "gameDate": game_date,
        "status": {"detailedState": ds},
        "teams": {
            "home": {"team": {"id": home_id, "abbreviation": home_tri, "triCode": home_tri}},
            "away": {"team": {"id": away_id, "abbreviation": away_tri, "triCode": away_tri}},
        },
        # also surface raw for name parsing
        "homeTeam": home,
        "awayTeam": away,
        "officialDate": g.get("gameDate", "")[:10],
    }

def fetch_schedule_apiweb(days_back: int, days_fwd: int) -> Optional[Dict]:
    """api-web 'season now' (broader) or 'month now' mapped to {dates:[{games:[...]}}]."""
    j = _req_json(NHL_WEB_TEAM_SEASON_NOW.format(tric=TEAM_TRICODE))
    if not j:
        j = _req_json(NHL_WEB_TEAM_MONTH_NOW.format(tric=TEAM_TRICODE))
    if not j:
        return None
    games = j.get("games") or j.get("gameWeek", []) or j.get("gameMonth", []) or []
    flat = []
    if isinstance(games, list):
        for g in games:
            if isinstance(g, dict) and ("id" in g or "gamePk" in g or "gameId" in g):
                flat.append(_map_apiweb_game(g))
            else:
                inner = g.get("games") if isinstance(g, dict) else None
                if isinstance(inner, list):
                    for gg in inner:
                        flat.append(_map_apiweb_game(gg))
    if not flat:
        return None
    return {"dates": [{"games": flat}]}

def fetch_schedule_legacy(days_back: int, days_fwd: int) -> Optional[Dict]:
    today = dt.date.today()
    start = (today - dt.timedelta(days=days_back)).strftime("%Y-%m-%d")
    end   = (today + dt.timedelta(days=days_fwd)).strftime("%Y-%m-%d")
    return _req_json(NHL_STATS_SCHEDULE, params={"teamId": TEAM_ID, "startDate": start, "endDate": end}, quiet=True)

def fetch_schedule(days_back: int, days_fwd: int) -> Optional[Dict]:
    j = fetch_schedule_apiweb(days_back, days_fwd)
    if j: return j
    return fetch_schedule_legacy(days_back, days_fwd)

def classify_games(schedule_json: Dict) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """Return (live, last_final, next_sched)."""
    dates = schedule_json.get("dates", [])
    games = [g for day in dates for g in day.get("games", [])]
    games.sort(key=lambda g: g.get("gameDate", ""))

    live = next((g for g in games if g.get("status", {}).get("detailedState") in ("In Progress", "In Progress - Critical")), None)

    now_iso = dt.datetime.utcnow().isoformat()
    finals = [g for g in games if g.get("status", {}).get("detailedState") in ("Final", "Game Over") and g.get("gameDate","") <= now_iso]
    last_final = finals[-1] if finals else None

    scheduled = [g for g in games if g.get("status", {}).get("detailedState") in ("Scheduled", "Pre-Game") and g.get("gameDate","") >= now_iso]
    next_sched = scheduled[0] if scheduled else None

    return live, last_final, next_sched

def fetch_game_feed(game_pk: int) -> Optional[Dict]:
    """Prefer api-web boxscore/landing (goals + SOG). Quiet legacy fallback."""
    box  = _req_json(NHL_WEB_GAME_BOXSCORE.format(gid=game_pk))
    land = None if box else _req_json(NHL_WEB_GAME_LANDING.format(gid=game_pk))
    payload = box or land
    if payload:
        home = payload.get("homeTeam") or payload.get("home") or {}
        away = payload.get("awayTeam") or payload.get("away") or {}

        def _tri(t: Dict, default: str) -> str:
            return t.get("abbrev") or t.get("triCode") or t.get("abbreviation") or default
        def _as_int(v):
            try: return int(v) if v is not None else None
            except Exception: return None

        period_desc = payload.get("periodDescriptor") or {}
        per_val = (
            period_desc.get("ordinalNum")
            or period_desc.get("ordinal")
            or period_desc.get("number")
            or ""
        )

        clock_payload = payload.get("clock") or {}
        clock_val = (
            clock_payload.get("timeRemaining")
            or clock_payload.get("remaining")
            or clock_payload.get("time")
            or clock_payload.get("displayValue")
            or clock_payload.get("label")
            or ""
        )

        return {
            "homeTri": _tri(home, "HOME"),
            "awayTri": _tri(away, "AWAY"),
            "homeScore": _as_int(home.get("score")),
            "awayScore": _as_int(away.get("score")),
            "homeSOG": _as_int(home.get("sog") or home.get("shotsOnGoal") or home.get("shots")),
            "awaySOG": _as_int(away.get("sog") or away.get("shotsOnGoal") or away.get("shots")),
            "perOrdinal": per_val,
            "clock": clock_val,
            "clockState": "INTERMISSION" if clock_payload.get("inIntermission") else "",
        }

    # legacy fallback (quiet)
    url = NHL_STATS_FEED.format(gamePk=game_pk)
    data = _req_json(url, quiet=True)
    if not data:
        return None

    lines = data.get("liveData", {}).get("linescore", {})
    teams = lines.get("teams", {})
    gd    = data.get("gameData", {}).get("teams", {})

    def _tri2(t: Dict, default: str) -> str:
        return t.get("abbreviation") or t.get("triCode") or default
    def _as_int(v):
        try: return int(v) if v is not None else None
        except Exception: return None

    intermission = (lines.get("intermissionInfo") or {}).get("inIntermission")

    return {
        "homeTri": _tri2(gd.get("home", {}), "HOME"),
        "awayTri": _tri2(gd.get("away", {}), "AWAY"),
        "homeScore": _as_int((teams.get("home") or {}).get("goals")),
        "awayScore": _as_int((teams.get("away") or {}).get("goals")),
        "homeSOG": _as_int((teams.get("home") or {}).get("shotsOnGoal")),
        "awaySOG": _as_int((teams.get("away") or {}).get("shotsOnGoal")),
        "perOrdinal": lines.get("currentPeriodOrdinal") or lines.get("currentPeriod") or "",
        "clock": lines.get("currentPeriodTimeRemaining") or "",
        "clockState": "INTERMISSION" if intermission else "",
    }

# ─────────────────────────────────────────────────────────────────────────────
# Team + logo helpers (local PNGs)

FALLBACK_LOGO = NHL_FALLBACK_LOGO

def _team_obj_from_any(t: Dict) -> Dict:
    """Return team dict with {'abbrev','id','name'} (and discover names)."""
    if not isinstance(t, dict):
        return {}
    raw = t.get("team") if isinstance(t.get("team"), dict) else t

    def _name_from(d: Dict) -> Optional[str]:
        v = d.get("name")
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            s = v.get("default") or v.get("en")
            if isinstance(s, str) and s.strip():
                return s
        cn = d.get("commonName")
        if isinstance(cn, dict):
            base = cn.get("default") or cn.get("en")
            if base:
                pn = d.get("placeName")
                city = ""
                if isinstance(pn, dict):
                    city = pn.get("default") or pn.get("en") or ""
                return f"{city} {base}".strip()
        tn = d.get("teamName")
        if isinstance(tn, str) and tn.strip():
            pn = d.get("placeName")
            city = ""
            if isinstance(pn, dict):
                city = pn.get("default") or pn.get("en") or ""
            if city:
                return f"{city} {tn}".strip()
            return tn
        return None

    name = _name_from(raw) or raw.get("clubName") or raw.get("shortName") or None
    abbr = raw.get("abbrev") or raw.get("triCode") or raw.get("abbreviation")
    tid  = raw.get("id") or raw.get("teamId")
    return {"abbrev": abbr, "id": tid, "name": name}

def _extract_tris_from_game(game: Dict) -> Tuple[str, str]:
    """(away_tri, home_tri) from a game-like dict."""
    away = game.get("awayTeam") or (game.get("teams") or {}).get("away") or {}
    home = game.get("homeTeam") or (game.get("teams") or {}).get("home") or {}
    a = _team_obj_from_any(away).get("abbrev") or "AWAY"
    h = _team_obj_from_any(home).get("abbrev") or "HOME"
    return a, h

def _load_logo_png(abbr: str, height: int) -> Optional[Image.Image]:
    """Load team logo from local repo PNG: images/nhl/{ABBR}.png; fallback NHL.jpg."""
    if not abbr:
        abbr = "NHL"
    png_path = os.path.join(NHL_DIR, f"{abbr.upper()}.png")
    try:
        if os.path.exists(png_path):
            img = Image.open(png_path).convert("RGBA")
            w0, h0 = img.size
            r = height / float(h0) if h0 else 1.0
            return img.resize((max(1, int(w0*r)), height), Image.LANCZOS)
    except Exception:
        pass
    # Generic fallback
    try:
        if os.path.exists(FALLBACK_LOGO):
            img = Image.open(FALLBACK_LOGO).convert("RGBA")
            w0, h0 = img.size
            r = height / float(h0) if h0 else 1.0
            return img.resize((max(1, int(w0*r)), height), Image.LANCZOS)
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Text helpers

def _text_h(d: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    _, _, _, h = d.textbbox((0,0), "Hg", font=font)
    return h

def _text_w(d: ImageDraw.ImageDraw, s: str, font: ImageFont.ImageFont) -> int:
    l,t,r,b = d.textbbox((0,0), s, font=font)
    return r - l

def _center_text(d: ImageDraw.ImageDraw, y: int, s: str, font: ImageFont.ImageFont):
    x = (WIDTH - _text_w(d, s, font)) // 2
    d.text((x, y), s, font=font, fill="white")


def _center_wrapped_text(
    d: ImageDraw.ImageDraw,
    y: int,
    s: str,
    font: ImageFont.ImageFont,
    *,
    max_width: Optional[int] = None,
    line_spacing: int = 1,
) -> int:
    """Draw text centered on the screen, wrapping to additional lines if needed."""
    if not s:
        return 0

    max_width = min(max_width or WIDTH, WIDTH)

    text_h = _text_h(d, font)

    if _text_w(d, s, font) <= max_width:
        _center_text(d, y, s, font)
        return text_h

    words = s.split()
    if not words:
        return 0

    lines = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}" if current else word
        if _text_w(d, candidate, font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    # If any individual word is wider than the max width, fall back to character wrapping.
    fixed_lines = []
    for line in lines:
        if _text_w(d, line, font) <= max_width:
            fixed_lines.append(line)
            continue

        chunk = ""
        for ch in line:
            test = f"{chunk}{ch}"
            if chunk and _text_w(d, test, font) > max_width:
                fixed_lines.append(chunk)
                chunk = ch
            else:
                chunk = test
        if chunk:
            fixed_lines.append(chunk)

    lines = fixed_lines or lines

    total_height = 0
    for idx, line in enumerate(lines):
        line_y = y + idx * (text_h + line_spacing)
        _center_text(d, line_y, line, font)
        total_height = (idx + 1) * text_h + idx * line_spacing

    return total_height


def _draw_title_line(
    img: Image.Image,
    d: ImageDraw.ImageDraw,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    *,
    extra_offset: int = 0,
) -> int:
    """Draw a centered title, reusing MLB's faux-bold helper when available."""
    top = y + extra_offset
    if callable(_MLB_DRAW_TITLE):
        # Render via MLB helper onto a temporary transparent strip so we can offset it.
        strip_h = _text_h(d, font) + 4
        strip = Image.new("RGBA", (WIDTH, strip_h), (0, 0, 0, 0))
        strip_draw = ImageDraw.Draw(strip)
        _, th = _MLB_DRAW_TITLE(strip_draw, text)
        img.paste(strip, (0, top), strip)
        return max(th, strip_h)

    _center_text(d, top, text, font)
    return _text_h(d, font)

# ─────────────────────────────────────────────────────────────────────────────
# Scoreboard (Live/Last) — wider col1, equal col2/col3, SOG label tight

def _draw_dotted_line(
    d: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color,
    *,
    dash: int = 3,
    gap: int = 3,
):
    """Draw a dotted (dash-gap) line supporting horizontal/vertical segments."""
    x0, y0 = start
    x1, y1 = end
    if x0 == x1:
        if y0 > y1:
            y0, y1 = y1, y0
        y = y0
        while y <= y1:
            segment_end = min(y + dash - 1, y1)
            d.line([(x0, y), (x1, segment_end)], fill=color)
            y += dash + gap
        return
    if y0 == y1:
        if x0 > x1:
            x0, x1 = x1, x0
        x = x0
        while x <= x1:
            segment_end = min(x + dash - 1, x1)
            d.line([(x, y0), (segment_end, y1)], fill=color)
            x += dash + gap
        return
    d.line([start, end], fill=color)


def _draw_dotted_rect(
    d: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    color,
    *,
    dash: int = 3,
    gap: int = 3,
):
    """Draw a dotted rectangle border."""
    left, top, right, bottom = bbox
    _draw_dotted_line(d, (left, top), (right, top), color, dash=dash, gap=gap)
    _draw_dotted_line(d, (right, top), (right, bottom), color, dash=dash, gap=gap)
    _draw_dotted_line(d, (right, bottom), (left, bottom), color, dash=dash, gap=gap)
    _draw_dotted_line(d, (left, bottom), (left, top), color, dash=dash, gap=gap)


def _team_scoreboard_label(team_like: Dict, fallback: str = "") -> str:
    """Prefer short team names ("Kings") for the scoreboard column."""
    if not isinstance(team_like, dict):
        return fallback
    raw = team_like.get("team") if isinstance(team_like.get("team"), dict) else team_like

    def _str_or_dict(val) -> str:
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            for key in ("default", "en", "shortName", "teamName", "fullName"):
                inner = val.get(key)
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()
        return ""

    for key in ("teamName", "shortName", "commonName", "name", "nickname", "clubName"):
        label = _str_or_dict(raw.get(key))
        if label:
            return label
    return fallback


def _draw_scoreboard(
    img: Image.Image,
    d: ImageDraw.ImageDraw,
    top_y: int,
    away_tri: str,
    away_score: Optional[int],
    away_sog: Optional[int],
    home_tri: str,
    home_score: Optional[int],
    home_sog: Optional[int],
    *,
    away_label: Optional[str] = None,
    home_label: Optional[str] = None,
    put_sog_label: bool = True,
    bottom_reserved_px: int = 0,
) -> int:
    """Draw a compact 2×3 scoreboard. Returns bottom y."""
    # Column widths: first column dominates for logo + name, remaining space split
    # for score/SOG with the score column slightly wider than SOG.
    col1_w = min(WIDTH - 32, max(84, int(WIDTH * 0.72)))
    remaining = max(24, WIDTH - col1_w)
    col2_w = max(12, int(round(remaining * 0.55)))
    col3_w = max(8, WIDTH - col1_w - col2_w)
    # Ensure we account for rounding adjustments.
    if col1_w + col2_w + col3_w != WIDTH:
        col3_w = WIDTH - col1_w - col2_w
    x0, x1, x2, x3 = 0, col1_w, col1_w + col2_w, WIDTH

    y = top_y

    header_h = _text_h(d, FONT_SMALL) + 4 if put_sog_label else 0
    table_top = y

    # Row heights — compact
    total_available = max(0, HEIGHT - bottom_reserved_px - table_top)
    available_for_rows = max(0, total_available - header_h)
    row_h = max(available_for_rows // 2, 32)
    row_h = min(row_h, 48)
    if row_h * 2 > available_for_rows and available_for_rows > 0:
        row_h = max(24, available_for_rows // 2)
    if row_h <= 0:
        row_h = 32

    table_height = header_h + (row_h * 2)
    if total_available:
        table_height = min(table_height, total_available)
    if table_height < (header_h + 2):
        table_height = header_h + 2
    table_bottom = min(table_top + table_height, HEIGHT - bottom_reserved_px)
    table_height = max(header_h + 2, table_bottom - table_top)
    table_bottom = table_top + table_height

    header_bottom = table_top + header_h
    row_area_height = max(2, table_height - header_h)
    row1_h = max(1, row_area_height // 2)
    row2_h = max(1, row_area_height - row1_h)
    row1_top = header_bottom
    split_y = row1_top + row1_h

    # We keep the invisible grid for layout math only—no rendered lines.

    # Column headers inside the table
    if header_h:
        header_y = table_top + (header_h - _text_h(d, FONT_SMALL)) // 2
        score_lbl = ""
        sog_lbl = "SOG"
        if score_lbl:
            d.text((x1 + (col2_w - _text_w(d, score_lbl, FONT_SMALL)) // 2, header_y), score_lbl, font=FONT_SMALL, fill="white")
        d.text((x2 + (col3_w - _text_w(d, sog_lbl, FONT_SMALL)) // 2, header_y), sog_lbl, font=FONT_SMALL, fill="white")

    def _prepare_row(
        row_top: int,
        row_height: int,
        tri: str,
        score: Optional[int],
        sog: Optional[int],
        label: Optional[str],
    ) -> Dict:
        base_logo_height = max(1, row_height - 4)
        logo_height = min(56, base_logo_height)
        if row_height >= 38:
            logo_height = min(56, max(logo_height, min(row_height - 2, 48)))
        logo_height = max(1, min(int(round(logo_height * 1.3)), row_height - 2, 64))
        logo = _load_logo_png(tri, height=logo_height)
        logo_w = logo.size[0] if logo else 0
        text = (label or "").strip() or (tri or "").upper() or "—"
        text_start = x0 + 6 + (logo_w + 6 if logo else 0)
        max_width = max(1, x1 - text_start - 4)
        return {
            "top": row_top,
            "height": row_height,
            "tri": tri,
            "score": score,
            "sog": sog,
            "base_text": text,
            "logo": logo,
            "max_width": max_width,
        }

    row_specs = [
        _prepare_row(row1_top, row1_h, away_tri, away_score, away_sog, away_label),
        _prepare_row(split_y, row2_h, home_tri, home_score, home_sog, home_label),
    ]

    def _fits(font: ImageFont.ImageFont) -> bool:
        return all(
            _text_w(d, spec["base_text"], font) <= spec["max_width"]
            for spec in row_specs
            if spec["max_width"] > 0 and spec["base_text"]
        )

    name_font = FONT_ABBR
    if not _fits(name_font):
        size = getattr(FONT_ABBR, "size", None) or _ABBR_FONT_SIZE
        min_size = max(8, int(round(_ABBR_FONT_SIZE * 0.5)))
        chosen = None
        for test_size in range(size - 1, min_size - 1, -1):
            candidate = _ts(test_size)
            if _fits(candidate):
                chosen = candidate
                break
        name_font = chosen or _ts(min_size)

    def _draw_row(spec: Dict):
        y_top = spec["top"]
        row_height = spec["height"]
        tri = spec["tri"]
        score = spec["score"]
        sog = spec["sog"]
        text = spec["base_text"]
        logo = spec["logo"]

        cy = y_top + row_height // 2
        lx = x0 + 6
        tx = lx
        if logo:
            lw, lh = logo.size
            ly = cy - lh//2
            try:
                img.paste(logo, (lx, ly), logo)
            except Exception:
                pass
            tx = lx + lw + 6

        max_width = spec["max_width"]
        font = name_font
        if _text_w(d, text, font) > max_width:
            ellipsis = "…"
            trimmed = text
            while trimmed and _text_w(d, trimmed + ellipsis, font) > max_width:
                trimmed = trimmed[:-1]
            text = (trimmed + ellipsis) if trimmed else ellipsis

        ah = _text_h(d, font)
        aw = _text_w(d, text, font)
        max_tx = x1 - aw - 4
        tx = min(tx, max_tx)
        tx = max(tx, x0 + 4)
        d.text((tx, cy - ah//2), text, font=font, fill="white")

        sc = "-" if score is None else str(score)
        sw = _text_w(d, sc, FONT_SCORE)
        sh = _text_h(d, FONT_SCORE)
        sx = x1 + (col2_w - sw)//2
        sy = cy - sh//2
        d.text((sx, sy), sc, font=FONT_SCORE, fill="white")

        sog_txt = "-" if sog is None else str(sog)
        gw = _text_w(d, sog_txt, FONT_SOG)
        gh = _text_h(d, FONT_SOG)
        gx = x2 + (col3_w - gw)//2
        gy = cy - gh//2
        d.text((gx, gy), sog_txt, font=FONT_SOG, fill="white")

    for spec in row_specs:
        _draw_row(spec)

    return table_bottom  # bottom of table


def _ordinal(n: int) -> str:
    try:
        num = int(n)
    except Exception:
        return str(n)

    if 10 <= num % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(num % 10, "th")
    return f"{num}{suffix}"


def _normalize_period(period_val) -> str:
    if period_val is None:
        return ""
    if isinstance(period_val, str):
        period = period_val.strip()
        if not period:
            return ""
        if period.isdigit():
            return _ordinal(int(period))
        return period
    if isinstance(period_val, (int, float)):
        return _ordinal(int(period_val))
    try:
        return str(period_val).strip()
    except Exception:
        return ""


def _format_live_dateline(feed: Dict) -> str:
    period = _normalize_period(feed.get("perOrdinal"))
    clock = str(feed.get("clock") or "").strip()
    clock_state = str(feed.get("clockState") or "").strip()

    if clock_state:
        state = clock_state.title() if clock_state.isupper() else clock_state
        if period:
            if "intermission" in state.lower():
                return f"{state} ({period})"
            return f"{period} {state}"
        return state

    if clock:
        if clock.upper() == "END" and period:
            return f"End of {period}"
        if period:
            return f"{period} {clock}"
        return clock

    return period

# ─────────────────────────────────────────────────────────────────────────────
# Date formatting (Last)

def _format_last_date_bottom(game_date_iso: str) -> str:
    """Return 'Yesterday' or 'Wed Sep 24' (no year)."""
    try:
        dt_utc = dt.datetime.fromisoformat(game_date_iso.replace("Z","+00:00"))
        local  = dt_utc.astimezone()
        gdate  = local.date()
    except Exception:
        return ""
    today = dt.datetime.now().astimezone().date()
    delta = (today - gdate).days
    if delta == 1:
        return "Yesterday"
    return local.strftime("%a %b %-d") if os.name != "nt" else local.strftime("%a %b %#d")


def _last_game_result_prefix(game: Dict, feed: Optional[Dict] = None) -> str:
    """Return "Final", "Final/OT", or "Final/SO" for a completed game."""

    def _norm(value: Optional[str]) -> str:
        return value.strip().upper() if isinstance(value, str) else ""

    linescore = (game or {}).get("linescore") or {}
    outcome = (game or {}).get("gameOutcome") or {}

    def _is_shootout(text: str) -> bool:
        return bool(text) and (text == "SO" or "SHOOTOUT" in text)

    def _is_overtime(text: str) -> bool:
        return bool(text) and not _is_shootout(text) and ("OT" in text or "OVERT" in text)

    # Shootout overrides any other period information.
    if linescore.get("hasShootout"):
        return "Final/SO"

    outcome_period = _norm(outcome.get("lastPeriodType"))
    if _is_shootout(outcome_period):
        return "Final/SO"

    # Check for overtime indicators in the schedule payload.
    period_ord = linescore.get("currentPeriodOrdinal")
    period_text = _norm(period_ord) if isinstance(period_ord, str) else ""
    if not period_text and isinstance(period_ord, (int, float)):
        period_text = f"{int(period_ord)}TH"

    if not period_text:
        period = (game or {}).get("period") or {}
        period_text = _norm(period.get("ordinal")) or _norm(period.get("ordinalNum"))
        if not period_text:
            period_text = _norm(period.get("periodType"))

    if not period_text and feed:
        period_text = _norm(feed.get("perOrdinal"))

    if _is_shootout(period_text):
        return "Final/SO"
    if _is_overtime(period_text):
        return "Final/OT"

    if period_text.endswith(("ST", "ND", "RD", "TH")):
        digits = "".join(ch for ch in period_text if ch.isdigit())
        if digits:
            try:
                if int(digits) >= 4:
                    return "Final/OT"
            except ValueError:
                pass

    period_number = linescore.get("currentPeriod")
    if period_number is None:
        period_number = (game or {}).get("period", {}).get("number")
    try:
        if int(period_number) >= 4:
            return "Final/OT"
    except Exception:
        pass

    if _is_overtime(outcome_period):
        return "Final/OT"

    return "Final"


def _format_last_bottom_line(game: Dict, feed: Optional[Dict] = None) -> str:
    prefix = _last_game_result_prefix(game, feed)

    if callable(_MLB_REL_DATE_ONLY):
        official = game.get("officialDate") or (game.get("gameDate") or "")[:10]
        date_str = _MLB_REL_DATE_ONLY(official)
    else:
        date_str = _format_last_date_bottom(game.get("gameDate", ""))

    if date_str:
        return f"{prefix} {date_str}"
    return prefix

# ─────────────────────────────────────────────────────────────────────────────
# Next-game helpers (names, local PNG logos, centered bigger logos)

def _team_full_name(team_like: Dict) -> Optional[str]:
    """Extract a full team name from a 'homeTeam'/'awayTeam' shape."""
    info = _team_obj_from_any(team_like)
    return info.get("name") or info.get("abbrev")

def _format_next_bottom(
    official_date: str,
    game_date_iso: str,
    start_time_central: Optional[str] = None,
) -> str:
    """
    Always include the time:
      "Today 7:30 PM", "Tonight 7:30 PM", "Tomorrow 6:00 PM", or "Wed Sep 24 7:30 PM".
    """
    local = None
    if game_date_iso:
        try:
            local = dt.datetime.fromisoformat(game_date_iso.replace("Z", "+00:00")).astimezone()
        except Exception:
            local = None

    # If the official date is missing, fall back to the localised game date so we
    # always have something for the MLB helper (otherwise it only shows the time).
    official = (official_date or "").strip()
    if not official and local:
        official = local.date().isoformat()

    # Determine a human readable start time we can pass to MLB or use locally.
    start = (start_time_central or "").strip()
    if not start and local:
        try:
            start = local.strftime("%-I:%M %p") if os.name != "nt" else local.strftime("%#I:%M %p")
        except Exception:
            start = ""
    if not start and game_date_iso:
        try:
            dt_utc = dt.datetime.fromisoformat(game_date_iso.replace("Z", "+00:00"))
            start_local = dt_utc.astimezone()
            start = (
                start_local.strftime("%-I:%M %p")
                if os.name != "nt"
                else start_local.strftime("%#I:%M %p")
            )
        except Exception:
            start = ""

    if callable(_MLB_FORMAT_GAME_LABEL):
        return _MLB_FORMAT_GAME_LABEL(official, start)

    if local is None and official:
        try:
            d = dt.datetime.strptime(official[:10], "%Y-%m-%d").date()
            local = dt.datetime.combine(d, dt.time(19, 0)).astimezone()  # default 7pm if time missing
        except Exception:
            local = None

    if not local:
        return ""

    today    = dt.datetime.now().astimezone()
    today_d  = today.date()
    game_d   = local.date()
    time_str = local.strftime("%-I:%M %p") if os.name != "nt" else local.strftime("%#I:%M %p")

    if game_d == today_d:
        return f"Tonight {time_str}" if local.hour >= 18 else f"Today {time_str}"
    if game_d == (today_d + dt.timedelta(days=1)):
        return f"Tomorrow {time_str}"
    # For later dates, include weekday+date **and** time
    date_str = local.strftime("%a %b %-d") if os.name != "nt" else local.strftime("%a %b %#d")
    return f"{date_str} {time_str}"

def _draw_next_card(display, game: Dict, *, title: str, transition: bool=False, log_label: str="hawks next"):
    """
    Next-game card with:
      - Title (MLB font)
      - Opponent line: "@ FULLNAME" or "vs. FULLNAME"
      - Logos row (AWAY @ HOME) centered vertically and larger (local PNGs)
      - Bottom line that always includes game time
    """
    if not isinstance(game, dict):
        logging.warning("%s: missing payload", log_label)
        return None

    # Raw teams (for names); tris for local logo filenames
    raw_away = game.get("awayTeam") or (game.get("teams") or {}).get("away") or {}
    raw_home = game.get("homeTeam") or (game.get("teams") or {}).get("home") or {}
    away_tri, home_tri = _extract_tris_from_game(game)

    away_info = _team_obj_from_any(raw_away)
    home_info = _team_obj_from_any(raw_home)

    is_hawks_away = (away_info.get("id") == TEAM_ID) or ((away_tri or "").upper() == TEAM_TRICODE)
    is_hawks_home = (home_info.get("id") == TEAM_ID) or ((home_tri or "").upper() == TEAM_TRICODE)

    # Build canvas
    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    d   = ImageDraw.Draw(img)

    # Title
    y_top = 2
    title_h = _draw_title_line(img, d, y_top, title, FONT_TITLE)
    y_top += title_h + 1

    # Opponent-only line (full name) with "@"/"vs."
    opp_full = _team_full_name(raw_home if is_hawks_away else raw_away) or (home_tri if is_hawks_away else away_tri)
    prefix   = "@ " if is_hawks_away else "vs. " if is_hawks_home else ""
    opp_line = f"{prefix}{opp_full or '—'}"
    wrapped_h = _center_wrapped_text(d, y_top, opp_line, FONT_NEXT_OPP, max_width=WIDTH - 4)
    y_top += wrapped_h + 1 if wrapped_h else _text_h(d, FONT_NEXT_OPP) + 1

    # Bottom label text (we need its height to avoid overlap)
    official_date = game.get("officialDate") or ""
    game_date_iso = game.get("gameDate") or ""
    start_time_central = game.get("startTimeCentral")
    bottom_text   = _format_next_bottom(official_date, game_date_iso, start_time_central)
    bottom_h      = _text_h(d, FONT_BOTTOM) if bottom_text else 0
    bottom_y      = HEIGHT - (bottom_h + 2) if bottom_text else HEIGHT

    # Desired logo height (bigger on 128px; adapt if smaller/other displays)
    desired_logo_h = standard_next_game_logo_height(HEIGHT)

    # Compute max logo height to fit between the top content and bottom line
    available_h = max(10, bottom_y - (y_top + 2))  # space for logos row
    logo_h = min(desired_logo_h, available_h)
    # Compute a row top such that the logos row is **centered vertically**.
    # But never allow overlap with top content nor with bottom label.
    centered_top = (HEIGHT - logo_h) // 2
    row_y = max(y_top + 1, min(centered_top, bottom_y - logo_h - 1))

    # Render logos at computed height (from local PNGs)
    away_logo = _load_logo_png(away_tri, height=logo_h)
    home_logo = _load_logo_png(home_tri, height=logo_h)

    # Center '@' between logos
    at_txt = "@"
    at_w   = _text_w(d, at_txt, FONT_NEXT_OPP)
    at_h   = _text_h(d, FONT_NEXT_OPP)
    at_x   = (WIDTH - at_w) // 2
    at_y   = row_y + (logo_h - at_h)//2
    d.text((at_x, at_y), at_txt, font=FONT_NEXT_OPP, fill="white")

    # Away logo left of '@'
    if away_logo:
        aw, ah = away_logo.size
        right_limit = at_x - 4
        ax = max(2, right_limit - aw)
        ay = row_y + (logo_h - ah)//2
        img.paste(away_logo, (ax, ay), away_logo)
    else:
        # fallback text
        txt = (away_tri or "AWY")
        tx  = (at_x - 6) // 2 - _text_w(d, txt, FONT_NEXT_OPP)//2
        ty  = row_y + (logo_h - at_h)//2
        d.text((tx, ty), txt, font=FONT_NEXT_OPP, fill="white")

    # Home logo right of '@'
    if home_logo:
        hw, hh = home_logo.size
        left_limit = at_x + at_w + 4
        hx = min(WIDTH - hw - 2, left_limit)
        hy = row_y + (logo_h - hh)//2
        img.paste(home_logo, (hx, hy), home_logo)
    else:
        # fallback text
        txt = (home_tri or "HME")
        tx  = at_x + at_w + ((WIDTH - (at_x + at_w)) // 2) - _text_w(d, txt, FONT_NEXT_OPP)//2
        ty  = row_y + (logo_h - at_h)//2
        d.text((tx, ty), txt, font=FONT_NEXT_OPP, fill="white")

    # Bottom label (always includes time)
    if bottom_text:
        _center_text(d, bottom_y, bottom_text, FONT_BOTTOM)

    return _push(display, img, transition=transition)

# ─────────────────────────────────────────────────────────────────────────────
# Public screens

def draw_last_hawks_game(display, game, transition: bool=False):
    """
    Ignores incoming 'game' and fetches most recent Final to ensure score+SOG.
    """
    sched = fetch_schedule(days_back=30, days_fwd=0)
    if not sched:
        logging.warning("hawks last: no schedule")
        return None
    _, last_final, _ = classify_games(sched)
    if not last_final:
        logging.warning("hawks last: no final found")
        return None

    game_pk = last_final.get("gamePk")
    feed = fetch_game_feed(game_pk) if game_pk else None
    if not feed:
        logging.warning("hawks last: no boxscore/feed")
        return None

    # Build the image
    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    d   = ImageDraw.Draw(img)

    # Title (MLB title font)
    y = 2
    title_h = _draw_title_line(img, d, y, "Last Hawks game:", FONT_TITLE)
    y += title_h

    # Reserve bottom for date (in MLB bottom font)
    bottom_str = _format_last_bottom_line(last_final, feed)
    reserve = (_text_h(d, FONT_BOTTOM) + 2) if bottom_str else 0

    raw_away = last_final.get("awayTeam") or (last_final.get("teams") or {}).get("away") or {}
    raw_home = last_final.get("homeTeam") or (last_final.get("teams") or {}).get("home") or {}
    away_label = _team_scoreboard_label(raw_away, feed.get("awayTri", ""))
    home_label = _team_scoreboard_label(raw_home, feed.get("homeTri", ""))

    # Scoreboard
    _draw_scoreboard(
        img, d, y,
        feed["awayTri"], feed["awayScore"], feed["awaySOG"],
        feed["homeTri"], feed["homeScore"], feed["homeSOG"],
        away_label=away_label,
        home_label=home_label,
        put_sog_label=True,
        bottom_reserved_px=reserve,
    )

    # Bottom date (MLB bottom font)
    if bottom_str:
        by = HEIGHT - _text_h(d, FONT_BOTTOM) - 1
        _center_text(d, by, bottom_str, FONT_BOTTOM)

    return _push(display, img, transition=transition)

def draw_live_hawks_game(display, game, transition: bool=False):
    """
    Ignores incoming 'game' and fetches current live game to ensure score+SOG.
    """
    sched = fetch_schedule(days_back=1, days_fwd=1)
    if not sched:
        logging.warning("hawks live: no schedule")
        return None
    live, _, _ = classify_games(sched)
    if not live:
        logging.info("hawks live: not in progress")
        return None

    game_pk = live.get("gamePk")
    feed = fetch_game_feed(game_pk) if game_pk else None
    if not feed:
        logging.warning("hawks live: no feed")
        return None

    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    d   = ImageDraw.Draw(img)

    dateline = _format_live_dateline(feed)

    # Title (MLB title font) + fallback live clock (TimesSquare small)
    y = 2
    title_h = _draw_title_line(img, d, y, "Hawks Live:", FONT_TITLE)
    y += title_h

    # Only show the inline clock if we don't have a dateline to reserve.
    if not dateline:
        per_inline = _normalize_period(feed.get("perOrdinal"))
        clock_inline = str(feed.get("clock") or "").strip()
        inline = " ".join(val for val in (per_inline, clock_inline) if val).strip()
        if inline:
            _center_text(d, y, inline, FONT_SMALL)
            y += _text_h(d, FONT_SMALL)

    reserve = (_text_h(d, FONT_BOTTOM) + 2) if dateline else 0

    raw_away = live.get("awayTeam") or (live.get("teams") or {}).get("away") or {}
    raw_home = live.get("homeTeam") or (live.get("teams") or {}).get("home") or {}
    away_label = _team_scoreboard_label(raw_away, feed.get("awayTri", ""))
    home_label = _team_scoreboard_label(raw_home, feed.get("homeTri", ""))

    _draw_scoreboard(
        img, d, y,
        feed["awayTri"], feed["awayScore"], feed["awaySOG"],
        feed["homeTri"], feed["homeScore"], feed["homeSOG"],
        away_label=away_label,
        home_label=home_label,
        put_sog_label=True,
        bottom_reserved_px=reserve,
    )

    if dateline:
        by = HEIGHT - _text_h(d, FONT_BOTTOM) - 1
        _center_text(d, by, dateline, FONT_BOTTOM)

    return _push(display, img, transition=transition)

def draw_sports_screen_hawks(display, game, transition: bool=False):
    """
    "Next Hawks game" card with '@ FULLNAME' / 'vs. FULLNAME', logos (local PNGs, centered and larger), and bottom time.
    Uses the provided 'game' payload from your scheduler for the next slot.
    """
    return _draw_next_card(display, game, title="Next Hawks game:", transition=transition, log_label="hawks next")


def draw_hawks_next_home_game(display, game, transition: bool=False):
    """Dedicated "Next at home..." card using the same layout as the next-game screen."""
    return _draw_next_card(display, game, title="Next at home...", transition=transition, log_label="hawks next home")
