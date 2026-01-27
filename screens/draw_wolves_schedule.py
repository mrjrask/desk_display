#!/usr/bin/env python3
"""
draw_wolves_schedule.py

Chicago Wolves (AHL) screens that mirror the Hawks layouts:

- Last Wolves game: compact 2×2 scoreboard (logo+abbr | score)
  * Title: "Last Wolves game:" (uses same title font as mlb_schedule if available)
  * Bottom date: "Yesterday" or "Wed Sep 24" (no year) using the same footer/small font as mlb_schedule if available

- Wolves Live: compact scoreboard (same), optional live clock line.

- Next Wolves game:
  * Title: "Next Wolves game:" (mlb title font)
  * Opponent line: "@ FULL TEAM NAME" (if Wolves are away) or "vs. FULL TEAM NAME" (if Wolves are home)
  * Logos row: AWAY logo  @  HOME logo from local PNGs: images/ahl/{ABBR}.png
    - Logos are centered vertically on the screen and auto-sized larger (up to ~44px on 128px tall panels)
  * Bottom: Always includes time ("Today 7:30 PM", "Tomorrow 6:00 PM", or "Wed Sep 24 7:30 PM")

- Next Wolves home game:
  * Title: "Next at home..."
  * Layout matches the standard next-game card

Function signatures (match main.py):
  - draw_last_wolves_game(display, game, transition=False)
  - draw_sports_screen_wolves(display, game, transition=False)
  - draw_wolves_next_home_game(display, game, transition=False)
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

import config
from config import (
    FONT_DATE_SPORTS,
    FONT_TEAM_SPORTS,
    FONT_TITLE_SPORTS,
    AHL_FALLBACK_LOGO,
    AHL_IMAGES_DIR,
    AHL_TEAM_ID,
    AHL_TEAM_NAME,
    AHL_TEAM_TRICODE,
    TIMES_SQUARE_FONT_PATH,
    WIDTH,
    HEIGHT,
)
from utils import (
    LED_INDICATOR_LEVEL,
    ScreenImage,
    standard_next_game_logo_height,
    temporary_display_led,
)

TS_PATH = TIMES_SQUARE_FONT_PATH
AHL_DIR = AHL_IMAGES_DIR
BACKGROUND_COLOR = (0, 0, 0)

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

TEAM_ID      = AHL_TEAM_ID
TEAM_TRICODE = (AHL_TEAM_TRICODE or "CHI").upper()

# Map full team names (as returned by the ICS feed) to the
# lowercase abbreviations used in ``images/ahl``.
# Keys are stored as lowercase strings stripped of punctuation for
# reliable lookups even if the feed varies the formatting.
_TEAM_NAME_TO_ABBR = {
    "abbotsford canucks": "ABB",
    "ahl": "AHL",
    "ahl all star": "AHL",
    "ahl all star challenge": "AHL",
    "ahl all star classic": "AHL",
    "ahl all star game": "AHL",
    "ahl all star skills competition": "AHL",
    "ahl skills competition": "AHL",
    "american hockey league": "AHL",
    "bakersfield condors": "BAK",
    "belleville senators": "BEL",
    "bridgeport islanders": "BRI",
    "calgary wranglers": "CGY",
    "charlotte checkers": "CLT",
    "chicago wolves": "CHI",
    "cleveland monsters": "CLE",
    "coachella valley firebirds": "CV",
    "colorado eagles": "COL",
    "grand rapids griffins": "GR",
    "hartford wolf pack": "HFD",
    "henderson silver knights": "HSK",
    "hershey bears": "HER",
    "iowa wild": "IA",
    "laval rocket": "LAV",
    "lehigh valley phantoms": "LV",
    "manitoba moose": "MM",
    "milwaukee admirals": "MIL",
    "ontario reign": "ONT",
    "providence bruins": "PRO",
    "rockford icehogs": "RFD",
    "rochester americans": "ROC",
    "san diego gulls": "SD",
    "san jose barracuda": "SJ",
    "springfield thunderbirds": "SPR",
    "syracuse crunch": "SYR",
    "texas stars": "TEX",
    "toronto marlies": "TOR",
    "tucson roadrunners": "TUC",
    "utica comets": "UTC",
    "wilkes barre scranton penguins": "WBS",
}


def _normalize_team_key(name: Optional[str]) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _map_team_name_to_abbr(*names: Optional[str]) -> Optional[str]:
    for candidate in names:
        key = _normalize_team_key(candidate)
        if not key:
            continue
        mapped = _TEAM_NAME_TO_ABBR.get(key)
        if mapped:
            return mapped.upper()
    return None


_CUSTOM_TEAM_NAME = _normalize_team_key(AHL_TEAM_NAME)
if _CUSTOM_TEAM_NAME and TEAM_TRICODE:
    _TEAM_NAME_TO_ABBR.setdefault(_CUSTOM_TEAM_NAME, TEAM_TRICODE)

# ─────────────────────────────────────────────────────────────────────────────
# Display helpers

def _clear_display(display):
    try:
        from utils import clear_display  # in your repo
        clear_display(display)
    except Exception:
        pass

def _push(
    display,
    img: Optional[Image.Image],
    *,
    transition: bool = False,
    led_override: Optional[Tuple[float, float, float]] = None,
):
    if img is None or display is None:
        return None
    if transition:
        return ScreenImage(img, displayed=False, led_override=led_override)

    def _show_image() -> None:
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

    if led_override is not None:
        with temporary_display_led(*led_override):
            _show_image()
    else:
        _show_image()
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Team + logo helpers (local PNGs)

FALLBACK_LOGO = AHL_FALLBACK_LOGO

def _team_obj_from_any(t: Dict) -> Dict:
    """Return team dict with {'abbrev','id','name','nickname'} from AHL structure."""
    if not isinstance(t, dict):
        return {}

    # AHL data structure uses 'abbr' not 'abbrev'
    abbr = t.get("abbr") or t.get("abbrev") or t.get("code")
    if isinstance(abbr, str):
        abbr = abbr.strip()
    tid  = t.get("id") or t.get("teamId") or t.get("team_id")
    name = t.get("name") or t.get("fullName")
    nickname = t.get("nickname") or t.get("shortName")

    resolved_abbr = abbr or _map_team_name_to_abbr(name, nickname)
    if isinstance(resolved_abbr, str):
        resolved_abbr = resolved_abbr.upper()

    return {"abbrev": resolved_abbr, "id": tid, "name": name, "nickname": nickname}

def _extract_tris_from_game(game: Dict) -> Tuple[str, str]:
    """(away_tri, home_tri) from a game-like dict."""
    away = game.get("away") or {}
    home = game.get("home") or {}
    a = _team_obj_from_any(away).get("abbrev") or "AWAY"
    h = _team_obj_from_any(home).get("abbrev") or "HOME"
    return a, h

def _load_logo_png(abbr: str, height: int) -> Optional[Image.Image]:
    """Load team logo from local repo PNG: images/ahl/{ABBR}.png; fallback AHL.png."""
    if not abbr:
        abbr = "AHL"

    # Try case-insensitive lookup: uppercase first, then lowercase
    for variant in [abbr.upper(), abbr.lower()]:
        png_path = os.path.join(AHL_DIR, f"{variant}.png")
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
    if not s:
        return 0
    try:
        l, t, r, b = d.textbbox((0, 0), s, font=font)
        tw, th = r - l, b - t
        tx = (WIDTH - tw) // 2 - l
        ty = y - t
    except Exception:
        tw, th = d.textsize(s, font=font)
        tx = (WIDTH - tw) // 2
        ty = y
    d.text((tx, ty), s, font=font, fill="white")
    return th


def _center_bottom_text(
    d: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    *,
    margin: int = 2,
    fill: str = "white",
):
    if not text:
        return 0
    try:
        l, t, r, b = d.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
        tx = (WIDTH - tw) // 2 - l
        ty = HEIGHT - th - margin - t
    except Exception:
        tw, th = d.textsize(text, font=font)
        tx = (WIDTH - tw) // 2
        ty = HEIGHT - th - margin
    d.text((tx, ty), text, font=font, fill=fill)
    return th


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
    """Prefer short team names for the scoreboard column."""
    if not isinstance(team_like, dict):
        return fallback

    abbr = team_like.get("abbr") or team_like.get("abbrev")
    if abbr and isinstance(abbr, str) and abbr.strip():
        return abbr.strip().upper()

    # AHL data structure - prefer nickname over name when no abbr exists
    nickname = team_like.get("nickname")
    if nickname and isinstance(nickname, str) and nickname.strip():
        return nickname.strip()

    name = team_like.get("name")
    if name and isinstance(name, str) and name.strip():
        return name.strip()

    return fallback


def _draw_scoreboard(
    img: Image.Image,
    d: ImageDraw.ImageDraw,
    top_y: int,
    away_tri: str,
    away_score: Optional[int],
    home_tri: str,
    home_score: Optional[int],
    *,
    away_label: Optional[str] = None,
    home_label: Optional[str] = None,
    bottom_reserved_px: int = 0,
) -> int:
    """Draw a compact 2×2 scoreboard. Returns bottom y."""
    # Column widths: first column dominates for logo + name, second column for score
    col1_w = min(WIDTH - 32, max(84, int(WIDTH * 0.72)))
    col2_w = WIDTH - col1_w
    x0, x1, x2 = 0, col1_w, WIDTH

    y = top_y

    header_h = 0
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

    def _prepare_row(
        row_top: int,
        row_height: int,
        tri: str,
        score: Optional[int],
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
            "base_text": text,
            "logo": logo,
            "max_width": max_width,
        }

    row_specs = [
        _prepare_row(row1_top, row1_h, away_tri, away_score, away_label),
        _prepare_row(split_y, row2_h, home_tri, home_score, home_label),
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


def _format_live_dateline(game: Dict) -> str:
    """Format live game status line from AHL data structure."""
    status = game.get("status") or {}
    period = _normalize_period(status.get("period"))
    clock = str(status.get("clock") or "").strip()
    note = str(status.get("note") or "").strip()

    if note and "intermission" in note.lower():
        state = note.title() if note.isupper() else note
        if period:
            return f"{state} ({period})"
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


def _last_game_result_prefix(game: Dict) -> str:
    """Return "Final", "Final/OT", or "Final/SO" for a completed game."""
    status = game.get("status") or {}
    detail = str(status.get("detail") or "").strip().upper()
    period = str(status.get("period") or "").strip().upper()

    # Check for shootout
    if "SO" in detail or "SHOOTOUT" in detail or "SO" in period or "SHOOTOUT" in period:
        return "Final/SO"

    # Check for overtime
    if "OT" in detail or "OVERTIME" in detail or "OT" in period or "OVERTIME" in period:
        return "Final/OT"

    # Check period number
    if period.isdigit() and int(period) >= 4:
        return "Final/OT"

    return "Final"


def _format_last_bottom_line(game: Dict) -> str:
    prefix = _last_game_result_prefix(game)

    if callable(_MLB_REL_DATE_ONLY):
        official = game.get("official_date") or ""
        date_str = _MLB_REL_DATE_ONLY(official)
    else:
        date_str = _format_last_date_bottom(game.get("start_iso", ""))

    parts = [p for p in (prefix, date_str) if p]
    return " • ".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# Next-game helpers (names, local PNG logos, centered bigger logos)

def _team_full_name(team_like: Dict) -> Optional[str]:
    """Extract a full team name from AHL team structure."""
    info = _team_obj_from_any(team_like)
    return info.get("name") or info.get("nickname") or info.get("abbrev")

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
        formatted = (_MLB_FORMAT_GAME_LABEL(official, start) or "").strip()
        return formatted

    if local is None and official:
        try:
            d = dt.datetime.strptime(official[:10], "%Y-%m-%d").date()
            local = dt.datetime.combine(d, dt.time(19, 0)).astimezone()  # default 7pm if time missing
        except Exception:
            local = None

    if not local:
        return ""

    today = dt.datetime.now().astimezone()
    today_d = today.date()
    game_d = local.date()
    time_str = (
        local.strftime("%-I:%M %p") if os.name != "nt" else local.strftime("%#I:%M %p")
    )

    if game_d == today_d:
        label = "Tonight" if local.hour >= 18 else "Today"
    elif game_d == (today_d + dt.timedelta(days=1)):
        label = "Tomorrow"
    else:
        label = (
            local.strftime("%a %b %-d")
            if os.name != "nt"
            else local.strftime("%a %b %#d")
        )

    parts = [p for p in (label, time_str) if p]
    return " • ".join(parts)

def _draw_next_card(
    display,
    game: Dict,
    *,
    title: str,
    transition: bool = False,
    log_label: str = "wolves next",
    logo_scale: float = 1.0,
):
    """
    Next-game card with:
      - Title (MLB font)
      - Opponent line: "@ FULLNAME" or "vs. FULLNAME"
      - Logos row (AWAY @ HOME) centered vertically and larger (local PNGs)
      - Bottom line that always includes game time

    ``logo_scale`` allows callers to shrink or enlarge
    the computed standard logo height without changing the helpers.
    """
    if not isinstance(game, dict):
        logging.warning("%s: missing payload", log_label)
        return None

    # Raw teams (for names); tris for local logo filenames
    raw_away = game.get("away") or {}
    raw_home = game.get("home") or {}
    away_tri, home_tri = _extract_tris_from_game(game)

    away_info = _team_obj_from_any(raw_away)
    home_info = _team_obj_from_any(raw_home)

    is_wolves_away = (str(away_info.get("id")) == str(TEAM_ID)) or ((away_tri or "").upper() == TEAM_TRICODE)
    is_wolves_home = (str(home_info.get("id")) == str(TEAM_ID)) or ((home_tri or "").upper() == TEAM_TRICODE)

    # Build canvas
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    d   = ImageDraw.Draw(img)

    # Title
    y_top = 2
    title_h = _draw_title_line(img, d, y_top, title, FONT_TITLE)
    y_top += title_h + 1

    # Opponent-only line (full name) with "@"/"vs."
    opp_full = _team_full_name(raw_home if is_wolves_away else raw_away) or (home_tri if is_wolves_away else away_tri)
    prefix   = "@ " if is_wolves_away else "vs. " if is_wolves_home else ""
    opp_line = f"{prefix}{opp_full or '—'}"
    wrapped_h = _center_wrapped_text(d, y_top, opp_line, FONT_NEXT_OPP, max_width=WIDTH - 4)
    y_top += wrapped_h + 1 if wrapped_h else _text_h(d, FONT_NEXT_OPP) + 1

    # Bottom label text (we need its height to avoid overlap)
    official_date = game.get("official_date") or ""
    game_date_iso = game.get("start_iso") or ""
    start_time_central = game.get("start_time_central")
    bottom_text = _format_next_bottom(official_date, game_date_iso, start_time_central)
    bottom_h = _text_h(d, FONT_BOTTOM) if bottom_text else 0
    bottom_y = HEIGHT - (bottom_h + 2) if bottom_text else HEIGHT

    # Desired logo height (bigger on 128px; adapt if smaller/other displays)
    clamped_scale = max(0.5, min(float(logo_scale or 1.0), 1.2))
    desired_logo_h = max(1, int(round(standard_next_game_logo_height(HEIGHT) * clamped_scale)))

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
        _center_bottom_text(d, bottom_text, FONT_BOTTOM)

    return _push(display, img, transition=transition)

# ─────────────────────────────────────────────────────────────────────────────
# Public screens

def draw_last_wolves_game(display, game, transition: bool=False):
    global BACKGROUND_COLOR
    BACKGROUND_COLOR = config.get_screen_background_color("wolves last", (0, 0, 0))
    """
    Display the last completed Wolves game with scoreboard.
    Expects AHL data structure with away/home dicts containing abbr, score, shots.
    """
    if not isinstance(game, dict):
        logging.warning("wolves last: missing game data")
        return None

    # Build the image
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    d   = ImageDraw.Draw(img)

    # Title (MLB title font)
    y = 2
    title_h = _draw_title_line(img, d, y, "Last Wolves game:", FONT_TITLE)
    y += title_h

    # Reserve bottom for date (in MLB bottom font)
    bottom_str = _format_last_bottom_line(game)
    reserve = (_text_h(d, FONT_BOTTOM) + 2) if bottom_str else 0

    raw_away = game.get("away") or {}
    raw_home = game.get("home") or {}
    away_info = _team_obj_from_any(raw_away)
    home_info = _team_obj_from_any(raw_home)

    away_tri = (away_info.get("abbrev") or "AWY").upper()
    home_tri = (home_info.get("abbrev") or "HME").upper()
    away_score = raw_away.get("score")
    home_score = raw_home.get("score")

    away_label = _team_scoreboard_label(away_info, away_tri)
    home_label = _team_scoreboard_label(home_info, home_tri)

    # Scoreboard
    _draw_scoreboard(
        img, d, y,
        away_tri, away_score,
        home_tri, home_score,
        away_label=away_label,
        home_label=home_label,
        bottom_reserved_px=reserve,
    )

    # LED indicator logic
    led_override: Optional[Tuple[float, float, float]] = None

    wolves_home = str(home_info.get("id")) == str(TEAM_ID)
    wolves_score = home_score if wolves_home else away_score
    opp_score = away_score if wolves_home else home_score

    if isinstance(wolves_score, int) and isinstance(opp_score, int) and wolves_score != opp_score:
        if wolves_score > opp_score:
            led_override = (0.0, LED_INDICATOR_LEVEL, 0.0)  # Green for win
        else:
            result_label = _last_game_result_prefix(game)
            is_ot_loss = (
                isinstance(result_label, str)
                and "OT" in result_label.upper()
                and "SO" not in result_label.upper()
            )
            led_override = (
                (LED_INDICATOR_LEVEL, LED_INDICATOR_LEVEL, 0.0)  # Yellow for OT loss
                if is_ot_loss
                else (LED_INDICATOR_LEVEL, 0.0, 0.0)  # Red for regulation loss
            )

    # Bottom date (MLB bottom font)
    if bottom_str:
        _center_bottom_text(d, bottom_str, FONT_BOTTOM)

    return _push(display, img, transition=transition, led_override=led_override)

def draw_live_wolves_game(display, game, transition: bool=False):
    global BACKGROUND_COLOR
    BACKGROUND_COLOR = config.get_screen_background_color("wolves live", (0, 0, 0))
    """
    Display a live Wolves game with scoreboard and live status.
    Expects AHL data structure with away/home dicts containing abbr and score.
    """
    if not isinstance(game, dict):
        logging.warning("wolves live: missing game data")
        return None

    # Build the image
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    d   = ImageDraw.Draw(img)

    # Title (MLB title font)
    y = 2
    title_h = _draw_title_line(img, d, y, "Wolves Live:", FONT_TITLE)
    y += title_h

    # Live status (period and clock)
    dateline = _format_live_dateline(game)
    if dateline:
        _center_text(d, y, dateline, FONT_SMALL)
        y += _text_h(d, FONT_SMALL)

    # Reserve bottom for live status if present
    reserve = (_text_h(d, FONT_BOTTOM) + 2) if dateline else 0

    raw_away = game.get("away") or {}
    raw_home = game.get("home") or {}
    away_info = _team_obj_from_any(raw_away)
    home_info = _team_obj_from_any(raw_home)

    away_tri = (away_info.get("abbrev") or "AWY").upper()
    home_tri = (home_info.get("abbrev") or "HME").upper()
    away_score = raw_away.get("score")
    home_score = raw_home.get("score")

    away_label = _team_scoreboard_label(away_info, away_tri)
    home_label = _team_scoreboard_label(home_info, home_tri)

    # Scoreboard
    _draw_scoreboard(
        img, d, y,
        away_tri, away_score,
        home_tri, home_score,
        away_label=away_label,
        home_label=home_label,
        bottom_reserved_px=reserve,
    )

    # Bottom status line if available
    if dateline:
        _center_bottom_text(d, dateline, FONT_BOTTOM)

    return _push(display, img, transition=transition)

def draw_sports_screen_wolves(display, game, transition: bool=False):
    global BACKGROUND_COLOR
    BACKGROUND_COLOR = config.get_screen_background_color("wolves next", (0, 0, 0))
    """
    "Next Wolves game" card with '@ FULLNAME' / 'vs. FULLNAME', logos (local PNGs, centered and larger), and bottom time.
    Uses the provided 'game' payload from your scheduler for the next slot.
    """
    return _draw_next_card(display, game, title="Next Wolves game:", transition=transition, log_label="wolves next", logo_scale=0.8)


def draw_wolves_next_home_game(display, game, transition: bool=False):
    global BACKGROUND_COLOR
    BACKGROUND_COLOR = config.get_screen_background_color("wolves next home", (0, 0, 0))
    """Dedicated "Next at home..." card using the same layout as the next-game screen."""
    return _draw_next_card(display, game, title="Next at home...", transition=transition, log_label="wolves next home", logo_scale=0.8)
