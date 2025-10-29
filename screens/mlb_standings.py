#!/usr/bin/env python3
"""
mlb_standings.py

Draw MLB division standings, overview, and Wild Card screens in RGB with:
- Drop-in animations on Overview (last place teams fall in first)
- Proper GB / WCGB formatting
- Wild Card screen scrolls bottom → top
"""

import os
import time
import requests
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw

import config
from config import (
    SCOREBOARD_SCROLL_STEP,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
)
from utils import clear_display, get_mlb_abbreviation, log_call
from screens.mlb_team_standings import format_games_back

# ─── Fonts / geometry from config ────────────────────────────────────────────
WIDTH  = config.WIDTH
HEIGHT = config.HEIGHT
FONT_DIV_HEADER = config.FONT_DIV_HEADER
FONT_DIV_RECORD = config.FONT_DIV_RECORD
FONT_GB_VALUE   = config.FONT_GB_VALUE
FONT_GB_LABEL   = config.FONT_GB_LABEL

# ─── Tunables ────────────────────────────────────────────────────────────────
LOGO_SIZE   = 52      # max width/height of a division logo
MARGIN      = 6       # left/right gutter
ROW_SPACING = 6       # vertical gap between rows

OV_COLS = 3           # East, Central, West columns on Overview
OV_ROWS = 5           # max teams to show per division on Overview
OVERVIEW_DROP_STEPS = 30
OVERVIEW_DROP_STAGGER = 0.4  # fraction of steps before next rank begins dropping
OVERVIEW_DROP_FRAME_DELAY = 0.02

LEAGUE_DIVISION_IDS: Dict[int, Dict[str, int]] = {
    104: {"East": 204, "Central": 205, "West": 203},  # National League
    103: {"East": 201, "Central": 202, "West": 200},  # American League
}

# Logos live in the shared images directory alongside config.py. Using the
# config.IMAGES_DIR constant keeps the standings screen aligned with the other
# MLB screens (schedule, scoreboard, etc.) and avoids looking for a nonexistent
# ./screens/images/mlb folder.
LOGOS_DIR = os.path.join(config.IMAGES_DIR, "mlb")
TIMEOUT   = 10
BACKGROUND_COLOR = config.SCOREBOARD_BACKGROUND_COLOR


# ─────────────────────────────────────────────────────────────────────────────
# Data fetchers
# ─────────────────────────────────────────────────────────────────────────────

def _sort_by_int_key(items: List[dict], key: str) -> List[dict]:
    def _k(x):  # MLB API sends ranks as strings
        try:
            return int(x.get(key, 999))
        except Exception:
            return 999
    return sorted(items, key=_k)

def fetch_division_records(league_id: int, division_id: int) -> List[dict]:
    """
    Return teamRecords for a given league+division, sorted by divisionRank (1..N).
    """
    url = (
        "https://statsapi.mlb.com/api/v1/standings"
        f"?season=2025&leagueId={league_id}&divisionId={division_id}"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        records = r.json().get("records", [])
        rec = next(
            (x for x in records if x.get("division", {}).get("id") == division_id),
            None
        )
        if not rec:
            return []
        teams = rec.get("teamRecords", []) or []
        return _sort_by_int_key(teams, "divisionRank")
    except Exception as e:
        logging.error(f"Fetch standings L{league_id} D{division_id} failed: {e}")
        return []

def fetch_wildcard_records(league_id: int) -> List[dict]:
    """
    Return teamRecords for league Wild Card, sorted by wildCardRank (1..N).
    """
    url = (
        "https://statsapi.mlb.com/api/v1/standings"
        f"?season=2025&leagueId={league_id}&standingsTypes=wildCard"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json().get("records", [])
        teams = (data[0].get("teamRecords", []) if data else []) or []
        return _sort_by_int_key(teams, "wildCardRank")
    except Exception as e:
        logging.error(f"Fetch wildcard standings L{league_id} failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_logo(abbr: str, target: int) -> Optional[Image.Image]:
    """
    Load a team logo (PNG) and resize to fit within target×target box.
    """
    fn = f"{abbr.upper()}.png"
    path = os.path.join(LOGOS_DIR, fn)
    if not os.path.exists(path):
        logging.warning(f"Logo missing: {fn}")
        return None
    try:
        img = Image.open(path).convert("RGBA")
        w0, h0 = img.size
        s = min(target / w0, target / h0)
        return img.resize((max(1, int(w0*s)), max(1, int(h0*s))), Image.LANCZOS)
    except Exception as e:
        logging.warning(f"Logo load error {fn}: {e}")
        return None

def _header_frame(title: str) -> Tuple[Image.Image, int]:
    """
    Create a header-only frame and return (image, header_height).
    """
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    d = ImageDraw.Draw(img)
    tw, th = d.textsize(title, font=FONT_DIV_HEADER)
    d.text(((WIDTH - tw)//2, 0), title, font=FONT_DIV_HEADER, fill=(255,255,255))
    return img, th + 6


def _ease_out_cubic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    inv = 1.0 - t
    return 1.0 - inv * inv * inv


# ─────────────────────────────────────────────────────────────────────────────
# Overview (drop-in animation; last place drops first)
# ─────────────────────────────────────────────────────────────────────────────

@log_call
def draw_overview(display, title: str, league_id: int, transition=False):
    """
    Animated overview showing 3 columns (East, Central, West). Each column drops
    logos from last place up to first, onto a header-only background.
    """
    divisions = ["East", "Central", "West"]

    # Load logos per division in standings order (1..N), trimmed to OV_ROWS
    logos_per_div: Dict[str, List[Optional[Image.Image]]] = {}
    for div in divisions:
        div_id = LEAGUE_DIVISION_IDS[league_id][div]
        recs = fetch_division_records(league_id, div_id)[:OV_ROWS]
        logos: List[Optional[Image.Image]] = []
        for rec in recs:
            abbr = get_mlb_abbreviation(rec["team"]["name"])
            logos.append(_load_logo(abbr, LOGO_SIZE))
        # ensure length OV_ROWS (pad with None if short)
        while len(logos) < OV_ROWS:
            logos.append(None)
        logos_per_div[div] = logos

    # Header-only base
    header, top_y = _header_frame(title)
    cell_h = (HEIGHT - top_y) // OV_ROWS
    col_w  = LOGO_SIZE
    margin_x = (WIDTH - OV_COLS * col_w) // (OV_COLS + 1)
    x_cols = [margin_x*(i+1) + col_w*i for i in range(OV_COLS)]

    row_positions: List[List[Tuple[Image.Image, int, int]]] = []
    for rank in range(OV_ROWS):
        placements: List[Tuple[Image.Image, int, int]] = []
        for ci, div in enumerate(divisions):
            ic = logos_per_div[div][rank]
            if not ic:
                continue
            x0 = x_cols[ci] + (col_w - ic.width)//2
            y_target = top_y + rank * cell_h + (cell_h - ic.height)//2
            placements.append((ic, x0, y_target))
        row_positions.append(placements)

    steps = max(2, OVERVIEW_DROP_STEPS)
    stagger = max(1, int(round(steps * OVERVIEW_DROP_STAGGER)))

    schedule: List[Tuple[int, List[Tuple[Image.Image, int, int]]]] = []
    start_step = 0
    for rank in range(len(row_positions) - 1, -1, -1):
        drops = row_positions[rank]
        if not drops:
            continue
        schedule.append((start_step, drops))
        start_step += stagger

    if schedule:
        total_duration = schedule[-1][0] + steps + 1
        placed: List[Tuple[Image.Image, int, int]] = []
        completed = [False] * len(schedule)

        for current_step in range(total_duration):
            for idx, (start, drops) in enumerate(schedule):
                if current_step >= start + steps and not completed[idx]:
                    placed.extend(drops)
                    completed[idx] = True

            frame = header.copy()
            for ic, x0, y0 in placed:
                frame.paste(ic, (x0, y0), ic)

            for idx, (start, drops) in enumerate(schedule):
                progress = current_step - start
                if progress < 0 or progress >= steps:
                    continue

                frac = progress / (steps - 1) if steps > 1 else 1.0
                eased = _ease_out_cubic(frac)
                for ic, x0, y_target in drops:
                    start_y = -LOGO_SIZE
                    y_pos = int(start_y + (y_target - start_y) * eased)
                    if y_pos > y_target:
                        y_pos = y_target
                    frame.paste(ic, (x0, y_pos), ic)

            display.image(frame)
            display.show()
            time.sleep(OVERVIEW_DROP_FRAME_DELAY)

    # Final static image
    final = header.copy()
    for ri in range(OV_ROWS):
        for ci, div in enumerate(divisions):
            ic = logos_per_div[div][ri]
            if ic:
                x0 = x_cols[ci] + (col_w - ic.width)//2
                y0 = top_y + ri * cell_h + (cell_h - ic.height)//2
                final.paste(ic, (x0, y0), ic)

    display.image(final)
    display.show()
    time.sleep(PAUSE_END)

    return final if transition else None


# ─────────────────────────────────────────────────────────────────────────────
# Division screen (static header + vertical scroll of rows)
# ─────────────────────────────────────────────────────────────────────────────

@log_call
def draw_division_screen(display, league_id: int, division_id: int, title: str, transition=False):
    teams = fetch_division_records(league_id, division_id)
    if not teams:
        clear_display(display)
        return None

    header, header_h = _header_frame(title)

    # Build the list canvas (all rows)
    row_h  = LOGO_SIZE + ROW_SPACING
    list_h = row_h * len(teams)
    canvas = Image.new("RGB", (WIDTH, list_h), BACKGROUND_COLOR)
    cd     = ImageDraw.Draw(canvas)

    for i, rec in enumerate(teams):
        y = i * row_h

        # Logo
        abbr = get_mlb_abbreviation(rec["team"]["name"])
        ic = _load_logo(abbr, LOGO_SIZE)
        if ic:
            logo_x = MARGIN + (LOGO_SIZE - ic.width)//2
            logo_y = y + (LOGO_SIZE - ic.height)//2
            canvas.paste(ic, (logo_x, logo_y), ic)

        # GB column (right-aligned, label "GB")
        dgb = rec.get("divisionGamesBack", "-")
        gb_val = format_games_back(dgb) if dgb != "-" else "--"
        num_w, num_h = cd.textsize(gb_val, font=FONT_GB_VALUE)
        lab_w, lab_h = cd.textsize("GB",   font=FONT_GB_LABEL)
        gb_x = WIDTH - MARGIN - (num_w + lab_w)
        gb_y = y + (LOGO_SIZE - num_h)//2
        cd.text((gb_x, gb_y), gb_val, font=FONT_GB_VALUE, fill=(255,255,255))
        cd.text((gb_x + num_w, gb_y + (num_h - lab_h)//2), "GB", font=FONT_GB_LABEL, fill=(255,255,255))

        # W-L centered between logo block and GB
        wins = rec["leagueRecord"]["wins"]
        loss = rec["leagueRecord"]["losses"]
        rec_txt = f"{wins}-{loss}"
        rw2, rh2 = cd.textsize(rec_txt, font=FONT_DIV_RECORD)
        left  = MARGIN + LOGO_SIZE + MARGIN
        right = gb_x - MARGIN
        rec_x = left + ((right - left) - rw2)//2
        rec_y = y + (LOGO_SIZE - rh2)//2
        cd.text((rec_x, rec_y), rec_txt, font=FONT_DIV_RECORD, fill=(255,255,255))

    # Show first slice
    slice_first = canvas.crop((0, 0, WIDTH, HEIGHT - header_h))
    frame = header.copy()
    frame.paste(slice_first, (0, header_h))
    display.image(frame)
    display.show()

    visible_h = HEIGHT - header_h
    max_offset = max(0, list_h - visible_h)

    time.sleep(SCOREBOARD_SCROLL_PAUSE_TOP)
    if max_offset <= 0:
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
        return None

    for off in range(
        SCOREBOARD_SCROLL_STEP,
        max_offset + SCOREBOARD_SCROLL_STEP,
        SCOREBOARD_SCROLL_STEP,
    ):
        offset = min(off, max_offset)
        f2 = header.copy()
        part = canvas.crop((0, offset, WIDTH, offset + visible_h))
        f2.paste(part, (0, header_h))
        display.image(f2)
        display.show()
        time.sleep(SCOREBOARD_SCROLL_DELAY)

    time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Wild Card (bottom → top scroll; WCGB column)
# ─────────────────────────────────────────────────────────────────────────────

@log_call
def draw_wildcard_screen(display, league_id: int, title: str, transition=False):
    teams = fetch_wildcard_records(league_id)
    if not teams:
        clear_display(display)
        return None

    header, header_h = _header_frame(title)

    row_h  = LOGO_SIZE + ROW_SPACING
    list_h = row_h * len(teams)
    canvas = Image.new("RGB", (WIDTH, list_h), BACKGROUND_COLOR)
    cd     = ImageDraw.Draw(canvas)

    for i, rec in enumerate(teams):
        y = i * row_h

        # Team logo
        abbr = get_mlb_abbreviation(rec["team"]["name"])
        ic = _load_logo(abbr, LOGO_SIZE)
        if ic:
            canvas.paste(ic, (MARGIN + (LOGO_SIZE - ic.width)//2,
                              y + (LOGO_SIZE - ic.height)//2), ic)

        # WCGB formatting
        raw_wcb = rec.get("wildCardGamesBack")
        try:
            wc_val = float(raw_wcb)
        except Exception:
            wc_val = None

        base = format_games_back(raw_wcb) if raw_wcb is not None else "--"
        if wc_val is None or wc_val == 0:
            s = "--"
        elif i < 3:
            s = f"+{base}"
        else:
            s = base

        # Right column labeled WCGB
        nw, nh = cd.textsize(s, font=FONT_GB_VALUE)
        lw, lh = cd.textsize("WCGB", font=FONT_GB_LABEL)
        start_x = WIDTH - MARGIN - (nw + lw)
        y_text  = y + (LOGO_SIZE - nh)//2
        cd.text((start_x, y_text), s, font=FONT_GB_VALUE, fill=(255,255,255))
        cd.text((start_x + nw, y + (LOGO_SIZE - lh)//2), "WCGB", font=FONT_GB_LABEL, fill=(255,255,255))

        # W-L centered between logo block and WCGB
        rw, rl = rec["leagueRecord"]["wins"], rec["leagueRecord"]["losses"]
        rt = f"{rw}-{rl}"
        tw2, th2 = cd.textsize(rt, font=FONT_DIV_RECORD)
        left  = MARGIN + LOGO_SIZE + MARGIN
        right = start_x - MARGIN
        rec_x = left + ((right - left) - tw2)//2
        cd.text((rec_x, y + (LOGO_SIZE - th2)//2), rt, font=FONT_DIV_RECORD, fill=(255,255,255))

        # Separator below 3rd team (between #3 and #4): green line
        if i == 2:
            sep_y = y + row_h - ROW_SPACING // 2
            cd.line((MARGIN, sep_y, WIDTH - MARGIN, sep_y), fill=(0, 255, 0))

    # Reverse scroll bottom → top
    visible_h = HEIGHT - header_h
    start_off = max(0, list_h - visible_h)
    first = header.copy()
    first_slice = canvas.crop((0, start_off, WIDTH, start_off + visible_h))
    first.paste(first_slice, (0, header_h))
    display.image(first)
    display.show()

    time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    if start_off <= 0:
        time.sleep(SCOREBOARD_SCROLL_PAUSE_TOP)
        return None

    for off in range(
        start_off - SCOREBOARD_SCROLL_STEP,
        -SCOREBOARD_SCROLL_STEP,
        -SCOREBOARD_SCROLL_STEP,
    ):
        offset = max(0, off)
        f2 = header.copy()
        part = canvas.crop((0, offset, WIDTH, offset + visible_h))
        f2.paste(part, (0, header_h))
        display.image(f2)
        display.show()
        time.sleep(SCOREBOARD_SCROLL_DELAY)

    time.sleep(SCOREBOARD_SCROLL_PAUSE_TOP)
    return None


# ─── Wrappers expected by main.py ────────────────────────────────────────────

@log_call
def draw_NL_Overview(display, transition=False):
    return draw_overview(display, "NL Overview", 104, transition)

@log_call
def draw_AL_Overview(display, transition=False):
    return draw_overview(display, "AL Overview", 103, transition)

@log_call
def draw_NL_East(display, transition=False):
    return draw_division_screen(display, 104, 204, "NL East", transition)

@log_call
def draw_NL_Central(display, transition=False):
    return draw_division_screen(display, 104, 205, "NL Central", transition)

@log_call
def draw_NL_West(display, transition=False):
    return draw_division_screen(display, 104, 203, "NL West", transition)

@log_call
def draw_AL_East(display, transition=False):
    return draw_division_screen(display, 103, 201, "AL East", transition)

@log_call
def draw_AL_Central(display, transition=False):
    return draw_division_screen(display, 103, 202, "AL Central", transition)

@log_call
def draw_AL_West(display, transition=False):
    return draw_division_screen(display, 103, 200, "AL West", transition)

@log_call
def draw_NL_WildCard(display, transition=False):
    return draw_wildcard_screen(display, 104, "NL Wild Card", transition)

@log_call
def draw_AL_WildCard(display, transition=False):
    return draw_wildcard_screen(display, 103, "AL Wild Card", transition)
