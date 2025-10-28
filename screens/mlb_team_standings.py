#!/usr/bin/env python3
"""
mlb_team_standings.py

Draw MLB team standings screens 1 & 2 in RGB.
Screen 1: logo at top center, then W-L, rank, GB, WCGB with:
  - “--” for 0 WCGB
  - “+n” for any of the top-3 wild card slots when WCGB > 0
  - “n” for everyone else
Screen 2: logo at top center, then overall record and splits.
"""
import os
import time
from PIL import Image, ImageDraw
from config import (
    WIDTH,
    HEIGHT,
    TEAM_STANDINGS_DISPLAY_SECONDS,
    FONT_STAND1_WL,
    FONT_STAND1_RANK,
    FONT_STAND1_GB_LABEL,
    FONT_STAND1_GB_VALUE,
    FONT_STAND1_WCGB_LABEL,
    FONT_STAND1_WCGB_VALUE,
    FONT_STAND2_RECORD,
    FONT_STAND2_VALUE,
    SCOREBOARD_BACKGROUND_COLOR,
)
from utils import clear_display, log_call

# Constants
LOGO_SZ = 59
MARGIN  = 6

# Helpers
def _ord(n):
    try:
        i = int(n)
    except:
        return f"{n}th"
    if 10 <= i % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1:"st", 2:"nd", 3:"rd"}.get(i % 10, "th")
    return f"{i}{suffix}"

def format_games_back(gb):
    """
    Convert raw games-back (float) into display string:
     - integer -> "5"
     - half games -> "½" or "3½"
    """
    try:
        v = float(gb)
        v_abs = abs(v)
        if v_abs.is_integer():
            return f"{int(v_abs)}"
        if abs(v_abs - int(v_abs) - 0.5) < 1e-3:
            return f"{int(v_abs)}½" if int(v_abs)>0 else "½"
    except:
        pass
    return str(gb)

@log_call
def draw_standings_screen1(display, rec, logo_path, division_name, transition=False):
    """
    Screen 1: logo, W/L, rank, GB, WCGB.
    """
    if not rec:
        return None

    clear_display(display)
    img  = Image.new("RGB", (WIDTH, HEIGHT), SCOREBOARD_BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Logo
    logo = None
    try:
        logo_img = Image.open(logo_path).convert("RGBA")
        ratio    = LOGO_SZ / logo_img.height
        logo     = logo_img.resize((int(logo_img.width*ratio), LOGO_SZ), Image.ANTIALIAS)
    except:
        pass
    if logo:
        x0 = (WIDTH - logo.width)//2
        img.paste(logo,(x0,0),logo)

    text_top     = (logo.height if logo else 0) + MARGIN
    bottom_limit = HEIGHT - MARGIN

    # W/L
    w = rec['leagueRecord'].get('wins','-')
    l = rec['leagueRecord'].get('losses','-')
    wl_txt = f"W: {w} L: {l}"

    # Division rank
    dr = rec.get('divisionRank','-')
    try:
        dr_lbl = "Last" if int(dr)==5 else _ord(dr)
    except:
        dr_lbl = dr
    rank_txt = f"{dr_lbl} in {division_name}"

    # GB
    gb_raw = rec.get('divisionGamesBack','-')
    gb_txt = f"{format_games_back(gb_raw)} GB" if gb_raw!='-' else "- GB"

    # WCGB
    wc_raw  = rec.get('wildCardGamesBack')
    wc_rank = rec.get('wildCardRank')
    wc_txt  = None
    if wc_raw is not None:
        base = format_games_back(wc_raw)
        try:
            rank_int = int(wc_rank)
        except:
            rank_int = None

        if wc_raw == 0:
            wc_txt = "-- WCGB"
        elif rank_int and rank_int <= 3:
            wc_txt = f"+{base} WCGB"
        else:
            wc_txt = f"{base} WCGB"

    # Lines to draw
    lines = [
        (wl_txt, FONT_STAND1_WL),
        (rank_txt, FONT_STAND1_RANK),
        (gb_txt, FONT_STAND1_GB_VALUE),
    ]
    if wc_txt:
        lines.append((wc_txt, FONT_STAND1_WCGB_VALUE))

    # Layout text
    heights = [draw.textsize(txt,font)[1] for txt,font in lines]
    total_h = sum(heights)
    avail_h = bottom_limit - text_top
    spacing = (avail_h - total_h) / (len(lines)+1)

    y = text_top + spacing
    for txt,font in lines:
        w0,h0 = draw.textsize(txt,font)
        draw.text(((WIDTH-w0)//2,int(y)),txt,font=font,fill=(255,255,255))
        y += h0 + spacing

    if transition:
        return img

    display.image(img)
    display.show()
    time.sleep(TEAM_STANDINGS_DISPLAY_SECONDS)
    return None


@log_call
def draw_standings_screen2(display, rec, logo_path, transition=False):
    """
    Screen 2: logo + overall record and splits.
    """
    if not rec:
        return None

    clear_display(display)
    img  = Image.new("RGB", (WIDTH, HEIGHT), SCOREBOARD_BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Logo
    logo = None
    try:
        logo_img = Image.open(logo_path).convert("RGBA")
        logo     = logo_img.resize((LOGO_SZ,LOGO_SZ), Image.ANTIALIAS)
    except:
        pass
    if logo:
        x0 = (WIDTH-LOGO_SZ)//2
        img.paste(logo,(x0,0),logo)

    text_top     = LOGO_SZ + MARGIN
    bottom_limit = HEIGHT - MARGIN

    # Overall record
    w = rec['leagueRecord'].get('wins','-')
    l = rec['leagueRecord'].get('losses','-')
    pct = str(rec['leagueRecord'].get('pct','-')).lstrip('0')
    rec_txt = f"{w}-{l} ({pct})"

    # Splits
    splits = rec.get('records',{}).get('splitRecords',[])
    def find_split(t):
        for sp in splits:
            if sp.get('type','').lower()==t.lower():
                return f"{sp.get('wins','-')}-{sp.get('losses','-')}"
        return "-"
    items = [
        f"Streak: {rec.get('streak',{}).get('streakCode','-')}",
        f"L10: {find_split('lastTen')}",
        f"Home: {find_split('home')}",
        f"Away: {find_split('away')}"
    ]

    lines2 = [(rec_txt, FONT_STAND2_RECORD)] + [(it, FONT_STAND2_VALUE) for it in items]
    heights2 = [draw.textsize(txt,font)[1] for txt,font in lines2]
    total2   = sum(heights2)
    avail2   = bottom_limit - text_top
    spacing2 = (avail2 - total2)/(len(lines2)+1)

    y = text_top + spacing2
    for txt,font in lines2:
        w0,h0 = draw.textsize(txt,font)
        draw.text(((WIDTH-w0)//2,int(y)),txt,font=font,fill=(255,255,255))
        y += h0+spacing2

    if transition:
        return img

    display.image(img)
    display.show()
    time.sleep(TEAM_STANDINGS_DISPLAY_SECONDS)
    return None
