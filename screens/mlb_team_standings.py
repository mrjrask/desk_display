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
from utils import clear_display, fit_logo_to_box, log_call

# Constants
LOGO_SZ = 71
MARGIN  = 6

# Helpers
def _ord(n):
    try:
        i = int(n)
    except:
        return f"{n}th"
    if i <= 0:
        return "-"
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


def _format_int(value, *, default="-") -> str:
    """Return an integer-like string with no decimal places."""

    try:
        if value in (None, ""):
            return default
        return f"{int(float(value))}"
    except Exception:
        return str(value) if value not in (None, "") else default


def _format_record_values(record, *, ot_label="OT"):
    w = record.get("wins", "-")
    l = record.get("losses", "-")
    t = record.get("ties")
    ot = record.get("ot")

    tie_val = t if t not in (None, "", "-") else ot
    tie_label = "T" if t is not None else ot_label

    parts = [f"W: {w}", f"L: {l}"]
    if tie_val not in (None, "", "-", 0, "0"):
        parts.append(f"{tie_label}: {tie_val}")

    return " ".join(parts)


def _format_streak(streak) -> str:
    """Return a streak string without decimal places in the numeric portion."""

    try:
        if streak in (None, "", "-"):
            return "-"

        code = str(streak)
        prefix = code[0] if code and code[0].isalpha() else ""
        number_part = code[len(prefix) :] if prefix else code

        if number_part == "":
            return code

        number_val = float(number_part)
        number_txt = f"{int(abs(number_val))}"

        if prefix:
            return f"{prefix}{number_txt}"

        if number_val > 0:
            return f"W{number_txt}"
        if number_val < 0:
            return f"L{number_txt}"
        return number_txt
    except Exception:
        return str(streak)


@log_call
def draw_standings_screen1(
    display,
    rec,
    logo_path,
    division_name,
    *,
    logo_size=None,
    show_games_back=True,
    show_wild_card=True,
    show_streak=False,
    ot_label="OT",
    points_label=None,
    conference_label=None,
    show_conference_rank=True,
    place_gb_before_rank=False,
    show_pct=False,
    pct_precision=None,
    record_details_fn=None,
    record_font=None,
    points_font=None,
    transition=False,
):
    """
    Screen 1: logo, W/L, rank, optional GB/WCGB.
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
        logo_target = LOGO_SZ if logo_size is None else logo_size
        logo = fit_logo_to_box(logo_img, logo_target)
    except:
        pass
    if logo:
        x0 = (WIDTH - logo.width)//2
        img.paste(logo,(x0,0),logo)

    text_top     = (logo.height if logo else 0) + MARGIN
    bottom_limit = HEIGHT - MARGIN

    # W/L
    record_line = _format_record_values(rec.get("leagueRecord", {}), ot_label=ot_label)

    record_font = FONT_STAND1_WL if record_font is None else record_font
    points_font = FONT_STAND1_GB_VALUE if points_font is None else points_font

    if record_details_fn:
        wl_txt = record_details_fn(rec, record_line)
    elif show_pct:
        pct_raw = rec.get("leagueRecord", {}).get("pct", "-")
        precision = 3 if pct_precision is None else pct_precision
        try:
            pct_txt = f"{float(pct_raw):.{precision}f}".lstrip("0")
        except Exception:
            pct_txt = str(pct_raw).lstrip("0")
        wl_txt = f"{record_line} ({pct_txt})"
    else:
        wl_txt = record_line

    points_txt = None
    if points_label is not None:
        pts_val = _format_int(rec.get("points"))
        points_txt = f"{pts_val} {points_label}"

    # Division rank
    dr_raw = rec.get('divisionRank')
    dr = dr_raw if dr_raw not in (None, "") else "-"
    try:
        dr_lbl = "Last" if int(dr)==5 else _ord(dr)
    except:
        dr_lbl = dr
    rank_txt = f"{dr_lbl} in {division_name}"

    # GB
    gb_txt = None
    if show_games_back:
        gb_raw = rec.get('divisionGamesBack','-')
        gb_txt = f"{format_games_back(gb_raw)} GB" if gb_raw!='-' else "- GB"

    # WCGB
    wc_txt  = None
    if show_wild_card:
        wc_raw  = rec.get('wildCardGamesBack')
        wc_rank = rec.get('wildCardRank')
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
        (wl_txt, record_font),
    ]
    if points_txt:
        lines.append((points_txt, points_font))
    if gb_txt and place_gb_before_rank:
        lines.append((gb_txt, FONT_STAND1_GB_VALUE))
    lines.append((rank_txt, FONT_STAND1_RANK))
    if conference_label and show_conference_rank:
        conf_raw = rec.get("conferenceRank")
        conf_rank = conf_raw if conf_raw not in (None, "") else "-"
        try:
            conf_lbl = "Last" if int(conf_rank) == 16 else _ord(conf_rank)
        except Exception:
            conf_lbl = conf_rank
        conf_name = rec.get("conferenceName") or rec.get("conferenceAbbrev") or "conference"
        lines.append((f"{conf_lbl} in {conf_name}", FONT_STAND1_RANK))
    if gb_txt and not place_gb_before_rank:
        lines.append((gb_txt, FONT_STAND1_GB_VALUE))
    if wc_txt:
        lines.append((wc_txt, FONT_STAND1_WCGB_VALUE))
    if show_streak:
        streak_raw = (rec.get("streak") or {}).get("streakCode", "-")
        lines.append((f"Streak: {_format_streak(streak_raw)}", FONT_STAND1_RANK))

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
def draw_standings_screen2(
    display,
    rec,
    logo_path,
    *,
    logo_size=None,
    pct_precision=None,
    record_details_fn=None,
    split_order=("lastTen", "home", "away"),
    split_overrides=None,
    show_streak=True,
    show_points=True,
    transition=False,
):
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
        logo_target = LOGO_SZ if logo_size is None else logo_size
        logo = fit_logo_to_box(logo_img, logo_target)
    except:
        pass
    if logo:
        x0 = (WIDTH - logo.width) // 2
        img.paste(logo, (x0, 0), logo)

    text_top     = (logo.height if logo else 0) + MARGIN
    bottom_limit = HEIGHT - MARGIN

    # Overall record
    record = rec.get('leagueRecord', {})
    w = record.get('wins','-')
    l = record.get('losses','-')
    t = record.get('ties') if record.get('ties') not in (0, '0') else None
    if t in (None, '', '-', 0, '0'):
        t = record.get('ot') if record.get('ot') not in (0, '0') else None
    pct_raw = record.get("pct", "-")
    precision = 3 if pct_precision is None else pct_precision
    try:
        pct = f"{float(pct_raw):.{precision}f}".lstrip("0")
    except Exception:
        pct = str(pct_raw).lstrip("0")

    base_rec = f"{w}-{l}"
    if t not in (None, '', '-', 0, '0'):
        base_rec = f"{base_rec}-{t}"
    if record_details_fn:
        rec_txt = record_details_fn(rec, base_rec)
    else:
        rec_txt = f"{base_rec} ({pct})"

    # Splits
    split_overrides = split_overrides or {}
    splits = rec.get('records',{}).get('splitRecords',[])

    def find_split(t):
        if t in split_overrides:
            return split_overrides[t]
        for sp in splits:
            if sp.get('type','').lower()==t.lower():
                return f"{sp.get('wins','-')}-{sp.get('losses','-')}"
        return "-"

    items = []
    if show_streak:
        streak_raw = rec.get("streak", {}).get("streakCode", "-")
        items.append(f"Streak: {_format_streak(streak_raw)}")
    pts = rec.get('points')
    if show_points and pts not in (None, ''):
        items.append(f"Pts: {_format_int(pts)}")
    for split in split_order:
        label = {
            "lastTen": "L10",
            "home": "Home",
            "away": "Away",
            "division": "Division",
            "conference": "Conference",
        }.get(split, split)
        items.append(f"{label}: {find_split(split)}")

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
