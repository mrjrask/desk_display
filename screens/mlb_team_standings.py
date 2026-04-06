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
import logging
from PIL import Image, ImageDraw
import config
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
    get_screen_background_color,
    is_hyperpixel_next_layout,
    scale_value,
    scale_value_width,
    DISPLAY_SCALE,
    is_hyperpixel_4_square_layout,
)
from utils import clear_display, fit_logo_to_box, log_call, clone_font

# Constants
_DISPLAY_OUTPUT = os.environ.get("DESK_DISPLAY_OUTPUT", "auto").strip().lower()
_IS_DISPLAY_HAT_MINI = _DISPLAY_OUTPUT in {
    "displayhatmini",
    "display-hat-mini",
    "hatmini",
    "hat",
}

LOGO_SZ_BASE = scale_value_width(27) if is_hyperpixel_next_layout() else scale_value(36)
if is_hyperpixel_4_square_layout():
    LOGO_SZ_BASE = max(LOGO_SZ_BASE, scale_value_width(80))
LOGO_SZ = LOGO_SZ_BASE * (3 if _IS_DISPLAY_HAT_MINI else 1)
_IS_1080P_LAYOUT = config.is_hdmi_1080p_layout()
_VRNOF_MATCH_LOGO_HEIGHT_1080 = 54 * 5
if _IS_1080P_LAYOUT:
    LOGO_SZ = min(LOGO_SZ, _VRNOF_MATCH_LOGO_HEIGHT_1080)
MARGIN  = scale_value(6)
FRACTION_FONT_SCALE = 0.6
_HAWKS_STAND1_MATCH_SCREEN_IDS = {
    "cubs stand1",
    "cubs stand2",
    "cubs stand3",
    "sox stand1",
    "sox stand2",
    "sox stand3",
}


def _restore_font(font):
    size = getattr(font, "size", None)
    if not isinstance(size, int) or size <= 0:
        return font
    scale = max(1.0, DISPLAY_SCALE)
    if _IS_1080P_LAYOUT:
        scale = max(1.0, scale / 1.8)
    return clone_font(font, max(1, int(round(size / scale))))


FONT_STAND1_WL_RESTORED = _restore_font(FONT_STAND1_WL)
FONT_STAND1_RANK_RESTORED = _restore_font(FONT_STAND1_RANK)
FONT_STAND1_GB_VALUE_RESTORED = _restore_font(FONT_STAND1_GB_VALUE)
FONT_STAND1_WCGB_VALUE_RESTORED = _restore_font(FONT_STAND1_WCGB_VALUE)
FONT_STAND2_RECORD_RESTORED = _restore_font(FONT_STAND2_RECORD)
FONT_STAND2_VALUE_RESTORED = _restore_font(FONT_STAND2_VALUE)

# Helpers
def _ord(n):
    try:
        i = int(n)
    except (TypeError, ValueError):
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
     - half games -> "1/2" or "3 1/2"
    """
    try:
        v = float(gb)
        v_abs = abs(v)
        if v_abs.is_integer():
            return f"{int(v_abs)}"
        if abs(v_abs - int(v_abs) - 0.5) < 1e-3:
            return f"{int(v_abs)} 1/2" if int(v_abs) > 0 else "1/2"
    except (TypeError, ValueError):
        logging.debug("format_games_back fallback for value=%r", gb)
    return str(gb)


def _format_int(value, *, default="-") -> str:
    """Return an integer-like string with no decimal places."""

    try:
        if value in (None, ""):
            return default
        return f"{int(float(value))}"
    except Exception:
        return str(value) if value not in (None, "") else default


def _format_wcgb_text(wc_raw, wc_rank) -> str | None:
    """Format WCGB display text with correct sign handling."""

    if wc_raw is None:
        return None

    base = format_games_back(wc_raw)
    try:
        rank_int = int(wc_rank)
    except (TypeError, ValueError):
        rank_int = None

    try:
        wc_value = float(wc_raw)
    except (TypeError, ValueError):
        wc_value = None

    if wc_value == 0:
        return "-- WCGB"
    if rank_int and rank_int <= 3 and wc_value is not None and wc_value > 0:
        return f"+{base} WCGB"
    return f"{base} WCGB"


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


def _hawks_stand1_match_metrics(screen_id: str | None) -> tuple[int | None, int]:
    """Return logo/font overrides that match the Hawks Stand 1 Hyperpixel sizing."""

    if not screen_id or str(screen_id).lower() not in _HAWKS_STAND1_MATCH_SCREEN_IDS:
        return None, 0

    if is_hyperpixel_4_square_layout():
        return max(1, int(round(LOGO_SZ * 1.5))), 8
    if is_hyperpixel_next_layout():
        return max(1, int(round(LOGO_SZ * 2.25))), 12
    return None, 0


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


def _fraction_font(font):
    base_size = max(1, int(getattr(font, "size", 20)))
    return clone_font(font, max(1, int(round(base_size * FRACTION_FONT_SCALE))))


def _measure_fraction_text(draw, text: str, font) -> tuple[int, int]:
    if "1/2" not in text:
        return draw.textsize(text, font)
    pre, frac, post = text.partition("1/2")
    frac_font = _fraction_font(font)
    pre_w, pre_h = draw.textsize(pre, font)
    frac_w, frac_h = draw.textsize(frac, frac_font)
    post_w, post_h = draw.textsize(post, font)
    return pre_w + frac_w + post_w, max(pre_h, frac_h, post_h)


def _draw_fraction_text_centered(draw, y: int, text: str, font, *, fill=(255, 255, 255)) -> None:
    if "1/2" not in text:
        w, _ = draw.textsize(text, font)
        draw.text(((WIDTH - w) // 2, int(y)), text, font=font, fill=fill)
        return

    pre, frac, post = text.partition("1/2")
    frac_font = _fraction_font(font)
    pre_w, pre_h = draw.textsize(pre, font)
    frac_w, frac_h = draw.textsize(frac, frac_font)
    post_w, post_h = draw.textsize(post, font)
    total_w = pre_w + frac_w + post_w
    x = (WIDTH - total_w) // 2
    base_h = max(pre_h, post_h)
    frac_y = int(y + max(0, base_h - frac_h))

    if pre:
        draw.text((x, int(y)), pre, font=font, fill=fill)
    draw.text((x + pre_w, frac_y), frac, font=frac_font, fill=fill)
    if post:
        draw.text((x + pre_w + frac_w, int(y)), post, font=font, fill=fill)


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
    division_last_rank=5,
    conference_last_rank=16,
    place_gb_before_rank=False,
    show_pct=False,
    pct_precision=None,
    record_details_fn=None,
    record_font=None,
    points_font=None,
    font_size_offset=0,
    screen_id=None,
    transition=False,
):
    """
    Screen 1: logo, W/L, rank, optional GB/WCGB.
    """
    if not rec:
        return None

    clear_display(display)
    background_color = (
        get_screen_background_color(screen_id, SCOREBOARD_BACKGROUND_COLOR)
        if screen_id
        else SCOREBOARD_BACKGROUND_COLOR
    )
    img  = Image.new("RGB", (WIDTH, HEIGHT), background_color)
    draw = ImageDraw.Draw(img)

    hawks_logo_size, hawks_font_offset = _hawks_stand1_match_metrics(screen_id)

    # Logo
    logo = None
    try:
        logo_img = Image.open(logo_path).convert("RGBA")
        logo_target = LOGO_SZ if logo_size is None else logo_size
        if hawks_logo_size is not None and logo_size is None:
            logo_target = hawks_logo_size
        if _IS_1080P_LAYOUT:
            logo_target = min(logo_target, _VRNOF_MATCH_LOGO_HEIGHT_1080)
        logo = fit_logo_to_box(logo_img, logo_target)
    except (FileNotFoundError, OSError) as exc:
        logging.warning(
            "draw_standings_screen1 logo fallback screen_id=%s logo_path=%s err=%s",
            screen_id,
            logo_path,
            exc,
        )
    if logo:
        x0 = (WIDTH - logo.width)//2
        img.paste(logo,(x0,0),logo)

    text_top     = (logo.height if logo else 0) + MARGIN
    bottom_limit = HEIGHT - MARGIN

    # W/L
    record_line = _format_record_values(rec.get("leagueRecord", {}), ot_label=ot_label)

    record_font = FONT_STAND1_WL_RESTORED if record_font is None else record_font
    points_font = FONT_STAND1_GB_VALUE_RESTORED if points_font is None else points_font

    rank_font = FONT_STAND1_RANK_RESTORED
    gb_font = FONT_STAND1_GB_VALUE_RESTORED
    wc_font = FONT_STAND1_WCGB_VALUE_RESTORED
    if is_hyperpixel_4_square_layout():
        record_font = clone_font(record_font, max(1, int(round(getattr(record_font, "size", 24) * 1.55))))
        points_font = clone_font(points_font, max(1, int(round(getattr(points_font, "size", 20) * 1.45))))
        rank_font = clone_font(rank_font, max(1, int(round(getattr(rank_font, "size", 20) * 1.45))))
        gb_font = clone_font(gb_font, max(1, int(round(getattr(gb_font, "size", 20) * 1.35))))
        wc_font = clone_font(wc_font, max(1, int(round(getattr(wc_font, "size", 20) * 1.35))))

    total_font_offset = int(round(font_size_offset)) + hawks_font_offset
    if total_font_offset:
        offset = total_font_offset
        record_font = clone_font(record_font, max(1, getattr(record_font, "size", 24) + offset))
        points_font = clone_font(points_font, max(1, getattr(points_font, "size", 20) + offset))
        rank_font = clone_font(rank_font, max(1, getattr(rank_font, "size", 20) + offset))
        gb_font = clone_font(gb_font, max(1, getattr(gb_font, "size", 20) + offset))
        wc_font = clone_font(wc_font, max(1, getattr(wc_font, "size", 20) + offset))

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
        dr_lbl = "Last" if int(dr) == int(division_last_rank) else _ord(dr)
    except (TypeError, ValueError):
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
        wc_txt = _format_wcgb_text(rec.get('wildCardGamesBack'), rec.get('wildCardRank'))

    # Lines to draw
    lines = [
        (wl_txt, record_font),
    ]
    if points_txt:
        lines.append((points_txt, points_font))
    if gb_txt and place_gb_before_rank:
        lines.append((gb_txt, gb_font))
    lines.append((rank_txt, rank_font))
    if conference_label and show_conference_rank:
        conf_raw = rec.get("conferenceRank")
        conf_rank = conf_raw if conf_raw not in (None, "") else "-"
        try:
            conf_lbl = "Last" if int(conf_rank) == int(conference_last_rank) else _ord(conf_rank)
        except Exception:
            conf_lbl = conf_rank
        conf_name = rec.get("conferenceName") or rec.get("conferenceAbbrev") or "conference"
        lines.append((f"{conf_lbl} in {conf_name}", rank_font))
    if gb_txt and not place_gb_before_rank:
        lines.append((gb_txt, gb_font))
    if wc_txt:
        lines.append((wc_txt, wc_font))
    if show_streak:
        streak_raw = (rec.get("streak") or {}).get("streakCode", "-")
        lines.append((f"Streak: {_format_streak(streak_raw)}", rank_font))

    # Layout text
    heights = [_measure_fraction_text(draw, txt, font)[1] for txt, font in lines]
    total_h = sum(heights)
    avail_h = bottom_limit - text_top
    spacing = (avail_h - total_h) / (len(lines)+1)

    y = text_top + spacing
    for txt,font in lines:
        _, h0 = _measure_fraction_text(draw, txt, font)
        _draw_fraction_text_centered(draw, int(y), txt, font, fill=(255, 255, 255))
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
    screen_id=None,
    transition=False,
):
    """
    Screen 2: logo + overall record and splits.
    """
    if not rec:
        return None

    clear_display(display)
    background_color = (
        get_screen_background_color(screen_id, SCOREBOARD_BACKGROUND_COLOR)
        if screen_id
        else SCOREBOARD_BACKGROUND_COLOR
    )
    img  = Image.new("RGB", (WIDTH, HEIGHT), background_color)
    draw = ImageDraw.Draw(img)

    hawks_logo_size, hawks_font_offset = _hawks_stand1_match_metrics(screen_id)

    # Logo
    logo = None
    try:
        logo_img = Image.open(logo_path).convert("RGBA")
        logo_target = LOGO_SZ if logo_size is None else logo_size
        if hawks_logo_size is not None and logo_size is None:
            logo_target = hawks_logo_size
        if _IS_1080P_LAYOUT:
            logo_target = min(logo_target, _VRNOF_MATCH_LOGO_HEIGHT_1080)
        logo = fit_logo_to_box(logo_img, logo_target)
    except (FileNotFoundError, OSError) as exc:
        logging.warning(
            "draw_standings_screen2 logo fallback screen_id=%s logo_path=%s err=%s",
            screen_id,
            logo_path,
            exc,
        )
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

    record_font = FONT_STAND2_RECORD_RESTORED
    value_font = FONT_STAND2_VALUE_RESTORED
    if hawks_font_offset:
        record_font = clone_font(record_font, max(1, getattr(record_font, "size", 24) + hawks_font_offset))
        value_font = clone_font(value_font, max(1, getattr(value_font, "size", 20) + hawks_font_offset))

    lines2 = [(rec_txt, record_font)] + [(it, value_font) for it in items]
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


@log_call
def draw_standings_screen3(
    display,
    rec,
    logo_path,
    division_name,
    *,
    logo_size=None,
    screen_id=None,
    transition=False,
):
    """
    Screen 3: same logo placement as Screen 1, then two columns.
      - Left column: Stand 1 data (record, rank, GB, WCGB)
      - Right column: Stand 2 data (record+pct, streak, L10, home, away)
    """
    if not rec:
        return None

    clear_display(display)
    background_color = (
        get_screen_background_color(screen_id, SCOREBOARD_BACKGROUND_COLOR)
        if screen_id
        else SCOREBOARD_BACKGROUND_COLOR
    )
    img = Image.new("RGB", (WIDTH, HEIGHT), background_color)
    draw = ImageDraw.Draw(img)

    hawks_logo_size, hawks_font_offset = _hawks_stand1_match_metrics(screen_id)

    logo = None
    try:
        logo_img = Image.open(logo_path).convert("RGBA")
        logo_target = LOGO_SZ if logo_size is None else logo_size
        if hawks_logo_size is not None and logo_size is None:
            logo_target = hawks_logo_size
        if _IS_1080P_LAYOUT:
            logo_target = min(logo_target, _VRNOF_MATCH_LOGO_HEIGHT_1080)
        logo = fit_logo_to_box(logo_img, logo_target)
    except Exception:
        pass
    if logo:
        x0 = (WIDTH - logo.width) // 2
        img.paste(logo, (x0, 0), logo)

    text_top = (logo.height if logo else 0) + MARGIN
    bottom_limit = HEIGHT - MARGIN

    left_header_font = FONT_STAND1_RANK_RESTORED
    left_value_font = FONT_STAND1_GB_VALUE_RESTORED
    right_header_font = FONT_STAND2_RECORD_RESTORED
    right_value_font = FONT_STAND2_VALUE_RESTORED

    if hawks_font_offset:
        left_header_font = clone_font(
            left_header_font, max(1, getattr(left_header_font, "size", 20) + hawks_font_offset)
        )
        left_value_font = clone_font(
            left_value_font, max(1, getattr(left_value_font, "size", 20) + hawks_font_offset)
        )
        right_header_font = clone_font(
            right_header_font, max(1, getattr(right_header_font, "size", 24) + hawks_font_offset)
        )
        right_value_font = clone_font(
            right_value_font, max(1, getattr(right_value_font, "size", 20) + hawks_font_offset)
        )

    # Left column = Stand 1 data
    record_line = _format_record_values(rec.get("leagueRecord", {}), ot_label="OT")
    dr_raw = rec.get("divisionRank")
    dr = dr_raw if dr_raw not in (None, "") else "-"
    try:
        dr_lbl = "Last" if int(dr) == 5 else _ord(dr)
    except Exception:
        dr_lbl = dr
    rank_txt = f"{dr_lbl} in {division_name}"

    gb_raw = rec.get("divisionGamesBack", "-")
    gb_txt = f"{format_games_back(gb_raw)} GB" if gb_raw != "-" else "- GB"

    wc_txt = None
    wc_raw = rec.get("wildCardGamesBack")
    wc_rank = rec.get("wildCardRank")
    if wc_raw is not None:
        base = format_games_back(wc_raw)
        try:
            rank_int = int(wc_rank)
        except Exception:
            rank_int = None
        if wc_raw == 0:
            wc_txt = "-- WCGB"
        elif rank_int and rank_int <= 3:
            wc_txt = f"+{base} WCGB"
        else:
            wc_txt = f"{base} WCGB"

    left_lines = [
        (record_line, left_header_font),
        (rank_txt, left_value_font),
        (gb_txt, left_value_font),
    ]
    if wc_txt:
        left_lines.append((wc_txt, left_value_font))

    # Right column = Stand 2 data
    record = rec.get("leagueRecord", {})
    w = record.get("wins", "-")
    l = record.get("losses", "-")
    pct_raw = record.get("pct", "-")
    try:
        pct = f"{float(pct_raw):.3f}".lstrip("0")
    except Exception:
        pct = str(pct_raw).lstrip("0")
    rec_txt = f"{w}-{l} ({pct})"

    def _find_split(split_type):
        for sp in rec.get("records", {}).get("splitRecords", []):
            if sp.get("type", "").lower() == split_type.lower():
                return f"{sp.get('wins', '-')}-{sp.get('losses', '-')}"
        return "-"

    streak_raw = rec.get("streak", {}).get("streakCode", "-")
    right_lines = [
        (rec_txt, right_header_font),
        (f"Streak: {_format_streak(streak_raw)}", right_value_font),
        (f"L10: {_find_split('lastTen')}", right_value_font),
        (f"Home: {_find_split('home')}", right_value_font),
        (f"Away: {_find_split('away')}", right_value_font),
    ]

    col_gap = max(scale_value_width(10), WIDTH // 24)
    col_width = (WIDTH - (2 * MARGIN) - col_gap) // 2
    left_x = MARGIN
    right_x = left_x + col_width + col_gap
    avail_h = max(1, bottom_limit - text_top)

    def draw_column(lines, x, width):
        heights = [_measure_fraction_text(draw, txt, font)[1] for txt, font in lines]
        total_h = sum(heights)
        spacing = (avail_h - total_h) / (len(lines) + 1)
        y = text_top + spacing
        for txt, font in lines:
            w0, h0 = _measure_fraction_text(draw, txt, font)
            tx = int(x + max(0, (width - w0) / 2))
            _draw_fraction_text_centered(
                draw,
                int(y),
                txt,
                font,
                fill=(255, 255, 255),
            ) if width >= WIDTH - (2 * MARGIN) else draw.text((tx, int(y)), txt, font=font, fill=(255, 255, 255))
            y += h0 + spacing

    draw_column(left_lines, left_x, col_width)
    draw_column(right_lines, right_x, col_width)

    if transition:
        return img

    display.image(img)
    display.show()
    time.sleep(TEAM_STANDINGS_DISPLAY_SECONDS)
    return None
