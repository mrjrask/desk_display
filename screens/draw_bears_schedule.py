#!/usr/bin/env python3
"""
draw_bears_schedule.py

Shows the next Chicago Bears game with:
  - Title at y=0
  - Opponent wrapped in up to two lines, prefixed by '@' if the Bears are away,
    or 'vs.' if the Bears are home.
  - Between those and the bottom line, a row of logos: AWAY @ HOME, each logo
    auto-sized similarly to the Hawks schedule screen.
  - Bottom line with week/date/time (no spaces around the dash).
"""

import datetime
import os
from PIL import Image, ImageDraw
import config
from config import BEARS_BOTTOM_MARGIN, BEARS_SCHEDULE, NFL_TEAM_ABBREVIATIONS
from utils import (
    load_team_logo,
    next_game_from_schedule,
    standard_next_game_logo_height,
    wrap_text,
)

NFL_LOGO_DIR = os.path.join(config.IMAGES_DIR, "nfl")
def show_bears_next_game(display, transition=False):
    game = next_game_from_schedule(BEARS_SCHEDULE)
    title = "Next for Da Bears:"
    img   = Image.new("RGB", (config.WIDTH, config.HEIGHT), "black")
    draw  = ImageDraw.Draw(img)

    # Title
    tw, th = draw.textsize(title, font=config.FONT_TITLE_SPORTS)
    draw.text(((config.WIDTH - tw)//2, 0), title,
              font=config.FONT_TITLE_SPORTS, fill=(255,255,255))

    if game:
        opp = game["opponent"]
        ha  = game["home_away"].lower()
        prefix = "@" if ha=="away" else "vs."

        # Opponent text (up to 2 lines)
        lines  = wrap_text(f"{prefix} {opp}", config.FONT_TEAM_SPORTS, config.WIDTH)[:2]
        y_txt  = th + 4
        for ln in lines:
            w_ln, h_ln = draw.textsize(ln, font=config.FONT_TEAM_SPORTS)
            draw.text(((config.WIDTH - w_ln)//2, y_txt),
                      ln, font=config.FONT_TEAM_SPORTS, fill=(255,255,255))
            y_txt += h_ln + 2

        # Logos row: AWAY @ HOME
        bears_ab = "chi"
        opp_key  = opp.split()[-1].lower()
        opp_ab   = NFL_TEAM_ABBREVIATIONS.get(opp_key, opp_key[:3])
        if opp_ab == "was":
            opp_ab = "wsh"
        if ha=="away":
            away_ab, home_ab, loc_sym = bears_ab, opp_ab, "@"
        else:
            away_ab, home_ab, loc_sym = opp_ab, bears_ab, "@"

        # Bottom line text — **no spaces around the dash**
        wk = game["week"]
        try:
            dt0 = datetime.datetime.strptime(game["date"], "%a, %b %d")
            date_txt = f"{dt0.month}/{dt0.day}"
        except:
            date_txt = game["date"]
        t_txt = game["time"].strip()
        bottom = f"{wk.replace('0.', 'Pre')}-{date_txt} {t_txt}"
        bw, bh = draw.textsize(bottom, font=config.FONT_DATE_SPORTS)
        bottom_y = config.HEIGHT - bh - BEARS_BOTTOM_MARGIN  # keep on-screen

        desired_logo_h = standard_next_game_logo_height(config.HEIGHT)
        available_h = max(10, bottom_y - (y_txt + 2))
        logo_h = min(desired_logo_h, available_h)

        logo_away = load_team_logo(NFL_LOGO_DIR, away_ab, height=logo_h)
        logo_home = load_team_logo(NFL_LOGO_DIR, home_ab, height=logo_h)

        elems   = [logo_away, loc_sym, logo_home]
        spacing = 8
        widths  = [
            el.width if isinstance(el, Image.Image)
            else draw.textsize(el, font=config.FONT_TEAM_SPORTS)[0]
            for el in elems
        ]
        total_w = sum(widths) + spacing*(len(widths)-1)
        x0      = (config.WIDTH - total_w)//2

        # Vertical center of logos/text block between opponent text and bottom label
        block_h = logo_h
        y_logo = y_txt + ((bottom_y - y_txt) - block_h)//2

        # Draw logos and '@'
        x = x0
        for el in elems:
            if isinstance(el, Image.Image):
                img.paste(el, (x, y_logo), el)
                x += el.width + spacing
            else:
                w_sy, h_sy = draw.textsize(el, font=config.FONT_TEAM_SPORTS)
                y_sy = y_logo + (block_h - h_sy)//2
                draw.text((x, y_sy), el,
                          font=config.FONT_TEAM_SPORTS, fill=(255,255,255))
                x += w_sy + spacing

        # Draw bottom text
        draw.text(((config.WIDTH - bw)//2, bottom_y),
                  bottom, font=config.FONT_DATE_SPORTS, fill=(255,255,255))

    if transition:
        return img

    display.image(img)
    display.show()
    return None
