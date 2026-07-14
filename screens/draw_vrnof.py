#!/usr/bin/env python3
"""
draw_vrnof.py

Displays VRNO stock price, change, and all-time percentage on the Display HAT Mini,
with a 10-minute freshness requirement. Title and all-time percentage remain fixed; price/change vertically centered on screen.
Exact cost-basis calculation from individual lots.
"""
import logging
import math
import os
import time

from PIL import Image, ImageDraw
import yfinance as yf

import config
from config import (
    WIDTH,
    HEIGHT,
    VRNO_CACHE_TTL,
    VRNO_FRESHNESS_LIMIT,
    VRNO_LOTS,
    FONT_STOCK_TITLE,
    FONT_STOCK_PRICE,
    FONT_STOCK_CHANGE,
    FONT_STOCK_TEXT,
    get_screen_background_color,
    IMAGES_DIR,
    is_hyperpixel_4_square_layout,
    is_hyperpixel_next_layout,
)
from utils import (
    LED_INDICATOR_LEVEL,
    ScreenImage,
    log_call,
)

# In-memory cache
_cache = {
    "price":         None,
    "change_val":    None,
    "change_pct":    None,
    "all_time":      None,
    "ts":            0.0,
    "active_symbol": None,
}

VRNO_SYMBOL = "VRNO"


def _candidate_symbols(symbol: str | None = None) -> list[str]:
    """Return the ticker symbol to fetch."""
    return [symbol or VRNO_SYMBOL]

_IS_1080P_LAYOUT = config.is_hdmi_1080p_layout()
_LOGO_SCALE_1080 = config.DISPLAY_PROFILE_LOGO_SCALE_CAP
LOGO_HEIGHT = 54 * _LOGO_SCALE_1080
LOGO_GAP = 4
BOTTOM_TEXT_MARGIN = (
    (18 if is_hyperpixel_4_square_layout() else 10 if is_hyperpixel_next_layout() else 6)
    + (30 if _IS_1080P_LAYOUT else 0)
)
BOTTOM_ALL_TIME_OFFSET = 10
LOGO_PATH = os.path.join(IMAGES_DIR, "verano.jpg")
_LOGO = None


def _get_logo() -> Image.Image | None:
    global _LOGO
    if _LOGO is not None:
        return _LOGO
    try:
        logo = Image.open(LOGO_PATH).convert("RGBA")
        target_height = max(1, int(round(LOGO_HEIGHT)))
        ratio = target_height / logo.height
        width = max(1, int(round(logo.width * ratio)))
        height = target_height
        _LOGO = logo.resize((width, height), Image.ANTIALIAS)
    except Exception as exc:
        logging.warning("VRNO: failed to load logo at %s: %s", LOGO_PATH, exc)
        _LOGO = None
    return _LOGO

def _fetch_price(symbol: str) -> bool:
    """Fetch latest price + change; update cache; return whether a price was found."""
    price = None
    change_val = None
    change_pct = None

    def _derive_change(current: float | None, previous: float | None):
        if current is None or previous is None:
            return None, None
        if previous <= 0:
            return None, None

        delta = current - previous
        pct = (delta / previous) * 100
        if not math.isfinite(pct) or abs(pct) > 500:
            return None, None
        return delta, pct

    # Try info first
    try:
        tk = yf.Ticker(symbol)
        info = tk.info
        prev = info.get("previousClose")
        cand = info.get("regularMarketPrice") or prev
        if cand is not None:
            price = float(cand)
            change_val, change_pct = _derive_change(price, float(prev) if prev is not None else None)
    except Exception as e:
        logging.warning(f"VRNO: info fetch failed: {e}")

    # Fallback to history when primary source didn't produce a usable change.
    if price is None or change_val is None:
        try:
            hist = yf.Ticker(symbol).history(period="2d", interval="1d")
            closes = hist.get("Close")
            if closes is not None and len(closes) >= 2:
                prev = float(closes.iloc[-2])
                hist_price = float(closes.iloc[-1])
                hist_change_val, hist_change_pct = _derive_change(hist_price, prev)
                if hist_change_val is not None:
                    price = hist_price
                    change_val = hist_change_val
                    change_pct = hist_change_pct
        except Exception as e:
            logging.warning(f"VRNO: history fetch failed: {e}")

    # calculate all-time percentage exactly per lot
    all_time_str = None
    if price is not None:
        total_pl = 0.0
        total_cost = 0.0
        for lot in VRNO_LOTS:
            shares = lot["shares"]
            cost_basis = lot["cost"]
            total_cost += shares * cost_basis
            total_pl += shares * (price - cost_basis)
        # percentage based on total cost
        all_time_pct = (total_pl / total_cost) * 100 if total_cost else 0
        all_time_str = f"{all_time_pct:.2f}%"

    # update cache
    _cache.update({
        "price":         price,
        "change_val":    change_val,
        "change_pct":    change_pct,
        "all_time":      all_time_str,
        "ts":            time.time(),
        "active_symbol": symbol if price is not None else None,
    })
    return price is not None


def _fetch_preferred_price(symbol: str | None = None) -> None:
    """Fetch the configured VRNO ticker symbol."""
    last_symbol = None
    for candidate in _candidate_symbols(symbol):
        last_symbol = candidate
        if _fetch_price(candidate):
            return

    # Keep the last attempted symbol visible on the unavailable screen.
    _cache["active_symbol"] = last_symbol


def _build_image(symbol: str | None = None) -> Image.Image:
    """Construct the PIL image for the stock screen."""
    background = get_screen_background_color("vrnof", (0, 0, 0))
    now = time.time()
    candidate_symbols = _candidate_symbols(symbol)
    if (
        _cache["price"] is None
        or (
            _cache.get("active_symbol") is not None
            and _cache.get("active_symbol") not in candidate_symbols
        )
        or (now - _cache["ts"] > VRNO_FRESHNESS_LIMIT)
    ):
        _fetch_preferred_price(symbol)

    display_symbol = _cache.get("active_symbol") or candidate_symbols[-1]

    # Fallback when no price
    logo = _get_logo()
    if _cache["price"] is None:
        img = Image.new("RGB", (WIDTH, HEIGHT), background)
        draw = ImageDraw.Draw(img)
        title = display_symbol
        title_top = 2
        if logo:
            logo_x = (WIDTH - logo.width) // 2
            img.paste(logo, (logo_x, 0), logo)
            title_top = logo.height + LOGO_GAP
        w_t, h_t = draw.textsize(title, font=FONT_STOCK_TITLE)
        draw.text(((WIDTH - w_t)//2, title_top), title, font=FONT_STOCK_TITLE, fill=(255,255,255))
        msg = "Price unavailable"
        w_m, h_m = draw.textsize(msg, font=FONT_STOCK_TEXT)
        draw.text(((WIDTH - w_m)//2, HEIGHT//2 - h_m//2), msg, font=FONT_STOCK_TEXT, fill=(200,200,200))
        retry = "Try again shortly"
        w_r, h_r = draw.textsize(retry, font=FONT_STOCK_TEXT)
        draw.text(
            ((WIDTH - w_r)//2, HEIGHT - h_r - BOTTOM_TEXT_MARGIN),
            retry,
            font=FONT_STOCK_TEXT,
            fill=(200,200,200),
        )
        return img

    price = _cache["price"]
    change_val = _cache["change_val"]
    change_pct = _cache["change_pct"]
    all_time = _cache["all_time"]
    chg_str = f"{change_val:+.3f} ({change_pct:+.2f}%)" if change_val is not None else "N/A"

    img = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)

    title_top = 2
    if logo:
        logo_x = (WIDTH - logo.width) // 2
        img.paste(logo, (logo_x, 0), logo)
        title_top = logo.height + LOGO_GAP

    # Title fixed at top (below logo when present)
    title = display_symbol
    w_title, h_title = draw.textsize(title, font=FONT_STOCK_TITLE)
    draw.text(((WIDTH - w_title)//2, title_top), title, font=FONT_STOCK_TITLE, fill=(255,255,255))

    # All-time percentage fixed at bottom
    if all_time:
        w_all, h_all = draw.textsize(all_time, font=FONT_STOCK_TEXT)
        draw.text(
            ((WIDTH - w_all)//2, HEIGHT - h_all - BOTTOM_TEXT_MARGIN - BOTTOM_ALL_TIME_OFFSET),
            all_time,
            font=FONT_STOCK_TEXT,
            fill=(255,255,255),
        )

    # Price and change centered in the available area between title/logo and
    # bottom all-time text to avoid overlap on short displays.
    price_str = f"${price:.3f}"
    w_price, h_price = draw.textsize(price_str, font=FONT_STOCK_PRICE)
    w_chg, h_chg = draw.textsize(chg_str, font=FONT_STOCK_CHANGE)
    pad = 2
    total_mid_h = h_price + pad + h_chg
    top_content_y = title_top + h_title + LOGO_GAP
    bottom_reserved = BOTTOM_TEXT_MARGIN + BOTTOM_ALL_TIME_OFFSET + (h_all if all_time else 0)
    bottom_content_y = HEIGHT - bottom_reserved

    available_mid_h = bottom_content_y - top_content_y
    if available_mid_h >= total_mid_h:
        y_mid = top_content_y + (available_mid_h - total_mid_h) // 2
    else:
        # Degenerate fallback for extremely constrained heights.
        y_mid = max(top_content_y, min((HEIGHT - total_mid_h) // 2, HEIGHT - total_mid_h))

    # Draw price
    draw.text(((WIDTH - w_price)//2, y_mid), price_str, font=FONT_STOCK_PRICE, fill=(255,255,255))
    # Determine change color
    if change_val is None:
        color = (255,255,255)
    elif change_val > 0:
        color = (0,255,0)
    elif change_val < 0:
        color = (255,0,0)
    else:
        color = (255,255,255)
    # Draw change below price
    draw.text(((WIDTH - w_chg)//2, y_mid + h_price + pad), chg_str, font=FONT_STOCK_CHANGE, fill=color)

    return img


def draw_vrnof_screen(display, symbol: str | None = None, transition: bool = False):
    img = _build_image(symbol)
    change_val = _cache.get("change_val")
    led_color = None
    if change_val is not None:
        if change_val > 0:
            led_color = (0.0, LED_INDICATOR_LEVEL, 0.0)
        elif change_val < 0:
            led_color = (LED_INDICATOR_LEVEL, 0.0, 0.0)

    return ScreenImage(img, displayed=False, led_override=led_color)
