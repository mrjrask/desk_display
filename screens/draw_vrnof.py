#!/usr/bin/env python3
"""
draw_vrnof.py

Displays VRNOF stock price, change, and all-time P/L on the Display HAT Mini,
with a 10-minute freshness requirement. Title and all-time P/L remain fixed; price/change vertically centered on screen.
Exact cost-basis calculation from individual lots.
"""
import logging
import os
import time

from PIL import Image, ImageDraw
import yfinance as yf

from config import (
    WIDTH,
    HEIGHT,
    VRNOF_CACHE_TTL,
    VRNOF_FRESHNESS_LIMIT,
    VRNOF_LOTS,
    FONT_STOCK_TITLE,
    FONT_STOCK_PRICE,
    FONT_STOCK_CHANGE,
    FONT_STOCK_TEXT,
    IMAGES_DIR,
)
from utils import (
    LED_INDICATOR_LEVEL,
    ScreenImage,
    clear_display,
    log_call,
    temporary_display_led,
)

# In-memory cache
_cache = {
    "price":       None,
    "change_val":  None,
    "change_pct":  None,
    "all_time":    None,
    "ts":          0.0
}

LOGO_HEIGHT = 65
LOGO_GAP = 4
LOGO_PATH = os.path.join(IMAGES_DIR, "verano.jpg")
_LOGO = None


def _get_logo() -> Image.Image | None:
    global _LOGO
    if _LOGO is not None:
        return _LOGO
    try:
        logo = Image.open(LOGO_PATH).convert("RGBA")
        ratio = LOGO_HEIGHT / logo.height
        width = max(1, int(round(logo.width * ratio)))
        height = LOGO_HEIGHT
        _LOGO = logo.resize((width, height), Image.ANTIALIAS)
    except Exception as exc:
        logging.warning("VRNOF: failed to load logo at %s: %s", LOGO_PATH, exc)
        _LOGO = None
    return _LOGO

def _fetch_price(symbol: str):
    """Fetch latest price + change; update cache."""
    price = None
    change_val = None
    change_pct = None

    # Try info first
    try:
        tk = yf.Ticker(symbol)
        info = tk.info
        prev = info.get("previousClose")
        cand = info.get("regularMarketPrice") or prev
        if cand is not None and prev is not None:
            price = float(cand)
            change_val = price - float(prev)
            change_pct = (change_val / float(prev)) * 100
    except Exception as e:
        logging.warning(f"VRNOF: info fetch failed: {e}")

    # Fallback to history
    if price is None:
        try:
            hist = yf.Ticker(symbol).history(period="2d", interval="1d")
            closes = hist.get("Close")
            if closes is not None and len(closes) >= 2:
                prev = float(closes.iloc[-2])
                price = float(closes.iloc[-1])
                change_val = price - prev
                change_pct = (change_val / prev) * 100
        except Exception as e:
            logging.warning(f"VRNOF: history fetch failed: {e}")

    # calculate all-time P/L exactly per lot
    all_time_str = None
    if price is not None:
        total_pl = 0.0
        total_cost = 0.0
        for lot in VRNOF_LOTS:
            shares = lot["shares"]
            cost_basis = lot["cost"]
            total_cost += shares * cost_basis
            total_pl += shares * (price - cost_basis)
        # percentage based on total cost
        all_time_pct = (total_pl / total_cost) * 100 if total_cost else 0
        all_time_str = f"${total_pl:.2f} ({all_time_pct:.2f}%)"

    # update cache
    _cache.update({
        "price":      price,
        "change_val": change_val,
        "change_pct": change_pct,
        "all_time":   all_time_str,
        "ts":         time.time()
    })


def _build_image(symbol: str = "VRNOF") -> Image.Image:
    """Construct the PIL image for the stock screen."""
    now = time.time()
    if _cache["price"] is None or (now - _cache["ts"] > VRNOF_FRESHNESS_LIMIT):
        _fetch_price(symbol)

    # Fallback when no price
    logo = _get_logo()
    if _cache["price"] is None:
        img = Image.new("RGB", (WIDTH, HEIGHT), "black")
        draw = ImageDraw.Draw(img)
        title = symbol
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
        draw.text(((WIDTH - w_r)//2, HEIGHT - h_r - 2), retry, font=FONT_STOCK_TEXT, fill=(200,200,200))
        return img

    price = _cache["price"]
    change_val = _cache["change_val"]
    change_pct = _cache["change_pct"]
    all_time = _cache["all_time"]
    chg_str = f"{change_val:+.3f} ({change_pct:+.2f}%)" if change_val is not None else "N/A"

    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    title_top = 2
    if logo:
        logo_x = (WIDTH - logo.width) // 2
        img.paste(logo, (logo_x, 0), logo)
        title_top = logo.height + LOGO_GAP

    # Title fixed at top (below logo when present)
    title = symbol
    w_title, h_title = draw.textsize(title, font=FONT_STOCK_TITLE)
    draw.text(((WIDTH - w_title)//2, title_top), title, font=FONT_STOCK_TITLE, fill=(255,255,255))

    # All-time P/L fixed at bottom
    if all_time:
        w_all, h_all = draw.textsize(all_time, font=FONT_STOCK_TEXT)
        draw.text(((WIDTH - w_all)//2, HEIGHT - h_all - 2), all_time, font=FONT_STOCK_TEXT, fill=(255,255,255))

    # Price and change vertically centered on entire screen
    price_str = f"${price:.3f}"
    w_price, h_price = draw.textsize(price_str, font=FONT_STOCK_PRICE)
    w_chg, h_chg = draw.textsize(chg_str, font=FONT_STOCK_CHANGE)
    pad = 2
    total_mid_h = h_price + pad + h_chg
    y_mid = (HEIGHT - total_mid_h) // 2

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


def draw_vrnof_screen(display, symbol: str = "VRNOF", transition: bool = False):
    img = _build_image(symbol)
    change_val = _cache.get("change_val")
    led_color = None
    if change_val is not None:
        if change_val > 0:
            led_color = (0.0, LED_INDICATOR_LEVEL, 0.0)
        elif change_val < 0:
            led_color = (LED_INDICATOR_LEVEL, 0.0, 0.0)

    if transition:
        return ScreenImage(img, displayed=False, led_override=led_color)

    def _render_screen():
        clear_display(display)
        display.image(img)
        display.show()
        time.sleep(4)

    if led_color is not None:
        with temporary_display_led(*led_color):
            _render_screen()
    else:
        _render_screen()
    return None
