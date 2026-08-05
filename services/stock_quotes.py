"""Stock/index quote fetching for the news-headlines ticker's market row.

Uses ``yfinance`` (the same source as the VRNO screen) to fetch a last price
and change for a small, fixed set of symbols: the three major US indices,
VRNO, AAPL, and the current mega-cap leaders by market cap. Results are
cached for STOCK_TICKER_CACHE_TTL_SECONDS so the marquee doesn't refetch on
every frame.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Optional

import yfinance as yf

import config

# Order here is the left-to-right order symbols appear in the ticker row.
INDEX_LABELS: dict[str, str] = {
    "^DJI": "DJIA",
    "^IXIC": "NASDAQ",
    "^GSPC": "S&P 500",
}

# Static "current mega-caps" list. Market-cap rankings shift over time; edit
# this list as needed rather than trying to rank live (that would require an
# extra API call per screen draw just to order five tickers).
TOP_MARKET_CAP_SYMBOLS: list[str] = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]

_FETCH_TIMEOUT_BUDGET_SECONDS = 8.0


def default_symbol_order() -> list[str]:
    """Indices, then VRNO/AAPL, then the top-market-cap list, de-duplicated."""

    ordered = [*INDEX_LABELS, "VRNO", "AAPL", *TOP_MARKET_CAP_SYMBOLS]
    seen: set[str] = set()
    result: list[str] = []
    for symbol in ordered:
        if symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)
    return result


@dataclass(frozen=True)
class StockQuote:
    symbol: str
    label: str
    price: Optional[float]
    change: Optional[float]
    change_pct: Optional[float]


def _label_for(symbol: str) -> str:
    return INDEX_LABELS.get(symbol, symbol)


def _derive_change(
    current: Optional[float], previous: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    if current is None or previous is None or previous <= 0:
        return None, None
    delta = current - previous
    pct = (delta / previous) * 100
    if not math.isfinite(pct) or abs(pct) > 500:
        return None, None
    return delta, pct


def fetch_quote(symbol: str) -> StockQuote:
    """Fetch a single symbol's last price and change. Never raises."""

    price: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        prev = info.get("previousClose")
        candidate = info.get("regularMarketPrice") or prev
        if candidate is not None:
            price = float(candidate)
            change, change_pct = _derive_change(price, float(prev) if prev is not None else None)
    except Exception as exc:
        logging.debug("stock_quotes: info fetch failed for %s: %s", symbol, exc)

    if price is None or change is None:
        try:
            hist = yf.Ticker(symbol).history(period="2d", interval="1d")
            closes = hist.get("Close")
            if closes is not None and len(closes) >= 2:
                prev = float(closes.iloc[-2])
                hist_price = float(closes.iloc[-1])
                hist_change, hist_change_pct = _derive_change(hist_price, prev)
                if hist_change is not None:
                    price, change, change_pct = hist_price, hist_change, hist_change_pct
        except Exception as exc:
            logging.debug("stock_quotes: history fetch failed for %s: %s", symbol, exc)

    return StockQuote(
        symbol=symbol, label=_label_for(symbol), price=price, change=change, change_pct=change_pct
    )


_quotes_cache_lock = threading.Lock()
_quotes_cache_value: dict[str, StockQuote] = {}
_quotes_cache_time: Optional[float] = None


def fetch_stock_quotes(
    symbols: Optional[list[str]] = None, *, force: bool = False
) -> list[StockQuote]:
    """Return quotes for *symbols* (default: default_symbol_order()).

    Cached for config.STOCK_TICKER_CACHE_TTL_SECONDS. Symbols that fail to
    fetch on a given round keep their last known-good quote in the cache
    (mirroring fetch_all_headlines's stale-is-better-than-empty behavior)
    rather than disappearing from the ticker for one cycle.
    """

    global _quotes_cache_time

    symbols = list(symbols) if symbols is not None else default_symbol_order()
    if not symbols:
        return []

    cache_ttl = max(60, config.STOCK_TICKER_CACHE_TTL_SECONDS)
    now = time.monotonic()
    with _quotes_cache_lock:
        if (
            not force
            and _quotes_cache_time is not None
            and (now - _quotes_cache_time) < cache_ttl
            and all(symbol in _quotes_cache_value for symbol in symbols)
        ):
            return [_quotes_cache_value[symbol] for symbol in symbols]

    results: dict[str, StockQuote] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(symbols))) as executor:
        future_to_symbol = {executor.submit(fetch_quote, symbol): symbol for symbol in symbols}
        done, pending = wait(future_to_symbol, timeout=_FETCH_TIMEOUT_BUDGET_SECONDS)
        for future in done:
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                logging.debug("stock_quotes: symbol %s raised: %s", symbol, exc)
        for future in pending:
            future.cancel()

    with _quotes_cache_lock:
        for symbol, quote in results.items():
            if quote.price is not None:
                _quotes_cache_value[symbol] = quote
        _quotes_cache_time = time.monotonic()
        return [_quotes_cache_value[symbol] for symbol in symbols if symbol in _quotes_cache_value]


def clear_quote_cache_for_tests() -> None:
    global _quotes_cache_value, _quotes_cache_time
    with _quotes_cache_lock:
        _quotes_cache_value = {}
        _quotes_cache_time = None
