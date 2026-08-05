import time

import pandas as pd
import pytest

import services.stock_quotes as sq


@pytest.fixture(autouse=True)
def _clear_cache():
    sq.clear_quote_cache_for_tests()
    yield
    sq.clear_quote_cache_for_tests()


def test_default_symbol_order_dedupes_and_leads_with_indices():
    order = sq.default_symbol_order()

    assert order[:3] == ["^DJI", "^IXIC", "^GSPC"]
    assert "VRNO" in order
    assert order.count("AAPL") == 1
    assert set(sq.TOP_MARKET_CAP_SYMBOLS) <= set(order)


def test_fetch_quote_uses_info_price_and_change(monkeypatch):
    class _FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def info(self):
            return {"regularMarketPrice": 210.5, "previousClose": 200.0}

    monkeypatch.setattr(sq.yf, "Ticker", _FakeTicker)

    quote = sq.fetch_quote("AAPL")

    assert quote.symbol == "AAPL"
    assert quote.label == "AAPL"
    assert quote.price == 210.5
    assert quote.change == pytest.approx(10.5)
    assert quote.change_pct == pytest.approx(5.25)


def test_fetch_quote_labels_indices_by_friendly_name(monkeypatch):
    class _FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def info(self):
            return {"regularMarketPrice": 44000.0, "previousClose": 43900.0}

    monkeypatch.setattr(sq.yf, "Ticker", _FakeTicker)

    quote = sq.fetch_quote("^DJI")

    assert quote.label == "DJIA"


def test_fetch_quote_falls_back_to_history_when_info_unusable(monkeypatch):
    class _FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def info(self):
            return {"regularMarketPrice": 0.994, "previousClose": 1e-9}

        def history(self, period, interval):
            return pd.DataFrame({"Close": [1.055, 0.994]})

    monkeypatch.setattr(sq.yf, "Ticker", _FakeTicker)

    quote = sq.fetch_quote("VRNO")

    assert quote.price == 0.994
    assert quote.change == pytest.approx(-0.061)


def test_fetch_quote_returns_none_fields_on_total_failure(monkeypatch):
    class _FakeTicker:
        def __init__(self, symbol):
            raise RuntimeError("network down")

    monkeypatch.setattr(sq.yf, "Ticker", _FakeTicker)

    quote = sq.fetch_quote("AAPL")

    assert quote.price is None
    assert quote.change is None
    assert quote.change_pct is None


def test_fetch_stock_quotes_caches_until_ttl_elapses(monkeypatch):
    calls = []

    def fake_fetch(symbol):
        calls.append(symbol)
        return sq.StockQuote(symbol=symbol, label=symbol, price=1.0, change=0.1, change_pct=10.0)

    monkeypatch.setattr(sq, "fetch_quote", fake_fetch)
    monkeypatch.setattr(sq.config, "STOCK_TICKER_CACHE_TTL_SECONDS", 900)

    first = sq.fetch_stock_quotes(["AAPL", "MSFT"])
    second = sq.fetch_stock_quotes(["AAPL", "MSFT"])

    assert sorted(calls) == ["AAPL", "MSFT"]
    assert [q.symbol for q in first] == [q.symbol for q in second] == ["AAPL", "MSFT"]


def test_fetch_stock_quotes_refetches_after_ttl_elapses(monkeypatch):
    def fake_fetch(symbol):
        return sq.StockQuote(symbol=symbol, label=symbol, price=1.0, change=0.1, change_pct=10.0)

    monkeypatch.setattr(sq, "fetch_quote", fake_fetch)
    monkeypatch.setattr(sq.config, "STOCK_TICKER_CACHE_TTL_SECONDS", 60)

    sq.fetch_stock_quotes(["AAPL"])
    with sq._quotes_cache_lock:
        sq._quotes_cache_time = time.monotonic() - 3600.0

    calls = []

    def _tracking_fetch(symbol):
        calls.append(symbol)
        return fake_fetch(symbol)

    monkeypatch.setattr(sq, "fetch_quote", _tracking_fetch)
    sq.fetch_stock_quotes(["AAPL"])

    assert calls == ["AAPL"]


def test_fetch_stock_quotes_keeps_stale_quote_when_refetch_fails(monkeypatch):
    good = sq.StockQuote(symbol="AAPL", label="AAPL", price=200.0, change=1.0, change_pct=0.5)
    monkeypatch.setattr(sq, "fetch_quote", lambda symbol: good)
    sq.fetch_stock_quotes(["AAPL"], force=True)

    failed = sq.StockQuote(symbol="AAPL", label="AAPL", price=None, change=None, change_pct=None)
    monkeypatch.setattr(sq, "fetch_quote", lambda symbol: failed)

    result = sq.fetch_stock_quotes(["AAPL"], force=True)

    assert result == [good]


def test_fetch_stock_quotes_handles_empty_symbol_list():
    assert sq.fetch_stock_quotes([]) == []
