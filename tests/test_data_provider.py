"""Tests for data provider stale fallback behavior."""

from services.data_provider import DataProvider


def test_read_cached_keeps_stale_value_when_fetch_returns_none():
    provider = DataProvider()
    calls = {"count": 0}

    def fetcher():
        calls["count"] += 1
        return {"temp": 70} if calls["count"] == 1 else None

    first = provider._read_cached("weather", fetcher, ttl_seconds=0)
    second = provider._read_cached("weather", fetcher, ttl_seconds=0)

    assert first == {"temp": 70}
    assert second == {"temp": 70}


def test_read_cached_does_not_cache_none_without_existing_value():
    provider = DataProvider()
    calls = {"count": 0}

    def fetcher():
        calls["count"] += 1
        return None

    first = provider._read_cached("weather", fetcher, ttl_seconds=300)
    second = provider._read_cached("weather", fetcher, ttl_seconds=300)

    assert first is None
    assert second is None
    assert calls["count"] == 2


def test_read_sports_payloads_excludes_wbc_scoreboard(monkeypatch):
    provider = DataProvider()

    monkeypatch.setattr("services.data_provider.fetch_nfl_week_scoreboard", lambda now=None: [])
    monkeypatch.setattr("services.data_provider.fetch_nfl_next_scoreboard", lambda start_date, max_days=370: [{"league": "nfl"}])
    monkeypatch.setattr("services.data_provider.fetch_mlb_scoreboard", lambda day=None, now=None: [{"league": "mlb"}])
    monkeypatch.setattr("services.data_provider.fetch_nba_scoreboard", lambda day=None, now=None: [{"league": "nba"}])
    monkeypatch.setattr("services.data_provider.fetch_nhl_scoreboard", lambda day=None, now=None: [{"league": "nhl"}])
    monkeypatch.setattr("services.data_provider.fetch_ncaam_scoreboard", lambda day=None, now=None, mode=None: [{"league": "ncaam"}])

    payload = provider.read_sports_payloads(ttl_seconds=0)

    assert "wbc" not in payload["scoreboards"]
