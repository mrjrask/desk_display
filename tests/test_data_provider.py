"""Tests for data provider stale fallback behavior."""

from concurrent.futures import ThreadPoolExecutor

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

    monkeypatch.setattr("services.data_provider._fetch_nfl_games_for_week", lambda now: [])
    monkeypatch.setattr("services.data_provider._fetch_nfl_next_games", lambda day: [{"league": "nfl"}])
    monkeypatch.setattr("services.data_provider._mlb_scoreboard_date", lambda now: now.date())
    monkeypatch.setattr("services.data_provider._nba_scoreboard_date", lambda now: now.date())
    monkeypatch.setattr("services.data_provider._nhl_scoreboard_date", lambda now: now.date())
    monkeypatch.setattr("services.data_provider._fetch_mlb_games_for_date", lambda day: [{"league": "mlb"}])
    monkeypatch.setattr("services.data_provider._fetch_nba_games_for_date", lambda day: [{"league": "nba"}])
    monkeypatch.setattr("services.data_provider._fetch_nhl_games_for_date", lambda day: [{"league": "nhl"}])

    payload = provider.read_sports_payloads(ttl_seconds=0)

    assert "wbc" not in payload["scoreboards"]


def test_read_weather_is_safe_under_concurrent_access(monkeypatch):
    provider = DataProvider()
    calls = {"count": 0}

    def fake_fetch_weather(*, force_refresh=True):
        calls["count"] += 1
        return {"temp": 72}

    monkeypatch.setattr("services.data_provider.data_fetch.fetch_weather", fake_fetch_weather)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: provider.read_weather(ttl_seconds=60), range(40)))

    assert all(result == {"temp": 72} for result in results)
    assert calls["count"] >= 1


def test_read_sports_payloads_is_safe_under_concurrent_access(monkeypatch):
    provider = DataProvider()

    monkeypatch.setattr("services.data_provider._fetch_nfl_games_for_week", lambda now: [{"league": "nfl"}])
    monkeypatch.setattr("services.data_provider._fetch_nfl_next_games", lambda day: [])
    monkeypatch.setattr("services.data_provider._mlb_scoreboard_date", lambda now: now.date())
    monkeypatch.setattr("services.data_provider._nba_scoreboard_date", lambda now: now.date())
    monkeypatch.setattr("services.data_provider._ncaam_scoreboard_date", lambda now: now.date())
    monkeypatch.setattr("services.data_provider._nhl_scoreboard_date", lambda now: now.date())
    monkeypatch.setattr("services.data_provider._fetch_mlb_games_for_date", lambda day: [{"league": "mlb"}])
    monkeypatch.setattr("services.data_provider._fetch_nba_games_for_date", lambda day: [{"league": "nba"}])
    monkeypatch.setattr("services.data_provider._fetch_ncaam_games_for_date", lambda day: [{"league": "ncaam"}])
    monkeypatch.setattr("services.data_provider._fetch_nhl_games_for_date", lambda day: [{"league": "nhl"}])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: provider.read_sports_payloads(ttl_seconds=60), range(24)))

    for payload in results:
        assert sorted(payload["scoreboards"].keys()) == ["mlb", "nba", "ncaam", "nfl", "nhl"]
