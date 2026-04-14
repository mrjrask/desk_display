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

    monkeypatch.setattr("services.data_provider.fetch_nfl_week_scoreboard", lambda now=None: [])
    monkeypatch.setattr("services.data_provider.fetch_nfl_next_scoreboard", lambda start_date, max_days=370: [{"league": "nfl"}])
    monkeypatch.setattr("services.data_provider.fetch_mlb_scoreboard", lambda day=None, now=None: [{"league": "mlb"}])
    monkeypatch.setattr("services.data_provider.fetch_nba_scoreboard", lambda day=None, now=None: [{"league": "nba"}])
    monkeypatch.setattr("services.data_provider.fetch_nhl_scoreboard", lambda day=None, now=None: [{"league": "nhl"}])
    monkeypatch.setattr("services.data_provider.fetch_ncaam_scoreboard", lambda day=None, now=None, mode=None: [{"league": "ncaam"}])

    payload = provider.read_sports_payloads(ttl_seconds=0)

    assert "wbc" not in payload["scoreboards"]


def test_read_sports_payloads_fetches_only_requested_leagues(monkeypatch):
    provider = DataProvider()
    calls = {"nfl": 0, "mlb": 0, "nba": 0, "ncaam": 0, "nhl": 0}

    def _track(league):
        def _fetch(*args, **kwargs):
            calls[league] += 1
            return [{"league": league}]

        return _fetch

    monkeypatch.setattr("services.data_provider.fetch_nfl_week_scoreboard", _track("nfl"))
    monkeypatch.setattr("services.data_provider.fetch_nfl_next_scoreboard", _track("nfl"))
    monkeypatch.setattr("services.data_provider.fetch_mlb_scoreboard", _track("mlb"))
    monkeypatch.setattr("services.data_provider.fetch_nba_scoreboard", _track("nba"))
    monkeypatch.setattr("services.data_provider.fetch_nhl_scoreboard", _track("nhl"))
    monkeypatch.setattr("services.data_provider.fetch_ncaam_scoreboard", _track("ncaam"))

    payload = provider.read_sports_payloads(ttl_seconds=0, leagues={"mlb"})

    assert payload["scoreboards"]["mlb"] == [{"league": "mlb"}]
    assert payload["scoreboards"]["nfl"] == []
    assert calls["mlb"] == 1
    assert calls["nfl"] == 0
    assert calls["nba"] == 0
    assert calls["ncaam"] == 0
    assert calls["nhl"] == 0


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

    monkeypatch.setattr("services.data_provider.fetch_nfl_week_scoreboard", lambda now=None: [{"league": "nfl"}])
    monkeypatch.setattr("services.data_provider.fetch_nfl_next_scoreboard", lambda start_date, max_days=370: [])
    monkeypatch.setattr("services.data_provider.fetch_mlb_scoreboard", lambda day=None, now=None: [{"league": "mlb"}])
    monkeypatch.setattr("services.data_provider.fetch_nba_scoreboard", lambda day=None, now=None: [{"league": "nba"}])
    monkeypatch.setattr("services.data_provider.fetch_ncaam_scoreboard", lambda day=None, now=None, mode=None: [{"league": "ncaam"}])
    monkeypatch.setattr("services.data_provider.fetch_nhl_scoreboard", lambda day=None, now=None: [{"league": "nhl"}])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: provider.read_sports_payloads(ttl_seconds=60), range(24)))

    for payload in results:
        assert sorted(payload["scoreboards"].keys()) == ["mlb", "nba", "ncaam", "nfl", "nhl"]
