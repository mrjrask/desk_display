"""Contract tests for sports service-layer scoreboard fetchers."""

from __future__ import annotations

import datetime as dt

from services.sports import mlb, nba, ncaam, nfl, nhl


def test_mlb_fetch_scoreboard_contract(monkeypatch):
    now = dt.datetime(2026, 3, 30, 12, 0)
    day = dt.date(2026, 3, 30)

    monkeypatch.setattr(mlb, "scoreboard_date", lambda current_now=None: day)
    monkeypatch.setattr(mlb, "_fetch_games_for_date", lambda requested_day: [{"id": "mlb-1", "day": requested_day.isoformat()}])

    payload = mlb.fetch_scoreboard(now=now)

    assert isinstance(payload, list)
    assert payload == [{"id": "mlb-1", "day": "2026-03-30"}]


def test_nba_fetch_scoreboard_contract(monkeypatch):
    day = dt.date(2026, 3, 30)
    monkeypatch.setattr(nba, "scoreboard_date", lambda current_now=None: day)
    monkeypatch.setattr(nba, "fetch_games_for_date", lambda requested_day: [{"id": "nba-1", "date": requested_day.isoformat()}])

    payload = nba.fetch_scoreboard(now=dt.datetime(2026, 3, 30, 8, 30))

    assert isinstance(payload, list)
    assert payload[0]["id"] == "nba-1"


def test_nhl_fetch_scoreboard_contract(monkeypatch):
    day = dt.date(2026, 3, 30)
    monkeypatch.setattr(nhl, "scoreboard_date", lambda current_now=None: day)
    monkeypatch.setattr(nhl, "_fetch_games_for_date", lambda requested_day: [{"id": "nhl-1", "date": requested_day.isoformat()}])

    payload = nhl.fetch_scoreboard(now=dt.datetime(2026, 3, 30, 8, 30))

    assert isinstance(payload, list)
    assert payload == [{"id": "nhl-1", "date": "2026-03-30"}]


def test_ncaam_fetch_scoreboard_contract(monkeypatch):
    day = dt.date(2026, 3, 30)
    monkeypatch.setattr(ncaam, "scoreboard_date", lambda current_now=None: day)
    monkeypatch.setattr(
        ncaam,
        "_fetch_games_for_date",
        lambda requested_day, mode=None: [{"id": "ncaam-1", "date": requested_day.isoformat(), "mode": mode}],
    )

    payload = ncaam.fetch_scoreboard(now=dt.datetime(2026, 3, 30, 8, 30), mode="top25")

    assert isinstance(payload, list)
    assert payload == [{"id": "ncaam-1", "date": "2026-03-30", "mode": "top25"}]


def test_nfl_fetch_scoreboard_contract(monkeypatch):
    now = dt.datetime(2026, 3, 30, 8, 30)
    monkeypatch.setattr(nfl, "fetch_week_scoreboard", lambda now=None: [])
    monkeypatch.setattr(
        nfl,
        "fetch_next_scoreboard",
        lambda start_date, max_days=370: [{"id": "nfl-1", "start_date": start_date.isoformat()}],
    )

    payload = nfl.fetch_scoreboard(now=now)

    assert isinstance(payload, list)
    assert payload == [{"id": "nfl-1", "start_date": "2026-03-30"}]


def test_nflverse_schedule_is_cached_across_long_range_scan(monkeypatch):
    class _Response:
        text = """gameday,gametime,away_team,home_team,game_id,away_score,home_score
2026-09-10,19:20,DAL,PHI,game-1,,
2026-09-17,19:15,LAC,KC,game-2,,
2026-09-24,19:15,SEA,LAR,game-3,,
"""

        def raise_for_status(self):
            return None

    class _Session:
        def __init__(self):
            self.calls = 0

        def get(self, url, timeout):
            self.calls += 1
            return _Response()

    session = _Session()
    cache = {}
    monotonic_times = iter((0.0, 40.0, 80.0))
    monkeypatch.setattr(nfl.time, "monotonic", lambda: next(monotonic_times))

    first_week = nfl._fetch_nflverse(
        dt.date(2026, 9, 7), dt.date(2026, 9, 13), session=session, cache=cache
    )
    second_week = nfl._fetch_nflverse(
        dt.date(2026, 9, 14), dt.date(2026, 9, 20), session=session, cache=cache
    )
    third_week = nfl._fetch_nflverse(
        dt.date(2026, 9, 21), dt.date(2026, 9, 27), session=session, cache=cache
    )

    assert [game["id"] for game in first_week] == ["game-1"]
    assert [game["id"] for game in second_week] == ["game-2"]
    assert [game["id"] for game in third_week] == ["game-3"]
    assert session.calls == 1


def test_nfl_range_falls_back_after_empty_espn_site_response(monkeypatch):
    day = dt.date(2026, 9, 10)
    calls = []

    def _fake_fetch_espn(url, dates, *, session):
        calls.append(url)
        if url == nfl._SITE_SCOREBOARD_URL:
            return []
        return [{"id": "cdn-game"}]

    monkeypatch.setattr(nfl, "_fetch_espn", _fake_fetch_espn)
    monkeypatch.setattr(
        nfl,
        "_fetch_nflverse",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("nflverse should not be used when the CDN has games")
        ),
    )

    cache = {}
    games = nfl.fetch_range(day, day, session=object(), cache=cache)

    assert games == [{"id": "cdn-game"}]
    assert calls == [nfl._SITE_SCOREBOARD_URL, nfl._CDN_SCOREBOARD_URL + "?xhr=1"]
    assert cache[(day, day, "nfl_scoreboard_range")][1] == games


def test_services_normalize_non_list_to_empty_list(monkeypatch):
    day = dt.date(2026, 3, 30)

    monkeypatch.setattr(mlb, "_fetch_games_for_date", lambda requested_day: None)
    monkeypatch.setattr(nba, "fetch_games_for_date", lambda requested_day: {"not": "a list"})
    monkeypatch.setattr(nhl, "_fetch_games_for_date", lambda requested_day: "bad")
    monkeypatch.setattr(ncaam, "_fetch_games_for_date", lambda requested_day, mode=None: 5)

    assert mlb.fetch_scoreboard(day=day) == []
    assert nba.fetch_scoreboard(day=day) == []
    assert nhl.fetch_scoreboard(day=day) == []
    assert ncaam.fetch_scoreboard(day=day) == []


def test_ncaam_tournament_mode_advances_to_next_day_with_games(monkeypatch):
    day = dt.date(2026, 3, 30)

    monkeypatch.setattr(ncaam, "scoreboard_date", lambda current_now=None: day)

    def _fake_fetch(requested_day, mode=None):
        if requested_day == day:
            return []
        if requested_day == day + dt.timedelta(days=1):
            return [{"id": "ncaam-2", "date": requested_day.isoformat(), "mode": mode}]
        return []

    monkeypatch.setattr(ncaam, "_fetch_games_for_date", _fake_fetch)

    payload = ncaam.fetch_scoreboard(now=dt.datetime(2026, 3, 30, 8, 30), mode="tournament")

    assert payload == [{"id": "ncaam-2", "date": "2026-03-31", "mode": "tournament"}]
