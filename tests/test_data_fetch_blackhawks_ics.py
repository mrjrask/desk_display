import datetime

import pytz

import data_fetch


def _FixedDatetime(fixed_now):
    """A datetime.datetime subclass whose now()/utcnow() return a fixed instant."""

    class _Fixed(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now

        @classmethod
        def utcnow(cls):
            return fixed_now.astimezone(pytz.UTC).replace(tzinfo=None)

    return _Fixed


def test_blackhawks_ics_url_converts_webcal_to_https():
    assert (
        data_fetch._calendar_url("webcal://ics.ecal.com/ecal-sub/example/NHL.ics")
        == "https://ics.ecal.com/ecal-sub/example/NHL.ics"
    )


def test_normalize_blackhawks_ics_game_maps_schedule_payload():
    event = {
        "UID": "game-1",
        "SUMMARY": "Chicago Blackhawks at St Louis Blues",
        "DTSTART": "20261001T000000Z",
        "LOCATION": "Enterprise Center",
        "__params__": {"DTSTART": {}},
    }

    game = data_fetch._normalize_blackhawks_ics_game(event)

    assert game is not None
    assert game["gameState"] == "FUT"
    assert game["awayTeam"]["abbrev"] == "CHI"
    assert game["awayTeam"]["id"] == data_fetch.NHL_TEAM_ID
    assert game["homeTeam"]["abbrev"] == "STL"
    assert game["startTimeUTC"] == "2026-10-01T00:00:00Z"
    assert game["venue"]["default"] == "Enterprise Center"


def test_fetch_blackhawks_next_game_prefers_ics_schedule(monkeypatch):
    now = datetime.datetime(2026, 7, 20, tzinfo=pytz.UTC)
    games = [
        {
            "gameDate": "2026-10-01T00:00:00Z",
            "gameState": "FUT",
            "startTimeUTC": "2026-10-01T00:00:00Z",
            "homeTeam": {"abbrev": "STL"},
            "awayTeam": {"abbrev": "CHI"},
        }
    ]
    monkeypatch.setattr(data_fetch, "_fetch_blackhawks_ics_schedule", lambda: games)

    result = data_fetch.fetch_blackhawks_next_game()

    assert result is games[0]
    assert result["startTimeCentral"] == (
        now.replace(month=9, day=30, hour=19).strftime("%I:%M %p").lstrip("0")
    )


def test_fetch_blackhawks_schedule_fallback_does_not_use_expired_season(monkeypatch):
    requested_urls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"games": []}

    def fake_get(url, **_kwargs):
        requested_urls.append(url)
        return Response()

    monkeypatch.setattr(data_fetch, "_fetch_blackhawks_ics_schedule", list)
    monkeypatch.setattr(data_fetch._session, "get", fake_get)

    assert data_fetch._fetch_blackhawks_schedule_games() == []
    assert requested_urls == [data_fetch.NHL_API_URL]
    assert "20252026" not in requested_urls[0]


def test_fetch_blackhawks_next_game_skips_already_played_ics_games(monkeypatch):
    """ICS games are always tagged gameState="FUT", even once they've been
    played, since the calendar feed carries no live-result data. The next
    game lookup must fall back to comparing start times against "now" so a
    played game doesn't get stuck showing as upcoming."""
    now = datetime.datetime(2026, 10, 10, tzinfo=pytz.UTC)
    played_game = {
        "gameDate": "2026-10-01T00:00:00Z",
        "gameState": "FUT",
        "startTimeUTC": "2026-10-01T00:00:00Z",
        "homeTeam": {"abbrev": "STL"},
        "awayTeam": {"abbrev": "CHI"},
    }
    upcoming_game = {
        "gameDate": "2026-10-15T00:00:00Z",
        "gameState": "FUT",
        "startTimeUTC": "2026-10-15T00:00:00Z",
        "homeTeam": {"abbrev": "CHI"},
        "awayTeam": {"abbrev": "NYR"},
    }
    monkeypatch.setattr(
        data_fetch, "_fetch_blackhawks_schedule_games", lambda: [played_game, upcoming_game]
    )
    monkeypatch.setattr(data_fetch.datetime, "datetime", _FixedDatetime(now))

    result = data_fetch.fetch_blackhawks_next_game()

    assert result is upcoming_game


def test_fetch_blackhawks_last_game_returns_recently_played_ics_game(monkeypatch):
    """fetch_blackhawks_last_game must not require gameState=="OFF", since
    ICS-sourced games never carry that state."""
    now = datetime.datetime(2026, 10, 10, tzinfo=pytz.UTC)
    played_game = {
        "gameDate": "2026-10-01T00:00:00Z",
        "gameState": "FUT",
        "startTimeUTC": "2026-10-01T00:00:00Z",
        "homeTeam": {"abbrev": "STL"},
        "awayTeam": {"abbrev": "CHI"},
    }
    upcoming_game = {
        "gameDate": "2026-10-15T00:00:00Z",
        "gameState": "FUT",
        "startTimeUTC": "2026-10-15T00:00:00Z",
        "homeTeam": {"abbrev": "CHI"},
        "awayTeam": {"abbrev": "NYR"},
    }
    monkeypatch.setattr(
        data_fetch, "_fetch_blackhawks_schedule_games", lambda: [played_game, upcoming_game]
    )
    monkeypatch.setattr(data_fetch.datetime, "datetime", _FixedDatetime(now))

    result = data_fetch.fetch_blackhawks_last_game()

    assert result is played_game


def test_fetch_blackhawks_next_home_game_does_not_duplicate_next_game(monkeypatch):
    """Once fetch_blackhawks_next_game correctly skips a played game, the
    next-home lookup's duplicate check must compare against that corrected
    game rather than an earlier already-played one."""
    now = datetime.datetime(2026, 10, 10, tzinfo=pytz.UTC)
    played_home_game = {
        "gameDate": "2026-10-01T00:00:00Z",
        "gameState": "FUT",
        "startTimeUTC": "2026-10-01T00:00:00Z",
        "homeTeam": {"abbrev": "CHI", "id": data_fetch.NHL_TEAM_ID},
        "awayTeam": {"abbrev": "STL"},
    }
    upcoming_away_game = {
        "gameDate": "2026-10-15T00:00:00Z",
        "gameState": "FUT",
        "startTimeUTC": "2026-10-15T00:00:00Z",
        "homeTeam": {"abbrev": "NYR"},
        "awayTeam": {"abbrev": "CHI", "id": data_fetch.NHL_TEAM_ID},
    }
    upcoming_home_game = {
        "gameDate": "2026-10-20T00:00:00Z",
        "gameState": "FUT",
        "startTimeUTC": "2026-10-20T00:00:00Z",
        "homeTeam": {"abbrev": "CHI", "id": data_fetch.NHL_TEAM_ID},
        "awayTeam": {"abbrev": "DAL"},
    }
    monkeypatch.setattr(
        data_fetch,
        "_fetch_blackhawks_schedule_games",
        lambda: [played_home_game, upcoming_away_game, upcoming_home_game],
    )
    monkeypatch.setattr(data_fetch.datetime, "datetime", _FixedDatetime(now))

    next_game = data_fetch.fetch_blackhawks_next_game()
    next_home_game = data_fetch.fetch_blackhawks_next_home_game()

    assert next_game is upcoming_away_game
    assert next_home_game is upcoming_home_game


def test_fetch_blackhawks_live_game_treats_crit_as_live(monkeypatch):
    game = {
        "gameDate": "2026-10-01",
        "gameState": "CRIT",
        "startTimeUTC": "2026-10-01T00:00:00Z",
    }
    monkeypatch.setattr(data_fetch, "_fetch_blackhawks_schedule_games", lambda: [game])

    result = data_fetch.fetch_blackhawks_live_game()

    assert result is game
    assert result["startTimeCentral"] == "7:00 PM"
