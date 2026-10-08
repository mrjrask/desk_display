import datetime

import data_fetch
from config import CENTRAL_TIME


def _game(game_id, game_date):
    return {
        "id": game_id,
        "gameDate": game_date,
        "teams": {
            "home": {"team": {"id": data_fetch._BULLS_TEAM_ID}},
            "away": {"team": {"id": "1234"}},
        },
        "status": {"abstractGameState": "Preview"},
    }


def test_bulls_next_home_game_skips_first_when_same_as_next(monkeypatch):
    next_game = _game("game-1", "2024-10-01T00:00:00Z")
    later_home_game = _game("game-2", "2024-10-05T00:00:00Z")
    games = [next_game, later_home_game]

    def fake_future_games(_):
        for game in games:
            yield game

    monkeypatch.setattr(data_fetch, "_future_bulls_games", fake_future_games)

    assert data_fetch.fetch_bulls_next_home_game() == later_home_game


def test_bulls_lookahead_fetches_schedule_once_instead_of_scanning_days(monkeypatch):
    """Regression test: Bulls lookups must pull one cached team schedule,
    not scan the ESPN scoreboard endpoint one day at a time.

    A 120-day forward scan (one HTTP request per day) was enough to trip
    ESPN's rate limiter and 403 the shared site.api.espn.com host, which
    then blocked every other sport's requests -- including the NFL
    scoreboard -- for the circuit breaker's cooldown window.
    """
    today = datetime.datetime.now(CENTRAL_TIME).date()
    near_game = _game("game-near", f"{today + datetime.timedelta(days=2)}T00:00:00Z")
    far_game = _game("game-far", f"{today + datetime.timedelta(days=100)}T00:00:00Z")
    past_game = _game("game-past", f"{today - datetime.timedelta(days=3)}T00:00:00Z")
    schedule = [past_game, far_game, near_game]

    call_count = 0

    def fake_fetch_team_schedule(team_id):
        nonlocal call_count
        call_count += 1
        assert team_id == data_fetch._BULLS_TEAM_ID
        return schedule

    monkeypatch.setattr(data_fetch, "_nba_fetch_team_schedule", fake_fetch_team_schedule)

    result = data_fetch.fetch_bulls_next_game()

    assert result["id"] == "game-near"
    assert call_count == 1

    future_games = list(data_fetch._future_bulls_games(data_fetch._NBA_LOOKAHEAD_DAYS))
    assert [g["id"] for g in future_games] == ["game-near", "game-far"]

    past_games = list(data_fetch._past_bulls_games(data_fetch._NBA_LOOKBACK_DAYS))
    assert [g["id"] for g in past_games] == ["game-past"]

    # One schedule fetch per call site above -- never one request per day.
    assert call_count == 3


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _espn_event(event_id, date, home_abbr, away_abbr, home_id, away_id):
    return {
        "id": event_id,
        "date": date,
        "competitions": [
            {
                "id": event_id,
                "date": date,
                "status": {"type": {"state": "pre", "shortDetail": "Scheduled"}},
                "competitors": [
                    {"homeAway": "home", "team": {"id": home_id, "abbreviation": home_abbr}},
                    {"homeAway": "away", "team": {"id": away_id, "abbreviation": away_abbr}},
                ],
            }
        ],
    }


def test_bulls_next_and_next_home_use_espn_id_and_skip_preseason(monkeypatch):
    """Regression: the schedule was requested with NBA.com's team id
    (1610612741), which ESPN does not know, and without ``seasontype``, so
    before opening night Bulls Next and Next Home both said "no games"."""

    from services.sports import nba

    today = datetime.datetime.now(CENTRAL_TIME).date()
    preseason_away = _espn_event(
        "pre-1", f"{today + datetime.timedelta(days=3)}T00:00Z", "DEN", "CHI", "7", "4"
    )
    regular_home = _espn_event(
        "reg-1", f"{today + datetime.timedelta(days=20)}T00:00Z", "CHI", "DET", "4", "8"
    )
    regular_away = _espn_event(
        "reg-2", f"{today + datetime.timedelta(days=22)}T00:00Z", "MIL", "CHI", "15", "4"
    )
    payloads = {
        "1": {"events": [preseason_away]},
        "2": {"events": [regular_home, regular_away]},
        "3": {"events": []},
    }
    requested = []

    def fake_get(url, timeout=None, **_kwargs):
        requested.append(url)
        assert "/teams/4/schedule" in url
        season_type = url.rsplit("seasontype=", 1)[1]
        return _FakeResponse(payloads[season_type])

    monkeypatch.setattr(nba._SESSION, "get", fake_get)
    monkeypatch.setattr(nba, "_team_schedule_cache", {})

    next_game = data_fetch.fetch_bulls_next_game()
    next_home = data_fetch.fetch_bulls_next_home_game()

    # Preseason games are never shown, and are not even requested.
    assert next_game["gamePk"] == "reg-1"
    # The next game is the home game, so next-home skips it (the registry falls
    # back to the next game); the preseason opener is not a candidate either.
    assert next_home is None
    assert sorted(url.rsplit("seasontype=", 1)[1] for url in requested) == ["2", "3"]


def test_espn_team_id_maps_nba_com_ids():
    from services.sports import nba

    assert nba.espn_team_id("1610612741") == "4"
    assert nba.espn_team_id("4") == "4"
