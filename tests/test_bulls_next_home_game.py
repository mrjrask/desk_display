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
