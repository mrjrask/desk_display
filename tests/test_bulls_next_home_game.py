import data_fetch


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
