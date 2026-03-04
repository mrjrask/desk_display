"""Tests for scoreboard live-refresh behavior in main."""

import importlib
import sys


def _load_main():
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def test_is_live_scoreboard_game_detects_in_progress_states():
    main = _load_main()

    assert main._is_live_scoreboard_game({"status": {"abstractGameState": "Live"}})
    assert main._is_live_scoreboard_game({"status": {"codedGameState": "I"}})
    assert main._is_live_scoreboard_game({"statusCode": "2"})


def test_is_live_scoreboard_game_excludes_final_and_scheduled_states():
    main = _load_main()

    assert not main._is_live_scoreboard_game({"status": {"detailedState": "Final"}})
    assert not main._is_live_scoreboard_game({"status": {"abstractGameState": "Preview"}})


def test_scoreboards_have_live_games_checks_all_leagues():
    main = _load_main()

    scoreboards = {
        "nfl": [{"status": {"detailedState": "Final"}}],
        "mlb": [{"status": {"statusCode": "I"}}],
        "nba": [],
        "nhl": [],
    }

    assert main._scoreboards_have_live_games(scoreboards)
