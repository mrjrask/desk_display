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


def test_should_force_refresh_scoreboards_for_any_scoreboard_screen_when_online():
    main = _load_main()

    assert main._should_force_refresh_scoreboards("NFL Scoreboard", offline=False)
    assert main._should_force_refresh_scoreboards("NBA Scoreboard v2", offline=False)


def test_should_force_refresh_scoreboards_skips_non_scoreboard_and_offline():
    main = _load_main()

    assert not main._should_force_refresh_scoreboards("date", offline=False)
    assert not main._should_force_refresh_scoreboards("MLB Scoreboard", offline=True)


def test_feed_to_force_refresh_for_screen_handles_scoreboards_and_live_team_screens():
    main = _load_main()

    assert main._feed_to_force_refresh_for_screen("MLB Scoreboard", offline=False) == "scoreboards"
    assert main._feed_to_force_refresh_for_screen("cubs live", offline=False) == "cubs"
    assert main._feed_to_force_refresh_for_screen("sox live", offline=False) == "sox"


def test_feed_to_force_refresh_for_screen_skips_offline_and_non_live_screens():
    main = _load_main()

    assert main._feed_to_force_refresh_for_screen("cubs live", offline=True) is None
    assert main._feed_to_force_refresh_for_screen("date", offline=False) is None


def test_requested_scoreboard_leagues_only_includes_enabled_scoreboard_screens():
    main = _load_main()
    original_requested = main._requested_screen_ids
    try:
        main._requested_screen_ids = {"MLB Scoreboard", "date", "hawks next"}
        assert main._requested_scoreboard_leagues() == {"mlb"}
    finally:
        main._requested_screen_ids = original_requested
