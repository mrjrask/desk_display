"""Tests for scoreboard live-refresh behavior in main."""

import datetime
import importlib
import sys


def _load_main():
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def test_is_live_scoreboard_game_detects_in_progress_states():
    main = _load_main()

    assert main._is_live_scoreboard_game({"status": {"abstractGameState": "Live"}})
    assert main._is_live_scoreboard_game({"status": {"codedGameState": "I"}})
    assert main._is_live_scoreboard_game({"status": {"type": {"state": "in"}}})
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


def test_should_force_refresh_scoreboards_during_live_game_when_online():
    main = _load_main()
    main.cache["scoreboards"]["nfl"] = [{"status": {"abstractGameState": "Live"}}]

    assert main._should_force_refresh_scoreboards("NFL Scoreboard", offline=False)
    assert main._should_force_refresh_scoreboards("NBA Scoreboard v2", offline=False)


def test_should_force_refresh_scoreboards_skips_non_scoreboard_and_offline():
    main = _load_main()

    assert not main._should_force_refresh_scoreboards("date", offline=False)
    assert not main._should_force_refresh_scoreboards("MLB Scoreboard", offline=True)


def test_should_force_refresh_scoreboards_during_scheduled_game_window():
    main = _load_main()
    now = datetime.datetime(2026, 9, 8, 20, 0, tzinfo=datetime.UTC)
    scoreboards = {
        "nfl": [
            {
                "_event_date": "2026-09-08T19:00:00Z",
                "status": {"type": {"state": "pre"}},
            }
        ]
    }

    assert main._scoreboards_in_live_window(scoreboards, now=now)
    assert not main._scoreboards_in_live_window(
        scoreboards, now=now + datetime.timedelta(hours=4)
    )


def test_scoreboard_live_window_starts_before_scheduled_game():
    main = _load_main()
    start = datetime.datetime(2026, 9, 8, 20, 0, tzinfo=datetime.UTC)
    scoreboards = {"nfl": [{"_event_date": start.isoformat()}]}

    assert main._scoreboards_in_live_window(
        scoreboards, now=start - datetime.timedelta(minutes=30)
    )
    assert not main._scoreboards_in_live_window(
        scoreboards, now=start - datetime.timedelta(minutes=31)
    )


def test_scoreboard_refresh_is_due_when_provider_date_rolls_over(monkeypatch):
    main = _load_main()
    main._requested_screen_ids = {"MLB Scoreboard"}
    main._last_scoreboard_refresh_dates.clear()
    main._last_scoreboard_refresh_dates["mlb"] = datetime.date(2026, 9, 7)
    monkeypatch.setattr(
        main,
        "_scoreboard_date_for_league",
        lambda league, now=None: datetime.date(2026, 9, 8),
    )

    assert main._scoreboard_refresh_dates_changed()


def test_record_scoreboard_refresh_dates_clears_rollover(monkeypatch):
    main = _load_main()
    main._requested_screen_ids = {"MLB Scoreboard"}
    main._last_scoreboard_refresh_dates.clear()
    monkeypatch.setattr(
        main,
        "_scoreboard_date_for_league",
        lambda league, now=None: datetime.date(2026, 9, 8),
    )

    main._record_scoreboard_refresh_dates()

    assert main._last_scoreboard_refresh_dates == {"mlb": datetime.date(2026, 9, 8)}
    assert not main._scoreboard_refresh_dates_changed()


def test_scoreboard_schedule_refresh_interval_is_daily():
    main = _load_main()

    assert main._FEED_REFRESH_INTERVALS["scoreboards"] == 24 * 60 * 60


def test_feed_to_force_refresh_for_screen_handles_scoreboards_and_live_team_screens():
    main = _load_main()
    main.cache["scoreboards"]["mlb"] = [{"status": {"abstractGameState": "Live"}}]

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
