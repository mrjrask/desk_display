import datetime

from screens.nhl_scoreboard import _map_api_web_game


def _game(**overrides):
    base = {
        "id": 1,
        "startTimeUTC": "2026-01-15T19:00:00Z",
        "awayTeam": {"abbrev": "CHI"},
        "homeTeam": {"abbrev": "STL"},
    }
    base.update(overrides)
    return base


def test_postponed_via_schedule_state_is_not_shown_as_scheduled():
    # Regression: `game.get("gameState") or game.get("gameScheduleState")`
    # only ever looked at gameScheduleState when gameState was falsy, so a
    # game reported as gameState "FUT" (not started) *and*
    # gameScheduleState "PPD" (postponed) at the same time showed as a
    # normal scheduled game.
    mapped = _map_api_web_game(
        _game(gameState="FUT", gameScheduleState="PPD"), datetime.date(2026, 1, 15)
    )

    assert mapped["status"]["detailedState"] == "Postponed"
    assert mapped["status"]["abstractGameState"] == "preview"


def test_normal_scheduled_game_is_unaffected():
    mapped = _map_api_web_game(
        _game(gameState="FUT", gameScheduleState="OK"), datetime.date(2026, 1, 15)
    )

    assert mapped["status"]["detailedState"] == "Scheduled"
    assert mapped["status"]["abstractGameState"] == "preview"


def test_live_game_is_unaffected():
    mapped = _map_api_web_game(
        _game(gameState="LIVE", gameScheduleState="OK"), datetime.date(2026, 1, 15)
    )

    assert mapped["status"]["abstractGameState"] == "live"
