"""Tests for Hawks schedule helpers."""

from screens.draw_hawks_schedule import _last_game_result_prefix


def test_last_game_result_overtime_from_outcome():
    game = {"gameOutcome": {"lastPeriodType": "Overtime"}}

    assert _last_game_result_prefix(game, None) == "Final/OT"


def test_last_game_result_shootout_variants():
    shootout_cases = [
        {"linescore": {"hasShootout": True}},
        {"gameOutcome": {"lastPeriodType": "Shootout"}},
        {"period": {"periodType": "Shootout"}},
    ]

    for game in shootout_cases:
        assert _last_game_result_prefix(game, None) == "Final/SO"


def test_last_game_result_overtime_period_text():
    game = {"period": {"ordinal": "Overtime"}}

    assert _last_game_result_prefix(game, None) == "Final/OT"


def test_last_game_result_overtime_from_feed():
    feed = {"perOrdinal": "Overtime"}

    assert _last_game_result_prefix({}, feed) == "Final/OT"
