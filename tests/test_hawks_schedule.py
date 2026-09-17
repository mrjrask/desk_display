"""Tests for Hawks schedule helpers."""

from screens.draw_hawks_schedule import _last_game_result_prefix, _team_full_name


def test_team_full_name_removes_feed_icons_and_hidden_characters():
    team = {"name": {"default": "\u200b🏒\u00a0Minnesota Wild™"}}

    assert _team_full_name(team) == "Minnesota Wild"


def test_team_full_name_preserves_valid_name_punctuation():
    team = {"name": {"default": "St. John's A&M"}}

    assert _team_full_name(team) == "St. John's A&M"


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


def test_last_game_result_from_feed_period_type():
    feed = {"periodType": "Shootout"}

    assert _last_game_result_prefix({}, feed) == "Final/SO"


def test_last_game_result_from_feed_numeric_period():
    feed = {"perOrdinal": 4}

    assert _last_game_result_prefix({}, feed) == "Final/OT"
