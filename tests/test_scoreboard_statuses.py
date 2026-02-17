"""Tests for scoreboard status string formatting."""

import datetime

import pytest

from screens.mlb_scoreboard import _format_status as mlb_format_status
from screens.mlb_scoreboard import _scoreboard_date as mlb_scoreboard_date
from screens.nfl_scoreboard import _format_status as nfl_format_status
from config import CENTRAL_TIME


def _mlb_game(*, detailed: str, abstract: str = "preview", start: bool = True) -> dict:
    game = {
        "status": {
            "abstractGameState": abstract,
            "detailedState": detailed,
            "statusCode": "",
        },
        "linescore": {},
    }
    if start:
        game["_start_local"] = datetime.datetime(2024, 6, 1, 12, 30)
    return game


@pytest.mark.parametrize(
    "detailed, expected",
    [
        ("Warmup", "Warmup"),
        ("Delayed", "Delayed"),
        ("Postponed", "Postponed"),
    ],
)
def test_mlb_status_overrides_start_time(detailed: str, expected: str):
    game = _mlb_game(detailed=detailed)
    assert mlb_format_status(game) == expected


def _nfl_game(*, state: str, short: str = "", detail: str = "", clock: str = "", period=None) -> dict:
    return {
        "status": {
            "type": {
                "state": state,
                "shortDetail": short,
                "detail": detail,
            },
            "displayClock": clock,
            "period": period,
        }
    }


@pytest.mark.parametrize(
    "short_detail",
    ["End of the 1st", "Halftime", "End of the 3rd"],
)
def test_nfl_in_game_status_overrides_clock(short_detail: str):
    period = {"End of the 1st": 1, "Halftime": 2, "End of the 3rd": 3}[short_detail]
    game = _nfl_game(state="in", short=short_detail, detail=short_detail, clock="0:00", period=period)
    assert nfl_format_status(game) == short_detail


def test_mlb_scoreboard_date_pinned_to_spring_training_before_opening_day():
    now = datetime.datetime(2026, 2, 19, 12, 0, tzinfo=CENTRAL_TIME)
    assert mlb_scoreboard_date(now) == datetime.date(2026, 2, 20)


def test_mlb_scoreboard_date_uses_normal_cutoff_on_opening_day():
    before_cutoff = datetime.datetime(2026, 2, 20, 8, 0, tzinfo=CENTRAL_TIME)
    after_cutoff = datetime.datetime(2026, 2, 20, 10, 0, tzinfo=CENTRAL_TIME)

    assert mlb_scoreboard_date(before_cutoff) == datetime.date(2026, 2, 19)
    assert mlb_scoreboard_date(after_cutoff) == datetime.date(2026, 2, 20)
