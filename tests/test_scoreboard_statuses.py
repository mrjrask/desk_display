"""Tests for scoreboard status string formatting."""

import datetime

import pytest

from screens.mlb_scoreboard import _format_status as mlb_format_status
from screens.nfl_scoreboard import _format_status as nfl_format_status


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

