"""Tests for scoreboard status string formatting."""

import datetime

import pytest

from config import CENTRAL_TIME
from screens.mlb_scoreboard import (
    _fetch_games_for_date as mlb_fetch_games_for_date,
    _format_status as mlb_format_status,
    _scoreboard_date as mlb_scoreboard_date,
)
from screens.nba_scoreboard import _format_status as nba_format_status
from screens.nfl_scoreboard import _format_status as nfl_format_status
from screens.nhl_scoreboard import _format_status as nhl_format_status


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
        ("Pre-Game Warmup", "Warmup"),
        ("Warm-Up", "Warmup"),
        ("Delayed", "Delayed"),
        ("Postponed", "Postponed"),
        ("Canceled", "Canceled"),
        ("Cancelled", "Canceled"),
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


def test_mlb_scoreboard_date_uses_temporary_override_window():
    now = datetime.datetime(2026, 2, 19, 12, 0, tzinfo=CENTRAL_TIME)
    assert mlb_scoreboard_date(now) == datetime.date(2026, 2, 20)


def test_mlb_scoreboard_date_uses_normal_cutoff_after_override_window():
    before_cutoff = datetime.datetime(2026, 2, 21, 10, 9, tzinfo=CENTRAL_TIME)
    after_cutoff = datetime.datetime(2026, 2, 21, 10, 11, tzinfo=CENTRAL_TIME)

    assert mlb_scoreboard_date(before_cutoff) == datetime.date(2026, 2, 20)
    assert mlb_scoreboard_date(after_cutoff) == datetime.date(2026, 2, 21)


def test_mlb_fetch_includes_spring_training_and_all_star_filters(monkeypatch):
    captured = {}

    class _DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"dates": []}

    class _DummySession:
        def get(self, url, timeout):
            captured["url"] = url
            captured["timeout"] = timeout
            return _DummyResponse()

    monkeypatch.setattr("screens.mlb_scoreboard._SESSION", _DummySession())
    monkeypatch.setattr("screens.mlb_scoreboard._GAMES_CACHE", {})

    mlb_fetch_games_for_date(datetime.date(2026, 3, 1))

    assert "sportId=1" in captured["url"]
    assert "gameTypes=S,E,R,A,F,D,L,W" in captured["url"]


def test_nba_live_status_drops_leading_zero_minutes():
    game = {
        "status": {"abstractGameState": "live", "detailedState": "In Progress", "statusCode": "2"},
        "linescore": {"currentPeriodOrdinal": "Q1", "currentPeriodTimeRemaining": "05:00"},
    }
    assert nba_format_status(game) == "5:00 Q1"


def test_nhl_live_status_drops_leading_zero_minutes():
    game = {
        "status": {"abstractGameState": "live", "detailedState": "In Progress", "statusCode": "3"},
        "linescore": {"currentPeriodOrdinal": "1st", "currentPeriodTimeRemaining": "05:00"},
    }
    assert nhl_format_status(game) == "1ST 5:00"


def test_mlb_live_detailed_status_drops_leading_zero_minutes():
    game = {
        "status": {"abstractGameState": "live", "detailedState": "Rain Delay 05:00", "statusCode": "I"},
        "linescore": {},
    }
    assert mlb_format_status(game) == "Rain Delay 5:00"
