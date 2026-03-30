from __future__ import annotations

import datetime as dt

from config import CENTRAL_TIME
from services.sports import nhl


def test_scoreboard_date_rolls_before_central_cutoff():
    before_cutoff = dt.datetime(2026, 3, 30, 9, 29, tzinfo=CENTRAL_TIME)
    at_cutoff = dt.datetime(2026, 3, 30, 9, 30, tzinfo=CENTRAL_TIME)

    assert nhl.scoreboard_date(before_cutoff) == dt.date(2026, 3, 29)
    assert nhl.scoreboard_date(at_cutoff) == dt.date(2026, 3, 30)


def test_prepare_scoreboard_data_hydrates_and_sorts_games():
    payload = [
        {"gamePk": 2, "gameDate": "2026-03-30T15:00:00Z"},
        {"gamePk": 1, "gameDate": "2026-03-30T14:00:00Z"},
    ]

    prepared = nhl.prepare_scoreboard_data(payload)

    assert [game["gamePk"] for game in prepared] == [1, 2]
    assert all("_start_local" in game for game in prepared)


def test_fetch_scoreboard_uses_service_fetch_and_normalizes_non_list(monkeypatch):
    day = dt.date(2026, 3, 30)
    monkeypatch.setattr(nhl, "scoreboard_date", lambda now=None: day)
    monkeypatch.setattr(nhl, "_fetch_games_for_date", lambda requested_day: {"bad": requested_day.isoformat()})

    assert nhl.fetch_scoreboard(now=dt.datetime(2026, 3, 30, 8, 30)) == []
