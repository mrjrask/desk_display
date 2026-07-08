import datetime as dt

from services.sports import world_cup
from screens import world_cup_scoreboard


def test_fetch_scoreboard_uses_round_dates_during_quarterfinals(monkeypatch):
    calls = []

    def fake_fetch(day):
        calls.append(day)
        return [{"id": day.isoformat(), "date": day.isoformat()}]

    monkeypatch.setattr(world_cup, "_fetch_games_for_date", fake_fetch)

    result = world_cup.fetch_scoreboard(
        now=dt.datetime(2026, 7, 8, 12, 0, tzinfo=world_cup_scoreboard.CENTRAL_TIME)
    )

    assert calls == list(
        world_cup_scoreboard._round_dates(world_cup_scoreboard.ROUND_QUARTERFINALS)
    )
    assert len(result) == 3
    assert all(
        game[world_cup_scoreboard.ROUND_LABEL_KEY] == world_cup_scoreboard.ROUND_QUARTERFINALS
        for game in result
    )


def test_explicit_day_keeps_single_day_scoreboard(monkeypatch):
    calls = []

    def fake_fetch(day):
        calls.append(day)
        return [{"id": "single"}]

    monkeypatch.setattr(world_cup, "_fetch_games_for_date", fake_fetch)

    result = world_cup.fetch_scoreboard(day=dt.date(2026, 7, 8))

    assert calls == [dt.date(2026, 7, 8)]
    assert result == [{"id": "single"}]
