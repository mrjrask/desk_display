import datetime as dt

from services.sports import ncaam


def test_tournament_pre_update_empty_combo_still_scans_forward(monkeypatch):
    yesterday = dt.date(2026, 3, 30)
    today = dt.date(2026, 3, 31)
    tomorrow = dt.date(2026, 4, 1)
    now = dt.datetime(2026, 3, 31, 8, 0)
    calls = []

    monkeypatch.setattr(ncaam, "scoreboard_date", lambda current_now: yesterday)

    def fetch(day, mode=None):
        calls.append((day, mode))
        if day == tomorrow:
            return [{"id": "tomorrow", "status": {"abstractGameState": "Preview"}}]
        return []

    monkeypatch.setattr(ncaam, "_fetch_games_for_date", fetch)

    assert ncaam.fetch_scoreboard(now=now, mode="tournament") == [
        {"id": "tomorrow", "status": {"abstractGameState": "Preview"}}
    ]
    assert calls[:3] == [(yesterday, "tournament"), (today, "tournament"), (today, "tournament")]
    assert (tomorrow, "tournament") in calls
