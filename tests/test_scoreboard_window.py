import datetime as dt

from services.sports.scoreboard_window import compose_pre_update_scoreboard


def test_pre_update_scoreboard_appends_todays_schedule_after_prior_finals():
    yesterday = dt.date(2026, 3, 30)
    today = dt.date(2026, 3, 31)
    calls = []

    def fetch(day):
        calls.append(day)
        if day == yesterday:
            return [
                {"id": "final", "status": {"abstractGameState": "Final"}},
                {"id": "live", "status": {"abstractGameState": "Live"}},
                {"id": "postponed-prior", "status": {"detailedState": "Postponed"}},
            ]
        return [
            {"id": "scheduled", "status": {"abstractGameState": "Preview"}},
            {"id": "in-progress", "status": {"statusCode": "I"}},
            {"id": "postponed", "status": {"detailedState": "Postponed"}},
        ]

    games = compose_pre_update_scoreboard(
        now=dt.datetime(2026, 3, 31, 8, 0),
        scoreboard_day=yesterday,
        fetch_games_for_date=fetch,
    )

    assert [game["id"] for game in games] == ["final", "scheduled"]
    assert calls == [yesterday, today]
