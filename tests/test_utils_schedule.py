import datetime

from utils import next_game_from_schedule


def test_next_game_prefers_earliest_future_week():
    schedule = [
        {
            "week": "Week 16",
            "date": "Sat, Dec 20",
            "opponent": "Green Bay Packers",
            "home_away": "Home",
            "time": "7:20PM",
        },
        {
            "week": "Week 17",
            "date": "Sun, Dec 28",
            "opponent": "San Francisco 49ers",
            "home_away": "Away",
            "time": "7:20PM",
        },
    ]

    today = datetime.date(2025, 12, 1)
    game = next_game_from_schedule(schedule, today)

    assert game["week"] == "Week 16"


def test_next_game_falls_back_to_week_when_date_missing():
    schedule = [
        {
            "week": "Week 16",
            "date": "Sat, Dec 20",  # Will fail parsing because of locale-free weekday
            "opponent": "Green Bay Packers",
            "home_away": "Home",
            "time": "7:20PM",
        },
        {
            "week": "Week 17",
            "date": "Sun, Dec 28",
            "opponent": "San Francisco 49ers",
            "home_away": "Away",
            "time": "7:20PM",
        },
    ]

    today = datetime.date(2025, 12, 1)

    class LocaleResistantDate(datetime.datetime):
        @classmethod
        def strptime(cls, date_string, format):  # pragma: no cover - defensive
            if "Sat" in date_string:
                raise ValueError("Cannot parse locale-specific weekday")
            return super().strptime(date_string, format)

    original = datetime.datetime
    datetime.datetime = LocaleResistantDate
    try:
        game = next_game_from_schedule(schedule, today)
    finally:
        datetime.datetime = original

    assert game["week"] == "Week 17"


def test_next_game_uses_game_number_when_no_future_dates():
    schedule = [
        {
            "game_no": "19",
            "week": "Week 19",
            "date": "Sun, Dec 1",
            "opponent": "Detroit Lions",
            "home_away": "Home",
            "time": "12:00PM",
        },
        {
            "game_no": "20",
            "week": "Week 20",
            "date": "Sun, Dec 8",
            "opponent": "Minnesota Vikings",
            "home_away": "Away",
            "time": "12:00PM",
        },
    ]

    today = datetime.date(2025, 12, 31)

    game = next_game_from_schedule(schedule, today)

    assert game["game_no"] == "19"
