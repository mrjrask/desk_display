import datetime

from config import CENTRAL_TIME
from data_fetch import _normalise_openweathermap_response, _sun_times_for, _sun_windows


def _timestamp(day: int, hour: int) -> int:
    return int(datetime.datetime(2026, 7, day, hour, tzinfo=CENTRAL_TIME).timestamp())


def test_openweathermap_morning_uses_same_calendar_days_sun_window():
    data = {
        "current": {},
        "daily": [
            {
                "dt": _timestamp(1, 12),
                "sunrise": _timestamp(1, 6),
                "sunset": _timestamp(1, 20),
            },
            {
                "dt": _timestamp(2, 12),
                "sunrise": _timestamp(2, 6),
                "sunset": _timestamp(2, 20),
            },
        ],
        "hourly": [
            {
                "dt": _timestamp(2, 7),
                "weather": [{"id": 800, "main": "Clear", "description": "clear sky"}],
            }
        ],
    }

    normalized = _normalise_openweathermap_response(data)

    assert normalized is not None
    assert normalized["hourly"][0]["weather"][0]["icon"] == "Clear"


def test_sun_window_selection_ignores_provider_daily_anchor():
    daily = [
        {
            "dt": _timestamp(1, 12),
            "sunrise": _timestamp(1, 6),
            "sunset": _timestamp(1, 20),
        },
        {
            "dt": _timestamp(2, 12),
            "sunrise": _timestamp(2, 7),
            "sunset": _timestamp(2, 21),
        },
    ]

    sunrise, sunset = _sun_times_for(_timestamp(2, 5), _sun_windows(daily, CENTRAL_TIME))

    assert sunrise == _timestamp(2, 7)
    assert sunset == _timestamp(2, 21)


def test_openweathermap_sun_windows_use_forecast_timezone():
    tokyo = datetime.timezone(datetime.timedelta(hours=9))

    def tokyo_timestamp(day: int, hour: int, minute: int = 0) -> int:
        return int(datetime.datetime(2026, 7, day, hour, minute, tzinfo=tokyo).timestamp())

    data = {
        "timezone_offset": 9 * 60 * 60,
        "current": {},
        "daily": [
            {
                "dt": tokyo_timestamp(1, 12),
                "sunrise": tokyo_timestamp(1, 4, 30),
                "sunset": tokyo_timestamp(1, 19),
            },
            {
                "dt": tokyo_timestamp(2, 12),
                "sunrise": tokyo_timestamp(2, 4, 30),
                "sunset": tokyo_timestamp(2, 19),
            },
        ],
        "hourly": [
            {
                "dt": tokyo_timestamp(1, 14),
                "weather": [{"id": 800, "main": "Clear", "description": "clear sky"}],
            }
        ],
    }

    normalized = _normalise_openweathermap_response(data)

    assert normalized is not None
    assert normalized["hourly"][0]["weather"][0]["icon"] == "Clear"
