import datetime

from config import CENTRAL_TIME
from screens.draw_weather import _gather_hourly_forecast


def _build_hourly_entry(dt: datetime.datetime, *, main: str = "Clouds", icon: str = "Cloudy") -> dict:
    return {
        "dt": int(dt.timestamp()),
        "temp": 70,
        "wind_speed": 8,
        "wind_deg": 90,
        "uvi": 3,
        "weather": [
            {
                "main": main,
                "icon": icon,
            }
        ],
    }


def test_gather_hourly_forecast_skips_past_entries():
    now = CENTRAL_TIME.localize(datetime.datetime(2024, 1, 1, 12, 0))

    weather = {
        "hourly": [
            _build_hourly_entry(now - datetime.timedelta(hours=3)),
            _build_hourly_entry(now - datetime.timedelta(hours=1)),
            _build_hourly_entry(now),
            _build_hourly_entry(now + datetime.timedelta(hours=1)),
            _build_hourly_entry(now + datetime.timedelta(hours=2)),
        ]
    }

    forecast = _gather_hourly_forecast(weather, 3, now=now)

    assert [entry["time"] for entry in forecast] == ["12pm", "1pm", "2pm"]


def test_gather_hourly_forecast_orders_future_entries():
    now = CENTRAL_TIME.localize(datetime.datetime(2024, 1, 1, 9, 0))

    weather = {
        "hourly": [
            _build_hourly_entry(now + datetime.timedelta(hours=3)),
            _build_hourly_entry(now + datetime.timedelta(hours=1)),
            _build_hourly_entry(now + datetime.timedelta(hours=2)),
        ]
    }

    forecast = _gather_hourly_forecast(weather, 3, now=now)

    assert [entry["time"] for entry in forecast] == ["10am", "11am", "12pm"]
