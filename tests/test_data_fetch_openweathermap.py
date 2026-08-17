import datetime

from config import CENTRAL_TIME
from data_fetch import (
    _normalise_openweathermap_response,
    _owm_moon_phase_label,
    _sun_times_for,
    _sun_windows,
)


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

    sunrise, sunset = _sun_times_for(_timestamp(2, 5), _sun_windows(daily))

    assert sunrise == _timestamp(2, 7)
    assert sunset == _timestamp(2, 21)


def test_openweathermap_daily_maps_moon_rise_set_and_phase():
    data = {
        "current": {},
        "daily": [
            {
                "dt": _timestamp(1, 12),
                "sunrise": _timestamp(1, 6),
                "sunset": _timestamp(1, 20),
                "moonrise": _timestamp(1, 21),
                "moonset": _timestamp(1, 9),
                "moon_phase": 0.5,
            },
        ],
        "hourly": [],
    }

    normalized = _normalise_openweathermap_response(data)

    assert normalized is not None
    day0 = normalized["daily"][0]
    assert day0["moonrise"] == _timestamp(1, 21)
    assert day0["moonset"] == _timestamp(1, 9)
    assert day0["moonPhase"] == "Full"


def test_owm_moon_phase_label_covers_full_cycle():
    assert _owm_moon_phase_label(0.0) == "New"
    assert _owm_moon_phase_label(1.0) == "New"
    assert _owm_moon_phase_label(0.1) == "WaxingCrescent"
    assert _owm_moon_phase_label(0.25) == "FirstQuarter"
    assert _owm_moon_phase_label(0.4) == "WaxingGibbous"
    assert _owm_moon_phase_label(0.5) == "Full"
    assert _owm_moon_phase_label(0.6) == "WaningGibbous"
    assert _owm_moon_phase_label(0.75) == "LastQuarter"
    assert _owm_moon_phase_label(0.9) == "WaningCrescent"
    assert _owm_moon_phase_label(None) is None
    assert _owm_moon_phase_label("bad") is None
