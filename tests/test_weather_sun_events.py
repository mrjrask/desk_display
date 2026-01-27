import datetime

from config import CENTRAL_TIME
from screens.draw_weather import SUN_EVENT_GRACE, _next_sun_event


def _ts(dt: datetime.datetime) -> int:
    return int(dt.timestamp())


def _day(sunrise: datetime.datetime, sunset: datetime.datetime) -> dict:
    return {
        "sunrise": _ts(sunrise),
        "sunset": _ts(sunset),
    }


def test_next_sun_event_holds_sunset_during_grace_window():
    today = CENTRAL_TIME.localize(datetime.datetime(2024, 1, 1))
    sunrise_today = today + datetime.timedelta(hours=7)
    sunset_today = today + datetime.timedelta(hours=18)
    sunrise_next = today + datetime.timedelta(days=1, hours=7, minutes=1)
    sunset_next = today + datetime.timedelta(days=1, hours=18, minutes=1)

    now = sunset_today + datetime.timedelta(minutes=10)
    label, event_time = _next_sun_event(
        [
            _day(sunrise_today, sunset_today),
            _day(sunrise_next, sunset_next),
        ],
        now=now,
    )

    assert label == "Sunset"
    assert event_time == sunset_today
    assert now <= event_time + SUN_EVENT_GRACE


def test_next_sun_event_switches_to_sunrise_after_grace():
    today = CENTRAL_TIME.localize(datetime.datetime(2024, 1, 1))
    sunrise_today = today + datetime.timedelta(hours=7)
    sunset_today = today + datetime.timedelta(hours=18)
    sunrise_next = today + datetime.timedelta(days=1, hours=7, minutes=1)
    sunset_next = today + datetime.timedelta(days=1, hours=18, minutes=1)

    now = sunset_today + SUN_EVENT_GRACE + datetime.timedelta(minutes=1)
    label, event_time = _next_sun_event(
        [
            _day(sunrise_today, sunset_today),
            _day(sunrise_next, sunset_next),
        ],
        now=now,
    )

    assert label == "Sunrise"
    assert event_time == sunrise_next
