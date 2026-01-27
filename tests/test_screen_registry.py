import datetime

from config import CENTRAL_TIME
from screens.registry import ScreenContext, build_screen_registry


class _DummyDisplay:
    pass


class _DummyLogos:
    def get(self, name: str):
        return None


def _make_context(weather: dict, now: datetime.datetime) -> ScreenContext:
    return ScreenContext(
        display=_DummyDisplay(),
        cache={"weather": weather},
        logos=_DummyLogos(),
        image_dir="",
        travel_requested=False,
        travel_active=False,
        travel_window=None,
        previous_travel_state=None,
        now=now,
    )


def _ts(dt: datetime.datetime) -> int:
    return int(dt.timestamp())


def test_weather_radar_available_with_precipitation():
    now = CENTRAL_TIME.localize(datetime.datetime(2024, 1, 1, 12, 0))
    weather = {
        "hourly": [
            {
                "dt": _ts(now + datetime.timedelta(hours=4)),
                "pop": 80,
            }
        ]
    }

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert registry["weather radar"].available is True


def test_weather_radar_unavailable_without_precipitation_window():
    now = CENTRAL_TIME.localize(datetime.datetime(2024, 1, 1, 12, 0))
    weather = {
        "hourly": [
            {
                "dt": _ts(now + datetime.timedelta(hours=1)),
                "pop": 0,
            },
            {
                "dt": _ts(now + datetime.timedelta(hours=9)),
                "pop": 90,
            },
        ]
    }

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert registry["weather radar"].available is False


def test_weather_radar_detects_precipitation_amount_without_pop():
    now = CENTRAL_TIME.localize(datetime.datetime(2024, 1, 1, 12, 0))
    weather = {
        "hourly": [
            {
                "dt": _ts(now + datetime.timedelta(hours=2)),
                "rain": {"1h": 0.2},
            }
        ]
    }

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert registry["weather radar"].available is True
