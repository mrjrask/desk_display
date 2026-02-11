import datetime

from config import CENTRAL_TIME
from screens.registry import ScreenContext, build_screen_registry


class _DummyDisplay:
    pass


class _DummyLogos:
    def get(self, name: str):
        return None


def _make_context(
    weather: dict,
    now: datetime.datetime,
    cache_updates: dict | None = None,
) -> ScreenContext:
    cache = {"weather": weather}
    if cache_updates:
        cache.update(cache_updates)
    return ScreenContext(
        display=_DummyDisplay(),
        cache=cache,
        logos=_DummyLogos(),
        image_dir="",
        now=now,
        now_utc=now.astimezone(datetime.timezone.utc),
        offline=False,
        weather_fetched_at=None,
        skip_scoreboards=False,
    )


def _ts(dt: datetime.datetime) -> int:
    return int(dt.timestamp())


def test_weather_radar_available_with_precipitation():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
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
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
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
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
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


def test_legacy_travel_alias_is_registered():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert "travel" in registry
    assert registry["travel"].available is True


def test_legacy_travel_v2_alias_is_registered():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert "travel v2" in registry
    assert registry["travel v2"].available is True


def test_legacy_travel_map_alias_is_registered():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert "travel map" in registry
    assert registry["travel map"].available is True


def test_legacy_travel_map_v2_alias_is_registered():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert "travel map v2" in registry
    assert registry["travel map v2"].available is True


def test_nhl_scoreboard_hidden_during_2026_break_window():
    now = datetime.datetime(2026, 2, 10, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(
        _make_context(weather, now, cache_updates={"hawks": {"next": {"id": 1}}})
    )

    assert registry["NHL Scoreboard"].available is False
    assert registry["NHL Scoreboard v2"].available is False
