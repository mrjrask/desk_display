import datetime

from config import CENTRAL_TIME
from screens.registry import (
    ScreenContext,
    _logo_scroll_speed_for_layout,
    _is_1080p_or_higher,
    build_screen_registry,
)


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


def test_logo_scroll_threshold_detects_1080p_and_higher():
    assert _is_1080p_or_higher(1920, 1080) is True
    assert _is_1080p_or_higher(1080, 1920) is True
    assert _is_1080p_or_higher(2560, 1440) is True
    assert _is_1080p_or_higher(800, 480) is False


def test_logo_scroll_speed_is_doubled_for_1080p_layout():
    assert _logo_scroll_speed_for_layout(1920, 1080) == 4.4
    assert _logo_scroll_speed_for_layout(1080, 1920) == 4.4
    assert _logo_scroll_speed_for_layout(800, 480) == 2.2


def test_date_time_screens_render_in_transition_mode(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(weather, now)

    calls = []

    def _fake_draw_date(_display, transition=False):
        calls.append(("date", transition))

    def _fake_draw_time(_display, transition=False):
        calls.append(("time", transition))

    monkeypatch.setattr("screens.registry.draw_date", _fake_draw_date)
    monkeypatch.setattr("screens.registry.draw_time", _fake_draw_time)

    registry, _ = build_screen_registry(context)
    registry["date"].render()
    registry["time"].render()

    assert calls == [("date", True), ("time", True)]


def test_quad_screen_is_registered(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    monkeypatch.setattr(
        "screens.registry._next_quad_page_tiles",
        lambda: (True, ["date", "weather1", "weather hourly", "inside"]),
    )

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert "quad" in registry
    assert registry["quad"].available is True


def test_quad_screen_can_be_disabled(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    monkeypatch.setattr(
        "screens.registry._next_quad_page_tiles",
        lambda: (False, ["date", "weather1", "weather hourly", "inside"]),
    )

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert registry["quad"].available is False


def test_quad_screen_uses_layout_tile_selection(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(weather, now)

    monkeypatch.setattr(
        "screens.registry._next_quad_page_tiles",
        lambda: (True, ["time", "time", "inside", "weather1"]),
    )

    captured = {}

    def _fake_draw_quad_screen(_display, tiles, transition=False):
        captured["labels"] = [tile.label for tile in tiles]
        return None

    monkeypatch.setattr("screens.registry.draw_quad_screen", _fake_draw_quad_screen)

    registry, _ = build_screen_registry(context)
    registry["quad"].render()

    assert captured["labels"] == ["time", "time", "inside", "weather1"]


def test_next_quad_page_tiles_rotates_pages(monkeypatch):
    monkeypatch.setattr(
        "screens.registry._quad_layout_from_layouts",
        lambda: (True, [["date", "date", "date", "date"], ["time", "time", "time", "time"]]),
    )
    monkeypatch.setattr("screens.registry._quad_page_index", 0)

    from screens import registry as registry_module

    enabled_first, first = registry_module._next_quad_page_tiles()
    enabled_second, second = registry_module._next_quad_page_tiles()

    assert enabled_first is True
    assert enabled_second is True
    assert first == ["date", "date", "date", "date"]
    assert second == ["time", "time", "time", "time"]
