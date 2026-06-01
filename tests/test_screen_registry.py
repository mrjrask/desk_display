import datetime

import pytest
import screens.registry as registry_module
from PIL import Image

from config import CENTRAL_TIME
from screens.registry import (
    ScreenContext,
    _logo_scroll_speed_for_layout,
    _is_1080p_or_higher,
    build_screen_registry,
)
from utils import ScreenImage


def _reset_quad_scroll_state():
    registry_module._quad_tile_scroll_cursor.clear()

@pytest.fixture(autouse=True)
def _clear_quad_scroll_cursor():
    _reset_quad_scroll_state()
    yield
    _reset_quad_scroll_state()


class _DummyDisplay:
    pass


class _DummyLogos:
    def get(self, name: str):
        return None


def _make_context(
    weather: dict,
    now: datetime.datetime,
    cache_updates: dict | None = None,
    *,
    offline: bool = False,
    weather_fetched_at: datetime.datetime | None = None,
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
        offline=offline,
        weather_fetched_at=weather_fetched_at,
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


def test_weather_current_screens_stay_available_with_cached_data_offline():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    stale_fetch = now - datetime.timedelta(hours=12)
    weather = {
        "current": {"temp": 65},
        "daily": [{"temp": {"max": 70, "min": 50}}],
        "hourly": [{"dt": _ts(now + datetime.timedelta(hours=1)), "pop": 0}],
    }

    registry, _ = build_screen_registry(
        _make_context(weather, now, offline=True, weather_fetched_at=stale_fetch)
    )

    assert registry["weather1"].available is True
    assert registry["weather2"].available is True


def test_weather_hourly_screens_stay_available_with_cached_data_offline():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    stale_fetch = now - datetime.timedelta(days=1)
    weather = {
        "hourly": [{"dt": _ts(now + datetime.timedelta(hours=2)), "pop": 0}],
        "daily": [{"temp": {"max": 70, "min": 50}}],
    }

    registry, _ = build_screen_registry(
        _make_context(weather, now, offline=True, weather_fetched_at=stale_fetch)
    )

    assert registry["weather hourly"].available is True
    assert registry["weather daily"].available is True
    assert registry["astronomical"].available is True


def test_weather_quad_screen_available_with_cached_weather_offline():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {
        "current": {"temp": 65},
        "hourly": [{"dt": _ts(now + datetime.timedelta(hours=2)), "pop": 0}],
        "daily": [{"temp": {"max": 70, "min": 50}}],
    }

    registry, _ = build_screen_registry(_make_context(weather, now, offline=True))

    assert registry["weather quad"].available is True


def test_travel_alias_is_not_registered():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert "travel" not in registry


def test_travel_v2_alias_is_not_registered():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert "travel v2" not in registry


def test_travel_map_alias_is_not_registered():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert "travel map" not in registry


def test_travel_map_v2_alias_is_not_registered():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert "travel map v2" not in registry


def test_mlb_current_series_screens_are_registered_and_available():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    cache_updates = {
        "cubs": {"current_series": [{"gamePk": 1}]},
        "sox": {"current_series": [{"gamePk": 2}]},
    }

    registry, _ = build_screen_registry(_make_context(weather, now, cache_updates=cache_updates))

    assert "cubs current series" in registry
    assert "sox current series" in registry
    assert registry["cubs current series"].available is True
    assert registry["sox current series"].available is True


def test_mlb_next_home_series_uses_following_home_series_title(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    titles = []

    def _fake_draw_series_screen(display, games, title, **kwargs):
        titles.append(title)

    monkeypatch.setattr("screens.registry.draw_series_screen", _fake_draw_series_screen)
    monkeypatch.setattr(registry_module, "is_display_profile", lambda *args, **kwargs: False)

    cache_updates = {
        "cubs": {"next_home_series": [{"gamePk": 5001}]},
        "sox": {"next_home_series": [{"gamePk": 6001}]},
    }
    registry, _ = build_screen_registry(_make_context(weather, now, cache_updates=cache_updates))

    registry["cubs next home series"].render()
    registry["sox next home series"].render()

    assert "Following Home Series" in titles
    assert "Cubs Following Home Series" not in titles
    assert "Sox Following Home Series" not in titles


def test_mlb_series_titles_keep_team_name_on_hyperpixel4(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    titles = []

    def _fake_draw_series_screen(display, games, title, **kwargs):
        titles.append(title)

    monkeypatch.setattr("screens.registry.draw_series_screen", _fake_draw_series_screen)
    monkeypatch.setattr(
        registry_module,
        "is_display_profile",
        lambda profile_id, *_args, **_kwargs: profile_id == "hyperpixel4",
    )

    cache_updates = {
        "cubs": {"next_series": [{"gamePk": 5001}]},
        "sox": {"next_series": [{"gamePk": 6001}]},
    }
    registry, _ = build_screen_registry(_make_context(weather, now, cache_updates=cache_updates))

    registry["cubs next series"].render()
    registry["sox next series"].render()

    assert "Cubs Next Series" in titles
    assert "Sox Next Series" in titles


def test_inside_screen_hidden_when_sensor_unavailable(monkeypatch):
    import screens.registry as registry_module

    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    monkeypatch.setattr(registry_module, "is_inside_sensor_available", lambda: False)

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert registry["inside"].available is False


@pytest.mark.parametrize(
    ("date_parts", "expected_available"),
    [
        ((2031, 2, 3), True),
        ((2031, 2, 10), False),
        ((2031, 2, 17), False),
        ((2031, 2, 25), True),
    ],
)
def test_nhl_scoreboard_availability_respects_configured_break_windows(
    monkeypatch, date_parts, expected_available
):
    monkeypatch.setenv(
        "NHL_BREAK_WINDOWS_JSON",
        '{"2030-2031":{"start":"2031-02-10","end":"2031-02-17"}}',
    )
    now = datetime.datetime(*date_parts, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(
        _make_context(weather, now, cache_updates={"hawks": {"next": {"id": 1}}})
    )

    assert registry["NHL Scoreboard"].available is expected_available
    assert registry["NHL Scoreboard v2"].available is expected_available


def test_invalid_nhl_break_config_falls_back_to_showing_nhl_scoreboards(monkeypatch):
    monkeypatch.setenv("NHL_BREAK_WINDOWS_JSON", "{this-is-not-json")
    now = datetime.datetime(2031, 2, 12, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(
        _make_context(weather, now, cache_updates={"hawks": {"next": {"id": 1}}})
    )

    assert registry["NHL Scoreboard"].available is True
    assert registry["NHL Scoreboard v2"].available is True


def test_nhl_registry_uses_prepared_service_payload(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    raw_nhl_payload = {"games": [{"gamePk": 1}]}
    prepared_games = [{"gamePk": 99}]
    captured = {}

    monkeypatch.setattr(
        registry_module,
        "prepare_nhl_scoreboard_data",
        lambda payload: prepared_games if payload == raw_nhl_payload else [],
    )

    def _capture_nhl(_display, games, transition=False):
        captured["games"] = games
        return Image.new("RGB", (10, 10))

    monkeypatch.setattr(registry_module, "render_nhl_scoreboard", _capture_nhl)

    registry, _ = build_screen_registry(
        _make_context(
            weather,
            now,
            cache_updates={"scoreboards": {"nhl": raw_nhl_payload}, "hawks": {"next": {"id": 1}}},
        )
    )
    registry["NHL Scoreboard"].render()

    assert captured["games"] == prepared_games


def test_active_screens_config_path_uses_local_override_when_present(tmp_path, monkeypatch):
    default_path = tmp_path / "screens_config.json"
    local_path = tmp_path / "screens_config.local.json"
    default_path.write_text('{"screens": {"date": 1}}', encoding="utf-8")
    local_path.write_text('{"screens": {"date": 2}}', encoding="utf-8")
    monkeypatch.setenv("SCREENS_CONFIG_PATH", str(default_path))
    monkeypatch.setenv("SCREENS_CONFIG_LOCAL_PATH", str(local_path))

    assert registry_module._active_screens_config_path() == str(local_path)


def test_active_screens_config_path_falls_back_to_default_when_local_missing(tmp_path, monkeypatch):
    default_path = tmp_path / "screens_config.json"
    local_path = tmp_path / "screens_config.local.json"
    default_path.write_text('{"screens": {"date": 1}}', encoding="utf-8")
    monkeypatch.setenv("SCREENS_CONFIG_PATH", str(default_path))
    monkeypatch.setenv("SCREENS_CONFIG_LOCAL_PATH", str(local_path))

    assert registry_module._active_screens_config_path() == str(default_path)


def test_adafruit_minipitft_routes_scoreboard_v2_ids_to_v1_renderers(monkeypatch):
    now = datetime.datetime(2024, 7, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    calls: list[str] = []

    def _mark(name: str):
        def _renderer(*_args, **_kwargs):
            calls.append(name)
            return None

        return _renderer

    monkeypatch.setattr(registry_module, "WIDTH", 240)
    monkeypatch.setattr(registry_module, "HEIGHT", 135)
    monkeypatch.setattr(registry_module, "render_nfl_scoreboard", _mark("nfl_v1"))
    monkeypatch.setattr(registry_module, "render_nfl_scoreboard_v2", _mark("nfl_v2"))
    monkeypatch.setattr(registry_module, "render_nhl_scoreboard", _mark("nhl_v1"))
    monkeypatch.setattr(registry_module, "render_nhl_scoreboard_v2", _mark("nhl_v2"))
    monkeypatch.setattr(registry_module, "render_mlb_scoreboard", _mark("mlb_v1"))
    monkeypatch.setattr(registry_module, "render_mlb_scoreboard_v2", _mark("mlb_v2"))
    monkeypatch.setattr(registry_module, "render_nba_scoreboard", _mark("nba_v1"))
    monkeypatch.setattr(registry_module, "render_nba_scoreboard_v2", _mark("nba_v2"))

    registry, _ = build_screen_registry(
        _make_context(
            weather,
            now,
            cache_updates={
                "hawks": {"next": {"id": 1}},
                "scoreboards": {"nfl": [{}], "nhl": [{}], "mlb": [{}], "nba": [{}]},
            },
        )
    )

    registry["NFL Scoreboard v2"].render()
    registry["NHL Scoreboard v2"].render()
    registry["MLB Scoreboard v2"].render()
    registry["NBA Scoreboard v2"].render()

    assert calls == ["nfl_v1", "nhl_v1", "mlb_v1", "nba_v1"]


def test_waveshare_routes_scoreboard_v2_ids_to_v1_renderers_except_mlb(monkeypatch):
    now = datetime.datetime(2024, 7, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    calls: list[str] = []

    def _mark(name: str):
        def _renderer(*_args, **_kwargs):
            calls.append(name)
            return None

        return _renderer

    monkeypatch.setenv("WAVESHARE_OLED_LCD_HAT_A_INSTALLED", "installed")
    monkeypatch.setattr(registry_module, "render_nfl_scoreboard", _mark("nfl_v1"))
    monkeypatch.setattr(registry_module, "render_nfl_scoreboard_v2", _mark("nfl_v2"))
    monkeypatch.setattr(registry_module, "render_nhl_scoreboard", _mark("nhl_v1"))
    monkeypatch.setattr(registry_module, "render_nhl_scoreboard_v2", _mark("nhl_v2"))
    monkeypatch.setattr(registry_module, "render_mlb_scoreboard", _mark("mlb_v1"))
    monkeypatch.setattr(registry_module, "render_mlb_scoreboard_v2", _mark("mlb_v2"))
    monkeypatch.setattr(registry_module, "render_nba_scoreboard", _mark("nba_v1"))
    monkeypatch.setattr(registry_module, "render_nba_scoreboard_v2", _mark("nba_v2"))

    registry, _ = build_screen_registry(
        _make_context(
            weather,
            now,
            cache_updates={
                "hawks": {"next": {"id": 1}},
                "scoreboards": {"nfl": [{}], "nhl": [{}], "mlb": [{}], "nba": [{}]},
            },
        )
    )

    registry["NFL Scoreboard v2"].render()
    registry["NHL Scoreboard v2"].render()
    registry["MLB Scoreboard v2"].render()
    registry["NBA Scoreboard v2"].render()

    assert calls == ["nfl_v1", "nhl_v1", "mlb_v2", "nba_v1"]


def test_cubs_live_is_available_when_status_is_warmup():
    now = datetime.datetime(2024, 7, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    registry, _ = build_screen_registry(
        _make_context(
            weather,
            now,
            cache_updates={
                "cubs": {
                    "live": {
                        "officialDate": "2024-07-01",
                        "status": {"detailedState": "Warmup"},
                    }
                }
            },
        )
    )

    assert registry["cubs live"].available is True


def test_sox_live_is_available_when_status_is_pre_game_warmup():
    now = datetime.datetime(2024, 7, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    registry, _ = build_screen_registry(
        _make_context(
            weather,
            now,
            cache_updates={
                "sox": {
                    "live": {
                        "officialDate": "2024-07-01",
                        "status": {"detailedState": "Pre-Game Warmup"},
                    }
                }
            },
        )
    )

    assert registry["sox live"].available is True


def test_mlb_next_alt_games_rotate_on_primary_next_screen(monkeypatch):
    now = datetime.datetime(2024, 3, 10, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    rendered_game_ids = []

    def _fake_draw_sports_screen(_display, game, *_args, **_kwargs):
        rendered_game_ids.append(game.get("gamePk"))

    monkeypatch.setattr("screens.registry.draw_sports_screen", _fake_draw_sports_screen)

    registry, _ = build_screen_registry(
        _make_context(
            weather,
            now,
            cache_updates={
                "cubs": {
                    "next": {"gamePk": 1, "officialDate": "2024-03-10"},
                    "next_alt": {"gamePk": 2, "officialDate": "2024-03-10"},
                },
                "sox": {
                    "next": {"gamePk": 3, "officialDate": "2024-03-10"},
                    "next_alt": {"gamePk": 4, "officialDate": "2024-03-10"},
                },
            },
        )
    )

    assert registry["cubs next"].available is True
    assert registry["sox next"].available is True
    registry["cubs next"].render()
    registry["cubs next"].render()
    registry["sox next"].render()
    registry["sox next"].render()
    assert rendered_game_ids == [1, 2, 3, 4]


def test_logo_scroll_threshold_detects_1080p_and_higher():
    assert _is_1080p_or_higher(1920, 1080) is True
    assert _is_1080p_or_higher(1080, 1920) is True
    assert _is_1080p_or_higher(2560, 1440) is True
    assert _is_1080p_or_higher(800, 480) is False


def test_logo_scroll_speed_is_doubled_for_1080p_layout():
    assert _logo_scroll_speed_for_layout(1920, 1080) == 4.4
    assert _logo_scroll_speed_for_layout(1080, 1920) == 4.4
    assert _logo_scroll_speed_for_layout(800, 480) == 2.2


def test_date_nixie_screens_render_with_live_color_cycle_mode(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(weather, now)

    calls = []

    def _fake_draw_date(_display, transition=False):
        calls.append(("date", transition))

    def _fake_draw_nixie(_display, transition=False):
        calls.append(("nixie", transition))

    monkeypatch.setattr("screens.registry.draw_date", _fake_draw_date)
    monkeypatch.setattr("screens.registry.draw_nixie", _fake_draw_nixie)

    registry, _ = build_screen_registry(context)
    registry["date"].render()
    registry["nixie"].render()

    assert calls == [("date", False), ("nixie", False)]


def test_quad_screen_is_registered(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    monkeypatch.setattr(
        "screens.registry._next_quad_page_tiles",
        lambda: (True, 1.0, ["date", "weather1", "weather hourly", "inside"]),
    )

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert "quad" in registry
    assert registry["quad"].available is True


def test_quad_screen_can_be_disabled(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    monkeypatch.setattr(
        "screens.registry._next_quad_page_tiles",
        lambda: (False, 1.0, ["date", "weather1", "weather hourly", "inside"]),
    )

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert registry["quad"].available is False


def test_quad_screen_uses_layout_tile_selection(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(weather, now)

    monkeypatch.setattr(
        "screens.registry._next_quad_page_tiles",
        lambda: (True, 1.0, ["nixie", "nixie", "inside", "weather1"]),
    )

    captured = {}

    def _fake_draw_quad_screen(_display, tiles, transition=False, scroll_speed=1.0):
        captured["labels"] = [tile.label for tile in tiles]
        return None

    monkeypatch.setattr("screens.registry.draw_quad_screen", _fake_draw_quad_screen)

    registry, _ = build_screen_registry(context)
    registry["quad"].render()

    assert captured["labels"] == ["nixie", "nixie", "inside", "weather1"]


def test_weather_quad_screen_uses_weather_tile_selection(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {
        "current": {"temp": 65},
        "hourly": [{"dt": _ts(now + datetime.timedelta(hours=2)), "pop": 0}],
        "daily": [{"temp": {"max": 70, "min": 50}}],
    }
    context = _make_context(weather, now)

    captured = {}

    def _fake_draw_quad_screen(_display, tiles, transition=False, scroll_speed=1.0):
        captured["labels"] = [tile.label for tile in tiles]
        return None

    monkeypatch.setattr("screens.registry.draw_quad_screen", _fake_draw_quad_screen)

    registry, _ = build_screen_registry(context)
    registry["weather quad"].render()

    assert captured["labels"] == ["weather1", "weather2", "weather hourly", "weather daily"]


def test_cubs_schedule_quad_uses_expected_tile_selection(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(
        weather,
        now,
        cache_updates={
            "cubs": {
                "next": {"gamePk": 1, "officialDate": "2024-01-01"},
                "current_series": [{"gamePk": 2, "officialDate": "2024-01-01"}],
                "next_series": [{"gamePk": 3}],
                "next_home_series": [{"gamePk": 4}],
            }
        },
    )
    captured = {}

    def _fake_draw_quad_screen(_display, tiles, transition=False, scroll_speed=1.0):
        captured["labels"] = [tile.label for tile in tiles]
        return None

    monkeypatch.setattr("screens.registry.draw_quad_screen", _fake_draw_quad_screen)

    registry, _ = build_screen_registry(context)
    assert registry["cubs schedule quad"].available is True
    registry["cubs schedule quad"].render()

    assert captured["labels"] == [
        "cubs next",
        "cubs current series",
        "cubs next series",
        "cubs next home series",
    ]


def test_sox_schedule_quad_uses_expected_tile_selection(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(
        weather,
        now,
        cache_updates={
            "sox": {
                "next": {"gamePk": 10, "officialDate": "2024-01-01"},
                "current_series": [{"gamePk": 20, "officialDate": "2024-01-01"}],
                "next_series": [{"gamePk": 30}],
                "next_home_series": [{"gamePk": 40}],
            }
        },
    )
    captured = {}

    def _fake_draw_quad_screen(_display, tiles, transition=False, scroll_speed=1.0):
        captured["labels"] = [tile.label for tile in tiles]
        return None

    monkeypatch.setattr("screens.registry.draw_quad_screen", _fake_draw_quad_screen)

    registry, _ = build_screen_registry(context)
    assert registry["sox schedule quad"].available is True
    registry["sox schedule quad"].render()

    assert captured["labels"] == [
        "sox next",
        "sox current series",
        "sox next series",
        "sox next home series",
    ]


def test_hawks_schedule_quad_uses_expected_tile_selection(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(
        weather,
        now,
        cache_updates={
            "hawks": {
                "stand": [{"team": {"id": 16}}],
                "last": {"id": 10},
                "next": {"id": 20},
                "next_home": {"id": 30},
            }
        },
    )
    captured = {}

    def _fake_draw_quad_screen(_display, tiles, transition=False, scroll_speed=1.0):
        captured["labels"] = [tile.label for tile in tiles]
        return None

    monkeypatch.setattr("screens.registry.draw_quad_screen", _fake_draw_quad_screen)

    registry, _ = build_screen_registry(context)
    assert registry["hawks schedule quad"].available is True
    registry["hawks schedule quad"].render()

    assert captured["labels"] == [
        "hawks stand1",
        "hawks last",
        "hawks next",
        "hawks next home",
    ]


def test_hawks_schedule_quad_uses_blank_tile_when_next_home_missing(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(
        weather,
        now,
        cache_updates={
            "hawks": {
                "stand": [{"team": {"id": 16}}],
                "last": {"id": 10},
                "next": {"id": 20},
                "next_home": {"id": 20},
            }
        },
    )
    captured = {}

    def _fake_draw_quad_screen(_display, tiles, transition=False, scroll_speed=1.0):
        captured["labels"] = [tile.label for tile in tiles]
        return None

    monkeypatch.setattr("screens.registry.draw_quad_screen", _fake_draw_quad_screen)

    registry, _ = build_screen_registry(context)
    assert "hawks next home" not in registry
    assert registry["hawks schedule quad"].available is True
    registry["hawks schedule quad"].render()

    assert captured["labels"] == [
        "hawks stand1",
        "hawks last",
        "hawks next",
        "blank",
    ]


def test_bulls_schedule_quad_uses_expected_tile_selection(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(
        weather,
        now,
        cache_updates={
            "bulls": {
                "stand": [{"team": {"id": 5}}],
                "last": {"id": 11},
                "next": {"id": 21},
                "next_home": {"id": 31},
            }
        },
    )
    captured = {}

    def _fake_draw_quad_screen(_display, tiles, transition=False, scroll_speed=1.0):
        captured["labels"] = [tile.label for tile in tiles]
        return None

    monkeypatch.setattr("screens.registry.draw_quad_screen", _fake_draw_quad_screen)

    registry, _ = build_screen_registry(context)
    assert registry["bulls schedule quad"].available is True
    registry["bulls schedule quad"].render()

    assert captured["labels"] == [
        "bulls stand1",
        "bulls last",
        "bulls next",
        "bulls next home",
    ]


def test_quad_screen_advances_scrolling_tiles_between_renders(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(weather, now)

    monkeypatch.setattr(
        "screens.registry._next_quad_page_tiles",
        lambda: (True, 1.0, ["date", "nixie", "inside", "weather1"]),
    )

    def _animated_date(display, transition=False):
        frames = [(255, 0, 0), (0, 255, 0)]
        for color in frames:
            if hasattr(display, "skip_requested") and display.skip_requested():
                break
            display.image(Image.new("RGB", (8, 8), color))
        return None

    def _single_frame(_display, transition=False):
        return Image.new("RGB", (8, 8), (0, 0, 0))

    sampled_colors = []

    def _fake_draw_quad_screen(_display, tiles, transition=False, scroll_speed=1.0):
        rendered = tiles[0].render()
        if isinstance(rendered, list):
            rendered = rendered[0]
        sampled_colors.append(rendered.getpixel((0, 0)))
        return None

    monkeypatch.setattr("screens.registry.draw_date", _animated_date)
    monkeypatch.setattr("screens.registry.draw_nixie", _single_frame)
    monkeypatch.setattr("screens.registry.draw_inside", _single_frame)
    monkeypatch.setattr("screens.registry.draw_weather_screen_1", _single_frame)
    monkeypatch.setattr("screens.registry.draw_quad_screen", _fake_draw_quad_screen)

    registry, _ = build_screen_registry(context)
    registry["quad"].render()
    registry["quad"].render()
    registry["quad"].render()

    assert sampled_colors == [(255, 0, 0), (0, 255, 0), (255, 0, 0)]


def test_quad_screen_prefers_captured_frames_over_screenimage_return(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(weather, now)

    monkeypatch.setattr(
        "screens.registry._next_quad_page_tiles",
        lambda: (True, 1.0, ["date", "nixie", "inside", "weather1"]),
    )

    def _animated_date(display, transition=False):
        colors = [(255, 0, 0), (0, 255, 0)]
        last = None
        for color in colors:
            if hasattr(display, "skip_requested") and display.skip_requested():
                break
            last = Image.new("RGB", (8, 8), color)
            display.image(last)
        assert last is not None
        return ScreenImage(last, displayed=True)

    def _single_frame(_display, transition=False):
        return Image.new("RGB", (8, 8), (0, 0, 0))

    sampled_colors = []

    def _fake_draw_quad_screen(_display, tiles, transition=False, scroll_speed=1.0):
        rendered = tiles[0].render()
        if isinstance(rendered, list):
            rendered = rendered[0]
        sampled_colors.append(rendered.getpixel((0, 0)))
        return None

    monkeypatch.setattr("screens.registry.draw_date", _animated_date)
    monkeypatch.setattr("screens.registry.draw_nixie", _single_frame)
    monkeypatch.setattr("screens.registry.draw_inside", _single_frame)
    monkeypatch.setattr("screens.registry.draw_weather_screen_1", _single_frame)
    monkeypatch.setattr("screens.registry.draw_quad_screen", _fake_draw_quad_screen)

    registry, _ = build_screen_registry(context)
    registry["quad"].render()
    registry["quad"].render()
    registry["quad"].render()

    assert sampled_colors == [(255, 0, 0), (0, 255, 0), (255, 0, 0)]


def test_quad_screen_samples_across_longer_animations(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(weather, now)

    monkeypatch.setattr(
        "screens.registry._next_quad_page_tiles",
        lambda: (True, 1.0, ["date", "nixie", "inside", "weather1"]),
    )

    def _long_animated_date(display, transition=False):
        for idx in range(40):
            if hasattr(display, "skip_requested") and display.skip_requested():
                break
            display.image(Image.new("RGB", (8, 8), (idx, 0, 0)))
        return None

    def _single_frame(_display, transition=False):
        return Image.new("RGB", (8, 8), (0, 0, 0))

    sampled_red = []

    def _fake_draw_quad_screen(_display, tiles, transition=False, scroll_speed=1.0):
        rendered = tiles[0].render()
        if isinstance(rendered, list):
            rendered = rendered[0]
        sampled_red.append(rendered.getpixel((0, 0))[0])
        return None

    monkeypatch.setattr("screens.registry.draw_date", _long_animated_date)
    monkeypatch.setattr("screens.registry.draw_nixie", _single_frame)
    monkeypatch.setattr("screens.registry.draw_inside", _single_frame)
    monkeypatch.setattr("screens.registry.draw_weather_screen_1", _single_frame)
    monkeypatch.setattr("screens.registry.draw_quad_screen", _fake_draw_quad_screen)

    registry, _ = build_screen_registry(context)
    for _ in range(10):
        registry["quad"].render()

    assert sampled_red == [0, 4, 9, 13, 17, 22, 26, 30, 35, 39]




def test_quad_screen_preserves_scrolling_cursor_across_registry_rebuilds(monkeypatch):
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    monkeypatch.setattr(
        "screens.registry._next_quad_page_tiles",
        lambda: (True, 1.0, ["date", "nixie", "inside", "weather1"]),
    )

    def _animated_date(display, transition=False):
        frames = [(255, 0, 0), (0, 255, 0)]
        for color in frames:
            if hasattr(display, "skip_requested") and display.skip_requested():
                break
            display.image(Image.new("RGB", (8, 8), color))
        return None

    def _single_frame(_display, transition=False):
        return Image.new("RGB", (8, 8), (0, 0, 0))

    sampled_colors = []

    def _fake_draw_quad_screen(_display, tiles, transition=False, scroll_speed=1.0):
        rendered = tiles[0].render()
        if isinstance(rendered, list):
            rendered = rendered[0]
        sampled_colors.append(rendered.getpixel((0, 0)))
        return None

    monkeypatch.setattr("screens.registry.draw_date", _animated_date)
    monkeypatch.setattr("screens.registry.draw_nixie", _single_frame)
    monkeypatch.setattr("screens.registry.draw_inside", _single_frame)
    monkeypatch.setattr("screens.registry.draw_weather_screen_1", _single_frame)
    monkeypatch.setattr("screens.registry.draw_quad_screen", _fake_draw_quad_screen)
    monkeypatch.setattr("screens.registry._quad_tile_scroll_cursor", {})

    first_registry, _ = build_screen_registry(_make_context(weather, now))
    first_registry["quad"].render()

    second_registry, _ = build_screen_registry(_make_context(weather, now))
    second_registry["quad"].render()

    assert sampled_colors == [(255, 0, 0), (0, 255, 0)]
def test_next_quad_page_tiles_rotates_pages(monkeypatch):
    monkeypatch.setattr(
        "screens.registry._quad_layout_from_layouts",
        lambda: (True, 1.0, [["date", "date", "date", "date"], ["nixie", "nixie", "nixie", "nixie"]]),
    )
    monkeypatch.setattr("screens.registry._quad_page_index", 0)

    from screens import registry as registry_module

    enabled_first, speed_first, first = registry_module._next_quad_page_tiles()
    enabled_second, speed_second, second = registry_module._next_quad_page_tiles()

    assert enabled_first is True
    assert enabled_second is True
    assert speed_first == 1.0
    assert speed_second == 1.0
    assert first == ["date", "date", "date", "date"]
    assert second == ["nixie", "nixie", "nixie", "nixie"]






def test_mlb_scoreboard_renders_with_empty_list_when_cache_is_none(monkeypatch):
    now = datetime.datetime(2024, 7, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}
    context = _make_context(
        weather,
        now,
        cache_updates={"scoreboards": {"mlb": None}},
    )

    captured = {}

    def _capture_mlb(_display, games, transition=False):
        captured["mlb_games"] = games
        return None

    monkeypatch.setattr("screens.registry.render_mlb_scoreboard", _capture_mlb)

    registry, _ = build_screen_registry(context)
    registry["MLB Scoreboard"].render()

    assert captured["mlb_games"] == []


def test_wbc_scoreboards_are_removed_from_registry():
    now = datetime.datetime(2024, 7, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(_make_context(weather, now))

    assert "WBC Scoreboard" not in registry
    assert "WBC Scoreboard v2" not in registry


def test_mlb_no_game_screens_available_when_no_game_today():
    now = datetime.datetime(2024, 7, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(
        _make_context(
            weather,
            now,
            cache_updates={
                "cubs": {"next": {"gamePk": 1, "officialDate": "2024-07-02"}},
                "sox": {"last": {"gamePk": 2, "officialDate": "2024-06-30"}},
            },
        )
    )

    assert registry["cubs no game"].available is True
    assert registry["sox no game"].available is True
    assert registry["cubs next"].available is True
    assert registry["sox next"].available is True
    assert registry["cubs next"].metadata["replaces_with"] == "cubs no game"
    assert registry["sox next"].metadata["replaces_with"] == "sox no game"


def test_mlb_no_game_screens_hidden_when_game_is_today():
    now = datetime.datetime(2024, 7, 1, 12, 0, tzinfo=CENTRAL_TIME)
    weather = {"hourly": []}

    registry, _ = build_screen_registry(
        _make_context(
            weather,
            now,
            cache_updates={
                "cubs": {"next": {"gamePk": 1, "officialDate": "2024-07-01"}},
                "sox": {"next": {"gamePk": 2, "officialDate": "2024-07-01"}},
            },
        )
    )

    assert registry["cubs no game"].available is False
    assert registry["sox no game"].available is False
    assert registry["cubs next"].available is True
    assert registry["sox next"].available is True
    assert registry["cubs next"].metadata["replaces_with"] is None
    assert registry["sox next"].metadata["replaces_with"] is None
