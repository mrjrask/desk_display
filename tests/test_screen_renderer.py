"""Tests for the hardware-free rendering boundary."""

from types import SimpleNamespace

from PIL import Image

import screens.mlb_league_standings as mlb_standings
from config import CENTRAL_TIME
from display_profiles import DISPLAY_PROFILE_DISPLAY_HAT_MINI, PROFILE_PRESETS
from rendering.screen_renderer import ScreenRenderer, ServerPreferenceSnapshot
from services.data_coordinator import DataCoordinator
from services.data_provider import DataProvider


def test_default_renderer_thaws_snapshot_for_legacy_screens(monkeypatch):
    coordinator = DataCoordinator()
    snapshot = coordinator.publish(
        "weather",
        {
            "hourly": [{"temperature": 72}],
            "alerts": {"active", "watch"},
        },
    )
    received = {}

    def fake_build_screen_registry(context):
        received["cache"] = context.cache
        received["now"] = context.now
        received["now_utc"] = context.now_utc
        context.cache["weather"]["hourly"][0]["temperature"] = 0
        definition = SimpleNamespace(
            available=True,
            metadata={},
            render=lambda: Image.new("RGB", (320, 240)),
        )
        return {"weather_hourly": definition}, None

    monkeypatch.setattr("screens.registry.build_screen_registry", fake_build_screen_registry)

    renderer = ScreenRenderer()
    renderer.render(
        "weather_hourly",
        PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI],
        ServerPreferenceSnapshot(revision=1),
        snapshot,
    )

    weather = received["cache"]["weather"]
    assert isinstance(received["cache"], dict)
    assert isinstance(weather, dict)
    assert isinstance(weather["hourly"], list)
    assert isinstance(weather["hourly"][0], dict)
    assert isinstance(weather["alerts"], set)
    assert snapshot["weather"]["hourly"][0]["temperature"] == 72
    assert received["now"].tzinfo is CENTRAL_TIME
    assert received["now_utc"] == received["now"].astimezone(received["now_utc"].tzinfo)


def test_snapshot_renderer_preserves_mlb_standings_with_numeric_league_ids(monkeypatch):
    standings = {
        mlb_standings.AL_LEAGUE_ID: {
            "East": [{"abbr": "NYY", "team_name": "Yankees", "wins": 1, "losses": 0}]
        }
    }
    monkeypatch.setattr(mlb_standings, "_fetch_league_standings", lambda: standings)

    coordinator = DataCoordinator(DataProvider())
    coordinator.read_mlb_league_standings()
    snapshot = coordinator.snapshot()
    assert str(mlb_standings.AL_LEAGUE_ID) in snapshot["mlb_league_standings"]

    captured_rows = []
    original_column_layout = mlb_standings._column_layout

    def capture_column_layout(draw, rows):
        captured_rows.extend(rows)
        return original_column_layout(draw, rows)

    monkeypatch.setattr(mlb_standings, "_column_layout", capture_column_layout)
    monkeypatch.setattr(mlb_standings, "_load_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_standings, "clear_display", lambda display: None)
    monkeypatch.setattr(mlb_standings, "scroll_vertical_content", lambda **kwargs: None)

    ScreenRenderer().render(
        "MLB AL Standings",
        PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI],
        ServerPreferenceSnapshot(revision=1),
        snapshot,
    )

    assert [row["abbr"] for row in captured_rows] == ["NYY"]
