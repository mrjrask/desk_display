"""Tests for the hardware-free rendering boundary."""

from types import SimpleNamespace

from PIL import Image

from display_profiles import DISPLAY_PROFILE_DISPLAY_HAT_MINI, PROFILE_PRESETS
from rendering.screen_renderer import ScreenRenderer, ServerPreferenceSnapshot
from services.data_coordinator import DataCoordinator


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
