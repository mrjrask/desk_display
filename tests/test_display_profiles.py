import importlib
import sys

import pytest

import config as base_config

PROFILE_IDS = tuple(base_config.get_available_display_profiles())


@pytest.fixture(params=PROFILE_IDS)
def profile_modules(monkeypatch, request):
    profile_id = request.param
    monkeypatch.setenv("DISPLAY_PROFILE", profile_id)

    # Reload configuration and dependent modules so they pick up the profile.
    if "config" in sys.modules:
        config_module = importlib.reload(sys.modules["config"])
    else:  # pragma: no cover - defensive
        config_module = importlib.import_module("config")

    for module_name in ("screens.draw_date_time", "screens.draw_inside"):
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:  # pragma: no cover - defensive
            importlib.import_module(module_name)

    draw_date_time = sys.modules["screens.draw_date_time"]
    draw_inside = sys.modules["screens.draw_inside"]
    return profile_id, config_module, draw_date_time, draw_inside


def test_date_screen_matches_canvas(profile_modules):
    profile_id, config_module, draw_date_time, _ = profile_modules

    image = draw_date_time.draw_date(None, transition=True)
    assert image.size == (config_module.WIDTH, config_module.HEIGHT), profile_id


def test_inside_screen_matches_canvas(monkeypatch, profile_modules):
    profile_id, config_module, _, draw_inside = profile_modules

    def _fake_probe_sensor():
        def reader():
            return {
                "temp_f": 71.2,
                "humidity": 48.5,
                "pressure_inhg": 29.92,
                "voc_ohms": 120000,
                "co2_ppm": 415,
            }

        return "Unit Test Sensor", reader

    monkeypatch.setattr(draw_inside, "_probe_sensor", _fake_probe_sensor)

    image = draw_inside.draw_inside(None, transition=True)
    assert image.size == (config_module.WIDTH, config_module.HEIGHT), profile_id
