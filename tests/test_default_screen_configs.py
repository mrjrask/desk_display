import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_default_config(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def _weather_playlist_steps(config: dict) -> list[str]:
    playlists = config["config"]["playlists"]
    for playlist in playlists.values():
        if playlist.get("label") == "weather":
            return [step["screen"] for step in playlist.get("steps", [])]
    raise AssertionError("weather playlist missing")


def test_default_screen_configs_include_weather_alert_screen():
    for filename in ("default_screens_large.json", "default_screens_small.json"):
        config = _load_default_config(filename)

        assert config["config"]["screens"].get("weather alert") == 1
        assert "weather alert" in _weather_playlist_steps(config)
