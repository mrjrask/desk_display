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


def _other_playlist_steps(config: dict) -> list[str]:
    playlists = config["config"]["playlists"]
    for playlist in playlists.values():
        if playlist.get("label") == "Other":
            return [step["screen"] for step in playlist.get("steps", [])]
    raise AssertionError("Other playlist missing")


def _wolves_playlist_steps(config: dict) -> list[str]:
    playlists = config.get("config", config)["playlists"]
    for playlist in playlists.values():
        if playlist.get("label") == "wolves":
            return [step["screen"] for step in playlist.get("steps", [])]
    raise AssertionError("wolves playlist missing")


def test_default_screen_configs_include_weather_alert_screen():
    for filename in ("default_screens_large.json", "default_screens_small.json"):
        config = _load_default_config(filename)

        assert config["config"]["screens"].get("weather alert") == 1
        assert "weather alert" in _weather_playlist_steps(config)


def test_default_screen_configs_alternate_news_headline_feeds():
    for filename in ("default_screens_large.json", "default_screens_small.json"):
        config = _load_default_config(filename)
        screens = config["config"]["screens"]

        assert screens["news headlines"] == {
            "frequency": 1,
            "alt": {"screen": "news headlines 2", "frequency": 2},
        }
        assert screens["news headlines 2"] == 0


def test_default_screen_configs_include_adsb_screens_at_end_of_other():
    for filename in ("default_screens_large.json", "default_screens_small.json"):
        config = _load_default_config(filename)

        assert config["config"]["screens"].get("adsb stats") == 3
        assert config["config"]["screens"].get("adsb live") == 3
        assert "adsb live airlines" not in config["config"]["screens"]
        other_steps = _other_playlist_steps(config)
        assert other_steps[-2:] == ["adsb stats", "adsb live"]


def test_default_screen_configs_rotate_nfl_overviews_with_conference_standings():
    for filename in ("default_screens_large.json", "default_screens_small.json"):
        config = _load_default_config(filename)
        screens = config["config"]["screens"]

        for conference in ("AFC", "NFC"):
            assert screens[f"NFL Overview {conference}"] == {
                "frequency": 4,
                "extra_seconds": 0,
                "alt": {
                    "screen": f"NFL Standings {conference}",
                    "frequency": 3,
                },
            }


def test_default_screen_configs_rotate_mlb_overviews_with_league_standings():
    for filename in ("default_screens_large.json", "default_screens_small.json"):
        config = _load_default_config(filename)
        screens = config["config"]["screens"]

        for league in ("NL", "AL"):
            assert screens[f"{league} Overview+WC"] == {
                "frequency": 2,
                "alt": {
                    "screen": [
                        f"MLB {league} Standings",
                        f"MLB {league}WC Standings",
                    ],
                    "frequency": 4,
                },
            }


def test_default_screen_configs_place_wolves_live_after_logo():
    for filename in (
        "default_screens_large.json",
        "default_screens_small.json",
        "screens_config.json",
    ):
        config = _load_default_config(filename)
        rotation_config = config.get("config", config)

        assert rotation_config["screens"].get("wolves live") == 0
        wolves_steps = _wolves_playlist_steps(config)
        assert wolves_steps[:2] == ["wolves logo", "wolves live"]
