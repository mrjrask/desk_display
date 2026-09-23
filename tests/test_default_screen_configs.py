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


def test_default_screen_configs_enable_hawks_screens_without_schedule_expiration():
    enabled_screens = ("hawks logo", "hawks last", "hawks next", "hawks next home")
    expected_frequencies = {
        "default_screens_large.json": 2,
        "default_screens_small.json": 1,
    }

    for filename, expected_frequency in expected_frequencies.items():
        config = _load_default_config(filename)
        screens = config["config"]["screens"]

        for screen in enabled_screens:
            assert screens[screen] == expected_frequency

        schedule_quad = screens["hawks schedule quad"]
        if isinstance(schedule_quad, dict):
            assert "hide_after_enabled" not in schedule_quad
            assert "hide_after_at" not in schedule_quad


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
            expected = {
                "frequency": 4,
                "alt": {
                    "screen": f"NFL Standings {conference}",
                    "frequency": 3,
                },
            }
            if filename == "default_screens_small.json":
                expected["extra_seconds"] = 0
            assert screens[f"NFL Overview {conference}"] == expected


def test_default_screen_configs_do_not_alternate_nfl_logo_or_scoreboard():
    for filename in ("default_screens_large.json", "default_screens_small.json"):
        config = _load_default_config(filename)
        screens = config["config"]["screens"]

        assert screens["nfl logo"] == 0
        assert screens["NFL Scoreboard"] == 2


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


def test_default_screens_large_screen_order_and_frequencies():
    """The large defaults match the rotation the user tuned in the web UI.

    The order/frequencies in ``screens_config.export-4.json`` (downloaded via
    the Screens export) were adopted as the repo-backed defaults in 2026-09:
    the news feeds moved to follow the weather/sensors block, "on this day"
    moved to the front of the "Other" rotation, and the hawks screens plus
    the NHL Scoreboard were re-weighted.
    """
    config = _load_default_config("default_screens_large.json")
    screens = config["config"]["screens"]
    screen_ids = list(screens)

    # Screens dict order: news feeds follow "inside"; "on this day" sits just
    # before the miscellaneous end-of-rotation screens.
    assert screen_ids.index("news headlines") == screen_ids.index("inside") + 1
    assert screen_ids.index("news headlines 2") == screen_ids.index("news headlines") + 1
    assert screen_ids.index("on this day") == screen_ids.index("NBA Playoffs") + 1

    assert screens["hawks logo"] == 2
    assert screens["hawks schedule quad"] == {"frequency": 2, "extra_seconds": 3}
    assert screens["NHL Scoreboard"] == 2
    assert screens["air quality"] == 2
    assert screens["astronomical"] == 2
    assert screens["wolves logo"] == 8
    assert screens["wolves next home"] == 8


def test_default_screens_large_playlist_step_order():
    config = _load_default_config("default_screens_large.json")
    playlists = {
        playlist["label"]: [step["screen"] for step in playlist["steps"]]
        for playlist in config["config"]["playlists"].values()
    }

    assert playlists["weather"] == [
        "weather logo",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather quad",
        "weather radar",
        "astronomical",
    ]
    assert playlists["news & stocks"] == [
        "news headlines",
        "news headlines 2",
        "verano logo",
        "vrnof",
    ]
    other = playlists["Other"]
    assert other[0] == "on this day"
    assert other[-2:] == ["adsb stats", "adsb live"]
