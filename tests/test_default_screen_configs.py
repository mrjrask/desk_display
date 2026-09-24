import json
from pathlib import Path

import config_ui
from schedule import REPLACEMENT_ONLY_SCREENS, build_scheduler

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


def test_default_screen_configs_include_wolves_live_in_approved_position():
    expected_prefixes = {
        "default_screens_large.json": ["wolves logo", "wolves last", "wolves live"],
        "default_screens_small.json": ["wolves logo", "wolves live"],
        "screens_config.json": ["wolves logo", "wolves live"],
    }
    for filename, expected_prefix in expected_prefixes.items():
        config = _load_default_config(filename)
        rotation_config = config.get("config", config)

        assert rotation_config["screens"].get("wolves live") == 0
        assert _wolves_playlist_steps(config)[: len(expected_prefix)] == expected_prefix


LARGE_SPREADSHEET_SEQUENCE = ['date',
 'nixie',
 'quad',
 'on this day',
 'news headlines',
 'news headlines 2',
 'weather logo',
 'weather1',
 'weather2',
 'air quality',
 'weather alert',
 'weather hourly',
 'weather daily',
 'astronomical',
 'weather quad',
 'weather radar',
 'inside',
 'verano logo',
 'vrnof',
 'bears logo',
 'bears stand1',
 'bears stand2',
 'bears next',
 'bears next season',
 'bears next season sched',
 'nfl logo',
 'NFL Scoreboard',
 'NFL Overview NFC',
 'NFL Overview AFC',
 'NFL Standings NFC',
 'NFL Standings AFC',
 'nba logo',
 'NBA Scoreboard',
 'NBA Playoffs',
 'NCAAM Scoreboard',
 'World Cup Scoreboard',
 'bulls logo',
 'bulls stand1',
 'bulls last',
 'bulls live',
 'bulls next',
 'bulls next home',
 'bulls schedule quad',
 'hawks logo',
 'hawks stand1',
 'hawks last',
 'hawks live',
 'hawks next',
 'hawks next home',
 'hawks schedule quad',
 'nhl logo',
 'NHL Scoreboard',
 'NHL Playoffs',
 'NHL Standings Overview West',
 'NHL Standings Overview East',
 'NHL Standings West',
 'NHL Standings West v2',
 'NHL Standings East',
 'NHL Standings East v2',
 'wolves logo',
 'wolves last',
 'wolves live',
 'wolves next',
 'wolves next home',
 'cubs logo',
 'cubs stand1',
 'cubs stand2',
 'cubs stand3',
 'cubs last',
 'cubs live',
 'cubs no game',
 'cubs next',
 'cubs next home',
 'cubs current series',
 'cubs next series',
 'cubs next home series',
 'cubs schedule quad',
 'sox logo',
 'sox stand1',
 'sox stand2',
 'sox stand3',
 'sox last',
 'sox live',
 'sox no game',
 'sox next',
 'sox next home',
 'sox current series',
 'sox next series',
 'sox next home series',
 'sox schedule quad',
 'mlb logo',
 'MLB Scoreboard',
 'NL Overview',
 'AL Overview',
 'NL Overview+WC',
 'AL Overview+WC',
 'MLB AL Standings',
 'MLB ALWC Standings',
 'MLB NL Standings',
 'MLB NLWC Standings',
 'adsb stats',
 'adsb live']


LARGE_PLAYLIST_SEQUENCE = ['starter',
 'Other',
 'news & stocks',
 'weather',
 'sensors',
 'bears',
 'nfl',
 'nba',
 'bulls',
 'hawks',
 'nhl',
 'wolves',
 'cubs',
 'sox',
 'mlb']


LARGE_RESOLVED_ORDER = ['date',
 'nixie',
 'quad',
 'on this day',
 'NCAAM Scoreboard',
 'World Cup Scoreboard',
 'adsb stats',
 'adsb live',
 'news headlines',
 'news headlines 2',
 'verano logo',
 'vrnof',
 'weather logo',
 'weather1',
 'weather2',
 'air quality',
 'weather alert',
 'weather hourly',
 'weather daily',
 'astronomical',
 'weather quad',
 'weather radar',
 'inside',
 'bears logo',
 'bears stand1',
 'bears stand2',
 'bears next',
 'bears next season',
 'bears next season sched',
 'nfl logo',
 'NFL Scoreboard',
 'NFL Overview NFC',
 'NFL Overview AFC',
 'NFL Standings NFC',
 'NFL Standings AFC',
 'nba logo',
 'NBA Scoreboard',
 'NBA Playoffs',
 'bulls logo',
 'bulls stand1',
 'bulls last',
 'bulls live',
 'bulls next',
 'bulls next home',
 'bulls schedule quad',
 'hawks logo',
 'hawks stand1',
 'hawks last',
 'hawks live',
 'hawks next',
 'hawks next home',
 'hawks schedule quad',
 'nhl logo',
 'NHL Scoreboard',
 'NHL Playoffs',
 'NHL Standings Overview West',
 'NHL Standings Overview East',
 'NHL Standings West',
 'NHL Standings West v2',
 'NHL Standings East',
 'NHL Standings East v2',
 'wolves logo',
 'wolves last',
 'wolves live',
 'wolves next',
 'wolves next home',
 'cubs logo',
 'cubs stand1',
 'cubs stand2',
 'cubs stand3',
 'cubs last',
 'cubs live',
 'cubs no game',
 'cubs next',
 'cubs next home',
 'cubs current series',
 'cubs next series',
 'cubs next home series',
 'cubs schedule quad',
 'sox logo',
 'sox stand1',
 'sox stand2',
 'sox stand3',
 'sox last',
 'sox live',
 'sox no game',
 'sox next',
 'sox next home',
 'sox current series',
 'sox next series',
 'sox next home series',
 'sox schedule quad',
 'mlb logo',
 'MLB Scoreboard',
 'NL Overview',
 'AL Overview',
 'NL Overview+WC',
 'AL Overview+WC',
 'MLB AL Standings',
 'MLB ALWC Standings',
 'MLB NL Standings',
 'MLB NLWC Standings']


def test_default_screens_large_matches_complete_approved_sequence():
    config = _load_default_config("default_screens_large.json")["config"]
    position = {screen_id: index for index, screen_id in enumerate(LARGE_SPREADSHEET_SEQUENCE)}

    assert list(config["screens"]) == LARGE_SPREADSHEET_SEQUENCE

    playlist_labels = []
    playlist_screen_ids = set()
    for sequence_item in config["sequence"]:
        playlist = config["playlists"][sequence_item["playlist"]]
        steps = [step["screen"] for step in playlist["steps"]]
        playlist_labels.append(playlist["label"])
        playlist_screen_ids.update(steps)
        assert steps == sorted(steps, key=position.__getitem__)

    assert playlist_labels == LARGE_PLAYLIST_SEQUENCE
    assert playlist_screen_ids == set(LARGE_SPREADSHEET_SEQUENCE)


def test_large_config_ui_resolves_complete_order_and_scheduler_resolves_enabled_order():
    config = _load_default_config("default_screens_large.json")["config"]
    ordered_ids = config_ui._ordered_screen_ids(
        config["screens"], exclude=config_ui.LEGACY_RETIRED_SCREEN_IDS
    )
    playlists, assignments = config_ui._build_playlist_assignments(config)

    # The Config page groups its ordered entries client-side with the same playlist
    # data. The Screenshots page calls this helper directly before rendering cards.
    config_page_order = config_ui._apply_playlist_grouping(
        [entry["id"] for entry in config_ui._build_screen_entries(config, {})],
        playlists,
        assignments,
    )
    screenshots_page_order = config_ui._apply_playlist_grouping(
        ordered_ids, playlists, assignments
    )
    runtime_order = [entry.screen_id for entry in build_scheduler(config)._entries]
    expected_runtime_order = [
        screen_id
        for screen_id in LARGE_RESOLVED_ORDER
        if screen_id not in REPLACEMENT_ONLY_SCREENS
        and (
            config["screens"][screen_id].get("frequency", 0)
            if isinstance(config["screens"][screen_id], dict)
            else config["screens"][screen_id]
        )
        > 0
    ]

    assert config_page_order == LARGE_RESOLVED_ORDER
    assert screenshots_page_order == LARGE_RESOLVED_ORDER
    assert runtime_order == expected_runtime_order
