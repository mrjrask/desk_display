import json
from pathlib import Path

from screens_catalog import RAW_SCREEN_IDS


def test_raw_screen_ids_are_unique():
    assert len(RAW_SCREEN_IDS) == len(set(RAW_SCREEN_IDS))


def test_weather_quad_screen_id_is_listed():
    assert "weather quad" in RAW_SCREEN_IDS


def test_astronomical_screen_is_listed_after_weather_daily():
    assert "astronomical" in RAW_SCREEN_IDS
    assert RAW_SCREEN_IDS.index("astronomical") == RAW_SCREEN_IDS.index("weather daily") + 1


def test_mlb_schedule_quad_screen_ids_are_listed():
    assert "cubs schedule quad" in RAW_SCREEN_IDS
    assert "sox schedule quad" in RAW_SCREEN_IDS


def test_mlb_no_game_screen_ids_are_listed():
    assert "cubs no game" in RAW_SCREEN_IDS
    assert "sox no game" in RAW_SCREEN_IDS


def test_mlb_no_game_default_frequency_matches_next_screen():
    config = json.loads(Path("screens_config.json").read_text())
    screens = config["screens"]

    assert screens["cubs no game"] == screens["cubs next"]
    assert screens["sox no game"] == screens["sox next"]


def test_nba_nhl_schedule_quad_screen_ids_are_listed():
    assert "bulls schedule quad" in RAW_SCREEN_IDS
    assert "hawks schedule quad" in RAW_SCREEN_IDS
