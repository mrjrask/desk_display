"""Regression matrix for the default Large display rotation."""

import json
from pathlib import Path

import pytest

from schedule import build_scheduler

ROOT = Path(__file__).resolve().parents[1]

# Startup is a hydration-only traversal: alternates must not replace base screens.
EXPECTED_STARTUP = ['date',
 'on this day',
 'adsb stats',
 'adsb live',
 'news headlines',
 'vrnof',
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
 'bears next',
 'NFL Scoreboard',
 'NFL Overview NFC',
 'NFL Overview AFC',
 'hawks logo',
 'hawks last',
 'hawks live',
 'hawks next',
 'hawks next home',
 'hawks schedule quad',
 'NHL Scoreboard',
 'cubs logo',
 'cubs stand3',
 'cubs last',
 'cubs live',
 'cubs next',
 'cubs next home',
 'cubs current series',
 'cubs next series',
 'cubs next home series',
 'cubs schedule quad',
 'sox logo',
 'sox stand3',
 'sox last',
 'sox live',
 'sox next',
 'sox next home',
 'sox current series',
 'sox next series',
 'sox next home series',
 'sox schedule quad',
 'MLB Scoreboard',
 'NL Overview+WC',
 'AL Overview+WC']

# Normal rotations transcribed as ordered rows from the approved scheduling matrix.
# Keeping every row explicit makes a changed default produce a useful list diff.
EXPECTED_NORMAL_PASSES = {1: ['date',
     'news headlines',
     'weather1',
     'weather2',
     'weather alert',
     'weather hourly',
     'weather quad',
     'weather radar',
     'inside',
     'cubs live',
     'sox live'],
 2: ['date',
     'news headlines 2',
     'weather1',
     'weather2',
     'air quality',
     'weather alert',
     'weather hourly',
     'astronomical',
     'weather quad',
     'weather radar',
     'inside',
     'NFL Scoreboard',
     'hawks logo',
     'hawks last',
     'hawks live',
     'hawks next',
     'hawks next home',
     'hawks schedule quad',
     'NHL Scoreboard',
     'cubs logo',
     'cubs stand3',
     'cubs last',
     'cubs live',
     'cubs schedule quad',
     'sox live',
     'MLB Scoreboard',
     'NL Overview+WC',
     'AL Overview+WC'],
 3: ['nixie',
     'adsb stats',
     'adsb live',
     'news headlines',
     'vrnof',
     'weather1',
     'weather2',
     'weather alert',
     'weather hourly',
     'weather daily',
     'weather quad',
     'weather radar',
     'inside',
     'bears logo',
     'bears next',
     'cubs live',
     'cubs next',
     'cubs next home',
     'cubs current series',
     'cubs next series',
     'cubs next home series',
     'sox live'],
 4: ['date',
     'news headlines 2',
     'weather1',
     'weather2',
     'air quality',
     'weather alert',
     'weather hourly',
     'astronomical',
     'weather quad',
     'weather radar',
     'inside',
     'NFL Scoreboard',
     'NFL Overview NFC',
     'NFL Overview AFC',
     'hawks logo',
     'hawks last',
     'hawks live',
     'hawks next',
     'hawks next home',
     'hawks schedule quad',
     'NHL Scoreboard',
     'cubs logo',
     'cubs stand3',
     'cubs last',
     'cubs live',
     'cubs schedule quad',
     'sox live',
     'MLB Scoreboard',
     'NL Overview+WC',
     'AL Overview+WC'],
 5: ['date',
     'news headlines',
     'weather1',
     'weather2',
     'weather alert',
     'weather hourly',
     'weather quad',
     'weather radar',
     'inside',
     'cubs live',
     'sox logo',
     'sox stand3',
     'sox last',
     'sox live',
     'sox schedule quad'],
 6: ['nixie',
     'on this day',
     'adsb stats',
     'adsb live',
     'news headlines 2',
     'vrnof',
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
     'bears next',
     'NFL Scoreboard',
     'hawks logo',
     'hawks last',
     'hawks live',
     'hawks next',
     'hawks next home',
     'hawks schedule quad',
     'NHL Scoreboard',
     'cubs logo',
     'cubs stand3',
     'cubs last',
     'cubs live',
     'cubs next',
     'cubs next home',
     'cubs current series',
     'cubs next series',
     'cubs next home series',
     'cubs schedule quad',
     'sox live',
     'MLB Scoreboard',
     'NL Overview+WC',
     'AL Overview+WC'],
 7: ['date',
     'news headlines',
     'weather1',
     'weather2',
     'weather alert',
     'weather hourly',
     'weather quad',
     'weather radar',
     'inside',
     'cubs live',
     'sox live',
     'sox next',
     'sox next home',
     'sox current series',
     'sox next series',
     'sox next home series'],
 8: ['date',
     'news headlines 2',
     'weather1',
     'weather2',
     'air quality',
     'weather alert',
     'weather hourly',
     'astronomical',
     'weather quad',
     'weather radar',
     'inside',
     'NFL Scoreboard',
     'NFL Overview NFC',
     'NFL Overview AFC',
     'hawks logo',
     'hawks last',
     'hawks live',
     'hawks next',
     'hawks next home',
     'hawks schedule quad',
     'NHL Scoreboard',
     'cubs logo',
     'cubs stand3',
     'cubs last',
     'cubs live',
     'cubs schedule quad',
     'sox live',
     'MLB Scoreboard',
     'MLB NL Standings',
     'MLB AL Standings'],
 9: ['nixie',
     'adsb stats',
     'adsb live',
     'news headlines',
     'vrnof',
     'weather1',
     'weather2',
     'weather alert',
     'weather hourly',
     'weather daily',
     'weather quad',
     'weather radar',
     'inside',
     'bears logo',
     'bears next',
     'cubs live',
     'cubs next',
     'cubs next home',
     'cubs current series',
     'cubs next series',
     'cubs next home series',
     'sox live'],
 10: ['date',
      'news headlines 2',
      'weather1',
      'weather2',
      'air quality',
      'weather alert',
      'weather hourly',
      'astronomical',
      'weather quad',
      'weather radar',
      'inside',
      'NFL Scoreboard',
      'hawks logo',
      'hawks last',
      'hawks live',
      'hawks next',
      'hawks next home',
      'hawks schedule quad',
      'NHL Scoreboard',
      'cubs logo',
      'cubs stand3',
      'cubs last',
      'cubs live',
      'cubs schedule quad',
      'sox logo',
      'sox stand3',
      'sox last',
      'sox live',
      'sox schedule quad',
      'MLB Scoreboard',
      'NL Overview+WC',
      'AL Overview+WC'],
 11: ['date',
      'news headlines',
      'weather1',
      'weather2',
      'weather alert',
      'weather hourly',
      'weather quad',
      'weather radar',
      'inside',
      'cubs live',
      'sox live'],
 12: ['nixie',
      'on this day',
      'adsb stats',
      'adsb live',
      'news headlines 2',
      'vrnof',
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
      'bears next',
      'NFL Scoreboard',
      'NFL Standings NFC',
      'NFL Standings AFC',
      'hawks logo',
      'hawks last',
      'hawks live',
      'hawks next',
      'hawks next home',
      'hawks schedule quad',
      'NHL Scoreboard',
      'cubs logo',
      'cubs stand3',
      'cubs last',
      'cubs live',
      'cubs next',
      'cubs next home',
      'cubs current series',
      'cubs next series',
      'cubs next home series',
      'cubs schedule quad',
      'sox live',
      'MLB Scoreboard',
      'NL Overview+WC',
      'AL Overview+WC']}


def _large_config() -> dict:
    with (ROOT / "default_screens_large.json").open(encoding="utf-8") as config_file:
        return json.load(config_file)["config"]


def _frequency(spec: object) -> int:
    if isinstance(spec, dict):
        return int(spec["frequency"])
    return int(spec)


def _resolved_config_page_order(config: dict) -> list[str]:
    """Resolve Ungrouped then sequenced playlists, like the config page/scheduler."""
    screens = config["screens"]
    playlists = config["playlists"]
    playlist_ids = [item["playlist"] for item in config["sequence"]]
    playlist_ids.extend(
        playlist_id for playlist_id in playlists if playlist_id not in playlist_ids
    )

    assignment = {}
    for playlist_id in playlist_ids:
        for step in playlists[playlist_id]["steps"]:
            screen_id = step["screen"]
            if screen_id in screens:
                assignment.setdefault(screen_id, playlist_id)

    return [
        screen_id
        for group_id in [None, *playlist_ids]
        for screen_id in screens
        if assignment.get(screen_id) == group_id
    ]


def _rotation_matrix() -> tuple[list[str], dict[int, list[str]]]:
    scheduler = build_scheduler(_large_config())
    entries = scheduler.preview_scheduled_entries(1_000)
    startup = [entry.screen_id for entry in entries if entry.phase == "startup"]
    normal = {
        pass_number: [
            entry.screen_id for entry in entries if entry.pass_number == pass_number
        ]
        for pass_number in EXPECTED_NORMAL_PASSES
    }
    return startup, normal


def test_large_default_startup_hydrates_each_enabled_base_once_in_resolved_order():
    config = _large_config()
    screens = config["screens"]
    resolved_order = _resolved_config_page_order(config)
    enabled_bases = [
        screen_id for screen_id in resolved_order if _frequency(screens[screen_id]) >= 1
    ]
    frequency_zero = {
        screen_id for screen_id, spec in screens.items() if _frequency(spec) == 0
    }

    startup, _ = _rotation_matrix()

    assert startup == EXPECTED_STARTUP
    assert startup == enabled_bases
    assert len(startup) == len(set(startup))
    assert frequency_zero.isdisjoint(startup)


@pytest.mark.parametrize(
    ("pass_number", "expected_ids"), EXPECTED_NORMAL_PASSES.items()
)
def test_large_default_normal_pass_matches_approved_matrix(pass_number, expected_ids):
    _, normal = _rotation_matrix()

    assert normal[pass_number] == expected_ids


def test_large_default_matrix_encodes_key_frequency_and_alternate_boundaries():
    _, normal = _rotation_matrix()

    assert [normal[number][0] for number in range(1, 13)] == [
        "date",
        "date",
        "nixie",
        "date",
        "date",
        "nixie",
        "date",
        "date",
        "nixie",
        "date",
        "date",
        "nixie",
    ]
    assert [
        "news headlines 2" if "news headlines 2" in normal[number] else "news headlines"
        for number in range(1, 13)
    ] == [
        "news headlines" if number % 2 else "news headlines 2"
        for number in range(1, 13)
    ]

    screens = _large_config()["screens"]
    for frequency in (2, 3, 6):
        screen_id = next(
            screen_id
            for screen_id, spec in screens.items()
            if _frequency(spec) == frequency
        )
        assert (
            next(number for number, ids in normal.items() if screen_id in ids)
            == frequency
        )

    for conference in ("NFC", "AFC"):
        overview = f"NFL Overview {conference}"
        standings = f"NFL Standings {conference}"
        assert [number for number, ids in normal.items() if overview in ids] == [4, 8]
        assert [number for number, ids in normal.items() if standings in ids] == [12]
