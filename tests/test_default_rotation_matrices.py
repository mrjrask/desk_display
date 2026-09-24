"""Regression matrices for the default Large and Small display rotations."""

import importlib
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


SMALL_EXPECTED_STARTUP = [
    "date",
    "weather1",
    "weather2",
    "air quality",
    "weather alert",
    "weather hourly",
    "weather daily",
    "weather radar",
    "astronomical",
    "inside",
    "news headlines",
    "vrnof",
    "hawks logo",
    "hawks last",
    "hawks next",
    "hawks next home",
    "cubs logo",
    "cubs stand3",
    "cubs last",
    "cubs live",
    "cubs next",
    "cubs next home",
    "cubs current series",
    "cubs next series",
    "cubs next home series",
    "cubs schedule quad",
    "sox logo",
    "sox stand3",
    "sox last",
    "sox live",
    "sox next",
    "sox next home",
    "sox current series",
    "sox next series",
    "sox next home series",
    "sox schedule quad",
    "MLB Scoreboard",
    "NL Overview+WC",
    "AL Overview+WC",
    "bears logo",
    "bears next",
    "NFL Scoreboard",
    "NFL Overview NFC",
    "NFL Overview AFC",
    "on this day",
    "adsb stats",
    "adsb live",
]

SMALL_EXPECTED_NORMAL_PASSES = {
    1: [
        "date",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather radar",
        "inside",
        "news headlines",
        "hawks logo",
        "hawks last",
        "hawks next",
        "hawks next home",
        "cubs live",
        "sox live",
    ],
    2: [
        "date",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather radar",
        "astronomical",
        "inside",
        "news headlines 2",
        "hawks logo",
        "hawks last",
        "hawks next",
        "hawks next home",
        "cubs logo",
        "cubs stand3",
        "cubs last",
        "cubs live",
        "cubs schedule quad",
        "sox live",
        "MLB Scoreboard",
        "NL Overview+WC",
        "AL Overview+WC",
        "NFL Scoreboard",
    ],
    3: [
        "date",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather radar",
        "inside",
        "news headlines",
        "vrnof",
        "hawks logo",
        "hawks last",
        "hawks next",
        "hawks next home",
        "cubs live",
        "cubs next",
        "cubs next home",
        "cubs current series",
        "cubs next series",
        "cubs next home series",
        "sox live",
        "bears logo",
        "bears next",
        "adsb stats",
        "adsb live",
    ],
    4: [
        "date",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather radar",
        "astronomical",
        "inside",
        "news headlines 2",
        "hawks logo",
        "hawks last",
        "hawks next",
        "hawks next home",
        "cubs logo",
        "cubs stand3",
        "cubs last",
        "cubs live",
        "cubs schedule quad",
        "sox live",
        "MLB Scoreboard",
        "NL Overview+WC",
        "AL Overview+WC",
        "NFL Scoreboard",
        "NFL Overview NFC",
        "NFL Overview AFC",
    ],
    5: [
        "date",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather radar",
        "inside",
        "news headlines",
        "hawks logo",
        "hawks last",
        "hawks next",
        "hawks next home",
        "cubs live",
        "sox logo",
        "sox stand3",
        "sox last",
        "sox live",
        "sox schedule quad",
    ],
    6: [
        "date",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather radar",
        "astronomical",
        "inside",
        "news headlines 2",
        "vrnof",
        "hawks logo",
        "hawks last",
        "hawks next",
        "hawks next home",
        "cubs logo",
        "cubs stand3",
        "cubs last",
        "cubs live",
        "cubs next",
        "cubs next home",
        "cubs current series",
        "cubs next series",
        "cubs next home series",
        "cubs schedule quad",
        "sox live",
        "MLB Scoreboard",
        "NL Overview+WC",
        "AL Overview+WC",
        "bears logo",
        "bears next",
        "NFL Scoreboard",
        "on this day",
        "adsb stats",
        "adsb live",
    ],
    7: [
        "date",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather radar",
        "inside",
        "news headlines",
        "hawks logo",
        "hawks last",
        "hawks next",
        "hawks next home",
        "cubs live",
        "sox live",
        "sox next",
        "sox next home",
        "sox current series",
        "sox next series",
        "sox next home series",
    ],
    8: [
        "date",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather radar",
        "astronomical",
        "inside",
        "news headlines 2",
        "hawks logo",
        "hawks last",
        "hawks next",
        "hawks next home",
        "cubs logo",
        "cubs stand3",
        "cubs last",
        "cubs live",
        "cubs schedule quad",
        "sox live",
        "MLB Scoreboard",
        "MLB NL Standings",
        "MLB AL Standings",
        "NFL Scoreboard",
        "NFL Overview NFC",
        "NFL Overview AFC",
    ],
    9: [
        "date",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather radar",
        "inside",
        "news headlines",
        "vrnof",
        "hawks logo",
        "hawks last",
        "hawks next",
        "hawks next home",
        "cubs live",
        "cubs next",
        "cubs next home",
        "cubs current series",
        "cubs next series",
        "cubs next home series",
        "sox live",
        "bears logo",
        "bears next",
        "adsb stats",
        "adsb live",
    ],
    10: [
        "date",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather radar",
        "astronomical",
        "inside",
        "news headlines 2",
        "hawks logo",
        "hawks last",
        "hawks next",
        "hawks next home",
        "cubs logo",
        "cubs stand3",
        "cubs last",
        "cubs live",
        "cubs schedule quad",
        "sox logo",
        "sox stand3",
        "sox last",
        "sox live",
        "sox schedule quad",
        "MLB Scoreboard",
        "NL Overview+WC",
        "AL Overview+WC",
        "NFL Scoreboard",
    ],
    11: [
        "date",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather radar",
        "inside",
        "news headlines",
        "hawks logo",
        "hawks last",
        "hawks next",
        "hawks next home",
        "cubs live",
        "sox live",
    ],
    12: [
        "date",
        "weather1",
        "weather2",
        "air quality",
        "weather alert",
        "weather hourly",
        "weather daily",
        "weather radar",
        "astronomical",
        "inside",
        "news headlines 2",
        "vrnof",
        "hawks logo",
        "hawks last",
        "hawks next",
        "hawks next home",
        "cubs logo",
        "cubs stand3",
        "cubs last",
        "cubs live",
        "cubs next",
        "cubs next home",
        "cubs current series",
        "cubs next series",
        "cubs next home series",
        "cubs schedule quad",
        "sox live",
        "MLB Scoreboard",
        "NL Overview+WC",
        "AL Overview+WC",
        "bears logo",
        "bears next",
        "NFL Scoreboard",
        "NFL Standings NFC",
        "NFL Standings AFC",
        "on this day",
        "adsb stats",
        "adsb live",
    ],
}


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
    playlist_ids.extend(playlist_id for playlist_id in playlists if playlist_id not in playlist_ids)

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
        pass_number: [entry.screen_id for entry in entries if entry.pass_number == pass_number]
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
    frequency_zero = {screen_id for screen_id, spec in screens.items() if _frequency(spec) == 0}

    startup, _ = _rotation_matrix()

    assert startup == EXPECTED_STARTUP
    assert startup == enabled_bases
    assert len(startup) == len(set(startup))
    assert frequency_zero.isdisjoint(startup)


@pytest.mark.parametrize(("pass_number", "expected_ids"), EXPECTED_NORMAL_PASSES.items())
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
    ] == ["news headlines" if number % 2 else "news headlines 2" for number in range(1, 13)]

    screens = _large_config()["screens"]
    for frequency in (2, 3, 6):
        screen_id = next(
            screen_id for screen_id, spec in screens.items() if _frequency(spec) == frequency
        )
        assert next(number for number, ids in normal.items() if screen_id in ids) == frequency

    for conference in ("NFC", "AFC"):
        overview = f"NFL Overview {conference}"
        standings = f"NFL Standings {conference}"
        assert [number for number, ids in normal.items() if overview in ids] == [4, 8]
        assert [number for number, ids in normal.items() if standings in ids] == [12]


def _small_config() -> dict:
    with (ROOT / "default_screens_small.json").open(encoding="utf-8") as config_file:
        return json.load(config_file)["config"]


def _small_rotation_matrix() -> tuple[list[str], dict[int, list[str]]]:
    scheduler = build_scheduler(_small_config())
    entries = scheduler.preview_scheduled_entries(1_000)
    startup = [entry.screen_id for entry in entries if entry.phase == "startup"]
    normal = {
        pass_number: [entry.screen_id for entry in entries if entry.pass_number == pass_number]
        for pass_number in SMALL_EXPECTED_NORMAL_PASSES
    }
    return startup, normal


def test_small_default_startup_hydrates_enabled_bases_without_changing_cadence():
    config = _small_config()
    screens = config["screens"]
    enabled_bases = [
        screen_id
        for screen_id in _resolved_config_page_order(config)
        if _frequency(screens[screen_id]) > 0
    ]
    startup, normal = _small_rotation_matrix()

    assert startup == SMALL_EXPECTED_STARTUP == enabled_bases
    assert len(startup) == len(set(startup))
    # An immediate pass 1 proves hydration did not consume a normal presentation.
    assert normal[1] == SMALL_EXPECTED_NORMAL_PASSES[1]


@pytest.mark.parametrize(("pass_number", "expected_ids"), SMALL_EXPECTED_NORMAL_PASSES.items())
def test_small_default_normal_pass_matches_approved_matrix(pass_number, expected_ids):
    _, normal = _small_rotation_matrix()

    assert normal[pass_number] == expected_ids


def test_small_default_zero_frequency_screens_only_appear_as_configured_alternates():
    config = _small_config()
    screens = config["screens"]
    _, normal = _small_rotation_matrix()
    scheduled = {screen_id for ids in normal.values() for screen_id in ids}
    alternates = {
        alternate
        for spec in screens.values()
        if isinstance(spec, dict) and isinstance(spec.get("alt"), dict)
        for alternate in (
            spec["alt"]["screen"]
            if isinstance(spec["alt"]["screen"], list)
            else [spec["alt"]["screen"]]
        )
    }
    zero_frequency = {screen_id for screen_id, spec in screens.items() if _frequency(spec) == 0}

    assert scheduled & zero_frequency <= alternates


def test_small_default_alternates_replace_bases_at_presentation_boundaries():
    _, normal = _small_rotation_matrix()

    for pass_number in range(1, 13):
        headlines = [
            screen_id
            for screen_id in normal[pass_number]
            if screen_id in {"news headlines", "news headlines 2"}
        ]
        assert headlines == ["news headlines 2" if pass_number % 2 == 0 else "news headlines"]

    for conference in ("NFC", "AFC"):
        overview = f"NFL Overview {conference}"
        standings = f"NFL Standings {conference}"
        assert [number for number, ids in normal.items() if overview in ids] == [4, 8]
        assert [number for number, ids in normal.items() if standings in ids] == [12]


def test_small_default_multiple_alternates_rotate_in_configured_order():
    scheduler = build_scheduler(_small_config())
    entries = scheduler.preview_scheduled_entries(2_000)

    for league in ("NL", "AL"):
        base = f"{league} Overview+WC"
        family = {base, f"MLB {league} Standings", f"MLB {league}WC Standings"}
        presentations = [
            entry.screen_id
            for entry in entries
            if entry.phase == "normal" and entry.screen_id in family
        ]
        assert presentations[:12] == [
            base,
            base,
            base,
            f"MLB {league} Standings",
            base,
            base,
            base,
            f"MLB {league}WC Standings",
            base,
            base,
            base,
            f"MLB {league} Standings",
        ]


def test_small_default_frequencies_are_independent_from_large_defaults():
    small = _small_config()["screens"]
    large = _large_config()["screens"]

    assert _frequency(small["air quality"]) == 1
    assert _frequency(large["air quality"]) == 2
    assert _frequency(small["weather daily"]) == 1
    assert _frequency(large["weather daily"]) == 3
    assert _frequency(small["hawks logo"]) == 1
    assert _frequency(large["hawks logo"]) == 2


def test_small_default_extra_seconds_changes_duration_not_pass_eligibility():
    config = _small_config()
    baseline = build_scheduler(config)
    changed_config = json.loads(json.dumps(config))
    changed_config["screens"]["date"] = {"frequency": 1, "extra_seconds": 17}
    changed = build_scheduler(changed_config)

    baseline_entries = baseline.preview_scheduled_entries(500)
    changed_entries = changed.preview_scheduled_entries(500)
    assert [(entry.phase, entry.pass_number, entry.screen_id) for entry in changed_entries] == [
        (entry.phase, entry.pass_number, entry.screen_id) for entry in baseline_entries
    ]
    assert baseline.extra_seconds_for("date") == 0
    assert changed.extra_seconds_for("date") == 17


def test_config_ui_small_defaults_match_scheduler_configuration():
    pytest.importorskip("flask")
    config_ui = importlib.import_module("config_ui")

    response = config_ui.app.test_client().get("/api/screens/defaults?profile=small")

    assert response.status_code == 200
    payload = response.get_json()
    # The defaults endpoint always injects a normalized "scroll" block (see
    # has_explicit_scroll / _validate_config_payload), even when the source
    # file omits one, so compare it separately from the rest of the config.
    expected_config = _small_config()
    expected_config["scroll"] = config_ui._normalize_scroll_settings(
        expected_config.get("scroll")
    )
    assert payload["config"] == expected_config
    assert build_scheduler(payload["config"]).preview_scheduled_entries(1_000) == (
        build_scheduler(expected_config).preview_scheduled_entries(1_000)
    )
