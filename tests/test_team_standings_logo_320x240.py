"""Team Stand screens give the logo the free room on plain 320x240 panels."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from display_profiles import (
    DISPLAY_PROFILE_DISPLAY_HAT_MINI,
    DISPLAY_PROFILE_FALLBACK_DEFAULT,
    DISPLAY_PROFILE_WAVESHARE_LCD_320X240,
    resolve_display_profile_by_id,
)
from rendering.profile_process import composition_env

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import screens.mlb_team_standings as m
from screens.nhl_team_standings import draw_nhl_standings_screen1
from screens.nba_team_standings import draw_nba_standings_screen1
from screens.nfl_team_standings import draw_nfl_standings_screen1, draw_nfl_standings_screen2
targets = []
fit = m.fit_logo_to_box
m.fit_logo_to_box = lambda img, size: targets.append(size) or fit(img, size)
m.clear_display = lambda display: None
rec = {"leagueRecord": {"wins": 85, "losses": 77, "pct": ".525"}, "divisionRank": 3,
       "divisionGamesBack": 7.5, "wildCardGamesBack": 2, "wildCardRank": 5,
       "records": {"splitRecords": []}}
draw_nhl_standings_screen1(None, rec, "images/nhl/CHI.png", "", screen_id="hawks stand1", transition=True)
draw_nba_standings_screen1(None, rec, "images/nba/CHI.png", "East", screen_id="bulls stand1", transition=True)
draw_nfl_standings_screen1(None, rec, "images/nfl/chi.png", "NFC North", screen_id="bears stand1", transition=True)
m.draw_standings_screen1(None, rec, "images/mlb/CUBS.png", "NL Central", screen_id="cubs stand1", transition=True)
draw_nfl_standings_screen2(None, rec, "images/nfl/chi.png", screen_id="bears stand2", transition=True)
m.draw_standings_screen2(None, rec, "images/mlb/CUBS.png", screen_id="cubs stand2", transition=True)
m.draw_standings_screen3(None, rec, "images/mlb/CUBS.png", "NL Central", screen_id="cubs stand3", transition=True)
print(" ".join(map(str, targets)))
"""


def _logo_targets(profile_id):
    env = dict(os.environ)
    for key, value in composition_env(resolve_display_profile_by_id(profile_id)).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
    )
    return [int(value) for value in result.stdout.split()[-7:]]


@pytest.mark.parametrize("profile_id", [DISPLAY_PROFILE_WAVESHARE_LCD_320X240, DISPLAY_PROFILE_FALLBACK_DEFAULT])
def test_plain_320x240_panels_draw_large_stand_logos(profile_id):
    # Stand 1 (Hawks, Bulls, Bears, Cubs), Stand 2 (Bears, Cubs), Stand 3 (Cubs)
    assert _logo_targets(profile_id) == [100, 100, 100, 100, 64, 64, 90]


def test_display_hat_mini_keeps_v01_logo_size():
    assert _logo_targets(DISPLAY_PROFILE_DISPLAY_HAT_MINI) == [108] * 7
