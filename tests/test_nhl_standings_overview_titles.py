"""NHL standings overview West/East screens use the short titles on every display."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from display_profiles import (
    DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114,
    DISPLAY_PROFILE_DISPLAY_HAT_MINI,
    DISPLAY_PROFILE_HDMI_1080P,
    DISPLAY_PROFILE_HYPERPIXEL4,
    DISPLAY_PROFILE_HYPERPIXEL4_SQUARE,
    DISPLAY_PROFILE_WAVESHARE_LCD_320X240,
)
from screens import nhl_standings

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "profile",
    [
        DISPLAY_PROFILE_DISPLAY_HAT_MINI,
        DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114,
        DISPLAY_PROFILE_HYPERPIXEL4,
        DISPLAY_PROFILE_HYPERPIXEL4_SQUARE,
        DISPLAY_PROFILE_WAVESHARE_LCD_320X240,
        DISPLAY_PROFILE_HDMI_1080P,
    ],
)
def test_overview_titles_for_every_profile(profile):
    env = dict(os.environ, DESK_DISPLAY_PROFILE=profile)
    code = (
        "import config\n"
        "from screens import nhl_standings as s\n"
        "print(config.DISPLAY_PROFILE_ID)\n"
        "print(s.OVERVIEW_TITLE_WEST)\n"
        "print(s.OVERVIEW_TITLE_EAST)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines()[-3:] == [profile, "NHL West", "NHL East"]


@pytest.mark.parametrize(
    ("draw", "expected"),
    [
        (nhl_standings.draw_nhl_standings_overview_west, "NHL West"),
        (nhl_standings.draw_nhl_standings_overview_east, "NHL East"),
    ],
)
def test_overview_screens_draw_short_titles(monkeypatch, draw, expected):
    titles: list[str] = []

    def fake_render_empty(title):
        titles.append(title)
        return Image.new("RGB", (nhl_standings.WIDTH, nhl_standings.HEIGHT))

    monkeypatch.setattr(nhl_standings, "_render_empty", fake_render_empty)
    monkeypatch.setattr(nhl_standings, "clear_display", lambda display: None)

    draw(object(), transition=True, standings={})

    assert titles == [expected]
