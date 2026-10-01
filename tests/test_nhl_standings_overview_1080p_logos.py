"""NHL West/East overview logos fill their cells on the 1080p display only."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_CODE = (
    "from screens import nhl_standings as s\n"
    "for sid in ('NHL Standings Overview West', 'NHL Standings Overview East'):\n"
    "    s._apply_style_overrides(sid)\n"
    "    print(s._row_logo_box(220.0, 3), s._row_logo_box(220.0, 8))\n"
)


def _logo_boxes(width: int, height: int, profile: str) -> list[tuple[int, int]]:
    env = dict(
        os.environ,
        DISPLAY_WIDTH=str(width),
        DISPLAY_HEIGHT=str(height),
        DESK_DISPLAY_PROFILE=profile,
    )
    result = subprocess.run(
        [sys.executable, "-c", _CODE],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.strip().splitlines()[-2:]
    return [tuple(int(v) for v in line.split()) for line in lines]


def test_1080p_overview_logos_use_the_space():
    for leaders, rest in _logo_boxes(1920, 1080, "hdmi_1080p"):
        # Before: capped at ~103px, leaving most of the screen empty.
        assert leaders >= 180
        assert rest >= 180


@pytest.mark.parametrize(
    ("width", "height", "profile"),
    [
        (320, 240, "waveshare_lcd_320x240"),
        (720, 720, "hyperpixel4_square"),
        (800, 480, "hyperpixel4"),
    ],
)
def test_other_profiles_keep_v01_overview_logo_sizing(width, height, profile):
    env = dict(
        os.environ,
        DISPLAY_WIDTH=str(width),
        DISPLAY_HEIGHT=str(height),
        DESK_DISPLAY_PROFILE=profile,
    )
    code = (
        "from screens import nhl_standings as s\n"
        "s._apply_style_overrides('NHL Standings Overview West')\n"
        "print(s._ACTIVE_OVERVIEW_LOGO_PADDING == s.OVERVIEW_LOGO_PADDING,"
        " s.OVERVIEW_MAX_LOGO_HEIGHT != s.OVERVIEW_MAX_LOGO_HEIGHT_1080P)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "True True"
