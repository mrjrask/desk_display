"""The render server draws what the v0.1 standalone display drew.

v0.1's fonts and layouts were tuned for every supported display, so they are
the baseline.  tests/fixtures/v01_reference holds, per profile, what v0.1
drew and the font sizes and layout constants it loaded (recorded by
scripts/make_v01_references.py from the v0.1 tag).  Each profile is probed
here in a process configured by rendering.profile_process, as the render
server's workers and the display clients are.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from display_profiles import PROFILE_PRESETS
from rendering.profile_process import ProfileProcessPool, composition_env

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "fixtures" / "v01_reference"
PROFILES = sorted(p.name for p in REFERENCE.iterdir() if p.is_dir())
IMAGES = ("mlb_al_standings", "al_overview", "date", "nixie")


def _probe_env(profile_id):
    import os

    env = dict(os.environ)
    for name, value in composition_env(PROFILE_PRESETS[profile_id]).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return env


@pytest.fixture(scope="module")
def probed(tmp_path_factory):
    out = tmp_path_factory.mktemp("v01_parity")
    for profile_id in PROFILES:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "v01_parity_probe.py"), str(ROOT), profile_id,
             str(REFERENCE / "standings.json"), str(out / profile_id)],
            env=_probe_env(profile_id), check=True, capture_output=True, timeout=300,
        )
    return out


def _close(actual: Image.Image, expected: Image.Image) -> tuple[bool, str]:
    """Equal size and all but a sliver of pixels within a small tolerance.

    The tolerance absorbs anti-aliasing differences between Pillow builds; a
    changed font size or layout moves far more pixels than it allows.
    """

    if actual.size != expected.size:
        return False, f"size {actual.size} != {expected.size}"
    diff = ImageChops.difference(actual.convert("RGB"), expected.convert("RGB")).convert("L")
    off = sum(count for level, count in enumerate(diff.histogram()) if level > 48)
    share = off / (actual.width * actual.height)
    return share <= 0.002, f"{share:.2%} of pixels differ"


def test_every_v01_profile_has_references():
    assert set(PROFILES) == {"display_hat_mini", "adafruit_minipitft_114", "hyperpixel4", "hyperpixel4_square",
                             "waveshare_lcd_320x240", "hdmi_1080p", "fallback_hd"}


@pytest.mark.parametrize("profile_id", PROFILES)
def test_font_sizes_and_layout_constants_match_v01(probed, profile_id):
    expected = json.loads((REFERENCE / profile_id / "constants.json").read_text())
    actual = json.loads((probed / profile_id / "constants.json").read_text())
    changed = [
        f"{module}.{name}: v0.1 {value!r}, now {actual[module][name]!r}"
        for module, names in expected.items() if module in actual
        for name, value in names.items() if name in actual[module] and actual[module][name] != value
    ]
    assert not changed, "\n".join(changed)


@pytest.mark.parametrize("profile_id", PROFILES)
@pytest.mark.parametrize("name", IMAGES)
def test_screens_render_like_v01(probed, profile_id, name):
    with Image.open(REFERENCE / profile_id / f"{name}.png") as expected, \
            Image.open(probed / profile_id / f"{name}.png") as actual:
        ok, detail = _close(actual, expected)
    assert ok, f"{profile_id} {name}: {detail}"


def test_the_render_server_worker_draws_like_v01():
    """The production path: ServerRendering with its per-profile workers."""

    from remote_display.models import RenderKey, ScreenRevisions
    from remote_display.server_rendering import ServerRendering
    from services.data_coordinator import DataCoordinator

    standings = {int(k): v for k, v in json.loads((REFERENCE / "standings.json").read_text()).items()}
    data = DataCoordinator()
    data.publish("mlb_league_standings", standings)
    rendering = ServerRendering(data, preferences="p-v01", profile_processes=ProfileProcessPool())
    try:
        for screen_id, name in (("MLB AL Standings", "mlb_al_standings"), ("AL Overview", "al_overview")):
            key = RenderKey.for_screen(screen_id, "hyperpixel4", ScreenRevisions("s", "d", "r"))
            output = rendering.render(key)
            with Image.open(REFERENCE / "hyperpixel4" / f"{name}.png") as expected:
                ok, detail = _close(output.image, expected)
            assert ok, f"{screen_id}: {detail}"
    finally:
        rendering.close()


def test_composition_env_matches_the_v01_installers():
    env = composition_env(PROFILE_PRESETS["hyperpixel4"])
    assert env["DISPLAY_WIDTH"] == "800" and env["DISPLAY_HEIGHT"] == "480"
    assert env["DESK_DISPLAY_OUTPUT"] == "kernel" and env["HYPERPIXEL_PANEL"] == "hyperpixel4"
    # v0.1 knew the Waveshare LCD as a 320x240 Display HAT Mini layout.
    waveshare = composition_env(PROFILE_PRESETS["waveshare_lcd_320x240"])
    assert waveshare["DESK_DISPLAY_PROFILE"] is None
    assert waveshare["DESK_DISPLAY_OUTPUT"] == "framebuffer"
