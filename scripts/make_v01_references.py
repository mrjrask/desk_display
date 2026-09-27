#!/usr/bin/env python3
"""Record the v0.1 reference renders that tests/test_v01_parity.py checks.

v0.1 is the layout baseline: its fonts and layouts were tuned for every
supported display.  This checks out the ``v0.1`` tag into a temporary
worktree and runs scripts/v01_parity_probe.py there once per profile, in a
process configured the way the v0.1 installer configured that display, and
writes the results to tests/fixtures/v01_reference/<profile>/.

Usage: python3 scripts/make_v01_references.py [--tag v0.1]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rendering.profile_process import composition_env  # noqa: E402
from display_profiles import PROFILE_PRESETS  # noqa: E402

REFERENCE_DIR = ROOT / "tests" / "fixtures" / "v01_reference"
# The displays v0.1 supported as its main panel.
V01_PROFILES = ("display_hat_mini", "adafruit_minipitft_114", "hyperpixel4", "hyperpixel4_square",
                "waveshare_lcd_320x240", "hdmi_1080p", "fallback_hd")


def probe_env(profile_id: str) -> dict[str, str]:
    env = dict(os.environ)
    for name, value in composition_env(PROFILE_PRESETS[profile_id]).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="v0.1")
    args = parser.parse_args()
    fixture = REFERENCE_DIR / "standings.json"
    with tempfile.TemporaryDirectory() as scratch:
        worktree = Path(scratch) / "v01"
        subprocess.run(["git", "worktree", "add", "--detach", str(worktree), args.tag], cwd=ROOT, check=True)
        try:
            for profile_id in V01_PROFILES:
                out = REFERENCE_DIR / profile_id
                shutil.rmtree(out, ignore_errors=True)
                subprocess.run([sys.executable, str(ROOT / "scripts" / "v01_parity_probe.py"), str(worktree),
                                profile_id, str(fixture), str(out), "--v01"],
                               env=probe_env(profile_id), check=True)
                print(f"recorded {profile_id}")
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
