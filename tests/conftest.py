import os
import sys
from collections.abc import MutableMapping
from pathlib import Path

# These settings are consumed while config.py and utils.py are imported.  Keep
# this setup at conftest module scope: an autouse fixture would run too late for
# test modules which import application modules during collection.
DETERMINISTIC_ENV = {
    "CONFIG_LOAD_DOTENV": "0",
    "DESK_DISPLAY_FORCE_HEADLESS": "1",
    "DESK_DISPLAY_LOW_POWER": "0",
    "DESK_DISPLAY_OUTPUT": "headless",
    "DISPLAY_FB_CONSOLE_GRAPHICS": "0",
    "DISPLAY_FB_DEVICE": "/dev/fb0",
    "DISPLAY_FB_HIDE_CONSOLE_CURSOR": "0",
    "DISPLAY_HEIGHT": "240",
    "DISPLAY_ROTATION": "0",
    "DISPLAY_WIDTH": "320",
}

# Host desktop sessions and hardware installation profiles must not select a
# backend or layout for hardware-independent tests.  Tests exercising these
# controls set their desired values with monkeypatch and reload the module that
# reads them.
HOST_ONLY_ENV = {
    "DESK_DISPLAY_PROFILE",
    "DESK_DISPLAY_SDL_DRIVERS",
    "DISPLAY",
    "DISPLAY_FB_PIXEL_FORMAT",
    "DISPLAY_FB_PIXEL_ORDER",
    "DISPLAY_ROTATION_STRICT",
    "HYPERPIXEL_PANEL",
    "SDL_VIDEODRIVER",
    "WAYLAND_DISPLAY",
    "XAUTHORITY",
    "XDG_RUNTIME_DIR",
}


def establish_deterministic_environment(environ: MutableMapping[str, str]) -> None:
    """Remove host display hints and install collection-safe test defaults."""

    for name in HOST_ONLY_ENV:
        environ.pop(name, None)
    environ.update(DETERMINISTIC_ENV)


establish_deterministic_environment(os.environ)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
