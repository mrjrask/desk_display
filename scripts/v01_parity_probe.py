#!/usr/bin/env python3
"""Record what one display profile draws, for comparison with the v0.1 release.

Run it in a process configured for one profile (see
``rendering.profile_process.composition_env``).  It writes to OUT_DIR:

* ``constants.json``: every font size and upper-case numeric layout constant
  the screen modules and ``config`` hold after import;
* PNG stills of the MLB AL standings (first frame), the AL overview (settled
  frame) and the ``date`` and ``nixie`` clock faces at a fixed time and
  colour, drawn from the standings fixture in ``FIXTURE``.

With ``--v01`` it runs against a checkout of the v0.1 tag (the standalone
display, which drew everything in its own process); without it, against
this tree the way the render server now does.  ``scripts/make_v01_references.py``
records the v0.1 side into ``tests/fixtures/v01_reference`` and
``tests/test_v01_parity.py`` compares the current side with it.

Usage: v01_parity_probe.py ROOT PROFILE_ID FIXTURE OUT_DIR [--v01]
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import logging
import os
import pkgutil
import sys
import time

NOW = dt.datetime(2026, 9, 27, 14, 5, 9, tzinfo=dt.timezone.utc)
COLORS = ((255, 128, 0), (0, 200, 255))


def _constants(config) -> dict:
    from PIL import ImageFont

    import screens

    def snap(namespace: dict) -> dict:
        found = {}
        for name, value in namespace.items():
            if isinstance(value, ImageFont.FreeTypeFont):
                found[name] = ["font", os.path.basename(value.path), value.size]
            elif not name.isupper() or "MTIME" in name or isinstance(value, bool):
                continue
            elif isinstance(value, int | float):
                found[name] = value
            elif isinstance(value, tuple) and value and all(
                    isinstance(v, int | float) and not isinstance(v, bool) for v in value):
                found[name] = list(value)
        return found

    result = {"config": snap(vars(config))}
    for info in sorted(pkgutil.iter_modules(screens.__path__), key=lambda i: i.name):
        try:
            module = importlib.import_module(f"screens.{info.name}")
        except Exception:  # noqa: BLE001 - a module that cannot import has no constants
            continue
        result[f"screens.{info.name}"] = snap(vars(module))
    return result


class _Display:
    def __init__(self, width: int, height: int, *, skip: bool) -> None:
        self.width, self.height, self.skip, self.frames = width, height, skip, []

    def clear(self) -> None:
        pass

    def image(self, image) -> None:
        self.frames.append(image.copy())

    def show(self) -> None:
        pass

    def skip_requested(self) -> bool:
        return self.skip and bool(self.frames)

    def wait_for_skip(self, *_args, **_kwargs) -> bool:
        return False


def _v01(config, standings: dict, out: str) -> None:
    import screens.draw_date_time as date_time
    import screens.draw_nixie as nixie
    import screens.mlb_league_standings as mlb

    mlb._fetch_league_standings = lambda **_kwargs: standings
    display = _Display(config.WIDTH, config.HEIGHT, skip=True)
    mlb.draw_mlb_al_standings(display)
    display.frames[0].convert("RGB").save(os.path.join(out, "mlb_al_standings.png"))
    display = _Display(config.WIDTH, config.HEIGHT, skip=False)
    mlb.draw_AL_Overview(display)
    display.frames[-1].convert("RGB").save(os.path.join(out, "al_overview.png"))
    local = NOW.astimezone(config.CENTRAL_TIME)
    date_time.display_datetime = lambda *_args, **_kwargs: local
    date_time._compose_frame("date_time", *COLORS, False, "date").convert("RGB").save(
        os.path.join(out, "date.png"))
    nixie._compose_frame(NOW).convert("RGB").save(os.path.join(out, "nixie.png"))


def _current(profile_id: str, standings: dict, out: str) -> None:
    from display_profiles import resolve_display_profile_by_id
    from remote_display.models import RenderKey, ScreenRevisions
    from remote_display.server_rendering import compose_screen
    from rendering.clock_faces import render_clock
    from rendering.logos import ProfileLogos
    from screens.registry import set_native_profile
    from services.data_coordinator import DataCoordinator

    profile = resolve_display_profile_by_id(profile_id)
    if not set_native_profile(profile):
        raise SystemExit(f"process is not configured for {profile_id}")
    data = DataCoordinator()
    data.publish("mlb_league_standings", standings)
    logos = ProfileLogos()
    for screen_id, name in (("MLB AL Standings", "mlb_al_standings"), ("AL Overview", "al_overview")):
        key = RenderKey.for_screen(screen_id, profile_id, ScreenRevisions("s", "d", "r"))
        image, _metadata, _package = compose_screen(key, profile, data.snapshot(), logos)
        image.convert("RGB").save(os.path.join(out, f"{name}.png"))
    layout = {"face": "date", "time_zone": "America/Chicago", "time_format": "12", "show_ip": False}
    render_clock(layout, profile, NOW, colors=COLORS).convert("RGB").save(os.path.join(out, "date.png"))
    render_clock({**layout, "face": "nixie"}, profile, NOW).convert("RGB").save(os.path.join(out, "nixie.png"))


def main(argv: list[str]) -> None:
    root, profile_id, fixture, out = argv[:4]
    v01 = "--v01" in argv
    os.chdir(root)
    sys.path.insert(0, root)
    os.environ["CONFIG_LOAD_DOTENV"] = "0"
    os.environ["IP_WITH_TIME"] = "0"
    logging.disable(logging.CRITICAL)
    time.sleep = lambda *_args: None  # frames, not pacing, are compared
    with open(fixture, encoding="utf-8") as handle:
        standings = {int(league): value for league, value in json.load(handle).items()}
    os.makedirs(out, exist_ok=True)
    import config

    if v01:
        _v01(config, standings, out)
    else:
        _current(profile_id, standings, out)
    with open(os.path.join(out, "constants.json"), "w", encoding="utf-8") as handle:
        json.dump(_constants(config), handle, indent=1, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
