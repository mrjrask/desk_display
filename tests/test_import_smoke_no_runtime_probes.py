import importlib
import socket
import subprocess
import sys

import pytest


def _guard(*_args, **_kwargs):
    raise AssertionError("runtime probe should not run during module import")


@pytest.mark.parametrize("module_name", ["config", "utils"])
def test_import_does_not_probe_runtime(monkeypatch, module_name):
    monkeypatch.setattr(subprocess, "run", _guard)
    monkeypatch.setattr(subprocess, "check_call", _guard)
    monkeypatch.setattr(subprocess, "check_output", _guard)
    monkeypatch.setattr(socket, "getaddrinfo", _guard)
    monkeypatch.setattr(socket, "create_connection", _guard)

    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)


def test_import_screen_registry_does_not_import_heavy_renderers(monkeypatch):
    heavy_modules = {
        "screens.draw_bears_schedule",
        "screens.draw_bulls_schedule",
        "screens.draw_hawks_schedule",
        "screens.draw_wolves_schedule",
        "screens.draw_vrnof",
        "screens.draw_nixie",
        "screens.on_this_day",
        "screens.draw_date_time",
        "screens.mlb_schedule",
        "screens.mlb_scoreboard",
        "screens.mlb_scoreboard_v2",
        "screens.mlb_league_standings",
        "screens.mlb_team_standings",
        "screens.nba_team_standings",
        "screens.nfl_team_standings",
        "screens.nhl_team_standings",
        "screens.ncaam_scoreboard",
        "screens.world_cup_scoreboard",
        "screens.nfl_scoreboard",
        "screens.nfl_scoreboard_v2",
        "screens.nhl_playoffs",
        "screens.nba_playoffs",
        "screens.draw_inside",
        "screens.nba_scoreboard",
        "screens.nhl_scoreboard",
        "screens.nfl_standings",
        "screens.nhl_standings",
        "services.sports.nhl",
    }
    module_names = ["screens.registry", *heavy_modules]
    missing = object()
    original_modules = {name: sys.modules.get(name, missing) for name in module_names}

    try:
        for module_name in module_names:
            sys.modules.pop(module_name, None)

        real_import_module = importlib.import_module
        imported = []

        def _tracking_import(name, package=None):
            imported.append(name)
            return real_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _tracking_import)
        importlib.import_module("screens.registry")

        assert heavy_modules.isdisjoint(imported)
        assert heavy_modules.isdisjoint(sys.modules)
    finally:
        for module_name in module_names:
            original = original_modules[module_name]
            if original is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original
