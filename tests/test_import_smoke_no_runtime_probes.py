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
        "screens.draw_inside",
        "screens.nba_scoreboard",
        "screens.nhl_scoreboard",
        "screens.nfl_standings",
        "screens.nhl_standings",
        "services.sports.nhl",
    }
    for module_name in ["screens.registry", *heavy_modules]:
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
