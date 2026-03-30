"""Tests for dynamic active config path selection in main."""

import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def main_module(monkeypatch):
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    yield main
    main.request_shutdown("tests")
    sys.modules.pop("main", None)


def test_load_scheduler_uses_active_config_path(main_module, monkeypatch):
    captured = {}

    def fake_load_schedule_config(path):
        captured["path"] = path
        return {"screens": {"date": 1}}

    monkeypatch.setattr(main_module, "_active_config_path", lambda: "/tmp/local.json")
    monkeypatch.setattr(main_module, "load_schedule_config", fake_load_schedule_config)
    monkeypatch.setattr(main_module, "sanitize_schedule_config", lambda config: (config, []))
    scheduler = SimpleNamespace(node_count=1, requested_ids={"date"})
    monkeypatch.setattr(main_module, "build_scheduler", lambda config: scheduler)

    loaded = main_module._load_scheduler_from_config()

    assert loaded is scheduler
    assert captured["path"] == "/tmp/local.json"


def test_refresh_schedule_rechecks_active_config_path(main_module, monkeypatch):
    paths = iter(["/tmp/default.json", "/tmp/local.json"])
    mtimes = {"/tmp/default.json": 1.0, "/tmp/local.json": 2.0}
    loader_calls = []

    def fake_active_path():
        return next(paths)

    def fake_loader():
        loader_calls.append(True)
        return SimpleNamespace(node_count=1, requested_ids={"date"})

    monkeypatch.setattr(main_module, "_active_config_path", fake_active_path)
    monkeypatch.setattr(main_module.os.path, "getmtime", lambda path: mtimes[path])
    monkeypatch.setattr(main_module, "_load_scheduler_from_config", fake_loader)

    main_module.screen_scheduler = None
    main_module._screen_config_mtime = None

    main_module.refresh_schedule_if_needed()
    main_module.refresh_schedule_if_needed()

    assert len(loader_calls) == 2
    assert main_module._screen_config_mtime == 2.0


def test_default_config_path_honors_screens_config_env(monkeypatch):
    sys.modules.pop("main", None)
    monkeypatch.setenv("SCREENS_CONFIG_PATH", "/tmp/custom_screens_config.json")

    main = importlib.import_module("main")
    try:
        assert main.DEFAULT_CONFIG_PATH == "/tmp/custom_screens_config.json"
        assert main._active_config_path() == "/tmp/custom_screens_config.json"
    finally:
        main.request_shutdown("tests")
        sys.modules.pop("main", None)


def test_active_config_path_prefers_local_file_when_present(tmp_path, monkeypatch):
    sys.modules.pop("main", None)
    default_path = tmp_path / "screens_config.json"
    local_path = tmp_path / "screens_config.local.json"
    default_path.write_text('{"screens":{"date":1}}', encoding="utf-8")
    local_path.write_text('{"screens":{"date":2}}', encoding="utf-8")

    monkeypatch.setenv("SCREENS_CONFIG_PATH", str(default_path))
    monkeypatch.setenv("SCREENS_CONFIG_LOCAL_PATH", str(local_path))

    main = importlib.import_module("main")
    try:
        assert main.DEFAULT_CONFIG_PATH == str(default_path)
        assert main.LOCAL_CONFIG_PATH == str(local_path)
        assert main._active_config_path() == str(local_path)
    finally:
        main.request_shutdown("tests")
        sys.modules.pop("main", None)
