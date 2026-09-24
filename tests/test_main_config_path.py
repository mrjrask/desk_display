"""Tests for dynamic active config path selection in main."""

import importlib
import os
import sys
import time
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


def test_sanitized_config_is_written_back_atomically(tmp_path, main_module, monkeypatch):
    """Regression test: the sanitized schedule used to be written back with
    a plain open(path, "w"), which is not atomic and could leave a
    truncated file on a mid-write crash. Confirm the write goes through
    _atomic_write_json (temp file + os.replace) instead."""

    config_path = tmp_path / "screens_config.json"
    config_path.write_text('{"screens": {"date": 1, "legacy": 1}}', encoding="utf-8")

    monkeypatch.setattr(main_module, "_active_config_path", lambda: str(config_path))
    monkeypatch.setattr(
        main_module,
        "sanitize_schedule_config",
        lambda config: ({"screens": {"date": 1}}, ["legacy"]),
    )
    scheduler = SimpleNamespace(node_count=1, requested_ids={"date"})
    monkeypatch.setattr(main_module, "build_scheduler", lambda config: scheduler)

    calls = []
    real_atomic_write = main_module._atomic_write_json

    def spying_atomic_write(path, data):
        calls.append((path, data))
        real_atomic_write(path, data)

    monkeypatch.setattr(main_module, "_atomic_write_json", spying_atomic_write)

    loaded = main_module._load_scheduler_from_config()

    assert loaded is scheduler
    assert calls == [(str(config_path), {"screens": {"date": 1}})]
    assert main_module.json.loads(config_path.read_text(encoding="utf-8")) == {
        "screens": {"date": 1}
    }


def test_sanitized_write_back_skipped_on_concurrent_edit(tmp_path, main_module, monkeypatch):
    """Regression test: writing the sanitized config back unconditionally
    could clobber a concurrent save from the config UI (a lost update).
    When the file's mtime changed between the read and the write-back
    decision, the write must be skipped."""

    config_path = tmp_path / "screens_config.json"
    config_path.write_text('{"screens": {"date": 1, "legacy": 1}}', encoding="utf-8")

    monkeypatch.setattr(main_module, "_active_config_path", lambda: str(config_path))
    monkeypatch.setattr(
        main_module,
        "sanitize_schedule_config",
        lambda config: ({"screens": {"date": 1}}, ["legacy"]),
    )
    scheduler = SimpleNamespace(node_count=1, requested_ids={"date"})
    monkeypatch.setattr(main_module, "build_scheduler", lambda config: scheduler)

    real_load = main_module.load_schedule_config

    def load_then_touch(path):
        data = real_load(path)
        # Simulate a concurrent writer (e.g. the config UI) saving a new
        # version of the file after this read but before the write-back.
        config_path.write_text('{"screens": {"date": 1, "nixie": 1}}', encoding="utf-8")
        os.utime(config_path, (time.time() + 5, time.time() + 5))
        return data

    monkeypatch.setattr(main_module, "load_schedule_config", load_then_touch)

    calls = []
    monkeypatch.setattr(
        main_module, "_atomic_write_json", lambda path, data: calls.append((path, data))
    )

    loaded = main_module._load_scheduler_from_config()

    assert loaded is scheduler
    assert calls == []
    # The concurrent writer's content on disk must be untouched.
    assert main_module.json.loads(config_path.read_text(encoding="utf-8")) == {
        "screens": {"date": 1, "nixie": 1}
    }


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
        assert str(default_path) == main.DEFAULT_CONFIG_PATH
        assert str(local_path) == main.LOCAL_CONFIG_PATH
        assert main._active_config_path() == str(local_path)
    finally:
        main.request_shutdown("tests")
        sys.modules.pop("main", None)
