"""Tests for startup refresh behavior."""

import importlib
import sys


class _FakeThread:
    def __init__(self, target=None, daemon=None):
        self.target = target
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started


def _load_main():
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def test_init_runtime_runs_startup_refresh_before_main_loop(monkeypatch):
    main = _load_main()

    started_targets = []
    startup_refresh_calls = []

    def _thread_factory(*, target, daemon):
        thread = _FakeThread(target=target, daemon=daemon)

        def _start():
            started_targets.append(target)
            thread.started = True

        thread.start = _start
        return thread

    monkeypatch.setattr(main, "Display", lambda: type("_D", (), {
        "register_skip_event": lambda self, _event: None,
        "set_button_callback": lambda self, _cb: None,
    })())
    monkeypatch.setattr(main, "clear_update_indicator", lambda _display: None)
    monkeypatch.setattr(main, "_start_config_ui", lambda: None)
    monkeypatch.setattr(main, "resolve_storage_paths", lambda logger=None: type("_P", (), {
        "screenshot_dir": "/tmp",
        "current_screenshot_dir": "/tmp",
        "archive_base": "/tmp",
    })())
    monkeypatch.setattr(main, "refresh_schedule_if_needed", lambda force=False: None)
    monkeypatch.setattr(main, "_startup_refresh", lambda: startup_refresh_calls.append(True))
    monkeypatch.setattr(main.threading, "Thread", _thread_factory)

    main._runtime_initialized = False
    main._background_refresh_thread = None
    main._startup_refresh_thread = None

    main.init_runtime()

    assert startup_refresh_calls == [True]
    assert started_targets == [main._background_refresh]
    assert main._startup_refresh_thread is None
