"""Regression coverage for explicit, idempotent main runtime setup."""

import importlib
import signal
import sys
import threading
from types import SimpleNamespace
from typing import ClassVar


class _FakeThread:
    instances: ClassVar[list["_FakeThread"]] = []

    def __init__(self, *, target, name=None, daemon=None):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        self.join_calls = []
        self.instances.append(self)

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        self.started = False


def _import_main(monkeypatch):
    thread_calls = []
    signal_calls = []

    def fake_thread(*args, **kwargs):
        thread_calls.append((args, kwargs))
        return _FakeThread(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", fake_thread)
    monkeypatch.setattr(
        signal,
        "signal",
        lambda *args: signal_calls.append(args),
    )
    sys.modules.pop("main", None)
    module = importlib.import_module("main")
    return module, thread_calls, signal_calls


def test_import_does_not_start_threads_or_register_sigterm(monkeypatch):
    main, thread_calls, signal_calls = _import_main(monkeypatch)

    assert thread_calls == []
    assert signal_calls == []
    assert main._button_monitor_thread is None

    sys.modules.pop("main", None)


def test_init_runtime_registers_sigterm_and_starts_one_monitor(monkeypatch, tmp_path):
    _FakeThread.instances = []
    main, thread_calls, signal_calls = _import_main(monkeypatch)

    paths = SimpleNamespace(
        screenshot_dir=tmp_path / "screenshots",
        current_screenshot_dir=tmp_path / "current",
        archive_base=tmp_path / "archive",
    )

    class FakeDisplay:
        def register_skip_event(self, event):
            self.skip_event = event

        def set_button_callback(self, callback):
            self.button_callback = callback

    monkeypatch.setattr(main, "initialise_runtime_probes", lambda: None)
    monkeypatch.setattr(main, "_start_config_ui", lambda: None)
    monkeypatch.setattr(main, "resolve_storage_paths", lambda **kwargs: paths)
    monkeypatch.setattr(main, "Display", FakeDisplay)
    monkeypatch.setattr(main, "clear_update_indicator", lambda display: None)
    monkeypatch.setattr(main, "refresh_schedule_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(main, "_refresh_startup_critical_feeds", lambda: None)
    monkeypatch.setattr(main, "ENABLE_SCREENSHOTS", False)
    monkeypatch.setattr(main, "ENABLE_VIDEO", False)
    monkeypatch.setattr(main, "ENABLE_WIFI_MONITOR", False)

    main.init_runtime()
    main.init_runtime()

    monitors = [
        thread
        for thread in _FakeThread.instances
        if thread.name == "control-button-monitor"
    ]
    assert signal_calls == [(signal.SIGTERM, main._handle_sigterm)]
    assert len(monitors) == 1
    assert monitors[0].started is True
    assert len(thread_calls) == 3  # button, background refresh, startup refresh

    main._handle_sigterm(signal.SIGTERM, None)
    assert main._shutdown_event.is_set()
    assert main._manual_skip_event.is_set()
    assert main._display_cleared.is_set()

    main._finalize_shutdown()
    assert monitors[0].join_calls == [1.0]
    assert main._button_monitor_thread is None
    assert main._shutdown_complete.is_set()

    sys.modules.pop("main", None)
