"""Tests for asynchronous startup refresh behavior."""

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


def test_init_runtime_starts_startup_refresh_thread(monkeypatch):
    main = _load_main()

    started_targets = []

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
    monkeypatch.setattr(main.threading, "Thread", _thread_factory)

    main._runtime_initialized = False
    main._background_refresh_thread = None
    main._startup_refresh_thread = None

    main.init_runtime()

    assert main._startup_refresh_thread is not None
    assert main._startup_refresh_thread.started is True
    assert main._startup_refresh in started_targets


def test_scheduled_startup_feed_order_prioritizes_upcoming_screens(monkeypatch):
    main = _load_main()

    class _Scheduler:
        def preview_scheduled_ids(self, limit):
            return ["weather1", "hawks next", "date"]

    monkeypatch.setattr(main, "screen_scheduler", _Scheduler())
    monkeypatch.setattr(main, "_requested_data_feeds", lambda: {"weather", "hawks", "bears"})

    ordered = main._scheduled_startup_feed_order()

    assert ordered[:2] == ["weather", "hawks"]
    assert ordered[-1] == "bears"


def test_startup_refresh_runs_first_wave_before_background(monkeypatch):
    main = _load_main()

    monkeypatch.setattr(main, "_scheduled_startup_feed_order", lambda limit=4: ["weather", "hawks", "bears"])
    monkeypatch.setattr(main._shutdown_event, "is_set", lambda: False)

    calls = []

    def _refresh(feeds):
        calls.append(list(feeds))

    monkeypatch.setattr(main, "_refresh_feeds_in_order", _refresh)

    main._startup_refresh()

    assert calls == [["weather", "hawks"], ["bears"]]


def test_air_quality_is_a_startup_critical_feed():
    main = _load_main()

    assert "air_quality" in main._STARTUP_CRITICAL_FEEDS


def test_startup_critical_feeds_includes_air_quality_when_requested(monkeypatch):
    main = _load_main()

    monkeypatch.setattr(main, "_requested_data_feeds", lambda: {"weather", "scoreboards", "air_quality"})

    assert main._startup_critical_feeds() == ["weather", "scoreboards", "air_quality"]


def test_refresh_startup_critical_feeds_fetches_air_quality_before_main_loop(monkeypatch):
    """AQI must be primed alongside weather/scoreboards so the "air quality"
    screen's registry entry never falls back to a synchronous, blocking
    AirNow/Open-Meteo fetch the first time it renders."""

    main = _load_main()

    monkeypatch.setattr(main, "_wifi_outage_active", False)
    monkeypatch.setattr(main, "_requested_data_feeds", lambda: {"weather", "scoreboards", "air_quality"})
    main._last_feed_refresh.clear()

    called = []
    monkeypatch.setitem(main._FEED_REFRESHERS, "weather", lambda: called.append("weather"))
    monkeypatch.setitem(main._FEED_REFRESHERS, "scoreboards", lambda: called.append("scoreboards"))
    monkeypatch.setitem(main._FEED_REFRESHERS, "air_quality", lambda: called.append("air_quality"))

    main._refresh_startup_critical_feeds()

    assert set(called) == {"weather", "scoreboards", "air_quality"}
    assert "air_quality" in main._last_feed_refresh
