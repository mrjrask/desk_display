"""Tests for asynchronous startup refresh behavior."""

import importlib
import sys
import threading
import time

from schedule import build_scheduler
from screens.registry import ScreenDefinition


class _FakeThread:
    """Stand-in for ``threading.Thread`` used to observe runtime workers.

    The signature mirrors ``threading.Thread`` so production code can pass any
    of the real constructor arguments (``name``, ``args``, ...) without this
    double having to be updated in lockstep.
    """

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        self.group = group
        self.target = target
        self.name = name
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started


def _load_main():
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def test_init_runtime_starts_startup_refresh_thread(monkeypatch, tmp_path):
    main = _load_main()

    started_targets = []

    def _thread_factory(*args, **kwargs):
        thread = _FakeThread(*args, **kwargs)
        target = thread.target

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
        "screenshot_dir": tmp_path / "screenshots",
        "current_screenshot_dir": tmp_path / "screenshots" / "current",
        "archive_base": tmp_path / "archive",
    })())
    monkeypatch.setattr(main, "initialise_runtime_probes", lambda: None)
    monkeypatch.setattr(main, "refresh_schedule_if_needed", lambda force=False: None)
    monkeypatch.setattr(main, "_refresh_startup_critical_feeds", lambda: None)
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


def test_startup_feed_preview_does_not_consume_scheduler_hydration(monkeypatch):
    main = _load_main()
    scheduler = build_scheduler({"screens": {"weather1": 1, "date": 1}})
    registry = {
        screen_id: ScreenDefinition(id=screen_id, render=lambda: None)
        for screen_id in ("weather1", "date")
    }

    monkeypatch.setattr(main, "screen_scheduler", scheduler)
    monkeypatch.setattr(main, "_requested_data_feeds", lambda: {"weather"})
    main._last_feed_refresh.clear()

    assert main._scheduled_startup_feed_order() == ["weather"]
    assert scheduler.next_available(registry).id == "weather1"
    assert scheduler.next_available(registry).id == "date"
    assert scheduler._pass_number == 0


def test_routine_refresh_preserves_scheduler_and_pending_hydration(monkeypatch):
    main = _load_main()
    config = {"screens": {"date": 1, "inside": 1}}
    monkeypatch.setattr(main, "_active_config_path", lambda: "/tmp/screens.json")
    monkeypatch.setattr(main.os.path, "getmtime", lambda _path: 10.0)
    monkeypatch.setattr(main, "load_schedule_config", lambda _path: config)

    main.screen_scheduler = None
    main._screen_config_mtime = None
    main._screen_config_path = None
    main.refresh_schedule_if_needed()
    scheduler = main.screen_scheduler
    registry = {
        screen_id: ScreenDefinition(id=screen_id, render=lambda: None)
        for screen_id in ("date", "inside")
    }

    assert scheduler.next_available(registry).id == "date"
    main._bump_registry_cache_nonce()
    main.refresh_schedule_if_needed()

    assert main.screen_scheduler is scheduler
    assert scheduler.next_available(registry).id == "inside"
    assert scheduler._pass_number == 0


def test_config_reload_constructs_scheduler_with_new_hydration(monkeypatch):
    main = _load_main()
    config_mtime = {"value": 10.0}
    config = {"screens": {"date": 1, "inside": 1}}
    monkeypatch.setattr(main, "_active_config_path", lambda: "/tmp/screens.json")
    monkeypatch.setattr(main.os.path, "getmtime", lambda _path: config_mtime["value"])
    monkeypatch.setattr(main, "load_schedule_config", lambda _path: config)
    registry = {
        screen_id: ScreenDefinition(id=screen_id, render=lambda: None)
        for screen_id in ("date", "inside")
    }

    main.screen_scheduler = None
    main._screen_config_mtime = None
    main._screen_config_path = None
    main.refresh_schedule_if_needed()
    original = main.screen_scheduler
    assert original.next_available(registry).id == "date"

    config_mtime["value"] = 11.0
    main.refresh_schedule_if_needed()

    replacement = main.screen_scheduler
    assert replacement is not original
    assert replacement.next_available(registry).id == "date"
    assert replacement._pass_number == 0


def test_bases_are_not_rehydrated_until_saved_config_rebuilds_scheduler(monkeypatch):
    main = _load_main()
    config_mtime = {"value": 10.0}
    config = {"screens": {"date": 1, "inside": 1}}
    monkeypatch.setattr(main, "_active_config_path", lambda: "/tmp/screens.json")
    monkeypatch.setattr(main.os.path, "getmtime", lambda _path: config_mtime["value"])
    monkeypatch.setattr(main, "load_schedule_config", lambda _path: config)
    registry = {
        screen_id: ScreenDefinition(id=screen_id, render=lambda: None)
        for screen_id in ("date", "inside")
    }

    main.screen_scheduler = None
    main._screen_config_mtime = None
    main._screen_config_path = None
    main.refresh_schedule_if_needed()
    scheduler = main.screen_scheduler

    assert [scheduler.next_available(registry).id for _ in range(2)] == ["date", "inside"]
    assert scheduler.preview_scheduled_entries(2)[0].phase == "normal"

    main.refresh_schedule_if_needed()

    assert main.screen_scheduler is scheduler
    assert scheduler.preview_scheduled_entries(2)[0].phase == "normal"

    config_mtime["value"] += 1
    main.refresh_schedule_if_needed()

    rebuilt = main.screen_scheduler
    assert rebuilt is not scheduler
    preview = rebuilt.preview_scheduled_entries(2)
    assert [(entry.screen_id, entry.phase) for entry in preview] == [
        ("date", "startup"),
        ("inside", "startup"),
    ]


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


def test_startup_feeds_include_team_data_for_stable_first_schedule(monkeypatch):
    main = _load_main()

    monkeypatch.setattr(
        main,
        "_requested_data_feeds",
        lambda: {"weather", "scoreboards", "hawks", "cubs"},
    )

    assert main._startup_critical_feeds() == [
        "weather",
        "scoreboards",
        "cubs",
        "hawks",
    ]


def test_async_startup_skips_feeds_loaded_before_first_frame(monkeypatch):
    main = _load_main()

    monkeypatch.setattr(main, "_requested_data_feeds", lambda: {"weather", "hawks"})
    main._last_feed_refresh.clear()
    main._last_feed_refresh["weather"] = time.monotonic()

    assert main._scheduled_startup_feed_order() == ["hawks"]


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


def test_refresh_startup_critical_feeds_returns_after_timeout_for_hung_worker(monkeypatch):
    main = _load_main()

    release = threading.Event()
    monkeypatch.setattr(main, "_wifi_outage_active", False)
    monkeypatch.setattr(main, "_requested_data_feeds", lambda: {"weather"})
    monkeypatch.setattr(main, "_STARTUP_CRITICAL_FEED_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setitem(main._FEED_REFRESHERS, "weather", lambda: release.wait())

    started_at = time.monotonic()
    try:
        main._refresh_startup_critical_feeds()
        elapsed = time.monotonic() - started_at
        assert elapsed < 0.5
    finally:
        release.set()
