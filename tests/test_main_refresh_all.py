"""Tests for concurrent data-feed refreshes."""

import importlib
import threading
import time


def _load_main():
    return importlib.reload(importlib.import_module("main"))


def _prepare_refresh_all(monkeypatch, main, feeds):
    monkeypatch.setattr(main, "_wifi_outage_active", False)
    monkeypatch.setattr(main, "_requested_data_feeds", lambda: set(feeds))
    monkeypatch.setattr(main, "_FEED_REFRESH_INTERVALS", {feed: 0 for feed in feeds})
    main._last_feed_refresh.clear()


def test_refresh_all_submits_due_feeds_independently(monkeypatch):
    main = _load_main()
    feeds = {"weather", "hawks"}
    _prepare_refresh_all(monkeypatch, main, feeds)
    monkeypatch.setattr(main, "DATA_REFRESH_MAX_WORKERS", 2)

    first_started = threading.Event()
    second_started = threading.Event()
    calls = []

    def refresh_weather():
        calls.append("weather-start")
        first_started.set()
        assert second_started.wait(1.0)
        calls.append("weather-done")

    def refresh_hawks():
        assert first_started.wait(1.0)
        calls.append("hawks-start")
        second_started.set()
        calls.append("hawks-done")

    monkeypatch.setattr(
        main,
        "_FEED_REFRESHERS",
        {"weather": refresh_weather, "hawks": refresh_hawks},
    )

    main.refresh_all()

    assert "weather-done" in calls
    assert "hawks-done" in calls
    assert calls.index("hawks-start") < calls.index("weather-done")
    assert set(main._last_feed_refresh) == feeds


def test_refresh_all_failing_feed_does_not_block_successes(monkeypatch, caplog):
    main = _load_main()
    feeds = {"weather", "hawks", "bears"}
    _prepare_refresh_all(monkeypatch, main, feeds)
    monkeypatch.setattr(main, "DATA_REFRESH_MAX_WORKERS", 3)

    calls = []
    nonce_bumps = []

    def fail_weather():
        calls.append("weather")
        raise RuntimeError("boom")

    def refresh_hawks():
        time.sleep(0.01)
        calls.append("hawks")

    def refresh_bears():
        calls.append("bears")

    monkeypatch.setattr(
        main,
        "_FEED_REFRESHERS",
        {"weather": fail_weather, "hawks": refresh_hawks, "bears": refresh_bears},
    )
    monkeypatch.setattr(main, "_bump_registry_cache_nonce", lambda: nonce_bumps.append("bump"))

    main.refresh_all()

    assert set(calls) == feeds
    assert set(main._last_feed_refresh) == {"hawks", "bears"}
    assert "weather" not in main._last_feed_refresh
    assert nonce_bumps == ["bump"]
    assert "Failed to refresh weather feed: boom" in caplog.text
