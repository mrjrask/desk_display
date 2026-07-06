"""Tests for weather feed refresh wiring in main."""

import importlib
import sys


def _load_main():
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def test_requested_data_feeds_includes_weather_daily(monkeypatch):
    main = _load_main()
    main._requested_screen_ids = {"weather daily"}
    monkeypatch.setattr(main, "ENABLE_WEATHER", True)

    feeds = main._requested_data_feeds()

    assert "weather" in feeds


def test_requested_data_feeds_includes_weather_quad(monkeypatch):
    main = _load_main()
    main._requested_screen_ids = {"weather quad"}
    monkeypatch.setattr(main, "ENABLE_WEATHER", True)

    feeds = main._requested_data_feeds()

    assert "weather" in feeds


def test_requested_data_feeds_includes_weather_alert(monkeypatch):
    main = _load_main()
    main._requested_screen_ids = {"weather alert"}
    monkeypatch.setattr(main, "ENABLE_WEATHER", True)

    feeds = main._requested_data_feeds()

    assert "weather" in feeds

def test_refresh_weather_logs_warning_when_payload_missing(monkeypatch):
    main = _load_main()
    main.cache["weather"] = {"current": {}}

    monkeypatch.setattr(main.data_provider, "read_weather", lambda ttl_seconds: None)

    warnings = []
    monkeypatch.setattr(main.logging, "warning", lambda msg, *args: warnings.append(msg % args if args else msg))

    main._refresh_weather()

    assert main.cache["weather"] is None
    assert any("Weather feed returned no data" in msg for msg in warnings)
