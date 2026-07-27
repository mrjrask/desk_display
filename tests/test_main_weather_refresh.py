"""Tests for weather feed refresh wiring in main."""

import importlib
import sys

from services.air_quality import AirQualityReport


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


def test_requested_data_feeds_includes_air_quality(monkeypatch):
    main = _load_main()
    main._requested_screen_ids = {"air quality"}
    monkeypatch.setattr(main.config, "ENABLE_AIR_QUALITY", True)

    assert "air_quality" in main._requested_data_feeds()


def test_requested_data_feeds_includes_air_quality_for_weather_quad(monkeypatch):
    main = _load_main()
    main._requested_screen_ids = {"weather quad"}
    monkeypatch.setattr(main.config, "ENABLE_AIR_QUALITY", True)

    assert "air_quality" in main._requested_data_feeds()


def test_refresh_air_quality_uses_configured_aqi_coordinates(monkeypatch):
    main = _load_main()
    report = object()
    captured = {}
    monkeypatch.setattr(main.config, "ENABLE_AIR_QUALITY", True)
    monkeypatch.setattr(main.config, "AIR_QUALITY_LATITUDE", 34.0522)
    monkeypatch.setattr(main.config, "AIR_QUALITY_LONGITUDE", -118.2437)
    monkeypatch.setattr(main.config, "AIRNOW_API_KEY", "test-key")
    monkeypatch.setattr(main.config, "AIR_QUALITY_ENABLE_POLLEN", False)

    def fake_fetch(latitude, longitude, *, api_key, include_pollen):
        captured.update(
            latitude=latitude,
            longitude=longitude,
            api_key=api_key,
            include_pollen=include_pollen,
        )
        return report

    monkeypatch.setattr(main, "fetch_air_quality", fake_fetch)

    main._refresh_air_quality()

    assert captured == {
        "latitude": 34.0522,
        "longitude": -118.2437,
        "api_key": "test-key",
        "include_pollen": False,
    }
    assert main.cache["air_quality"] is report


def test_refresh_air_quality_retains_component_history_for_charts(monkeypatch, tmp_path):
    main = _load_main()
    report = AirQualityReport(
        72,
        "Moderate",
        "PM2.5",
        us_aqi_pm2_5=72,
        us_aqi_pm10=31,
        us_aqi_ozone=44,
    )
    monkeypatch.setattr(main.config, "ENABLE_AIR_QUALITY", True)
    monkeypatch.setattr(main.config, "AIR_QUALITY_LATITUDE", 34.0522)
    monkeypatch.setattr(main.config, "AIR_QUALITY_LONGITUDE", -118.2437)
    monkeypatch.setattr(main.config, "AIRNOW_API_KEY", "test-key")
    monkeypatch.setattr(main, "fetch_air_quality", lambda *args, **kwargs: report)
    monkeypatch.setattr(main.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(main, "_AIR_QUALITY_HISTORY_PATH", str(tmp_path / "aq.json"))

    main._refresh_air_quality()

    assert main.cache["air_quality"].component_history == ((1_000.0, 72, 31, 44),)


def test_refresh_air_quality_restores_persisted_chart_history(monkeypatch, tmp_path):
    main = _load_main()
    report = AirQualityReport(
        72,
        "Moderate",
        "PM2.5",
        us_aqi_pm2_5=72,
        us_aqi_pm10=31,
        us_aqi_ozone=44,
    )
    history_path = tmp_path / "aq.json"
    history_path.write_text('{"history": [[300.0, 65, 28, 40]]}', encoding="utf-8")
    monkeypatch.setattr(main.config, "ENABLE_AIR_QUALITY", True)
    monkeypatch.setattr(main.config, "AIR_QUALITY_LATITUDE", 34.0522)
    monkeypatch.setattr(main.config, "AIR_QUALITY_LONGITUDE", -118.2437)
    monkeypatch.setattr(main.config, "AIRNOW_API_KEY", "test-key")
    monkeypatch.setattr(main, "fetch_air_quality", lambda *args, **kwargs: report)
    monkeypatch.setattr(main.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(main, "_AIR_QUALITY_HISTORY_PATH", str(history_path))

    main._refresh_air_quality()

    assert main.cache["air_quality"].component_history == (
        (300.0, 65, 28, 40),
        (1_000.0, 72, 31, 44),
    )
    persisted = main._load_air_quality_history(1_000.0)
    assert persisted == [(300.0, 65, 28, 40), (1_000.0, 72, 31, 44)]


def test_refresh_air_quality_keeps_last_report_when_request_fails(monkeypatch):
    main = _load_main()
    previous_report = object()
    main.cache["air_quality"] = previous_report
    monkeypatch.setattr(main.config, "ENABLE_AIR_QUALITY", True)
    monkeypatch.setattr(main.config, "AIR_QUALITY_LATITUDE", 34.0522)
    monkeypatch.setattr(main.config, "AIR_QUALITY_LONGITUDE", -118.2437)
    monkeypatch.setattr(main, "fetch_air_quality", lambda *args, **kwargs: None)

    main._refresh_air_quality()

    assert main.cache["air_quality"] is previous_report
