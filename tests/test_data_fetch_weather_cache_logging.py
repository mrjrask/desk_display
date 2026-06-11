import datetime
import logging

import data_fetch


def test_fetch_weather_logs_warning_when_returning_cached_weather(monkeypatch, caplog):
    cached_payload = {"current": {"temperature": 70}}
    cached_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=5)

    monkeypatch.setattr(data_fetch, "_weather_cache", cached_payload)
    monkeypatch.setattr(data_fetch, "_weather_cache_fetched_at", cached_at)

    caplog.set_level(logging.WARNING)
    returned = data_fetch.fetch_weather(force_refresh=False)

    assert returned == cached_payload
    assert "Using cached weather data last retrieved at" in caplog.text
    assert cached_at.isoformat() in caplog.text


def test_fetch_weather_logs_warning_when_returning_stale_cached_weather(monkeypatch, caplog):
    cached_payload = {"current": {"temperature": 68}}
    cached_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=3)

    monkeypatch.setattr(data_fetch, "_weather_cache", cached_payload)
    monkeypatch.setattr(data_fetch, "_weather_cache_fetched_at", cached_at)
    monkeypatch.setattr(data_fetch, "_fetch_weatherkit", lambda _now: None)
    monkeypatch.setattr(data_fetch, "_fetch_openweathermap", lambda _now: None)

    caplog.set_level(logging.WARNING)
    returned = data_fetch.fetch_weather(force_refresh=True)

    assert returned == cached_payload
    assert "Using stale cached weather data last retrieved at" in caplog.text
    assert cached_at.isoformat() in caplog.text


def test_fetch_weather_uses_openweathermap_without_weatherkit_private_key(monkeypatch, caplog):
    owm_payload = {"current": {"temperature": 71}}

    monkeypatch.setattr(data_fetch, "_weather_cache", None)
    monkeypatch.setattr(data_fetch, "_weather_cache_fetched_at", None)
    monkeypatch.setattr(data_fetch, "WEATHERKIT_TEAM_ID", "team-id")
    monkeypatch.setattr(data_fetch, "WEATHERKIT_KEY_ID", "key-id")
    monkeypatch.setattr(data_fetch, "WEATHERKIT_SERVICE_ID", "service-id")
    monkeypatch.setattr(data_fetch, "WEATHERKIT_PRIVATE_KEY", None)
    monkeypatch.setattr(data_fetch, "WEATHERKIT_KEY_PATH", None)
    monkeypatch.setattr(data_fetch, "OWM_API_KEY", "owm-key")
    monkeypatch.setattr(data_fetch, "_weatherkit_config_warning_logged", False)
    monkeypatch.setattr(
        data_fetch,
        "_fetch_weatherkit",
        lambda _now: (_ for _ in ()).throw(AssertionError("WeatherKit should be skipped")),
    )
    monkeypatch.setattr(data_fetch, "_fetch_openweathermap", lambda _now: owm_payload)

    caplog.set_level(logging.INFO)
    returned = data_fetch.fetch_weather(force_refresh=True)

    assert returned == owm_payload
    assert "WeatherKit not fully configured" in caplog.text
    assert "Using OpenWeatherMap fallback" in caplog.text
