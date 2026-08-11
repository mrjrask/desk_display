import datetime
import logging

import data_fetch


def test_fetch_weather_logs_warning_when_returning_cached_weather(monkeypatch, caplog):
    cached_payload = {"current": {"temperature": 70}}
    cached_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=5)

    monkeypatch.setattr(data_fetch, "_weather_cache", cached_payload)
    monkeypatch.setattr(data_fetch, "_weather_cache_fetched_at", cached_at)
    monkeypatch.setattr(data_fetch, "_weather_cache_source", "Apple WeatherKit")

    caplog.set_level(logging.WARNING)
    returned = data_fetch.fetch_weather(force_refresh=False)

    assert returned == cached_payload
    assert "Using cached weather data from Apple WeatherKit last retrieved at" in caplog.text
    assert cached_at.isoformat() in caplog.text


def test_fetch_weather_logs_warning_when_returning_stale_cached_weather(monkeypatch, caplog):
    cached_payload = {"current": {"temperature": 68}}
    cached_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=3)

    monkeypatch.setattr(data_fetch, "_weather_cache", cached_payload)
    monkeypatch.setattr(data_fetch, "_weather_cache_fetched_at", cached_at)
    monkeypatch.setattr(data_fetch, "_weather_cache_source", "OpenWeatherMap")
    monkeypatch.setattr(data_fetch, "_fetch_weatherkit", lambda _now: None)
    monkeypatch.setattr(data_fetch, "_fetch_openweathermap", lambda _now: None)

    caplog.set_level(logging.WARNING)
    returned = data_fetch.fetch_weather(force_refresh=True)

    assert returned == cached_payload
    assert "Using stale cached weather data from OpenWeatherMap last retrieved at" in caplog.text
    assert cached_at.isoformat() in caplog.text


def test_fetch_weather_uses_openweathermap_without_weatherkit_private_key(monkeypatch, caplog):
    owm_payload = {"current": {"temperature": 71}, "source": "OpenWeatherMap"}

    monkeypatch.setattr(data_fetch, "_weather_cache", None)
    monkeypatch.setattr(data_fetch, "_weather_cache_fetched_at", None)
    monkeypatch.setattr(data_fetch, "_weather_cache_source", None)
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


def test_fetch_weather_logs_successful_weather_source(monkeypatch, caplog):
    weatherkit_payload = {"current": {"temperature": 72}, "source": "Apple WeatherKit"}

    monkeypatch.setattr(data_fetch, "_weather_cache", None)
    monkeypatch.setattr(data_fetch, "_weather_cache_fetched_at", None)
    monkeypatch.setattr(data_fetch, "_weather_cache_source", None)
    monkeypatch.setattr(data_fetch, "_weatherkit_configured", lambda: True)
    monkeypatch.setattr(data_fetch, "_fetch_weatherkit", lambda _now: weatherkit_payload)
    monkeypatch.setattr(
        data_fetch,
        "_fetch_openweathermap",
        lambda _now: (_ for _ in ()).throw(AssertionError("OpenWeatherMap should not be used")),
    )

    caplog.set_level(logging.INFO)
    returned = data_fetch.fetch_weather(force_refresh=True)

    assert returned == weatherkit_payload
    assert data_fetch._weather_cache_source == "Apple WeatherKit"
    assert "Fetched weather data from Apple WeatherKit" in caplog.text


def test_normalized_weather_payloads_include_source():
    weatherkit_payload = data_fetch._normalise_weatherkit_response({})
    owm_payload = data_fetch._normalise_openweathermap_response({})

    assert weatherkit_payload["source"] == "Apple WeatherKit"
    assert owm_payload["source"] == "OpenWeatherMap"
