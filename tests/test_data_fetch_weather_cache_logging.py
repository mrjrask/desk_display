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
