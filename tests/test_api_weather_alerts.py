import scripts.test_api_connections as api_checks


def test_check_weather_alerts_reports_no_alerts(monkeypatch):
    monkeypatch.setattr(
        api_checks,
        "fetch_weather",
        lambda: {
            "current": {"temp": 72},
            "alerts": [],
            "source": "Apple WeatherKit",
        },
    )

    status, detail = api_checks.check_weather_alerts()

    assert status == "ok"
    assert detail == "no active weather alerts reported (source=Apple WeatherKit, alerts=0)"


def test_check_weather_alerts_reports_active_alert(monkeypatch):
    monkeypatch.setattr(
        api_checks,
        "fetch_weather",
        lambda: {
            "current": {"temp": 72},
            "alerts": [
                {
                    "event": "Flood Advisory",
                    "description": "Minor flooding is possible.",
                },
                {
                    "event": "Tornado Warning",
                    "description": "Move to an interior room now.",
                },
            ],
            "source": "OpenWeatherMap",
        },
    )

    status, detail = api_checks.check_weather_alerts()

    assert status == "ok"
    assert detail == (
        "active warning alert reported (source=OpenWeatherMap, alerts=2): "
        "Tornado Warning: Move to an interior room now."
    )


def test_check_weather_alerts_fails_without_weather_payload(monkeypatch):
    monkeypatch.setattr(api_checks, "fetch_weather", lambda: None)

    status, detail = api_checks.check_weather_alerts()

    assert status == "fail"
    assert detail == "fetch_weather returned empty payload"


def test_weather_alerts_check_is_registered_after_weather_helper():
    check_names = [check.name for check in api_checks.CHECKS]

    weather_index = check_names.index("weather (weatherkit/owm via app helper)")

    assert check_names[weather_index + 1] == "weather alerts"
