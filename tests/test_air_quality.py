from services.air_quality import advisory_for, normalize_airnow


def test_normalize_airnow_uses_highest_pollutant_aqi_as_overall_aqi():
    report = normalize_airnow(
        [
            {"ParameterName": "OZONE", "AQI": 44, "Category": {"Name": "Good"}},
            {"ParameterName": "PM2.5", "AQI": 72, "Category": {"Name": "Moderate"}},
            {"ParameterName": "PM10", "AQI": 31, "Category": {"Name": "Good"}},
        ]
    )

    assert report is not None
    assert report.aqi_value == 72
    assert report.aqi_category == "Moderate"
    assert report.primary_pollutant == "PM2.5"
    assert report.us_aqi_pm2_5 == 72
    assert report.us_aqi_pm10 == 31
    assert report.us_aqi_ozone == 44
    assert report.pollutant_breakdown == (("Ozone", 44.0), ("PM2.5", 72.0), ("PM10", 31.0))
    assert report.advisory_text == "Okay for most outdoor plans."


def test_normalize_airnow_ignores_unknown_or_invalid_records():
    assert normalize_airnow([]) is None
    assert normalize_airnow(
        [{"ParameterName": "CO", "AQI": 10}, {"ParameterName": "PM2.5", "AQI": None}]
    ) is None


def test_advisory_for_is_outdoor_activity_focused():
    assert advisory_for("Hazardous", None) == "Avoid outdoor activity."
    assert advisory_for("Good", "High") == "Good for outdoor activity."


def test_fetch_air_quality_requests_airnow_current_observations(monkeypatch):
    from services import air_quality

    captured = {}

    def fake_request_json(url, *, params, timeout, quiet):
        captured.update(url=url, params=params, timeout=timeout, quiet=quiet)
        return [{"ParameterName": "PM2.5", "AQI": 20, "Category": {"Name": "Good"}}]

    monkeypatch.setattr(air_quality, "request_json", fake_request_json)

    report = air_quality.fetch_air_quality(
        42.1373, -87.8446, api_key="test-key", include_pollen=True, timeout=3.0
    )

    assert report is not None
    assert report.aqi_value == 20
    assert captured["url"] == air_quality.AIRNOW_CURRENT_OBSERVATION_URL
    assert captured["params"] == {
        "format": "application/json",
        "latitude": 42.1373,
        "longitude": -87.8446,
        "distance": 25,
        "API_KEY": "test-key",
    }
    assert captured["timeout"] == 3.0
    assert captured["quiet"] is True


def test_fetch_air_quality_does_not_request_without_api_key(monkeypatch):
    from services import air_quality

    monkeypatch.setattr(
        air_quality,
        "request_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )

    assert air_quality.fetch_air_quality(42.1373, -87.8446, api_key="") is None
