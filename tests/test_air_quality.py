from screens.draw_air_quality import _chart_layout, _component_history_points, _display_metrics
from services.air_quality import AirQualityReport, advisory_for, normalize_airnow


def test_air_quality_screen_shows_only_compact_metrics():
    report = AirQualityReport(
        aqi_value=72,
        aqi_category="Moderate",
        primary_pollutant="PM2.5",
        pollen_level="High",
        advisory_text="Okay for most outdoor plans.",
        pm2_5_value=18.4,
        us_aqi_pm2_5=72,
        us_aqi_pm10=31,
        us_aqi_ozone=44,
        trend_text="Rising",
    )

    assert _display_metrics(report) == [
        ("Top Pollutant", "PM2.5"),
        ("PM2.5", "72"),
        ("PM10", "31"),
        ("Ozone", "44"),
        ("Advice", "Okay for most outdoor plans."),
    ]


def test_air_quality_component_charts_use_history_and_available_width():
    report = AirQualityReport(
        aqi_value=72,
        aqi_category="Moderate",
        primary_pollutant="PM2.5",
        component_history=((100.0, 65, 30, 42), (200.0, 72, None, 44)),
    )

    assert _component_history_points(report, 0) == [(100.0, 65.0), (200.0, 72.0)]
    assert _component_history_points(report, 1) == [(100.0, 30.0)]
    assert _chart_layout(
        [150, 175], value_x=110, right_edge=310, chart_gap=6, chart_min_w=40
    ) == (181, 129, True)


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

    captured = []

    def fake_request_json(url, *, params, timeout, quiet):
        captured.append((url, params, timeout, quiet))
        return [{"ParameterName": "PM2.5", "AQI": 20, "Category": {"Name": "Good"}}]

    monkeypatch.setattr(air_quality, "request_json", fake_request_json)

    report = air_quality.fetch_air_quality(
        42.1373, -87.8446, api_key="test-key", include_pollen=True, timeout=3.0
    )

    assert report is not None
    assert report.aqi_value == 20
    assert captured[0][0] == air_quality.AIRNOW_CURRENT_OBSERVATION_URL
    assert captured[0][1] == {
        "format": "application/json",
        "latitude": 42.1373,
        "longitude": -87.8446,
        "distance": 25,
        "API_KEY": "test-key",
    }
    assert captured[0][2] == 3.0
    assert captured[0][3] is True


def test_fetch_air_quality_supplements_missing_airnow_components(monkeypatch):
    from services import air_quality

    requests = []

    def fake_request_json(url, *, params, timeout, quiet):
        requests.append((url, params, timeout, quiet))
        if url == air_quality.AIRNOW_CURRENT_OBSERVATION_URL:
            return [{"ParameterName": "PM2.5", "AQI": 411, "Category": {"Name": "Hazardous"}}]
        assert url == air_quality.OPEN_METEO_AIR_QUALITY_URL
        return {"current": {"us_aqi_pm2_5": 400, "us_aqi_pm10": 78, "us_aqi_ozone": 52}}

    monkeypatch.setattr(air_quality, "request_json", fake_request_json)

    report = air_quality.fetch_air_quality(42.1373, -87.8446, api_key="test-key", timeout=3.0)

    assert report is not None
    assert report.aqi_value == 411
    assert report.aqi_category == "Hazardous"
    assert report.us_aqi_pm2_5 == 411
    assert report.us_aqi_pm10 == 78
    assert report.us_aqi_ozone == 52
    assert requests[1] == (
        air_quality.OPEN_METEO_AIR_QUALITY_URL,
        {
            "latitude": 42.1373,
            "longitude": -87.8446,
            "current": "us_aqi_pm2_5,us_aqi_pm10,us_aqi_ozone",
        },
        3.0,
        True,
    )


def test_fetch_air_quality_skips_component_fallback_when_airnow_is_complete(monkeypatch):
    from services import air_quality

    def fake_request_json(url, **_kwargs):
        assert url == air_quality.AIRNOW_CURRENT_OBSERVATION_URL
        return [
            {"ParameterName": "PM2.5", "AQI": 72},
            {"ParameterName": "PM10", "AQI": 31},
            {"ParameterName": "OZONE", "AQI": 44},
        ]

    monkeypatch.setattr(air_quality, "request_json", fake_request_json)

    report = air_quality.fetch_air_quality(42.1373, -87.8446, api_key="test-key")

    assert report is not None
    assert report.us_aqi_pm10 == 31
    assert report.us_aqi_ozone == 44


def test_fetch_air_quality_does_not_request_without_api_key(monkeypatch):
    from services import air_quality

    monkeypatch.setattr(
        air_quality,
        "request_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )

    assert air_quality.fetch_air_quality(42.1373, -87.8446, api_key="") is None
