from services.air_quality import (
    advisory_for,
    normalize_open_meteo,
    trend_for,
)


def test_normalize_open_meteo_includes_requested_air_quality_metrics():
    report = normalize_open_meteo(
        {
            "current": {
                "us_aqi": 72,
                "pm2_5": 12.4,
                "pm10": 20.0,
                "ozone": 60.0,
                "grass_pollen": 15.0,
            },
            "hourly": {
                "us_aqi": [72, 75, 78, 81, 85, 89],
            },
        }
    )

    assert report.aqi_value == 72
    assert report.aqi_category == "Moderate"
    assert report.primary_pollutant == "PM2.5"
    assert report.pm2_5_value == 12.4
    assert report.pollen_level == "Moderate"
    assert report.advisory_text == "Okay for most outdoor plans."
    assert report.trend_text == "Rising next 6h"


def test_normalize_open_meteo_categories_match_the_rounded_displayed_aqi():
    report = normalize_open_meteo({"current": {"us_aqi": 50.6}})

    assert report.aqi_value == 51
    assert report.aqi_category == "Moderate"


def test_trend_for_handles_steady_and_improving_forecasts():
    assert trend_for({"hourly": {"us_aqi": [42, 43, 44, 43]}}) == "Steady next 4h"
    assert trend_for({"hourly": {"us_aqi": [90, 84, 80, 75]}}) == "Improving next 4h"


def test_advisory_for_is_outdoor_activity_focused():
    assert advisory_for("Hazardous", None) == "Avoid outdoor activity."
    assert advisory_for("Good", "High") == "Okay outside; pollen high."


def test_fetch_air_quality_uses_open_meteo_for_all_coordinates(monkeypatch):
    from services import air_quality

    calls: list[tuple[float, float, bool, float]] = []

    def fake_fetch_open_meteo(latitude, longitude, *, include_pollen, timeout):
        calls.append((latitude, longitude, include_pollen, timeout))
        return air_quality.AirQualityReport(55, "Moderate", "PM2.5")

    monkeypatch.setattr(air_quality, "_fetch_open_meteo_air_quality", fake_fetch_open_meteo)

    report = air_quality.fetch_air_quality(41.9037, -87.6357, include_pollen=False, timeout=3.0)

    assert report is not None
    assert report.aqi_value == 55
    assert calls == [(41.9037, -87.6357, False, 3.0)]


def test_fetch_open_meteo_requests_current_and_hourly_metrics(monkeypatch):
    from services import air_quality

    captured = {}

    def fake_request_json(url, *, params, timeout, quiet):
        captured.update(url=url, params=params, timeout=timeout, quiet=quiet)
        return {"current": {"us_aqi": 20}, "hourly": {"us_aqi": [20, 21]}}

    monkeypatch.setattr(air_quality, "request_json", fake_request_json)

    report = air_quality._fetch_open_meteo_air_quality(
        42.1373, -87.8446, include_pollen=True, timeout=3.0
    )

    assert report is not None
    assert report.aqi_value == 20
    assert captured["url"] == air_quality.OPEN_METEO_AIR_QUALITY_URL
    assert captured["params"]["latitude"] == 42.1373
    assert captured["params"]["longitude"] == -87.8446
    assert "us_aqi" in captured["params"]["current"]
    assert "pm2_5" in captured["params"]["hourly"]
    assert "grass_pollen" in captured["params"]["current"]
    assert captured["timeout"] == 3.0
    assert captured["quiet"] is True
