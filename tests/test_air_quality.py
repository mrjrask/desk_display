from services.air_quality import (
    advisory_for,
    iqair_url_for_coordinates,
    normalize_iqair_page,
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


def test_trend_for_handles_steady_and_improving_forecasts():
    assert trend_for({"hourly": {"us_aqi": [42, 43, 44, 43]}}) == "Steady next 4h"
    assert trend_for({"hourly": {"us_aqi": [90, 84, 80, 75]}}) == "Improving next 4h"


def test_advisory_for_is_outdoor_activity_focused():
    assert advisory_for("Hazardous", None) == "Avoid outdoor activity."
    assert advisory_for("Good", "High") == "Okay outside; pollen high."


def test_iqair_url_for_configured_coordinates():
    assert iqair_url_for_coordinates(42.1373, -87.8446) == "https://www.iqair.com/air-quality/usa/illinois/northbrook"
    assert iqair_url_for_coordinates(41.9037, -87.6357) == "https://www.iqair.com/air-quality/usa/illinois/chicago"
    assert iqair_url_for_coordinates(41.0, -87.0) is None


def test_normalize_iqair_page_extracts_current_conditions_and_forecast_trend():
    report = normalize_iqair_page(
        """
        <h1>Air quality in Northbrook</h1>
        <div>437</div><div>US AQI⁺</div><div>Hazardous</div>
        <div>Main pollutant:</div><div>PM2.5</div><div>294 µg/m³</div>
        <h2>Hourly forecast</h2>
        <div>Now</div><div>437</div><div>25°</div>
        <div>10:00</div><div>407</div><div>27°</div>
        <div>11:00</div><div>376</div><div>28°</div>
        <div>12:00</div><div>346</div><div>29°</div>
        """
    )

    assert report is not None
    assert report.aqi_value == 437
    assert report.aqi_category == "Hazardous"
    assert report.primary_pollutant == "PM2.5"
    assert report.pm2_5_value == 294.0
    assert report.pollutant_breakdown == (("PM2.5", 294.0),)
    assert report.advisory_text == "Avoid outdoor activity."
    assert report.trend_text == "Improving next 4h"


def test_fetch_air_quality_prefers_iqair_for_configured_coordinates(monkeypatch):
    from services import air_quality

    calls = []

    def fake_fetch_iqair(url, *, timeout=8.0):
        calls.append((url, timeout))
        return air_quality.AirQualityReport(55, "Moderate", "PM2.5")

    monkeypatch.setattr(air_quality, "fetch_iqair_air_quality", fake_fetch_iqair)

    report = air_quality.fetch_air_quality(42.1373, -87.8446, timeout=3.0)

    assert report is not None
    assert report.aqi_value == 55
    assert calls == [("https://www.iqair.com/air-quality/usa/illinois/northbrook", 3.0)]
