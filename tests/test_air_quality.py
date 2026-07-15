from services.air_quality import advisory_for, normalize_open_meteo, trend_for


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
