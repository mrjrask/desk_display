import pytest

from data_fetch import _normalise_weatherkit_response


def _iso(dt: str) -> str:
    """Helper to keep sample payload readable."""
    return f"2024-01-01T{dt}Z"


def test_weatherkit_measurements_extract_wind_gust_and_speed():
    data = {
        "currentWeather": {
            "temperature": 10,
            "temperatureApparent": 8,
            "conditionCode": "Clear",
            "windSpeed": {"value": 4.2},
            "windGust": {"value": 9.6},
            "windDirection": 180,
            "humidity": 0.5,
            "pressure": 1012,
            "uvIndex": 3,
            "asOf": _iso("12:00:00"),
            "cloudCover": 0.1,
        },
        "forecastDaily": {
            "days": [
                {
                    "sunrise": _iso("12:00:00"),
                    "sunset": _iso("22:00:00"),
                    "temperatureMax": 20,
                    "temperatureMin": 5,
                    "precipitationChance": 0,
                    "conditionCode": "Clear",
                    "forecastStart": _iso("00:00:00"),
                }
            ]
        },
        "forecastHourly": {
            "hours": [
                {
                    "forecastStart": _iso("12:00:00"),
                    "temperature": 10,
                    "temperatureApparent": 9,
                    "precipitationChance": 0,
                    "windSpeed": {"value": 5.1},
                    "windGust": {"value": 11.2},
                    "windDirection": 200,
                    "uvIndex": 2,
                    "conditionCode": "Clear",
                }
            ]
        },
        "weatherAlerts": {"alerts": []},
    }

    normalized = _normalise_weatherkit_response(data)

    assert normalized is not None
    assert normalized["current"]["wind_speed"] == pytest.approx(4.2)
    assert normalized["current"]["wind_gust"] == pytest.approx(9.6)

    hourly = normalized["hourly"][0]
    assert hourly["wind_speed"] == pytest.approx(5.1)
    assert hourly["wind_gust"] == pytest.approx(11.2)
