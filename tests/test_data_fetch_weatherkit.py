import pytest

from data_fetch import _fetch_weatherkit, _normalise_weatherkit_response


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
            "precipitationIntensity": {"value": 2.54},
            "visibility": {"value": 16093.44},
        },
        "forecastDaily": {
            "days": [
                {
                    "sunrise": _iso("12:00:00"),
                    "sunset": _iso("22:00:00"),
                    "temperatureMax": 20,
                    "temperatureMin": 5,
                    "precipitationChance": 0,
                    "windSpeed": {"value": 6.8},
                    "windDirection": 225,
                    "uvIndex": 5,
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
    assert normalized["current"]["precipitation_intensity"] == pytest.approx(2.54)
    assert normalized["current"]["visibility"] == pytest.approx(16093.44)

    hourly = normalized["hourly"][0]
    assert hourly["wind_speed"] == pytest.approx(5.1)
    assert hourly["wind_gust"] == pytest.approx(11.2)
    daily = normalized["daily"][0]
    assert daily["wind_speed"] == pytest.approx(6.8)
    assert daily["wind_deg"] == 225
    assert daily["uvi"] == 5


def test_save_pressure_history_creates_parent_directory(tmp_path, monkeypatch):
    import data_fetch

    target = tmp_path / "nested" / "pressure_history.json"
    monkeypatch.setattr(data_fetch, "_PRESSURE_HISTORY_PATH", str(target))
    monkeypatch.setattr(data_fetch, "_PRESSURE_HISTORY_LAST_SAVE", 0.0)

    data_fetch._PRESSURE_HISTORY.clear()
    data_fetch._PRESSURE_HISTORY.append((1700000000.0, 1012.3))

    data_fetch._save_pressure_history(1700000100.0)

    assert target.exists()


def test_weatherkit_hourly_daylight_does_not_force_night_icon():
    data = {
        "currentWeather": {
            "temperature": 10,
            "conditionCode": "Clear",
            "asOf": _iso("12:00:00"),
        },
        "forecastDaily": {
            "days": [
                {
                    "sunrise": _iso("06:00:00"),
                    "sunset": _iso("13:00:00"),
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
                    "forecastStart": _iso("14:00:00"),
                    "temperature": 10,
                    "precipitationChance": 0,
                    "conditionCode": "Clear",
                    "isDaylight": True,
                }
            ]
        },
        "weatherAlerts": {"alerts": []},
    }

    normalized = _normalise_weatherkit_response(data)

    assert normalized is not None
    assert normalized["hourly"][0]["weather"][0]["icon"] == "Clear"


def test_weatherkit_daily_maps_astronomical_and_moon_fields():
    data = {
        "currentWeather": {
            "temperature": 10,
            "conditionCode": "Clear",
            "asOf": _iso("12:00:00"),
        },
        "forecastDaily": {
            "days": [
                {
                    "sunrise": _iso("12:00:00"),
                    "sunset": _iso("22:00:00"),
                    "sunriseCivil": _iso("11:30:00"),
                    "sunriseNautical": _iso("11:00:00"),
                    "sunriseAstronomical": _iso("10:30:00"),
                    "sunsetCivil": _iso("22:30:00"),
                    "sunsetNautical": _iso("23:00:00"),
                    "sunsetAstronomical": _iso("23:30:00"),
                    "moonrise": _iso("15:15:00"),
                    "moonset": _iso("05:45:00"),
                    "moonPhase": "WaxingGibbous",
                    "temperatureMax": 20,
                    "temperatureMin": 5,
                    "precipitationChance": 0,
                    "conditionCode": "Clear",
                    "forecastStart": _iso("00:00:00"),
                }
            ]
        },
        "forecastHourly": {"hours": []},
        "weatherAlerts": {"alerts": []},
    }

    normalized = _normalise_weatherkit_response(data)

    assert normalized is not None
    day = normalized["daily"][0]
    assert day["sunriseCivil"] is not None
    assert day["sunriseNautical"] is not None
    assert day["sunriseAstronomical"] is not None
    assert day["sunsetCivil"] is not None
    assert day["sunsetNautical"] is not None
    assert day["sunsetAstronomical"] is not None
    assert day["moonrise"] is not None
    assert day["moonset"] is not None
    assert day["moonPhase"] == "WaxingGibbous"


def test_fetch_weatherkit_retries_without_alerts_on_404(monkeypatch):
    import datetime
    import requests
    import data_fetch

    now = datetime.datetime(2026, 4, 28, tzinfo=datetime.timezone.utc)
    calls: list[str] = []

    class _Resp:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(response=self)

        def json(self):
            return self._payload

    class _Session:
        def get(self, _url, params, headers, timeout):
            assert headers["Authorization"] == "Bearer fake-token"
            assert timeout == 10
            calls.append(params["dataSets"])
            if "weatherAlerts" in params["dataSets"]:
                return _Resp(404, {})
            return _Resp(
                200,
                {
                    "currentWeather": {"temperature": 10, "conditionCode": "Clear", "asOf": _iso("12:00:00")},
                    "forecastDaily": {"days": []},
                    "forecastHourly": {"hours": []},
                },
            )

    monkeypatch.setattr(data_fetch, "_build_weatherkit_token", lambda _now: "fake-token")
    monkeypatch.setattr(data_fetch, "_session", _Session())

    normalized = _fetch_weatherkit(now)

    assert normalized is not None
    assert calls == [
        "currentWeather,forecastDaily,forecastHourly,weatherAlerts",
        "currentWeather,forecastDaily,forecastHourly",
    ]


def test_weatherkit_hourly_rounds_sunset_to_detected_two_hour_increment():
    data = {
        "currentWeather": {"temperature": 10, "conditionCode": "Clear", "asOf": _iso("12:00:00")},
        "forecastDaily": {
            "days": [
                {
                    "sunrise": _iso("06:00:00"),
                    "sunset": _iso("19:10:00"),
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
                {"forecastStart": _iso("18:00:00"), "temperature": 10, "precipitationChance": 0, "conditionCode": "Clear"},
                {"forecastStart": _iso("20:00:00"), "temperature": 9, "precipitationChance": 0, "conditionCode": "Clear"},
            ]
        },
        "weatherAlerts": {"alerts": []},
    }

    normalized = _normalise_weatherkit_response(data)

    assert normalized is not None
    assert normalized["hourly"][0]["weather"][0]["icon"] == "Clear"
    assert normalized["hourly"][1]["weather"][0]["icon"] == "Clear_night"


def test_weatherkit_hourly_rounds_sunset_to_detected_one_hour_increment():
    data = {
        "currentWeather": {"temperature": 10, "conditionCode": "Clear", "asOf": _iso("12:00:00")},
        "forecastDaily": {
            "days": [
                {
                    "sunrise": _iso("06:00:00"),
                    "sunset": _iso("19:10:00"),
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
                {"forecastStart": _iso("18:00:00"), "temperature": 10, "precipitationChance": 0, "conditionCode": "Clear"},
                {"forecastStart": _iso("19:00:00"), "temperature": 10, "precipitationChance": 0, "conditionCode": "Clear"},
                {"forecastStart": _iso("20:00:00"), "temperature": 9, "precipitationChance": 0, "conditionCode": "Clear"},
            ]
        },
        "weatherAlerts": {"alerts": []},
    }

    normalized = _normalise_weatherkit_response(data)

    assert normalized is not None
    assert normalized["hourly"][0]["weather"][0]["icon"] == "Clear"
    assert normalized["hourly"][1]["weather"][0]["icon"] == "Clear"
    assert normalized["hourly"][2]["weather"][0]["icon"] == "Clear_night"


def test_weatherkit_current_switches_to_night_only_after_actual_sunset():
    data = {
        "currentWeather": {
            "temperature": 10,
            "conditionCode": "Clear",
            "asOf": _iso("19:30:00"),
        },
        "forecastDaily": {
            "days": [
                {
                    "sunrise": _iso("06:00:00"),
                    "sunset": _iso("20:00:00"),
                    "temperatureMax": 20,
                    "temperatureMin": 5,
                    "precipitationChance": 0,
                    "conditionCode": "Clear",
                    "forecastStart": _iso("00:00:00"),
                }
            ]
        },
        "forecastHourly": {"hours": []},
        "weatherAlerts": {"alerts": []},
    }

    normalized = _normalise_weatherkit_response(data)

    assert normalized is not None
    assert normalized["current"]["weather"][0]["icon"] == "Clear"
