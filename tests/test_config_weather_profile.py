import config


def test_resolve_weather_coordinates_from_env(monkeypatch):
    monkeypatch.setenv("WEATHER_LATITUDE", "41.1000")
    monkeypatch.setenv("WEATHER_LONGITUDE", "-87.2000")
    lat, lon, errors = config._resolve_weather_coordinates()
    assert (lat, lon, errors) == (41.1, -87.2, [])


def test_resolve_weather_coordinates_reports_missing_vars(monkeypatch):
    monkeypatch.delenv("WEATHER_LATITUDE", raising=False)
    monkeypatch.delenv("WEATHER_LONGITUDE", raising=False)
    lat, lon, errors = config._resolve_weather_coordinates()
    assert lat is None
    assert lon is None
    assert "WEATHER_LATITUDE is missing" in errors
    assert "WEATHER_LONGITUDE is missing" in errors


def test_owm_api_key_randomly_selected_from_pool(monkeypatch):
    monkeypatch.setenv("OWM_API_KEY", "primary-key")
    monkeypatch.setenv("OWM_API_KEY_DEFAULT", "default-key")
    monkeypatch.setenv("OWM_API_KEY_WIFFY", "wiffy-key")
    monkeypatch.setenv("OWM_API_KEY_VERANO", "verano-key")
    monkeypatch.setattr(config.random, "choice", lambda values: values[-1])

    assert config._get_owm_api_key() == "verano-key"


def test_weather_config_accepts_single_openweathermap_key(monkeypatch):
    monkeypatch.setattr(config, "OWM_API_KEY", "default-key")
    monkeypatch.setattr(config, "WEATHERKIT_TEAM_ID", "team-id")
    monkeypatch.setattr(config, "WEATHERKIT_KEY_ID", "key-id")
    monkeypatch.setattr(config, "WEATHERKIT_SERVICE_ID", "service-id")
    monkeypatch.setattr(config, "WEATHERKIT_PRIVATE_KEY", None)
    monkeypatch.setattr(config, "WEATHERKIT_KEY_PATH", None)

    assert config._build_weather_config_errors([]) == []


def test_weather_config_requires_at_least_one_provider(monkeypatch):
    monkeypatch.setattr(config, "OWM_API_KEY", None)
    monkeypatch.setattr(config, "WEATHERKIT_TEAM_ID", "team-id")
    monkeypatch.setattr(config, "WEATHERKIT_KEY_ID", "key-id")
    monkeypatch.setattr(config, "WEATHERKIT_SERVICE_ID", "service-id")
    monkeypatch.setattr(config, "WEATHERKIT_PRIVATE_KEY", None)
    monkeypatch.setattr(config, "WEATHERKIT_KEY_PATH", None)

    errors = config._build_weather_config_errors([])

    assert len(errors) == 1
    assert "No weather provider configured" in errors[0]
    assert "WEATHERKIT_PRIVATE_KEY or WEATHERKIT_KEY_PATH" in errors[0]
