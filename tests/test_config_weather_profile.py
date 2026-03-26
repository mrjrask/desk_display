import config


def test_weather_profile_uses_home_destination_for_wiffy(monkeypatch):
    monkeypatch.setenv("TRAVEL_TO_HOME_DESTINATION", "41.1000,-87.2000")
    lat, lon, mode = config._weather_profile_for_ssid("wiffy")
    assert (lat, lon, mode) == (41.1, -87.2, "to_work")


def test_weather_profile_uses_work_destination_for_verano_variations(monkeypatch):
    monkeypatch.setenv("TRAVEL_TO_WORK_DESTINATION", "42.2000,-88.3000")
    lat, lon, mode = config._weather_profile_for_ssid("Verano-Guest")
    assert (lat, lon, mode) == (42.2, -88.3, "to_home")


def test_owm_api_key_uses_verano_bucket_for_variations(monkeypatch):
    monkeypatch.setenv("OWM_API_KEY_VERANO", "verano-key")
    monkeypatch.setenv("OWM_API_KEY_DEFAULT", "default-key")
    assert config._get_owm_api_key("my Verano network") == "verano-key"
