import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_script_module(name: str, filename: str):
    script_path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


api_checks = _load_script_module("desk_display_test_api_connections", "test_api_connections.py")


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
