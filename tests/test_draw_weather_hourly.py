import datetime

from config import CENTRAL_TIME
from screens.draw_weather import (
    _gather_daily_forecast,
    _gather_hourly_forecast,
    _temperature_chart_color,
    draw_weather_daily,
)


def _build_hourly_entry(dt: datetime.datetime, *, main: str = "Clouds", icon: str = "Cloudy") -> dict:
    return {
        "dt": int(dt.timestamp()),
        "temp": 70,
        "wind_speed": 8,
        "wind_deg": 90,
        "uvi": 3,
        "weather": [
            {
                "main": main,
                "icon": icon,
            }
        ],
    }


def test_gather_hourly_forecast_skips_past_entries():
    now = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=CENTRAL_TIME)

    weather = {
        "hourly": [
            _build_hourly_entry(now - datetime.timedelta(hours=3)),
            _build_hourly_entry(now - datetime.timedelta(hours=1)),
            _build_hourly_entry(now),
            _build_hourly_entry(now + datetime.timedelta(hours=1)),
            _build_hourly_entry(now + datetime.timedelta(hours=2)),
        ]
    }

    forecast = _gather_hourly_forecast(weather, 3, now=now)

    assert [entry["time"] for entry in forecast] == ["12pm", "1pm", "2pm"]


def test_gather_hourly_forecast_orders_future_entries():
    now = datetime.datetime(2024, 1, 1, 9, 0, tzinfo=CENTRAL_TIME)

    weather = {
        "hourly": [
            _build_hourly_entry(now + datetime.timedelta(hours=3)),
            _build_hourly_entry(now + datetime.timedelta(hours=1)),
            _build_hourly_entry(now + datetime.timedelta(hours=2)),
        ]
    }

    forecast = _gather_hourly_forecast(weather, 3, now=now)

    assert [entry["time"] for entry in forecast] == ["10am", "11am", "12pm"]


def test_gather_hourly_forecast_uses_two_hour_steps_when_stable():
    now = datetime.datetime(2024, 1, 1, 9, 0, tzinfo=CENTRAL_TIME)
    weather = {
        "hourly": [_build_hourly_entry(now + datetime.timedelta(hours=offset)) for offset in range(6)]
    }

    forecast = _gather_hourly_forecast(weather, 4, now=now)

    assert [entry["time"] for entry in forecast] == ["9am", "11am", "1pm"]


def test_gather_hourly_forecast_keeps_one_hour_steps_on_significant_changes():
    now = datetime.datetime(2024, 1, 1, 9, 0, tzinfo=CENTRAL_TIME)
    weather = {
        "hourly": [
            _build_hourly_entry(now + datetime.timedelta(hours=0), main="Clouds"),
            _build_hourly_entry(now + datetime.timedelta(hours=1), main="Rain"),
            _build_hourly_entry(now + datetime.timedelta(hours=2), main="Clear"),
            _build_hourly_entry(now + datetime.timedelta(hours=3), main="Clouds"),
            _build_hourly_entry(now + datetime.timedelta(hours=4), main="Clouds"),
        ]
    }

    forecast = _gather_hourly_forecast(weather, 4, now=now)

    assert [entry["time"] for entry in forecast] == ["9am", "10am", "11am", "12pm"]


def test_gather_daily_forecast_includes_icon_metadata():
    now = datetime.datetime(2024, 1, 1, 9, 0, tzinfo=CENTRAL_TIME)
    weather = {
        "daily": [
            {
                "dt": int(now.timestamp()),
                "temp": {"max": 50, "min": 31},
                "weather": [{"main": "Clouds", "icon": "Cloudy", "condition_code": "Cloudy"}],
            },
            {
                "dt": int((now + datetime.timedelta(days=1)).timestamp()),
                "temp": {"max": 55, "min": 35},
                "pop": 0.2,
                "weather": [{"main": "Rain", "icon": "Rain", "condition_code": "Rain"}],
            },
        ]
    }

    forecast = _gather_daily_forecast(weather, 1)

    assert forecast == [
        {
            "day": "Tmrw",
            "hi": 55,
            "lo": 35,
            "pop": 20,
            "is_snow": False,
            "condition": "Rain",
            "icon": "Rain",
            "condition_code": "Rain",
            "wind_speed": None,
            "wind_dir": "",
            "uvi": None,
        }
    ]


def test_gather_daily_forecast_falls_back_to_hourly_wind_and_uv():
    now = datetime.datetime(2024, 1, 1, 9, 0, tzinfo=CENTRAL_TIME)
    weather = {
        "daily": [
            {
                "dt": int(now.timestamp()),
                "temp": {"max": 50, "min": 31},
                "weather": [{"main": "Clouds", "icon": "Cloudy", "condition_code": "Cloudy"}],
            },
            {
                "dt": int((now + datetime.timedelta(days=1)).timestamp()),
                "temp": {"max": 55, "min": 35},
                "pop": 0.2,
                "weather": [{"main": "Rain", "icon": "Rain", "condition_code": "Rain"}],
            },
        ],
        "hourly": [
            _build_hourly_entry(now + datetime.timedelta(days=1, hours=1)),
            _build_hourly_entry(now + datetime.timedelta(days=1, hours=2)),
        ],
    }
    weather["hourly"][0]["wind_speed"] = 10
    weather["hourly"][0]["wind_deg"] = 135
    weather["hourly"][0]["uvi"] = 4
    weather["hourly"][1]["wind_speed"] = 14
    weather["hourly"][1]["wind_deg"] = 180
    weather["hourly"][1]["uvi"] = 6

    forecast = _gather_daily_forecast(weather, 1)

    assert forecast[0]["wind_speed"] == 12
    assert forecast[0]["wind_dir"] in {"↘", "SE"}
    assert forecast[0]["uvi"] == 6


def test_temperature_chart_color_uses_expected_band_colors():
    assert _temperature_chart_color(-20) == (211, 46, 179)
    assert _temperature_chart_color(60) == (255, 214, 0)
    assert _temperature_chart_color(105) == (204, 0, 0)


def test_temperature_chart_color_interpolates_between_bands():
    # 65°F sits halfway between the 60°F and 70°F chart colors.
    assert _temperature_chart_color(65) == (249, 192, 45)


def test_draw_weather_daily_renders_without_tuple_unpack_errors():
    now = datetime.datetime(2024, 1, 1, 9, 0, tzinfo=CENTRAL_TIME)
    weather = {
        "daily": [
            {
                "dt": int(now.timestamp()),
                "temp": {"max": 50, "min": 31},
                "weather": [{"main": "Clouds", "icon": "Cloudy", "condition_code": "Cloudy"}],
            },
            {
                "dt": int((now + datetime.timedelta(days=1)).timestamp()),
                "temp": {"max": 55, "min": 35},
                "pop": 0.2,
                "weather": [{"main": "Rain", "icon": "Rain", "condition_code": "Rain"}],
            },
        ]
    }

    rendered = draw_weather_daily(None, weather, transition=False, days=1)

    assert rendered is not None
    assert rendered.image.size[0] > 0
    assert rendered.image.size[1] > 0


def test_alert_message_text_prefers_event_and_description():
    from screens.draw_weather import _alert_message_text

    alert = {
        "event": "Severe Thunderstorm Warning",
        "description": "Take shelter now. Damaging winds are expected.",
    }

    assert (
        _alert_message_text(alert)
        == "Severe Thunderstorm Warning: Take shelter now. Damaging winds are expected."
    )


def test_selected_alert_returns_highest_priority_alert_with_message():
    from screens.draw_weather import _alert_message_text, _selected_alert

    weather = {
        "alerts": [
            {"event": "Flood Advisory", "description": "Minor flooding is possible."},
            {"event": "Tornado Warning", "description": "Move to an interior room now."},
        ]
    }

    severity, alert = _selected_alert(weather)

    assert severity == "warning"
    assert _alert_message_text(alert) == "Tornado Warning: Move to an interior room now."


def test_selected_alert_uses_provider_severity_when_text_is_generic():
    from screens.draw_weather import _selected_alert

    weather = {"alerts": [{"description": "Flooding", "severity": "extreme"}]}

    severity, alert = _selected_alert(weather)

    assert severity == "warning"
    assert alert == weather["alerts"][0]


def test_selected_alert_maps_provider_severity_priority():
    from screens.draw_weather import _selected_alert

    weather = {
        "alerts": [
            {"description": "Coastal flooding", "severity": "minor"},
            {"description": "River flooding", "severity": "moderate"},
        ]
    }

    severity, alert = _selected_alert(weather)

    assert severity == "watch"
    assert alert == weather["alerts"][1]


def test_alert_indicator_stays_above_bottom_safe_buffer():
    from PIL import Image, ImageDraw

    from screens import draw_weather

    img = Image.new("RGB", (draw_weather.WIDTH, draw_weather.HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    draw_weather._draw_alert_indicator(img, draw, "warning")

    bottom_rows = [
        img.getpixel((x, y))
        for y in range(max(0, draw_weather.HEIGHT - 6), draw_weather.HEIGHT)
        for x in range(draw_weather.WIDTH)
    ]
    top_rows = [
        img.getpixel((x, y))
        for y in range(0, min(24, draw_weather.HEIGHT))
        for x in range(draw_weather.WIDTH)
    ]

    assert any(pixel != (0, 0, 0) for pixel in top_rows)
    assert all(pixel == (0, 0, 0) for pixel in bottom_rows)
