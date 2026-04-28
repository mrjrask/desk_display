import datetime

from config import CENTRAL_TIME
import screens.draw_weather as draw_weather
from screens.draw_weather import _astronomical_layout_details
from screens.draw_weather import _astronomy_time_text


def test_astronomical_layout_handles_supported_display_profiles():
    # displayhat mini + waveshare lcd/oled defaults
    display_hat = _astronomical_layout_details(320, 240)
    assert display_hat["split_columns"] is True
    assert display_hat["compact"] is True

    # hyperpixel rectangular
    hyperpixel = _astronomical_layout_details(800, 480)
    assert hyperpixel["split_columns"] is True
    assert hyperpixel["compact"] is False

    # hyperpixel square
    hyperpixel_square = _astronomical_layout_details(720, 720)
    assert hyperpixel_square["split_columns"] is True
    assert hyperpixel_square["compact"] is False

    # miniTFT
    minipitft = _astronomical_layout_details(240, 135)
    assert minipitft["ultra_compact"] is True
    assert minipitft["split_columns"] is False


def test_astronomical_layout_compacts_vertical_small_panels():
    portrait_compact = _astronomical_layout_details(240, 320)
    assert portrait_compact["compact"] is True
    assert portrait_compact["split_columns"] is False


def test_astronomy_time_text_is_12_hour_central_without_leading_zero():
    utc_event = datetime.datetime(2026, 1, 1, 7, 5, tzinfo=datetime.timezone.utc)
    assert _astronomy_time_text(utc_event) == "1:05 AM"

    central_event = datetime.datetime(2026, 1, 1, 13, 5, tzinfo=CENTRAL_TIME)
    assert _astronomy_time_text(central_event) == "1:05 PM"


def test_astronomy_time_text_accepts_iso_timestamp_strings():
    assert _astronomy_time_text("2026-01-01T07:05:00Z") == "1:05 AM"


def test_astronomical_single_column_240x135_keeps_at_least_one_sun_row(monkeypatch):
    monkeypatch.setattr(draw_weather, "WIDTH", 240)
    monkeypatch.setattr(draw_weather, "HEIGHT", 135)

    rendered_text: list[str] = []
    original_text = draw_weather.ImageDraw.ImageDraw.text

    def _recording_text(self, xy, text, *args, **kwargs):
        rendered_text.append(text)
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(draw_weather.ImageDraw.ImageDraw, "text", _recording_text)

    weather = {
        "daily": [
            {
                "sunriseAstronomical": "2026-01-01T11:00:00Z",
                "sunriseCivil": "2026-01-01T11:30:00Z",
                "sunsetAstronomical": "2026-01-01T23:00:00Z",
                "sunsetCivil": "2026-01-01T22:30:00Z",
                "moonPhase": 0.5,
                "moonrise": "2026-01-01T20:00:00Z",
                "moonset": "2026-01-01T09:00:00Z",
            }
        ]
    }
    draw_weather.draw_weather_astronomical(None, weather)

    sun_labels = {label for label, _ in _astronomical_layout_details(240, 135)["sun_labels"]}
    assert any(text in sun_labels for text in rendered_text)
