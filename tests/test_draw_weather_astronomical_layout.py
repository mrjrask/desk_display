import datetime

from config import CENTRAL_TIME
from screens.draw_weather import _astronomical_layout_details
from screens.draw_weather import _astronomy_time_text
from screens.draw_weather import _moon_illumination_mask
from screens.draw_weather import _moon_phase_is_waxing


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


def test_astronomy_time_text_formats_midnight_and_noon_without_platform_specific_directives():
    assert _astronomy_time_text("2026-01-01T06:00:00Z") == "12:00 AM"
    assert _astronomy_time_text("2026-01-01T18:00:00Z") == "12:00 PM"


def test_astronomical_sun_rows_use_civil_times_only():
    layout = _astronomical_layout_details(640, 480)
    assert layout["sun_labels"] == (("Civil Rise", "sunrise_civil"), ("Civil Set", "sunset_civil"))


def test_moon_phase_direction_controls_illuminated_side():
    waxing = _moon_illumination_mask(10, 0.5, waxing=True)
    waning = _moon_illumination_mask(10, 0.5, waxing=False)

    assert waxing.getpixel((15, 10)) == 255
    assert waxing.getpixel((5, 10)) == 0
    assert waning.getpixel((5, 10)) == 255
    assert waning.getpixel((15, 10)) == 0


def test_moon_phase_name_identifies_waning_labels():
    assert _moon_phase_is_waxing("WaxingCrescent", "Waxing Crescent") is True
    assert _moon_phase_is_waxing("WaningGibbous", "Waning Gibbous") is False
    assert _moon_phase_is_waxing("ThirdQuarter", "Third Quarter") is False
