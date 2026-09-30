import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from config import CENTRAL_TIME
from screens.draw_weather import (
    FONT_WEATHER_DETAILS_SMALL_BOLD,
    FONT_WEATHER_DETAILS_TINY_LARGE,
    _astronomical_layout_details,
    _astronomy_row_x_positions,
    _astronomy_time_text,
    _draw_weather_history_chart,
    _fit_text_and_font_to_width,
    _moon_illumination_mask,
    _moon_phase_is_waxing,
    _normalise_moon_phase,
    _safe_textbbox,
    _weather_detail_chart_layout,
    _weather_history_points,
)


def test_astronomical_layout_handles_supported_display_profiles():
    # displayhat mini + waveshare lcd/oled defaults
    display_hat = _astronomical_layout_details(320, 240)
    assert display_hat["split_columns"] is True
    assert display_hat["compact"] is True
    assert display_hat["group_rows"] is False

    # hyperpixel rectangular
    hyperpixel = _astronomical_layout_details(800, 480)
    assert hyperpixel["split_columns"] is True
    assert hyperpixel["compact"] is False
    assert hyperpixel["group_rows"] is False

    # hyperpixel square
    hyperpixel_square = _astronomical_layout_details(720, 720)
    assert hyperpixel_square["split_columns"] is True
    assert hyperpixel_square["compact"] is False
    # Rise/Set sit together instead of spreading down the tall columns.
    assert hyperpixel_square["group_rows"] is True
    assert hyperpixel_square["coords_below"] is True
    assert hyperpixel["coords_below"] is False

    # miniTFT
    minipitft = _astronomical_layout_details(240, 135)
    assert minipitft["ultra_compact"] is True
    assert minipitft["split_columns"] is False


def test_astronomical_layout_compacts_vertical_small_panels():
    portrait_compact = _astronomical_layout_details(240, 320)
    assert portrait_compact["compact"] is True
    assert portrait_compact["split_columns"] is False


def test_astronomy_time_text_is_12_hour_central_without_leading_zero():
    utc_event = datetime.datetime(2026, 1, 1, 7, 5, tzinfo=datetime.UTC)
    assert _astronomy_time_text(utc_event) == "1:05 AM"

    central_event = datetime.datetime(2026, 1, 1, 13, 5, tzinfo=CENTRAL_TIME)
    assert _astronomy_time_text(central_event) == "1:05 PM"


def test_astronomy_time_text_accepts_iso_timestamp_strings():
    assert _astronomy_time_text("2026-01-01T07:05:00Z") == "1:05 AM"


def test_astronomy_time_text_formats_midnight_and_noon_without_platform_specific_directives():
    assert _astronomy_time_text("2026-01-01T06:00:00Z") == "12:00 AM"
    assert _astronomy_time_text("2026-01-01T18:00:00Z") == "12:00 PM"


def test_astronomical_sun_rows_use_civil_times_without_civil_label():
    layout = _astronomical_layout_details(640, 480)
    assert layout["sun_labels"] == (("Rise", "sunrise_civil"), ("Set", "sunset_civil"))


def test_astronomical_row_centers_label_and_time_with_compact_gap():
    label_x, value_x = _astronomy_row_x_positions((10, 20, 210, 220), 30, 60, compact=True)

    assert (label_x, value_x) == (61, 99)
    assert value_x - (label_x + 30) == 8


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


def test_moon_phase_label_splits_camel_case_names():
    fraction, label = _normalise_moon_phase("waxingGibbous")

    assert fraction == 0.75
    assert label == "Waxing Gibbous"


def test_moon_phase_label_uses_smaller_font_before_truncating():
    image = Image.new("RGB", (240, 80))
    draw = ImageDraw.Draw(image)
    phase = "Waxing Gibbous"
    tiny_bbox = _safe_textbbox(draw, phase, FONT_WEATHER_DETAILS_TINY_LARGE)
    max_width = tiny_bbox[2] - tiny_bbox[0]

    fitted_text, fitted_font = _fit_text_and_font_to_width(
        draw,
        phase,
        (FONT_WEATHER_DETAILS_SMALL_BOLD, FONT_WEATHER_DETAILS_TINY_LARGE),
        max_width,
    )

    assert fitted_text == phase
    assert fitted_font == FONT_WEATHER_DETAILS_TINY_LARGE


def test_weather_history_points_filters_and_sorts_metric_values():
    weather = {
        "current_history": [
            {"dt": 1200, "wind_speed": 12},
            {"dt": 600, "wind_speed": 8},
            {"dt": 1800, "humidity": 55},
            {"dt": "bad", "wind_speed": 10},
        ]
    }

    assert _weather_history_points(weather, "wind_speed") == [(600.0, 8.0), (1200.0, 12.0)]


def test_weather_history_points_uses_hourly_and_current_when_history_is_sparse():
    weather = {
        "current_history": [{"dt": 1200, "wind_speed": 8}],
        "hourly": [
            {"dt": 600, "wind_speed": 4},
            {"dt": 900, "humidity": 50},
            {"dt": "bad", "wind_speed": 6},
        ],
        "current": {"dt": 1800, "wind_speed": 12},
    }

    assert _weather_history_points(weather, "wind_speed") == [
        (600.0, 4.0),
        (1200.0, 8.0),
        (1800.0, 12.0),
    ]


def test_weather_history_chart_draws_placeholder_for_fewer_than_two_points():
    image = Image.new("RGB", (40, 24), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    _draw_weather_history_chart(draw, (4, 4, 35, 19), [(600.0, 8.0)], (255, 0, 0))

    assert image.getpixel((4, 11)) == (28, 64, 88)
    assert image.getpixel((20, 11)) == (68, 105, 130)
    # The bottom ticks show that the chart's horizontal axis represents time.
    assert image.getpixel((20, 17)) == (68, 105, 130)


def test_weather_detail_charts_start_after_the_longest_value_with_padding():
    chart_x, chart_width, charts_enabled = _weather_detail_chart_layout(
        [115, 168, 143],
        value_x=80,
        right_edge=304,
        chart_gap=8,
        chart_min_w=40,
    )

    assert charts_enabled is True
    assert chart_x == 176
    assert chart_width == 128


def test_weather_detail_chart_layout_keeps_a_shared_minimum_chart_on_narrow_displays():
    chart_x, chart_width, charts_enabled = _weather_detail_chart_layout(
        [140],
        value_x=80,
        right_edge=180,
        chart_gap=8,
        chart_min_w=40,
    )

    assert charts_enabled is True
    assert (chart_x, chart_width) == (140, 40)


def test_astronomical_layout_groups_rows_on_1080p_hdmi():
    # Jason: on 1920x1080 Rise and Set sat far apart; keep them together there.
    assert _astronomical_layout_details(1920, 1080)["group_rows"] is True
    assert _astronomical_layout_details(800, 480)["group_rows"] is False


_COORDS_PROBE = """
import json, os, sys
os.environ["CONFIG_LOAD_DOTENV"] = "0"
for name in ("WEATHER_LATITUDE", "WEATHER_LONGITUDE"):
    os.environ.pop(name, None)
from screens.draw_weather import draw_weather_astronomical
weather = {
    "daily": [{"sunrise": "2026-09-30T06:50:00-05:00", "sunset": "2026-09-30T18:40:00-05:00",
               "moonPhase": 0.1}],
    "location": {"latitude": 41.9037, "longitude": -87.6357},
}
img = draw_weather_astronomical(None, weather).image
coord = (132, 149, 180)
rows = [y for y in range(img.height) if any(img.getpixel((x, y)) == coord for x in range(img.width))]
print(json.dumps({"size": img.size, "rows": [min(rows), max(rows)] if rows else None}))
"""


@pytest.mark.parametrize("profile_id", ["hyperpixel4_square", "hyperpixel4"])
def test_sun_and_moon_shows_the_coordinates_on_hyperpixel_panels(profile_id):
    from display_profiles import PROFILE_PRESETS
    from rendering.profile_process import composition_env

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    for name, value in composition_env(PROFILE_PRESETS[profile_id]).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    result = subprocess.run([sys.executable, "-c", _COORDS_PROBE], cwd=root, env=env, check=True,
                            capture_output=True, text=True, timeout=120)
    drawn = json.loads(result.stdout.strip().splitlines()[-1])
    width, height = drawn["size"]
    assert drawn["rows"], f"{profile_id}: coordinates not drawn"
    top, bottom = drawn["rows"]
    if height >= width:
        # The square panel's title leaves no room beside it: a line under the cards.
        assert top > height - 40
    else:
        assert bottom < 60
