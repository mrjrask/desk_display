"""Tests for the hardware presentation boundary."""

from PIL import Image

from display.hardware_presenter import HardwarePresenter
from display_profiles import DISPLAY_PROFILE_WAVESHARE_LCD_320X240, PROFILE_PRESETS


def test_presenter_leaves_physical_rotation_to_display_driver():
    profile = PROFILE_PRESETS[DISPLAY_PROFILE_WAVESHARE_LCD_320X240]
    presenter = HardwarePresenter(display=object(), profile=profile)

    converted = presenter.convert(Image.new("RGB", (profile.width, profile.height)))

    assert profile.constraints.physical_rotation == 90
    assert converted.size == (profile.width, profile.height)
