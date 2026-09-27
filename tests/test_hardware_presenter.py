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


class _LedDisplay:
    def __init__(self):
        self.leds = []

    def set_led(self, r=0.0, g=0.0, b=0.0):
        self.leds.append((r, g, b))

    def apply_indicator_border(self, image):
        return "bordered"


def test_presenter_dims_screen_led_colors_by_its_own_level(monkeypatch):
    import utils

    monkeypatch.setattr(utils, "LED_INDICATOR_LEVEL", 0.1)
    display = _LedDisplay()
    presenter = HardwarePresenter(display=display)

    presenter.set_led((1.0, 0.5, 0.0))

    assert display.leds == [(0.1, 0.05, 0.0)]
    assert presenter.apply_indicator_border(Image.new("RGB", (2, 2))) == "bordered"


def test_presenter_led_none_restores_the_update_status(monkeypatch):
    import utils

    refreshed = []
    monkeypatch.setattr(utils, "_refresh_led_indicator", lambda display=None: refreshed.append(display))
    display = _LedDisplay()

    HardwarePresenter(display=display).set_led(None)

    assert refreshed == [display] and display.leds == []
