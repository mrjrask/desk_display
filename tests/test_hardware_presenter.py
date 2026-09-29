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


class _PushDisplay:
    def __init__(self, image_pushes_frame):
        self.image_pushes_frame = image_pushes_frame
        self.pushes = 0
        self.images = []

    def image(self, image):
        self.images.append(image)
        if self.image_pushes_frame:
            self.pushes += 1

    def show(self):
        self.pushes += 1


def test_presenter_pushes_each_frame_to_the_panel_once():
    # utils.Display.image() already sends the frame over SPI; calling show()
    # as well sent it twice, halving the frame rate on a Display HAT Mini.
    import utils

    assert utils.Display.image_pushes_frame
    profile = PROFILE_PRESETS[DISPLAY_PROFILE_WAVESHARE_LCD_320X240]
    for pushes_in_image in (True, False):
        display = _PushDisplay(pushes_in_image)
        HardwarePresenter(display=display, profile=profile).present(
            Image.new("RGB", (profile.width, profile.height)))
        assert display.pushes == 1


def test_presenter_passes_matching_frames_through_without_copying():
    profile = PROFILE_PRESETS[DISPLAY_PROFILE_WAVESHARE_LCD_320X240]
    presenter = HardwarePresenter(display=object(), profile=profile)
    frame = Image.new(profile.color_mode, (profile.width, profile.height))

    assert presenter.convert(frame) is frame
    assert presenter.convert(Image.new("L", (10, 10))).size == (profile.width, profile.height)
    assert presenter.convert(Image.new("L", (10, 10))).mode == profile.color_mode
