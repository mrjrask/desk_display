from dataclasses import FrozenInstanceError

import pytest

from display_profiles import (
    DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114,
    DISPLAY_PROFILE_DISPLAY_HAT_MINI,
    DISPLAY_PROFILE_FALLBACK_DEFAULT,
    DISPLAY_PROFILE_FALLBACK_HD,
    DISPLAY_PROFILE_HDMI_1080P,
    DISPLAY_PROFILE_HYPERPIXEL4,
    DISPLAY_PROFILE_HYPERPIXEL4_SQUARE,
    DISPLAY_PROFILE_WAVESHARE_LCD_320X240,
    DISPLAY_PROFILE_WAVESHARE_OLED_128X64,
    PROFILE_PRESETS,
    resolve_display_profile,
)


def test_all_supported_render_profiles_are_explicit_and_frozen():
    expected = {
        DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114,
        DISPLAY_PROFILE_DISPLAY_HAT_MINI,
        DISPLAY_PROFILE_HYPERPIXEL4,
        DISPLAY_PROFILE_HYPERPIXEL4_SQUARE,
        DISPLAY_PROFILE_WAVESHARE_LCD_320X240,
        DISPLAY_PROFILE_WAVESHARE_OLED_128X64,
        DISPLAY_PROFILE_HDMI_1080P,
        DISPLAY_PROFILE_FALLBACK_HD,
        DISPLAY_PROFILE_FALLBACK_DEFAULT,
    }
    assert expected <= PROFILE_PRESETS.keys()
    with pytest.raises(FrozenInstanceError):
        PROFILE_PRESETS[DISPLAY_PROFILE_HYPERPIXEL4].width = 1


@pytest.mark.parametrize(
    ("dimensions", "profile_id", "mode", "compact"),
    [
        ((800, 480), DISPLAY_PROFILE_HYPERPIXEL4, "RGB", True),
        ((720, 720), DISPLAY_PROFILE_HYPERPIXEL4_SQUARE, "RGB", True),
        ((320, 240), DISPLAY_PROFILE_DISPLAY_HAT_MINI, "RGB", False),
        ((240, 135), DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114, "RGB", True),
        ((1920, 1080), DISPLAY_PROFILE_HDMI_1080P, "RGB", False),
        ((240, 320), DISPLAY_PROFILE_DISPLAY_HAT_MINI, "RGB", False),
        ((135, 240), DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114, "RGB", True),
        ((480, 800), DISPLAY_PROFILE_HYPERPIXEL4, "RGB", True),
        ((1080, 1920), DISPLAY_PROFILE_HDMI_1080P, "RGB", False),
    ],
)
def test_resolution_profiles(dimensions, profile_id, mode, compact):
    profile = resolve_display_profile(*dimensions)
    assert (profile.width, profile.height) == dimensions
    assert profile.profile_id == profile_id
    assert profile.color_mode == mode
    assert profile.use_compact_layout is compact


def test_custom_fallback_dimensions_do_not_leak_between_profiles():
    first = resolve_display_profile(1024, 600)
    second = resolve_display_profile(1366, 768)
    third = resolve_display_profile(1024, 600)
    assert (first.width, first.height) == (1024, 600)
    assert (second.width, second.height) == (1366, 768)
    assert first == third
    assert first is not third


def test_display_hat_mini_is_not_kernel_driven():
    profile = PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI]

    assert profile.constraints.framebuffer is False
