"""Immutable descriptions of the displays supported by desk-display.

The values in this module are deliberately independent of :mod:`config`.  A
profile is render input, rather than process configuration, which makes it safe
to build registries for several displays in the same interpreter.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

DISPLAY_PROFILE_DISPLAY_HAT_MINI = "display_hat_mini"
DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114 = "adafruit_minipitft_114"
DISPLAY_PROFILE_HYPERPIXEL4 = "hyperpixel4"
DISPLAY_PROFILE_HYPERPIXEL4_SQUARE = "hyperpixel4_square"
DISPLAY_PROFILE_WAVESHARE_LCD_320X240 = "waveshare_lcd_320x240"
DISPLAY_PROFILE_WAVESHARE_OLED_128X64 = "waveshare_oled_128x64"
DISPLAY_PROFILE_HDMI_1080P = "hdmi_1080p"
DISPLAY_PROFILE_FALLBACK_HD = "fallback_hd"
DISPLAY_PROFILE_FALLBACK_DEFAULT = "fallback_default"


@dataclass(frozen=True)
class HardwareRenderingConstraints:
    """Capabilities which can affect the final frame sent to hardware."""

    physical_rotation: int = 0
    max_refresh_hz: float | None = None
    monochrome: bool = False
    framebuffer: bool = False


@dataclass(frozen=True)
class RenderProfile:
    """All display-dependent inputs used while composing a frame.

    ``width`` and ``height`` are the logical canvas dimensions, before the
    output driver's physical rotation is applied.
    """

    profile_id: str
    width: int
    height: int
    color_mode: str
    image_formats: tuple[str, ...]
    use_compact_layout: bool
    font_scale: float
    logo_scale: float
    logo_scale_cap: float
    scoreboard_scroll_step: int
    scoreboard_scroll_delay: float
    fade_in_steps: int
    animation_delay: float
    constraints: HardwareRenderingConstraints

    @property
    def canonical_profile_id(self) -> str:
        return self.profile_id

    @property
    def logical_width(self) -> int:
        return self.width

    @property
    def logical_height(self) -> int:
        return self.height

    @property
    def output_color_mode(self) -> str:
        return self.color_mode

    @property
    def supported_image_formats(self) -> tuple[str, ...]:
        return self.image_formats

    @property
    def compact_layout(self) -> bool:
        return self.use_compact_layout

    @property
    def is_hyperpixel_next_layout(self) -> bool:
        return self.use_compact_layout

    @property
    def is_hyperpixel_4_square_layout(self) -> bool:
        return self.profile_id == DISPLAY_PROFILE_HYPERPIXEL4_SQUARE


# Compatibility name for integrations written before RenderProfile acquired
# dimensions and hardware capabilities.
DisplayProfilePreset = RenderProfile


def _profile(profile_id: str, width: int, height: int, *, compact: bool,
             fade: int, scroll_step: int, scroll_delay: float, logo_cap: float,
             animation_delay: float, color_mode: str = "RGB", font_scale: float = 1.0,
             logo_scale: float = 1.0,
             constraints: HardwareRenderingConstraints | None = None) -> RenderProfile:
    return RenderProfile(
        profile_id, width, height, color_mode,
        ("PNG", "JPEG", "WEBP") if color_mode == "RGB" else ("PNG",),
        compact, font_scale, logo_scale, logo_cap, scroll_step, scroll_delay,
        fade, animation_delay, constraints or HardwareRenderingConstraints(),
    )


PROFILE_PRESETS: dict[str, RenderProfile] = {
    DISPLAY_PROFILE_DISPLAY_HAT_MINI: _profile(DISPLAY_PROFILE_DISPLAY_HAT_MINI, 320, 240, compact=False, fade=10, scroll_step=1, scroll_delay=.020, logo_cap=1., animation_delay=.06, constraints=HardwareRenderingConstraints(max_refresh_hz=30, framebuffer=True)),
    DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114: _profile(DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114, 240, 135, compact=True, fade=6, scroll_step=1, scroll_delay=.028, logo_cap=1.1, animation_delay=.06, font_scale=.75, logo_scale=.75, constraints=HardwareRenderingConstraints(max_refresh_hz=30)),
    DISPLAY_PROFILE_HYPERPIXEL4: _profile(DISPLAY_PROFILE_HYPERPIXEL4, 800, 480, compact=True, fade=6, scroll_step=1, scroll_delay=.030, logo_cap=3., animation_delay=.05, font_scale=2., logo_scale=2., constraints=HardwareRenderingConstraints(max_refresh_hz=60, framebuffer=True)),
    DISPLAY_PROFILE_HYPERPIXEL4_SQUARE: _profile(DISPLAY_PROFILE_HYPERPIXEL4_SQUARE, 720, 720, compact=True, fade=6, scroll_step=1, scroll_delay=.030, logo_cap=3., animation_delay=.05, font_scale=2.25, logo_scale=2.25, constraints=HardwareRenderingConstraints(max_refresh_hz=60, framebuffer=True)),
    DISPLAY_PROFILE_WAVESHARE_LCD_320X240: _profile(DISPLAY_PROFILE_WAVESHARE_LCD_320X240, 320, 240, compact=False, fade=10, scroll_step=1, scroll_delay=.020, logo_cap=1., animation_delay=.06, constraints=HardwareRenderingConstraints(physical_rotation=90, max_refresh_hz=30, framebuffer=True)),
    DISPLAY_PROFILE_WAVESHARE_OLED_128X64: _profile(DISPLAY_PROFILE_WAVESHARE_OLED_128X64, 128, 64, compact=True, fade=8, scroll_step=1, scroll_delay=.035, logo_cap=.5, animation_delay=.035, color_mode="1", font_scale=.4, logo_scale=.4, constraints=HardwareRenderingConstraints(max_refresh_hz=30, monochrome=True)),
    DISPLAY_PROFILE_HDMI_1080P: _profile(DISPLAY_PROFILE_HDMI_1080P, 1920, 1080, compact=False, fade=0, scroll_step=2, scroll_delay=.016, logo_cap=5., animation_delay=.04, font_scale=4.5, logo_scale=4.5),
    DISPLAY_PROFILE_FALLBACK_HD: _profile(DISPLAY_PROFILE_FALLBACK_HD, 1280, 720, compact=True, fade=0, scroll_step=1, scroll_delay=.016, logo_cap=1.2, animation_delay=.04, font_scale=3., logo_scale=3.),
    DISPLAY_PROFILE_FALLBACK_DEFAULT: _profile(DISPLAY_PROFILE_FALLBACK_DEFAULT, 320, 240, compact=False, fade=10, scroll_step=1, scroll_delay=.020, logo_cap=1., animation_delay=.06),
}


def _is_hd_widescreen_layout(width: int, height: int) -> bool:
    return width > 0 and height > 0 and max(width, height) >= 1280 and min(width, height) >= 720 and max(width, height) / min(width, height) >= 16 / 10


def resolve_display_profile_by_id(profile_id: str) -> RenderProfile | None:
    return PROFILE_PRESETS.get(profile_id.strip().lower())


def resolve_display_profile(width: int, height: int) -> RenderProfile:
    dimensions = (width, height)
    if dimensions in {(320, 240), (240, 320)}:
        preset = PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI]
        return replace(preset, width=width, height=height)
    if dimensions in {(240, 135), (135, 240)}:
        preset = PROFILE_PRESETS[DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114]
        return replace(preset, width=width, height=height)
    if dimensions == (720, 720):
        return PROFILE_PRESETS[DISPLAY_PROFILE_HYPERPIXEL4_SQUARE]
    if dimensions in {(800, 480), (480, 800)}:
        preset = PROFILE_PRESETS[DISPLAY_PROFILE_HYPERPIXEL4]
        return replace(preset, width=width, height=height)
    if sorted(dimensions) == [1080, 1920]:
        preset = PROFILE_PRESETS[DISPLAY_PROFILE_HDMI_1080P]
        return replace(preset, width=width, height=height)
    if _is_hd_widescreen_layout(width, height):
        preset = PROFILE_PRESETS[DISPLAY_PROFILE_FALLBACK_HD]
    else:
        preset = PROFILE_PRESETS[DISPLAY_PROFILE_FALLBACK_DEFAULT]
    # Fallbacks preserve the caller's actual logical canvas.
    return replace(preset, width=width, height=height)
