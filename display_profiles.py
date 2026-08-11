from __future__ import annotations

from dataclasses import dataclass

DISPLAY_PROFILE_DISPLAY_HAT_MINI = "display_hat_mini"
DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114 = "adafruit_minipitft_114"
DISPLAY_PROFILE_HYPERPIXEL4 = "hyperpixel4"
DISPLAY_PROFILE_HYPERPIXEL4_SQUARE = "hyperpixel4_square"
DISPLAY_PROFILE_HDMI_1080P = "hdmi_1080p"
DISPLAY_PROFILE_FALLBACK_HD = "fallback_hd"
DISPLAY_PROFILE_FALLBACK_DEFAULT = "fallback_default"


@dataclass(frozen=True)
class DisplayProfilePreset:
    profile_id: str
    fade_in_steps: int
    scoreboard_scroll_step: int
    scoreboard_scroll_delay: float
    logo_scale_cap: float
    animation_delay: float
    use_compact_layout: bool = False

    @property
    def is_hyperpixel_next_layout(self) -> bool:
        return self.use_compact_layout

    @property
    def is_hyperpixel_4_square_layout(self) -> bool:
        return self.profile_id == DISPLAY_PROFILE_HYPERPIXEL4_SQUARE


PROFILE_PRESETS: dict[str, DisplayProfilePreset] = {
    DISPLAY_PROFILE_DISPLAY_HAT_MINI: DisplayProfilePreset(
        profile_id=DISPLAY_PROFILE_DISPLAY_HAT_MINI,
        fade_in_steps=10,
        scoreboard_scroll_step=1,
        scoreboard_scroll_delay=0.020,
        logo_scale_cap=1.0,
        animation_delay=0.06,
        use_compact_layout=False,
    ),
    DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114: DisplayProfilePreset(
        profile_id=DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114,
        fade_in_steps=6,
        scoreboard_scroll_step=1,
        scoreboard_scroll_delay=0.028,
        logo_scale_cap=1.1,
        animation_delay=0.06,
        use_compact_layout=True,
    ),
    DISPLAY_PROFILE_HYPERPIXEL4: DisplayProfilePreset(
        profile_id=DISPLAY_PROFILE_HYPERPIXEL4,
        fade_in_steps=6,
        scoreboard_scroll_step=1,
        scoreboard_scroll_delay=0.030,
        logo_scale_cap=3.0,
        animation_delay=0.05,
        use_compact_layout=True,
    ),
    DISPLAY_PROFILE_HYPERPIXEL4_SQUARE: DisplayProfilePreset(
        profile_id=DISPLAY_PROFILE_HYPERPIXEL4_SQUARE,
        fade_in_steps=6,
        scoreboard_scroll_step=1,
        scoreboard_scroll_delay=0.030,
        logo_scale_cap=3.0,
        animation_delay=0.05,
        use_compact_layout=True,
    ),
    DISPLAY_PROFILE_HDMI_1080P: DisplayProfilePreset(
        profile_id=DISPLAY_PROFILE_HDMI_1080P,
        fade_in_steps=0,
        scoreboard_scroll_step=2,
        scoreboard_scroll_delay=0.016,
        logo_scale_cap=5.0,
        animation_delay=0.04,
        use_compact_layout=False,
    ),
    DISPLAY_PROFILE_FALLBACK_HD: DisplayProfilePreset(
        profile_id=DISPLAY_PROFILE_FALLBACK_HD,
        fade_in_steps=0,
        scoreboard_scroll_step=1,
        scoreboard_scroll_delay=0.016,
        logo_scale_cap=1.2,
        animation_delay=0.04,
        use_compact_layout=True,
    ),
    DISPLAY_PROFILE_FALLBACK_DEFAULT: DisplayProfilePreset(
        profile_id=DISPLAY_PROFILE_FALLBACK_DEFAULT,
        fade_in_steps=10,
        scoreboard_scroll_step=1,
        scoreboard_scroll_delay=0.020,
        logo_scale_cap=1.0,
        animation_delay=0.06,
        use_compact_layout=False,
    ),
}


def _is_hd_widescreen_layout(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    long_edge = max(width, height)
    short_edge = min(width, height)
    if long_edge < 1280 or short_edge < 720:
        return False
    return (long_edge / short_edge) >= (16 / 10)


def resolve_display_profile_by_id(profile_id: str) -> DisplayProfilePreset | None:
    return PROFILE_PRESETS.get(profile_id.strip().lower())


def resolve_display_profile(width: int, height: int) -> DisplayProfilePreset:
    if (width, height) in {(320, 240), (240, 320)}:
        return PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI]
    if (width, height) in {(240, 135), (135, 240)}:
        return PROFILE_PRESETS[DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114]
    if (width, height) == (720, 720):
        return PROFILE_PRESETS[DISPLAY_PROFILE_HYPERPIXEL4_SQUARE]
    if (width, height) in {(800, 480), (480, 800)}:
        return PROFILE_PRESETS[DISPLAY_PROFILE_HYPERPIXEL4]
    if sorted((width, height)) == [1080, 1920]:
        return PROFILE_PRESETS[DISPLAY_PROFILE_HDMI_1080P]
    if _is_hd_widescreen_layout(width, height):
        return PROFILE_PRESETS[DISPLAY_PROFILE_FALLBACK_HD]
    return PROFILE_PRESETS[DISPLAY_PROFILE_FALLBACK_DEFAULT]
