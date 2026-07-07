import datetime as dt

from PIL import Image, ImageDraw

import screens.on_this_day as otd
from screens.on_this_day import _build_sections, draw_on_this_day


class DummyDisplay:
    width = 320
    height = 240

    def __init__(self):
        self.frames = []

    def clear(self):
        pass

    def image(self, img):
        self.frames.append(img.copy())


def test_on_this_day_exposes_editable_font_defaults_for_requested_profiles():
    expected_profiles = {
        "hdmi_1080p",
        "fallback_hd",
        "fallback_default",
        "display_hat_mini",
        "hyperpixel4_square",
        "hyperpixel4",
    }

    assert expected_profiles <= set(otd.ON_THIS_DAY_FONT_SIZES_BY_PROFILE)
    for profile in expected_profiles:
        sizes = otd._font_sizes_for_profile(profile)
        assert {"title", "section", "body", "year"} <= set(sizes)
        assert all(isinstance(value, int) and value > 0 for value in sizes.values())


def test_on_this_day_has_requested_curated_sections_for_july_6(monkeypatch):
    monkeypatch.setattr("screens.on_this_day._wiki_items", lambda *args, **kwargs: [])

    sections = _build_sections(dt.date(2026, 7, 6))

    assert "🇺🇸 American History" in sections
    assert "🏙️ Chicago History" in sections
    assert "🏟️ Sports History" in sections
    assert "🎂 Famous Birthdays" in sections
    assert "💾 Tech & Science" in sections


def test_on_this_day_uses_curated_sections_without_wikimedia_calls(monkeypatch):
    def fail_wiki_call(*args, **kwargs):
        raise AssertionError("fallback date should not call Wikimedia feeds")

    monkeypatch.setattr("screens.on_this_day._wiki_items", fail_wiki_call)

    sections = _build_sections(dt.date(2026, 7, 6))

    assert "🌎 General History" in sections
    assert "💾 Tech & Science" in sections


def test_on_this_day_renderer_produces_frame(monkeypatch):
    monkeypatch.setattr("screens.on_this_day._wiki_items", lambda *args, **kwargs: [])
    display = DummyDisplay()

    screen = draw_on_this_day(display, transition=False, today=dt.date(2026, 7, 6))

    assert screen.image.size == (320, 240)
    assert display.frames


def test_on_this_day_full_image_is_tall_enough_for_curated_july_6(monkeypatch):
    monkeypatch.setattr("screens.on_this_day._wiki_items", lambda *args, **kwargs: [])
    monkeypatch.setattr("screens.on_this_day._download_thumbnail", lambda *args, **kwargs: None)

    full_img = otd._render_full_image(dt.date(2026, 7, 6))

    assert full_img.height >= 1211


def test_on_this_day_year_items_wrap_within_remaining_card_width(monkeypatch):
    monkeypatch.setattr("screens.on_this_day._wiki_items", lambda *args, **kwargs: [])
    sections = otd._build_sections(dt.date(2026, 7, 6))
    draw = ImageDraw.Draw(Image.new("RGB", (otd.W, otd.H), otd._BG))
    pad = max(8, otd.W // 32)
    thumb_size = 34 if otd.W >= 300 else 0

    for items in sections.values():
        for item in items:
            if item.year is None:
                continue
            _, text_width = otd._item_text_layout(draw, item, pad, otd.W - pad, thumb_size)
            lines = otd.wrap_text(item.text, otd.BODY_FONT, text_width)

            assert lines
            assert all(
                otd.measure_text(draw, line, otd.BODY_FONT)[0] <= text_width
                for line in lines
            )
            assert text_width <= 210


def test_on_this_day_scroll_uses_smooth_readable_tuning(monkeypatch):
    monkeypatch.setattr("screens.on_this_day._wiki_items", lambda *args, **kwargs: [])
    monkeypatch.setattr("screens.on_this_day._download_thumbnail", lambda *args, **kwargs: None)
    captured = {}

    def fake_scroll_vertical_content(**kwargs):
        captured.update(kwargs)
        kwargs["render_at_offset"](0)

    monkeypatch.setattr(otd, "scroll_vertical_content", fake_scroll_vertical_content)
    display = DummyDisplay()

    draw_on_this_day(display, transition=True, today=dt.date(2026, 7, 6))

    assert captured["base_step"] == 1
    assert captured["min_frame_time"] == 0.030
    assert captured["page_jump_mode"] is False
