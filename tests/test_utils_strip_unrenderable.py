from PIL import ImageFont

import config
import pytest

import utils
from utils import strip_unrenderable

FONT = config.FONT_WEATHER_DETAILS_SMALL


def test_strip_unrenderable_drops_emoji_and_collapses_spaces():
    text = "Jewish holiday: \U0001F33F\U0001F34B Sukkot IV"

    assert strip_unrenderable(text, FONT) == "Jewish holiday: Sukkot IV"


def test_strip_unrenderable_drops_joiners_and_variation_selectors():
    text = "Rosh Hashana \U0001F34F‍\U0001F36F️ 5787"

    assert strip_unrenderable(text, FONT) == "Rosh Hashana 5787"


def test_strip_unrenderable_keeps_characters_the_font_has():
    text = "Café — “quote” שלום ☀"

    assert strip_unrenderable(text, FONT) == text


def test_strip_unrenderable_leaves_ascii_untouched():
    text = "Plain  ASCII text"

    assert strip_unrenderable(text, FONT) is text


def test_strip_unrenderable_works_with_default_bitmap_font():
    assert strip_unrenderable("abc \U0001F33F", ImageFont.load_default()) == "abc"


def _no_emoji_font(monkeypatch):
    monkeypatch.setattr(utils, "_COLOR_EMOJI_FONT_CACHE", {"font": None})
    monkeypatch.setattr(utils, "_EMOJI_SUPPORT_CACHE", {})


def test_emoji_text_runs_drop_emoji_without_emoji_font(monkeypatch):
    _no_emoji_font(monkeypatch)

    runs = utils.emoji_text_runs("Holiday: \U0001F33F\U0001F34B Sukkot 中", FONT)

    assert runs == [("Holiday: Sukkot", False)]


def test_emoji_text_width_matches_plain_text_without_emoji_font(monkeypatch):
    _no_emoji_font(monkeypatch)

    assert utils.emoji_text_width("a \U0001F33F b", FONT) == round(FONT.getlength("a b"))


@pytest.mark.skipif(utils.color_emoji_font() is None, reason="Noto Color Emoji not installed")
def test_emoji_text_runs_keep_emoji_sequences_with_emoji_font():
    text = "Go \U0001F44D\U0001F3FD team \U0001F3F3️‍\U0001F308 中!"

    runs = utils.emoji_text_runs(text, FONT)

    assert runs == [
        ("Go ", False),
        ("\U0001F44D\U0001F3FD", True),
        (" team ", False),
        ("\U0001F3F3️‍\U0001F308", True),
        (" !", False),
    ]


@pytest.mark.skipif(utils.color_emoji_font() is None, reason="Noto Color Emoji not installed")
def test_draw_emoji_text_pastes_color_emoji_within_line_height():
    from PIL import Image

    img = Image.new("RGB", (200, 40), (0, 0, 0))
    utils.draw_emoji_text(img, (0, 0), "\U0001F34B lemon", FONT, (255, 255, 255))

    ascent, descent = FONT.getmetrics()
    colored = [
        (x, y)
        for y in range(img.height)
        for x in range(img.width)
        if len(set(img.getpixel((x, y)))) > 1
    ]
    assert colored
    assert max(y for _, y in colored) < ascent + descent
    assert utils.wrap_emoji_text("\U0001F34B lemon", FONT, 1000) == ["\U0001F34B lemon"]
