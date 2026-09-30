from PIL import ImageFont

import config
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
