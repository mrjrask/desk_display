from pathlib import Path

from PIL import ImageFont

import utils

FONT_PATH = str(Path(__file__).resolve().parents[1] / "fonts" / "DejaVuSans.ttf")


def test_clone_font_reuses_cached_font_for_same_path_and_size(monkeypatch):
    utils._FONT_CLONE_CACHE.clear()
    real_truetype = ImageFont.truetype
    calls = []

    def tracked_truetype(path, size, *args, **kwargs):
        calls.append((path, size))
        return real_truetype(path, size, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", tracked_truetype)

    base_font = real_truetype(FONT_PATH, 16)

    first = utils.clone_font(base_font, 24)
    second = utils.clone_font(base_font, 24)

    assert first is second
    assert calls == [(FONT_PATH, 24)]


def test_clone_font_loads_each_distinct_size_once(monkeypatch):
    utils._FONT_CLONE_CACHE.clear()
    real_truetype = ImageFont.truetype
    calls = []

    def tracked_truetype(path, size, *args, **kwargs):
        calls.append((path, size))
        return real_truetype(path, size, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", tracked_truetype)

    base_font = real_truetype(FONT_PATH, 16)

    sizes = [12, 14, 16, 12, 14, 16]
    fonts = [utils.clone_font(base_font, size) for size in sizes]

    assert len(calls) == 3
    assert fonts[0] is fonts[3]
    assert fonts[1] is fonts[4]
    assert fonts[2] is fonts[5]
