from PIL import Image

import utils


def test_load_weather_icon_reuses_cached_image_without_reopening(monkeypatch):
    utils._WEATHER_ICON_CACHE.clear()
    original_open = utils.Image.open
    open_calls = []

    def counting_open(path, *args, **kwargs):
        open_calls.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(utils.Image, "open", counting_open)

    first = utils.load_weather_icon("Clear", "clear", True, 24)
    assert isinstance(first, Image.Image)
    first.putpixel((0, 0), (255, 0, 0, 255))

    second = utils.load_weather_icon("Clear", "clear", True, 24)

    assert isinstance(second, Image.Image)
    assert len(open_calls) == 1
    assert second is not first
    assert second.getpixel((0, 0)) != (255, 0, 0, 255)
