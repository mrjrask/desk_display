from PIL import Image

from screens.draw_quad import _TileSpec, draw_quad_screen


class _DisplayStub:
    def __init__(self):
        self.frames = []
        self.show_calls = 0

    def image(self, img):
        self.frames.append(img.copy())

    def show(self):
        self.show_calls += 1

    def wait_for_skip(self, _duration):
        return False


def test_draw_quad_animates_when_tile_has_multiple_frames(monkeypatch):
    display = _DisplayStub()
    monkeypatch.setattr("screens.draw_quad.SCREEN_DELAY", 0.03)

    red = Image.new("RGB", (20, 20), "red")
    blue = Image.new("RGB", (20, 20), "blue")
    tiles = [_TileSpec("animated", lambda: [red, blue])]

    result = draw_quad_screen(display, tiles, transition=True)

    assert result.displayed is True
    assert result.consumed_delay is True
    assert len(display.frames) >= 2
    assert display.show_calls >= 2


def test_draw_quad_static_tile_does_not_consume_delay(monkeypatch):
    display = _DisplayStub()
    monkeypatch.setattr("screens.draw_quad.SCREEN_DELAY", 0.03)

    green = Image.new("RGB", (20, 20), "green")
    tiles = [_TileSpec("static", lambda: green)]

    result = draw_quad_screen(display, tiles, transition=True)

    assert result.displayed is True
    assert result.consumed_delay is False
    assert len(display.frames) == 1
    assert display.show_calls == 1
