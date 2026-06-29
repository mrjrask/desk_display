from PIL import Image

from screens import world_cup_scoreboard


class DummyDisplay:
    def __init__(self):
        self.images = []

    def image(self, img):
        self.images.append(img)


def test_scroll_display_scrolls_through_scoreboard_twice_with_continuous_content(monkeypatch):
    calls = []

    def fake_scroll_vertical_content(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(world_cup_scoreboard, "scroll_vertical_content", fake_scroll_vertical_content)

    img = Image.new(
        "RGB",
        (world_cup_scoreboard.WIDTH, world_cup_scoreboard.HEIGHT * 2),
        (0, 0, 0),
    )
    world_cup_scoreboard._scroll_display(object(), img)

    assert len(calls) == 1
    assert world_cup_scoreboard.SCROLL_REPEAT_COUNT == 2
    assert calls[0]["content_height"] == img.height * world_cup_scoreboard.SCROLL_REPEAT_COUNT


def test_single_game_scoreboard_displays_without_scrolling(monkeypatch):
    def fail_scroll(*args, **kwargs):
        raise AssertionError("single-game World Cup scoreboard should not scroll")

    full_img = Image.new(
        "RGB",
        (world_cup_scoreboard.WIDTH, world_cup_scoreboard.HEIGHT * 2),
        (0, 0, 0),
    )
    display = DummyDisplay()

    monkeypatch.setattr(world_cup_scoreboard, "_render_scoreboard", lambda games: full_img)
    monkeypatch.setattr(world_cup_scoreboard, "_scroll_display", fail_scroll)
    monkeypatch.setattr(world_cup_scoreboard.time, "sleep", lambda duration: None)

    result = world_cup_scoreboard.render_world_cup_scoreboard(display, [{}])

    assert result.displayed is True
    assert display.images == [result.image]
    assert result.image.size == (world_cup_scoreboard.WIDTH, world_cup_scoreboard.HEIGHT)


def test_multi_game_second_scroll_cycle_replaces_title_with_line(monkeypatch):
    calls = []
    rendered_styles = []

    def fake_render_scoreboard(games, *, title_style="title"):
        rendered_styles.append(title_style)
        color = (255, 255, 255) if title_style == "title" else (45, 45, 45)
        return Image.new(
            "RGB",
            (world_cup_scoreboard.WIDTH, world_cup_scoreboard.HEIGHT + 1),
            color,
        )

    def fake_scroll_display(display, img, repeat_images=None):
        calls.append({"img": img, "repeat_images": repeat_images})

    monkeypatch.setattr(world_cup_scoreboard, "_render_scoreboard", fake_render_scoreboard)
    monkeypatch.setattr(world_cup_scoreboard, "_scroll_display", fake_scroll_display)

    world_cup_scoreboard.render_world_cup_scoreboard(object(), [{}, {}])

    assert rendered_styles == ["title", "line"]
    assert len(calls) == 1
    repeat_images = calls[0]["repeat_images"]
    assert len(repeat_images) == 2
    assert repeat_images[0] is calls[0]["img"]
    assert repeat_images[0].getpixel((0, 0)) == (255, 255, 255)
    assert repeat_images[1].getpixel((0, 0)) == (45, 45, 45)
