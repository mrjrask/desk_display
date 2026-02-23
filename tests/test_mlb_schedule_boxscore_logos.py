from PIL import Image, ImageDraw

import screens.mlb_schedule as mlb_schedule


def test_should_show_team_logo_boxscore_only_for_display_hat_mini(monkeypatch):
    monkeypatch.setattr(mlb_schedule.config, "get_display_profile_id", lambda: "display_hat_mini")

    assert mlb_schedule._should_show_team_logo_boxscore("cubs live")
    assert mlb_schedule._should_show_team_logo_boxscore("sox last")
    assert not mlb_schedule._should_show_team_logo_boxscore("cubs next")

    monkeypatch.setattr(mlb_schedule.config, "get_display_profile_id", lambda: "hyperpixel4")
    assert not mlb_schedule._should_show_team_logo_boxscore("cubs live")


def test_draw_left_team_cell_with_logo_stays_inside_cell(monkeypatch):
    img = Image.new("RGB", (80, 40), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    logo = Image.new("RGBA", (12, 12), (255, 0, 0, 255))
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: logo)

    mlb_schedule._draw_left_team_cell_with_logo(
        img,
        draw,
        team={"name": "Cubs"},
        abbr="CHICAGO",
        x=10,
        y=10,
        w=24,
        h=16,
        font=mlb_schedule.FONT_TEAM_SPORTS,
    )

    # Ensure logo/text introduced pixels inside the target cell.
    assert any(img.getpixel((x, y)) != (0, 0, 0) for x in range(10, 34) for y in range(10, 26))

    # Ensure helper never paints past the right edge of the target cell.
    assert all(img.getpixel((x, y)) == (0, 0, 0) for x in range(34, 80) for y in range(0, 40))
