from PIL import Image, ImageDraw

import screens.mlb_schedule as mlb_schedule


def test_should_show_team_logo_boxscore_for_supported_display_profiles(monkeypatch):
    monkeypatch.setattr(mlb_schedule.config, "get_display_profile_id", lambda: "display_hat_mini")

    assert mlb_schedule._should_show_team_logo_boxscore("cubs live")
    assert mlb_schedule._should_show_team_logo_boxscore("sox last")
    assert not mlb_schedule._should_show_team_logo_boxscore("cubs next")

    monkeypatch.setattr(mlb_schedule.config, "get_display_profile_id", lambda: "hyperpixel4")
    assert mlb_schedule._should_show_team_logo_boxscore("cubs live")

    monkeypatch.setattr(mlb_schedule.config, "get_display_profile_id", lambda: "other")
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


def test_draw_box_score_reserves_flag_block_for_live_layout(monkeypatch):
    captured = {}

    def _fake_draw_table(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mlb_schedule, "_draw_boxscore_table", _fake_draw_table)

    game = {
        "linescore": {
            "inningState": "Top",
            "currentInningOrdinal": "3rd",
            "teams": {
                "away": {"hits": 1, "errors": 0},
                "home": {"hits": 2, "errors": 0},
            },
        },
        "teams": {
            "away": {"score": 2, "team": {"name": "Chicago Cubs"}},
            "home": {"score": 3, "team": {"name": "St. Louis Cardinals"}},
        },
    }

    mlb_schedule.draw_box_score(None, game, title="Cubs Live...", screen_id="cubs live")

    assert captured["reserve_flag_block"] is True



def test_draw_box_score_centers_content_vertically_for_live_screens(monkeypatch):
    captured = {}

    def _fake_draw_table(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mlb_schedule, "_draw_boxscore_table", _fake_draw_table)

    game = {
        "linescore": {
            "inningState": "Top",
            "currentInningOrdinal": "3rd",
            "teams": {
                "away": {"hits": 1, "errors": 0},
                "home": {"hits": 2, "errors": 0},
            },
        },
        "teams": {
            "away": {"score": 2, "team": {"name": "Chicago Cubs"}},
            "home": {"score": 3, "team": {"name": "St. Louis Cardinals"}},
        },
    }

    mlb_schedule.draw_box_score(None, game, title="Sox Live...", screen_id="sox live")
    assert captured["center_content_vertically"] is True

    mlb_schedule.draw_box_score(None, game, title="Sox Live...", screen_id="sox live")
    assert captured["center_content_vertically"] is True

    mlb_schedule.draw_box_score(None, game, title="Live Game...", screen_id="cubs next")
    assert captured["center_content_vertically"] is False


def test_draw_last_game_centers_content_vertically_for_sox_last(monkeypatch):
    captured = {}

    def _fake_draw_table(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mlb_schedule, "_draw_boxscore_table", _fake_draw_table)

    game = {
        "officialDate": "2026-05-01",
        "teams": {
            "away": {"score": 4, "team": {"id": 145, "name": "Chicago White Sox"}},
            "home": {"score": 2, "team": {"id": 121, "name": "New York Mets"}},
        },
        "linescore": {
            "teams": {
                "away": {"hits": 7, "errors": 1},
                "home": {"hits": 5, "errors": 0},
            },
        },
    }

    mlb_schedule.draw_last_game(None, game, title="Last Sox game...", screen_id="sox last")
    assert captured["center_content_vertically"] is True

    mlb_schedule.draw_last_game(None, game, title="Last Cubs game...", screen_id="cubs last")
    assert captured["center_content_vertically"] is False
