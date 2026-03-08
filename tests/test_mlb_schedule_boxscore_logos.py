from PIL import Image, ImageDraw

import screens.mlb_schedule as mlb_schedule


def test_should_show_team_logo_boxscore_for_hyperpixel_4_square_only(monkeypatch):
    monkeypatch.setattr(mlb_schedule, "is_hyperpixel_4_square_layout", lambda: True)

    assert mlb_schedule._should_show_team_logo_boxscore("cubs live")
    assert mlb_schedule._should_show_team_logo_boxscore("sox last")
    assert not mlb_schedule._should_show_team_logo_boxscore("cubs next")

    monkeypatch.setattr(mlb_schedule, "is_hyperpixel_4_square_layout", lambda: False)
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




def test_draw_left_team_cell_logo_is_centered_when_replacing_abbreviation(monkeypatch):
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
        replace_abbreviation_with_logo=True,
    )

    # With abbreviation replacement enabled, the logo should be centered in the cell.
    assert img.getpixel((16, 12)) != (0, 0, 0)
    assert img.getpixel((12, 12)) == (0, 0, 0)

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
    assert captured["center_ignores_reserved_flag_block"] is True


def test_draw_box_score_uses_warmup_status_over_inning_state(monkeypatch):
    captured = {}

    def _fake_draw_table(*args, **kwargs):
        captured["bottom"] = args[11]

    monkeypatch.setattr(mlb_schedule, "_draw_boxscore_table", _fake_draw_table)

    game = {
        "status": {"detailedState": "Warmup"},
        "linescore": {
            "inningState": "Top",
            "currentInningOrdinal": "1st",
            "teams": {
                "away": {"hits": 0, "errors": 0},
                "home": {"hits": 0, "errors": 0},
            },
        },
        "teams": {
            "away": {"score": 0, "team": {"name": "Chicago Cubs"}},
            "home": {"score": 0, "team": {"name": "Chicago White Sox"}},
        },
    }

    mlb_schedule.draw_box_score(None, game, title="Cubs Live...", screen_id="cubs live")

    assert captured["bottom"] == "Warmup"


def test_draw_box_score_normalizes_hyphenated_warmup_status(monkeypatch):
    captured = {}

    def _fake_draw_table(*args, **kwargs):
        captured["bottom"] = args[11]

    monkeypatch.setattr(mlb_schedule, "_draw_boxscore_table", _fake_draw_table)

    game = {
        "status": {"detailedState": "Pre-Game Warmup"},
        "linescore": {
            "inningState": "Top",
            "currentInningOrdinal": "1st",
            "teams": {
                "away": {"hits": 0, "errors": 0},
                "home": {"hits": 0, "errors": 0},
            },
        },
        "teams": {
            "away": {"score": 0, "team": {"name": "Chicago Cubs"}},
            "home": {"score": 0, "team": {"name": "Chicago White Sox"}},
        },
    }

    mlb_schedule.draw_box_score(None, game, title="Sox Live...", screen_id="sox live")

    assert captured["bottom"] == "Warmup"



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
    assert captured["center_ignores_reserved_flag_block"] is True

    mlb_schedule.draw_box_score(None, game, title="Cubs Live...", screen_id="cubs live")
    assert captured["center_content_vertically"] is True
    assert captured["center_ignores_reserved_flag_block"] is True

    mlb_schedule.draw_box_score(None, game, title="Live Game...", screen_id="cubs next")
    assert captured["center_content_vertically"] is False
    assert captured["center_ignores_reserved_flag_block"] is False


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
    assert captured["center_ignores_reserved_flag_block"] is True

    mlb_schedule.draw_last_game(None, game, title="Last Cubs game...", screen_id="cubs last")
    assert captured["center_content_vertically"] is False
    assert captured["center_ignores_reserved_flag_block"] is False



def test_draw_last_game_moves_cubs_result_flag_inline_on_hyperpixel4(monkeypatch):
    captured = {}

    def _fake_draw_table(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mlb_schedule, "_draw_boxscore_table", _fake_draw_table)
    monkeypatch.setattr(mlb_schedule.config, "get_display_profile_id", lambda: "hyperpixel4")
    monkeypatch.setattr(mlb_schedule, "is_hyperpixel_4_square_layout", lambda: False)

    game = {
        "officialDate": "2026-05-01",
        "teams": {
            "away": {"score": 5, "team": {"id": 121, "name": "New York Mets"}},
            "home": {"score": 2, "team": {"id": 112, "name": "Chicago Cubs"}},
        },
        "linescore": {
            "teams": {
                "away": {"hits": 7, "errors": 1},
                "home": {"hits": 5, "errors": 0},
            },
        },
    }

    mlb_schedule.draw_last_game(None, game, title="Last Cubs game...", screen_id="cubs last")

    assert captured["reserve_flag_block"] is False
    assert captured["winner_flag"] is None
    assert captured["inline_team_id"] == int(mlb_schedule.MLB_CUBS_TEAM_ID)
    assert captured["inline_winner_flag"] == "L"

def test_centered_boxscore_accounts_for_header_height(monkeypatch):
    img = Image.new("RGB", (mlb_schedule.WIDTH, mlb_schedule.HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    geometry = {
        "hdr_h": 10,
        "grid_top": 20,
        "row_h": 30,
        "team_w": 80,
        "square": 30,
        "xs": [0, 80, 110, 140, 170],
        "grid_w": 170,
        "grid_h": 60,
    }
    monkeypatch.setattr(mlb_schedule, "_compute_table_geometry", lambda *args, **kwargs: geometry)
    monkeypatch.setattr(mlb_schedule, "_draw_title_with_bold_result", lambda *args, **kwargs: (0, 20))
    monkeypatch.setattr(mlb_schedule, "_should_show_team_logo_boxscore", lambda *args, **kwargs: False)

    header_row_y = {}

    def _capture_bbox_center(_draw, x, y, w, h, text, font, fill=(255, 255, 255)):
        if text == "R" and "y" not in header_row_y:
            header_row_y["y"] = y

    monkeypatch.setattr(mlb_schedule, "_bbox_center", _capture_bbox_center)

    bottom_text = "7:10 PM"
    _, t, _, b = draw.textbbox((0, 0), bottom_text, font=mlb_schedule.FONT_DATE_SPORTS)
    bh = b - t
    bottom_y = mlb_schedule.HEIGHT - bh - mlb_schedule.BOTTOM_MARGIN
    flag_block_h = mlb_schedule.SMALL_RESULT_FLAG_H + mlb_schedule.FLAG_BLOCK_PAD
    content_top = 20 + mlb_schedule.TITLE_TO_HEADER_GAP
    content_bottom = bottom_y - flag_block_h
    scoreboard_block_h = (
        geometry["hdr_h"] + mlb_schedule.HEADER_GAP + geometry["grid_h"]
    )
    expected_grid_top = max(
        geometry["grid_top"],
        content_top
        + max(0, (content_bottom - content_top - scoreboard_block_h) // 2)
        + geometry["hdr_h"]
        + mlb_schedule.HEADER_GAP,
    )

    mlb_schedule._draw_boxscore_table(
        img,
        draw,
        "Sox Live...",
        "SOX",
        1,
        2,
        0,
        "DET",
        0,
        1,
        0,
        bottom_text,
        reserve_flag_block=True,
        live=True,
        center_content_vertically=True,
    )

    assert header_row_y["y"] + mlb_schedule.HEADER_GAP + geometry["hdr_h"] == expected_grid_top
