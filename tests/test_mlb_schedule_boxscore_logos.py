from PIL import Image, ImageDraw
import pytest

import screens.mlb_schedule as mlb_schedule


def test_extract_probable_pitcher_falls_back_to_game_level_probables():
    game = {
        "probablePitchers": {
            "away": {"id": 12345, "fullName": "Jane Smith", "record": "4-2"},
        }
    }
    team_block = {"team": {"name": "Chicago Cubs"}}

    name, record, image_url = mlb_schedule._extract_probable_pitcher(team_block, game=game, side="away")

    assert name == "Jane Smith"
    assert record == "(4-2)"
    assert "/12345/" in image_url


def test_extract_probable_pitcher_builds_record_from_stat_splits():
    team_block = {
        "probablePitcher": {
            "person": {"id": 777, "fullName": "John Doe"},
            "stats": {"splits": [{"stat": {"wins": 6, "losses": 1}}]},
        }
    }

    name, record, image_url = mlb_schedule._extract_probable_pitcher(team_block)

    assert name == "John Doe"
    assert record == "(6-1)"
    assert "/777/" in image_url



def test_extract_probable_pitcher_formats_record_before_era():
    team_block = {
        "probablePitcher": {
            "person": {"id": 888, "fullName": "Jane Doe"},
            "stats": {"splits": [{"stat": {"wins": 5, "losses": 3, "era": "3.21"}}]},
        }
    }

    name, record, image_url = mlb_schedule._extract_probable_pitcher(team_block)

    assert name == "Jane Doe"
    assert record == "(5-3) 3.21 ERA"
    assert "/888/" in image_url


def test_extract_probable_pitcher_reads_season_stats_record_and_era():
    team_block = {
        "probablePitcher": {
            "person": {"id": 999, "fullName": "Alex Sample"},
            "seasonStats": {"wins": 7, "losses": 4, "era": "2.98"},
        }
    }

    name, record, image_url = mlb_schedule._extract_probable_pitcher(team_block)

    assert name == "Alex Sample"
    assert record == "(7-4) 2.98 ERA"
    assert "/999/" in image_url


def test_extract_probable_pitcher_handles_stats_list_and_split_record_era():
    team_block = {
        "probablePitcher": {
            "person": {"id": 111, "fullName": "List Stats"},
            "stats": [{"splits": [{"stat": {"record": "8-2", "era": "2.67"}}]}],
        }
    }

    name, record, image_url = mlb_schedule._extract_probable_pitcher(team_block)

    assert name == "List Stats"
    assert record == "(8-2) 2.67 ERA"
    assert "/111/" in image_url

def test_fit_image_within_box_preserves_headshot_aspect_ratio():
    wide = Image.new("RGBA", (120, 60), (255, 0, 0, 255))

    fitted = mlb_schedule._fit_image_within_box(wide, 40)

    assert fitted.size == (40, 20)


def test_should_show_team_logo_boxscore_for_supported_mlb_screens():
    assert mlb_schedule._should_show_team_logo_boxscore("cubs live")
    assert mlb_schedule._should_show_team_logo_boxscore("sox last")
    assert not mlb_schedule._should_show_team_logo_boxscore("cubs next")


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


def test_draw_box_score_hides_outs_during_warmup(monkeypatch):
    captured = {}

    def _fake_draw_table(*args, **kwargs):
        captured["bottom"] = args[11]

    monkeypatch.setattr(mlb_schedule, "_draw_boxscore_table", _fake_draw_table)

    game = {
        "status": {"detailedState": "Warmup"},
        "linescore": {
            "inningState": "Top",
            "currentInningOrdinal": "1st",
            "outs": 2,
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


def test_draw_box_score_hides_outs_during_delay(monkeypatch):
    captured = {}

    def _fake_draw_table(*args, **kwargs):
        captured["bottom"] = args[11]

    monkeypatch.setattr(mlb_schedule, "_draw_boxscore_table", _fake_draw_table)

    game = {
        "status": {"detailedState": "Rain Delay"},
        "linescore": {
            "inningState": "Top",
            "currentInningOrdinal": "5th",
            "outs": 1,
            "teams": {
                "away": {"hits": 4, "errors": 0},
                "home": {"hits": 3, "errors": 1},
            },
        },
        "teams": {
            "away": {"score": 2, "team": {"name": "Chicago Cubs"}},
            "home": {"score": 1, "team": {"name": "Chicago White Sox"}},
        },
    }

    mlb_schedule.draw_box_score(None, game, title="Sox Live...", screen_id="sox live")

    assert captured["bottom"] == "Top 5th"



def test_live_base_state_reads_offense_base_keys():
    linescore = {
        "offense": {
            "first": {"id": 1},
            "second": None,
            "third": {"id": 3},
        }
    }

    assert mlb_schedule._live_base_state(linescore) == (True, False, True)


def test_live_base_state_accepts_bases_occupied_fallback():
    linescore = {
        "offense": {
            "basesOccupied": [2],
        }
    }

    assert mlb_schedule._live_base_state(linescore) == (False, True, False)


def test_draw_box_score_passes_live_bases_for_live_screens(monkeypatch):
    captured = {}

    def _fake_draw_table(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mlb_schedule, "_draw_boxscore_table", _fake_draw_table)

    game = {
        "linescore": {
            "inningState": "Top",
            "currentInningOrdinal": "3rd",
            "outs": 1,
            "offense": {"first": {"id": 10}, "second": None, "third": {"id": 30}},
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
    assert captured["live_bases"] == (True, False, True)

    mlb_schedule.draw_box_score(None, game, title="Live Game...", screen_id="cubs next")
    assert captured["live_bases"] is None


def test_center_bottom_text_with_live_bases_uses_space_gap(monkeypatch):
    captured = {}
    img = Image.new("RGB", (mlb_schedule.WIDTH, mlb_schedule.HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    text = "Top 5th, 2 outs"
    font = mlb_schedule.FONT_DATE_SPORTS

    def _capture_indicator(_draw, *, x, y, size, on_first, on_second, on_third):
        captured.update({"x": x, "y": y, "size": size})

    monkeypatch.setattr(mlb_schedule, "_draw_live_basepaths_indicator", _capture_indicator)

    mlb_schedule._center_bottom_text_with_live_bases(
        draw,
        text,
        font,
        live_bases=(True, False, True),
    )

    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    base_size = max(3, int(round(th * 0.35)))
    step = max(3, base_size)
    half = max(2, step // 2)
    indicator_w = (2 * step) + (2 * half)
    sl, _, sr, _ = draw.textbbox((0, 0), "   ", font=font)
    min_gap = sr - sl
    group_w = tw + min_gap + indicator_w
    group_x = max(0, (mlb_schedule.WIDTH - group_w) // 2)
    text_right = (group_x - l) + tw

    assert captured["x"] >= text_right + min_gap


def test_center_bottom_text_with_live_bases_aligns_indicator_vertically(monkeypatch):
    captured = {}
    img = Image.new("RGB", (mlb_schedule.WIDTH, mlb_schedule.HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    text = "Bot 7th, 1 out"
    font = mlb_schedule.FONT_DATE_SPORTS

    def _capture_indicator(_draw, *, x, y, size, on_first, on_second, on_third):
        captured.update({"x": x, "y": y, "size": size})

    monkeypatch.setattr(mlb_schedule, "_draw_live_basepaths_indicator", _capture_indicator)

    mlb_schedule._center_bottom_text_with_live_bases(
        draw,
        text,
        font,
        live_bases=(False, True, False),
    )

    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    step = max(3, captured["size"])
    half = max(2, step // 2)
    indicator_h = step + (2 * half)

    sl, _, sr, _ = draw.textbbox((0, 0), "   ", font=font)
    gap = sr - sl
    indicator_w = (2 * step) + (2 * half)
    group_w = tw + gap + indicator_w
    group_x = max(0, (mlb_schedule.WIDTH - group_w) // 2)
    text_x = group_x - l
    text_y = mlb_schedule.HEIGHT - th - mlb_schedule.BOTTOM_MARGIN - t
    _, text_top, _, text_bottom = draw.textbbox((text_x, text_y), text, font=font)
    text_center_y = (text_top + text_bottom) // 2

    indicator_center_y = captured["y"] + (indicator_h // 2)
    assert abs(indicator_center_y - text_center_y) <= 1


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


def test_draw_series_screen_uses_result_icon_on_cubs_next_series(monkeypatch):
    opened_paths = []

    def _fake_open(path):
        opened_paths.append(path)
        return Image.new("RGBA", (10, 10), (255, 255, 255, 255))

    monkeypatch.setattr(mlb_schedule.Image, "open", _fake_open)
    monkeypatch.setattr(mlb_schedule.os.path, "exists", lambda _path: True)

    game = {
        "officialDate": "2026-05-01",
        "status": {"detailedState": "Final"},
        "teams": {
            "away": {"score": 1, "team": {"id": 121, "name": "New York Mets"}},
            "home": {"score": 4, "team": {"id": 112, "name": "Chicago Cubs"}},
        },
    }

    mlb_schedule.draw_series_screen(
        None,
        [game],
        title="Cubs Next Series",
        screen_id="cubs next series",
    )

    assert any(path.endswith("mlb/W.png") for path in opened_paths)


def test_draw_series_screen_does_not_use_result_icon_on_sox_next_series(monkeypatch):
    opened_paths = []

    def _fake_open(path):
        opened_paths.append(path)
        return Image.new("RGBA", (10, 10), (255, 255, 255, 255))

    monkeypatch.setattr(mlb_schedule.Image, "open", _fake_open)
    monkeypatch.setattr(mlb_schedule.os.path, "exists", lambda _path: True)

    game = {
        "officialDate": "2026-05-01",
        "status": {"detailedState": "Final"},
        "teams": {
            "away": {"score": 1, "team": {"id": 121, "name": "New York Mets"}},
            "home": {"score": 4, "team": {"id": 145, "name": "Chicago White Sox"}},
        },
    }

    mlb_schedule.draw_series_screen(
        None,
        [game],
        title="Sox Next Series",
        screen_id="sox next series",
    )

    assert not any(path.endswith("/mlb/W.png") or path.endswith("/mlb/L.png") for path in opened_paths)




def test_draw_series_screen_renders_all_games_for_sox_series(monkeypatch):
    drawn_rows = []
    original_text = ImageDraw.ImageDraw.text

    def _capture_text(self, xy, text, *args, **kwargs):
        if isinstance(text, str) and text.startswith("SOXROW "):
            drawn_rows.append(text)
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_series_final_result_parts", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(
        mlb_schedule,
        "_series_line",
        lambda game, _focus_id: f"SOXROW {game.get('gamePk')}",
    )

    def _game(game_pk: int):
        return {
            "gamePk": game_pk,
            "status": {"detailedState": "Scheduled"},
            "teams": {
                "away": {"team": {"id": 121, "name": "New York Mets"}},
                "home": {"team": {"id": 145, "name": "Chicago White Sox"}},
            },
        }

    games = [_game(idx) for idx in range(1, 5)]
    mlb_schedule.draw_series_screen(
        None,
        games,
        title="Sox Next Series",
        screen_id="sox next series",
    )

    assert drawn_rows == ["SOXROW 1", "SOXROW 2", "SOXROW 3", "SOXROW 4"]
def test_draw_series_screen_renders_more_than_four_games(monkeypatch):
    drawn_rows = []
    original_text = ImageDraw.ImageDraw.text

    def _capture_text(self, xy, text, *args, **kwargs):
        if isinstance(text, str) and text.startswith("ROW "):
            drawn_rows.append(text)
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_series_final_result_parts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mlb_schedule,
        "_series_line",
        lambda game, _focus_id: f"ROW {game.get('gamePk')}",
    )

    def _game(game_pk: int):
        return {
            "gamePk": game_pk,
            "status": {"detailedState": "Scheduled"},
            "teams": {
                "away": {"team": {"id": 121, "name": "New York Mets"}},
                "home": {"team": {"id": 112, "name": "Chicago Cubs"}},
            },
        }

    games = [_game(idx) for idx in range(1, 6)]
    mlb_schedule.draw_series_screen(
        None,
        games,
        title="Cubs Next Series",
        screen_id="cubs next series",
    )

    assert drawn_rows == ["ROW 1", "ROW 2", "ROW 3", "ROW 4", "ROW 5"]


def test_draw_series_screen_shrinks_row_font_to_fit_three_game_sox_series(monkeypatch):
    drawn_rows = []
    original_text = ImageDraw.ImageDraw.text
    original_textsize = ImageDraw.ImageDraw.textsize

    class _FakeFont:
        def __init__(self, size):
            self.size = size

        def font_variant(self, size):
            return _FakeFont(size)

    def _capture_text(self, xy, text, *args, **kwargs):
        if isinstance(text, str) and text.startswith("FITROW "):
            drawn_rows.append(text)
        if isinstance(kwargs.get("font"), _FakeFont):
            return None
        return original_text(self, xy, text, *args, **kwargs)

    def _fake_textsize(self, text, font=None, *args, **kwargs):
        if isinstance(font, _FakeFont):
            size = int(getattr(font, "size", 30) or 30)
            if isinstance(text, str) and text == "Tonight • 7 PM":
                # Keep row text too tall until the font gets small enough.
                text_h = 40 if size >= 30 else 32 if size >= 24 else 24
                return (120, text_h)
            return (max(20, int(len(str(text)) * (size * 0.55))), max(12, int(size * 0.9)))
        if isinstance(text, str) and text == "Tonight • 7 PM":
            size = int(getattr(font, "size", 30) or 30)
            return (120, max(12, int(size * 0.9)))
        return original_textsize(self, text, font=font, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)
    monkeypatch.setattr(ImageDraw.ImageDraw, "textsize", _fake_textsize)
    monkeypatch.setattr(mlb_schedule, "FONT_DATE_SPORTS", _FakeFont(30))
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_series_final_result_parts", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(mlb_schedule, "HEIGHT", 265)
    monkeypatch.setattr(
        mlb_schedule,
        "_series_line",
        lambda game, _focus_id: f"FITROW {game.get('gamePk')}",
    )

    def _game(game_pk: int):
        return {
            "gamePk": game_pk,
            "status": {"detailedState": "Scheduled"},
            "teams": {
                "away": {"team": {"id": 121, "name": "Boston Red Sox"}},
                "home": {"team": {"id": 145, "name": "Chicago White Sox"}},
            },
        }

    games = [_game(idx) for idx in range(1, 4)]
    mlb_schedule.draw_series_screen(
        None,
        games,
        title="Sox Next Home Series",
        screen_id="sox next home series",
    )

    assert drawn_rows == ["FITROW 1", "FITROW 2", "FITROW 3"]


def test_draw_series_screen_distributes_rows_vertically_on_hyperpixel(monkeypatch):
    row_positions = []
    original_text = ImageDraw.ImageDraw.text

    def _capture_text(self, xy, text, *args, **kwargs):
        if isinstance(text, str) and text.startswith("SPREADROW "):
            row_positions.append(xy[1])
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)
    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: True)
    monkeypatch.setattr(mlb_schedule.config, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_series_final_result_parts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mlb_schedule,
        "_series_line",
        lambda game, _focus_id: f"SPREADROW {game.get('gamePk')}",
    )

    def _game(game_pk: int):
        return {
            "gamePk": game_pk,
            "status": {"detailedState": "Scheduled"},
            "teams": {
                "away": {"team": {"id": 121, "name": "Boston Red Sox"}},
                "home": {"team": {"id": 145, "name": "Chicago White Sox"}},
            },
        }

    games = [_game(idx) for idx in range(1, 4)]
    mlb_schedule.draw_series_screen(
        None,
        games,
        title="Sox Next Series",
        screen_id="sox next series",
    )

    assert len(row_positions) == 3
    first_gap = row_positions[1] - row_positions[0]
    second_gap = row_positions[2] - row_positions[1]
    assert first_gap == second_gap
    assert first_gap >= 28


def test_draw_series_screen_centers_three_blocks_on_display_hat_mini(monkeypatch):
    captured = {}
    original_text = ImageDraw.ImageDraw.text

    def _capture_text(self, xy, text, *args, **kwargs):
        if text == "Sox Next Series":
            captured["title_y"] = xy[1]
        elif text == "Boston Red Sox":
            captured["opponent_y"] = xy[1]
        elif isinstance(text, str) and text.startswith("BLOCKROW "):
            captured.setdefault("row_ys", []).append(xy[1])
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)
    monkeypatch.setattr(mlb_schedule, "WIDTH", 320)
    monkeypatch.setattr(mlb_schedule, "HEIGHT", 240)
    monkeypatch.setattr(mlb_schedule.config, "get_display_profile_id", lambda: mlb_schedule.DISPLAY_PROFILE_DISPLAY_HAT_MINI)
    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(mlb_schedule.config, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_schedule, "wrap_text", lambda text, *_args, **_kwargs: [text])
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_series_final_result_parts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mlb_schedule,
        "_series_line",
        lambda game, _focus_id: f"BLOCKROW {game.get('gamePk')}",
    )

    def _game(game_pk: int):
        return {
            "gamePk": game_pk,
            "status": {"detailedState": "Scheduled"},
            "teams": {
                "away": {"team": {"id": 111, "name": "Boston Red Sox"}},
                "home": {"team": {"id": 145, "name": "Chicago White Sox"}},
            },
        }

    games = [_game(idx) for idx in range(1, 4)]
    mlb_schedule.draw_series_screen(
        None,
        games,
        title="Sox Next Series",
        screen_id="sox next series",
    )

    assert captured["title_y"] == 0
    assert len(captured.get("row_ys", [])) == 3

    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    prefix_font = mlb_schedule.FONT_TEAM_SPORTS.font_variant(
        size=max(8, int(round(getattr(mlb_schedule.FONT_TEAM_SPORTS, "size", 20) * 0.6)))
    )
    opponent_h = max(
        measure_draw.textsize("vs. ", font=prefix_font)[1],
        measure_draw.textsize("Boston Red Sox", font=mlb_schedule.FONT_TEAM_SPORTS)[1],
    )
    row_text_h = measure_draw.textsize("Tonight • 7 PM", font=mlb_schedule.FONT_DATE_SPORTS)[1]

    block_h_title = captured["opponent_y"] + opponent_h - captured["title_y"]
    logo_h = min(mlb_schedule.standard_next_game_logo_height(mlb_schedule.HEIGHT), max(16, mlb_schedule.HEIGHT // 5))
    row_h = row_text_h + 1 + int(round(row_text_h * 0.25))
    block_h_games = 3 * row_h

    used_h = block_h_title + logo_h + block_h_games
    between_block_gap = max(0, (mlb_schedule.HEIGHT - mlb_schedule.BOTTOM_MARGIN - used_h) // 2)
    expected_first_row_y = (captured["opponent_y"] + opponent_h) + logo_h + (between_block_gap * 2)
    assert abs(captured["row_ys"][0] - expected_first_row_y) <= 1


def test_draw_series_screen_series_variants_scale_opponent_to_single_line(monkeypatch):
    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: True)
    monkeypatch.setattr(mlb_schedule.config, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    wrap_calls = []

    def _capture_wrap(text, font, width):
        wrap_calls.append(text)
        return [text]

    monkeypatch.setattr(mlb_schedule, "wrap_text", _capture_wrap)

    game = {
        "status": {"detailedState": "Scheduled"},
        "teams": {
            "away": {"team": {"id": 121, "name": "Boston Red Sox"}},
            "home": {"team": {"id": 145, "name": "Chicago White Sox"}},
        },
    }
    screen_ids = (
        "cubs current series",
        "cubs next series",
        "cubs next home series",
        "sox current series",
        "sox next series",
        "sox next home series",
    )

    for screen_id in screen_ids:
        mlb_schedule.draw_series_screen(
            None,
            [game],
            title=f"{screen_id.title()}",
            screen_id=screen_id,
        )

    assert wrap_calls == []


def test_series_line_fill_uses_live_scoreboard_color_for_in_progress_game():
    game = {
        "status": {
            "abstractGameState": "Live",
            "detailedState": "In Progress",
            "statusCode": "I",
        }
    }

    assert mlb_schedule._series_line_fill(game) == mlb_schedule.config.SCOREBOARD_IN_PROGRESS_SCORE_COLOR


def test_series_line_fill_uses_white_for_non_live_game():
    game = {
        "status": {
            "abstractGameState": "Final",
            "detailedState": "Final",
            "statusCode": "F",
        }
    }

    assert mlb_schedule._series_line_fill(game) == (255, 255, 255)


@pytest.mark.parametrize(
    ("scores", "expected_result", "expected_fill"),
    [
        ((5, 2), "W", (0, 180, 0)),
        ((1, 4), "L", (220, 0, 0)),
    ],
)
def test_draw_series_screen_sox_current_series_colors_result_letter(monkeypatch, scores, expected_result, expected_fill):
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)

    home_score, away_score = scores
    games = [
        {
            "officialDate": "2026-05-01",
            "status": {"abstractGameState": "Final", "detailedState": "Final", "statusCode": "F"},
            "teams": {
                "away": {"score": away_score, "team": {"id": 121, "name": "Boston Red Sox"}},
                "home": {"score": home_score, "team": {"id": 145, "name": "Chicago White Sox"}},
            },
        }
    ]

    text_calls = []
    original_text = ImageDraw.ImageDraw.text

    def _capture_text(self, xy, text, *args, **kwargs):
        text_calls.append((text, kwargs.get("fill")))
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)

    mlb_schedule.draw_series_screen(
        None,
        games,
        "Sox Current Series",
        screen_id="sox current series",
    )

    assert (expected_result, expected_fill) in text_calls

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


def test_draw_sports_screen_uses_postponed_bottom_label(monkeypatch):
    captured = {}

    def _fake_center_bottom_text(_draw, text, _font, *, margin, fill=(255, 255, 255)):
        captured["bottom"] = text

    monkeypatch.setattr(mlb_schedule, "_center_bottom_text", _fake_center_bottom_text)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)

    game = {
        "officialDate": "2026-05-01",
        "startTimeCentral": "6:40 PM",
        "status": {"detailedState": "Postponed", "abstractGameState": "Postponed", "statusCode": "PPD"},
        "teams": {
            "away": {"team": {"id": 112, "name": "Chicago Cubs"}},
            "home": {"team": {"id": 111, "name": "Boston Red Sox"}},
        },
    }

    mlb_schedule.draw_sports_screen(None, game, "Next Game...", screen_id="cubs next")

    assert captured["bottom"] == "Postponed"


def test_draw_sports_screen_cubs_next_uses_current_series_title_spacing_on_square_layout(monkeypatch):
    captured = {}
    original_text = ImageDraw.ImageDraw.text

    def _capture_text(self, xy, text, *args, **kwargs):
        if text == "Next Cubs game...":
            captured["title_y"] = xy[1]
        elif text == "Philadelphia Phillies":
            captured["opponent_y"] = xy[1]
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)
    monkeypatch.setattr(mlb_schedule, "is_hyperpixel_4_square_layout", lambda: True)
    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(mlb_schedule.config, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_center_bottom_text", lambda *args, **kwargs: None)

    game = {
        "officialDate": "2026-05-01",
        "startTimeCentral": "6:40 PM",
        "status": {"detailedState": "Scheduled"},
        "teams": {
            "away": {"team": {"id": 143, "name": "Philadelphia Phillies"}},
            "home": {"team": {"id": 112, "name": "Chicago Cubs"}},
        },
    }

    mlb_schedule.draw_sports_screen(None, game, "Next Cubs game...", screen_id="cubs next")

    reference_h = ImageDraw.Draw(Image.new("RGB", (1, 1))).textsize(
        "Cubs Current Series",
        font=mlb_schedule.FONT_TITLE_SPORTS,
    )[1]
    assert captured["title_y"] == 0
    assert captured["opponent_y"] == reference_h + 8


def test_draw_sports_screen_sox_next_uses_current_series_title_spacing(monkeypatch):
    captured = {}
    original_text = ImageDraw.ImageDraw.text

    def _capture_text(self, xy, text, *args, **kwargs):
        if text == "Next Sox game...":
            captured["title_y"] = xy[1]
        elif text == "Detroit Tigers":
            captured["opponent_y"] = xy[1]
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)
    monkeypatch.setattr(mlb_schedule, "is_hyperpixel_4_square_layout", lambda: False)
    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(mlb_schedule.config, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_center_bottom_text", lambda *args, **kwargs: None)

    game = {
        "officialDate": "2026-05-01",
        "startTimeCentral": "6:40 PM",
        "status": {"detailedState": "Scheduled"},
        "teams": {
            "away": {"team": {"id": 145, "name": "Chicago White Sox"}},
            "home": {"team": {"id": 116, "name": "Detroit Tigers"}},
        },
    }

    mlb_schedule.draw_sports_screen(None, game, "Next Sox game...", screen_id="sox next")

    reference_h = ImageDraw.Draw(Image.new("RGB", (1, 1))).textsize(
        "Sox Current Series",
        font=mlb_schedule.FONT_TITLE_SPORTS,
    )[1]
    assert captured["title_y"] == 0
    assert captured["opponent_y"] == reference_h + 8


@pytest.mark.parametrize(
    ("title", "screen_id", "expected_opponent_text", "reference_title", "game"),
    [
        (
            "Next at home...",
            "cubs next home",
            "vs. St. Louis Cardinals",
            "Cubs Current Series",
            {
                "teams": {
                    "away": {"team": {"id": 138, "name": "St. Louis Cardinals"}},
                    "home": {"team": {"id": 112, "name": "Chicago Cubs"}},
                },
            },
        ),
        (
            "Next at home...",
            "sox next home",
            "vs. Kansas City Royals",
            "Sox Current Series",
            {
                "teams": {
                    "away": {"team": {"id": 118, "name": "Kansas City Royals"}},
                    "home": {"team": {"id": 145, "name": "Chicago White Sox"}},
                },
            },
        ),
    ],
)
def test_draw_sports_screen_next_home_uses_current_series_title_spacing(
    monkeypatch,
    title,
    screen_id,
    expected_opponent_text,
    reference_title,
    game,
):
    captured = {}
    line_calls = []
    original_text = ImageDraw.ImageDraw.text

    def _capture_text(self, xy, text, *args, **kwargs):
        line_calls.append((xy[0], xy[1], text))
        if text == title:
            captured["title_y"] = xy[1]
        elif text == expected_opponent_text:
            captured["opponent_y"] = xy[1]
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)
    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(mlb_schedule.config, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_center_bottom_text", lambda *args, **kwargs: None)

    payload = {
        "officialDate": "2026-05-01",
        "startTimeCentral": "6:40 PM",
        "status": {"detailedState": "Scheduled"},
        **game,
    }
    mlb_schedule.draw_sports_screen(None, payload, title, screen_id=screen_id)

    reference_h = ImageDraw.Draw(Image.new("RGB", (1, 1))).textsize(
        reference_title,
        font=mlb_schedule.FONT_TITLE_SPORTS,
    )[1]
    if "opponent_y" not in captured:
        opponent = expected_opponent_text.split(". ", 1)[1]
        for x, y, text in line_calls:
            if text != opponent:
                continue
            for prefix_x, prefix_y, prefix_text in line_calls:
                if (
                    prefix_text in {"vs. ", "at "}
                    and abs(prefix_y - y) <= 10
                    and prefix_x < x
                ):
                    captured["opponent_y"] = max(y, prefix_y)
                    break
            if "opponent_y" in captured:
                break
    assert captured["title_y"] == 0
    assert captured["opponent_y"] == reference_h + 8


def test_draw_sports_screen_next_variants_scale_prefix_font(monkeypatch):
    captured = []
    original_text = ImageDraw.ImageDraw.text

    def _capture_text(self, xy, text, *args, **kwargs):
        font = kwargs.get("font")
        font_size = getattr(font, "size", None)
        captured.append((text, font_size))
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)
    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(mlb_schedule.config, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_center_bottom_text", lambda *args, **kwargs: None)

    game = {
        "officialDate": "2026-05-01",
        "startTimeCentral": "6:40 PM",
        "status": {"detailedState": "Scheduled"},
        "teams": {
            "away": {"team": {"id": 143, "name": "Philadelphia Phillies"}},
            "home": {"team": {"id": 112, "name": "Chicago Cubs"}},
        },
    }

    mlb_schedule.draw_sports_screen(None, game, "Next Cubs game...", screen_id="cubs next")

    prefix_draw = next((entry for entry in captured if entry[0] == "vs. "), None)
    opponent_draw = next((entry for entry in captured if entry[0] == "Philadelphia Phillies"), None)
    assert prefix_draw is not None
    assert opponent_draw is not None
    assert prefix_draw[1] < opponent_draw[1]


def test_draw_series_screen_next_variants_use_current_series_title_spacing(monkeypatch):
    cases = [
        ("Cubs Next Series", "cubs next series", "vs. Milwaukee Brewers", "Cubs Current Series"),
        ("Cubs Next Home Series", "cubs next home series", "vs. Milwaukee Brewers", "Cubs Current Series"),
        ("Sox Next Series", "sox next series", "vs. Cleveland Guardians", "Sox Current Series"),
        ("Sox Next Home Series", "sox next home series", "vs. Cleveland Guardians", "Sox Current Series"),
    ]

    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(mlb_schedule.config, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_series_final_result_parts", lambda *args, **kwargs: None)

    original_text = ImageDraw.ImageDraw.text
    for title, screen_id, opponent_text, reference_title in cases:
        captured = {}
        line_calls = []

        def _capture_text(self, xy, text, *args, **kwargs):
            line_calls.append((xy[0], xy[1], text))
            if text == title:
                captured["title_y"] = xy[1]
            elif text == opponent_text:
                captured["opponent_y"] = xy[1]
            return original_text(self, xy, text, *args, **kwargs)

        monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)

        focus_id = 112 if "cubs" in screen_id else 145
        games = [
            {
                "status": {"detailedState": "Scheduled"},
                "teams": {
                    "away": {"team": {"id": 158 if focus_id == 112 else 114, "name": "Milwaukee Brewers" if focus_id == 112 else "Cleveland Guardians"}},
                    "home": {"team": {"id": focus_id, "name": "Chicago Cubs" if focus_id == 112 else "Chicago White Sox"}},
                },
            }
        ]

        mlb_schedule.draw_series_screen(None, games, title=title, screen_id=screen_id)

        reference_h = ImageDraw.Draw(Image.new("RGB", (1, 1))).textsize(
            reference_title,
            font=mlb_schedule.FONT_TITLE_SPORTS,
        )[1]
        if "opponent_y" not in captured:
            opponent = opponent_text.split(". ", 1)[1]
            for x, y, text in line_calls:
                if text != opponent:
                    continue
                for prefix_x, prefix_y, prefix_text in line_calls:
                    if (
                        prefix_text in {"vs. ", "at "}
                        and abs(prefix_y - y) <= 10
                        and prefix_x < x
                    ):
                        captured["opponent_y"] = max(y, prefix_y)
                        break
                if "opponent_y" in captured:
                    break
        assert captured["title_y"] == 0
        assert captured["opponent_y"] == reference_h + 8


def test_draw_series_screen_current_matches_next_game_title_spacing_on_hyperpixel(monkeypatch):
    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: True)
    monkeypatch.setattr(mlb_schedule.config, "scale_value", lambda value: value * 3)
    monkeypatch.setattr(mlb_schedule, "is_hyperpixel_4_square_layout", lambda: False)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_center_bottom_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_series_final_result_parts", lambda *args, **kwargs: None)

    original_text = ImageDraw.ImageDraw.text
    captured = {
        "next_title_y": None,
        "next_opponent_y": None,
        "series_title_y": None,
        "series_opponent_y": None,
    }

    def _capture_text(self, xy, text, *args, **kwargs):
        if text == "Next Cubs game...":
            captured["next_title_y"] = xy[1]
        elif text == "Cubs Current Series":
            captured["series_title_y"] = xy[1]
        elif text == "Philadelphia Phillies":
            if captured["next_title_y"] is not None and captured["next_opponent_y"] is None:
                captured["next_opponent_y"] = xy[1]
            elif captured["series_title_y"] is not None and captured["series_opponent_y"] is None:
                captured["series_opponent_y"] = xy[1]
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)

    game = {
        "officialDate": "2026-05-01",
        "startTimeCentral": "6:40 PM",
        "status": {"detailedState": "Scheduled"},
        "teams": {
            "away": {"team": {"id": 143, "name": "Philadelphia Phillies"}},
            "home": {"team": {"id": 112, "name": "Chicago Cubs"}},
        },
    }
    games = [{"status": {"detailedState": "Scheduled"}, "teams": game["teams"]}]

    mlb_schedule.draw_sports_screen(None, game, "Next Cubs game...", screen_id="cubs next")
    mlb_schedule.draw_series_screen(None, games, "Cubs Current Series", screen_id="cubs current series")

    next_reference_h = ImageDraw.Draw(Image.new("RGB", (1, 1))).textsize(
        "Cubs Current Series",
        font=mlb_schedule.FONT_TITLE_SPORTS,
    )[1]

    assert captured["next_title_y"] is not None
    assert captured["next_opponent_y"] is not None
    assert captured["series_title_y"] is not None
    assert captured["series_opponent_y"] is not None
    assert captured["next_opponent_y"] == captured["next_title_y"] + next_reference_h + 16
    assert captured["series_opponent_y"] == captured["series_title_y"] + next_reference_h + 16


@pytest.mark.parametrize(
    ("title", "screen_id", "expected_opponent_text", "reference_title", "game"),
    [
        (
            "Next at home...",
            "cubs next home",
            "vs. St. Louis Cardinals",
            "Cubs Current Series",
            {
                "teams": {
                    "away": {"team": {"id": 138, "name": "St. Louis Cardinals"}},
                    "home": {"team": {"id": 112, "name": "Chicago Cubs"}},
                },
            },
        ),
        (
            "Next at home...",
            "sox next home",
            "vs. Kansas City Royals",
            "Sox Current Series",
            {
                "teams": {
                    "away": {"team": {"id": 118, "name": "Kansas City Royals"}},
                    "home": {"team": {"id": 145, "name": "Chicago White Sox"}},
                },
            },
        ),
    ],
)
def test_draw_sports_screen_next_home_uses_current_series_title_spacing(
    monkeypatch,
    title,
    screen_id,
    expected_opponent_text,
    reference_title,
    game,
):
    captured = {}
    line_calls = []
    original_text = ImageDraw.ImageDraw.text

    def _capture_text(self, xy, text, *args, **kwargs):
        line_calls.append((xy[0], xy[1], text))
        if text == title:
            captured["title_y"] = xy[1]
        elif text == expected_opponent_text:
            captured["opponent_y"] = xy[1]
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)
    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(mlb_schedule.config, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_center_bottom_text", lambda *args, **kwargs: None)

    payload = {
        "officialDate": "2026-05-01",
        "startTimeCentral": "6:40 PM",
        "status": {"detailedState": "Scheduled"},
        **game,
    }
    mlb_schedule.draw_sports_screen(None, payload, title, screen_id=screen_id)

    reference_h = ImageDraw.Draw(Image.new("RGB", (1, 1))).textsize(
        reference_title,
        font=mlb_schedule.FONT_TITLE_SPORTS,
    )[1]
    expected_y = reference_h + 8
    if "opponent_y" not in captured:
        opponent = expected_opponent_text.split(". ", 1)[1]
        for x, y, text in line_calls:
            if text != opponent:
                continue
            for prefix_x, prefix_y, prefix_text in line_calls:
                if (
                    prefix_text in {"vs. ", "at "}
                    and abs(prefix_y - y) <= 10
                    and prefix_x < x
                ):
                    captured["opponent_y"] = y
                    captured["prefix_y"] = prefix_y
                    break
            if "opponent_y" in captured:
                break
    assert captured["title_y"] == 0
    assert any(
        value is not None and abs(value - expected_y) <= 1
        for value in (captured.get("opponent_y"), captured.get("prefix_y"))
    )


def test_draw_series_screen_next_variants_use_current_series_title_spacing(monkeypatch):
    cases = [
        ("Cubs Next Series", "cubs next series", "vs. Milwaukee Brewers", "Cubs Current Series"),
        ("Cubs Next Home Series", "cubs next home series", "vs. Milwaukee Brewers", "Cubs Current Series"),
        ("Sox Next Series", "sox next series", "vs. Cleveland Guardians", "Sox Current Series"),
        ("Sox Next Home Series", "sox next home series", "vs. Cleveland Guardians", "Sox Current Series"),
    ]

    monkeypatch.setattr(mlb_schedule.config, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(mlb_schedule.config, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_schedule, "load_team_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_schedule, "_series_final_result_parts", lambda *args, **kwargs: None)

    original_text = ImageDraw.ImageDraw.text
    for title, screen_id, opponent_text, reference_title in cases:
        captured = {}
        line_calls = []

        def _capture_text(self, xy, text, *args, **kwargs):
            line_calls.append((xy[0], xy[1], text))
            if text == title:
                captured["title_y"] = xy[1]
            elif text == opponent_text:
                captured["opponent_y"] = xy[1]
            return original_text(self, xy, text, *args, **kwargs)

        monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)

        focus_id = 112 if "cubs" in screen_id else 145
        games = [
            {
                "status": {"detailedState": "Scheduled"},
                "teams": {
                    "away": {"team": {"id": 158 if focus_id == 112 else 114, "name": "Milwaukee Brewers" if focus_id == 112 else "Cleveland Guardians"}},
                    "home": {"team": {"id": focus_id, "name": "Chicago Cubs" if focus_id == 112 else "Chicago White Sox"}},
                },
            }
        ]

        mlb_schedule.draw_series_screen(None, games, title=title, screen_id=screen_id)

        reference_h = ImageDraw.Draw(Image.new("RGB", (1, 1))).textsize(
            reference_title,
            font=mlb_schedule.FONT_TITLE_SPORTS,
        )[1]
        expected_y = reference_h + 8
        if "opponent_y" not in captured:
            opponent = opponent_text.split(". ", 1)[1]
            for x, y, text in line_calls:
                if text != opponent:
                    continue
                for prefix_x, prefix_y, prefix_text in line_calls:
                    if (
                        prefix_text in {"vs. ", "at "}
                        and abs(prefix_y - y) <= 10
                        and prefix_x < x
                    ):
                        captured["opponent_y"] = y
                        captured["prefix_y"] = prefix_y
                        break
                if "opponent_y" in captured:
                    break
        assert captured["title_y"] == 0
        assert any(
            value is not None and abs(value - expected_y) <= 1
            for value in (captured.get("opponent_y"), captured.get("prefix_y"))
        )


def test_is_postponed_game_detects_common_status_shapes():
    assert mlb_schedule._is_postponed_game({"status": {"detailedState": "Postponed"}})
    assert mlb_schedule._is_postponed_game({"status": {"abstractGameState": "Postponed"}})
    assert mlb_schedule._is_postponed_game({"status": {"statusCode": "PPD"}})
    assert not mlb_schedule._is_postponed_game({"status": {"statusCode": "P"}})
    assert not mlb_schedule._is_postponed_game({"status": {"detailedState": "Final"}})


def test_draw_last_game_postponed_uses_dashes_and_no_result_flags(monkeypatch):
    captured = {}

    def _fake_draw_table(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(mlb_schedule, "_draw_boxscore_table", _fake_draw_table)

    game = {
        "officialDate": "2026-05-01",
        "status": {"detailedState": "Postponed", "statusCode": "PPD"},
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

    result = mlb_schedule.draw_last_game(None, game, title="Last Sox game...", screen_id="sox last")

    assert result is not None
    args = captured["args"]
    kwargs = captured["kwargs"]
    assert args[2] == "Last Sox game..."
    assert args[4:7] == ("-", "-", "-")
    assert args[8:11] == ("-", "-", "-")
    assert args[11] == "Postponed"
    assert kwargs["winner_flag"] is None
    assert kwargs["inline_winner_flag"] is None
