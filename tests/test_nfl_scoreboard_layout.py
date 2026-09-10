from screens import nfl_scoreboard, nfl_scoreboard_v2


def test_nfl_single_game_columns_match_mlb_spacing(monkeypatch):
    monkeypatch.setattr(nfl_scoreboard, "HYPERPIXEL_4_SQUARE", True)
    monkeypatch.setattr(nfl_scoreboard, "scale_value_width", lambda value: value)

    assert nfl_scoreboard._scoreboard_column_widths() == [76, 60, 48, 60, 76]


def test_nfl_dual_game_columns_match_mlb_spacing(monkeypatch):
    monkeypatch.setattr(nfl_scoreboard_v2, "_IS_HYPERPIXEL_4_SQUARE_LAYOUT", True)
    monkeypatch.setattr(nfl_scoreboard_v2, "scale_value_width", lambda value: value)

    assert nfl_scoreboard_v2._game_column_widths() == [42, 30, 16, 30, 42]


def test_nfl_column_spacing_is_unchanged_on_other_displays(monkeypatch):
    monkeypatch.setattr(nfl_scoreboard, "HYPERPIXEL_4_SQUARE", False)
    monkeypatch.setattr(nfl_scoreboard, "scale_value_width", lambda value: value)
    monkeypatch.setattr(nfl_scoreboard_v2, "_IS_HYPERPIXEL_4_SQUARE_LAYOUT", False)
    monkeypatch.setattr(nfl_scoreboard_v2, "scale_value_width", lambda value: value)

    assert nfl_scoreboard._scoreboard_column_widths() == [80, 60, 40, 60, 80]
    assert nfl_scoreboard_v2._game_column_widths() == [44, 30, 12, 30, 44]
