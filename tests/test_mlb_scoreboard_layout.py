import screens.mlb_scoreboard as mlb_scoreboard
import screens.mlb_scoreboard_v2 as mlb_scoreboard_v2


def test_mlb_logo_height_is_capped_to_score_row(monkeypatch):
    monkeypatch.setattr(mlb_scoreboard, "TEAM_LOGO_BASE_HEIGHT", 120)
    monkeypatch.setattr(mlb_scoreboard, "SCORE_ROW_H", 56)
    monkeypatch.setattr(mlb_scoreboard, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_scoreboard, "get_screen_image_scale", lambda *args, **kwargs: 1.0)

    assert mlb_scoreboard._team_logo_height() == 48


def test_mlb_v2_logo_height_is_capped_to_score_row(monkeypatch):
    monkeypatch.setattr(mlb_scoreboard_v2, "TEAM_LOGO_BASE_HEIGHT", 80)
    monkeypatch.setattr(mlb_scoreboard_v2, "SCORE_ROW_H", 30)
    monkeypatch.setattr(mlb_scoreboard_v2, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_scoreboard_v2, "get_screen_image_scale", lambda *args, **kwargs: 1.0)

    mlb_scoreboard_v2._apply_style_overrides()

    assert mlb_scoreboard_v2.LOGO_HEIGHT == 26
