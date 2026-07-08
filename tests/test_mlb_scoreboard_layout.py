import screens.mlb_scoreboard as mlb_scoreboard
import screens.mlb_scoreboard_v2 as mlb_scoreboard_v2
import screens.nhl_scoreboard as nhl_scoreboard
import screens.nhl_scoreboard_v2 as nhl_scoreboard_v2


def test_mlb_logo_height_is_capped_to_score_row(monkeypatch):
    monkeypatch.setattr(mlb_scoreboard, "TEAM_LOGO_BASE_HEIGHT", 120)
    monkeypatch.setattr(mlb_scoreboard, "SCORE_ROW_H", 56)
    monkeypatch.setattr(mlb_scoreboard, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_scoreboard, "get_screen_image_scale", lambda *args, **kwargs: 1.0)

    assert mlb_scoreboard._team_logo_height() == 48


def test_mlb_hyperpixel_square_adds_space_around_at_marker(monkeypatch):
    monkeypatch.setattr(mlb_scoreboard, "HYPERPIXEL_LAYOUT", True)
    monkeypatch.setattr(mlb_scoreboard, "HYPERPIXEL_4_SQUARE_LAYOUT", True)
    monkeypatch.setattr(mlb_scoreboard, "scale_value_width", lambda value: value)

    assert mlb_scoreboard._scoreboard_column_widths() == [76, 60, 48, 60, 76]


def test_mlb_non_square_keeps_original_scoreboard_columns(monkeypatch):
    monkeypatch.setattr(mlb_scoreboard, "HYPERPIXEL_LAYOUT", True)
    monkeypatch.setattr(mlb_scoreboard, "HYPERPIXEL_4_SQUARE_LAYOUT", False)
    monkeypatch.setattr(mlb_scoreboard, "scale_value_width", lambda value: value)

    assert mlb_scoreboard._scoreboard_column_widths() == [80, 60, 40, 60, 80]


def test_mlb_v2_hyperpixel_square_adds_space_around_at_marker(monkeypatch):
    monkeypatch.setattr(mlb_scoreboard_v2, "HYPERPIXEL_LAYOUT", True)
    monkeypatch.setattr(mlb_scoreboard_v2, "HYPERPIXEL_4_SQUARE_LAYOUT", True)
    monkeypatch.setattr(mlb_scoreboard_v2, "scale_value_width", lambda value: value)

    assert mlb_scoreboard_v2._game_column_widths() == [42, 30, 16, 30, 42]


def test_mlb_v2_non_square_keeps_original_scoreboard_columns(monkeypatch):
    monkeypatch.setattr(mlb_scoreboard_v2, "HYPERPIXEL_LAYOUT", True)
    monkeypatch.setattr(mlb_scoreboard_v2, "HYPERPIXEL_4_SQUARE_LAYOUT", False)
    monkeypatch.setattr(mlb_scoreboard_v2, "scale_value_width", lambda value: value)

    assert mlb_scoreboard_v2._game_column_widths() == [44, 30, 12, 30, 44]


def test_mlb_v2_logo_height_is_capped_to_score_row(monkeypatch):
    monkeypatch.setattr(mlb_scoreboard_v2, "TEAM_LOGO_BASE_HEIGHT", 80)
    monkeypatch.setattr(mlb_scoreboard_v2, "SCORE_ROW_H", 30)
    monkeypatch.setattr(mlb_scoreboard_v2, "scale_value", lambda value: value)
    monkeypatch.setattr(mlb_scoreboard_v2, "get_screen_image_scale", lambda *args, **kwargs: 1.0)

    mlb_scoreboard_v2._apply_style_overrides()

    assert mlb_scoreboard_v2.LOGO_HEIGHT == 26


def test_nhl_logo_height_is_capped_to_score_row(monkeypatch):
    monkeypatch.setattr(nhl_scoreboard, "TEAM_LOGO_BASE_HEIGHT", 120)
    monkeypatch.setattr(nhl_scoreboard, "SCORE_ROW_H", 56)
    monkeypatch.setattr(nhl_scoreboard, "scale_value", lambda value: value)
    monkeypatch.setattr(nhl_scoreboard, "get_screen_image_scale", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(nhl_scoreboard, "is_kernel_driven_display", lambda: False)

    nhl_scoreboard._apply_style_overrides()

    assert nhl_scoreboard.LOGO_HEIGHT == 48


def test_nhl_v2_logo_height_is_capped_to_score_row(monkeypatch):
    monkeypatch.setattr(nhl_scoreboard_v2, "TEAM_LOGO_BASE_HEIGHT", 80)
    monkeypatch.setattr(nhl_scoreboard_v2, "SCORE_ROW_H", 30)
    monkeypatch.setattr(nhl_scoreboard_v2, "scale_value", lambda value: value)
    monkeypatch.setattr(nhl_scoreboard_v2, "get_screen_image_scale", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(nhl_scoreboard_v2, "is_kernel_driven_display", lambda: False)

    nhl_scoreboard_v2._apply_style_overrides()

    assert nhl_scoreboard_v2.LOGO_HEIGHT == 26


def _game(game_pk: int, game_date: str, away_team: dict, home_team: dict) -> dict:
    return {
        "gamePk": game_pk,
        "gameDate": game_date,
        "teams": {
            "away": {"team": away_team},
            "home": {"team": home_team},
        },
    }


def test_mlb_hydrate_games_keeps_all_mlb_feed_games_sorted_by_start_time():
    mlb_vs_mlb_late = _game(
        1,
        "2026-03-05T21:00:00Z",
        {"triCode": "NYY"},
        {"triCode": "BOS"},
    )
    mlb_vs_mlb_early = _game(
        2,
        "2026-03-05T18:00:00Z",
        {"triCode": "CUBS"},
        {"triCode": "LAD"},
    )
    mixed_game = _game(
        3,
        "2026-03-05T17:00:00Z",
        {"triCode": "SEA"},
        {"name": "Japan"},
    )
    intl_vs_intl = _game(
        4,
        "2026-03-05T16:00:00Z",
        {"name": "Korea"},
        {"name": "Japan"},
    )

    hydrated = mlb_scoreboard._hydrate_games([
        intl_vs_intl,
        mlb_vs_mlb_late,
        mixed_game,
        mlb_vs_mlb_early,
    ])

    assert [g["gamePk"] for g in hydrated] == [4, 3, 2, 1]


class _DisplayStub:
    def __init__(self):
        self.last_image = None

    def image(self, img):
        self.last_image = img


def test_mlb_v2_render_shows_loading_message_when_games_not_hydrated(monkeypatch):
    display = _DisplayStub()
    monkeypatch.setattr(mlb_scoreboard_v2, "clear_display", lambda _display: None)
    monkeypatch.setattr(mlb_scoreboard_v2, "_get_league_logo", lambda: None)

    result = mlb_scoreboard_v2.render_mlb_scoreboard_v2(display, None, transition=False)

    assert result.displayed is True
    assert display.last_image is not None


