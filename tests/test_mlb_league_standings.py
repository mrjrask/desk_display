from PIL import Image, ImageDraw

import screens.mlb_league_standings as mlb_league_standings


def test_normalize_row_uses_scoreboard_logo_code_mapping():
    record = {
        "team": {"name": "Chicago Cubs", "abbreviation": "CHC", "teamName": "Cubs"},
        "wins": 10,
        "losses": 7,
        "winningPercentage": ".588",
        "gamesBack": "2.5",
    }

    row = mlb_league_standings._normalize_row(record)

    assert row["abbr"] == "CUBS"
    assert row["team_name"] == "Cubs"


def test_stat_columns_omit_winning_pct_when_disabled(monkeypatch):
    monkeypatch.setattr(mlb_league_standings, "SHOW_WIN_PCT", False)

    assert mlb_league_standings._stat_columns() == ("record", "gb")


def test_normalize_row_keeps_red_sox_nickname():
    record = {
        "team": {"name": "Boston Red Sox", "abbreviation": "BOS"},
        "wins": 12,
        "losses": 8,
        "winningPercentage": ".600",
        "gamesBack": "1.0",
    }

    row = mlb_league_standings._normalize_row(record)

    assert row["abbr"] == "BOS"
    assert row["team_name"] == "Red Sox"


def test_normalize_row_formats_zero_games_back_as_dash():
    record = {
        "team": {"name": "New York Mets", "abbreviation": "NYM"},
        "wins": 15,
        "losses": 10,
        "winningPercentage": ".600",
        "gamesBack": "-0.0",
    }

    row = mlb_league_standings._normalize_row(record)

    assert row["gb"] == "-"


def test_split_gb_text_uses_superscript_fraction_suffix():
    assert mlb_league_standings._split_gb_text("1.5") == ("1", "1/2")
    assert mlb_league_standings._split_gb_text("0.5") == ("", "1/2")


def test_draw_gb_renders_without_name_error():
    image = Image.new("RGB", (200, 100), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    mlb_league_standings._draw_gb(draw, "1.5", x=180, y=40)
