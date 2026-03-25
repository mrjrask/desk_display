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

