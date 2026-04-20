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
    monkeypatch.setattr(mlb_league_standings, "SHOW_LAST_10", False)
    monkeypatch.setattr(mlb_league_standings, "SHOW_WIN_PCT", False)

    assert mlb_league_standings._stat_columns() == ("record", "gb")

def test_stat_columns_show_last_10_for_wide_layout(monkeypatch):
    monkeypatch.setattr(mlb_league_standings, "SHOW_LAST_10", True)
    monkeypatch.setattr(mlb_league_standings, "SHOW_WIN_PCT", True)

    assert mlb_league_standings._stat_columns() == ("record", "last10", "gb")


def test_show_win_pct_for_layout_matches_display_hat_mini_behavior():
    assert mlb_league_standings._show_win_pct_for_layout(320, 240) is False
    assert mlb_league_standings._show_win_pct_for_layout(240, 135) is False


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


def test_normalize_row_extracts_last_ten_record():
    record = {
        "team": {"name": "Boston Red Sox", "abbreviation": "BOS"},
        "wins": 12,
        "losses": 8,
        "winningPercentage": ".600",
        "gamesBack": "1.0",
        "records": {
            "splitRecords": [
                {"type": "home", "wins": 7, "losses": 3},
                {"type": "lastTen", "wins": 6, "losses": 4},
            ]
        },
    }

    row = mlb_league_standings._normalize_row(record)

    assert row["last10"] == "6-4"


def test_normalize_row_expands_boston_sox_short_name():
    record = {
        "team": {"name": "Boston", "abbreviation": "BOS", "teamName": "Sox"},
        "wins": 12,
        "losses": 8,
        "winningPercentage": ".600",
        "gamesBack": "1.0",
    }

    row = mlb_league_standings._normalize_row(record)

    assert row["team_name"] == "Red Sox"


def test_normalize_row_keeps_white_sox_short_name():
    record = {
        "team": {"name": "Chicago", "abbreviation": "CWS", "teamName": "Sox"},
        "wins": 12,
        "losses": 8,
        "winningPercentage": ".600",
        "gamesBack": "1.0",
    }

    row = mlb_league_standings._normalize_row(record)

    assert row["team_name"] == "Sox"


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


def test_draw_stat_uses_vertical_middle_anchor():
    class _DrawProbe:
        def __init__(self):
            self.anchor = None

        def text(self, _xy, _value, font=None, fill=None, anchor=None):
            _ = font, fill
            self.anchor = anchor

    probe = _DrawProbe()

    mlb_league_standings._draw_stat(probe, ".600", x=150, y=30)

    assert probe.anchor == "rm"


def test_compact_layout_increases_record_to_gb_gap(monkeypatch):
    monkeypatch.setattr(mlb_league_standings, "SHOW_WIN_PCT", False)
    monkeypatch.setattr(mlb_league_standings, "WIDTH", 320)
    monkeypatch.setattr(mlb_league_standings, "RIGHT_MARGIN", 8)
    monkeypatch.setattr(mlb_league_standings, "LEFT_MARGIN", 5)
    monkeypatch.setattr(mlb_league_standings, "LOGO_SIZE", 24)
    monkeypatch.setattr(mlb_league_standings, "TEAM_GAP", 6)
    monkeypatch.setattr(mlb_league_standings, "_STAT_COLUMN_GAP", 30)
    monkeypatch.setattr(mlb_league_standings, "_RECORD_TO_GB_EXTRA_GAP", 22)

    draw = ImageDraw.Draw(Image.new("RGB", (320, 240), (0, 0, 0)))
    rows = [{"wins": "99", "losses": "62", "gb": "33 1/2"}]

    layout = mlb_league_standings._column_layout(draw, rows)
    gb_whole, gb_frac = mlb_league_standings._split_gb_text("33 1/2")
    gb_w = mlb_league_standings._gb_value_width(draw, gb_whole, gb_frac)

    inter_column_gap = layout["gb"] - layout["record"] - gb_w

    assert inter_column_gap >= 52


def test_stat_header_labels_include_record_l10_and_gb():
    labels = mlb_league_standings._stat_header_labels()

    assert labels["record"] == "Record"
    assert labels["last10"] == "L10"
    assert labels["gb"] == "GB"


def test_draw_stat_headers_centers_labels_in_column(monkeypatch):
    monkeypatch.setattr(mlb_league_standings, "_stat_columns", lambda: ("record", "last10", "gb"))

    class _DrawProbe:
        def __init__(self):
            self.calls = []

        def text(self, xy, value, font=None, fill=None, anchor=None):
            _ = font, fill
            self.calls.append((xy, value, anchor))

        def textbbox(self, xy, value, font=None):
            _ = xy, font
            return (0, 0, len(value) * 6, 10)

    probe = _DrawProbe()
    layout = {
        "record": 210,
        "record_width": 60,
        "last10": 300,
        "last10_width": 40,
        "gb": 380,
        "gb_width": 30,
    }

    mlb_league_standings._draw_stat_headers(probe, layout, y=50)

    assert probe.calls[0] == ((180, 50), "Record", "mm")
    assert probe.calls[1] == ((280, 50), "L10", "mm")
    assert probe.calls[2] == ((365, 50), "GB", "mm")


def test_wide_layout_centers_record_and_midpoints_l10(monkeypatch):
    monkeypatch.setattr(mlb_league_standings, "SHOW_LAST_10", True)
    monkeypatch.setattr(mlb_league_standings, "SHOW_WIN_PCT", True)
    monkeypatch.setattr(mlb_league_standings, "WIDTH", 480)
    monkeypatch.setattr(mlb_league_standings, "RIGHT_MARGIN", 8)
    monkeypatch.setattr(mlb_league_standings, "LEFT_MARGIN", 5)
    monkeypatch.setattr(mlb_league_standings, "LOGO_SIZE", 24)
    monkeypatch.setattr(mlb_league_standings, "TEAM_GAP", 6)

    draw = ImageDraw.Draw(Image.new("RGB", (480, 320), (0, 0, 0)))
    rows = [{"wins": "99", "losses": "62", "pct": ".615", "last10": "8-2", "gb": "12 1/2"}]
    layout = mlb_league_standings._column_layout(draw, rows)

    record_center = layout["record"] - (layout["record_width"] / 2.0)
    last10_center = layout["last10"] - (layout["last10_width"] / 2.0)
    gb_center = layout["gb"] - (layout["gb_width"] / 2.0)

    assert abs(record_center - (mlb_league_standings.WIDTH / 2.0)) <= 1.0
    assert abs(last10_center - ((record_center + gb_center) / 2.0)) <= 1.0
