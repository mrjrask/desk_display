import pytest
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

    assert mlb_league_standings._stat_columns() == ("record", "streak", "gb")


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


def test_normalize_row_extracts_streak_code():
    record = {
        "team": {"name": "Boston Red Sox", "abbreviation": "BOS"},
        "streak": {"streakCode": "W5"},
    }

    row = mlb_league_standings._normalize_row(record)

    assert row["streak"] == "W5"


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


def test_normalize_row_expands_toronto_jays_short_name():
    record = {
        "team": {"name": "Toronto", "abbreviation": "TOR", "teamName": "Jays"},
        "wins": 12,
        "losses": 8,
        "winningPercentage": ".600",
        "gamesBack": "1.0",
    }

    row = mlb_league_standings._normalize_row(record)

    assert row["team_name"] == "Blue Jays"


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
    assert labels["streak"] == "STRK (L10)"
    assert labels["gb"] == "GB"


def test_stat_columns_include_streak_with_last_10(monkeypatch):
    monkeypatch.setattr(mlb_league_standings, "SHOW_LAST_10", True)

    assert mlb_league_standings._stat_columns() == ("record", "streak", "gb")


def test_render_screen_draws_combined_recent_column(monkeypatch):
    draw_calls = []
    monkeypatch.setattr(
        mlb_league_standings,
        "_draw_league_screen",
        lambda title, league_id, screen_id, *, wild_card_only=False: draw_calls.append(
            (title, league_id, screen_id, wild_card_only)
        )
        or Image.new("RGB", (1, 1)),
    )
    monkeypatch.setattr(mlb_league_standings, "clear_display", lambda _display: None)
    monkeypatch.setattr(mlb_league_standings, "scroll_vertical_content", lambda **_kwargs: None)

    mlb_league_standings._render_screen(object(), "AL", 103, "MLB AL Standings")

    assert draw_calls == [("AL", 103, "MLB AL Standings", False)]


def test_render_screen_passes_through_wild_card_only(monkeypatch):
    draw_calls = []
    monkeypatch.setattr(
        mlb_league_standings,
        "_draw_league_screen",
        lambda title, league_id, screen_id, *, wild_card_only=False: draw_calls.append(
            (title, league_id, screen_id, wild_card_only)
        )
        or Image.new("RGB", (1, 1)),
    )
    monkeypatch.setattr(mlb_league_standings, "clear_display", lambda _display: None)
    monkeypatch.setattr(mlb_league_standings, "scroll_vertical_content", lambda **_kwargs: None)

    mlb_league_standings._render_screen(
        object(), "ALWC", 103, "MLB ALWC Standings", wild_card_only=True
    )

    assert draw_calls == [("ALWC", 103, "MLB ALWC Standings", True)]


def test_draw_stat_headers_centers_labels_in_column(monkeypatch):
    monkeypatch.setattr(
        mlb_league_standings,
        "_stat_columns",
        lambda: ("record", "streak", "gb"),
    )

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
        "streak": 300,
        "streak_width": 40,
        "gb": 380,
        "gb_width": 30,
    }

    mlb_league_standings._draw_stat_headers(probe, layout, y=50)

    assert probe.calls[0] == ((180, 50), "Record", "mm")
    assert probe.calls[1] == ((280, 50), "STRK (L10)", "mm")
    assert probe.calls[2] == ((365, 50), "GB", "mm")


def test_draw_streak_places_parenthesized_last_10_to_right_in_pct_font(monkeypatch):
    calls = []

    class _DrawProbe:
        def textbbox(self, _xy, value, font=None):
            return (0, 0, len(value) * 6, 10)

        def text(self, xy, value, font=None, fill=None, anchor=None):
            calls.append((xy, value, font, fill, anchor))

    probe = _DrawProbe()
    mlb_league_standings._draw_streak_with_last10(probe, "W2", "8-2", 100, 30)

    assert calls[0][1:3] == ("W2", mlb_league_standings.STATS_FONT)
    assert calls[1][1:3] == ("(8-2)", mlb_league_standings.RECORD_PCT_FONT)
    assert calls[1][0][0] > calls[0][0][0]


def test_streak_width_accounts_for_last_10_parentheses():
    class _DrawProbe:
        def textbbox(self, _xy, value, font=None):
            return (0, 0, len(value) * 6, 10)

    width = mlb_league_standings._streak_with_last10_width(_DrawProbe(), "W2", "8-2")

    expected = (len("W2") * 6) + mlb_league_standings.scale_value(4) + (len("(8-2)") * 6)
    assert width == expected


def test_wide_layout_centers_record_and_midpoints_streak_with_last_10(monkeypatch):
    monkeypatch.setattr(mlb_league_standings, "SHOW_LAST_10", True)
    monkeypatch.setattr(mlb_league_standings, "SHOW_WIN_PCT", True)
    monkeypatch.setattr(mlb_league_standings, "WIDTH", 480)
    monkeypatch.setattr(mlb_league_standings, "RIGHT_MARGIN", 8)
    monkeypatch.setattr(mlb_league_standings, "LEFT_MARGIN", 5)
    monkeypatch.setattr(mlb_league_standings, "LOGO_SIZE", 24)
    monkeypatch.setattr(mlb_league_standings, "TEAM_GAP", 6)

    draw = ImageDraw.Draw(Image.new("RGB", (480, 320), (0, 0, 0)))
    rows = [{"wins": "99", "losses": "62", "pct": ".615", "streak": "W2", "last10": "8-2", "gb": "12 1/2"}]
    layout = mlb_league_standings._column_layout(draw, rows)

    record_center = layout["record"] - (layout["record_width"] / 2.0)
    streak_center = layout["streak"] - (layout["streak_width"] / 2.0)
    gb_center = layout["gb"] - (layout["gb_width"] / 2.0)

    assert abs(record_center - (mlb_league_standings.WIDTH / 2.0)) <= 1.0
    assert abs(streak_center - ((record_center + gb_center) / 2.0)) <= 1.0


def test_square_layout_team_width_stops_before_three_digit_record(monkeypatch):
    monkeypatch.setattr(mlb_league_standings, "SHOW_LAST_10", True)
    monkeypatch.setattr(mlb_league_standings, "SHOW_WIN_PCT", True)
    monkeypatch.setattr(mlb_league_standings, "WIDTH", 720)

    draw = ImageDraw.Draw(Image.new("RGB", (720, 720), (0, 0, 0)))
    rows = [
        {
            "team_name": "Guardians",
            "wins": "100",
            "losses": "62",
            "pct": ".617",
            "streak": "W2",
            "last10": "0-10",
            "gb": "-",
        }
    ]
    layout = mlb_league_standings._column_layout(draw, rows)

    team_right = layout["team"] + layout["team_max"]
    record_left = layout["record"] - layout["record_width"]

    assert team_right + mlb_league_standings._TEAM_TO_RECORD_GAP_WIDE <= record_left


@pytest.mark.parametrize(("width", "height"), [(135, 240), (240, 320)])
def test_compact_portrait_layout_preserves_team_name_width(monkeypatch, width, height):
    monkeypatch.setattr(mlb_league_standings, "SHOW_LAST_10", False)
    monkeypatch.setattr(mlb_league_standings, "SHOW_WIN_PCT", False)
    monkeypatch.setattr(mlb_league_standings, "WIDTH", width)

    draw = ImageDraw.Draw(Image.new("RGB", (width, height), (0, 0, 0)))
    rows = [{"team_name": "Guardians", "wins": "99", "losses": "63", "gb": "12.5"}]
    layout = mlb_league_standings._column_layout(draw, rows)

    assert layout["team_max"] >= mlb_league_standings.scale_value(70)


def test_wild_card_rows_excludes_division_leaders_and_defaults_to_five():
    standings = {
        "East": [
            {"team_name": "East Leader", "wins": "90", "losses": "60", "pct": ".600"},
            {"team_name": "East Two", "wins": "88", "losses": "62", "pct": ".587"},
            {"team_name": "East Three", "wins": "80", "losses": "70", "pct": ".533"},
        ],
        "Central": [
            {"team_name": "Central Leader", "wins": "85", "losses": "65", "pct": ".567"},
            {"team_name": "Central Two", "wins": "89", "losses": "61", "pct": ".593"},
            {"team_name": "Central Three", "wins": "81", "losses": "69", "pct": ".540"},
        ],
        "West": [
            {"team_name": "West Leader", "wins": "92", "losses": "58", "pct": ".613"},
            {"team_name": "West Two", "wins": "82", "losses": "68", "pct": ".547"},
            {"team_name": "West Three", "wins": "79", "losses": "71", "pct": ".527"},
        ],
    }

    rows = mlb_league_standings._wild_card_rows(standings)

    assert [row["team_name"] for row in rows] == [
        "Central Two",
        "East Two",
        "West Two",
        "Central Three",
        "East Three",
    ]
    assert all("Leader" not in row["team_name"] for row in rows)


def test_wcgb_text_preserves_positive_sign_for_current_wild_card_teams():
    assert mlb_league_standings._wcgb_text("1.5", "2") == "+1 1/2"
    assert mlb_league_standings._split_gb_text("+1 1/2") == ("+1", "1/2")


def test_wcgb_text_does_not_add_positive_sign_below_wild_card_cutoff():
    assert mlb_league_standings._wcgb_text("1.5", "4") == "1 1/2"


def test_wcgb_text_normalizes_negative_top_three_values_to_magnitude():
    assert mlb_league_standings._wcgb_text("-1.5", "2") == "1 1/2"


def test_postseason_elimination_prefers_wild_card_elimination_field():
    assert (
        mlb_league_standings._is_postseason_eliminated(
            {"wildCardEliminationNumber": "-", "eliminationNumber": "E"}
        )
        is False
    )
    assert (
        mlb_league_standings._is_postseason_eliminated(
            {"wildCardEliminationNumber": "E", "eliminationNumber": "-"}
        )
        is True
    )


def test_wild_card_rows_can_include_all_non_eliminated_teams_and_uses_wcgb():
    standings = {
        "East": [
            {"team_name": "East Leader", "wins": "90", "losses": "60", "pct": ".600", "wcgb": "-"},
            {"team_name": "East Two", "wins": "88", "losses": "62", "pct": ".587", "wcgb": "-", "wildCardRank": "1"},
            {"team_name": "East Out", "wins": "70", "losses": "80", "pct": ".467", "wcgb": "18", "wildCardRank": "6"},
        ],
        "Central": [
            {"team_name": "Central Leader", "wins": "85", "losses": "65", "pct": ".567", "wcgb": "-"},
            {"team_name": "Central Two", "wins": "84", "losses": "66", "pct": ".560", "wcgb": "1", "wildCardRank": "2"},
            {"team_name": "Central Eliminated", "wins": "60", "losses": "90", "pct": ".400", "wcgb": "28", "wildCardRank": "7", "wildCardEliminationNumber": "E"},
            {"team_name": "Central Division Eliminated", "wins": "83", "losses": "67", "pct": ".553", "wcgb": "2", "wildCardRank": "3", "wildCardEliminationNumber": "-", "eliminationNumber": "E"},
        ],
        "West": [
            {"team_name": "West Leader", "wins": "92", "losses": "58", "pct": ".613", "wcgb": "-"},
            {"team_name": "West Two", "wins": "83", "losses": "67", "pct": ".553", "wcgb": "2", "wildCardRank": "3"},
            {"team_name": "West Three", "wins": "82", "losses": "68", "pct": ".547", "wcgb": "3", "wildCardRank": "4", "wildCardEliminationNumber": "-"},
            {"team_name": "West Four", "wins": "81", "losses": "69", "pct": ".540", "wcgb": "4", "wildCardRank": "5", "eliminationNumber": "--"},
        ],
    }

    rows = mlb_league_standings._wild_card_rows(standings, limit=None)

    assert [row["team_name"] for row in rows] == [
        "East Two",
        "Central Two",
        "Central Division Eliminated",
        "West Two",
        "West Three",
        "West Four",
        "East Out",
    ]
    assert rows[1]["gb"] == "1"
    assert "Central Eliminated" not in [row["team_name"] for row in rows]


def test_wild_card_rows_uses_api_wild_card_rank_for_tied_records():
    standings = {
        "East": [
            {"team_name": "East Leader", "wins": "90", "losses": "60", "pct": ".600"},
            {"team_name": "East Four", "wins": "80", "losses": "70", "pct": ".533", "wildCardRank": "4"},
        ],
        "Central": [
            {"team_name": "Central Leader", "wins": "91", "losses": "59", "pct": ".607"},
            {"team_name": "Central Three", "wins": "80", "losses": "70", "pct": ".533", "wildCardRank": "3"},
        ],
        "West": [
            {"team_name": "West Leader", "wins": "92", "losses": "58", "pct": ".613"},
        ],
    }

    rows = mlb_league_standings._wild_card_rows(standings)

    assert [row["team_name"] for row in rows[:2]] == ["Central Three", "East Four"]


def test_should_not_draw_wild_card_cut_line_through_record_tie_without_api_rank():
    rows = [
        {"wins": "91", "losses": "59", "pct": ".607"},
        {"wins": "88", "losses": "62", "pct": ".587"},
        {"wins": "80", "losses": "70", "pct": ".533"},
        {"wins": "80", "losses": "70", "pct": ".533"},
    ]

    assert mlb_league_standings._should_draw_wild_card_cut_line(rows) is False


def test_should_draw_wild_card_cut_line_between_distinct_api_ranks():
    rows = [
        {"wins": "91", "losses": "59", "pct": ".607", "wildCardRank": "1"},
        {"wins": "88", "losses": "62", "pct": ".587", "wildCardRank": "2"},
        {"wins": "80", "losses": "70", "pct": ".533", "wildCardRank": "3"},
        {"wins": "80", "losses": "70", "pct": ".533", "wildCardRank": "4"},
    ]

    assert mlb_league_standings._should_draw_wild_card_cut_line(rows) is True


def test_draw_wild_card_cut_line_uses_dotted_segments():
    image = Image.new("RGB", (120, 40), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    mlb_league_standings._draw_wild_card_cut_line(draw, center_x=60, col_width=80, y=20)

    lit_pixels = [x for x in range(120) if image.getpixel((x, 20)) != (0, 0, 0)]
    assert lit_pixels
    assert min(lit_pixels) == 20
    assert max(lit_pixels) == 100
    assert any((x + 1) not in lit_pixels for x in lit_pixels[:-1])


def test_league_screen_uses_wild_card_tie_check_before_drawing_cut_line(monkeypatch):
    standings = {
        mlb_league_standings.AL_LEAGUE_ID: {
            "East": [
                {"team_name": "East Leader", "abbr": "NYY", "wins": "90", "losses": "60", "pct": ".600", "gb": "-"},
                {"team_name": "East Two", "abbr": "BOS", "wins": "88", "losses": "62", "pct": ".587", "gb": "2", "wcgb": "-"},
                {"team_name": "East Three", "abbr": "TOR", "wins": "80", "losses": "70", "pct": ".533", "gb": "10", "wcgb": "8"},
            ],
            "Central": [
                {"team_name": "Central Leader", "abbr": "CLE", "wins": "85", "losses": "65", "pct": ".567", "gb": "-"},
                {"team_name": "Central Two", "abbr": "DET", "wins": "83", "losses": "67", "pct": ".553", "gb": "2", "wcgb": "5"},
            ],
            "West": [
                {"team_name": "West Leader", "abbr": "HOU", "wins": "92", "losses": "58", "pct": ".613", "gb": "-"},
                {"team_name": "West Two", "abbr": "SEA", "wins": "80", "losses": "70", "pct": ".533", "gb": "12", "wcgb": "8"},
            ],
        }
    }
    calls = []

    monkeypatch.setattr(mlb_league_standings, "_fetch_league_standings", lambda: standings)
    monkeypatch.setattr(mlb_league_standings, "_load_mlb_logo", lambda: None)
    monkeypatch.setattr(mlb_league_standings, "_load_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mlb_league_standings,
        "_draw_wild_card_cut_line",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    mlb_league_standings._draw_league_screen(
        "MLB AL Standings", mlb_league_standings.AL_LEAGUE_ID, "MLB AL Standings"
    )

    assert calls == []


def test_wild_card_only_screen_draws_cut_line_when_applicable(monkeypatch):
    standings = {
        mlb_league_standings.AL_LEAGUE_ID: {
            "East": [
                {"team_name": "East Leader", "abbr": "NYY", "wins": "90", "losses": "60", "pct": ".600", "gb": "-"},
                {"team_name": "East Two", "abbr": "BOS", "wins": "88", "losses": "62", "pct": ".587", "gb": "2", "wcgb": "-", "wildCardRank": "1"},
                {"team_name": "East Three", "abbr": "TOR", "wins": "80", "losses": "70", "pct": ".533", "gb": "10", "wcgb": "8", "wildCardRank": "3"},
            ],
            "Central": [
                {"team_name": "Central Leader", "abbr": "CLE", "wins": "85", "losses": "65", "pct": ".567", "gb": "-"},
                {"team_name": "Central Two", "abbr": "DET", "wins": "83", "losses": "67", "pct": ".553", "gb": "2", "wcgb": "5", "wildCardRank": "2"},
            ],
            "West": [
                {"team_name": "West Leader", "abbr": "HOU", "wins": "92", "losses": "58", "pct": ".613", "gb": "-"},
                {"team_name": "West Two", "abbr": "SEA", "wins": "78", "losses": "72", "pct": ".520", "gb": "14", "wcgb": "9", "wildCardRank": "4"},
            ],
        }
    }
    calls = []

    monkeypatch.setattr(mlb_league_standings, "_fetch_league_standings", lambda: standings)
    monkeypatch.setattr(mlb_league_standings, "_load_mlb_logo", lambda: None)
    monkeypatch.setattr(mlb_league_standings, "_load_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mlb_league_standings,
        "_draw_wild_card_cut_line",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    mlb_league_standings._draw_league_screen(
        "MLB ALWC Standings",
        mlb_league_standings.AL_LEAGUE_ID,
        "MLB ALWC Standings",
        wild_card_only=True,
    )

    assert calls


def test_draw_mlb_wc_standings_route_to_wild_card_only_render(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mlb_league_standings,
        "_render_screen",
        lambda display, title, league_id, screen_id, **kwargs: calls.append(
            (title, league_id, screen_id, kwargs)
        ),
    )

    mlb_league_standings.draw_mlb_al_wc_standings(object())
    mlb_league_standings.draw_mlb_nl_wc_standings(object())
    mlb_league_standings.draw_mlb_al_standings(object())
    mlb_league_standings.draw_mlb_nl_standings(object())

    assert calls == [
        ("MLB ALWC Standings", mlb_league_standings.AL_LEAGUE_ID, "MLB ALWC Standings", {"wild_card_only": True}),
        ("MLB NLWC Standings", mlb_league_standings.NL_LEAGUE_ID, "MLB NLWC Standings", {"wild_card_only": True}),
        ("MLB AL Standings", mlb_league_standings.AL_LEAGUE_ID, "MLB AL Standings", {}),
        ("MLB NL Standings", mlb_league_standings.NL_LEAGUE_ID, "MLB NL Standings", {}),
    ]


def test_draw_overview_wc_non_hyperpixel_has_column_width(monkeypatch):
    class _Display:
        def image(self, _image):
            pass

        def show(self):
            pass

        def wait_for_skip(self, _duration):
            return False

    monkeypatch.setattr(mlb_league_standings.config, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(mlb_league_standings.config, "is_hyperpixel_4_square_layout", lambda: False)
    monkeypatch.setattr(mlb_league_standings, "_fetch_league_standings", lambda: {mlb_league_standings.NL_LEAGUE_ID: {}})
    monkeypatch.setattr(mlb_league_standings, "_load_mlb_logo", lambda: None)
    monkeypatch.setattr(mlb_league_standings, "OVERVIEW_PAUSE_END", 0)

    result = mlb_league_standings.draw_overview(
        _Display(),
        "NL Overview+WC",
        mlb_league_standings.NL_LEAGUE_ID,
        transition=True,
        include_wc=True,
    )

    assert result is not None


def test_overview_wc_cut_line_drops_with_logos(monkeypatch):
    class _Display:
        def image(self, _image):
            pass

        def show(self):
            pass

    rows = [
        {
            "abbr": f"T{i}",
            "team_name": f"Team {i}",
            "wins": str(90 - i),
            "losses": str(60 + i),
            "pct": ".600",
            "wildCardRank": str(i + 1),
        }
        for i in range(5)
    ]
    standings = {
        mlb_league_standings.NL_LEAGUE_ID: {
            "East": rows,
            "Central": rows,
            "West": rows,
        }
    }

    line_ys = []

    def _capture_line(_draw, _center_x, _col_width, y):
        line_ys.append(y)

    monkeypatch.setattr(mlb_league_standings.config, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(mlb_league_standings.config, "is_hyperpixel_4_square_layout", lambda: False)
    monkeypatch.setattr(mlb_league_standings, "_fetch_league_standings", lambda: standings)
    monkeypatch.setattr(mlb_league_standings, "_load_logo", lambda *_args, **_kwargs: Image.new("RGBA", (2, 2), (255, 255, 255, 255)))
    monkeypatch.setattr(mlb_league_standings, "_draw_wild_card_cut_line", _capture_line)
    monkeypatch.setattr(mlb_league_standings, "OVERVIEW_DROP_STEPS", 3)
    monkeypatch.setattr(mlb_league_standings, "OVERVIEW_DROP_STAGGER", 0.5)
    monkeypatch.setattr(mlb_league_standings, "OVERVIEW_DROP_FRAME_DELAY", 0)
    monkeypatch.setattr(mlb_league_standings, "OVERVIEW_PAUSE_END", 0)

    mlb_league_standings.draw_overview(
        _Display(),
        "NL Overview+WC",
        mlb_league_standings.NL_LEAGUE_ID,
        transition=True,
        include_wc=True,
    )

    _header, top_y = mlb_league_standings._overview_header_frame("NL Overview+WC", (0, 0, 0))
    cell_h = (mlb_league_standings.HEIGHT - top_y) // mlb_league_standings.OV_ROWS
    final_line_y = int(top_y + 3 * cell_h)
    assert line_ys
    assert line_ys[0] < final_line_y
    assert final_line_y in line_ys
