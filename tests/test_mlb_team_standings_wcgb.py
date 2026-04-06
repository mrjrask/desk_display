import screens.mlb_team_standings as mlb_team_standings


def test_format_wcgb_text_uses_plus_only_for_positive_top_three():
    assert mlb_team_standings._format_wcgb_text(1, 2) == "+1 WCGB"
    assert mlb_team_standings._format_wcgb_text("1.5", "3") == "+1 1/2 WCGB"


def test_format_wcgb_text_never_shows_plus_minus_combo():
    assert mlb_team_standings._format_wcgb_text("-", 2) == "- WCGB"
    assert mlb_team_standings._format_wcgb_text(-1, 2) == "1 WCGB"


def test_format_wcgb_text_shows_double_dash_at_zero():
    assert mlb_team_standings._format_wcgb_text(0, 1) == "-- WCGB"
