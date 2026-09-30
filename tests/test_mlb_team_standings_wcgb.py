import screens.mlb_team_standings as mlb_team_standings


def test_format_wcgb_text_uses_plus_only_for_positive_top_three():
    assert mlb_team_standings._format_wcgb_text(1, 2) == "+1 WCGB"
    assert mlb_team_standings._format_wcgb_text("1.5", "3") == "+1 1/2 WCGB"


def test_format_wcgb_text_never_shows_plus_minus_combo():
    assert mlb_team_standings._format_wcgb_text("-", 2) == "- WCGB"
    assert mlb_team_standings._format_wcgb_text(-1, 2) == "1 WCGB"


def test_format_wcgb_text_shows_double_dash_at_zero():
    assert mlb_team_standings._format_wcgb_text(0, 1) == "-- WCGB"


def test_normalize_half_text_uses_half_glyph_for_cubs_and_sox_stands():
    assert (
        mlb_team_standings._normalize_half_text("3 1/2 GB", screen_id="Cubs Stand1")
        == "3 ½ GB"
    )
    assert (
        mlb_team_standings._normalize_half_text("1/2 WCGB", screen_id="sox stand3")
        == "½ WCGB"
    )


def test_normalize_half_text_keeps_default_fraction_for_other_screens():
    assert (
        mlb_team_standings._normalize_half_text("3 1/2 GB", screen_id="MLB AL Standings")
        == "3 1/2 GB"
    )


def test_cubs_and_sox_stands_use_the_1080p_logo_and_text_sizes(monkeypatch):
    # v0.1 drew these for a 320px panel; at 1080p they were a small logo
    # over small text.
    monkeypatch.setattr(mlb_team_standings, "_IS_1080P_LAYOUT", True)
    logo = mlb_team_standings.TEAM_STAND_LOGO_SZ_1080P
    assert mlb_team_standings._hawks_stand1_match_metrics("cubs stand1") == (
        logo, mlb_team_standings.TEAM_STAND1_FONT_OFFSET_1080P)
    assert mlb_team_standings._hawks_stand1_match_metrics("sox stand2") == (
        logo, mlb_team_standings.TEAM_STAND2_FONT_OFFSET_1080P)
    assert mlb_team_standings._hawks_stand1_match_metrics("bears stand1") == (None, 0)
