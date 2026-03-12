from screens import ncaam_scoreboard


def test_mode_title_has_no_emoji(monkeypatch):
    monkeypatch.setattr(ncaam_scoreboard, "NCAAM_SCOREBOARD_MODE", "top25")
    title, _ = ncaam_scoreboard._mode_title_and_logo()
    assert title == "Top 25 - NCAAM"


def test_team_logo_url_supports_logo_fallback():
    team = {
        "team": {"logo": "https://example.com/logo.png"},
    }
    assert ncaam_scoreboard._team_logo_url(team) == "https://example.com/logo.png"


def test_extract_rank_from_nested_curated_rank():
    team = {"curatedRank": {"current": "7"}}
    assert ncaam_scoreboard._extract_rank(team) == 7
