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


def test_seed_text_for_display_suppresses_duplicate_rank():
    team = {"curatedRank": {"current": "9"}}
    assert ncaam_scoreboard._seed_text_for_display(team) == ""


def test_seed_text_for_display_keeps_tournament_seed():
    team = {"seed": "11", "curatedRank": {"current": "7"}}
    assert ncaam_scoreboard._seed_text_for_display(team) == "11"


def test_extract_seed_ignores_curated_rank_placeholder():
    team = {"curatedRank": {"current": "99"}}
    assert ncaam_scoreboard._extract_seed(team) == ""


def test_ncaam_v2_logo_height_is_capped_to_score_row(monkeypatch):
    from screens import ncaam_scoreboard_v2

    monkeypatch.setattr(ncaam_scoreboard_v2, "SCORE_ROW_H", 28)
    monkeypatch.setattr(ncaam_scoreboard_v2, "scale_value", lambda value: value)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_team_logo_height", lambda: 100)

    assert ncaam_scoreboard_v2._v2_team_logo_height() == 24


def test_draw_rank_places_text_bottom_right_of_logo():
    class DummyDraw:
        def __init__(self):
            self.coords = None
            self.kwargs = None

        def textbbox(self, *_args, **_kwargs):
            return (0, 0, 12, 8)

        def text(self, coords, text, **kwargs):
            self.coords = coords
            self.kwargs = {"text": text, **kwargs}

    draw = DummyDraw()
    ncaam_scoreboard._draw_rank(draw, 4, x_logo=10, y_logo=20, logo_w=30, logo_h=40)

    assert draw.coords == (28 + ncaam_scoreboard.RANK_GAP, 52)
    assert draw.kwargs["text"] == "#4"
    assert draw.kwargs["font"] == ncaam_scoreboard.RANK_FONT
