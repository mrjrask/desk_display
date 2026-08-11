from PIL import Image

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


def test_team_logo_url_uses_iowa_override():
    team = {
        "team": {
            "abbreviation": "IOWA",
            "logo": "https://example.com/original.png",
        },
    }
    assert ncaam_scoreboard._team_logo_url(team) == (
        "https://brand.uiowa.edu/sites/brand.uiowa.edu/files/styles/widescreen__1920_x_1080/public/2020-05/Tigerhawk-gold%20on%20black%402x.png?h=e39f7b2b&itok=TdYKif5p"
    )


def test_extract_rank_from_nested_curated_rank():
    team = {"curatedRank": {"current": "7"}}
    assert ncaam_scoreboard._extract_rank(team) == 7


def test_seed_text_for_display_suppresses_duplicate_rank():
    team = {"curatedRank": {"current": "9"}}
    assert ncaam_scoreboard._seed_text_for_display(team) == ""


def test_seed_text_for_display_keeps_tournament_seed():
    team = {"seed": "11", "curatedRank": {"current": "7"}}
    assert ncaam_scoreboard._seed_text_for_display(team) == "11"


def test_seed_text_for_display_keeps_tournament_seed_when_rank_matches(monkeypatch):
    monkeypatch.setattr(ncaam_scoreboard, "_scoreboard_mode", lambda: ncaam_scoreboard.MODE_TOURNAMENT)
    team = {"seed": "9", "curatedRank": {"current": "9"}}
    assert ncaam_scoreboard._seed_text_for_display(team) == "9"


def test_extract_seed_ignores_curated_rank_placeholder():
    team = {"curatedRank": {"current": "99"}}
    assert ncaam_scoreboard._extract_seed(team) == ""


def test_extract_seed_from_nested_team_blob():
    team = {"team": {"tournamentSeed": {"value": "12"}}}
    assert ncaam_scoreboard._extract_seed(team) == "12"


def test_extract_seed_parses_embedded_numeric_seed():
    team = {"team": {"tournamentSeed": {"displayValue": "No. 11"}}}
    assert ncaam_scoreboard._extract_seed(team) == "11"


def test_score_text_uses_dash_for_scheduled_games():
    assert ncaam_scoreboard._score_text({"score": None}, show=False) == "—"
    assert ncaam_scoreboard._score_text({"score": ""}, show=True) == "—"


def test_parse_start_time_central_includes_day_and_date():
    game = {"date": "2026-03-31T23:00:00Z"}
    assert ncaam_scoreboard._parse_start_time_central(game) == "Tue 3/31 6:00 PM"


def test_ncaam_v2_logo_height_is_capped_to_score_row(monkeypatch):
    from screens import ncaam_scoreboard_v2

    monkeypatch.setattr(ncaam_scoreboard_v2, "SCORE_ROW_H", 28)
    monkeypatch.setattr(ncaam_scoreboard_v2, "scale_value", lambda value: value)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_team_logo_height", lambda: 100)

    assert ncaam_scoreboard_v2._v2_team_logo_height() == 24


def test_draw_rank_places_text_bottom_left_of_logo():
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
    ncaam_scoreboard._draw_rank(draw, 4, x_logo=10, y_logo=20, logo_w=30, logo_h=40, position="left")

    assert draw.coords == (10 - ncaam_scoreboard.RANK_GAP - 12, 52)
    assert draw.kwargs["text"] == "#4"
    assert draw.kwargs["font"] == ncaam_scoreboard.RANK_FONT


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
    ncaam_scoreboard._draw_rank(draw, 4, x_logo=10, y_logo=20, logo_w=30, logo_h=40, position="right")

    assert draw.coords == (40 + ncaam_scoreboard.RANK_GAP, 52)
    assert draw.kwargs["text"] == "#4"
    assert draw.kwargs["font"] == ncaam_scoreboard.RANK_FONT


def test_v2_draw_single_game_passes_logo_dimensions_to_rank(monkeypatch):
    from screens import ncaam_scoreboard_v2

    class Logo:
        width = 18
        height = 24

    class DummyCanvas:
        def paste(self, *_args, **_kwargs):
            return None

    class DummyDraw:
        def textbbox(self, *_args, **_kwargs):
            return (0, 0, 0, 0)

        def text(self, *_args, **_kwargs):
            return None

    rank_calls = []

    monkeypatch.setattr(ncaam_scoreboard_v2, "_should_display_scores", lambda _game: False)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_score_text", lambda _team, show=False: "")
    monkeypatch.setattr(ncaam_scoreboard_v2, "_is_in_progress", lambda _game: False)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_is_final", lambda _game: False)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_center_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_v2_team_logo_height", lambda: 24)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_load_remote_logo", lambda _url, _h: Logo())
    monkeypatch.setattr(ncaam_scoreboard_v2, "_team_logo_url", lambda _team: "https://example.com/logo.png")
    monkeypatch.setattr(ncaam_scoreboard_v2, "_rank_for_display", lambda _team: 5)
    monkeypatch.setattr(
        ncaam_scoreboard_v2,
        "_draw_rank",
        lambda _draw, _rank, _x, _y, logo_w, logo_h, position="right": rank_calls.append((logo_w, logo_h, position)),
    )
    monkeypatch.setattr(ncaam_scoreboard_v2, "_draw_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_seed_text_for_display", lambda _team: "")
    monkeypatch.setattr(ncaam_scoreboard_v2, "_status_text", lambda _game: "Scheduled")

    game = {
        "teams": {
            "away": {"team": {"id": "1"}},
            "home": {"team": {"id": "2"}},
        },
        "status": {"type": {"state": "pre"}},
    }

    ncaam_scoreboard_v2._draw_single_game(DummyCanvas(), DummyDraw(), game, x_offset=0, top=0)

    assert rank_calls == [(18, 24, "left"), (18, 24, "right")]


def test_v1_tournament_mode_draws_seed(monkeypatch):
    draw_seed_calls = []

    monkeypatch.setattr(ncaam_scoreboard, "_scoreboard_mode", lambda: ncaam_scoreboard.MODE_TOURNAMENT)
    monkeypatch.setattr(ncaam_scoreboard, "_team_logo_height", lambda: 24)
    monkeypatch.setattr(ncaam_scoreboard, "_load_remote_logo", lambda _url, _h: Image.new("RGBA", (18, 24), (0, 0, 0, 0)))
    monkeypatch.setattr(ncaam_scoreboard, "_team_logo_url", lambda _team: "https://example.com/logo.png")
    monkeypatch.setattr(ncaam_scoreboard, "_should_display_scores", lambda _game: False)
    monkeypatch.setattr(ncaam_scoreboard, "_score_text", lambda _team, show=False: "")
    monkeypatch.setattr(ncaam_scoreboard, "_is_in_progress", lambda _game: False)
    monkeypatch.setattr(ncaam_scoreboard, "_is_final", lambda _game: False)
    monkeypatch.setattr(ncaam_scoreboard, "_status_text", lambda _game: "Scheduled")
    monkeypatch.setattr(ncaam_scoreboard, "_center_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(ncaam_scoreboard, "_draw_rank", lambda *args, **kwargs: None)
    monkeypatch.setattr(ncaam_scoreboard, "_seed_text_for_display", lambda _team: "11")
    monkeypatch.setattr(ncaam_scoreboard, "_draw_seed", lambda *args, **kwargs: draw_seed_calls.append(True))
    monkeypatch.setattr(ncaam_scoreboard, "_get_league_logo", lambda *_args, **_kwargs: None)

    game = {
        "teams": {"away": {"team": {"id": "1"}}, "home": {"team": {"id": "2"}}},
        "status": {"type": {"state": "pre"}},
    }

    ncaam_scoreboard._render_scoreboard([game], mode=ncaam_scoreboard.MODE_TOURNAMENT)

    assert len(draw_seed_calls) == 2


def test_v2_tournament_mode_draws_seed(monkeypatch):
    from screens import ncaam_scoreboard_v2

    class Logo:
        width = 18
        height = 24

    class DummyCanvas:
        def paste(self, *_args, **_kwargs):
            return None

    class DummyDraw:
        def textbbox(self, *_args, **_kwargs):
            return (0, 0, 0, 0)

        def text(self, *_args, **_kwargs):
            return None

    draw_seed_calls = []

    monkeypatch.setattr(ncaam_scoreboard_v2, "_scoreboard_mode", lambda: ncaam_scoreboard.MODE_TOURNAMENT)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_should_display_scores", lambda _game: False)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_score_text", lambda _team, show=False: "")
    monkeypatch.setattr(ncaam_scoreboard_v2, "_is_in_progress", lambda _game: False)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_is_final", lambda _game: False)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_center_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_v2_team_logo_height", lambda: 24)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_load_remote_logo", lambda _url, _h: Logo())
    monkeypatch.setattr(ncaam_scoreboard_v2, "_team_logo_url", lambda _team: "https://example.com/logo.png")
    monkeypatch.setattr(ncaam_scoreboard_v2, "_draw_rank", lambda *args, **kwargs: None)
    monkeypatch.setattr(ncaam_scoreboard_v2, "_seed_text_for_display", lambda _team: "8")
    monkeypatch.setattr(ncaam_scoreboard_v2, "_draw_seed", lambda *args, **kwargs: draw_seed_calls.append(True))
    monkeypatch.setattr(ncaam_scoreboard_v2, "_status_text", lambda _game: "Scheduled")

    game = {
        "teams": {"away": {"team": {"id": "1"}}, "home": {"team": {"id": "2"}}},
        "status": {"type": {"state": "pre"}},
    }

    ncaam_scoreboard_v2._draw_single_game(DummyCanvas(), DummyDraw(), game, x_offset=0, top=0)

    assert len(draw_seed_calls) == 2
