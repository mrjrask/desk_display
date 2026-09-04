import screens.mlb_scoreboard_v2 as mlb_scoreboard_v2
import screens.nba_scoreboard_v2 as nba_scoreboard_v2
import screens.ncaam_scoreboard as ncaam_scoreboard
import screens.ncaam_scoreboard_v2 as ncaam_scoreboard_v2
import screens.nfl_scoreboard_v2 as nfl_scoreboard_v2
import screens.nhl_scoreboard_v2 as nhl_scoreboard_v2


class _DisplayStub:
    def image(self, _img):
        return None


def test_nfl_v2_uses_v1_renderer_when_fewer_than_six_games(monkeypatch):
    sentinel = object()

    monkeypatch.setattr(
        nfl_scoreboard_v2,
        "render_nfl_scoreboard_v1",
        lambda display, games, transition=False: sentinel,
    )

    result = nfl_scoreboard_v2.render_nfl_scoreboard_v2(_DisplayStub(), [{}] * 5, transition=True)

    assert result is sentinel


def test_nfl_v2_reads_espn_competitors_by_home_away():
    away = {"homeAway": "away", "team": {"abbreviation": "CHI"}}
    home = {"homeAway": "home", "team": {"abbreviation": "GB"}}

    assert nfl_scoreboard_v2._competitors_by_side(
        {"competitors": [home, away]}
    ) == (away, home)


def test_nhl_v2_uses_v1_renderer_when_fewer_than_six_games(monkeypatch):
    sentinel = object()

    monkeypatch.setattr(
        nhl_scoreboard_v2,
        "render_nhl_scoreboard_v1",
        lambda display, games, transition=False: sentinel,
    )

    result = nhl_scoreboard_v2.render_nhl_scoreboard_v2(_DisplayStub(), [{}] * 5, transition=False)

    assert result is sentinel


def test_mlb_v2_uses_v1_renderer_when_fewer_than_six_games(monkeypatch):
    sentinel = object()

    monkeypatch.setattr(
        mlb_scoreboard_v2,
        "render_mlb_scoreboard_v1",
        lambda display, games, transition=False: sentinel,
    )

    result = mlb_scoreboard_v2.render_mlb_scoreboard_v2(_DisplayStub(), [{}] * 5, transition=False)

    assert result is sentinel


def test_nba_v2_uses_v1_renderer_when_fewer_than_six_games(monkeypatch):
    sentinel = object()

    monkeypatch.setattr(
        nba_scoreboard_v2,
        "render_nba_scoreboard_v1",
        lambda display, games, transition=False: sentinel,
    )

    result = nba_scoreboard_v2.render_nba_scoreboard_v2(_DisplayStub(), [{}] * 5, transition=False)

    assert result is sentinel



def test_ncaam_v1_entry_routes_to_non_recursive_v1_fallback_on_320x240(monkeypatch):
    sentinel = object()

    monkeypatch.setattr(ncaam_scoreboard, "WIDTH", 320)
    monkeypatch.setattr(ncaam_scoreboard, "HEIGHT", 240)
    monkeypatch.setattr(ncaam_scoreboard_v2, "WIDTH", 320)
    monkeypatch.setattr(ncaam_scoreboard_v2, "HEIGHT", 240)
    monkeypatch.setattr(
        ncaam_scoreboard_v2,
        "_render_ncaam_scoreboard_v1",
        lambda display, games, transition=False: sentinel,
    )

    result = ncaam_scoreboard.render_ncaam_scoreboard(_DisplayStub(), [{}] * 6, transition=False)

    assert result is sentinel
