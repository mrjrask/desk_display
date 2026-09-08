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


def test_nfl_v2_caps_team_logos_to_compact_score_cell(monkeypatch):
    monkeypatch.setattr(nfl_scoreboard_v2, "TEAM_LOGO_BASE_HEIGHT", 100)
    monkeypatch.setattr(nfl_scoreboard_v2, "SCORE_ROW_H", 80)
    monkeypatch.setattr(nfl_scoreboard_v2, "GAME_COL_WIDTHS", [44, 30, 12, 30, 44])
    monkeypatch.setattr(nfl_scoreboard_v2, "scale_value_width", lambda value: value)
    monkeypatch.setattr(
        nfl_scoreboard_v2,
        "get_screen_image_scale",
        lambda _screen, image, default: 3.0 if image == "team_logo" else default,
    )
    monkeypatch.setattr(nfl_scoreboard_v2, "is_kernel_driven_display", lambda: False)

    nfl_scoreboard_v2._apply_style_overrides()

    assert nfl_scoreboard_v2.LOGO_HEIGHT == 26


def test_nfl_v2_composes_and_scrolls_entire_16_game_week(monkeypatch):
    games = [{"id": f"event-{index:02d}"} for index in range(16)]
    composed_ids = []
    canvas_heights = []
    scroll_calls = []
    original_compose = nfl_scoreboard_v2._compose_canvas

    def record_pair(_canvas, _draw, first, second, _top):
        composed_ids.append(first["id"])
        if second is not None:
            composed_ids.append(second["id"])

    def record_compose(all_games, *, show_super_bowl_logo):
        canvas = original_compose(
            all_games, show_super_bowl_logo=show_super_bowl_logo
        )
        canvas_heights.append(canvas.height)
        return canvas

    monkeypatch.setattr(nfl_scoreboard_v2, "HEIGHT", 120)
    monkeypatch.setattr(nfl_scoreboard_v2, "V2_DISABLED_RESOLUTIONS", set())
    monkeypatch.setattr(nfl_scoreboard_v2, "_apply_style_overrides", lambda: None)
    monkeypatch.setattr(nfl_scoreboard_v2, "_get_league_logo", lambda: None)
    monkeypatch.setattr(nfl_scoreboard_v2, "_get_super_bowl_logo", lambda: None)
    monkeypatch.setattr(nfl_scoreboard_v2, "_draw_game_pair", record_pair)
    monkeypatch.setattr(nfl_scoreboard_v2, "_compose_canvas", record_compose)
    monkeypatch.setattr(
        nfl_scoreboard_v2,
        "scroll_vertical_content",
        lambda **kwargs: scroll_calls.append(kwargs),
    )

    result = nfl_scoreboard_v2.render_nfl_scoreboard_v2(
        _DisplayStub(), games, transition=False
    )

    assert composed_ids == [game["id"] for game in games]
    assert canvas_heights and canvas_heights[0] > nfl_scoreboard_v2.HEIGHT
    assert result.image.height > nfl_scoreboard_v2.HEIGHT
    assert len(scroll_calls) == 1
    assert scroll_calls[0]["content_height"] == result.image.height


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
