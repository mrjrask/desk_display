from PIL import Image, ImageDraw

from screens import nhl_standings


def test_draw_division_centers_all_stat_values(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_draw_text(draw, text, font, x, top, height, align="left"):
        calls.append((str(text), align))

    monkeypatch.setattr(nhl_standings, "_draw_text", fake_draw_text)
    monkeypatch.setattr(nhl_standings, "_draw_centered_text", lambda *args, **kwargs: 0)
    monkeypatch.setattr(nhl_standings, "_load_logo_cached", lambda abbr: None)

    image = Image.new("RGB", (nhl_standings.WIDTH, nhl_standings.HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    column_layout = {
        "team": 10,
        "wins": 100,
        "losses": 130,
        "ot": 160,
        "points": 190,
    }
    nhl_standings._draw_division(
        image,
        draw,
        top=0,
        title="Division",
        teams=[{"name": "Test Team", "wins": 1, "losses": 2, "ot": 3, "points": 4}],
        column_layout=column_layout,
        team_name_max_width=200,
    )

    align_by_text = {text: align for text, align in calls}
    assert align_by_text["1"] == "center"
    assert align_by_text["2"] == "center"
    assert align_by_text["3"] == "center"
    assert align_by_text["4"] == "center"


def test_draw_division_right_aligns_points_on_hyperpixel_4(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_draw_text(draw, text, font, x, top, height, align="left"):
        calls.append((str(text), align))

    monkeypatch.setattr(nhl_standings, "_draw_text", fake_draw_text)
    monkeypatch.setattr(nhl_standings, "_draw_centered_text", lambda *args, **kwargs: 0)
    monkeypatch.setattr(nhl_standings, "_load_logo_cached", lambda abbr: None)
    monkeypatch.setattr(nhl_standings, "_IS_HYPERPIXEL_4", True)

    image = Image.new("RGB", (nhl_standings.WIDTH, nhl_standings.HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    column_layout = {
        "team": 10,
        "wins": 100,
        "losses": 130,
        "ot": 160,
        "points": 190,
    }
    nhl_standings._draw_division(
        image,
        draw,
        top=0,
        title="Division",
        teams=[{"name": "Test Team", "wins": 1, "losses": 2, "ot": 3, "points": 4}],
        column_layout=column_layout,
        team_name_max_width=200,
    )

    align_by_text = {text: align for text, align in calls}
    assert align_by_text["1"] == "center"
    assert align_by_text["2"] == "center"
    assert align_by_text["3"] == "center"
    assert align_by_text["4"] == "right"
    assert align_by_text["PTS"] == "right"
