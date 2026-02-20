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


def test_draw_division_right_aligns_points_on_hyperpixel_4_square(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_draw_text(draw, text, font, x, top, height, align="left"):
        calls.append((str(text), align))

    monkeypatch.setattr(nhl_standings, "_draw_text", fake_draw_text)
    monkeypatch.setattr(nhl_standings, "_draw_centered_text", lambda *args, **kwargs: 0)
    monkeypatch.setattr(nhl_standings, "_load_logo_cached", lambda abbr: None)
    monkeypatch.setattr(nhl_standings, "_IS_HYPERPIXEL_4", False)
    monkeypatch.setattr(nhl_standings, "is_hyperpixel_4_square_layout", lambda: True)

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
    assert align_by_text["4"] == "right"
    assert align_by_text["PTS"] == "right"


def test_build_column_layout_keeps_stats_separated_from_team_name():
    layout, team_name_width = nhl_standings._build_column_layout(max_team_name_width=500)

    team_x = nhl_standings.LEFT_MARGIN + nhl_standings.LOGO_HEIGHT + nhl_standings.TEAM_COLUMN_GAP
    first_stats_key = nhl_standings.STATS_COLUMNS[0]
    assert team_x + team_name_width <= layout[first_stats_key] - nhl_standings.TEAM_COLUMN_GAP

    previous_right = None
    for key in nhl_standings.STATS_COLUMNS:
        anchor_x = layout[key]
        header_label = next(label for label, col_key, _ in nhl_standings.COLUMN_HEADERS if col_key == key)
        header_font = nhl_standings.COLUMN_HEADER_FONTS.get(key, nhl_standings.COLUMN_FONT)
        header_width = nhl_standings._text_size(header_label, header_font)[0] if header_label else 0
        sample_value = "999" if key == "points" else "99"
        value_width = nhl_standings._text_size(sample_value, nhl_standings.ROW_STATS_FONT)[0]
        width = max(header_width, value_width)

        align = nhl_standings._stat_text_align(key)
        if align == "right":
            left, right = anchor_x - width, anchor_x
        elif align == "center":
            left, right = anchor_x - width / 2.0, anchor_x + width / 2.0
        else:
            left, right = anchor_x, anchor_x + width

        if previous_right is not None:
            assert left >= previous_right + 2
        previous_right = right

    assert layout[nhl_standings.STATS_COLUMNS[-1]] <= nhl_standings._table_right_edge()


def test_build_column_layout_falls_back_to_max_first_when_gap_is_too_large(monkeypatch):
    monkeypatch.setattr(nhl_standings, "STATS_COLUMN_MIN_STEP", 120)
    monkeypatch.setattr(nhl_standings, "STATS_FIRST_COLUMN_GAP", 260)

    layout, team_name_width = nhl_standings._build_column_layout(max_team_name_width=500)

    required_total = nhl_standings.STATS_COLUMN_MIN_STEP * (len(nhl_standings.STATS_COLUMNS) - 1)
    max_first = nhl_standings._table_right_edge() - required_total
    assert layout[nhl_standings.STATS_COLUMNS[0]] == max_first
    assert team_name_width >= 0


def test_build_column_layout_team_width_respects_first_stat_text_extent(monkeypatch):
    monkeypatch.setattr(nhl_standings, "_stat_text_align", lambda _key: "center")
    monkeypatch.setattr(nhl_standings, "_text_size", lambda _text, _font: (30, 10))

    _layout, team_name_width = nhl_standings._build_column_layout(max_team_name_width=500)

    team_x = nhl_standings.LEFT_MARGIN + nhl_standings.LOGO_HEIGHT + nhl_standings.TEAM_COLUMN_GAP
    first_key = nhl_standings.STATS_COLUMNS[0]
    first_anchor = _layout[first_key]
    first_left_extent = 15  # half of mocked 30 px text width due center alignment

    assert team_x + team_name_width <= first_anchor - first_left_extent - nhl_standings.TEAM_COLUMN_GAP
