from PIL import Image

from screens import world_cup_scoreboard


class FixedDateTime(world_cup_scoreboard.datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        current = cls(2026, 6, 12, 12, 0, tzinfo=world_cup_scoreboard.datetime.timezone.utc)
        return current.astimezone(tz) if tz else current.replace(tzinfo=None)


class DummyDisplay:
    def __init__(self):
        self.images = []

    def image(self, img):
        self.images.append(img)


def test_scroll_display_scrolls_through_scoreboard_twice_with_continuous_content(monkeypatch):
    calls = []

    def fake_scroll_vertical_content(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(world_cup_scoreboard, "scroll_vertical_content", fake_scroll_vertical_content)

    img = Image.new(
        "RGB",
        (world_cup_scoreboard.WIDTH, world_cup_scoreboard.HEIGHT * 2),
        (0, 0, 0),
    )
    world_cup_scoreboard._scroll_display(object(), img)

    assert len(calls) == 1
    assert world_cup_scoreboard.SCROLL_REPEAT_COUNT == 2
    assert calls[0]["content_height"] == img.height * world_cup_scoreboard.SCROLL_REPEAT_COUNT


def test_single_game_scoreboard_displays_without_scrolling(monkeypatch):
    def fail_scroll(*args, **kwargs):
        raise AssertionError("single-game World Cup scoreboard should not scroll")

    full_img = Image.new(
        "RGB",
        (world_cup_scoreboard.WIDTH, world_cup_scoreboard.HEIGHT * 2),
        (0, 0, 0),
    )
    display = DummyDisplay()

    monkeypatch.setattr(world_cup_scoreboard, "_render_scoreboard", lambda games: full_img)
    monkeypatch.setattr(world_cup_scoreboard, "_scroll_display", fail_scroll)
    monkeypatch.setattr(world_cup_scoreboard.time, "sleep", lambda duration: None)

    result = world_cup_scoreboard.render_world_cup_scoreboard(display, [{}])

    assert result.displayed is True
    assert display.images == [result.image]
    assert result.image.size == (world_cup_scoreboard.WIDTH, world_cup_scoreboard.HEIGHT)


def test_multi_game_second_scroll_cycle_replaces_title_with_line(monkeypatch):
    calls = []
    rendered_styles = []

    def fake_render_scoreboard(games, *, title_style="title"):
        rendered_styles.append(title_style)
        color = (255, 255, 255) if title_style == "title" else (45, 45, 45)
        return Image.new(
            "RGB",
            (world_cup_scoreboard.WIDTH, world_cup_scoreboard.HEIGHT + 1),
            color,
        )

    def fake_scroll_display(display, img, repeat_images=None):
        calls.append({"img": img, "repeat_images": repeat_images})

    monkeypatch.setattr(world_cup_scoreboard, "_render_scoreboard", fake_render_scoreboard)
    monkeypatch.setattr(world_cup_scoreboard, "_scroll_display", fake_scroll_display)

    world_cup_scoreboard.render_world_cup_scoreboard(object(), [{}, {}])

    assert rendered_styles == ["title", "line"]
    assert len(calls) == 1
    repeat_images = calls[0]["repeat_images"]
    assert len(repeat_images) == 2
    assert repeat_images[0] is calls[0]["img"]
    assert repeat_images[0].getpixel((0, 0)) == (255, 255, 255)
    assert repeat_images[1].getpixel((0, 0)) == (45, 45, 45)


def test_parse_start_time_central_uses_today_for_same_day_match(monkeypatch):
    monkeypatch.setattr(world_cup_scoreboard.datetime, "datetime", FixedDateTime)
    game = {"date": "2026-06-12T17:00:00Z"}

    assert world_cup_scoreboard._parse_start_time_central(game) == "Today 12:00 PM"


def test_parse_start_time_central_uses_tonight_for_same_day_evening_match(monkeypatch):
    monkeypatch.setattr(world_cup_scoreboard.datetime, "datetime", FixedDateTime)
    game = {"date": "2026-06-12T23:00:00Z"}

    assert world_cup_scoreboard._parse_start_time_central(game) == "Tonight 6:00 PM"


def test_parse_start_time_central_keeps_date_for_later_match(monkeypatch):
    monkeypatch.setattr(world_cup_scoreboard.datetime, "datetime", FixedDateTime)
    game = {"date": "2026-06-13T17:00:00Z"}

    assert world_cup_scoreboard._parse_start_time_central(game) == "Sat 6/13 12:00 PM"


def test_score_text_includes_penalty_score_in_parentheses():
    team = {"score": "1", "shootoutScore": "4"}

    assert world_cup_scoreboard._score_text(team, show=True) == "1 (4)"


def test_score_fill_uses_penalty_score_to_choose_final_winner():
    away = {"score": "1", "shootoutScore": "4"}
    home = {"score": "1", "shootoutScore": "2"}

    assert world_cup_scoreboard._score_fill(
        "away", in_progress=False, final=True, away=away, home=home
    ) == world_cup_scoreboard.FINAL_WINNING_SCORE_COLOR
    assert world_cup_scoreboard._score_fill(
        "home", in_progress=False, final=True, away=away, home=home
    ) == world_cup_scoreboard.FINAL_LOSING_SCORE_COLOR


def test_score_text_supports_nested_penalty_score_payloads():
    team = {"score": "2", "penaltyScore": {"score": "5"}}

    assert world_cup_scoreboard._score_text(team, show=True) == "2 (5)"


def test_score_text_preserves_nested_zero_penalty_score():
    team = {"score": "1", "penaltyScore": {"score": 0, "value": None}}

    assert world_cup_scoreboard._score_text(team, show=True) == "1 (0)"


def test_score_fill_uses_nested_zero_penalty_score_to_choose_final_loser():
    away = {"score": "1", "penaltyScore": {"score": 0}}
    home = {"score": "1", "penaltyScore": {"score": 3}}

    assert world_cup_scoreboard._score_fill(
        "away", in_progress=False, final=True, away=away, home=home
    ) == world_cup_scoreboard.FINAL_LOSING_SCORE_COLOR
    assert world_cup_scoreboard._score_fill(
        "home", in_progress=False, final=True, away=away, home=home
    ) == world_cup_scoreboard.FINAL_WINNING_SCORE_COLOR


def test_penalty_score_font_is_30_percent_smaller_than_regular_score():
    expected_size = max(1, int(round(world_cup_scoreboard.SCORE_FONT.size * 0.7)))

    assert world_cup_scoreboard.PK_SCORE_FONT.size == expected_size


def test_score_text_segments_use_smaller_penalty_score_font():
    team = {"score": "1", "shootoutScore": "4"}

    assert world_cup_scoreboard._score_text_segments(team, show=True) == [
        ("1", world_cup_scoreboard.SCORE_FONT),
        (" (4)", world_cup_scoreboard.PK_SCORE_FONT),
    ]


def test_no_games_message_matches_requested_empty_state():
    assert world_cup_scoreboard.NO_GAMES_TEXT == "No Games Today"


def test_round_for_date_advances_through_world_cup_final_rounds():
    assert (
        world_cup_scoreboard._round_for_date(world_cup_scoreboard.datetime.date(2026, 7, 8))
        == world_cup_scoreboard.ROUND_QUARTERFINALS
    )
    assert (
        world_cup_scoreboard._round_for_date(world_cup_scoreboard.datetime.date(2026, 7, 12))
        == world_cup_scoreboard.ROUND_SEMIFINALS
    )
    assert (
        world_cup_scoreboard._round_for_date(world_cup_scoreboard.datetime.date(2026, 7, 17))
        == world_cup_scoreboard.ROUND_FINALS
    )


def test_finals_metadata_orders_championship_before_third_place():
    games = [
        {"id": "third", "date": "2026-07-18T20:00:00Z"},
        {"id": "championship", "date": "2026-07-19T19:00:00Z"},
    ]

    result = world_cup_scoreboard._with_round_metadata(games, world_cup_scoreboard.ROUND_FINALS)

    assert [game["id"] for game in result] == ["championship", "third"]
    assert [game[world_cup_scoreboard.ROUND_GAME_KEY] for game in result] == [
        "Championship",
        "3rd Place",
    ]
    assert all(
        game[world_cup_scoreboard.ROUND_LABEL_KEY] == world_cup_scoreboard.ROUND_FINALS
        for game in result
    )


def test_pregame_score_text_defaults_to_dash(monkeypatch):
    monkeypatch.setattr(world_cup_scoreboard, "PREGAME_SCORE_DISPLAY", "dash")

    assert world_cup_scoreboard._score_text({"team": {"abbreviation": "USA"}}, show=False) == "—"


def test_pregame_score_text_can_show_team_abbreviation(monkeypatch):
    monkeypatch.setattr(world_cup_scoreboard, "PREGAME_SCORE_DISPLAY", "abbreviation")

    assert world_cup_scoreboard._score_text({"team": {"abbreviation": "usa"}}, show=False) == "USA"


def test_pregame_score_segments_use_status_font_for_team_abbreviation(monkeypatch):
    monkeypatch.setattr(world_cup_scoreboard, "PREGAME_SCORE_DISPLAY", "abbreviation")
    team = {"team": {"abbreviation": "can"}}

    assert world_cup_scoreboard._score_text_segments(team, show=False) == [
        ("CAN", world_cup_scoreboard.STATUS_FONT)
    ]
