import datetime

from PIL import Image, ImageDraw

import config
from screens import draw_bears_schedule


def test_format_game_date_includes_day_only_for_non_sundays():
    assert draw_bears_schedule._format_game_date("Sun, Sep 13, 2026") == "9/13"
    assert draw_bears_schedule._format_game_date("Mon, Sep 28, 2026") == "Mon 9/28"
    assert draw_bears_schedule._format_game_date("Fri, Dec 25, 2026") == "Fri 12/25"


def test_bears_schedule_screen_filters_to_preseason_and_regular_season_by_default():
    today = datetime.date(2026, 6, 1)
    assert draw_bears_schedule._should_show_bears_schedule_game(
        {"week": "Preseason 1", "date": "TBD", "opponent": "Cleveland Browns"},
        today=today,
    )
    assert draw_bears_schedule._should_show_bears_schedule_game(
        {"week": "Week 18", "date": "TBD", "opponent": "Minnesota Vikings"},
        today=today,
    )
    assert not draw_bears_schedule._should_show_bears_schedule_game(
        {"week": "Wild Card", "date": "Sat, Jan 10, 2026", "opponent": "Green Bay Packers"},
        today=today,
    )
    assert draw_bears_schedule._should_show_bears_schedule_game(
        {"week": "Week 10", "date": "BYE", "opponent": "—"},
        today=today,
    )


def test_bears_schedule_screen_can_show_future_postseason_games():
    assert draw_bears_schedule._should_show_bears_schedule_game(
        {"week": "Wild Card", "date": "Sat, Jan 10, 2026", "opponent": "Green Bay Packers"},
        today=datetime.date(2026, 1, 1),
    )


def test_bears_schedule_week_labels_are_compact():
    assert draw_bears_schedule._format_bears_schedule_week_label("Preseason 3") == "P3"
    assert draw_bears_schedule._format_bears_schedule_week_label("Week 18") == "W18"


def test_bears_schedule_identifies_bye_week():
    assert draw_bears_schedule._is_bye_week({"week": "Week 10", "date": "BYE", "opponent": "—"})
    assert not draw_bears_schedule._is_bye_week(
        {"week": "Week 11", "date": "Sun, Nov 22, 2026", "opponent": "New Orleans Saints"}
    )


def test_fit_font_to_width_shrinks_without_truncating_text():
    img = Image.new("RGB", (config.WIDTH, config.HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    name = "Washington Commanders"
    fitted = draw_bears_schedule._fit_font_to_width(draw, name, config.FONT_DATE_SPORTS, 160)
    assert draw.textsize(name, font=fitted)[0] <= 160


def test_bears_schedule_opponent_team_name_removes_city_names():
    assert draw_bears_schedule._opponent_team_name("Washington Commanders") == "Commanders"
    assert draw_bears_schedule._opponent_team_name("San Francisco 49ers") == "49ers"
    assert draw_bears_schedule._opponent_team_name("Green Bay Packers") == "Packers"


def test_bears_next_game_reads_same_config_schedule_as_schedule_screen(monkeypatch):
    schedule = [
        {
            "game_no": "0.1",
            "week": "Preseason 1",
            "date": "TBD",
            "opponent": "Cleveland Browns",
            "home_away": "Home",
            "time": "TBD",
        }
    ]
    captured = {}

    def fake_next_game(schedule_arg, today=None):
        captured["schedule"] = schedule_arg
        return schedule_arg[0]

    monkeypatch.setattr(config, "BEARS_SCHEDULE", schedule)
    monkeypatch.setattr(draw_bears_schedule, "_next_bears_game_from_schedule", fake_next_game)

    image = draw_bears_schedule.show_bears_next_game(None, transition=True)

    assert image.size == (config.WIDTH, config.HEIGHT)
    assert captured["schedule"] == schedule


def test_next_bears_game_uses_tbd_preseason_before_dated_regular_season():
    schedule = [
        {
            "game_no": "0.1",
            "week": "Preseason 1",
            "date": "TBD",
            "opponent": "Cleveland Browns",
            "home_away": "Home",
            "time": "TBD",
        },
        {
            "game_no": "1",
            "week": "Week 1",
            "date": "Sun, Sep 13, 2026",
            "opponent": "Carolina Panthers",
            "home_away": "Away",
            "time": "Noon",
        },
    ]

    game = draw_bears_schedule._next_bears_game_from_schedule(
        schedule,
        today=datetime.date(2026, 6, 3),
    )

    assert game["week"] == "Preseason 1"


def test_next_bears_game_skips_bye_week_after_previous_game():
    schedule = [
        {
            "game_no": "9",
            "week": "Week 9",
            "date": "Sun, Nov 8, 2026",
            "opponent": "Tampa Bay Buccaneers",
            "home_away": "Home",
            "time": "7:20PM",
        },
        {
            "game_no": "10",
            "week": "Week 10",
            "date": "BYE",
            "opponent": "—",
            "home_away": "—",
            "time": "—",
        },
        {
            "game_no": "11",
            "week": "Week 11",
            "date": "Sun, Nov 22, 2026",
            "opponent": "New Orleans Saints",
            "home_away": "Home",
            "time": "Noon",
        },
    ]

    game = draw_bears_schedule._next_bears_game_from_schedule(
        schedule,
        today=datetime.date(2026, 11, 10),
    )

    assert game["week"] == "Week 11"


def test_next_bears_game_moves_past_same_day_final(monkeypatch):
    today = datetime.date(2026, 9, 13)
    schedule = [
        {
            "game_no": "1",
            "week": "Week 1",
            "date": "Sun, Sep 13, 2026",
            "opponent": "Carolina Panthers",
            "home_away": "Away",
            "time": "Noon",
        },
        {
            "game_no": "2",
            "week": "Week 2",
            "date": "Sun, Sep 20, 2026",
            "opponent": "Minnesota Vikings",
            "home_away": "Home",
            "time": "Noon",
        },
    ]

    def fake_score_text(game):
        return "F 28-31" if game["week"] == "Week 1" else ""

    monkeypatch.setattr(draw_bears_schedule, "_bears_schedule_score_text", fake_score_text)

    game = draw_bears_schedule._next_bears_game_from_schedule(schedule, today=today)

    assert game["week"] == "Week 2"


def test_bears_schedule_score_text_uses_nfl_scoreboard_feed(monkeypatch):
    past = datetime.date.today() - datetime.timedelta(days=1)
    game = {
        "week": "Week 1",
        "date": past.strftime("%a, %b %d, %Y"),
        "opponent": "Minnesota Vikings",
        "home_away": "Home",
    }

    def fake_fetch(day):
        assert day == past
        return [
            {
                "status": {"type": {"state": "post", "completed": True, "description": "Final"}},
                "competitors": [
                    {"homeAway": "away", "score": "17", "team": {"abbreviation": "MIN"}},
                    {"homeAway": "home", "score": "24", "team": {"abbreviation": "CHI"}},
                ],
            }
        ]

    monkeypatch.setattr(draw_bears_schedule, "_fetch_bears_scoreboard_games_for_date", fake_fetch)

    assert draw_bears_schedule._scoreboard_scores_for_bears_game(game) == (24, 17)
    assert draw_bears_schedule._bears_schedule_score_text(game) == "F 17-24"


def test_bears_schedule_score_text_uses_scoreboard_perspective_for_away_game(monkeypatch):
    past = datetime.date.today() - datetime.timedelta(days=1)
    game = {
        "week": "Week 1",
        "date": past.strftime("%a, %b %d, %Y"),
        "opponent": "Carolina Panthers",
        "home_away": "Away",
    }

    monkeypatch.setattr(
        draw_bears_schedule,
        "_fetch_bears_scoreboard_games_for_date",
        lambda day: [
            {
                "status": {"type": {"state": "post", "completed": True}},
                "competitors": [
                    {"homeAway": "away", "score": 31, "team": {"abbreviation": "CHI"}},
                    {"homeAway": "home", "score": 28, "team": {"abbreviation": "CAR"}},
                ],
            }
        ],
    )

    assert draw_bears_schedule._scoreboard_scores_for_bears_game(game) == (28, 31)
    assert draw_bears_schedule._bears_schedule_score_text(game) == "F 28-31"


def test_bears_schedule_score_fetch_skips_future_dates(monkeypatch):
    future = datetime.date.today() + datetime.timedelta(days=30)
    calls = []
    monkeypatch.setattr(
        draw_bears_schedule,
        "_fetch_bears_scoreboard_games_for_date",
        lambda day: calls.append(day),
    )

    assert draw_bears_schedule._scoreboard_scores_for_bears_game(
        {
            "week": "Week 1",
            "date": future.strftime("%a, %b %d, %Y"),
            "opponent": "Minnesota Vikings",
            "home_away": "Home",
        }
    ) is None
    assert calls == []


def test_bears_schedule_scroll_uses_scoreboard_scroll_settings(monkeypatch):
    calls = []

    def fake_scroll_vertical_content(**kwargs):
        calls.append(kwargs)

    class FakeDisplay:
        def image(self, image):
            self.last_image = image

    monkeypatch.setattr(draw_bears_schedule, "scroll_vertical_content", fake_scroll_vertical_content)

    result = draw_bears_schedule.show_bears_next_season_sched(FakeDisplay())

    assert result.displayed is True
    assert calls
    assert "page_jump_mode" not in calls[0]
    assert calls[0]["base_step"] == config.SCOREBOARD_SCROLL_STEP
    assert calls[0]["pause_start"] == config.SCOREBOARD_SCROLL_PAUSE_TOP
    assert calls[0]["pause_end"] == config.SCOREBOARD_SCROLL_PAUSE_BOTTOM
    assert calls[0]["min_frame_time"] == config.SCOREBOARD_SCROLL_DELAY
