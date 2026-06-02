import datetime

from PIL import Image, ImageDraw

import config
from screens import draw_bears_schedule


def test_format_game_date_includes_day_only_for_non_sundays():
    assert draw_bears_schedule._format_game_date("Sun, Sep 14, 2025") == "9/14"
    assert draw_bears_schedule._format_game_date("Mon, Sep 8, 2025") == "Mon 9/8"
    assert draw_bears_schedule._format_game_date("Fri, Nov 28, 2025") == "Fri 11/28"


def test_bears_schedule_screen_filters_to_preseason_and_regular_season_by_default():
    today = datetime.date(2026, 6, 1)
    assert draw_bears_schedule._should_show_bears_schedule_game(
        {"week": "Preseason 1", "date": "Sat, Aug 9, 2025", "opponent": "Miami Dolphins"},
        today=today,
    )
    assert draw_bears_schedule._should_show_bears_schedule_game(
        {"week": "Week 18", "date": "Sun, Jan 4, 2026", "opponent": "Detroit Lions"},
        today=today,
    )
    assert not draw_bears_schedule._should_show_bears_schedule_game(
        {"week": "Wild Card", "date": "Sat, Jan 10, 2026", "opponent": "Green Bay Packers"},
        today=today,
    )
    assert not draw_bears_schedule._should_show_bears_schedule_game(
        {"week": "Week 5", "date": "BYE", "opponent": "—"},
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
