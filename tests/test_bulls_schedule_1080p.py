"""Bulls Last/Next cards fill the 1080p HDMI canvas."""

import datetime as dt

from PIL import Image

import screens.draw_bulls_schedule as bulls


def _team(tri, name, city, score=None):
    return {"team": {"triCode": tri, "teamName": name, "teamCity": city}, "score": score}


def test_last_game_logos_use_the_row_height_at_1080p(monkeypatch):
    sizes = []
    monkeypatch.setattr(bulls, "_IS_1080P_LAYOUT", True)
    monkeypatch.setattr(bulls, "WIDTH", 1920)
    monkeypatch.setattr(bulls, "HEIGHT", 1080)
    monkeypatch.setattr(bulls, "_load_logo_png", lambda tri, h: sizes.append(h) or Image.new("RGBA", (h, h), "red"))
    game = {
        "gameDate": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gameStatusText": "Final",
        "teams": {"away": _team("CHI", "Bulls", "Chicago", 112), "home": _team("MIL", "Bucks", "Milwaukee", 105)},
    }
    img = bulls._render_scoreboard(game, title="Last Bulls game:", footer="Final")
    assert img.size == (1920, 1080)
    # Other profiles cap these logos at 64px; 1080p fills 80% of each row.
    assert len(sizes) == 2 and min(sizes) > 250
