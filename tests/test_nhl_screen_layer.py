from __future__ import annotations

from PIL import Image

import screens.nhl_scoreboard as nhl_screen


class _DisplayStub:
    def image(self, _img):
        return None


def test_draw_nhl_scoreboard_fetches_prepared_data_from_service(monkeypatch):
    captured = {}

    monkeypatch.setattr(nhl_screen, "fetch_scoreboard", lambda: [{"gamePk": 7}])

    def _fake_render(display, games, transition=False):
        captured["display"] = display
        captured["games"] = games
        captured["transition"] = transition
        return Image.new("RGB", (10, 10))

    monkeypatch.setattr(nhl_screen, "render_nhl_scoreboard", _fake_render)

    display = _DisplayStub()
    nhl_screen.draw_nhl_scoreboard(display, transition=True)

    assert captured["display"] is display
    assert captured["games"] == [{"gamePk": 7}]
    assert captured["transition"] is True
