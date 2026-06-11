import screens.nba_scoreboard as nba_scoreboard
import screens.nhl_scoreboard as nhl_scoreboard


class _DisplayStub:
    def image(self, _img):
        return None


def _capture_no_games_row(monkeypatch, module):
    captured = {}

    def fake_center_text(_draw, text, _font, x, width, y, height, **_kwargs):
        captured["text"] = text
        captured["row"] = (x, width, y, height)

    monkeypatch.setattr(module, "clear_display", lambda _display: None)
    monkeypatch.setattr(module, "_get_league_logo", lambda: None)
    monkeypatch.setattr(module, "_center_text", fake_center_text)

    return captured


def test_nba_and_nhl_no_games_messages_use_same_vertical_spacing(monkeypatch):
    nba_capture = _capture_no_games_row(monkeypatch, nba_scoreboard)
    nhl_capture = _capture_no_games_row(monkeypatch, nhl_scoreboard)

    nba_scoreboard.render_nba_scoreboard(_DisplayStub(), [], transition=True)
    nhl_scoreboard.render_nhl_scoreboard(_DisplayStub(), [], transition=True)

    assert nba_capture["text"] == "No games today"
    assert nhl_capture["text"] == "No games"
    assert nba_capture["row"][2:] == nhl_capture["row"][2:]
    assert nba_capture["row"][2] == nba_scoreboard.HEIGHT // 2 - nba_scoreboard.STATUS_ROW_H // 2
    assert nhl_capture["row"][2] == nhl_scoreboard.HEIGHT // 2 - nhl_scoreboard.STATUS_ROW_H // 2
