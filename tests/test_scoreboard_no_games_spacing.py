import screens.nba_scoreboard as nba_scoreboard
import screens.nhl_scoreboard as nhl_scoreboard
import screens.scoreboard_components as scoreboard_components


class _DisplayStub:
    def image(self, _img):
        return None


def _capture_no_games_row(monkeypatch, module, center_text_owner, center_text_name):
    captured = {}

    def fake_center_text(_draw, text, _font, x, width, y, height, **_kwargs):
        captured["text"] = text
        captured["row"] = (x, width, y, height)

    monkeypatch.setattr(module, "clear_display", lambda _display: None)
    monkeypatch.setattr(module, "_get_league_logo", lambda: None)
    monkeypatch.setattr(center_text_owner, center_text_name, fake_center_text)

    return captured


def test_nba_and_nhl_no_games_messages_use_same_vertical_spacing(monkeypatch):
    nba_capture = _capture_no_games_row(
        monkeypatch,
        nba_scoreboard,
        nba_scoreboard,
        "_center_text",
    )
    nhl_capture = _capture_no_games_row(
        monkeypatch,
        nhl_scoreboard,
        scoreboard_components,
        "center_text",
    )

    nba_scoreboard.render_nba_scoreboard(_DisplayStub(), [], transition=True)
    nhl_scoreboard.render_nhl_scoreboard(_DisplayStub(), [], transition=True)

    assert nba_capture["text"] == "No games today"
    assert nhl_capture["text"] == "No games"
    assert nba_capture["row"] == (
        0,
        nba_scoreboard.WIDTH,
        nba_scoreboard.HEIGHT // 2 - nba_scoreboard.STATUS_ROW_H // 2,
        nba_scoreboard.STATUS_ROW_H,
    )
    assert nhl_capture["row"] == (
        0,
        nhl_scoreboard.WIDTH,
        nhl_scoreboard.HEIGHT // 2 - nhl_scoreboard.STATUS_ROW_H // 2,
        nhl_scoreboard.STATUS_ROW_H,
    )
