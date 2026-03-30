import logging

import data_fetch
import screens.mlb_team_standings as mlb_team_standings
from services.network import ConnectivityMonitor


def test_connectivity_monitor_check_internet_logs_oserror(monkeypatch, caplog):
    class _FakeSocket:
        def close(self):
            return None

    def _raise_oserror(*_args, **_kwargs):
        raise OSError("network unreachable")

    monitor = ConnectivityMonitor.__new__(ConnectivityMonitor)

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr("socket.create_connection", _raise_oserror)

    assert monitor._check_internet() is False
    assert "connectivity_check failed" in caplog.text

    monkeypatch.setattr("socket.create_connection", lambda *_args, **_kwargs: _FakeSocket())
    assert monitor._check_internet() is True


def test_standings_screen1_logs_logo_fallback(monkeypatch, caplog):
    rec = {
        "leagueRecord": {"wins": 10, "losses": 5, "pct": ".667"},
        "divisionRank": "1",
        "divisionGamesBack": "0",
        "wildCardGamesBack": 0,
        "records": {"splitRecords": []},
    }

    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(mlb_team_standings.Image, "open", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad image")))

    image = mlb_team_standings.draw_standings_screen1(
        display=None,
        rec=rec,
        logo_path="/tmp/invalid-logo.png",
        division_name="NL Central",
        screen_id="cubs stand1",
        transition=True,
    )

    assert image is not None
    assert "draw_standings_screen1 logo fallback" in caplog.text
    assert "cubs stand1" in caplog.text


def test_parse_datetime_candidates_falls_back_after_bad_iso():
    row = {
        "game_date_time": "not-a-date",
        "game_date": "2026-03-30",
        "game_time": "7:15 PM",
    }

    parsed = data_fetch._parse_datetime_candidates(row)

    assert parsed is not None
    assert parsed.hour == 19
    assert parsed.minute == 15
