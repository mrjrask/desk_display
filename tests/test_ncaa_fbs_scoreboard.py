import datetime as dt
import logging

from screens import ncaa_fbs_scoreboard


def test_fetch_uses_fbs_group_and_keeps_only_top_25_games(monkeypatch):
    captured = {}

    def fake_fetch(params):
        captured.update(params)
        return {
            "events": [
                {"id": "ranked", "competitions": [{"competitors": [{"curatedRank": {"current": 8}}]}]},
                {"id": "unranked", "competitions": [{"competitors": [{"curatedRank": {"current": 99}}]}]},
            ]
        }

    monkeypatch.setattr(ncaa_fbs_scoreboard, "_fetch_json", fake_fetch)

    games = ncaa_fbs_scoreboard._fetch_games_for_date(dt.date(2026, 9, 26))

    assert captured == {"dates": "20260926", "limit": 300, "groups": 80}
    assert [game["id"] for game in games] == ["ranked"]


def test_missing_team_logo_logs_expected_filename(monkeypatch, caplog):
    monkeypatch.setattr(ncaa_fbs_scoreboard.os.path, "exists", lambda path: False)
    ncaa_fbs_scoreboard._REMOTE_LOGO_CACHE.clear()

    with caplog.at_level(logging.WARNING):
        logo = ncaa_fbs_scoreboard._load_team_logo(
            {"team": {"abbreviation": "MINN"}}, 32
        )

    assert logo is None
    assert "expected filename: MINN.png" in caplog.text
