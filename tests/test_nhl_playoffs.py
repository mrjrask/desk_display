from screens import nhl_playoffs


def _row(team, conference, division, points):
    return {
        "teamAbbrev": {"default": team},
        "teamName": {"default": team},
        "conferenceAbbrev": conference,
        "divisionAbbrev": division,
        "points": points,
    }


def test_projected_matchups_from_standings_builds_first_round_bracket():
    standings = [
        _row("TOR", "E", "A", 110),
        _row("TBL", "E", "A", 104),
        _row("FLA", "E", "A", 101),
        _row("MTL", "E", "A", 88),
        _row("CAR", "E", "M", 111),
        _row("NJD", "E", "M", 103),
        _row("NYR", "E", "M", 100),
        _row("CBJ", "E", "M", 86),
        _row("DAL", "W", "C", 112),
        _row("COL", "W", "C", 106),
        _row("MIN", "W", "C", 102),
        _row("STL", "W", "C", 84),
        _row("VGK", "W", "P", 109),
        _row("LAK", "W", "P", 105),
        _row("EDM", "W", "P", 99),
        _row("SEA", "W", "P", 82),
        _row("OTT", "E", "A", 98),  # East WC1
        _row("DET", "E", "M", 97),  # East WC2
        _row("WPG", "W", "C", 101),  # West WC1
        _row("VAN", "W", "P", 100),  # West WC2
    ]

    projected = nhl_playoffs._projected_matchups_from_standings(standings)

    assert len(projected) == 8
    assert all(item.get("status_text") == "Projected" for item in projected)

    east_atlantic_top_vs_wc1 = projected[0]
    assert east_atlantic_top_vs_wc1["teams"]["home"]["team"]["abbreviation"] == "TOR"
    assert east_atlantic_top_vs_wc1["teams"]["away"]["team"]["abbreviation"] == "OTT"

    east_metro_top_vs_wc2 = projected[2]
    assert east_metro_top_vs_wc2["teams"]["home"]["team"]["abbreviation"] == "CAR"
    assert east_metro_top_vs_wc2["teams"]["away"]["team"]["abbreviation"] == "DET"


def test_render_uses_projected_standings_when_no_live_playoff_series(monkeypatch):
    monkeypatch.setattr(nhl_playoffs, "_fetch_playoff_matchups", lambda: [])
    monkeypatch.setattr(
        nhl_playoffs,
        "_fetch_projected_matchups_from_standings",
        lambda: [
            {
                "teams": {
                    "away": {"team": {"abbreviation": "AWY"}, "score": 90},
                    "home": {"team": {"abbreviation": "HME"}, "score": 100},
                },
                "status_text": "Projected",
            }
        ],
    )
    monkeypatch.setattr(nhl_playoffs, "_derive_playoff_matchups_from_games", lambda games: [])

    class _DisplayStub:
        def image(self, _img):
            return None

    rendered = nhl_playoffs.render_nhl_playoffs(_DisplayStub(), games=[], transition=False)
    assert rendered.displayed is True
