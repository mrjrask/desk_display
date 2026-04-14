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

    east_metro_top_vs_wc2 = projected[1]
    assert east_metro_top_vs_wc2["teams"]["home"]["team"]["abbreviation"] == "CAR"
    assert east_metro_top_vs_wc2["teams"]["away"]["team"]["abbreviation"] == "DET"

    assert projected[0]["higher_seed"] == 1
    assert projected[1]["higher_seed"] == 2
    assert projected[2]["higher_seed"] == 3
    assert projected[3]["higher_seed"] == 4


def test_format_next_text_uses_cdt():
    text = nhl_playoffs._format_next_text({"nextGameStartTimeUTC": "2026-04-20T00:30:00Z"})
    assert text == "Next: 4/19 7:30 PM CDT"


def test_conference_buckets_order_by_seed():
    series = [
        {"conference": "west", "higher_seed": 3, "lower_seed": 6, "teams": {"away": {"team": {"abbreviation": "AAA"}}, "home": {"team": {"abbreviation": "BBB"}}}},
        {"conference": "west", "higher_seed": 1, "lower_seed": 8, "teams": {"away": {"team": {"abbreviation": "CCC"}}, "home": {"team": {"abbreviation": "DDD"}}}},
        {"conference": "east", "higher_seed": 2, "lower_seed": 7, "teams": {"away": {"team": {"abbreviation": "EEE"}}, "home": {"team": {"abbreviation": "FFF"}}}},
        {"conference": "east", "higher_seed": 1, "lower_seed": 8, "teams": {"away": {"team": {"abbreviation": "GGG"}}, "home": {"team": {"abbreviation": "HHH"}}}},
    ]
    west, east = nhl_playoffs._conference_buckets(series)
    assert [item["higher_seed"] for item in west] == [1, 3]
    assert [item["higher_seed"] for item in east] == [1, 2]


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


def test_recompute_series_layout_keeps_two_columns_on_screen(monkeypatch):
    monkeypatch.setattr(nhl_playoffs, "WIDTH", 800)
    monkeypatch.setattr(nhl_playoffs, "PAIR_SPACING_BASE", 40)
    monkeypatch.setattr(nhl_playoffs, "SERIES_COL_WIDTHS_BASE", [105, 80, 40, 80, 105])

    nhl_playoffs._recompute_series_layout()

    assert nhl_playoffs.CONTENT_WIDTH <= nhl_playoffs.WIDTH
    assert nhl_playoffs.EAST_X + nhl_playoffs.SERIES_WIDTH <= nhl_playoffs.WIDTH


def test_fit_widths_to_total_preserves_total_and_min_width():
    fitted = nhl_playoffs._fit_widths_to_total([105, 80, 40, 80, 105], 400)

    assert sum(fitted) == 400
    assert all(width >= 1 for width in fitted)
