import datetime

import pytest

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
    assert all(item.get("status_text") == "" for item in projected)

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


def test_format_next_text_omits_timezone():
    text = nhl_playoffs._format_next_text({"nextGameStartTimeUTC": "2026-04-20T00:30:00Z"})
    assert text == "Next: 4/19 7:30 PM"


def test_format_next_text_supports_date_only_without_time():
    text = nhl_playoffs._format_next_text({"nextGameDate": "2026-04-27"})
    assert text == "Next: 4/27"


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


def test_normalize_series_item_reads_nested_wins_and_hides_projected_status():
    normalized = nhl_playoffs._normalize_series_item(
        {
            "topSeedTeam": {"abbreviation": "DAL", "wins": 3},
            "bottomSeedTeam": {"abbreviation": "COL", "seriesWins": 2},
            "seriesStatus": "Projected",
            "nextGameDateTimeUTC": "2026-04-20T02:00:00Z",
        }
    )

    assert normalized is not None
    assert normalized["teams"]["away"]["score"] == 3
    assert normalized["teams"]["home"]["score"] == 2
    assert normalized["status_text"] == ""
    assert normalized["next_text"] == "Next: 4/19 9:00 PM"


def test_normalize_series_item_supports_rounds_series_topseed_bottomseed_shape():
    normalized = nhl_playoffs._normalize_series_item(
        {
            "seriesLetter": "B",
            "topSeed": {"abbrev": "TBL", "wins": 1},
            "bottomSeed": {"abbrev": "MTL", "wins": 2},
        }
    )

    assert normalized is not None
    assert normalized["teams"]["away"]["team"]["abbrev"] == "TBL"
    assert normalized["teams"]["home"]["team"]["abbrev"] == "MTL"
    assert normalized["teams"]["away"]["score"] == 1
    assert normalized["teams"]["home"]["score"] == 2
    assert normalized["conference"] == "east"


def test_series_next_text_from_games_uses_upcoming_matchup_time(monkeypatch):
    series = {
        "teams": {
            "away": {"team": {"abbrev": "BUF"}, "score": 1},
            "home": {"team": {"abbrev": "BOS"}, "score": 0},
        },
        "next_text": "Next: TBD",
    }
    games = [
        {
            "gameDate": "2026-04-18T01:00:00Z",
            "teams": {
                "away": {"team": {"abbreviation": "BUF"}},
                "home": {"team": {"abbreviation": "BOS"}},
            },
        },
        {
            "gameDate": "2026-04-21T01:00:00Z",
            "teams": {
                "away": {"team": {"abbreviation": "BOS"}},
                "home": {"team": {"abbreviation": "BUF"}},
            },
        },
    ]

    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 20, 12, 0, tzinfo=tz)

    monkeypatch.setattr(nhl_playoffs.datetime, "datetime", _FixedNow)
    assert nhl_playoffs._series_next_text_from_games(series, games) == "Next: Tonight 8:00 PM"


def test_series_next_text_from_games_uses_date_only_when_time_is_tbd(monkeypatch):
    series = {
        "teams": {
            "away": {"team": {"abbrev": "BUF"}, "score": 1},
            "home": {"team": {"abbrev": "BOS"}, "score": 0},
        },
        "next_text": "Next: TBD",
    }
    games = [
        {
            "gameDate": "2026-04-27",
            "teams": {
                "away": {"team": {"abbreviation": "BUF"}},
                "home": {"team": {"abbreviation": "BOS"}},
            },
        },
    ]

    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 20, 12, 0, tzinfo=tz)

    monkeypatch.setattr(nhl_playoffs.datetime, "datetime", _FixedNow)
    assert nhl_playoffs._series_next_text_from_games(series, games) == "Next: 4/27"


def test_normalize_next_text_strips_known_timezone_and_leading_zeroes():
    assert nhl_playoffs._normalize_next_text("Next: 04/09 8:00 PM ET") == "Next: 4/9 8:00 PM"


def test_normalize_next_text_keeps_text_without_timezone_unchanged():
    assert nhl_playoffs._normalize_next_text("Next: 4/9 8:00 PM") == "Next: 4/9 8:00 PM"


def test_normalize_next_text_keeps_meridiem_suffix():
    assert nhl_playoffs._normalize_next_text("Next: 4/9 8:00 AM") == "Next: 4/9 8:00 AM"


def test_normalize_next_text_preserves_tbd():
    assert nhl_playoffs._normalize_next_text("Next: TBD") == "Next: TBD"


def test_normalize_next_text_uses_tonight_when_date_matches_today(monkeypatch):
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 9, 10, 0, tzinfo=tz)

    monkeypatch.setattr(nhl_playoffs.datetime, "datetime", _FixedNow)
    assert nhl_playoffs._normalize_next_text("Next: 04/09 8:00 PM ET") == "Next: Tonight 8:00 PM"


def test_normalize_next_text_uses_tomorrow_when_date_matches_next_day(monkeypatch):
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 8, 10, 0, tzinfo=tz)

    monkeypatch.setattr(nhl_playoffs.datetime, "datetime", _FixedNow)
    assert nhl_playoffs._normalize_next_text("Next: 04/09 8:00 PM ET") == "Next: Tomorrow 8:00 PM"


def test_team_abbr_supports_localized_team_abbrev_shapes():
    assert nhl_playoffs._team_abbr({"teamAbbrev": {"default": "WPG"}}) == "WPG"
    assert nhl_playoffs._team_abbr({"abbrev": {"default": "TOR"}}) == "TOR"
    assert nhl_playoffs._team_abbr({"team": {"triCode": "car"}}) == "CAR"


def test_team_abbr_prefers_nested_canonical_abbreviation_over_localized_short_name():
    team = {"shortName": {"default": "Rangers"}, "team": {"abbreviation": "NYR"}}

    assert nhl_playoffs._team_abbr(team) == "NYR"


def test_first_present_int_supports_record_text():
    assert nhl_playoffs._first_present_int(["3-2", None]) == 3


def test_apply_style_overrides_reduces_logo_height_by_ten_percent(monkeypatch):
    monkeypatch.setattr(nhl_playoffs, "TEAM_LOGO_BASE_HEIGHT", 20)
    monkeypatch.setattr(nhl_playoffs, "PLAYOFF_LOGO_SCALE", 0.9)
    monkeypatch.setattr(nhl_playoffs, "SCORE_ROW_H", 56)
    monkeypatch.setattr(nhl_playoffs, "get_screen_image_scale", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(nhl_playoffs, "scale_value", lambda value: value)
    monkeypatch.setattr(nhl_playoffs, "_scoreboard_fonts", lambda: (None, None, None, None))
    monkeypatch.setattr(nhl_playoffs, "_recompute_series_layout", lambda: None)

    nhl_playoffs._apply_style_overrides()

    assert nhl_playoffs.LOGO_HEIGHT == 18


def test_format_next_text_supports_nested_next_game_schedule(monkeypatch):
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 20, 12, 0, tzinfo=tz)

    monkeypatch.setattr(nhl_playoffs.datetime, "datetime", _FixedNow)
    text = nhl_playoffs._format_next_text(
        {
            "nextGameSchedule": {
                "scheduledStartTimeUTC": "2026-04-22T01:00:00Z",
            }
        }
    )
    assert text == "Next: Tomorrow 8:00 PM"
