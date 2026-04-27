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
    assert text == "4/19 7:30 PM"


def test_format_next_text_supports_date_only_without_time(monkeypatch):
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 20, 12, 0, tzinfo=tz)

    monkeypatch.setattr(nhl_playoffs.datetime, "datetime", _FixedNow)
    text = nhl_playoffs._format_next_text({"nextGameDate": "2026-04-27"})
    assert text == "4/27"


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
    assert normalized["next_text"] == "4/19 9 PM"


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
        "next_text": "TBD",
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
    assert nhl_playoffs._series_next_text_from_games(series, games) == "Tonight 8 PM"


def test_series_next_text_from_games_uses_date_only_when_time_is_tbd(monkeypatch):
    series = {
        "teams": {
            "away": {"team": {"abbrev": "BUF"}, "score": 1},
            "home": {"team": {"abbrev": "BOS"}, "score": 0},
        },
        "next_text": "TBD",
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
    assert nhl_playoffs._series_next_text_from_games(series, games) == "4/27"


def test_normalize_next_text_strips_known_timezone_and_leading_zeroes():
    assert nhl_playoffs._normalize_next_text("Next: 04/09 8:00 PM ET") == "4/9 8 PM"


def test_normalize_next_text_keeps_text_without_timezone_unchanged():
    assert nhl_playoffs._normalize_next_text("Next: 4/9 8:00 PM") == "4/9 8 PM"


def test_normalize_next_text_keeps_meridiem_suffix():
    assert nhl_playoffs._normalize_next_text("Next: 4/9 8:00 AM") == "4/9 8 AM"


def test_normalize_next_text_preserves_tbd():
    assert nhl_playoffs._normalize_next_text("Next: TBD") == "TBD"


def test_normalize_next_text_uses_tonight_when_date_matches_today(monkeypatch):
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 9, 10, 0, tzinfo=tz)

    monkeypatch.setattr(nhl_playoffs.datetime, "datetime", _FixedNow)
    assert nhl_playoffs._normalize_next_text("Next: 04/09 8:00 PM ET") == "Tonight 8 PM"


def test_normalize_next_text_uses_tomorrow_when_date_matches_next_day(monkeypatch):
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 8, 10, 0, tzinfo=tz)

    monkeypatch.setattr(nhl_playoffs.datetime, "datetime", _FixedNow)
    assert nhl_playoffs._normalize_next_text("Next: 04/09 8:00 PM ET") == "Tomorrow 8 PM"


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
    assert text == "Tomorrow 8 PM"


def test_fetch_remaining_playoff_schedule_games_includes_upcoming_through_june(monkeypatch):
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 20, 12, 0, tzinfo=tz)

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    payload = {
        "gameWeek": [
            {
                "games": [
                    {
                        "id": 1,
                        "startTimeUTC": "2026-05-02T00:00:00Z",
                        "gameType": 3,
                        "awayTeam": {"abbrev": "WPG"},
                        "homeTeam": {"abbrev": "DAL"},
                    },
                    {
                        "id": 2,
                        "gameDate": "2026-07-02T00:00:00Z",
                        "gameType": 3,
                        "awayTeam": {"abbrev": "NYR"},
                        "homeTeam": {"abbrev": "CAR"},
                    },
                ]
            }
        ]
    }

    monkeypatch.setattr(nhl_playoffs.datetime, "datetime", _FixedNow)
    monkeypatch.setattr(
        nhl_playoffs._SESSION,
        "get",
        lambda *_args, **_kwargs: _Response(payload),
    )

    games = nhl_playoffs._fetch_remaining_playoff_schedule_games()
    assert len(games) == 1
    assert games[0]["gamePk"] == 1


def test_fetch_remaining_playoff_schedule_games_excludes_non_playoff_game_types(monkeypatch):
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 20, 12, 0, tzinfo=tz)

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    payload = {
        "gameWeek": [
            {
                "games": [
                    {
                        "id": 10,
                        "startTimeUTC": "2026-05-03T00:00:00Z",
                        "gameType": 2,
                        "awayTeam": {"abbrev": "WPG"},
                        "homeTeam": {"abbrev": "DAL"},
                    },
                    {
                        "id": 11,
                        "startTimeUTC": "2026-05-04T00:00:00Z",
                        "gameType": 3,
                        "awayTeam": {"abbrev": "VGK"},
                        "homeTeam": {"abbrev": "EDM"},
                    },
                ]
            }
        ]
    }

    monkeypatch.setattr(nhl_playoffs.datetime, "datetime", _FixedNow)
    monkeypatch.setattr(
        nhl_playoffs._SESSION,
        "get",
        lambda *_args, **_kwargs: _Response(payload),
    )

    games = nhl_playoffs._fetch_remaining_playoff_schedule_games()
    assert len(games) == 1
    assert games[0]["gamePk"] == 11


def test_render_nhl_playoffs_enriches_next_text_with_official_schedule(monkeypatch):
    class _DisplayStub:
        def image(self, _img):
            return None

    monkeypatch.setattr(nhl_playoffs, "_apply_style_overrides", lambda: None)
    monkeypatch.setattr(
        nhl_playoffs,
        "_fetch_playoff_matchups",
        lambda: [
            {
                "teams": {
                    "away": {"team": {"abbreviation": "WPG"}, "score": 1},
                    "home": {"team": {"abbreviation": "DAL"}, "score": 1},
                },
                "next_text": "TBD",
                "conference": "west",
            }
        ],
    )
    monkeypatch.setattr(nhl_playoffs, "_fetch_projected_matchups_from_standings", lambda: [])
    monkeypatch.setattr(
        nhl_playoffs,
        "_fetch_remaining_playoff_schedule_games",
        lambda: [{"gameDate": "2026-04-28", "teams": {"away": {"team": {"abbreviation": "WPG"}}, "home": {"team": {"abbreviation": "DAL"}}}}],
    )
    captured = {}

    def _render(series):
        captured["series"] = series
        return __import__("PIL").Image.new("RGB", (10, 10))

    monkeypatch.setattr(nhl_playoffs, "_render_playoff_screen", _render)
    monkeypatch.setattr(nhl_playoffs.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nhl_playoffs, "HEIGHT", 100)
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 20, 12, 0, tzinfo=tz)
    monkeypatch.setattr(nhl_playoffs.datetime, "datetime", _FixedNow)

    rendered = nhl_playoffs.render_nhl_playoffs(_DisplayStub(), games=[], transition=False)
    assert rendered.displayed is True
    assert captured["series"][0]["next_text"] == "4/28"


def test_select_current_round_series_keeps_completed_first_round_visible():
    series = [
        {"teams": {"away": {"team": {"abbreviation": "DAL"}, "score": 4}, "home": {"team": {"abbreviation": "COL"}, "score": 1}}, "round_rank": 1},
        {"teams": {"away": {"team": {"abbreviation": "WPG"}, "score": 3}, "home": {"team": {"abbreviation": "STL"}, "score": 2}}, "round_rank": 1},
        {"teams": {"away": {"team": {"abbreviation": "DAL"}, "score": 0}, "home": {"team": {"abbreviation": "WPG"}, "score": 0}, "next_text": "TBD"}, "round_rank": 2},
    ]

    selected = nhl_playoffs._select_current_round_series(series)

    assert len(selected) == 2
    assert all(item["round_rank"] == 1 for item in selected)


def test_select_current_round_series_ignores_opponentless_next_round_series():
    series = [
        {"teams": {"away": {"team": {"abbreviation": "DAL"}, "score": 4}, "home": {"team": {"abbreviation": "COL"}, "score": 1}}, "round_rank": 1},
        {"teams": {"away": {"team": {"abbreviation": "WPG"}, "score": 2}, "home": {"team": {"abbreviation": "STL"}, "score": 2}}, "round_rank": 1},
        {"teams": {"away": {"team": {"abbreviation": "DAL"}, "score": 0}, "home": {"team": {}, "score": 0}, "next_text": "Tomorrow 7 PM"}, "round_rank": 2},
    ]

    selected = nhl_playoffs._select_current_round_series(series)

    assert len(selected) == 2
    assert all(item["round_rank"] == 1 for item in selected)


def test_select_current_round_series_advances_only_when_next_round_started():
    series = [
        {"teams": {"away": {"team": {"abbreviation": "DAL"}, "score": 4}, "home": {"team": {"abbreviation": "COL"}, "score": 1}}, "round_rank": 1},
        {"teams": {"away": {"team": {"abbreviation": "WPG"}, "score": 4}, "home": {"team": {"abbreviation": "STL"}, "score": 2}}, "round_rank": 1},
        {"teams": {"away": {"team": {"abbreviation": "DAL"}, "score": 0}, "home": {"team": {"abbreviation": "WPG"}, "score": 0}, "next_text": "TBD"}, "round_rank": 2},
    ]
    assert all(item["round_rank"] == 1 for item in nhl_playoffs._select_current_round_series(series))

    series[2]["next_text"] = "Tomorrow 7 PM"
    selected = nhl_playoffs._select_current_round_series(series)
    assert len(selected) == 1
    assert selected[0]["round_rank"] == 2
