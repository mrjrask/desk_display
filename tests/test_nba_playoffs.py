import datetime

from screens import nba_playoffs


def test_normalize_series_item_supports_matchup_shape_and_next_game_time():
    normalized = nba_playoffs._normalize_series_item(
        {
            "topSeed": {"team": {"teamTricode": "BOS"}, "wins": 3},
            "bottomSeed": {"team": {"teamTricode": "MIA"}, "seriesWins": 1},
            "seriesStatus": "BOS leads 3-1",
            "nextGameDateTime": "2026-04-21T00:00:00Z",
        }
    )

    assert normalized is not None
    assert normalized["teams"]["away"]["team"]["teamTricode"] == "BOS"
    assert normalized["teams"]["home"]["team"]["teamTricode"] == "MIA"
    assert normalized["teams"]["away"]["score"] == 3
    assert normalized["teams"]["home"]["score"] == 1
    assert normalized["next_text"]


def test_extract_series_reads_matchups_key():
    extracted = nba_playoffs._extract_series(
        {
            "matchups": [
                {
                    "awayTeam": {"teamTricode": "NYK"},
                    "homeTeam": {"teamTricode": "DET"},
                    "awayWins": 2,
                    "homeWins": 2,
                }
            ]
        }
    )

    assert len(extracted) == 1
    assert extracted[0]["teams"]["away"]["team"]["teamTricode"] == "NYK"
    assert extracted[0]["teams"]["home"]["team"]["teamTricode"] == "DET"


def test_extract_series_ignores_non_series_matchups_without_playoff_shape():
    extracted = nba_playoffs._extract_series(
        {
            "matchups": [
                {
                    "awayTeam": {"teamTricode": "BOS"},
                    "homeTeam": {"teamTricode": "NYK"},
                    "gameDate": "2026-04-21T00:00:00Z",
                }
            ]
        }
    )

    assert extracted == []


def test_extract_series_accepts_nested_team_wins_without_top_level_series_keys():
    extracted = nba_playoffs._extract_series(
        {
            "matchups": [
                {
                    "awayTeam": {"teamTricode": "BOS", "wins": 2},
                    "homeTeam": {"teamTricode": "NYK", "seriesWins": 1},
                }
            ]
        }
    )

    assert len(extracted) == 1
    assert extracted[0]["teams"]["away"]["score"] == 2
    assert extracted[0]["teams"]["home"]["score"] == 1


def test_normalize_next_text_strips_timezone_suffix():
    assert nba_playoffs._normalize_next_text("Next: 04/09 8:00 PM ET") == "4/9 8:00 PM"


def test_format_next_text_uses_tonight_label(monkeypatch):
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 20, 10, 0, tzinfo=tz)

    monkeypatch.setattr(nba_playoffs.datetime, "datetime", _FixedNow)
    text = nba_playoffs._format_next_text({"nextGameStartTimeUTC": "2026-04-21T02:30:00Z"})
    assert text == "Tonight 9:30 PM"


def test_format_next_text_supports_next_game_datetime_utc_key():
    text = nba_playoffs._format_next_text({"nextGameDateTimeUTC": "2026-04-22T00:00:00Z"})
    assert text == "4/21 7:00 PM"


def test_format_next_text_supports_nested_next_game_payload(monkeypatch):
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 20, 10, 0, tzinfo=tz)

    monkeypatch.setattr(nba_playoffs.datetime, "datetime", _FixedNow)
    text = nba_playoffs._format_next_text({"nextGame": {"gameDateTimeUTC": "2026-04-23T00:30:00Z"}})
    assert text == "4/22 7:30 PM"


def test_derive_playoff_matchups_supports_scoreboard_games_shape():
    games = [
        {
            "gamePk": "0042600001",
            "teams": {
                "away": {"team": {"abbreviation": "BOS"}},
                "home": {"team": {"abbreviation": "NYK"}},
            },
        }
    ]

    derived = nba_playoffs._derive_playoff_matchups_from_games(games)

    assert len(derived) == 1
    assert derived[0]["teams"]["away"]["team"]["abbreviation"] == "BOS"
    assert derived[0]["teams"]["home"]["team"]["abbreviation"] == "NYK"


def test_derive_playoff_matchups_counts_wins_from_finals():
    games = [
        {
            "gamePk": "0042600101",
            "gameDate": "2026-04-18T00:00:00Z",
            "status": {"statusCode": "3", "detailedState": "Final"},
            "teams": {
                "away": {"team": {"abbreviation": "BOS"}, "score": 100},
                "home": {"team": {"abbreviation": "NYK"}, "score": 90},
            },
        },
        {
            "gamePk": "0042600102",
            "gameDate": "2026-04-20T00:00:00Z",
            "status": {"statusCode": "3", "detailedState": "Final"},
            "teams": {
                "away": {"team": {"abbreviation": "BOS"}, "score": 95},
                "home": {"team": {"abbreviation": "NYK"}, "score": 101},
            },
        },
    ]

    derived = nba_playoffs._derive_playoff_matchups_from_games(games)

    assert len(derived) == 1
    assert derived[0]["teams"]["away"]["score"] == 1
    assert derived[0]["teams"]["home"]["score"] == 1


def test_parse_series_record_ignores_non_series_score_text():
    assert nba_playoffs._parse_series_record_from_text("Final 117-99") is None


def test_derive_playoff_matchups_filters_non_playoff_games_when_playoff_game_present():
    games = [
        {
            "gamePk": "0022500001",
            "status": {"statusCode": "3", "detailedState": "Final"},
            "teams": {
                "away": {"team": {"abbreviation": "BOS"}, "score": 120},
                "home": {"team": {"abbreviation": "NYK"}, "score": 100},
            },
        },
        {
            "gamePk": "0042500101",
            "status": {"statusCode": "3", "detailedState": "Final"},
            "teams": {
                "away": {"team": {"abbreviation": "BOS"}, "score": 98},
                "home": {"team": {"abbreviation": "NYK"}, "score": 95},
            },
        },
    ]

    derived = nba_playoffs._derive_playoff_matchups_from_games(games)

    assert len(derived) == 1
    assert derived[0]["teams"]["away"]["score"] == 1
    assert derived[0]["teams"]["home"]["score"] == 0


def test_is_current_series_filters_completed_series():
    series = {
        "teams": {
            "away": {"score": 4},
            "home": {"score": 2},
        }
    }

    assert nba_playoffs._is_current_series(series) is False


def test_looks_like_playoff_game_accepts_espn_postseason_type():
    game = {
        "gamePk": "401999999",
        "seasonType": "3",
        "status": {"detailedState": "Scheduled"},
    }

    assert nba_playoffs._looks_like_playoff_game(game) is True


def test_looks_like_playoff_game_rejects_play_in_game_id_prefix():
    game = {
        "gamePk": "0052600101",
        "status": {"detailedState": "Final"},
    }

    assert nba_playoffs._looks_like_playoff_game(game) is False


def test_extract_series_dedupes_when_home_away_flipped():
    extracted = nba_playoffs._extract_series(
        {
            "series": [
                {
                    "awayTeam": {"teamTricode": "GSW"},
                    "homeTeam": {"teamTricode": "LAL"},
                    "awayWins": 1,
                    "homeWins": 1,
                },
                {
                    "awayTeam": {"teamTricode": "LAL"},
                    "homeTeam": {"teamTricode": "GSW"},
                    "awayWins": 1,
                    "homeWins": 1,
                },
            ]
        }
    )

    assert len(extracted) == 1


def test_select_current_round_series_prefers_lowest_round_rank():
    series = [
        {"teams": {"away": {"score": 1}, "home": {"score": 1}}, "round_rank": 1},
        {"teams": {"away": {"score": 2}, "home": {"score": 0}}, "round_rank": 1},
        {"teams": {"away": {"score": 0}, "home": {"score": 0}}, "round_rank": 2},
    ]

    selected = nba_playoffs._select_current_round_series(series)

    assert len(selected) == 2
    assert all(item["round_rank"] == 1 for item in selected)


def test_select_current_round_series_keeps_all_when_round_unknown():
    series = [
        {"teams": {"away": {"score": 1}, "home": {"score": 1}}},
        {"teams": {"away": {"score": 2}, "home": {"score": 0}}},
    ]

    selected = nba_playoffs._select_current_round_series(series)

    assert selected == series


def test_round_rank_from_text_supports_common_labels():
    assert nba_playoffs._round_rank_from_text("Western Conference First Round") == 1
    assert nba_playoffs._round_rank_from_text("Conference Semifinals") == 2
    assert nba_playoffs._round_rank_from_text("Eastern Conference Finals") == 3
    assert nba_playoffs._round_rank_from_text("NBA Finals") == 4


def test_select_current_round_series_keeps_completed_first_round_visible():
    series = [
        {"teams": {"away": {"team": {"abbreviation": "BOS"}, "score": 4}, "home": {"team": {"abbreviation": "ORL"}, "score": 1}}, "round_rank": 1},
        {"teams": {"away": {"team": {"abbreviation": "NYK"}, "score": 3}, "home": {"team": {"abbreviation": "DET"}, "score": 2}}, "round_rank": 1},
        {"teams": {"away": {"team": {"abbreviation": "CLE"}, "score": 0}, "home": {"team": {"abbreviation": "MIA"}, "score": 0}}, "round_rank": 2},
    ]

    selected = nba_playoffs._select_current_round_series(series)

    assert len(selected) == 2
    assert all(item["round_rank"] == 1 for item in selected)


def test_select_current_round_series_ignores_opponentless_next_round_series():
    series = [
        {"teams": {"away": {"team": {"abbreviation": "BOS"}, "score": 4}, "home": {"team": {"abbreviation": "ORL"}, "score": 1}}, "round_rank": 1},
        {"teams": {"away": {"team": {"abbreviation": "NYK"}, "score": 2}, "home": {"team": {"abbreviation": "DET"}, "score": 2}}, "round_rank": 1},
        {"teams": {"away": {"team": {"abbreviation": "BOS"}, "score": 0}, "home": {"team": {}, "score": 0}}, "round_rank": 2},
    ]

    selected = nba_playoffs._select_current_round_series(series)

    assert len(selected) == 2
    assert all(item["round_rank"] == 1 for item in selected)


def test_select_current_round_series_advances_only_when_next_round_started():
    series = [
        {"teams": {"away": {"team": {"abbreviation": "BOS"}, "score": 4}, "home": {"team": {"abbreviation": "ORL"}, "score": 1}}, "round_rank": 1},
        {"teams": {"away": {"team": {"abbreviation": "NYK"}, "score": 4}, "home": {"team": {"abbreviation": "DET"}, "score": 2}}, "round_rank": 1},
        {"teams": {"away": {"team": {"abbreviation": "BOS"}, "score": 0}, "home": {"team": {"abbreviation": "NYK"}, "score": 0}, "next_text": "TBD"}, "round_rank": 2},
    ]
    assert all(item["round_rank"] == 1 for item in nba_playoffs._select_current_round_series(series))

    series[2]["next_text"] = "Tonight 8:00 PM"
    selected = nba_playoffs._select_current_round_series(series)
    assert len(selected) == 1
    assert selected[0]["round_rank"] == 2
