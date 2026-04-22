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
