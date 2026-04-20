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
    assert normalized["teams"]["away"]["team"]["team"]["teamTricode"] == "BOS"
    assert normalized["teams"]["home"]["team"]["team"]["teamTricode"] == "MIA"
    assert normalized["teams"]["away"]["score"] == 3
    assert normalized["teams"]["home"]["score"] == 1
    assert normalized["next_text"].startswith("Next: ")


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


def test_normalize_next_text_strips_timezone_suffix():
    assert nba_playoffs._normalize_next_text("Next: 04/09 8:00 PM ET") == "Next: 4/9 8:00 PM"


def test_format_next_text_uses_tonight_label(monkeypatch):
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 20, 10, 0, tzinfo=tz)

    monkeypatch.setattr(nba_playoffs.datetime, "datetime", _FixedNow)
    text = nba_playoffs._format_next_text({"nextGameStartTimeUTC": "2026-04-21T02:30:00Z"})
    assert text == "Next: Tonight 9:30 PM"


def test_derive_playoff_matchups_supports_scoreboard_games_shape():
    games = [
        {
            "teams": {
                "away": {"team": {"abbreviation": "BOS"}},
                "home": {"team": {"abbreviation": "NYK"}},
            }
        }
    ]

    derived = nba_playoffs._derive_playoff_matchups_from_games(games)

    assert len(derived) == 1
    assert derived[0]["teams"]["away"]["team"]["abbreviation"] == "BOS"
    assert derived[0]["teams"]["home"]["team"]["abbreviation"] == "NYK"
