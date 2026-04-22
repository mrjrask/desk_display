import datetime

from screens import nba_scoreboard


def test_map_espn_game_includes_season_metadata_for_playoff_detection():
    day = datetime.date(2026, 4, 22)
    event = {
        "id": "401999999",
        "uid": "s:40~l:46~e:401999999",
        "date": "2026-04-22T23:00Z",
        "season": {"type": 3, "slug": "post-season"},
    }
    competition = {
        "id": "401999999",
        "date": "2026-04-22T23:00Z",
        "status": {"type": {"state": "pre"}},
        "competitors": [
            {"homeAway": "away", "team": {"abbreviation": "BOS", "displayName": "Boston Celtics"}},
            {"homeAway": "home", "team": {"abbreviation": "NYK", "displayName": "New York Knicks"}},
        ],
    }

    mapped = nba_scoreboard._map_espn_game(event, competition, day)

    assert mapped is not None
    assert mapped["seasonType"] == 3
    assert mapped["seasonStage"] == "post-season"
