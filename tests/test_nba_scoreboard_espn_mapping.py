import datetime

from services.sports import nba


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

    mapped = nba._map_espn_game(event, competition, day)

    assert mapped is not None
    assert mapped["seasonType"] == 3
    assert mapped["seasonStage"] == "post-season"


def test_map_game_preserves_espn_season_metadata_for_downstream_detectors():
    mapped = nba._map_game(
        {
            "id": "401999999",
            "gameTimeUTC": "2026-04-22T23:00Z",
            "statusNum": "1",
            "statusText": "Scheduled",
            "awayTeam": {"teamAbbr": "BOS"},
            "homeTeam": {"teamAbbr": "NYK"},
            "seasonType": 3,
            "seasonStage": "post-season",
        }
    )

    assert mapped["seasonType"] == 3
    assert mapped["seasonStage"] == "post-season"


def test_fetch_games_for_date_caches_repeated_calls_for_same_day(monkeypatch):
    day = datetime.date(2026, 8, 24)
    nba._games_for_date_cache.clear()
    calls = []

    def fake_fetch_from_espn(requested_day):
        calls.append(requested_day)
        return [{"id": "game-1"}]

    monkeypatch.setattr(nba, "_fetch_games_from_espn", fake_fetch_from_espn)

    first = nba._fetch_games_for_date(day)
    second = nba._fetch_games_for_date(day)

    assert first == [{"id": "game-1"}]
    assert second == [{"id": "game-1"}]
    assert calls == [day], "second call within the TTL should reuse the cached result"


def test_fetch_games_for_date_refetches_after_cache_expires(monkeypatch):
    day = datetime.date(2026, 8, 24)
    nba._games_for_date_cache.clear()
    calls = []

    def fake_fetch_from_espn(requested_day):
        calls.append(requested_day)
        return [{"id": f"game-{len(calls)}"}]

    monkeypatch.setattr(nba, "_fetch_games_from_espn", fake_fetch_from_espn)

    nba._fetch_games_for_date(day)
    # Simulate the cache entry having aged past its TTL.
    cached_at, cached_games = nba._games_for_date_cache[day]
    nba._games_for_date_cache[day] = (
        cached_at - nba._GAMES_FOR_DATE_CACHE_TTL_SECONDS - 1,
        cached_games,
    )

    nba._fetch_games_for_date(day)

    assert calls == [day, day]


def test_map_espn_competitor_flattens_schedule_score_objects():
    # The team schedule endpoint (Bulls Last) sends score as an object.
    obj = {"team": {"abbreviation": "CHI"}, "score": {"value": 98.0, "displayValue": "98"}}
    assert nba._map_espn_competitor(obj)["score"] == "98"
    assert nba._map_espn_competitor({"team": {}, "score": {"value": 101.0}})["score"] == 101
    assert nba._map_espn_competitor({"team": {}, "score": "110"})["score"] == "110"
    assert nba._map_espn_competitor({"team": {}})["score"] is None


def test_final_schedule_game_keeps_scores_through_map_game():
    event = {"id": "1", "date": "2026-10-07T01:00Z", "season": {"type": 1, "slug": "preseason"}}
    competition = {
        "id": "1",
        "date": "2026-10-07T01:00Z",
        "status": {"type": {"state": "post", "completed": True, "name": "STATUS_FINAL"}},
        "competitors": [
            {"homeAway": "home", "team": {"abbreviation": "CHI"}, "score": {"value": 98.0, "displayValue": "98"}},
            {"homeAway": "away", "team": {"abbreviation": "PHX"}, "score": {"value": 105.0, "displayValue": "105"}},
        ],
    }
    mapped = nba._map_game(nba._map_espn_game(event, competition, None))
    assert str(mapped["teams"]["home"]["score"]) == "98"
    assert str(mapped["teams"]["away"]["score"]) == "105"
