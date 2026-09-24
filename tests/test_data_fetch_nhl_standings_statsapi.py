import data_fetch


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _team(abbr, wins, division_rank):
    return {
        "team": {"abbreviation": abbr},
        "leagueRecord": {"wins": wins, "losses": 10, "ot": 2, "pct": ".500"},
        "streak": {"streakCode": "W1"},
        "records": {"splitRecords": []},
        "points": wins * 2,
        "divisionRank": division_rank,
        "divisionGamesBack": "0",
        "wildCardRank": "1",
        "conferenceRank": division_rank,
        "conference": {"name": "Western"},
        "division": {"name": "Central"},
    }


def test_returns_matching_team_not_last_team_in_division(monkeypatch):
    # Regression: a `return` mistakenly indented one level too shallow used
    # to fire unconditionally after the first division's team loop finished,
    # returning whichever team happened to be last in that division instead
    # of the one actually matching team_abbr.
    payload = {
        "records": [
            {
                "teamRecords": [
                    _team("STL", 10, 3),
                    _team("CHI", 40, 1),
                ]
            }
        ]
    }
    monkeypatch.setattr(
        data_fetch._session, "get", lambda *args, **kwargs: _DummyResponse(payload)
    )

    result = data_fetch._fetch_nhl_team_standings_statsapi("CHI")

    assert result is not None
    assert result["leagueRecord"]["wins"] == 40
    assert result["divisionRank"] == 1


def test_returns_match_from_a_later_division_record(monkeypatch):
    # Regression: the same indentation bug also caused the function to
    # return after only the first "records" entry (division), even when the
    # requested team was in a later one.
    payload = {
        "records": [
            {"teamRecords": [_team("STL", 10, 3)]},
            {"teamRecords": [_team("CHI", 40, 1)]},
        ]
    }
    monkeypatch.setattr(
        data_fetch._session, "get", lambda *args, **kwargs: _DummyResponse(payload)
    )

    result = data_fetch._fetch_nhl_team_standings_statsapi("CHI")

    assert result is not None
    assert result["leagueRecord"]["wins"] == 40


def test_returns_none_when_team_not_found(monkeypatch):
    payload = {"records": [{"teamRecords": [_team("STL", 10, 3)]}]}
    monkeypatch.setattr(
        data_fetch._session, "get", lambda *args, **kwargs: _DummyResponse(payload)
    )

    result = data_fetch._fetch_nhl_team_standings_statsapi("CHI")

    assert result is None
