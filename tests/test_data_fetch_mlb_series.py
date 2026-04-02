import datetime

import data_fetch


class _FrozenDateTime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        base = cls(2026, 4, 1, 12, 0, 0)
        return base.replace(tzinfo=tz) if tz is not None else base


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _game(game_pk, date, state, home_id, away_id):
    return {
        "gamePk": game_pk,
        "gameDate": f"{date}T19:05:00Z",
        "officialDate": date,
        "status": {
            "statusCode": state,
            "abstractGameState": (
                "Live" if state == "I" else "Preview" if state == "S" else "Final"
            ),
            "detailedState": "In Progress" if state == "I" else "Scheduled",
        },
        "teams": {
            "home": {"team": {"id": home_id}},
            "away": {"team": {"id": away_id}},
        },
    }


def test_next_series_skips_current_live_series(monkeypatch):
    monkeypatch.setattr(data_fetch.datetime, "datetime", _FrozenDateTime)

    payload = {
        "dates": [
            {
                "date": "2026-04-01",
                "games": [
                    _game(1, "2026-04-01", "I", 112, 121),
                ],
            },
            {
                "date": "2026-04-02",
                "games": [
                    _game(2, "2026-04-02", "S", 112, 121),
                ],
            },
            {
                "date": "2026-04-05",
                "games": [
                    _game(3, "2026-04-05", "S", 140, 112),
                ],
            },
            {
                "date": "2026-04-08",
                "games": [
                    _game(4, "2026-04-08", "S", 112, 118),
                ],
            },
        ]
    }

    monkeypatch.setattr(data_fetch._session, "get", lambda *args, **kwargs: _DummyResponse(payload))

    result = data_fetch._fetch_mlb_schedule(112)

    assert [g["gamePk"] for g in (result["current_series_games"] or [])] == [1, 2]
    assert [g["gamePk"] for g in (result["next_series_games"] or [])] == [3]
    assert [g["gamePk"] for g in (result["next_home_series_games"] or [])] == [4]


def test_next_series_advances_when_current_series_uses_next_block(monkeypatch):
    monkeypatch.setattr(data_fetch.datetime, "datetime", _FrozenDateTime)

    payload = {
        "dates": [
            {
                "date": "2026-04-03",
                "games": [_game(10, "2026-04-03", "S", 112, 121)],
            },
            {
                "date": "2026-04-04",
                "games": [_game(11, "2026-04-04", "S", 112, 121)],
            },
            {
                "date": "2026-04-07",
                "games": [_game(12, "2026-04-07", "S", 112, 118)],
            },
        ]
    }

    monkeypatch.setattr(data_fetch._session, "get", lambda *args, **kwargs: _DummyResponse(payload))

    result = data_fetch._fetch_mlb_schedule(112)

    assert [g["gamePk"] for g in (result["current_series_games"] or [])] == [10, 11]
    assert [g["gamePk"] for g in (result["next_series_games"] or [])] == [12]
    assert result["next_home_series_games"] is None


def test_next_home_series_advances_when_matching_next_series(monkeypatch):
    monkeypatch.setattr(data_fetch.datetime, "datetime", _FrozenDateTime)

    payload = {
        "dates": [
            {
                "date": "2026-04-01",
                "games": [_game(20, "2026-04-01", "I", 112, 121)],
            },
            {
                "date": "2026-04-02",
                "games": [_game(21, "2026-04-02", "S", 112, 121)],
            },
            {
                "date": "2026-04-05",
                "games": [_game(22, "2026-04-05", "S", 112, 118)],
            },
            {
                "date": "2026-04-07",
                "games": [_game(23, "2026-04-07", "S", 140, 112)],
            },
            {
                "date": "2026-04-09",
                "games": [_game(24, "2026-04-09", "S", 112, 147)],
            },
        ]
    }

    monkeypatch.setattr(data_fetch._session, "get", lambda *args, **kwargs: _DummyResponse(payload))

    result = data_fetch._fetch_mlb_schedule(112)

    assert [g["gamePk"] for g in (result["current_series_games"] or [])] == [20, 21]
    assert [g["gamePk"] for g in (result["next_series_games"] or [])] == [22]
    assert [g["gamePk"] for g in (result["next_home_series_games"] or [])] == [24]


def test_next_home_series_keeps_all_games_when_opponent_stays_same_but_venue_changes(monkeypatch):
    monkeypatch.setattr(data_fetch.datetime, "datetime", _FrozenDateTime)

    payload = {
        "dates": [
            {
                "date": "2026-04-01",
                "games": [_game(30, "2026-04-01", "I", 112, 121)],
            },
            {
                "date": "2026-04-02",
                "games": [_game(31, "2026-04-02", "S", 112, 121)],
            },
            # Next series is away vs STL
            {
                "date": "2026-04-05",
                "games": [_game(32, "2026-04-05", "S", 138, 112)],
            },
            {
                "date": "2026-04-06",
                "games": [_game(33, "2026-04-06", "S", 138, 112)],
            },
            {
                "date": "2026-04-07",
                "games": [_game(34, "2026-04-07", "S", 138, 112)],
            },
            # Following home series is also vs STL and should remain a distinct block
            {
                "date": "2026-04-09",
                "games": [_game(35, "2026-04-09", "S", 112, 138)],
            },
            {
                "date": "2026-04-10",
                "games": [_game(36, "2026-04-10", "S", 112, 138)],
            },
            {
                "date": "2026-04-11",
                "games": [_game(37, "2026-04-11", "S", 112, 138)],
            },
        ]
    }

    monkeypatch.setattr(data_fetch._session, "get", lambda *args, **kwargs: _DummyResponse(payload))

    result = data_fetch._fetch_mlb_schedule(112)

    assert [g["gamePk"] for g in (result["next_series_games"] or [])] == [32, 33, 34]
    assert [g["gamePk"] for g in (result["next_home_series_games"] or [])] == [35, 36, 37]
