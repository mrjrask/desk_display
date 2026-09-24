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


def _game(game_pk, date, time_utc, state, home_id, away_id):
    return {
        "gamePk": game_pk,
        "gameDate": f"{date}T{time_utc}Z",
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


def test_next_game_picks_earliest_of_a_doubleheader(monkeypatch):
    # A traditional doubleheader against the same opponent: the API lists
    # Game 2 (later start) after Game 1 in iteration order. next_game must
    # be the earlier game, not whichever one was iterated last.
    monkeypatch.setattr(data_fetch.datetime, "datetime", _FrozenDateTime)

    payload = {
        "dates": [
            {
                "date": "2026-04-01",
                "games": [
                    _game(1, "2026-04-01", "18:05:00", "S", 112, 121),
                    _game(2, "2026-04-01", "21:40:00", "S", 112, 121),
                ],
            },
        ]
    }
    monkeypatch.setattr(data_fetch._session, "get", lambda *args, **kwargs: _DummyResponse(payload))

    result = data_fetch._fetch_mlb_schedule(112)

    assert result["next_game"]["gamePk"] == 1


def test_next_game_alt_picks_split_squad_second_game(monkeypatch):
    # Split-squad games against different opponents: next_game is the
    # earlier one and next_game_alt exposes the other.
    monkeypatch.setattr(data_fetch.datetime, "datetime", _FrozenDateTime)

    payload = {
        "dates": [
            {
                "date": "2026-04-01",
                "games": [
                    _game(2, "2026-04-01", "21:40:00", "S", 112, 118),
                    _game(1, "2026-04-01", "18:05:00", "S", 112, 121),
                ],
            },
        ]
    }
    monkeypatch.setattr(data_fetch._session, "get", lambda *args, **kwargs: _DummyResponse(payload))

    result = data_fetch._fetch_mlb_schedule(112)

    assert result["next_game"]["gamePk"] == 1
    assert result["next_game_alt"]["gamePk"] == 2
