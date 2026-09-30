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


def _with_status(game, code, abstract, detailed):
    game["status"] = {
        "statusCode": code,
        "abstractGameState": abstract,
        "detailedState": detailed,
    }
    return game


def test_next_game_keeps_todays_game_until_first_pitch(monkeypatch):
    # Today's game stays the next game until the first pitch, even with
    # tomorrow's game in the schedule. Warmup and a delayed start report
    # abstractGameState "Live" before the first pitch.
    monkeypatch.setattr(data_fetch.datetime, "datetime", _FrozenDateTime)

    for code, abstract, detailed in (
        ("S", "Preview", "Scheduled"),
        ("P", "Preview", "Pre-Game"),
        ("PW", "Live", "Warmup"),
        ("PR", "Live", "Delayed Start: Rain"),
    ):
        payload = {
            "dates": [
                {
                    "date": "2026-04-01",
                    "games": [
                        _with_status(
                            _game(1, "2026-04-01", "18:05:00", "S", 112, 121),
                            code,
                            abstract,
                            detailed,
                        ),
                    ],
                },
                {
                    "date": "2026-04-02",
                    "games": [_game(2, "2026-04-02", "18:05:00", "S", 112, 121)],
                },
            ]
        }
        monkeypatch.setattr(
            data_fetch._session, "get", lambda *args, p=payload, **kwargs: _DummyResponse(p)
        )

        result = data_fetch._fetch_mlb_schedule(112)

        assert result["next_game"]["gamePk"] == 1, detailed


def test_next_game_rolls_forward_once_todays_game_starts(monkeypatch):
    monkeypatch.setattr(data_fetch.datetime, "datetime", _FrozenDateTime)

    payload = {
        "dates": [
            {
                "date": "2026-04-01",
                "games": [_game(1, "2026-04-01", "17:05:00", "I", 112, 121)],
            },
            {
                "date": "2026-04-02",
                "games": [_game(2, "2026-04-02", "18:05:00", "S", 112, 121)],
            },
        ]
    }
    monkeypatch.setattr(data_fetch._session, "get", lambda *args, **kwargs: _DummyResponse(payload))

    result = data_fetch._fetch_mlb_schedule(112)

    assert result["live_game"]["gamePk"] == 1
    assert result["next_game"]["gamePk"] == 2
