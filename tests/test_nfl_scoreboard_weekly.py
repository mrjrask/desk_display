"""Regression coverage for the NFL weekly scoreboard fetch pipeline.

Exercises ``_fetch_games_for_week`` / ``_fetch_games_for_date`` against a fake
HTTP session shaped like ESPN's real scoreboard response, instead of mocking
those functions away -- the previous "No games" regressions (permanent
no-upcoming-games latch, and the shared-host circuit breaker getting tripped
by an unrelated sport) both lived inside this parsing/aggregation path, and
none of the existing tests exercised it end to end.
"""

from __future__ import annotations

import datetime

import screens.nfl_scoreboard as nfl_scoreboard
from services.sports import nfl as nfl_service


def _event(
    *,
    event_id: str,
    date: str,
    away: str,
    home: str,
    away_score: str | None = None,
    home_score: str | None = None,
    state: str = "pre",
) -> dict:
    completed = state == "post"
    return {
        "id": event_id,
        "date": date,
        "name": f"{away} at {home}",
        "shortName": f"{away} @ {home}",
        "competitions": [
            {
                "id": event_id,
                "competitors": [
                    {
                        "homeAway": "away",
                        "team": {"abbreviation": away},
                        "score": away_score,
                    },
                    {
                        "homeAway": "home",
                        "team": {"abbreviation": home},
                        "score": home_score,
                    },
                ],
                "status": {
                    "type": {
                        "state": state,
                        "completed": completed,
                        "description": "Final" if completed else "Scheduled",
                        "shortDetail": "Final" if completed else "8:15 PM",
                    }
                },
            }
        ],
    }


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    """Serves canned ESPN events for single-day scoreboard queries."""

    def __init__(self, events_by_date: dict[str, list[dict]]):
        self._events_by_date = events_by_date
        self.requested_dates: list[str] = []

    def get(self, url: str, timeout: float | None = None):
        date_key = url.rsplit("dates=", 1)[-1]
        self.requested_dates.append(date_key)
        start, _, end = date_key.partition("-")
        end = end or start
        events = [
            event
            for day, day_events in self._events_by_date.items()
            if start <= day <= end
            for event in day_events
        ]
        return _FakeResponse({"events": events})


def _install_fake_session(monkeypatch, events_by_date: dict[str, list[dict]]) -> _FakeSession:
    fake_session = _FakeSession(events_by_date)
    monkeypatch.setattr(nfl_scoreboard, "_SESSION", fake_session)
    monkeypatch.setattr(nfl_scoreboard, "_GAMES_CACHE", {})
    return fake_session


def test_fetch_games_for_week_returns_thursday_through_monday_games(monkeypatch):
    # Preseason week: Thu 8/20 - Mon 8/24, 2026.
    events_by_date = {
        "20260820": [
            _event(event_id="1", date="2026-08-20T23:20Z", away="DET", home="CHI", state="post",
                   away_score="17", home_score="24"),
        ],
        "20260823": [
            _event(event_id="2", date="2026-08-24T00:25Z", away="GB", home="MIN", state="pre"),
        ],
        "20260824": [
            _event(event_id="3", date="2026-08-25T00:15Z", away="SF", home="SEA", state="in"),
        ],
    }
    _install_fake_session(monkeypatch, events_by_date)

    # A Monday well inside the Thu 8/20 - Mon 8/24 window.
    now = datetime.datetime(2026, 8, 24, 12, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)

    games = nfl_scoreboard._fetch_games_for_week(now)

    assert [game["id"] for game in games] == ["1", "2", "3"]


def test_fetch_games_for_week_requests_each_day_thursday_to_wednesday(monkeypatch):
    events_by_date = {
        "20260902": [
            _event(event_id="wednesday", date="2026-09-03T00:15Z", away="DAL", home="NYG"),
        ],
    }
    session = _install_fake_session(monkeypatch, events_by_date)

    games = nfl_scoreboard._fetch_games_for_week(
        datetime.datetime(2026, 9, 1, 12, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)
    )

    assert [game["id"] for game in games] == ["wednesday"]
    assert session.requested_dates == [
        "20260827",
        "20260828",
        "20260829",
        "20260830",
        "20260831",
        "20260901",
        "20260902",
    ]


def test_fetch_games_for_week_empty_this_week_does_not_fabricate_games(monkeypatch):
    # Dead week between preseason and the regular season: no events on any day.
    _install_fake_session(monkeypatch, {})

    now = datetime.datetime(2026, 8, 24, 12, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)

    games = nfl_scoreboard._fetch_games_for_week(now)

    assert games == []


def test_fetch_scoreboard_falls_back_to_next_games_when_week_is_empty(monkeypatch):
    # No games this week, but the regular season opener is a few days out.
    events_by_date = {
        "20260904": [
            _event(event_id="opener", date="2026-09-04T23:20Z", away="CHI", home="MIN", state="pre"),
        ],
    }
    _install_fake_session(monkeypatch, events_by_date)
    nfl_scoreboard._NO_UPCOMING_GAMES_COOLDOWN.reset()

    now = datetime.datetime(2026, 8, 24, 12, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)

    games = nfl_service.fetch_scoreboard(now=now)

    assert [game["id"] for game in games] == ["opener"]


def test_next_games_fallback_returns_the_full_week(monkeypatch):
    events_by_date = {
        "20260910": [_event(event_id="thursday", date="2026-09-10T23:20Z", away="CHI", home="GB")],
        "20260913": [_event(event_id="sunday", date="2026-09-13T17:00Z", away="MIN", home="DET")],
    }
    _install_fake_session(monkeypatch, events_by_date)
    nfl_scoreboard._NO_UPCOMING_GAMES_COOLDOWN.reset()

    games = nfl_scoreboard._fetch_next_games(datetime.date(2026, 9, 1), max_days=20)

    assert [game["id"] for game in games] == ["thursday", "sunday"]


def test_next_games_keeps_discovered_games_when_full_week_refetch_fails(monkeypatch):
    events_by_date = {
        "20260910": [
            _event(
                event_id="discovered",
                date="2026-09-10T23:20Z",
                away="CHI",
                home="GB",
            ),
        ],
    }
    _install_fake_session(monkeypatch, events_by_date)
    nfl_scoreboard._NO_UPCOMING_GAMES_COOLDOWN.reset()
    monkeypatch.setattr(nfl_scoreboard, "_fetch_week_from_start", lambda _start: [])

    games = nfl_scoreboard._fetch_next_games(datetime.date(2026, 9, 8), max_days=6)

    assert [game["id"] for game in games] == ["discovered"]


def test_wednesday_morning_cutover_advances_to_the_upcoming_week(monkeypatch):
    events_by_date = {
        # This week's (Thu 8/20 - Mon 8/24) Monday night game -- should still show
        # up to Wednesday morning, before the cutover advances the window.
        "20260824": [
            _event(event_id="this-week", date="2026-08-25T00:15Z", away="LAR", home="LV", state="post",
                   away_score="20", home_score="13"),
        ],
        # Next week's (Thu 8/27 - Mon 8/31) Thursday game -- should appear only
        # after the Wednesday 9am cutover.
        "20260827": [
            _event(event_id="next-week", date="2026-08-27T23:20Z", away="ATL", home="MIA", state="pre"),
        ],
    }
    _install_fake_session(monkeypatch, events_by_date)

    before_cutover = datetime.datetime(2026, 8, 26, 8, 59, tzinfo=nfl_scoreboard.CENTRAL_TIME)
    after_cutover = datetime.datetime(2026, 8, 26, 9, 1, tzinfo=nfl_scoreboard.CENTRAL_TIME)

    games_before = nfl_scoreboard._fetch_games_for_week(before_cutover)
    assert [game["id"] for game in games_before] == ["this-week"]

    games_after = nfl_scoreboard._fetch_games_for_week(after_cutover)
    assert [game["id"] for game in games_after] == ["next-week"]


def test_wednesday_game_prevents_morning_cutover(monkeypatch):
    events_by_date = {
        "20260826": [
            _event(
                event_id="wednesday-game",
                date="2026-08-27T00:15Z",
                away="DAL",
                home="NYG",
                state="pre",
            ),
        ],
        "20260827": [
            _event(
                event_id="next-week",
                date="2026-08-27T23:20Z",
                away="ATL",
                home="MIA",
                state="pre",
            ),
        ],
    }
    session = _install_fake_session(monkeypatch, events_by_date)
    after_cutover = datetime.datetime(2026, 8, 26, 12, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)

    games = nfl_scoreboard._fetch_games_for_week(after_cutover)

    assert [game["id"] for game in games] == ["wednesday-game"]
    assert session.requested_dates == [
        "20260820",
        "20260821",
        "20260822",
        "20260823",
        "20260824",
        "20260825",
        "20260826",
    ]


def test_wednesday_game_prevents_playoff_month_cutover(monkeypatch):
    events_by_date = {
        "20260114": [
            _event(
                event_id="rescheduled-playoff",
                date="2026-01-15T01:00Z",
                away="BUF",
                home="KC",
                state="pre",
            ),
        ],
        "20260115": [
            _event(
                event_id="following-week",
                date="2026-01-16T01:00Z",
                away="SF",
                home="SEA",
                state="pre",
            ),
        ],
    }
    session = _install_fake_session(monkeypatch, events_by_date)
    wednesday_evening = datetime.datetime(
        2026,
        1,
        14,
        18,
        0,
        tzinfo=nfl_scoreboard.CENTRAL_TIME,
    )

    games = nfl_scoreboard._fetch_games_for_week(wednesday_evening)

    assert [game["id"] for game in games] == ["rescheduled-playoff"]
    assert session.requested_dates == [
        "20260108",
        "20260109",
        "20260110",
        "20260111",
        "20260112",
        "20260113",
        "20260114",
    ]
