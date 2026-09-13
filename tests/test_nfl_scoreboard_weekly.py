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
    def __init__(self, payload: dict, *, error: bool = False):
        self._payload = payload
        self._error = error
        self.status_code = 200
        self.text = ""

    def raise_for_status(self) -> None:
        if self._error:
            raise RuntimeError("HTTP 503")
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


def test_fetch_games_for_week_requests_each_day_wednesday_to_tuesday(monkeypatch):
    events_by_date = {
        "20260902": [
            _event(event_id="wednesday", date="2026-09-03T00:15Z", away="DAL", home="NYG"),
        ],
        "20260908": [
            _event(event_id="tuesday", date="2026-09-08T23:15Z", away="CHI", home="GB"),
        ],
    }
    session = _install_fake_session(monkeypatch, events_by_date)

    games = nfl_scoreboard._fetch_games_for_week(
        datetime.datetime(2026, 9, 2, 12, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)
    )

    assert [game["id"] for game in games] == ["wednesday", "tuesday"]
    assert session.requested_dates == [
        "20260902",
        "20260903",
        "20260904",
        "20260905",
        "20260906",
        "20260907",
        "20260908",
    ]


def test_fetch_games_for_week_empty_this_week_does_not_fabricate_games(monkeypatch):
    # Dead week between preseason and the regular season: no events on any day.
    _install_fake_session(monkeypatch, {})

    now = datetime.datetime(2026, 8, 24, 12, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)

    games = nfl_scoreboard._fetch_games_for_week(now)

    assert games == []


def test_fetch_games_for_date_preserves_list_contract(monkeypatch):
    event = _event(
        event_id="single-date",
        date="2026-09-04T00:20Z",
        away="DAL",
        home="PHI",
    )
    _install_fake_session(monkeypatch, {"20260903": [event]})

    games = nfl_scoreboard._fetch_games_for_date(datetime.date(2026, 9, 3))

    assert isinstance(games, list)
    assert [game["id"] for game in games] == ["single-date"]


def test_incomplete_dates_use_nonempty_whole_week_fallback(monkeypatch):
    event = _event(event_id="fallback", date="2026-09-04T00:20Z", away="DAL", home="PHI")

    class Session:
        def get(self, url, timeout=None):
            if "dates=20260904" in url and "-" not in url.rsplit("dates=", 1)[-1]:
                return _FakeResponse({}, error=True)
            if "dates=20260903-20260909" in url:
                return _FakeResponse({"events": [event]})
            return _FakeResponse({"events": []})

    monkeypatch.setattr(nfl_scoreboard, "_SESSION", Session())
    monkeypatch.setattr(nfl_scoreboard, "_GAMES_CACHE", {})
    result = nfl_scoreboard._fetch_week_result_from_start(datetime.date(2026, 9, 3))

    assert [game["id"] for game in result.games] == ["fallback"]
    assert result.failed_dates == 1
    assert result.stale is False


def test_incomplete_dates_use_shared_provider_chain(monkeypatch):
    fallback = _event(
        event_id="nflverse-fallback",
        date="2026-09-04T00:20Z",
        away="DAL",
        home="PHI",
    )["competitions"][0]
    fallback["_event_date"] = "2026-09-04T00:20Z"

    class FailedSession:
        def get(self, url, timeout=None):
            return _FakeResponse({}, error=True)

    calls = []

    def fake_fetch_range_result(start, end, **kwargs):
        calls.append((start, end, kwargs))
        return nfl_service.WeeklyResult(games=[fallback])

    monkeypatch.setattr(nfl_scoreboard, "_SESSION", FailedSession())
    monkeypatch.setattr(nfl_scoreboard, "_GAMES_CACHE", {})
    monkeypatch.setattr(nfl_service, "fetch_range_result", fake_fetch_range_result)

    result = nfl_scoreboard._fetch_week_result_from_start(datetime.date(2026, 9, 3))

    assert [game["id"] for game in result.games] == ["nflverse-fallback"]
    assert result.failed_dates == 7
    assert result.stale is False
    assert calls[0][:2] == (datetime.date(2026, 9, 3), datetime.date(2026, 9, 9))
    assert calls[0][2]["failed_providers"] == {"ESPN Site"}


def test_stale_range_fallback_remains_stale_and_does_not_replace_week_cache(monkeypatch):
    fallback = _event(
        event_id="stale-range",
        date="2026-09-04T00:20Z",
        away="DAL",
        home="PHI",
    )["competitions"][0]
    fallback["_event_date"] = "2026-09-04T00:20Z"
    last_complete = nfl_service.WeeklyResult(games=[{"id": "last-complete"}])
    cache = {("nfl", "last_complete_week"): (10.0, last_complete)}

    class FailedSession:
        def get(self, url, timeout=None):
            return _FakeResponse({}, error=True)

    monkeypatch.setattr(nfl_scoreboard, "_SESSION", FailedSession())
    monkeypatch.setattr(nfl_scoreboard, "_GAMES_CACHE", cache)
    monkeypatch.setattr(
        nfl_service,
        "fetch_range_result",
        lambda *args, **kwargs: nfl_service.WeeklyResult(
            games=[fallback], stale=True
        ),
    )

    result = nfl_scoreboard._fetch_week_result_from_start(datetime.date(2026, 9, 3))

    assert [game["id"] for game in result.games] == ["stale-range"]
    assert result.stale is True
    assert cache[("nfl", "last_complete_week")][1] is last_complete


def test_incomplete_dates_ignore_empty_whole_week_fallback(monkeypatch):
    class Session:
        def get(self, url, timeout=None):
            date_value = url.rsplit("dates=", 1)[-1]
            if date_value == "20260904":
                return _FakeResponse({}, error=True)
            return _FakeResponse({"events": []})

    monkeypatch.setattr(nfl_scoreboard, "_SESSION", Session())
    monkeypatch.setattr(nfl_scoreboard, "_GAMES_CACHE", {})
    result = nfl_scoreboard._fetch_week_result_from_start(datetime.date(2026, 9, 3))

    assert result.games == []
    assert result.stale is True


def test_total_failure_retains_last_complete_week(monkeypatch):
    week_start = datetime.date(2026, 9, 3)
    monotonic_time = [100.0]
    cached_event = _event(
        event_id="cached", date="2026-09-04T00:20Z", away="DAL", home="PHI"
    )
    session = _FakeSession({"20260903": [cached_event]})
    monkeypatch.setattr(nfl_scoreboard, "_SESSION", session)
    monkeypatch.setattr(nfl_scoreboard, "_GAMES_CACHE", {})
    monkeypatch.setattr(nfl_scoreboard.time, "monotonic", lambda: monotonic_time[0])
    complete = nfl_scoreboard._fetch_week_result_from_start(week_start)
    assert [game["id"] for game in complete.games] == ["cached"]

    class FailedSession:
        def get(self, url, timeout=None):
            return _FakeResponse({}, error=True)

    monkeypatch.setattr(nfl_scoreboard, "_SESSION", FailedSession())
    monotonic_time[0] += nfl_scoreboard.FETCH_CACHE_TTL_SECONDS + 1
    stale = nfl_scoreboard._fetch_week_result_from_start(week_start)

    assert [game["id"] for game in stale.games] == ["cached"]
    assert stale.failed_dates == 7
    assert stale.stale is True


def test_total_failure_is_negatively_cached_for_refresh_ttl(monkeypatch):
    class FailedSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, timeout=None):
            self.calls += 1
            return _FakeResponse({}, error=True)

    session = FailedSession()
    monkeypatch.setattr(nfl_scoreboard, "_SESSION", session)
    monkeypatch.setattr(nfl_scoreboard, "_GAMES_CACHE", {})

    first = nfl_scoreboard._fetch_week_result_from_start(datetime.date(2026, 9, 3))
    call_count = session.calls
    second = nfl_scoreboard._fetch_week_result_from_start(datetime.date(2026, 9, 3))

    assert first.stale is True
    assert second is first
    assert session.calls == call_count


def test_complete_week_is_cached_between_rapid_render_loops(monkeypatch):
    event = _event(
        event_id="cached-week", date="2026-09-10T23:20Z", away="CHI", home="GB"
    )
    session = _install_fake_session(monkeypatch, {"20260910": [event]})
    week_start = datetime.date(2026, 9, 9)

    first = nfl_scoreboard._fetch_week_result_from_start(week_start)
    request_count = len(session.requested_dates)
    second = nfl_scoreboard._fetch_week_result_from_start(week_start)

    assert second is first
    assert request_count == 7
    assert len(session.requested_dates) == request_count


def test_force_refresh_bypasses_complete_week_cache(monkeypatch):
    scheduled = _event(
        event_id="live-game", date="2026-09-10T23:20Z", away="CHI", home="GB"
    )
    live = _event(
        event_id="live-game",
        date="2026-09-10T23:20Z",
        away="CHI",
        home="GB",
        state="in",
    )
    session = _install_fake_session(monkeypatch, {"20260910": [scheduled]})
    now = datetime.datetime(2026, 9, 10, 19, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)

    first = nfl_scoreboard._fetch_games_for_week(now)
    request_count = len(session.requested_dates)
    session._events_by_date["20260910"] = [live]
    refreshed = nfl_scoreboard._fetch_games_for_week(now, force_refresh=True)

    assert first[0]["status"]["type"]["state"] == "pre"
    assert refreshed[0]["status"]["type"]["state"] == "in"
    assert session.requested_dates[request_count:] == ["20260910"]


def test_live_refresh_merges_todays_scores_into_complete_week(monkeypatch):
    thursday = _event(
        event_id="thursday", date="2026-09-10T23:20Z", away="CHI", home="GB"
    )
    sunday = _event(
        event_id="sunday", date="2026-09-13T17:00Z", away="DET", home="MIN"
    )
    session = _install_fake_session(
        monkeypatch,
        {"20260910": [thursday], "20260913": [sunday]},
    )
    initial = nfl_scoreboard._fetch_games_for_week(
        datetime.datetime(2026, 9, 10, 18, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)
    )
    session._events_by_date["20260910"] = [
        _event(
            event_id="thursday",
            date="2026-09-10T23:20Z",
            away="CHI",
            home="GB",
            away_score="7",
            home_score="3",
            state="in",
        )
    ]

    refreshed = nfl_scoreboard._fetch_games_for_week(
        datetime.datetime(2026, 9, 10, 19, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME),
        force_refresh=True,
    )

    assert [game["id"] for game in initial] == ["thursday", "sunday"]
    assert [game["id"] for game in refreshed] == ["thursday", "sunday"]
    assert refreshed[0]["scores"] == {"away": "7", "home": "3"}


def test_force_refresh_honors_failed_week_retry_cooldown(monkeypatch):
    class FailedSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, timeout=None):
            self.calls += 1
            return _FakeResponse({}, error=True)

    session = FailedSession()
    monkeypatch.setattr(nfl_scoreboard, "_SESSION", session)
    monkeypatch.setattr(nfl_scoreboard, "_GAMES_CACHE", {})
    week_start = datetime.date(2026, 9, 9)

    first = nfl_scoreboard._fetch_week_result_from_start(
        week_start, force_refresh=True
    )
    call_count = session.calls
    second = nfl_scoreboard._fetch_week_result_from_start(
        week_start, force_refresh=True
    )

    assert first.failed_dates == 7
    assert second is first
    assert session.calls == call_count


def test_failed_focused_refresh_is_cached_for_retry_cooldown(monkeypatch):
    event = _event(
        event_id="live-game",
        date="2026-09-11T00:20Z",
        away="CHI",
        home="GB",
        state="in",
    )
    clock = [100.0]
    _install_fake_session(monkeypatch, {"20260910": [event]})
    monkeypatch.setattr(nfl_scoreboard.time, "monotonic", lambda: clock[0])
    week_start = datetime.date(2026, 9, 9)
    nfl_scoreboard._fetch_week_result_from_start(week_start)

    class FailedSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, timeout=None):
            self.calls += 1
            return _FakeResponse({}, error=True)

    failed_session = FailedSession()
    monkeypatch.setattr(nfl_scoreboard, "_SESSION", failed_session)
    clock[0] += nfl_scoreboard.FETCH_CACHE_TTL_SECONDS + 1
    now = datetime.datetime(2026, 9, 10, 20, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)

    first = nfl_scoreboard._fetch_games_for_week(now, force_refresh=True)
    calls_after_failure = failed_session.calls
    second = nfl_scoreboard._fetch_games_for_week(now, force_refresh=True)

    cached_result = nfl_scoreboard._GAMES_CACHE[
        ("nfl", "display_week", week_start, datetime.date(2026, 9, 15))
    ][1]
    assert first == second
    assert cached_result.failed_dates == 1
    assert cached_result.stale is True
    assert failed_session.calls == calls_after_failure == 1


def test_focused_refresh_uses_scheduled_date_after_midnight(monkeypatch):
    event = _event(
        event_id="long-game",
        date="2026-09-11T00:20Z",
        away="CHI",
        home="GB",
        state="in",
    )
    session = _install_fake_session(monkeypatch, {"20260910": [event]})
    week_start = datetime.date(2026, 9, 9)
    nfl_scoreboard._fetch_week_result_from_start(week_start)
    session.requested_dates.clear()
    after_midnight = datetime.datetime(
        2026, 9, 11, 1, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME
    )

    games = nfl_scoreboard._fetch_games_for_week(after_midnight, force_refresh=True)

    assert [game["id"] for game in games] == ["long-game"]
    assert session.requested_dates == ["20260910"]


def test_focused_refresh_requests_every_active_game_date(monkeypatch):
    events = {
        "20260910": [
            _event(
                event_id="suspended-thursday",
                date="2026-09-11T00:20Z",
                away="CHI",
                home="GB",
                state="in",
            )
        ],
        "20260913": [
            _event(
                event_id="live-sunday",
                date="2026-09-13T17:00Z",
                away="DET",
                home="MIN",
                state="in",
            )
        ],
    }
    session = _install_fake_session(monkeypatch, events)
    week_start = datetime.date(2026, 9, 9)
    nfl_scoreboard._fetch_week_result_from_start(week_start)
    session.requested_dates.clear()
    sunday_afternoon = datetime.datetime(
        2026, 9, 13, 14, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME
    )

    games = nfl_scoreboard._fetch_games_for_week(
        sunday_afternoon, force_refresh=True
    )

    assert {game["id"] for game in games} == {
        "suspended-thursday",
        "live-sunday",
    }
    assert session.requested_dates == ["20260910", "20260913"]


def test_focused_refresh_preserves_updates_when_another_date_fails(monkeypatch):
    thursday = _event(
        event_id="suspended-thursday",
        date="2026-09-11T00:20Z",
        away="CHI",
        home="GB",
        state="in",
    )
    sunday_live = _event(
        event_id="live-sunday",
        date="2026-09-13T17:00Z",
        away="DET",
        home="MIN",
        away_score="7",
        home_score="3",
        state="in",
    )
    sunday_updated = _event(
        event_id="live-sunday",
        date="2026-09-13T17:00Z",
        away="DET",
        home="MIN",
        away_score="14",
        home_score="3",
        state="in",
    )
    session = _install_fake_session(
        monkeypatch,
        {"20260910": [thursday], "20260913": [sunday_live]},
    )
    week_start = datetime.date(2026, 9, 9)
    nfl_scoreboard._fetch_week_result_from_start(week_start)

    class PartialFailureSession(_FakeSession):
        def get(self, url, timeout=None):
            date_key = url.rsplit("dates=", 1)[-1]
            self.requested_dates.append(date_key)
            if date_key == "20260910":
                return _FakeResponse({}, error=True)
            return super().get(url, timeout=timeout)

    partial_session = PartialFailureSession({"20260913": [sunday_updated]})
    monkeypatch.setattr(nfl_scoreboard, "_SESSION", partial_session)
    sunday_afternoon = datetime.datetime(
        2026, 9, 13, 14, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME
    )

    games = nfl_scoreboard._fetch_games_for_week(
        sunday_afternoon, force_refresh=True
    )

    by_id = {game["id"]: game for game in games}
    assert by_id["suspended-thursday"]["scores"] == {"away": None, "home": None}
    assert by_id["live-sunday"]["scores"] == {"away": "14", "home": "3"}
    cached = nfl_scoreboard._GAMES_CACHE[
        ("nfl", "display_week", week_start, datetime.date(2026, 9, 15))
    ][1]
    assert cached.failed_dates == 1
    assert cached.successful_dates == 1
    assert cached.stale is True


def test_successful_focused_refresh_replaces_all_games_on_requested_date(monkeypatch):
    event = _event(
        event_id="removed-game",
        date="2026-09-11T00:20Z",
        away="CHI",
        home="GB",
        state="in",
    )
    session = _install_fake_session(monkeypatch, {"20260910": [event]})
    week_start = datetime.date(2026, 9, 9)
    nfl_scoreboard._fetch_week_result_from_start(week_start)
    session._events_by_date.clear()
    now = datetime.datetime(2026, 9, 10, 20, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)

    games = nfl_scoreboard._fetch_games_for_week(now, force_refresh=True)

    assert games == []
    assert session.requested_dates[-1] == "20260910"


def test_successful_focused_refresh_updates_last_complete_week(monkeypatch):
    live = _event(
        event_id="finishing-game",
        date="2026-09-11T00:20Z",
        away="CHI",
        home="GB",
        state="in",
    )
    final = _event(
        event_id="finishing-game",
        date="2026-09-11T00:20Z",
        away="CHI",
        home="GB",
        away_score="20",
        home_score="24",
        state="post",
    )
    session = _install_fake_session(monkeypatch, {"20260910": [live]})
    week_start = datetime.date(2026, 9, 9)
    nfl_scoreboard._fetch_week_result_from_start(week_start)
    session._events_by_date["20260910"] = [final]
    now = datetime.datetime(2026, 9, 10, 20, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)

    nfl_scoreboard._fetch_games_for_week(now, force_refresh=True)

    last_complete = nfl_scoreboard._GAMES_CACHE[("nfl", "last_complete_week")][1]
    assert last_complete.games[0]["status"]["type"]["state"] == "post"
    assert last_complete.games[0]["scores"] == {"away": "20", "home": "24"}


def test_partial_failure_suppresses_all_provider_retries_for_five_minutes(monkeypatch):
    event = _event(
        event_id="fallback", date="2026-09-10T23:20Z", away="CHI", home="GB"
    )

    class Session:
        def __init__(self):
            self.calls = 0

        def get(self, url, timeout=None):
            self.calls += 1
            if "dates=20260910" in url and "-" not in url.rsplit("dates=", 1)[-1]:
                return _FakeResponse({}, error=True)
            return _FakeResponse({"events": [event]})

    session = Session()
    clock = iter([100.0, 100.0, 100.0, 100.0, 399.0, 401.0, 401.0, 401.0])
    monkeypatch.setattr(nfl_scoreboard, "_SESSION", session)
    monkeypatch.setattr(nfl_scoreboard, "_GAMES_CACHE", {})
    monkeypatch.setattr(nfl_scoreboard.time, "monotonic", lambda: next(clock))
    week_start = datetime.date(2026, 9, 9)

    first = nfl_scoreboard._fetch_week_result_from_start(week_start)
    calls_after_failure = session.calls
    cached = nfl_scoreboard._fetch_week_result_from_start(week_start)

    assert first.failed_dates == 1
    assert cached is first
    assert session.calls == calls_after_failure


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


def test_next_games_uses_provider_fallback_when_site_range_is_empty(monkeypatch):
    """Week 1 may appear on ESPN's CDN before its Site date-range feed."""

    opener = _event(
        event_id="week-one-opener",
        date="2026-09-11T00:20Z",
        away="DAL",
        home="PHI",
    )

    class Session:
        def get(self, url, timeout=None):
            if "cdn.espn.com/core/nfl/scoreboard" in url:
                return _FakeResponse({"events": [opener]})
            return _FakeResponse({"events": []})

    monkeypatch.setattr(nfl_scoreboard, "_SESSION", Session())
    monkeypatch.setattr(nfl_scoreboard, "_GAMES_CACHE", {})
    nfl_scoreboard._NO_UPCOMING_GAMES_COOLDOWN.reset()

    games = nfl_scoreboard._fetch_next_games(datetime.date(2026, 9, 7), max_days=6)

    assert [game["id"] for game in games] == ["week-one-opener"]


def test_bulk_range_reuses_successful_cached_result(monkeypatch):
    """A successful discovery range remains usable on the next refresh."""

    calls = 0
    discovered = [{"id": "cached-opener"}]

    def fetch_range(start, end, *, session, cache, failed_providers=None):
        nonlocal calls
        calls += 1
        return discovered

    monkeypatch.setattr("services.sports.nfl.fetch_range", fetch_range)
    monkeypatch.setattr(nfl_scoreboard, "_hydrate_games", lambda games: games)
    monkeypatch.setattr(nfl_scoreboard, "_GAMES_CACHE", {})
    start = datetime.date(2026, 9, 7)
    end = datetime.date(2026, 9, 13)

    first = nfl_scoreboard._fetch_games_for_bulk_range(start, end)
    second = nfl_scoreboard._fetch_games_for_bulk_range(start, end)

    assert first == discovered
    assert second == discovered
    assert calls == 1


def test_next_games_disables_failed_fallbacks_for_remainder_of_scan(monkeypatch):
    """A long scan must not retry an unavailable fallback in every window."""

    class Session:
        def __init__(self):
            self.urls = []

        def get(self, url, timeout=None):
            self.urls.append(url)
            if url.startswith("https://site.api.espn.com"):
                return _FakeResponse({"events": []})
            raise RuntimeError("provider unavailable")

    session = Session()
    monkeypatch.setattr(nfl_scoreboard, "_SESSION", session)
    monkeypatch.setattr(nfl_scoreboard, "_GAMES_CACHE", {})
    nfl_scoreboard._NO_UPCOMING_GAMES_COOLDOWN.reset()

    games = nfl_scoreboard._fetch_next_games(
        datetime.date(2026, 2, 9),
        max_days=13,
    )

    assert games == []
    site_urls = [url for url in session.urls if "site.api.espn.com" in url]
    cdn_urls = [url for url in session.urls if "cdn.espn.com" in url]
    nflverse_urls = [url for url in session.urls if "github.com/nflverse" in url]
    assert len(site_urls) == 2
    assert len(cdn_urls) == 1
    assert len(nflverse_urls) == 1


def test_next_games_year_long_fallback_uses_bounded_range_requests(monkeypatch):
    session = _install_fake_session(monkeypatch, {})
    nfl_scoreboard._NO_UPCOMING_GAMES_COOLDOWN.reset()

    games = nfl_scoreboard._fetch_next_games(datetime.date(2026, 2, 9))

    assert games == []
    # Site and CDN each receive one bounded request per window; nflverse's
    # complete schedule is downloaded once and then reused from the cache.
    range_requests = [value for value in session.requested_dates if "-" in value]
    assert len(range_requests) == 106
    assert session.requested_dates[0] == "20260209-20260215"
    assert range_requests[-1] == "20270208-20270214"


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


def test_wednesday_loads_the_entire_upcoming_nfl_week(monkeypatch):
    events_by_date = {
        "20260826": [
            _event(event_id="wednesday", date="2026-08-27T00:15Z", away="NE", home="SEA"),
        ],
        "20260827": [
            _event(event_id="thursday", date="2026-08-27T23:20Z", away="ATL", home="MIA"),
        ],
        "20260830": [
            _event(event_id="sunday", date="2026-08-30T18:00Z", away="CHI", home="GB"),
        ],
        "20260831": [
            _event(event_id="monday", date="2026-09-01T00:15Z", away="LAR", home="LV"),
        ],
    }
    session = _install_fake_session(monkeypatch, events_by_date)

    games = nfl_scoreboard._fetch_games_for_week(
        datetime.datetime(2026, 8, 26, 8, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME)
    )

    assert [game["id"] for game in games] == ["wednesday", "thursday", "sunday", "monday"]
    assert session.requested_dates == [
        "20260826", "20260827", "20260828", "20260829",
        "20260830", "20260831", "20260901",
    ]


def test_wednesday_and_thursday_games_share_the_same_display_week(monkeypatch):
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

    assert [game["id"] for game in games] == ["wednesday-game", "next-week"]
    assert session.requested_dates == [
        "20260826",
        "20260827",
        "20260828",
        "20260829",
        "20260830",
        "20260831",
        "20260901",
    ]


def test_force_refresh_after_playoff_cutoff_loads_next_week(monkeypatch):
    events_by_date = {
        "20260117": [
            _event(
                event_id="divisional-one",
                date="2026-01-18T01:00Z",
                away="HOU",
                home="DEN",
                state="post",
            ),
        ],
        "20260118": [
            _event(
                event_id="divisional-two",
                date="2026-01-19T01:00Z",
                away="LAR",
                home="SEA",
                state="post",
            ),
        ],
        "20260121": [
            _event(
                event_id="conference-championship",
                date="2026-01-22T01:00Z",
                away="BUF",
                home="KC",
                state="pre",
            ),
        ],
    }
    session = _install_fake_session(monkeypatch, events_by_date)
    after_cutoff = datetime.datetime(
        2026, 1, 19, 16, 0, tzinfo=nfl_scoreboard.CENTRAL_TIME
    )

    games = nfl_scoreboard._fetch_games_for_week(after_cutoff, force_refresh=True)

    assert [game["id"] for game in games] == ["conference-championship"]
    assert "20260121" in session.requested_dates


def test_playoff_week_also_starts_on_wednesday(monkeypatch):
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

    assert [game["id"] for game in games] == ["rescheduled-playoff", "following-week"]
    assert session.requested_dates == [
        "20260114",
        "20260115",
        "20260116",
        "20260117",
        "20260118",
        "20260119",
        "20260120",
    ]
