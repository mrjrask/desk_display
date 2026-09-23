import data_fetch


def _empty_classification():
    return {
        "last_game": None,
        "live_game": None,
        "next_game": None,
        "next_home_game": None,
    }


def _season_payload(rows):
    return {"SiteKit": {"Seasons": {"rows": rows}}}


def test_sanitize_ahl_payload_strips_guard_prefixes():
    raw = "while(1);\n\ufeff{\"foo\": \"bar\"}"
    assert data_fetch._sanitize_ahl_payload(raw) == '{"foo": "bar"}'


def test_sanitize_ahl_payload_handles_comments_and_whitespace():
    raw = "/* comment */  \n  {\n  \"ok\": true\n}"
    assert data_fetch._sanitize_ahl_payload(raw) == '{\n  "ok": true\n}'


def test_current_ahl_season_explicit_override_skips_discovery(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_SEASON_ID", "override")
    monkeypatch.setattr(
        data_fetch,
        "_ahl_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected request")),
    )

    assert data_fetch._current_ahl_season_id() == "override"


def test_current_ahl_season_prefers_current_flag(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_SEASON_ID", "")
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE", None)
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE_YEAR", None)
    monkeypatch.setattr(data_fetch, "_expected_ahl_season_year", lambda: 2025)
    monkeypatch.setattr(
        data_fetch,
        "_ahl_request",
        lambda *args, **kwargs: _season_payload(
            [
                {"id": "newer", "name": "2026-27"},
                {"id": "current", "name": "2025-26", "isCurrent": True},
            ]
        ),
    )

    assert data_fetch._current_ahl_season_id() == "current"


def test_current_ahl_season_sorts_unordered_rows_by_dates_or_identifier(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_SEASON_ID", "")
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE", None)
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE_YEAR", None)
    monkeypatch.setattr(data_fetch, "_expected_ahl_season_year", lambda: 2025)
    monkeypatch.setattr(
        data_fetch,
        "_ahl_request",
        lambda *args, **kwargs: _season_payload(
            [
                {"id": "old", "end_date": "2024-06-30"},
                {"id": "new", "startDate": "2025-09-01"},
                {"id": "middle", "name": "2024-25"},
            ]
        ),
    )

    assert data_fetch._current_ahl_season_id() == "new"


def test_current_ahl_season_excludes_future_unflagged_season(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_SEASON_ID", "")
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE", None)
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE_YEAR", None)
    monkeypatch.setattr(data_fetch, "_expected_ahl_season_year", lambda: 2025)
    monkeypatch.setattr(
        data_fetch,
        "_ahl_request",
        lambda *args, **kwargs: _season_payload(
            [
                {"id": "active", "name": "2025-26"},
                {"id": "future", "name": "2026-27"},
                {"id": "old", "name": "2024-25"},
            ]
        ),
    )

    assert data_fetch._current_ahl_season_id() == "active"
    assert data_fetch._AHL_SEASON_CACHE == "active"
    assert data_fetch._AHL_SEASON_CACHE_YEAR == 2025


def test_current_ahl_season_parses_four_digit_year_ranges(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_SEASON_ID", "")
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE", None)
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE_YEAR", None)
    monkeypatch.setattr(data_fetch, "_expected_ahl_season_year", lambda: 2025)
    monkeypatch.setattr(
        data_fetch,
        "_ahl_request",
        lambda *args, **kwargs: _season_payload(
            [
                {"id": "active", "name": "2025-2026"},
                {"id": "future", "name": "2026-2027"},
            ]
        ),
    )

    assert data_fetch._current_ahl_season_id() == "active"


def test_current_ahl_season_does_not_combine_years_from_separate_fields(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_SEASON_ID", "")
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE", None)
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE_YEAR", None)
    monkeypatch.setattr(data_fetch, "_expected_ahl_season_year", lambda: 2025)
    monkeypatch.setattr(
        data_fetch,
        "_ahl_request",
        lambda *args, **kwargs: _season_payload(
            [
                {"id": "active", "name": "2025-2026"},
                {
                    "id": "future",
                    "copyright": "Copyright 2025",
                    "name": "2026-2027",
                },
            ]
        ),
    )

    assert data_fetch._current_ahl_season_id() == "active"


def test_current_ahl_season_cache_expires_at_hockey_year_rollover(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_SEASON_ID", "")
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE", None)
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE_YEAR", None)
    expected_year = 2025
    requests = 0

    def fake_request(*args, **kwargs):
        nonlocal requests
        requests += 1
        return _season_payload([{"id": f"season-{expected_year}", "current": "yes"}])

    monkeypatch.setattr(data_fetch, "_expected_ahl_season_year", lambda: expected_year)
    monkeypatch.setattr(data_fetch, "_ahl_request", fake_request)

    assert data_fetch._current_ahl_season_id() == "season-2025"
    assert data_fetch._current_ahl_season_id() == "season-2025"
    expected_year = 2026
    assert data_fetch._current_ahl_season_id() == "season-2026"
    assert requests == 2


def test_current_ahl_season_does_not_cache_stale_rollover_flag(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_SEASON_ID", "")
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE", None)
    monkeypatch.setattr(data_fetch, "_AHL_SEASON_CACHE_YEAR", None)
    monkeypatch.setattr(data_fetch, "_expected_ahl_season_year", lambda: 2026)
    responses = iter(
        [
            _season_payload(
                [{"id": "outgoing", "name": "2025-26", "current": "yes"}]
            ),
            _season_payload(
                [
                    {"id": "outgoing", "name": "2025-26", "current": "yes"},
                    {"id": "new", "name": "2026-27"},
                ]
            ),
        ]
    )
    monkeypatch.setattr(data_fetch, "_ahl_request", lambda *args, **kwargs: next(responses))

    assert data_fetch._current_ahl_season_id() == "outgoing"
    assert data_fetch._AHL_SEASON_CACHE is None
    assert data_fetch._current_ahl_season_id() == "new"
    assert data_fetch._AHL_SEASON_CACHE == "new"
    assert data_fetch._AHL_SEASON_CACHE_YEAR == 2026


def test_fetch_ahl_schedule_discovers_season_then_falls_back_when_empty(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_SEASON_ID", "")
    monkeypatch.setattr(data_fetch, "_current_ahl_season_id", lambda: "discovered")
    calls = []

    def fake_request(view, *, feed="statviewfeed", **params):
        calls.append((view, feed, params.get("season_id")))
        if view == "schedule" and params.get("season_id") is None:
            return {"SiteKit": {"Schedule": {"rows": [{"not": "a game"}]}}}
        return None

    monkeypatch.setattr(data_fetch, "_ahl_request", fake_request)

    assert data_fetch._fetch_ahl_schedule() == []
    assert calls == [
        ("schedule", "statviewfeed", "discovered"),
        ("schedule", "modulekit", "discovered"),
        ("schedule", "statviewfeed", None),
    ]


def test_fetch_ahl_schedule_raises_when_all_schedule_requests_fail(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_SEASON_ID", "current")
    monkeypatch.setattr(data_fetch, "_ahl_request", lambda *args, **kwargs: None)

    try:
        data_fetch._fetch_ahl_schedule()
    except RuntimeError as exc:
        assert "schedule refresh failed" in str(exc)
    else:
        raise AssertionError("failed schedule requests should not look like an empty feed")


def test_normalize_ahl_future_statuses_before_final_prefix():
    assert data_fetch._normalize_status("FUT") == "FUT"
    assert data_fetch._normalize_status("Future") == "FUT"
    assert data_fetch._normalize_status("Final") == "FINAL"


def test_classify_wolves_games_excludes_canceled_and_postponed_fixtures(monkeypatch):
    now = data_fetch.datetime.datetime(2026, 1, 1, tzinfo=data_fetch.pytz.UTC)

    class FixedDatetime(data_fetch.datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(data_fetch.datetime, "datetime", FixedDatetime)
    future = now + data_fetch.datetime.timedelta(days=1)
    later = now + data_fetch.datetime.timedelta(days=2)
    games = [
        {"start_utc": future, "status": {"state": "CANCELED"}, "is_home": True},
        {"start_utc": future, "status": {"state": "POSTPONED"}, "is_home": False},
        {"start_utc": later, "status": {"state": "FUT"}, "is_home": True},
    ]

    classified = data_fetch._classify_wolves_games(games)

    assert classified["next_game"] is games[2]
    assert classified["next_home_game"] is games[2]


def test_fetch_wolves_games_uses_hockeytech_when_calendar_is_unconfigured(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_API_KEY", "configured-key")
    monkeypatch.setattr(data_fetch, "_fetch_wolves_ics_games", list)
    monkeypatch.setattr(data_fetch, "_fetch_ahl_schedule", lambda: [{"source": "api"}])
    expected = {**_empty_classification(), "next_game": {"source": "api"}}
    monkeypatch.setattr(data_fetch, "_classify_wolves_games", lambda games: expected)

    assert data_fetch.fetch_wolves_games(force_refresh=True) == expected


def test_fetch_wolves_games_prefers_authoritative_api_fields(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_API_KEY", "configured-key")
    monkeypatch.setattr(data_fetch, "_fetch_wolves_ics_games", lambda: [{"source": "ics"}])
    monkeypatch.setattr(data_fetch, "_fetch_ahl_schedule", lambda: [{"source": "api"}])
    calendar = {
        **_empty_classification(),
        "last_game": {"source": "ics"},
        "next_game": {"source": "ics"},
    }
    api = {
        **_empty_classification(),
        "last_game": {"source": "api"},
        "live_game": {"source": "api"},
    }
    monkeypatch.setattr(data_fetch, "_classify_wolves_ics_games", lambda games: calendar)
    monkeypatch.setattr(data_fetch, "_classify_wolves_games", lambda games: api)

    assert data_fetch.fetch_wolves_games(force_refresh=True) == {
        **_empty_classification(),
        "last_game": {"source": "api"},
        "live_game": {"source": "api"},
        "next_game": {"source": "ics"},
    }


def test_fetch_wolves_games_does_not_probe_hockeytech_without_key(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_API_KEY", "")
    monkeypatch.setattr(data_fetch, "_fetch_wolves_ics_games", list)
    monkeypatch.setattr(
        data_fetch,
        "_fetch_ahl_schedule",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected API request")),
    )

    assert data_fetch.fetch_wolves_games(force_refresh=True) == _empty_classification()


def test_fetch_wolves_games_bypasses_cache_during_live_game(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_API_KEY", "configured-key")
    monkeypatch.setattr(data_fetch, "_fetch_wolves_ics_games", list)
    monkeypatch.setattr(data_fetch.time, "time", lambda: 100.0)
    cached = {
        **_empty_classification(),
        "live_game": {"status": {"state": "LIVE"}, "away_score": 1},
    }
    monkeypatch.setattr(
        data_fetch,
        "_wolves_cache",
        {"expires": 100.0 + data_fetch._WOLVES_CACHE_TTL, "data": cached},
    )
    refreshed = {
        **_empty_classification(),
        "live_game": {"status": {"state": "LIVE"}, "away_score": 2},
    }
    requests = 0

    def fetch_schedule():
        nonlocal requests
        requests += 1
        return [{"source": "api"}]

    monkeypatch.setattr(data_fetch, "_fetch_ahl_schedule", fetch_schedule)
    monkeypatch.setattr(data_fetch, "_classify_wolves_games", lambda games: refreshed)

    assert data_fetch.fetch_wolves_games() == refreshed
    assert requests == 1


def test_fetch_wolves_games_preserves_cached_live_game_when_refresh_fails(monkeypatch):
    monkeypatch.setattr(data_fetch, "AHL_API_KEY", "configured-key")
    monkeypatch.setattr(data_fetch.time, "time", lambda: 100.0)
    cached = {
        **_empty_classification(),
        "live_game": {"status": {"state": "LIVE"}, "away_score": 3},
    }
    monkeypatch.setattr(
        data_fetch,
        "_wolves_cache",
        {"expires": 100.0 + data_fetch._WOLVES_CACHE_TTL, "data": cached},
    )
    monkeypatch.setattr(
        data_fetch,
        "_fetch_wolves_ics_games",
        lambda: [{"source": "calendar"}],
    )
    monkeypatch.setattr(
        data_fetch,
        "_classify_wolves_ics_games",
        lambda games: {**_empty_classification(), "next_game": games[0]},
    )
    monkeypatch.setattr(
        data_fetch,
        "_fetch_ahl_schedule",
        lambda: (_ for _ in ()).throw(RuntimeError("temporary failure")),
    )

    assert data_fetch.fetch_wolves_games() is cached
    assert data_fetch._wolves_cache["data"] is cached
