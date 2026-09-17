import data_fetch


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
