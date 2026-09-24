import data_fetch


def _reset_wolves_cache():
    with data_fetch._wolves_cache_lock:
        data_fetch._wolves_cache["expires"] = 0.0
        data_fetch._wolves_cache["data"] = None


def test_hockeytech_clears_a_stale_ics_live_game(monkeypatch):
    # Regression: the ICS calendar has no real live state, so it should never
    # be trusted to *set* live_game; but the merge used to only overwrite a
    # key when HockeyTech's value was non-None, so a live_game the ICS pass
    # happened to populate was never cleared once HockeyTech reported the
    # game was no longer live. That also permanently defeated the cache TTL,
    # since the cache is bypassed whenever live_game is set.
    _reset_wolves_cache()
    monkeypatch.setattr(data_fetch, "AHL_API_KEY", "test-key")
    monkeypatch.setattr(data_fetch, "_fetch_wolves_ics_games", lambda: [{"uid": "1"}])
    monkeypatch.setattr(
        data_fetch,
        "_classify_wolves_ics_games",
        lambda games: {
            "last_game": None,
            "live_game": {"source": "ics"},
            "next_game": None,
            "next_home_game": None,
        },
    )
    monkeypatch.setattr(data_fetch, "_fetch_ahl_schedule", lambda: [{"id": "1"}])
    monkeypatch.setattr(
        data_fetch,
        "_classify_wolves_games",
        lambda games: {
            "last_game": {"source": "hockeytech", "final": True},
            "live_game": None,
            "next_game": None,
            "next_home_game": None,
        },
    )

    result = data_fetch.fetch_wolves_games(force_refresh=True)

    assert result["live_game"] is None
    assert result["last_game"] == {"source": "hockeytech", "final": True}


def test_hockeytech_live_game_still_wins_when_present(monkeypatch):
    _reset_wolves_cache()
    monkeypatch.setattr(data_fetch, "AHL_API_KEY", "test-key")
    monkeypatch.setattr(data_fetch, "_fetch_wolves_ics_games", lambda: [{"uid": "1"}])
    monkeypatch.setattr(
        data_fetch,
        "_classify_wolves_ics_games",
        lambda games: {
            "last_game": None,
            "live_game": {"source": "ics"},
            "next_game": None,
            "next_home_game": None,
        },
    )
    monkeypatch.setattr(data_fetch, "_fetch_ahl_schedule", lambda: [{"id": "1"}])
    monkeypatch.setattr(
        data_fetch,
        "_classify_wolves_games",
        lambda games: {
            "last_game": None,
            "live_game": {"source": "hockeytech", "score": "2-1"},
            "next_game": None,
            "next_home_game": None,
        },
    )

    result = data_fetch.fetch_wolves_games(force_refresh=True)

    assert result["live_game"] == {"source": "hockeytech", "score": "2-1"}
