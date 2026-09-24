import screens.mlb_schedule as mlb_schedule


def _postponed_game(*, abstract="final", detailed="postponed"):
    return {
        "officialDate": "2026-05-10",
        "gameDate": "2026-05-10T23:00:00Z",
        "status": {
            "abstractGameState": abstract,
            "detailedState": detailed,
            "statusCode": "PPD",
        },
        "teams": {
            "away": {"team": {"id": 112}},
            "home": {"team": {"id": 121}},
        },
    }


def test_postponed_game_reported_as_final_is_not_treated_as_final():
    # Regression: MLB can report a postponed game that won't be made up
    # with abstractGameState "Final" (or a detailedState combining "final"
    # and "postponed"), which _is_final_game used to treat as an actually
    # played, final game.
    game = _postponed_game(abstract="final", detailed="final: postponed")

    assert mlb_schedule._is_postponed_game(game)
    assert not mlb_schedule._is_final_game(game)


def test_series_line_shows_postponed_not_final():
    game = _postponed_game(abstract="final", detailed="final: postponed")

    assert mlb_schedule._series_line(game, focus_id=112) == "Sun 5/10 • Postponed"


def test_series_line_still_shows_final_for_a_real_final_game():
    game = {
        "officialDate": "2026-05-10",
        "gameDate": "2026-05-10T23:00:00Z",
        "status": {
            "abstractGameState": "final",
            "detailedState": "final",
            "statusCode": "F",
        },
        "teams": {
            "away": {"team": {"id": 112}, "score": 4},
            "home": {"team": {"id": 121}, "score": 2},
        },
    }

    assert mlb_schedule._is_final_game(game)
    assert mlb_schedule._series_line(game, focus_id=112) == "Sun 5/10 • W 4-2"
