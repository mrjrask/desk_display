import datetime

import pytz

import data_fetch


def test_blackhawks_ics_url_converts_webcal_to_https():
    assert (
        data_fetch._calendar_url("webcal://ics.ecal.com/ecal-sub/example/NHL.ics")
        == "https://ics.ecal.com/ecal-sub/example/NHL.ics"
    )


def test_normalize_blackhawks_ics_game_maps_schedule_payload():
    event = {
        "UID": "game-1",
        "SUMMARY": "Chicago Blackhawks at St Louis Blues",
        "DTSTART": "20261001T000000Z",
        "LOCATION": "Enterprise Center",
        "__params__": {"DTSTART": {}},
    }

    game = data_fetch._normalize_blackhawks_ics_game(event)

    assert game is not None
    assert game["gameState"] == "FUT"
    assert game["awayTeam"]["abbrev"] == "CHI"
    assert game["awayTeam"]["id"] == data_fetch.NHL_TEAM_ID
    assert game["homeTeam"]["abbrev"] == "STL"
    assert game["startTimeUTC"] == "2026-10-01T00:00:00Z"
    assert game["venue"]["default"] == "Enterprise Center"


def test_fetch_blackhawks_next_game_prefers_ics_schedule(monkeypatch):
    now = datetime.datetime(2026, 7, 20, tzinfo=pytz.UTC)
    games = [
        {
            "gameDate": "2026-10-01T00:00:00Z",
            "gameState": "FUT",
            "startTimeUTC": "2026-10-01T00:00:00Z",
            "homeTeam": {"abbrev": "STL"},
            "awayTeam": {"abbrev": "CHI"},
        }
    ]
    monkeypatch.setattr(data_fetch, "_fetch_blackhawks_ics_schedule", lambda: games)

    result = data_fetch.fetch_blackhawks_next_game()

    assert result is games[0]
    assert result["startTimeCentral"] == now.replace(month=9, day=30, hour=19).strftime("%I:%M %p").lstrip("0")
