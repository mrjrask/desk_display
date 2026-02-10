import datetime as dt

import pytest

from requests import HTTPError

from screens.data_sources import olympic_hockey
from screens.data_sources.olympic_hockey import normalize_espn_olympic_response, resolve_display_date


def test_resolve_display_date_uses_previous_day_before_cutoff():
    now = dt.datetime(2026, 2, 15, 9, 15, tzinfo=dt.timezone.utc)
    date_value = resolve_display_date(tz_name="UTC", now=now)
    assert date_value == dt.date(2026, 2, 14)


def test_normalize_espn_olympic_response_shape():
    payload = {
        "events": [
            {
                "id": "401",
                "date": "2026-02-15T18:00:00Z",
                "status": {"type": {"state": "in", "shortDetail": "2nd"}, "displayClock": "10:14"},
                "competitions": [
                    {
                        "venue": {"fullName": "Milano Arena"},
                        "competitors": [
                            {"homeAway": "away", "score": "2", "team": {"abbreviation": "USA", "displayName": "United States"}},
                            {"homeAway": "home", "score": "3", "team": {"abbreviation": "CAN", "displayName": "Canada"}},
                        ],
                    }
                ],
            }
        ]
    }

    games = normalize_espn_olympic_response(payload, league_key="olympic_mhockey")

    assert len(games) == 1
    game = games[0]
    assert game["leagueKey"] == "olympic_mhockey"
    assert game["status"] == "live"
    assert game["period"] == "2nd"
    assert game["clock"] == "10:14"
    assert game["home"]["code3"] == "CAN"
    assert game["away"]["code3"] == "USA"
    assert game["source"]["providerName"] == "espn"


def test_espn_provider_retries_without_date_filter_on_400(monkeypatch: pytest.MonkeyPatch):
    payload = {
        "events": [
            {
                "id": "402",
                "date": "2026-02-15T18:00:00Z",
                "status": {"type": {"state": "pre", "shortDetail": "Scheduled"}, "displayClock": ""},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "away", "score": "0", "team": {"abbreviation": "SWE", "displayName": "Sweden"}},
                            {"homeAway": "home", "score": "0", "team": {"abbreviation": "FIN", "displayName": "Finland"}},
                        ],
                    }
                ],
            }
        ]
    }
    calls: list[dict[str, object]] = []

    def fake_http_json(url: str, *, params=None, provider_name: str):
        calls.append({"url": url, "params": params, "provider_name": provider_name})
        if len(calls) == 1:
            err = HTTPError("400")
            err.response = type("Response", (), {"status_code": 400})()
            raise err
        return payload

    monkeypatch.setattr(olympic_hockey, "_http_json", fake_http_json)

    result = olympic_hockey._espn_provider(dt.date(2026, 2, 9), "women")

    assert result.provider_name == "espn"
    assert len(result.games) == 1
    assert calls[0]["params"] == {"dates": "20260209"}
    assert calls[1]["params"] is None


def test_extract_embedded_events_from_html_parses_balanced_json():
    html = """
        <html><script>
        window.__DATA__ = {"events":[
          {"id":"1","name":"Men Group A","competitions":[{"competitors":[]}]} ,
          {"id":"2","name":"Women Group A","competitions":[{"competitors":[]}]}
        ],"other":1}
        </script></html>
    """

    events = olympic_hockey._extract_embedded_events_from_html(html)

    assert [event["id"] for event in events] == ["1", "2"]


def test_extract_embedded_events_from_nested_next_data_json():
    html = """
        <script id="__NEXT_DATA__" type="application/json">
        {
          "props": {
            "pageProps": {
              "scoreboard": {
                "events": [
                  {"id": "w1", "name": "Women Preliminary", "date": "2026-02-15T20:00:00Z", "competitions": [{"competitors": []}]},
                  {"id": "w2", "name": "Women Preliminary", "date": "2026-02-15T22:00:00Z", "competitions": [{"competitors": []}]}
                ]
              }
            }
          }
        }
        </script>
    """

    events = olympic_hockey._extract_embedded_events_from_html(html)

    assert [event["id"] for event in events] == ["w1", "w2"]


def test_espn_results_page_provider_filters_women(monkeypatch: pytest.MonkeyPatch):
    payload = {
        "events": [
            {
                "id": "401",
                "name": "Men Preliminary Round",
                "date": "2026-02-15T18:00:00Z",
                "status": {"type": {"state": "pre", "shortDetail": "Scheduled"}, "displayClock": ""},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "away", "score": "0", "team": {"abbreviation": "SWE", "displayName": "Sweden"}},
                            {"homeAway": "home", "score": "0", "team": {"abbreviation": "FIN", "displayName": "Finland"}},
                        ]
                    }
                ],
            },
            {
                "id": "402",
                "name": "Women Preliminary Round",
                "date": "2026-02-15T20:00:00Z",
                "status": {"type": {"state": "in", "shortDetail": "2nd"}, "displayClock": "10:14"},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "away", "score": "1", "team": {"abbreviation": "USA", "displayName": "United States"}},
                            {"homeAway": "home", "score": "2", "team": {"abbreviation": "CAN", "displayName": "Canada"}},
                        ]
                    }
                ],
            },
        ]
    }
    html = f"<script>window.__DATA__ = {payload!r}</script>".replace("'", '"')

    monkeypatch.setattr(olympic_hockey, "_http_text", lambda *args, **kwargs: html)

    result = olympic_hockey._espn_results_page_provider(dt.date(2026, 2, 15), "women")

    assert result.provider_name == "espn_results_page"
    assert len(result.games) == 1
    assert result.games[0]["gameId"] == "402"


def test_espn_provider_filters_division_when_using_generic_olympic_endpoint(monkeypatch: pytest.MonkeyPatch):
    payload = {
        "events": [
            {
                "id": "m1",
                "name": "Men Preliminary Round",
                "date": "2026-02-15T18:00:00Z",
                "status": {"type": {"state": "pre", "shortDetail": "Scheduled"}, "displayClock": ""},
                "competitions": [{"competitors": [
                    {"homeAway": "away", "score": "0", "team": {"abbreviation": "SWE", "displayName": "Sweden"}},
                    {"homeAway": "home", "score": "0", "team": {"abbreviation": "FIN", "displayName": "Finland"}},
                ]}],
            },
            {
                "id": "w1",
                "name": "Women Preliminary Round",
                "date": "2026-02-15T20:00:00Z",
                "status": {"type": {"state": "pre", "shortDetail": "Scheduled"}, "displayClock": ""},
                "competitions": [{"competitors": [
                    {"homeAway": "away", "score": "0", "team": {"abbreviation": "USA", "displayName": "United States"}},
                    {"homeAway": "home", "score": "0", "team": {"abbreviation": "CAN", "displayName": "Canada"}},
                ]}],
            },
        ]
    }

    calls: list[dict[str, object]] = []

    def fake_http_json(url: str, *, params=None, provider_name: str):
        calls.append({"url": url, "params": params})
        if "womens-olympics" in url:
            err = HTTPError("400")
            err.response = type("Response", (), {"status_code": 400})()
            raise err
        return payload

    monkeypatch.setattr(olympic_hockey, "_http_json", fake_http_json)

    result = olympic_hockey._espn_provider(dt.date(2026, 2, 9), "women")

    assert len(result.games) == 1
    assert result.games[0]["gameId"] == "w1"
    assert any("/sports/hockey/olympics/scoreboard" in str(call["url"]) for call in calls)
