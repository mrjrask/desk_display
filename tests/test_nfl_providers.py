from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from services.sports import nfl

FIXTURES = Path(__file__).with_name("fixtures")


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


class Response:
    def __init__(self, payload=None, *, error=False, text=""):
        self.payload, self.error, self.text = payload, error, text

    def raise_for_status(self):
        if self.error:
            raise RuntimeError("HTTP 503")

    def json(self):
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        return next(self.responses)


@pytest.fixture
def dates():
    return dt.date(2026, 9, 3), dt.date(2026, 9, 9)


def test_primary_success_uses_site_only(dates):
    session = Session([Response(fixture("nfl_espn_site.json"))])
    games = nfl.fetch_range(*dates, session=session, cache={})
    assert [game["id"] for game in games] == ["401"]
    assert len(session.urls) == 1 and session.urls[0].startswith(nfl.ESPN_SITE_URL)


@pytest.mark.parametrize("primary", [Response({"events": []}), Response(error=True)])
def test_empty_or_http_failure_primary_uses_cdn(primary, dates):
    session = Session([primary, Response(fixture("nfl_espn_cdn.json"))])
    assert [game["id"] for game in nfl.fetch_range(*dates, session=session, cache={})] == ["402"]
    assert session.urls[1].startswith(nfl._CDN_SCOREBOARD_URL)


def test_both_espn_formats_normalize_to_same_contract():
    games = nfl.normalize_espn_site(fixture("nfl_espn_site.json")) + nfl.normalize_espn_cdn(fixture("nfl_espn_cdn.json"))
    for game in games:
        assert {
            "event_id", "start_time", "event_name", "competitors", "scores", "status"
        } <= game.keys()
        assert {side["homeAway"] for side in game["competitors"]} == {"away", "home"}


def test_duplicate_event_ids_are_removed():
    payload = fixture("nfl_espn_site.json")
    payload["events"].append(payload["events"][0])
    assert len(nfl.normalize_espn_site(payload)) == 1


def test_weekly_dates_count_one_failure_without_discarding_successes(dates):
    session = Session([
        Response(fixture("nfl_espn_site.json")),
        Response(error=True),
        Response({"events": []}),
    ])

    result = nfl.fetch_week_dates(
        [dates[0], dates[0] + dt.timedelta(days=1), dates[0] + dt.timedelta(days=2)],
        session=session,
    )

    assert [game["id"] for game in result.games] == ["401"]
    assert result.successful_dates == 2
    assert result.failed_dates == 1


def test_weekly_dates_treat_empty_events_as_success(dates):
    result = nfl.fetch_week_dates(
        [dates[0], dates[0] + dt.timedelta(days=1)],
        session=Session([Response({"events": []}), Response({"events": []})]),
    )

    assert result.games == []
    assert result.successful_dates == 2
    assert result.failed_dates == 0


def test_stale_cache_is_retained_when_every_provider_fails(dates):
    stale = [{"id": "cached"}]
    cache = {(dates[0], f"nfl_providers:{dates[1].isoformat()}"): (0.0, stale)}
    session = Session([Response(error=True), Response(error=True), Response(error=True)])
    assert nfl.fetch_range(*dates, session=session, cache=cache) == stale


def test_total_provider_failure_without_cache_returns_empty(dates):
    session = Session([Response(error=True), Response(error=True), Response(error=True)])
    assert nfl.fetch_range(*dates, session=session, cache={}) == []


def test_range_skips_providers_that_failed_in_an_earlier_window(dates):
    failures = set()
    session = Session([
        Response({"events": []}),
        Response(error=True),
        Response(error=True),
        Response({"events": []}),
    ])

    assert nfl.fetch_range(
        *dates, session=session, cache={}, failed_providers=failures
    ) == []
    next_week = tuple(day + dt.timedelta(days=7) for day in dates)
    assert nfl.fetch_range(
        *next_week, session=session, cache={}, failed_providers=failures
    ) == []

    assert failures == {"ESPN CDN", "nflverse"}
    assert len(session.urls) == 4
    assert session.urls[-1].startswith(nfl.ESPN_SITE_URL)


def test_range_retries_primary_after_an_earlier_window_failure(dates):
    failures = set()
    session = Session(
        [
            Response(error=True),
            Response({"events": []}),
            Response(text=""),
            Response(fixture("nfl_espn_site.json")),
        ]
    )

    assert nfl.fetch_range(
        *dates, session=session, cache={}, failed_providers=failures
    ) == []
    next_week = tuple(day + dt.timedelta(days=7) for day in dates)
    games = nfl.fetch_range(
        *next_week, session=session, cache={}, failed_providers=failures
    )

    assert [game["id"] for game in games] == ["401"]
    assert failures == set()
    assert session.urls[-1].startswith(nfl.ESPN_SITE_URL)


def test_nflverse_schedule_converts_eastern_kickoff_to_utc():
    schedule = nfl._fetch_nflverse_schedule(
        session=Session(
            [
                Response(
                    text=(
                        "game_id,gameday,gametime,away_team,home_team,"
                        "away_score,home_score\n"
                        "week-one,2026-09-10,20:20,DAL,PHI,,\n"
                    )
                )
            ]
        ),
        cache={},
    )

    assert schedule[0]["_event_date"] == "2026-09-11T00:20:00Z"


def test_nflverse_range_filters_utc_rollover_by_local_game_day():
    cache = {}
    session = Session(
        [
            Response(
                text=(
                    "game_id,gameday,gametime,away_team,home_team,"
                    "away_score,home_score,result\n"
                    "sunday-night,2026-09-13,20:20,CHI,GB,,,\n"
                )
            )
        ]
    )

    games = nfl._fetch_nflverse(
        dt.date(2026, 9, 7),
        dt.date(2026, 9, 13),
        session=session,
        cache=cache,
    )

    assert [game["id"] for game in games] == ["sunday-night"]
    assert games[0]["_event_date"] == "2026-09-14T00:20:00Z"


def test_nflverse_schedule_does_not_expose_scores_before_result_is_final():
    schedule = nfl._fetch_nflverse_schedule(
        session=Session(
            [
                Response(
                    text=(
                        "game_id,gameday,gametime,away_team,home_team,"
                        "away_score,home_score,result\n"
                        "live,2026-09-13,13:00,CHI,GB,7,3,\n"
                    )
                )
            ]
        ),
        cache={},
    )

    assert schedule[0]["status"]["type"]["state"] == "pre"
    assert [team["score"] for team in schedule[0]["competitors"]] == [None, None]


def test_nflverse_does_not_treat_unfinalized_scores_as_live_results():
    csv_payload = (
        "game_id,gameday,gametime,away_team,home_team,away_score,home_score,result\n"
        "live,2026-09-06,13:00,CHI,GB,7,3,\n"
    )
    games = nfl._nflverse_games(
        Session([Response(text=csv_payload)]),
        dt.date(2026, 9, 6),
        dt.date(2026, 9, 6),
    )
    assert games[0]["status"]["type"]["state"] == "pre"
    assert games[0]["scores"] == {"away": None, "home": None}
