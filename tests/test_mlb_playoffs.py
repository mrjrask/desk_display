"""Tests for the MLB Playoffs screen and its postseason feed."""
from __future__ import annotations

import datetime
import json
from types import SimpleNamespace

import pytest

from config import CENTRAL_TIME, HEIGHT, WIDTH
from screens import mlb_playoffs, playoff_bracket
from services import feeds
from services.data_coordinator import DataCoordinator
from services.server_feeds import LIVE_REFRESH_SECONDS, ServerFeedService
from services.sports import mlb_postseason as mp

_LEAGUE_IDS = {
    "BAL": 103, "KC": 103, "HOU": 103, "DET": 103, "NYY": 103, "CLE": 103,
    "SD": 104, "ATL": 104, "MIL": 104, "NYM": 104, "LAD": 104, "PHI": 104, "CHC": 104,
}
_NAMES = {
    "BAL": "Baltimore Orioles", "KC": "Kansas City Royals", "HOU": "Houston Astros", "DET": "Detroit Tigers",
    "NYY": "New York Yankees", "CLE": "Cleveland Guardians", "SD": "San Diego Padres", "ATL": "Atlanta Braves",
    "MIL": "Milwaukee Brewers", "NYM": "New York Mets", "LAD": "Los Angeles Dodgers",
    "PHI": "Philadelphia Phillies", "CHC": "Chicago Cubs",
}
SEEDS = {
    "AL": {"NYY": 1, "CLE": 2, "HOU": 3, "BAL": 4, "KC": 5, "DET": 6},
    "NL": {"LAD": 1, "PHI": 2, "MIL": 3, "SD": 4, "ATL": 5, "NYM": 6},
}
_STATES = {
    "final": {"abstractGameState": "Final", "detailedState": "Final"},
    "live": {"abstractGameState": "Live", "detailedState": "In Progress"},
    "pre": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
    "ppd": {"abstractGameState": "Final", "detailedState": "Postponed"},
}
_PK = iter(range(1, 10_000))


def _team(abbr):
    return {"id": 1, "name": _NAMES[abbr], "abbreviation": abbr, "league": {"id": _LEAGUE_IDS[abbr]}}


def game(round_code, away, home, number, state="final", away_score=None, home_score=None,
         date="2026-10-01T18:08:00Z", of=None):
    return {
        "gamePk": next(_PK),
        "gameType": round_code,
        "gameDate": date,
        "status": dict(_STATES[state]),
        "seriesGameNumber": number,
        "gamesInSeries": of or mp.DEFAULT_BEST_OF[round_code],
        "teams": {
            "away": {"team": _team(away), "score": away_score},
            "home": {"team": _team(home), "score": home_score},
        },
    }


def postseason(games, seeds=SEEDS):
    series = mp.series_from_games({"dates": [{"games": games}]})
    data = {"season": 2026, "series": series, "seeds": seeds, "names": {}}
    for item in series:
        data["names"].update(item["names"])
    # The feed is saved as JSON; the screen must work from that form.
    return json.loads(json.dumps(data))


# ── Series ──────────────────────────────────────────────────────────────────


def test_series_count_wins_live_and_next_game():
    series = mp.series_from_games({"dates": [{"games": [
        game("F", "KC", "BAL", 1, "final", 1, 0),
        game("F", "KC", "BAL", 2, "live"),
        game("F", "KC", "BAL", 3, "pre", date="2026-10-02T23:08:00Z"),
        game("F", "DET", "HOU", 1, "final", 3, 1),
        game("F", "DET", "HOU", 2, "ppd"),
        game("F", "DET", "HOU", 2, "pre", date="2026-10-03T17:08:00Z"),
    ]}]})
    by_teams = {tuple(s["teams"]): s for s in series}

    kc = by_teams[("BAL", "KC")]  # game 1's home team is listed first
    assert kc["wins"] == {"BAL": 0, "KC": 1}
    assert kc["live"] is True and kc["league"] == "AL" and kc["best_of"] == 3
    assert kc["next_game"] == 3 and kc["winner"] is None

    det = by_teams[("HOU", "DET")]
    assert det["live"] is False
    assert det["next_start"].startswith("2026-10-03T17:08")  # the postponed game is skipped


def test_series_winner_clears_live_and_next_game():
    series = mp.series_from_games([
        game("F", "KC", "BAL", 1, "final", 1, 0),
        game("F", "KC", "BAL", 2, "final", 2, 1),
        game("F", "KC", "BAL", 3, "pre", date="2026-10-02T23:08:00Z"),  # if necessary
    ])
    assert series[0]["winner"] == "KC"
    assert series[0]["next_start"] is None and series[0]["next_game"] is None


def test_series_ignore_placeholder_teams_and_use_logo_codes():
    placeholder = game("D", "NYY", "NYY", 1, "pre")
    placeholder["teams"]["away"]["team"] = {"id": 0, "name": "AL Wild Card Series A Winner"}
    cubs = game("F", "SD", "CHC", 1, "final", 1, 3)
    series = mp.series_from_games([placeholder, cubs])
    assert [s["teams"] for s in series] == [["CUBS", "SD"]]
    assert series[0]["names"]["CUBS"] == "Cubs"


# ── Seeds ───────────────────────────────────────────────────────────────────


def _standings_record(league_id, division_id, rows):
    return {
        "league": {"id": league_id},
        "division": {"id": division_id},
        "teamRecords": [
            {"team": _team(abbr), "divisionRank": str(div_rank), "leagueRank": str(league_rank),
             "wildCardRank": None if wc_rank is None else str(wc_rank), "winningPercentage": ".500"}
            for abbr, div_rank, league_rank, wc_rank in rows
        ],
    }


def test_seeds_are_division_winners_then_wild_cards():
    payload = {"records": [
        _standings_record(103, 201, [("NYY", 1, 1, None), ("BAL", 2, 4, 1)]),
        _standings_record(103, 202, [("CLE", 1, 2, None), ("KC", 2, 5, 2), ("DET", 3, 6, 3)]),
        _standings_record(103, 200, [("HOU", 1, 3, None)]),
    ]}
    assert mp.seeds_from_standings(payload) == {"AL": SEEDS["AL"]}


# ── Bracket ─────────────────────────────────────────────────────────────────


def test_projected_bracket_places_seeds_like_mlb_com():
    bracket = mp.build_bracket({"series": [], "seeds": SEEDS})
    al = bracket["AL"]
    assert [slot["teams"] for slot in al["F"]] == [["BAL", "KC"], ["HOU", "DET"]]
    assert [slot["teams"] for slot in al["D"]] == [["NYY", None], ["CLE", None]]
    assert al["L"][0]["teams"] == [None, None]
    assert all(slot["series"] is None for slot in al["F"])


def test_bracket_advances_winners_and_aligns_later_rounds():
    data = postseason([
        game("F", "KC", "BAL", 1, "final", 1, 0), game("F", "KC", "BAL", 2, "final", 2, 1),
        game("F", "DET", "HOU", 1, "final", 3, 1), game("F", "DET", "HOU", 2, "final", 5, 2),
        # Only the 2 seed's Division Series has been scheduled.
        game("D", "DET", "CLE", 1, "final", 0, 7),
    ])
    al = mp.build_bracket(data)["AL"]
    assert al["F"][0]["winner"] == "KC" and al["F"][0]["wins"] == [0, 2]
    assert al["D"][0]["teams"] == ["NYY", "KC"] and al["D"][0]["series"] is None
    assert al["D"][1]["teams"] == ["CLE", "DET"] and al["D"][1]["wins"] == [1, 0]
    assert mp.current_round(data) == "D"


def test_bracket_without_seeds_still_links_rounds():
    data = postseason([
        game("F", "KC", "BAL", 1, "final", 1, 0), game("F", "KC", "BAL", 2, "final", 2, 1),
        game("F", "DET", "HOU", 1, "final", 3, 1), game("F", "DET", "HOU", 2, "final", 5, 2),
        game("D", "KC", "NYY", 1, "final", 1, 2),
        game("D", "DET", "CLE", 1, "final", 1, 2),
    ], seeds={})
    al = mp.build_bracket(data)["AL"]
    feeders = [set(slot["teams"]) for slot in al["F"]]
    for index, slot in enumerate(al["D"]):
        assert set(slot["teams"]) & feeders[index]


def test_world_series_lists_the_american_league_first():
    data = postseason([game("W", "NYY", "LAD", 1, "final", 3, 6)])
    world = mp.build_bracket(data)["W"][0]
    assert world["teams"] == ["NYY", "LAD"] and world["wins"] == [0, 1]


# ── Screen ──────────────────────────────────────────────────────────────────


NOW = datetime.datetime(2026, 10, 1, 12, 0, tzinfo=CENTRAL_TIME)


def _slot(data, league, round_code, index=0):
    return mp.build_bracket(data)[league][round_code][index]


def test_status_lines():
    data = postseason([
        game("F", "KC", "BAL", 1, "final", 1, 0), game("F", "KC", "BAL", 2, "final", 2, 1),
        game("F", "DET", "HOU", 1, "final", 3, 1),
        game("F", "DET", "HOU", 2, "pre", date="2026-10-02T00:08:00Z"),
        game("F", "ATL", "SD", 1, "live"),
    ])
    assert mlb_playoffs.series_status(_slot(data, "AL", "F", 0), data, NOW)[0] == "Royals win 2-0"
    assert mlb_playoffs.series_status(_slot(data, "AL", "F", 1), data, NOW)[0] == "Game 2 · Tonight 7:08 PM"
    text, fill = mlb_playoffs.series_status(_slot(data, "NL", "F", 0), data, NOW)
    assert text == "Game 1 · LIVE" and fill == playoff_bracket.SCOREBOARD_IN_PROGRESS_SCORE_COLOR
    assert mlb_playoffs.series_status(_slot(data, "NL", "F", 1), data, NOW)[0] == "Projected"


@pytest.mark.parametrize("data", [
    {},
    {"series": [], "seeds": SEEDS, "names": {}},
    postseason([game("F", "KC", "BAL", 1, "final", 1, 0), game("W", "NYY", "LAD", 1, "live")]),
])
def test_screen_image_fills_the_display_width(data):
    image = mlb_playoffs.compose_playoffs_image(data, now=NOW)
    assert image.width == WIDTH and image.height >= HEIGHT


class _Display:
    def __init__(self):
        self.images = []

    def image(self, img):
        self.images.append(img)

    def clear(self):
        pass


def test_render_uses_given_feed_data_without_fetching(monkeypatch):
    monkeypatch.setattr(mp, "fetch_postseason", lambda **_: pytest.fail("fetched upstream"))
    monkeypatch.setattr(playoff_bracket, "scroll_vertical_content", lambda **kwargs: kwargs["render_at_offset"](0))
    monkeypatch.setattr(playoff_bracket.time, "sleep", lambda _s: None)
    display = _Display()
    result = mlb_playoffs.render_mlb_playoffs(display, {"series": [], "seeds": SEEDS}, transition=True)
    assert result.displayed and display.images


def test_registry_hands_the_server_feed_to_the_screen(monkeypatch):
    import importlib

    from display_profiles import resolve_display_profile

    # Other tests drop screens.registry from sys.modules; use the live module.
    registry_module = importlib.import_module("screens.registry")
    ScreenContext, build_screen_registry = registry_module.ScreenContext, registry_module.build_screen_registry

    received = []
    monkeypatch.setattr(registry_module, "render_mlb_playoffs",
                        lambda display, data, transition=False: received.append(data))
    feed = {"series": [], "seeds": SEEDS}
    context = ScreenContext(
        display=_Display(), cache={"mlb_postseason": feed}, logos={}, image_dir="", now=NOW,
        now_utc=NOW.astimezone(datetime.UTC), offline=False, weather_fetched_at=None, skip_scoreboards=False,
        render_profile=resolve_display_profile(320, 240), allow_upstream_requests=False,
    )
    registry, _ = build_screen_registry(context)
    registry["MLB Playoffs"].render()
    assert received == [feed]


# ── Server feed ─────────────────────────────────────────────────────────────


def test_mlb_playoffs_is_catalogued_with_its_feed():
    from rendering.screen_classes import SCROLLING_CANVAS, classify
    from screens_catalog import SCREEN_IDS

    assert "MLB Playoffs" in SCREEN_IDS
    assert classify("MLB Playoffs").kind == SCROLLING_CANVAS
    assert feeds.feeds_for_screen("MLB Playoffs", feeds.SERVER_FEED_DEPENDENCIES) == {"mlb_postseason"}
    assert feeds.feeds_for_screen("MLB Playoffs") == set()  # standalone fetches while drawing


class _Clock:
    now = 1_000.0

    def __call__(self):
        return self.now


def test_server_refreshes_the_bracket_faster_while_a_game_is_live():
    clock = _Clock()
    value = {"series": [{"round": "F", "live": False}], "seeds": {}}
    calls = []

    def fetch(*, force=False):
        calls.append(force)
        return {"mlb_postseason": json.loads(json.dumps(value))}

    service = ServerFeedService(
        DataCoordinator(SimpleNamespace()), SimpleNamespace(), fetch_air_quality=lambda *a, **k: None,
        settings=SimpleNamespace(ENABLE_WEATHER=False, ENABLE_AIR_QUALITY=False),
        standings_fetchers={"mlb_postseason": fetch}, history_path="/nonexistent/aq.json",
        clock=clock, wall_clock=clock,
    )
    assert service.refresh({"MLB Playoffs"}) == {"mlb_postseason": True}
    clock.now += LIVE_REFRESH_SECONDS
    assert service.refresh({"MLB Playoffs"}) == {}
    value["series"][0]["live"] = True
    clock.now += feeds.SERVER_FEED_REFRESH_INTERVALS["mlb_postseason"]
    assert service.refresh({"MLB Playoffs"}) == {"mlb_postseason": True}
    clock.now += LIVE_REFRESH_SECONDS
    assert service.refresh({"MLB Playoffs"}) == {"mlb_postseason": True}
    assert calls == [False, False, True]


def test_fetch_fails_only_when_no_source_answers(monkeypatch):
    def boom(url, params):
        raise OSError("offline")

    monkeypatch.setattr(mp, "_get_json", boom)
    with pytest.raises(RuntimeError):
        mp.fetch_postseason(force=True)

    def standings_only(url, params):
        if url == mp.STANDINGS_URL:
            return {"records": [
                _standings_record(103, 201, [("NYY", 1, 1, None), ("BAL", 2, 4, 1)]),
                _standings_record(103, 202, [("CLE", 1, 2, None), ("KC", 2, 5, 2), ("DET", 3, 6, 3)]),
                _standings_record(103, 200, [("HOU", 1, 3, None)]),
            ]}
        return {"dates": []}

    monkeypatch.setattr(mp, "_get_json", standings_only)
    data = mp.fetch_postseason(force=True)
    assert data["series"] == [] and data["seeds"] == {"AL": SEEDS["AL"]}
    assert data["names"]["NYY"] == "Yankees"


def test_server_renders_the_screen_from_its_snapshot(monkeypatch):
    from display_profiles import DISPLAY_PROFILE_DISPLAY_HAT_MINI, PROFILE_PRESETS
    from rendering.screen_renderer import ScreenRenderer, ServerPreferenceSnapshot

    monkeypatch.setattr(mp, "fetch_postseason", lambda **_: pytest.fail("fetched upstream"))
    monkeypatch.setattr(playoff_bracket.time, "sleep", lambda _s: None)
    monkeypatch.setattr(playoff_bracket, "scroll_vertical_content",
                        lambda **kwargs: kwargs["render_at_offset"](0))
    data = postseason([game("F", "KC", "BAL", 1, "final", 1, 0), game("F", "DET", "HOU", 1, "live")])
    snapshot = DataCoordinator().publish("mlb_postseason", data)
    artifact = ScreenRenderer().render(
        "MLB Playoffs", PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI], ServerPreferenceSnapshot(revision=1), snapshot,
    )
    assert artifact.image.size == (320, 240)
    assert mp.has_live_series(snapshot["mlb_postseason"])


def test_series_list_uses_one_full_width_line_per_series(monkeypatch):
    calls = []
    monkeypatch.setattr(playoff_bracket, "draw_series_row",
                        lambda canvas, draw, spec, slot, names, seeds, x, width, y, now: calls.append((x, width, y)))
    data = {"series": [], "seeds": SEEDS}
    mlb_playoffs._compose_series_list(1280, data, mlb_playoffs._bracket(data), NOW)
    assert len(calls) == 4
    assert {(x, width) for x, width, _y in calls} == {(calls[0][0], 1280 - 2 * calls[0][0])}
    assert [y for _x, _w, y in calls] == sorted({y for _x, _w, y in calls})
