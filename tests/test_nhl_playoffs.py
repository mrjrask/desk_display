"""Tests for the NHL Playoffs screen, its bracket feed and the shared 16-team bracket."""
from __future__ import annotations

import datetime
import importlib
import json
from types import SimpleNamespace

import pytest

from config import CENTRAL_TIME, HEIGHT, WIDTH
from screens import nhl_playoffs, playoff_bracket
from services import feeds
from services.data_coordinator import DataCoordinator
from services.server_feeds import LIVE_REFRESH_SECONDS, ServerFeedService
from services.sports import nhl_postseason as np_
from services.sports import playoff_bracket16 as pb

NOW = datetime.datetime(2024, 4, 22, 12, 0, tzinfo=CENTRAL_TIME)
_IDS = iter(range(1, 1000))
_TEAM_IDS: dict[str, int] = {}

# The 2024 playoffs: letter -> (top, top seed, bottom, bottom seed, top wins, bottom wins).
SERIES_2024 = {
    "A": ("FLA", "D1", "TBL", "WC1", 4, 1), "B": ("BOS", "D2", "TOR", "D3", 4, 3),
    "C": ("NYR", "D1", "WSH", "WC2", 4, 0), "D": ("CAR", "D2", "NYI", "D3", 4, 1),
    "E": ("DAL", "D1", "VGK", "WC1", 4, 3), "F": ("WPG", "D2", "COL", "D3", 1, 4),
    "G": ("VAN", "D1", "NSH", "WC1", 4, 2), "H": ("EDM", "D2", "LAK", "D3", 4, 1),
    "I": ("FLA", "", "BOS", "", 4, 2), "J": ("NYR", "", "CAR", "", 4, 2),
    "K": ("DAL", "", "COL", "", 4, 2), "L": ("VAN", "", "EDM", "", 3, 4),
    "M": ("NYR", "", "FLA", "", 2, 4), "N": ("DAL", "", "EDM", "", 2, 4),
    "O": ("FLA", "", "EDM", "", 4, 3),
}


def _team(abbr):
    team_id = _TEAM_IDS.setdefault(abbr, next(_IDS))
    return {"id": team_id, "abbrev": abbr, "commonName": {"default": f"{abbr} name"}}


def bracket_payload(letters=SERIES_2024, wins=None):
    """``playoff-bracket`` JSON for *letters*, with optional ``{letter: (top, bottom)}`` wins."""

    series = []
    for letter, (top, top_seed, bottom, bottom_seed, top_wins, bottom_wins) in letters.items():
        top_wins, bottom_wins = (wins or {}).get(letter, (top_wins, bottom_wins))
        item = {
            "seriesLetter": letter, "playoffRound": np_.SERIES_LETTERS[letter][0],
            "topSeedRankAbbrev": top_seed, "bottomSeedRankAbbrev": bottom_seed,
            "topSeedWins": top_wins, "bottomSeedWins": bottom_wins,
            "topSeedTeam": _team(top), "bottomSeedTeam": _team(bottom),
        }
        if top_wins == 4 or bottom_wins == 4:
            item["winningTeamId"] = _team(top if top_wins == 4 else bottom)["id"]
        series.append(item)
    return {"series": series}


def first_round(wins=None):
    return bracket_payload({k: v for k, v in SERIES_2024.items() if k <= "H"},
                           wins or dict.fromkeys("ABCDEFGH", (1, 1)))


def schedule_payload(games):
    return {"gameWeek": [{"date": "2024-04-22", "games": [
        {"gameType": 3, "awayTeam": {"abbrev": away}, "homeTeam": {"abbrev": home}, "gameState": state,
         "startTimeUTC": start, "seriesStatus": {"gameNumberOfSeries": number}}
        for away, home, state, start, number in games
    ]}]}


def data_from(payload, schedule=None):
    series, names, seeds = np_.series_from_bracket(payload)
    if schedule is not None:
        np_.apply_schedule(series, schedule)
    return {"season": 2024, "series": series, "names": names, "seeds": seeds}


# ── Feed parsing ────────────────────────────────────────────────────────────


def test_bracket_places_series_by_letter_west_left():
    bracket = pb.build_bracket(data_from(bracket_payload()))
    assert [slot["teams"] for slot in bracket["left"][0]] == [
        ["DAL", "VGK"], ["WPG", "COL"], ["VAN", "NSH"], ["EDM", "LAK"]]
    assert [slot["teams"] for slot in bracket["right"][0]][0] == ["FLA", "TBL"]
    assert bracket["left"][2][0]["teams"] == ["DAL", "EDM"] and bracket["left"][2][0]["winner"] == "EDM"
    # Each slot lists the team from its upper feeder first, so the West leads the final.
    assert bracket["center"]["teams"] == ["EDM", "FLA"] and bracket["center"]["wins"] == [3, 4]
    assert bracket["center"]["winner"] == "FLA"
    assert bracket["seeds"]["VGK"] == "WC1" and bracket["seeds"]["FLA"] == "D1"


def test_rounds_not_reached_show_the_winners_who_will_meet():
    data = data_from(first_round({"A": (4, 0), "B": (4, 2), "C": (2, 2), "D": (0, 3),
                                  "E": (1, 1), "F": (1, 1), "G": (1, 1), "H": (1, 1)}))
    bracket = pb.build_bracket(data)
    east_r2 = bracket["right"][1]
    assert east_r2[0]["teams"] == ["FLA", "BOS"] and east_r2[0]["series"] is None
    assert east_r2[1]["teams"] == [None, None]
    assert pb.current_round(data) == 1


def test_schedule_adds_live_games_and_the_next_game():
    schedule = schedule_payload([
        ("TBL", "FLA", "LIVE", "2024-04-22T23:00:00Z", 3),
        ("TOR", "BOS", "FUT", "2024-04-24T23:30:00Z", 3),
        ("TOR", "BOS", "FUT", "2024-04-23T00:00:00Z", 3),
        ("VGK", "DAL", "OFF", "2024-04-21T00:00:00Z", 2),
    ])
    data = data_from(first_round(), schedule)
    by_letter = {s["letter"]: s for s in data["series"]}
    assert by_letter["A"]["live"] is True and by_letter["B"]["live"] is False
    assert by_letter["B"]["next_start"] == "2024-04-23T00:00:00Z" and by_letter["B"]["next_game"] == 3
    assert by_letter["E"]["next_start"] is None
    assert pb.has_live_series(data)


def test_status_lines_follow_the_mlb_screen():
    data = data_from(first_round({**dict.fromkeys("ABCDEFGH", (1, 1)), "C": (4, 0), "D": (0, 2)}),
                     schedule_payload([("TBL", "FLA", "LIVE", "2024-04-22T23:00:00Z", 3),
                                       ("TOR", "BOS", "FUT", "2024-04-23T00:00:00Z", 3)]))
    data["names"]["NYR"] = "Rangers"
    east = pb.build_bracket(data)["right"][0]
    status = [playoff_bracket.series_status(slot, data["names"], NOW)[0] for slot in east]
    assert status == ["Game 3 · LIVE", "Game 3 · Tonight 7 PM", "Rangers win 4-0", "NYI leads 2-0"]


def _standings_row(abbr, conference, division, points, games=10):
    return {"teamAbbrev": {"default": abbr}, "teamCommonName": {"default": abbr.title()},
            "conferenceAbbrev": conference, "divisionAbbrev": division, "points": points, "gamesPlayed": games}


def test_projection_from_standings_uses_the_division_format():
    rows = []
    for conference, divisions in (("E", ("A", "M")), ("W", ("C", "P"))):
        for d_index, division in enumerate(divisions):
            for rank in range(8):
                rows.append(_standings_row(f"{conference}{division}{rank}", conference, division,
                                           100 - rank * 5 - d_index))
    series, _names, seeds = np_.projected_from_standings({"standings": rows})
    east = {s["letter"]: s["teams"] for s in series if s["conference"] == "east"}
    # Atlantic's leader has more points, so it plays the second wild card.
    assert east == {"A": ["EA0", "EM3"], "B": ["EA1", "EA2"], "C": ["EM0", "EA3"], "D": ["EM1", "EM2"]}
    assert seeds["EM3"] == "WC2" and seeds["EA3"] == "WC1" and seeds["EA0"] == "D1"
    data = {"series": series, "seeds": seeds, "names": {}}
    assert pb.current_round(data) is None
    slot = pb.build_bracket(data)["right"][0][0]
    assert slot["teams"] == ["EA0", "EM3"] and slot["series"] is None
    assert playoff_bracket.series_status(slot, {}, NOW)[0] == "Projected"


def test_no_projection_before_games_are_played():
    rows = [_standings_row("BOS", "E", "A", 0, games=0)]
    assert np_.projected_from_standings({"standings": rows}) == ([], {}, {})


def test_fetch_falls_back_to_last_seasons_bracket(monkeypatch):
    requested = []

    def fake(url):
        requested.append(url)
        if url == np_.BRACKET_URL.format(year=2026):
            return bracket_payload()
        raise OSError("not yet")

    monkeypatch.setattr(np_, "_get_json", fake)
    data = np_.fetch_postseason(force=True, now=datetime.datetime(2026, 9, 28, tzinfo=datetime.UTC))
    assert data["season"] == 2026 and len(data["series"]) == 15
    assert requested[0] == np_.BRACKET_URL.format(year=2027)


def test_fetch_fails_when_nothing_answers(monkeypatch):
    monkeypatch.setattr(np_, "_get_json", lambda url: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(RuntimeError):
        np_.fetch_postseason(force=True, now=NOW)


# ── Screen ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("data", [
    {},
    data_from(bracket_payload()),
    data_from(first_round(), schedule_payload([("TBL", "FLA", "LIVE", "2024-04-22T23:00:00Z", 3)])),
])
def test_screen_image_fills_the_display_width(data):
    image = nhl_playoffs.compose_playoffs_image(data, now=NOW)
    assert image.width == WIDTH and image.height >= HEIGHT


def test_series_list_shows_the_current_round(monkeypatch):
    rows = []
    monkeypatch.setattr(playoff_bracket, "draw_series_row",
                        lambda canvas, draw, spec, slot, names, seeds, x, width, y, now: rows.append(slot["teams"]))
    headings = []
    real = playoff_bracket.compose_series_list

    def spy(spec, width, heading, *args, **kwargs):
        headings.append(heading)
        return real(spec, width, heading, *args, **kwargs)

    monkeypatch.setattr(playoff_bracket, "compose_series_list", spy)
    nhl_playoffs.compose_playoffs_image(data_from(bracket_payload()), now=NOW)
    assert headings == ["Stanley Cup Final · Best of 7"] and rows == [["EDM", "FLA"]]


class _Display:
    def __init__(self):
        self.images = []

    def image(self, img):
        self.images.append(img)

    def clear(self):
        pass


def test_render_uses_given_feed_data_without_fetching(monkeypatch):
    monkeypatch.setattr(np_, "fetch_postseason", lambda **_: pytest.fail("fetched upstream"))
    monkeypatch.setattr(playoff_bracket, "scroll_vertical_content", lambda **kwargs: kwargs["render_at_offset"](0))
    monkeypatch.setattr(playoff_bracket.time, "sleep", lambda _s: None)
    display = _Display()
    result = nhl_playoffs.render_nhl_playoffs(display, data_from(bracket_payload()), transition=True)
    assert result.displayed and display.images


def test_render_fetches_in_standalone_mode(monkeypatch):
    monkeypatch.setattr(np_, "fetch_postseason", lambda **_: data_from(bracket_payload()))
    monkeypatch.setattr(playoff_bracket, "scroll_vertical_content", lambda **kwargs: kwargs["render_at_offset"](0))
    monkeypatch.setattr(playoff_bracket.time, "sleep", lambda _s: None)
    assert nhl_playoffs.render_nhl_playoffs(_Display()).image.height > HEIGHT


@pytest.mark.parametrize("screen_id,feed,attr", [
    ("NHL Playoffs", "nhl_playoffs", "render_nhl_playoffs"),
    ("NBA Playoffs", "nba_playoffs", "render_nba_playoffs"),
])
def test_registry_hands_the_server_feed_to_the_screen(monkeypatch, screen_id, feed, attr):
    from display_profiles import resolve_display_profile

    # Other tests drop screens.registry from sys.modules; use the live module.
    registry_module = importlib.import_module("screens.registry")
    received = []
    monkeypatch.setattr(registry_module, attr, lambda display, data, transition=False: received.append(data))
    value = {"series": [{"round": 1}]}
    context = registry_module.ScreenContext(
        display=_Display(), cache={feed: value}, logos={}, image_dir="", now=NOW,
        now_utc=NOW.astimezone(datetime.UTC), offline=False, weather_fetched_at=None, skip_scoreboards=False,
        render_profile=resolve_display_profile(320, 240), allow_upstream_requests=False,
    )
    registry, _ = registry_module.build_screen_registry(context)
    registry[screen_id].render()
    assert received == [value]


# ── Server feed ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("screen_id,feed", [("NHL Playoffs", "nhl_playoffs"), ("NBA Playoffs", "nba_playoffs")])
def test_playoff_screens_have_a_server_feed(screen_id, feed):
    assert feeds.feeds_for_screen(screen_id, feeds.SERVER_FEED_DEPENDENCIES) == {feed}
    assert feeds.SERVER_FEED_REFRESH_INTERVALS[feed] == 600


class _Clock:
    now = 1_000.0

    def __call__(self):
        return self.now


@pytest.mark.parametrize("screen_id,feed", [("NHL Playoffs", "nhl_playoffs"), ("NBA Playoffs", "nba_playoffs")])
def test_server_refreshes_the_bracket_faster_while_a_game_is_live(screen_id, feed):
    clock = _Clock()
    value = {"series": [{"round": 1, "live": False}]}

    def fetch(*, force=False):
        return {feed: json.loads(json.dumps(value))}

    service = ServerFeedService(
        DataCoordinator(SimpleNamespace()), SimpleNamespace(), fetch_air_quality=lambda *a, **k: None,
        settings=SimpleNamespace(ENABLE_WEATHER=False, ENABLE_AIR_QUALITY=False),
        standings_fetchers={feed: fetch}, history_path="/nonexistent/aq.json",
        clock=clock, wall_clock=clock,
    )
    assert service.refresh({screen_id}) == {feed: True}
    clock.now += LIVE_REFRESH_SECONDS
    assert service.refresh({screen_id}) == {}
    value["series"][0]["live"] = True
    clock.now += feeds.SERVER_FEED_REFRESH_INTERVALS[feed]
    assert service.refresh({screen_id}) == {feed: True}
    clock.now += LIVE_REFRESH_SECONDS
    assert service.refresh({screen_id}) == {feed: True}


def test_default_fetchers_cover_every_bracket_feed(monkeypatch):
    from services import server_feeds

    fetchers = server_feeds._default_standings_fetchers()
    monkeypatch.setattr(np_, "fetch_postseason", lambda **_: {"series": [1]})
    assert fetchers["nhl_playoffs"](force=False) == {"nhl_playoffs": {"series": [1]}}
    assert set(feeds.POSTSEASON_FEED_MODULES) <= set(fetchers)


def test_server_renders_the_screen_from_its_snapshot(monkeypatch):
    from display_profiles import DISPLAY_PROFILE_DISPLAY_HAT_MINI, PROFILE_PRESETS
    from rendering.screen_renderer import ScreenRenderer, ServerPreferenceSnapshot

    monkeypatch.setattr(np_, "fetch_postseason", lambda **_: pytest.fail("fetched upstream"))
    monkeypatch.setattr(playoff_bracket.time, "sleep", lambda _s: None)
    monkeypatch.setattr(playoff_bracket, "scroll_vertical_content", lambda **kwargs: kwargs["render_at_offset"](0))
    data = data_from(first_round(), schedule_payload([("TBL", "FLA", "LIVE", "2024-04-22T23:00:00Z", 3)]))
    snapshot = DataCoordinator().publish("nhl_playoffs", data)
    artifact = ScreenRenderer().render(
        "NHL Playoffs", PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI], ServerPreferenceSnapshot(revision=1), snapshot,
    )
    assert artifact.image.size == (320, 240)
    assert pb.has_live_series(snapshot["nhl_playoffs"])
