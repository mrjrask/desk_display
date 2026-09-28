"""Tests for the NBA Playoffs screen and its bracket feed."""
from __future__ import annotations

import datetime

import pytest

from config import CENTRAL_TIME, HEIGHT, WIDTH
from screens import nba_playoffs, playoff_bracket
from services.sports import nba_postseason as nb
from services.sports import playoff_bracket16 as pb

NOW = datetime.datetime(2024, 4, 22, 12, 0, tzinfo=CENTRAL_TIME)
_TEAM_IDS: dict[str, int] = {}

# The 2024 playoffs: (round, conference, high, high seed, low, low seed, high wins, low wins).
SERIES_2024 = [
    (1, "East", "BOS", 1, "MIA", 8, 4, 1), (1, "East", "CLE", 4, "ORL", 5, 4, 3),
    (1, "East", "MIL", 3, "IND", 6, 2, 4), (1, "East", "NYK", 2, "PHI", 7, 4, 2),
    (1, "West", "OKC", 1, "NOP", 8, 4, 0), (1, "West", "LAC", 4, "DAL", 5, 2, 4),
    (1, "West", "MIN", 3, "PHX", 6, 4, 0), (1, "West", "DEN", 2, "LAL", 7, 4, 1),
    (2, "East", "BOS", 1, "CLE", 4, 4, 1), (2, "East", "NYK", 2, "IND", 6, 3, 4),
    (2, "West", "OKC", 1, "DAL", 5, 2, 4), (2, "West", "DEN", 2, "MIN", 3, 3, 4),
    (3, "East", "BOS", 1, "IND", 6, 4, 0), (3, "West", "MIN", 3, "DAL", 5, 1, 4),
    (4, "NBA Finals", "BOS", 1, "DAL", 5, 4, 1),
]


def _id(code):
    return _TEAM_IDS.setdefault(code, 1610612700 + len(_TEAM_IDS))


def bracket_payload(series=SERIES_2024, next_games=None):
    """The live bracket JSON; *next_games* maps a high seed's tricode to ``(status, iso, number)``."""

    items = []
    for round_number, conference, high, high_rank, low, low_rank, high_wins, low_wins in series:
        item = {
            "roundNumber": round_number, "seriesConference": conference,
            "highSeedId": _id(high), "highSeedTricode": high, "highSeedName": f"{high} name",
            "highSeedRank": high_rank, "highSeedSeriesWins": high_wins,
            "lowSeedId": _id(low), "lowSeedTricode": low, "lowSeedName": f"{low} name",
            "lowSeedRank": low_rank, "lowSeedSeriesWins": low_wins,
            "seriesWinner": _id(high) if high_wins == 4 else (_id(low) if low_wins == 4 else 0),
            "nextGameStatus": 3,
        }
        if next_games and high in next_games:
            status, start, number = next_games[high]
            item.update(nextGameStatus=status, nextGameDateTimeUTC=start, nextGameNumber=number)
        items.append(item)
    return {"meta": {}, "bracket": {"playoffBracketSeries": items}}


def first_round(next_games=None):
    series = [(r, c, h, hr, lo, lr, 1, 1) for r, c, h, hr, lo, lr, _hw, _lw in SERIES_2024 if r == 1]
    return bracket_payload(series, next_games)


def data_from(payload):
    series, names, seeds = nb.series_from_bracket(payload)
    return {"series": series, "names": names, "seeds": seeds}


def test_first_round_follows_the_seed_order():
    bracket = pb.build_bracket(data_from(bracket_payload()))
    assert [slot["teams"] for slot in bracket["right"][0]] == [
        ["BOS", "MIA"], ["CLE", "ORL"], ["MIL", "IND"], ["NY", "PHI"]]
    assert [slot["teams"] for slot in bracket["left"][0]] == [
        ["OKC", "NO"], ["LAC", "DAL"], ["MIN", "PHX"], ["DEN", "LAL"]]
    assert bracket["left"][1][0]["teams"] == ["OKC", "DAL"] and bracket["left"][1][0]["winner"] == "DAL"
    assert bracket["right"][1][1]["teams"] == ["IND", "NY"]  # upper feeder first
    assert bracket["center"]["teams"] == ["DAL", "BOS"] and bracket["center"]["winner"] == "BOS"
    assert bracket["seeds"]["NY"] == 2 and bracket["seeds"]["NO"] == 8


def test_tricodes_use_the_logo_file_names():
    assert [nb.team_code(code) for code in ("NYK", "GSW", "SAS", "BKN", "NOP", "WAS", "CHI")] == [
        "NY", "GS", "SA", "BRK", "NO", "WSH", "CHI"]


def test_play_in_games_are_left_out():
    payload = bracket_payload([(0, "East", "PHI", 7, "MIA", 8, 1, 0), *SERIES_2024[:1]])
    assert [s["teams"] for s in data_from(payload)["series"]] == [["BOS", "MIA"]]


def test_live_and_next_games():
    data = data_from(first_round({"BOS": (2, "2024-04-22T23:00:00Z", 3), "CLE": (1, "2024-04-23T00:00:00Z", 3)}))
    by_high = {s["teams"][0]: s for s in data["series"]}
    assert by_high["BOS"]["live"] and by_high["BOS"]["next_start"] is None
    assert by_high["CLE"]["next_start"] == "2024-04-23T00:00:00Z" and by_high["CLE"]["next_game"] == 3
    assert pb.has_live_series(data)
    east = pb.build_bracket(data)["right"][0]
    status = [playoff_bracket.series_status(slot, data["names"], NOW)[0] for slot in east[:3]]
    assert status == ["Game 3 · LIVE", "Game 3 · Tonight 7 PM", "Series tied 1-1"]


def test_current_round_is_the_earliest_unfinished_one():
    series = [s for s in SERIES_2024 if s[0] <= 2]
    series[-1] = (2, "West", "DEN", 2, "MIN", 3, 3, 3)
    data = data_from(bracket_payload(series))
    assert pb.current_round(data) == 2
    assert pb.current_round(data_from(bracket_payload())) == 4


def test_fetch_tries_the_mirror_and_fails_when_neither_answers(monkeypatch):
    calls = []

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return bracket_payload()

    def get(url, **_kwargs):
        calls.append(url)
        if url == nb.BRACKET_URLS[0]:
            raise OSError("blocked")
        return _Response()

    monkeypatch.setattr(nb._SESSION, "get", get)
    assert len(nb.fetch_postseason(force=True)["series"]) == 15
    assert calls == list(nb.BRACKET_URLS)

    monkeypatch.setattr(nb._SESSION, "get", lambda url, **_k: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(RuntimeError):
        nb.fetch_postseason(force=True)


@pytest.mark.parametrize("data", [{}, data_from(bracket_payload()), data_from(first_round())])
def test_screen_image_fills_the_display_width(data):
    image = nba_playoffs.compose_playoffs_image(data, now=NOW)
    assert image.width == WIDTH and image.height >= HEIGHT


def test_render_fetches_in_standalone_mode(monkeypatch):
    class _Display:
        def __init__(self):
            self.images = []

        def image(self, img):
            self.images.append(img)

    monkeypatch.setattr(nb, "fetch_postseason", lambda **_: data_from(bracket_payload()))
    monkeypatch.setattr(playoff_bracket, "scroll_vertical_content", lambda **kwargs: kwargs["render_at_offset"](0))
    monkeypatch.setattr(playoff_bracket.time, "sleep", lambda _s: None)
    display = _Display()
    assert nba_playoffs.render_nba_playoffs(display).displayed and display.images
