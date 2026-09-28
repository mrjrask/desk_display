"""MLB postseason bracket data.

Fetches the season's postseason games (Wild Card, Division Series, League
Championship Series and World Series) from the MLB Stats API, groups them into
series, and seeds each league's six playoff teams from the regular-season
standings.  Before the postseason schedule is published the standings alone
give a projected bracket ("if the season ended today").

The result is plain JSON so the render server can publish it as the
``mlb_postseason`` feed and save it between restarts.
"""
from __future__ import annotations

import datetime
import logging
import threading
import time
from collections.abc import Mapping
from typing import Any, Optional

from config import CENTRAL_TIME
from services.http_client import get_session
from utils import get_mlb_tricode

REQUEST_TIMEOUT = 10
CACHE_TTL_SECONDS = 120
LIVE_CACHE_TTL_SECONDS = 30

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
SERIES_URL = "https://statsapi.mlb.com/api/v1/schedule/postseason/series"
STANDINGS_URL = "https://statsapi.mlb.com/api/v1/standings"

AL_LEAGUE_ID = 103
NL_LEAGUE_ID = 104
LEAGUE_BY_ID = {AL_LEAGUE_ID: "AL", NL_LEAGUE_ID: "NL"}
LEAGUES = ("AL", "NL")

# Round codes are the Stats API game types, in bracket order.
ROUNDS = ("F", "D", "L", "W")
ROUND_NAMES = {
    "F": "Wild Card Series",
    "D": "Division Series",
    "L": "League Championship Series",
    "W": "World Series",
}
DEFAULT_BEST_OF = {"F": 3, "D": 5, "L": 7, "W": 7}
_ROUND_FROM_DESCRIPTION = (
    ("wild card", "F"),
    ("division", "D"),
    ("championship", "L"),
    ("world series", "W"),
)

# Logo abbreviations (see utils.get_mlb_tricode) by league, for teams whose
# payload carries no league.
AL_TEAMS = frozenset(
    {"BAL", "BOS", "NYY", "TB", "TOR", "CLE", "DET", "KC", "MIN", "SOX", "HOU", "LAA", "ATH", "SEA", "TEX"}
)
NL_TEAMS = frozenset(
    {"ATL", "MIA", "NYM", "PHI", "WSH", "CUBS", "CIN", "MIL", "PIT", "STL", "ARI", "COL", "LAD", "SD", "SF"}
)

_SESSION = get_session()
_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def postseason_season(now: Optional[datetime.datetime] = None) -> int:
    """The season whose bracket to show: the previous one until March."""

    current = now or datetime.datetime.now(CENTRAL_TIME)
    return current.year if current.month >= 3 else current.year - 1


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def team_league(abbr: str, team: Any = None) -> str:
    if isinstance(team, dict):
        league = team.get("league")
        if isinstance(league, dict):
            found = LEAGUE_BY_ID.get(_as_int(league.get("id")) or 0)
            if found:
                return found
    if abbr in AL_TEAMS:
        return "AL"
    if abbr in NL_TEAMS:
        return "NL"
    return ""


def _team_abbr(team: Any) -> str:
    if not isinstance(team, dict):
        return ""
    abbr = get_mlb_tricode(team)
    if abbr in AL_TEAMS or abbr in NL_TEAMS:
        return abbr
    # Placeholders ("TBD", "AL Wild Card Series A Winner") are not teams.
    return ""


def _team_name(team: Any, abbr: str) -> str:
    if isinstance(team, dict):
        for key in ("teamName", "clubName"):
            value = team.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        name = team.get("name")
        if isinstance(name, str) and name.strip():
            parts = name.strip().split()
            if len(parts) > 2 and parts[-2] in {"Red", "White", "Blue"}:
                return " ".join(parts[-2:])
            return parts[-1]
    return abbr


def _round_code(game: dict) -> str:
    code = str(game.get("gameType") or "").strip().upper()
    if code in ROUNDS:
        return code
    description = str(game.get("seriesDescription") or "").strip().lower()
    for token, round_code in _ROUND_FROM_DESCRIPTION:
        if token in description:
            return round_code
    return ""


def _iter_games(payload: Any):
    """Every game object in a schedule or postseason-series payload."""

    if isinstance(payload, list):
        for item in payload:
            yield from _iter_games(item)
    elif isinstance(payload, dict):
        if "gamePk" in payload and isinstance(payload.get("teams"), dict):
            yield payload
            return
        for value in payload.values():
            if isinstance(value, (list, dict)):
                yield from _iter_games(value)


def _game_state(game: dict) -> str:
    """One of ``final``, ``live``, ``scheduled`` or ``void`` (postponed/cancelled)."""

    status = game.get("status") if isinstance(game.get("status"), dict) else {}
    detailed = str(status.get("detailedState") or "").strip().lower()
    abstract = str(status.get("abstractGameState") or "").strip().lower()
    if any(token in detailed for token in ("postponed", "cancelled", "canceled")):
        return "void"
    if abstract == "final" or detailed in {"final", "game over", "completed early"}:
        return "final"
    if abstract == "live":
        if detailed in {"warmup", "pre-game", "delayed start"}:
            return "scheduled"
        return "live"
    return "scheduled"


def _winner_side(game: dict) -> Optional[str]:
    teams = game.get("teams") or {}
    for side in ("away", "home"):
        entry = teams.get(side) or {}
        if entry.get("isWinner") is True:
            return side
    away = _as_int((teams.get("away") or {}).get("score"))
    home = _as_int((teams.get("home") or {}).get("score"))
    if away is None or home is None or away == home:
        return None
    return "away" if away > home else "home"


def _parse_iso(value: Any) -> Optional[datetime.datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


# ─── Series ───────────────────────────────────────────────────────────────────


def series_from_games(payload: Any) -> list[dict]:
    """Group postseason games into series with wins, live state and next game."""

    grouped: dict[tuple[str, frozenset], dict] = {}
    seen_games: set[Any] = set()
    for game in _iter_games(payload):
        pk = game.get("gamePk")
        if pk in seen_games:
            continue
        seen_games.add(pk)
        round_code = _round_code(game)
        if not round_code:
            continue
        teams = game.get("teams") or {}
        away_team = (teams.get("away") or {}).get("team")
        home_team = (teams.get("home") or {}).get("team")
        away = _team_abbr(away_team)
        home = _team_abbr(home_team)
        if not away or not home or away == home:
            continue
        key = (round_code, frozenset((away, home)))
        series = grouped.get(key)
        if series is None:
            series = {
                "round": round_code,
                "league": "",
                "best_of": 0,
                "teams": [],
                "names": {},
                "wins": {away: 0, home: 0},
                "live": False,
                "next_start": None,
                "next_time_tbd": False,
                "next_game": None,
                "_first": None,
                "_next_dt": None,
            }
            grouped[key] = series
        for abbr, team in ((away, away_team), (home, home_team)):
            series["names"].setdefault(abbr, _team_name(team, abbr))
        leagues = {team_league(away, away_team), team_league(home, home_team)}
        if round_code != "W" and len(leagues) == 1:
            series["league"] = leagues.pop()
        series["best_of"] = max(series["best_of"], _as_int(game.get("gamesInSeries")) or 0)

        number = _as_int(game.get("seriesGameNumber"))
        start = _parse_iso(game.get("gameDate"))
        order = (number if number is not None else 99, start or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc))
        if series["_first"] is None or order < series["_first"][0]:
            # Game 1's home team is the higher seed.
            series["_first"] = (order, home, away)

        state = _game_state(game)
        if state == "final":
            winner = _winner_side(game)
            if winner:
                series["wins"][away if winner == "away" else home] += 1
        elif state == "live":
            series["live"] = True
        elif state == "scheduled" and start is not None:
            if series["_next_dt"] is None or start < series["_next_dt"]:
                series["_next_dt"] = start
                series["next_start"] = start.isoformat()
                series["next_game"] = number
                status = game.get("status") if isinstance(game.get("status"), dict) else {}
                series["next_time_tbd"] = bool(status.get("startTimeTBD"))

    results = []
    for series in grouped.values():
        first = series.pop("_first")
        series.pop("_next_dt")
        series["teams"] = [first[1], first[2]]
        if not series["best_of"]:
            series["best_of"] = DEFAULT_BEST_OF[series["round"]]
        needed = series["best_of"] // 2 + 1
        series["winner"] = next((abbr for abbr, wins in series["wins"].items() if wins >= needed), None)
        if series["winner"]:
            series["live"] = False
            series["next_start"] = None
            series["next_game"] = None
        results.append(series)
    results.sort(key=lambda s: (ROUNDS.index(s["round"]), s["league"], s["teams"]))
    return results


# ─── Seeds ────────────────────────────────────────────────────────────────────


def seeds_from_standings(payload: Any) -> dict[str, dict[str, int]]:
    """Playoff seeds 1-6 per league: three division leaders, then three wild cards."""

    rows: dict[str, list[dict]] = {"AL": [], "NL": []}
    for record in (payload or {}).get("records") or []:
        if not isinstance(record, dict):
            continue
        league = LEAGUE_BY_ID.get(_as_int((record.get("league") or {}).get("id")) or 0)
        if not league:
            continue
        for team_record in record.get("teamRecords") or []:
            if not isinstance(team_record, dict):
                continue
            abbr = _team_abbr(team_record.get("team"))
            if not abbr:
                continue
            rows[league].append(
                {
                    "abbr": abbr,
                    "division": _as_int((record.get("division") or {}).get("id")) or 0,
                    "division_rank": _as_int(team_record.get("divisionRank")) or 99,
                    "league_rank": _as_int(team_record.get("leagueRank")) or 99,
                    "wild_card_rank": _as_int(team_record.get("wildCardRank")),
                    "pct": _as_float(team_record.get("winningPercentage")) or 0.0,
                }
            )

    seeds: dict[str, dict[str, int]] = {}
    for league, league_rows in rows.items():
        leaders: dict[int, dict] = {}
        for row in league_rows:
            current = leaders.get(row["division"])
            if row["division_rank"] == 1 and current is None:
                leaders[row["division"]] = row
        if len(leaders) != 3:
            continue
        by_rank = lambda row: (row["league_rank"], -row["pct"])  # noqa: E731
        winners = sorted(leaders.values(), key=by_rank)
        leader_abbrs = {row["abbr"] for row in winners}
        others = [row for row in league_rows if row["abbr"] not in leader_abbrs]
        ranked = [row for row in others if row["wild_card_rank"] is not None]
        if len(ranked) >= 3:
            wild_cards = sorted(ranked, key=lambda row: (row["wild_card_rank"], -row["pct"]))[:3]
        else:
            wild_cards = sorted(others, key=by_rank)[:3]
        if len(wild_cards) != 3:
            continue
        seeds[league] = {row["abbr"]: seed for seed, row in enumerate([*winners, *wild_cards], start=1)}
    return seeds


def _standings_names(payload: Any) -> dict[str, str]:
    names: dict[str, str] = {}
    for record in (payload or {}).get("records") or []:
        for team_record in (record or {}).get("teamRecords") or []:
            team = (team_record or {}).get("team")
            abbr = _team_abbr(team)
            if abbr:
                names[abbr] = _team_name(team, abbr)
    return names


# ─── Fetch ────────────────────────────────────────────────────────────────────


def _get_json(url: str, params: dict[str, Any]) -> Any:
    response = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _fetch_series(season: int) -> tuple[list[dict], bool]:
    """Return ``(series, fetched)``; ``fetched`` is False when every source failed."""

    sources = (
        (SCHEDULE_URL, {"sportId": 1, "season": season, "gameTypes": "F,D,L,W", "hydrate": "team"}),
        (SERIES_URL, {"sportId": 1, "season": season, "hydrate": "team"}),
    )
    fetched = False
    for url, params in sources:
        try:
            payload = _get_json(url, params)
        except Exception as exc:  # noqa: BLE001 - try the next source
            logging.debug("MLB postseason source failed (%s): %s", url, exc)
            continue
        fetched = True
        series = series_from_games(payload)
        if series:
            return series, True
    return [], fetched


def _fetch_standings(season: int) -> Any:
    params = {
        "leagueId": f"{AL_LEAGUE_ID},{NL_LEAGUE_ID}",
        "season": season,
        "standingsType": "regularSeason",
        "hydrate": "team",
    }
    try:
        return _get_json(STANDINGS_URL, params)
    except Exception as exc:  # noqa: BLE001 - seeds are optional
        logging.debug("MLB standings for postseason seeds failed: %s", exc)
        return None


def fetch_postseason(*, force: bool = False, now: Optional[datetime.datetime] = None) -> dict[str, Any]:
    """Return the bracket data; raises ``RuntimeError`` when nothing could be fetched."""

    with _CACHE_LOCK:
        cached = _CACHE.get("data")
        age = time.monotonic() - float(_CACHE.get("fetched_at", 0.0))
    if cached and not force:
        ttl = LIVE_CACHE_TTL_SECONDS if has_live_series(cached) else CACHE_TTL_SECONDS
        if age < ttl:
            return cached

    season = postseason_season(now)
    series, series_fetched = _fetch_series(season)
    standings = _fetch_standings(season)
    if not series_fetched and standings is None:
        raise RuntimeError("MLB postseason data could not be fetched")
    seeds = seeds_from_standings(standings) if standings else {}
    if not series and not seeds:
        raise RuntimeError("MLB postseason returned no series or standings")
    names = _standings_names(standings) if standings else {}
    for item in series:
        names.update(item.get("names") or {})
    data = {
        "season": season,
        "series": series,
        "seeds": seeds,
        "names": names,
    }
    with _CACHE_LOCK:
        _CACHE["data"] = data
        _CACHE["fetched_at"] = time.monotonic()
    return data


def has_live_series(data: Any) -> bool:
    # The render server's snapshot holds frozen mappings, not dicts.
    if not isinstance(data, Mapping):
        return False
    return any(isinstance(s, Mapping) and s.get("live") for s in data.get("series") or ())


# ─── Bracket ──────────────────────────────────────────────────────────────────


def _slot(top: Optional[str], bottom: Optional[str]) -> dict:
    return {
        "teams": [top, bottom],
        "wins": [None, None],
        "series": None,
        "winner": None,
    }


def _slot_winner(slot: dict) -> Optional[str]:
    return slot.get("winner")


def _fill_slots(slots: list[dict], candidates: list[dict], seeds: dict[str, int]) -> None:
    """Match actual series onto the expected slots of one round."""

    unmatched = list(candidates)
    open_slots = []
    for slot in slots:
        expected = {team for team in slot["teams"] if team}
        match = next((s for s in unmatched if expected & set(s["teams"])), None) if expected else None
        if match is None:
            open_slots.append(slot)
            continue
        unmatched.remove(match)
        _apply_series(slot, match, seeds)
    for slot, series in zip(open_slots, unmatched):
        _apply_series(slot, series, seeds)


def _apply_series(slot: dict, series: dict, seeds: dict[str, int]) -> None:
    teams = list(series["teams"])
    expected_top = slot["teams"][0]
    if expected_top and expected_top in teams:
        teams.sort(key=lambda team: team != expected_top)
    elif seeds and all(team in seeds for team in teams):
        teams.sort(key=lambda team: seeds[team])
    slot["teams"] = teams
    slot["wins"] = [series["wins"].get(team, 0) for team in teams]
    slot["series"] = series
    slot["winner"] = series.get("winner")


def build_bracket(data: Any) -> dict[str, Any]:
    """Lay the series out in bracket slots, mlb.com style.

    Each league has two Wild Card slots (4 v 5, then 3 v 6), two Division
    Series slots (1 v the 4/5 winner, 2 v the 3/6 winner) and one LCS slot;
    the World Series slot pairs the two pennant winners, AL first.  Slots the
    postseason has not reached show the teams the seeds and earlier winners
    put there, with no series score.
    """

    data = data if isinstance(data, dict) else {}
    series = [s for s in data.get("series") or [] if isinstance(s, dict)]
    all_seeds = data.get("seeds") or {}
    bracket: dict[str, Any] = {}
    for league in LEAGUES:
        seeds = all_seeds.get(league) or {}
        by_seed = {seed: abbr for abbr, seed in seeds.items()}
        wild_card = [_slot(by_seed.get(4), by_seed.get(5)), _slot(by_seed.get(3), by_seed.get(6))]
        _fill_slots(wild_card, [s for s in series if s["round"] == "F" and s.get("league") == league], seeds)
        division = [
            _slot(by_seed.get(1), _slot_winner(wild_card[0])),
            _slot(by_seed.get(2), _slot_winner(wild_card[1])),
        ]
        _fill_slots(division, [s for s in series if s["round"] == "D" and s.get("league") == league], seeds)
        championship = [_slot(_slot_winner(division[0]), _slot_winner(division[1]))]
        _fill_slots(championship, [s for s in series if s["round"] == "L" and s.get("league") == league], seeds)
        bracket[league] = {"F": wild_card, "D": division, "L": championship}

    world_series = [_slot(_slot_winner(bracket["AL"]["L"][0]), _slot_winner(bracket["NL"]["L"][0]))]
    _fill_slots(world_series, [s for s in series if s["round"] == "W"], {})
    world = world_series[0]
    if world["series"] is not None and team_league(world["teams"][0] or "") == "NL":
        world["teams"].reverse()
        world["wins"].reverse()
    bracket["W"] = world_series
    bracket["seeds"] = {abbr: seed for league in LEAGUES for abbr, seed in (all_seeds.get(league) or {}).items()}
    return bracket


def current_round(data: Any) -> Optional[str]:
    """The round to list under the bracket: the earliest one still being played."""

    series = [s for s in (data or {}).get("series") or [] if isinstance(s, dict)]
    rounds = [r for r in ROUNDS if any(s["round"] == r for s in series)]
    if not rounds:
        return None
    for round_code in rounds:
        if any(not s.get("winner") for s in series if s["round"] == round_code):
            return round_code
    return rounds[-1]


__all__ = [
    "LEAGUES",
    "ROUNDS",
    "ROUND_NAMES",
    "build_bracket",
    "current_round",
    "fetch_postseason",
    "has_live_series",
    "postseason_season",
    "seeds_from_standings",
    "series_from_games",
    "team_league",
]
