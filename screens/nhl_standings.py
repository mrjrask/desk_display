#!/usr/bin/env python3
"""Render NHL standings screens for Western and Eastern conferences."""

from __future__ import annotations

import logging
import os
import socket
import time
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from config import (
    WIDTH,
    HEIGHT,
    FONT_TITLE_SPORTS,
    FONT_STATUS,
    NHL_IMAGES_DIR,
    SCOREBOARD_SCROLL_STEP,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
    SCOREBOARD_BACKGROUND_COLOR,
)
from services.http_client import NHL_HEADERS, get_session
from utils import ScreenImage, clear_display, clone_font, log_call

# ─── Constants ────────────────────────────────────────────────────────────────
TITLE_WEST = "Western Conference"
TITLE_EAST = "Eastern Conference"
STANDINGS_URL = "https://statsapi.web.nhl.com/api/v1/standings"
API_WEB_STANDINGS_URL = "https://api-web.nhle.com/v1/standings/now"
API_WEB_STANDINGS_PARAMS = {"site": "en_nhl"}
REQUEST_TIMEOUT = 10
CACHE_TTL = 15 * 60  # seconds

CONFERENCE_WEST_KEY = "Western"
CONFERENCE_EAST_KEY = "Eastern"

LOGO_DIR = NHL_IMAGES_DIR
LOGO_HEIGHT = 41  # ~10% larger logos for standings rows
LEFT_MARGIN = 4
RIGHT_MARGIN = 6
TEAM_COLUMN_GAP = 6
STATS_FIRST_COLUMN_GAP = 24
STATS_COLUMN_MIN_STEP = 36
ROW_PADDING = 2
ROW_SPACING = 2
SECTION_GAP = 10
TITLE_MARGIN_TOP = 4
TITLE_MARGIN_BOTTOM = 6
DIVISION_MARGIN_TOP = 4
DIVISION_MARGIN_BOTTOM = 4
COLUMN_GAP_BELOW = 3
DIVISION_HEADER_GAP = 6

TITLE_FONT = FONT_TITLE_SPORTS
DIVISION_FONT = clone_font(FONT_TITLE_SPORTS, 26)
COLUMN_FONT = clone_font(FONT_STATUS, 24)
_COLUMN_POINTS_SIZE = max(8, getattr(COLUMN_FONT, "size", 24) - 6)
COLUMN_FONT_POINTS = clone_font(COLUMN_FONT, _COLUMN_POINTS_SIZE)
ROW_FONT = clone_font(FONT_STATUS, 28)
_TEAM_NAME_FONT_SIZE = max(8, getattr(ROW_FONT, "size", 28) - 3)
TEAM_NAME_FONT = clone_font(ROW_FONT, _TEAM_NAME_FONT_SIZE)

OVERVIEW_TITLE = "NHL Overview"
OVERVIEW_DIVISIONS = [
    (CONFERENCE_EAST_KEY, "Metropolitan", "Metro"),
    (CONFERENCE_EAST_KEY, "Atlantic", "Atlantic"),
    (CONFERENCE_WEST_KEY, "Central", "Central"),
    (CONFERENCE_WEST_KEY, "Pacific", "Pacific"),
]
OVERVIEW_MARGIN_X = 4
OVERVIEW_TITLE_MARGIN_BOTTOM = 6
OVERVIEW_BOTTOM_MARGIN = 2
OVERVIEW_MIN_LOGO_HEIGHT = 33
OVERVIEW_MAX_LOGO_HEIGHT = 67
OVERVIEW_LOGO_PADDING = 4
OVERVIEW_LOGO_OVERLAP = 6
BACKGROUND_COLOR = SCOREBOARD_BACKGROUND_COLOR
OVERVIEW_DROP_STEPS = 30
OVERVIEW_DROP_STAGGER = 0.4  # fraction of steps before next team starts
DROP_FRAME_DELAY = 0.02


WHITE = (255, 255, 255)

_SESSION = get_session()

_MEASURE_IMG = Image.new("RGB", (1, 1))
_MEASURE_DRAW = ImageDraw.Draw(_MEASURE_IMG)

_STANDINGS_CACHE: dict[str, object] = {"timestamp": 0.0, "data": None}
_LOGO_CACHE: dict[str, Optional[Image.Image]] = {}
_OVERVIEW_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}

STATSAPI_HOST = "statsapi.web.nhl.com"
_DNS_RETRY_INTERVAL = 600  # seconds
_dns_block_until = 0.0

DIVISION_ORDER_WEST = ["Central", "Pacific"]
DIVISION_ORDER_EAST = ["Metropolitan", "Atlantic"]
VALID_DIVISIONS = set(DIVISION_ORDER_WEST + DIVISION_ORDER_EAST)

STATS_COLUMNS = ("wins", "losses", "ot", "points")


TEAM_NICKNAMES = {
    "ANA": "Ducks",
    "BOS": "Bruins",
    "BUF": "Sabres",
    "CGY": "Flames",
    "CAR": "Hurricanes",
    "CHI": "Blackhawks",
    "COL": "Avalanche",
    "CBJ": "Blue Jackets",
    "DAL": "Stars",
    "DET": "Red Wings",
    "EDM": "Oilers",
    "FLA": "Panthers",
    "LAK": "Kings",
    "MIN": "Wild",
    "MTL": "Canadiens",
    "NJD": "Devils",
    "NSH": "Predators",
    "NYI": "Islanders",
    "NYR": "Rangers",
    "OTT": "Senators",
    "PHI": "Flyers",
    "PIT": "Penguins",
    "SEA": "Kraken",
    "SJS": "Sharks",
    "STL": "Blues",
    "TBL": "Lightning",
    "TOR": "Maple Leafs",
    "VAN": "Canucks",
    "UTA": "Mammoth",
    "VGK": "Knights",
    "WSH": "Capitals",
    "WPG": "Jets",
}


def _build_column_layout(max_team_name_width: int) -> tuple[dict[str, int], int]:
    team_x = LEFT_MARGIN + LOGO_HEIGHT + TEAM_COLUMN_GAP
    stats_right = WIDTH - RIGHT_MARGIN

    layout: dict[str, int] = {"team": team_x}
    if not STATS_COLUMNS:
        return layout, max(0, max_team_name_width)

    column_count = len(STATS_COLUMNS)
    min_spacing = STATS_COLUMN_MIN_STEP * max(0, column_count - 1)
    max_team_space = max(0, stats_right - team_x - STATS_FIRST_COLUMN_GAP - min_spacing)
    allowed_team_space = max(0, min(max_team_name_width, max_team_space))

    first_column = min(stats_right, team_x + allowed_team_space + STATS_FIRST_COLUMN_GAP)

    if column_count == 1:
        layout[STATS_COLUMNS[0]] = stats_right
    else:
        available_space = max(0.0, stats_right - first_column)
        step = available_space / (column_count - 1) if column_count > 1 else 0.0

        positions: list[int] = []
        for idx in range(column_count):
            if idx == column_count - 1:
                pos = stats_right
            else:
                pos = first_column + step * idx
            positions.append(int(round(pos)))

        for key, pos in zip(STATS_COLUMNS, positions):
            layout[key] = pos

    team_name_width = max(
        0,
        min(max_team_name_width, first_column - TEAM_COLUMN_GAP - team_x),
    )
    return layout, team_name_width

COLUMN_HEADERS = [
    ("", "team", "left"),
    ("W", "wins", "right"),
    ("L", "losses", "right"),
    ("O", "ot", "right"),
    ("PTS", "points", "right"),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _text_size(text: str, font) -> tuple[int, int]:
    try:
        l, t, r, b = _MEASURE_DRAW.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:
        return _MEASURE_DRAW.textsize(text, font)


ROW_TEXT_HEIGHT = _text_size("PTS", ROW_FONT)[1]
ROW_HEIGHT = max(LOGO_HEIGHT, ROW_TEXT_HEIGHT) + ROW_PADDING * 2
COLUMN_HEADER_FONTS = {"points": COLUMN_FONT_POINTS}

COLUMN_TEXT_HEIGHT = max(
    _text_size(label, COLUMN_HEADER_FONTS.get(key, COLUMN_FONT))[1]
    for label, key, _ in COLUMN_HEADERS
)
COLUMN_ROW_HEIGHT = COLUMN_TEXT_HEIGHT + 2
DIVISION_TEXT_HEIGHT = _text_size("Metropolitan", DIVISION_FONT)[1]


def _conference_column_layout(
    standings: Dict[str, List[dict]],
    divisions: Sequence[str],
) -> tuple[dict[str, int], int]:
    max_team_name_width = 0
    for division in divisions:
        for team in standings.get(division, []):
            team_label = _coerce_text(team.get("name")) or team.get("abbr", "")
            if team_label:
                max_team_name_width = max(
                    max_team_name_width, _text_size(team_label, TEAM_NAME_FONT)[0]
                )

    if max_team_name_width <= 0:
        max_team_name_width = _text_size("Team", TEAM_NAME_FONT)[0]

    return _build_column_layout(max_team_name_width)


def _load_logo_cached(abbr: str) -> Optional[Image.Image]:
    key = (abbr or "").strip()
    if not key:
        return None
    cache_key = key.upper()
    if cache_key in _LOGO_CACHE:
        return _LOGO_CACHE[cache_key]

    candidates = [cache_key, cache_key.lower(), cache_key.title()]
    for candidate in candidates:
        path = os.path.join(LOGO_DIR, f"{candidate}.png")
        if os.path.exists(path):
            logo = _load_logo(candidate)
            _LOGO_CACHE[cache_key] = logo
            return logo

    _LOGO_CACHE[cache_key] = None
    return None


def _load_overview_logo(abbr: str, height: int) -> Optional[Image.Image]:
    abbr_key = (abbr or "").strip().upper()
    if not abbr_key or height <= 0:
        return None

    cache_key = (abbr_key, height)
    if cache_key in _OVERVIEW_LOGO_CACHE:
        return _OVERVIEW_LOGO_CACHE[cache_key]

    try:
        from utils import load_team_logo

        logo = load_team_logo(LOGO_DIR, abbr_key, height=height)
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.debug("NHL overview logo load failed for %s@%s: %s", abbr_key, height, exc)
        logo = None

    _OVERVIEW_LOGO_CACHE[cache_key] = logo
    return logo


def _load_logo(abbr: str) -> Optional[Image.Image]:
    try:
        from utils import load_team_logo

        return load_team_logo(LOGO_DIR, abbr, height=LOGO_HEIGHT)
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.debug("NHL logo load failed for %s: %s", abbr, exc)
        return None


def _team_abbreviation(team: dict) -> str:
    if not isinstance(team, dict):
        return ""
    for key in ("abbreviation", "abbrev", "triCode", "teamCode"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    name = (team.get("teamName") or team.get("name") or "").strip()
    return name[:3].upper() if name else ""


def _resolve_team_nickname(text: str, abbr: str) -> str:
    cleaned = text.strip() if isinstance(text, str) else ""
    if not cleaned:
        return ""

    abbr_key = abbr.strip().upper() if isinstance(abbr, str) else ""
    if not abbr_key:
        return cleaned

    nickname = TEAM_NICKNAMES.get(abbr_key)
    if not nickname:
        return cleaned

    text_fold = cleaned.casefold()
    nickname_fold = nickname.casefold()
    if text_fold == nickname_fold or nickname_fold in text_fold:
        return nickname

    if cleaned.upper() == abbr_key:
        return nickname

    return ""


def _team_display_name(team: dict, abbr: str = "") -> str:
    if not isinstance(team, dict):
        nickname = TEAM_NICKNAMES.get(abbr.strip().upper()) if abbr else ""
        return nickname or ""

    keys = ("teamName", "teamNickname", "nickname", "teamCommonName", "clubName", "name")
    for key in keys:
        value = team.get(key)
        text = _coerce_text(value)
        if not text:
            continue
        resolved = _resolve_team_nickname(text, abbr)
        if resolved:
            return resolved

    if abbr:
        nickname = TEAM_NICKNAMES.get(abbr.strip().upper())
        if nickname:
            return nickname

    return ""


def _normalize_int(value) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _division_sort_key(team: dict) -> tuple[int, int, int, int, str]:
    points = _normalize_int(team.get("points"))
    wins = _normalize_int(team.get("wins"))
    ot = _normalize_int(team.get("ot"))
    rank = _normalize_int(team.get("_rank", 99)) or 99
    abbr = str(team.get("abbr", ""))
    # Sort by points (desc), wins (desc), overtime losses (asc), then fallback rank and abbr.
    return (-points, -wins, ot, rank, abbr)


def _normalize_conference_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    text = name.strip()
    if not text:
        return ""
    if text.lower().endswith("conference"):
        text = text[: -len("conference")].strip()
    lowered = text.lower()
    if lowered == "western":
        return CONFERENCE_WEST_KEY
    if lowered == "eastern":
        return CONFERENCE_EAST_KEY
    return text.title()


def _normalize_division_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    text = name.strip()
    if not text:
        return ""
    if text.lower().endswith("division"):
        text = text[: -len("division")].strip()
    if not text:
        return ""
    return text.title()


def _division_section_height(team_count: int) -> int:
    height = DIVISION_MARGIN_TOP + DIVISION_TEXT_HEIGHT
    height += COLUMN_ROW_HEIGHT + COLUMN_GAP_BELOW
    if team_count > 0:
        height += team_count * ROW_HEIGHT + max(0, team_count - 1) * ROW_SPACING
    height += DIVISION_MARGIN_BOTTOM
    return height


def _walk_nodes(payload: object) -> Iterable[dict]:
    """Yield every mapping from *payload* using an iterative DFS."""

    stack: list[object] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            yield current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def _fetch_standings_data() -> dict[str, dict[str, list[dict]]]:
    now = time.time()
    cached = _STANDINGS_CACHE.get("data")
    timestamp = float(_STANDINGS_CACHE.get("timestamp", 0.0))
    if cached and now - timestamp < CACHE_TTL:
        return cached  # type: ignore[return-value]

    standings: Optional[dict[str, dict[str, list[dict]]]] = None

    if _statsapi_available():
        standings = _fetch_standings_statsapi()
    else:
        logging.info("Using api-web NHL standings endpoint (statsapi DNS failure)")

    if not standings:
        standings = _fetch_standings_api_web()

    if standings:
        _STANDINGS_CACHE["timestamp"] = now
        _STANDINGS_CACHE["data"] = standings
        return standings

    return cached or {}


def _statsapi_available() -> bool:
    global _dns_block_until

    now = time.time()
    if now < _dns_block_until:
        return False

    try:
        socket.getaddrinfo(STATSAPI_HOST, None)
    except socket.gaierror as exc:
        logging.warning("NHL statsapi DNS lookup failed: %s", exc)
        _dns_block_until = now + _DNS_RETRY_INTERVAL
        return False
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.debug("Unexpected error checking NHL statsapi DNS: %s", exc)
    else:
        _dns_block_until = 0.0
        return True

    return True


def _fetch_standings_statsapi() -> Optional[dict[str, dict[str, list[dict]]]]:
    try:
        response = _SESSION.get(STANDINGS_URL, timeout=REQUEST_TIMEOUT, headers=NHL_HEADERS)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logging.error("Failed to fetch NHL standings: %s", exc)
        return None

    records = payload.get("records", []) if isinstance(payload, dict) else []
    conferences: dict[str, dict[str, list[dict]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        div = record.get("division", {}) or {}
        conf = record.get("conference", {}) or {}
        conf_name = _normalize_conference_name(conf.get("name"))
        div_name = _normalize_division_name(div.get("name"))
        if not conf_name or not div_name:
            continue
        teams = record.get("teamRecords", []) or []
        parsed: list[dict] = []
        for team_record in sorted(teams, key=lambda t: _normalize_int(t.get("divisionRank", 99))):
            if not isinstance(team_record, dict):
                continue
            team_info = team_record.get("team", {}) or {}
            abbr = _team_abbreviation(team_info)
            record_info = team_record.get("leagueRecord", {}) or {}
            parsed.append(
                {
                    "abbr": abbr,
                    "name": _team_display_name(team_info, abbr) or abbr,
                    "wins": _normalize_int(record_info.get("wins")),
                    "losses": _normalize_int(record_info.get("losses")),
                    "ot": _normalize_int(record_info.get("ot")),
                    "points": _normalize_int(team_record.get("points")),
                }
            )
        if parsed:
            conferences.setdefault(conf_name, {})[div_name] = parsed

    return conferences if conferences else None


def _parse_grouped_standings(groups: Iterable[dict]) -> dict[str, dict[str, list[dict]]]:
    conferences: dict[str, dict[str, list[dict]]] = {}

    for group in groups:
        if not isinstance(group, dict):
            continue
        rows = None
        for key in ("teamRecords", "rows", "standings", "standingsRows", "teams"):
            candidate = group.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            team_info = row.get("team") if isinstance(row.get("team"), dict) else {}
            conference_name = (
                _extract_from_candidates(row, ("conferenceName", "conference", "conferenceAbbrev", "conferenceId"))
                or _extract_from_candidates(team_info, ("conferenceName", "conference"))
            )
            division_name = (
                _extract_from_candidates(row, ("divisionName", "division", "divisionAbbrev", "divisionId"))
                or _extract_from_candidates(team_info, ("divisionName", "division"))
            )
            conference_name = _normalize_conference_name(conference_name)
            division_name = _normalize_division_name(division_name)
            if not conference_name or not division_name or division_name not in VALID_DIVISIONS:
                continue

            abbr = (
                _extract_from_candidates(row, ("teamAbbrev", "abbrev", "triCode", "teamTricode"))
                or _extract_from_candidates(team_info, ("abbrev", "triCode", "teamTricode"))
                or _team_abbreviation(team_info)
            )

            if not abbr:
                continue

            wins = _extract_stat(row, ("wins", "w"))
            losses = _extract_stat(row, ("losses", "l"))
            ot = _extract_stat(row, ("ot", "otLosses", "otl"))
            points = _extract_stat(row, ("points", "pts"))

            team_entry = {
                "abbr": abbr,
                "name": _team_display_name(team_info, abbr) or abbr,
                "wins": wins,
                "losses": losses,
                "ot": ot,
                "points": points,
                "_rank": _extract_rank(row),
            }

            divisions = conferences.setdefault(conference_name, {})
            divisions.setdefault(division_name, []).append(team_entry)

    return conferences


def _parse_generic_standings(payload: object) -> dict[str, dict[str, list[dict]]]:
    conferences: dict[str, dict[str, list[dict]]] = {}
    seen: set[tuple[str, str, str]] = set()

    for node in _walk_nodes(payload):
        team_info = {}
        for key in ("team", "teamRecord", "club", "clubInfo", "teamData"):
            candidate = node.get(key)
            if isinstance(candidate, dict):
                team_info = candidate
                break
        if not team_info and isinstance(node.get("teams"), dict):
            team_info = node.get("teams", {})  # type: ignore[assignment]

        conference_name = (
            _extract_from_candidates(node, ("conferenceName", "conference", "conferenceAbbrev", "conferenceId"))
            or _extract_from_candidates(team_info, ("conferenceName", "conference"))
        )
        division_name = (
            _extract_from_candidates(node, ("divisionName", "division", "divisionAbbrev", "divisionId"))
            or _extract_from_candidates(team_info, ("divisionName", "division"))
        )
        conference_name = _normalize_conference_name(conference_name)
        division_name = _normalize_division_name(division_name)
        if not conference_name or not division_name or division_name not in VALID_DIVISIONS:
            continue

        abbr = (
            _extract_from_candidates(node, ("teamAbbrev", "abbrev", "triCode", "teamTricode", "teamTriCode"))
            or _extract_from_candidates(team_info, ("teamAbbrev", "abbrev", "triCode", "teamTricode", "teamTriCode"))
            or _team_abbreviation(team_info)
        )
        if not abbr:
            continue

        wins = _extract_stat(node, ("wins", "w"))
        losses = _extract_stat(node, ("losses", "l"))
        ot = _extract_stat(node, ("ot", "otLosses", "otl"))
        points = _extract_stat(node, ("points", "pts"))

        key = (conference_name, division_name, abbr)
        if key in seen:
            continue
        seen.add(key)

        entry = {
            "abbr": abbr,
            "name": _team_display_name(team_info, abbr) or abbr,
            "wins": wins,
            "losses": losses,
            "ot": ot,
            "points": points,
            "_rank": _extract_rank(node),
        }

        conference = conferences.setdefault(conference_name, {})
        conference.setdefault(division_name, []).append(entry)

    return conferences


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
        return ""
    if isinstance(value, dict):
        for key in ("default", "en", "english", "abbr", "abbrev", "code", "name", "value"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return ""


def _extract_from_candidates(payload: dict, keys: Iterable[str]) -> str:
    for key in keys:
        if not isinstance(payload, dict):
            continue
        text = _coerce_text(payload.get(key))
        if text:
            return text
    return ""


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("value", "default", "amount", "num", "number", "statValue"):
            nested = value.get(key)
            result = _coerce_int(nested)
            if result is not None:
                return result
    return None


def _extract_stat(row: dict, names: Iterable[str]) -> int:
    name_candidates = [name.lower() for name in names]
    for key in names:
        value = row.get(key)
        result = _coerce_int(value)
        if result is not None:
            return result

    stats_iterables = [
        row.get("stats"),
        row.get("teamStats"),
        row.get("teamStatsLeaders"),
        row.get("splits"),
    ]
    for stats in stats_iterables:
        if not isinstance(stats, Iterable) or isinstance(stats, (str, bytes)):
            continue
        for stat in stats:
            if not isinstance(stat, dict):
                continue
            identifier = _coerce_text(stat.get("name")) or _coerce_text(stat.get("type"))
            abbreviation = _coerce_text(stat.get("abbr") or stat.get("abbreviation"))
            identifier = identifier.lower() if identifier else ""
            abbreviation = abbreviation.lower() if abbreviation else ""
            for candidate in name_candidates:
                if identifier == candidate or abbreviation == candidate:
                    result = _coerce_int(stat.get("value") or stat.get("statValue") or stat.get("amount"))
                    if result is not None:
                        return result
    return 0


def _extract_rank(row: dict) -> int:
    for key in (
        "divisionRank",
        "conferenceRank",
        "leagueRank",
        "rank",
        "sequence",
        "position",
        "order",
    ):
        value = _coerce_int(row.get(key))
        if value is not None and value > 0:
            return value
    return 99


def _fetch_standings_api_web() -> Optional[dict[str, dict[str, list[dict]]]]:
    try:
        response = _SESSION.get(
            API_WEB_STANDINGS_URL,
            timeout=REQUEST_TIMEOUT,
            headers=NHL_HEADERS,
            params=API_WEB_STANDINGS_PARAMS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logging.error("Failed to fetch NHL standings (api-web fallback): %s", exc)
        return None

    standings_payload: list = []
    if isinstance(payload, dict):
        for key in ("standings", "records", "groups"):
            value = payload.get(key)
            if isinstance(value, list):
                standings_payload = value
                break
        else:
            if isinstance(payload.get("rows"), list):
                standings_payload = [payload]
    elif isinstance(payload, list):
        standings_payload = payload

    conferences = _parse_grouped_standings(standings_payload)

    if not conferences and isinstance(payload, dict):
        alternative_groups: list = []
        for key in (
            "standingsByConference",
            "standingsByDivision",
            "standingsByType",
            "divisionStandings",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                alternative_groups.extend(value)
        if alternative_groups:
            conferences = _parse_grouped_standings(alternative_groups)

    if not conferences:
        conferences = _parse_generic_standings(payload)

    if not conferences:
        return None

    for conference in conferences.values():
        for teams in conference.values():
            teams.sort(key=_division_sort_key)
            for item in teams:
                item.pop("_rank", None)

    return conferences


def _draw_centered_text(draw: ImageDraw.ImageDraw, text: str, font, top: int) -> int:
    tw, th = _text_size(text, font)
    draw.text(((WIDTH - tw) // 2, top), text, font=font, fill=WHITE)
    return th


def _draw_text(draw: ImageDraw.ImageDraw, text: str, font, x: int, top: int, height: int, align: str) -> None:
    if not text:
        return
    tw, th = _text_size(text, font)
    y = top + (height - th) // 2
    if align == "right":
        draw.text((x - tw, y), text, font=font, fill=WHITE)
    else:
        draw.text((x, y), text, font=font, fill=WHITE)


def _truncate_text_to_width(text: str, font, max_width: int) -> str:
    if max_width <= 0 or not text:
        return text
    if _text_size(text, font)[0] <= max_width:
        return text
    ellipsis = "…"
    trimmed = text.strip()
    while trimmed and _text_size(trimmed + ellipsis, font)[0] > max_width:
        trimmed = trimmed[:-1].rstrip()
    return (trimmed + ellipsis) if trimmed else ellipsis


def _draw_division(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    top: int,
    title: str,
    teams: Iterable[dict],
    column_layout: dict[str, int],
    team_name_max_width: int,
) -> int:
    y = top + DIVISION_MARGIN_TOP
    y += _draw_centered_text(draw, title, DIVISION_FONT, y)
    y += DIVISION_HEADER_GAP
    header_top = y
    for label, key, align in COLUMN_HEADERS:
        font = COLUMN_HEADER_FONTS.get(key, COLUMN_FONT)
        if key not in column_layout:
            continue
        _draw_text(draw, label, font, column_layout[key], header_top, COLUMN_ROW_HEIGHT, align)
    y += COLUMN_ROW_HEIGHT + COLUMN_GAP_BELOW

    for team in teams:
        row_top = y
        abbr = team.get("abbr", "")
        logo = _load_logo_cached(abbr)
        if logo:
            logo_y = row_top + (ROW_HEIGHT - logo.height) // 2
            img.paste(logo, (LEFT_MARGIN, logo_y), logo)
        team_label = _coerce_text(team.get("name")) or abbr
        team_label = _truncate_text_to_width(
            team_label, TEAM_NAME_FONT, team_name_max_width
        )
        _draw_text(
            draw,
            team_label,
            TEAM_NAME_FONT,
            column_layout.get("team", LEFT_MARGIN + LOGO_HEIGHT + TEAM_COLUMN_GAP),
            row_top,
            ROW_HEIGHT,
            "left",
        )
        for key in STATS_COLUMNS:
            if key not in column_layout:
                continue
            _draw_text(
                draw,
                str(team.get(key, "")),
                ROW_FONT,
                column_layout[key],
                row_top,
                ROW_HEIGHT,
                "right",
            )
        y += ROW_HEIGHT + ROW_SPACING

    y -= ROW_SPACING
    y += DIVISION_MARGIN_BOTTOM
    return y


def _render_conference(
    title: str, division_order: List[str], standings: Dict[str, List[dict]]
) -> Image.Image:
    divisions = [division for division in division_order if standings.get(division)]
    if not divisions:
        divisions = division_order

    column_layout, team_name_max_width = _conference_column_layout(standings, divisions)

    total_height = TITLE_MARGIN_TOP + _text_size(title, TITLE_FONT)[1] + TITLE_MARGIN_BOTTOM
    for idx, division in enumerate(divisions):
        team_count = len(standings.get(division, []))
        total_height += _division_section_height(team_count)
        if idx < len(divisions) - 1:
            total_height += SECTION_GAP
    total_height = max(total_height, HEIGHT)

    img = Image.new("RGB", (WIDTH, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    y = TITLE_MARGIN_TOP
    y += _draw_centered_text(draw, title, TITLE_FONT, y)
    y += TITLE_MARGIN_BOTTOM

    for idx, division in enumerate(divisions):
        teams = standings.get(division, [])
        if not teams:
            continue
        y = _draw_division(
            img,
            draw,
            y,
            f"{division} Division",
            teams,
            column_layout,
            team_name_max_width,
        )
        if idx < len(divisions) - 1:
            y += SECTION_GAP

    return img


Placement = Tuple[str, Image.Image, int, int]


def _overview_layout(
    divisions: Sequence[tuple[str, List[dict]]]
) -> tuple[Image.Image, List[float], float, float, int, int]:
    base = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(base)

    y = TITLE_MARGIN_TOP
    y += _draw_centered_text(draw, OVERVIEW_TITLE, TITLE_FONT, y)
    y += OVERVIEW_TITLE_MARGIN_BOTTOM

    logos_top = y
    available_height = max(1.0, HEIGHT - logos_top - OVERVIEW_BOTTOM_MARGIN)

    max_rows = max((len(teams) for _, teams in divisions), default=0)
    if max_rows <= 0:
        max_rows = 1

    col_count = max(1, len(divisions))
    available_width = max(1.0, WIDTH - 2 * OVERVIEW_MARGIN_X)
    col_width = available_width / col_count
    col_centers = [OVERVIEW_MARGIN_X + col_width * (idx + 0.5) for idx in range(col_count)]

    cell_height = available_height / max_rows if max_rows else available_height
    logo_width_limit = max(6, int(col_width - OVERVIEW_LOGO_PADDING))
    logo_base_height = cell_height + OVERVIEW_LOGO_OVERLAP
    logo_target_height = int(
        min(
            OVERVIEW_MAX_LOGO_HEIGHT,
            max(OVERVIEW_MIN_LOGO_HEIGHT, logo_base_height),
            logo_width_limit,
        )
    )
    logo_target_height = max(6, logo_target_height)

    return base, col_centers, logos_top, cell_height, logo_target_height, max_rows


def _overview_logo_position(
    col_idx: int,
    row_idx: int,
    col_centers: Sequence[float],
    logos_top: float,
    cell_height: float,
    logo: Image.Image,
) -> tuple[int, int]:
    col_center = col_centers[col_idx]
    y_center = logos_top + cell_height * (row_idx + 0.5)
    x0 = int(col_center - logo.width / 2)
    y0 = int(y_center - logo.height / 2)
    return x0, y0


def _build_overview_rows(
    divisions: Sequence[tuple[str, List[dict]]],
    col_centers: Sequence[float],
    logos_top: float,
    cell_height: float,
    logo_height: int,
    max_rows: int,
) -> List[List[Placement]]:
    rows: List[List[Placement]] = [[] for _ in range(max_rows)]

    for col_idx, (_, teams) in enumerate(divisions):
        limited = teams[:max_rows]
        for row_idx, team in enumerate(limited):
            abbr = (team.get("abbr") or "").upper()
            if not abbr:
                continue
            logo = _load_overview_logo(abbr, logo_height)
            if not logo:
                continue
            x0, y0 = _overview_logo_position(col_idx, row_idx, col_centers, logos_top, cell_height, logo)
            rows[row_idx].append((abbr, logo, x0, y0))

    return rows


def _ensure_blackhawks_top_layer(canvas: Image.Image, placements: Sequence[Placement]) -> None:
    for abbr, logo, x0, y0 in placements:
        if logo and abbr.upper() == "CHI":
            canvas.paste(logo, (x0, y0), logo)


def _ease_out_cubic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    inv = 1.0 - t
    return 1.0 - inv * inv * inv


def _compose_overview_image(
    base: Image.Image, row_positions: Sequence[Sequence[Placement]]
) -> tuple[Image.Image, List[Placement]]:
    final = base.copy()
    placements: List[Placement] = []

    for row in reversed(row_positions):
        for placement in row:
            abbr, logo, x0, y0 = placement
            final.paste(logo, (x0, y0), logo)
            placements.append(placement)

    _ensure_blackhawks_top_layer(final, placements)
    return final, placements


def _animate_overview_drop(
    display, base: Image.Image, row_positions: Sequence[Sequence[Placement]]
) -> None:
    has_logos = any(row for row in row_positions)
    if not has_logos:
        return

    steps = max(2, OVERVIEW_DROP_STEPS)
    stagger = max(1, int(round(steps * OVERVIEW_DROP_STAGGER)))

    schedule: List[tuple[int, Sequence[Placement]]] = []
    start_step = 0
    for rank in range(len(row_positions) - 1, -1, -1):
        drops = row_positions[rank]
        if not drops:
            continue
        schedule.append((start_step, drops))
        start_step += stagger

    if not schedule:
        return

    total_duration = schedule[-1][0] + steps + 1
    placed: List[Placement] = []
    completed = [False] * len(schedule)

    for current_step in range(total_duration):
        for idx, (start, drops) in enumerate(schedule):
            if current_step >= start + steps and not completed[idx]:
                placed.extend(drops)
                completed[idx] = True

        frame = base.copy()
        dynamic: List[Placement] = []

        for abbr, logo, x0, y0 in placed:
            frame.paste(logo, (x0, y0), logo)

        for idx, (start, drops) in enumerate(schedule):
            progress = current_step - start
            if progress < 0 or progress >= steps:
                continue

            frac = progress / (steps - 1) if steps > 1 else 1.0
            eased = _ease_out_cubic(frac)
            for abbr, logo, x0, y_target in drops:
                start_y = -logo.height
                y_pos = int(start_y + (y_target - start_y) * eased)
                if y_pos > y_target:
                    y_pos = y_target
                frame.paste(logo, (x0, y_pos), logo)
                dynamic.append((abbr, logo, x0, y_pos))

        _ensure_blackhawks_top_layer(frame, [*placed, *dynamic])
        display.image(frame)
        if hasattr(display, "show"):
            display.show()
        time.sleep(DROP_FRAME_DELAY)


def _prepare_overview(divisions: List[tuple[str, List[dict]]]) -> tuple[Image.Image, List[List[Placement]]]:
    base, col_centers, logos_top, cell_height, logo_height, max_rows = _overview_layout(divisions)
    row_positions = _build_overview_rows(divisions, col_centers, logos_top, cell_height, logo_height, max_rows)
    return base, row_positions


def _render_empty(title: str) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    _draw_centered_text(draw, title, TITLE_FONT, 10)
    _draw_centered_text(draw, "No standings", ROW_FONT, HEIGHT // 2 - ROW_TEXT_HEIGHT // 2)
    return img


def _scroll_vertical(display, image: Image.Image) -> None:
    if image.height <= HEIGHT:
        display.image(image)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
        return

    max_offset = image.height - HEIGHT
    display.image(image.crop((0, 0, WIDTH, HEIGHT)))
    time.sleep(SCOREBOARD_SCROLL_PAUSE_TOP)

    for offset in range(
        SCOREBOARD_SCROLL_STEP, max_offset + 1, SCOREBOARD_SCROLL_STEP
    ):
        frame = image.crop((0, offset, WIDTH, offset + HEIGHT))
        display.image(frame)
        time.sleep(SCOREBOARD_SCROLL_DELAY)

    time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)


# ─── Public API ───────────────────────────────────────────────────────────────
@log_call
def draw_nhl_standings_overview(display, transition: bool = False) -> ScreenImage:
    standings_by_conf = _fetch_standings_data()

    divisions: List[tuple[str, List[dict]]] = []
    for conference_key, division_name, label in OVERVIEW_DIVISIONS:
        conference = standings_by_conf.get(conference_key, {})
        teams = conference.get(division_name, [])
        divisions.append((label, teams))

    if not any(teams for _, teams in divisions):
        clear_display(display)
        img = _render_empty(OVERVIEW_TITLE)
        if transition:
            return ScreenImage(img, displayed=False)
        display.image(img)
        return ScreenImage(img, displayed=True)

    base, row_positions = _prepare_overview(divisions)
    final_img, _ = _compose_overview_image(base, row_positions)

    clear_display(display)
    _animate_overview_drop(display, base, row_positions)
    display.image(final_img)
    if hasattr(display, "show"):
        display.show()

    return ScreenImage(final_img, displayed=True)


@log_call
def draw_nhl_standings_west(display, transition: bool = False) -> ScreenImage:
    standings_by_conf = _fetch_standings_data()
    conference = standings_by_conf.get(CONFERENCE_WEST_KEY, {})
    divisions = [d for d in DIVISION_ORDER_WEST if conference.get(d)]
    if not divisions:
        clear_display(display)
        img = _render_empty(TITLE_WEST)
        if transition:
            return ScreenImage(img, displayed=False)
        display.image(img)
        return ScreenImage(img, displayed=True)

    full_img = _render_conference(TITLE_WEST, divisions, conference)
    clear_display(display)
    _scroll_vertical(display, full_img)
    return ScreenImage(full_img, displayed=True)


@log_call
def draw_nhl_standings_east(display, transition: bool = False) -> ScreenImage:
    standings_by_conf = _fetch_standings_data()
    conference = standings_by_conf.get(CONFERENCE_EAST_KEY, {})
    divisions = [d for d in DIVISION_ORDER_EAST if conference.get(d)]
    if not divisions:
        clear_display(display)
        img = _render_empty(TITLE_EAST)
        if transition:
            return ScreenImage(img, displayed=False)
        display.image(img)
        return ScreenImage(img, displayed=True)

    full_img = _render_conference(TITLE_EAST, divisions, conference)
    clear_display(display)
    _scroll_vertical(display, full_img)
    return ScreenImage(full_img, displayed=True)


if __name__ == "__main__":  # pragma: no cover
    from utils import Display

    disp = Display()
    try:
        draw_nhl_standings_west(disp)
        draw_nhl_standings_east(disp)
    finally:
        clear_display(disp)
