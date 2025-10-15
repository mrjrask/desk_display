#!/usr/bin/env python3
"""Render NHL standings screens for Western and Eastern conferences."""

from __future__ import annotations

import logging
import os
import socket
import time
from collections.abc import Iterable
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw

from config import (
    WIDTH,
    HEIGHT,
    FONT_TITLE_SPORTS,
    FONT_STATUS,
    NHL_IMAGES_DIR,
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
LOGO_HEIGHT = 20
LEFT_MARGIN = 4
ROW_PADDING = 2
ROW_SPACING = 2
SECTION_GAP = 10
TITLE_MARGIN_TOP = 4
TITLE_MARGIN_BOTTOM = 6
DIVISION_MARGIN_TOP = 4
DIVISION_MARGIN_BOTTOM = 4
COLUMN_GAP_BELOW = 3
SCROLL_STEP = 1
SCROLL_DELAY = 0.04
SCROLL_PAUSE_TOP = 0.75
SCROLL_PAUSE_BOTTOM = 0.5

TITLE_FONT = FONT_TITLE_SPORTS
DIVISION_FONT = clone_font(FONT_TITLE_SPORTS, 14)
COLUMN_FONT = clone_font(FONT_STATUS, 13)
COLUMN_FONT_POINTS = clone_font(FONT_STATUS, 9)
ROW_FONT = clone_font(FONT_STATUS, 14)
OVERVIEW_LABEL_FONT = clone_font(FONT_STATUS, 10)

OVERVIEW_TITLE = "NHL Standings Overview"
OVERVIEW_DIVISIONS = [
    (CONFERENCE_EAST_KEY, "Metropolitan", "Metro"),
    (CONFERENCE_EAST_KEY, "Atlantic", "Atlantic"),
    (CONFERENCE_WEST_KEY, "Central", "Central"),
    (CONFERENCE_WEST_KEY, "Pacific", "Pacific"),
]
OVERVIEW_MARGIN_X = 4
OVERVIEW_TITLE_MARGIN_BOTTOM = 4
OVERVIEW_LABEL_GAP = 2
OVERVIEW_BOTTOM_MARGIN = 2
OVERVIEW_MAX_LOGO_HEIGHT = 24
OVERVIEW_LOGO_PADDING = 2


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

COLUMN_LAYOUT = {
    "team": LEFT_MARGIN + LOGO_HEIGHT + 4,
    "wins": 72,
    "losses": 88,
    "ot": 104,
    "points": WIDTH - LEFT_MARGIN,
}
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


def _draw_division(img: Image.Image, draw: ImageDraw.ImageDraw, top: int, title: str, teams: Iterable[dict]) -> int:
    y = top + DIVISION_MARGIN_TOP
    y += _draw_centered_text(draw, title, DIVISION_FONT, y)
    y += 2
    header_top = y
    for label, key, align in COLUMN_HEADERS:
        font = COLUMN_HEADER_FONTS.get(key, COLUMN_FONT)
        _draw_text(draw, label, font, COLUMN_LAYOUT[key], header_top, COLUMN_ROW_HEIGHT, align)
    y += COLUMN_ROW_HEIGHT + COLUMN_GAP_BELOW

    for team in teams:
        row_top = y
        abbr = team.get("abbr", "")
        logo = _load_logo_cached(abbr)
        if logo:
            logo_y = row_top + (ROW_HEIGHT - logo.height) // 2
            img.paste(logo, (LEFT_MARGIN, logo_y), logo)
        _draw_text(draw, abbr, ROW_FONT, COLUMN_LAYOUT["team"], row_top, ROW_HEIGHT, "left")
        _draw_text(draw, str(team.get("wins", "")), ROW_FONT, COLUMN_LAYOUT["wins"], row_top, ROW_HEIGHT, "right")
        _draw_text(draw, str(team.get("losses", "")), ROW_FONT, COLUMN_LAYOUT["losses"], row_top, ROW_HEIGHT, "right")
        _draw_text(draw, str(team.get("ot", "")), ROW_FONT, COLUMN_LAYOUT["ot"], row_top, ROW_HEIGHT, "right")
        _draw_text(draw, str(team.get("points", "")), ROW_FONT, COLUMN_LAYOUT["points"], row_top, ROW_HEIGHT, "right")
        y += ROW_HEIGHT + ROW_SPACING

    y -= ROW_SPACING
    y += DIVISION_MARGIN_BOTTOM
    return y


def _render_conference(title: str, division_order: List[str], standings: Dict[str, List[dict]]) -> Image.Image:
    total_height = TITLE_MARGIN_TOP + _text_size(title, TITLE_FONT)[1] + TITLE_MARGIN_BOTTOM
    for idx, division in enumerate(division_order):
        team_count = len(standings.get(division, []))
        total_height += _division_section_height(team_count)
        if idx < len(division_order) - 1:
            total_height += SECTION_GAP
    total_height = max(total_height, HEIGHT)

    img = Image.new("RGB", (WIDTH, total_height), "black")
    draw = ImageDraw.Draw(img)

    y = TITLE_MARGIN_TOP
    y += _draw_centered_text(draw, title, TITLE_FONT, y)
    y += TITLE_MARGIN_BOTTOM

    for idx, division in enumerate(division_order):
        teams = standings.get(division, [])
        if not teams:
            continue
        y = _draw_division(img, draw, y, f"{division} Division", teams)
        if idx < len(division_order) - 1:
            y += SECTION_GAP

    return img


def _render_overview(divisions: List[tuple[str, List[dict]]]) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    y = TITLE_MARGIN_TOP
    y += _draw_centered_text(draw, OVERVIEW_TITLE, TITLE_FONT, y)
    y += OVERVIEW_TITLE_MARGIN_BOTTOM

    label_height = _text_size("Metro", OVERVIEW_LABEL_FONT)[1]
    logos_top = y + label_height + OVERVIEW_LABEL_GAP
    available_height = max(0, HEIGHT - logos_top - OVERVIEW_BOTTOM_MARGIN)

    max_rows = max((len(teams) for _, teams in divisions), default=0)
    if max_rows <= 0:
        max_rows = 1

    col_count = max(1, len(divisions))
    available_width = max(1.0, WIDTH - 2 * OVERVIEW_MARGIN_X)
    col_width = available_width / col_count
    cell_height = available_height / max_rows if max_rows else available_height

    logo_from_height = max(6, int(cell_height) - OVERVIEW_LOGO_PADDING)
    logo_from_width = max(6, int(col_width) - OVERVIEW_LOGO_PADDING)
    logo_target_height = max(6, min(OVERVIEW_MAX_LOGO_HEIGHT, logo_from_height, logo_from_width))

    for idx, (label, teams) in enumerate(divisions):
        col_left = OVERVIEW_MARGIN_X + idx * col_width
        col_center = col_left + col_width / 2

        lw, lh = _text_size(label, OVERVIEW_LABEL_FONT)
        label_y = y + (label_height - lh) // 2 if label_height > 0 else y
        draw.text((int(col_center - lw / 2), int(label_y)), label, font=OVERVIEW_LABEL_FONT, fill=WHITE)

        for row in range(max_rows):
            y_center = logos_top + cell_height * (row + 0.5)
            logo = None
            if row < len(teams):
                abbr = teams[row].get("abbr") or ""
                logo = _load_overview_logo(abbr, logo_target_height)
            if not logo:
                continue
            x = int(col_center - logo.width / 2)
            y0 = int(y_center - logo.height / 2)
            img.paste(logo, (x, y0), logo)

    return img


def _render_empty(title: str) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)
    _draw_centered_text(draw, title, TITLE_FONT, 10)
    _draw_centered_text(draw, "No standings", ROW_FONT, HEIGHT // 2 - ROW_TEXT_HEIGHT // 2)
    return img


def _scroll_vertical(display, image: Image.Image) -> None:
    if image.height <= HEIGHT:
        display.image(image)
        time.sleep(SCROLL_PAUSE_BOTTOM)
        return

    max_offset = image.height - HEIGHT
    display.image(image.crop((0, 0, WIDTH, HEIGHT)))
    time.sleep(SCROLL_PAUSE_TOP)

    for offset in range(SCROLL_STEP, max_offset + 1, SCROLL_STEP):
        frame = image.crop((0, offset, WIDTH, offset + HEIGHT))
        display.image(frame)
        time.sleep(SCROLL_DELAY)

    time.sleep(SCROLL_PAUSE_BOTTOM)


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

    img = _render_overview(divisions)
    clear_display(display)
    display.image(img)
    return ScreenImage(img, displayed=True)


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
