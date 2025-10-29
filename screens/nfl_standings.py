#!/usr/bin/env python3
"""Render NFL standings screens for the NFC and AFC conferences."""

from __future__ import annotations

import csv
import datetime
import io
import logging
import os
import re
import time
from collections.abc import Iterable
from typing import Any, Dict, Iterable as _Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw

from config import (
    WIDTH,
    HEIGHT,
    FONT_TITLE_SPORTS,
    FONT_STATUS,
    IMAGES_DIR,
    SCOREBOARD_SCROLL_STEP,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
    SCOREBOARD_BACKGROUND_COLOR,
)
from services.http_client import get_session
from utils import ScreenImage, clear_display, clone_font, load_team_logo, log_call

# ─── Constants ────────────────────────────────────────────────────────────────
TITLE_NFC = "NFC Standings"
TITLE_AFC = "AFC Standings"
STANDINGS_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/standings.csv"
REQUEST_TIMEOUT = 10
CACHE_TTL = 15 * 60  # seconds

OFFSEASON_START = (2, 15)  # Feb 15
OFFSEASON_END = (8, 1)  # Aug 1
FALLBACK_MESSAGE_OFFSEASON = "NFL standings return this fall"
FALLBACK_MESSAGE_UNAVAILABLE = "NFL standings unavailable"

CONFERENCE_NFC_KEY = "NFC"
CONFERENCE_AFC_KEY = "AFC"

LOGO_DIR = os.path.join(IMAGES_DIR, "nfl")
LOGO_HEIGHT = 45

# Overview animation geometry
OVERVIEW_LOGO_HEIGHT = 65
OVERVIEW_VERTICAL_STEP = 45
OVERVIEW_COLUMN_MARGIN = 2
OVERVIEW_DROP_MARGIN = 6
OVERVIEW_DROP_STEPS = 30
OVERVIEW_DROP_STAGGER = 0.4  # fraction of steps before next rank begins dropping
OVERVIEW_FRAME_DELAY = 0.02
OVERVIEW_PAUSE_END = 0.5

LEFT_MARGIN = 4
RIGHT_MARGIN = 6
BACKGROUND_COLOR = SCOREBOARD_BACKGROUND_COLOR
ROW_PADDING = 3
ROW_SPACING = 2
TITLE_MARGIN_TOP = 4
TITLE_MARGIN_BOTTOM = 6
DIVISION_MARGIN_TOP = 2
DIVISION_MARGIN_BOTTOM = 4
COLUMN_GAP_BELOW = 3
RECORD_COLUMN_SPACING = 10
TEAM_COLUMN_PADDING = 6

TITLE_FONT = FONT_TITLE_SPORTS
DIVISION_FONT = clone_font(FONT_TITLE_SPORTS, 28)
COLUMN_FONT = clone_font(FONT_STATUS, 26)
ROW_FONT = clone_font(FONT_STATUS, 30)

TEAM_NAMES_BY_ABBR: dict[str, str] = {
    "ARI": "Cardinals",
    "ATL": "Falcons",
    "BAL": "Ravens",
    "BUF": "Bills",
    "CAR": "Panthers",
    "CHI": "Bears",
    "CIN": "Bengals",
    "CLE": "Browns",
    "DAL": "Cowboys",
    "DEN": "Broncos",
    "DET": "Lions",
    "GB": "Packers",
    "HOU": "Texans",
    "IND": "Colts",
    "JAX": "Jaguars",
    "KC": "Chiefs",
    "LAC": "Chargers",
    "LA": "Rams",
    "LAR": "Rams",
    "LV": "Raiders",
    "OAK": "Raiders",
    "MIA": "Dolphins",
    "MIN": "Vikings",
    "NE": "Patriots",
    "NO": "Saints",
    "NYG": "Giants",
    "NYJ": "Jets",
    "PHI": "Eagles",
    "PIT": "Steelers",
    "SEA": "Seahawks",
    "SF": "49ers",
    "TB": "Buccaneers",
    "TEN": "Titans",
    "WAS": "Commanders",
    "WSH": "Commanders",
    "JAC": "Jaguars",
    "SD": "Chargers",
    "STL": "Rams",
}

WHITE = (255, 255, 255)

_SESSION = get_session()

_MEASURE_IMG = Image.new("RGB", (1, 1))
_MEASURE_DRAW = ImageDraw.Draw(_MEASURE_IMG)

_STANDINGS_CACHE: Dict[str, Any] = {"timestamp": 0.0, "data": None, "message": None}
_LOGO_CACHE: Dict[str, Optional[Image.Image]] = {}
_OVERVIEW_LOGO_CACHE: Dict[str, Optional[Image.Image]] = {}

_CONFERENCE_ALIASES = {
    "american football conference": CONFERENCE_AFC_KEY,
    "national football conference": CONFERENCE_NFC_KEY,
}

_DIRECTION_KEYWORDS = ("EAST", "WEST", "NORTH", "SOUTH")
_DIVISION_PATTERN = re.compile(r"\b(AFC|NFC)\s+(EAST|WEST|NORTH|SOUTH)\b", re.IGNORECASE)

DIVISION_ORDER_NFC = ["NFC North", "NFC East", "NFC South", "NFC West"]
DIVISION_ORDER_AFC = ["AFC North", "AFC East", "AFC South", "AFC West"]

COLUMN_HEADERS: List[tuple[str, str, str]] = [
    ("", "team", "left"),
    ("W", "wins", "right"),
    ("L", "losses", "right"),
    ("T", "ties", "right"),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _text_size(text: str, font) -> tuple[int, int]:
    try:
        l, t, r, b = _MEASURE_DRAW.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:  # pragma: no cover - PIL fallback
        return _MEASURE_DRAW.textsize(text, font)


if TEAM_NAMES_BY_ABBR:
    ROW_TEXT_HEIGHT = max(_text_size(name, ROW_FONT)[1] for name in TEAM_NAMES_BY_ABBR.values())
else:
    ROW_TEXT_HEIGHT = _text_size("CHI", ROW_FONT)[1]
ROW_HEIGHT = max(LOGO_HEIGHT, ROW_TEXT_HEIGHT) + ROW_PADDING * 2
COLUMN_TEXT_HEIGHT = max(_text_size(label, COLUMN_FONT)[1] for label, _, _ in COLUMN_HEADERS)
COLUMN_ROW_HEIGHT = COLUMN_TEXT_HEIGHT + 2
DIVISION_TEXT_HEIGHT = _text_size("NFC North", DIVISION_FONT)[1]
TITLE_TEXT_HEIGHT = _text_size(TITLE_NFC, TITLE_FONT)[1]


def _record_column_width(label: str, sample: str) -> int:
    label_width = _text_size(label, COLUMN_FONT)[0]
    value_width = _text_size(sample, ROW_FONT)[0]
    return max(label_width, value_width)


def _team_name_for_abbr(abbr: str, *, fallback: str = "") -> str:
    key = (abbr or "").strip().upper()
    if not key:
        return fallback
    if key in TEAM_NAMES_BY_ABBR:
        return TEAM_NAMES_BY_ABBR[key]
    if fallback:
        return fallback
    if key.isalpha():
        return key.title()
    return key


def _build_column_layout(team_names: Iterable[str] | None = None) -> dict[str, int]:
    """Compute dynamic column layout based on the configured WIDTH."""

    samples = {
        "wins": "17",
        "losses": "17",
        "ties": "9",
    }

    record_columns: List[Tuple[str, str, int]] = []
    for label, key, _ in COLUMN_HEADERS:
        if key == "team":
            continue

        sample = samples.get(key, "17")
        width = _record_column_width(label or sample, sample)
        record_columns.append((label, key, width))

    team_left = LEFT_MARGIN + LOGO_HEIGHT + TEAM_COLUMN_PADDING
    names = list(team_names) if team_names is not None else list(TEAM_NAMES_BY_ABBR.values())
    if names:
        team_sample_width = max(_text_size(name, ROW_FONT)[0] for name in names)
    else:
        team_sample_width = _text_size("Commanders", ROW_FONT)[0]
    min_gap = 8
    record_area_left = team_left + team_sample_width + min_gap
    record_area_right = WIDTH - RIGHT_MARGIN
    record_area_width = max(0, record_area_right - record_area_left)

    layout: dict[str, int] = {}

    record_count = len(record_columns)
    if record_count and record_area_width > 0:
        max_column_width = max(width for _, _, width in record_columns)
        slot_width = record_area_width / record_count
        if slot_width >= max_column_width:
            for idx, (_, key, _width) in enumerate(record_columns):
                right_edge = record_area_left + slot_width * (idx + 1)
                layout[key] = int(round(min(record_area_right, right_edge)))
        else:
            spacing = RECORD_COLUMN_SPACING
            x = WIDTH - RIGHT_MARGIN
            for _label, key, width in reversed(record_columns):
                layout[key] = x
                x -= width + spacing
    else:
        spacing = RECORD_COLUMN_SPACING
        x = WIDTH - RIGHT_MARGIN
        for _label, key, width in reversed(record_columns):
            layout[key] = x
            x -= width + spacing

    layout["team"] = team_left
    return layout


def _load_logo_for_height(
    abbr: str, height: int, cache: Dict[str, Optional[Image.Image]]
) -> Optional[Image.Image]:
    key = (abbr or "").strip()
    if not key:
        return None

    cache_key = key.upper()
    if cache_key in cache:
        return cache[cache_key]

    candidates = [cache_key, cache_key.lower(), cache_key.title()]
    for candidate in candidates:
        path = os.path.join(LOGO_DIR, f"{candidate}.png")
        if os.path.exists(path):
            try:
                logo = load_team_logo(LOGO_DIR, candidate, height=height)
            except Exception as exc:  # pragma: no cover - defensive guard
                logging.debug("NFL logo load failed for %s: %s", candidate, exc)
                logo = None
            cache[cache_key] = logo
            return logo

    cache[cache_key] = None
    return None


def _load_logo_cached(abbr: str) -> Optional[Image.Image]:
    return _load_logo_for_height(abbr, LOGO_HEIGHT, _LOGO_CACHE)


def _load_overview_logo(abbr: str) -> Optional[Image.Image]:
    return _load_logo_for_height(abbr, OVERVIEW_LOGO_HEIGHT, _OVERVIEW_LOGO_CACHE)


def _normalize_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, str) and not value.strip():
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _normalize_conference(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    text = name.strip()
    if not text:
        return ""
    lowered = text.lower()
    for alias, replacement in _CONFERENCE_ALIASES.items():
        if alias in lowered:
            text = re.sub(alias, replacement, text, flags=re.IGNORECASE)
            break
    upper = text.upper()
    if "AFC" in upper:
        return CONFERENCE_AFC_KEY
    if "NFC" in upper:
        return CONFERENCE_NFC_KEY
    return text.title()


def _normalize_division(name: Any, conference: str = "") -> str:
    if not isinstance(name, str):
        name = ""
    text = (name or "").strip()
    if not text and conference:
        return conference
    for alias, replacement in _CONFERENCE_ALIASES.items():
        text = re.sub(alias, replacement, text, flags=re.IGNORECASE)
    if text.lower().endswith("division"):
        text = text[: -len("division")].strip()
    parts = text.split()
    normalized: List[str] = []
    seen_conference = False
    for part in parts:
        upper = part.upper()
        if upper in {CONFERENCE_AFC_KEY, CONFERENCE_NFC_KEY}:
            normalized.append(upper)
            seen_conference = True
        else:
            normalized.append(part.title())
    if not normalized and conference:
        normalized = [conference]
    if conference and not seen_conference:
        normalized.insert(0, conference)
    return " ".join(normalized).strip()


def _extract_division_from_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    match = _DIVISION_PATTERN.search(text)
    if not match:
        return ""
    conference, direction = match.groups()
    return f"{conference.upper()} {direction.title()}"


def _target_season_year(today: Optional[datetime.date] = None) -> int:
    today = today or datetime.datetime.now().date()
    if today.month >= 8:
        return today.year
    return today.year - 1


def _build_standings_from_rows(rows: Iterable[dict], *, conference_key: str) -> Dict[str, List[dict]]:
    divisions: Dict[str, List[dict]] = {}
    for row in rows:
        division = _normalize_division(row.get("division"), conference_key)
        if not division:
            continue

        abbr = (row.get("team") or "").strip().upper()
        if not abbr:
            continue

        wins = _normalize_int(row.get("wins"))
        losses = _normalize_int(row.get("losses"))
        ties = _normalize_int(row.get("ties"))
        order = _normalize_int(row.get("div_rank"))
        name = _team_name_for_abbr(abbr)

        bucket = divisions.setdefault(division, [])
        entry = {
            "abbr": abbr,
            "name": name,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "order": order if order > 0 else len(bucket) + 1,
        }
        bucket.append(entry)

    for teams in divisions.values():
        teams.sort(
            key=lambda item: (
                item.get("order", 999),
                -item.get("wins", 0),
                item.get("losses", 0),
                -item.get("ties", 0),
                item.get("abbr", ""),
            )
        )

    return divisions


def _parse_csv_standings(text: str, season: int) -> Tuple[dict[str, dict[str, List[dict]]], Optional[int]]:
    reader = csv.DictReader(io.StringIO(text))
    rows = [row for row in reader if row]

    standings: dict[str, dict[str, List[dict]]] = {
        CONFERENCE_NFC_KEY: {},
        CONFERENCE_AFC_KEY: {},
    }

    used_season: Optional[int] = None
    for candidate in (season, season - 1):
        filtered = [row for row in rows if _normalize_int(row.get("season")) == candidate]
        if not filtered:
            continue

        used_season = candidate
        grouped: Dict[str, List[dict]] = {CONFERENCE_NFC_KEY: [], CONFERENCE_AFC_KEY: []}
        for row in filtered:
            conference = (row.get("conf") or "").strip().upper()
            if conference not in standings:
                continue
            grouped[conference].append(row)

        for conference, conference_rows in grouped.items():
            standings[conference] = _build_standings_from_rows(
                conference_rows,
                conference_key=conference,
            )
        break

    return standings, used_season


def _extract_groups_info(data: Any) -> tuple[str, str]:
    conference_name = ""
    division_name = ""
    stack: list[Any] = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            type_value = str(node.get("type") or node.get("typeId") or "").lower()
            label = _first_string(node, ("displayName", "name", "abbreviation", "shortName", "label"))
            if label:
                label = label.strip()
                division_guess = _extract_division_from_text(label)
                if division_guess and not division_name:
                    division_name = division_guess
                if "division" in type_value and not division_name:
                    division_name = label
                if "conference" in type_value and not conference_name:
                    conference_name = label
                if not conference_name:
                    conference_guess = _normalize_conference(label)
                    if conference_guess in {CONFERENCE_AFC_KEY, CONFERENCE_NFC_KEY}:
                        conference_name = conference_guess

            for key in ("parent", "children", "items", "leagues"):
                value = node.get(key)
                if isinstance(value, list):
                    stack.extend(value)
                elif isinstance(value, dict):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)

    return conference_name, division_name


def _stat_map(stats: Iterable[dict]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for stat in stats or []:
        if not isinstance(stat, dict):
            continue
        name = stat.get("name") or stat.get("abbreviation")
        if not isinstance(name, str):
            continue
        value = stat.get("value")
        if value is None:
            value = stat.get("displayValue")
        mapping[name.lower()] = value
    return mapping


def _first_string(source: Any, keys: _Iterable[str]) -> str:
    if not isinstance(source, dict):
        return ""
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_team_info(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None

    team = entry.get("team") if isinstance(entry.get("team"), dict) else {}
    if not isinstance(team, dict):
        team = {}

    nickname = ""
    for key in ("nickname", "name"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            nickname = value.strip()
            break
    if not nickname:
        display_name = team.get("displayName")
        if isinstance(display_name, str) and display_name.strip():
            parts = display_name.strip().split()
            if parts:
                nickname = parts[-1]

    abbr = team.get("abbreviation") or team.get("shortDisplayName") or team.get("displayName")
    if isinstance(abbr, str):
        abbr = abbr.strip().upper()
    else:
        abbr = ""
    if not abbr:
        name_source = nickname or team.get("displayName") or ""
        abbr = name_source[:3].upper() if isinstance(name_source, str) else ""
    if not abbr:
        return None

    team_name = nickname or _team_name_for_abbr(abbr)

    stats = _stat_map(entry.get("stats") or [])
    wins = _normalize_int(stats.get("wins") or stats.get("overallwins"))
    losses = _normalize_int(stats.get("losses") or stats.get("overalllosses"))
    ties = _normalize_int(stats.get("ties") or stats.get("overallties") or stats.get("draws"))
    rank = _normalize_int(stats.get("rank") or stats.get("overallrank") or stats.get("playoffseed"))

    conference_name = ""
    conference_value = entry.get("conference")
    if isinstance(conference_value, dict):
        conference_name = _first_string(conference_value, ("name", "displayName", "abbreviation"))
    elif isinstance(conference_value, str):
        conference_name = conference_value
    if not conference_name and isinstance(team.get("conference"), dict):
        conference_name = _first_string(team["conference"], ("name", "displayName", "abbreviation"))

    division_name = ""
    for key in ("division", "group"):
        value = entry.get(key)
        if isinstance(value, dict):
            division_name = _first_string(value, ("displayName", "name", "abbreviation"))
            if division_name:
                break
        elif isinstance(value, str) and value.strip():
            division_name = value.strip()
            break
    if not division_name and isinstance(team.get("division"), dict):
        division_name = _first_string(team["division"], ("displayName", "name", "abbreviation"))

    if not division_name or not conference_name:
        groups_data = team.get("groups")
        if isinstance(groups_data, (dict, list)):
            group_conf, group_div = _extract_groups_info(groups_data)
            if not division_name and group_div:
                division_name = group_div
            if not conference_name and group_conf:
                conference_name = group_conf
            if not conference_name and group_div:
                conference_name = group_div

    if not division_name:
        summary = team.get("standingSummary")
        division_guess = _extract_division_from_text(summary)
        if not division_guess and isinstance(summary, str) and " in " in summary.lower():
            division_guess = summary.split(" in ", 1)[1].strip()
        if division_guess:
            division_name = division_guess
            if not conference_name:
                conference_name = division_guess

    if not division_name:
        note = entry.get("note") if isinstance(entry.get("note"), dict) else None
        if isinstance(note, dict):
            for key in ("headline", "shortHeadline", "description", "detail"):
                division_guess = _extract_division_from_text(note.get(key))
                if division_guess:
                    division_name = division_guess
                    if not conference_name:
                        conference_name = division_guess
                    break

    return {
        "abbr": abbr,
        "name": team_name,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "rank": rank,
        "conference_name": conference_name,
        "division_name": division_name,
    }


def _collect_division_groups(data: Any) -> List[Tuple[str, str, List[dict]]]:
    groups: List[Tuple[str, str, List[dict]]] = []
    stack: List[Any] = [data]
    seen_nodes: set[int] = set()
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            node_id = id(node)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            standings = node.get("standings")
            if isinstance(standings, dict):
                node_label = _first_string(node, ("displayName", "name", "abbreviation", "label"))
                standings_label = _first_string(
                    standings, ("displayName", "name", "abbreviation", "label")
                )
                base_label = node_label or standings_label
                base_conf_hint = _normalize_conference(base_label) or _normalize_conference(
                    standings_label
                )

                entries = standings.get("entries")
                if isinstance(entries, list) and entries:
                    label = base_label or ""
                    if label:
                        upper = label.upper()
                        if any(direction in upper for direction in _DIRECTION_KEYWORDS):
                            conference_hint = base_conf_hint or _normalize_conference(label)
                            groups.append((conference_hint, label, entries))

                entries_by_group = standings.get("entriesByGroup")
                if isinstance(entries_by_group, list):
                    for group in entries_by_group:
                        if not isinstance(group, dict):
                            continue
                        group_entries = group.get("entries")
                        if not isinstance(group_entries, list) or not group_entries:
                            continue

                        group_label = _first_string(
                            group, ("displayName", "name", "abbreviation", "label")
                        )
                        group_info = group.get("group")
                        if isinstance(group_info, dict):
                            if not group_label:
                                group_label = _first_string(
                                    group_info,
                                    (
                                        "displayName",
                                        "name",
                                        "abbreviation",
                                        "shortName",
                                        "label",
                                    ),
                                )
                            parent_info = group_info.get("parent")
                        else:
                            parent_info = None

                        if not group_label:
                            group_label = base_label or ""

                        conference_hint = _normalize_conference(
                            _first_string(group, ("conference", "conferenceName"))
                        )
                        if not conference_hint and isinstance(group_info, dict):
                            conference_hint = _normalize_conference(
                                _first_string(
                                    group_info,
                                    (
                                        "conference",
                                        "conferenceName",
                                        "parentConference",
                                        "parentDisplayName",
                                        "parentName",
                                    ),
                                )
                            )
                        if not conference_hint and isinstance(parent_info, dict):
                            conference_hint = _normalize_conference(
                                _first_string(
                                    parent_info,
                                    (
                                        "displayName",
                                        "name",
                                        "abbreviation",
                                        "shortName",
                                        "label",
                                    ),
                                )
                            )
                        if not conference_hint:
                            conference_hint = _normalize_conference(group_label)
                        if not conference_hint:
                            conference_hint = base_conf_hint

                        groups.append((conference_hint or "", group_label or "", group_entries))

                stack.extend(value for value in standings.values() if isinstance(value, (dict, list)))

            stack.extend(value for value in node.values() if isinstance(value, (dict, list)))
        elif isinstance(node, list):
            stack.extend(item for item in node if isinstance(item, (dict, list)))
    return groups


def _extract_entries(payload: Any) -> List[dict]:
    """Return the first set of overall standings entries found in *payload*."""

    if isinstance(payload, dict):
        entries = payload.get("entries")
        if isinstance(entries, list):
            return entries  # type: ignore[return-value]

        grouped_entries = payload.get("entriesByGroup")
        if isinstance(grouped_entries, list):
            for group in grouped_entries:
                if isinstance(group, dict) and isinstance(group.get("entries"), list):
                    return group["entries"]  # type: ignore[return-value]

    stack: List[Any] = [payload]
    candidates: List[tuple[str, List[dict]]] = []
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            entries = node.get("entries")
            if isinstance(entries, list):
                label = str(node.get("name") or node.get("type") or node.get("displayName") or "").lower()
                candidates.append((label, entries))
            for value in node.values():
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)

    if not candidates:
        return []

    for label, entries in candidates:
        if "overall" in label or "league" in label:
            return entries
    return candidates[0][1]


def _find_all_team_entries(payload: Any) -> List[dict]:
    """Return every dict that looks like a standings entry within *payload*."""

    entries: List[dict] = []
    stack: List[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            team = node.get("team")
            stats = node.get("stats")
            if isinstance(team, dict) and isinstance(stats, list):
                entries.append(node)

            stack.extend(value for value in node.values() if isinstance(value, (dict, list)))
        elif isinstance(node, list):
            stack.extend(item for item in node if isinstance(item, (dict, list)))

    return entries


def _parse_standings(data: Any) -> dict[str, dict[str, List[dict]]]:
    standings: dict[str, dict[str, List[dict]]] = {
        CONFERENCE_NFC_KEY: {},
        CONFERENCE_AFC_KEY: {},
    }

    groups = _collect_division_groups(data)
    added_from_groups = False
    if groups:
        for conference_hint, label, entries in groups:
            for entry in entries:
                info = _extract_team_info(entry)
                if not info:
                    continue

                conference_name = (
                    _normalize_conference(conference_hint)
                    or _normalize_conference(label)
                    or _normalize_conference(info.get("conference_name"))
                    or _normalize_conference(info.get("division_name"))
                )
                if conference_name not in standings:
                    continue

                division_label = label or info.get("division_name") or ""
                division = _normalize_division(division_label, conference_name)
                if not division:
                    division = _normalize_division(info.get("division_name"), conference_name)
                if not division:
                    continue

                conference_bucket = standings[conference_name]
                division_bucket = conference_bucket.setdefault(division, [])
                order = info["rank"] if info["rank"] > 0 else len(division_bucket) + 1
                division_bucket.append(
                    {
                        "abbr": info["abbr"],
                        "name": info.get("name") or _team_name_for_abbr(info["abbr"]),
                        "wins": info["wins"],
                        "losses": info["losses"],
                        "ties": info["ties"],
                        "order": order,
                    }
                )
                added_from_groups = True

    if not added_from_groups:
        entries = _extract_entries(data)
        if not entries:
            entries = _find_all_team_entries(data)
        if not entries:
            logging.warning("NFL standings response missing entries")
            return standings

        seen: set[Tuple[str, str, str]] = set()
        for entry in entries:
            info = _extract_team_info(entry)
            if not info:
                continue

            conference_name = _normalize_conference(info.get("conference_name"))
            if not conference_name and info.get("division_name"):
                conference_name = _normalize_conference(info.get("division_name"))
            if conference_name not in standings:
                if conference_name:
                    logging.debug(
                        "NFL standings skipping team %s with unknown conference %s",
                        info["abbr"],
                        conference_name,
                    )
                continue

            division = _normalize_division(info.get("division_name"), conference_name)
            if not division:
                logging.debug("NFL standings skipping team %s without division", info["abbr"])
                continue

            conference_bucket = standings[conference_name]
            division_bucket = conference_bucket.setdefault(division, [])
            order = info["rank"] if info["rank"] > 0 else len(division_bucket) + 1
            key = (conference_name, division, info["abbr"])
            if key in seen:
                continue
            seen.add(key)
            division_bucket.append(
                {
                    "abbr": info["abbr"],
                    "name": info.get("name") or _team_name_for_abbr(info["abbr"]),
                    "wins": info["wins"],
                    "losses": info["losses"],
                    "ties": info["ties"],
                    "order": order,
                }
            )

    # Sort each division by rank fallback to record
    for conference in standings.values():
        for division, teams in conference.items():
            teams.sort(
                key=lambda item: (
                    item.get("order", 999),
                    -item.get("wins", 0),
                    item.get("losses", 0),
                    -item.get("ties", 0),
                    item.get("abbr", ""),
                )
            )

    return standings


def _in_offseason(today: Optional[datetime.date] = None) -> bool:
    today = today or datetime.datetime.now().date()
    start = datetime.date(today.year, *OFFSEASON_START)
    end = datetime.date(today.year, *OFFSEASON_END)
    return start <= today < end


def _fetch_standings_data() -> Tuple[dict[str, dict[str, List[dict]]], Optional[str]]:
    now = time.time()
    cached = _STANDINGS_CACHE.get("data")
    timestamp = float(_STANDINGS_CACHE.get("timestamp", 0.0))
    cached_message = _STANDINGS_CACHE.get("message")
    if cached and now - timestamp < CACHE_TTL:
        return cached, cached_message  # type: ignore[return-value]

    if _in_offseason():
        standings = {
            CONFERENCE_NFC_KEY: {},
            CONFERENCE_AFC_KEY: {},
        }
        _STANDINGS_CACHE["data"] = standings
        _STANDINGS_CACHE["timestamp"] = now
        _STANDINGS_CACHE["message"] = FALLBACK_MESSAGE_OFFSEASON
        logging.info("NFL standings offseason fallback engaged; suppressing data display")
        return standings, FALLBACK_MESSAGE_OFFSEASON

    try:
        response = _SESSION.get(STANDINGS_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload_text = response.text
    except Exception as exc:  # pragma: no cover - network guard
        logging.error("Failed to fetch NFL standings: %s", exc)
        if isinstance(cached, dict):
            _STANDINGS_CACHE["timestamp"] = now
            _STANDINGS_CACHE["message"] = cached_message or FALLBACK_MESSAGE_UNAVAILABLE
            return cached, _STANDINGS_CACHE["message"]  # type: ignore[return-value]
        standings = {
            CONFERENCE_NFC_KEY: {},
            CONFERENCE_AFC_KEY: {},
        }
        _STANDINGS_CACHE["data"] = standings
        _STANDINGS_CACHE["timestamp"] = now
        _STANDINGS_CACHE["message"] = FALLBACK_MESSAGE_UNAVAILABLE
        return standings, FALLBACK_MESSAGE_UNAVAILABLE

    target_season = _target_season_year()
    standings, used_season = _parse_csv_standings(payload_text, target_season)
    if used_season and used_season != target_season:
        logging.info(
            "NFL standings using fallback season %s instead of %s",
            used_season,
            target_season,
        )

    _STANDINGS_CACHE["data"] = standings
    _STANDINGS_CACHE["timestamp"] = now
    fallback_message = None
    if not any(standings.values()) or used_season is None:
        fallback_message = FALLBACK_MESSAGE_UNAVAILABLE
    _STANDINGS_CACHE["message"] = fallback_message
    return standings, fallback_message


def _division_section_height(team_count: int) -> int:
    height = DIVISION_MARGIN_TOP + DIVISION_TEXT_HEIGHT
    height += COLUMN_ROW_HEIGHT + COLUMN_GAP_BELOW
    if team_count > 0:
        height += team_count * ROW_HEIGHT + max(0, team_count - 1) * ROW_SPACING
    height += DIVISION_MARGIN_BOTTOM
    return height


def _render_conference(title: str, division_order: List[str], standings: Dict[str, List[dict]]) -> Image.Image:
    sections = [
        _division_section_height(len(standings.get(division, [])))
        for division in division_order
    ]
    content_height = sum(sections)
    total_height = max(
        HEIGHT,
        TITLE_MARGIN_TOP + TITLE_TEXT_HEIGHT + TITLE_MARGIN_BOTTOM + content_height,
    )
    img = Image.new("RGB", (WIDTH, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Title
    try:
        l, t, r, b = draw.textbbox((0, 0), title, font=TITLE_FONT)
        tw, th = r - l, b - t
        tx = (WIDTH - tw) // 2 - l
        ty = TITLE_MARGIN_TOP - t
    except Exception:  # pragma: no cover - PIL fallback
        tw, th = draw.textsize(title, font=TITLE_FONT)
        tx = (WIDTH - tw) // 2
        ty = TITLE_MARGIN_TOP
    draw.text((tx, ty), title, font=TITLE_FONT, fill=WHITE)

    y = TITLE_MARGIN_TOP + TITLE_TEXT_HEIGHT + TITLE_MARGIN_BOTTOM

    team_names: List[str] = []
    for division in division_order:
        for team in standings.get(division, []) or []:
            name = team.get("name") or _team_name_for_abbr(team.get("abbr", ""))
            if name:
                team_names.append(name)

    column_layout = _build_column_layout(team_names)

    for division, section_height in zip(division_order, sections):
        teams = standings.get(division, [])

        # Division header
        try:
            l, t, r, b = draw.textbbox((0, 0), division, font=DIVISION_FONT)
            tw, th = r - l, b - t
            tx = (WIDTH - tw) // 2 - l
            ty = y + DIVISION_MARGIN_TOP - t
        except Exception:  # pragma: no cover - PIL fallback
            tw, th = draw.textsize(division, font=DIVISION_FONT)
            tx = (WIDTH - tw) // 2
            ty = y + DIVISION_MARGIN_TOP
        draw.text((tx, ty), division, font=DIVISION_FONT, fill=WHITE)

        y_division_bottom = y + section_height
        row_y = y + DIVISION_MARGIN_TOP + DIVISION_TEXT_HEIGHT + COLUMN_GAP_BELOW

        # Column headers
        column_y = row_y
        for label, key, align in COLUMN_HEADERS:
            x = column_layout[key]
            if align == "right":
                try:
                    l, t, r, b = draw.textbbox((0, 0), label, font=COLUMN_FONT)
                    tw, th = r - l, b - t
                    tx = x - tw
                    ty = column_y - t
                except Exception:  # pragma: no cover - PIL fallback
                    tw, th = draw.textsize(label, font=COLUMN_FONT)
                    tx = x - tw
                    ty = column_y
            else:
                try:
                    l, t, r, b = draw.textbbox((0, 0), label, font=COLUMN_FONT)
                    tx = x - l
                    ty = column_y - t
                except Exception:  # pragma: no cover - PIL fallback
                    tx = x
                    ty = column_y
            draw.text((tx, ty), label, font=COLUMN_FONT, fill=WHITE)
        row_y += COLUMN_ROW_HEIGHT + ROW_SPACING

        # Team rows
        for team in teams:
            abbr = team.get("abbr", "")
            team_name = team.get("name") or _team_name_for_abbr(abbr)
            display_text = team_name or abbr
            wins = str(team.get("wins", 0))
            losses = str(team.get("losses", 0))
            ties = str(team.get("ties", 0))

            # Team name
            text_top = row_y + ROW_PADDING
            text_center = text_top
            try:
                l, t, r, b = draw.textbbox((0, 0), display_text, font=ROW_FONT)
                tw, th = r - l, b - t
                tx = column_layout["team"] - l
                ty = row_y + ROW_PADDING - t
                text_top = ty
                text_center = text_top + th / 2
            except Exception:  # pragma: no cover - PIL fallback
                tw, th = draw.textsize(display_text, font=ROW_FONT)
                tx = column_layout["team"]
                ty = row_y + ROW_PADDING
                text_top = ty
                text_center = text_top + th / 2

            # Logo
            logo = _load_logo_cached(abbr)
            if logo:
                logo_y = int(text_center - logo.height / 2)
                img.paste(logo, (LEFT_MARGIN, logo_y), logo)

            draw.text((tx, ty), display_text, font=ROW_FONT, fill=WHITE)

            # Record columns
            for value, key in ((wins, "wins"), (losses, "losses"), (ties, "ties")):
                x = column_layout[key]
                try:
                    l, t, r, b = draw.textbbox((0, 0), value, font=ROW_FONT)
                    tw, th = r - l, b - t
                    tx = x - tw
                    ty = row_y + ROW_PADDING - t
                except Exception:  # pragma: no cover - PIL fallback
                    tw, th = draw.textsize(value, font=ROW_FONT)
                    tx = x - tw
                    ty = row_y + ROW_PADDING
                draw.text((tx, ty), value, font=ROW_FONT, fill=WHITE)

            row_y += ROW_HEIGHT + ROW_SPACING

        y = y_division_bottom

    return img


def _overview_header_frame(title: str) -> Tuple[Image.Image, int]:
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    try:
        l, t, r, b = draw.textbbox((0, 0), title, font=TITLE_FONT)
        tw, th = r - l, b - t
        tx = (WIDTH - tw) // 2 - l
        ty = TITLE_MARGIN_TOP - t
    except Exception:  # pragma: no cover - PIL fallback
        tw, th = draw.textsize(title, font=TITLE_FONT)
        tx = (WIDTH - tw) // 2
        ty = TITLE_MARGIN_TOP
    draw.text((tx, ty), title, font=TITLE_FONT, fill=WHITE)
    content_top = TITLE_MARGIN_TOP + TITLE_TEXT_HEIGHT + TITLE_MARGIN_BOTTOM
    return img, content_top


def _paste_overview_logos(canvas: Image.Image, placements: Iterable[Dict[str, Any]]):
    ordered = sorted(
        (
            placement
            for placement in placements
            if placement and placement.get("logo") is not None
        ),
        key=lambda item: (
            1 if item.get("abbr", "").upper() == "CHI" else 0,
            -int(item.get("y", 0)),
        ),
    )
    for placement in ordered:
        logo = placement["logo"]
        x = int(placement.get("x", 0))
        y = int(placement.get("y", 0))
        canvas.paste(logo, (x, y), logo)


def _prepare_overview_columns(
    division_order: List[str],
    standings: Dict[str, List[dict]],
    content_top: int,
) -> Tuple[List[Dict[int, Optional[Dict[str, Any]]]], int]:
    column_count = max(1, len(division_order))
    column_width = WIDTH / column_count
    available_height = max(0, HEIGHT - content_top)

    columns: List[Dict[int, Optional[Dict[str, Any]]]] = []
    max_rows = 0

    for idx, division in enumerate(division_order):
        teams = standings.get(division, []) or []
        team_count = len(teams)
        max_rows = max(max_rows, team_count)

        stack_height = (
            OVERVIEW_LOGO_HEIGHT + (team_count - 1) * OVERVIEW_VERTICAL_STEP
            if team_count
            else OVERVIEW_LOGO_HEIGHT
        )
        top_offset = 0
        if available_height > stack_height:
            top_offset = (available_height - stack_height) // 2
        start_center = content_top + top_offset + OVERVIEW_LOGO_HEIGHT // 2
        col_center = int((idx + 0.5) * column_width)
        width_limit = max(0, int(column_width - 2 * OVERVIEW_COLUMN_MARGIN))

        column: Dict[int, Optional[Dict[str, Any]]] = {}
        for rank, team in enumerate(teams):
            abbr = team.get("abbr", "")
            logo_source = _load_overview_logo(abbr)
            if not logo_source:
                column[rank] = None
                continue

            logo = logo_source.copy()
            if width_limit and logo.width > width_limit:
                ratio = width_limit / float(logo.width)
                new_size = (
                    max(1, int(logo.width * ratio)),
                    max(1, int(logo.height * ratio)),
                )
                logo = logo.resize(new_size, Image.LANCZOS)

            center_y = start_center + rank * OVERVIEW_VERTICAL_STEP
            y_target = int(center_y - logo.height / 2)
            x_target = int(col_center - logo.width / 2)
            drop_start = min(-logo.height, content_top - logo.height - OVERVIEW_DROP_MARGIN)

            column[rank] = {
                "logo": logo,
                "x": x_target,
                "y": y_target,
                "abbr": abbr,
                "drop_start": drop_start,
            }

        columns.append(column)

    return columns, max_rows


def _ease_out_cubic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    inv = 1.0 - t
    return 1.0 - inv * inv * inv


def _render_overview(
    display,
    title: str,
    division_order: List[str],
    standings: Dict[str, List[dict]],
    transition: bool,
    fallback_message: Optional[str],
) -> ScreenImage:
    if not any(standings.get(division) for division in division_order):
        return _render_overview_fallback(display, title, fallback_message, transition)

    header, content_top = _overview_header_frame(title)
    columns, max_rows = _prepare_overview_columns(division_order, standings, content_top)

    if max_rows == 0:
        return _render_overview_fallback(display, title, fallback_message, transition)

    row_positions: List[List[Dict[str, Any]]] = []
    for rank in range(max_rows):
        row: List[Dict[str, Any]] = []
        for column in columns:
            placement = column.get(rank)
            if placement:
                row.append(placement)
        row_positions.append(row)

    steps = max(2, OVERVIEW_DROP_STEPS)
    stagger = max(1, int(round(steps * OVERVIEW_DROP_STAGGER)))

    schedule: List[Tuple[int, List[Dict[str, Any]]]] = []
    start_step = 0
    for rank in range(len(row_positions) - 1, -1, -1):
        drops = row_positions[rank]
        if not drops:
            continue
        schedule.append((start_step, drops))
        start_step += stagger

    if schedule:
        total_duration = schedule[-1][0] + steps + 1
        placed: List[Dict[str, Any]] = []
        completed = [False] * len(schedule)

        for current_step in range(total_duration):
            for idx, (start, drops) in enumerate(schedule):
                if current_step >= start + steps and not completed[idx]:
                    placed.extend(
                        {
                            "logo": placement["logo"],
                            "x": placement["x"],
                            "y": placement["y"],
                            "abbr": placement.get("abbr", ""),
                        }
                        for placement in drops
                    )
                    completed[idx] = True

            frame = header.copy()
            if placed:
                _paste_overview_logos(frame, placed)

            animated: List[Dict[str, Any]] = []
            for idx, (start, drops) in enumerate(schedule):
                progress = current_step - start
                if progress < 0 or progress >= steps:
                    continue

                frac = progress / (steps - 1) if steps > 1 else 1.0
                eased = _ease_out_cubic(frac)
                for placement in drops:
                    start_y = placement["drop_start"]
                    target_y = placement["y"]
                    y_pos = int(start_y + (target_y - start_y) * eased)
                    if y_pos > target_y:
                        y_pos = target_y
                    animated.append(
                        {
                            "logo": placement["logo"],
                            "x": placement["x"],
                            "y": y_pos,
                            "abbr": placement.get("abbr", ""),
                        }
                    )

            if animated:
                _paste_overview_logos(frame, animated)

            display.image(frame)
            display.show()
            time.sleep(OVERVIEW_FRAME_DELAY)

    final = header.copy()
    all_placements: List[Dict[str, Any]] = []
    for column in columns:
        for placement in column.values():
            if placement:
                all_placements.append(placement)
    _paste_overview_logos(final, all_placements)

    display.image(final)
    display.show()
    time.sleep(OVERVIEW_PAUSE_END)
    return ScreenImage(final, displayed=True)


def _render_overview_fallback(
    display,
    title: str,
    fallback_message: Optional[str],
    transition: bool,
) -> ScreenImage:
    clear_display(display)
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    message = fallback_message or "No standings"

    try:
        l, t, r, b = draw.textbbox((0, 0), title, font=TITLE_FONT)
        tw, th = r - l, b - t
        tx = (WIDTH - tw) // 2 - l
        ty = 0 - t
    except Exception:  # pragma: no cover - PIL fallback
        tw, th = draw.textsize(title, font=TITLE_FONT)
        tx = (WIDTH - tw) // 2
        ty = 0
    draw.text((tx, ty), title, font=TITLE_FONT, fill=WHITE)

    try:
        l, t, r, b = draw.textbbox((0, 0), message, font=ROW_FONT)
        tw, th = r - l, b - t
        mx = (WIDTH - tw) // 2 - l
        my = (HEIGHT - th) // 2 - t
    except Exception:  # pragma: no cover - PIL fallback
        tw, th = draw.textsize(message, font=ROW_FONT)
        mx = (WIDTH - tw) // 2
        my = (HEIGHT - th) // 2
    draw.text((mx, my), message, font=ROW_FONT, fill=WHITE)

    if transition:
        return ScreenImage(img, displayed=False)

    display.image(img)
    display.show()
    time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    return ScreenImage(img, displayed=True)


def _scroll_display(display, full_img: Image.Image):
    if full_img.height <= HEIGHT:
        display.image(full_img)
        display.show()
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
        return

    max_offset = full_img.height - HEIGHT
    frame = full_img.crop((0, 0, WIDTH, HEIGHT))
    display.image(frame)
    display.show()
    time.sleep(SCOREBOARD_SCROLL_PAUSE_TOP)

    for offset in range(
        SCOREBOARD_SCROLL_STEP, max_offset + 1, SCOREBOARD_SCROLL_STEP
    ):
        frame = full_img.crop((0, offset, WIDTH, offset + HEIGHT))
        display.image(frame)
        display.show()
        time.sleep(SCOREBOARD_SCROLL_DELAY)

    time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)


def _render_and_display(
    display,
    title: str,
    division_order: List[str],
    standings: Dict[str, List[dict]],
    transition: bool,
    fallback_message: Optional[str] = None,
) -> ScreenImage:
    if not any(standings.values()):
        clear_display(display)
        img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        message = fallback_message or "No standings"
        try:
            l, t, r, b = draw.textbbox((0, 0), title, font=TITLE_FONT)
            tw, th = r - l, b - t
            tx = (WIDTH - tw) // 2 - l
            ty = 0 - t
        except Exception:  # pragma: no cover - PIL fallback
            tw, th = draw.textsize(title, font=TITLE_FONT)
            tx = (WIDTH - tw) // 2
            ty = 0
        draw.text((tx, ty), title, font=TITLE_FONT, fill=WHITE)

        try:
            l, t, r, b = draw.textbbox((0, 0), message, font=ROW_FONT)
            tw, th = r - l, b - t
            tx = (WIDTH - tw) // 2 - l
            ty = (HEIGHT - th) // 2 - t
        except Exception:  # pragma: no cover - PIL fallback
            tw, th = draw.textsize(message, font=ROW_FONT)
            tx = (WIDTH - tw) // 2
            ty = (HEIGHT - th) // 2
        draw.text((tx, ty), message, font=ROW_FONT, fill=WHITE)

        if transition:
            return ScreenImage(img, displayed=False)
        display.image(img)
        display.show()
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
        return ScreenImage(img, displayed=True)

    full_img = _render_conference(title, division_order, standings)
    if transition:
        _scroll_display(display, full_img)
        return ScreenImage(full_img, displayed=True)

    _scroll_display(display, full_img)
    return ScreenImage(full_img, displayed=True)


# ─── Public API ───────────────────────────────────────────────────────────────
@log_call
def draw_nfl_overview_nfc(display, transition: bool = False) -> ScreenImage:
    standings_by_conf, fallback_message = _fetch_standings_data()
    conference = standings_by_conf.get(CONFERENCE_NFC_KEY, {})
    return _render_overview(
        display,
        "NFC Overview",
        DIVISION_ORDER_NFC,
        conference,
        transition,
        fallback_message,
    )


@log_call
def draw_nfl_overview_afc(display, transition: bool = False) -> ScreenImage:
    standings_by_conf, fallback_message = _fetch_standings_data()
    conference = standings_by_conf.get(CONFERENCE_AFC_KEY, {})
    return _render_overview(
        display,
        "AFC Overview",
        DIVISION_ORDER_AFC,
        conference,
        transition,
        fallback_message,
    )


@log_call
def draw_nfl_standings_nfc(display, transition: bool = False) -> ScreenImage:
    standings_by_conf, fallback_message = _fetch_standings_data()
    conference = standings_by_conf.get(CONFERENCE_NFC_KEY, {})
    return _render_and_display(
        display,
        TITLE_NFC,
        DIVISION_ORDER_NFC,
        conference,
        transition,
        fallback_message,
    )


@log_call
def draw_nfl_standings_afc(display, transition: bool = False) -> ScreenImage:
    standings_by_conf, fallback_message = _fetch_standings_data()
    conference = standings_by_conf.get(CONFERENCE_AFC_KEY, {})
    return _render_and_display(
        display,
        TITLE_AFC,
        DIVISION_ORDER_AFC,
        conference,
        transition,
        fallback_message,
    )


if __name__ == "__main__":  # pragma: no cover
    from utils import Display

    disp = Display()
    try:
        draw_nfl_standings_nfc(disp)
        draw_nfl_standings_afc(disp)
    finally:
        clear_display(disp)
