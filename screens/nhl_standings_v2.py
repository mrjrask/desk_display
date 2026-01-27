#!/usr/bin/env python3
"""Wild card NHL standings screens (v2) using GP, RW, and Points columns."""
from __future__ import annotations

from contextlib import contextmanager
from typing import List, Sequence, Tuple
from utils import ScreenImage, clear_display, log_call

import screens.nhl_standings as nhl_standings
from screens.nhl_standings import (
    CONFERENCE_EAST_KEY,
    CONFERENCE_WEST_KEY,
    DIVISION_ORDER_EAST,
    DIVISION_ORDER_WEST,
    TITLE_EAST,
    TITLE_WEST,
    _conference_overview_rows,
    _division_sequence_sort_key,
    _animate_overview_drop,
    _apply_style_overrides,
    _compose_overview_image,
    _fetch_standings_data,
    _normalize_int,
    _prepare_overview_horizontal,
    _render_conference,
    _render_empty,
    _scroll_vertical,
    OVERVIEW_TITLE_EAST,
    OVERVIEW_TITLE_WEST,
)

WILDCARD_SECTION_NAME = "Wild Card"
OVERVIEW_TITLE_WEST_V3 = "NHL West Wild Card"
OVERVIEW_TITLE_EAST_V3 = "NHL East Wild Card"
TITLE_SUBTITLE_WILDCARD = "Wild Card Standings"
DIVISION_LEADERS_LABELS = {
    "Central": "Central Leaders",
    "Pacific": "Pacific Leaders",
    "Metropolitan": "Metropolitan Leaders",
    "Atlantic": "Atlantic Leaders",
}


@contextmanager
def _wildcard_columns() -> None:
    original_stats = nhl_standings.STATS_COLUMNS
    original_headers = nhl_standings.COLUMN_HEADERS
    original_column_text_height = nhl_standings.COLUMN_TEXT_HEIGHT
    original_column_row_height = nhl_standings.COLUMN_ROW_HEIGHT
    try:
        nhl_standings.STATS_COLUMNS = ("gamesPlayed", "regulationWins", "points")
        nhl_standings.COLUMN_HEADERS = [
            ("", "team", "left"),
            ("GP", "gamesPlayed", "right"),
            ("RW", "regulationWins", "right"),
            ("PTS", "points", "right"),
        ]
        yield
    finally:
        nhl_standings.STATS_COLUMNS = original_stats
        nhl_standings.COLUMN_HEADERS = original_headers
        nhl_standings.COLUMN_TEXT_HEIGHT = original_column_text_height
        nhl_standings.COLUMN_ROW_HEIGHT = original_column_row_height


def _update_column_metrics() -> None:
    nhl_standings.COLUMN_TEXT_HEIGHT = max(
        nhl_standings._text_size(
            label,
            nhl_standings.COLUMN_HEADER_FONTS.get(key, nhl_standings.COLUMN_FONT),
        )[1]
        for label, key, _ in nhl_standings.COLUMN_HEADERS
    )
    nhl_standings.COLUMN_ROW_HEIGHT = nhl_standings.COLUMN_TEXT_HEIGHT + 2


@contextmanager
def _cap_wildcard_column_spacing(max_step: int | None) -> None:
    original_max_step = nhl_standings.STATS_COLUMN_MAX_STEP
    try:
        nhl_standings.STATS_COLUMN_MAX_STEP = max_step
        yield
    finally:
        nhl_standings.STATS_COLUMN_MAX_STEP = original_max_step


def _normalize_wildcard_team(team: dict) -> dict:
    normalized = dict(team)
    wins = _normalize_int(normalized.get("wins"))
    losses = _normalize_int(normalized.get("losses"))
    ot = _normalize_int(normalized.get("ot"))
    normalized.setdefault("gamesPlayed", wins + losses + ot)
    normalized.setdefault("regulationWins", 0)
    normalized.setdefault(
        "regulationPlusOvertimeWins",
        _normalize_int(normalized.get("row", wins)),
    )
    return normalized


def _wildcard_sort_key(team: dict) -> Tuple[int, int, int, int, int, str]:
    points = _normalize_int(team.get("points"))
    regulation_wins = _normalize_int(team.get("regulationWins"))
    regulation_plus_overtime_wins = _normalize_int(
        team.get("regulationPlusOvertimeWins", team.get("row", team.get("wins")))
    )
    wins = _normalize_int(team.get("wins"))
    games_played = _normalize_int(team.get("gamesPlayed"))
    abbr = str(team.get("abbr", ""))
    # Sort by points (desc), regulation wins (desc), regulation+OT wins (desc),
    # overall wins (desc), games played (asc), then abbreviation for determinism.
    return (
        -points,
        -regulation_wins,
        -regulation_plus_overtime_wins,
        -wins,
        games_played,
        abbr,
    )


def _wildcard_order_sort_key(team: dict) -> Tuple:
    wildcard_rank = _normalize_int(team.get("wildcardRank") or team.get("wildCardRank"))
    if wildcard_rank > 0:
        return (0, wildcard_rank) + _wildcard_sort_key(team)
    return (1,) + _wildcard_sort_key(team)


def _conference_wildcard_standings(
    conference: dict[str, list[dict]],
    division_order: Sequence[str],
    wildcard_order: Sequence[str] | None = None,
) -> dict[str, list[dict]]:
    wildcard_conf: dict[str, list[dict]] = {}
    remaining: list[dict] = []
    all_teams: list[dict] = []

    for division in division_order:
        teams = [_normalize_wildcard_team(team) for team in conference.get(division, [])]
        teams.sort(key=_division_sequence_sort_key)
        wildcard_conf[division] = teams[:3]
        remaining.extend(teams[3:])
        all_teams.extend(teams)

    wildcard_ranked = [
        team
        for team in all_teams
        if _normalize_int(team.get("wildcardRank") or team.get("wildCardRank")) > 0
    ]
    if wildcard_ranked:
        if wildcard_order:
            order_map = {abbr.upper(): idx for idx, abbr in enumerate(wildcard_order)}

            def _order_key(team: dict) -> tuple[int, Tuple[int, int, int, int, int, str]]:
                abbr = str(team.get("abbr", "")).upper()
                return (order_map.get(abbr, 999), _wildcard_order_sort_key(team))

            wildcard_ranked.sort(key=_order_key)
        else:
            wildcard_ranked.sort(key=_wildcard_order_sort_key)
        if len(wildcard_ranked) >= 3:
            wildcard_ranked[2]["_wildcard_cutoff_before"] = True
        wildcard_conf[WILDCARD_SECTION_NAME] = wildcard_ranked
        return wildcard_conf

    if wildcard_order:
        order_map = {abbr.upper(): idx for idx, abbr in enumerate(wildcard_order)}

        def _order_key(team: dict) -> tuple[int, Tuple[int, int, int, int, int, str]]:
            abbr = str(team.get("abbr", "")).upper()
            return (order_map.get(abbr, 999), _wildcard_order_sort_key(team))

        remaining.sort(key=_order_key)
    else:
        remaining.sort(key=_wildcard_order_sort_key)
    if len(remaining) >= 3:
        remaining[2]["_wildcard_cutoff_before"] = True
    if remaining:
        wildcard_conf[WILDCARD_SECTION_NAME] = remaining

    return wildcard_conf


def _build_wildcard_standings(
    standings_by_conf: dict[str, dict[str, list[dict]]],
    wildcard_order_by_conf: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, list[dict]]]:
    wildcard: dict[str, dict[str, list[dict]]] = {}
    conference_orders = {
        CONFERENCE_WEST_KEY: DIVISION_ORDER_WEST,
        CONFERENCE_EAST_KEY: DIVISION_ORDER_EAST,
    }

    for conf_key, order in conference_orders.items():
        conference = standings_by_conf.get(conf_key, {})
        if conference:
            wildcard[conf_key] = _conference_wildcard_standings(
                conference,
                order,
                (wildcard_order_by_conf or {}).get(conf_key),
            )

    return wildcard


@log_call
def draw_nhl_standings_overview_v2_west(display, transition: bool = False) -> ScreenImage:
    with _wildcard_columns():
        standings_by_conf = _fetch_standings_data()
        _apply_style_overrides("NHL Standings Overview v2 West")
        _update_column_metrics()

        conference = standings_by_conf.get(CONFERENCE_WEST_KEY, {})
        rows = _conference_overview_rows(conference, DIVISION_ORDER_WEST, "West")

        if not any(teams for _, teams in rows):
            clear_display(display)
            img = _render_empty(OVERVIEW_TITLE_WEST)
            if transition:
                return ScreenImage(img, displayed=False)
            display.image(img)
            return ScreenImage(img, displayed=True)

        base, row_positions = _prepare_overview_horizontal(
            rows,
            title=OVERVIEW_TITLE_WEST,
            conference_key=CONFERENCE_WEST_KEY,
        )
        final_img, _ = _compose_overview_image(base, row_positions)

        clear_display(display)
        _animate_overview_drop(display, base, row_positions)
        display.image(final_img)
        if hasattr(display, "show"):
            display.show()

    return ScreenImage(final_img, displayed=True)


@log_call
def draw_nhl_overview_west_v3(display, transition: bool = False) -> ScreenImage:
    with _wildcard_columns():
        standings_by_conf = _fetch_standings_data()
        _apply_style_overrides("NHL Overview West v3")
        _update_column_metrics()

        conference = standings_by_conf.get(CONFERENCE_WEST_KEY, {})
        rows = _conference_overview_rows(conference, DIVISION_ORDER_WEST, "West")

        if not any(teams for _, teams in rows):
            clear_display(display)
            img = _render_empty(OVERVIEW_TITLE_WEST_V3)
            if transition:
                return ScreenImage(img, displayed=False)
            display.image(img)
            return ScreenImage(img, displayed=True)

        base, row_positions = _prepare_overview_horizontal(
            rows,
            title=OVERVIEW_TITLE_WEST_V3,
            conference_key=CONFERENCE_WEST_KEY,
        )
        final_img, _ = _compose_overview_image(base, row_positions)

        clear_display(display)
        _animate_overview_drop(display, base, row_positions)
        display.image(final_img)
        if hasattr(display, "show"):
            display.show()

    return ScreenImage(final_img, displayed=True)


@log_call
def draw_nhl_standings_overview_v2_east(display, transition: bool = False) -> ScreenImage:
    with _wildcard_columns():
        standings_by_conf = _fetch_standings_data()
        _apply_style_overrides("NHL Standings Overview v2 East")
        _update_column_metrics()

        conference = standings_by_conf.get(CONFERENCE_EAST_KEY, {})
        rows = _conference_overview_rows(conference, DIVISION_ORDER_EAST, "East")

        if not any(teams for _, teams in rows):
            clear_display(display)
            img = _render_empty(OVERVIEW_TITLE_EAST)
            if transition:
                return ScreenImage(img, displayed=False)
            display.image(img)
            return ScreenImage(img, displayed=True)

        base, row_positions = _prepare_overview_horizontal(
            rows,
            title=OVERVIEW_TITLE_EAST,
            conference_key=CONFERENCE_EAST_KEY,
        )
        final_img, _ = _compose_overview_image(base, row_positions)

        clear_display(display)
        _animate_overview_drop(display, base, row_positions)
        display.image(final_img)
        if hasattr(display, "show"):
            display.show()

    return ScreenImage(final_img, displayed=True)


@log_call
def draw_nhl_overview_east_v3(display, transition: bool = False) -> ScreenImage:
    with _wildcard_columns():
        standings_by_conf = _fetch_standings_data()
        _apply_style_overrides("NHL Overview East v3")
        _update_column_metrics()

        conference = standings_by_conf.get(CONFERENCE_EAST_KEY, {})
        rows = _conference_overview_rows(conference, DIVISION_ORDER_EAST, "East")

        if not any(teams for _, teams in rows):
            clear_display(display)
            img = _render_empty(OVERVIEW_TITLE_EAST_V3)
            if transition:
                return ScreenImage(img, displayed=False)
            display.image(img)
            return ScreenImage(img, displayed=True)

        base, row_positions = _prepare_overview_horizontal(
            rows,
            title=OVERVIEW_TITLE_EAST_V3,
            conference_key=CONFERENCE_EAST_KEY,
        )
        final_img, _ = _compose_overview_image(base, row_positions)

        clear_display(display)
        _animate_overview_drop(display, base, row_positions)
        display.image(final_img)
        if hasattr(display, "show"):
            display.show()

    return ScreenImage(final_img, displayed=True)


@log_call
def draw_nhl_standings_west_v2(display, transition: bool = False) -> ScreenImage:
    with _wildcard_columns(), _cap_wildcard_column_spacing(nhl_standings.STATS_COLUMN_MIN_STEP):
        standings_by_conf = _fetch_standings_data()
        wildcard_order = nhl_standings._fetch_wildcard_order_api_web()
        wildcard_standings = _build_wildcard_standings(standings_by_conf, wildcard_order)
        _apply_style_overrides("NHL Standings West v2")
        _update_column_metrics()
        conference = wildcard_standings.get(CONFERENCE_WEST_KEY, {})
        divisions = [d for d in DIVISION_ORDER_WEST if conference.get(d)]
        if conference.get(WILDCARD_SECTION_NAME):
            divisions.append(WILDCARD_SECTION_NAME)
        if not divisions:
            clear_display(display)
            img = _render_empty(
                TITLE_WEST,
                TITLE_SUBTITLE_WILDCARD,
                conference_key=CONFERENCE_WEST_KEY,
            )
            if transition:
                return ScreenImage(img, displayed=False)
            display.image(img)
            return ScreenImage(img, displayed=True)

        full_img = _render_conference(
            TITLE_WEST,
            divisions,
            conference,
            subtitle=TITLE_SUBTITLE_WILDCARD,
            division_labels=DIVISION_LEADERS_LABELS,
            conference_key=CONFERENCE_WEST_KEY,
        )
        clear_display(display)
        _scroll_vertical(display, full_img)
    return ScreenImage(full_img, displayed=True)


@log_call
def draw_nhl_standings_east_v2(display, transition: bool = False) -> ScreenImage:
    with _wildcard_columns(), _cap_wildcard_column_spacing(nhl_standings.STATS_COLUMN_MIN_STEP):
        standings_by_conf = _fetch_standings_data()
        wildcard_order = nhl_standings._fetch_wildcard_order_api_web()
        wildcard_standings = _build_wildcard_standings(standings_by_conf, wildcard_order)
        _apply_style_overrides("NHL Standings East v2")
        _update_column_metrics()
        conference = wildcard_standings.get(CONFERENCE_EAST_KEY, {})
        divisions = [d for d in DIVISION_ORDER_EAST if conference.get(d)]
        if conference.get(WILDCARD_SECTION_NAME):
            divisions.append(WILDCARD_SECTION_NAME)
        if not divisions:
            clear_display(display)
            img = _render_empty(
                TITLE_EAST,
                TITLE_SUBTITLE_WILDCARD,
                conference_key=CONFERENCE_EAST_KEY,
            )
            if transition:
                return ScreenImage(img, displayed=False)
            display.image(img)
            return ScreenImage(img, displayed=True)

        full_img = _render_conference(
            TITLE_EAST,
            divisions,
            conference,
            subtitle=TITLE_SUBTITLE_WILDCARD,
            division_labels=DIVISION_LEADERS_LABELS,
            conference_key=CONFERENCE_EAST_KEY,
        )
        clear_display(display)
        _scroll_vertical(display, full_img)
    return ScreenImage(full_img, displayed=True)


if __name__ == "__main__":  # pragma: no cover
    from utils import Display

    disp = Display()
    try:
        draw_nhl_standings_west_v2(disp)
        draw_nhl_standings_east_v2(disp)
    finally:
        clear_display(disp)
