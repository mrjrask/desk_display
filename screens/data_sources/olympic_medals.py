#!/usr/bin/env python3
"""Olympic medal standings provider with local fallback data."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from services.http_client import get_session

SESSION = get_session()
REQUEST_TIMEOUT = 10

# Optional live source (settable in env). Must return list-like JSON rows.
MEDAL_URL = os.getenv("OLYMPIC_MEDALS_URL", "")

# Offline-safe fallback: 2022 Winter Olympics medal table top 20 by gold.
FALLBACK_MEDALS: list[dict[str, Any]] = [
    {"rank": 1, "country": "NOR", "gold": 16, "silver": 8, "bronze": 13},
    {"rank": 2, "country": "GER", "gold": 12, "silver": 10, "bronze": 5},
    {"rank": 3, "country": "CHN", "gold": 9, "silver": 4, "bronze": 2},
    {"rank": 4, "country": "USA", "gold": 8, "silver": 10, "bronze": 7},
    {"rank": 5, "country": "SWE", "gold": 8, "silver": 5, "bronze": 5},
    {"rank": 6, "country": "NED", "gold": 8, "silver": 5, "bronze": 4},
    {"rank": 7, "country": "AUT", "gold": 7, "silver": 7, "bronze": 4},
    {"rank": 8, "country": "SUI", "gold": 7, "silver": 2, "bronze": 5},
    {"rank": 9, "country": "ROC", "gold": 6, "silver": 12, "bronze": 14},
    {"rank": 10, "country": "FRA", "gold": 5, "silver": 7, "bronze": 2},
    {"rank": 11, "country": "CAN", "gold": 4, "silver": 8, "bronze": 14},
    {"rank": 12, "country": "JPN", "gold": 3, "silver": 6, "bronze": 9},
    {"rank": 13, "country": "ITA", "gold": 2, "silver": 7, "bronze": 8},
    {"rank": 14, "country": "KOR", "gold": 2, "silver": 5, "bronze": 2},
    {"rank": 15, "country": "SLO", "gold": 2, "silver": 3, "bronze": 2},
    {"rank": 16, "country": "FIN", "gold": 2, "silver": 2, "bronze": 4},
    {"rank": 17, "country": "NZL", "gold": 2, "silver": 1, "bronze": 0},
    {"rank": 18, "country": "AUS", "gold": 1, "silver": 2, "bronze": 1},
    {"rank": 19, "country": "GBR", "gold": 1, "silver": 1, "bronze": 0},
    {"rank": 20, "country": "HUN", "gold": 1, "silver": 0, "bronze": 2},
]


def _normalize_row(row: dict[str, Any]) -> dict[str, int | str] | None:
    country = str(row.get("country") or row.get("abbr") or row.get("code") or "").strip().upper()
    if not country:
        return None

    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    gold = _to_int(row.get("gold"))
    silver = _to_int(row.get("silver"))
    bronze = _to_int(row.get("bronze"))
    total = _to_int(row.get("total")) or (gold + silver + bronze)
    rank = _to_int(row.get("rank"))
    return {
        "rank": rank,
        "country": country,
        "gold": gold,
        "silver": silver,
        "bronze": bronze,
        "total": total,
    }


def _normalize_payload(payload: Any) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    if isinstance(payload, dict):
        for key in ("medals", "countries", "rows", "table", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                payload = value
                break
    if not isinstance(payload, list):
        return rows

    for raw in payload:
        if not isinstance(raw, dict):
            continue
        row = _normalize_row(raw)
        if row:
            rows.append(row)

    rows.sort(key=lambda row: (-int(row["gold"]), -int(row["silver"]), -int(row["bronze"]), str(row["country"])))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def _load_local_fallback() -> list[dict[str, int | str]]:
    path = Path(__file__).resolve().parent / "olympic_medals_fallback.json"
    if path.exists():
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            rows = _normalize_payload(content)
            if rows:
                return rows
        except Exception:
            logging.exception("Failed parsing %s", path)

    rows = _normalize_payload(FALLBACK_MEDALS)
    return rows


def fetch_olympic_medal_table(*, top_n: int = 20) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    if MEDAL_URL:
        try:
            response = SESSION.get(MEDAL_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            rows = _normalize_payload(response.json())
        except Exception as exc:
            logging.warning("Olympic medal feed failed (%s): %s", MEDAL_URL, exc)

    if not rows:
        rows = _load_local_fallback()

    return rows[: max(1, top_n)]
