#!/usr/bin/env python3
"""Olympic medal standings provider (live data only, no fallback)."""

from __future__ import annotations

import logging
import os
from typing import Any

from services.http_client import get_session

SESSION = get_session()
REQUEST_TIMEOUT = 10

# Live source URL (set this to a reliable 2026 Winter Olympics medal feed).
MEDAL_URL = os.getenv("OLYMPIC_MEDALS_URL", "")


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


def fetch_olympic_medal_table(*, top_n: int = 20) -> list[dict[str, int | str]]:
    if not MEDAL_URL:
        logging.info("OLYMPIC_MEDALS_URL is not configured; skipping Olympic medal table render")
        return []

    try:
        response = SESSION.get(MEDAL_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        rows = _normalize_payload(response.json())
    except Exception as exc:
        logging.warning("Olympic medal feed failed (%s): %s", MEDAL_URL, exc)
        return []

    return rows[: max(1, top_n)]
