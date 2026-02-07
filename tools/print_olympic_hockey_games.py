#!/usr/bin/env python3
"""Print normalized Olympic hockey games for a date."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from screens.data_sources.olympic_hockey import (
    fetch_olympic_scoreboard_men,
    fetch_olympic_scoreboard_women,
    resolve_display_date,
)


def _parse_date(raw: str | None) -> dt.date:
    if raw:
        return dt.datetime.strptime(raw, "%Y-%m-%d").date()
    return resolve_display_date()


def main() -> int:
    parser = argparse.ArgumentParser(description="Print normalized Olympic hockey games")
    parser.add_argument("--date", help="Date in YYYY-MM-DD (default uses display-date logic)")
    parser.add_argument("--division", choices=["men", "women", "both"], default="both")
    parser.add_argument("--timezone", default=None, help="Override timezone for display-date logic")
    args = parser.parse_args()

    date_value = _parse_date(args.date)
    out: dict[str, list[dict]] = {}
    if args.division in {"men", "both"}:
        out["men"] = fetch_olympic_scoreboard_men(date_value, tz_name=args.timezone)
    if args.division in {"women", "both"}:
        out["women"] = fetch_olympic_scoreboard_women(date_value, tz_name=args.timezone)

    print(json.dumps({"date": date_value.isoformat(), "games": out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
