#!/usr/bin/env python3
"""Run lightweight checks against the external APIs used by the project.

Each check calls the same fetch helpers that the application uses and reports
whether the call returned data or raised an error. Use ``--json`` for a
machine-readable summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Ensure the project's .env is loaded for API credentials and feature flags.
os.environ.setdefault("CONFIG_LOAD_DOTENV", "1")

from data_fetch import (
    fetch_bears_standings,
    fetch_blackhawks_last_game,
    fetch_blackhawks_live_game,
    fetch_blackhawks_next_game,
    fetch_blackhawks_next_home_game,
    fetch_blackhawks_standings,
    fetch_bulls_last_game,
    fetch_bulls_live_game,
    fetch_bulls_next_game,
    fetch_bulls_next_home_game,
    fetch_bulls_standings,
    fetch_cubs_games,
    fetch_cubs_standings,
    fetch_sox_games,
    fetch_sox_standings,
    fetch_weather,
    fetch_wolves_games,
)
from screens.draw_travel_time import get_travel_times
from screens.nhl_scoreboard import dns_diagnostics


@dataclass
class ApiCheck:
    """Definition for a single API check."""

    name: str
    func: Callable[[], Any]
    expect_data: bool = True
    description: str | None = None


def _has_data(payload: Any) -> bool:
    if payload is None:
        return False
    if isinstance(payload, (list, tuple, set, dict)):
        return len(payload) > 0
    return True


def _summarize(payload: Any) -> str:
    if payload is None:
        return "None"
    if isinstance(payload, list):
        return f"list(len={len(payload)})"
    if isinstance(payload, dict):
        keys: List[str] = list(payload.keys())
        preview = ", ".join(keys[:5])
        return f"dict(keys=[{preview}]{'...' if len(keys) > 5 else ''})"
    return str(type(payload).__name__)


def _run_check(check: ApiCheck) -> dict:
    start = time.perf_counter()
    result: dict[str, Any] = {
        "name": check.name,
    }
    try:
        payload = check.func()
        has_data = _has_data(payload)
        status = "ok" if (has_data or not check.expect_data) else "no_data"
        result.update(
            {
                "status": status,
                "summary": _summarize(payload),
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic script
        logging.exception("API check %s failed", check.name)
        result.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        result["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
    if check.description:
        result["description"] = check.description
    return result


CHECKS: Iterable[ApiCheck] = [
    ApiCheck("weather", fetch_weather, description="Apple WeatherKit"),
    ApiCheck("nhl_next_game", fetch_blackhawks_next_game, expect_data=False),
    ApiCheck("nhl_next_home_game", fetch_blackhawks_next_home_game, expect_data=False),
    ApiCheck("nhl_last_game", fetch_blackhawks_last_game, expect_data=False),
    ApiCheck("nhl_live_game", fetch_blackhawks_live_game, expect_data=False),
    ApiCheck("nhl_standings", fetch_blackhawks_standings),
    ApiCheck("nba_next_game", fetch_bulls_next_game, expect_data=False),
    ApiCheck("nba_next_home_game", fetch_bulls_next_home_game, expect_data=False),
    ApiCheck("nba_last_game", fetch_bulls_last_game, expect_data=False),
    ApiCheck("nba_live_game", fetch_bulls_live_game, expect_data=False),
    ApiCheck("nba_standings", fetch_bulls_standings),
    ApiCheck("nfl_bears_standings", fetch_bears_standings),
    ApiCheck("mlb_cubs_games", fetch_cubs_games, expect_data=False),
    ApiCheck("mlb_sox_games", fetch_sox_games, expect_data=False),
    ApiCheck("mlb_cubs_standings", fetch_cubs_standings),
    ApiCheck("mlb_sox_standings", fetch_sox_standings),
    ApiCheck(
        "ahl_wolves_games",
        fetch_wolves_games,
        expect_data=False,
        description="AHL Wolves schedule",
    ),
    ApiCheck(
        "nhl_network_diagnostics",
        dns_diagnostics,
        description="DNS and HTTP reachability for NHL endpoints",
    ),
    ApiCheck(
        "google_travel_times",
        get_travel_times,
        expect_data=False,
        description="Google Directions travel times",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Print JSON instead of text"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    results = [_run_check(check) for check in CHECKS]

    if args.json_output:
        print(json.dumps({"checks": results}, indent=2))
    else:
        for res in results:
            status = res.get("status")
            name = res.get("name")
            duration = res.get("duration_ms")
            summary = res.get("summary")
            description = res.get("description")
            line = f"{name:25} {status:8} ({duration} ms)"
            if description:
                line += f" - {description}"
            if summary:
                line += f" :: {summary}"
            if res.get("error"):
                line += f" :: {res['error']}"
            print(line)

    failures = [r for r in results if r.get("status") == "error"]
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
