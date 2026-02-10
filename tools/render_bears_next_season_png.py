#!/usr/bin/env python3
"""Interactive Bears Next Season PNG generator for supported displays."""

from __future__ import annotations

import argparse
import curses
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NFL_TEAMS = [
    ("Arizona Cardinals", "ari"), ("Atlanta Falcons", "atl"), ("Baltimore Ravens", "bal"),
    ("Buffalo Bills", "buf"), ("Carolina Panthers", "car"), ("Chicago Bears", "chi"),
    ("Cincinnati Bengals", "cin"), ("Cleveland Browns", "cle"), ("Dallas Cowboys", "dal"),
    ("Denver Broncos", "den"), ("Detroit Lions", "det"), ("Green Bay Packers", "gb"),
    ("Houston Texans", "hou"), ("Indianapolis Colts", "ind"), ("Jacksonville Jaguars", "jax"),
    ("Kansas City Chiefs", "kc"), ("Las Vegas Raiders", "lv"), ("Los Angeles Chargers", "lac"),
    ("Los Angeles Rams", "lar"), ("Miami Dolphins", "mia"), ("Minnesota Vikings", "min"),
    ("New England Patriots", "ne"), ("New Orleans Saints", "no"), ("New York Giants", "nyg"),
    ("New York Jets", "nyj"), ("Philadelphia Eagles", "phi"), ("Pittsburgh Steelers", "pit"),
    ("San Francisco 49ers", "sf"), ("Seattle Seahawks", "sea"), ("Tampa Bay Buccaneers", "tb"),
    ("Tennessee Titans", "ten"), ("Washington Commanders", "wsh"),
]

TARGETS = {
    "dhm": ((320, 240), Path("images/bears_next_season_dhm.png")),
    "h4": ((800, 480), Path("images/bears_next_season_h4.png")),
    "h4sq": ((720, 720), Path("images/bears_next_season_h4sq.png")),
}

DEFAULT_HOME = ["det", "gb", "jax", "min", "ne", "no", "nyj", "phi", "tb"]
DEFAULT_AWAY = ["atl", "buf", "car", "det", "gb", "mia", "min", "sea"]


def _parse_team_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    valid = {abbr for _, abbr in NFL_TEAMS}
    selected = sorted({item.strip().lower() for item in raw.split(",") if item.strip()})
    bad = [abbr for abbr in selected if abbr not in valid]
    if bad:
        raise ValueError(f"Unknown team abbreviation(s): {', '.join(bad)}")
    return selected


def _interactive_select() -> tuple[list[str], list[str]]:
    home: set[str] = set()
    away: set[str] = set()

    def _ui(stdscr):
        nonlocal home, away
        curses.curs_set(0)
        stdscr.keypad(True)
        row, col = 0, 0
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            stdscr.addnstr(0, 0, "Select teams (SPACE=toggle, arrows=move, ENTER=render, q=quit)", w - 1, curses.A_BOLD)
            stdscr.addnstr(1, 0, "      Home Away Team", w - 1, curses.A_UNDERLINE)

            visible_rows = max(1, h - 4)
            start = max(0, row - visible_rows + 1)
            for idx, (name, abbr) in enumerate(NFL_TEAMS[start:start + visible_rows], start=start):
                y = 2 + idx - start
                line = f"  {'[*]' if abbr in home else '[ ]'}  {'[*]' if abbr in away else '[ ]'}  {name}"
                stdscr.addnstr(y, 0, line, w - 1, curses.A_REVERSE if idx == row else curses.A_NORMAL)
                if idx == row:
                    stdscr.chgat(y, 2 if col == 0 else 7, 3, curses.A_REVERSE | curses.A_BOLD)

            stdscr.addnstr(h - 1, 0, f"Selected: home={len(home)} away={len(away)}", w - 1, curses.A_DIM)
            stdscr.refresh()

            key = stdscr.getch()
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Cancelled by user")
            if key in (curses.KEY_UP, ord("k")):
                row = max(0, row - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                row = min(len(NFL_TEAMS) - 1, row + 1)
            elif key in (curses.KEY_LEFT, ord("h")):
                col = 0
            elif key in (curses.KEY_RIGHT, ord("l"), 9):
                col = 1 - col if key == 9 else 1
            elif key == ord(" "):
                abbr = NFL_TEAMS[row][1]
                target = home if col == 0 else away
                target.remove(abbr) if abbr in target else target.add(abbr)
            elif key in (10, 13, curses.KEY_ENTER):
                return

    curses.wrapper(_ui)
    return sorted(home), sorted(away)


def _render_one(output: Path, home: list[str], away: list[str]) -> None:
    import config
    from screens.draw_bears_schedule import render_bears_next_season_image

    background = tuple(config.get_screen_background_color("bears next season", (0, 0, 0)))
    img = render_bears_next_season_image(
        config.WIDTH,
        config.HEIGHT,
        background,
        home_opponents=home,
        away_opponents=away,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output, format="PNG")


def _run_target(key: str, home: list[str], away: list[str]) -> None:
    (width, height), output = TARGETS[key]
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--render-target",
        key,
        "--home",
        ",".join(home),
        "--away",
        ",".join(away),
    ]
    env = dict(**__import__("os").environ)
    env["DISPLAY_WIDTH"] = str(width)
    env["DISPLAY_HEIGHT"] = str(height)
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO_ROOT))
    print(f"Saved {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Bears Next Season images for DHM/H4/H4SQ.")
    parser.add_argument("--home", help="Comma-separated home team abbreviations")
    parser.add_argument("--away", help="Comma-separated away team abbreviations")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive selector instead of using defaults",
    )
    parser.add_argument("--render-target", choices=sorted(TARGETS.keys()), help=argparse.SUPPRESS)
    args = parser.parse_args()

    home = _parse_team_list(args.home) if args.home is not None else list(DEFAULT_HOME)
    away = _parse_team_list(args.away) if args.away is not None else list(DEFAULT_AWAY)

    if args.render_target:
        output = TARGETS[args.render_target][1]
        _render_one(output, home, away)
        return 0

    if args.interactive:
        home, away = _interactive_select()

    for key in ("dhm", "h4", "h4sq"):
        _run_target(key, home, away)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
