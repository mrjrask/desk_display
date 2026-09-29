#!/usr/bin/env python3
"""Download team logos for the NCAA scoreboard screens.

Saves the logos the screens look for, named the way they look them up:

* Folder: ``~/Desktop/desk_display_logos/<mode>/`` by default, so you can
  review them first. The screens read ``images/ncaa/`` (``LOGO_DIR`` in
  ``screens/ncaa_fbs_scoreboard.py`` and ``screens/ncaam_scoreboard.py``), a
  flat folder: copy the files you want straight into it, or pass
  ``--output-dir images/ncaa`` to save there directly.
* Filename: ``{ABBREVIATION}.png`` (ESPN abbreviation, uppercased), or
  ``{team_id}.png`` when ESPN gives a team no abbreviation. This is
  ``_team_logo_filename`` in ``screens/ncaa_fbs_scoreboard.py``, which is also
  the name the board logs as "expected filename" when a logo is missing.
* Image: transparent padding trimmed (the board trims it again when it
  draws) and the longest edge capped at 128 px, the team-logo limit in
  ``images/README.md``. The board scales every logo to the same area itself.
* Iowa uses the university's own Tigerhawk, like ``_TEAM_LOGO_URL_OVERRIDES``
  in both scoreboard screens.

Team sets:

  1. NCAA FBS this week: every team in this Monday-Sunday week's games that
     include a Top 25 team, which is exactly what the FBS board draws
     (unranked opponents included).
  2. NCAA FBS Top 25 poll.
  3. NCAA Men's Basketball Top 25 poll.
  4. NCAA Men's Basketball Tournament field ("March Madness").

The NCAAM board currently draws team logos straight from ESPN's game data
and reads only its league logos (``NCAA.png``, ``MM.png``) from
``images/ncaa/``, so sets 3 and 4 fill a local copy it does not read yet.

ESPN is asked on ``site.web.api.espn.com`` first and ``site.api.espn.com``
second, like ``services/sports/ncaa_fbs.py``: some networks get 403 from
``site.api``. ``site.web.api`` rejects date ranges, so games are fetched
one day at a time.

Logos that already exist in the output folder are left alone unless
``--overwrite`` is given, so hand-tuned files stay put.

No venv setup is needed. Inside the project the script re-runs itself under
the project's ``venv``/``.venv``. Elsewhere, if ``requests`` or ``Pillow`` is
missing, it builds a throwaway venv in a temp folder, installs the two
packages there, runs, and deletes the venv when it finishes.

Usage:
    python3 scripts/logo_getter.py                 # interactive prompt
    python3 scripts/logo_getter.py --mode fbs_week
    python3 scripts/logo_getter.py --mode fbs_top25 --dry-run
    python3 scripts/logo_getter.py --mode fbs_week --output-dir images/ncaa
    python3 scripts/logo_getter.py --mode ncaam_tournament --season-year 2027

Requires only ``requests`` and ``Pillow`` (already project dependencies);
it sets them up itself when they are missing (see above).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import venv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

# requests and Pillow may be missing when the script runs outside the
# project's venv; the __main__ block below sets up a temporary venv then.
try:
    import requests
    from PIL import Image
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover - exercised by running without the packages
    requests = Image = HTTPAdapter = Retry = None  # type: ignore[assignment]

if Image is None:  # pragma: no cover
    RESAMPLE = None
elif hasattr(Image, "Resampling"):  # Pillow >= 9.1
    RESAMPLE = Image.Resampling.LANCZOS
else:  # pragma: no cover - older Pillow
    RESAMPLE = Image.LANCZOS


REQUIRED_PACKAGES = {"requests": "requests", "PIL": "Pillow"}  # import name: pip name
# Set in the child run inside the temporary venv so it does not build another.
TEMP_VENV_ENV = "DESK_DISPLAY_LOGO_GETTER_TEMP_VENV"


def missing_packages() -> list[str]:
    """Return the pip names of required packages this Python cannot import."""

    return [
        pip for module, pip in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None
    ]


def run_in_temp_venv(argv: list[str], packages: list[str]) -> int:
    """Run this script in a throwaway venv with *packages*, then delete the venv."""

    venv_dir = Path(tempfile.mkdtemp(prefix="logo_getter_venv_"))
    try:
        print(f"Missing {', '.join(packages)}; setting up a temporary venv in {venv_dir}...")
        try:
            venv.EnvBuilder(with_pip=True).create(venv_dir)
        except Exception as exc:
            print(
                f"Could not create a venv ({exc}). On Raspberry Pi OS/Debian, "
                "install it with: sudo apt install python3-venv"
            )
            return 1
        python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        install = subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                *packages,
            ],
            check=False,
        )
        if install.returncode != 0:
            print("Could not install the packages into the temporary venv.")
            return install.returncode or 1
        env = {**os.environ, TEMP_VENV_ENV: "1"}
        return subprocess.run(
            [str(python), str(Path(__file__).resolve()), *argv], env=env, check=False
        ).returncode
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)
        print(f"Removed the temporary venv {venv_dir}.")


def bootstrap_dependencies() -> None:
    """Run under the project venv, or a temporary one when packages are missing."""

    if os.environ.get(TEMP_VENV_ENV):
        return
    # Same as scripts/_venv_bootstrap.py, inlined so a copy of this file
    # outside the repo still runs.
    project_root = Path(__file__).resolve().parents[1]
    for candidate in (project_root / ".venv", project_root / "venv"):
        python = candidate / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.exists():
            continue
        if Path(sys.prefix).resolve() != candidate.resolve():
            os.execv(str(python), [str(python), *sys.argv])
        break
    # Still here: no project venv (or it is the one running). Fall back to a
    # temporary venv when this Python lacks the packages.
    _missing = missing_packages()
    if _missing:
        try:
            raise SystemExit(run_in_temp_venv(sys.argv[1:], _missing))
        except KeyboardInterrupt:
            raise SystemExit(130) from None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Where screens/ncaa_fbs_scoreboard.py and screens/ncaam_scoreboard.py read logos.
PROJECT_LOGO_DIR = PROJECT_ROOT / "images" / "ncaa"
DEFAULT_OUTPUT_BASE = Path.home() / "Desktop" / "desk_display_logos"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 desk_display-logo-getter/1.0"
)
REQUEST_TIMEOUT = 15
TEAM_LOGO_MAX_DIMENSION = 128  # matches images/README.md team-logo policy
CENTRAL_TIME = ZoneInfo("America/Chicago")

MODE_FBS_WEEK = "fbs_week"
MODE_FBS_TOP25 = "fbs_top25"
MODE_NCAAM_TOP25 = "ncaam_top25"
MODE_NCAAM_TOURNAMENT = "ncaam_tournament"
MODES = (MODE_FBS_WEEK, MODE_FBS_TOP25, MODE_NCAAM_TOP25, MODE_NCAAM_TOURNAMENT)

MODE_CHOICES = {str(index): mode for index, mode in enumerate(MODES, start=1)}
MODE_LABELS = {
    MODE_FBS_WEEK: "NCAA FBS this week (every team on the FBS scoreboard)",
    MODE_FBS_TOP25: "NCAA FBS Top 25 poll (football)",
    MODE_NCAAM_TOP25: "NCAA Men's Basketball Top 25 poll",
    MODE_NCAAM_TOURNAMENT: "NCAA Men's Basketball Tournament field",
}
MODE_SPORTS = {
    MODE_FBS_WEEK: "football",
    MODE_FBS_TOP25: "football",
    MODE_NCAAM_TOP25: "basketball",
    MODE_NCAAM_TOURNAMENT: "basketball",
}
# League logos each board reads from images/ncaa (``_mode_title_and_logo``).
MODE_LEAGUE_LOGOS = {
    MODE_FBS_WEEK: ("NCAA.png",),
    MODE_FBS_TOP25: ("NCAA.png",),
    MODE_NCAAM_TOP25: ("NCAA.png",),
    MODE_NCAAM_TOURNAMENT: ("MM.png",),
}

# site.web.api first: site.api answers 403 on some networks (see
# screens/ncaa_fbs_scoreboard.py ESPN_URL / ESPN_FALLBACK_URLS).
ESPN_HOSTS = ("https://site.web.api.espn.com", "https://site.api.espn.com")
ESPN_SPORT_PATHS = {
    "football": "/apis/site/v2/sports/football/college-football",
    "basketball": "/apis/site/v2/sports/basketball/mens-college-basketball",
}
# ESPN scoreboard ``groups`` the boards ask for: 80 is FBS, 100 the NCAA tournament.
FBS_GROUP = 80
NCAAM_TOURNAMENT_GROUP = 100

# Same as _TEAM_LOGO_URL_OVERRIDES in both scoreboard screens: ESPN's stock
# Iowa logo renders poorly, so the university's brand asset is used instead.
LOGO_URL_OVERRIDES: dict[str, str] = {
    "iowa": "https://brand.uiowa.edu/sites/brand.uiowa.edu/files/styles/widescreen__1920_x_1080/public/2020-05/Tigerhawk-gold%20on%20black%402x.png?h=e39f7b2b&itok=TdYKif5p",
}


@dataclass
class TeamResult:
    label: str
    filename: str
    ok: bool
    reason: str = ""


@dataclass
class RunSummary:
    mode: str
    out_dir: Path
    results: list[TeamResult] = field(default_factory=list)

    @property
    def succeeded(self) -> list[TeamResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[TeamResult]:
        return [r for r in self.results if not r.ok]


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def prompt_for_mode() -> str:
    print("Which set of logos do you want to download?")
    for key, mode in MODE_CHOICES.items():
        print(f"  {key}) {MODE_LABELS[mode]}")
    keys = ", ".join(MODE_CHOICES)
    while True:
        choice = input(f"Enter {keys} (or 'q' to quit): ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Cancelled.")
            raise SystemExit(0)
        if choice in MODE_CHOICES:
            return MODE_CHOICES[choice]
        print(f"Sorry, '{choice}' isn't a valid choice.")


def espn_urls(sport: str, endpoint: str) -> list[str]:
    """Return *endpoint* on every ESPN host, in the order the project tries them."""

    return [f"{host}{ESPN_SPORT_PATHS[sport]}/{endpoint}" for host in ESPN_HOSTS]


def _fetch_json(
    session: requests.Session,
    sport: str,
    endpoint: str,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return the first JSON object an ESPN host answers with; raise if none do."""

    errors: list[str] = []
    for url in espn_urls(sport, endpoint):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        return payload if isinstance(payload, dict) else {}
    raise requests.RequestException("; ".join(errors))


def _dump_debug(out_dir: Path, name: str, payload: Any) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    with contextlib.suppress(Exception):
        path.write_text(json.dumps(payload, indent=2)[:200_000])
    return path


def team_filename(team: dict[str, Any]) -> str:
    """Match screens/ncaa_fbs_scoreboard.py::_team_logo_filename exactly.

    *team* may be an ESPN team object or a scoreboard competitor that wraps
    one under ``"team"``.
    """

    team_blob = team.get("team") if isinstance(team.get("team"), dict) else team
    abbreviation = str(team_blob.get("abbreviation") or "").strip().upper()
    if abbreviation:
        return f"{abbreviation}.png"
    team_id = str(team_blob.get("id") or team.get("id") or "unknown").strip()
    return f"{team_id}.png"


def _team_label(team: dict[str, Any]) -> str:
    name = (
        team.get("displayName")
        or team.get("shortDisplayName")
        or team.get("location")
        or team.get("name")
    )
    abbrev = team.get("abbreviation")
    if name and abbrev:
        return f"{name} ({abbrev})"
    return str(name or abbrev or team.get("id") or "unknown team")


def resolve_logo_url(team: dict[str, Any]) -> Optional[str]:
    candidates = [
        str(team.get(key) or "").strip().lower()
        for key in ("abbreviation", "shortDisplayName", "displayName", "name", "location")
    ]
    for candidate in candidates:
        override = LOGO_URL_OVERRIDES.get(candidate)
        if override:
            return override

    logos = team.get("logos")
    if isinstance(logos, list):
        for logo in logos:
            if isinstance(logo, dict) and logo.get("href"):
                return str(logo["href"])
    logo = team.get("logo")
    if isinstance(logo, str) and logo.strip():
        return logo.strip()
    return None


def extract_rank(competitor: dict[str, Any]) -> Optional[int]:
    """Match screens/ncaa_fbs_scoreboard.py::_extract_rank (1-25, else None)."""

    rank = competitor.get("curatedRank")
    if isinstance(rank, dict):
        value = rank.get("current")
        try:
            parsed = int(value)
            return parsed if 1 <= parsed <= 25 else None
        except (TypeError, ValueError):
            pass
    for key in ("rank", "curatedRankCurrent"):
        try:
            parsed = int(competitor.get(key))
            return parsed if 1 <= parsed <= 25 else None
        except (TypeError, ValueError):
            continue
    return None


def scoreboard_date(now: Optional[datetime.datetime] = None) -> datetime.date:
    """Match screens/ncaa_fbs_scoreboard.py::_scoreboard_date (10:10 AM Central cutoff)."""

    if now is None:
        now = datetime.datetime.now(CENTRAL_TIME)
    cutoff = now.replace(hour=10, minute=10, second=0, microsecond=0)
    return (now.date() - datetime.timedelta(days=1)) if now < cutoff else now.date()


def week_dates(day: datetime.date) -> list[datetime.date]:
    """Return the Monday-Sunday week containing *day* (services/sports/ncaa_fbs.py)."""

    start = day - datetime.timedelta(days=day.weekday())
    return [start + datetime.timedelta(days=offset) for offset in range(7)]


def fetch_team_detail(
    session: requests.Session, sport: str, team_id: str
) -> Optional[dict[str, Any]]:
    try:
        payload = _fetch_json(session, sport, f"teams/{team_id}")
    except Exception:
        return None
    team = payload.get("team")
    return team if isinstance(team, dict) else None


def fetch_top25_teams(
    session: requests.Session, sport: str, *, debug_dir: Path
) -> list[dict[str, Any]]:
    payload = _fetch_json(session, sport, "rankings")
    rankings = payload.get("rankings")
    if not isinstance(rankings, list) or not rankings:
        debug_path = _dump_debug(debug_dir, "_debug_rankings_response.json", payload)
        raise RuntimeError(
            "ESPN's rankings response didn't include a 'rankings' list. "
            f"Saved the raw response to {debug_path} for a closer look."
        )

    def _poll_score(poll: dict[str, Any]) -> int:
        name = str(poll.get("name") or poll.get("shortName") or "").lower()
        poll_type = str(poll.get("type") or "").lower()
        if "ap" in poll_type or "ap top 25" in name:
            return 0
        return 1

    chosen = sorted((p for p in rankings if isinstance(p, dict)), key=_poll_score)
    if not chosen:
        raise RuntimeError("ESPN's rankings response had no usable poll entries.")
    poll = chosen[0]

    ranks = poll.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        debug_path = _dump_debug(debug_dir, "_debug_rankings_response.json", payload)
        raise RuntimeError(
            f"Poll '{poll.get('name')}' had no 'ranks' entries. "
            f"Saved the raw response to {debug_path}."
        )

    teams: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in ranks:
        if not isinstance(entry, dict):
            continue
        team = entry.get("team")
        if not isinstance(team, dict) or not team:
            continue
        team_id = str(team.get("id") or "")
        if team_id and team_id in seen_ids:
            continue
        if team_id:
            seen_ids.add(team_id)
        merged = dict(team)
        merged["_rank"] = entry.get("current")
        teams.append(merged)
    return teams


def _events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = payload.get("events")
    return (
        [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []
    )


def _competitors(event: dict[str, Any]) -> list[dict[str, Any]]:
    competitions = event.get("competitions") or [{}]
    comp = competitions[0] if isinstance(competitions, list) and competitions else {}
    if not isinstance(comp, dict):
        return []
    return [c for c in (comp.get("competitors") or []) if isinstance(c, dict)]


def fetch_fbs_week_teams(
    session: requests.Session,
    *,
    day: Optional[datetime.date] = None,
    debug_dir: Path,
) -> list[dict[str, Any]]:
    """Return both teams of every game this week that has a Top 25 team.

    Same games as services/sports/ncaa_fbs.py::fetch_scoreboard: the
    Monday-Sunday week of the board's scoreboard date, one day per request,
    ESPN ``groups=80`` (FBS), kept when either competitor is ranked.
    """

    days = week_dates(day or scoreboard_date())
    collected: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    last_payload: dict[str, Any] = {}
    for week_day in days:
        stamp = week_day.strftime("%Y%m%d")
        try:
            payload = _fetch_json(
                session, "football", "scoreboard", {"dates": stamp, "groups": FBS_GROUP}
            )
        except requests.RequestException as exc:
            print(f"  {week_day}: {exc}")
            failures.append(stamp)
            continue
        last_payload = payload
        for event in _events(payload):
            competitors = _competitors(event)
            if not any(extract_rank(c) is not None for c in competitors):
                continue
            for competitor in competitors:
                team = competitor.get("team")
                if not isinstance(team, dict) or not team:
                    continue
                key = team_filename(competitor)
                merged = collected.setdefault(key, dict(team))
                rank = extract_rank(competitor)
                if rank is not None:
                    merged["_rank"] = rank
    if len(failures) == len(days):
        raise requests.RequestException(f"every ESPN host failed for the week of {days[0]}")
    if not collected:
        _dump_debug(debug_dir, "_debug_fbs_week_response.json", last_payload)
    return list(collected.values())


def _extract_seed(competitor: dict[str, Any]) -> str:
    def _parse(value: Any) -> str:
        if isinstance(value, bool):
            return ""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed if trimmed.isdigit() else ""
        if isinstance(value, dict):
            for key in ("seed", "current", "value", "displayValue"):
                parsed = _parse(value.get(key))
                if parsed:
                    return parsed
        return ""

    for source in (
        competitor,
        competitor.get("team") if isinstance(competitor.get("team"), dict) else None,
    ):
        if not isinstance(source, dict):
            continue
        for key in ("seed", "tournamentSeed", "playoffSeed"):
            parsed = _parse(source.get(key))
            if parsed:
                return parsed
    return ""


def _is_tournament_event(event: dict[str, Any], competitors: list[dict[str, Any]]) -> bool:
    season = event.get("season")
    if isinstance(season, dict):
        try:
            if int(season.get("type")) == 3:
                return True
        except (TypeError, ValueError):
            pass
    text = " ".join(str(event.get(key, "")) for key in ("name", "shortName")).lower()
    if "ncaa" in text and ("tournament" in text or "first four" in text):
        return True
    if "march madness" in text:
        return True
    return any(_extract_seed(competitor) for competitor in competitors)


def _date_range(start_date: str, end_date: str) -> list[str]:
    start = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    end = datetime.datetime.strptime(end_date, "%Y%m%d").date()
    days = (end - start).days
    return [
        (start + datetime.timedelta(days=offset)).strftime("%Y%m%d") for offset in range(days + 1)
    ]


def fetch_tournament_teams(
    session: requests.Session,
    *,
    start_date: str,
    end_date: str,
    debug_dir: Path,
) -> list[dict[str, Any]]:
    # One day per request: site.web.api rejects ranged dates.
    collected: dict[str, dict[str, Any]] = {}
    stamps = _date_range(start_date, end_date)
    failures = 0
    last_payload: dict[str, Any] = {}
    for stamp in stamps:
        try:
            payload = _fetch_json(
                session,
                "basketball",
                "scoreboard",
                {"dates": stamp, "groups": NCAAM_TOURNAMENT_GROUP},
            )
        except requests.RequestException:
            failures += 1
            continue
        last_payload = payload
        for event in _events(payload):
            competitors = _competitors(event)
            if not _is_tournament_event(event, competitors):
                continue
            for competitor in competitors:
                team = competitor.get("team")
                if isinstance(team, dict) and team.get("id"):
                    merged = collected.setdefault(str(team["id"]), dict(team))
                    seed = _extract_seed(competitor)
                    if seed:
                        merged["_seed"] = seed

    if stamps and failures == len(stamps):
        raise requests.RequestException(f"every ESPN host failed for {start_date}-{end_date}")
    if not collected:
        _dump_debug(debug_dir, "_debug_tournament_response.json", last_payload)
    return list(collected.values())


def prepare_logo(data: bytes, max_dimension: int) -> Image.Image:
    """Trim transparent padding and cap the longest edge at *max_dimension*."""

    img = Image.open(io.BytesIO(data)).convert("RGBA")
    bbox = img.getchannel("A").getbbox()
    if bbox:
        img = img.crop(bbox)
    width, height = img.size
    if max(width, height) > max_dimension:
        ratio = max_dimension / float(max(width, height))
        new_size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        img = img.resize(new_size, RESAMPLE)
    return img


def download_logo(session: requests.Session, url: str, max_dimension: int) -> Image.Image:
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return prepare_logo(resp.content, max_dimension)


def process_teams(
    session: requests.Session,
    teams: list[dict[str, Any]],
    *,
    mode: str,
    out_dir: Path,
    max_dimension: int,
    dry_run: bool,
    overwrite: bool = False,
) -> RunSummary:
    sport = MODE_SPORTS[mode]
    summary = RunSummary(mode=mode, out_dir=out_dir)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    total = len(teams)
    seen_files: set[str] = set()
    for idx, team in enumerate(teams, start=1):
        label = _team_label(team)
        filename = team_filename(team)
        prefix = f"[{idx}/{total}]"

        url = resolve_logo_url(team)
        if not url or not team.get("abbreviation"):
            team_id = str(team.get("id") or "")
            detail = fetch_team_detail(session, sport, team_id) if team_id else None
            if detail:
                url = url or resolve_logo_url(detail)
                if not team.get("abbreviation") and detail.get("abbreviation"):
                    # The board names the file from the game data, which
                    # normally carries the abbreviation this detail has.
                    filename = team_filename(detail)

        if filename in seen_files:
            continue
        seen_files.add(filename)
        path = out_dir / filename

        if path.exists() and not overwrite:
            print(f"{prefix} {label} -> {filename} already saved, keeping it")
            summary.results.append(TeamResult(label, filename, ok=True, reason="kept existing"))
            continue

        if not url:
            print(f"{prefix} {label}: no logo URL found, skipping")
            summary.results.append(TeamResult(label, filename, ok=False, reason="no logo URL"))
            continue

        if dry_run:
            print(f"{prefix} {label} -> {filename}  (would fetch {url})")
            summary.results.append(TeamResult(label, filename, ok=True, reason="dry-run"))
            continue

        try:
            img = download_logo(session, url, max_dimension)
            img.save(path, format="PNG", optimize=True)
            print(f"{prefix} {label} -> {filename} ({img.width}x{img.height})")
            summary.results.append(TeamResult(label, filename, ok=True, reason="saved"))
        except Exception as exc:
            print(f"{prefix} {label}: failed to download/save logo: {exc}")
            summary.results.append(TeamResult(label, filename, ok=False, reason=str(exc)))
        time.sleep(0.1)  # be polite to ESPN's API

    return summary


def missing_league_logos(mode: str, logo_dir: Path = PROJECT_LOGO_DIR) -> list[str]:
    return [name for name in MODE_LEAGUE_LOGOS[mode] if not (logo_dir / name).exists()]


def print_summary(summary: RunSummary) -> None:
    saved = [r for r in summary.succeeded if r.reason == "saved"]
    kept = [r for r in summary.succeeded if r.reason == "kept existing"]
    print()
    print(f"Mode: {MODE_LABELS.get(summary.mode, summary.mode)}")
    print(f"Output folder: {summary.out_dir}")
    print(
        f"Saved {len(saved)} new logo(s), kept {len(kept)} already there, "
        f"of {len(summary.results)} team(s)."
    )
    if summary.failed:
        print("Skipped/failed:")
        for result in summary.failed:
            print(f"  - {result.label}: {result.reason}")
    missing = missing_league_logos(summary.mode) if PROJECT_LOGO_DIR.is_dir() else []
    if missing:
        print(f"Missing league logo(s) in {PROJECT_LOGO_DIR}: {', '.join(missing)}")
    if MODE_SPORTS[summary.mode] == "basketball":
        print(
            "Note: the NCAAM board draws team logos straight from ESPN, so it does not "
            "read these files yet; only its league logos come from images/ncaa."
        )


def default_tournament_dates(season_year: int) -> tuple[str, str]:
    # NCAA men's tournament (Selection Sunday through the championship) runs
    # roughly mid-March through the first weekend of April. Widen with
    # --start-date/--end-date if the bracket for a given year falls outside
    # this window, or if ESPN hasn't published it yet.
    start = datetime.date(season_year, 3, 10)
    end = datetime.date(season_year, 4, 10)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _parse_date(value: str) -> datetime.date:
    return datetime.datetime.strptime(value, "%Y%m%d").date()


def default_output_dir(mode: str) -> Path:
    """Return ~/Desktop/desk_display_logos/<mode>, the default review folder."""

    return DEFAULT_OUTPUT_BASE / mode


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        help="Which set of logos to download. Omit to be prompted interactively.",
    )
    parser.add_argument(
        "--output-dir",
        help="Folder to save logos in, used as is (for example images/ncaa, where the "
        "screens read them). Defaults to a folder per mode in "
        f"{DEFAULT_OUTPUT_BASE.name} on the Desktop.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace logos that are already saved. By default existing files are kept.",
    )
    parser.add_argument(
        "--week-of",
        type=_parse_date,
        help="Any date (YYYYMMDD) in the Monday-Sunday week to use (fbs_week mode only). "
        "Defaults to the week the FBS board is showing now.",
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=datetime.date.today().year,
        help="Calendar year for the default tournament date window (ncaam_tournament mode only).",
    )
    parser.add_argument(
        "--start-date",
        help="Override tournament search start date (YYYYMMDD). ncaam_tournament mode only.",
    )
    parser.add_argument(
        "--end-date",
        help="Override tournament search end date (YYYYMMDD). ncaam_tournament mode only.",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=TEAM_LOGO_MAX_DIMENSION,
        help=f"Max edge length in pixels for saved logos (default {TEAM_LOGO_MAX_DIMENSION}, "
        "matching the project's team-logo policy).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the teams and filenames that would be produced, without downloading anything.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    mode = args.mode or prompt_for_mode()
    out_dir = (
        default_output_dir(mode) if args.output_dir is None else Path(args.output_dir).expanduser()
    )
    session = build_session()

    try:
        if mode == MODE_FBS_WEEK:
            print(f"Fetching {MODE_LABELS[mode]} from ESPN...")
            teams = fetch_fbs_week_teams(session, day=args.week_of, debug_dir=out_dir)
        elif mode in (MODE_FBS_TOP25, MODE_NCAAM_TOP25):
            print(f"Fetching {MODE_LABELS[mode]} from ESPN...")
            teams = fetch_top25_teams(session, MODE_SPORTS[mode], debug_dir=out_dir)
        else:
            if args.start_date and args.end_date:
                start_date, end_date = args.start_date, args.end_date
            else:
                start_date, end_date = default_tournament_dates(args.season_year)
            print(
                f"Fetching {MODE_LABELS[mode]} from ESPN "
                f"(searching {start_date}-{end_date} for tournament games)..."
            )
            teams = fetch_tournament_teams(
                session, start_date=start_date, end_date=end_date, debug_dir=out_dir
            )
            if not teams:
                print(
                    "No tournament games found in that date range. This usually means the "
                    "bracket hasn't been announced yet, or the window needs adjusting -- try "
                    "--season-year, or --start-date/--end-date (format YYYYMMDD)."
                )
                return 1
    except requests.RequestException as exc:
        print(f"Network error talking to ESPN: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    if not teams:
        print("No teams found; nothing to download.")
        return 1

    print(f"Found {len(teams)} team(s). Saving logos to {out_dir}\n")
    summary = process_teams(
        session,
        teams,
        mode=mode,
        out_dir=out_dir,
        max_dimension=args.max_dimension,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print_summary(summary)
    return 0 if summary.succeeded or args.dry_run else 1


if __name__ == "__main__":
    bootstrap_dependencies()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130) from None
