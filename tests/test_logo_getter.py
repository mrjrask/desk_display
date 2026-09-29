"""scripts/logo_getter.py saves the logos the NCAA scoreboards look for."""

from __future__ import annotations

import datetime
import io
import os
from pathlib import Path

import pytest
import requests
from PIL import Image

import screens.ncaa_fbs_scoreboard as fbs
import screens.ncaam_scoreboard as ncaam
from scripts import logo_getter as lg


class FakeResponse:
    def __init__(self, status: int = 200, payload=None, content: bytes = b""):
        self.status_code = status
        self._payload = payload
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Forbidden")

    def json(self):
        return self._payload


class FakeSession:
    """Answers GETs from a routing function and records every call."""

    def __init__(self, route):
        self.route = route
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.route(url, params)


def png_bytes(size=(300, 200), box=(50, 40, 250, 160)) -> bytes:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    img.paste((200, 30, 30, 255), box)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def competitor(abbr, team_id, rank=99, home_away="home"):
    return {
        "homeAway": home_away,
        "curatedRank": {"current": rank},
        "team": {
            "id": team_id,
            "abbreviation": abbr,
            "displayName": abbr.title(),
            "logo": f"https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png",
        },
    }


# ── Parity with the screens ────────────────────────────────────────────────


def test_writes_to_the_folder_the_screens_read():
    assert Path(fbs.LOGO_DIR).resolve() == lg.PROJECT_LOGO_DIR
    assert Path(ncaam.LOGO_DIR).resolve() == lg.PROJECT_LOGO_DIR


def test_default_output_is_a_desktop_folder_per_mode():
    assert lg.parse_args([]).output_dir is None
    assert (
        lg.default_output_dir("fbs_week")
        == Path.home() / "Desktop" / "desk_display_logos" / "fbs_week"
    )


@pytest.mark.parametrize(
    "team",
    [
        competitor("lsu", "99"),
        {"team": {"abbreviation": " Tex ", "id": "251"}},
        {"team": {"id": "2294"}},
        {"id": "333", "team": {}},
        {"abbreviation": "uga", "id": "61"},
        {"id": "61"},
        {},
    ],
)
def test_filename_matches_the_fbs_board(team):
    assert lg.team_filename(team) == fbs._team_logo_filename(team)


def test_downloaded_logo_size_matches_the_fbs_board():
    assert lg.TEAM_LOGO_MAX_DIMENSION == fbs.AUTO_LOGO_MAX_DIMENSION


def test_logo_overrides_match_both_boards():
    assert lg.LOGO_URL_OVERRIDES == fbs._TEAM_LOGO_URL_OVERRIDES
    assert lg.LOGO_URL_OVERRIDES == ncaam._TEAM_LOGO_URL_OVERRIDES


@pytest.mark.parametrize(
    "comp",
    [
        {"curatedRank": {"current": 7}},
        {"curatedRank": {"current": 99}},
        {"rank": "12"},
        {"rank": 0},
        {},
    ],
)
def test_rank_matches_the_fbs_board(comp):
    assert lg.extract_rank(comp) == fbs._extract_rank(comp)


@pytest.mark.parametrize("hour,minute", [(9, 0), (10, 9), (10, 10), (23, 59)])
def test_scoreboard_date_matches_the_fbs_board(hour, minute):
    now = datetime.datetime(2026, 9, 28, hour, minute, tzinfo=lg.CENTRAL_TIME)
    assert lg.scoreboard_date(now) == fbs._scoreboard_date(now)


def test_espn_hosts_match_the_fbs_board():
    assert lg.espn_urls("football", "scoreboard") == [fbs.ESPN_URL, *fbs.ESPN_FALLBACK_URLS][:2]


def test_league_logos_match_the_boards(monkeypatch):
    assert fbs._mode_title_and_logo()[1] + ".png" in lg.MODE_LEAGUE_LOGOS[lg.MODE_FBS_WEEK]
    monkeypatch.setattr(ncaam, "NCAAM_SCOREBOARD_MODE", "top25")
    assert ncaam._mode_title_and_logo()[1] + ".png" in lg.MODE_LEAGUE_LOGOS[lg.MODE_NCAAM_TOP25]
    monkeypatch.setattr(ncaam, "NCAAM_SCOREBOARD_MODE", "tournament")
    assert (
        ncaam._mode_title_and_logo()[1] + ".png" in lg.MODE_LEAGUE_LOGOS[lg.MODE_NCAAM_TOURNAMENT]
    )


# ── ESPN fetching ──────────────────────────────────────────────────────────


def test_falls_back_to_site_api_when_site_web_api_refuses():
    def route(url, params):
        if url.startswith("https://site.web.api.espn.com"):
            return FakeResponse(403)
        return FakeResponse(
            payload={
                "rankings": [{"name": "AP Top 25", "ranks": [{"current": 1, "team": {"id": "1"}}]}]
            }
        )

    session = FakeSession(route)
    teams = lg.fetch_top25_teams(session, "football", debug_dir=Path("unused"))
    assert [t["id"] for t in teams] == ["1"]
    assert [url.split("/apis")[0] for url, _ in session.calls] == list(lg.ESPN_HOSTS)


def test_fbs_week_collects_both_teams_of_ranked_games_one_day_at_a_time(tmp_path):
    games_by_day = {
        "20260926": [
            {
                "competitions": [
                    {
                        "competitors": [
                            competitor("uga", "61", 3),
                            competitor("ksu", "2306", 99, "away"),
                        ]
                    }
                ]
            },
            {
                "competitions": [
                    {
                        "competitors": [
                            competitor("army", "349"),
                            competitor("navy", "2426", 99, "away"),
                        ]
                    }
                ]
            },
        ],
        "20260927": [
            {
                "competitions": [
                    {
                        "competitors": [
                            competitor("uga", "61", 3),
                            competitor("lsu", "99", 12, "away"),
                        ]
                    }
                ]
            },
        ],
    }

    def route(url, params):
        assert "limit" not in params and "-" not in params["dates"]
        return FakeResponse(payload={"events": games_by_day.get(params["dates"], [])})

    session = FakeSession(route)
    teams = lg.fetch_fbs_week_teams(session, day=datetime.date(2026, 9, 27), debug_dir=tmp_path)

    assert sorted(lg.team_filename(t) for t in teams) == ["KSU.png", "LSU.png", "UGA.png"]
    assert [params["dates"] for _, params in session.calls] == [
        f"202609{d}" for d in ("21", "22", "23", "24", "25", "26", "27")
    ]
    assert {params["groups"] for _, params in session.calls} == {lg.FBS_GROUP}
    assert all(url == fbs.ESPN_URL for url, _ in session.calls)


def test_fbs_week_fails_only_when_every_day_fails(tmp_path):
    session = FakeSession(lambda url, params: FakeResponse(403))
    with pytest.raises(requests.RequestException):
        lg.fetch_fbs_week_teams(session, day=datetime.date(2026, 9, 27), debug_dir=tmp_path)


def test_tournament_asks_one_day_at_a_time(tmp_path):
    event = {
        "season": {"type": 3},
        "competitions": [
            {"competitors": [competitor("duke", "150"), competitor("conn", "41", 99, "away")]}
        ],
    }

    def route(url, params):
        return FakeResponse(payload={"events": [event] if params["dates"] == "20270320" else []})

    session = FakeSession(route)
    teams = lg.fetch_tournament_teams(
        session, start_date="20270319", end_date="20270321", debug_dir=tmp_path
    )
    assert sorted(t["abbreviation"] for t in teams) == ["conn", "duke"]
    assert [params["dates"] for _, params in session.calls] == ["20270319", "20270320", "20270321"]


# ── Saving ─────────────────────────────────────────────────────────────────


def test_prepare_logo_trims_padding_and_caps_the_edge():
    img = lg.prepare_logo(png_bytes(), 128)
    assert img.size == (128, 77)
    assert img.getchannel("A").getbbox() == (0, 0, 128, 77)
    small = lg.prepare_logo(png_bytes(size=(60, 60), box=(10, 10, 50, 30)), 128)
    assert small.size == (40, 20)


def test_saved_logo_is_found_by_the_fbs_board(tmp_path, monkeypatch):
    session = FakeSession(lambda url, params: FakeResponse(content=png_bytes()))
    team = competitor("miss", "145")["team"]
    summary = lg.process_teams(
        session, [team], mode=lg.MODE_FBS_WEEK, out_dir=tmp_path, max_dimension=128, dry_run=False
    )
    assert [r.filename for r in summary.succeeded] == ["MISS.png"]

    monkeypatch.setattr(fbs, "LOGO_DIR", str(tmp_path))
    monkeypatch.setattr(fbs, "_REMOTE_LOGO_CACHE", {})
    assert fbs._load_team_logo({"team": team}, 30) is not None


def test_existing_logos_are_kept_unless_overwrite(tmp_path):
    existing = tmp_path / "UGA.png"
    existing.write_bytes(b"hand tuned")
    session = FakeSession(lambda url, params: FakeResponse(content=png_bytes()))
    team = competitor("uga", "61")["team"]

    summary = lg.process_teams(
        session, [team], mode=lg.MODE_FBS_WEEK, out_dir=tmp_path, max_dimension=128, dry_run=False
    )
    assert summary.results[0].reason == "kept existing"
    assert existing.read_bytes() == b"hand tuned"
    assert session.calls == []

    lg.process_teams(
        session,
        [team],
        mode=lg.MODE_FBS_WEEK,
        out_dir=tmp_path,
        max_dimension=128,
        dry_run=False,
        overwrite=True,
    )
    assert Image.open(existing).size == (128, 77)


def test_iowa_uses_the_brand_logo(tmp_path):
    session = FakeSession(lambda url, params: FakeResponse(content=png_bytes()))
    team = {"id": "2294", "abbreviation": "IOWA", "logo": "https://a.espncdn.com/iowa.png"}
    lg.process_teams(
        session, [team], mode=lg.MODE_FBS_TOP25, out_dir=tmp_path, max_dimension=128, dry_run=False
    )
    assert session.calls[0][0] == lg.LOGO_URL_OVERRIDES["iowa"]
    assert (tmp_path / "IOWA.png").exists()


def test_dry_run_writes_nothing(tmp_path):
    out = tmp_path / "logos"
    session = FakeSession(lambda url, params: pytest.fail("dry run downloaded"))
    summary = lg.process_teams(
        session,
        [competitor("uga", "61")["team"]],
        mode=lg.MODE_FBS_WEEK,
        out_dir=out,
        max_dimension=128,
        dry_run=True,
    )
    assert summary.results[0].reason == "dry-run"
    assert not out.exists()


def test_main_fills_the_output_folder(tmp_path, monkeypatch, capsys):
    events = {
        "events": [
            {
                "competitions": [
                    {"competitors": [competitor("byu", "252", 20), competitor("utah", "254")]}
                ]
            }
        ]
    }

    def route(url, params):
        if "scoreboard" in url:
            return FakeResponse(payload=events)
        return FakeResponse(content=png_bytes())

    monkeypatch.setattr(lg, "build_session", lambda: FakeSession(route))
    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    assert (
        lg.main(["--mode", "fbs_week", "--output-dir", str(tmp_path), "--week-of", "20260927"]) == 0
    )
    assert sorted(os.listdir(tmp_path)) == ["BYU.png", "UTAH.png"]
    assert "Saved 2 new logo(s)" in capsys.readouterr().out


# ── Dependencies ───────────────────────────────────────────────────────────


def test_missing_packages_reports_pip_names(monkeypatch):
    real = lg.importlib.util.find_spec
    monkeypatch.setattr(
        lg.importlib.util, "find_spec", lambda name: None if name == "PIL" else real(name)
    )
    assert lg.missing_packages() == ["Pillow"]


def test_temp_venv_runs_the_script_and_is_deleted(tmp_path, monkeypatch):
    venv_dir = tmp_path / "venv"
    venv_dir.mkdir()
    monkeypatch.setattr(lg.tempfile, "mkdtemp", lambda prefix: str(venv_dir))
    monkeypatch.setattr(
        lg.venv, "EnvBuilder", lambda with_pip: type("B", (), {"create": lambda self, d: None})()
    )
    calls = []

    def fake_run(cmd, env=None, check=False):
        calls.append((cmd, env))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(lg.subprocess, "run", fake_run)
    assert lg.run_in_temp_venv(["--mode", "fbs_week"], ["Pillow"]) == 0
    (install, _), (run, env) = calls
    assert install[1:] == [
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        "Pillow",
    ]
    assert run[1:] == [str(Path(lg.__file__).resolve()), "--mode", "fbs_week"]
    assert env[lg.TEMP_VENV_ENV] == "1"
    assert not venv_dir.exists()
