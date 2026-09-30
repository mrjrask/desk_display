import datetime as dt
import io
import logging
import os

import pytest
from PIL import Image

from screens import ncaa_fbs_scoreboard


@pytest.fixture(autouse=True)
def _isolated_logo_dirs(monkeypatch, tmp_path):
    """Keep downloaded logos out of the checkout and start every test cold."""

    monkeypatch.setattr(ncaa_fbs_scoreboard, "AUTO_LOGO_DIR", str(tmp_path / "auto"))
    ncaa_fbs_scoreboard._DOWNLOAD_FAILURES.clear()
    ncaa_fbs_scoreboard._REMOTE_LOGO_CACHE.clear()
    ncaa_fbs_scoreboard._LOGO_MISSES.clear()


def test_fetch_uses_fbs_group_and_keeps_only_top_25_games(monkeypatch):
    captured = {}

    def fake_fetch(params, url=ncaa_fbs_scoreboard.ESPN_URL):
        captured.update(params)
        return {
            "events": [
                {"id": "ranked", "competitions": [{"competitors": [{"curatedRank": {"current": 8}}]}]},
                {"id": "unranked", "competitions": [{"competitors": [{"curatedRank": {"current": 99}}]}]},
            ]
        }

    monkeypatch.setattr(ncaa_fbs_scoreboard, "_fetch_json", fake_fetch)

    games = ncaa_fbs_scoreboard._fetch_games_for_date(dt.date(2026, 9, 26))

    assert captured == {"dates": "20260926", "groups": 80}
    assert [game["id"] for game in games] == ["ranked"]


def test_forbidden_primary_host_falls_back_to_other_espn_hosts(monkeypatch):
    requested = []
    ranked = {"id": "ranked", "competitions": [{"competitors": [{"curatedRank": {"current": 3}}]}]}

    def fake_fetch(params, url=ncaa_fbs_scoreboard.ESPN_URL):
        requested.append(url)
        if "cdn.espn.com" not in url:
            raise RuntimeError("403 Client Error: Forbidden")
        return {"content": {"sbData": {"events": [ranked]}}}

    monkeypatch.setattr(ncaa_fbs_scoreboard, "_fetch_json", fake_fetch)

    games = ncaa_fbs_scoreboard.fetch_games_for_day(dt.date(2026, 9, 26))

    assert requested == [ncaa_fbs_scoreboard.ESPN_URL, *ncaa_fbs_scoreboard.ESPN_FALLBACK_URLS]
    assert [game["id"] for game in games] == ["ranked"]


def test_cdn_request_adds_xhr_flag(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"content": {"sbData": {"events": []}}}

    def fake_get(url, params=None, timeout=None):
        captured.update(url=url, params=params)
        return Response()

    monkeypatch.setattr(ncaa_fbs_scoreboard._SESSION, "get", fake_get)

    ncaa_fbs_scoreboard._fetch_json({"dates": "20260926"}, ncaa_fbs_scoreboard.ESPN_FALLBACK_URLS[-1])

    assert captured["params"] == {"dates": "20260926", "xhr": 1}


def test_all_hosts_failing_raises(monkeypatch):
    def fake_fetch(params, url=ncaa_fbs_scoreboard.ESPN_URL):
        raise RuntimeError("403 Client Error: Forbidden")

    monkeypatch.setattr(ncaa_fbs_scoreboard, "_fetch_json", fake_fetch)

    try:
        ncaa_fbs_scoreboard.fetch_games_for_day(dt.date(2026, 9, 26))
    except RuntimeError as exc:
        assert "Forbidden" in str(exc)
    else:  # pragma: no cover - the fetch must not look like an empty week
        raise AssertionError("expected RuntimeError")


def test_missing_team_logo_logs_expected_filename(monkeypatch, caplog):
    monkeypatch.setattr(ncaa_fbs_scoreboard.os.path, "exists", lambda path: False)
    ncaa_fbs_scoreboard._REMOTE_LOGO_CACHE.clear()

    with caplog.at_level(logging.WARNING):
        logo = ncaa_fbs_scoreboard._load_team_logo(
            {"team": {"abbreviation": "MINN"}}, 32
        )

    assert logo is None
    assert "expected filename: MINN.png" in caplog.text
    assert ncaa_fbs_scoreboard._team_abbreviation(
        {"team": {"abbreviation": "MINN"}}
    ) == "MINN"


def test_team_abbreviation_matches_id_based_expected_filename():
    team = {"team": {"id": "123", "abbreviation": ""}}

    assert ncaa_fbs_scoreboard._team_logo_filename(team) == "123.png"
    assert ncaa_fbs_scoreboard._team_abbreviation(team) == "123"


def _game(game_id, date, *, state="post", away_rank=None, home_rank=None):
    def team(abbr, rank):
        blob = {"team": {"abbreviation": abbr}, "score": "10"}
        if rank is not None:
            blob["curatedRank"] = {"current": rank}
        return blob

    return {
        "id": game_id,
        "date": date,
        "status": {"type": {"state": state, "completed": state == "post", "shortDetail": "Final"}},
        "teams": {"away": team("AAA", away_rank), "home": team("BBB", home_rank)},
    }


def test_primary_host_is_site_web_api():
    assert ncaa_fbs_scoreboard.ESPN_URL.startswith("https://site.web.api.espn.com/")


def test_sunday_shows_the_weekends_games_one_day_per_request(monkeypatch):
    from services.sports import ncaa_fbs

    days = {
        dt.date(2026, 9, 26): [_game("sat-late", "2026-09-27T01:00Z"), _game("sat", "2026-09-26T16:00Z")],
        dt.date(2026, 9, 25): [_game("fri", "2026-09-26T00:00Z")],
    }
    requested = []

    def fake_fetch(day):
        requested.append(day)
        return days.get(day, [])

    monkeypatch.setattr(ncaa_fbs, "fetch_games_for_day", fake_fetch)
    ncaa_fbs._FINAL_DAY_CACHE.clear()

    sunday = dt.datetime(2026, 9, 27, 12, 0, tzinfo=ncaa_fbs.CENTRAL_TIME)
    games = ncaa_fbs.fetch_scoreboard(now=sunday)

    assert requested == [dt.date(2026, 9, 21) + dt.timedelta(days=offset) for offset in range(7)]
    assert [game["id"] for game in games] == ["fri", "sat", "sat-late"]


def test_games_are_ordered_by_best_ranked_team(monkeypatch):
    from services.sports import ncaa_fbs

    games_by_day = {
        dt.date(2026, 10, 1): [_game("thu-12v20", "2026-10-01T23:00Z", away_rank=20, home_rank=12)],
        dt.date(2026, 10, 3): [
            _game("sat-3", "2026-10-03T16:00Z", home_rank=3),
            _game("sat-1", "2026-10-03T20:00Z", away_rank=1),
            _game("sat-12v7", "2026-10-03T19:00Z", away_rank=7, home_rank=12),
            _game("sat-3-late", "2026-10-03T23:00Z", away_rank=3),
            _game("sat-none", "2026-10-03T15:00Z"),
            _game("sat-12", "2026-10-03T17:00Z", away_rank=12),
        ],
    }
    monkeypatch.setattr(ncaa_fbs, "fetch_games_for_day", lambda day: games_by_day.get(day, []))
    ncaa_fbs._FINAL_DAY_CACHE.clear()
    friday = dt.datetime(2026, 10, 2, 12, 0, tzinfo=ncaa_fbs.CENTRAL_TIME)

    games = ncaa_fbs.fetch_scoreboard(now=friday)

    assert [game["id"] for game in games] == [
        "sat-1",
        "sat-3",
        "sat-3-late",
        "sat-12v7",
        "thu-12v20",
        "sat-12",
        "sat-none",
    ]


def test_finished_past_days_are_not_refetched(monkeypatch):
    from services.sports import ncaa_fbs

    calls = []

    def fake_fetch(day):
        calls.append(day)
        if day == dt.date(2026, 10, 1):
            return [_game("thu", "2026-10-01T23:00Z")]
        if day == dt.date(2026, 10, 3):
            return [_game("sat", "2026-10-03T16:00Z", state="pre")]
        return []

    monkeypatch.setattr(ncaa_fbs, "fetch_games_for_day", fake_fetch)
    ncaa_fbs._FINAL_DAY_CACHE.clear()
    friday = dt.datetime(2026, 10, 2, 12, 0, tzinfo=ncaa_fbs.CENTRAL_TIME)

    ncaa_fbs.fetch_scoreboard(now=friday)
    calls.clear()
    games = ncaa_fbs.fetch_scoreboard(now=friday)

    assert dt.date(2026, 10, 1) not in calls
    assert dt.date(2026, 10, 3) in calls
    assert [game["id"] for game in games] == ["thu", "sat"]


def test_week_matches_espn_monday_to_sunday_calendar():
    from services.sports import ncaa_fbs

    assert ncaa_fbs.week_dates(dt.date(2026, 9, 28)) == [
        dt.date(2026, 9, 28) + dt.timedelta(days=offset) for offset in range(7)
    ]
    assert ncaa_fbs.week_start_for_date(dt.date(2026, 10, 4)) == dt.date(2026, 9, 28)


def test_week_rolls_over_on_monday_morning_cutoff():
    from services.sports import ncaa_fbs

    before = dt.datetime(2026, 9, 28, 9, 0, tzinfo=ncaa_fbs.CENTRAL_TIME)
    after = dt.datetime(2026, 9, 28, 11, 0, tzinfo=ncaa_fbs.CENTRAL_TIME)

    assert ncaa_fbs.week_start_for_date(ncaa_fbs.scoreboard_date(before)) == dt.date(2026, 9, 21)
    assert ncaa_fbs.week_start_for_date(ncaa_fbs.scoreboard_date(after)) == dt.date(2026, 9, 28)


def test_failed_fetch_keeps_last_good_games_for_the_same_week(monkeypatch):
    from services.sports import ncaa_fbs

    fail = {"on": False}

    def fake_fetch(day):
        if fail["on"]:
            raise RuntimeError("403 Client Error: Forbidden")
        return [_game("sat", "2026-10-03T16:00Z", state="pre")] if day == dt.date(2026, 10, 3) else []

    monkeypatch.setattr(ncaa_fbs, "fetch_games_for_day", fake_fetch)
    ncaa_fbs._LAST_GOOD.clear()
    ncaa_fbs._FINAL_DAY_CACHE.clear()
    friday = dt.datetime(2026, 10, 2, 12, 0, tzinfo=ncaa_fbs.CENTRAL_TIME)
    next_week = dt.datetime(2026, 10, 6, 12, 0, tzinfo=ncaa_fbs.CENTRAL_TIME)

    assert [game["id"] for game in ncaa_fbs.fetch_scoreboard(now=friday)] == ["sat"]
    fail["on"] = True
    assert [game["id"] for game in ncaa_fbs.fetch_scoreboard(now=friday)] == ["sat"]
    assert ncaa_fbs.fetch_scoreboard(now=next_week) == []


def test_rank_is_drawn_as_superscript_before_logo():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (200, 100), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    x_logo, y_logo, logo_w, logo_h = 120, 30, 40, 40

    ncaa_fbs_scoreboard._draw_rank(draw, 7, x_logo, y_logo, logo_w, logo_h)

    bbox = img.getbbox()
    assert bbox is not None
    left, top, right, bottom = bbox
    assert right <= x_logo
    assert top >= y_logo
    assert bottom <= y_logo + logo_h // 2


def test_unranked_team_gets_no_rank():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (200, 100), (0, 0, 0))
    ncaa_fbs_scoreboard._draw_rank(ImageDraw.Draw(img), None, 120, 30, 40, 40)

    assert img.getbbox() is None


def _solid_logo(width, height, *, pad=0):
    from PIL import Image

    img = Image.new("RGBA", (width + 2 * pad, height + 2 * pad), (0, 0, 0, 0))
    img.paste(Image.new("RGBA", (width, height), (200, 0, 0, 255)), (pad, pad))
    return img


def test_fit_logo_trims_padding_and_matches_area_across_shapes():
    area = 80 * 80
    square = ncaa_fbs_scoreboard._fit_logo(_solid_logo(50, 50, pad=40), 200, 120, area)
    wide = ncaa_fbs_scoreboard._fit_logo(_solid_logo(300, 100), 200, 120, area)

    assert square.size == (80, 80)
    assert abs(wide.width * wide.height - area) / area < 0.05
    assert wide.height < square.height


def test_fit_logo_never_exceeds_the_box():
    fitted = ncaa_fbs_scoreboard._fit_logo(_solid_logo(600, 100), 150, 120, 120 * 120)

    assert fitted.width <= 150 and fitted.height <= 120


def test_missing_local_logo_uses_espn_logo_from_payload(monkeypatch):
    import io

    buf = io.BytesIO()
    _solid_logo(40, 40).save(buf, format="PNG")

    class Response:
        content = buf.getvalue()

        def raise_for_status(self):
            return None

    requested = []

    def fake_get(url, timeout=None):
        requested.append(url)
        return Response()

    monkeypatch.setattr(ncaa_fbs_scoreboard.os.path, "exists", lambda path: False)
    monkeypatch.setattr(ncaa_fbs_scoreboard._SESSION, "get", fake_get)
    ncaa_fbs_scoreboard._REMOTE_LOGO_CACHE.clear()
    ncaa_fbs_scoreboard._LOGO_MISSES.clear()
    team = {"team": {"abbreviation": "UNC", "logo": "https://a.espncdn.com/i/teamlogos/ncaa/500/153.png"}}

    logo = ncaa_fbs_scoreboard._load_team_logo(team, 60, 90, 50 * 50)

    assert requested == ["https://a.espncdn.com/i/teamlogos/ncaa/500/153.png"]
    assert logo is not None and logo.size == (50, 50)


class _LogoResponse:
    def __init__(self, img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.content = buf.getvalue()

    def raise_for_status(self):
        return None


def _padded_logo(size, mark):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = (size - mark) // 2
    img.paste(Image.new("RGBA", (mark, mark), (200, 0, 0, 255)), (offset, offset))
    return img


def _week_game(away_abbr, home_abbr, **away_extra):
    away = {"team": {"abbreviation": away_abbr, "logo": f"https://espn.test/{away_abbr}.png", **away_extra}}
    home = {"team": {"abbreviation": home_abbr, "logo": f"https://espn.test/{home_abbr}.png"}}
    return {"id": f"{away_abbr}-{home_abbr}", "teams": {"away": away, "home": home}}


def test_download_missing_team_logos_saves_only_missing_trimmed_and_capped(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    Image.new("RGBA", (20, 20), (0, 0, 255, 255)).save(repo_dir / "OSU.png")
    monkeypatch.setattr(ncaa_fbs_scoreboard, "LOGO_DIR", str(repo_dir))
    requested = []

    def fake_get(url, timeout=None):
        requested.append(url)
        return _LogoResponse(_padded_logo(500, 400))

    monkeypatch.setattr(ncaa_fbs_scoreboard._SESSION, "get", fake_get)

    saved = ncaa_fbs_scoreboard.download_missing_team_logos(
        [_week_game("OSU", "MICH"), _week_game("MICH", "PSU")]
    )

    assert saved == ["MICH.png", "PSU.png"]
    assert requested == ["https://espn.test/MICH.png", "https://espn.test/PSU.png"]
    auto_dir = ncaa_fbs_scoreboard.AUTO_LOGO_DIR
    assert sorted(os.listdir(auto_dir)) == ["MICH.png", "PSU.png"]
    with Image.open(os.path.join(auto_dir, "MICH.png")) as img:
        assert img.size == (128, 128)
    # The committed logo is never replaced.
    with Image.open(repo_dir / "OSU.png") as img:
        assert img.size == (20, 20)

    # A second refresh finds everything saved and asks ESPN for nothing.
    requested.clear()
    assert ncaa_fbs_scoreboard.download_missing_team_logos([_week_game("OSU", "MICH")]) == []
    assert requested == []


def test_download_missing_team_logos_backs_off_after_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(ncaa_fbs_scoreboard, "LOGO_DIR", str(tmp_path / "repo"))
    calls = []

    def failing_get(url, timeout=None):
        calls.append(url)
        raise RuntimeError("403 Client Error: Forbidden")

    monkeypatch.setattr(ncaa_fbs_scoreboard._SESSION, "get", failing_get)

    games = [_week_game("UNC", "DUKE")]
    assert ncaa_fbs_scoreboard.download_missing_team_logos(games) == []
    assert ncaa_fbs_scoreboard.download_missing_team_logos(games) == []
    assert len(calls) == 2


def test_board_reads_downloaded_logo_without_network(monkeypatch, tmp_path):
    monkeypatch.setattr(ncaa_fbs_scoreboard, "LOGO_DIR", str(tmp_path / "repo"))
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir()
    Image.new("RGBA", (40, 40), (0, 200, 0, 255)).save(auto_dir / "UNC.png")

    def no_network(url, timeout=None):  # pragma: no cover - must not be called
        raise AssertionError("downloaded logo should be read from disk")

    monkeypatch.setattr(ncaa_fbs_scoreboard._SESSION, "get", no_network)

    logo = ncaa_fbs_scoreboard._load_team_logo({"team": {"abbreviation": "UNC"}}, 60, 90, 50 * 50)

    assert logo is not None and logo.size == (50, 50)


def test_logo_url_uses_team_override(monkeypatch):
    team = {"team": {"abbreviation": "IOWA", "logo": "https://espn.test/IOWA.png"}}

    assert ncaa_fbs_scoreboard._team_logo_url(team) == ncaa_fbs_scoreboard._TEAM_LOGO_URL_OVERRIDES["iowa"]


def test_league_logo_matches_team_logo_height_on_1080p(monkeypatch):
    # The league logo used to multiply in the display scale twice at 1080p,
    # drawing an NCAA logo thousands of pixels tall over the whole screen.
    monkeypatch.setattr(ncaa_fbs_scoreboard, "HDMI_1080P_LAYOUT", True)
    monkeypatch.setattr(ncaa_fbs_scoreboard, "_team_logo_height", lambda: 246)
    assert ncaa_fbs_scoreboard._league_logo_height() == 246
