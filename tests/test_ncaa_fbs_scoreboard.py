import datetime as dt
import logging

from screens import ncaa_fbs_scoreboard


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
