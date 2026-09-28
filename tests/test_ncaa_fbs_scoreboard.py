import datetime as dt
import logging

from screens import ncaa_fbs_scoreboard


def test_fetch_uses_fbs_group_and_keeps_only_top_25_games(monkeypatch):
    captured = {}

    def fake_fetch(params):
        captured.update(params)
        return {
            "events": [
                {"id": "ranked", "competitions": [{"competitors": [{"curatedRank": {"current": 8}}]}]},
                {"id": "unranked", "competitions": [{"competitors": [{"curatedRank": {"current": 99}}]}]},
            ]
        }

    monkeypatch.setattr(ncaa_fbs_scoreboard, "_fetch_json", fake_fetch)

    games = ncaa_fbs_scoreboard._fetch_games_for_date(dt.date(2026, 9, 26))

    assert captured == {"dates": "20260926", "limit": 300, "groups": 80}
    assert [game["id"] for game in games] == ["ranked"]


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


def test_sunday_shows_the_weekends_games(monkeypatch):
    from services.sports import ncaa_fbs

    days = {
        dt.date(2026, 9, 26): [_game("sat-late", "2026-09-27T01:00Z"), _game("sat", "2026-09-26T16:00Z")],
        dt.date(2026, 9, 25): [_game("fri", "2026-09-26T00:00Z")],
    }
    requested = []

    def fake_fetch(day):
        requested.append(day)
        return days.get(day, [])

    monkeypatch.setattr(ncaa_fbs, "_fetch_games_for_date", fake_fetch)
    ncaa_fbs._FINAL_DAY_CACHE.clear()

    sunday = dt.datetime(2026, 9, 27, 12, 0, tzinfo=ncaa_fbs.CENTRAL_TIME)
    games = ncaa_fbs.fetch_scoreboard(now=sunday)

    assert requested[0] == dt.date(2026, 9, 21)
    assert requested[-1] == dt.date(2026, 9, 27)
    assert [game["id"] for game in games] == ["fri", "sat", "sat-late"]


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

    monkeypatch.setattr(ncaa_fbs, "_fetch_games_for_date", fake_fetch)
    ncaa_fbs._FINAL_DAY_CACHE.clear()
    friday = dt.datetime(2026, 10, 2, 12, 0, tzinfo=ncaa_fbs.CENTRAL_TIME)

    ncaa_fbs.fetch_scoreboard(now=friday)
    calls.clear()
    games = ncaa_fbs.fetch_scoreboard(now=friday)

    assert dt.date(2026, 10, 1) not in calls
    assert dt.date(2026, 10, 3) in calls
    assert [game["id"] for game in games] == ["thu", "sat"]


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
