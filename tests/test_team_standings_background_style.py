import screens.mlb_team_standings as mlb_team_standings


def _record():
    return {
        "leagueRecord": {"wins": 10, "losses": 5, "pct": ".667"},
        "divisionRank": "1",
        "divisionGamesBack": "0",
        "wildCardGamesBack": 0,
        "records": {"splitRecords": []},
        "streak": {"streakCode": "W2"},
    }


def test_standings_screen1_uses_screen_background_override(monkeypatch):
    monkeypatch.setattr(
        mlb_team_standings,
        "get_screen_background_color",
        lambda screen_id, default: (17, 34, 51) if screen_id == "cubs stand1" else default,
    )

    image = mlb_team_standings.draw_standings_screen1(
        display=None,
        rec=_record(),
        logo_path="/tmp/does-not-exist.png",
        division_name="NL Central",
        screen_id="cubs stand1",
        transition=True,
    )

    assert image is not None
    assert image.getpixel((0, 0)) == (17, 34, 51)


def test_standings_screen2_uses_screen_background_override(monkeypatch):
    monkeypatch.setattr(
        mlb_team_standings,
        "get_screen_background_color",
        lambda screen_id, default: (68, 85, 102) if screen_id == "cubs stand2" else default,
    )

    image = mlb_team_standings.draw_standings_screen2(
        display=None,
        rec=_record(),
        logo_path="/tmp/does-not-exist.png",
        screen_id="cubs stand2",
        transition=True,
    )

    assert image is not None
    assert image.getpixel((0, 0)) == (68, 85, 102)
