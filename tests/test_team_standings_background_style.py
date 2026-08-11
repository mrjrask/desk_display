import pytest

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


def test_standings_screen3_uses_w_l_markers_for_cubs_only(monkeypatch):
    calls = []

    def _fake_load_marker(letter, target_height):
        calls.append((letter, target_height))
        return None

    monkeypatch.setattr(mlb_team_standings, "_load_marker_icon", _fake_load_marker)

    cubs_image = mlb_team_standings.draw_standings_screen3(
        display=None,
        rec=_record(),
        logo_path="/tmp/does-not-exist.png",
        division_name="NL Central",
        screen_id="cubs stand3",
        transition=True,
    )
    cubs_calls = list(calls)

    calls.clear()
    sox_image = mlb_team_standings.draw_standings_screen3(
        display=None,
        rec=_record(),
        logo_path="/tmp/does-not-exist.png",
        division_name="AL Central",
        screen_id="sox stand3",
        transition=True,
    )

    assert cubs_image is not None
    assert sox_image is not None
    assert [letter for letter, _ in cubs_calls] == ["W", "L"]
    assert calls == []


@pytest.mark.parametrize("screen_id", ["cubs stand3", "sox stand3"])
def test_standings_screen3_wcgb_tied_lead_never_shows_plus_minus(monkeypatch, screen_id):
    rec = _record()
    rec["wildCardGamesBack"] = "-"
    rec["wildCardRank"] = "2"

    normalized_inputs = []

    def _capture_normalize(text, *, screen_id=None):
        normalized_inputs.append(text)
        return text

    monkeypatch.setattr(mlb_team_standings, "_normalize_half_text", _capture_normalize)

    image = mlb_team_standings.draw_standings_screen3(
        display=None,
        rec=rec,
        logo_path="/tmp/does-not-exist.png",
        division_name="AL Central",
        screen_id=screen_id,
        transition=True,
    )

    assert image is not None
    assert "- WCGB" in normalized_inputs
    assert "+- WCGB" not in normalized_inputs


@pytest.mark.parametrize("screen_id", ["cubs stand1", "sox stand1"])
def test_standings_screen1_wcgb_tied_lead_never_shows_plus_minus(monkeypatch, screen_id):
    rec = _record()
    rec["wildCardGamesBack"] = "-"
    rec["wildCardRank"] = "2"

    normalized_inputs = []

    def _capture_normalize(text, *, screen_id=None):
        normalized_inputs.append(text)
        return text

    monkeypatch.setattr(mlb_team_standings, "_normalize_half_text", _capture_normalize)

    image = mlb_team_standings.draw_standings_screen1(
        display=None,
        rec=rec,
        logo_path="/tmp/does-not-exist.png",
        division_name="AL Central",
        screen_id=screen_id,
        transition=True,
    )

    assert image is not None
    assert "- WCGB" in normalized_inputs
    assert "+- WCGB" not in normalized_inputs
