import screens.mlb_team_standings as mlb_team_standings


def _base_record(*, division_rank, conference_rank):
    return {
        "leagueRecord": {"wins": 10, "losses": 5, "pct": ".667"},
        "divisionRank": str(division_rank),
        "conferenceRank": str(conference_rank),
        "divisionGamesBack": "0",
        "wildCardGamesBack": 0,
        "records": {"splitRecords": []},
    }


def test_standings_screen1_uses_configurable_last_ranks(monkeypatch):
    rendered_lines = []

    monkeypatch.setattr(
        mlb_team_standings,
        "_draw_fraction_text_centered",
        lambda draw, y, text, font, fill=(255, 255, 255): rendered_lines.append(text),
    )

    image = mlb_team_standings.draw_standings_screen1(
        display=None,
        rec=_base_record(division_rank=8, conference_rank=16),
        logo_path="/tmp/does-not-exist.png",
        division_name="Central",
        show_games_back=False,
        show_wild_card=False,
        points_label=None,
        conference_label="conference",
        show_conference_rank=True,
        division_last_rank=8,
        conference_last_rank=16,
        transition=True,
    )

    assert image is not None
    assert "Last in Central" in rendered_lines
    assert "Last in conference" in rendered_lines

