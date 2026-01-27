import textwrap

from screens.nfl_standings import (
    CONFERENCE_AFC_KEY,
    CONFERENCE_NFC_KEY,
    _parse_csv_standings,
)


CSV_SAMPLE = textwrap.dedent(
    """\
    season,conf,division,team,wins,losses,ties,pct,div_rank,scored,allowed,net,sov,sos,seed,playoff
    2024,AFC,AFC East,BUF,11,6,0,0.647,1,451,327,124,0.5,0.5,1,
    2024,AFC,AFC East,MIA,10,7,0,0.588,2,400,350,50,0.4,0.5,5,
    2024,NFC,NFC North,DET,12,5,0,0.706,1,450,350,100,0.6,0.5,1,
    2024,NFC,NFC North,GB,9,8,0,0.529,2,400,380,20,0.4,0.5,6,
    """
)


def test_parse_csv_standings_groups_by_conference():
    standings, used_season = _parse_csv_standings(CSV_SAMPLE, 2024)

    assert used_season == 2024

    afc_east = standings[CONFERENCE_AFC_KEY]["AFC East"]
    nfc_north = standings[CONFERENCE_NFC_KEY]["NFC North"]

    assert [team["abbr"] for team in afc_east] == ["BUF", "MIA"]
    assert afc_east[0]["name"] == "Bills"
    assert afc_east[0]["wins"] == 11

    assert [team["abbr"] for team in nfc_north] == ["DET", "GB"]
    assert nfc_north[1]["name"] == "Packers"
    assert nfc_north[1]["losses"] == 8


CSV_PREVIOUS_SEASON = textwrap.dedent(
    """\
    season,conf,division,team,wins,losses,ties,pct,div_rank,scored,allowed,net,sov,sos,seed,playoff
    2023,AFC,AFC East,BUF,13,4,0,0.765,1,450,300,150,0.6,0.5,1,
    2023,AFC,AFC East,MIA,11,6,0,0.647,2,400,350,50,0.5,0.6,5,
    2023,NFC,NFC East,PHI,12,5,0,0.706,1,420,330,90,0.6,0.6,2,
    2023,NFC,NFC East,DAL,12,5,0,0.706,2,430,310,120,0.5,0.6,5,
    """
)


def test_parse_csv_standings_falls_back_to_previous_season():
    standings, used_season = _parse_csv_standings(CSV_PREVIOUS_SEASON, 2024)

    assert used_season == 2023

    afc_east = standings[CONFERENCE_AFC_KEY]["AFC East"]
    assert [team["abbr"] for team in afc_east] == ["BUF", "MIA"]


CSV_NO_MATCH = textwrap.dedent(
    """\
    season,conf,division,team,wins,losses,ties,pct,div_rank,scored,allowed,net,sov,sos,seed,playoff
    2022,AFC,AFC West,DEN,5,12,0,0.294,4,287,359,-72,0.3,0.5,,
    2022,NFC,NFC West,SF,13,4,0,0.765,1,450,300,150,0.6,0.6,2,
    """
)


def test_parse_csv_standings_returns_empty_when_no_data():
    standings, used_season = _parse_csv_standings(CSV_NO_MATCH, 2024)

    assert used_season is None
    assert standings[CONFERENCE_AFC_KEY] == {}
    assert standings[CONFERENCE_NFC_KEY] == {}
