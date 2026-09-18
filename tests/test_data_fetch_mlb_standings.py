import datetime
import inspect
import re

import pytest

import data_fetch


class _DummyResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "records": [
                {
                    "teamRecords": [
                        {"team": {"id": 112}, "wins": 1, "losses": 0},
                    ]
                }
            ]
        }


@pytest.mark.parametrize(
    ("current_date", "expected_season"),
    [
        (datetime.date(2031, 7, 15), 2031),
        (datetime.date(2032, 1, 1), 2032),
    ],
)
def test_mlb_standings_uses_applicable_calendar_year(monkeypatch, current_date, expected_season):
    request = {}

    def fake_get(url, *, params, timeout):
        request.update(url=url, params=params, timeout=timeout)
        return _DummyResponse()

    monkeypatch.setattr(data_fetch._session, "get", fake_get)

    standings = data_fetch._fetch_mlb_standings(104, 205, 112, current_date=current_date)

    assert standings == {"team": {"id": 112}, "wins": 1, "losses": 0}
    assert request == {
        "url": "https://statsapi.mlb.com/api/v1/standings",
        "params": {"season": expected_season, "leagueId": 104, "divisionId": 205},
        "timeout": 10,
    }


def test_mlb_standings_production_code_has_no_fixed_season():
    source = inspect.getsource(data_fetch._fetch_mlb_standings)

    assert re.search(r'["\']season["\']\s*:\s*20\d{2}', source) is None
    assert re.search(r"season=20\d{2}", source) is None
    assert '"season": applicable_date.year' in source
