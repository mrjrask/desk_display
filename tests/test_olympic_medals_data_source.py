from screens.data_sources import olympic_medals


def test_default_medal_url_is_espn_page():
    assert olympic_medals.DEFAULT_MEDAL_URL == "https://www.espn.com/olympics/winter/2026/medals"


def test_extract_espn_medal_table_from_html_parses_balanced_json():
    html = (
        '<script>window.__DATA__={"x":1,"medals":['
        '{"country":"NOR","gold":4,"silver":1,"bronze":2},'
        '{"country":"USA","gold":3,"silver":2,"bronze":1}'
        ']}</script>'
    )

    rows = olympic_medals._extract_espn_medal_table_from_html(html)

    assert len(rows) == 2
    assert rows[0]["country"] == "NOR"
    assert rows[1]["country"] == "USA"


def test_fetch_olympic_medal_table_uses_html_fallback(monkeypatch):
    class FakeResponse:
        text = (
            '<script>window.__DATA__={"medals":['
            '{"country":"CAN","gold":5,"silver":0,"bronze":1},'
            '{"country":"SWE","gold":2,"silver":3,"bronze":0}'
            ']}</script>'
        )

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("not json")

    class FakeSession:
        def get(self, url, timeout):
            return FakeResponse()

    monkeypatch.setattr(olympic_medals, "SESSION", FakeSession())

    rows = olympic_medals.fetch_olympic_medal_table(top_n=1)

    assert len(rows) == 1
    assert rows[0]["country"] == "CAN"
    assert rows[0]["gold"] == 5
