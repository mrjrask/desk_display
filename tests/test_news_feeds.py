import datetime as dt
import json
import time

import services.news_feeds as nf
from services.news_feeds import (
    ArticleContent,
    NewsTopic,
    _ArticleTextExtractor,
    fetch_all_headlines,
    fetch_article_text,
    fetch_topic_headlines,
    load_news_feed_config,
    parse_feed_headlines,
)


class _FakeResponse:
    def __init__(self, text: str, status_ok: bool = True):
        self.text = text
        self.content = text.encode("utf-8")
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("bad status")


def _rss(items_xml: str) -> str:
    return (
        '<?xml version="1.0"?>'
        '<rss version="2.0" '
        'xmlns:media="http://search.yahoo.com/mrss/" '
        'xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        f"<channel><title>Test</title>{items_xml}</channel></rss>"
    )


def test_parse_feed_headlines_sorts_newest_first_and_respects_limit():
    xml = _rss(
        "<item><title>Older</title><link>https://example.com/older</link>"
        "<pubDate>Mon, 03 Aug 2026 09:00:00 GMT</pubDate></item>"
        "<item><title>Newer</title><link>https://example.com/newer</link>"
        "<pubDate>Mon, 03 Aug 2026 15:00:00 GMT</pubDate></item>"
        "<item><title>Undated</title><link>https://example.com/undated</link></item>"
    )

    items = parse_feed_headlines(xml, "world", limit=2)

    assert [item.title for item in items] == ["Newer", "Older"]


def test_parse_feed_headlines_prefers_largest_media_content():
    xml = _rss(
        "<item><title>Widgets</title><link>https://example.com/a</link>"
        '<media:content url="https://example.com/small.jpg" width="100"/>'
        '<media:content url="https://example.com/big.jpg" width="800"/>'
        "</item>"
    )

    items = parse_feed_headlines(xml, "world", limit=5)

    assert items[0].image_url == "https://example.com/big.jpg"


def test_parse_feed_headlines_falls_back_to_media_thumbnail():
    xml = _rss(
        "<item><title>Widgets</title><link>https://example.com/a</link>"
        '<media:thumbnail url="https://example.com/thumb.jpg"/>'
        "</item>"
    )

    items = parse_feed_headlines(xml, "world", limit=5)

    assert items[0].image_url == "https://example.com/thumb.jpg"


def test_parse_feed_headlines_falls_back_to_enclosure():
    xml = _rss(
        "<item><title>Widgets</title><link>https://example.com/a</link>"
        '<enclosure url="https://example.com/enc.jpg" type="image/jpeg"/>'
        "</item>"
    )

    items = parse_feed_headlines(xml, "world", limit=5)

    assert items[0].image_url == "https://example.com/enc.jpg"


def test_parse_feed_headlines_falls_back_to_img_in_description_html():
    xml = _rss(
        "<item><title>Widgets</title><link>https://example.com/a</link>"
        "<description>&lt;div&gt;text &lt;img src=\"https://example.com/inline.jpg\"/&gt;&lt;/div&gt;</description>"
        "</item>"
    )

    items = parse_feed_headlines(xml, "world", limit=5)

    assert items[0].image_url == "https://example.com/inline.jpg"
    assert items[0].summary == "text"


def test_parse_feed_headlines_skips_entries_without_titles():
    xml = _rss("<item><link>https://example.com/a</link></item>")

    assert parse_feed_headlines(xml, "world", limit=5) == []


def test_parse_feed_headlines_handles_malformed_xml():
    assert parse_feed_headlines("<rss><channel><item>", "world", limit=5) == []


def test_parse_feed_headlines_handles_atom_entries():
    xml = (
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
        "<title>Atom Feed</title>"
        "<entry>"
        "<title>Atom Headline</title>"
        '<link rel="alternate" href="https://example.com/atom1"/>'
        "<summary>Atom summary</summary>"
        "<updated>2026-08-03T10:00:00Z</updated>"
        "</entry>"
        "</feed>"
    )

    items = parse_feed_headlines(xml, "technology", limit=5)

    assert len(items) == 1
    assert items[0].title == "Atom Headline"
    assert items[0].link == "https://example.com/atom1"
    assert items[0].published == dt.datetime(2026, 8, 3, 10, 0, tzinfo=dt.UTC)


def test_fetch_topic_headlines_returns_empty_on_request_failure(monkeypatch):
    def fail_get(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(nf, "http_get", fail_get)
    topic = NewsTopic(id="world", label="World News", name="Test", url="https://example.com/feed")

    assert fetch_topic_headlines(topic, limit=5) == []


def test_load_news_feed_config_reads_topics_and_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "news_feeds.json"
    config_path.write_text(
        json.dumps(
            {
                "headline_count": 3,
                "refresh_minutes": 10,
                "topics": [
                    {"id": "local", "label": "Local News", "name": "Test Local", "url": "https://example.com/local.xml"},
                    {"id": "missing_url"},
                ],
            }
        )
    )
    monkeypatch.setattr(nf, "resolve_news_feeds_config_path", lambda: config_path)
    with nf._config_cache_lock:
        nf._config_cache_path = None
        nf._config_cache_mtime = None
        nf._config_cache_value = None

    topics, headline_count, refresh_minutes = load_news_feed_config()

    assert [t.id for t in topics] == ["local"]
    assert headline_count == 3
    assert refresh_minutes == 10


def test_load_news_feed_config_falls_back_when_file_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(nf, "resolve_news_feeds_config_path", lambda: missing_path)
    with nf._config_cache_lock:
        nf._config_cache_path = None
        nf._config_cache_mtime = None
        nf._config_cache_value = None

    topics, headline_count, refresh_minutes = load_news_feed_config()

    assert topics == []
    assert headline_count == nf._DEFAULT_HEADLINE_COUNT
    assert refresh_minutes == nf._DEFAULT_REFRESH_MINUTES


def test_fetch_all_headlines_caches_until_refresh_interval_elapses(monkeypatch):
    # Deliberately avoids patching time.monotonic: concurrent.futures.wait()
    # calls the real time module internally during fetch_all_headlines, and a
    # patched global monotonic() would desync those internal calls too.
    nf.clear_headline_cache_for_tests()
    topics = [NewsTopic(id="local", label="Local", name="Test", url="https://example.com/local.xml")]
    monkeypatch.setattr(nf, "load_news_feed_config", lambda: (topics, 5, 20))

    calls = []

    def fake_fetch(topic, limit, timeout=5.0):
        calls.append(topic.id)
        return [nf.NewsHeadline(topic_id=topic.id, title=f"Headline {len(calls)}", link="https://example.com/x")]

    monkeypatch.setattr(nf, "fetch_topic_headlines", fake_fetch)

    first = fetch_all_headlines()
    second = fetch_all_headlines()

    assert len(calls) == 1
    assert first["local"][0].title == second["local"][0].title


def test_fetch_all_headlines_refetches_after_refresh_interval(monkeypatch):
    nf.clear_headline_cache_for_tests()
    topics = [NewsTopic(id="local", label="Local", name="Test", url="https://example.com/local.xml")]
    monkeypatch.setattr(nf, "load_news_feed_config", lambda: (topics, 5, 1))

    calls = []

    def fake_fetch(topic, limit, timeout=5.0):
        calls.append(topic.id)
        return [nf.NewsHeadline(topic_id=topic.id, title=f"Headline {len(calls)}", link="https://example.com/x")]

    monkeypatch.setattr(nf, "fetch_topic_headlines", fake_fetch)

    # Seed a cache entry that is older than the 1-minute refresh window
    # (refresh_minutes=1) using the real clock, so the next call must refetch.
    with nf._headlines_cache_lock:
        nf._headlines_cache_value = {
            "local": [nf.NewsHeadline(topic_id="local", title="Stale headline", link="https://example.com/stale")]
        }
        nf._headlines_cache_time = time.monotonic() - 3600.0

    result = fetch_all_headlines()

    assert len(calls) == 1
    assert result["local"][0].title == "Headline 1"


def test_article_text_extractor_prefers_article_tag_and_finds_og_image():
    html = (
        "<html><head>"
        '<meta property="og:image" content="https://example.com/hero.jpg"/>'
        "<title>Page Title</title></head><body>"
        "<footer><p>Footer boilerplate text that is definitely long enough to pass the filter"
        " easily.</p></footer>"
        "<article><p>Real paragraph one has plenty of length to clear the extraction filter"
        " threshold.</p>"
        "<p>Real paragraph two also has plenty of length to clear the extraction filter"
        " threshold.</p></article>"
        "</body></html>"
    )

    parser = _ArticleTextExtractor()
    parser.feed(html)

    assert parser.title == "Page Title"
    assert parser.og_image == "https://example.com/hero.jpg"
    assert len(parser.paragraphs_in_article) == 2
    assert "Footer boilerplate" not in " ".join(parser.paragraphs_all)


def test_article_text_extractor_falls_back_to_all_paragraphs_without_article_tag():
    html = (
        "<html><body>"
        "<div><p>First standalone paragraph long enough to clear the extraction filter"
        " threshold here.</p>"
        "<p>Second standalone paragraph long enough to clear the extraction filter"
        " threshold here.</p></div>"
        "</body></html>"
    )

    parser = _ArticleTextExtractor()
    parser.feed(html)

    assert parser.paragraphs_in_article == []
    assert len(parser.paragraphs_all) == 2


def test_fetch_article_text_returns_none_on_failure_and_caches(monkeypatch):
    calls = []

    def fail_get(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("network down")

    monkeypatch.setattr(nf, "http_get", fail_get)
    with nf._article_cache_lock:
        nf._article_cache.clear()

    assert fetch_article_text("https://example.com/article") is None
    assert fetch_article_text("https://example.com/article") is None
    assert len(calls) == 1


def test_fetch_article_text_parses_and_caches_successful_fetch(monkeypatch):
    html = (
        "<html><head><title>Great Article</title>"
        '<meta property="og:image" content="https://example.com/hero.jpg"/></head>'
        "<body><article><p>This is the body paragraph with plenty of characters to pass"
        " the length filter.</p>"
        "</article></body></html>"
    )
    calls = []

    def fake_get(url, timeout=6.0):
        calls.append(url)
        return _FakeResponse(html)

    monkeypatch.setattr(nf, "http_get", fake_get)
    with nf._article_cache_lock:
        nf._article_cache.clear()

    first = fetch_article_text("https://example.com/article")
    second = fetch_article_text("https://example.com/article")

    assert isinstance(first, ArticleContent)
    assert first.title == "Great Article"
    assert first.image_url == "https://example.com/hero.jpg"
    assert len(first.paragraphs) == 1
    assert second == first
    assert len(calls) == 1
