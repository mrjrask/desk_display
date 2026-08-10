"""News headline feed loading, RSS/Atom parsing, and article text extraction.

Feed sources (name + URL) are *not* hardcoded here. They live in the editable
``news_feeds.json`` file (see ``paths.resolve_news_feeds_config_path``) so an
operator can add, remove, rename, or re-point topics without touching code.
Every feed is a plain, free, key-less RSS/Atom URL.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import email.utils
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import ClassVar, Optional
from xml.etree import ElementTree

from paths import resolve_news_feeds_config_path, resolve_news_feeds_config_path_2
from services.http_client import http_get

_DEFAULT_HEADLINE_COUNT = 5
_DEFAULT_REFRESH_MINUTES = 20
_MEDIA_NAMESPACE_HINTS = ("search.yahoo.com/mrss", "media")
_IMG_SRC_RE = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.IGNORECASE)


@dataclass(frozen=True)
class NewsTopic:
    id: str
    label: str
    name: str
    url: str


@dataclass(frozen=True)
class NewsHeadline:
    topic_id: str
    title: str
    link: str
    summary: str = ""
    image_url: Optional[str] = None
    published: Optional[dt.datetime] = None
    content_html: str = ""


@dataclass(frozen=True)
class ArticleContent:
    title: Optional[str]
    image_url: Optional[str]
    paragraphs: tuple[str, ...]
    source_url: str


# ─── Feed config loading (mtime-cached, matches screens/registry.py's layouts pattern) ──

_config_cache_lock = threading.Lock()
_config_cache_path: Optional[str] = None
_config_cache_mtime: Optional[float] = None
_config_cache_value: Optional[tuple[list[NewsTopic], int, int]] = None

# Cache slot for the second "news headlines 2" screen (news_feeds_2.json),
# kept separate from the primary screen's slot above so the two screens
# refresh/reload independently of each other.
_config_cache_path_2: Optional[str] = None
_config_cache_mtime_2: Optional[float] = None
_config_cache_value_2: Optional[tuple[list[NewsTopic], int, int]] = None


def _fallback_config() -> tuple[list[NewsTopic], int, int]:
    return [], _DEFAULT_HEADLINE_COUNT, _DEFAULT_REFRESH_MINUTES


def _parse_news_feed_config(path: str) -> tuple[list[NewsTopic], int, int]:
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        logging.warning("news_feeds: could not read config at %s: %s", path, exc)
        return _fallback_config()

    topics: list[NewsTopic] = []
    for raw in payload.get("topics", []) if isinstance(payload, dict) else []:
        if not isinstance(raw, dict):
            continue
        topic_id = str(raw.get("id") or "").strip()
        url = str(raw.get("url") or "").strip()
        if not topic_id or not url:
            continue
        label = str(raw.get("label") or topic_id.title()).strip()
        name = str(raw.get("name") or label).strip()
        topics.append(NewsTopic(id=topic_id, label=label, name=name, url=url))

    try:
        headline_count = max(1, int(payload.get("headline_count", _DEFAULT_HEADLINE_COUNT)))
    except Exception:
        headline_count = _DEFAULT_HEADLINE_COUNT
    try:
        refresh_minutes = max(1, int(payload.get("refresh_minutes", _DEFAULT_REFRESH_MINUTES)))
    except Exception:
        refresh_minutes = _DEFAULT_REFRESH_MINUTES

    return topics, headline_count, refresh_minutes


def load_news_feed_config() -> tuple[list[NewsTopic], int, int]:
    """Return (topics, headline_count, refresh_minutes) from news_feeds.json.

    Reloads only when the file's mtime changes so repeated calls during a
    render pass are cheap. Missing/invalid config degrades to an empty topic
    list rather than raising, so a typo cannot take down the whole screen.
    """

    global _config_cache_path, _config_cache_mtime, _config_cache_value

    path = str(resolve_news_feeds_config_path())
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None

    with _config_cache_lock:
        if (
            _config_cache_path == path
            and _config_cache_mtime == mtime
            and _config_cache_value is not None
        ):
            return _config_cache_value

        result = _parse_news_feed_config(path)
        _config_cache_path, _config_cache_mtime, _config_cache_value = path, mtime, result
        return result


def load_news_feed_config_2() -> tuple[list[NewsTopic], int, int]:
    """Same as ``load_news_feed_config`` but for the "news headlines 2" screen.

    Reads/caches ``news_feeds_2.json`` (see
    ``paths.resolve_news_feeds_config_path_2``) independently of the primary
    screen's config.
    """

    global _config_cache_path_2, _config_cache_mtime_2, _config_cache_value_2

    path = str(resolve_news_feeds_config_path_2())
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None

    with _config_cache_lock:
        if (
            _config_cache_path_2 == path
            and _config_cache_mtime_2 == mtime
            and _config_cache_value_2 is not None
        ):
            return _config_cache_value_2

        result = _parse_news_feed_config(path)
        _config_cache_path_2, _config_cache_mtime_2, _config_cache_value_2 = path, mtime, result
        return result


# ─── RSS 2.0 / Atom parsing ─────────────────────────────────────────────────


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _namespace_uri(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else ""


def _is_media_namespace(tag: str) -> bool:
    ns = _namespace_uri(tag).lower()
    return any(hint in ns for hint in _MEDIA_NAMESPACE_HINTS)


def _clean_text(value: Optional[str]) -> str:
    return " ".join(str(value or "").split())


def _strip_html(html_text: str) -> str:
    """Collapse an HTML fragment down to plain text (used for RSS summaries)."""

    if not html_text:
        return ""

    class _TextOnly(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.chunks: list[str] = []

        def handle_data(self, data: str) -> None:
            self.chunks.append(data)

    parser = _TextOnly()
    with contextlib.suppress(Exception):
        parser.feed(html_text)
    return _clean_text(" ".join(parser.chunks))


def _first_image_in_html(html_text: str) -> Optional[str]:
    if not html_text:
        return None
    match = _IMG_SRC_RE.search(html_text)
    return match.group(1) if match else None


def _parse_pub_date(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
    except Exception:
        pass
    try:
        iso_value = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = dt.datetime.fromisoformat(iso_value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
    except Exception:
        return None


def _extract_link(item: ElementTree.Element) -> str:
    link_el = item.find("{*}link")
    if link_el is not None and (link_el.text or "").strip():
        return link_el.text.strip()

    href_links = [el for el in item.findall("{*}link") if el.get("href")]
    for el in href_links:
        if el.get("rel", "alternate") == "alternate":
            return str(el.get("href"))
    if href_links:
        return str(href_links[0].get("href"))

    guid = item.findtext("{*}guid") or item.findtext("{*}id")
    if guid and guid.strip().startswith("http"):
        return guid.strip()
    return ""


def _extract_image(item: ElementTree.Element, description: str, content_html: str) -> Optional[str]:
    media_content = [
        el
        for el in item.iter()
        if _local_name(el.tag) == "content" and _is_media_namespace(el.tag) and el.get("url")
    ]
    if media_content:
        def _width(el: ElementTree.Element) -> int:
            try:
                return int(el.get("width") or 0)
            except (TypeError, ValueError):
                return 0

        media_content.sort(key=_width, reverse=True)
        return media_content[0].get("url")

    media_thumb = [
        el
        for el in item.iter()
        if _local_name(el.tag) == "thumbnail" and _is_media_namespace(el.tag) and el.get("url")
    ]
    if media_thumb:
        return media_thumb[0].get("url")

    for enclosure in item.findall("{*}enclosure"):
        enclosure_type = str(enclosure.get("type") or "")
        if enclosure_type.startswith("image") and enclosure.get("url"):
            return enclosure.get("url")

    return _first_image_in_html(content_html) or _first_image_in_html(description)


def _entries_from_root(root: ElementTree.Element) -> list[ElementTree.Element]:
    root_local = _local_name(root.tag)
    if root_local == "feed":
        return root.findall("{*}entry")
    return root.findall(".//{*}item")


def parse_feed_headlines(xml_text: str, topic_id: str, limit: int) -> list[NewsHeadline]:
    """Parse an RSS 2.0 or Atom feed body into up to *limit* headlines.

    Returns newest-first when publish dates are present; otherwise preserves
    feed order. Tolerant of malformed/unexpected XML: returns an empty list
    rather than raising, so one bad feed cannot break the whole screen.
    """

    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        logging.debug("news_feeds: failed to parse feed for topic %s: %s", topic_id, exc)
        return []

    headlines: list[NewsHeadline] = []
    for item in _entries_from_root(root):
        title = _clean_text(
            item.findtext("{*}title") or ""
        )
        if not title:
            continue
        link = _extract_link(item)
        description_raw = item.findtext("{*}description") or item.findtext("{*}summary") or ""
        content_html = item.findtext("{*}encoded") or item.findtext("{*}content") or ""
        summary = _strip_html(description_raw)
        if len(summary) > 220:
            summary = summary[:217].rstrip() + "..."
        image_url = _extract_image(item, description_raw, content_html)
        published = _parse_pub_date(
            item.findtext("{*}pubDate")
            or item.findtext("{*}published")
            or item.findtext("{*}updated")
            or item.findtext("{*}date")
        )
        headlines.append(
            NewsHeadline(
                topic_id=topic_id,
                title=title,
                link=link,
                summary=summary,
                image_url=image_url,
                published=published,
                content_html=content_html,
            )
        )

    dated = [h for h in headlines if h.published is not None]
    undated = [h for h in headlines if h.published is None]
    dated.sort(key=lambda h: h.published, reverse=True)
    return (dated + undated)[:limit]


def fetch_topic_headlines(
    topic: NewsTopic, limit: int, *, timeout: float = 5.0
) -> list[NewsHeadline]:
    try:
        response = http_get(topic.url, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        logging.debug("news_feeds: fetch failed for topic %s (%s): %s", topic.id, topic.url, exc)
        return []
    return parse_feed_headlines(response.text, topic.id, limit)


# ─── Parallel fetch across all configured topics, cached by refresh interval ───

_headlines_cache_lock = threading.Lock()
_headlines_cache_value: dict[str, list[NewsHeadline]] = {}
_headlines_cache_time: Optional[float] = None
_headlines_cache_refresh_seconds: float = _DEFAULT_REFRESH_MINUTES * 60

# Cache slot for the second "news headlines 2" screen, kept separate from the
# primary screen's slot above so the two screens refresh independently.
_headlines_cache_value_2: dict[str, list[NewsHeadline]] = {}
_headlines_cache_time_2: Optional[float] = None
_headlines_cache_refresh_seconds_2: float = _DEFAULT_REFRESH_MINUTES * 60

_FEED_FETCH_TIMEOUT_BUDGET_SECONDS = 6.0


def _fetch_topics_parallel(
    topics: list[NewsTopic], headline_count: int
) -> dict[str, list[NewsHeadline]]:
    """Fetch every topic's headlines in parallel with a shared time budget.

    Mirrors the pattern used by the "on this day" screen so one slow/
    unreachable feed cannot stall the whole screen.
    """

    results: dict[str, list[NewsHeadline]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(topics))) as executor:
        future_to_topic = {
            executor.submit(fetch_topic_headlines, topic, headline_count): topic
            for topic in topics
        }
        done, pending = wait(future_to_topic, timeout=_FEED_FETCH_TIMEOUT_BUDGET_SECONDS)
        for future in done:
            topic = future_to_topic[future]
            try:
                results[topic.id] = future.result() or []
            except Exception as exc:
                logging.debug("news_feeds: topic %s raised: %s", topic.id, exc)
        for future in pending:
            future.cancel()
    return results


def fetch_all_headlines(*, force: bool = False) -> dict[str, list[NewsHeadline]]:
    """Return {topic_id: [NewsHeadline, ...]} for every configured topic.

    Fetches run in parallel with a shared time budget (mirroring the pattern
    used by the "on this day" screen) so one slow/unreachable feed cannot
    stall the whole screen. Results are cached for ``refresh_minutes``
    (from news_feeds.json) between refreshes.
    """

    global _headlines_cache_value, _headlines_cache_time, _headlines_cache_refresh_seconds

    topics, headline_count, refresh_minutes = load_news_feed_config()
    _headlines_cache_refresh_seconds = max(60, refresh_minutes * 60)

    now = time.monotonic()
    with _headlines_cache_lock:
        if (
            not force
            and _headlines_cache_time is not None
            and (now - _headlines_cache_time) < _headlines_cache_refresh_seconds
            and _headlines_cache_value
        ):
            return {topic_id: list(items) for topic_id, items in _headlines_cache_value.items()}

    if not topics:
        return {}

    results = _fetch_topics_parallel(topics, headline_count)

    with _headlines_cache_lock:
        if results:
            # Merge per-topic rather than replacing the whole cache: a topic
            # whose feed fails/times out this cycle would otherwise wipe out
            # its last known-good headlines (and thus its ticker row) even
            # though every other topic refreshed fine.
            merged_cache = dict(_headlines_cache_value)
            for topic_id, items in results.items():
                if items or topic_id not in merged_cache:
                    merged_cache[topic_id] = items
            _headlines_cache_value = merged_cache
            _headlines_cache_time = time.monotonic()
        elif not _headlines_cache_value:
            _headlines_cache_time = time.monotonic()
        return {topic_id: list(items) for topic_id, items in _headlines_cache_value.items()}


def fetch_all_headlines_2(*, force: bool = False) -> dict[str, list[NewsHeadline]]:
    """Same as ``fetch_all_headlines`` but for the "news headlines 2" screen."""

    global _headlines_cache_value_2, _headlines_cache_time_2, _headlines_cache_refresh_seconds_2

    topics, headline_count, refresh_minutes = load_news_feed_config_2()
    _headlines_cache_refresh_seconds_2 = max(60, refresh_minutes * 60)

    now = time.monotonic()
    with _headlines_cache_lock:
        if (
            not force
            and _headlines_cache_time_2 is not None
            and (now - _headlines_cache_time_2) < _headlines_cache_refresh_seconds_2
            and _headlines_cache_value_2
        ):
            return {topic_id: list(items) for topic_id, items in _headlines_cache_value_2.items()}

    if not topics:
        return {}

    results = _fetch_topics_parallel(topics, headline_count)

    with _headlines_cache_lock:
        if results:
            merged_cache = dict(_headlines_cache_value_2)
            for topic_id, items in results.items():
                if items or topic_id not in merged_cache:
                    merged_cache[topic_id] = items
            _headlines_cache_value_2 = merged_cache
            _headlines_cache_time_2 = time.monotonic()
        elif not _headlines_cache_value_2:
            _headlines_cache_time_2 = time.monotonic()
        return {topic_id: list(items) for topic_id, items in _headlines_cache_value_2.items()}


def clear_headline_cache_for_tests() -> None:
    global _headlines_cache_value, _headlines_cache_time
    global _headlines_cache_value_2, _headlines_cache_time_2
    with _headlines_cache_lock:
        _headlines_cache_value = {}
        _headlines_cache_time = None
        _headlines_cache_value_2 = {}
        _headlines_cache_time_2 = None


# ─── Reader-style article text extraction ──────────────────────────────────


class _ArticleTextExtractor(HTMLParser):
    """Minimal readability-style extractor: pulls <article> paragraph text.

    This intentionally avoids a full readability/BeautifulSoup dependency;
    it favors text inside an <article> element when present, falls back to
    every <p> in the document otherwise, and skips common boilerplate
    containers (nav/header/footer/aside/script/style/form).
    """

    _SKIP_TAGS: ClassVar[set[str]] = {
        "script", "style", "nav", "header", "footer", "aside",
        "form", "noscript", "figcaption", "button", "iframe", "svg",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._article_depth = 0
        self._in_title = False
        self._current: list[str] = []
        self._title_chars: list[str] = []
        self.title: Optional[str] = None
        self.og_image: Optional[str] = None
        self.paragraphs_in_article: list[str] = []
        self.paragraphs_all: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_d = dict(attrs)
        if tag == "meta":
            prop = (attrs_d.get("property") or attrs_d.get("name") or "").lower()
            if prop in ("og:image", "twitter:image") and not self.og_image:
                content = attrs_d.get("content")
                if content:
                    self.og_image = content.strip()
            return
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "article":
            self._article_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "p":
            self._current = []
        if tag == "br" and not self._skip_depth:
            self._current.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "article":
            self._article_depth = max(0, self._article_depth - 1)
        if tag == "title":
            self._in_title = False
            if self.title is None:
                text = _clean_text("".join(self._title_chars))
                if text:
                    self.title = text
        if tag == "p":
            text = _clean_text("".join(self._current))
            self._current = []
            if len(text) >= 40:
                self.paragraphs_all.append(text)
                if self._article_depth > 0:
                    self.paragraphs_in_article.append(text)

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_chars.append(data)
        self._current.append(data)


_article_cache_lock = threading.Lock()
_article_cache: dict[str, tuple[float, Optional[ArticleContent]]] = {}
_ARTICLE_CACHE_TTL_SECONDS = 3600.0
_ARTICLE_CACHE_MAX_ENTRIES = 64


def fetch_article_text(
    url: str, *, timeout: float = 6.0, max_paragraphs: int = 40
) -> Optional[ArticleContent]:
    """Fetch *url* and extract Reader-style article text for the touch overlay.

    Returns None on any network/parse failure so callers can fall back to the
    RSS summary. Results are cached in-memory for a while since a user may
    reopen the same headline within one rotation.
    """

    if not url:
        return None

    now = time.monotonic()
    with _article_cache_lock:
        cached = _article_cache.get(url)
        if cached is not None and (now - cached[0]) < _ARTICLE_CACHE_TTL_SECONDS:
            return cached[1]

    try:
        response = http_get(url, timeout=timeout)
        response.raise_for_status()
        html_text = response.text
    except Exception as exc:
        logging.debug("news_feeds: article fetch failed for %s: %s", url, exc)
        with _article_cache_lock:
            _article_cache[url] = (now, None)
        return None

    parser = _ArticleTextExtractor()
    try:
        parser.feed(html_text)
    except Exception as exc:
        logging.debug("news_feeds: article parse failed for %s: %s", url, exc)

    paragraphs = (
        parser.paragraphs_in_article
        if len(parser.paragraphs_in_article) >= 2
        else parser.paragraphs_all
    )
    result = ArticleContent(
        title=parser.title,
        image_url=parser.og_image,
        paragraphs=tuple(paragraphs[:max_paragraphs]),
        source_url=url,
    )

    with _article_cache_lock:
        if len(_article_cache) >= _ARTICLE_CACHE_MAX_ENTRIES:
            oldest_key = min(_article_cache, key=lambda key: _article_cache[key][0])
            _article_cache.pop(oldest_key, None)
        _article_cache[url] = (now, result)

    return result
