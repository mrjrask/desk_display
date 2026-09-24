import time

import services.news_feeds as nf
from services.news_feeds import NewsTopic


def test_fetch_topics_parallel_does_not_block_past_the_budget(monkeypatch):
    # Regression: `with ThreadPoolExecutor(...) as executor:` calls
    # shutdown(wait=True) on exit, which blocks until every submitted thread
    # finishes -- including ones `future.cancel()` couldn't actually cancel
    # because they'd already started. A single slow/hung feed used to stall
    # the whole function well past the shared budget.
    monkeypatch.setattr(nf, "_FEED_FETCH_TIMEOUT_BUDGET_SECONDS", 0.2)

    topics = [
        NewsTopic(id="fast", label="Fast", name="Fast", url="https://example.com/fast"),
        NewsTopic(id="slow", label="Slow", name="Slow", url="https://example.com/slow"),
    ]

    def _fake_fetch(topic, count):
        if topic.id == "slow":
            time.sleep(2.0)
            return [("slow-headline",)]
        return ["fast-headline"]

    monkeypatch.setattr(nf, "fetch_topic_headlines", _fake_fetch)

    started = time.monotonic()
    results = nf._fetch_topics_parallel(topics, 5)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert results.get("fast") == ["fast-headline"]
    assert "slow" not in results
