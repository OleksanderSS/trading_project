import asyncio

import pytest

from src.data.collectors.reddit_sentiment_collector import RedditSentimentCollector


def _collector(config=None):
    return RedditSentimentCollector(config or {}, http_client_factory=None, db_manager=None)


def test_reddit_collector_disabled_by_default():
    collector = _collector()

    assert collector.enabled is False
    assert asyncio.run(collector.run()) is None


def test_reddit_collector_enabled_without_adapter_raises():
    # The collector fetches real posts from Reddit's public RSS feeds (no
    # synthetic-data fallback exists anymore) -- enabling it without a real
    # http_client_factory fails when it tries to open a client, and run()'s
    # broad except wraps that into a RuntimeError rather than returning None.
    collector = _collector({"enabled": True})

    with pytest.raises(RuntimeError, match="Reddit sentiment collection failed"):
        asyncio.run(collector.run())
