import asyncio

from src.data.collectors.reddit_sentiment_collector import RedditSentimentCollector


def _collector(config=None):
    return RedditSentimentCollector(config or {}, http_client_factory=None, db_manager=None)


def test_reddit_collector_disabled_by_default():
    collector = _collector()

    assert collector.enabled is False
    assert collector.use_synthetic_data is False
    assert asyncio.run(collector.run()) is None


def test_reddit_collector_enabled_without_adapter_returns_no_data():
    collector = _collector({"enabled": True})

    assert asyncio.run(collector._fetch_reddit_sentiment_data()) == []
    assert asyncio.run(collector.run()) is None


def test_reddit_synthetic_data_requires_explicit_flag():
    collector = _collector(
        {
            "enabled": True,
            "use_synthetic_data": True,
            "subreddits": ["stocks"],
        }
    )

    records = asyncio.run(collector._fetch_reddit_sentiment_data())
    df = asyncio.run(collector.run())

    assert records
    assert all(record["is_synthetic"] for record in records)
    assert df is not None
    assert not df.empty
    assert df["data_source"].eq("synthetic_reddit_simulation").all()
