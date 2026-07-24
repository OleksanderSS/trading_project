from pathlib import Path

import yaml


def test_enabled_collectors_cannot_use_synthetic_data():
    payload = yaml.safe_load(Path("src/config/collectors.yaml").read_text(encoding="utf-8"))
    collectors = payload.get("collectors") or payload
    violations = [
        name
        for name, config in collectors.items()
        if isinstance(config, dict)
        and config.get("enabled") is True
        and config.get("use_synthetic_data") is True
    ]
    assert violations == []


def test_reddit_sentiment_uses_a_real_adapter_not_synthetic_data():
    # src/data/collectors/reddit_sentiment_collector.py now fetches real
    # posts from Reddit's public RSS feeds (no API key, no PRAW) --
    # legitimately enabled since this test was written for the
    # not-yet-built state. What must never regress is use_synthetic_data.
    payload = yaml.safe_load(Path("src/config/collectors.yaml").read_text(encoding="utf-8"))
    collectors = payload.get("collectors") or payload
    config = collectors["reddit_sentiment"]
    assert config["enabled"] is True
    assert config["use_synthetic_data"] is False
