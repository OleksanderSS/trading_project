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


def test_reddit_sentiment_stays_disabled_until_real_adapter_exists():
    payload = yaml.safe_load(Path("src/config/collectors.yaml").read_text(encoding="utf-8"))
    collectors = payload.get("collectors") or payload
    config = collectors["reddit_sentiment"]
    assert config["enabled"] is False
    assert config["use_synthetic_data"] is False
