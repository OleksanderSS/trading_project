from __future__ import annotations

from dean_os.analyst_core.domain_analyst_pipeline_news_taxonomy import classify_pipeline_news_context


def test_pipeline_news_taxonomy_classifies_crisis_patterns_without_prediction_adjustment():
    payload = classify_pipeline_news_context(
        "Silicon Valley Bank collapses amid liquidity crisis as Fed rates pressure credit conditions.",
        sentiment_label="negative",
    )

    assert payload["adapter_id"] == "domain_analyst_pipeline_news_taxonomy_v1"
    assert payload["allowed_output"] == "pipeline_news_context_for_review"
    assert "prediction_adjustment" in payload["forbidden_outputs"]
    assert "trade_signal" in payload["forbidden_outputs"]
    assert any(item["classification_id"] == "financial" for item in payload["impact_classifications"])
    assert any(item["pattern_id"] == "banking_crisis_2023" for item in payload["crisis_pattern_matches"])
    assert any(item["pattern_id"] == "banking_crisis" for item in payload["learned_pattern_matches"])
    assert "pipeline_market_crisis_context" in payload["context_tags"]
    assert "pipeline_crisis_analogy_requires_human_review" in payload["review_flags"]
    assert "credit_conditions" in payload["watch_metrics"]


def test_pipeline_news_taxonomy_classifies_semiconductor_geopolitical_context():
    payload = classify_pipeline_news_context(
        "Export control sanctions create China semiconductor equipment and HBM supply-chain risk.",
        sentiment_label="negative",
    )

    assert any(item["classification_id"] == "geopolitical" for item in payload["impact_classifications"])
    assert any(item["classification_id"] == "technology" for item in payload["impact_classifications"])
    assert any(item["pattern_id"] == "geopolitical_crisis" for item in payload["crisis_pattern_matches"])
    assert "pipeline_news_geopolitical" in payload["context_tags"]
    assert "pipeline_news_technology" in payload["context_tags"]
    assert "pipeline_learned_pattern_is_analogy_not_prediction" in payload["review_flags"]
    assert "export_controls" in payload["watch_metrics"]
