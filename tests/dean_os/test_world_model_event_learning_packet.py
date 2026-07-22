from __future__ import annotations

import pytest

from dean_os.analyst_core import OUTCOME_HORIZONS
from dean_os.schemas import MarketContext
from dean_os.world_model_event_learning import (
    WORLD_MODEL_EVENT_LEARNING_CONTRACT,
    WorldModelEventLearningPacket,
)

AS_OF = "2026-07-01T12:00:00+00:00"
DOMAIN_ID = "semiconductor_ai_infrastructure"


def _semantic_news(**overrides):
    payload = {
        "title": (
            "Nvidia AI demand growth confirms semiconductor memory shortage "
            "and data center capex pressure"
        ),
        "summary": (
            "AI demand growth is increasing HBM memory shortage risk, "
            "supporting data center capex but raising supply-chain constraints."
        ),
        "published_at": "2026-07-01T10:00:00+00:00",
        "url": "https://example.test/news/ai-memory-shortage",
        "tickers": ["NVDA"],
        "_dean_semantic_evidence": {
            "producer_contract": "test_news_contract",
            "evidence_type": "sector_demand",
            "matched_terms": ["ai demand", "data center demand"],
            "required_lane_eligible": True,
            "source_tier": "tier_2_strong_context",
            "source_identity": "reuters",
            "candidate_sha256": "abc123",
        },
    }
    payload.update(overrides)
    return payload


def test_world_model_event_learning_builds_hypotheses_graph_and_replay_tasks():
    context = MarketContext(
        as_of=AS_OF,
        tickers=["NVDA"],
        news=[_semantic_news()],
    )

    payload = WorldModelEventLearningPacket().build(
        context,
        domain_id=DOMAIN_ID,
        save=False,
    )

    assert payload["contract"] == WORLD_MODEL_EVENT_LEARNING_CONTRACT
    assert payload["summary"]["packet_status"] == (
        "world_model_event_learning_ready_pending_replay"
    )
    assert payload["summary"]["classified_event_count"] >= 1
    assert payload["summary"]["hypothesis_count"] >= 1
    assert payload["summary"]["replay_task_count"] == len(OUTCOME_HORIZONS)
    assert payload["summary"]["can_trade"] is False
    assert payload["summary"]["can_write_learning_memory"] is False
    assert payload["safety"]["learning_memory_write_performed"] is False
    assert payload["safety"]["outcome_registration_performed"] is False

    event_classes = {
        item["event_class"] for item in payload["classified_events"]
    }
    assert "demand_driver" in event_classes
    hypothesis = payload["hypotheses"][0]
    assert hypothesis["as_of"] == AS_OF
    assert hypothesis["status"] == "open"
    assert hypothesis["invalidation_signals"]
    assert tuple(hypothesis["horizons_to_check"]) == OUTCOME_HORIZONS
    assert hypothesis["hypothesis_scope"] == "event_response"
    assert hypothesis["horizon_family"] == "event_response_fixed_v1"
    assert hypothesis["trigger_evidence_ids"]
    assert hypothesis["supporting_evidence_ids"] == []
    assert hypothesis["evidence_relationship_status"] == (
        "trigger_only_pending_claim_review"
    )
    assert payload["replay_tasks"][0]["replay_scope"] == "event_response"
    assert payload["replay_tasks"][0]["horizon_family"] == (
        "event_response_fixed_v1"
    )
    assert payload["replay_tasks"][0]["source_evidence_role"] == "trigger_only"
    first_task = payload["replay_tasks"][0]
    assert first_task["as_of"] == "2026-07-01T10:00:00+00:00"
    assert first_task["packet_as_of"] == AS_OF
    assert first_task["trigger_event_at"] == "2026-07-01T10:00:00+00:00"
    assert first_task["trigger_event_timestamp_basis"] == (
        "provenance.published_at"
    )
    assert first_task["due_at"] == "2026-07-02T10:00:00+00:00"
    assert first_task["checkpoint_state_at_packet"] == "scheduled"
    assert payload["summary"]["event_anchored_replay_task_count"] == len(
        OUTCOME_HORIZONS
    )
    assert payload["summary"]["matured_replay_checkpoint_count"] == 0

    graph = payload["scenario_outcome_graph"]
    assert graph["probability_mass_check"] is True
    scenario_prob_sum = sum(
        node["probability"]
        for node in graph["nodes"]
        if node["node_type"] == "scenario"
    )
    assert scenario_prob_sum == pytest.approx(1.0)
    assert any(
        node["node_type"] == "expectation_gap"
        for node in graph["nodes"]
    )
    assert any(
        "Expectation Graph missing" in gap["description"]
        for gap in payload["evidence_gaps"]
    )


def test_world_model_event_learning_conditions_hypotheses_on_pipeline_context():
    pipeline_context = {
        "regime": "risk_off",
        "confidence": 0.72,
        "context_tags": ["high volatility", "semiconductor cycle watch"],
        "metrics": {
            "vix": 32.0,
            "volatility_ratio": 1.6,
            "inflation_yoy": 4.2,
            "yield_curve_slope": -0.35,
            "credit_spread": 2.1,
            "macro_score": -0.28,
            "news_impact_score": -0.85,
        },
    }
    expectation_context = {
        "crowdedness": 0.82,
        "surprise_magnitude": -0.4,
        "tags": ["crowded ai"],
        "watch_metrics": ["earnings_revisions"],
    }
    context = MarketContext(
        as_of=AS_OF,
        tickers=["NVDA"],
        news=[_semantic_news()],
        metadata={
            "pipeline_context": pipeline_context,
            "expectation_context": expectation_context,
        },
        pipeline_result=pipeline_context,
    )

    payload = WorldModelEventLearningPacket().build(
        context,
        domain_id=DOMAIN_ID,
        save=False,
    )

    summary = payload["summary"]
    assert summary["pipeline_indicator_context_status"] == (
        "pipeline_indicator_context_ready"
    )
    assert summary["indicator_metric_count"] >= 7
    assert summary["regime_label"] == "risk_off"
    assert summary["expectation_context_available"] is True
    assert "pipeline_volatility_high" in summary["pipeline_context_tags"]
    assert "expectation_crowded" in summary["pipeline_context_tags"]
    assert "vix" in summary["watch_metrics"]
    assert "earnings_revisions" in summary["watch_metrics"]
    assert not any(
        "Indicator State Grid is not supplied" in gap["description"]
        for gap in payload["evidence_gaps"]
    )
    assert not any(
        "Expectation Graph missing" in gap["description"]
        for gap in payload["evidence_gaps"]
    )

    graph = payload["scenario_outcome_graph"]
    assert any(
        node["label"] == "expectation_context_supplied"
        for node in graph["nodes"]
    )
    assert any(
        item.startswith("pipeline_context_tags=")
        for item in graph["evidence_gaps"]
    )
    replay_task = payload["replay_tasks"][0]
    assert replay_task["manual_review_gate_required"] is True
    assert replay_task["registration_status"] == "candidate_pending_manual_review"
    replay_context = replay_task["pipeline_context_snapshot"]
    assert replay_context["indicator_metric_count"] >= 7
    assert replay_context["expectation_context_available"] is True
    assert "vix" in replay_context["watch_metrics"]


def test_world_model_event_learning_blocks_without_stable_news_locator():
    context = MarketContext(
        as_of=AS_OF,
        tickers=["NVDA"],
        news=[
            _semantic_news(
                url=None,
                link=None,
                title="Headline alone should not pass source audit",
            )
        ],
    )

    payload = WorldModelEventLearningPacket().build(
        context,
        domain_id=DOMAIN_ID,
        save=False,
    )

    assert payload["summary"]["packet_status"] == (
        "blocked_no_point_in_time_event_evidence"
    )
    assert payload["summary"]["classified_event_count"] == 0
    assert payload["summary"]["hypothesis_count"] == 0
    assert payload["scenario_outcome_graph"] is None
    assert any(
        "stable_source_locator_missing" in exclusion.get("reasons", [])
        for exclusion in payload["source_evidence_audit"]["exclusions"]
    )


def test_world_model_event_learning_requires_timezone_aware_as_of():
    context = MarketContext(
        as_of="2026-07-01T12:00:00",
        news=[_semantic_news()],
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        WorldModelEventLearningPacket().build(
            context,
            domain_id=DOMAIN_ID,
            save=False,
        )


def test_world_model_event_selection_represents_lanes_and_deduplicates_sources():
    lanes = [
        ("capex_cycle", "Capex plan expands semiconductor capacity"),
        ("sector_demand", "AI demand accelerates chip orders"),
        ("supply_chain", "Memory shortage constrains chip supply"),
        (
            "policy_or_geopolitical",
            "Export control license requirement applies to advanced computing items",
        ),
        ("market_confirmation", "Semiconductor sector rotation confirms demand"),
    ]
    news = []
    for index, (lane, title) in enumerate(lanes):
        record = _semantic_news(
            title=title,
            summary=title,
            url=f"https://example.test/news/{index}",
        )
        record["_dean_semantic_evidence"] = {
            **record["_dean_semantic_evidence"],
            "evidence_type": lane,
            "matched_terms": [title.split()[0].lower()],
            "candidate_sha256": f"candidate-{index}",
        }
        news.append(record)
    duplicate = _semantic_news(
        title="Duplicate capex rendering of the demand source",
        summary="Duplicate capex rendering of the demand source",
        url="https://example.test/news/1",
    )
    duplicate["_dean_semantic_evidence"] = {
        **duplicate["_dean_semantic_evidence"],
        "evidence_type": "capex_cycle",
        "matched_terms": ["capex"],
        "candidate_sha256": "duplicate-candidate",
    }
    news.append(duplicate)

    payload = WorldModelEventLearningPacket().build(
        MarketContext(as_of=AS_OF, news=news),
        domain_id=DOMAIN_ID,
        max_events=5,
        save=False,
    )

    audit = payload["event_selection_audit"]
    assert audit["all_input_lanes_represented"] is True
    assert audit["selected_event_count"] == 5
    assert audit["selected_unique_source_count"] == 5
    assert set(audit["selected_lane_counts"]) == {lane for lane, _ in lanes}
    assert "sanctions_change" in {
        event["event_class"] for event in payload["classified_events"]
    }
    assert any(
        hypothesis["hypothesis"].startswith("Sanctions will constrain")
        for hypothesis in payload["hypotheses"]
    )
