from __future__ import annotations

import asyncio
import json

import pytest

from dean_os.agents.domain_analyst import DomainAnalystAgent
from dean_os.agents.pipeline_manager import PipelineManagerAgent
from dean_os.agents.pipeline_readiness import load_pipeline_readiness
from dean_os.consensus import ConsensusEngine
from dean_os.orchestrator import DEANOrchestrator
from dean_os.registry import AgentRegistry
from dean_os.schemas import MarketContext, PipelineReport


def test_registry_honors_run_phases_before_loading_agent(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
agents:
  domain_manager:
    class_path: dean_os.agents.pipeline_manager:PipelineManagerAgent
    branch: pipeline
    enabled: true
    veto_level: none
    error_behavior: skip
    domain_id: energy
    run_phases: [pre_trade]
""".strip(),
        encoding="utf-8",
    )
    registry = AgentRegistry(registry_path, project_root=tmp_path)

    assert registry.load_branch(
        "pipeline",
        MarketContext(phase="pre_pipeline"),
    ) == []
    assert len(
        registry.load_branch(
            "pipeline",
            MarketContext(phase="pre_trade"),
        )
    ) == 1


def test_registry_rejects_duplicate_domain_analysis_execution_group(
    tmp_path,
):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
agents:
  direct:
    class_path: dean_os.agents.domain_analyst:DomainAnalystAgent
    branch: pipeline
    enabled: true
    execution_group: energy_analysis
    run_phases: [pre_trade]
  composite:
    class_path: dean_os.agents.pipeline_manager:PipelineManagerAgent
    branch: pipeline
    enabled: true
    execution_group: energy_analysis
    run_phases: [pre_trade]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exclusive execution group"):
        AgentRegistry(registry_path, project_root=tmp_path)


def test_domain_agent_is_review_only_and_requires_explicit_as_of():
    agent = DomainAnalystAgent(
        name="energy_analyst",
        config={"domain_id": "energy"},
    )

    report = asyncio.run(agent.run(MarketContext()))

    assert report.verdict == "needs_more_data"
    assert report.metrics_snapshot["analysis_executed"] is False
    assert report.metrics_snapshot["decision_influence"] is False
    assert report.metrics_snapshot["can_trade"] is False
    assert report.input_hash
    assert report.config_hash


def test_composite_pipeline_manager_uses_context_artifact_binding(
    tmp_path,
):
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    (news_dir / "latest.json").write_text(
        json.dumps(
            {
                "market_context_fragment": {
                    "news": [
                        {
                            "title": "Oil supply update",
                            "summary": (
                                "Energy supply remains constrained"
                            ),
                            "source": "reuters",
                            "published_at": (
                                "2026-07-01T12:00:00+00:00"
                            ),
                            "_dean_semantic_evidence": {
                                "evidence_type": "sector_demand",
                                "source_tier": (
                                    "tier_2_strong_context"
                                ),
                                "source_identity": "reuters",
                                "matched_terms": ["supply"],
                                "stance_hint": "positive",
                            },
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    agent = PipelineManagerAgent(
        name="energy_pipeline_manager",
        config={
            "domain_id": "energy",
            "horizon_days": 180,
            "output_dir": str(tmp_path / "composite_report"),
        },
    )
    context = MarketContext(
        phase="pre_trade",
        as_of="2026-07-02T00:00:00+00:00",
        metadata={
            "domain_artifacts": {
                "energy": {"news": str(news_dir)}
            }
        },
    )

    report = asyncio.run(agent.run(context))

    assert report.metrics_snapshot["artifact_count"] == 1
    binding = report.metrics_snapshot["artifact_bindings"]["news"]
    assert binding["available"] is True
    assert len(binding["sha256"]) == 64
    assert report.metrics_snapshot["decision_influence"] is False
    assert report.metrics_snapshot["supporting_review_only"] is True
    assert report.metrics_snapshot["can_trade"] is False
    assert report.input_hash
    saved_payload = json.loads(
        (
            tmp_path / "composite_report" / "latest.json"
        ).read_text(encoding="utf-8")
    )
    assert saved_payload["mode"] == "pipeline_manager_agent_report"
    assert saved_payload["safety"]["decision_influence"] is False
    assert saved_payload["safety"]["can_trade"] is False


def test_composite_pipeline_manager_saves_final_agent_contract(
    tmp_path,
):
    agent = PipelineManagerAgent(
        name="energy_pipeline_manager",
        config={
            "domain_id": "energy",
            "output_dir": str(tmp_path / "reports"),
        },
    )

    report = asyncio.run(
        agent.run(
            MarketContext(
                phase="pre_trade",
                as_of="2026-07-02T00:00:00+00:00",
            )
        )
    )

    assert report.verdict == "needs_more_data"
    assert "saved_paths" not in report.metrics_snapshot

    # No final artifact is written when required analysis inputs are absent.
    assert not (tmp_path / "reports" / "latest.json").exists()


def test_consensus_ignores_review_only_domain_caution():
    report = PipelineReport(
        agent_name="domain_manager",
        agent_version="test",
        verdict="caution",
        confidence=1.0,
        data_quality_score=1.0,
        signal_strength=1.0,
        metrics_snapshot={"decision_influence": False},
    )

    decision = ConsensusEngine().combine(
        [report],
        {"model_score": 0.0, "timeframe": "1d"},
        [],
    )

    assert decision.decision == "no_trade"
    assert "domain_manager" in decision.agent_report_hashes


def test_pipeline_readiness_keeps_analysis_and_ticker_authority_separate(
    tmp_path,
):
    stage5_sha = "a" * 64
    feature_path = tmp_path / "feature_audit.json"
    feature_path.write_text(
        json.dumps(
            {
                "mode": "pipeline_feature_timeframe_audit",
                "status": (
                    "pipeline_feature_timeframe_audit_blocked_mismatch"
                ),
                "summary": {
                    "timeframe_mismatch_ticker_count": 1,
                    "timeframe_mismatch_tickers": ["AMD"],
                    "timezone_aware_ticker_count": 0,
                    "can_use_for_stage4": False,
                    "can_use_for_stage5": False,
                    "can_trade": False,
                },
                "stage5_candidate_binding": {
                    "sha256": stage5_sha,
                    "relationship_status": (
                        "co_located_same_batch_candidate_not_hash_bound"
                    ),
                },
                "safety": {"can_trade": False},
            }
        ),
        encoding="utf-8",
    )
    prediction_path = tmp_path / "prediction_review.json"
    prediction_path.write_text(
        json.dumps(
            {
                "mode": "pipeline_prediction_review_packet",
                "status": "stage5_prediction_review_partial",
                "context_count": 10,
                "complete_context_count": 0,
                "source_artifact": {"sha256": stage5_sha},
                "safety": {"can_trade": False},
            }
        ),
        encoding="utf-8",
    )
    sector_path = tmp_path / "sector_review.json"
    sector_path.write_text(
        json.dumps(
            {
                "mode": "sector_to_ticker_review_packet",
                "summary": {
                    "packet_status": "review_ready_with_limitations",
                    "ticker_count": 1,
                    "review_ready_count": 0,
                    "blocked_or_context_count": 1,
                    "can_create_ticker_forecast": False,
                    "can_trade": False,
                },
                "safety": {"can_trade": False},
            }
        ),
        encoding="utf-8",
    )

    readiness = load_pipeline_readiness(
        {
            "feature_timeframe_audit": feature_path,
            "prediction_review": prediction_path,
            "sector_to_ticker_review": sector_path,
        }
    )

    assert readiness["status"] == "pipeline_readiness_blocked"
    assert readiness["blocking_reasons"] == [
        "feature_timeframe_cadence_mismatch",
        "stage5_prediction_contexts_quarantined_or_incomplete",
        "zero_review_ready_ticker_candidates",
    ]
    assert readiness["can_use_ticker_pipeline"] is False
    assert readiness["decision_influence"] is False
    assert readiness["can_trade"] is False


def test_pipeline_readiness_binds_target_audit_to_exact_features(
    tmp_path,
):
    feature_sha = "f" * 64
    feature_path = tmp_path / "feature_audit.json"
    feature_path.write_text(
        json.dumps(
            {
                "mode": "pipeline_feature_timeframe_audit",
                "status": "pipeline_feature_timeframe_audit_ready",
                "inputs": {"features_sha256": feature_sha},
                "summary": {
                    "timeframe_mismatch_ticker_count": 0,
                    "timezone_aware_ticker_count": 1,
                    "can_use_for_stage4": True,
                    "can_use_for_stage5": True,
                    "can_trade": False,
                },
                "safety": {"can_trade": False},
            }
        ),
        encoding="utf-8",
    )
    target_path = tmp_path / "target_audit.json"
    target_path.write_text(
        json.dumps(
            {
                "mode": "pipeline_target_readiness_audit",
                "status": "pipeline_target_readiness_ready",
                "summary": {
                    "target_count": 2,
                    "ready_target_count": 2,
                    "blocked_target_count": 0,
                    "can_use_for_stage4": True,
                    "can_trade": False,
                },
                "lineage_bindings": {
                    "target_sha256": "t" * 64,
                    "feature_artifact": {"sha256": feature_sha},
                },
                "safety": {"can_trade": False},
            }
        ),
        encoding="utf-8",
    )

    readiness = load_pipeline_readiness(
        {
            "feature_timeframe_audit": feature_path,
            "target_readiness": target_path,
        }
    )

    assert readiness["errors"] == []
    assert readiness["target_count"] == 2
    assert readiness["ready_target_count"] == 2
    assert readiness["can_use_for_stage4"] is True
    assert readiness["blocking_reasons"] == [
        "stage5_prediction_review_not_supplied"
    ]


def test_pipeline_readiness_surfaces_stage4_validation_failure(tmp_path):
    feature_sha = "f" * 64
    target_sha = "t" * 64
    stage4_path = tmp_path / "stage4_review.json"
    stage4_path.write_text(
        json.dumps(
            {
                "mode": "pipeline_stage4_exact_context_review",
                "status": (
                    "walk_forward_candidate_blocked_by_validation_contract"
                ),
                "scope": {
                    "ticker": "NVDA",
                    "timeframe": "15m",
                    "target_name": "target_intraday_up_15m",
                },
                "parent_lineage": {
                    "features": {"sha256": feature_sha},
                    "targets": {"sha256": target_sha},
                },
                "summary": {
                    "fold_count": 3,
                    "contract_passed": False,
                    "failed_contract_checks": [
                        "mean_train_validation_gap_at_most_0_20"
                    ],
                    "can_promote_model": False,
                    "can_trade": False,
                },
                "safety": {"can_trade": False},
            }
        ),
        encoding="utf-8",
    )

    readiness = load_pipeline_readiness(
        {"stage4_review": stage4_path}
    )

    assert readiness["status"] == "pipeline_readiness_blocked"
    assert readiness["stage4_contract_passed"] is False
    assert readiness["stage4_failed_contract_checks"] == [
        "mean_train_validation_gap_at_most_0_20"
    ]
    assert readiness["blocking_reasons"] == [
        "stage4_validation_contract_failed",
        "stage5_prediction_review_not_supplied",
    ]
    assert readiness["can_use_ticker_pipeline"] is False


def test_dynamic_hard_veto_name_blocks_orchestrator(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
agents:
  custom_guard:
    class_path: dean_os.agents.pipeline_audit:PipelineAuditAgent
    branch: pipeline
    enabled: true
    veto_level: hard
    error_behavior: block
    required_inputs: [missing.json]
""".strip(),
        encoding="utf-8",
    )
    registry = AgentRegistry(registry_path, project_root=tmp_path)
    pipeline_calls = []
    orchestrator = DEANOrchestrator(
        registry=registry,
        pipeline_runner=lambda context: pipeline_calls.append(
            context.phase
        ),
        consensus=ConsensusEngine(
            hard_veto_agents=registry.hard_veto_agent_names()
        ),
    )

    decision = asyncio.run(
        orchestrator.run(MarketContext(tickers=["NVDA"]))
    )

    assert pipeline_calls == []
    assert decision.decision == "blocked"
    assert decision.blocking_agents == ["custom_guard"]
