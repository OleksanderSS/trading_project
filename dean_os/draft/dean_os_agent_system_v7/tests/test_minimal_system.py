from __future__ import annotations

import asyncio
import sys

import pytest
from pathlib import Path

from dean_os.agents.domain_analytical import DomainAnalyticalAgent
from dean_os.agents.pipeline_control import PipelineControlAgent
from dean_os.analysts.profiles import get_domain_profile, list_domain_profiles
from dean_os.draft.dean_os_agent_system_v7.dean_os.minimal_system import create_minimal_system
from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_adapter import HybridPipelineAdapter
from dean_os.schemas import AnalyticalReport, MarketContext

AS_OF = "2026-07-12T12:00:00+00:00"


def run(coro):
    return asyncio.run(coro)


def test_semiconductor_profile_is_packaged_and_loadable() -> None:
    profile = get_domain_profile("semiconductor_ai_infrastructure")

    assert profile.domain_id == "semiconductor_ai_infrastructure"
    assert profile.horizon_days_default == 180
    assert "demand" in profile.required_evidence_types
    assert "semiconductor_ai_infrastructure" in list_domain_profiles()


def test_agents_package_uses_lazy_optional_dependency_imports() -> None:
    import dean_os.agents as agents

    assert agents.PipelineControlAgent is PipelineControlAgent
    assert "dean_os.agents.unified_research_agent" not in sys.modules
    assert "duckdb" not in sys.modules


def test_pipeline_control_agent_writes_bounded_execution_policy(tmp_path: Path) -> None:
    context = MarketContext(as_of=AS_OF)
    agent = PipelineControlAgent(
        name="pipeline_control",
        config={
            "allowed_stages": [0, 1, 2, 3, 4, 5, 7],
            "save_surface": False,
            "output_dir": str(tmp_path),
            "constraints": {
                "max_drawdown": 0.25,
                "max_train_test_gap": 0.15,
            },
        },
    )

    report = run(agent.run(context))
    policy = context.metadata["pipeline_execution_policy"]

    assert report.branch == "pipeline"
    assert policy["allowed_stages"] == [0, 1, 2, 3, 4, 5, 7]
    assert policy["production_config_write_allowed"] is False
    assert policy["model_promotion_allowed"] is False
    assert policy["learning_memory_write_allowed"] is False
    assert policy["paper_or_live_trade_allowed"] is False
    assert policy["requires_human_review"] is True


def test_pipeline_adapter_degrades_on_first_missing_dependency_call() -> None:
    def unavailable_factory():
        raise ModuleNotFoundError("optional pipeline dependency is absent")

    adapter = HybridPipelineAdapter(orchestrator_factory=unavailable_factory)
    context = MarketContext(
        as_of=AS_OF,
        tickers=["NVDA"],
        timeframes=["1d"],
    )

    result = run(adapter(context))

    assert result["status"] == "pipeline_skipped"
    assert result["pipeline_skipped"] is True
    assert "ModuleNotFoundError" in result["skip_reason"]
    assert context.pipeline_result["status"] == "pipeline_skipped"


def test_pipeline_policy_blocks_runner_before_backend_call() -> None:
    class Backend:
        called = False

        async def run_local_pipeline(self, **kwargs):
            self.called = True
            return {"status": "success"}

    backend = Backend()
    adapter = HybridPipelineAdapter(orchestrator=backend)
    context = MarketContext(
        as_of=AS_OF,
        tickers=["NVDA"],
        metadata={
            "pipeline_execution_policy": {
                "pipeline_run_allowed": False,
                "allowed_stages": [0, 1],
            }
        },
    )

    result = run(adapter(context))

    assert backend.called is False
    assert result["status"] == "pipeline_skipped"
    assert result["skip_reason"] == "blocked_by_pipeline_execution_policy"


def test_domain_agent_is_analytical_and_review_only() -> None:
    agent = DomainAnalyticalAgent(
        name="domain_analyst",
        config={
            "domain_id": "semiconductor_ai_infrastructure",
            "horizon_days": 180,
        },
    )
    context = MarketContext(as_of=AS_OF)

    report = run(agent.run(context))

    assert isinstance(report, AnalyticalReport)
    assert report.branch == "analytical"
    assert report.asset_or_sector == "semiconductor_ai_infrastructure"
    assert report.signal_strength == 0.0
    assert report.analysis_payload["authority_boundary"]["review_only"] is True
    assert report.analysis_payload["authority_boundary"]["can_trade"] is False


def test_minimal_system_runs_both_branches_with_bounded_fake_pipeline(tmp_path: Path) -> None:
    class FakeHybridOrchestrator:
        def __init__(self):
            self.calls: list[dict] = []

        async def run_local_pipeline(self, *, tickers, timeframes, stages_to_run=None):
            self.calls.append(
                {
                    "tickers": tickers,
                    "timeframes": timeframes,
                    "stages_to_run": stages_to_run,
                }
            )
            return {
                "status": "success",
                "as_of": AS_OF,
                "results": {
                    "model_metrics": {
                        "pnl": 1.0,
                        "sharpe": 1.2,
                    }
                },
            }

    backend = FakeHybridOrchestrator()
    system = create_minimal_system(
        project_root=tmp_path,
        domain_id="semiconductor_ai_infrastructure",
        orchestrator=backend,
        soft_mode=True,
    )
    context = MarketContext(
        as_of=AS_OF,
        tickers=["NVDA"],
        timeframes=["1d"],
    )

    result = run(system.run(context))
    report_names = {item["agent_name"] for item in result.agent_reports}

    assert result.status == "completed"
    assert result.decision.decision == "watchlist"
    assert result.review_only is True
    assert backend.calls == [
        {
            "tickers": ["NVDA"],
            "timeframes": ["1d"],
            "stages_to_run": [0, 1, 2, 3, 4, 5, 7],
        }
    ]
    assert result.pipeline_result["status"] == "success"
    assert result.pipeline_execution_policy["pipeline_run_allowed"] is True
    assert "pipeline_control" in report_names
    assert "domain_analyst" in report_names
    assert result.world_model_event_learning["summary"]["domain_id"] == (
        "semiconductor_ai_infrastructure"
    )
    assert result.safety["can_trade"] is False


def test_world_model_creates_hypothesis_and_probability_graph_from_point_in_time_news() -> None:
    from dean_os.world_model.world_model_event_learning import WorldModelEventLearningPacket

    context = MarketContext(
        as_of=AS_OF,
        tickers=["TSM"],
        news=[
            {
                "title": (
                    "TSMC expands advanced packaging capacity as AI accelerator "
                    "demand grows"
                ),
                "summary": (
                    "Additional CoWoS capacity is intended to address HBM and "
                    "advanced-packaging constraints."
                ),
                "published_at": "2026-07-11T08:00:00+00:00",
                "url": "https://example.test/tsmc-cowos-capacity",
                "tickers": ["TSM"],
                "_dean_semantic_evidence": {
                    "producer_contract": "test_semantic_evidence_v1",
                    "evidence_type": "supply",
                    "matched_terms": ["advanced packaging", "cowos"],
                    "required_lane_eligible": True,
                    "source_tier": "tier_2_strong_context",
                    "source_identity": "example_test",
                    "candidate_sha256": "fixture-candidate",
                    "stance_hint": "positive",
                },
            }
        ],
        metadata={
            "stage7_regime_review": {"regime": "AI_CAPEX_BOOM"},
            "expectation_context": {
                "status": "crowded_positive_expectations"
            },
        },
    )

    payload = WorldModelEventLearningPacket().build(
        context,
        domain_id="semiconductor_ai_infrastructure",
        save=False,
    )

    assert payload["summary"]["packet_status"] == (
        "world_model_event_learning_ready_pending_replay"
    )
    assert payload["summary"]["classified_event_count"] == 1
    assert payload["summary"]["hypothesis_count"] == 1
    assert payload["summary"]["scenario_probability_mass_valid"] is True
    assert payload["classified_events"][0]["event_class"] == (
        "supply_disruption"
    )
    assert payload["hypotheses"][0]["horizons_to_check"]
    assert payload["scenario_outcome_graph"] is not None
    assert payload["safety"]["can_trade"] is False


def test_pipeline_metric_snapshot_normalizes_current_run_without_cross_family_leakage() -> None:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_metrics import PipelineMetricNormalizer

    raw = {
        "status": "success",
        "as_of": AS_OF,
        "tickers": ["NVDA"],
        "timeframes": ["1d"],
        "results": {
            "model_metrics": {
                "pnl": 2.5,
                "total_return_pct": 0.12,
                "sharpe_ratio": 1.4,
                "max_drawdown_pct": 0.08,
                "train_score": 0.78,
                "validation_score": 0.71,
                "sample_count": 240,
                "feature_importance": {"hbm_supply": 0.4, "capex": 0.35, "rates": 0.25},
                "feature_stability_score": 0.82,
                "warnings": [],
                "leakage_flags": [],
            }
        },
    }

    snapshot = PipelineMetricNormalizer().from_pipeline_result(raw)

    assert snapshot.identity.tickers == ["NVDA"]
    assert snapshot.profitability.total_return == 0.12
    assert snapshot.risk.max_drawdown == 0.08
    assert snapshot.validation.train_test_gap == pytest.approx(0.07)
    assert snapshot.feature_stability.feature_concentration == 0.4
    assert snapshot.data_quality.warning_count == 0
    assert snapshot.evidence_availability["data_quality"] is True
    # sample_count is a model-validation metric and must not be silently reused
    # as a replay sample count.
    assert snapshot.replay.clear_evaluated_runs is None
    assert snapshot.evidence_availability["replay"] is False


def test_pipeline_control_reassesses_current_run_and_controls_next_run() -> None:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_metrics import PipelineMetricNormalizer

    context = MarketContext(as_of=AS_OF, tickers=["NVDA"], timeframes=["1d"])
    agent = PipelineControlAgent(
        name="pipeline_control",
        config={
            "allowed_stages": [0, 1, 2, 3, 4, 5, 7],
            "save_surface": False,
            "constraints": {"min_pnl": 0.0},
        },
    )

    context.phase = "pre_pipeline"
    run(agent.run(context))
    assert context.metadata["pipeline_execution_policy"]["pipeline_run_allowed"] is True

    snapshot = PipelineMetricNormalizer().from_pipeline_result(
        {
            "status": "success",
            "as_of": AS_OF,
            "tickers": ["NVDA"],
            "timeframes": ["1d"],
            "results": {
                "model_metrics": {
                    "pnl": -1.0,
                    "max_drawdown": 0.1,
                    "train_score": 0.7,
                    "validation_score": 0.65,
                    "sample_count": 100,
                    "warnings": [],
                    "leakage_flags": [],
                }
            },
        }
    )
    context.metadata["pipeline_metric_snapshot"] = snapshot.model_dump(mode="json")
    context.phase = "pre_trade"
    report = run(agent.run(context))
    policy = context.metadata["pipeline_execution_policy"]

    assert report.verdict == "blocked"
    # The completed run was allowed by preflight; post-run evidence controls the
    # next run rather than rewriting history.
    assert policy["pipeline_run_allowed"] is True
    assert policy["next_pipeline_run_allowed"] is False
    assert policy["post_run_assessment"]["current_run_evidence_accepted"] is True
    assert set(context.metadata["pipeline_control_surfaces"]) == {"pre_pipeline", "pre_trade"}


def test_context_and_indicator_grids_are_domain_portable_and_point_in_time() -> None:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.context_grids import ContextIndicatorGridBuilder
    from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_metrics import PipelineMetricNormalizer

    snapshot = PipelineMetricNormalizer().from_pipeline_result(
        {
            "status": "success",
            "as_of": AS_OF,
            "tickers": ["TSM"],
            "timeframes": ["1d"],
            "results": {
                "model_metrics": {
                    "total_return": 0.09,
                    "max_drawdown": 0.07,
                    "validation_score": 0.68,
                    "sample_count": 120,
                }
            },
        }
    )
    context = MarketContext(
        as_of=AS_OF,
        tickers=["TSM"],
        macro={
            "policy_rate": {
                "value": 5.25,
                "unit": "percent",
                "period": "2026-07",
                "available_at": "2026-07-11T12:00:00+00:00",
                "source_url": "https://example.test/official-rate",
            }
        },
        metadata={
            "regime_dimensions": {
                "economic_phase": {"state": "expansion", "confidence": 0.65},
                "ai_cycle": {"state": "capex_boom", "confidence": 0.8},
                "geopolitical_phase": {"state": "localized_war", "confidence": 0.9},
            }
        },
    )
    report = AnalyticalReport(
        agent_name="domain_analyst",
        agent_version="1.0.0",
        verdict="caution",
        confidence=0.7,
        data_quality_score=0.6,
        signal_strength=0.0,
        asset_or_sector="semiconductor_ai_infrastructure",
        horizon_years=0.5,
        thesis="HBM and packaging remain the main bottleneck.",
        data_quality="partial",
        position_bias="neutral",
        analysis_payload={
            "domain_id": "semiconductor_ai_infrastructure",
            "domain_metrics": {
                "stance": "mixed",
                "expectation_gap": {"status": "crowded_positive_expectations"},
                "evidence_gaps": ["Foundry utilization evidence is incomplete."],
            },
        },
    )

    packet = ContextIndicatorGridBuilder().build(
        context,
        domain_id="semiconductor_ai_infrastructure",
        agent_reports=[report],
        pipeline_metric_snapshot=snapshot,
    )

    node_ids = {node.node_id for node in packet.context_grid.nodes}
    assert "global" in node_ids
    assert "sector:semiconductor_ai_infrastructure" in node_ids
    assert any(node.level == "adjacent_sector" for node in packet.context_grid.nodes)
    global_node = next(node for node in packet.context_grid.nodes if node.node_id == "global")
    assert global_node.dimensions["economic_phase"].state == "expansion"
    assert global_node.dimensions["credit_phase"].state == "unknown"
    assert packet.indicator_state_grid.family_counts["macro"] == 1
    assert packet.indicator_state_grid.family_counts["pipeline_profitability"] >= 1
    assert packet.authority_boundary["can_trade"] is False


def test_src_pipeline_bridge_emits_canonical_metric_snapshot() -> None:
    from src.agents.pipeline_bridge import PipelineBridge

    context = PipelineBridge(project_root=".").from_pipeline_result(
        {
            "status": "success",
            "ticker": "NVDA",
            "timeframe": "1d",
            "financial_metrics": {
                "total_return_pct": 0.11,
                "sharpe_ratio": 1.25,
                "max_drawdown_pct": 0.09,
            },
            "validation_metrics": {
                "validation_score": 0.67,
                "sample_count": 90,
            },
        },
        as_of=AS_OF,
    )

    snapshot = context.metadata["pipeline_metric_snapshot"]
    assert snapshot["schema_version"] == "dean_pipeline_metric_snapshot_v1"
    assert snapshot["profitability"]["total_return"] == 0.11
    assert snapshot["risk"]["max_drawdown"] == 0.09
    assert context.pipeline_result["dean_os_pipeline_metric_snapshot"]["identity"]["tickers"] == ["NVDA"]
