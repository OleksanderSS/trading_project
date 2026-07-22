from __future__ import annotations

import asyncio

from dean_os.agents.tuning import TuningAgent
from dean_os.schemas import MarketContext


def test_tuning_agent_blocks_tuning_when_control_surface_blocks():
    context = MarketContext(
        tickers=["AMD"],
        timeframes=["1d"],
        metadata={
            "model_performance": {
                "threshold_failures": ["drawdown_above_threshold"],
                "performance_score": 0.42,
                "evaluation_scope": {
                    "ticker": "AMD",
                    "model": "random_forest",
                    "target_name": "target_intraday_up_15m",
                    "timeframe": "1d",
                    "context_fingerprint": "ctx-amd-1d",
                },
            },
            "pipeline_control_surface": {
                "surface": {"status": "blocked", "allowed_variation": {"max_trials": 0}},
                "proposal_gate": {
                    "status": "blocked",
                    "can_propose_tuning": False,
                    "reason": "Blocked axes remain.",
                },
            },
        },
    )

    report = asyncio.run(TuningAgent(name="tuning", config={}).run(context))

    assert report.metrics_snapshot["status"] == "control_surface_blocked"
    assert context.metadata["tuning"]["proposal_count"] == 1
    assert context.action_proposals[0].action_type == "validate"
    assert context.action_proposals[0].target == "pipeline_control_surface"
    assert all(proposal.action_type != "tune" for proposal in context.action_proposals)


def test_tuning_agent_adds_control_surface_bounds_to_tuning_proposal():
    context = MarketContext(
        tickers=["AMD"],
        timeframes=["1d"],
        metadata={
            "model_performance": {
                "threshold_failures": ["drawdown_above_threshold"],
                "performance_score": 0.42,
                "evaluation_scope": {
                    "ticker": "AMD",
                    "model": "random_forest",
                    "target_name": "target_intraday_up_15m",
                    "timeframe": "1d",
                    "context_fingerprint": "ctx-amd-1d",
                },
            },
            "pipeline_control_surface": {
                "surface": {
                    "status": "caution",
                    "allowed_variation": {
                        "max_trials": 10,
                        "parameter_delta_pct": 0.1,
                        "max_feature_additions": 1,
                        "production_write_allowed": False,
                    },
                },
                "proposal_gate": {
                    "status": "review_required",
                    "can_propose_tuning": True,
                    "reason": "Reviewed bounded experiment only.",
                },
            },
        },
    )

    report = asyncio.run(TuningAgent(name="tuning", config={}).run(context))

    assert report.metrics_snapshot["status"] == "tuning_experiment_proposed"
    assert context.action_proposals[0].action_type == "tune"
    command = context.action_proposals[0].command_preview
    assert "--max-trials 10" in command
    assert "--parameter-delta-pct 0.1" in command
    assert "--max-feature-additions 1" in command
    assert "--no-production-write" in command
