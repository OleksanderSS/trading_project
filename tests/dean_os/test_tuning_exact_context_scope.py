from __future__ import annotations

import asyncio

from dean_os.agents.model_performance import (
    _extract_evaluation_scope,
)
from dean_os.agents.tuning import TuningAgent
from dean_os.schemas import MarketContext


def _scope():
    return {
        "ticker": "AMD",
        "model": "random_forest",
        "target_name": "target_intraday_up_15m",
        "timeframe": "15m",
        "context_fingerprint": "ctx-amd-15m",
    }


def _surface():
    return {
        "surface": {
            "status": "caution",
            "allowed_variation": {
                "max_trials": 5,
                "production_write_allowed": False,
            },
        },
        "proposal_gate": {
            "status": "review_required",
            "can_propose_tuning": True,
        },
    }


def test_one_ticker_failure_does_not_broaden_to_sector_cohort():
    context = MarketContext(
        tickers=["NVDA", "AMD", "INTC", "TSM"],
        timeframes=["15m"],
        metadata={
            "model_performance": {
                "threshold_failures": [
                    "validation_score_below_threshold"
                ],
                "performance_score": 0.42,
                "evaluation_scope": _scope(),
            },
            "pipeline_control_surface": _surface(),
        },
    )

    report = asyncio.run(
        TuningAgent(name="tuning", config={}).run(context)
    )

    assert report.metrics_snapshot["status"] == (
        "tuning_experiment_proposed"
    )
    scope = report.metrics_snapshot["experiment_scope"]
    assert scope["tickers"] == ["AMD"]
    assert scope["timeframes"] == ["15m"]
    assert scope["domain_or_sector_scope_inherited"] is False
    command = context.action_proposals[0].command_preview
    assert "--tickers AMD " in command
    assert "NVDA" not in command
    assert "INTC" not in command
    assert "TSM" not in command


def test_actionable_failure_without_exact_scope_only_validates():
    context = MarketContext(
        tickers=["NVDA", "AMD", "INTC", "TSM"],
        timeframes=["15m"],
        metadata={
            "model_performance": {
                "threshold_failures": [
                    "validation_score_below_threshold"
                ],
                "performance_score": 0.42,
            },
            "pipeline_control_surface": _surface(),
        },
    )

    report = asyncio.run(
        TuningAgent(name="tuning", config={}).run(context)
    )

    assert report.metrics_snapshot["status"] == (
        "validate_exact_model_scope_first"
    )
    proposal = context.action_proposals[0]
    assert proposal.action_type == "validate"
    assert proposal.target == "model_evaluation_scope"


def test_config_cannot_broaden_evaluated_ticker_scope():
    context = MarketContext(
        tickers=["NVDA", "AMD", "INTC", "TSM"],
        timeframes=["15m"],
        metadata={
            "model_performance": {
                "threshold_failures": [
                    "drawdown_above_threshold"
                ],
                "evaluation_scope": _scope(),
            },
            "pipeline_control_surface": _surface(),
        },
    )

    report = asyncio.run(
        TuningAgent(
            name="tuning",
            config={
                "tickers": ["NVDA", "AMD", "INTC", "TSM"],
            },
        ).run(context)
    )

    assert report.metrics_snapshot["status"] == (
        "validate_exact_model_scope_first"
    )
    assert (
        "configured_tickers_broaden_evaluated_scope"
        in report.metrics_snapshot["scope_mismatches"]
    )
    assert all(
        proposal.action_type != "tune"
        for proposal in context.action_proposals
    )


def test_model_performance_preserves_joined_lineage_scope():
    scope = _extract_evaluation_scope(
        {
            "joined_lineage": {
                **_scope(),
                "evaluation_window": {
                    "start": "2026-06-18T00:00:00+00:00"
                },
            }
        }
    )

    assert scope == _scope()
