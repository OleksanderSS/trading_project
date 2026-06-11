from __future__ import annotations

from typing import Any

from dean_os.base import BaseAgent
from dean_os.regime_context import normalize_context_tags
from dean_os.schemas import MarketContext, PipelineActionProposal, PipelineReport


ACTIONABLE_MODEL_FAILURES = {
    "validation_score_below_threshold",
    "sharpe_below_threshold",
    "drawdown_above_threshold",
}
VALIDATION_FIRST_FAILURES = {
    "missing_evaluation_metrics",
    "missing_recognized_metrics",
    "missing_evaluation_timestamp",
    "evaluation_artifact_stale",
    "sample_count_below_threshold",
}


class TuningAgent(BaseAgent):
    """Creates reviewable tuning experiment proposals without changing the pipeline."""

    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        plan = self._build_plan(context)
        proposals = plan["proposals"]
        context.action_proposals.extend(proposals)
        context.metadata["tuning"] = {
            **{key: value for key, value in plan.items() if key != "proposals"},
            "proposal_count": len(proposals),
            "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
        }

        verdict = "caution" if proposals else "clear"
        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=0.82 if proposals else 0.65,
            data_quality_score=plan["data_quality_score"],
            signal_strength=0.0,
            reasons=plan["reasons"] or ["No tuning experiment proposed."],
            risks=plan["risks"],
            blind_spots=[
                "TuningAgent proposes experiments only; it does not run Optuna, retrain models, or write production config."
            ],
            evidence=[
                self.evidence("operation", "context.action_proposals", "proposal_count", len(proposals)),
                self.evidence("metric", "context.metadata.model_performance", "threshold_failures", plan["model_failures"]),
                self.evidence("metric", "context.metadata.regime_context", "regime", plan["regime"]),
                self.evidence("metric", "context.metadata.data_freshness", "stale_sources", plan["stale_sources"]),
                self.evidence("config", "agent_config", "proposal_only", True),
            ],
            input_hash=self.context_hash(context),
            metrics_snapshot={
                **{key: value for key, value in plan.items() if key != "proposals"},
                "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
            },
        )

    def _build_plan(self, context: MarketContext) -> dict[str, Any]:
        model_performance = _as_dict(context.metadata.get("model_performance"))
        regime_context = _as_dict(context.metadata.get("regime_context"))
        context_performance = _as_dict(context.metadata.get("context_performance"))
        model_failures = list(model_performance.get("threshold_failures") or [])
        stale_sources = _stale_sources(context)
        regime = str(regime_context.get("regime") or "UNKNOWN")
        regime_tags = normalize_context_tags(
            [*context.metadata.get("regime_tags", []), *(regime_context.get("context_tags") or [])]
        )
        weak_contexts = _matching_weak_contexts(context_performance, regime_tags)
        tickers = self.config.get("tickers") or context.tickers or ["<approved>"]
        timeframes = self.config.get("timeframes") or context.timeframes or ([context.timeframe] if context.timeframe else ["<approved>"])

        plan: dict[str, Any] = {
            "status": "no_action",
            "reasons": [],
            "risks": [
                "Any tuning experiment must preserve locked holdout windows, walk-forward validation, and human review.",
                "Production config and model artifacts must not be changed by this proposal.",
            ],
            "data_quality_score": 0.7,
            "model_failures": model_failures,
            "stale_sources": stale_sources,
            "regime": regime,
            "regime_tags": regime_tags,
            "weak_contexts": weak_contexts,
            "experiment_scope": {
                "tickers": tickers,
                "timeframes": timeframes,
                "regime_tags": regime_tags,
                "target": "walk_forward_tuning_experiment",
            },
            "guardrails": [
                "walk_forward_validation",
                "locked_holdout",
                "no_production_config_write",
                "risk_constraints",
                "human_approval_required",
            ],
            "proposals": [],
        }

        if stale_sources:
            plan["status"] = "validate_inputs_first"
            plan["data_quality_score"] = 0.45
            plan["reasons"].append(f"Stale data sources must be refreshed or validated before tuning: {', '.join(stale_sources)}.")
            plan["proposals"].append(self._validation_proposal(plan, target="tuning_inputs"))
            return plan

        if not model_performance:
            plan["status"] = "validate_metrics_first"
            plan["data_quality_score"] = 0.35
            plan["model_failures"] = ["missing_evaluation_metrics"]
            plan["reasons"].append("No model performance metrics are available for tuning review.")
            plan["proposals"].append(self._validation_proposal(plan, target="model_performance"))
            return plan

        actionable_failures = [failure for failure in model_failures if failure in ACTIONABLE_MODEL_FAILURES]
        validation_failures = [failure for failure in model_failures if failure in VALIDATION_FIRST_FAILURES]
        if actionable_failures:
            plan["status"] = "tuning_experiment_proposed"
            plan["reasons"].append(
                f"Model performance is below tuning thresholds: {', '.join(actionable_failures)}."
            )
            if regime_tags:
                plan["reasons"].append(f"Scope experiment by regime tags: {', '.join(regime_tags)}.")
            if weak_contexts:
                plan["risks"].append("Historical weak contexts require stronger validation before promotion.")
            plan["proposals"].append(self._tuning_proposal(plan, model_performance))
            return plan

        if validation_failures or model_performance.get("verdict") == "caution":
            plan["status"] = "validate_metrics_first"
            plan["data_quality_score"] = 0.5
            plan["reasons"].append("Model evidence is incomplete or stale; validate metrics before tuning.")
            plan["proposals"].append(self._validation_proposal(plan, target="model_performance"))
            return plan

        plan["reasons"].append("Model performance is acceptable under current thresholds; no tuning proposal needed.")
        return plan

    def _tuning_proposal(self, plan: dict[str, Any], model_performance: dict[str, Any]) -> PipelineActionProposal:
        tickers = " ".join(str(ticker) for ticker in plan["experiment_scope"]["tickers"])
        timeframes = " ".join(str(timeframe) for timeframe in plan["experiment_scope"]["timeframes"])
        return PipelineActionProposal(
            agent_name=self.name,
            action_type="tune",
            target="walk_forward_tuning_experiment",
            reason="; ".join(plan["reasons"]),
            command_preview=(
                "approved experiment only: walk-forward tuning "
                f"--tickers {tickers} --timeframes {timeframes} --locked-holdout --no-production-write"
            ),
            expected_effect="Produce candidate hyperparameter or model-selection changes for review, not automatic promotion.",
            risks=plan["risks"],
            evidence=[
                self.evidence("metric", "context.metadata.model_performance", "threshold_failures", plan["model_failures"]),
                self.evidence("metric", "context.metadata.model_performance", "performance_score", model_performance.get("performance_score")),
                self.evidence("metric", "context.metadata.regime_context", "regime_tags", plan["regime_tags"]),
            ],
        )

    def _validation_proposal(self, plan: dict[str, Any], target: str) -> PipelineActionProposal:
        return PipelineActionProposal(
            agent_name=self.name,
            action_type="validate",
            target=target,
            reason="; ".join(plan["reasons"]),
            command_preview="review/refresh evaluation inputs before any tuning experiment is approved",
            expected_effect="Prevent tuning on stale, missing, or incomplete evidence.",
            risks=plan["risks"],
            evidence=[
                self.evidence("metric", "context.metadata.model_performance", "threshold_failures", plan["model_failures"]),
                self.evidence("metric", "context.metadata.data_freshness", "stale_sources", plan["stale_sources"]),
            ],
        )


def _stale_sources(context: MarketContext) -> list[str]:
    freshness = context.metadata.get("data_freshness", {})
    if not isinstance(freshness, dict):
        return []
    return sorted(name for name, info in freshness.items() if isinstance(info, dict) and info.get("stale"))


def _matching_weak_contexts(context_performance: dict[str, Any], regime_tags: list[str]) -> list[dict[str, Any]]:
    weak_contexts = context_performance.get("weak_contexts", [])
    if not isinstance(weak_contexts, list):
        return []
    if not regime_tags:
        return weak_contexts[:3]
    matches = []
    for bucket in weak_contexts:
        if not isinstance(bucket, dict):
            continue
        tags = normalize_context_tags(
            [
                str(bucket.get("context_tag", "")),
                str(bucket.get("regime_tag", "")),
                *bucket.get("context_tags", []),
                *bucket.get("regime_tags", []),
            ]
        )
        if any(tag in tags for tag in regime_tags):
            matches.append(bucket)
    return matches[:3]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
