from __future__ import annotations

from typing import Any

from dean_os.base import BaseAgent
from dean_os.draft.dean_os_agent_system_v7.dean_os.regime_context import normalize_context_tags
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
EXACT_SCOPE_FIELDS = (
    "ticker",
    "model",
    "target_name",
    "timeframe",
    "context_fingerprint",
)


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
        evaluation_scope = _as_dict(
            model_performance.get("evaluation_scope")
        )
        scope_complete = all(
            evaluation_scope.get(field)
            for field in EXACT_SCOPE_FIELDS
        )
        tickers = (
            [str(evaluation_scope["ticker"])]
            if scope_complete
            else []
        )
        timeframes = (
            [str(evaluation_scope["timeframe"])]
            if scope_complete
            else []
        )
        scope_mismatches = _scope_mismatches(
            evaluation_scope=evaluation_scope,
            context=context,
            configured_tickers=self.config.get("tickers"),
            configured_timeframes=self.config.get("timeframes"),
        )
        control_surface = _as_dict(context.metadata.get("pipeline_control_surface"))
        surface = _as_dict(control_surface.get("surface"))
        proposal_gate = _as_dict(control_surface.get("proposal_gate"))
        allowed_variation = _as_dict(surface.get("allowed_variation"))

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
            "evaluation_scope": evaluation_scope,
            "evaluation_scope_complete": scope_complete,
            "scope_mismatches": scope_mismatches,
            "control_surface_status": surface.get("status"),
            "control_surface_gate": proposal_gate.get("status"),
            "allowed_variation": allowed_variation,
            "experiment_scope": {
                "tickers": tickers,
                "timeframes": timeframes,
                "regime_tags": regime_tags,
                "target": "walk_forward_tuning_experiment",
                "model": evaluation_scope.get("model"),
                "target_name": evaluation_scope.get("target_name"),
                "context_fingerprint": evaluation_scope.get(
                    "context_fingerprint"
                ),
                "domain_or_sector_scope_inherited": False,
            },
            "guardrails": [
                "walk_forward_validation",
                "locked_holdout",
                "no_production_config_write",
                "risk_constraints",
                "human_approval_required",
                "single_exact_model_context_only",
            ],
            "proposals": [],
        }

        if control_surface and not proposal_gate.get("can_propose_tuning", False):
            plan["status"] = "control_surface_blocked"
            plan["data_quality_score"] = 0.35
            plan["reasons"].append(
                f"Pipeline control surface blocks tuning proposals: {proposal_gate.get('reason', 'no reason supplied')}."
            )
            plan["risks"].append("Tuning outside the approved control surface can overfit or bypass data-quality gates.")
            plan["proposals"].append(self._validation_proposal(plan, target="pipeline_control_surface"))
            return plan

        if self.config.get("require_control_surface") and not control_surface:
            plan["status"] = "validate_control_surface_first"
            plan["data_quality_score"] = 0.45
            plan["reasons"].append("Pipeline control surface is required before tuning proposals can be created.")
            plan["proposals"].append(self._validation_proposal(plan, target="pipeline_control_surface"))
            return plan

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
        if actionable_failures and (
            not scope_complete or scope_mismatches
        ):
            plan["status"] = "validate_exact_model_scope_first"
            plan["data_quality_score"] = 0.35
            if not scope_complete:
                plan["reasons"].append(
                    "Actionable model failures lack exact ticker, model, "
                    "target, timeframe, and context-fingerprint lineage."
                )
            if scope_mismatches:
                plan["reasons"].append(
                    "Configured/context tuning scope conflicts with the "
                    "evaluated model context: "
                    + ", ".join(scope_mismatches)
                    + "."
                )
            plan["risks"].append(
                "A single ticker/model failure must never broaden into a "
                "sector, domain, or multi-ticker tuning experiment."
            )
            plan["proposals"].append(
                self._validation_proposal(
                    plan,
                    target="model_evaluation_scope",
                )
            )
            return plan
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
        variation = plan.get("allowed_variation") or {}
        variation_preview = _variation_preview(variation)
        return PipelineActionProposal(
            agent_name=self.name,
            action_type="tune",
            target="walk_forward_tuning_experiment",
            reason="; ".join(plan["reasons"]),
            command_preview=(
                "approved experiment only: walk-forward tuning "
                f"--tickers {tickers} --timeframes {timeframes} --locked-holdout --no-production-write"
                f"{variation_preview}"
            ),
            expected_effect="Produce candidate hyperparameter or model-selection changes for review, not automatic promotion.",
            risks=plan["risks"],
            evidence=[
                self.evidence("metric", "context.metadata.model_performance", "threshold_failures", plan["model_failures"]),
                self.evidence("metric", "context.metadata.model_performance", "performance_score", model_performance.get("performance_score")),
                self.evidence("metric", "context.metadata.regime_context", "regime_tags", plan["regime_tags"]),
                self.evidence("metric", "context.metadata.pipeline_control_surface", "proposal_gate", plan.get("control_surface_gate")),
                self.evidence("metric", "context.metadata.pipeline_control_surface", "allowed_variation", variation),
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
                self.evidence("metric", "context.metadata.pipeline_control_surface", "proposal_gate", plan.get("control_surface_gate")),
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


def _scope_mismatches(
    *,
    evaluation_scope: dict[str, Any],
    context: MarketContext,
    configured_tickers: Any,
    configured_timeframes: Any,
) -> list[str]:
    if not evaluation_scope:
        return []
    mismatches: list[str] = []
    ticker = str(evaluation_scope.get("ticker") or "")
    timeframe = str(evaluation_scope.get("timeframe") or "")
    context_tickers = {
        str(item).upper() for item in context.tickers if str(item)
    }
    context_timeframes = {
        str(item)
        for item in (
            context.timeframes
            or ([context.timeframe] if context.timeframe else [])
        )
        if str(item)
    }
    if context_tickers and ticker.upper() not in context_tickers:
        mismatches.append("evaluated_ticker_not_in_context")
    if context_timeframes and timeframe not in context_timeframes:
        mismatches.append("evaluated_timeframe_not_in_context")
    if configured_tickers:
        configured = {
            str(item).upper()
            for item in configured_tickers
            if str(item)
        }
        if configured != {ticker.upper()}:
            mismatches.append(
                "configured_tickers_broaden_evaluated_scope"
            )
    if configured_timeframes:
        configured = {
            str(item)
            for item in configured_timeframes
            if str(item)
        }
        if configured != {timeframe}:
            mismatches.append(
                "configured_timeframes_broaden_evaluated_scope"
            )
    return mismatches


def _variation_preview(variation: dict[str, Any]) -> str:
    if not variation:
        return ""
    pieces = []
    if variation.get("max_trials") is not None:
        pieces.append(f"--max-trials {variation['max_trials']}")
    if variation.get("parameter_delta_pct") is not None:
        pieces.append(f"--parameter-delta-pct {variation['parameter_delta_pct']}")
    if variation.get("max_feature_additions") is not None:
        pieces.append(f"--max-feature-additions {variation['max_feature_additions']}")
    if not pieces:
        return ""
    return " " + " ".join(pieces)
