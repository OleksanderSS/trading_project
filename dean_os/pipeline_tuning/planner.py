from __future__ import annotations

from typing import Any

from .planes import get_default_tuning_planes
from .schemas import PipelineTuningPlan, TuningPlaneDecision, TuningPlaneProfile


class PipelineTuningPlanner:
    """Builds a bounded, review-only pipeline tuning plan.

    It does not run training, Optuna, model promotion, or production config writes.
    """

    def __init__(self, planes: list[TuningPlaneProfile] | None = None):
        self.planes = planes or get_default_tuning_planes()

    def build_plan(self, context_metadata: dict[str, Any]) -> PipelineTuningPlan:
        model_performance = _as_dict(context_metadata.get("model_performance"))
        control_surface = _as_dict(context_metadata.get("pipeline_control_surface"))
        data_freshness = _as_dict(context_metadata.get("data_freshness"))
        risk_report = _as_dict(context_metadata.get("risk_report") or context_metadata.get("risk"))

        failures = list(model_performance.get("threshold_failures") or [])
        stale_sources = _stale_sources(data_freshness)
        control_gate = _as_dict(control_surface.get("proposal_gate"))
        control_blocked = control_surface and not control_gate.get("can_propose_tuning", False)

        reasons: list[str] = []
        risks = [
            "Tuning plan is review-only.",
            "No production config write is allowed.",
            "Locked holdout and walk-forward validation must be preserved.",
        ]

        if not model_performance:
            reasons.append("No model performance metrics are available.")
            return self._plan("validate_first", [], reasons, risks, "review/collect model performance metrics first")

        if stale_sources:
            reasons.append(f"Stale inputs must be validated before tuning: {', '.join(stale_sources)}.")
            return self._plan("validate_first", [], reasons, risks, "review/refresh stale inputs before tuning")

        if control_blocked:
            reasons.append(f"Pipeline control surface blocks tuning: {control_gate.get('reason', 'no reason supplied')}.")
            return self._plan("blocked", [], reasons, risks, "control surface blocks tuning proposal")

        decisions = [self._decide_plane(plane, failures, model_performance, risk_report, control_surface) for plane in self.planes]
        candidate_count = sum(1 for decision in decisions if decision.status == "candidate")

        if candidate_count:
            reasons.append(f"{candidate_count} tuning plane(s) are candidates for bounded review.")
            status = "tuning_candidate"
            command = "approved experiment only: walk-forward tuning --locked-holdout --no-production-write"
        else:
            reasons.append("No tuning plane is currently needed under supplied metrics.")
            status = "no_action"
            command = "no tuning command proposed"

        return self._plan(status, decisions, reasons, risks, command)

    def _decide_plane(
        self,
        plane: TuningPlaneProfile,
        failures: list[str],
        model_performance: dict[str, Any],
        risk_report: dict[str, Any],
        control_surface: dict[str, Any],
    ) -> TuningPlaneDecision:
        guardrails = [
            "human_review_required",
            "locked_holdout",
            "walk_forward_validation",
            "no_production_config_write",
        ]

        if any(blocker in failures for blocker in plane.blocked_if):
            return TuningPlaneDecision(
                plane_id=plane.plane_id,
                status="blocked",
                reasons=[f"Blocked by failure(s): {', '.join(sorted(set(failures).intersection(plane.blocked_if)))}"],
                guardrails=guardrails,
                risks=["Do not tune this plane until blockers are resolved."],
            )

        if plane.plane_id == "hyperparameters" and any(item in failures for item in {"validation_score_below_threshold", "sharpe_below_threshold"}):
            return self._candidate(plane, "Performance failures are actionable by bounded hyperparameter review.", guardrails)

        if plane.plane_id == "model_selection" and any(item in failures for item in {"validation_score_below_threshold", "sharpe_below_threshold"}):
            return self._candidate(plane, "Model comparison may be needed under walk-forward validation.", guardrails)

        if plane.plane_id == "ensemble_weights" and any(item in failures for item in {"sharpe_below_threshold", "drawdown_above_threshold"}):
            return self._candidate(plane, "Risk-adjusted performance failure may require bounded ensemble-weight review.", guardrails)

        if plane.plane_id == "risk_thresholds" and "drawdown_above_threshold" in failures:
            return TuningPlaneDecision(
                plane_id=plane.plane_id,
                status="needs_validation",
                reasons=["Drawdown failure requires risk review; do not relax limits automatically."],
                proposed_bounds={"max_change_pct": 0.0, "direction": "tighten_or_validate_only"},
                guardrails=guardrails,
                risks=["Relaxing risk thresholds after drawdown failure is forbidden without review."],
            )

        if plane.plane_id == "feature_space" and "validation_score_below_threshold" in failures:
            return self._candidate(plane, "Validation weakness may require feature-space review after leakage checks.", guardrails)

        return TuningPlaneDecision(
            plane_id=plane.plane_id,
            status="not_needed",
            reasons=["No supplied metric failure requires this plane."],
            guardrails=guardrails,
            risks=[],
        )

    def _candidate(self, plane: TuningPlaneProfile, reason: str, guardrails: list[str]) -> TuningPlaneDecision:
        return TuningPlaneDecision(
            plane_id=plane.plane_id,
            status="candidate",
            reasons=[reason],
            proposed_bounds={
                "allowed_parameters": plane.allowed_parameters,
                "max_change_pct": plane.max_change_pct,
            },
            guardrails=guardrails,
            risks=[
                "Candidate plane only; implementation must be a separate approved experiment.",
                "No production artifacts can be promoted from this plan automatically.",
            ],
        )

    def _plan(
        self,
        status: str,
        decisions: list[TuningPlaneDecision],
        reasons: list[str],
        risks: list[str],
        command: str,
    ) -> PipelineTuningPlan:
        return PipelineTuningPlan(
            status=status,  # type: ignore[arg-type]
            planes=decisions,
            reasons=reasons,
            guardrails=[
                "human_review_required",
                "locked_holdout",
                "walk_forward_validation",
                "no_live_execution",
                "no_production_config_write",
            ],
            risks=risks,
            command_preview=command,
            review_required=True,
            live_execution_allowed=False,
            production_config_write_allowed=False,
        )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _stale_sources(data_freshness: dict[str, Any]) -> list[str]:
    return sorted(name for name, info in data_freshness.items() if isinstance(info, dict) and info.get("stale"))
