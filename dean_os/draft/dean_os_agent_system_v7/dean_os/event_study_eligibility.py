from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EventStudyDesign(BaseModel):
    event_id: str
    event_timestamp: str
    asset_ids: list[str] = Field(min_length=1)
    benchmark_id: str
    timeframe: str
    event_window_start_bars: int = -1
    event_window_end_bars: int = 1
    estimation_window_start_bars: int = -120
    estimation_window_end_bars: int = -20
    post_event_drift_end_bars: int = 20
    expected_return_model: Literal[
        "constant_mean",
        "market_model",
        "factor_model",
    ] = "market_model"
    volatility_adjustment: Literal[
        "none",
        "pre_event_residual_volatility",
        "garch_review_only",
    ] = "pre_event_residual_volatility"
    event_time_alignment: Literal[
        "market_session_exact",
        "before_open",
        "after_close",
        "unknown",
    ] = "unknown"

    @model_validator(mode="after")
    def validate_windows(self) -> "EventStudyDesign":
        timestamp = datetime.fromisoformat(self.event_timestamp.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("event_timestamp must be timezone-aware")
        if not (
            self.event_window_start_bars <= 0 <= self.event_window_end_bars
        ):
            raise ValueError("event window must contain event bar zero")
        if self.estimation_window_start_bars >= self.estimation_window_end_bars:
            raise ValueError("estimation window start must precede its end")
        if self.estimation_window_end_bars >= self.event_window_start_bars:
            raise ValueError("estimation window must end before event window")
        if self.post_event_drift_end_bars < self.event_window_end_bars:
            raise ValueError("post-event drift horizon cannot precede event window end")
        return self


class EventStudyReadiness(BaseModel):
    release_timestamp_verified: bool = False
    market_data_hash: str = ""
    benchmark_data_hash: str = ""
    available_estimation_observations: int = 0
    event_window_complete: bool = False
    benchmark_window_complete: bool = False
    liquidity_evidence_available: bool = False
    volatility_evidence_available: bool = False
    unresolved_confounding_events: list[str] = Field(default_factory=list)
    anticipation_risk: Literal["low", "medium", "high", "unknown"] = "unknown"
    overlapping_event_windows: list[str] = Field(default_factory=list)
    source_cutoff: str = ""


class EventStudyEligibility(BaseModel):
    contract: str = "dean_event_study_eligibility_v1"
    event_id: str
    status: Literal[
        "blocked",
        "descriptive_only",
        "eligible_for_abnormal_return_estimation",
    ]
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    can_estimate_abnormal_returns: bool = False
    can_compute_car: bool = False
    causal_attribution_allowed: bool = False
    required_outputs: list[str] = Field(default_factory=list)
    interpretation_limits: list[str] = Field(default_factory=list)


class EventStudyEligibilityGate:
    def __init__(self, *, min_estimation_observations: int = 60) -> None:
        self.min_estimation_observations = min_estimation_observations

    def evaluate(
        self,
        design: EventStudyDesign,
        readiness: EventStudyReadiness,
    ) -> EventStudyEligibility:
        blockers: list[str] = []
        warnings: list[str] = []
        if not readiness.release_timestamp_verified:
            blockers.append("release_timestamp_not_verified")
        if design.event_time_alignment == "unknown":
            blockers.append("event_time_market_session_alignment_unknown")
        if not readiness.market_data_hash:
            blockers.append("market_data_hash_missing")
        if design.expected_return_model != "constant_mean" and not readiness.benchmark_data_hash:
            blockers.append("benchmark_data_hash_missing")
        if readiness.available_estimation_observations < self.min_estimation_observations:
            blockers.append("insufficient_estimation_window_observations")
        if not readiness.event_window_complete:
            blockers.append("event_window_incomplete")
        if not readiness.benchmark_window_complete:
            blockers.append("benchmark_window_incomplete")

        if readiness.unresolved_confounding_events:
            warnings.append("unresolved_confounding_events")
        if readiness.overlapping_event_windows:
            warnings.append("overlapping_event_windows")
        if readiness.anticipation_risk in {"high", "unknown"}:
            warnings.append(f"anticipation_risk_{readiness.anticipation_risk}")
        if not readiness.liquidity_evidence_available:
            warnings.append("liquidity_effect_not_measured")
        if not readiness.volatility_evidence_available:
            warnings.append("volatility_adjustment_evidence_missing")

        can_estimate = not blockers
        if blockers:
            status = "blocked"
        elif any(
            warning in warnings
            for warning in (
                "unresolved_confounding_events",
                "overlapping_event_windows",
                "anticipation_risk_high",
                "anticipation_risk_unknown",
            )
        ):
            status = "descriptive_only"
        else:
            status = "eligible_for_abnormal_return_estimation"

        return EventStudyEligibility(
            event_id=design.event_id,
            status=status,
            blockers=blockers,
            warnings=warnings,
            can_estimate_abnormal_returns=can_estimate,
            can_compute_car=can_estimate,
            causal_attribution_allowed=False,
            required_outputs=[
                "expected_return",
                "abnormal_return_by_bar",
                "cumulative_abnormal_return",
                "volatility_adjusted_statistic",
                "liquidity_effect",
                "post_event_drift",
                "confounding_event_review",
            ],
            interpretation_limits=[
                "Abnormal return is not proof that the event caused the move",
                "Confounders, anticipation and positioning must be reviewed",
                "Market reaction and fundamental change are separate outcomes",
            ],
        )


__all__ = [
    "EventStudyDesign",
    "EventStudyEligibility",
    "EventStudyEligibilityGate",
    "EventStudyReadiness",
]
