from __future__ import annotations

from .schemas import TuningPlaneProfile

DEFAULT_TUNING_PLANES = [
    TuningPlaneProfile(
        plane_id="model_selection",
        display_name="Model Selection",
        description="Bounded comparison of already-supported model families under walk-forward validation.",
        allowed_parameters=["model_family", "model_alias", "ensemble_member_enablement"],
        required_preconditions=["valid_metrics", "locked_holdout", "walk_forward_validation"],
        blocked_if=["missing_evaluation_metrics", "sample_count_below_threshold"],
        max_change_pct=0.0,
    ),
    TuningPlaneProfile(
        plane_id="feature_space",
        display_name="Feature Space",
        description="Bounded feature inclusion/exclusion within approved feature families.",
        allowed_parameters=["feature_family", "max_feature_count", "feature_drop_list"],
        required_preconditions=["feature_schema_manifest", "leakage_check_passed", "locked_holdout"],
        blocked_if=["target_leakage_detected", "missing_feature_manifest"],
        max_change_pct=0.15,
    ),
    TuningPlaneProfile(
        plane_id="hyperparameters",
        display_name="Hyperparameters",
        description="Small bounded hyperparameter search for approved models.",
        allowed_parameters=["learning_rate", "max_depth", "regularization", "n_estimators"],
        required_preconditions=["valid_metrics", "walk_forward_validation", "control_surface_clear"],
        blocked_if=["missing_evaluation_metrics", "control_surface_blocked"],
        max_change_pct=0.20,
    ),
    TuningPlaneProfile(
        plane_id="ensemble_weights",
        display_name="Ensemble Weights",
        description="Bounded adjustment of ensemble weights without adding unsupported models.",
        allowed_parameters=["member_weight", "regime_weight", "fallback_weight"],
        required_preconditions=["ensemble_manifest", "per_model_metrics", "regime_context"],
        blocked_if=["missing_model_metrics", "missing_regime_context"],
        max_change_pct=0.10,
    ),
    TuningPlaneProfile(
        plane_id="risk_thresholds",
        display_name="Risk Thresholds",
        description="Review-only proposal for risk gate threshold review, not automatic relaxation.",
        allowed_parameters=["max_drawdown", "var_limit", "exposure_limit"],
        required_preconditions=["risk_report", "drawdown_report", "human_review"],
        blocked_if=["risk_gate_blocked", "missing_risk_report"],
        max_change_pct=0.05,
    ),
]


def get_default_tuning_planes() -> list[TuningPlaneProfile]:
    return list(DEFAULT_TUNING_PLANES)
