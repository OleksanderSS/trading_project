from __future__ import annotations

from typing import Any

CONTRACT_SCHEMA_VERSION = "dean_stage5_model_output_contract_v1"


def build_model_output_contract(
    *,
    target_name: str | None,
    target_type: str | None,
    model_count: int,
    contextual_adjustment_applied: bool,
    nlp_adjustment_applied: bool,
    target_scaler_applied: bool,
    classification_predict_semantics: str | None = None,
) -> dict[str, Any]:
    """Describe Stage 5 scalar meaning without treating scores as probabilities."""
    normalized_type = str(target_type or "").strip().lower()
    issues: list[str] = []
    if normalized_type not in {"classification", "regression"}:
        issues.append("target_type_missing_or_unsupported")

    is_ensemble = int(model_count) > 1
    if normalized_type == "classification":
        if classification_predict_semantics == "class_label":
            raw_scale = (
                "ensemble_decision_signal_from_class_labels"
                if is_ensemble
                else "class_label_from_predict"
            )
            final_scale = (
                "adjusted_classification_score"
                if contextual_adjustment_applied
                or nlp_adjustment_applied
                else raw_scale
            )
        else:
            raw_scale = "classification_predict_output_unknown_scale"
            final_scale = "unknown_adjusted_classification_output"
            issues.append("classification_predict_scale_unverified")
        if target_scaler_applied:
            issues.append("classification_target_scaler_applied")
    elif normalized_type == "regression":
        raw_scale = (
            "ensemble_regression_signal_from_predict_outputs"
            if is_ensemble
            else "regression_value_from_predict"
        )
        final_scale = "regression_target_value"
    else:
        raw_scale = "unknown_predict_output"
        final_scale = "unknown_adjusted_model_output"

    scale_known = not issues
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "status": (
            "model_output_contract_ready"
            if scale_known
            else "model_output_contract_partial"
        ),
        "target_name": target_name,
        "target_type": normalized_type or "unknown",
        "prediction_method": "predict",
        "model_count": int(model_count),
        "is_ensemble": is_ensemble,
        "raw_output": {
            "scale": raw_scale,
            "positive_class_probability": False,
        },
        "adjustments": {
            "contextual_adjustment_applied": bool(
                contextual_adjustment_applied
            ),
            "nlp_pattern_adjustment_applied": bool(
                nlp_adjustment_applied
            ),
            "target_inverse_scaler_applied": bool(
                target_scaler_applied
            ),
        },
        "final_output": {
            "scale": final_scale,
            "model_output_scale_known": scale_known,
            "positive_class_probability": False,
            "directional_inference_allowed": False,
        },
        "issues": issues,
        "calibration": {
            "realized_outcome_required": True,
            "calibration_executed": False,
            "calibration_eligible_now": False,
        },
        "decision_influence": False,
        "can_trade": False,
    }


def infer_classification_predict_semantics(
    models: dict[str, Any],
) -> str | None:
    """Confirm label-returning classifiers from their runtime class contract."""
    predictive_models = [
        model
        for name, model in models.items()
        if "autoencoder" not in str(name).lower()
    ]
    if predictive_models and all(
        getattr(model, "classes_", None) is not None
        for model in predictive_models
    ):
        return "class_label"
    return None
