from __future__ import annotations

from dean_os.prediction_target_semantics import (
    PredictionTargetSemanticsRegistry,
)
from src.pipeline.stages.prediction.output_contract import (
    build_model_output_contract,
)


def test_intraday_binary_target_semantics_are_explicit():
    semantics = PredictionTargetSemanticsRegistry().resolve(
        target_name="target_intraday_up_15m",
        timeframe="15m",
        prediction_as_of="2026-06-29T12:00:00+00:00",
    )

    assert semantics["status"] == "target_semantics_ready"
    assert semantics["target_type"] == "classification_binary"
    assert semantics["target_unit"] == "class"
    assert semantics["shift_bars"] == 1
    assert semantics["horizon_seconds"] == 900
    assert semantics["threshold"] == {
        "value": 0.001,
        "unit": "return_ratio",
        "percent": 0.1,
    }
    assert semantics["class_semantics"]["positive_class"] == 1
    assert semantics["class_semantics"]["positive_condition"] == (
        "future_close_return > 0.001"
    )
    assert semantics["realization_window"]["expected_end"] == (
        "2026-06-29T12:15:00+00:00"
    )
    assert semantics["numeric_metadata_complete"] is True
    assert semantics["stage5_scalar_semantics"] == (
        "model_output_scale_not_declared"
    )
    assert semantics["calibration"][
        "directional_inference_allowed"
    ] is False
    assert semantics["calibration"][
        "calibration_eligible_now"
    ] is False


def test_hourly_target_resolves_against_actual_bar_timeframe():
    registry = PredictionTargetSemanticsRegistry()

    on_15m = registry.resolve(
        target_name="target_hourly_up_1h",
        timeframe="15m",
        prediction_as_of="2026-06-29T12:00:00Z",
    )
    on_60m = registry.resolve(
        target_name="target_hourly_up_1h",
        timeframe="60m",
        prediction_as_of="2026-06-29T12:00:00Z",
    )

    assert on_15m["shift_bars"] == 4
    assert on_60m["shift_bars"] == 1
    assert on_15m["horizon_seconds"] == on_60m["horizon_seconds"]
    assert on_15m["timeframe_compatible"] is True
    assert on_60m["timeframe_compatible"] is True


def test_target_semantics_validates_stage5_output_contract():
    contract = build_model_output_contract(
        target_name="target_intraday_up_15m",
        target_type="classification",
        model_count=1,
        contextual_adjustment_applied=True,
        nlp_adjustment_applied=False,
        target_scaler_applied=False,
        classification_predict_semantics="class_label",
    )

    semantics = PredictionTargetSemanticsRegistry().resolve(
        target_name="target_intraday_up_15m",
        timeframe="15m",
        prediction_as_of="2026-06-29T12:00:00Z",
        model_output_contract=contract,
    )

    assert semantics["stage5_scalar_semantics"] == (
        "adjusted_classification_score"
    )
    assert semantics["model_output_contract_validation"]["status"] == (
        "model_output_contract_validated"
    )
    assert semantics["calibration"]["model_output_scale_known"] is True
    assert semantics["calibration"]["directional_inference_allowed"] is False


def test_target_timeframe_mismatch_is_fail_closed():
    semantics = PredictionTargetSemanticsRegistry().resolve(
        target_name="target_intraday_up_15m",
        timeframe="1d",
        prediction_as_of="2026-06-29T00:00:00Z",
    )

    assert semantics["status"] == "target_semantics_partial"
    assert semantics["timeframe_compatible"] is False
    assert "target_timeframe_mismatch" in semantics["issues"]
    assert "horizon_not_exact_bar_multiple" in semantics["issues"]
    assert semantics["calibration"]["calibration_eligible_now"] is False


def test_unknown_target_cannot_be_interpreted():
    semantics = PredictionTargetSemanticsRegistry().resolve(
        target_name="target_not_registered",
        timeframe="15m",
        prediction_as_of="2026-06-29T12:00:00Z",
    )

    assert semantics["status"] == "target_semantics_unavailable"
    assert semantics["numeric_metadata_complete"] is False
    assert semantics["issues"] == [
        "target_not_found_in_canonical_config"
    ]
    assert semantics["decision_influence"] is False
    assert semantics["can_trade"] is False
