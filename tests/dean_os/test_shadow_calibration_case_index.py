from __future__ import annotations

import json

import pandas as pd

from dean_os.pipeline_prediction_review_packet import (
    PipelinePredictionReviewPacket,
)
from dean_os.shadow_calibration_case_index import (
    ShadowCalibrationCaseIndexBuilder,
    validate_shadow_calibration_case,
)
from src.pipeline.stages.prediction.output_contract import (
    build_model_output_contract,
)


def _pipeline_result():
    return {
        "prediction_results": {
            "ctx_nvda": {
                "ticker": "NVDA",
                "model_context_id": "ctx_nvda",
                "target_name": "target_intraday_up_15m",
                "model_type": "random_forest",
                "timeframe": "15m",
                "context_fingerprint": "fingerprint_nvda",
                "selected_primary_model": "random_forest",
                "model_output_contract": build_model_output_contract(
                    target_name="target_intraday_up_15m",
                    target_type="classification",
                    model_count=1,
                    contextual_adjustment_applied=True,
                    nlp_adjustment_applied=False,
                    target_scaler_applied=False,
                    classification_predict_semantics="class_label",
                ),
                "predictions": 0.8,
                "raw_forecast": 1,
                "predictions_by_model": {"random_forest": 1},
                "confidence": 0.73,
                "anomaly_score": 0.91,
                "last_price": 100.0,
                "timestamp": "2026-06-29T12:00:00+00:00",
            }
        }
    }


def _prediction_review(tmp_path):
    pipeline_result = _pipeline_result()
    source = tmp_path / "pipeline_result.json"
    source.write_text(json.dumps(pipeline_result), encoding="utf-8")
    output = tmp_path / "prediction_review"
    packet = PipelinePredictionReviewPacket(output_dir=output).build(
        pipeline_result,
        source_artifact_path=source,
        save=True,
    )
    assert packet["status"] == "stage5_prediction_review_ready"
    return source, output / "latest.json"


def test_case_index_binds_exact_outcome_without_using_later_rows(
    tmp_path,
):
    _, review = _prediction_review(tmp_path)
    outcome = tmp_path / "outcomes.csv"
    pd.DataFrame([
        {
            "ticker": "NVDA",
            "interval": "15m",
            "datetime": "2026-06-29T12:15:00Z",
            "close": 100.2,
        },
        {
            "ticker": "NVDA",
            "interval": "15m",
            "datetime": "2026-06-29T12:30:00Z",
            "close": 99.0,
        },
    ]).to_csv(outcome, index=False)

    payload = ShadowCalibrationCaseIndexBuilder(
        prediction_review_path=review,
        outcome_source_path=outcome,
        output_dir=tmp_path / "case_index",
    ).build(save=False)

    assert payload["status"] == "shadow_calibration_case_index_ready"
    assert payload["record_count"] == 1
    record = payload["records"][0]
    assert validate_shadow_calibration_case(record) == []
    assert record["realization"]["realized_return"] == (
        100.2 / 100.0 - 1.0
    )
    assert record["realization"]["realized_target"] == 1
    assert record["prediction"]["raw_value"] == 1.0
    assert record["prediction"]["raw_output_scale"] == (
        "class_label_from_predict"
    )
    assert record["prediction"][
        "positive_class_probability"
    ] is False
    outcome_source = record["source_provenance"]["outcome_source"]
    assert outcome_source["source_contains_later_rows"] is True
    assert outcome_source["later_rows_used"] is False
    assert record["safety"]["future_evidence_used"] is False
    assert payload["safety"]["automatic_weight_change_allowed"] is False
    assert payload["safety"]["can_trade"] is False


def test_case_index_rejects_non_exact_realization_timestamp(tmp_path):
    _, review = _prediction_review(tmp_path)
    outcome = tmp_path / "outcomes.csv"
    pd.DataFrame([{
        "ticker": "NVDA",
        "timeframe": "15m",
        "observed_at": "2026-06-29T12:16:00Z",
        "close": 100.2,
    }]).to_csv(outcome, index=False)

    payload = ShadowCalibrationCaseIndexBuilder(
        prediction_review_path=review,
        outcome_source_path=outcome,
    ).build(save=False)

    assert payload["status"] == "shadow_calibration_case_index_blocked"
    assert payload["record_count"] == 0
    assert payload["rejected_contexts"][0]["issues"] == [
        "outcome_row_missing"
    ]


def test_case_index_rejects_changed_pipeline_source(tmp_path):
    source, review = _prediction_review(tmp_path)
    source.write_text(
        json.dumps({"status": "mutated_after_review"}),
        encoding="utf-8",
    )
    outcome = tmp_path / "outcomes.csv"
    pd.DataFrame([{
        "ticker": "NVDA",
        "timeframe": "15m",
        "observed_at": "2026-06-29T12:15:00Z",
        "close": 100.2,
    }]).to_csv(outcome, index=False)

    payload = ShadowCalibrationCaseIndexBuilder(
        prediction_review_path=review,
        outcome_source_path=outcome,
    ).build(save=False)

    assert payload["record_count"] == 0
    assert "pipeline_result_source_hash_mismatch" in (
        payload["rejected_contexts"][0]["issues"]
    )
