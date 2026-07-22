from __future__ import annotations

import json

from dean_os.pipeline_prediction_review_packet import (
    PipelinePredictionReviewPacket,
)
from src.pipeline.stages.prediction.output_contract import (
    build_model_output_contract,
)


def _prediction(
    ticker: str,
    timeframe: str,
    context_id: str,
    *,
    target_name: str = "target_intraday_up_15m",
    model: str = "random_forest",
) -> dict:
    return {
        "ticker": ticker,
        "model_context_id": context_id,
        "target_name": target_name,
        "model_type": model,
        "timeframe": timeframe,
        "context_fingerprint": f"fingerprint_{context_id}",
        "selected_primary_model": model,
        "model_output_contract": build_model_output_contract(
            target_name=target_name,
            target_type="classification",
            model_count=2,
            contextual_adjustment_applied=True,
            nlp_adjustment_applied=False,
            target_scaler_applied=False,
            classification_predict_semantics="class_label",
        ),
        "predictions": [0.64],
        "raw_forecast": 0.59,
        "predictions_by_model": {
            model: 0.64,
            "baseline": 0.51,
        },
        "confidence": 0.74,
        "anomaly_score": 0.92,
        "last_price": 145.2,
        "timestamp": "2026-06-29T12:00:00+00:00",
    }


def _sector_ticker_review(
    *,
    context_fingerprint: str = "fingerprint_ctx_amd",
    can_create_ticker_forecast: bool = False,
) -> dict:
    return {
        "run_id": "sector_ticker_review_fixture",
        "mode": "sector_to_ticker_review_packet",
        "summary": {
            "packet_status": "review_ready_with_limitations",
            "sector": "semiconductor",
            "domain_profile": "semiconductor_ai_infrastructure",
            "sector_stance": "mixed",
            "can_create_ticker_forecast": (
                can_create_ticker_forecast
            ),
            "can_write_learning_memory": False,
            "can_trade": False,
        },
        "ticker_review_map": [
            {
                "ticker": "AMD",
                "review_status": (
                    "ticker_evidence_ready_pipeline_blocked"
                ),
                "allowed_use": (
                    "manual_review_of_ticker_evidence_not_forecast"
                ),
                "ticker_specific_evidence": {
                    "status": (
                        "company_mechanism_corroborated"
                    ),
                    "eligible_record_count": 3,
                    "corroborated_lane_count": 1,
                },
                "sector_context": {
                    "allowed_use": "supporting_context_only",
                    "can_influence_ticker_direction": False,
                    "verified_reasoning": {
                        "available": True,
                        "status": (
                            "reasoning_snapshot_ready_with_cautions"
                        ),
                        "runtime_hash_bound": True,
                        "classified_event_count": 152,
                        "transmission_channel_count": 62,
                        "directional_ticker_event_count": 0,
                        "scenario_graph_status": "not_generated",
                        "can_change_prediction": False,
                    },
                },
                "feature_timeframe_audit": {
                    "status": "timeframe_cadence_mismatch",
                    "declared_timeframe": "1d",
                    "observed_timeframe": "15m",
                    "datetime_timezone_aware": False,
                    "can_assert_feature_parentage": False,
                    "can_override_timeframe": False,
                },
                "exact_pipeline_contexts": [
                    {
                        "ticker": "AMD",
                        "model": "random_forest",
                        "target_name": (
                            "target_intraday_up_15m"
                        ),
                        "timeframe": "15m",
                        "context_fingerprint": (
                            context_fingerprint
                        ),
                        "case_classification": (
                            "negative_evaluation_block_case"
                        ),
                        "blocked_metric_planes": [
                            "validation",
                            "feature_stability",
                        ],
                    }
                ],
                "required_next_inputs": [
                    (
                        "realized_outcome_calibration_for_target_"
                        "and_horizon"
                    )
                ],
            }
        ],
    }


def test_prediction_review_preserves_per_context_lineage():
    result = {
        "tickers": ["NVDA", "MSFT"],
        "timeframes": ["15m"],
        "results": {
            "prediction_results": {
                "ticker=NVDA|interval=15m": _prediction(
                    "NVDA",
                    "15m",
                    "ctx_nvda",
                ),
                "ticker=MSFT|interval=15m": _prediction(
                    "MSFT",
                    "15m",
                    "ctx_msft",
                ),
            }
        },
    }

    payload = PipelinePredictionReviewPacket().build(result)

    assert payload["status"] == "stage5_prediction_review_ready"
    assert payload["context_count"] == 2
    assert payload["complete_context_count"] == 2
    assert [
        item["ticker"] for item in payload["contexts"]
    ] == ["MSFT", "NVDA"]
    nvda = payload["contexts"][1]
    assert nvda["model_context_id"] == "ctx_nvda"
    assert nvda["target_name"] == "target_intraday_up_15m"
    assert nvda["context_fingerprint"] == "fingerprint_ctx_nvda"
    assert nvda["prediction"]["value"] == 0.64
    assert nvda["prediction"]["shape"] == {
        "kind": "single_item_sequence",
        "count": 1,
    }
    assert nvda["model_contribution_count"] == 2
    assert nvda["target_semantics"]["status"] == (
        "target_semantics_ready"
    )
    assert nvda["target_semantics"]["horizon_seconds"] == 900
    assert nvda["target_semantics"]["class_semantics"][
        "positive_class"
    ] == 1
    assert nvda["target_semantics"]["calibration"][
        "directional_inference_allowed"
    ] is False
    assert nvda["target_semantics"]["calibration"][
        "model_output_scale_known"
    ] is True
    assert nvda["target_semantics"]["stage5_scalar_semantics"] == (
        "adjusted_classification_score"
    )
    assert nvda["decision_influence"] is False
    assert payload["safety"]["is_realized_outcome"] is False
    assert payload["safety"]["can_promote_model"] is False
    assert payload["safety"]["can_trade"] is False


def test_prediction_review_rejects_incomplete_and_wrong_context():
    prediction = _prediction("AMD", "1d", "ctx_wrong")
    prediction.pop("context_fingerprint")
    prediction["confidence"] = 1.4
    prediction["anomaly_score"] = "not-a-number"
    prediction["predictions"] = [0.1, 0.2, 0.3]
    result = {
        "results": {
            "prediction_results": {"ctx_wrong": prediction}
        }
    }

    payload = PipelinePredictionReviewPacket().build(
        result,
        requested_tickers=["NVDA"],
        requested_timeframes=["15m"],
    )
    item = payload["contexts"][0]

    assert payload["status"] == "stage5_prediction_review_partial"
    assert item["lineage_status"] == "incomplete"
    assert item["missing_lineage_fields"] == [
        "context_fingerprint"
    ]
    assert item["prediction"]["value"] is None
    assert item["prediction"]["shape"]["count"] == 3
    assert set(item["review_issues"]) == {
        "ticker_outside_requested_context",
        "timeframe_outside_requested_context",
        "invalid_confidence",
            "invalid_anomaly_score",
            "prediction_not_single_scalar",
            "target_semantics_incomplete",
        }


def test_prediction_review_quarantines_timeframe_cadence_mismatch():
    prediction = _prediction("AMD", "15m", "ctx_amd")
    prediction["timeframe_lineage"] = {
        "status": "timeframe_cadence_mismatch",
        "declared_timeframe": "1d",
        "observed_timeframe": "15m",
        "resolved_timeframe": None,
        "safe_for_prediction_lineage": False,
    }

    payload = PipelinePredictionReviewPacket().build(
        {
            "results": {
                "prediction_results": {
                    "ctx_amd": prediction,
                }
            }
        }
    )

    assert payload["status"] == "stage5_prediction_review_partial"
    assert payload["complete_context_count"] == 0
    assert payload["contexts"][0]["timeframe_lineage"][
        "status"
    ] == "timeframe_cadence_mismatch"
    assert "timeframe_cadence_mismatch" in payload["contexts"][0][
        "review_issues"
    ]


def test_prediction_review_marks_duplicate_lineage_without_collapsing():
    shared = _prediction("NVDA", "15m", "ctx_nvda")
    result = {
        "prediction_results": {
            "first": shared,
            "second": dict(shared),
        }
    }

    payload = PipelinePredictionReviewPacket().build(result)

    assert payload["status"] == "stage5_prediction_review_partial"
    assert payload["context_count"] == 2
    assert all(
        "duplicate_prediction_lineage" in item["review_issues"]
        for item in payload["contexts"]
    )


def test_prediction_review_prefers_raw_stage5_results_over_summary_copy():
    result = {
        "results": {
            "prediction_results": {
                "raw": _prediction("NVDA", "15m", "raw_ctx")
            }
        },
        "summary": {
            "prediction_results": {
                "summary": _prediction("MSFT", "1d", "summary_ctx")
            }
        },
    }

    payload = PipelinePredictionReviewPacket().build(result)

    assert payload["source_path"] == "results.prediction_results"
    assert payload["contexts"][0]["ticker"] == "NVDA"


def test_prediction_review_reports_absent_stage5_output():
    payload = PipelinePredictionReviewPacket().build(
        {"status": "completed"}
    )

    assert payload["status"] == "stage5_predictions_not_available"
    assert payload["context_count"] == 0
    assert payload["safety"]["decision_influence"] is False


def test_prediction_review_can_filter_to_requested_ticker_scope():
    result = {
        "prediction_results": {
            "ctx_amd": _prediction("AMD", "15m", "ctx_amd"),
            "ctx_nvda": _prediction("NVDA", "15m", "ctx_nvda"),
        }
    }

    payload = PipelinePredictionReviewPacket().build(
        result,
        requested_tickers=["AMD"],
        filter_to_requested_scope=True,
    )

    assert payload["source_context_count"] == 2
    assert payload["excluded_by_scope_count"] == 1
    assert payload["context_count"] == 1
    assert payload["contexts"][0]["ticker"] == "AMD"
    assert "ticker_outside_requested_context" not in payload[
        "contexts"
    ][0]["review_issues"]


def test_prediction_review_quarantines_missing_time_and_pattern_fingerprint():
    prediction = _prediction("AMD", "15m", "ctx_amd")
    prediction["timestamp"] = None
    prediction["context_fingerprint"] = "normal"

    payload = PipelinePredictionReviewPacket().build(
        {"prediction_results": {"ctx_amd": prediction}}
    )

    issues = payload["contexts"][0]["review_issues"]
    assert "prediction_as_of_missing" in issues
    assert "context_fingerprint_placeholder_or_pattern" in issues
    assert payload["review_issue_counts"][
        "prediction_as_of_missing"
    ] == 1


def test_prediction_review_blocks_missing_model_output_contract():
    prediction = _prediction("NVDA", "15m", "ctx_nvda")
    prediction.pop("model_output_contract")

    payload = PipelinePredictionReviewPacket().build(
        {"prediction_results": {"ctx_nvda": prediction}}
    )

    assert payload["status"] == "stage5_prediction_review_partial"
    assert "model_output_contract_incomplete" in (
        payload["contexts"][0]["review_issues"]
    )


def test_prediction_review_binds_saved_pipeline_source(tmp_path):
    source = tmp_path / "pipeline_result.json"
    result = {
        "prediction_results": {
            "ctx_nvda": _prediction("NVDA", "15m", "ctx_nvda")
        }
    }
    source.write_text(json.dumps(result), encoding="utf-8")

    payload = PipelinePredictionReviewPacket().build(
        result,
        source_artifact_path=source,
    )

    assert payload["source_artifact"]["path"] == str(source)
    assert payload["source_artifact"]["available"] is True
    assert len(payload["source_artifact"]["sha256"]) == 64
    assert payload["source_artifact"][
        "immutable_binding_ready"
    ] is True


def test_prediction_review_attaches_sector_ticker_context_without_influence(
    tmp_path,
):
    review_path = tmp_path / "sector_ticker_review.json"
    review_path.write_text(
        json.dumps(_sector_ticker_review()),
        encoding="utf-8",
    )
    result = {
        "prediction_results": {
            "ctx_amd": _prediction("AMD", "15m", "ctx_amd")
        }
    }

    payload = PipelinePredictionReviewPacket().build(
        result,
        sector_to_ticker_review_path=review_path,
    )

    context = payload["contexts"][0][
        "supporting_sector_ticker_context"
    ]
    assert context["status"] == "supporting_context_attached"
    assert context["sector_stance"] == "mixed"
    assert context["sector_reasoning_context"][
        "runtime_hash_bound"
    ] is True
    assert context["sector_reasoning_context"][
        "transmission_channel_count"
    ] == 62
    assert "verified_sector_reasoning_supporting_only" in (
        context["context_flags"]
    )
    assert context[
        "ticker_evidence_eligible_record_count"
    ] == 3
    assert context["aligned_pipeline_case_count"] == 1
    assert "negative_pipeline_evaluation_case_aligned" in (
        context["context_flags"]
    )
    assert "candidate_feature_timeframe_cadence_mismatch" in (
        context["context_flags"]
    )
    assert "legacy_stage5_feature_parentage_unverified" in (
        context["context_flags"]
    )
    assert context["decision_influence"] is False
    assert context["can_change_prediction"] is False
    assert context["can_fill_missing_lineage"] is False
    assert context["can_clear_model_evaluation"] is False
    assert payload["sector_context_overlay_summary"][
        "ticker_evidence_context_count"
    ] == 1
    assert payload["safety"][
        "sector_context_decision_influence"
    ] is False
    assert payload["status"] == "stage5_prediction_review_ready"


def test_prediction_review_flags_nonmatching_pipeline_case_as_context_only(
    tmp_path,
):
    review_path = tmp_path / "sector_ticker_review.json"
    review_path.write_text(
        json.dumps(
            _sector_ticker_review(
                context_fingerprint="different_context"
            )
        ),
        encoding="utf-8",
    )

    payload = PipelinePredictionReviewPacket().build(
        {
            "prediction_results": {
                "ctx_amd": _prediction(
                    "AMD", "15m", "ctx_amd"
                )
            }
        },
        sector_to_ticker_review_path=review_path,
    )
    context = payload["contexts"][0][
        "supporting_sector_ticker_context"
    ]

    assert context["aligned_pipeline_case_count"] == 0
    assert (
        "attached_pipeline_cases_do_not_match_prediction_identity"
        in context["context_flags"]
    )
    assert context["decision_influence"] is False


def test_prediction_review_rejects_context_that_can_create_forecast(
    tmp_path,
):
    review_path = tmp_path / "unsafe_sector_ticker_review.json"
    review_path.write_text(
        json.dumps(
            _sector_ticker_review(
                can_create_ticker_forecast=True
            )
        ),
        encoding="utf-8",
    )

    try:
        PipelinePredictionReviewPacket().build(
            {
                "prediction_results": {
                    "ctx_amd": _prediction(
                        "AMD", "15m", "ctx_amd"
                    )
                }
            },
            sector_to_ticker_review_path=review_path,
        )
    except ValueError as exc:
        assert "safety boundary invalid" in str(exc)
    else:
        raise AssertionError("Unsafe sector context was accepted")
