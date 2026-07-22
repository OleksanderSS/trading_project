from __future__ import annotations

import json

from dean_os.shadow_component_case_producer import (
    ShadowComponentCaseProducer,
    validate_component_case,
)


def _prediction_case():
    return {
        "schema_version": "dean_shadow_calibration_case_v1",
        "case_id": "shadow_case:prediction",
        "case_validation_status": "accepted",
        "component": "prediction",
        "identity": {
            "ticker": "NVDA",
            "timeframe": "15m",
            "target_name": "target_intraday_up_15m",
            "model_context_id": "ctx_nvda",
            "context_fingerprint": "fingerprint_nvda",
        },
        "market_regime": "unknown",
        "prediction": {
            "as_of": "2026-06-29T12:00:00+00:00",
            "value": 0.8,
            "output_scale": "adjusted_classification_score",
        },
        "realization": {
            "expected_end": "2026-06-29T12:15:00+00:00",
            "observed_at": "2026-06-29T12:15:00+00:00",
            "realized_return": 0.002,
            "realized_target": 1,
        },
        "source_provenance": {
            "prediction_review": {
                "path": "prediction.json",
                "sha256": "a" * 64,
            },
            "pipeline_result": {
                "path": "pipeline.json",
                "sha256": "b" * 64,
            },
            "outcome_source": {
                "path": "outcomes.csv",
                "sha256": "c" * 64,
            },
        },
        "safety": {
            "exact_context_match": True,
            "exact_realization_timestamp_match": True,
            "time_leakage_detected": False,
            "future_evidence_used": False,
            "sector_to_ticker_leakage_detected": False,
            "unsafe_output_detected": False,
            "decision_influence": False,
            "can_trade": False,
        },
    }


def _base_index(tmp_path):
    path = tmp_path / "base_index.json"
    path.write_text(json.dumps({
        "mode": "shadow_calibration_case_index",
        "schema_version": "dean_shadow_calibration_case_index_v1",
        "status": "shadow_calibration_case_index_ready",
        "records": [_prediction_case()],
    }), encoding="utf-8")
    return path


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _regime_review(as_of="2026-06-29T12:00:00+00:00"):
    return {
        "schema_version": "dean_stage7_regime_review_v1",
        "status": "stage7_regime_contexts_recorded",
        "context_partitioned": True,
        "contexts": [{
            "context_key": "ticker=NVDA|interval=15m",
            "ticker": "NVDA",
            "timeframe": "15m",
            "identity_status": "exact_context_key",
            "regime": "TRENDING_UP",
            "confidence": 0.8,
            "metrics": {"trend_strength": 0.4},
            "as_of": as_of,
            "decision_influence": False,
        }],
    }


def test_regime_case_producer_adds_exact_pre_prediction_assessment(
    tmp_path,
):
    artifact = _write(
        tmp_path,
        "regime.json",
        _regime_review(),
    )
    payload = ShadowComponentCaseProducer(
        base_case_index_path=_base_index(tmp_path),
        component="regime",
        component_artifact_path=artifact,
    ).build(save=False)

    assert payload["status"] == "shadow_component_cases_added"
    assert payload["new_record_count"] == 1
    regime_case = next(
        item for item in payload["records"]
        if item["component"] == "regime"
    )
    assert validate_component_case(regime_case) == []
    assert regime_case["assessment"]["regime"] == "TRENDING_UP"
    assert regime_case["market_regime"] == "TRENDING_UP"
    assert regime_case["safety"]["future_evidence_used"] is False


def test_regime_case_producer_rejects_post_prediction_evidence(
    tmp_path,
):
    artifact = _write(
        tmp_path,
        "regime_future.json",
        _regime_review("2026-06-29T12:01:00+00:00"),
    )
    payload = ShadowComponentCaseProducer(
        base_case_index_path=_base_index(tmp_path),
        component="regime",
        component_artifact_path=artifact,
    ).build(save=False)

    assert payload["new_record_count"] == 0
    assert payload["rejected_cases"][0]["issues"] == [
        "regime_uses_post_prediction_evidence"
    ]


def test_specialist_case_requires_exact_approved_context(tmp_path):
    specialist = {
        "schema_version": "dean_specialist_context_review_v1",
        "status": "specialist_context_exact_match_ready",
        "requested_context": {
            "ticker": "NVDA",
            "timeframe": "15m",
            "as_of": "2026-06-29T12:00:00+00:00",
        },
        "domain_scope": {
            "domain_id": "semiconductor_ai_infrastructure",
        },
        "ticker_scope": {
            "evidence_scope": "direct_ticker_review_candidate",
        },
        "point_in_time": {"status": "point_in_time_compatible"},
        "timeframe_alignment": {"status": "aligned"},
        "safety": {
            "eligible_for_exact_pipeline_context": True,
            "manual_review_required": False,
            "decision_influence": False,
        },
    }
    artifact = _write(tmp_path, "specialist.json", specialist)
    payload = ShadowComponentCaseProducer(
        base_case_index_path=_base_index(tmp_path),
        component="specialist",
        component_artifact_path=artifact,
    ).build(save=False)

    assert payload["new_record_count"] == 1
    case = next(
        item for item in payload["records"]
        if item["component"] == "specialist"
    )
    assert validate_component_case(case) == []
    assert case["assessment"][
        "eligible_for_exact_pipeline_context"
    ] is True


def test_specialist_case_rejects_manual_pending_context(tmp_path):
    specialist = {
        "schema_version": "dean_specialist_context_review_v1",
        "requested_context": {
            "ticker": "NVDA",
            "timeframe": "15m",
            "as_of": "2026-06-29T12:00:00+00:00",
        },
        "ticker_scope": {
            "evidence_scope": "direct_ticker_review_candidate",
        },
        "point_in_time": {"status": "point_in_time_compatible"},
        "timeframe_alignment": {"status": "aligned"},
        "safety": {
            "eligible_for_exact_pipeline_context": False,
            "manual_review_required": True,
            "decision_influence": False,
        },
    }
    artifact = _write(tmp_path, "specialist.json", specialist)
    payload = ShadowComponentCaseProducer(
        base_case_index_path=_base_index(tmp_path),
        component="specialist",
        component_artifact_path=artifact,
    ).build(save=False)

    assert payload["new_record_count"] == 0
    assert set(payload["rejected_cases"][0]["issues"]) == {
        "specialist_manual_review_not_complete",
        "specialist_not_exact_pipeline_eligible",
    }


def test_synthesis_case_uses_exact_prediction_lineage(tmp_path):
    synthesis = {
        "schema_version": "dean_pipeline_context_synthesis_v1",
        "status": "context_synthesis_ready",
        "ticker": "NVDA",
        "timeframe": "15m",
        "regime": {
            "regime": "TRENDING_UP",
            "as_of": "2026-06-29T11:59:00+00:00",
        },
        "prediction_assessments": [{
            "model_context_id": "ctx_nvda",
            "target_name": "target_intraday_up_15m",
            "context_fingerprint": "fingerprint_nvda",
            "prediction_as_of": "2026-06-29T12:00:00+00:00",
            "regime_as_of": "2026-06-29T11:59:00+00:00",
            "freshness_status": "compatible",
            "as_of_skew_minutes": 1.0,
        }],
        "conflicts": [],
        "review_confidence": 0.8,
        "directional_synthesis_performed": False,
        "decision_influence": False,
    }
    artifact = _write(tmp_path, "synthesis.json", synthesis)
    payload = ShadowComponentCaseProducer(
        base_case_index_path=_base_index(tmp_path),
        component="context_synthesis",
        component_artifact_path=artifact,
    ).build(save=False)

    assert payload["new_record_count"] == 1
    case = next(
        item for item in payload["records"]
        if item["component"] == "context_synthesis"
    )
    assert validate_component_case(case) == []
    assert case["assessment"]["freshness_status"] == "compatible"
    assert case["assessment"][
        "directional_synthesis_performed"
    ] is False


def test_component_producers_chain_without_dropping_prior_cases(
    tmp_path,
):
    current_index = _base_index(tmp_path)
    artifacts = {
        "regime": _regime_review(),
        "specialist": {
            "schema_version": "dean_specialist_context_review_v1",
            "status": "specialist_context_exact_match_ready",
            "requested_context": {
                "ticker": "NVDA",
                "timeframe": "15m",
                "as_of": "2026-06-29T12:00:00+00:00",
            },
            "ticker_scope": {
                "evidence_scope": "direct_ticker_review_candidate",
            },
            "point_in_time": {
                "status": "point_in_time_compatible",
            },
            "timeframe_alignment": {"status": "aligned"},
            "safety": {
                "eligible_for_exact_pipeline_context": True,
                "manual_review_required": False,
                "decision_influence": False,
            },
        },
        "context_synthesis": {
            "schema_version": "dean_pipeline_context_synthesis_v1",
            "status": "context_synthesis_ready",
            "ticker": "NVDA",
            "timeframe": "15m",
            "regime": {"regime": "TRENDING_UP"},
            "prediction_assessments": [{
                "model_context_id": "ctx_nvda",
                "target_name": "target_intraday_up_15m",
                "context_fingerprint": "fingerprint_nvda",
                "prediction_as_of": (
                    "2026-06-29T12:00:00+00:00"
                ),
                "regime_as_of": (
                    "2026-06-29T11:59:00+00:00"
                ),
                "freshness_status": "compatible",
            }],
            "conflicts": [],
            "directional_synthesis_performed": False,
            "decision_influence": False,
        },
    }

    for component, artifact_payload in artifacts.items():
        artifact = _write(
            tmp_path,
            f"{component}.json",
            artifact_payload,
        )
        result = ShadowComponentCaseProducer(
            base_case_index_path=current_index,
            component=component,
            component_artifact_path=artifact,
        ).build(save=False)
        current_index = _write(
            tmp_path,
            f"after_{component}.json",
            result,
        )

    assert result["component_counts"] == {
        "prediction": 1,
        "regime": 1,
        "specialist": 1,
        "context_synthesis": 1,
    }
    assert result["record_count"] == 4
    assert all(
        not validate_component_case(item)
        for item in result["records"]
        if item["component"] != "prediction"
    )
