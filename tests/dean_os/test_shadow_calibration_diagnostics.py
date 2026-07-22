from __future__ import annotations

import json

import yaml

from dean_os.shadow_calibration_diagnostics import (
    ShadowCalibrationDiagnostics,
)


def _policy(tmp_path, minimum=2):
    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump({
            "schema_version": "dean_shadow_calibration_policy_v1",
            "case_requirements": {
                "diagnostic_min_cases_per_context": minimum,
            },
            "component_metrics": {
                "prediction": {
                    "classification_label": [
                        "balanced_accuracy",
                        "precision",
                        "recall",
                    ],
                },
            },
            "safety_thresholds": {
                "unsafe_output_rate_max": 0.0,
                "time_leakage_rate_max": 0.0,
            },
        }),
        encoding="utf-8",
    )
    return path


def _identity():
    return {
        "ticker": "NVDA",
        "timeframe": "15m",
        "target_name": "target_intraday_up_15m",
        "model_context_id": "ctx_nvda",
        "context_fingerprint": "ctx-nvda",
    }


def _safety():
    return {
        "exact_context_match": True,
        "exact_realization_timestamp_match": True,
        "time_leakage_detected": False,
        "future_evidence_used": False,
        "sector_to_ticker_leakage_detected": False,
        "unsafe_output_detected": False,
        "decision_influence": False,
        "can_trade": False,
    }


def _provenance(component=None):
    result = {
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
    }
    if component:
        result["component_assessment"] = {
            "path": f"{component}.json",
            "sha256": "d" * 64,
        }
    return result


def _episode(index, *, predicted, actual, realized_return, regime):
    prediction_id = f"prediction:{index}"
    prediction = {
        "as_of": f"2026-06-29T12:{index:02d}:00+00:00",
        "value": 0.8 if predicted else 0.2,
        "output_scale": "adjusted_classification_score",
        "raw_value": predicted,
        "raw_output_scale": "class_label_from_predict",
        "positive_class_probability": False,
        "target_type": "classification_binary",
    }
    realization = {
        "expected_end": f"2026-06-29T12:{index + 15:02d}:00+00:00",
        "observed_at": f"2026-06-29T12:{index + 15:02d}:00+00:00",
        "realized_return": realized_return,
        "realized_target": actual,
        "target_type": "classification_binary",
    }
    common = {
        "schema_version": "dean_shadow_calibration_case_v1",
        "case_validation_status": "accepted",
        "identity": _identity(),
        "prediction": prediction,
        "realization": realization,
        "safety": _safety(),
    }
    records = [{
        **common,
        "case_id": prediction_id,
        "component": "prediction",
        "market_regime": regime,
        "source_provenance": _provenance(),
    }]
    assessments = {
        "regime": {
            "status": "accepted",
            "regime": regime,
            "as_of": prediction["as_of"],
        },
        "specialist": {
            "status": "accepted",
            "point_in_time_status": "point_in_time_compatible",
            "timeframe_alignment_status": "aligned",
            "manual_review_required": False,
            "eligible_for_exact_pipeline_context": True,
        },
        "context_synthesis": {
            "status": "accepted",
            "freshness_status": "compatible",
            "directional_synthesis_performed": False,
            "conflict_codes": [],
        },
    }
    for component, assessment in assessments.items():
        records.append({
            **common,
            "case_id": f"{component}:{index}",
            "component": component,
            "base_prediction_case_id": prediction_id,
            "market_regime": regime,
            "assessment": assessment,
            "source_provenance": _provenance(component),
        })
    return records


def _index(tmp_path, records):
    path = tmp_path / "case_index.json"
    path.write_text(json.dumps({
        "mode": "shadow_calibration_case_index",
        "schema_version": "dean_shadow_calibration_case_index_v1",
        "records": records,
    }), encoding="utf-8")
    return path


def test_diagnostics_compute_only_semantically_available_metrics(
    tmp_path,
):
    records = [
        *_episode(
            0,
            predicted=1,
            actual=1,
            realized_return=0.02,
            regime="TRENDING_UP",
        ),
        *_episode(
            1,
            predicted=0,
            actual=0,
            realized_return=-0.01,
            regime="RANGING",
        ),
    ]
    payload = ShadowCalibrationDiagnostics(
        case_index_path=_index(tmp_path, records),
        policy_path=_policy(tmp_path),
        output_dir=tmp_path / "reports",
    ).build(save=False)

    assert payload["status"] == "shadow_diagnostics_ready_for_review"
    assert payload["diagnostic_context_count"] == 1
    diagnostic = payload["diagnostics"][0]
    labels = diagnostic["prediction"]["available"][
        "classification_label"
    ]
    assert labels["balanced_accuracy"] == 1.0
    assert labels["precision"] == 1.0
    assert labels["recall"] == 1.0
    unavailable = diagnostic["prediction"]["unavailable"]
    assert unavailable["classification_probability"] == (
        "final_output_is_not_validated_positive_class_probability"
    )
    assert "adjusted_score_directional_accuracy" in unavailable
    regimes = diagnostic["regime"]["available"][
        "conditional_forward_return"
    ]
    assert regimes["TRENDING_UP"]["mean_forward_return"] == 0.02
    assert regimes["RANGING"]["mean_forward_return"] == -0.01
    assert diagnostic["specialist"]["available"][
        "point_in_time_valid_rate"
    ] == 1.0
    assert diagnostic["context_synthesis"]["available"][
        "freshness_compatibility_rate"
    ] == 1.0
    assert diagnostic["safety_metrics"]["time_leakage_rate"][
        "rate"
    ] == 0.0
    assert payload["safety"]["consensus_weight_eligible"] is False
    assert payload["safety"]["can_trade"] is False


def test_diagnostics_block_zero_or_missing_case_index(tmp_path):
    payload = ShadowCalibrationDiagnostics(
        case_index_path=tmp_path / "missing.json",
        policy_path=_policy(tmp_path),
    ).build(save=False)

    assert payload["status"] == "shadow_diagnostics_blocked"
    assert payload["diagnostic_context_count"] == 0
    assert "case_index_unavailable" in payload["blocking_gaps"]
    assert "no_exact_context_with_minimum_aligned_episodes" in (
        payload["blocking_gaps"]
    )


def test_diagnostics_block_below_minimum_aligned_episodes(tmp_path):
    records = _episode(
        0,
        predicted=1,
        actual=1,
        realized_return=0.02,
        regime="TRENDING_UP",
    )
    payload = ShadowCalibrationDiagnostics(
        case_index_path=_index(tmp_path, records),
        policy_path=_policy(tmp_path, minimum=2),
    ).build(save=False)

    assert payload["status"] == "shadow_diagnostics_blocked"
    coverage = next(iter(payload["context_coverage"].values()))
    assert coverage["common_episode_count"] == 1
    assert payload["diagnostics"] == []


def test_diagnostics_reject_duplicate_component_episode(tmp_path):
    records = [
        *_episode(
            0,
            predicted=1,
            actual=1,
            realized_return=0.02,
            regime="TRENDING_UP",
        ),
        *_episode(
            1,
            predicted=0,
            actual=0,
            realized_return=-0.01,
            regime="RANGING",
        ),
    ]
    duplicate = dict(
        next(
            item for item in records
            if item["component"] == "regime"
            and item["base_prediction_case_id"] == "prediction:0"
        )
    )
    duplicate["case_id"] = "regime:duplicate"
    records.append(duplicate)

    payload = ShadowCalibrationDiagnostics(
        case_index_path=_index(tmp_path, records),
        policy_path=_policy(tmp_path),
    ).build(save=False)

    assert payload["status"] == "shadow_diagnostics_blocked"
    assert any(
        "duplicate_regime_records_per_episode" in item
        for item in payload["blocking_gaps"]
    )
