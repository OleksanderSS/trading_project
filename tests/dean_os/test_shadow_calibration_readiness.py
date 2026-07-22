from __future__ import annotations

import json

import yaml

from dean_os.shadow_calibration_readiness import (
    ShadowCalibrationReadinessPacket,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _policy(tmp_path, minimum=2):
    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump({
            "schema_version": "dean_shadow_calibration_policy_v1",
            "case_requirements": {
                "diagnostic_min_cases_per_context": minimum,
                "weight_review_min_cases_per_context": minimum + 1,
                "minimum_distinct_regimes": 2,
            },
            "component_metrics": {},
            "safety_thresholds": {
                "unsafe_output_rate_max": 0.0,
                "time_leakage_rate_max": 0.0,
            },
        }),
        encoding="utf-8",
    )
    return path


def _sources(tmp_path, *, case_records=None, output_scale=False):
    prediction = _write_json(
        tmp_path / "prediction.json",
        {
            "mode": "pipeline_prediction_review_packet",
            "status": "stage5_prediction_review_ready",
            "contexts": [
                {
                    "target_semantics": {
                        "status": "target_semantics_ready",
                        "calibration": {
                            "model_output_scale_known": output_scale
                        },
                    }
                }
            ],
        },
    )
    specialist = _write_json(
        tmp_path / "specialist.json",
        {
            "mode": "specialist_context_review_packet",
            "status": "specialist_context_exact_match_ready",
            "safety": {
                "eligible_for_exact_pipeline_context": True
            },
        },
    )
    capability = _write_json(
        tmp_path / "capability.json",
        {"mode": "agent_capability_matrix"},
    )
    cases = _write_json(
        tmp_path / "cases.json",
        {
            "mode": "shadow_calibration_case_index",
            "schema_version": (
                "dean_shadow_calibration_case_index_v1"
            ),
            "records": [
                _valid_case(item)
                for item in (case_records or [])
            ],
        },
    )
    return {
        "prediction_review": prediction,
        "specialist_context": specialist,
        "capability_matrix": capability,
        "historical_case_index": cases,
    }


def _valid_case(item):
    component = item["component"]
    record = {
        "schema_version": "dean_shadow_calibration_case_v1",
        "case_id": item["case_id"],
        "case_validation_status": "accepted",
        "component": component,
        "identity": {
            "ticker": item.get("ticker", "NVDA"),
            "timeframe": "15m",
            "target_name": "target_intraday_up_15m",
            "context_fingerprint": item.get(
                "context_fingerprint",
                "ctx-nvda",
            ),
        },
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
    if component != "prediction":
        index = str(item["case_id"]).rsplit(":", 1)[-1]
        record["base_prediction_case_id"] = item.get(
            "base_prediction_case_id",
            f"prediction:{index}",
        )
        record["source_provenance"]["component_assessment"] = {
            "path": f"{component}.json",
            "sha256": "d" * 64,
        }
    if component == "regime":
        record["assessment"] = {
            "status": "accepted",
            "regime": "TRENDING_UP",
            "as_of": "2026-06-29T12:00:00+00:00",
        }
    elif component == "specialist":
        record["assessment"] = {
            "status": "accepted",
            "eligible_for_exact_pipeline_context": True,
            "manual_review_required": False,
        }
    elif component == "context_synthesis":
        record["assessment"] = {
            "status": "accepted",
            "freshness_status": "compatible",
            "directional_synthesis_performed": False,
        }
    return record


def test_shadow_calibration_readiness_blocks_missing_cases_and_scale(
    tmp_path,
):
    packet = ShadowCalibrationReadinessPacket(
        policy_path=_policy(tmp_path),
        sources=_sources(tmp_path),
        output_dir=tmp_path / "reports",
    ).build(save=False)

    assert packet["status"] == "shadow_calibration_blocked"
    assert "model_output_scale_contract_missing" in (
        packet["blocking_gaps"]
    )
    assert "prediction_outcome_case_count_below_policy" in (
        packet["blocking_gaps"]
    )
    assert packet["component_readiness"]["regime"][
        "consensus_weight_eligible"
    ] is False
    assert packet["safety_counters"] == {
        "unsafe_output_count": 0,
        "time_leakage_count": 0,
        "sector_to_ticker_leakage_count": 0,
        "future_evidence_use_count": 0,
        "context_mismatch_accepted_count": 0,
    }
    assert packet["safety"]["automatic_weight_change_allowed"] is False
    assert packet["safety"]["can_trade"] is False


def test_shadow_calibration_contract_can_reach_diagnostic_readiness(
    tmp_path,
):
    records = [
        {
            "component": component,
            "case_id": f"{component}:{index}",
        }
        for component in (
            "prediction",
            "regime",
            "specialist",
            "context_synthesis",
        )
        for index in range(2)
    ]
    packet = ShadowCalibrationReadinessPacket(
        policy_path=_policy(tmp_path, minimum=2),
        sources=_sources(
            tmp_path,
            case_records=records,
            output_scale=True,
        ),
        output_dir=tmp_path / "reports",
    ).build(save=False)

    assert packet["status"] == (
        "shadow_calibration_ready_for_diagnostic_review"
    )
    assert packet["blocking_gaps"] == []
    assert all(
        item["status"] == "ready"
        for item in packet["component_readiness"].values()
    )
    assert all(
        item["consensus_weight_eligible"] is False
        for item in packet["component_readiness"].values()
    )
    assert packet["diagnostic_ready_contexts"] == [
        "NVDA|15m|target_intraday_up_15m|ctx-nvda"
    ]


def test_shadow_calibration_requires_one_common_exact_context(
    tmp_path,
):
    tickers = {
        "prediction": "NVDA",
        "regime": "MSFT",
        "specialist": "AMD",
        "context_synthesis": "TSM",
    }
    records = [
        {
            "component": component,
            "case_id": f"{component}:{index}",
            "ticker": tickers[component],
            "context_fingerprint": f"ctx-{tickers[component].lower()}",
        }
        for component in tickers
        for index in range(2)
    ]
    packet = ShadowCalibrationReadinessPacket(
        policy_path=_policy(tmp_path, minimum=2),
        sources=_sources(
            tmp_path,
            case_records=records,
            output_scale=True,
        ),
        output_dir=tmp_path / "reports",
    ).build(save=False)

    assert packet["diagnostic_ready_contexts"] == []
    assert "no_common_exact_context_meets_diagnostic_policy" in (
        packet["blocking_gaps"]
    )


def test_shadow_calibration_requires_aligned_outcome_episodes(
    tmp_path,
):
    records = []
    for component in (
        "prediction",
        "regime",
        "specialist",
        "context_synthesis",
    ):
        for index in range(2):
            item = {
                "component": component,
                "case_id": f"{component}:{index}",
            }
            if component != "prediction":
                item["base_prediction_case_id"] = (
                    f"prediction:{index + 10}"
                )
            records.append(item)
    packet = ShadowCalibrationReadinessPacket(
        policy_path=_policy(tmp_path, minimum=2),
        sources=_sources(
            tmp_path,
            case_records=records,
            output_scale=True,
        ),
        output_dir=tmp_path / "reports",
    ).build(save=False)

    context_key = "NVDA|15m|target_intraday_up_15m|ctx-nvda"
    assert packet["context_case_counts"]["prediction"][
        context_key
    ] == 2
    assert packet["common_episode_counts"][context_key] == 0
    assert packet["diagnostic_ready_contexts"] == []
    assert "no_common_exact_context_meets_diagnostic_policy" in (
        packet["blocking_gaps"]
    )


def test_shadow_calibration_rejects_invalid_case_index_records(
    tmp_path,
):
    sources = _sources(tmp_path)
    _write_json(
        sources["historical_case_index"],
        {
            "mode": "shadow_calibration_case_index",
            "schema_version": (
                "dean_shadow_calibration_case_index_v1"
            ),
            "records": [
                {"component": "prediction", "case_id": "fake"}
            ],
        },
    )

    packet = ShadowCalibrationReadinessPacket(
        policy_path=_policy(tmp_path, minimum=1),
        sources=sources,
        output_dir=tmp_path / "reports",
    ).build(save=False)

    assert packet["invalid_case_record_count"] == 1
    assert "historical_case_index_contains_invalid_records" in (
        packet["blocking_gaps"]
    )
