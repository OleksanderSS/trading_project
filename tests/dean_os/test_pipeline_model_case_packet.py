from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dean_os.packets.pipeline_model_case_packet import (
    PipelineModelCasePacket,
    inspect_pipeline_model_case,
)
from dean_os.chief_review_index import classify_review_index
from dean_os.review_index import ReviewIndexBuilder


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_payload() -> dict:
    return {
        "run_id": "locked_model_run",
        "artifact_class": "locked_model_evaluation",
        "metrics": {
            "max_drawdown": 0.11,
            "train_score": 0.89,
            "validation_score": 0.58,
            "sample_count": 95,
            "total_return": 0.17,
            "sharpe": 1.2,
        },
        "joined_lineage": {
            "ticker": "AMD",
            "model": "random_forest",
            "target_name": "target_intraday_up_15m",
            "timeframe": "15m",
            "context_fingerprint": "ctx-1",
            "evaluation_window": {
                "training": {
                    "start": "2026-06-01T00:00:00+00:00",
                    "end": "2026-06-20T00:00:00+00:00",
                },
                "evaluation": {
                    "start": "2026-06-21T00:00:00+00:00",
                    "end": "2026-06-24T19:30:00+00:00",
                },
            },
        },
        "evaluated_at": "2026-06-24T19:30:00+00:00",
        "join_contract": {
            "join_status": "same_window_lineage_proven",
        },
    }


def _feature_payload() -> dict:
    return {
        "run_id": "locked_feature_run",
        "artifact_class": "locked_feature_stability_report",
        "training_lineage": {
            "ticker": "AMD",
            "model": "random_forest",
            "target_name": "target_intraday_up_15m",
            "timeframe": "15m",
            "context_fingerprint": "ctx-1",
        },
        "feature_importance": {
            "feature_a": 0.6,
            "feature_b": 0.4,
        },
        "feature_stability_score": 0.598726,
        "unstable_feature_count": 0,
        "materialization_contract": {
            "source_locked_artifact_verified": True,
            "source_provenance_proof": "measured_stability_assembled",
        },
    }


def _readiness_payload(
    model_path: Path,
    feature_path: Path,
) -> dict:
    return {
        "run_id": "readiness_run",
        "mode": "pipeline_metric_input_readiness_gate",
        "inputs": {
            "model_performance_path": str(model_path),
            "feature_report_path": str(feature_path),
        },
        "summary": {
            "blocked_metric_planes": [
                "validation",
                "feature_stability",
            ],
            "caution_metric_planes": [],
        },
        "metric_plane_readiness": [
            {
                "name": "profitability",
                "status": "clear",
                "score": 1.0,
                "metrics": {
                    "total_return": 0.17,
                    "sharpe": 1.2,
                },
                "constraints": {
                    "min_total_return": 0.0,
                    "min_sharpe": 0.0,
                },
                "reasons": ["Profitability floor is satisfied."],
            },
            {
                "name": "risk",
                "status": "clear",
                "score": 0.56,
                "metrics": {"max_drawdown": 0.11},
                "constraints": {"max_drawdown": 0.25},
                "reasons": ["Drawdown is inside the cap."],
            },
            {
                "name": "validation",
                "status": "blocked",
                "score": 0.0,
                "metrics": {
                    "train_score": 0.89,
                    "validation_score": 0.58,
                    "train_test_gap": 0.31,
                    "sample_count": 95,
                },
                "constraints": {
                    "min_validation_score": 0.55,
                    "max_train_test_gap": 0.15,
                    "min_sample_count": 50,
                },
                "reasons": [
                    "Train/test gap exceeds the configured cap."
                ],
            },
            {
                "name": "feature_stability",
                "status": "blocked",
                "score": 0.598726,
                "metrics": {
                    "feature_stability_score": 0.598726,
                    "feature_concentration": 0.6,
                    "max_feature_weight_abs": 0.6,
                    "unstable_feature_count": 0,
                },
                "constraints": {
                    "min_feature_stability_score": 0.7,
                    "max_feature_concentration": 0.7,
                    "max_feature_weight_abs": 0.7,
                    "max_unstable_features": 0,
                },
                "reasons": [
                    "Feature stability is below the configured floor."
                ],
            },
        ],
    }


def _write_bound_inputs(
    tmp_path: Path,
    *,
    chain_created_at: str = "2026-06-28T12:00:00+00:00",
) -> dict[str, Path]:
    model_path = _write_json(
        tmp_path / "model" / "latest.json",
        _model_payload(),
    )
    feature_path = _write_json(
        tmp_path / "feature" / "latest.json",
        _feature_payload(),
    )
    readiness_path = _write_json(
        tmp_path / "readiness" / "latest.json",
        _readiness_payload(model_path, feature_path),
    )
    chain = {
        "run_id": "real_chain_run",
        "created_at": chain_created_at,
        "mode": "pipeline_control_real_metric_evidence_run",
        "inputs": {
            "model_evaluation_json": str(model_path),
            "model_evaluation_sha256": _sha(model_path),
            "feature_stability_report": str(feature_path),
            "feature_stability_sha256": _sha(feature_path),
        },
        "summary": {
            "real_metric_evidence_status": (
                "real_metric_evidence_blocked_by_metric_planes"
            ),
            "can_use_as_metric_evidence": True,
            "can_clear_current_real_cautions": False,
            "blocked_metric_planes": [
                "validation",
                "feature_stability",
            ],
            "caution_metric_planes": [],
        },
        "chain_results": [
            {
                "step_id": "pipeline_metric_input_readiness",
                "status": "blocked_metric_inputs",
                "latest_json": str(readiness_path),
                "latest_json_sha256": _sha(readiness_path),
                "blocked_metric_planes": [
                    "validation",
                    "feature_stability",
                ],
                "caution_metric_planes": [],
            }
        ],
    }
    chain_path = _write_json(
        tmp_path / "chain" / "latest.json",
        chain,
    )
    return {
        "model": model_path,
        "feature": feature_path,
        "readiness": readiness_path,
        "chain": chain_path,
    }


def test_pipeline_model_case_builds_negative_evaluation_case(tmp_path):
    paths = _write_bound_inputs(tmp_path)

    payload = PipelineModelCasePacket(
        tmp_path / "reports"
    ).build(
        real_metric_evidence_json=paths["chain"],
        model_evaluation_json=paths["model"],
        feature_stability_json=paths["feature"],
        save=False,
    )

    assert payload["summary"]["case_status"] == (
        "evaluation_block_case_ready"
    )
    assert payload["summary"]["case_classification"] == (
        "negative_evaluation_block_case"
    )
    assert payload["summary"]["case_scope"] == (
        "ticker_model_evaluation_only"
    )
    assert payload["summary"]["domain_profile_association"] is None
    assert payload["summary"]["eligible_as_domain_evidence"] is False
    assert payload["case"]["sector_scope"] is None
    assert payload["summary"]["result_label"] == (
        "failed_validation_and_feature_stability"
    )
    assert payload["summary"]["root_cause_categories"] == [
        "generalization_gap",
        "feature_instability",
    ]
    assert payload["case"]["forecast_outcome_label"] is None
    assert payload["new_data_requirement"]["status"] == (
        "wait_for_new_forward_development_data"
    )
    assert payload["new_data_requirement"]["same_fold_retry_allowed"] is False
    assert payload["learning_bridge"]["learning_candidate_created"] is False
    assert payload["summary"]["can_launch_model_variant_now"] is False
    assert payload["summary"]["can_write_learning_memory"] is False
    assert payload["summary"]["can_trade"] is False
    assert {
        item["test_id"]
        for item in payload["evaluation_test_candidates"]
    } == {
        "regression_validation_train_test_gap",
        "regression_feature_stability_feature_stability_score",
    }


def test_pipeline_model_case_dedupe_ignores_wrapper_generation_time(
    tmp_path,
):
    first_paths = _write_bound_inputs(
        tmp_path / "first",
        chain_created_at="2026-06-28T12:00:00+00:00",
    )
    second_paths = _write_bound_inputs(
        tmp_path / "second",
        chain_created_at="2026-06-28T13:00:00+00:00",
    )

    first = PipelineModelCasePacket().build(
        real_metric_evidence_json=first_paths["chain"],
        model_evaluation_json=first_paths["model"],
        feature_stability_json=first_paths["feature"],
        save=False,
    )
    second = PipelineModelCasePacket().build(
        real_metric_evidence_json=second_paths["chain"],
        model_evaluation_json=second_paths["model"],
        feature_stability_json=second_paths["feature"],
        save=False,
    )

    assert first["case"]["dedupe_fingerprint"] == (
        second["case"]["dedupe_fingerprint"]
    )
    assert first["case"]["case_id"] == second["case"]["case_id"]


def test_pipeline_model_case_rejects_stale_readiness_binding(tmp_path):
    paths = _write_bound_inputs(tmp_path)
    readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    readiness["metric_plane_readiness"][2]["metrics"][
        "train_test_gap"
    ] = 0.01
    paths["readiness"].write_text(
        json.dumps(readiness),
        encoding="utf-8",
    )

    payload = PipelineModelCasePacket().build(
        real_metric_evidence_json=paths["chain"],
        model_evaluation_json=paths["model"],
        feature_stability_json=paths["feature"],
        save=False,
    )

    assert payload["summary"]["case_status"] == (
        "pipeline_model_case_rejected"
    )
    assert "metric_readiness_sha256_matches_chain" in (
        payload["summary"]["failed_review_checks"]
    )


def test_pipeline_model_case_inspection_binds_agent_inputs_and_cli(
    tmp_path,
):
    paths = _write_bound_inputs(tmp_path)
    report_dir = tmp_path / "reports"
    payload = PipelineModelCasePacket(report_dir).build(
        real_metric_evidence_json=paths["chain"],
        model_evaluation_json=paths["model"],
        feature_stability_json=paths["feature"],
    )

    inspection = inspect_pipeline_model_case(
        payload["saved_paths"]["latest_json"],
        expected_model_evaluation_path=paths["model"],
        expected_evidence_chain_path=paths["chain"],
    )

    assert inspection["status"] == "evaluation_block_case_ready"
    assert inspection["usable_for_review"] is True
    assert inspection["can_write_learning_memory"] is False
    assert inspection["can_trade"] is False

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(
                repo_root
                / "run_agent_pipeline_model_case_packet.py"
            ),
            "--real-metric-evidence-json",
            str(paths["chain"]),
            "--model-evaluation-json",
            str(paths["model"]),
            "--feature-stability-json",
            str(paths["feature"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Case status: evaluation_block_case_ready" in result.stdout
    assert "Can write learning memory: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()


def test_pipeline_model_case_routes_to_review_index_and_blocks_chief(
    tmp_path,
):
    paths = _write_bound_inputs(tmp_path)
    payload = PipelineModelCasePacket(
        tmp_path / "case_reports"
    ).build(
        real_metric_evidence_json=paths["chain"],
        model_evaluation_json=paths["model"],
        feature_stability_json=paths["feature"],
    )
    review_index = ReviewIndexBuilder(
        sources={
            "pipeline_model_case": payload["saved_paths"][
                "latest_json"
            ]
        }
    ).build(save=False)

    entry = review_index["entries"][0]
    decision = classify_review_index(review_index)

    assert entry["status"] == "evaluation_block_case_ready"
    assert entry["summary"]["root_cause_categories"] == [
        "generalization_gap",
        "feature_instability",
    ]
    assert decision["decision"] == "model_candidate_blocked"
    assert decision["model_case_state"]["blocked"] is True
    assert "wait for accepted new forward data" in (
        decision["next_actions"][0]
    )
