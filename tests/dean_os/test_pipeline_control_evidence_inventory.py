from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.pipeline_control.pipeline_control_evidence_inventory import PipelineControlEvidenceInventory


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_pipeline_control_evidence_inventory_keeps_partial_outputs_out_of_metric_evidence(tmp_path):
    partial_model = _write_json(
        tmp_path / "stage_5_results.json",
        {
            "ticker": "AMD",
            "model_name": "random_forest",
            "target_column": "AMD_target_up_1d",
            "accuracy": 0.6346,
        },
    )
    selected_features = _write_json(
        tmp_path / "selected_features.json",
        {"ticker": "AMD", "selected_features": ["momentum_20d", "news_sentiment", "macro_rate"]},
    )
    replay = _write_json(
        tmp_path / "replay" / "latest.json",
        {
            "summary": {
                "clear_hit_rate": 0.57,
                "clear_evaluated_runs": 26,
                "quality_blocked_runs": 0,
                "max_drawdown": 0.11,
            }
        },
    )
    data_quality = _write_json(
        tmp_path / "feature_lineage_report.json",
        {"columns": ["datetime", "ticker", "interval"], "warnings": [], "leakage_flags": []},
    )

    payload = PipelineControlEvidenceInventory(tmp_path / "reports").build(
        candidate_paths=[partial_model, selected_features, replay, data_quality],
        save=False,
    )

    summary = payload["summary"]
    assert summary["inventory_status"] == "real_pipeline_outputs_found_but_metric_evidence_incomplete"
    assert summary["ready_model_evaluation_candidate_count"] == 0
    assert summary["ready_feature_stability_candidate_count"] == 0
    assert summary["partial_model_metadata_count"] == 1
    assert summary["selected_feature_manifest_count"] == 1
    assert summary["supporting_artifact_count"] == 3
    assert summary["can_run_real_metric_evidence_now"] is False
    assert summary["can_clear_current_real_cautions"] is False
    assert summary["can_trade"] is False

    classifications = {item["artifact_id"]: item["classification"] for item in payload["candidate_artifacts"]}
    assert classifications["stage_5_results"] == "partial_model_metadata_not_locked_evaluation"
    assert classifications["selected_features"] == "selected_feature_manifest_only"
    assert classifications["latest"] == "supporting_replay_batch"
    assert classifications["feature_lineage_report"] == "supporting_data_quality_or_lineage"
    assert "max_drawdown" in payload["real_metric_evidence_gap"]["missing_for_model_evaluation"]
    assert payload["next_runner_inputs"]["can_invoke_pipeline_control_real_metric_evidence_run"] is False


def test_pipeline_control_evidence_inventory_detects_ready_locked_inputs(tmp_path):
    model = _write_json(
        tmp_path / "model_evaluation" / "latest.json",
        {
            "artifact_class": "locked_model_evaluation",
            "joined_lineage": {
                "ticker": "AMD",
                "model": "random_forest",
                "target_name": "target_up_1d",
                "timeframe": "1d",
                "context_fingerprint": "ctx_ready",
                "evaluation_window": {
                    "training": {
                        "start": "2026-01-01",
                        "end": "2026-02-01",
                    },
                    "evaluation": {
                        "start": "2026-01-01",
                        "end": "2026-02-01",
                    },
                },
            },
            "join_contract": {
                "join_status": "same_window_lineage_proven"
            },
            "metrics": {
                "total_return": 0.12,
                "max_drawdown": 0.08,
                "train_score": 0.72,
                "validation_score": 0.68,
                "sample_count": 250,
            },
        },
    )
    feature = _write_json(
        tmp_path / "feature_stability" / "latest.json",
        {
            "artifact_class": "locked_feature_stability_report",
            "training_lineage": {
                "ticker": "AMD",
                "model": "random_forest",
                "target_name": "target_up_1d",
                "timeframe": "1d",
                "context_fingerprint": "ctx_ready",
            },
            "assembly_contract": {
                "measured_stability_signal_required": True
            },
            "feature_importance": {"momentum": 0.2, "volume": 0.19},
            "feature_stability_score": 0.82,
            "unstable_features": [],
        },
    )

    payload = PipelineControlEvidenceInventory(tmp_path / "reports").build(
        candidate_paths=[model, feature],
        save=False,
    )

    summary = payload["summary"]
    assert summary["inventory_status"] == "real_metric_evidence_inputs_available"
    assert summary["ready_model_evaluation_candidate_count"] == 1
    assert summary["ready_feature_stability_candidate_count"] == 1
    assert summary["can_run_real_metric_evidence_now"] is True
    assert summary["can_clear_current_real_cautions"] is False
    assert summary["real_metric_evidence_run_required"] is True
    assert summary["can_trade"] is False
    assert payload["real_metric_evidence_gap"]["missing_for_model_evaluation"] == []
    assert payload["real_metric_evidence_gap"]["missing_for_feature_stability"] == []
    assert payload["next_runner_inputs"]["model_evaluation_json"] == str(model)
    assert payload["next_runner_inputs"]["feature_stability_report"] == str(feature)


def test_pipeline_control_evidence_inventory_rejects_synthetic_metric_candidate(tmp_path):
    model = _write_json(
        tmp_path / "synthetic_model.json",
        {
            "mode": "synthetic_pipeline_metric_fixture",
            "fixture_not_evidence": True,
            "metrics": {
                "max_drawdown": 0.08,
                "train_score": 0.72,
                "validation_score": 0.68,
                "sample_count": 250,
            },
        },
    )

    payload = PipelineControlEvidenceInventory(tmp_path / "reports").build(candidate_paths=[model], save=False)

    assert payload["summary"]["inventory_status"] == "real_pipeline_outputs_found_but_metric_evidence_incomplete"
    assert payload["summary"]["ready_model_evaluation_candidate_count"] == 0
    assert payload["candidate_artifacts"][0]["classification"] == "synthetic_or_fixture_not_metric_evidence"
    assert payload["summary"]["can_trade"] is False


def test_pipeline_control_evidence_inventory_rejects_complete_shape_without_locked_provenance(
    tmp_path,
):
    model = _write_json(
        tmp_path / "complete_but_unproven.json",
        {
            "metrics": {
                "max_drawdown": 0.08,
                "train_score": 0.72,
                "validation_score": 0.68,
                "sample_count": 250,
            }
        },
    )

    payload = PipelineControlEvidenceInventory(
        tmp_path / "reports"
    ).build(candidate_paths=[model], save=False)
    record = payload["candidate_artifacts"][0]

    assert record["classification"] == (
        "complete_model_shape_without_locked_provenance"
    )
    assert record["usable_as_model_evaluation"] is False
    assert record["recognized_fields"]["model_provenance"]["valid"] is False
    assert payload["summary"]["can_run_real_metric_evidence_now"] is False


def test_pipeline_control_evidence_inventory_keeps_walk_forward_development_only(tmp_path):
    walk_forward = _write_json(
        tmp_path / "walk_forward" / "latest.json",
        {
            "mode": "pipeline_control_walk_forward_validation_run",
            "summary": {
                "validation_status": "walk_forward_candidate_blocked_by_validation_contract",
                "contract_passed": False,
                "test_rows_loaded": 0,
                "past_evaluation_rows_loaded": 0,
            },
            "walk_forward_candidate": {
                "artifact_class": "pipeline_control_walk_forward_validation_candidate",
                "evidence_class": "development_train_validation_only",
                "contract_status": "walk_forward_candidate_blocked_by_validation_contract",
                "metrics": {
                    "max_drawdown": 0.08,
                    "train_score": 0.81,
                    "validation_score": 0.52,
                    "sample_count": 480,
                    "contract_passed": False,
                },
                "feature_importance": {"momentum": 0.4},
                "feature_stability_score": 0.53,
                "test_contract": {
                    "eligible_as_locked_test_evidence": False,
                    "test_rows_loaded": 0,
                },
            },
        },
    )

    payload = PipelineControlEvidenceInventory(tmp_path / "reports").build(
        candidate_paths=[walk_forward],
        save=False,
    )

    record = payload["candidate_artifacts"][0]
    assert record["classification"] == "supporting_walk_forward_train_validation"
    assert record["development_only_candidate"] is True
    assert record["eligible_as_locked_test_evidence"] is False
    assert record["usable_as_model_evaluation"] is False
    assert record["usable_as_feature_stability"] is False
    assert payload["summary"]["walk_forward_validation_candidate_count"] == 1
    assert payload["summary"]["blocked_walk_forward_candidate_count"] == 1
    assert payload["summary"]["ready_model_evaluation_candidate_count"] == 0
    assert payload["summary"]["ready_feature_stability_candidate_count"] == 0
    assert payload["summary"]["can_clear_current_real_cautions"] is False
    assert payload["next_runner_inputs"]["model_evaluation_json"] is None
    assert payload["next_runner_inputs"]["feature_stability_report"] is None
    assert "max_drawdown" in payload["real_metric_evidence_gap"]["missing_for_model_evaluation"]


def test_pipeline_control_evidence_inventory_keeps_forward_accrual_plan_supporting_only(tmp_path):
    plan = _write_json(
        tmp_path / "forward_accrual" / "latest.json",
        {
            "mode": "pipeline_control_forward_data_accrual_plan",
            "summary": {
                "plan_status": "forward_development_accrual_plan_ready",
                "can_call_next_data_virgin_holdout": False,
            },
            "accrual_plan": {
                "artifact_class": "pipeline_control_forward_data_accrual_plan",
                "evidence_class": "prospective_development_data_boundary",
                "lane": "development_refresh_only",
            },
        },
    )

    payload = PipelineControlEvidenceInventory(tmp_path / "reports").build(
        candidate_paths=[plan],
        save=False,
    )

    record = payload["candidate_artifacts"][0]
    assert record["classification"] == "supporting_forward_development_accrual_plan"
    assert record["eligible_as_locked_test_evidence"] is False
    assert record["usable_as_model_evaluation"] is False
    assert record["usable_as_feature_stability"] is False
    assert payload["summary"]["forward_development_accrual_plan_count"] == 1
    assert payload["summary"]["can_run_real_metric_evidence_now"] is False
    assert payload["next_runner_inputs"]["model_evaluation_json"] is None
    assert payload["next_runner_inputs"]["feature_stability_report"] is None


def test_pipeline_control_evidence_inventory_keeps_forward_accrual_gate_supporting_only(tmp_path):
    gate = _write_json(
        tmp_path / "forward_accrual_gate" / "latest.json",
        {
            "mode": "pipeline_control_forward_data_accrual_gate",
            "summary": {
                "gate_status": "blocked_forward_development_artifact",
                "can_supply_next_development_run": False,
                "can_use_as_locked_test_evidence": False,
            },
            "eligible_development_artifact": None,
        },
    )

    payload = PipelineControlEvidenceInventory(tmp_path / "reports").build(
        candidate_paths=[gate],
        save=False,
    )

    record = payload["candidate_artifacts"][0]
    assert record["classification"] == "supporting_forward_development_accrual_gate"
    assert record["eligible_as_locked_test_evidence"] is False
    assert record["usable_as_model_evaluation"] is False
    assert record["usable_as_feature_stability"] is False
    assert payload["summary"]["forward_development_accrual_gate_count"] == 1
    assert payload["summary"]["blocked_forward_development_artifact_count"] == 1
    assert payload["summary"]["can_run_real_metric_evidence_now"] is False


def test_pipeline_control_evidence_inventory_saves_markdown_and_cli_runs(tmp_path):
    lineage = {
        "ticker": "AMD",
        "model": "random_forest",
        "target_name": "target_up_1d",
        "timeframe": "1d",
        "context_fingerprint": "ctx_cli",
    }
    model = _write_json(
        tmp_path / "model.json",
        {
            "artifact_class": "locked_model_evaluation",
            "joined_lineage": {
                **lineage,
                "evaluation_window": {
                    "training": {
                        "start": "2026-01-01",
                        "end": "2026-02-01",
                    },
                    "evaluation": {
                        "start": "2026-01-01",
                        "end": "2026-02-01",
                    },
                },
            },
            "join_contract": {
                "join_status": "same_window_lineage_proven"
            },
            "metrics": {
                "max_drawdown": 0.08,
                "train_score": 0.72,
                "validation_score": 0.68,
                "sample_count": 250,
            },
        },
    )
    feature = _write_json(
        tmp_path / "features.json",
        {
            "artifact_class": "locked_feature_stability_report",
            "training_lineage": lineage,
            "assembly_contract": {
                "measured_stability_signal_required": True
            },
            "feature_importance": {"momentum": 0.2},
            "unstable_features": [],
        },
    )

    payload = PipelineControlEvidenceInventory(tmp_path / "reports").build(candidate_paths=[model, feature])
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Pipeline Control Evidence Inventory" in markdown
    assert "Can run real metric evidence now: True" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_pipeline_control_evidence_inventory.py"),
            "--candidate-paths",
            str(model),
            str(feature),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Inventory status: real_metric_evidence_inputs_available" in result.stdout
    assert "Can run real metric evidence now: True" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
