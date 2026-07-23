from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.pipeline_control.pipeline_control_locked_evaluation_assembler import PipelineControlLockedEvaluationAssembler


def test_locked_evaluation_assembler_blocks_partial_stage4_stage7_join(tmp_path):
    training = _write_json(tmp_path / "training_candidate.json", _training_candidate())
    evaluation = _write_json(
        tmp_path / "evaluation_candidate.json",
        {
            "artifact_class": "pipeline_control_evaluation_metric_candidate",
            "evidence_class": "pipeline_stage_7_evaluation_output",
            "metrics": {"max_drawdown": 0.08, "total_return": 0.12, "sharpe": 1.1},
            "evaluation_window": {"start": "2026-01-01", "end": "2026-02-01", "sample_count": 22},
            "lineage": {"tickers": ["AMD"], "selected_primary_models": ["AMD_target_up_1d_random_forest"]},
        },
    )

    payload = PipelineControlLockedEvaluationAssembler(tmp_path / "reports").build(
        training_candidate_json=training,
        evaluation_candidate_json=evaluation,
        save=False,
    )

    summary = payload["summary"]
    assert summary["assembly_status"] == "blocked_missing_same_window_lineage"
    assert summary["same_window_lineage_proven"] is False
    assert summary["locked_model_evaluation_written"] is False
    assert summary["can_supply_model_evaluation_to_real_runner"] is False
    assert summary["can_run_real_metric_evidence_now"] is False
    assert summary["can_trade"] is False
    assert "training_evaluation_window_present" in summary["blocked_check_codes"]
    assert "evaluation_target_name_present" in summary["blocked_check_codes"]
    assert payload["next_runner_inputs"]["model_evaluation_json"] is None


def test_locked_evaluation_assembler_writes_only_when_lineage_matches(tmp_path):
    window = {"start": "2026-01-01", "end": "2026-02-01", "sample_count": 22}
    training_payload = _training_candidate()
    training_payload["evaluation_window"] = window
    training = _write_json(tmp_path / "training_candidate.json", training_payload)
    evaluation = _write_json(
        tmp_path / "evaluation_candidate.json",
        {
            "artifact_class": "pipeline_control_evaluation_metric_candidate",
            "evidence_class": "pipeline_stage_7_evaluation_output",
            "ticker": "AMD",
            "target_name": "target_up_1d",
            "model_type": "random_forest",
            "timeframe": "1d",
            "context_fingerprint": "ctx_semis_2026_01",
            "metrics": {"max_drawdown": 0.08, "total_return": 0.12, "sharpe": 1.1},
            "evaluation_window": window,
            "lineage": {"tickers": ["AMD"], "selected_primary_models": ["AMD_target_up_1d_random_forest"]},
        },
    )

    payload = PipelineControlLockedEvaluationAssembler(tmp_path / "reports").build(
        training_candidate_json=training,
        evaluation_candidate_json=evaluation,
    )

    summary = payload["summary"]
    assert summary["assembly_status"] == "locked_model_evaluation_assembled"
    assert summary["same_window_lineage_proven"] is True
    assert summary["locked_model_evaluation_written"] is True
    assert summary["can_supply_model_evaluation_to_real_runner"] is True
    assert summary["feature_stability_report_required_separately"] is True
    assert summary["can_run_real_metric_evidence_now"] is False
    assert summary["can_trade"] is False

    next_inputs = payload["next_runner_inputs"]
    assert Path(next_inputs["model_evaluation_json"]).exists()
    locked = json.loads(Path(next_inputs["model_evaluation_json"]).read_text(encoding="utf-8"))
    assert locked["artifact_class"] == "locked_model_evaluation"
    assert locked["metrics"]["max_drawdown"] == 0.08
    assert locked["metrics"]["train_score"] == 0.72
    assert locked["metrics"]["validation_score"] == 0.68
    assert locked["metrics"]["sample_count"] == 250.0
    assert locked["join_contract"]["join_status"] == "same_window_lineage_proven"
    assert locked["evaluated_at"] == "2026-02-01"
    assert len(
        locked["source_artifact_hashes"]["training_candidate_sha256"]
    ) == 64
    assert len(
        locked["source_artifact_hashes"]["evaluation_candidate_sha256"]
    ) == 64


def test_locked_evaluation_assembler_rejects_synthetic_candidates(tmp_path):
    training_payload = _training_candidate()
    training_payload["evaluation_window"] = {"start": "2026-01-01", "end": "2026-02-01"}
    training_payload["fixture_not_evidence"] = True
    training = _write_json(tmp_path / "training_candidate.json", training_payload)
    evaluation = _write_json(
        tmp_path / "evaluation_candidate.json",
        {
            "artifact_class": "pipeline_control_evaluation_metric_candidate",
            "ticker": "AMD",
            "target_name": "target_up_1d",
            "model_type": "random_forest",
            "timeframe": "1d",
            "context_fingerprint": "ctx_semis_2026_01",
            "metrics": {"max_drawdown": 0.08},
            "evaluation_window": {"start": "2026-01-01", "end": "2026-02-01"},
        },
    )

    payload = PipelineControlLockedEvaluationAssembler(tmp_path / "reports").build(
        training_candidate_json=training,
        evaluation_candidate_json=evaluation,
        save=False,
    )

    assert payload["summary"]["assembly_status"] == "blocked_missing_same_window_lineage"
    assert "training_candidate_not_synthetic" in payload["summary"]["blocked_check_codes"]
    assert payload["summary"]["can_trade"] is False


def test_locked_evaluation_assembler_cli_runs(tmp_path):
    window = {"start": "2026-01-01", "end": "2026-02-01"}
    training_payload = _training_candidate()
    training_payload["evaluation_window"] = window
    training = _write_json(tmp_path / "training_candidate.json", training_payload)
    evaluation = _write_json(
        tmp_path / "evaluation_candidate.json",
        {
            "artifact_class": "pipeline_control_evaluation_metric_candidate",
            "ticker": "AMD",
            "target_name": "target_up_1d",
            "model_type": "random_forest",
            "timeframe": "1d",
            "context_fingerprint": "ctx_semis_2026_01",
            "metrics": {"max_drawdown": 0.08},
            "evaluation_window": window,
        },
    )
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_pipeline_control_locked_evaluation_assembler.py"),
            "--training-candidate-json",
            str(training),
            "--evaluation-candidate-json",
            str(evaluation),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Assembly status: locked_model_evaluation_assembled" in result.stdout
    assert "Can run real metric evidence now: False" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()


def _training_candidate() -> dict:
    return {
        "artifact_class": "pipeline_control_model_evaluation_candidate",
        "evidence_class": "pipeline_training_output",
        "ticker": "AMD",
        "target_name": "target_up_1d",
        "model_type": "random_forest",
        "timeframe": "1d",
        "context_fingerprint": "ctx_semis_2026_01",
        "metrics": {
            "train_score": 0.72,
            "validation_score": 0.68,
            "test_score": 0.68,
            "sample_count": 250,
        },
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
