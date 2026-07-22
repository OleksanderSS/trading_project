from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.pipeline_control_real_metric_evidence_run import PipelineControlRealMetricEvidenceRun


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _clean_inputs(tmp_path: Path) -> dict[str, Path]:
    model_path = _write_json(
        tmp_path / "model_evaluation" / "latest.json",
        {
            "artifact_class": "locked_model_evaluation",
            "joined_lineage": {
                "ticker": "AMD",
                "model": "random_forest",
                "target_name": "target_up_1d",
                "timeframe": "1d",
                "context_fingerprint": "ctx_semis_2026_03",
                "evaluation_window": {
                    "training": {
                        "start": "2026-03-01",
                        "end": "2026-04-01",
                    },
                    "evaluation": {
                        "start": "2026-03-01",
                        "end": "2026-04-01",
                    },
                },
            },
            "join_contract": {
                "join_status": "same_window_lineage_proven"
            },
            "metrics": {
                "total_return": 0.12,
                "pnl": 1200.0,
                "sharpe": 1.1,
                "max_drawdown": 0.08,
                "train_score": 0.72,
                "validation_score": 0.68,
                "sample_count": 250,
            },
        },
    )
    replay_path = _write_json(
        tmp_path / "replay" / "latest.json",
        {"summary": {"clear_hit_rate": 0.72, "clear_evaluated_runs": 8, "quality_blocked_runs": 0}},
    )
    feature_path = _write_json(
        tmp_path / "feature_stability" / "latest.json",
        {
            "artifact_class": "locked_feature_stability_report",
            "training_lineage": {
                "ticker": "AMD",
                "model": "random_forest",
                "target_name": "target_up_1d",
                "timeframe": "1d",
                "context_fingerprint": "ctx_semis_2026_03",
            },
            "assembly_contract": {
                "measured_stability_signal_required": True
            },
            "feature_importance": {"momentum": 0.2, "volume": 0.19, "sentiment": 0.18, "macro": 0.17},
            "feature_stability_score": 0.82,
            "unstable_features": [],
        },
    )
    quality_path = _write_json(tmp_path / "quality" / "latest.json", {"warnings": [], "leakage_flags": []})
    architecture_path = _write_json(
        tmp_path / "architecture" / "latest.json",
        {"summary": {"can_write_production_config_now": False, "can_trade": False}},
    )
    domain_path = _write_json(
        tmp_path / "domain" / "latest.json",
        {"summary": {"can_scale_to_other_domains_now": False, "can_trade": False}},
    )
    return {
        "model_evaluation_json": model_path,
        "feature_stability_report": feature_path,
        "replay_batch_json": replay_path,
        "data_quality_json": quality_path,
        "architecture_map_json": architecture_path,
        "domain_instance_contract_json": domain_path,
    }


def test_pipeline_control_real_metric_evidence_run_clears_with_locked_inputs(tmp_path):
    payload = PipelineControlRealMetricEvidenceRun(tmp_path / "reports").build(
        save=False,
        **_clean_inputs(tmp_path),
    )

    assert payload["summary"]["real_metric_evidence_status"] == "real_metric_evidence_chain_ready"
    assert payload["summary"]["can_use_as_metric_evidence"] is True
    assert payload["summary"]["can_clear_current_real_cautions"] is True
    assert payload["summary"]["readiness_status"] == "metric_inputs_ready"
    assert payload["summary"]["surface_status"] == "clear"
    assert payload["summary"]["instance_status"] == "pipeline_control_instance_review_ready"
    assert payload["summary"]["caution_review_status"] == "pipeline_ready_for_manual_proposal_review"
    assert payload["summary"]["can_write_production_config"] is False
    assert payload["summary"]["can_trade"] is False
    assert len(payload["inputs"]["model_evaluation_sha256"]) == 64
    assert len(payload["inputs"]["feature_stability_sha256"]) == 64
    assert all(check["status"] == "pass" for check in payload["input_evidence_checks"])


def test_pipeline_control_real_metric_evidence_run_rejects_mismatched_metric_pair_lineage(tmp_path):
    inputs = _clean_inputs(tmp_path)
    feature_payload = json.loads(inputs["feature_stability_report"].read_text(encoding="utf-8"))
    feature_payload["training_lineage"]["context_fingerprint"] = "ctx_different_model"
    _write_json(inputs["feature_stability_report"], feature_payload)

    payload = PipelineControlRealMetricEvidenceRun(tmp_path / "reports").build(
        save=False,
        **inputs,
    )

    assert payload["summary"]["real_metric_evidence_status"] == "real_metric_evidence_rejected"
    assert payload["summary"]["can_use_as_metric_evidence"] is False
    assert payload["summary"]["can_clear_current_real_cautions"] is False
    assert "metric_pair_context_fingerprint_matches" in payload["summary"]["failed_input_evidence_checks"]
    assert payload["summary"]["can_trade"] is False


def test_pipeline_control_real_metric_evidence_run_rejects_synthetic_metric_artifact(tmp_path):
    inputs = _clean_inputs(tmp_path)
    _write_json(
        inputs["model_evaluation_json"],
        {
            "mode": "synthetic_pipeline_metric_fixture",
            "fixture_not_evidence": True,
            "metrics": {
                "total_return": 0.12,
                "pnl": 1200.0,
                "sharpe": 1.1,
                "max_drawdown": 0.08,
                "train_score": 0.72,
                "validation_score": 0.68,
                "sample_count": 250,
            },
        },
    )

    payload = PipelineControlRealMetricEvidenceRun(tmp_path / "reports").build(
        save=False,
        **inputs,
    )

    assert payload["summary"]["real_metric_evidence_status"] == "real_metric_evidence_rejected"
    assert payload["summary"]["can_use_as_metric_evidence"] is False
    assert payload["summary"]["can_clear_current_real_cautions"] is False
    assert "model_evaluation_json_not_synthetic" in payload["summary"]["failed_input_evidence_checks"]
    assert payload["summary"]["surface_status"] == "clear"
    assert payload["summary"]["can_trade"] is False


def test_pipeline_control_real_metric_evidence_run_saves_markdown_and_cli_runs(tmp_path):
    inputs = _clean_inputs(tmp_path)

    payload = PipelineControlRealMetricEvidenceRun(tmp_path / "reports").build(**inputs)
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Pipeline Control Real Metric Evidence Run" in markdown
    assert "Can use as metric evidence: True" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_pipeline_control_real_metric_evidence_run.py"),
            "--model-evaluation-json",
            str(inputs["model_evaluation_json"]),
            "--feature-stability-report",
            str(inputs["feature_stability_report"]),
            "--replay-batch-json",
            str(inputs["replay_batch_json"]),
            "--data-quality-json",
            str(inputs["data_quality_json"]),
            "--architecture-map-json",
            str(inputs["architecture_map_json"]),
            "--domain-instance-contract-json",
            str(inputs["domain_instance_contract_json"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Real metric evidence status: real_metric_evidence_chain_ready" in result.stdout
    assert "Can use as metric evidence: True" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
