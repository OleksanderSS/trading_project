from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.pipeline_control.pipeline_control_metric_artifact_materializer import PipelineControlMetricArtifactMaterializer
from dean_os.pipeline_control.pipeline_control_real_metric_evidence_run import PipelineControlRealMetricEvidenceRun


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_metric_artifact_materializer_blocks_partial_real_outputs(tmp_path):
    stage_result = _write_json(
        tmp_path / "stage_5_results.json",
        {
            "models_metadata": {
                "AMD_target_up_1d_random_forest": {
                    "ticker": "AMD",
                    "model_type": "random_forest",
                    "metrics": {"accuracy": 0.6346},
                    "selected_features": ["open", "close", "news_sentiment"],
                }
            }
        },
    )
    backtest_summary = _write_json(
        tmp_path / "summary.json",
        {
            "metrics": {"total_return": 0.03, "sharpe_ratio": 1.8, "max_drawdown": -0.02},
            "backtest_stats": {"win_rate": 0.2},
        },
    )
    selected_features = _write_json(
        tmp_path / "selected_features.json",
        {"selected_features": ["open", "close", "news_sentiment"]},
    )

    payload = PipelineControlMetricArtifactMaterializer(tmp_path / "reports").build(
        candidate_paths=[stage_result, backtest_summary, selected_features],
        save=False,
    )

    summary = payload["summary"]
    assert summary["materialization_status"] == "blocked_missing_locked_metric_artifacts"
    assert summary["ready_model_candidate_found"] is False
    assert summary["ready_feature_candidate_found"] is False
    assert summary["materialized_model_evaluation_json"] is False
    assert summary["materialized_feature_stability_report"] is False
    assert summary["can_run_real_metric_evidence_now"] is False
    assert summary["can_clear_cautions_without_real_runner"] is False
    assert summary["can_trade"] is False
    assert "max_drawdown" in payload["materialization_gap"]["missing_for_model_evaluation"]
    assert "feature_importance" in payload["materialization_gap"]["missing_for_feature_stability"]
    assert payload["next_runner_inputs"]["can_invoke_pipeline_control_real_metric_evidence_run"] is False


def test_metric_artifact_materializer_writes_locked_pair_and_real_runner_accepts_it(tmp_path):
    model = _write_json(
        tmp_path / "locked_model_eval.json",
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
                        "start": "2026-03-01T00:00:00+00:00",
                        "end": "2026-04-01T00:00:00+00:00",
                    },
                    "evaluation": {
                        "start": "2026-03-01T00:00:00+00:00",
                        "end": "2026-04-01T00:00:00+00:00",
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
                "test_score": 0.68,
                "sample_count": 250,
            },
        },
    )
    feature = _write_json(
        tmp_path / "locked_features.json",
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
            "feature_importances": [
                {"feature": "momentum", "importance": 0.2},
                {"feature": "volume", "importance": 0.19},
                {"feature": "sentiment", "importance": 0.18},
                {"feature": "macro", "importance": 0.17},
            ],
            "feature_stability_score": 0.82,
            "unstable_features": [],
        },
    )
    replay = _write_json(
        tmp_path / "replay.json",
        {"summary": {"clear_hit_rate": 0.72, "clear_evaluated_runs": 8, "quality_blocked_runs": 0}},
    )
    quality = _write_json(tmp_path / "quality.json", {"warnings": [], "leakage_flags": []})
    architecture = _write_json(tmp_path / "architecture.json", {"summary": {"can_write_production_config_now": False, "can_trade": False}})
    domain = _write_json(tmp_path / "domain.json", {"summary": {"can_scale_to_other_domains_now": False, "can_trade": False}})

    payload = PipelineControlMetricArtifactMaterializer(tmp_path / "reports").build(candidate_paths=[model, feature])

    summary = payload["summary"]
    assert summary["materialization_status"] == "materialized_real_metric_artifacts_ready"
    assert summary["materialized_model_evaluation_json"] is True
    assert summary["materialized_feature_stability_report"] is True
    assert summary["compatible_metric_pair_found"] is True
    assert summary["can_run_real_metric_evidence_now"] is True
    assert summary["can_trade"] is False

    next_inputs = payload["next_runner_inputs"]
    assert Path(next_inputs["model_evaluation_json"]).exists()
    assert Path(next_inputs["feature_stability_report"]).exists()
    model_payload = json.loads(Path(next_inputs["model_evaluation_json"]).read_text(encoding="utf-8"))
    assert model_payload["metrics"]["validation_score"] == 0.68
    assert model_payload["evaluated_at"] == "2026-04-01T00:00:00+00:00"
    assert model_payload["materialization_contract"][
        "source_locked_artifact_verified"
    ] is True
    assert len(model_payload["source_artifact"]["sha256"]) == 64

    real_run = PipelineControlRealMetricEvidenceRun(tmp_path / "real_run").build(
        model_evaluation_json=next_inputs["model_evaluation_json"],
        feature_stability_report=next_inputs["feature_stability_report"],
        replay_batch_json=replay,
        data_quality_json=quality,
        architecture_map_json=architecture,
        domain_instance_contract_json=domain,
        save=False,
    )
    assert real_run["summary"]["real_metric_evidence_status"] == "real_metric_evidence_chain_ready"
    assert real_run["summary"]["can_clear_current_real_cautions"] is True
    assert real_run["summary"]["can_trade"] is False


def test_metric_artifact_materializer_rejects_synthetic_ready_shape(tmp_path):
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
    feature = _write_json(
        tmp_path / "feature.json",
        {"feature_importance": {"momentum": 0.2}, "feature_stability_score": 0.8},
    )

    payload = PipelineControlMetricArtifactMaterializer(tmp_path / "reports").build(
        candidate_paths=[model, feature],
        save=False,
    )

    assert payload["summary"]["materialization_status"] == "blocked_missing_locked_metric_artifacts"
    assert payload["summary"]["ready_model_candidate_found"] is False
    assert payload["summary"]["ready_feature_candidate_found"] is False
    assert payload["summary"]["can_trade"] is False


def test_metric_artifact_materializer_blocks_mismatched_lineage_pair(tmp_path):
    model = _write_json(
        tmp_path / "model.json",
        {
            "artifact_class": "locked_model_evaluation",
            "joined_lineage": {
                "ticker": "AMD",
                "model": "random_forest",
                "target_name": "target_up_1d",
                "timeframe": "1d",
                "context_fingerprint": "ctx_a",
                "evaluation_window": {
                    "training": {
                        "start": "2026-03-01T00:00:00+00:00",
                        "end": "2026-04-01T00:00:00+00:00",
                    },
                    "evaluation": {
                        "start": "2026-03-01T00:00:00+00:00",
                        "end": "2026-04-01T00:00:00+00:00",
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
        tmp_path / "feature.json",
        {
            "artifact_class": "locked_feature_stability_report",
            "training_lineage": {
                "ticker": "AMD",
                "model": "random_forest",
                "target_name": "target_up_1d",
                "timeframe": "1d",
                "context_fingerprint": "ctx_b",
            },
            "assembly_contract": {
                "measured_stability_signal_required": True
            },
            "feature_importance": {"momentum": 0.2},
            "feature_stability_score": 0.8,
        },
    )

    payload = PipelineControlMetricArtifactMaterializer(tmp_path / "reports").build(
        candidate_paths=[model, feature],
        save=False,
    )

    assert payload["summary"]["ready_model_candidate_found"] is True
    assert payload["summary"]["ready_feature_candidate_found"] is True
    assert payload["summary"]["compatible_metric_pair_found"] is False
    assert payload["summary"]["materialization_status"] == "blocked_missing_locked_metric_artifacts"
    assert payload["summary"]["can_run_real_metric_evidence_now"] is False


def test_metric_artifact_materializer_saves_markdown_and_cli_runs(tmp_path):
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
                        "start": "2026-03-01T00:00:00+00:00",
                        "end": "2026-04-01T00:00:00+00:00",
                    },
                    "evaluation": {
                        "start": "2026-03-01T00:00:00+00:00",
                        "end": "2026-04-01T00:00:00+00:00",
                    },
                },
            },
            "join_contract": {
                "join_status": "same_window_lineage_proven"
            },
            "metrics": {"max_drawdown": 0.08, "train_score": 0.72, "validation_score": 0.68, "sample_count": 250},
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

    payload = PipelineControlMetricArtifactMaterializer(tmp_path / "reports").build(candidate_paths=[model, feature])
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Pipeline Control Metric Artifact Materializer" in markdown
    assert "Can run real metric evidence now: True" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_pipeline_control_metric_artifact_materializer.py"),
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

    assert "Materialization status: materialized_real_metric_artifacts_ready" in result.stdout
    assert "Can run real metric evidence now: True" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
