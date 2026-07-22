from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.pipeline_metric_input_readiness_gate import PipelineMetricInputReadinessGate


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _clean_inputs(tmp_path: Path) -> dict[str, Path]:
    model_path = _write_json(
        tmp_path / "model.json",
        {
            "metrics": {
                "total_return": 0.12,
                "pnl": 1200.0,
                "sharpe": 1.1,
                "max_drawdown": 0.08,
                "train_score": 0.72,
                "validation_score": 0.68,
                "sample_count": 250,
            }
        },
    )
    replay_path = _write_json(
        tmp_path / "replay.json",
        {"summary": {"clear_hit_rate": 0.72, "clear_evaluated_runs": 8, "quality_blocked_runs": 0}},
    )
    feature_path = _write_json(
        tmp_path / "features.json",
        {
            "feature_importance": {"momentum": 0.2, "volume": 0.19, "sentiment": 0.18, "macro": 0.17},
            "feature_stability_score": 0.82,
            "unstable_features": [],
        },
    )
    quality_path = _write_json(tmp_path / "quality.json", {"warnings": [], "leakage_flags": []})
    return {
        "model_performance_path": model_path,
        "replay_batch_path": replay_path,
        "feature_report_path": feature_path,
        "data_quality_path": quality_path,
    }


def test_pipeline_metric_input_readiness_reports_clean_inputs(tmp_path):
    payload = PipelineMetricInputReadinessGate(output_dir=tmp_path / "reports").build(
        save=False,
        **_clean_inputs(tmp_path),
    )

    assert payload["summary"]["readiness_status"] == "metric_inputs_ready"
    assert payload["summary"]["can_refresh_pipeline_control_surface_now"] is True
    assert payload["summary"]["can_propose_reviewed_tuning_after_surface_and_manual_review"] is True
    assert payload["summary"]["can_run_autonomous_tuning_now"] is False
    assert payload["summary"]["can_write_production_config"] is False
    assert payload["summary"]["can_trade"] is False
    assert {axis["status"] for axis in payload["metric_plane_readiness"]} == {"clear"}


def test_pipeline_metric_input_readiness_blocks_leakage_and_dirty_replay(tmp_path):
    model_path = _write_json(tmp_path / "model.json", {"metrics": {"total_return": 0.11}})
    replay_path = _write_json(
        tmp_path / "replay.json",
        {"summary": {"clear_hit_rate": 0.75, "clear_evaluated_runs": 12, "quality_blocked_runs": 2}},
    )
    quality_path = _write_json(tmp_path / "quality.json", {"warnings": [], "leakage_flags": ["future target"]})

    payload = PipelineMetricInputReadinessGate(output_dir=tmp_path / "reports").build(
        model_performance_path=model_path,
        replay_batch_path=replay_path,
        feature_report_path=None,
        data_quality_path=quality_path,
        save=False,
    )

    assert payload["summary"]["readiness_status"] == "blocked_metric_inputs"
    assert payload["summary"]["can_refresh_pipeline_control_surface_now"] is True
    assert payload["summary"]["can_propose_reviewed_tuning_after_surface_and_manual_review"] is False
    assert {"data_quality", "replay_repeatability"}.issubset(payload["summary"]["blocked_metric_planes"])
    assert {"risk", "validation", "feature_stability"}.issubset(payload["summary"]["caution_metric_planes"])
    assert payload["summary"]["missing_input_count"] == 1


def test_pipeline_metric_input_readiness_allows_surface_refresh_with_feature_caution(tmp_path):
    inputs = _clean_inputs(tmp_path)
    inputs["feature_report_path"] = None

    payload = PipelineMetricInputReadinessGate(output_dir=tmp_path / "reports").build(save=False, **inputs)

    assert payload["summary"]["readiness_status"] == "metric_inputs_ready_with_cautions"
    assert payload["summary"]["can_refresh_pipeline_control_surface_now"] is True
    assert payload["summary"]["can_propose_reviewed_tuning_after_surface_and_manual_review"] is True
    assert payload["summary"]["caution_metric_planes"] == ["feature_stability"]


def test_pipeline_metric_input_readiness_saves_markdown_and_cli_runs(tmp_path):
    paths = _clean_inputs(tmp_path)
    output_dir = tmp_path / "cli_reports"
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_pipeline_metric_input_readiness_gate.py"),
            "--model-performance",
            str(paths["model_performance_path"]),
            "--replay-batch",
            str(paths["replay_batch_path"]),
            "--feature-report",
            str(paths["feature_report_path"]),
            "--data-quality",
            str(paths["data_quality_path"]),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Readiness status: metric_inputs_ready" in result.stdout
    assert (output_dir / "latest.json").exists()
    assert "Pipeline Metric Input Readiness Gate" in (output_dir / "latest.md").read_text(encoding="utf-8")
