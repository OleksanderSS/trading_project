from __future__ import annotations

import json

from dean_os.pipeline_control_surface import PipelineControlSurface


def test_pipeline_control_surface_allows_reviewed_experiment_when_axes_clear(tmp_path):
    model_path = tmp_path / "model.json"
    replay_path = tmp_path / "replay.json"
    feature_path = tmp_path / "features.json"
    quality_path = tmp_path / "quality.json"
    model_path.write_text(
        json.dumps(
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
            }
        ),
        encoding="utf-8",
    )
    replay_path.write_text(
        json.dumps({"summary": {"clear_hit_rate": 0.72, "clear_evaluated_runs": 8, "quality_blocked_runs": 0}}),
        encoding="utf-8",
    )
    feature_path.write_text(
        json.dumps(
            {
                "feature_importance": {"momentum": 0.2, "volume": 0.19, "sentiment": 0.18, "macro": 0.17},
                "feature_stability_score": 0.82,
                "unstable_features": [],
            }
        ),
        encoding="utf-8",
    )
    quality_path.write_text(json.dumps({"warnings": [], "leakage_flags": []}), encoding="utf-8")

    payload = PipelineControlSurface(output_dir=tmp_path / "reports").run(
        model_performance_path=model_path,
        replay_batch_path=replay_path,
        feature_report_path=feature_path,
        data_quality_path=quality_path,
    )

    assert payload["surface"]["status"] == "clear"
    assert payload["surface"]["feasible"] is True
    assert payload["proposal_gate"]["can_propose_tuning"] is True
    assert payload["surface"]["allowed_variation"]["production_write_allowed"] is False


def test_pipeline_control_surface_blocks_when_axes_violate_bounds(tmp_path):
    model_path = tmp_path / "model.json"
    replay_path = tmp_path / "replay.json"
    feature_path = tmp_path / "features.json"
    quality_path = tmp_path / "quality.json"
    model_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "total_return": -0.04,
                    "sharpe": -0.2,
                    "max_drawdown": 0.42,
                    "train_score": 0.9,
                    "validation_score": 0.54,
                    "sample_count": 40,
                }
            }
        ),
        encoding="utf-8",
    )
    replay_path.write_text(
        json.dumps({"summary": {"clear_hit_rate": 0.5, "clear_evaluated_runs": 4, "quality_blocked_runs": 2}}),
        encoding="utf-8",
    )
    feature_path.write_text(
        json.dumps({"feature_importance": {"one_feature": 0.9, "other": 0.1}, "unstable_features": ["one_feature"]}),
        encoding="utf-8",
    )
    quality_path.write_text(json.dumps({"warnings": ["stale prices"], "leakage_flags": ["future column"]}), encoding="utf-8")

    payload = PipelineControlSurface(output_dir=tmp_path / "reports").run(
        model_performance_path=model_path,
        replay_batch_path=replay_path,
        feature_report_path=feature_path,
        data_quality_path=quality_path,
    )

    assert payload["surface"]["status"] == "blocked"
    assert payload["surface"]["feasible"] is False
    assert payload["proposal_gate"]["can_propose_tuning"] is False
    assert payload["surface"]["allowed_variation"]["max_trials"] == 0
    blocked_axes = {axis["name"] for axis in payload["surface"]["axes"] if axis["status"] == "blocked"}
    assert {"profitability", "risk", "validation", "feature_stability", "data_quality", "replay_repeatability"}.issubset(blocked_axes)
