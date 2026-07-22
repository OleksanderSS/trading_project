from __future__ import annotations

import json
from pathlib import Path

from dean_os.pipeline_control_evidence_inventory import PipelineControlEvidenceInventory
from dean_os.pipeline_control_metric_artifact_materializer import PipelineControlMetricArtifactMaterializer
from src.pipeline.stages.modeling.pipeline_control_artifacts import (
    build_feature_distribution_stability_analysis,
    build_feature_stability_candidate,
    build_model_evaluation_candidate,
    build_split_evaluation_window,
    extract_native_feature_importance,
    write_pipeline_control_metric_artifact_candidates,
)


class _NativeModel:
    feature_importances_ = [0.2, 0.3, 0.5]


class _WrappedModel:
    model = _NativeModel()


class _IndexedFrame:
    index = ["2026-01-05", "2026-01-06", "2026-01-07"]


class _FrameWithoutIndex:
    pass


def test_pipeline_training_candidate_writes_partial_evidence_without_fake_drawdown_or_stability(tmp_path):
    importances = extract_native_feature_importance(_WrappedModel(), ["open", "close", "volume"])
    model_candidate = build_model_evaluation_candidate(
        ticker="AMD",
        target_name="target_up_1d",
        model_type="random_forest",
        timeframe="1d",
        context_fingerprint="ctx-real",
        market_regime="neutral",
        volatility_regime="normal",
        train_metrics={"accuracy": 0.75, "score": 0.75},
        validation_metrics={"accuracy": 0.62, "score": 0.62},
        train_sample_count=120,
        validation_sample_count=30,
    )
    feature_candidate = build_feature_stability_candidate(
        ticker="AMD",
        target_name="target_up_1d",
        model_type="random_forest",
        timeframe="1d",
        context_fingerprint="ctx-real",
        market_regime="neutral",
        volatility_regime="normal",
        feature_importance=importances,
    )

    paths = write_pipeline_control_metric_artifact_candidates(
        batch_dir=tmp_path,
        context_key="AMD_target_up_1d_random_forest",
        model_evaluation=model_candidate,
        feature_stability=feature_candidate,
    )

    feature_payload = json.loads(Path(paths["feature_stability_report"]).read_text(encoding="utf-8"))
    assert "unstable_features" not in feature_payload
    assert feature_payload["stability_signal_status"] == "not_measured"

    payload = PipelineControlEvidenceInventory(tmp_path / "reports").build(
        candidate_paths=[paths["manifest"]],
        save=False,
    )

    assert payload["summary"]["ready_model_evaluation_candidate_count"] == 0
    assert payload["summary"]["ready_feature_stability_candidate_count"] == 0
    assert payload["summary"]["can_clear_current_real_cautions"] is False
    assert payload["real_metric_evidence_gap"]["missing_for_model_evaluation"] == ["max_drawdown"]
    assert payload["real_metric_evidence_gap"]["missing_for_feature_stability"] == ["stability_signal"]


def test_training_model_candidate_records_real_split_window_without_drawdown_synthesis():
    evaluation_window = build_split_evaluation_window(_IndexedFrame())

    model_candidate = build_model_evaluation_candidate(
        ticker="AMD",
        target_name="target_up_1d",
        model_type="random_forest",
        timeframe="1d",
        context_fingerprint="ctx-real",
        market_regime="neutral",
        volatility_regime="normal",
        train_metrics={"accuracy": 0.75, "score": 0.75},
        validation_metrics={"accuracy": 0.62, "score": 0.62},
        train_sample_count=120,
        validation_sample_count=3,
        evaluation_window=evaluation_window,
    )

    assert evaluation_window == {
        "start": "2026-01-05",
        "end": "2026-01-07",
        "sample_count": 3,
        "source": "validation_feature_index",
    }
    assert model_candidate["evaluation_window"] == evaluation_window
    assert model_candidate["same_window_contract"]["evaluation_window_source"] == "validation_feature_index"
    assert model_candidate["same_window_contract"]["max_drawdown_source"] == "not_supplied_by_training_stage"
    assert model_candidate["contract_status"] == "partial_model_evaluation_candidate"
    assert model_candidate["missing_for_locked_model_evaluation"] == ["max_drawdown"]


def test_training_model_candidate_does_not_invent_window_when_split_index_is_missing():
    evaluation_window = build_split_evaluation_window(_FrameWithoutIndex())

    model_candidate = build_model_evaluation_candidate(
        ticker="AMD",
        target_name="target_up_1d",
        model_type="random_forest",
        timeframe="1d",
        context_fingerprint="ctx-real",
        market_regime="neutral",
        volatility_regime="normal",
        train_metrics={"accuracy": 0.75, "score": 0.75},
        validation_metrics={"accuracy": 0.62, "score": 0.62},
        train_sample_count=120,
        validation_sample_count=3,
        evaluation_window=evaluation_window,
    )

    assert evaluation_window is None
    assert "evaluation_window" not in model_candidate
    assert model_candidate["same_window_contract"]["evaluation_window_source"] == "not_supplied_by_training_stage"


def test_distribution_stability_analysis_promotes_candidate_only_when_split_signal_is_measured():
    stability_analysis = build_feature_distribution_stability_analysis(
        train_features={
            "macro_pressure": [10.0, 11.0, 9.0, 10.0],
            "news_pressure": [1.0, 1.0, 1.0, 1.0],
        },
        validation_features={
            "macro_pressure": [10.1, 10.9, 9.2, 9.8],
            "news_pressure": [4.0, 4.0, 4.0, 4.0],
        },
        feature_names=["macro_pressure", "news_pressure"],
        drift_threshold=1.0,
    )

    feature_candidate = build_feature_stability_candidate(
        ticker="AMD",
        target_name="target_up_1d",
        model_type="random_forest",
        timeframe="1d",
        context_fingerprint="ctx-real",
        market_regime="neutral",
        volatility_regime="normal",
        feature_importance={"macro_pressure": 0.7, "news_pressure": 0.3},
        stability_analysis=stability_analysis,
    )

    assert stability_analysis["measurement_status"] == "measured"
    assert stability_analysis["measured_feature_count"] == 2
    assert stability_analysis["unstable_features"] == ["news_pressure"]
    assert 0.0 <= stability_analysis["feature_stability_score"] < 1.0
    assert feature_candidate["contract_status"] == "ready_feature_stability_candidate"
    assert feature_candidate["stability_signal_status"] == "measured"
    assert feature_candidate["missing_for_locked_feature_stability"] == []
    assert feature_candidate["stability_signal"]["source"] == "train_validation_distribution_drift_v1"
    assert feature_candidate["feature_stability_analysis"]["measurement_status"] == "measured"


def test_distribution_stability_analysis_does_not_synthesize_signal_with_incomplete_split_coverage():
    stability_analysis = build_feature_distribution_stability_analysis(
        train_features={"macro_pressure": [10.0], "news_pressure": [1.0, 1.1]},
        validation_features={"macro_pressure": [10.0, 10.1], "news_pressure": [1.0, 1.2]},
        feature_names=["macro_pressure", "news_pressure"],
    )

    feature_candidate = build_feature_stability_candidate(
        ticker="AMD",
        target_name="target_up_1d",
        model_type="random_forest",
        timeframe="1d",
        context_fingerprint="ctx-real",
        market_regime="neutral",
        volatility_regime="normal",
        feature_importance={"macro_pressure": 0.7, "news_pressure": 0.3},
        stability_analysis=stability_analysis,
    )

    assert stability_analysis["measurement_status"] == "not_measured_incomplete_feature_coverage"
    assert "feature_stability_score" not in stability_analysis
    assert feature_candidate["contract_status"] == "partial_feature_stability_candidate"
    assert feature_candidate["stability_signal_status"] == "not_measured"
    assert feature_candidate["missing_for_locked_feature_stability"] == ["stability_signal"]


def test_metric_materializer_expands_pipeline_manifest_when_locked_pair_exists(tmp_path):
    model_candidate = build_model_evaluation_candidate(
        ticker="AMD",
        target_name="target_up_1d",
        model_type="random_forest",
        timeframe="1d",
        context_fingerprint="ctx-real",
        market_regime="neutral",
        volatility_regime="normal",
        train_metrics={"accuracy": 0.75, "score": 0.75},
        validation_metrics={"accuracy": 0.62, "score": 0.62},
        train_sample_count=120,
        validation_sample_count=30,
        max_drawdown=0.08,
    )
    feature_candidate = build_feature_stability_candidate(
        ticker="AMD",
        target_name="target_up_1d",
        model_type="random_forest",
        timeframe="1d",
        context_fingerprint="ctx-real",
        market_regime="neutral",
        volatility_regime="normal",
        feature_importance={"open": 0.2, "close": 0.3, "volume": 0.5},
        stability_analysis={"feature_stability_score": 0.82, "unstable_features": []},
    )
    paths = write_pipeline_control_metric_artifact_candidates(
        batch_dir=tmp_path,
        context_key="AMD_target_up_1d_random_forest",
        model_evaluation=model_candidate,
        feature_stability=feature_candidate,
    )

    payload = PipelineControlMetricArtifactMaterializer(tmp_path / "reports").build(
        candidate_paths=[paths["manifest"]],
    )

    assert payload["summary"]["materialization_status"] == "materialized_real_metric_artifacts_ready"
    assert payload["summary"]["can_run_real_metric_evidence_now"] is True
    assert payload["summary"]["can_trade"] is False
    assert Path(payload["next_runner_inputs"]["model_evaluation_json"]).exists()
    assert Path(payload["next_runner_inputs"]["feature_stability_report"]).exists()
