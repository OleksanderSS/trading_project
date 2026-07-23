from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dean_os.pipeline_control.pipeline_control_evidence_inventory import PipelineControlEvidenceInventory
from dean_os.pipeline_control.pipeline_control_locked_evaluation_assembler import PipelineControlLockedEvaluationAssembler
from dean_os.pipeline_control.pipeline_control_metric_artifact_materializer import PipelineControlMetricArtifactMaterializer
from src.pipeline.stages.modeling.pipeline_control_artifacts import build_model_evaluation_candidate
from src.pipeline.stages.evaluation.pipeline_control_artifacts import (
    build_evaluation_metric_candidate,
    write_evaluation_metric_artifact_candidate,
)


def test_stage_7_evaluation_candidate_is_supporting_drawdown_evidence_not_locked_model_eval(tmp_path):
    signals = pd.DataFrame(
        {
            "ticker": ["AMD", "AMD", "NVDA"],
            "selected_primary_model": ["random_forest", "random_forest", "lightgbm"],
            "signal": ["BUY", "HOLD", "SELL"],
            "price": [100.0, 101.0, 200.0],
        },
        index=pd.date_range("2026-01-01", periods=3),
    )
    portfolio = pd.DataFrame(
        {"total_value": [100000.0, 104000.0, 98000.0]},
        index=pd.date_range("2026-01-01", periods=3),
    )
    candidate = build_evaluation_metric_candidate(
        financial_metrics={"max_drawdown": -0.0577, "total_return": -0.02, "sharpe_ratio": 0.4},
        backtest_results={"performance": {"win_rate": 0.5}},
        evaluation_summary={"metrics": {"max_drawdown": -0.0577}},
        signals_df=signals,
        portfolio_history=portfolio,
        summary_path=tmp_path / "summary.json",
    )
    paths = write_evaluation_metric_artifact_candidate(
        output_dir=tmp_path,
        candidate=candidate,
        context_key="summary_20260626_001",
    )

    saved = json.loads(Path(paths["evaluation_metric_candidate"]).read_text(encoding="utf-8"))
    assert saved["metrics"]["max_drawdown"] == -0.0577
    assert saved["same_window_join_status"] == "requires_matching_training_candidate_before_locked_model_evaluation"
    assert saved["lineage"]["tickers"] == ["AMD", "NVDA"]
    assert saved["lineage"]["single_context_join_candidate"] is False
    assert "ticker" not in saved
    assert "target_name" not in saved

    inventory = PipelineControlEvidenceInventory(tmp_path / "reports").build(
        candidate_paths=[paths["manifest"]],
        save=False,
    )

    assert inventory["summary"]["ready_model_evaluation_candidate_count"] == 0
    assert inventory["summary"]["ready_feature_stability_candidate_count"] == 0
    assert inventory["summary"]["supporting_artifact_count"] == 1
    assert inventory["summary"]["can_clear_current_real_cautions"] is False
    assert inventory["candidate_artifacts"][0]["classification"] == "supporting_backtest_or_portfolio_performance"
    assert inventory["candidate_artifacts"][0]["recognized_fields"]["model_metrics"]["max_drawdown"] is True
    assert "train_score" in inventory["real_metric_evidence_gap"]["missing_for_model_evaluation"]


def test_stage_7_single_context_lineage_can_join_generated_stage4_candidate(tmp_path):
    index = pd.date_range("2026-01-01", periods=3)
    signals = pd.DataFrame(
        {
            "ticker": ["AMD", "AMD", "AMD"],
            "selected_primary_model": ["random_forest", "random_forest", "random_forest"],
            "model_context_id": ["AMD_target_up_1d_random_forest"] * 3,
            "context_fingerprint": ["ctx_semis_2026_01"] * 3,
            "signal": ["BUY", "HOLD", "SELL"],
            "price": [100.0, 101.0, 99.0],
        },
        index=index,
    )
    portfolio = pd.DataFrame({"total_value": [100000.0, 104000.0, 98000.0]}, index=index)
    evaluation_candidate = build_evaluation_metric_candidate(
        financial_metrics={"max_drawdown": 0.0577, "total_return": -0.02, "sharpe_ratio": 0.4},
        backtest_results={"performance": {"win_rate": 0.5}},
        evaluation_summary={"metrics": {"max_drawdown": 0.0577}},
        signals_df=signals,
        portfolio_history=portfolio,
        summary_path=tmp_path / "summary.json",
    )
    training_candidate = build_model_evaluation_candidate(
        ticker="AMD",
        target_name="target_up_1d",
        model_type="random_forest",
        timeframe="1d",
        context_fingerprint="ctx_semis_2026_01",
        market_regime="neutral",
        volatility_regime="normal",
        train_metrics={"accuracy": 0.75, "score": 0.75},
        validation_metrics={"accuracy": 0.62, "score": 0.62},
        train_sample_count=120,
        validation_sample_count=3,
        evaluation_window=evaluation_candidate["evaluation_window"],
    )
    training_path = _write_json(tmp_path / "training_candidate.json", training_candidate)
    evaluation_path = _write_json(tmp_path / "evaluation_candidate.json", evaluation_candidate)

    payload = PipelineControlLockedEvaluationAssembler(tmp_path / "reports").build(
        training_candidate_json=training_path,
        evaluation_candidate_json=evaluation_path,
    )

    assert evaluation_candidate["ticker"] == "AMD"
    assert evaluation_candidate["target_name"] == "target_up_1d"
    assert evaluation_candidate["model_type"] == "random_forest"
    assert evaluation_candidate["timeframe"] == "1d"
    assert evaluation_candidate["context_fingerprint"] == "ctx_semis_2026_01"
    assert evaluation_candidate["lineage"]["single_context_join_candidate"] is True
    assert payload["summary"]["assembly_status"] == "locked_model_evaluation_assembled"
    assert payload["summary"]["same_window_lineage_proven"] is True
    assert payload["summary"]["can_trade"] is False


def test_materializer_does_not_join_evaluation_drawdown_without_matching_training_candidate(tmp_path):
    candidate = build_evaluation_metric_candidate(
        financial_metrics={"max_drawdown": 0.08, "total_return": 0.03, "sharpe_ratio": 1.1},
        backtest_results={"performance": {"win_rate": 0.55}},
        evaluation_summary={"metrics": {"max_drawdown": 0.08}},
        signals_df=pd.DataFrame({"ticker": ["AMD"], "price": [100.0], "signal": ["BUY"]}),
        portfolio_history=pd.DataFrame({"total_value": [100000.0, 108000.0]}),
        summary_path=tmp_path / "summary.json",
    )
    paths = write_evaluation_metric_artifact_candidate(
        output_dir=tmp_path,
        candidate=candidate,
        context_key="summary_20260626_002",
    )

    payload = PipelineControlMetricArtifactMaterializer(tmp_path / "reports").build(
        candidate_paths=[paths["manifest"]],
        save=False,
    )

    assert payload["summary"]["materialization_status"] == "blocked_missing_locked_metric_artifacts"
    assert payload["summary"]["ready_model_candidate_found"] is False
    assert payload["summary"]["materialized_model_evaluation_json"] is False
    assert payload["summary"]["can_run_real_metric_evidence_now"] is False
    assert payload["summary"]["can_trade"] is False


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
