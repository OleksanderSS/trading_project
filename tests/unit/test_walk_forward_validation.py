import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.modeling.walk_forward_validation import (
    PipelineWalkForwardValidationEvaluator,
    WalkForwardValidationConfig,
    build_purged_expanding_folds,
)


def test_build_purged_expanding_folds_uses_latest_windows_and_real_gap():
    config = WalkForwardValidationConfig(
        min_train_rows=240,
        validation_rows=80,
        step_rows=80,
        purge_rows=4,
        max_folds=4,
    )

    folds = build_purged_expanding_folds(800, config=config)

    assert len(folds) == 4
    assert folds[0] == {
        "train_start": 0,
        "train_end": 476,
        "validation_start": 480,
        "validation_end": 560,
    }
    assert folds[-1]["validation_end"] == 800
    assert all(
        fold["validation_start"] - fold["train_end"] == 4
        for fold in folds
    )


def test_walk_forward_evaluator_never_uses_targets_as_features_or_test_rows():
    rows = 800
    index = pd.date_range(
        "2025-01-01",
        periods=rows,
        freq="15min",
        tz="UTC",
    )
    signal = np.sin(np.arange(rows) / 9.0)
    frame = pd.DataFrame(
        {
            "datetime": index,
            "ticker": ["NVDA"] * rows,
            "interval": ["15m"] * rows,
            "signal_15m": signal,
            "signal_copy_15m": signal,
            "slow_feature_15m": np.cos(np.arange(rows) / 31.0),
            "ctx_60m_regime": np.sin(np.arange(rows) / 55.0),
            "all_missing": [float("nan")] * rows,
            "target_intraday_up_15m": (signal > 0).astype(int),
            "target_hourly_up_1h": (signal < 0).astype(int),
        }
    )
    context_report = {
        "status": "causal_timeframe_context_ready",
        "join_direction": "backward",
        "allow_future_context": False,
        "summary": {
            "future_context_violations": 0,
            "row_identity_preserved": True,
        },
        "base_contexts": [
            {
                "base_timeframe": "15m",
                "context_joins": [
                    {
                        "context_timeframe": "60m",
                        "future_context_violations": 0,
                    }
                ],
            }
        ],
    }
    evaluator = PipelineWalkForwardValidationEvaluator(
        WalkForwardValidationConfig(
            min_train_rows=240,
            validation_rows=80,
            step_rows=80,
            purge_rows=4,
            max_folds=4,
            max_features=4,
        )
    )

    candidate = evaluator.evaluate(
        frame,
        ticker="NVDA",
        timeframe="15m",
        target_name="target_intraday_up_15m",
        timeframe_context_report=context_report,
        source_lineage={"development_15m": {"sha256": "abc"}},
    )

    assert candidate["artifact_class"] == (
        "pipeline_control_walk_forward_validation_candidate"
    )
    assert candidate["metrics"]["fold_count"] == 4
    assert candidate["metrics"]["test_rows_loaded"] == 0
    assert candidate["test_contract"]["past_evaluation_rows_loaded"] == 0
    assert candidate["test_contract"]["eligible_as_locked_test_evidence"] is False
    assert candidate["feature_selection"]["validation_labels_used"] is False
    assert candidate["feature_selection"]["feature_set_frozen_across_folds"] is True
    assert all(
        not feature.startswith("target_")
        and "_target_" not in feature
        for feature in candidate["selected_features"]
    )
    assert not {
        "signal_15m",
        "signal_copy_15m",
    }.issubset(candidate["selected_features"])
    assert all(
        fold["temporal_contract"]["train_precedes_validation"]
        and fold["temporal_contract"]["purge_rows"] == 4
        for fold in candidate["folds"]
    )
    assert candidate["timeframe_context_lineage"]["join_direction"] == "backward"
    assert len(candidate["context_fingerprint"]) == 64


def test_walk_forward_evaluator_rejects_insufficient_rows():
    rows = 200
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2025-01-01",
                periods=rows,
                freq="15min",
                tz="UTC",
            ),
            "ticker": ["NVDA"] * rows,
            "interval": ["15m"] * rows,
            "signal": np.arange(rows),
            "target_intraday_up_15m": [0, 1] * (rows // 2),
        }
    )
    evaluator = PipelineWalkForwardValidationEvaluator(
        WalkForwardValidationConfig(
            min_train_rows=180,
            validation_rows=60,
        )
    )

    with pytest.raises(ValueError, match="Insufficient rows"):
        evaluator.evaluate(
            frame,
            ticker="NVDA",
            timeframe="15m",
            target_name="target_intraday_up_15m",
        )
