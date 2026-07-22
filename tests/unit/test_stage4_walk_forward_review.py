import numpy as np
import pandas as pd


def test_stage4_walk_forward_review_bypasses_normal_model_promotion():
    from src.pipeline.stages.stage_4_modeling import ModelingStage

    rows = 480
    signal = np.sin(np.arange(rows) / 8.0)
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
            "signal": signal,
            "target_intraday_up_15m": (signal > 0).astype(int),
        }
    )
    stage = object.__new__(ModelingStage)

    result = stage._run_walk_forward_review_only(
        frame,
        acknowledge_no_test=True,
        target_column="target_intraday_up_15m",
        min_train_rows=200,
        validation_rows=60,
        step_rows=60,
        purge_rows=4,
        max_folds=3,
        max_features=3,
        timeframe_context_report={
            "status": "causal_timeframe_context_ready",
            "join_direction": "backward",
        },
    )

    assert result["status"] == "walk_forward_review_only_complete"
    assert result["models_metadata"] == {}
    assert result["can_promote_model"] is False
    assert result["can_trade"] is False
    candidate = result["walk_forward_validation_candidates"][
        "NVDA_15m_target_intraday_up_15m"
    ]
    assert candidate["metrics"]["fold_count"] == 3
    assert candidate["test_contract"]["test_rows_loaded"] == 0
