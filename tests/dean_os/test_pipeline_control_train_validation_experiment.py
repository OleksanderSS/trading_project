from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from dean_os.pipeline_control_train_validation_experiment import (
    PipelineControlTrainValidationExperiment,
    _select_candidate_features,
)


def test_train_only_selector_removes_calendar_duplicates_and_correlation():
    rows = 220
    index = pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC")
    signal = np.sin(np.arange(rows) / 7.0)
    target = (signal > 0).astype(int)
    train = pd.DataFrame(
        {
            "target_intraday_up_15m": target,
            "signal_15m": signal,
            "signal_copy_15m": signal,
            "noise_15m": np.cos(np.arange(rows) / 5.0),
            "day_of_month_15m": index.day,
            "state_day_of_month_15m": index.day,
            "market_context_hour_of_day_15m": index.hour,
            "hour_15m": index.hour,
            "close": 100.0 + np.arange(rows) * 0.01,
            "__forward_return": 0.001,
        },
        index=index,
    )

    selected, details = _select_candidate_features(
        train,
        target_name="target_intraday_up_15m",
        max_features=3,
    )

    assert "day_of_month_15m" not in selected
    assert "state_day_of_month_15m" not in selected
    assert not {"signal_15m", "signal_copy_15m"}.issubset(selected)
    assert details["validation_labels_used_for_selection"] is False
    assert details["test_rows_used"] == 0


def test_train_validation_experiment_requires_no_test_acknowledgement(tmp_path):
    with pytest.raises(ValueError, match="no-test acknowledgement"):
        asyncio.run(
            PipelineControlTrainValidationExperiment(tmp_path / "reports").run(
                batch_json=tmp_path / "missing_batch.json",
                diagnostic_json=tmp_path / "missing_diagnostic.json",
                acknowledge_no_test=False,
            )
        )
