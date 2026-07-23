from __future__ import annotations

import numpy as np
import pandas as pd

from dean_os.pipeline_control.pipeline_control_feature_causality_audit import (
    compare_feature_prefix_invariance,
)


def test_feature_causality_comparison_detects_prefix_changes_and_ignores_targets():
    timestamps = pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC")
    prefix = pd.DataFrame(
        {
            "datetime": timestamps[:8],
            "stable_15m": np.arange(8, dtype=float),
            "future_sensitive_15m": np.arange(8, dtype=float),
            "target_intraday_up_15m": [0, 1] * 4,
            "close": 100.0 + np.arange(8),
        }
    )
    full = pd.DataFrame(
        {
            "datetime": timestamps,
            "stable_15m": np.arange(10, dtype=float),
            "future_sensitive_15m": np.arange(10, dtype=float) + 1.0,
            "target_intraday_up_15m": [0, 1] * 5,
            "close": 100.0 + np.arange(10),
        }
    )

    result = compare_feature_prefix_invariance(
        full,
        prefix,
        start=timestamps[0],
        end=timestamps[7],
        mismatch_ratio_limit=0.01,
    )

    assert result["compared_row_count"] == 8
    assert result["datetime_sources"] == {"full": "datetime", "prefix": "datetime"}
    assert result["service_columns_invariant"] is True
    assert result["noncausal_feature_count"] == 1
    assert result["noncausal_features"][0]["feature"] == "future_sensitive_15m"
    compared = {item["feature"] for item in result["highest_difference_features"]}
    assert "target_intraday_up_15m" not in compared
    assert "close" not in compared


def test_feature_causality_comparison_ignores_float_roundoff_noise():
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    prefix = pd.DataFrame(
        {
            "datetime": timestamps,
            "rolling_kurt_15m": np.linspace(-1.0, 1.0, 8),
        }
    )
    full = prefix.copy()
    full["rolling_kurt_15m"] += 3e-8

    result = compare_feature_prefix_invariance(
        full,
        prefix,
        start=timestamps[0],
        end=timestamps[-1],
        mismatch_ratio_limit=0.01,
    )

    assert result["noncausal_feature_count"] == 0
