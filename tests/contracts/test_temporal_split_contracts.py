"""
Temporal split / leakage contracts.

These tests encode rules for time-series train/validation/test splitting.
"""

from __future__ import annotations

import pandas as pd


def assert_temporal_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, time_col: str = "timestamp") -> None:
    assert train[time_col].max() < val[time_col].min(), "train must end before validation starts"
    assert val[time_col].max() < test[time_col].min(), "validation must end before test starts"

    train_times = set(train[time_col])
    val_times = set(val[time_col])
    test_times = set(test[time_col])

    assert not train_times & val_times, "train/validation timestamps overlap"
    assert not val_times & test_times, "validation/test timestamps overlap"
    assert not train_times & test_times, "train/test timestamps overlap"


def test_temporal_splits_good_example():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=9, freq="D"), "x": range(9)})
    train, val, test = df.iloc[:3], df.iloc[3:6], df.iloc[6:]
    assert_temporal_splits(train, val, test)


def test_temporal_splits_reject_overlap():
    train = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    val = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"])})
    test = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-04"])})

    try:
        assert_temporal_splits(train, val, test)
    except AssertionError:
        return

    raise AssertionError("Overlapping temporal split was not rejected")
