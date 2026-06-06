"""
Data schema contracts.

These tests validate generic OHLCV/trading dataframe assumptions.
They do not require your project imports; they can be reused by project-specific fixtures later.
"""

from __future__ import annotations

import pandas as pd
import pytest


REQUIRED_COLUMNS = {"ticker", "timestamp", "open", "high", "low", "close", "volume"}


def assert_market_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing required market columns: {sorted(missing)}"

    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), "timestamp must be datetime"
    assert df["ticker"].notna().all(), "ticker must not contain nulls"

    for col in ["open", "high", "low", "close", "volume"]:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"

    duplicates = df.duplicated(["ticker", "timestamp"]).sum()
    assert duplicates == 0, "Duplicate ticker+timestamp rows are not allowed"

    sorted_df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    assert df.reset_index(drop=True).equals(sorted_df), "Market dataframe must be sorted by ticker,timestamp"

    assert (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all(), "high must be >= open/close/low"
    assert (df["low"] <= df[["open", "close", "high"]].min(axis=1)).all(), "low must be <= open/close/high"


def test_market_schema_good_example():
    df = pd.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
        "open": [10, 11, 20, 21],
        "high": [12, 12, 22, 22],
        "low": [9, 10, 19, 20],
        "close": [11, 11.5, 21, 21.5],
        "volume": [100, 110, 200, 210],
    })
    assert_market_schema(df)


def test_market_schema_rejects_cross_ticker_duplicate_timestamp_unsorted():
    df = pd.DataFrame({
        "ticker": ["A", "B", "A"],
        "timestamp": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01"]),
        "open": [10, 20, 11],
        "high": [12, 22, 12],
        "low": [9, 19, 10],
        "close": [11, 21, 11.5],
        "volume": [100, 200, 110],
    })
    with pytest.raises(AssertionError):
        assert_market_schema(df)
