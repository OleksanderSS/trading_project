from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data.collectors.yf_collector import YFCollector
from src.data.management.data_manager import DataManager
from src.pipeline.stages.processing.data_handler import ProcessingDataHandler
from src.processing.filters.price_filter import PriceFilter


def test_macro_normalizer_preserves_macro_schema_and_deduplicates():
    handler = ProcessingDataHandler(normalization_manager=None, data_filter=None)
    source = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
            "series": ["DGS10", "DGS10", "VIXCLS"],
            "value": ["4.1", "4.2", "18.0"],
            "realtime_start": ["2025-01-03"] * 3,
        }
    )

    result = handler.clean_and_normalize_macro_data(source)

    assert {"datetime", "series_id", "value", "realtime_start"}.issubset(result.columns)
    assert len(result) == 2
    assert result.loc[result["series_id"].eq("DGS10"), "value"].item() == 4.2
    assert "open" not in result.columns
    assert "ticker" not in result.columns


def test_macro_normalizer_keeps_empty_macro_schema_out_of_price_shape():
    handler = ProcessingDataHandler(normalization_manager=None, data_filter=None)

    result = handler.clean_and_normalize_macro_data(
        pd.DataFrame(columns=["date", "series_id", "value"])
    )

    assert {"datetime", "series_id", "value"}.issubset(result.columns)
    assert "close" not in result.columns
    assert result.empty




def test_data_manager_numeric_fill_never_crosses_ticker_interval_boundaries():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-01 10:00",
                    "2025-01-01 10:15",
                    "2025-01-01 10:00",
                    "2025-01-01 11:00",
                ],
                utc=True,
            ),
            "ticker": ["AAA", "AAA", "BBB", "AAA"],
            "interval": ["15m", "15m", "15m", "60m"],
            "close": [100.0, np.nan, np.nan, np.nan],
        }
    )

    result = DataManager._clean_numeric_data(object(), frame, "market_data_raw")

    assert result.loc[1, "close"] == 100.0
    assert pd.isna(result.loc[2, "close"])
    assert pd.isna(result.loc[3, "close"])


def test_data_manager_numeric_fill_stays_within_macro_series():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-02"]),
            "series_id": ["DGS10", "DGS10", "VIXCLS"],
            "value": [4.1, np.nan, np.nan],
        }
    )

    result = DataManager._clean_numeric_data(object(), frame, "fred_data")

    assert result.loc[1, "value"] == 4.1
    assert pd.isna(result.loc[2, "value"])


def test_price_filter_rejects_cross_ticker_identity_contamination():
    timestamps = pd.date_range("2025-01-01 10:00", periods=8, freq="15min", tz="UTC")
    rows = []
    for ticker in ("AAA", "BBB"):
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "datetime": timestamp,
                    "ticker": ticker,
                    "interval": "15m",
                    "open": 100.0 + index,
                    "high": 101.0 + index,
                    "low": 99.0 + index,
                    "close": 100.5 + index,
                    "volume": 1000.0 + index,
                }
            )
    contaminated = pd.DataFrame(rows)

    filtered, quality = PriceFilter({}).filter_price_data({"15m": contaminated})

    assert "15m" not in filtered
    assert quality["15m"]["status"] == "low_quality"
    assert "cross_ticker_duplicate_ohlcv" in quality["15m"]["hard_failures"]


def test_yf_collector_gate_rejects_mixed_cadence_without_network():
    frame = _valid_price_frame()
    frame.loc[3, "datetime"] = frame.loc[2, "datetime"] + pd.Timedelta(minutes=5)

    issues = YFCollector._validate_collected_price_data(object(), frame)

    assert any(issue.startswith("cadence_mismatch=") for issue in issues)


def test_yf_collector_gate_accepts_clean_saved_shape_without_network():
    issues = YFCollector._validate_collected_price_data(object(), _valid_price_frame())

    assert issues == []


def test_yf_collector_gate_rejects_cross_identity_ohlcv_without_network():
    first = _valid_price_frame()
    second = first.assign(ticker="BBB", interval="60m")

    issues = YFCollector._validate_collected_price_data(
        object(),
        pd.concat([first, second], ignore_index=True),
    )

    assert any(
        issue.startswith("cross_identity_ohlcv_rows=")
        for issue in issues
    )


def test_yf_download_disables_internal_threads(monkeypatch):
    captured = {}
    source = pd.DataFrame(
        {"Close": [100.0]},
        index=pd.DatetimeIndex(["2025-01-01"], name="Date"),
    )

    def fake_download(**kwargs):
        captured.update(kwargs)
        return source

    monkeypatch.setattr("src.data.collectors.yf_collector.yf.download", fake_download)
    monkeypatch.setattr(
        "src.data.collectors.yf_collector._configure_yfinance_cache",
        lambda: None,
    )
    collector = object.__new__(YFCollector)
    collector.logger = logging.getLogger("test_yf_collector")

    result = collector._single_ticker_download_with_retry(
        "AAA",
        "15m",
        datetime(2025, 1, 1),
        datetime(2025, 1, 2),
        retries=1,
        delay=0,
    )

    assert not result.empty
    assert result is not source
    assert captured["threads"] is False


def test_yf_collector_rejects_mismatched_multiindex_ticker():
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["BBB"]],
        names=["Price", "Ticker"],
    )
    source = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 1000.0]],
        columns=columns,
        index=pd.DatetimeIndex(["2025-01-01"], name="Date"),
    )
    collector = object.__new__(YFCollector)
    collector.logger = logging.getLogger("test_yf_collector")

    with pytest.raises(RuntimeError, match="source ticker mismatch"):
        collector._flatten_multiindex_columns(source, "AAA", "15m")


def _valid_price_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 10:00", periods=8, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "datetime": timestamps,
            "ticker": "AAA",
            "interval": "15m",
            "open": np.arange(8) + 100.0,
            "high": np.arange(8) + 101.0,
            "low": np.arange(8) + 99.0,
            "close": np.arange(8) + 100.5,
            "volume": np.arange(8) + 1000.0,
        }
    )
