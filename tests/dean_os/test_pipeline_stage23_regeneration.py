from __future__ import annotations

import pandas as pd

from dean_os.pipeline_stage23_regeneration import (
    _load_saved_stage1_market,
    _select_bounded_market_frame,
    _source_checks,
    _build_or_load_stage3_shard_cache,
    _stage3_cache_key,
)


def _market_frame(
    *,
    interval: str = "15m",
    frequency: str = "15min",
) -> pd.DataFrame:
    rows = []
    for ticker, offset in (("AMD", 0.0), ("NVDA", 20.0)):
        timestamps = pd.date_range(
            "2026-06-01T13:30:00Z",
            periods=10,
            freq=frequency,
        )
        for index, timestamp in enumerate(timestamps):
            close = 100.0 + offset + index
            rows.append(
                {
                    "ticker": ticker,
                    "datetime": timestamp,
                    "interval": interval,
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000.0 + offset + index,
                }
            )
    return pd.DataFrame(rows)


def test_stage23_loader_recognizes_legacy_pickle_and_bounds_each_ticker(
    tmp_path,
):
    source = tmp_path / "legacy_stage1.parquet"
    pd.to_pickle({"market_data": _market_frame()}, source)

    market, source_format = _load_saved_stage1_market(source)
    selected = _select_bounded_market_frame(
        market,
        tickers=["AMD", "NVDA"],
        timeframe="15m",
        max_rows_per_ticker=6,
    )
    checks = _source_checks(
        selected,
        tickers=["AMD", "NVDA"],
        timeframe="15m",
    )

    assert source_format == "legacy_pickle_with_parquet_extension"
    assert len(selected) == 12
    assert {item["status"] for item in checks} == {"pass"}


def test_stage23_source_checks_reject_daily_label_on_intraday_rows():
    frame = _market_frame(interval="1d", frequency="15min")

    checks = _source_checks(
        frame,
        tickers=["AMD", "NVDA"],
        timeframe="1d",
    )
    failed = {
        item["code"]
        for item in checks
        if item["status"] == "fail"
    }

    assert failed == {"timeframe_cadence"}


def test_stage23_source_checks_reject_cross_ticker_ohlcv_clone():
    frame = _market_frame()
    amd = frame.loc[frame["ticker"].eq("AMD")].iloc[0]
    nvda_index = frame.index[frame["ticker"].eq("NVDA")][0]
    for column in (
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        frame.loc[nvda_index, column] = amd[column]

    checks = _source_checks(
        frame,
        tickers=["AMD", "NVDA"],
        timeframe="15m",
    )
    failed = {
        item["code"]
        for item in checks
        if item["status"] == "fail"
    }

    assert "cross_ticker_ohlcv_identity" in failed


def test_stage23_stage3_shard_cache_reuses_saved_outputs(tmp_path):
    source = tmp_path / "stage1.parquet"
    source.write_bytes(b"source-bytes")
    frame = _market_frame()

    class Processor:
        def __init__(self):
            self.calls = 0

        def process_enriched_data(self, enriched_data):
            self.calls += 1
            return {
                "features": enriched_data[["ticker", "datetime", "interval", "close"]].copy(),
                "targets": enriched_data[["ticker", "datetime", "interval"]].copy(),
            }

    processor = Processor()
    cache_dir = tmp_path / "cache"
    first = _build_or_load_stage3_shard_cache(
        frame,
        source_sha256="abc123",
        source_path=source,
        tickers=["AMD", "NVDA"],
        timeframe="15m",
        max_rows_per_ticker=6,
        cache_dir=cache_dir,
        feature_processor=processor,
        save=True,
    )
    second = _build_or_load_stage3_shard_cache(
        frame,
        source_sha256="abc123",
        source_path=source,
        tickers=["AMD", "NVDA"],
        timeframe="15m",
        max_rows_per_ticker=6,
        cache_dir=cache_dir,
        feature_processor=processor,
        save=True,
    )

    assert processor.calls == 2
    assert first["features"].equals(second["features"])
    assert first["targets"].equals(second["targets"])
    assert first["cache"]["shard_count"] == 2
    assert all("cache_key" in shard for shard in first["cache"]["shards"])


def test_stage23_cache_key_changes_with_scope():
    key_a = _stage3_cache_key(
        source_sha256="abc",
        ticker="AMD",
        timeframe="15m",
        max_rows_per_ticker=600,
        selected_sha256="one",
    )
    key_b = _stage3_cache_key(
        source_sha256="abc",
        ticker="AMD",
        timeframe="15m",
        max_rows_per_ticker=300,
        selected_sha256="one",
    )

    assert key_a != key_b
