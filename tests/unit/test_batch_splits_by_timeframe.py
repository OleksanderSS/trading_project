"""One file per timeframe, because 69% of the combined one is empty.

The combined frame carries every timeframe's columns on every row: a daily bar
holds NaN in each 15-minute column and the other way round. Measured on the v9
batch of 2026-08-24, 154,069 daily rows carry 1,836 unused columns -- 283
million empty cells, and that is only the daily frame.

The saving is in MEMORY, not on disk. Parquet compresses an all-null column to
almost nothing, so the files together weigh 198 MiB against the original's 200.
Loaded, the daily slice is 0.27 GiB against the combined frame's 2.25 -- 8.3
times smaller -- and reading the combined one costs 4.85 GiB of resident memory
because pyarrow materialises before it converts.

At 110 tickers that is the difference between roughly 24 GiB and 3, which is
why this is the change that allows a wider universe rather than a tidier file.

The splitter itself never holds the union, and that is deliberate: one that did
would fix the storage and leave the memory exactly where it was -- and memory
is what stops 110 tickers, because stage 3 would die before writing anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.batch_timeframe_split import (
    columns_with_data,
    split_batch,
    timeframe_slices,
)


@pytest.fixture
def batch(tmp_path):
    """A miniature of the real shape: shared columns plus per-timeframe ones."""
    rows = []
    for timeframe, count in (("1d", 6), ("15m", 4)):
        for index in range(count):
            rows.append({
                "datetime": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index),
                "ticker": "AAPL",
                "interval": timeframe,
                "close": 100.0 + index,
                "only_daily_1d": index * 2.0 if timeframe == "1d" else np.nan,
                "only_intraday_15m": index * 3.0 if timeframe == "15m" else np.nan,
                "empty_everywhere": np.nan,
            })
    features = pd.DataFrame(rows)
    features.to_parquet(tmp_path / "features.parquet", index=False)

    targets = features[["datetime", "ticker", "interval"]].copy()
    targets["target_return_1d"] = np.where(
        features["interval"].eq("1d"), 0.01, np.nan
    )
    targets.to_parquet(tmp_path / "targets.parquet", index=False)
    return tmp_path


def test_each_slice_keeps_only_the_columns_it_uses(batch):
    report = split_batch(batch)

    daily = pd.read_parquet(batch / "features_1d.parquet")
    assert "only_daily_1d" in daily.columns
    assert "only_intraday_15m" not in daily.columns, (
        "a column belonging to another timeframe was carried along"
    )
    assert "empty_everywhere" not in daily.columns

    intraday = pd.read_parquet(batch / "features_15m.parquet")
    assert "only_intraday_15m" in intraday.columns
    assert "only_daily_1d" not in intraday.columns

    assert report["1d"]["rows"] == 6
    assert report["15m"]["rows"] == 4


def test_identity_columns_survive_even_when_they_look_empty(batch):
    """Without them a slice cannot be joined back or read in time order."""
    split_batch(batch)
    for timeframe in ("1d", "15m"):
        frame = pd.read_parquet(batch / f"features_{timeframe}.parquet")
        for column in ("datetime", "ticker", "interval"):
            assert column in frame.columns, f"{column} lost from {timeframe}"


def test_the_values_are_the_same_ones(batch):
    """A split that quietly reorders or drops rows is worse than none."""
    original = pd.read_parquet(batch / "features.parquet")
    split_batch(batch)
    daily = pd.read_parquet(batch / "features_1d.parquet")

    expected = original[original["interval"].eq("1d")].reset_index(drop=True)
    assert len(daily) == len(expected)
    for column in ("close", "only_daily_1d"):
        assert np.allclose(daily[column], expected[column], equal_nan=True)


def test_the_original_is_left_alone(batch):
    """Written alongside, not in place: the combined file is still the contract.

    `_load_prepared_batch`, training and stages 5 to 7 all read it. Replacing
    it before the slices are verified against it would be a contract change
    made on trust.
    """
    before = (batch / "features.parquet").read_bytes()
    split_batch(batch)
    assert (batch / "features.parquet").read_bytes() == before


def test_targets_are_split_too(batch):
    split_batch(batch)
    daily = pd.read_parquet(batch / "targets_1d.parquet")
    assert "target_return_1d" in daily.columns
    assert len(daily) == 6


def test_a_column_empty_on_one_slice_but_not_another_is_kept_where_it_matters(batch):
    """The whole rule in one case, and the one it would be easy to get backwards."""
    masks = timeframe_slices(batch / "features.parquet")
    daily_columns = columns_with_data(batch / "features.parquet", masks["1d"])
    intraday_columns = columns_with_data(batch / "features.parquet", masks["15m"])

    assert "only_daily_1d" in daily_columns
    assert "only_daily_1d" not in intraday_columns
    assert "only_intraday_15m" in intraday_columns
    assert "only_intraday_15m" not in daily_columns


def test_the_pipeline_writes_the_slices_itself(tmp_path, monkeypatch):
    """Wiring, not arithmetic: a splitter nobody calls produces nothing.

    The slices existed on 2026-08-24 only because they were produced by hand.
    The loader prefers them when present, so unless the pipeline writes them
    the cheap path silently never happens.
    """
    import logging

    from src.pipeline.hybrid.colab_manager import ColabManager

    manager = ColabManager.__new__(ColabManager)
    manager.logger = logging.getLogger("probe")

    frame = pd.DataFrame({
        "datetime": pd.date_range("2026-01-01", periods=4),
        "ticker": ["AAPL"] * 4,
        "interval": ["1d", "1d", "15m", "15m"],
        "close": [1.0, 2.0, 3.0, 4.0],
        "only_daily_1d": [1.0, 2.0, np.nan, np.nan],
    })
    frame.to_parquet(tmp_path / "features.parquet", index=False)
    frame[["datetime", "ticker", "interval"]].assign(
        target_return_1d=0.01
    ).to_parquet(tmp_path / "targets.parquet", index=False)

    manager._write_timeframe_slices(tmp_path)

    assert (tmp_path / "features_1d.parquet").exists()
    assert (tmp_path / "features_15m.parquet").exists()


def test_a_failed_split_does_not_lose_the_batch(tmp_path, monkeypatch, caplog):
    """The combined batch is already written; an optimisation must not cost it."""
    import logging

    from src.pipeline.hybrid.colab_manager import ColabManager

    manager = ColabManager.__new__(ColabManager)
    manager.logger = logging.getLogger("probe")

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "src.pipeline.batch_timeframe_split.split_batch", explode
    )
    with caplog.at_level(logging.ERROR):
        manager._write_timeframe_slices(tmp_path)   # must not raise

    assert any("per-timeframe slices" in record.message
               for record in caplog.records), (
        "the failure was swallowed, so the loader would quietly keep reading "
        "the expensive file with nothing saying why"
    )
