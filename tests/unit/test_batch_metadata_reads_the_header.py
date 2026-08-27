"""Metadata about a batch must not cost the batch.

`pd.read_parquet(features_path)` pulled all 2,305 columns of 1,232,907 rows
into memory to fill in a shape and a list of timeframes, and ended the
2026-08-27 run with `malloc of size 7308672704 failed` -- 6.8 GiB, requested
after everything of value was already on disk. Both answers live in the parquet
header.

It is the third time this project has paid for a whole frame to answer a
question about its description: `select_dtypes` was called to list column
names, `drop_duplicates` reloaded a batch to discover it had nothing to do, and
now this. So the cost is what these tests pin, not just the values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.hybrid.colab_manager import ColabManager


@pytest.fixture
def batch(tmp_path):
    """A file shaped like the real batch: three timeframes, wide, stacked."""
    frames = []
    for interval, rows in (("15m", 40), ("1d", 25), ("60m", 15)):
        frame = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "ticker": ["AAPL"] * rows,
            "interval": [interval] * rows,
        })
        for index in range(30):
            frame[f"f{index}"] = np.zeros(rows, dtype=np.float32)
        frames.append(frame)

    path = tmp_path / "features.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return path


def test_the_shape_matches_the_file(batch):
    assert ColabManager._parquet_shape(batch) == pd.read_parquet(batch).shape


def test_the_metadata_columns_carry_every_timeframe(batch):
    intervals = ColabManager._metadata_columns(batch)

    # interval and datetime, not the other 2,300 columns: the metadata step
    # reads which timeframes arrived and what timezone the stamps carry.
    assert set(intervals.columns) == {"interval", "datetime"}
    assert set(intervals["interval"]) == {"15m", "1d", "60m"}
    assert len(intervals) == 80


def test_only_the_metadata_columns_are_read(batch, monkeypatch):
    """The point is the cost, so the cost is what gets pinned.

    If a later change drops the `columns=` argument the timeframes still come
    out right and the 6.8 GiB comes back, with only this test to notice.
    """
    seen = {}
    original = pd.read_parquet

    def spy(source, *args, **kwargs):
        seen["columns"] = kwargs.get("columns")
        return original(source, *args, **kwargs)

    monkeypatch.setattr("src.pipeline.hybrid.colab_manager.pd.read_parquet", spy)
    ColabManager._metadata_columns(batch)

    assert seen["columns"] == ["interval", "datetime"]


def test_a_missing_interval_column_is_not_an_error(tmp_path):
    """No interval column means no timeframes delivered, which is an answer."""
    path = tmp_path / "features.parquet"
    pd.DataFrame({"close": [1.0, 2.0]}).to_parquet(path, index=False)

    # None of the metadata columns are present, so nothing is read at all --
    # asking parquet for an absent column raises.
    assert ColabManager._metadata_columns(path).empty
    assert ColabManager._parquet_shape(path) == (2, 1)


def test_an_unreadable_file_reports_zero_rather_than_raising(tmp_path):
    """Metadata is not worth ending a run whose batch is already written."""
    broken = tmp_path / "features.parquet"
    broken.write_bytes(b"not a parquet file")

    assert ColabManager._parquet_shape(broken) == (0, 0)
    assert ColabManager._metadata_columns(broken).empty
