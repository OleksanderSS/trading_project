"""The union file must be identical to the concat it replaces, and cost less.

`features.parquet` is a contract -- fingerprints, cache validation, historical
replay and the timeframe splitter all read it -- so replacing `pd.concat` with
an incremental writer is only allowed if the file that comes out is the same
file. These tests hold the concat as the reference and compare against it.

There is no deliberate difference. The first version of this file asserted one
-- that concat widens float32 to float64 wherever a frame lacks the column --
and the test disproved it: pandas pads with NaN and keeps float32, since
float32 holds NaN perfectly well. So the comparison below is exact, dtypes
included, which is a stronger guarantee than the one originally claimed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.pipeline.parquet_union_writer import union_schema, write_union


def _frame(rows: int, own: list[str], seed: int) -> pd.DataFrame:
    """A timeframe's slice: shared identity columns plus columns of its own."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=rows, freq="h"),
        "ticker": ["AAPL"] * rows,
        "close": rng.standard_normal(rows).astype(np.float32),
    })
    for name in own:
        frame[name] = rng.standard_normal(rows).astype(np.float32)
    return frame


@pytest.fixture
def frames() -> dict[str, pd.DataFrame]:
    return {
        "15m": _frame(40, ["rsi_15m", "atr_15m"], seed=1),
        "60m": _frame(25, ["rsi_60m"], seed=2),
        "1d": _frame(10, ["fund_pe", "fund_debt_to_equity"], seed=3),
    }


def test_the_file_holds_every_row_and_column(frames, tmp_path):
    path = tmp_path / "features.parquet"
    written = write_union(frames, path)

    assert written == {"15m": 40, "60m": 25, "1d": 10}

    back = pd.read_parquet(path)
    reference = pd.concat(frames.values(), ignore_index=True, sort=False)

    assert len(back) == len(reference) == 75
    assert set(back.columns) == set(reference.columns)


def test_values_match_the_concat_it_replaces(frames, tmp_path):
    """Same numbers in the same places -- the whole point of the exercise."""
    path = tmp_path / "features.parquet"
    write_union(frames, path)

    back = pd.read_parquet(path)
    reference = pd.concat(frames.values(), ignore_index=True, sort=False)
    back = back[reference.columns]

    # Exact, dtypes included. See the module docstring: the widening this was
    # originally written to tolerate does not happen.
    pd.testing.assert_frame_equal(back, reference)


def test_absent_columns_are_null_only_on_the_rows_that_lack_them(frames, tmp_path):
    """A daily column must be null on 15-minute rows and present on daily ones.

    If padding leaked the wrong way -- a value where there should be none --
    the frame would look richer than it is and nothing downstream would
    notice, which is the failure this file exists to prevent.
    """
    path = tmp_path / "features.parquet"
    write_union(frames, path)
    back = pd.read_parquet(path)

    daily_rows = back.index >= 65          # 40 + 25 written before the daily frame
    assert back.loc[daily_rows, "fund_pe"].notna().all()
    assert back.loc[~daily_rows, "fund_pe"].isna().all()
    assert back.loc[daily_rows, "rsi_15m"].isna().all()


def test_float32_is_not_widened_on_the_way_through(frames, tmp_path):
    """Half the file's size rides on this, so the stored type is pinned.

    Arrow padding an absent column with nulls must not promote it to double.
    A silent widening would double `features.parquet` and put back exactly the
    memory the incremental writer exists to avoid -- and nothing else would
    report it.
    """
    path = tmp_path / "features.parquet"
    write_union(frames, path)

    schema = pq.ParquetFile(path).schema_arrow
    assert str(schema.field("fund_pe").type) == "float"
    assert str(schema.field("close").type) == "float"

    # pandas keeps float32 too when padding with NaN, so the file matches the
    # concat here rather than improving on it.
    reference = pd.concat(frames.values(), ignore_index=True, sort=False)
    assert reference["fund_pe"].dtype == np.float32


def test_rows_are_written_in_bounded_chunks(frames, tmp_path):
    """Chunked row groups are the mechanism: peak is a chunk, not a frame.

    If a later change gathers the tables and writes them in one go the output
    stays correct and the memory problem comes back with nothing to show it.
    The row-group count is the only visible trace, so it is asserted here.
    """
    path = tmp_path / "features.parquet"
    write_union(frames, path, row_group_rows=10)

    # 40, 25 and 10 rows in chunks of ten: 4 + 3 + 1.
    assert pq.ParquetFile(path).num_row_groups == 8
    assert len(pd.read_parquet(path)) == 75


def test_chunking_does_not_change_the_contents(frames, tmp_path):
    """Whatever the chunk size, the file must be the same file."""
    whole = tmp_path / "whole.parquet"
    chunked = tmp_path / "chunked.parquet"
    write_union(frames, whole, row_group_rows=10_000_000)
    write_union(frames, chunked, row_group_rows=7)

    pd.testing.assert_frame_equal(
        pd.read_parquet(whole), pd.read_parquet(chunked)
    )


def test_empty_frames_are_skipped_not_written(frames, tmp_path):
    path = tmp_path / "features.parquet"
    frames["60m"] = frames["60m"].iloc[:0]
    written = write_union(frames, path)

    assert "60m" not in written
    assert len(pd.read_parquet(path)) == 50


def test_nothing_at_all_writes_no_file(tmp_path):
    """A zero-row features.parquet reads downstream as a successful empty batch.

    That is worse than no file: the run looks finished. So an all-empty input
    must leave nothing behind rather than an empty artifact.
    """
    path = tmp_path / "features.parquet"
    assert write_union({"1d": pd.DataFrame()}, path) == {}
    assert not path.exists()


def test_the_schema_covers_frames_that_disagree_on_type(tmp_path):
    """float32 in one timeframe and float64 in another must both survive.

    Enrichment downcasts per timeframe, so the same column genuinely arrives
    at two widths. Unifying to the narrower one would truncate real values.
    """
    narrow = pd.DataFrame({"x": np.array([1.0, 2.0], dtype=np.float32)})
    wide = pd.DataFrame({"x": np.array([1e300, 2.0], dtype=np.float64)})

    schema = union_schema({"a": narrow, "b": wide})
    assert str(schema.field("x").type) == "double"

    path = tmp_path / "u.parquet"
    write_union({"a": narrow, "b": wide}, path)
    assert pd.read_parquet(path)["x"].iloc[2] == 1e300
