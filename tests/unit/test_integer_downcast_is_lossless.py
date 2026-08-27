"""Half the batch is integers that need one byte, and narrowing them must be free.

Measured on the 110-ticker checkpoints of 2026-08-26: 224 of the 15-minute
frame's 466 columns are int64, 234 of the daily frame's 480, 235 of the hourly
frame's 473. They hold hours, weekdays, months and a long row of `*_available_*`
flags that are 0 or 1 -- eight bytes apiece, about 2.3 GiB across the three
frames where roughly 0.3 GiB would do. The existing downcast only ever looked
at floats.

Narrowing a stored value is exactly the kind of change that goes wrong quietly:
a flag that becomes -1 instead of 255, a count that wraps at 128, and nothing
downstream would report it. So these tests assert equality of values, not just
of dtypes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_orchestrator import FeatureOrchestrator


def _downcast(frame: pd.DataFrame) -> pd.DataFrame:
    return FeatureOrchestrator._downcast_integer_columns(frame.copy())


def test_the_values_survive_unchanged():
    """The whole point: narrower storage, identical numbers."""
    frame = pd.DataFrame({
        "hour": np.arange(24, dtype=np.int64),
        "day_of_week": np.tile(np.arange(6, dtype=np.int64), 4),
        "available_flag": np.tile(np.array([0, 1], dtype=np.int64), 12),
        "signed_small": np.arange(-12, 12, dtype=np.int64),
    })
    narrowed = _downcast(frame)

    for column in frame.columns:
        assert narrowed[column].tolist() == frame[column].tolist(), column


def test_the_columns_actually_get_narrower():
    frame = pd.DataFrame({
        "hour": np.arange(24, dtype=np.int64),
        "available_flag": np.tile(np.array([0, 1], dtype=np.int64), 12),
    })
    narrowed = _downcast(frame)

    # Per column, not per frame: on a 24-row fixture the index and the block
    # overhead dominate whole-frame memory and drown the effect being
    # asserted. Eight bytes to one is the claim, so that is what is measured.
    for column in frame.columns:
        assert frame[column].dtype.itemsize == 8
        assert narrowed[column].dtype.itemsize == 1, column


def test_a_column_that_does_not_fit_is_left_alone():
    """Only what fits is narrowed; the rest stays exactly as it was.

    A timestamp in nanoseconds stored as an integer is the realistic case --
    truncating one would be silent and catastrophic.
    """
    frame = pd.DataFrame({
        "epoch_ns": np.array([1_700_000_000_000_000_000, 1], dtype=np.int64),
    })
    narrowed = _downcast(frame)

    # int64, not uint64. Non-negative values tempt `to_numeric` into an
    # unsigned type of exactly the same width: no saving, and a signedness
    # change that later arithmetic would have to account for. Caught by this
    # test on the first run.
    assert narrowed["epoch_ns"].dtype == np.int64
    assert narrowed["epoch_ns"].tolist() == frame["epoch_ns"].tolist()


def test_negative_values_do_not_become_large_positive_ones():
    """The failure this file exists for: a signed column read as unsigned."""
    frame = pd.DataFrame({"drift": np.array([-1, -128, 0, 127], dtype=np.int64)})
    narrowed = _downcast(frame)

    assert narrowed["drift"].tolist() == [-1, -128, 0, 127]
    assert narrowed["drift"].min() < 0


def test_floats_and_strings_are_not_touched():
    frame = pd.DataFrame({
        "close": np.array([1.5, 2.5], dtype=np.float64),
        "ticker": ["AAPL", "MSFT"],
        "hour": np.array([1, 2], dtype=np.int64),
    })
    narrowed = _downcast(frame)

    assert narrowed["close"].dtype == np.float64
    assert narrowed["ticker"].tolist() == ["AAPL", "MSFT"]
    assert narrowed["hour"].dtype.itemsize == 1


def test_int8_is_left_where_it_is():
    """Already narrow columns must not be walked over for nothing."""
    frame = pd.DataFrame({"flag": np.array([0, 1], dtype=np.int8)})
    assert _downcast(frame)["flag"].dtype == np.int8


def test_an_empty_frame_is_returned_as_it_came():
    frame = pd.DataFrame({"hour": pd.Series([], dtype=np.int64)})
    narrowed = _downcast(frame)
    assert len(narrowed) == 0


def test_nullable_integers_keep_their_missing_values():
    """A pandas nullable Int64 column must not lose its NAs to the narrowing.

    Enrichment produces these wherever a count is absent rather than zero, and
    turning an absent count into 0 would be a value change wearing the costume
    of a memory optimisation.
    """
    frame = pd.DataFrame({"count": pd.array([1, None, 3], dtype="Int64")})
    narrowed = _downcast(frame)

    assert narrowed["count"].isna().tolist() == [False, True, False]
    assert narrowed["count"].dropna().tolist() == [1, 3]
