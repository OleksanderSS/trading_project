"""Normalising one column asked for a duplicate of the whole frame.

`ensure_datetime_column` opened with `df = df.copy()`. On the batch that is
2,200 columns by 259,133 rows, and it ended a rebuild that had already
finished stage 3 -- 146 minutes of work -- while validating the frame for
export:

    numpy._core._exceptions._ArrayMemoryError: Unable to allocate 4.25 GiB
    for an array with shape (2200, 259133)

The depth was never needed. Every branch replaces a whole column or calls
reset_index/rename, all of which build a new frame rather than writing into a
shared block. Measured on 30,000 x 400: 92.3 MiB peak deep against 0.5 MiB
shallow, with the caller's frame unchanged either way.

This is the third instance in one session of the same shape -- a frame
duplicated to read or set one thing. The others were `_initial_feature_columns`
and `_select_features`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.utils.datetime_utils import ensure_datetime_column


def _frame(rows=200, cols=20):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.standard_normal((rows, cols)),
                         columns=[f"c{i}" for i in range(cols)])
    frame["ticker"] = "AAPL"
    frame["datetime"] = pd.date_range("2026-01-01", periods=rows, freq="h")
    return frame


def test_the_callers_frame_is_not_modified():
    """The only thing the deep copy was protecting."""
    frame = _frame()
    before = frame["datetime"].copy()
    ensure_datetime_column(frame)
    pd.testing.assert_series_equal(frame["datetime"], before)


def test_the_returned_frame_is_a_different_object():
    frame = _frame()
    assert ensure_datetime_column(frame) is not frame


def test_writing_to_the_result_does_not_reach_the_original():
    """Shallow sharing must not leak an edit backwards."""
    frame = _frame()
    result = ensure_datetime_column(frame)
    result["c0"] = 999.0
    assert frame["c0"].iloc[0] != 999.0


def test_it_still_normalises_the_column():
    frame = _frame()
    frame["datetime"] = frame["datetime"].astype(str)
    result = ensure_datetime_column(frame)
    assert pd.api.types.is_datetime64_any_dtype(result["datetime"])


def test_it_still_recovers_datetime_from_the_index():
    frame = _frame().set_index("datetime")
    result = ensure_datetime_column(frame)
    assert "datetime" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["datetime"])


def test_it_still_finds_an_alternative_column_name():
    frame = _frame().drop(columns=["datetime"])
    frame["published_at"] = pd.date_range("2026-01-01", periods=len(frame), freq="h")
    result = ensure_datetime_column(frame)
    assert "datetime" in result.columns


def test_a_missing_datetime_still_raises_when_asked():
    frame = _frame().drop(columns=["datetime"])
    with pytest.raises(ValueError):
        ensure_datetime_column(frame, raise_on_missing=True)


def test_the_copy_is_shallow():
    """Pins the fix: a deep copy here is what killed a 146-minute run."""
    import inspect

    source = inspect.getsource(ensure_datetime_column)
    assert "df.copy(deep=False)" in source
