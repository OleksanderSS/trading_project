"""Encoding a handful of categorical columns must not copy 245 numeric ones.

`handle_categorical_features` opened with `df_out = df.copy()` — a DEEP copy
of the whole frame. On the pooled daily context that is 704,210 rows by 245
float64 columns; pandas consolidates blocks while copying, so it asked for
1.29 GiB twice and killed the modelling stage on 2026-08-31, at a different
line from the MemoryError that had killed it that morning.

The function only ever touches categorical columns. In that frame there were
none it could touch at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.adapters.data_preparation import handle_categorical_features


def _numeric_frame(rows=2000, cols=60):
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(
        rng.normal(size=(rows, cols)),
        columns=[f"f{i}" for i in range(cols)],
    )
    frame["target_up_1d"] = (rng.random(rows) > 0.5).astype(int)
    return frame


def test_a_frame_with_nothing_to_encode_is_returned_untouched():
    frame = _numeric_frame()
    out, info = handle_categorical_features(frame, ["target_up_1d"])

    assert info == {}
    # The same object, not a copy of it: nothing had to be built.
    assert out is frame


def test_the_frame_is_rebuilt_once_not_once_per_categorical_column():
    """`pd.concat` inside the loop rebuilt everything per column.

    Five categorical columns meant five reconstructions of 704,210 x 245
    float64 — 1.29 GiB apiece. Peak memory is what matters here, not which
    buffer ends up shared, so it is measured rather than asserted about.
    """
    import tracemalloc

    def peak_for(n_categoricals):
        frame = _numeric_frame(rows=20_000, cols=40)
        for i in range(n_categoricals):
            frame[f"cat{i}"] = np.where(
                np.arange(len(frame)) % (i + 2) == 0, "low", "high")
        tracemalloc.start()
        handle_categorical_features(frame, ["target_up_1d"])
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak

    one, five = peak_for(1), peak_for(5)

    # Four more categorical columns add four dummy columns, not four copies
    # of forty numeric ones. Anything near 5x means the loop is rebuilding.
    assert five < one * 2.0, (one, five)


def test_the_caller_s_frame_is_not_mutated():
    """A shallow copy is only safe if nothing writes through it."""
    frame = _numeric_frame(rows=300, cols=5)
    frame["regime"] = np.where(np.arange(300) % 2 == 0, "low", "high")
    columns_before = list(frame.columns)
    values_before = frame["f0"].to_numpy().copy()

    out, _info = handle_categorical_features(frame, ["target_up_1d"])

    assert list(frame.columns) == columns_before
    assert np.array_equal(frame["f0"].to_numpy(), values_before)
    assert "regime" not in out.columns


def test_a_high_cardinality_column_is_still_dropped():
    """The behaviour the copy was incidental to must be unchanged."""
    frame = _numeric_frame(rows=300, cols=3)
    frame["many"] = [f"v{i % 40}" for i in range(300)]

    out, info = handle_categorical_features(frame, ["target_up_1d"])

    assert info == {"many": "dropped_unpersisted_encoding"}
    assert "many" not in out.columns


def test_the_time_index_is_attached_without_copying_the_frame():
    """Four whole-frame copies in one chain, 1.29 GiB apiece.

    `filtered_df.assign(_model_datetime=...).dropna(...).sort_values(...)
    .set_index(...)` — `assign` opens with `self.copy(deep=None)` and each of
    the other three builds a new frame. On the pooled daily context that is
    704,724 rows by 245 float64 columns, and it killed the modelling stage on
    2026-08-31 twenty-five seconds into the daily frame: the THIRD MemoryError
    of the day in this one function.

    The two ways of attaching the index are measured against each other —
    same frame, same result — rather than against the whole of preparation,
    which does far more and would make the comparison meaningless.
    """
    import tracemalloc

    rows, cols = 60_000, 80
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        rng.normal(size=(rows, cols)),
        columns=[f"f{i}" for i in range(cols)],
    )
    frame["datetime"] = pd.date_range(
        "2020-01-01", periods=rows, freq="15min", tz="UTC")

    def old_chain(df):
        stamps = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        out = (
            df.assign(_model_datetime=stamps)
            .dropna(subset=["_model_datetime"])
            .sort_values("_model_datetime", kind="mergesort")
            .set_index("_model_datetime", drop=True)
        )
        out.index.name = "model_datetime"
        return out

    def new_way(df):
        stamps_series = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        usable = stamps_series.notna()
        if not bool(usable.all()):
            df = df.loc[usable]
            stamps_series = stamps_series.loc[usable]
        stamps = stamps_series.to_numpy()
        if not pd.Index(stamps).is_monotonic_increasing:
            order = np.argsort(stamps, kind="stable")
            df = df.iloc[order]
            stamps = stamps[order]
        df = df.copy(deep=False)
        df.index = pd.DatetimeIndex(stamps, name="model_datetime")
        return df

    def peak(fn):
        tracemalloc.start()
        out = fn(frame)
        used = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return used, out

    old_peak, old = peak(old_chain)
    new_peak, new = peak(new_way)

    # Same answer.
    assert new.index.name == old.index.name == "model_datetime"
    assert len(new) == len(old) == rows
    assert new.index.equals(old.index)

    # A fraction of the cost. The chain copies the values; attaching an index
    # to a shallow copy does not touch them.
    assert new_peak < old_peak / 4, (old_peak, new_peak)
