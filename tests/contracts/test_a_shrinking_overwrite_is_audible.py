"""Replacing a big batch with a small one must not be silent.

REGISTER #138. `features.parquet` is written by more than one component:
`colab_manager` and `feature_processor` both call `write_union`, and
`colab_manager` also writes the path directly. On the v26 run it was written
twice -- 22:32:46 by one and 23:15:57 by the other -- identically, so nothing
was lost that day.

The danger was always structural rather than observed. The second writer
builds its frame from its own path, and NOTHING compared what was about to be
written against what already lay on disk. A filtered or partial frame would
have replaced a good 1,242,693 x 2,284 batch in complete silence, and every
later stage would have reported success on the remains.

WHAT THIS DOES AND DOES NOT DO. A smaller write is not always wrong:
`--tickers AAPL` and single-timeframe runs are legitimate and must keep
working. So the shapes are compared and named, a shrinking overwrite is an
ERROR, and nothing is refused. Removing the silence is the fix available
without deciding who owns the artefact -- which is the open half of #138 and
belongs to the owner.

The comparison reads parquet METADATA only, so it costs nothing on a 969 MiB
file.
"""
from __future__ import annotations

import logging

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.pipeline.parquet_union_writer import write_union


def _frame(rows: int, cols: int) -> pd.DataFrame:
    return pd.DataFrame({f"c{i}": range(rows) for i in range(cols)})


def test_a_shrinking_overwrite_is_reported_at_error(tmp_path, caplog):
    path = tmp_path / "features.parquet"
    write_union({"1d": _frame(500, 6)}, path)

    with caplog.at_level(logging.ERROR):
        write_union({"1d": _frame(100, 6)}, path)

    said = " ".join(r.getMessage() for r in caplog.records
                    if r.levelno >= logging.ERROR)
    assert "OVERWRITING" in said and "WITH LESS" in said
    assert "500" in said and "100" in said, (
        "the shapes are not named, so a reader cannot tell how much was lost"
    )
    assert "#138" in said, (
        "the message does not point at why more than one writer exists"
    )


def test_losing_columns_counts_as_shrinking(tmp_path, caplog):
    """Rows are the obvious axis. A frame with the same rows and fewer columns
    is the one that looks fine in a row count and drops whole feature
    families."""
    path = tmp_path / "features.parquet"
    write_union({"1d": _frame(200, 8)}, path)

    with caplog.at_level(logging.ERROR):
        write_union({"1d": _frame(200, 3)}, path)

    said = " ".join(r.getMessage() for r in caplog.records
                    if r.levelno >= logging.ERROR)
    assert "OVERWRITING" in said
    assert "-5 columns" in said or "-5" in said


def test_a_growing_write_does_not_cry_wolf(tmp_path, caplog):
    """A rebuild that adds rows is the ordinary case. A check that fires on it
    gets switched off -- `|| true` sat in ci.yml six weeks for that reason."""
    path = tmp_path / "features.parquet"
    write_union({"1d": _frame(100, 4)}, path)

    with caplog.at_level(logging.ERROR):
        write_union({"1d": _frame(500, 4)}, path)

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not errors, f"a growing write was reported as loss: {errors}"


def test_the_first_write_says_nothing_about_a_previous_one(tmp_path, caplog):
    path = tmp_path / "features.parquet"
    with caplog.at_level(logging.INFO):
        write_union({"1d": _frame(100, 4)}, path)
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "OVERWRITING" not in said
    assert "Replacing" not in said


def test_the_written_file_is_still_correct(tmp_path):
    """The comparison must not disturb what it measures."""
    path = tmp_path / "features.parquet"
    write_union({"1d": _frame(300, 5)}, path)
    written = write_union({"1d": _frame(120, 5)}, path)

    assert written == {"1d": 120}
    assert pq.ParquetFile(path).metadata.num_rows == 120


def test_an_unreadable_existing_file_does_not_stop_the_write(tmp_path, caplog):
    """The check must never become the reason a run dies: it exists to make a
    loss audible, not to add a failure."""
    path = tmp_path / "features.parquet"
    path.write_bytes(b"this is not parquet")

    with caplog.at_level(logging.WARNING):
        written = write_union({"1d": _frame(50, 3)}, path)

    assert written == {"1d": 50}, "a corrupt existing file blocked the write"
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "unchecked" in said.lower(), (
        "the write proceeded without saying that no comparison was possible"
    )
