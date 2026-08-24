"""Accumulation read back the batch it had just written, and joined it to itself.

Two components write the same artifact in one `--mode prepare` run:
`pipeline_runner` writes `features.parquet` at the end of stages 0-3, and
`colab_manager._save_and_accumulate_data` then "accumulates" into the same
path. By the time the second one runs, the file it reads as "existing" IS the
new batch -- so the concat joined 259,133 rows to the identical 259,133 rows,
and `drop_duplicates` was left to remove exactly what the concat had added.

On 2026-08-22 it ended the run: 518,266 x 2,238 float64, and pandas died with
"Unable to allocate 437. MiB" inside `drop_duplicates`, which takes a fresh
copy through boolean indexing. Five minutes AFTER the batch had already been
written and was sitting complete on disk.

Measured against that real batch: 0 rows on disk were absent from the incoming
frame, and reading the three identity columns answered it in 2.3 seconds. The
whole load-concat-dedup was 9 GiB spent to rediscover it had nothing to do.

Accumulation is not removed here -- a previous batch really can hold tickers or
older bars this run did not collect. It is only asked the cheap question first.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline.hybrid.colab_manager import ColabManager


def _frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"datetime": stamp, "ticker": ticker, "interval": interval,
             "some_feature": 1.0}
            for stamp, ticker, interval in rows
        ]
    )


@pytest.fixture
def written(tmp_path):
    def write(frame: pd.DataFrame):
        path = tmp_path / "features.parquet"
        frame.to_parquet(path, index=False)
        return path
    return write


def test_a_batch_written_over_itself_carries_nothing(written):
    """The case that killed the run: disk and incoming are the same rows."""
    batch = _frame([("2026-01-01", "AAPL", "1d"), ("2026-01-02", "AAPL", "1d")])
    path = written(batch)
    assert ColabManager._rows_not_already_present(path, batch) == 0


def test_rows_only_on_disk_are_counted(written):
    """Real accumulation still has something to do, and this must see it."""
    on_disk = _frame([
        ("2026-01-01", "AAPL", "1d"),
        ("2026-01-01", "MSFT", "1d"),   # a ticker this run did not collect
        ("2025-06-01", "AAPL", "1d"),   # a bar older than this run's window
    ])
    incoming = _frame([("2026-01-01", "AAPL", "1d")])
    assert ColabManager._rows_not_already_present(written(on_disk), incoming) == 2


def test_the_same_bar_at_another_timeframe_is_a_different_row(written):
    """`interval` is part of the identity; collapsing it would drop timeframes."""
    on_disk = _frame([("2026-01-01", "AAPL", "15m")])
    incoming = _frame([("2026-01-01", "AAPL", "1d")])
    assert ColabManager._rows_not_already_present(written(on_disk), incoming) == 1


def test_it_says_cannot_tell_rather_than_nothing_to_do(written, tmp_path):
    """A 0 that means "unreadable" would silently discard the previous batch.

    Returning 0 skips accumulation, so 0 must only ever mean "checked, and
    there is nothing to carry over". When the question cannot be answered the
    old behaviour has to stand: fall through and accumulate properly.
    """
    # No identity columns in the incoming frame at all.
    keyless = pd.DataFrame({"some_feature": [1.0, 2.0]})
    path = written(_frame([("2026-01-01", "AAPL", "1d")]))
    assert ColabManager._rows_not_already_present(path, keyless) == 0

    # A file that is not parquet.
    broken = tmp_path / "broken.parquet"
    broken.write_bytes(b"not a parquet file")
    incoming = _frame([("2026-01-01", "AAPL", "1d")])
    assert ColabManager._rows_not_already_present(broken, incoming) == 0


def test_only_the_identity_columns_are_read(written, monkeypatch):
    """The point is the cost, so the cost is what gets pinned.

    Reading all 2,238 columns is the 4.6 GiB this exists to avoid. If a later
    change drops the `columns=` argument the function still returns the right
    number, and only this test would notice.
    """
    wide = _frame([("2026-01-01", "AAPL", "1d")])
    for index in range(30):
        wide[f"filler_{index}"] = 1.0
    path = written(wide)

    seen = {}
    original = pd.read_parquet

    def spy(source, *args, **kwargs):
        seen["columns"] = kwargs.get("columns")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(
        "src.pipeline.hybrid.colab_manager.pd.read_parquet", spy
    )
    ColabManager._rows_not_already_present(path, wide)

    assert seen["columns"] == ["datetime", "ticker", "interval"], (
        f"the whole frame was read, not just the keys: {seen['columns']}"
    )


def test_timezone_disagreement_does_not_kill_the_run(tmp_path):
    """pandas refuses the merge outright, and that ended a rebuild.

    The file on disk and the frame in memory need not agree on timezone. When
    they did not, pandas raised "You are trying to merge on datetime64[ns] and
    datetime64[ns, UTC] columns for key 'datetime'" -- two and a half hours
    into the 2026-08-23 rebuild, after the batch had already been written, so
    the work survived and the run still reported failure.
    """
    aware = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        "ticker": ["AAPL", "AAPL"],
        "interval": ["1d", "1d"],
        "some_feature": [1.0, 2.0],
    })
    naive = aware.copy()
    naive["datetime"] = naive["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)

    path = tmp_path / "features.parquet"
    naive.to_parquet(path, index=False)

    # Same rows, one side tz-aware: nothing to carry over, and no exception.
    assert ColabManager._rows_not_already_present(path, aware) == 0

    extra = pd.concat([aware, pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-03"], utc=True),
        "ticker": ["MSFT"], "interval": ["1d"], "some_feature": [3.0],
    })], ignore_index=True)
    extra.to_parquet(path, index=False)
    assert ColabManager._rows_not_already_present(path, aware) == 1
