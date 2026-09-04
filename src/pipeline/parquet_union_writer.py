"""Write the union of several frames to one parquet file, one frame at a time.

`features.parquet` holds every timeframe stacked: 15-minute rows, hourly rows
and daily rows in one file, with each timeframe's own columns null on the other
timeframes' rows. Measured on the v9 batch, 69.3% of its cells are empty by
construction, and at 110 tickers the whole thing comes to about 11 GiB.

Building it with `pd.concat` needs all 11 GiB resident at once, and on a machine
with roughly 10 GB usable it does not fit. Four separate attempts on 2026-08-26
trimmed allocations around that concat and all four failed for the same reason:
the union does not fit, and no trimming makes it fit.

The file itself is not the problem -- it is a contract. Fingerprints, cache
validation, historical replay and the timeframe splitter all read
`features.parquet`, so it keeps existing with exactly the rows and columns it
had. What changes is that it is now written incrementally, one timeframe per
row group, so peak memory is one timeframe rather than their sum.

**Why the padding is free.** Aligning a frame to the union's columns would
normally mean adding the other timeframes' columns as NaN -- for the daily
frame that is 705k rows times ~1,400 absent columns, several gigabytes of
nothing, which would put the problem straight back. Arrow stores an all-null
column as a null count and a validity bitmap: one bit per row rather than four
or eight bytes. The same padding costs about 123 MB instead of 3.9 GiB, and
parquet then compresses the nulls to almost nothing on disk.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def union_schema(frames: Mapping[str, pd.DataFrame]) -> pa.Schema:
    """One schema covering every frame, without concatenating them.

    Column ORDER follows first appearance, so a file written from the same
    frames in the same order is byte-comparable run to run. Types are unified
    permissively: the same column may arrive as float32 from one timeframe and
    float64 from another -- enrichment downcasts per timeframe -- and the union
    has to hold both without silently truncating either.
    """
    # Inferred from real rows, not from `frame.iloc[:0]`.
    #
    # An object column with no rows to look at infers as Arrow's `null` type,
    # not as string -- so a schema taken from an empty slice types `ticker` as
    # null, and writing the actual tickers into it then fails with "Invalid
    # null value". A small head is enough for inference and costs nothing.
    schemas = [
        pa.Schema.from_pandas(frame.head(64), preserve_index=False)
        for frame in frames.values()
        if frame is not None and not frame.empty
    ]
    if not schemas:
        return pa.schema([])
    try:
        return pa.unify_schemas(schemas, promote_options="permissive")
    except TypeError:
        # pyarrow < 14 has no promote_options; its unify_schemas is already
        # permissive about widening.
        return pa.unify_schemas(schemas)


def _aligned_table(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """One frame as a table with the union's columns, absent ones all-null."""
    columns: list[pa.Array | pa.ChunkedArray] = []
    for field in schema:
        # A column that was empty everywhere unifies to Arrow's `null` type,
        # which holds no values by definition -- writing its data into it
        # raises rather than silently dropping, so it is nulls either way.
        if field.name in frame.columns and not pa.types.is_null(field.type):
            columns.append(
                pa.array(frame[field.name], type=field.type, from_pandas=True)
            )
        else:
            # The cheap part: a validity bitmap, not a column of NaN.
            columns.append(pa.nulls(len(frame), type=field.type))
    return pa.Table.from_arrays(columns, schema=schema)


#: Rows converted to Arrow at a time. The peak is one chunk of the union's
#: width, not one frame of it -- measured at a fifth of the real batch, writing
#: whole frames cost +0.65 GiB above the frames themselves and chunking cut
#: that to a fraction, while `pd.concat` cost +1.96 GiB. 128k rows of 1,411
#: float32 columns is roughly a quarter of a gigabyte, which is small enough to
#: stop mattering and large enough that row groups stay usefully sized for
#: readers.
ROW_GROUP_ROWS = 128_000


def _report_if_this_write_shrinks(present: Mapping[str, pd.DataFrame],
                                  schema, destination: Path) -> None:
    """Say so when this write would replace a bigger file with a smaller one.

    REGISTER #138: `features.parquet` is written by more than one component --
    `colab_manager` and `feature_processor` both call `write_union`, and
    `colab_manager` also writes the path directly. On the v26 run it was
    written twice, identically, so nothing was lost. The danger was always
    structural: the second writer builds its frame from its own path, and
    NOTHING compared what was about to be written with what already lay on
    disk. A filtered or partial frame would have replaced a good batch in
    silence.

    A smaller write is not always wrong -- `--tickers AAPL` and single-frame
    runs are legitimate -- so this reports rather than refuses. What it removes
    is the silence: the shapes are named, and a shrinking overwrite is an
    ERROR rather than an unremarked one.

    Reads parquet METADATA only, so it costs nothing on a 969 MiB file.
    """
    if not destination.exists():
        return
    try:
        existing = pq.ParquetFile(destination)
        old_rows = existing.metadata.num_rows
        old_cols = len(existing.schema_arrow.names)
    except Exception as error:  # noqa: BLE001 - reported, never swallowed
        logger.warning(
            "Could not read the existing %s to compare shapes (%s: %s); "
            "writing over it unchecked.",
            destination.name, type(error).__name__, error,
        )
        return

    new_rows = sum(len(frame) for frame in present.values())
    new_cols = len(schema.names)
    if new_rows < old_rows or new_cols < old_cols:
        logger.error(
            "OVERWRITING %s WITH LESS: on disk %d rows x %d columns, about to "
            "write %d x %d (%+d rows, %+d columns). More than one component "
            "writes this file (REGISTER #138); if this write came from a "
            "filtered or partial run, a good batch is being replaced.",
            destination.name, old_rows, old_cols, new_rows, new_cols,
            new_rows - old_rows, new_cols - old_cols,
        )
    else:
        logger.info(
            "Replacing %s: on disk %d rows x %d columns, writing %d x %d.",
            destination.name, old_rows, old_cols, new_rows, new_cols,
        )


def write_union(frames: Mapping[str, pd.DataFrame], destination: Path,
                compression: str = "snappy",
                row_group_rows: int = ROW_GROUP_ROWS) -> dict[str, int]:
    """Write every frame into one parquet file, a chunk of rows at a time.

    Returns rows written per key. An empty mapping writes nothing and returns
    an empty report rather than an empty file, because a zero-row
    `features.parquet` reads downstream as a successful empty batch.
    """
    present = {
        name: frame for name, frame in frames.items()
        if frame is not None and not frame.empty
    }
    if not present:
        logger.warning("Nothing to write to %s: every frame was empty.", destination)
        return {}

    schema = union_schema(present)
    _report_if_this_write_shrinks(present, schema, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    written: dict[str, int] = {}
    writer = pq.ParquetWriter(destination, schema, compression=compression)
    try:
        for name, frame in present.items():
            for start in range(0, len(frame), row_group_rows):
                chunk = frame.iloc[start:start + row_group_rows]
                table = _aligned_table(chunk, schema)
                writer.write_table(table)
                del table, chunk
            written[name] = len(frame)
            logger.info(
                "Wrote %s into %s: %d rows, %d of %d columns carried data.",
                name, destination.name, len(frame),
                len(frame.columns), len(schema.names),
            )
    finally:
        writer.close()

    logger.info(
        "%s: %d rows across %d frames, %d columns, %.0f MiB on disk.",
        destination.name, sum(written.values()), len(written),
        len(schema.names), destination.stat().st_size / 2 ** 20,
    )
    return written
