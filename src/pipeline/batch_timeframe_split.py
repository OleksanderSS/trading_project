"""One batch file per timeframe, because 69% of the combined one is empty.

The combined frame carries every timeframe's columns on every row, so a daily
bar holds NaN in each 15-minute column and vice versa. Measured on the v9
batch of 2026-08-24:

    15m   29,097 rows   1,386 of 2,302 columns carry anything   60%
    60m   75,967 rows     932                                   40%
    1d   154,069 rows     466                                   20%

Splitting removes 69.3% of the cells: 2.22 GiB becomes 0.68, and at 110
tickers 11.1 GiB becomes 3.4. That last number is the point -- 11 does not fit
on this machine and 3.4 does, which makes this the change that allows a wider
universe rather than a tidier file.

The daily frame is where the waste concentrates: 154,069 rows times 1,836
unused columns, 283 million empty cells. It is also where the fundamentals and
the cross-sectional targets live.

**This module never holds the union.** It reads the interval column once, then
takes each timeframe's columns in blocks, so its own peak is one timeframe's
slice. A splitter that materialised the combined frame would solve the storage
problem and leave the memory one exactly where it was -- and the memory one is
what stops 110 tickers, because stage 3 would die before writing anything.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

#: Columns every slice keeps regardless of whether they look empty: without
#: them a slice cannot be joined back to anything or read in time order.
IDENTITY_COLUMNS = ("datetime", "ticker", "interval", "hash", "timestamp", "date")

#: Read this many columns at a time. The point of the module is never to hold
#: the whole width at once.
BLOCK = 200


def timeframe_slices(path: Path) -> dict[str, np.ndarray]:
    """Row masks per timeframe, read from the interval column alone."""
    intervals = pd.read_parquet(path, columns=["interval"])["interval"].astype(str)
    return {
        timeframe: (intervals == timeframe).to_numpy()
        for timeframe in sorted(intervals.unique())
    }


def columns_with_data(path: Path, mask: np.ndarray) -> list[str]:
    """Which columns carry anything at all on these rows.

    An all-null column on this slice belongs to another timeframe. Identity
    columns are kept even when they look empty, because dropping them would
    leave a slice that cannot be joined or ordered.
    """
    names = pq.ParquetFile(path).schema_arrow.names
    keep: list[str] = []
    for start in range(0, len(names), BLOCK):
        block_names = names[start:start + BLOCK]
        try:
            block = pd.read_parquet(path, columns=block_names)
        except (OSError, ValueError) as error:
            logger.warning("Could not read %s: %s", block_names[:3], error)
            continue
        for name in block_names:
            if name not in block.columns:
                continue
            if name in IDENTITY_COLUMNS or block[name].to_numpy()[mask].size and \
                    block[name].notna().to_numpy()[mask].any():
                keep.append(name)
        del block
    return keep


def write_timeframe_slice(source: Path, destination: Path,
                          mask: np.ndarray, keep: list[str]) -> int:
    """Write one timeframe's rows and its own columns, in column blocks."""
    pieces = []
    for start in range(0, len(keep), BLOCK):
        block = pd.read_parquet(source, columns=keep[start:start + BLOCK])
        pieces.append(block.loc[mask].reset_index(drop=True))
        del block
    slice_frame = pd.concat(pieces, axis=1) if len(pieces) > 1 else pieces[0]
    del pieces
    slice_frame.to_parquet(destination, index=False)
    rows = len(slice_frame)
    del slice_frame
    return rows


def split_batch(directory: Path, features_name: str = "features.parquet",
                targets_name: str = "targets.parquet") -> dict[str, dict]:
    """Write features_<tf>.parquet and targets_<tf>.parquet beside the originals.

    Written ALONGSIDE rather than in place. The combined file is read by
    `_load_prepared_batch`, by training and by stages 5 to 7; replacing it
    before the slices are verified against it would be a contract change made
    on trust.
    """
    features = directory / features_name
    targets = directory / targets_name
    if not features.exists():
        raise FileNotFoundError(f"{features} does not exist")

    masks = timeframe_slices(features)
    total_width = len(pq.ParquetFile(features).schema_arrow.names)
    report: dict[str, dict] = {}

    for timeframe, mask in masks.items():
        keep = columns_with_data(features, mask)
        out = directory / f"features_{timeframe}.parquet"
        rows = write_timeframe_slice(features, out, mask, keep)
        entry = {
            "rows": rows,
            "columns": len(keep),
            "of": total_width,
            "features_path": out,
            "megabytes": out.stat().st_size / 2 ** 20,
        }

        if targets.exists():
            target_mask = timeframe_slices(targets).get(timeframe)
            if target_mask is not None and target_mask.any():
                target_keep = columns_with_data(targets, target_mask)
                target_out = directory / f"targets_{timeframe}.parquet"
                entry["target_rows"] = write_timeframe_slice(
                    targets, target_out, target_mask, target_keep
                )
                entry["target_columns"] = len(target_keep)
                entry["targets_path"] = target_out

        report[timeframe] = entry
        logger.info(
            "%s: %d rows, %d of %d columns, %.0f MiB",
            timeframe, rows, len(keep), total_width, entry["megabytes"],
        )
    return report
