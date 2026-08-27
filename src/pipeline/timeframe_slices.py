"""Load a batch as one frame per timeframe instead of as their union.

`features.parquet` stacks every timeframe, so each row carries nulls in the
other timeframes' columns -- 69.3% of its cells on the v9 batch, and at 110
tickers it expands to about 6.8 GiB in memory. `features_<tf>.parquet` beside
it holds the same rows with only the columns that timeframe uses.

Two places need this and had one copy between them: the hybrid orchestrator,
which loads a prepared batch for training, and the cache check, which was still
reading the union and undoing the whole point of writing the slices.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: A timeframe label is short and has no underscore. The batch directory also
#: holds `features_all110_20260806.parquet` and `features_CORRUPT_dates_...`,
#: which match `features_*.parquet` and are not timeframes.
MAX_TIMEFRAME_LENGTH = 4


def slice_paths(base: Path) -> dict[str, tuple[Path, Path]]:
    """Timeframe -> (features path, targets path), for complete pairs only."""
    candidates: dict[str, tuple[Path, Path]] = {}
    for path in sorted(base.glob("features_*.parquet")):
        timeframe = path.stem.removeprefix("features_")
        if len(timeframe) > MAX_TIMEFRAME_LENGTH or "_" in timeframe:
            continue
        targets_path = base / f"targets_{timeframe}.parquet"
        if targets_path.exists():
            candidates[timeframe] = (path, targets_path)
    return candidates


def load_timeframe_slices(
    base: Path, log: logging.Logger | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]] | None:
    """Features and targets as dicts keyed by timeframe, or None.

    None when the slices are absent -- a caller that gets None should fall back
    to the combined file rather than proceed on part of the batch. A partial
    set would silently train on fewer timeframes than the batch holds, which is
    worse than being slow.
    """
    log = log or logger
    candidates = slice_paths(base)
    if not candidates:
        return None

    features: dict[str, pd.DataFrame] = {}
    targets: dict[str, pd.DataFrame] = {}
    for timeframe, (feature_path, target_path) in candidates.items():
        features[timeframe] = pd.read_parquet(feature_path)
        targets[timeframe] = pd.read_parquet(target_path)
        log.info(
            "Loaded %s slice: %d rows, %d feature columns.",
            timeframe, len(features[timeframe]), features[timeframe].shape[1],
        )
    return features, targets
