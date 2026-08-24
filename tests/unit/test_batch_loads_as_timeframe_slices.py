"""The batch loads as per-timeframe slices, and everything downstream copes.

The combined frame carries every timeframe's columns on every row: 154,069
daily rows holding 1,836 unused ones. Loading it costs 4.85 GiB of resident
memory against 0.27 GiB for the daily slice, and at 110 tickers that is roughly
24 GiB against 3 -- the difference between a wider universe being possible and
not.

`iter_model_contexts` has always accepted `DataFrame | dict[str, DataFrame]`.
Two places threw the shape away before it got there, and both are cheap to get
wrong in a way that reports success:

  the emptiness check   `getattr(data, "empty", True)` on a dict returns True,
                        so the cheap path would report "missing_features" --
                        a silent fallback rather than a readable error.
  the merge             it read `enriched_data.shape` and would raise on a dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.batch_timeframe_split import split_batch
from src.pipeline.hybrid_orchestrator import HybridOrchestrator
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator


def _batch(tmp_path):
    rows = []
    for timeframe, count in (("1d", 5), ("15m", 3)):
        for index in range(count):
            rows.append({
                "datetime": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index),
                "ticker": "AAPL",
                "interval": timeframe,
                "close": 100.0 + index,
                "only_daily_1d": index if timeframe == "1d" else np.nan,
                # Without a column belonging to the OTHER timeframe there is
                # nothing for the split to drop, and the test would pass on a
                # slice that is no narrower than the original.
                "only_intraday_15m": index if timeframe == "15m" else np.nan,
            })
    features = pd.DataFrame(rows)
    features.to_parquet(tmp_path / "features.parquet", index=False)
    targets = features[["datetime", "ticker", "interval"]].copy()
    targets["target_return_1d"] = 0.01
    targets.to_parquet(tmp_path / "targets.parquet", index=False)
    split_batch(tmp_path)
    return tmp_path


def test_slices_are_preferred_and_keyed_by_timeframe(tmp_path):
    import logging

    base = _batch(tmp_path)
    orchestrator = HybridOrchestrator.__new__(HybridOrchestrator)
    orchestrator.logger = logging.getLogger("probe")

    loaded = orchestrator._load_timeframe_slices(base)
    assert loaded is not None, "the slices were on disk and were not used"
    features, targets = loaded
    assert set(features) == {"1d", "15m"}
    assert len(features["1d"]) == 5 and len(features["15m"]) == 3
    assert features["1d"].shape[1] < pd.read_parquet(
        base / "features.parquet"
    ).shape[1], "the slice is no narrower than the combined frame"


def test_a_partial_set_of_slices_falls_back_rather_than_training_on_less(tmp_path):
    """Half the timeframes silently is worse than the combined file."""
    import logging

    base = _batch(tmp_path)
    (base / "targets_15m.parquet").unlink()

    orchestrator = HybridOrchestrator.__new__(HybridOrchestrator)
    orchestrator.logger = logging.getLogger("probe")
    loaded = orchestrator._load_timeframe_slices(base)

    assert loaded is not None
    features, _ = loaded
    assert set(features) == {"1d"}, (
        "a timeframe with no targets must not be presented as trainable"
    )


def test_unrelated_files_named_features_something_are_not_slices(tmp_path):
    """`features_all110_20260806.parquet` sits in the same directory."""
    import logging

    base = _batch(tmp_path)
    pd.DataFrame({"x": [1]}).to_parquet(
        base / "features_all110_20260806.parquet", index=False
    )
    pd.DataFrame({"x": [1]}).to_parquet(
        base / "targets_all110_20260806.parquet", index=False
    )

    orchestrator = HybridOrchestrator.__new__(HybridOrchestrator)
    orchestrator.logger = logging.getLogger("probe")
    features, _ = orchestrator._load_timeframe_slices(base)
    assert set(features) == {"1d", "15m"}, (
        f"a dated export was mistaken for a timeframe: {sorted(features)}"
    )


def test_the_merge_pairs_each_bar_with_its_own_outcome_per_timeframe():
    """Per timeframe, and still refusing to pair one bar with another's."""
    import logging

    orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orchestrator.logger = logging.getLogger("probe")

    features = pd.DataFrame({
        "ticker": ["AAPL"] * 3,
        "datetime": pd.date_range("2026-01-01", periods=3),
        "interval": ["1d"] * 3,
        "close": [1.0, 2.0, 3.0],
    })
    targets = features[["ticker", "datetime", "interval"]].copy()
    targets["target_return_1d"] = [0.1, 0.2, 0.3]

    merged = orchestrator._merge_features_and_targets(features, targets, label="1d")
    assert list(merged["target_return_1d"]) == [0.1, 0.2, 0.3]
    assert list(merged["close"]) == [1.0, 2.0, 3.0]


def test_a_reordered_target_frame_is_merged_on_keys_not_position():
    """The guarantee the extraction must not have lost.

    Enrichers in this pipeline do reorder -- "returned the same 28856 rows in a
    DIFFERENT ORDER" appears in the logs. Positional concat would survive that
    and pair each bar's features with another bar's outcome.
    """
    import logging

    orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orchestrator.logger = logging.getLogger("probe")

    features = pd.DataFrame({
        "ticker": ["AAPL"] * 3,
        "datetime": pd.date_range("2026-01-01", periods=3),
        "interval": ["1d"] * 3,
        "close": [1.0, 2.0, 3.0],
    })
    targets = features[["ticker", "datetime", "interval"]].copy()
    targets["target_return_1d"] = [0.1, 0.2, 0.3]
    shuffled = targets.iloc[::-1].reset_index(drop=True)

    merged = orchestrator._merge_features_and_targets(features, shuffled, label="1d")
    assert list(merged["target_return_1d"]) == [0.1, 0.2, 0.3], (
        "the reordered targets were concatenated positionally"
    )
