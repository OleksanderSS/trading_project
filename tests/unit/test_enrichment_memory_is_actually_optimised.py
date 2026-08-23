"""The function named "optimize memory" was making a full deep copy.

`_optimize_dataframe_memory` read:

    if df.shape[1] > 100 and iteration % 3 == 0:
        df = df.copy()

-- a deep copy of the enrichment frame every third enricher, seven times over
twenty-two of them, each doubling the peak.

The intent was defensible. Pandas fragments a frame after hundreds of
single-column insertions, and its own PerformanceWarning recommends
`frame.copy()` to consolidate. The advice was taken and applied inside the
loop. Measured on 300 insertions into a 20,000-row frame:

    without the copies    0.23 s   peak  54 MiB   later op  41.8 ms
    copying every third   5.70 s   peak 213 MiB   later op  62.5 ms

Worse on all three axes, including the operation consolidation was supposed to
speed up, so there was no tradeoff to weigh.

What does reduce the frame is downcasting, which the function was not doing.
2,200 of the batch's 2,238 columns are float64 and account for 4.25 of its 4.36
GiB. Across all 2,181 with finite values the largest relative error from
float32 is 5.96e-08 -- float32's own epsilon -- and none overflows.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd

from src.features.feature_orchestrator import FeatureOrchestrator


def test_the_periodic_deep_copy_is_gone():
    """Read the function, not its name. The name was the misleading part."""
    import textwrap

    source = textwrap.dedent(
        inspect.getsource(FeatureOrchestrator._optimize_dataframe_memory)
    )
    copies = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy"
        and not any(keyword.arg == "deep" for keyword in node.keywords)
    ]
    assert not copies, (
        "a deep copy is back inside the memory-optimisation step; measured, "
        "it costs 25x the build time and 4x the peak, and even slows the "
        "operation the consolidation was meant to speed up"
    )


def test_downcasting_halves_the_frame_and_keeps_the_numbers():
    frame = pd.DataFrame({
        "price": np.random.uniform(10, 500, 5_000),
        "volume": np.random.uniform(1e6, 5e9, 5_000),
        "ratio": np.random.uniform(-1, 1, 5_000),
    })
    before = frame.memory_usage(deep=False).sum()
    original = frame.copy(deep=True)

    out = FeatureOrchestrator._downcast_float_columns(frame)
    after = out.memory_usage(deep=False).sum()

    assert after < before * 0.6, "the frame did not actually shrink"
    for column in original.columns:
        relative = (out[column].astype(np.float64) - original[column]).abs() / \
            original[column].abs().clip(lower=1e-12)
        assert relative.max() < 1e-6, (
            f"{column} lost more than 1e-6 of relative precision"
        )


def test_non_float_columns_are_left_alone():
    """Timestamps, tickers and integer flags must survive untouched."""
    frame = pd.DataFrame({
        "ticker": ["AAPL"] * 10,
        "datetime": pd.date_range("2024-01-01", periods=10),
        "flag": np.arange(10, dtype=np.int64),
        "value": np.random.rand(10),
    })
    out = FeatureOrchestrator._downcast_float_columns(frame)

    assert out["ticker"].dtype == object
    assert str(out["datetime"].dtype).startswith("datetime64")
    assert out["flag"].dtype == np.int64
    assert out["value"].dtype == np.float32


def test_a_frame_with_nothing_to_downcast_is_returned_unchanged():
    frame = pd.DataFrame({"a": np.arange(5, dtype=np.int32)})
    out = FeatureOrchestrator._downcast_float_columns(frame)
    assert out["a"].dtype == np.int32
