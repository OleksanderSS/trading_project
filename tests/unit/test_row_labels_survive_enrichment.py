"""191 features lost because two tickers share a timestamp.

Several enrichers set `datetime` as their index on the way through --
macro_features first, then sentiment_features and hype_features -- and a
multi-ticker frame therefore comes back with duplicate labels. Measured
across a real chain on two tickers: 0 duplicates in, 147 after
macro_features, 299 from nlp_features onward.

ContextMapEnricher reindexes internally and cannot:

    ContextMapEnricher validation error: cannot reindex on an axis with
    duplicate labels
    Enricher 'context_map' completed: +0 columns in 0.10s

That was every timeframe of the 2026-08-13 rebuild, in the run where every
other enricher had just been repaired.

Fixing it inside each enricher fixes it once per enricher, and the next one
written reintroduces it. The rows are the same rows in the same order by
this point, so the orchestrator hands the labels back.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.feature_orchestrator import FeatureOrchestrator


class _Enricher:
    name = "test_enricher"


def _bars(tickers=("AAPL", "MSFT"), periods=5):
    stamps = pd.date_range("2026-07-01", periods=periods, freq="D", tz="UTC")
    return pd.DataFrame({
        "ticker": [t for t in tickers for _ in range(periods)],
        "datetime": list(stamps) * len(tickers),
        "close": np.arange(periods * len(tickers), dtype=float),
    })


def test_duplicate_labels_are_replaced_with_the_callers(  ):
    before = _bars()
    # What macro_features hands back: datetime as the index, shared across
    # tickers, so every label appears twice.
    after = before.set_index("datetime")
    assert after.index.has_duplicates

    restored = FeatureOrchestrator._restore_input_row_labels(
        _Enricher(), before, after
    )

    assert not restored.index.has_duplicates
    assert restored.index.equals(before.index)


def test_the_timestamps_are_not_destroyed_in_the_process():
    """Overwriting an index that HOLDS the datetime loses it outright.

    The next enricher then finds neither a column nor a DatetimeIndex and
    adds nothing -- which is how the first attempt at this fix broke the
    enricher after it.
    """
    before = _bars()
    after = before.drop(columns=["datetime"]).set_index(before["datetime"])

    restored = FeatureOrchestrator._restore_input_row_labels(
        _Enricher(), before, after
    )

    assert "datetime" in restored.columns
    assert pd.api.types.is_datetime64_any_dtype(restored["datetime"])
    assert list(restored["datetime"]) == list(before["datetime"])


def test_a_frame_that_is_already_fine_is_left_alone():
    before = _bars()
    after = before.assign(extra=1.0)

    restored = FeatureOrchestrator._restore_input_row_labels(
        _Enricher(), before, after
    )

    assert restored is after, "no copy, no work, when there is nothing to fix"


def test_a_changed_row_count_is_not_touched():
    """Different rows are a different question; guessing an alignment here
    is how columns get attached to the wrong bars."""
    before = _bars()
    after = before.head(3).set_index("datetime")

    restored = FeatureOrchestrator._restore_input_row_labels(
        _Enricher(), before, after
    )

    assert len(restored) == 3
    assert restored.index.name == "datetime"


def test_duplicates_the_caller_already_had_are_respected():
    """If the input itself carried duplicate labels, they are not ours to
    remove -- returning something the caller did not hand over is its own
    surprise."""
    before = _bars().set_index("datetime")
    after = before.assign(extra=1.0)

    restored = FeatureOrchestrator._restore_input_row_labels(
        _Enricher(), before, after
    )

    assert restored.index.has_duplicates
