"""Service columns must never be restored onto rows they don't belong to.

The bug this pins down destroyed the 2026-08-06 training batch. Stage 3's
`_restore_service_columns` copied `source_df[column].to_numpy()` onto the
enriched frame, guarded only by equal row counts. An enricher that returns the
same rows in a different order therefore got every restored value pasted onto
the wrong row.

Measured on the batch it produced, for AAPL 1d: all 327 exported bars are real
bars from the database (matched by their collector `hash`), every OHLCV field
and every calendar feature belongs to that real bar, and ZERO rows carry the
date the bar actually has -- offsets up to 686 days. `apply_guards` then sorted
by the corrupted date, so the file looked orderly while every indicator and
every shift(-n) target was computed across shuffled bars.
"""
import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.feature_engineering.orchestrator import FeatureEngineeringStage


@pytest.fixture
def stage():
    return object.__new__(FeatureEngineeringStage)


def _source(n=6):
    return pd.DataFrame({
        "datetime": pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC"),
        "ticker": ["AAPL"] * n,
        "close": np.arange(100.0, 100.0 + n),
        "volume": np.arange(1000, 1000 + n, dtype=float),
        "hash": [f"h{i}" for i in range(n)],
    })


def test_reordered_enricher_output_is_restored_by_hash_not_position(stage):
    """The real defect: same rows, different order, datetime dropped."""
    src = _source()
    # An enricher that sorted by close descending and dropped datetime.
    enriched = src.drop(columns=["datetime"]).sort_values(
        "close", ascending=False
    ).reset_index(drop=True)

    out = stage._restore_service_columns(enriched, src)

    # Every row must carry ITS OWN date, matched through the hash.
    expected = enriched["hash"].map(src.set_index("hash")["datetime"])
    assert list(out["datetime"]) == list(expected)

    # And specifically NOT the positional paste that caused the incident.
    positional = src["datetime"].to_numpy()
    assert not np.array_equal(out["datetime"].to_numpy(), positional)


def test_unchanged_order_still_restores(stage):
    src = _source()
    enriched = src.drop(columns=["datetime"]).copy()

    out = stage._restore_service_columns(enriched, src)

    assert list(out["datetime"]) == list(src["datetime"])


def test_reorder_without_hash_raises_instead_of_guessing(stage):
    """No hash and a moved anchor: refuse. A silent wrong date hid for weeks."""
    src = _source().drop(columns=["hash"])
    enriched = (
        src.drop(columns=["datetime"])
        .sort_values("close", ascending=False)
        .reset_index(drop=True)
    )

    with pytest.raises(ValueError, match="reordered its rows"):
        stage._restore_service_columns(enriched, src)


def test_no_anchor_at_all_raises(stage):
    src = _source().drop(columns=["hash"])
    enriched = pd.DataFrame({"rsi": np.arange(6.0)})

    with pytest.raises(ValueError, match="no shared anchor column"):
        stage._restore_service_columns(enriched, src)


def test_nothing_missing_is_a_no_op(stage):
    src = _source()
    enriched = src.copy()

    out = stage._restore_service_columns(enriched, src)

    pd.testing.assert_frame_equal(out, enriched)


def test_row_count_change_is_left_alone(stage):
    """A different length means the enricher filtered; not this method's job."""
    src = _source()
    enriched = src.drop(columns=["datetime"]).head(3)

    out = stage._restore_service_columns(enriched, src)

    assert "datetime" not in out.columns
