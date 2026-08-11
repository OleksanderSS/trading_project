"""An enricher may add columns. It may not silently permute rows.

Found by instrumenting the real pipeline: FIVE of the twenty enrichers
returned the same 24,395 rows in a different sequence -- macro_features,
nlp_features, keyword_entity, news_quality and sentiment_features. The
reordering itself is innocent (`merge_asof` demands sorted inputs; a
per-ticker groupby emits groups in key order); letting it escape is not,
because every consumer downstream assumes row i still describes bar i.

macro_features additionally dropped `datetime`, and that combination is
exactly what attached 54,000 bars to other days' dates in the 2026-08-06
training batch.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from src.features.feature_orchestrator import FeatureOrchestrator


class _Enricher:
    name = "test_enricher"


def _frame(n=6):
    return pd.DataFrame({
        "hash": [f"h{i}" for i in range(n)],
        "ticker": ["AAPL"] * (n // 2) + ["MSFT"] * (n - n // 2),
        "datetime": pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC"),
        "close": np.arange(100.0, 100.0 + n),
    })


def test_reordered_output_is_put_back_in_input_order():
    before = _frame()
    after = before.sort_values("close", ascending=False).copy()
    after["new_feature"] = 1.0

    restored = FeatureOrchestrator._restore_input_row_order(_Enricher(), before, after)

    assert list(restored["hash"]) == list(before["hash"])
    # The row's own values travel with it — this is a reorder, not a re-paste.
    assert list(restored["close"]) == list(before["close"])
    assert "new_feature" in restored.columns


def test_untouched_order_is_returned_unchanged():
    before = _frame()
    after = before.copy()

    restored = FeatureOrchestrator._restore_input_row_order(_Enricher(), before, after)

    assert list(restored["hash"]) == list(before["hash"])


def test_a_filtering_enricher_is_left_alone():
    """Dropping rows is an enricher's right; only permutation is corrected."""
    before = _frame()
    after = before.head(3).copy()

    restored = FeatureOrchestrator._restore_input_row_order(_Enricher(), before, after)

    assert len(restored) == 3


def test_without_a_usable_hash_nothing_is_guessed():
    before = _frame().drop(columns=["hash"])
    after = before.sort_values("close", ascending=False).copy()

    restored = FeatureOrchestrator._restore_input_row_order(_Enricher(), before, after)

    # Unchanged: with no identity column there is no safe way to realign.
    assert list(restored["close"]) == list(after["close"])


def test_duplicate_hashes_are_not_realigned():
    before = _frame()
    before.loc[1, "hash"] = "h0"
    after = before.sort_values("close", ascending=False).copy()

    restored = FeatureOrchestrator._restore_input_row_order(_Enricher(), before, after)

    assert list(restored["close"]) == list(after["close"])


def test_dropped_identity_column_is_reported(caplog):
    before = _frame()
    after = before.drop(columns=["datetime"])

    with caplog.at_level(logging.WARNING):
        FeatureOrchestrator._warn_if_row_identity_changed(_Enricher(), before, after)

    text = " ".join(record.getMessage() for record in caplog.records)
    assert "dropped identity column" in text
    assert "datetime" in text


def test_reordering_is_reported(caplog):
    before = _frame()
    after = before.sort_values("close", ascending=False)

    with caplog.at_level(logging.WARNING):
        FeatureOrchestrator._warn_if_row_identity_changed(_Enricher(), before, after)

    text = " ".join(record.getMessage() for record in caplog.records)
    assert "DIFFERENT ORDER" in text
    assert _Enricher.name in text
