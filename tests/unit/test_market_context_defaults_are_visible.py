"""Three of eighteen market-context columns were constants nobody mentioned.

Measured on the 2026-08-13 batch, on all 25,172 15m rows:

    market_context_put_call_ratio    the constant 1.0
    market_context_dollar_strength   the constant 100.0

put_call_ratio because CBOE serves HTTP 403 and that collector is now off;
dollar_strength because no DXY or FRED_DTWEXBGS column reaches the enricher.
Each feature is filled with a neutral default when its source is absent, and
the default is the right call -- NaN would only move the same problem into
the imputer. What was missing is that nothing said so.

A constant column is inert for a model, so the cost is not today. It is the
first run where a feed does start arriving: the column then acquires a
discontinuity at the boundary between fabricated neutral and measured value,
with nothing marking where it falls.

Also removed here: `self.analyzer = MarketContextAnalyzer(...)`, described in
a comment as "kept available to callers that need a latest snapshot". No
caller ever read it. This enricher computes its own columns causally in
`_build_single_series_context`; the analyzer was construction cost and a
false impression that it participated.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.market_context_enricher import MarketContextEnricher


@pytest.fixture
def enricher():
    return MarketContextEnricher()


def _bars(n=60, **extra):
    frame = pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "datetime": pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC"),
        "close": np.linspace(100, 120, n),
        "volume": np.linspace(1e6, 2e6, n),
    })
    for name, values in extra.items():
        frame[name] = values
    return frame


def test_features_with_no_source_are_named_in_the_log(enricher, caplog):
    with caplog.at_level(logging.WARNING):
        enricher._enrich_impl(_bars())

    warnings = "\n".join(r.message for r in caplog.records)
    assert "filled entirely by their default" in warnings
    assert "put_call_ratio" in warnings
    assert "dollar_strength" in warnings


def test_a_feature_with_a_real_source_is_not_reported(enricher, caplog):
    """The warning has to distinguish absent from present, or it is noise."""
    with caplog.at_level(logging.WARNING):
        enricher._enrich_impl(_bars(put_call_ratio=np.linspace(0.8, 1.2, 60)))

    warnings = "\n".join(r.message for r in caplog.records)
    assert "put_call_ratio" not in warnings
    assert "dollar_strength" in warnings, "that one is still absent"


def test_the_default_still_fills_the_column(enricher):
    """Behaviour is unchanged: this commit adds visibility, not NaN."""
    enriched = enricher._enrich_impl(_bars())

    ratio = pd.to_numeric(enriched["market_context_put_call_ratio"], errors="coerce")
    assert ratio.notna().all()
    assert ratio.nunique() == 1
    assert float(ratio.iloc[0]) == pytest.approx(1.0)


def test_the_superseded_analyzer_is_no_longer_constructed(enricher):
    assert not hasattr(enricher, "analyzer"), (
        "the point-in-time analyzer was set and never read; this enricher "
        "builds its own causal context"
    )


def test_all_eighteen_context_columns_are_still_produced(enricher):
    enriched = enricher._enrich_impl(_bars())

    produced = [c for c in enriched.columns if c.startswith("market_context_")]
    assert len(produced) == len(enricher.context_features)
