"""One feature was 90% of the technical-analysis step and failed its own test.

Measured on 2026-08-28 from mid-frame cycles of the live run: 68.7 of 75.8
seconds across four tickers went to `_add_market_regime_features`. The cost is
real work -- `detect_regime` takes 23.69 ms and runs once per row, so a daily
ticker of 7,507 bars costs 178 seconds and the 110-name daily frame costs about
5.4 hours of a twelve-hour rebuild.

Computing it less often was measured away rather than assumed: the stored value
changes on 47.1% of consecutive rows, one change every 2.1 bars, across 213,120
distinct values. It is the detector's continuous confidence, not a slow label,
so sampling it would make a different feature rather than a cheaper one.

What decided it is that the leading-feature report returned "sign flipped out
of sample" for `MARKET_REGIME_ENCODED_1d` -- the direction reversed on the
holdout. And its one non-model consumer reads `MARKET_REGIME.iloc[-1]`: a
single cell, for five hours of computation.

So it is opt-in, and the consumer says so out loud instead of quietly deciding
every regime is normal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.technical_analysis_enricher import TechnicalAnalysisEnricher


@pytest.fixture
def enricher():
    made = TechnicalAnalysisEnricher.__new__(TechnicalAnalysisEnricher)
    made._calculators_loaded = False
    made.config = {}
    return made


def _frame(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, rows)))
    return pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=rows, freq="D"),
        "ticker": "AAPL",
        "close": close,
    })


def test_it_does_nothing_unless_asked(enricher, monkeypatch):
    monkeypatch.delenv("MARKET_REGIME_FEATURES", raising=False)
    frame = _frame()

    enricher._add_market_regime_features(frame, frame["close"].pct_change())

    assert "MARKET_REGIME" not in frame.columns
    assert "MARKET_REGIME_ENCODED" not in frame.columns


def test_it_still_works_when_asked(enricher, monkeypatch):
    """Off by default is not the same as removed."""
    monkeypatch.setenv("MARKET_REGIME_FEATURES", "1")
    enricher._load_calculators()
    frame = _frame()

    enricher._add_market_regime_features(frame, frame["close"].pct_change())

    assert "MARKET_REGIME" in frame.columns
    assert "MARKET_REGIME_ENCODED" in frame.columns
    assert frame["MARKET_REGIME"].notna().all()


@pytest.mark.parametrize("value", ["0", "", "no", "off"])
def test_only_an_explicit_yes_turns_it_on(enricher, monkeypatch, value):
    monkeypatch.setenv("MARKET_REGIME_FEATURES", value)
    frame = _frame()

    enricher._add_market_regime_features(frame, frame["close"].pct_change())

    assert "MARKET_REGIME" not in frame.columns


def test_the_selector_says_so_instead_of_deciding_quietly(caplog):
    """'normal' is a real answer; an absent column is not, and it must say so."""
    from src.features.selection.enhanced_smart_selector import EnhancedSmartFeatureSelector

    selector = EnhancedSmartFeatureSelector.__new__(EnhancedSmartFeatureSelector)
    import logging
    selector.logger = logging.getLogger("selector-test")

    with caplog.at_level(logging.WARNING, logger="selector-test"):
        resolved = selector._resolve_market_regime(pd.DataFrame({"close": [1.0]}))

    assert resolved == "normal"
    assert any("MARKET_REGIME" in record.message for record in caplog.records), (
        "the fallback happened without saying anything"
    )


def test_a_present_column_is_still_read(caplog):
    """The opt-in path must keep working end to end."""
    from src.features.selection.enhanced_smart_selector import EnhancedSmartFeatureSelector

    selector = EnhancedSmartFeatureSelector.__new__(EnhancedSmartFeatureSelector)
    import logging
    selector.logger = logging.getLogger("selector-test")

    frame = pd.DataFrame({"MARKET_REGIME": ["RANGING", "TRENDING_UP"]})
    assert selector._resolve_market_regime(frame) == "trending"

    frame = pd.DataFrame({"MARKET_REGIME": ["RANGING", "VOLATILE"]})
    assert selector._resolve_market_regime(frame) == "volatile"
