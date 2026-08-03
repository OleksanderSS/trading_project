from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.enrichers.market_context_enricher import MarketContextEnricher
from src.features.enrichers.significance_features_enricher import (
    SignificanceFeaturesEnricher,
)
from src.features.enrichers.technical_analysis_enricher import (
    TechnicalAnalysisEnricher,
)


def test_market_context_features_are_prefix_invariant():
    rows = 80
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC")
    close = 100.0 + np.sin(np.arange(rows) / 4.0) + np.arange(rows) * 0.03
    frame = pd.DataFrame(
        {
            "datetime": timestamps,
            "ticker": ["NVDA"] * rows,
            "close": close,
            "volume": 1_000 + np.arange(rows) * 3,
            "RSI_14": 45.0 + np.sin(np.arange(rows) / 5.0) * 10.0,
            "FRED_DGS10": np.linspace(4.0, 4.2, rows),
            "FRED_GS2": np.linspace(4.3, 4.1, rows),
            "FRED_FEDFUNDS": np.full(rows, 4.5),
        }
    )
    enricher = MarketContextEnricher()

    full = enricher._enrich_impl(frame)
    prefix = enricher._enrich_impl(frame.iloc[:60].copy())
    context_columns = [
        column for column in prefix.columns if column.startswith("market_context_")
    ]

    pd.testing.assert_frame_equal(
        full.loc[:59, context_columns].reset_index(drop=True),
        prefix.loc[:, context_columns].reset_index(drop=True),
    )


def test_significance_features_are_prefix_invariant():
    returns = pd.Series(
        [np.nan, 0.01, -0.02, 0.015, 0.04, -0.01, 0.03] * 12,
        dtype=float,
    )
    frame = pd.DataFrame(
        {
            "ticker": ["NVDA"] * len(returns),
            "returns": returns,
        }
    )
    enricher = SignificanceFeaturesEnricher()

    full = enricher._enrich_impl(frame)
    prefix = enricher._enrich_impl(frame.iloc[:60].copy())
    significance_columns = [
        column
        for column in prefix.columns
        if "significant" in column or "significance" in column
    ]

    pd.testing.assert_frame_equal(
        full.loc[:59, significance_columns].reset_index(drop=True),
        prefix.loc[:, significance_columns].reset_index(drop=True),
    )


def test_market_regime_features_are_prefix_invariant():
    class DeterministicRegimeDetector:
        min_samples_for_clustering = 252

        @staticmethod
        def detect_regime(returns):
            mean = float(np.mean(returns))
            return {
                "regime": "UP" if mean >= 0 else "DOWN",
                "confidence": min(1.0, abs(mean) * 100.0),
            }

    returns = pd.Series(
        np.sin(np.arange(80) / 5.0) / 100.0,
        dtype=float,
    )
    full = pd.DataFrame({"close": 100.0 * (1.0 + returns).cumprod()})
    prefix = full.iloc[:60].copy()
    enricher = object.__new__(TechnicalAnalysisEnricher)
    enricher.regime_detector = DeterministicRegimeDetector()

    enricher._add_market_regime_features(full, returns=returns)
    enricher._add_market_regime_features(prefix, returns=returns.iloc[:60])

    assert full.loc[:59, "MARKET_REGIME"].tolist() == prefix["MARKET_REGIME"].tolist()
    np.testing.assert_allclose(
        full.loc[:59, "MARKET_REGIME_ENCODED"],
        prefix["MARKET_REGIME_ENCODED"],
        equal_nan=True,
    )
