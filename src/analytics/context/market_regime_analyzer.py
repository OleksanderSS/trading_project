# src/analytics/context/market_regime_analyzer.py
"""
MarketRegimeAnalyzer - wrapper around MarketRegimeDetector that implements IAnalyzer.
Used by UnifiedAnalyticsEngine (Stage 7) via analysis.yaml registration.
"""

import logging
from typing import Any

import pandas as pd

from src.algorithms.regime_detector import MarketRegimeDetector
from src.analytics.interfaces import IAnalyzer

logger = logging.getLogger(__name__)


class MarketRegimeAnalyzer(IAnalyzer):
    """
    Analyzes price data to detect the current market regime.

    Wraps MarketRegimeDetector so it can be registered in UnifiedAnalyticsEngine
    via analysis.yaml and called uniformly through the IAnalyzer interface.

    Expected input (data): pd.DataFrame with at least a 'close' column.
    Returns: dict with 'regime', 'confidence', and supporting metrics.
    """

    def __init__(self, window_size: int = 20, entropy_window: int = 50):
        self.window_size = window_size
        self.entropy_window = entropy_window
        self._detector = MarketRegimeDetector()
        logger.info(f"MarketRegimeAnalyzer initialized (window={window_size}, entropy_window={entropy_window})")

    def analyze(self, data: Any, **kwargs) -> dict[str, Any]:
        """
        Detect market regime from price data.

        Args:
            data: pd.DataFrame with 'close' column, or dict with key 'price_data'.

        Returns:
            dict with keys: regime, confidence, volatility, mean_return, adx
        """
        df = self._resolve_dataframe(data)
        if df is None or df.empty:
            logger.warning("MarketRegimeAnalyzer: no valid price data received.")
            return self._empty_result()

        if "close" not in df.columns:
            logger.warning(f"MarketRegimeAnalyzer: 'close' column missing. Available: {df.columns.tolist()}")
            return self._empty_result()

        try:
            returns_series = df["close"].pct_change(fill_method=None).replace(
                [float("inf"), float("-inf")], pd.NA).dropna()
            returns = returns_series.values
            if len(returns) < 30:
                logger.warning(f"MarketRegimeAnalyzer: insufficient data ({len(returns)} rows, need ≥30).")
                return self._empty_result()

            data_bundle: dict[str, Any] = {
                "prices": df["close"].values,
            }
            if "volume" in df.columns:
                data_bundle["volume"] = df["volume"].values

            result = self._detector.detect_regime(
                returns=returns,
                data_bundle=data_bundle,
            )
            logger.info(
                f"MarketRegimeAnalyzer: regime={result.get('regime')}, confidence={result.get('confidence', 0):.3f}"
            )
            return result

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"MarketRegimeAnalyzer error: {e}", exc_info=True)
            return self._empty_result()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_dataframe(self, data: Any) -> pd.DataFrame | None:
        """Accept DataFrame directly or dict with 'price_data' key."""
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            for key in ("price_data", "prices", "market_data"):
                if key in data and isinstance(data[key], pd.DataFrame):
                    return data[key]
        return None

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "regime": "UNKNOWN",
            "confidence": 0.0,
            "volatility": None,
            "mean_return": None,
            "adx": None,
        }
