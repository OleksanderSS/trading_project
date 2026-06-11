"""
Market Context Enricher - adds macroeconomic and market indicators to the dataset.

Integrates MarketContextAnalyzer as an enricher to add context features
directly to the features DataFrame.
"""

from typing import Any

import pandas as pd

from src.analytics.context.market_context_analyzer import MarketContextAnalyzer
from src.core.logging.logger import ProjectLogger

from .base import BaseEnricher

logger = ProjectLogger.get_logger("MarketContextEnricher")

class MarketContextEnricher(BaseEnricher):
    """
    Adds macroeconomic and market indicators to the dataset.

    Uses MarketContextAnalyzer to calculate 18 context metrics:
    - Volatility metrics (5d, 20d, ratio)
    - Trend metrics (5d, 20d, alignment)
    - Technical indicators (RSI, volume ratio, price to MA20)
    - Temporal features (hour, day of week)
    - Macro indicators (yield curve, Fed Funds, market breadth, dollar strength, put/call ratio)
    """

    @property
    def name(self) -> str:
        return "market_context"

    @property
    def priority(self) -> int:
        return 85  # After context_map (80), before final enrichers

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()  # ✅ FIX: Initialize BaseEnricher
        self.config = config or {}

        # List of context features to calculate
        self.context_features = self.config.get('context_features', [
            # Volatility
            "volatility_5d", "volatility_20d", "volatility_ratio",
            # Trend
            "trend_5d", "trend_20d", "trend_alignment",
            # Technical
            "rsi_current", "volume_ratio", "price_to_ma20",
            # Temporal
            "hour_of_day", "day_of_week",
            # Macro (new)
            "yield_curve_slope", "yield_curve_inverted",
            "fed_funds_trend", "fed_funds_velocity",
            "market_breadth", "dollar_strength", "put_call_ratio"
        ])

        # Initialize analyzer
        self.analyzer = MarketContextAnalyzer(context_features=self.context_features)

        logger.info(f"MarketContextEnricher initialized with {len(self.context_features)} features")

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Adds market context features to DataFrame.

        Args:
            df: DataFrame with OHLCV and other features
            **kwargs: Additional parameters (VIX, DGS10, DXY, etc.)

        Returns:
            DataFrame with added market_context_* columns
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, skipping enrichment")
            return df

        result_df = df.copy()

        try:
            analysis_result = self.analyzer.analyze(df, **kwargs)
            context_vector = analysis_result.get('market_context_vector')

            if context_vector is not None:
                self._add_context_features(result_df, context_vector, df)
                self._log_latest_values(context_vector)
            else:
                logger.warning("⚠️ MarketContextAnalyzer returned None")

        except Exception as e:
            logger.error(f"❌ Failed to calculate market context: {e}", exc_info=True)

        return result_df

    def _add_context_features(self, result_df: pd.DataFrame, context_vector: dict[str, Any],
                             original_df: pd.DataFrame) -> None:
        """Додає контекстні характеристики до DataFrame."""
        for feature_name, feature_value in context_vector.items():
            col_name = f"market_context_{feature_name}"

            if feature_name in ['hour_of_day', 'day_of_week']:
                self._add_temporal_features(result_df, original_df, col_name, feature_name, feature_value)
            else:
                # For other features - forward-fill (they change rarely)
                result_df[col_name] = feature_value

        # ✅ FIX: Add volume_ratio if not present (required by targets.yaml)
        if 'market_context_volume_ratio' not in result_df.columns:
            if 'volume' in result_df.columns:
                # Calculate volume ratio: current volume / 20-day average volume
                result_df['market_context_volume_ratio'] = (
                    result_df['volume'] / result_df['volume'].rolling(20, min_periods=1).mean()
                )
                logger.info("✅ Added market_context_volume_ratio (volume / 20-day avg)")
            else:
                # Fallback: set to 1.0 if volume column is missing
                result_df['market_context_volume_ratio'] = 1.0
                logger.warning("⚠️ Volume column missing, set market_context_volume_ratio to 1.0")

        logger.info(f"✅ Added {len(context_vector)} market context features")

    def _add_temporal_features(self, result_df: pd.DataFrame, original_df: pd.DataFrame,
                              col_name: str, feature_name: str, feature_value: Any) -> None:
        """Додає тимчасові характеристики (hour, day_of_week)."""
        if isinstance(original_df.index, pd.DatetimeIndex):
            if feature_name == 'hour_of_day':
                result_df[col_name] = original_df.index.hour
            elif feature_name == 'day_of_week':
                result_df[col_name] = original_df.index.weekday
        else:
            result_df[col_name] = feature_value

    def _log_latest_values(self, context_vector: dict[str, Any]) -> None:
        """Логує останні значення для верифікації."""
        last_values = dict(context_vector.items())
        logger.debug(f"Latest market context: {last_values}")
