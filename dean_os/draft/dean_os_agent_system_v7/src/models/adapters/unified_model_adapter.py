"""
Unified Model Adapter
Consolidates logic from multiple legacy adapters (integrated, dynamic, simple).
Provides a robust bridge between FeatureSelector and ML models.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.features.feature_selector import FeatureSelector
from src.models import constants

logger = ProjectLogger.get_logger("UnifiedModelAdapter")


class UnifiedModelAdapter:
    """
    Unified adapter for model features.
    Handles selection, fallback heuristics, and formatting for specific model types.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config_manager = get_current_config()
        self.config = config or self.config_manager.get_config("models", {}).get("adapter", {})
        self.feature_selector = FeatureSelector()

        # Default target feature counts per model type (legacy '42' fallback)
        self.model_feature_counts = self.config.get(
            "feature_counts",
            {
                constants.MLP: 42,
                constants.LSTM: 42,
                constants.GRU: 42,
                constants.CNN: 42,
                constants.TRANSFORMER: 42,
                constants.CATBOOST: 64,
                constants.LIGHTGBM: 64,
                constants.XGBOOST: 64,
                constants.RANDOM_FOREST: 42,
                'ensemble': 42,
                'linear': 20,
            },
        )

        # Heuristic priority list for fallback selection
        self.priority_features = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "sma_5",
            "sma_10",
            "sma_20",
            "ema_5",
            "ema_10",
            "ema_20",
            "rsi_14",
            "rsi_7",
            "bb_upper",
            "bb_lower",
            "bb_middle",
            "atr_14",
            "macd",
            "macd_signal",
            "volume_sma_5",
            "volume_sma_10",
            "hour_of_day",
            "day_of_week",
            "month",
            "news_sentiment",
            "news_impact_score",
            "sentiment_score",
            "sentiment_ma_5",
            "price_change_1d",
            "price_change_5d",
            "volatility_5d",
            "volatility_20d",
            "momentum_5d",
            "momentum_10d",
            "trend_5d",
            "trend_20d",
            "state_market_regime",
            "state_volatility_regime",
        ]

    def get_features_for_model(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        ticker: str,
        target_col: str,
        model_type: str,
        market_regime: str = "normal",
    ) -> tuple[np.ndarray, list[str]]:
        """
        Primary entry point for getting adapted features for a specific model.

        Args:
            features_df: Full features DataFrame.
            targets_df: Full targets DataFrame (contains labels).
            ticker: Asset ticker.
            target_col: Target column name.
            model_type: Model architecture type.
            market_regime: Detected market regime.

        Returns:
            Tuple of (features_numpy_array, list_of_feature_names)
        """
        try:
            # 1. Use the Unified FeatureSelector (Phase 1)
            selected_array, selected_names = self.feature_selector.select(
                features_df=features_df,
                targets_df=targets_df,
                ticker=ticker,
                target_col=target_col,
                model_type=model_type,
                market_regime=market_regime,
            )

            # 2. If selection failed or returned too few features, use fallback
            target_count = self.model_feature_counts.get(model_type, 42)
            if len(selected_names) < 5:
                logger.warning(
                    f"⚠️ Feature selection returned too few features ({len(selected_names)}). Using heuristic fallback."
                )
                return self._fallback_selection(features_df, ticker, target_count)

            return selected_array, selected_names

        except Exception as e:
            logger.error(f"❌ Error in UnifiedModelAdapter: {e}", exc_info=True)
            raise DataProcessingError(f"Error in UnifiedModelAdapter: {e}") from e

    def _fallback_selection(self, df: pd.DataFrame, ticker: str, target_count: int) -> tuple[np.ndarray, list[str]]:
        """
        Heuristic-based fallback selection when smart selection fails.
        Uses a combination of priority features and high-variance numeric columns.
        """
        ticker_df = self._filter_for_ticker(df, ticker)
        if ticker_df.empty:
            return np.array([]), []

        selected = []
        # Add priority features if available
        for feature in self.priority_features:
            if feature in ticker_df.columns:
                selected.append(feature)
            else:
                # Try to find alternative via fuzzy match
                alt = self._find_alternative_feature(ticker_df, feature)
                if alt:
                    selected.append(alt)

        # If still need more, add high-variance numeric features
        if len(selected) < target_count:
            numeric_cols = ticker_df.select_dtypes(include=[np.number]).columns
            remaining = [
                c for c in numeric_cols if c not in selected and c not in ["datetime", "ticker", "hash", "interval"]
            ]

            variances = ticker_df[remaining].var().sort_values(ascending=False)
            additional = variances.index[: target_count - len(selected)].tolist()
            selected.extend(additional)

        final_list = selected[:target_count]
        return ticker_df[final_list].values, final_list

    def _find_alternative_feature(self, df: pd.DataFrame, target_feature: str) -> str | None:
        """Fuzzy match to find alternative features (e.g., 'close' -> 'adj_close')."""
        target_lower = target_feature.lower()
        for col in df.columns:
            col_str = str(col).lower()
            if target_lower in col_str or col_str in target_lower:
                return str(col)
        return None

    def _filter_for_ticker(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Filter DataFrame by ticker (supports column or multi-index)."""
        if "ticker" in df.columns:
            return df[df["ticker"] == ticker].copy()
        elif hasattr(df.index, "levels") and "ticker" in df.index.names:
            try:
                return df.xs(ticker, level="ticker")
            except KeyError as e:
                logger.debug(f"Ticker {ticker} not found in index, returning full dataframe: {e}")
        return df

    def adapt_for_heavy_model(self, X: np.ndarray, seq_len: int) -> np.ndarray:
        """
        Transforms 2D features into 3D sequences for LSTM/GRU/CNN.
        Delegates to optimized stride-based windowing.
        """
        if len(X) <= seq_len:
            return np.array([])

        shape = (X.shape[0] - seq_len, seq_len, X.shape[1])
        strides = (X.strides[0], X.strides[0], X.strides[1])
        return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
