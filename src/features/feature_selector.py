"""
Unified Feature Selector Facade
Provides a single entry point for feature selection across the entire system.
"""
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.selection.enhanced_smart_selector import EnhancedSmartFeatureSelector

logger = ProjectLogger.get_logger("FeatureSelector")

class FeatureSelector:
    """
    Unified entry point for feature selection.
    Delegates implementation to EnhancedSmartFeatureSelector.
    """
    def __init__(self, config: Any = None):
        """
        Initialize the unified feature selector.

        Args:
            config: Optional configuration object. If None, uses global config.
        """
        self.selector = EnhancedSmartFeatureSelector()
        self.logger = logger
        self.logger.info("✅ Unified FeatureSelector initialized with EnhancedSmartFeatureSelector")

    def select(self,
               features_df: pd.DataFrame,
               targets_df: pd.DataFrame,
               ticker: str,
               target_col: str,
               model_type: str = 'mlp',
               market_regime: str = 'normal') -> tuple[np.ndarray, list[str]]:
        """
        Unifies feature selection across assets and models.

        Args:
            features_df: Full features DataFrame.
            targets_df: Full targets DataFrame.
            ticker: Asset ticker to filter for.
            target_col: Name of the target column.
            model_type: Model identifier for selection parameters.
            market_regime: Detected market regime.

        Returns:
            Tuple of (selected_features_array, list_of_feature_names)
        """
        self.logger.info(f"🔍 Selecting features for {ticker} (Target: {target_col}, Model: {model_type})")

        # 1. Filter data for ticker
        ticker_features = self._filter_for_ticker(features_df, ticker)
        ticker_targets = self._filter_for_ticker(targets_df, ticker)

        if ticker_features.empty or ticker_targets.empty:
            self.logger.warning(f"⚠️ No data for {ticker}. Returning empty selection.")
            return np.array([]), []

        # 2. Get target series
        if target_col not in ticker_targets.columns:
            self.logger.error(f"❌ Target '{target_col}' not found for {ticker}")
            return np.array([]), []

        target_series = ticker_targets[target_col]

        # 3. Handle model-specific max features
        max_features = self.selector.get_model_max_features(model_type)

        # 4. Perform smart selection
        context_id = f"{ticker}_{target_col}_{model_type}"
        selected_names = self.selector.select(
            features_df=ticker_features,
            target_series=target_series,
            context_id=context_id,
            market_regime=market_regime,
            max_features=max_features
        )

        # 5. Prepare output
        if not selected_names:
            self.logger.warning(f"⚠️ Selection returned no features for {ticker}. Using fallback.")
            numeric_cols = ticker_features.select_dtypes(include=[np.number]).columns.tolist()
            selected_names = numeric_cols[:min(len(numeric_cols), 50)]

        selected_array = np.array(ticker_features[selected_names])

        self.logger.info(f"✅ Selected {len(selected_names)} features for {ticker}")
        return selected_array, selected_names

    def _filter_for_ticker(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Helper to filter DataFrame by ticker."""
        if 'ticker' in df.columns:
            return df[df['ticker'] == ticker].copy()
        elif hasattr(df.index, 'levels') and 'ticker' in df.index.names:
            try:
                return df.xs(ticker, level='ticker')
            except KeyError:
                pass
        return df
