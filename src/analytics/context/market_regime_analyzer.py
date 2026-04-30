"""
Market regime analysis module for classifying market conditions.
"""


from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from src.core.logging.logger import ProjectLogger
from src.analytics.interfaces import IAnalyzer
from .market_regime_calculator import MarketRegimeCalculator

logger = ProjectLogger.get_logger(__name__)

class MarketRegimeAnalyzer(IAnalyzer):
    """
    Analyzes and classifies the market into regimes (e.g., Trend, Volatile, Range)
    by using calculated metrics and a set of configurable rules.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the analyzer with configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - 'window_size' (int): The main rolling window for calculations.
                - 'entropy_window' (int): Window for entropy calculation.
                - 'rules' (Dict): Thresholds for classifying regimes.
        """
        self.config = config or {}
        self.window_size = self.config.get('window_size', 20)
        self.entropy_window = self.config.get('entropy_window', 50)
        self.rules = self.config.get('rules', {
            'trend_threshold_multiplier': 2.0,
            'volatile_threshold_multiplier': 2.0,
            'range_volume_threshold': 0.1
        })
        logger.info("MarketRegimeAnalyzer initialized.")

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs the full market regime analysis.

        Args:
            data (pd.DataFrame): Input DataFrame containing at least
                               'close' and 'volume' columns.

        Returns:
            Dict[str, Any]: A dictionary containing the calculated
                              regimes and other metrics.
        """
        if not all(col in data.columns for col in ['close', 'volume']):
            logger.error(
                "Input data for MarketRegimeAnalyzer must contain "
                "'close' and 'volume' columns."
            )
            return {}

        # 1. Calculate all necessary indicators using the vectorized calculator
        regime_indicators = MarketRegimeCalculator.get_regime_indicators(
            data, window=self.window_size
        )

        # 2. Classify regime based on indicators and rules
        regimes = self._classify_regime(regime_indicators)

        # 3. (Optional) Calculate other metrics like entropy or reversal
        # probability if needed
        entropy = MarketRegimeCalculator.calculate_market_entropy(
            data['close'], window=self.entropy_window
        )
        reversal_prob = MarketRegimeCalculator.calculate_reversal_probability(
            data['close']
        )

        return {
            "market_regime": regimes,
            "market_entropy": entropy,
            "reversal_probability": reversal_prob,
            "regime_indicators": regime_indicators  # For debugging
        }

    def _classify_regime(self, indicators: pd.DataFrame) -> pd.Series:
        """
        Applies the configured rules to classify the market regime.
        """
        conditions = [
            (
                indicators['trend_strength'] >
                indicators['volatility'] * self.rules.get(
                    'trend_threshold_multiplier', 2
                )
            ),
            (
                indicators['volatility'] >
                indicators['trend_strength'] * self.rules.get(
                    'volatile_threshold_multiplier', 2
                )
            ),
            (
                indicators['volume_trend'].abs() <
                self.rules.get('range_volume_threshold', 0.1)
            ),
        ]

        choices = ['Trend', 'Volatile', 'Range']

        # numpy.select is a vectorized equivalent of if/elif/else
        regime_series = pd.Series(
            np.select(conditions, choices, default='Transition'),
            index=indicators.index
        )

        # Set initial periods to 'Unknown' as they are unreliable
        regime_series.iloc[:self.window_size] = 'Unknown'

        logger.info("Market regime classification complete.")
        return regime_series