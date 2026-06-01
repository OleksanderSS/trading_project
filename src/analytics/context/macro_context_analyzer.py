from ..interfaces import IAnalyzer
import pandas as pd
from typing import Dict, Any, Optional
import logging

from ..calculators.macro_score_calculator import MacroScoreCalculator

logger = logging.getLogger(__name__)

class MacroContextAnalyzer(IAnalyzer):
    """
    Analyzes macroeconomic indicators to determine the overall market context or regime.
    This analyzer uses the MacroScoreCalculator to compute indicator scores.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initializes the analyzer with configuration for indicators and regime thresholds.

        Args:
            config (Dict[str, Any]): Configuration dictionary that contains:
                - 'indicators': A dictionary mapping indicator names to their configs.
                - 'regime_thresholds': Thresholds for classifying market regimes.
        """
        self.config = config or {}
        self.indicators_config = self.config.get('indicators', {})
        self.regime_thresholds = self.config.get('regime_thresholds', {})
        logger.info(f"MacroContextAnalyzer initialized for {len(self.indicators_config)} indicators.")

    def analyze(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates a weighted macro score and determines the market regime.

        Args:
            data (pd.DataFrame): DataFrame where each column is a macro indicator time series.
            **kwargs: Not used in this implementation.

        Returns:
            pd.DataFrame: DataFrame with the calculated 'macro_score' and 'market_regime'.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            logger.warning("Input data is not a valid DataFrame or is empty. Returning empty DataFrame.")
            return pd.DataFrame(columns=['macro_score', 'market_regime'])

        total_score = pd.Series(0.0, index=data.index)
        
        for indicator_name, indicator_cfg in self.indicators_config.items():
            if indicator_name in data.columns:
                indicator_series = data[indicator_name]
                score = MacroScoreCalculator.calculate_indicator_score(indicator_series, indicator_cfg)
                total_score += score
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Indicator '{indicator_name}' not found in data. Skipping.")
        
        result_df = pd.DataFrame(index=data.index)
        result_df['macro_score'] = total_score
        result_df['market_regime'] = self._get_regime(total_score)
        
        logger.info("Macro context analysis complete.")
        return result_df

    def _get_regime(self, score_series: pd.Series) -> pd.Series:
        """Classifies the macro score into a market regime."""
        
        # Define default thresholds if not provided
        defaults = {
            'strong_expansion': 0.5,
            'expansion': 0.2,
            'strong_contraction': -0.5,
            'contraction': -0.2
        }
        thresholds = {key: self.regime_thresholds.get(key, defaults[key]) for key in defaults}

        def classify(score):
            if score >= thresholds['strong_expansion']:
                return 'strong_expansion'
            elif score >= thresholds['expansion']:
                return 'expansion'
            elif score <= thresholds['strong_contraction']:
                return 'strong_contraction'
            elif score <= thresholds['contraction']:
                return 'contraction'
            else:
                return 'neutral'

        return score_series.apply(classify)
