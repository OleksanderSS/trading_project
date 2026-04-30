# src/analytics/calculators/market_regime_calculator.py
"""
Market Regime Calculator
Categorizes market states based on price volatility and trend dynamics.
Identifies states such as Bullish, Bearish, and Consolidation for strategy adaptation.
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class MarketRegimeCalculator:
    """
    Analyzes market regimes to determine the dominant environmental state.
    Facilitates regime-specific logic for algorithmic execution and risk management.
    """

    # Regime encoding map: label -> numeric identifier
    REGIME_ENCODING = {
        'High Volatility': 2,    # Maximum variance, elevated risk environment
        'Bullish': 1,            # Aggressive expansionary trend
        'Consolidation': 0,      # Neutral, sideways mean-reverting movement
        'Bearish': -1,           # Aggressive contractionary trend
        'Low Volatility': -2     # Stable, compressed variance environment
    }
    
    @staticmethod
    def calculate_regime(price_series: pd.Series, window: int = 20, return_encoded: bool = False) -> pd.Series:
        """
        Determines the current market regime for each observation in the time series.

        Args:
            price_series: Series of historical prices.
            window: Rolling window for volatility and trend estimation.
            return_encoded: Returns numeric identifiers if True, else text labels.

        Returns:
            Series of regime markers.
        """
        if not isinstance(price_series, pd.Series) or price_series.empty:
            logger.warning("Regime calculation skipped: Input series empty.")
            return pd.Series(dtype=str if not return_encoded else float)

        logger.info(f"Identifying market regimes using a {window}-period rolling window.")

        # Core Statistical Indicators
        returns = price_series.pct_change()
        rolling_std = returns.rolling(window=window).std()
        rolling_mean = returns.rolling(window=window).mean()

        # Dynamic Threshold Estimation (Percentile-based)
        vol_high = rolling_std.quantile(0.75)
        vol_low = rolling_std.quantile(0.25)
        trend_pos = rolling_mean.quantile(0.7)
        trend_neg = rolling_mean.quantile(0.3)

        # Vectorized Classification logic
        regime = pd.Series('Consolidation', index=price_series.index)
        regime[(rolling_mean > trend_pos) & (rolling_std > vol_low)] = 'Bullish'
        regime[(rolling_mean < trend_neg) & (rolling_std > vol_low)] = 'Bearish'
        regime[rolling_std > vol_high] = 'High Volatility'
        regime[rolling_std < vol_low] = 'Low Volatility'
        
        regime = regime.fillna('Consolidation')
        
        # Numeric transformation for ML inputs/monitoring
        if return_encoded:
            regime = regime.map(MarketRegimeCalculator.REGIME_ENCODING).fillna(0)
            logger.debug("Regime calculation finalized (numeric).")
        else:
            logger.debug("Regime calculation finalized (categorical).")
        
        return regime

    @staticmethod
    def calculate_entropy(price_series: pd.Series, window: int = 50, num_bins: int = 10) -> pd.Series:
        """
        Calculates rolling Shannon entropy to measure market uncertainty/disorder.

        Args:
            price_series: Historical price sequence.
            window: Rolling window for distribution estimation.
            num_bins: Histogram bins for return discretization.

        Returns:
            Series of entropy coefficients (measured in bits).
        """
        if not isinstance(price_series, pd.Series) or price_series.empty:
            return pd.Series(dtype=float)
            
        returns = price_series.pct_change().dropna()
        
        def _compute_entropy(window_slice):
            if len(window_slice) < window * 0.8:
                return np.nan
            hist, _ = np.histogram(window_slice, bins=num_bins, density=True)
            # Normalize to probability distribution for entropy calculation
            prob_dist = hist * np.diff(np.histogram_bin_edges(window_slice, bins=num_bins))
            return entropy(prob_dist, base=2)

        result = returns.rolling(window=window).apply(_compute_entropy, raw=True)
        logger.debug(f"Entropy estimation completed (window={window}, bins={num_bins}).")
        return result

    @staticmethod
    def calculate_reversal_probability(price_series: pd.Series, down_day_threshold: float = -0.01, window: int = 5) -> pd.Series:
        """
        Estimates local reversal probability following sequences of expansionary/contractionary days.

        Args:
            price_series: Historical price sequence.
            down_day_threshold: Return threshold defining a 'down' state.
            window: Lookback for streak identification.

        Returns:
            Series of probabilities [0, 1].
        """
        if not isinstance(price_series, pd.Series) or price_series.empty:
            return pd.Series(dtype=float)

        returns = price_series.pct_change()
        is_down = (returns < down_day_threshold).astype(int)

        # Identify consecutive streak length
        consecutive_down = is_down.rolling(window=window).sum()

        # Heuristic probabilistic model (increases with streak exhaustion)
        base_probability = 0.1
        probability_estimate = base_probability + (consecutive_down / window) * 0.5
        
        # Isolate probabilities for relevant streaks only
        reversal_series = probability_estimate.where(consecutive_down > 1, 0.0)
        
        logger.debug(f"Reversal probability estimated across {window}-day streaks.")
        return reversal_series.clip(0, 1)
