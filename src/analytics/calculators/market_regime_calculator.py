import pandas as pd
import numpy as np
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)

class MarketRegimeCalculator:
    """
    Analyzes market regime based on price volatility and trend dynamics.
    The goal is to identify the dominant market state (e.g., Bull, Bear, Consolidation).
    """

    @staticmethod
    def calculate_regime(price_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Determines the market regime for each point in a time series.

        Args:
            price_series (pd.Series): Series of prices.
            window (int): Rolling window for calculations.

        Returns:
            pd.Series: A series of strings representing the market regime.
        """
        if not isinstance(price_series, pd.Series) or price_series.empty:
            logger.warning("Price series is empty or not a Series. Returning empty Series.")
            return pd.Series(dtype=str)

        logger.info(f"Calculating market regime with window size {window}...")

        # Calculate indicators
        rolling_std = price_series.pct_change(fill_method=None).fillna(0).rolling(window=window).std()
        rolling_mean = price_series.pct_change(fill_method=None).fillna(0).rolling(window=window).mean()

        # Define regime thresholds (these might be calibrated)
        volatility_threshold_high = rolling_std.quantile(0.75)
        volatility_threshold_low = rolling_std.quantile(0.25)
        trend_threshold_positive = rolling_mean.quantile(0.7)
        trend_threshold_negative = rolling_mean.quantile(0.3)

        # Classify regime
        regime = pd.Series('Consolidation', index=price_series.index)
        regime[ (rolling_mean > trend_threshold_positive) & (rolling_std > volatility_threshold_low) ] = 'Bullish'
        regime[ (rolling_mean < trend_threshold_negative) & (rolling_std > volatility_threshold_low) ] = 'Bearish'
        regime[ rolling_std > volatility_threshold_high ] = 'High Volatility'
        regime[ rolling_std < volatility_threshold_low ] = 'Low Volatility'
        
        logger.info("Market regime calculation completed.")
        return regime.fillna('Consolidation')

    @staticmethod
    def calculate_entropy(price_series: pd.Series, window: int = 50, num_bins: int = 10) -> pd.Series:
        """
        Calculates the rolling Shannon entropy of price returns.
        Entropy can be a measure of uncertainty or randomness in the market.

        Args:
            price_series (pd.Series): Series of prices.
            window (int): Rolling window for entropy calculation.
            num_bins (int): Number of bins to discretize returns.

        Returns:
            pd.Series: A series of entropy values.
        """
        if not isinstance(price_series, pd.Series) or price_series.empty:
            return pd.Series(dtype=float)
            
        returns = price_series.pct_change().dropna()
        
        def rolling_entropy(window_data):
            if len(window_data) < window * 0.8: # Require enough data
                return np.nan
            hist, _ = np.histogram(window_data, bins=num_bins, density=True)
            prob_dist = hist * np.diff(np.histogram_bin_edges(window_data, bins=num_bins))
            return entropy(prob_dist, base=2)

        result = returns.rolling(window=window).apply(rolling_entropy, raw=True)
        logger.info(f"Calculated entropy with window {window} and {num_bins} bins.")
        return result

    @staticmethod
    def calculate_reversal_probability(price_series: pd.Series, down_day_threshold: float = -0.01, window: int = 5) -> pd.Series:
        """
        Estimates the probability of a reversal after a sequence of down days.
        This is a heuristic-based indicator.

        Args:
            price_series (pd.Series): Series of prices.
            down_day_threshold (float): The return threshold to consider a "down day".
            window (int): How many consecutive days to look back.

        Returns:
            pd.Series: A series of estimated reversal probabilities.
        """
        if not isinstance(price_series, pd.Series) or price_series.empty:
            return pd.Series(dtype=float)

        returns = price_series.pct_change()
        is_down_day = (returns < down_day_threshold).astype(int)

        # Count consecutive down days
        consecutive_down_days = is_down_day.rolling(window=window).sum()

        # Simple probabilistic model: probability increases with more consecutive down days
        # This is a very basic model and can be significantly improved.
        base_prob = 0.1
        prob = base_prob + (consecutive_down_days / window) * 0.5
        
        # We only care about the probability at the end of a streak
        reversal_prob = prob.where(consecutive_down_days > 1, 0)
        
        logger.info(f"Calculated reversal probability with window {window}.")
        return reversal_prob.clip(0, 1)
