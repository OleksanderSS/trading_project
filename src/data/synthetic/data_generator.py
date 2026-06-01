"""
Synthetic Data Generation Module

Responsible for generating synthetic training data for the trading pipeline.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger


class DataGenerator:
    """
    Synthetic data generator for trading pipeline.
    
    Generates realistic market data with various regimes and scenarios
    for model training and testing purposes.
    """

    def __init__(self, config_manager: UnifiedConfigManager):
        self.config_manager = config_manager
        self.logger = ProjectLogger.get_logger("DataGenerator")
        self.rng = np.random.default_rng(42)  # Reproducible results

    def generate_synthetic_data(self) -> dict[str, Any]:
        """
        Generate complete synthetic dataset
        
        Returns:
            Dictionary containing features and targets DataFrames
        """
        self.logger.info("Generating synthetic trading data...")

        features_df = self.generate_synthetic_features()
        targets_df = self.generate_synthetic_targets()

        return {
            'status': 'success',
            'features_df': features_df,
            'targets_df': targets_df,
            'message': 'Generated synthetic data successfully',
            'data_points': len(features_df),
            'features_count': len(features_df.columns),
            'targets_count': len(targets_df.columns)
        }

    def generate_synthetic_features(self) -> pd.DataFrame:
        """
        Generate synthetic features DataFrame
        
        Returns:
            DataFrame with technical indicators and market features
        """
        # Generate base price data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='1H')
        n_points = len(dates)

        # Generate price series with different regimes
        price_data = self._generate_price_series(n_points)

        # Calculate technical indicators
        features = pd.DataFrame(index=dates)
        features['close'] = price_data
        features['high'] = price_data * (1 + self.rng.uniform(0, 0.02, n_points))
        features['low'] = price_data * (1 - self.rng.uniform(0, 0.02, n_points))
        features['open'] = features['close'].shift(1).fillna(features['close'].iloc[0])
        features['volume'] = self.rng.lognormal(10, 1, n_points)

        # Add technical indicators
        features['sma_20'] = features['close'].rolling(window=20, min_periods=1).mean().shift(1)
        features['sma_50'] = features['close'].rolling(window=50, min_periods=1).mean().shift(1)
        features['ema_12'] = features['close'].ewm(span=12).mean().shift(1)
        features['ema_26'] = features['close'].ewm(span=26).mean().shift(1)

        # MACD
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean().shift(1)
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # RSI
        delta = features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean().shift(1)
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean().shift(1)
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma_20 = features['close'].rolling(window=20, min_periods=1).mean().shift(1)
        std_20 = features['close'].rolling(window=20, min_periods=2).std().shift(1)
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20

        # Volatility
        features['volatility'] = features['close'].pct_change(fill_method=None).fillna(0).rolling(window=20, min_periods=2).std().shift(1)


        # Price change features
        features['returns_1h'] = features['close'].pct_change(fill_method=None).fillna(0)
        features['returns_4h'] = features['close'].pct_change(4, fill_method=None).fillna(0)
        features['returns_24h'] = features['close'].pct_change(24, fill_method=None).fillna(0)

        # Time-based features
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['month'] = features.index.month
        features['quarter'] = features.index.quarter

        # Drop NaN values
        features = features.dropna()

        self.logger.info(f"Generated {len(features)} data points with {len(features.columns)} features")

        return features

    def generate_synthetic_targets(self) -> pd.DataFrame:
        """
        Generate synthetic targets DataFrame
        
        Returns:
            DataFrame with target variables for prediction
        """
        # Generate features first to align indices
        features_df = self.generate_synthetic_features()

        targets = pd.DataFrame(index=features_df.index)

        # Generate targets carefully: forward price / current price - 1
        # To avoid look-ahead bias, we ensure the index corresponds to the time of prediction
        close_prices = features_df['close']
        targets['return_1h'] = (close_prices.shift(-1) / close_prices) - 1  # audit-ignore: NEGATIVE_SHIFT_LOOKAHEAD
        targets['return_4h'] = (close_prices.shift(-4) / close_prices) - 1  # audit-ignore: NEGATIVE_SHIFT_LOOKAHEAD
        targets['return_24h'] = (close_prices.shift(-24) / close_prices) - 1  # audit-ignore: NEGATIVE_SHIFT_LOOKAHEAD
        
        # Drop rows where we don't have enough future data (the look-ahead window)
        targets = targets.dropna()
        features_df = features_df.loc[targets.index]

        # Direction targets (classification)
        targets['direction_1h'] = (targets['return_1h'] > 0).astype(int)
        targets['direction_4h'] = (targets['return_4h'] > 0).astype(int)
        targets['direction_24h'] = (targets['return_24h'] > 0).astype(int)

        # Volatility targets
        future_1h_returns = close_prices.pct_change(fill_method=None).shift(-1)  # audit-ignore: NEGATIVE_SHIFT_LOOKAHEAD
        targets['volatility_1h'] = future_1h_returns.abs()
        targets['volatility_4h'] = (
            future_1h_returns.iloc[::-1].rolling(window=4, min_periods=2).std().iloc[::-1]
        )

        # Trend strength
        targets['trend_strength'] = (
            (features_df['sma_20'] > features_df['sma_50']).astype(int) +
            (features_df['macd'] > features_df['macd_signal']).astype(int) +
            (features_df['rsi'] > 50).astype(int)
        ) / 3

        # Regime classification
        volatility = features_df['volatility']
        targets['regime'] = pd.cut(
            volatility,
            bins=[0, 0.01, 0.02, float('inf')],
            labels=['low_volatility', 'medium_volatility', 'high_volatility']
        ).astype('category')

        # Drop NaN values from future targets
        targets = targets.dropna()

        self.logger.info(f"Generated {len(targets)} target samples with {len(targets.columns)} targets")

        return targets

    def _generate_price_series(self, n_points: int) -> np.ndarray:
        """
        Generate realistic price series with market regimes
        
        Args:
            n_points: Number of data points to generate
            
        Returns:
            Array of price values
        """
        # Start with base price
        base_price = 100.0
        prices = [base_price]

        # Define market regimes
        regimes = [
            {'name': 'bull_market', 'probability': 0.3, 'trend': 0.0001, 'volatility': 0.015},
            {'name': 'bear_market', 'probability': 0.2, 'trend': -0.0001, 'volatility': 0.025},
            {'name': 'sideways', 'probability': 0.5, 'trend': 0.0, 'volatility': 0.010}
        ]

        current_regime = None
        regime_duration = 0

        for i in range(1, n_points):
            # Switch regimes occasionally
            if current_regime is None or regime_duration <= 0:
                current_regime = self.rng.choice(regimes, p=[r['probability'] for r in regimes])
                regime_duration = self.rng.integers(50, 200)  # Regime lasts 50-200 periods

            # Generate return based on current regime
            return_val = (
                current_regime['trend'] +
                self.rng.normal(0, current_regime['volatility'])
            )

            # Add some mean reversion
            if len(prices) > 20:
                avg_price = np.mean(prices[-20:])
                mean_reversion = 0.001 * (avg_price - prices[-1]) / prices[-1]
                return_val += mean_reversion

            # Calculate new price
            new_price = prices[-1] * (1 + return_val)
            prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative

            regime_duration -= 1

        return np.array(prices)
