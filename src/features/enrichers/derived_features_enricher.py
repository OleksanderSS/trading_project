import pandas as pd
import logging
from typing import List, Dict, Any

from src.config.unified_config_manager import get_current_config
from src.features.enrichers.base import BaseEnricher
from src.analytics.calculators.volatility_calculator import VolatilityCalculator

logger = logging.getLogger(__name__)

class DerivedFeaturesEnricher(BaseEnricher):
    """
    Enriches the DataFrame with derived features (lags, velocity, rolling stats, and forward-looking targets).
    """

    def __init__(self, target_column: str = 'close', returns_column: str = 'returns'):
        self.config = get_current_config().get('derived_features', {})
        # Default configuration if not found
        if not self.config:
            self.config = {
                'lags': [1, 5, 10, 20],
                'velocity': [1, 5, 10],
                'acceleration': [5, 10],
                'rolling_skew': [10, 20],
                'rolling_kurtosis': [10, 20],
                'rolling_volatility': [10, 20],
                'forward_targets': {
                    'periods': [1, 5, 10],
                    'include_returns': True,
                    'include_direction': True
                }
            }
            logger.info("Using default configuration for derived features.")
        self.target_column = target_column # Used for price-based features
        self.returns_column = returns_column # Used for returns-based features
        self.periods_per_year = get_current_config().get('market_data', {}).get('trading_days_per_year', 252)
        if not self.config:
            logger.warning("Configuration for derived features ('derived_features') not found.")

    @property
    def name(self) -> str:
        """Unique identifier for the enricher."""
        return "derived_features"

    @property
    def priority(self) -> int:
        """Execution order (Lower = earlier, Higher = later)."""
        return 25  # After technical (20), before NLP (30)

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Adds derived features and target labels to the DataFrame.

        Args:
            df: DataFrame with data. Must contain the target_column or returns_column.

        Returns:
            DataFrame with added features.
        """
        df_enriched = df.copy()
        
        # Use 'close' as default if not provided
        price_target_col = kwargs.get('target_column', self.target_column)
        if not isinstance(price_target_col, str) or price_target_col not in df_enriched.columns:
            # Fallback to 'close' if target_column is invalid
            price_target_col = 'close' if 'close' in df_enriched.columns else None
        
        if price_target_col and price_target_col in df_enriched.columns:
            logger.info(f"Generating price-based derived features on column '{price_target_col}'...")
            
            # ✅ Calculate returns if missing
            if self.returns_column not in df_enriched.columns:
                df_enriched[self.returns_column] = df_enriched[price_target_col].pct_change()
                logger.info(f"Calculated '{self.returns_column}' from '{price_target_col}'")
            
            if 'lags' in self.config:
                self._add_lags(df_enriched, price_target_col, self.config['lags'])
            if 'velocity' in self.config:
                self._add_velocity(df_enriched, price_target_col, self.config['velocity'])
            if 'acceleration' in self.config:
                self._add_acceleration(df_enriched, price_target_col, self.config['acceleration'])
            if 'rolling_skew' in self.config:
                self._add_rolling_stat(df_enriched, price_target_col, 'rolling_skew', self.config['rolling_skew'])
            if 'rolling_kurtosis' in self.config:
                self._add_rolling_stat(df_enriched, price_target_col, 'rolling_kurtosis', self.config['rolling_kurtosis'])
        else:
            logger.warning(f"Target column '{price_target_col}' not found. Skipping price-based derived features.")

        returns_col = kwargs.get('returns_column', self.returns_column)
        if isinstance(returns_col, str) and returns_col in df_enriched.columns:
            logger.info(f"Generating returns-based derived features on column '{returns_col}'...")
            if 'rolling_volatility' in self.config:
                self._add_rolling_stat(df_enriched, returns_col, 'rolling_volatility', self.config['rolling_volatility'])
            if 'forward_targets' in self.config:
                self._add_forward_targets(df_enriched, returns_col, self.config['forward_targets'])
        else:
            logger.warning(f"Returns column '{returns_col}' not found. Skipping returns-based features and targets.")

        logger.info(f"Derived features enrichment complete. Added {len(df_enriched.columns) - len(df.columns)} features.")
        return df_enriched

    def _add_lags(self, df: pd.DataFrame, target_col: str, lags: List[int]):
        for lag in lags:
            df[f'LAG_{lag}'] = df[target_col].shift(lag)

    def _add_velocity(self, df: pd.DataFrame, target_col: str, periods: List[int]):
        for p in periods:
            df[f'VELOCITY_{p}'] = df[target_col].diff(p)

    def _add_acceleration(self, df: pd.DataFrame, target_col: str, periods: List[int]):
        for p in periods:
            df[f'ACCELERATION_{p}'] = df[target_col].diff(p).diff(p)

    def _add_rolling_stat(self, df: pd.DataFrame, col: str, stat_name: str, windows: List[int]):
        for window in windows:
            if stat_name == 'rolling_volatility':
                df[f'ROLLING_VOL_{window}'] = VolatilityCalculator.calculate_rolling_volatility(df[col], window, self.periods_per_year)
            elif stat_name == 'rolling_skew':
                df[f'ROLLING_SKEW_{window}'] = df[col].rolling(window=window).skew()
            elif stat_name == 'rolling_kurtosis':
                df[f'ROLLING_KURT_{window}'] = df[col].rolling(window=window).kurt()
    
    def _add_forward_targets(self, df: pd.DataFrame, returns_col: str, config: Dict[str, Any]):
        """Adds forward-looking returns and direction as target labels."""
        periods = config.get('periods', [])
        if not periods: return

        price = (1 + df[returns_col]).cumprod()
        for p in periods:
            if p <= 0: continue
            forward_price = price.shift(-p)
            forward_returns = (forward_price - price) / price
            if config.get('include_returns', True):
                df[f'TARGET_RETURN_{p}P'] = forward_returns
            if config.get('include_direction', True):
                df[f'TARGET_DIRECTION_{p}P'] = (forward_returns > 0).astype(int)