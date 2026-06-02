"""
Smart Missing Data Handler - Intelligent filling of missing values based on data type and temporal patterns.
Replaces zero-filling with context-aware interpolation and caching.
"""
import logging

from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("SmartMissingDataHandler")


class SmartMissingDataHandler:
    """
    Intelligent missing data handler that uses temporal patterns and data type awareness
    instead of destructive zero-filling.

    Key features:
    - Type-aware filling (price, volume, indicator, macro)
    - Temporal interpolation with distance weighting
    - Cache-based fallback for extended gaps
    - Anomaly detection for fill quality monitoring
    """

    def __init__(self):
        self.price_cache = {}  # Cache last known prices
        self.indicator_cache = {}  # Cache indicator states
        self.macro_cache = {}  # Cache macro values
        self.volume_cache = {}  # Cache volume patterns

        # Column type mappings
        self.column_types = {
            # Price columns
            "open": "price",
            "high": "price",
            "low": "price",
            "close": "price",
            # Volume columns
            "volume": "volume",
            "turnover": "volume",
            "liquidity": "volume",
            # Technical indicators
            "rsi": "indicator",
            "macd": "indicator",
            "signal": "indicator",
            "histogram": "indicator",
            "bb_upper": "indicator",
            "bb_middle": "indicator",
            "bb_lower": "indicator",
            "atr": "indicator",
            "sma": "indicator",
            "ema": "indicator",
            "wma": "indicator",
            "hma": "indicator",
            "tema": "indicator",
            "vwap": "indicator",
            "obv": "indicator",
            "mfi": "indicator",
            "roc": "indicator",
            "momentum": "indicator",
            "stoch_k": "indicator",
            "stoch_d": "indicator",
            "williams_r": "indicator",
            "cci": "indicator",
            "volatility": "indicator",
            "sharpe": "indicator",
            "sortino": "indicator",
            "drawdown": "indicator",
            "beta": "indicator",
            "alpha": "indicator",
            "correlation": "indicator",
            # Economic indicators
            "gdp": "macro",
            "inflation": "macro",
            "cpi": "macro",
            "unemployment": "macro",
            "fed_funds": "macro",
            "interest_rate": "macro",
            "yield_curve": "macro",
            "credit_spread": "macro",
            # Market regime
            "market_regime": "regime",
            "regime_confidence": "regime",
        }

        # Fill strategies for each type
        self.fill_strategies = {
            "price": self._fill_price_data,
            "volume": self._fill_volume_data,
            "indicator": self._fill_indicator_data,
            "macro": self._fill_macro_data,
            "regime": self._fill_regime_data,
        }

    def handle_missing_data(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Main method to handle missing data across all columns.

        Args:
            df: DataFrame with potential missing values
            verbose: Whether to log detailed information

        Returns:
            DataFrame with intelligently filled missing values
        """
        if verbose:
            logger.info(f"SmartMissingDataHandler: Processing {df.shape[0]} rows, {df.shape[1]} columns")

        # Detect column types
        column_type_map = self._detect_column_types(df.columns)

        # Apply appropriate fill strategy for each column
        filled_df = df.copy()
        missing_stats = {}

        for col in df.columns:
            if df[col].isna().any():
                col_type = column_type_map.get(col, "indicator")  # Default to indicator
                fill_strategy = self.fill_strategies.get(col_type, self._fill_indicator_data)

                # Apply fill strategy
                filled_col = fill_strategy(df[col], col)
                filled_df[col] = filled_col

                # Track statistics
                missing_count = df[col].isna().sum()
                missing_stats[col] = {
                    "type": col_type,
                    "missing_count": missing_count,
                    "fill_method": fill_strategy.__name__,
                }

        # Log summary
        if verbose and missing_stats:
            logger.info(f"SmartMissingDataHandler: Filled {len(missing_stats)} columns")
            for col, stats in missing_stats.items():
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"  {col}: {stats['missing_count']} missing, type={stats['type']}, method={stats['fill_method']}"
                    )

        # Detect fill anomalies
        anomalies = self._detect_fill_anomalies(filled_df, df)
        if anomalies and verbose:
            logger.warning(f"SmartMissingDataHandler: Detected {len(anomalies)} potential fill anomalies")
            for anomaly in anomalies[:5]:  # Show first 5
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"    {anomaly}")

        return filled_df

    def _detect_column_types(self, columns: list[str]) -> dict[str, str]:
        """Detect column types based on naming patterns."""
        column_type_map = {}

        for col in columns:
            col_lower = col.lower()

            # Check for exact matches first
            if col_lower in self.column_types:
                column_type_map[col] = self.column_types[col_lower]
                continue

            # Check for partial matches
            for pattern, col_type in self.column_types.items():
                if pattern in col_lower:
                    column_type_map[col] = col_type
                    break
            else:
                # Default to indicator for unknown columns
                column_type_map[col] = "indicator"

        return column_type_map

    def _fill_price_data(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Fill price data with forward fill + linear interpolation.
        Prices should maintain temporal continuity.
        """
        # Update cache with last known values
        if not series.empty and not pd.isna(series.iloc[-1]):
            self.price_cache[col_name] = series.iloc[-1]

        # Forward fill first (maintains last known price)
        filled = series.ffill()

        # Linear interpolation for remaining gaps (short gaps only)
        filled = filled.interpolate(method="linear", limit=10)

        # For remaining gaps, use cached value with decay
        remaining_na = filled.isna()
        if remaining_na.any():
            if col_name in self.price_cache:
                # Apply time decay to cached value
                decay_factor = 0.95  # Slight decay per period
                filled[remaining_na] = self.price_cache[col_name] * decay_factor
            else:
                # Use reasonable default based on column
                default_values = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
                default = default_values.get(col_name.split("_")[-1], 100.0)
                filled[remaining_na] = default

        return filled

    def _fill_volume_data(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Fill volume data with zero fill + seasonal patterns.
        Volume can legitimately be zero (no trading).
        """
        # Update cache with recent volume patterns
        if not series.empty and not pd.isna(series.iloc[-1]):
            self.volume_cache[col_name] = series.iloc[-1]

        # Zero is an explicit domain fallback for volume: no reported trading.
        filled = series.where(series.notna(), 0.0)

        # Apply seasonal smoothing if we have cache
        if col_name in self.volume_cache:
            # Smooth sudden zeros that might be data errors
            recent_avg = series.dropna().tail(20).mean()
            if recent_avg > 0:
                # Replace isolated zeros with small value.
                # FIX: Removed series.shift(-1) to prevent look-ahead bias.
                # Only check previous state or rely on interpolation.
                zero_mask = (filled == 0) & (series.shift(1) > 0)
                filled[zero_mask] = recent_avg * 0.1  # 10% of average

        return filled

    def _fill_indicator_data(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Fill indicator data with causal forward fill and neutral fallback.
        Indicators depend on historical calculations.
        """
        last_valid_values = series.dropna()

        # Forward fill first to avoid pulling future indicator values backward.
        filled = series.ffill(limit=5)

        # Extend short gaps causally; do not interpolate using future values.
        filled = filled.ffill(limit=7)

        # For remaining gaps, use cached value with mean reversion
        remaining_na = filled.isna()
        if remaining_na.any():
            if col_name in self.indicator_cache:
                cached_value = self.indicator_cache[col_name]

                # Apply mean reversion for oscillating indicators
                if any(indicator in col_name.lower() for indicator in ["rsi", "stoch", "williams"]):
                    # Oscillators: revert to 50 (neutral)
                    filled[remaining_na] = (cached_value + 50) / 2
                elif any(indicator in col_name.lower() for indicator in ["macd", "momentum", "roc"]):
                    # Momentum indicators: revert to 0
                    filled[remaining_na] = cached_value * 0.5
                else:
                    # Other indicators: use cached value
                    filled[remaining_na] = cached_value
            else:
                # Use reasonable defaults for different indicator types
                defaults = self._get_indicator_defaults(col_name)
                filled[remaining_na] = defaults.get(col_name.lower(), 0.0)

        if not last_valid_values.empty:
            self.indicator_cache[col_name] = last_valid_values.iloc[-1]

        return filled

    def _fill_macro_data(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Fill macro data with forward fill + interpolation + cache.
        Macro data changes slowly and has predictable patterns.
        """
        # Update cache with latest macro values
        if not series.empty and not pd.isna(series.iloc[-1]):
            self.macro_cache[col_name] = series.iloc[-1]

        # Forward fill (macro data persists until changed)
        filled = series.ffill()

        # Linear interpolation for short gaps
        filled = filled.interpolate(method="linear", limit=30)  # 30 days max

        # For remaining gaps, use cached value with time decay
        remaining_na = filled.isna()
        if remaining_na.any():
            if col_name in self.macro_cache:
                # Apply slight decay for long gaps
                decay_factor = 0.99  # Very slow decay for macro
                filled[remaining_na] = self.macro_cache[col_name] * decay_factor
            else:
                # Use realistic macro defaults
                macro_defaults = {
                    "gdp": 2.0,
                    "inflation": 2.5,
                    "cpi": 2.5,
                    "unemployment": 5.0,
                    "fed_funds": 2.0,
                    "interest_rate": 2.0,
                    "yield_curve": 1.0,
                    "credit_spread": 2.0,
                }

                # Find matching default
                for key, default in macro_defaults.items():
                    if key in col_name.lower():
                        filled[remaining_na] = default
                        break
                else:
                    filled[remaining_na] = 0.0  # Fallback

        return filled

    def _fill_regime_data(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Fill regime data with forward fill + confidence decay.
        Regime should persist but confidence may decay.
        """
        # Forward fill regime labels
        filled = series.ffill()

        # For regime confidence, apply decay over time
        if "confidence" in col_name.lower():
            remaining_na = filled.isna()
            if remaining_na.any():
                # Decay confidence for missing periods
                filled[remaining_na] = 0.5  # Neutral confidence
        else:
            # For regime labels, use 'UNKNOWN' for extended gaps
            remaining_na = filled.isna()
            if remaining_na.any():
                filled[remaining_na] = "UNKNOWN"

        return filled

    def _get_indicator_defaults(self, col_name: str) -> dict[str, float]:
        """Get default values for different indicator types."""
        defaults = {
            # Oscillators (0-100 range)
            "rsi": 50.0,
            "stoch": 50.0,
            "williams_r": -50.0,
            # Momentum indicators (centered around 0)
            "macd": 0.0,
            "signal": 0.0,
            "histogram": 0.0,
            "roc": 0.0,
            "momentum": 0.0,
            # Price-based indicators
            "sma": 100.0,
            "ema": 100.0,
            "wma": 100.0,
            "hma": 100.0,
            "tema": 100.0,
            "vwap": 100.0,
            # Volume indicators
            "obv": 0.0,
            "mfi": 50.0,
            # Volatility indicators
            "atr": 1.0,
            "volatility": 0.02,
            # Risk indicators
            "sharpe": 0.5,
            "sortino": 0.8,
            "drawdown": 0.0,
            # Correlation indicators
            "beta": 1.0,
            "alpha": 0.0,
            "correlation": 0.0,
            # Other indicators
            "cci": 0.0,
        }

        return defaults

    def _detect_fill_anomalies(self, filled_df: pd.DataFrame, original_df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Detect potential anomalies in filled data.
        """
        anomalies = []

        for col in filled_df.columns:
            # Check for sudden changes after fills
            if original_df[col].isna().any():
                # Find where original was NaN but filled is not
                fill_mask = original_df[col].isna() & filled_df[col].notna()

                if fill_mask.any():
                    # Calculate change magnitude
                    filled_values = filled_df[col][fill_mask]

                    # Check for extreme values
                    if len(filled_values) > 0:
                        mean_val = filled_values.mean()
                        std_val = filled_values.std()

                        # Flag if filled values are extreme
                        if abs(mean_val) > 3 * std_val:
                            anomalies.append(
                                {
                                    "column": col,
                                    "type": "extreme_fill",
                                    "mean_filled": mean_val,
                                    "std_filled": std_val,
                                    "severity": "high" if abs(mean_val) > 5 * std_val else "medium",
                                }
                            )

        return anomalies

    def get_fill_statistics(self, original_df: pd.DataFrame, filled_df: pd.DataFrame) -> dict[str, Any]:
        """
        Get statistics about the filling process.
        """
        stats = {
            "total_columns": len(original_df.columns),
            "columns_filled": 0,
            "total_missing_before": original_df.isna().sum().sum(),
            "total_missing_after": filled_df.isna().sum().sum(),
            "fill_efficiency": 0.0,
            "column_details": {},
        }

        for col in original_df.columns:
            missing_before = original_df[col].isna().sum()
            missing_after = filled_df[col].isna().sum()

            if missing_before > 0:
                stats["columns_filled"] += 1

                stats["column_details"][col] = {
                    "missing_before": missing_before,
                    "missing_after": missing_after,
                    "filled_count": missing_before - missing_after,
                    "fill_rate": (missing_before - missing_after) / missing_before if missing_before > 0 else 0,
                }

        if stats["total_missing_before"] > 0:
            stats["fill_efficiency"] = 1.0 - (stats["total_missing_after"] / stats["total_missing_before"])

        return stats
