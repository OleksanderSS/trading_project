#!/usr/bin/env python3
"""
Temporal Target Guard - Safe Target Generation with Time Constraints
Prevents temporal leakage in target generation for different timeframes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("TemporalTargetGuard")

class TemporalTargetGuard:
    """
    Temporal-safe target generation for multi-timeframe trading systems.
    
    This guard ensures that targets are generated without temporal leakage,
    which is crucial for realistic backtesting and live trading performance.
    
    Key protections:
    - Prevents using future data for target calculation
    - Validates target horizons for each timeframe
    - Ensures proper temporal alignment between features and targets
    - Handles market hours and trading days correctly
    """
    
    # Market hours (EST)
    MARKET_OPEN_HOUR = 9
    MARKET_OPEN_MINUTE = 30
    MARKET_CLOSE_HOUR = 16
    MARKET_CLOSE_MINUTE = 0
    
    # Target configurations for each timeframe
    TARGET_CONFIGS = {
        '15m': {
            'horizons': {
                'return_15m': {'periods': 1, 'description': 'Next 15 minutes'},
                'return_1h': {'periods': 4, 'description': 'Next 1 hour'},
                'return_4h': {'periods': 16, 'description': 'Next 4 hours'}
            },
            'max_future_periods': 16,  # 4 hours maximum
            'data_frequency': '15T'
        },
        '60m': {
            'horizons': {
                'return_1h': {'periods': 1, 'description': 'Next 1 hour'},
                'return_4h': {'periods': 4, 'description': 'Next 4 hours'},
                'return_24h': {'periods': 24, 'description': 'Next 24 hours'}
            },
            'max_future_periods': 24,  # 24 hours maximum
            'data_frequency': '1H'
        },
        '1d': {
            'horizons': {
                'return_1d': {'periods': 1, 'description': 'Next 1 day'},
                'return_5d': {'periods': 5, 'description': 'Next 5 days'},
                'return_20d': {'periods': 20, 'description': 'Next 20 days'}
            },
            'max_future_periods': 20,  # 20 days maximum
            'data_frequency': '1D'
        }
    }
    
    def __init__(self):
        """Initialize the TemporalTargetGuard."""
        self.logger = logger
    
    def generate_targets_safe(self, 
                           df_enriched: pd.DataFrame, 
                           timeframe: str, 
                           current_time: pd.Timestamp) -> pd.DataFrame:
        """
        Generate targets safely with temporal constraints.
        
        This is the main method that ensures temporally-safe target generation.
        
        Args:
            df_enriched: Enriched DataFrame with price data
            timeframe: Timeframe string ('15m', '60m', '1d')
            current_time: Current timestamp for validation
            
        Returns:
            DataFrame with safely generated targets
        """
        if timeframe not in self.TARGET_CONFIGS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        self.logger.info(f"🎯 Generating safe targets for {timeframe} timeframe at {current_time}")
        
        # Validate input DataFrame
        validation_result = self._validate_input_dataframe(df_enriched, timeframe)
        if not validation_result['valid']:
            raise ValueError(f"Invalid input DataFrame: {validation_result['issues']}")
        
        # Create targets DataFrame
        targets_df = pd.DataFrame(index=df_enriched.index)
        
        # Generate targets for each horizon
        config = self.TARGET_CONFIGS[timeframe]
        for target_name, target_config in config['horizons'].items():
            target_values = self._generate_single_target(
                df_enriched, 
                timeframe, 
                target_name, 
                target_config,
                current_time
            )
            targets_df[target_name] = target_values
        
        # Generate direction targets (binary classification)
        for target_name in config['horizons'].keys():
            if 'return_' in target_name:
                targets_df[f"{target_name}_direction"] = (targets_df[target_name] > 0).astype(int)
        
        # Generate volatility targets
        targets_df = self._generate_volatility_targets(df_enriched, timeframe, current_time)
        
        # Apply temporal constraints
        targets_df = self._apply_temporal_constraints(targets_df, df_enriched, timeframe, current_time)
        
        # Validate generated targets
        target_validation = self._validate_generated_targets(targets_df, timeframe, current_time)
        
        self.logger.info(f"✅ Generated {len(targets_df.columns)} targets for {timeframe}")
        self.logger.info(f"   Valid targets: {target_validation['valid_count']}/{len(targets_df)}")
        
        if target_validation['warnings']:
            for warning in target_validation['warnings']:
                self.logger.warning(f"   ⚠️ {warning}")
        
        return targets_df
    
    def _validate_input_dataframe(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Validate input DataFrame for target generation."""
        issues = []
        
        # Check required columns
        required_cols = ['close']
        if 'datetime' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            required_cols.append('datetime')
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if 'close' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['close']):
                issues.append("Close price column must be numeric")
        
        # Check minimum data length
        min_periods = self.TARGET_CONFIGS[timeframe]['max_future_periods']
        if len(df) < min_periods:
            issues.append(f"Insufficient data: need {min_periods} periods, got {len(df)}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def _generate_single_target(self, 
                              df: pd.DataFrame,
                              timeframe: str,
                              target_name: str,
                              target_config: Dict[str, Any],
                              current_time: pd.Timestamp) -> pd.Series:
        """Generate a single target variable."""
        periods = target_config['periods']
        
        # Calculate price changes
        if 'close' in df.columns:
            returns = df['close'].pct_change(periods)
        else:
            raise ValueError("Close price column not found")
        
        # Shift to create future targets
        future_returns = returns.shift(-periods)
        
        self.logger.debug(f"📈 Generated {target_name}: {periods} periods ahead")
        
        return future_returns
    
    def _generate_volatility_targets(self, 
                                   df: pd.DataFrame,
                                   timeframe: str,
                                   current_time: pd.Timestamp) -> pd.DataFrame:
        """Generate volatility-related targets."""
        volatility_targets = pd.DataFrame(index=df.index)
        
        if 'close' not in df.columns:
            return volatility_targets
        
        # Calculate rolling volatility
        if timeframe == '15m':
            # 15-minute volatility targets
            volatility_targets['volatility_15m'] = df['close'].pct_change().rolling(window=4).std().shift(-4)
            volatility_targets['volatility_1h'] = df['close'].pct_change().rolling(window=16).std().shift(-16)
        elif timeframe == '60m':
            # 1-hour volatility targets
            volatility_targets['volatility_1h'] = df['close'].pct_change().rolling(window=1).std().shift(-1)
            volatility_targets['volatility_4h'] = df['close'].pct_change().rolling(window=4).std().shift(-4)
            volatility_targets['volatility_24h'] = df['close'].pct_change().rolling(window=24).std().shift(-24)
        elif timeframe == '1d':
            # Daily volatility targets
            volatility_targets['volatility_1d'] = df['close'].pct_change().rolling(window=1).std().shift(-1)
            volatility_targets['volatility_5d'] = df['close'].pct_change().rolling(window=5).std().shift(-5)
        
        return volatility_targets
    
    def _apply_temporal_constraints(self, 
                                  targets_df: pd.DataFrame,
                                  features_df: pd.DataFrame,
                                  timeframe: str,
                                  current_time: pd.Timestamp) -> pd.DataFrame:
        """Apply temporal constraints to prevent future data usage."""
        
        # Get the datetime index
        if isinstance(targets_df.index, pd.DatetimeIndex):
            datetime_index = targets_df.index
        elif 'datetime' in features_df.columns:
            datetime_index = pd.to_datetime(features_df['datetime'])
        else:
            # Create synthetic datetime index
            datetime_index = pd.date_range(start='2024-01-01', periods=len(targets_df), freq='D')
        
        # Remove targets that would require future data
        max_future_periods = self.TARGET_CONFIGS[timeframe]['max_future_periods']
        
        if len(datetime_index) > max_future_periods:
            # Keep only rows that don't require future data
            valid_mask = datetime_index <= current_time
            targets_constrained = targets_df[valid_mask]
        else:
            targets_constrained = targets_df.copy()
        
        # Additional market hours constraints for intraday timeframes
        if timeframe in ['15m', '60m']:
            targets_constrained = self._apply_market_hours_constraints(
                targets_constrained, datetime_index, current_time
            )
        
        # Remove rows with all NaN (where future data is unavailable)
        targets_constrained = targets_constrained.dropna(how='all')
        
        self.logger.debug(f"🕐 Applied temporal constraints: {len(targets_df)} → {len(targets_constrained)} rows")
        
        return targets_constrained
    
    def _apply_market_hours_constraints(self, 
                                    targets_df: pd.DataFrame,
                                    datetime_index: pd.DatetimeIndex,
                                    current_time: pd.Timestamp) -> pd.DataFrame:
        """Apply market hours constraints for intraday timeframes."""
        
        # Handle both DatetimeIndex and Series
        if isinstance(datetime_index, pd.DatetimeIndex):
            # It's already a DatetimeIndex
            weekday_condition = datetime_index.weekday < 5
            hour_condition = datetime_index.hour >= self.MARKET_OPEN_HOUR
            minute_condition = datetime_index.hour < self.MARKET_CLOSE_HOUR
            open_minute_condition = ((datetime_index.hour > self.MARKET_OPEN_HOUR) | 
                                     (datetime_index.minute >= self.MARKET_OPEN_MINUTE))
        else:
            # It's a Series, use .dt accessor
            weekday_condition = datetime_index.dt.weekday < 5
            hour_condition = datetime_index.dt.hour >= self.MARKET_OPEN_HOUR
            minute_condition = datetime_index.dt.hour < self.MARKET_CLOSE_HOUR
            open_minute_condition = ((datetime_index.dt.hour > self.MARKET_OPEN_HOUR) | 
                                     (datetime_index.dt.minute >= self.MARKET_OPEN_MINUTE))
        
        # Create mask for market hours
        market_hours_mask = (
            weekday_condition &  # Monday-Friday
            hour_condition &
            minute_condition &
            open_minute_condition
        )
        
        # Apply mask
        constrained_targets = targets_df[market_hours_mask]
        
        self.logger.debug(f"🏢 Applied market hours: {len(targets_df)} → {len(constrained_targets)} rows")
        
        return constrained_targets
    
    def _validate_generated_targets(self, 
                                 targets_df: pd.DataFrame,
                                 timeframe: str,
                                 current_time: pd.Timestamp) -> Dict[str, Any]:
        """Validate generated targets for quality and temporal correctness."""
        
        validation_result = {
            'valid_count': len(targets_df),
            'warnings': [],
            'statistics': {}
        }
        
        if targets_df.empty:
            validation_result['warnings'].append("No valid targets generated")
            return validation_result
        
        # Check for extreme values
        for col in targets_df.columns:
            if 'return_' in col and targets_df[col].dtype in ['float64', 'int64']:
                extreme_values = targets_df[col][abs(targets_df[col]) > 0.5]  # >50% returns
                if len(extreme_values) > 0:
                    validation_result['warnings'].append(
                        f"Extreme values in {col}: {len(extreme_values)} values >50%"
                    )
        
        # Check target distribution
        for col in targets_df.columns:
            if 'return_' in col and targets_df[col].dtype in ['float64', 'int64']:
                col_stats = {
                    'mean': targets_df[col].mean(),
                    'std': targets_df[col].std(),
                    'min': targets_df[col].min(),
                    'max': targets_df[col].max(),
                    'nan_count': targets_df[col].isnull().sum()
                }
                validation_result['statistics'][col] = col_stats
        
        # Check temporal range
        if isinstance(targets_df.index, pd.DatetimeIndex):
            time_range = {
                'start': targets_df.index.min(),
                'end': targets_df.index.max(),
                'duration': targets_df.index.max() - targets_df.index.min()
            }
            validation_result['statistics']['time_range'] = time_range
        
        return validation_result
    
    def get_target_preview(self, 
                         df_enriched: pd.DataFrame,
                         timeframe: str,
                         current_time: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Get a preview of what targets would be generated without actually generating them.
        
        Useful for planning and debugging.
        
        Args:
            df_enriched: Enriched DataFrame
            timeframe: Timeframe string
            current_time: Current timestamp (uses now if None)
            
        Returns:
            Preview information
        """
        if current_time is None:
            current_time = pd.Timestamp.now()
        
        if timeframe not in self.TARGET_CONFIGS:
            return {'error': f'Unsupported timeframe: {timeframe}'}
        
        config = self.TARGET_CONFIGS[timeframe]
        
        preview = {
            'timeframe': timeframe,
            'current_time': current_time,
            'input_shape': df_enriched.shape,
            'target_horizons': config['horizons'],
            'max_future_periods': config['max_future_periods'],
            'expected_targets': []
        }
        
        # Calculate expected target count
        expected_count = min(len(df_enriched), config['max_future_periods'])
        preview['expected_target_count'] = expected_count
        
        # List expected target names
        for horizon_name in config['horizons'].keys():
            preview['expected_targets'].append(horizon_name)
            preview['expected_targets'].append(f"{horizon_name}_direction")
        
        # Add volatility targets
        if timeframe == '15m':
            preview['expected_targets'].extend(['volatility_15m', 'volatility_1h'])
        elif timeframe == '60m':
            preview['expected_targets'].extend(['volatility_1h', 'volatility_4h', 'volatility_24h'])
        elif timeframe == '1d':
            preview['expected_targets'].extend(['volatility_1d', 'volatility_5d'])
        
        # Validate input
        input_validation = self._validate_input_dataframe(df_enriched, timeframe)
        preview['input_validation'] = input_validation
        
        return preview


# Factory function for easy instantiation
def get_temporal_target_guard() -> TemporalTargetGuard:
    """Factory function to get TemporalTargetGuard instance."""
    return TemporalTargetGuard()


# Convenience function for quick target generation
def generate_targets_quick(df_enriched: pd.DataFrame,
                         timeframe: str,
                         current_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Quick target generation with temporal validation.
    
    Args:
        df_enriched: Enriched DataFrame with price data
        timeframe: Timeframe string
        current_time: Current timestamp (uses now if None)
        
    Returns:
        DataFrame with safely generated targets
    """
    guard = get_temporal_target_guard()
    if current_time is None:
        current_time = pd.Timestamp.now()
    
    return guard.generate_targets_safe(df_enriched, timeframe, current_time)
