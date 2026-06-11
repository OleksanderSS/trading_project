#!/usr/bin/env python3
"""
Market Conditions Analyzer - Regime Detection and Market Analysis
Handles market regime detection and market conditions calculation.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("MarketConditionsAnalyzer")


class MarketConditionsAnalyzer:
    """
    Market conditions analyzer.

    Handles:
    - Market regime detection
    - Volatility calculation
    - Trend calculation
    - Market conditions analysis
    """

    def __init__(self, regime_types: dict[str, Any] | None = None):
        """
        Initialize Market Conditions Analyzer.

        Args:
            regime_types: Dictionary of regime type configurations
        """
        self.logger = logger
        self.regime_types = regime_types or {
            'normal': {
                'description': 'Normal market conditions',
                'volatility_range': (0.01, 0.02),
                'trend_strength': (-0.001, 0.001)
            },
            'volatile': {
                'description': 'High volatility market',
                'volatility_range': (0.02, 0.05),
                'trend_strength': (-0.003, 0.003)
            },
            'trending_up': {
                'description': 'Strong uptrend market',
                'volatility_range': (0.015, 0.025),
                'trend_strength': (0.002, 0.005)
            },
            'trending_down': {
                'description': 'Strong downtrend market',
                'volatility_range': (0.015, 0.025),
                'trend_strength': (-0.005, -0.002)
            },
            'crisis': {
                'description': 'Market crisis conditions',
                'volatility_range': (0.04, 0.1),
                'trend_strength': (-0.01, 0.01)
            }
        }
        self.logger.info("✅ MarketConditionsAnalyzer initialized")

    def detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime based on market conditions."""
        try:
            volatility = self.calculate_volatility(market_data)
            trend = self.calculate_trend(market_data)

            for regime_name, regime_config in self.regime_types.items():
                vol_range = regime_config['volatility_range']
                trend_range = regime_config['trend_strength']

                if (vol_range[0] <= float(volatility) <= vol_range[1] and
                    trend_range[0] <= float(trend) <= trend_range[1]):
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f'Detected regime: {regime_name} (vol={volatility:.4f}, trend={trend:.4f})'
                        )
                    return regime_name

            return 'normal'
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error detecting market regime: {e}')
            return 'normal'

    def calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate market volatility."""
        try:
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change(fill_method=None).dropna()
                return float(returns.std() * np.sqrt(252))
            else:
                price_cols = [col for col in market_data.columns
                            if 'price' in col.lower() or col in ['open', 'high', 'low', 'close']]
                if price_cols:
                    returns = market_data[price_cols[0]].pct_change(fill_method=None).dropna()
                    return float(returns.std() * np.sqrt(252))
                return 0.02
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error calculating volatility: {e}')
            return 0.02

    def calculate_trend(self, market_data: pd.DataFrame) -> float:
        """Calculate market trend."""
        try:
            if 'close' in market_data.columns:
                recent_prices = market_data['close'].tail(20)
                if len(recent_prices) >= 2:
                    x = np.arange(len(recent_prices))
                    slope = np.polyfit(x, recent_prices, 1)[0]
                    normalized_trend = slope / recent_prices.mean()
                    return float(normalized_trend)
            return 0.0
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error calculating trend: {e}')
            return 0.0

    def calculate_market_conditions(self, market_data: pd.DataFrame) -> dict[str, float]:
        """Calculate comprehensive market conditions."""
        conditions = {}
        try:
            conditions['volatility'] = self.calculate_volatility(market_data)
            conditions['trend'] = self.calculate_trend(market_data)

            if 'volume' in market_data.columns:
                recent_volume = market_data['volume'].tail(10).mean()
                historical_volume = market_data['volume'].mean()
                conditions['volume_ratio'] = (recent_volume / historical_volume
                                           if historical_volume > 0 else 1.0)
            else:
                conditions['volume_ratio'] = 1.0

            if 'close' in market_data.columns:
                momentum_5d = (market_data['close'].iloc[-1] / market_data['close'].iloc[-6] - 1
                              if len(market_data) >= 6 else 0)
                momentum_20d = (market_data['close'].iloc[-1] / market_data['close'].iloc[-21] - 1
                               if len(market_data) >= 21 else 0)
                conditions['momentum_5d'] = momentum_5d
                conditions['momentum_20d'] = momentum_20d
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error calculating market conditions: {e}')

        return conditions
