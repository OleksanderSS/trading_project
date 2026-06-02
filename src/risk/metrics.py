#!/usr/bin/env python3
"""
Risk metrics helpers extracted from KillSwitchManager.

Contains pure functions for calculating portfolio and position
metrics and analysing market conditions. These are intended to
be small, testable units called by the orchestration layer.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.utils.data_safety import safe_rolling
from src.utils.math_safe import safe_div

logger = ProjectLogger.get_logger("RiskMetrics")


def calculate_portfolio_returns(portfolio_data: dict[str, Any], market_data: pd.DataFrame) -> list[float]:
    """Calculate daily returns for portfolio assets and return a flattened list of returns."""
    try:
        # Support market_data passed as dict-like {'close': DataFrame} or as object with ['close']
        close_df = market_data["close"] if isinstance(market_data, dict) else market_data["close"]

        if not portfolio_data or close_df.empty:
            return []

        returns = []
        portfolio_symbols = list(portfolio_data.keys())

        for symbol in portfolio_symbols:
            if symbol in close_df.columns:
                symbol_returns = close_df[symbol].pct_change(fill_method=None).dropna()
                returns.extend(symbol_returns.tolist())

        return returns

    except Exception as e:
        logger.error(f"Error calculating portfolio returns: {e}", exc_info=True)
        raise RuntimeError("Failed to calculate portfolio returns") from e


def calculate_portfolio_metrics(portfolio_data: dict[str, Any], market_data: pd.DataFrame) -> dict[str, Any]:
    """Calculate portfolio-level risk metrics."""
    try:
        if not portfolio_data:
            return {}

        portfolio_value = sum(position.get("current_value", 0.0) for position in portfolio_data.values())

        portfolio_returns = calculate_portfolio_returns(portfolio_data, market_data)

        if len(portfolio_returns) < 2:
            return {"portfolio_value": portfolio_value, "daily_returns": []}

        daily_var = np.var(portfolio_returns) if len(portfolio_returns) > 1 else 0
        portfolio_volatility = np.sqrt(daily_var) * np.sqrt(252) if daily_var > 0 else 0

        cumulative_returns = np.cumprod(1 + np.array(portfolio_returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown_signed = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        max_drawdown_pct = abs(max_drawdown_signed)

        current_drawdown = 0.0
        if len(portfolio_returns) > 0:
            peak = float(np.max(np.maximum.accumulate(cumulative_returns)))
            current = float(cumulative_returns[-1])
            current_drawdown = (peak - current) / peak if peak > 0 else 0.0

        return {
            "portfolio_value": portfolio_value,
            "daily_returns": portfolio_returns,
            "daily_var": daily_var,
            "portfolio_volatility": portfolio_volatility,
            "max_drawdown": max_drawdown_signed,
            "max_drawdown_signed": max_drawdown_signed,
            "max_drawdown_pct": max_drawdown_pct,
            "current_drawdown": current_drawdown,
            "current_drawdown_pct": current_drawdown,
        }

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise RuntimeError("Failed to calculate portfolio metrics") from e


def calculate_position_metrics(portfolio_data: dict[str, Any], market_data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Calculate per-position risk metrics."""
    try:
        position_metrics = {}

        # Support market_data passed as dict-like {'close': DataFrame} or as object with ['close']
        close_df = market_data["close"] if isinstance(market_data, dict) else market_data["close"]

        for symbol, _position in portfolio_data.items():
            if symbol not in close_df.columns:
                continue

            symbol_prices = close_df[symbol]

            if len(symbol_prices) < 2:
                position_metrics[symbol] = {
                    "returns": [],
                    "volatility": 0.0,
                    "max_drawdown": 0.0,
                    "max_drawdown_signed": 0.0,
                    "max_drawdown_pct": 0.0,
                    "correlation_risk": 0.0,
                }
                continue

            symbol_returns = symbol_prices.pct_change(fill_method=None).dropna()

            volatility = symbol_returns.std() * np.sqrt(252)

            cumulative_returns = (1 + symbol_returns).cumprod()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown_signed = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            max_drawdown_pct = abs(max_drawdown_signed)

            current_drawdown = 0.0
            if len(symbol_returns) > 0:
                peak = float(np.max(np.maximum.accumulate(cumulative_returns)))
                current = float(cumulative_returns.iloc[-1])
                current_drawdown = (peak - current) / peak if peak > 0 else 0.0

            position_metrics[symbol] = {
                "returns": symbol_returns.tolist(),
                "volatility": volatility,
                "max_drawdown": max_drawdown_signed,
                "max_drawdown_signed": max_drawdown_signed,
                "max_drawdown_pct": max_drawdown_pct,
                "current_drawdown": current_drawdown,
                "current_drawdown_pct": current_drawdown,
            }

        return position_metrics

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise RuntimeError("Failed to calculate position metrics") from e


def analyze_market_conditions(market_data: pd.DataFrame) -> dict[str, Any]:
    """Analyze market conditions (volatility regime, trend, stress indicators)."""
    try:
        close_df = market_data["close"] if isinstance(market_data, dict) else market_data["close"]

        if close_df.empty:
            return {
                "volatility_regime": "unknown",
                "trend_regime": "unknown",
                "volatility_level": "unknown",
                "trend_strength": 0.0,
                "market_stress": False,
            }

        returns = close_df.pct_change(fill_method=None).dropna()

        # returns.std() returns a Series for multiple assets; reduce to a scalar
        volatility_series = returns.std()
        volatility = float(volatility_series.mean() * np.sqrt(252))

        if volatility < 0.01:
            volatility_regime = "low"
            volatility_level = "low"
        elif volatility < 0.02:
            volatility_regime = "normal"
            volatility_level = "normal"
        elif volatility < 0.04:
            volatility_regime = "elevated"
            volatility_level = "elevated"
        else:
            volatility_regime = "high"
            volatility_level = "high"

        prices = close_df
        short_ma = safe_rolling(prices, window=20, agg="mean")
        long_ma = safe_rolling(prices, window=50, agg="mean")

        if (long_ma > short_ma).any().any():
            trend_regime = "uptrend"
            trend_strength = (long_ma - short_ma) / short_ma
        elif (long_ma < short_ma).any().any():
            trend_regime = "downtrend"
            trend_strength = (short_ma - long_ma) / long_ma
        else:
            trend_regime = "sideways"
            trend_strength = 0.0

        recent_volatility = safe_rolling(returns, window=5, agg="std")
        historical_volatility = safe_rolling(returns, window=20, agg="std")

        # Determine market stress as a boolean (any asset showing a 2x spike)
        try:
            stress_mask = recent_volatility > (historical_volatility * 2)
            market_stress = bool(stress_mask.any().any())
        except Exception as e:
            logger.error(f"Error determining market stress: {e}", exc_info=True)
            market_stress = False

        # Volatility spike: take the maximum ratio across assets as a scalar
        try:
            ratio = (recent_volatility / historical_volatility).replace([float("inf"), -float("inf")], np.nan)
            finite_values = ratio.to_numpy(dtype=float)
            finite_values = finite_values[np.isfinite(finite_values)]
            volatility_spike = float(finite_values.max()) if finite_values.size > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating volatility spike: {e}", exc_info=True)
            volatility_spike = 0.0

        # Trend strength: mean relative MA difference across assets
        try:
            ma_diff = long_ma - short_ma
            if len(ma_diff) > 0:
                trend_strength = float(safe_div(ma_diff.mean(axis=1).iloc[-1], short_ma.mean(axis=1).iloc[-1]))
            else:
                trend_strength = 0.0
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}", exc_info=True)
            trend_strength = 0.0

        return {
            "volatility_regime": volatility_regime,
            "trend_regime": trend_regime,
            "volatility_level": volatility_level,
            "trend_strength": trend_strength,
            "market_stress": market_stress,
            "current_volatility": volatility,
            "historical_volatility": historical_volatility,
            "volatility_spike": volatility_spike,
        }

    except Exception as e:
        logger.error(f"Error analyzing market conditions: {e}", exc_info=True)
        raise RuntimeError("Failed to analyze market conditions") from e
