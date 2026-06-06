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

    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
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

    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
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

    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise RuntimeError("Failed to calculate position metrics") from e


def _determine_volatility_regime(volatility: float) -> tuple[str, str]:
    """Determine volatility regime and level from volatility value."""
    if volatility < 0.01:
        return "low", "low"
    elif volatility < 0.02:
        return "normal", "normal"
    elif volatility < 0.04:
        return "elevated", "elevated"
    else:
        return "high", "high"


def _determine_trend_regime(prices: pd.DataFrame | pd.Series) -> tuple[str, float]:
    """Determine trend regime and strength from price data."""
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    short_ma = safe_rolling(prices, window=20, agg="mean")
    long_ma = safe_rolling(prices, window=50, agg="mean")

    if len(prices) < 2:
        return "sideways", 0.0

    latest_short = short_ma.iloc[-1].mean()
    latest_long = long_ma.iloc[-1].mean()

    if latest_short > latest_long:
        trend_regime = "uptrend"
        trend_strength = float((latest_short - latest_long) / latest_long) if latest_long > 0 else 0.0
    elif latest_short < latest_long:
        trend_regime = "downtrend"
        trend_strength = float((latest_long - latest_short) / latest_short) if latest_short > 0 else 0.0
    else:
        trend_regime = "sideways"
        trend_strength = 0.0

    return trend_regime, trend_strength


def _calculate_market_stress(recent_volatility: pd.DataFrame, historical_volatility: pd.DataFrame) -> bool:
    """Calculate market stress indicator."""
    try:
        stress_mask = recent_volatility > (historical_volatility * 2)
        return bool(stress_mask.any().any())
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.error(f"Error determining market stress: {e}", exc_info=True)
        return False


def _calculate_volatility_spike(recent_volatility: pd.DataFrame, historical_volatility: pd.DataFrame) -> float:
    """Calculate volatility spike ratio."""
    try:
        ratio = (recent_volatility / historical_volatility).replace([float("inf"), -float("inf")], np.nan)
        finite_values = ratio.to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        return float(finite_values.max()) if finite_values.size > 0 else 0.0
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.error(f"Error calculating volatility spike: {e}", exc_info=True)
        return 0.0


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
        if isinstance(volatility_series, (pd.Series, pd.DataFrame)):
            volatility = float(volatility_series.mean() * np.sqrt(252))
        else:
            volatility = float(volatility_series * np.sqrt(252))

        volatility_regime, volatility_level = _determine_volatility_regime(volatility)
        trend_regime, trend_strength = _determine_trend_regime(close_df)

        recent_volatility = safe_rolling(returns, window=5, agg="std")
        historical_volatility = safe_rolling(returns, window=20, agg="std")

        market_stress = _calculate_market_stress(recent_volatility, historical_volatility)
        volatility_spike = _calculate_volatility_spike(recent_volatility, historical_volatility)

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

    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.error(f"Error analyzing market conditions: {e}", exc_info=True)
        raise RuntimeError("Failed to analyze market conditions") from e
