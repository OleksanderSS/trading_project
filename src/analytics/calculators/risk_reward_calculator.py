import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

from .volatility_calculator import VolatilityCalculator

logger = logging.getLogger(__name__)

class RiskRewardCalculator:
    """A collection of static methods for calculating standard risk-reward metrics and dynamic trade parameters."""

    @staticmethod
    def calculate_trade_parameters(df: pd.DataFrame, 
                                 signal_type: str, 
                                 entry_price: float, 
                                 atr_multiplier: float = 2.0, 
                                 tp_multiplier: float = 3.0) -> Dict[str, float]:
        """
        Calculates dynamic Stop Loss and Take Profit levels based on market volatility (ATR).
        
        Args:
            df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns.
            signal_type (str): 'BUY' or 'SELL'.
            entry_price (float): Price at which the trade is entered.
            atr_multiplier (float): Multiplier for ATR to set Stop Loss.
            tp_multiplier (float): Multiplier for ATR to set Take Profit (or target ratio).

        Returns:
            Dict[str, float]: Dictionary containing stop_loss, take_profit, and risk_reward_ratio.
        """
        atr = VolatilityCalculator.calculate_atr(df, window=14).iloc[-1]
        
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.01 # Fallback to 1% of price
            
        risk = atr * atr_multiplier
        
        if signal_type == 'BUY':
            sl = entry_price - risk
            # Dynamic TP based on trend/volatility or simple multiplier of risk
            tp = entry_price + (risk * tp_multiplier)
        elif signal_type == 'SELL':
            sl = entry_price + risk
            tp = entry_price - (risk * tp_multiplier)
        else:
            return {'stop_loss': 0.0, 'take_profit': 0.0, 'risk_reward_ratio': 0.0}

        rr_ratio = abs(tp - entry_price) / abs(entry_price - sl) if abs(entry_price - sl) != 0 else 0.0
        
        return {
            'stop_loss': float(sl),
            'take_profit': float(tp),
            'risk_reward_ratio': float(rr_ratio),
            'risk_amount': float(risk)
        }

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Calculates the annualized Sharpe Ratio."""
        if returns.std() == 0:
            return np.nan
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe_ratio = excess_returns.mean() / excess_returns.std()
        annualized_sharpe = sharpe_ratio * np.sqrt(periods_per_year)
        return float(annualized_sharpe)

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Calculates the annualized Sortino Ratio."""
        target_return = risk_free_rate / periods_per_year
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) < 2:
            return np.nan

        downside_std = downside_returns.std()
        if downside_std == 0:
            excess_return = returns.mean() - target_return
            return np.inf if excess_return > 0 else 0.0

        expected_return = returns.mean()
        sortino_ratio = (expected_return - target_return) / downside_std
        annualized_sortino = sortino_ratio * np.sqrt(periods_per_year)
        return float(annualized_sortino)

    @staticmethod
    def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculates the Beta of an asset relative to the market."""
        common_index = asset_returns.dropna().index.intersection(market_returns.dropna().index)
        if len(common_index) < 2:
            return np.nan
            
        asset_returns = asset_returns.loc[common_index]
        market_returns = market_returns.loc[common_index]

        market_variance = market_returns.var()
        if market_variance == 0:
            return np.nan
            
        covariance = asset_returns.cov(market_returns)
        beta = covariance / market_variance
        return float(beta)

    @staticmethod
    def calculate_treynor_ratio(asset_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Calculates the Treynor Ratio."""
        beta = RiskRewardCalculator.calculate_beta(asset_returns, market_returns)
        if beta == 0 or pd.isna(beta):
            return np.nan

        excess_return = (asset_returns.mean() * periods_per_year) - risk_free_rate
        treynor_ratio = excess_return / beta
        return float(treynor_ratio)

    @staticmethod
    def calculate_var_cvar(returns: pd.Series, confidence_level: float = 0.95) -> dict:
        """Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR)."""
        if returns.empty:
            return {'var': np.nan, 'cvar': np.nan}
            
        quantile = 1 - confidence_level
        var = returns.quantile(quantile)
        cvar = returns[returns <= var].mean()
        return {'var': float(var), 'cvar': float(cvar)}

    @staticmethod
    def calculate_information_ratio(asset_returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculates the annualized Information Ratio."""
        active_returns = asset_returns - benchmark_returns
        tracking_error = active_returns.std()

        if tracking_error == 0:
            return np.inf if active_returns.mean() > 0 else 0.0
        
        information_ratio = active_returns.mean() / tracking_error
        annualized_ir = information_ratio * np.sqrt(periods_per_year)
        return float(annualized_ir)