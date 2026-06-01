import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .volatility_calculator import VolatilityCalculator

logger = logging.getLogger(__name__)

@dataclass
class TradeConfig:
    """Configuration for trade parameter calculations."""
    atr_multiplier: float = 2.0
    tp_multiplier: float = 3.0
    risk_free_rate: float = 0.0
    periods_per_year: int = 252
    confidence_level: float = 0.95

@dataclass
class TradeParameters:
    """Parameters for a specific trade."""
    df: pd.DataFrame
    signal_type: str
    entry_price: float

class RiskRewardCalculator:
    """A collection of static methods for calculating standard risk-reward metrics and dynamic trade parameters."""

    @staticmethod
    def calculate_trade_parameters(trade_params: TradeParameters, config: Optional[TradeConfig] = None) -> Dict[str, float]:
        """
        Calculates dynamic Stop Loss and Take Profit levels based on market volatility (ATR).
        
        Args:
            trade_params (TradeParameters): Trade parameters including data, signal type, and entry price.
            config (Optional[TradeConfig]): Configuration for multipliers and settings.

        Returns:
            Dict[str, float]: Dictionary containing stop_loss, take_profit, and risk_reward_ratio.
        """
        if config is None:
            config = TradeConfig()
            
        return RiskRewardCalculator._calculate_trade_with_config(trade_params, config)

    @staticmethod
    def _calculate_trade_with_config(trade_params: TradeParameters, config: TradeConfig) -> Dict[str, float]:
        """Calculate trade parameters with given configuration."""
        atr = VolatilityCalculator.calculate_atr(trade_params.df, window=14).iloc[-1]
        
        if pd.isna(atr) or atr <= 0:
            atr = trade_params.entry_price * 0.01 # Fallback to 1% of price
            
        risk = atr * config.atr_multiplier
        
        if trade_params.signal_type == 'BUY':
            sl = trade_params.entry_price - risk
            tp = trade_params.entry_price + (risk * config.tp_multiplier)
        elif trade_params.signal_type == 'SELL':
            sl = trade_params.entry_price + risk
            tp = trade_params.entry_price - (risk * config.tp_multiplier)
        else:
            return {'stop_loss': 0.0, 'take_profit': 0.0, 'risk_reward_ratio': 0.0}

        rr_ratio = abs(tp - trade_params.entry_price) / abs(trade_params.entry_price - sl) if abs(trade_params.entry_price - sl) != 0 else 0.0
        
        return {
            'stop_loss': float(sl),
            'take_profit': float(tp),
            'risk_reward_ratio': float(rr_ratio),
            'risk_amount': float(risk)
        }

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, config: Optional[TradeConfig] = None) -> float:
        """Calculates the annualized Sharpe Ratio."""
        if config is None:
            config = TradeConfig()
            
        if returns.std() == 0:
            return np.nan
        
        excess_returns = returns - (config.risk_free_rate / config.periods_per_year)
        annualized_sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(config.periods_per_year)
        return float(annualized_sharpe)

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, config: Optional[TradeConfig] = None) -> float:
        """Calculates the annualized Sortino Ratio."""
        if config is None:
            config = TradeConfig()
            
        target_return = config.risk_free_rate / config.periods_per_year
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) < 2:
            return np.nan

        downside_std = downside_returns.std()
        if downside_std == 0:
            excess_return = returns.mean() - target_return
            return np.inf if excess_return > 0 else 0.0

        expected_return = returns.mean()
        sortino_ratio = (expected_return - target_return) / downside_std
        annualized_sortino = sortino_ratio * np.sqrt(config.periods_per_year)
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
    def calculate_treynor_ratio(asset_returns: pd.Series, market_returns: pd.Series, config: Optional[TradeConfig] = None) -> float:
        """Calculates the Treynor Ratio."""
        if config is None:
            config = TradeConfig()
            
        beta = RiskRewardCalculator.calculate_beta(asset_returns, market_returns)
        if beta == 0 or pd.isna(beta):
            return np.nan

        excess_return = (asset_returns.mean() * config.periods_per_year) - config.risk_free_rate
        treynor_ratio = excess_return / beta
        return float(treynor_ratio)

    @staticmethod
    def calculate_var_cvar(returns: pd.Series, config: Optional[TradeConfig] = None) -> dict:
        """Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR)."""
        if config is None:
            config = TradeConfig()
            
        if returns.empty:
            return {'var': np.nan, 'cvar': np.nan}
            
        quantile = 1 - config.confidence_level
        # audit-ignore: VAR_SIGN_OR_EMPTY_DATA_REVIEW
        var = returns.quantile(quantile)
        cvar = returns[returns <= var].mean()
        return {'var': float(var), 'cvar': float(cvar)}

    @staticmethod
    def calculate_information_ratio(asset_returns: pd.Series, benchmark_returns: pd.Series, config: Optional[TradeConfig] = None) -> float:
        """Calculates the annualized Information Ratio."""
        if config is None:
            config = TradeConfig()
            
        active_returns = asset_returns - benchmark_returns
        tracking_error = active_returns.std()

        if tracking_error == 0:
            return np.inf if active_returns.mean() > 0 else 0.0
        
        information_ratio = active_returns.mean() / tracking_error
        annualized_ir = information_ratio * np.sqrt(config.periods_per_year)
        return float(annualized_ir)