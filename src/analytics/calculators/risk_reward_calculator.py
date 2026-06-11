"""
Risk Reward Calculator Proxy
Delegates core metrics to FinancialMetricsLibrary while keeping trade-specific logic.
"""
from dataclasses import dataclass

import pandas as pd

from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary

from .volatility_calculator import VolatilityCalculator


@dataclass
class TradeConfig:
    atr_multiplier: float = 2.0
    tp_multiplier: float = 3.0
    risk_free_rate: float = 0.0
    periods_per_year: int = 252
    confidence_level: float = 0.95

@dataclass
class TradeParameters:
    df: pd.DataFrame
    signal_type: str
    entry_price: float

class RiskRewardCalculator:
    """Proxy for risk-reward calculations."""

    @staticmethod
    def calculate_trade_parameters(trade_params: TradeParameters, config: TradeConfig | None = None) -> dict[str, float]:
        """Keeps local logic for SL/TP based on ATR."""
        if config is None: config = TradeConfig()
        atr = VolatilityCalculator.calculate_atr(trade_params.df).iloc[-1]
        if pd.isna(atr) or atr <= 0: atr = trade_params.entry_price * 0.01

        risk = atr * config.atr_multiplier
        if trade_params.signal_type == 'BUY':
            sl, tp = trade_params.entry_price - risk, trade_params.entry_price + (risk * config.tp_multiplier)
        elif trade_params.signal_type == 'SELL':
            sl, tp = trade_params.entry_price + risk, trade_params.entry_price - (risk * config.tp_multiplier)
        else: return {'stop_loss': 0.0, 'take_profit': 0.0, 'risk_reward_ratio': 0.0}

        return {
            'stop_loss': float(sl), 'take_profit': float(tp),
            'risk_reward_ratio': abs(tp - trade_params.entry_price) / risk if risk != 0 else 0.0,
            'risk_amount': float(risk)
        }

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, config: TradeConfig | None = None) -> float:
        c = config or TradeConfig()
        return FinancialMetricsLibrary.calculate_sharpe_ratio(returns, c.risk_free_rate, c.periods_per_year)

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, config: TradeConfig | None = None) -> float:
        c = config or TradeConfig()
        return FinancialMetricsLibrary.calculate_sortino_ratio(returns, c.risk_free_rate, c.periods_per_year)

    @staticmethod
    def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        return FinancialMetricsLibrary.calculate_beta(asset_returns, market_returns)

    @staticmethod
    def calculate_treynor_ratio(asset_returns: pd.Series, market_returns: pd.Series, config: TradeConfig | None = None) -> float:
        c = config or TradeConfig()
        return FinancialMetricsLibrary.calculate_treynor_ratio(asset_returns, market_returns, c.risk_free_rate, c.periods_per_year)

    @staticmethod
    def calculate_var_cvar(returns: pd.Series, config: TradeConfig | None = None) -> dict[str, float]:
        c = config or TradeConfig()
        return FinancialMetricsLibrary.calculate_var_cvar(returns, c.confidence_level)

    @staticmethod
    def calculate_information_ratio(asset_returns: pd.Series, benchmark_returns: pd.Series, config: TradeConfig | None = None) -> float:
        c = config or TradeConfig()
        return FinancialMetricsLibrary.calculate_information_ratio(asset_returns, benchmark_returns, c.periods_per_year)
