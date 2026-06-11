#!/usr/bin/env python3
"""
Advanced Simulation Engine - Розширена система бектестингу

Включає:
- Walk-Forward Optimization
- Transaction Cost Modeling
- Bias Detection (Look-ahead, Survivorship)
- Portfolio Backtesting with Multi-Assets
- Stable RNG for Parallel Simulations
"""

import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.metrics.calculator import MetricsCalculator
from src.trading.virtual_portfolio import VirtualPortfolio

logger = ProjectLogger.get_logger("SimulationEngine")

# Configuration constants
RANDOM_SEED_CONFIG_KEY = 'performance.random_seed'
DEFAULT_RANDOM_SEED = 42

class SimulationGranularity(Enum):
    TICKER_LEVEL = "ticker"
    SECTOR_LEVEL = "sector"
    MARKET_LEVEL = "market"

@dataclass
class SimulationContext:
    ticker: str
    timestamp: datetime
    granularity: SimulationGranularity
    features: dict[str, Any] = field(default_factory=dict)
    market_conditions: dict[str, Any] = field(default_factory=dict)
    historical_returns: pd.Series | None = None

@dataclass
class SimulationRiskReport:
    ticker: str
    var_95: float
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    max_drawdown: float

class SimulationEngine:
    def __init__(self, max_workers: int = None):
        self.logger = ProjectLogger.get_logger("SimulationEngine")
        self.max_workers = max_workers or min(mp.cpu_count(), 4)
        self.config = get_current_config()
        self.sim_config = self.config.get('simulation', {})
        self.optimization_config = self.sim_config.get('optimization', {})
        self.defaults_config = self.sim_config.get('defaults', {})
        self.slippage_pct = self.config.get('backtest.transaction_costs.slippage_pct', 0.0005)

    def run_monte_carlo_for_strategy(
        self,
        strategy_logic: Callable[[pd.DataFrame], pd.Series],
        initial_context: SimulationContext,
        horizon: int,
        runs: int = None
    ) -> list[SimulationRiskReport]:
        runs = runs or self.defaults_config.get('monte_carlo_runs', 1000)

        reports = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._run_single_path_simulation, strategy_logic, initial_context, horizon) for _ in range(runs)]
            for future in futures:
                report = future.result()
                if report:
                    reports.append(report)
        return reports

    def _run_single_path_simulation(
        self,
        strategy_logic: Callable[[pd.DataFrame], pd.Series],
        context: SimulationContext,
        horizon: int
    ) -> SimulationRiskReport | None:
        try:
            # 1. Generate synthetic market data path
            price_path_df = self._generate_price_path(context, horizon)

            # 2. Run the strategy logic on the generated data
            signals = strategy_logic(price_path_df)

            # 3. Simulate trading path in-memory
            cash = 100000.0
            shares = 0.0
            equity_history = []
            
            commission_pct = self.config.get('backtest.transaction_costs.commission_pct', 0.001)

            for timestamp, row in price_path_df.iterrows():
                close_price = float(row['close'])
                signal = signals.get(timestamp)

                if signal == 1 and cash >= 10 * close_price:
                    # Buy 10 shares
                    cost = 10 * close_price
                    tx_cost = commission_pct * cost
                    cash -= (cost + tx_cost)
                    shares += 10
                elif signal == -1 and shares >= 10:
                    # Sell 10 shares
                    revenue = 10 * close_price
                    tx_cost = commission_pct * revenue
                    cash += (revenue - tx_cost)
                    shares -= 10

                equity_history.append(cash + shares * close_price)

            equity_curve = pd.Series(equity_history, index=price_path_df.index)
            returns = equity_curve.pct_change().dropna()

            # 4. Calculate risk metrics using unified FinancialMetricsLibrary
            from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary
            lib = FinancialMetricsLibrary

            var_cvar_95 = lib.calculate_var_cvar(returns, confidence_level=0.95)
            var_cvar_99 = lib.calculate_var_cvar(returns, confidence_level=0.99)

            report = SimulationRiskReport(
                ticker=context.ticker,
                var_95=float(var_cvar_95['var']),
                var_99=float(var_cvar_99['var']),
                expected_shortfall=float(var_cvar_95['cvar']),
                sharpe_ratio=float(lib.calculate_sharpe_ratio(returns, risk_free_rate=0.02, trading_days_per_year=252)),
                max_drawdown=float(lib.calculate_max_drawdown(equity_curve))
            )
            return report
        except Exception as e:
            self.logger.error(f"Error in single path simulation: {e}", exc_info=True)
            return None

    def _generate_price_path(self, context: SimulationContext, horizon: int) -> pd.DataFrame:
        """Generates a DataFrame with OHLCV data for a single path."""
        market_params = self._extract_market_parameters(context)
        daily_returns = self._generate_daily_returns(context, horizon, market_params)
        dates = self._generate_dates(context.timestamp, horizon)
        prices = self._calculate_price_path(market_params['current_price'], daily_returns)
        return self._create_ohlcv_dataframe(prices, daily_returns, dates)

    def _extract_market_parameters(self, context: SimulationContext) -> dict:
        return {
            'current_price': context.features.get('close', 100),
            'volatility': context.market_conditions.get('volatility', 0.02),
            'trend': context.market_conditions.get('trend', 0)
        }

    def _ensure_determinism(self) -> np.random.Generator:
        """Returns a thread-safe deterministic random number generator."""
        seed = self.config.get(RANDOM_SEED_CONFIG_KEY, DEFAULT_RANDOM_SEED)
        return np.random.default_rng(seed)

    def _generate_bootstrap_returns(self, context: SimulationContext, horizon: int) -> np.ndarray:
        """Generate returns using historical bootstrap."""
        rng = self._ensure_determinism()
        hist_rets = context.historical_returns.values
        return rng.choice(hist_rets, size=horizon, replace=True)

    def _generate_t_distribution_returns(self, horizon: int, market_params: dict) -> np.ndarray:
        """Generate returns using t-distribution."""
        rng = self._ensure_determinism()
        df = self.optimization_config.get('t_distribution_df', 3.0)
        scale = self._calculate_t_distribution_scale(market_params['volatility'], df)
        # Using rng for thread-safe random state
        seed_val = int(rng.integers(0, 2**32))
        return stats.t.rvs(df, loc=market_params['trend'] / horizon, scale=scale, size=horizon, random_state=seed_val)

    def _calculate_t_distribution_scale(self, volatility: float, df: float) -> float:
        return volatility * np.sqrt((df - 2) / df) if df > 2 else volatility

    def _generate_dates(self, start_timestamp: datetime, horizon: int) -> pd.DatetimeIndex:
        return pd.to_datetime([start_timestamp + timedelta(days=i) for i in range(horizon)])

    def _calculate_price_path(self, current_price: float, daily_returns: np.ndarray) -> list:
        prices = [current_price]
        for r in daily_returns[:-1]:
            prices.append(prices[-1] * (1 + r))
        return prices

    def _create_ohlcv_dataframe(self, prices: list, daily_returns: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
        rng = self._ensure_determinism()
        return pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(r) * 0.5) for p, r in zip(prices, daily_returns, strict=False)],
            'low': [p * (1 - abs(r) * 0.5) for p, r in zip(prices, daily_returns, strict=False)],
            'close': prices,
            'volume': [rng.integers(1000, 10000) for _ in range(len(prices))]
        }, index=dates)

_simulation_engine = None

def get_simulation_engine() -> 'SimulationEngine':
    global _simulation_engine
    if _simulation_engine is None:
        _simulation_engine = SimulationEngine()
    return _simulation_engine
