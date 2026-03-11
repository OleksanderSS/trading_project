"""
ADVANCED SIMULATION ENGINE
Розширена симуляційна система для моделювання складних ринкових ситуацій
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from scipy import stats

from src.config.unified_config_manager import get_current_config
from src.trading.virtual_portfolio import VirtualPortfolio
from src.metrics.calculator import MetricsCalculator

logger = logging.getLogger(__name__)

class SimulationGranularity(Enum):
    TICKER_LEVEL = "ticker"
    SECTOR_LEVEL = "sector"
    MARKET_LEVEL = "market"

@dataclass
class SimulationContext:
    ticker: str
    timestamp: datetime
    granularity: SimulationGranularity
    features: Dict[str, Any] = field(default_factory=dict)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    historical_returns: Optional[pd.Series] = None

@dataclass
class SimulationRiskReport:
    ticker: str
    var_95: float
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    max_drawdown: float
    # ... other relevant metrics

class SimulationEngine:
    def __init__(self, max_workers: int = None):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers or min(mp.cpu_count(), 4)
        self.config = get_current_config()
        self.sim_config = self.config.get('simulation', {})
        self.optimization_config = self.sim_config.get('optimization', {})
        self.defaults_config = self.sim_config.get('defaults', {})

    def run_monte_carlo_for_strategy(
        self, 
        strategy_logic: Callable[[pd.DataFrame], pd.Series], # Expects a function that takes market data and returns trade signals
        initial_context: SimulationContext,
        horizon: int,
        runs: int = None
    ) -> List[SimulationRiskReport]:
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
    ) -> Optional[SimulationRiskReport]:
        try:
            # 1. Generate synthetic market data path
            price_path_df = self._generate_price_path(context, horizon)
            
            # 2. Initialize a virtual portfolio for this simulation run
            portfolio = VirtualPortfolio(initial_cash=100000) # Configurable

            # 3. Run the strategy logic on the generated data
            signals = strategy_logic(price_path_df)
            for timestamp, row in price_path_df.iterrows():
                signal = signals.get(timestamp)
                if signal is not None:
                    # Simple logic: 1 for buy, -1 for sell
                    if signal == 1:
                        portfolio.execute_order(context.ticker, 10, row['close'], timestamp, 'buy')
                    elif signal == -1:
                        portfolio.execute_order(context.ticker, 10, row['close'], timestamp, 'sell')
            
            # 4. Calculate metrics for this path
            metrics_calculator = MetricsCalculator(portfolio.get_history_df())
            report = SimulationRiskReport(
                ticker=context.ticker,
                var_95=metrics_calculator.calculate_value_at_risk(0.05),
                var_99=metrics_calculator.calculate_value_at_risk(0.01),
                expected_shortfall=metrics_calculator.calculate_expected_shortfall(0.05),
                sharpe_ratio=metrics_calculator.calculate_sharpe_ratio(),
                max_drawdown=metrics_calculator.calculate_max_drawdown()
            )
            return report
        except Exception as e:
            self.logger.error(f"Error in single path simulation: {e}", exc_info=True)
            return None

    def _generate_price_path(self, context: SimulationContext, horizon: int) -> pd.DataFrame:
        """Generates a DataFrame with OHLCV data for a single path."""
        current_price = context.features.get('close', 100)
        volatility = context.market_conditions.get('volatility', 0.02)
        trend = context.market_conditions.get('trend', 0)
        
        use_bootstrap = self.optimization_config.get('use_historical_bootstrap', True)
        if use_bootstrap and context.historical_returns is not None and not context.historical_returns.empty:
            hist_rets = context.historical_returns.values
            daily_returns = np.random.choice(hist_rets, size=horizon, replace=True)
        else:
            df = self.optimization_config.get('t_distribution_df', 3.0)
            scale = volatility * np.sqrt((df - 2) / df) if df > 2 else volatility
            daily_returns = stats.t.rvs(df, loc=trend / horizon, scale=scale, size=horizon)

        dates = pd.to_datetime([context.timestamp + timedelta(days=i) for i in range(horizon)])
        prices = [current_price]
        for r in daily_returns[:-1]:
            prices.append(prices[-1] * (1 + r))
        
        price_path = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(r) * 0.5) for p, r in zip(prices, daily_returns)],
            'low': [p * (1 - abs(r) * 0.5) for p, r in zip(prices, daily_returns)],
            'close': prices,
            'volume': [np.random.randint(1000, 10000) for _ in range(horizon)]
        }, index=dates)
        
        return price_path

_simulation_engine = None

def get_simulation_engine() -> 'SimulationEngine':
    global _simulation_engine
    if _simulation_engine is None:
        _simulation_engine = SimulationEngine()
    return _simulation_engine
