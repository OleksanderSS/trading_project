"""
ADVANCED SIMULATION ENGINE
Advanced simulation system for modeling complex market situations
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

from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config
from src.trading.virtual_portfolio import VirtualPortfolio
from src.metrics.calculator import MetricsCalculator

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
        self.logger = ProjectLogger.get_logger("SimulationEngine")
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
        market_params = self._extract_market_parameters(context)
        
        self._ensure_determinism()
        
        daily_returns = self._generate_daily_returns(context, horizon, market_params)
        
        dates = self._generate_dates(context.timestamp, horizon)
        prices = self._calculate_price_path(market_params['current_price'], daily_returns)
        
        return self._create_ohlcv_dataframe(prices, daily_returns, dates)
    
    def _extract_market_parameters(self, context: SimulationContext) -> dict:
        """Extract market parameters from context."""
        return {
            'current_price': context.features.get('close', 100),
            'volatility': context.market_conditions.get('volatility', 0.02),
            'trend': context.market_conditions.get('trend', 0)
        }
    
    def _ensure_determinism(self):
        """Ensure deterministic random number generation using thread-safe RNG."""
        seed = self.config.get(RANDOM_SEED_CONFIG_KEY, DEFAULT_RANDOM_SEED)
        self.rng = np.random.default_rng(seed)
    
    def _generate_daily_returns(self, context: SimulationContext, horizon: int, market_params: dict) -> np.ndarray:
        """Generate daily returns based on configuration."""
        if self._should_use_bootstrap(context):
            return self._generate_bootstrap_returns(context, horizon)
        else:
            return self._generate_t_distribution_returns(horizon, market_params)
    
    def _should_use_bootstrap(self, context: SimulationContext) -> bool:
        """Check if historical bootstrap should be used."""
        use_bootstrap = self.optimization_config.get('use_historical_bootstrap', True)
        return (use_bootstrap and 
                context.historical_returns is not None and 
                not context.historical_returns.empty)
    
    def _generate_bootstrap_returns(self, context: SimulationContext, horizon: int) -> np.ndarray:
        """Generate returns using historical bootstrap."""
        hist_rets = context.historical_returns.values
        return self.rng.choice(hist_rets, size=horizon, replace=True)
    
    def _generate_t_distribution_returns(self, horizon: int, market_params: dict) -> np.ndarray:
        """Generate returns using t-distribution."""
        df = self.optimization_config.get('t_distribution_df', 3.0)
        scale = self._calculate_t_distribution_scale(market_params['volatility'], df)
        return stats.t.rvs(df, loc=market_params['trend'] / horizon, scale=scale, size=horizon, random_state=self.rng)
    
    def _calculate_t_distribution_scale(self, volatility: float, df: float) -> float:
        """Calculate scale parameter for t-distribution."""
        return volatility * np.sqrt((df - 2) / df) if df > 2 else volatility
    
    def _generate_dates(self, start_timestamp: datetime, horizon: int) -> pd.DatetimeIndex:
        """Generate date series for the simulation horizon."""
        return pd.to_datetime([start_timestamp + timedelta(days=i) for i in range(horizon)])
    
    def _calculate_price_path(self, current_price: float, daily_returns: np.ndarray) -> list:
        """Calculate price path from daily returns."""
        prices = [current_price]
        for r in daily_returns:
            prices.append(prices[-1] * (1 + r))
        return prices
    
    def _create_ohlcv_dataframe(self, prices: list, daily_returns: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create OHLCV DataFrame from prices and returns."""
        return pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(r) * 0.5) for p, r in zip(prices, daily_returns)],
            'low': [p * (1 - abs(r) * 0.5) for p, r in zip(prices, daily_returns)],
            'close': prices,
            'volume': [self.rng.integers(1000, 10000) for _ in range(len(prices))]
        }, index=dates)

_simulation_engine = None

def get_simulation_engine() -> 'SimulationEngine':
    global _simulation_engine
    if _simulation_engine is None:
        _simulation_engine = SimulationEngine()
    return _simulation_engine
