"""
ADVANCED SIMULATION ENGINE
Advanced simulation system for modeling complex market situations
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
from src.data.collectors.synthetic_generator import (
    BUILTIN_SCENARIOS,
    GeneratorConfig,
    SyntheticGenerator,
)
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
    # ... other relevant metrics

class SimulationEngine:
    def __init__(self, max_workers: int = None):
        self.logger = ProjectLogger.get_logger("SimulationEngine")
        self.max_workers = max_workers or min(mp.cpu_count(), 4)
        self.config = get_current_config()
        self.sim_config = self.config.get('simulation', {})
        self.optimization_config = self.sim_config.get('optimization', {})
        self.defaults_config = self.sim_config.get('defaults', {})

        # Initialize synthetic generator for stress tests
        random_seed = self.config.get(RANDOM_SEED_CONFIG_KEY, DEFAULT_RANDOM_SEED)
        gen_config = GeneratorConfig(
            n_paths=self.defaults_config.get('monte_carlo_runs', 1000),
            horizon_days=self.defaults_config.get('horizon_days', 252),
            random_seed=random_seed,
        )
        self.synthetic_generator = SyntheticGenerator(config=gen_config)
        self.rng = np.random.default_rng(seed=random_seed)

    def run_monte_carlo_for_strategy(
        self,
        strategy_logic: Callable[[pd.DataFrame], pd.Series], # Expects a function that takes market data and returns trade signals
        initial_context: SimulationContext,
        horizon: int,
        runs: int = None,
        scenario_name: str | None = None
    ) -> list[SimulationRiskReport]:
        runs = runs or self.defaults_config.get('monte_carlo_runs', 1000)

        reports = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._run_single_path_simulation, strategy_logic, initial_context, horizon, scenario_name) for _ in range(runs)]
            for future in futures:
                report = future.result()
                if report:
                    reports.append(report)
        return reports

    def _run_single_path_simulation(
        self,
        strategy_logic: Callable[[pd.DataFrame], pd.Series],
        context: SimulationContext,
        horizon: int,
        scenario_name: str | None = None
    ) -> SimulationRiskReport | None:
        try:
            # 1. Generate synthetic market data path
            price_path_df = self._generate_price_path(context, horizon, scenario_name)

            # 2. Initialize a virtual portfolio for this simulation run
            portfolio = VirtualPortfolio(initial_balance=100000.0)

            # 3. Run the strategy logic on the generated data
            signals = strategy_logic(price_path_df)
            equity_curve = []
            
            for timestamp, row in price_path_df.iterrows():
                signal = signals.get(timestamp)
                price = row['close']
                
                if signal is not None:
                    if signal == 1:
                        portfolio.buy_stock({'ticker': context.ticker, 'quantity': 10, 'price': price})
                    elif signal == -1:
                        portfolio.sell_stock(ticker=context.ticker, quantity=10, price=price, reason='signal')
                
                # Calculate current equity
                pos_value = portfolio.positions.get(context.ticker, {}).get('quantity', 0) * price
                equity_curve.append(portfolio.current_balance + pos_value)

            # 4. Calculate metrics for this path
            eq_series = pd.Series(equity_curve)
            returns = eq_series.pct_change().dropna()
            
            if len(returns) > 0:
                var_95 = float(np.percentile(returns, 5))
                var_99 = float(np.percentile(returns, 1))
                es_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95
                
                # Sharpe (annualized assuming daily data for simplicity in simulation)
                sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0.0
                
                # Max Drawdown
                cum_returns = (1 + returns).cumprod()
                running_max = cum_returns.cummax()
                drawdowns = (cum_returns - running_max) / running_max
                max_drawdown = float(drawdowns.min())
            else:
                var_95 = var_99 = es_95 = sharpe_ratio = max_drawdown = 0.0

            report = SimulationRiskReport(
                ticker=context.ticker,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=es_95,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown
            )
            return report
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error in single path simulation: {e}", exc_info=True)
            raise RuntimeError(
                f"Single path simulation failed for {context.ticker}"
            ) from e

    def _generate_price_path(self, context: SimulationContext, horizon: int, scenario_name: str | None = None) -> pd.DataFrame:
        """Generates a DataFrame with OHLCV data for a single path."""
        
        if scenario_name and scenario_name in BUILTIN_SCENARIOS:
            # Use the robust SyntheticGenerator for specific scenarios
            # Build a base_df from historical returns so the generator can calibrate
            base_df = None
            if context.historical_returns is not None and not context.historical_returns.empty:
                base_df = pd.DataFrame({'close': (1 + context.historical_returns).cumprod() * 100})
            paths = self.synthetic_generator.generate_scenarios(
                scenario_names=[scenario_name],
                base_df=base_df,
            )
            path_list = paths.get(scenario_name, [])
            if path_list:
                df = path_list[0]  # take the first generated path
                dates = self._generate_dates(context.timestamp, horizon)
                df = df.iloc[:horizon] if len(df) >= horizon else df
                if len(df) == len(dates):
                    df.index = dates
                return df
            # fallthrough to GBM if no paths generated

        # Fallback to simple logic (historical bootstrap or simple GBM)
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

    # _ensure_determinism is removed — RNG is now initialized once in __init__

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
            'high': [p * (1 + abs(r) * 0.5) for p, r in zip(prices, daily_returns, strict=False)],
            'low': [p * (1 - abs(r) * 0.5) for p, r in zip(prices, daily_returns, strict=False)],
            'close': prices,
            'volume': [self.rng.integers(1000, 10000) for _ in range(len(prices))]
        }, index=dates)

    def run_stress_test_suite(
        self,
        base_df: pd.DataFrame | None = None,
        scenario_names: list[str] | None = None,
    ) -> dict[str, dict]:
        """Run a full stress test suite using the SyntheticGenerator.

        Generates Monte Carlo paths for each scenario and returns
        summary statistics (VaR, max drawdown, loss probability, etc.).

        Args:
            base_df: Optional real data to calibrate from.
            scenario_names: List of scenarios to test. Defaults to all built-in.

        Returns:
            Dict mapping scenario_name -> summary statistics dict.
        """
        if scenario_names is None:
            scenario_names = list(BUILTIN_SCENARIOS.keys())

        self.logger.info(
            f"🧪 Running stress test suite: {len(scenario_names)} scenarios, "
            f"{self.synthetic_generator.config.n_paths} paths each"
        )

        all_paths = self.synthetic_generator.generate_scenarios(
            scenario_names=scenario_names,
            base_df=base_df,
        )

        results = {}
        for name, paths in all_paths.items():
            summary = self.synthetic_generator.summarise_paths(paths)
            results[name] = summary
            self.logger.info(
                f"  📊 {name}: mean_ret={summary['mean_return']:.2%}, "
                f"VaR95={summary['var_95']:.2%}, "
                f"worst_dd={summary['worst_max_drawdown']:.2%}, "
                f"P(loss)={summary['prob_loss']:.1%}"
            )

        return results


_simulation_engine = None

def get_simulation_engine() -> 'SimulationEngine':
    global _simulation_engine
    if _simulation_engine is None:
        _simulation_engine = SimulationEngine()
    return _simulation_engine

