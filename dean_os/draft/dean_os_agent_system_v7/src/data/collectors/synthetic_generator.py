"""
Synthetic Market Data Generator for Stress Testing & Data Augmentation.

Generates realistic synthetic OHLCV price paths using:
- Geometric Brownian Motion (GBM) for typical market conditions
- Monte Carlo simulation with configurable shocks
- Scenario-based stress tests (Flash Crash, Liquidity Crisis, Black Swan, etc.)

The generated data can be injected into the pipeline via CSVCollector
or used directly in Stage 7 (Evaluation) for robustness testing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ScenarioConfig:
    """Configuration for a single stress-test scenario."""
    name: str
    description: str
    # GBM overrides
    mu_override: float | None = None  # Drift override (annualised)
    sigma_override: float | None = None  # Vol override (annualised)
    # Shock injection
    shock_magnitude: float = 0.0  # e.g. -0.15 for a 15% crash
    shock_day: int | None = None  # Day of the shock (None = mid-series)
    shock_duration_bars: int = 1  # How many bars the shock lasts
    # Fat tails
    t_distribution_df: float | None = None  # Use t-dist instead of normal
    # Volatility regime change
    vol_multiplier: float = 1.0  # Multiplier applied to base sigma
    # Mean reversion after shock
    mean_reversion_speed: float = 0.0  # 0 = no reversion, 1 = instant


@dataclass
class GeneratorConfig:
    """Global configuration for the synthetic data generator."""
    n_paths: int = 1000
    horizon_days: int = 252  # 1 trading year
    base_price: float = 100.0
    base_mu: float = 0.0003  # ~7.5% annual drift (daily)
    base_sigma: float = 0.015  # ~24% annual vol (daily)
    random_seed: int | None = 42
    synthetic_ratio: float = 0.1  # 10% of real data size by default
    scenarios: list[ScenarioConfig] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-built scenarios
# ---------------------------------------------------------------------------
BUILTIN_SCENARIOS: dict[str, ScenarioConfig] = {
    "typical": ScenarioConfig(
        name="typical",
        description="Normal market conditions (GBM with historical params)",
    ),
    "high_volatility": ScenarioConfig(
        name="high_volatility",
        description="Elevated volatility regime (2x base vol)",
        vol_multiplier=2.0,
    ),
    "flash_crash": ScenarioConfig(
        name="flash_crash",
        description="Sudden 10% drop over 5 bars, partial recovery",
        shock_magnitude=-0.10,
        shock_duration_bars=5,
        mean_reversion_speed=0.3,
    ),
    "black_swan": ScenarioConfig(
        name="black_swan",
        description="Extreme 20% crash in 1 bar (tail event)",
        shock_magnitude=-0.20,
        shock_duration_bars=1,
        vol_multiplier=3.0,
        t_distribution_df=3.0,
    ),
    "liquidity_crisis": ScenarioConfig(
        name="liquidity_crisis",
        description="Gradual 15% decline over 20 bars with spiking vol",
        shock_magnitude=-0.15,
        shock_duration_bars=20,
        vol_multiplier=2.5,
    ),
    "bull_run": ScenarioConfig(
        name="bull_run",
        description="Strong upward momentum, low vol",
        mu_override=0.002,  # ~50% annual drift
        vol_multiplier=0.7,
    ),
    "bear_market": ScenarioConfig(
        name="bear_market",
        description="Sustained downtrend with elevated vol",
        mu_override=-0.001,  # ~-25% annual drift
        vol_multiplier=1.8,
    ),
    "sector_rotation": ScenarioConfig(
        name="sector_rotation",
        description="Choppy sideways market with vol spikes",
        mu_override=0.0,
        vol_multiplier=1.5,
        t_distribution_df=5.0,
    ),
}


class SyntheticGenerator(BaseCollector):
    """Generates synthetic market scenarios for stress-testing and data augmentation.

    Supports:
    - Geometric Brownian Motion (GBM) price paths
    - Monte Carlo simulation (N paths)
    - Configurable shock injection (flash crash, black swan, etc.)
    - Fat-tailed returns via t-distribution
    - Mean reversion after shocks
    - Augmentation mode: generate extra data proportional to real dataset size
    """

    collector_type = "synthetic"

    def __init__(self, config: GeneratorConfig | None = None, **kwargs):
        # SyntheticGenerator can work without BaseCollector dependencies
        # so we handle the case where no kwargs are passed for standalone use
        if kwargs:
            super().__init__(**kwargs)
        else:
            self.logger = logging.getLogger("SyntheticGenerator")
            self.configs = {}

        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        logger.info(
            f"✅ SyntheticGenerator initialized: "
            f"{self.config.n_paths} paths × {self.config.horizon_days} days"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_scenarios(
        self,
        scenario_names: list[str] | None = None,
        base_df: pd.DataFrame | None = None,
    ) -> dict[str, list[pd.DataFrame]]:
        """Generate Monte Carlo paths for requested scenarios.

        Args:
            scenario_names: List of scenario names (from BUILTIN_SCENARIOS).
                            If None, generates all built-in scenarios.
            base_df: Optional real DataFrame to calibrate mu/sigma from.

        Returns:
            Dict mapping scenario name -> list of DataFrames (one per path).
        """
        if scenario_names is None:
            scenario_names = list(BUILTIN_SCENARIOS.keys())

        # Calibrate from real data if provided
        calibrated_mu, calibrated_sigma = self.config.base_mu, self.config.base_sigma
        if base_df is not None and "close" in base_df.columns:
            calibrated_mu, calibrated_sigma = self._calibrate_from_real_data(base_df)
            logger.info(
                f"📊 Calibrated from real data: μ={calibrated_mu:.6f}, σ={calibrated_sigma:.6f}"
            )

        results: dict[str, list[pd.DataFrame]] = {}
        for name in scenario_names:
            scenario = BUILTIN_SCENARIOS.get(name)
            if scenario is None:
                logger.warning(f"Unknown scenario '{name}', skipping")
                continue

            paths = self._generate_monte_carlo_paths(scenario, calibrated_mu, calibrated_sigma)
            results[name] = paths
            logger.info(
                f"✅ Generated {len(paths)} paths for scenario '{name}' "
                f"({scenario.description})"
            )

        return results

    def generate_augmentation_data(
        self,
        real_df: pd.DataFrame,
        ratio: float | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic data to augment a real dataset.

        This creates additional rows that look statistically similar
        to the real data, for training data augmentation.

        Args:
            real_df: The real DataFrame with at least a 'close' column.
            ratio: Fraction of real data size to generate (default: config.synthetic_ratio).

        Returns:
            pd.DataFrame with synthetic rows (same column schema as real_df).
        """
        ratio = ratio or self.config.synthetic_ratio
        n_synthetic = max(1, int(len(real_df) * ratio))
        logger.info(
            f"📊 Generating {n_synthetic} augmentation rows "
            f"({ratio:.0%} of {len(real_df)} real rows)"
        )

        mu, sigma = self._calibrate_from_real_data(real_df)
        base_price = float(real_df["close"].iloc[-1])

        # Generate a single path of n_synthetic length
        returns = self._generate_returns(
            n_synthetic, mu, sigma,
            t_df=None, vol_multiplier=1.0
        )
        prices = self._returns_to_prices(base_price, returns)

        # Build OHLCV
        synthetic_df = self._build_ohlcv_df(prices, returns, n_synthetic)

        # Copy non-OHLCV columns from real data (forward-fill from last row)
        ohlcv_cols = {"open", "high", "low", "close", "volume", "datetime", "date"}
        extra_cols = [c for c in real_df.columns if c.lower() not in ohlcv_cols]
        if extra_cols:
            last_row = real_df[extra_cols].iloc[-1]
            for col in extra_cols:
                synthetic_df[col] = last_row[col]

        synthetic_df["is_synthetic"] = True
        return synthetic_df

    def generate_flash_crash(
        self, base_df: pd.DataFrame, drop_percent: float = 0.1, duration_bars: int = 5
    ) -> pd.DataFrame:
        """Legacy API: inject a flash crash into an existing DataFrame."""
        df = base_df.copy()
        crash_start = len(df) // 2
        for i in range(min(duration_bars, len(df) - crash_start)):
            idx = crash_start + i
            col_idx = df.columns.get_loc("close")
            df.iloc[idx, col_idx] *= 1 - drop_percent / duration_bars
        return df

    def generate_high_volatility(
        self, base_df: pd.DataFrame, noise_level: float = 0.05
    ) -> pd.DataFrame:
        """Legacy API: add noise to close prices."""
        df = base_df.copy()
        noise = self.rng.normal(0, noise_level, len(df))
        df["close"] *= 1 + noise
        return df

    # ------------------------------------------------------------------
    # Core generation logic
    # ------------------------------------------------------------------

    def _generate_monte_carlo_paths(
        self,
        scenario: ScenarioConfig,
        base_mu: float,
        base_sigma: float,
    ) -> list[pd.DataFrame]:
        """Generate N Monte Carlo price paths for a given scenario."""
        mu = scenario.mu_override if scenario.mu_override is not None else base_mu
        sigma = base_sigma * scenario.vol_multiplier

        paths: list[pd.DataFrame] = []
        for i in range(self.config.n_paths):
            returns = self._generate_returns(
                self.config.horizon_days, mu, sigma,
                t_df=scenario.t_distribution_df,
                vol_multiplier=1.0,  # already applied above
            )

            # Inject shock if configured
            if scenario.shock_magnitude != 0:
                returns = self._inject_shock(
                    returns, scenario.shock_magnitude,
                    scenario.shock_day, scenario.shock_duration_bars,
                    scenario.mean_reversion_speed,
                )

            prices = self._returns_to_prices(self.config.base_price, returns)
            df = self._build_ohlcv_df(prices, returns, self.config.horizon_days)
            df["scenario"] = scenario.name
            df["path_id"] = i
            paths.append(df)

        return paths

    def _generate_returns(
        self,
        n: int,
        mu: float,
        sigma: float,
        t_df: float | None = None,
        vol_multiplier: float = 1.0,
    ) -> np.ndarray:
        """Generate daily returns from GBM or t-distribution."""
        effective_sigma = sigma * vol_multiplier
        if t_df is not None and t_df > 2:
            # Fat-tailed returns via t-distribution
            scale = effective_sigma * np.sqrt((t_df - 2) / t_df)
            returns = stats.t.rvs(
                t_df, loc=mu, scale=scale,
                size=n, random_state=self.rng,
            )
        else:
            # Standard GBM: dS/S = μdt + σdW
            returns = self.rng.normal(mu, effective_sigma, n)
        return returns

    def _inject_shock(
        self,
        returns: np.ndarray,
        magnitude: float,
        shock_day: int | None,
        duration: int,
        reversion_speed: float,
    ) -> np.ndarray:
        """Inject a price shock into the returns series."""
        returns = returns.copy()
        if shock_day is None:
            shock_day = len(returns) // 2

        shock_day = min(shock_day, len(returns) - duration)
        if shock_day < 0:
            shock_day = 0

        # Distribute shock across duration bars
        daily_shock = magnitude / max(duration, 1)
        for i in range(duration):
            idx = shock_day + i
            if idx < len(returns):
                returns[idx] = daily_shock

        # Mean reversion after shock
        if reversion_speed > 0:
            recovery_start = shock_day + duration
            recovery_length = min(duration * 3, len(returns) - recovery_start)
            for i in range(recovery_length):
                idx = recovery_start + i
                if idx < len(returns):
                    reversion_return = -magnitude * reversion_speed * np.exp(-i / max(duration, 1))
                    returns[idx] += reversion_return / recovery_length

        return returns

    def _returns_to_prices(self, base_price: float, returns: np.ndarray) -> np.ndarray:
        """Convert returns to a price series via cumulative product."""
        price_factors = np.cumprod(1 + returns)
        return base_price * np.insert(price_factors, 0, 1.0)

    def _build_ohlcv_df(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        n: int,
    ) -> pd.DataFrame:
        """Build OHLCV DataFrame from price path."""
        # prices has n+1 elements; use [1:] for close prices
        close_prices = prices[1:]
        open_prices = prices[:-1]

        # Simulate high/low spread based on return magnitude
        abs_returns = np.abs(returns)
        spread = abs_returns * 0.5 + 0.001  # minimum spread

        high_prices = np.maximum(open_prices, close_prices) * (1 + spread)
        low_prices = np.minimum(open_prices, close_prices) * (1 - spread)

        # Synthetic volume: higher volume on larger moves
        base_volume = 1_000_000
        volume = (base_volume * (1 + abs_returns * 10)).astype(int)

        dates = pd.date_range(
            start=datetime.now() - timedelta(days=n),
            periods=n,
            freq="B",  # Business days
        )

        return pd.DataFrame(
            {
                "datetime": dates,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
            }
        )

    def _calibrate_from_real_data(
        self, real_df: pd.DataFrame
    ) -> tuple[float, float]:
        """Extract mu (drift) and sigma (volatility) from real price data."""
        close = real_df["close"].dropna()
        if len(close) < 10:
            return self.config.base_mu, self.config.base_sigma

        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        mu = float(returns.mean())
        sigma = float(returns.std())

        # Sanity clamp
        mu = max(-0.01, min(0.01, mu))
        sigma = max(0.001, min(0.1, sigma))

        return mu, sigma

    # ------------------------------------------------------------------
    # Stress-test summary helpers
    # ------------------------------------------------------------------

    def summarise_paths(
        self, paths: list[pd.DataFrame]
    ) -> dict[str, Any]:
        """Compute summary statistics across Monte Carlo paths."""
        if not paths:
            return {}

        final_returns = []
        max_drawdowns = []
        for path_df in paths:
            close = path_df["close"].values
            total_return = (close[-1] / close[0]) - 1
            final_returns.append(total_return)

            # Max drawdown
            running_max = np.maximum.accumulate(close)
            drawdowns = (close - running_max) / running_max
            max_drawdowns.append(float(np.min(drawdowns)))

        final_returns = np.array(final_returns)
        max_drawdowns = np.array(max_drawdowns)

        return {
            "n_paths": len(paths),
            "mean_return": float(np.mean(final_returns)),
            "median_return": float(np.median(final_returns)),
            "std_return": float(np.std(final_returns)),
            "var_95": float(np.percentile(final_returns, 5)),
            "var_99": float(np.percentile(final_returns, 1)),
            "worst_return": float(np.min(final_returns)),
            "best_return": float(np.max(final_returns)),
            "mean_max_drawdown": float(np.mean(max_drawdowns)),
            "worst_max_drawdown": float(np.min(max_drawdowns)),
            "prob_loss": float(np.mean(final_returns < 0)),
            "prob_large_loss": float(np.mean(final_returns < -0.1)),
        }
