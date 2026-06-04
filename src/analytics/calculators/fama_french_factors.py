import logging

# src/analytics/calculators/fama_french_factors.py
"""
Fama-French Factor Provider
Provides accessibility to systematic risk factors including Market, Size, Value, and Momentum.
Uses Yahoo Finance as the primary data source for benchmark ETF proxies.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


def _load_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance dependency missing. Required for systematic factor retrieval."
        ) from exc
    return yf

class FamaFrenchFactors:
    """
    Data provider and calculator for Fama-French systematic risk factors.
    Downloads public benchmark data and computes relative factor returns for attribution analysis.
    """

    def __init__(self, benchmark_tickers: dict[str, str] | None = None, use_cache: bool = True):
        """
        Initializes the Fama-French factor engine.

        Args:
            benchmark_tickers: Custom ticker mapping for proxy ETFs.
            use_cache: Enables temporal caching of downloaded market data.
        """
        # Default ETF proxies for Fama-French factors
        self.benchmark_tickers = benchmark_tickers or {
            'market': '^GSPC',      # S&P 500 Index (Market Baseline)
            'size_small': 'IWM',    # Russell 2000 (Small Cap Proxy)
            'size_big': 'VV',       # Vanguard Large-Cap (Large Cap Proxy)
            'value': 'VTV',         # Vanguard Value (Value Style Proxy)
            'growth': 'VUG',        # Vanguard Growth (Growth Style Proxy)
            'momentum_up': 'MTUM',  # MSCI USA Momentum (Upward Momentum Proxy)
            'momentum_down': 'VIG', # Dividend Appreciation (Counter-momentum baseline)
        }

        self.use_cache = use_cache
        self.cache: dict[str, pd.DataFrame] = {}
        self.cache_expiry = timedelta(hours=24)
        self.last_cache_time: datetime | None = None

        logger.info(f"FamaFrenchFactors initialized with {len(self.benchmark_tickers)} benchmark proxies.")

    def get_factors(self, start_date: str, end_date: str) -> pd.DataFrame | None:
        """
        Retrieves and calculates primary Fama-French risk factors (MKT, SMB, HML, UMD).

        Args:
            start_date: Period start (YYYY-MM-DD).
            end_date: Period end (YYYY-MM-DD).

        Returns:
            DataFrame of factor returns or None if retrieval fails.
        """
        cache_key = f"{start_date}_{end_date}"

        cached_factors = self._check_factor_cache(cache_key)
        if cached_factors is not None:
            return cached_factors

        try:
            prices = self._download_data(start_date, end_date)
            if prices.empty:
                logger.info("Factor calculation skipped: Insufficient historical depth (Benchmark dataset empty).")
                return None

            returns = prices.pct_change(fill_method=None).dropna()

            if not self._validate_ticker_coverage(returns):
                return None

            factors_df = self._calculate_factors(returns)
            self._update_factor_cache(cache_key, factors_df)

            logger.info(f"Fama-French factor calculation successful ({len(factors_df)} points).")
            return factors_df

        except Exception as e:
            logger.error(f"Systematic factor computation failed: {e}", exc_info=True)
            raise RuntimeError(f"Systematic factor computation failed: {e}") from e

    def _check_factor_cache(self, cache_key: str) -> pd.DataFrame | None:
        """Check if factors are cached and valid."""
        if self.use_cache and cache_key in self.cache:
            if self.last_cache_time and (datetime.now() - self.last_cache_time) < self.cache_expiry:
                return self.cache[cache_key]
        return None

    def _validate_ticker_coverage(self, returns: pd.DataFrame) -> bool:
        """Validate that all required tickers are present in returns."""
        required_tickers = set(self.benchmark_tickers.values())
        if not required_tickers.issubset(returns.columns):
            missing = required_tickers - set(returns.columns)
            logger.error(f"Incomplete benchmark coverage. Missing: {missing}")
            return False
        return True

    def _calculate_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fama-French factors from returns data."""
        factors_df = pd.DataFrame(index=returns.index)

        factors_df['MKT'] = self._calculate_market_factor(returns)
        factors_df['SMB'] = self._calculate_size_factor(returns)
        factors_df['HML'] = self._calculate_value_factor(returns)
        factors_df['UMD'] = self._calculate_momentum_factor(returns)

        return factors_df.dropna()

    def _calculate_market_factor(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate Market factor (MKT)."""
        return returns[self.benchmark_tickers['market']]

    def _calculate_size_factor(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate Size factor (SMB): Small Minus Big."""
        return returns[self.benchmark_tickers['size_small']] - returns[self.benchmark_tickers['size_big']]

    def _calculate_value_factor(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate Value factor (HML): High Minus Low (Value vs. Growth)."""
        return returns[self.benchmark_tickers['value']] - returns[self.benchmark_tickers['growth']]

    def _calculate_momentum_factor(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate Momentum factor (UMD): Up Minus Down."""
        return returns[self.benchmark_tickers['momentum_up']] - returns[self.benchmark_tickers['momentum_down']]

    def _update_factor_cache(self, cache_key: str, factors_df: pd.DataFrame) -> None:
        """Update cache with calculated factors."""
        if self.use_cache:
            self.cache[cache_key] = factors_df
            self.last_cache_time = datetime.now()

    def _download_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Internal worker for Yahoo Finance data ingestion."""
        # Validate date range
        if not self._validate_date_range(start_date, end_date):
            return pd.DataFrame()

        # Setup cache and tickers
        tickers = list(set(self.benchmark_tickers.values()))
        cache_key = f"raw_{start_date}_{end_date}"

        # Check cache first
        cached_data = self._check_cache(cache_key, start_date)
        if cached_data is not None:
            return cached_data

        # Download from yfinance
        try:
            result = self._download_from_yfinance(tickers, start_date, end_date)
            if not result.empty:
                self._update_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Remote benchmark ingestion failed (yfinance): {e}")
            return self._get_fallback_cache(cache_key)

    def _validate_date_range(self, start_date: str, end_date: str) -> bool:
        """Validate date range parameters"""
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            if start_dt >= end_dt:
                logger.info(f"Temporal range too narrow for factor analysis: {start_date} to {end_date}")
                return False
            return True
        except Exception as e:
            logger.error(f"Temporal parameters malformed: {e}")
            return False

    def _check_cache(self, cache_key: str, start_date: str) -> pd.DataFrame | None:
        """Check if cached data is available and valid"""
        if not self.use_cache or cache_key not in self.cache:
            return None

        if self.last_cache_time and (datetime.now() - self.last_cache_time) < self.cache_expiry:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Utilizing cached benchmark data for {start_date} period.")
            return self.cache[cache_key]

        return None

    def _download_from_yfinance(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download data from yfinance"""
        logger.info(f"Ingesting historical benchmarks from yfinance ({len(tickers)} assets)...")
        yf = _load_yfinance()
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True, group_by='ticker')

        if len(tickers) == 1:
            result = data[['Close']].rename(columns={'Close': tickers[0]})
        else:
            # Reconstruct flat DataFrame from multi-index download
            close_prices = pd.DataFrame({
                ticker: data[ticker]['Close']
                for ticker in tickers
                if ticker in data.columns.get_level_values(0) and not data[ticker].empty
            })
            result = close_prices.dropna(how='all')

        if result.empty or len(result) < 5:
            logger.warning(f"Insufficient historical depth returned for {tickers}")
            return pd.DataFrame()

        return result

    def _update_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Update cache with new data"""
        if self.use_cache and not data.empty:
            self.cache[cache_key] = data
            self.last_cache_time = datetime.now()

    def _get_fallback_cache(self, cache_key: str) -> pd.DataFrame:
        """Get fallback cache data during outage"""
        if self.use_cache and cache_key in self.cache:
            logger.warning("Utilizing stale cache due to connectivity/retrieval error.")
            return self.cache[cache_key]
        return pd.DataFrame()

    def analyze_factor_performance(self, factors: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Calculates statistical properties of the systematic factor streams."""
        performance_stats = {}
        for factor_name in factors.columns:
            f_series = pd.to_numeric(factors[factor_name], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            if f_series.empty:
                continue
            factor_std = f_series.std()
            annualized_sharpe = (
                float((f_series.mean() / factor_std) * np.sqrt(252))
                if np.isfinite(factor_std) and factor_std > 1e-12
                else np.nan
            )

            performance_stats[factor_name] = {
                'mean_return': float(f_series.mean()),
                'volatility': float(factor_std),
                'annualized_sharpe': annualized_sharpe,
                'annualized_return': float(f_series.mean() * 252),
                'annualized_vol': float(factor_std * np.sqrt(252))
            }
        return performance_stats
