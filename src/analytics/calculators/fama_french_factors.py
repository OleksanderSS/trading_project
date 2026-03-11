import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)

class FamaFrenchFactors:
    """
    Provides Fama-French market factors (Market, Size, Value, Momentum, etc.).
    Acts as a data provider and calculator, downloading benchmark data and computing factors.
    """

    def __init__(self, benchmark_tickers: Optional[Dict[str, str]] = None, use_cache: bool = True):
        """
        Initializes the Fama-French factor provider.

        Args:
            benchmark_tickers (Optional[Dict[str, str]]): Tickers for benchmark indices.
            use_cache (bool): Flag to enable/disable caching of downloaded data.
        """
        if yf is None:
            raise ImportError("yfinance is not installed. Please install it with: pip install yfinance")

        self.benchmark_tickers = benchmark_tickers or {
            'market': '^GSPC',      # S&P 500
            'size_small': '^RUT',   # Russell 2000 (small caps)
            'size_big': '^OEX',     # S&P 100 (large caps)
            'value': 'IWD',         # iShares Russell 1000 Value
            'growth': 'IWF',        # iShares Russell 1000 Growth
            'momentum_up': 'MTUM',  # iShares MSCI USA Momentum
            'momentum_down': 'MDLA', # iShares MSCI USA Minimum Volatility (proxy for down)
        }
        
        self.use_cache = use_cache
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_expiry = timedelta(hours=24)
        self.last_cache_time: Optional[datetime] = None
        
        logger.info(f"FamaFrenchFactors provider initialized with {len(self.benchmark_tickers)} benchmark tickers.")

    def get_factors(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Retrieves and calculates key Fama-French factors (MKT, SMB, HML, UMD).

        Args:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            Optional[pd.DataFrame]: A DataFrame with calculated factors, or None on failure.
        """
        cache_key = f"{start_date}_{end_date}"
        if self.use_cache and cache_key in self.cache:
            if self.last_cache_time and (datetime.now() - self.last_cache_time) < self.cache_expiry:
                return self.cache[cache_key]

        try:
            prices = self._download_data(start_date, end_date)
            if prices.empty:
                logger.error("No data downloaded for Fama-French factors.")
                return None

            returns = prices.pct_change().dropna()
            
            # Ensure all required columns are present after download and pct_change
            required_tickers = set(self.benchmark_tickers.values())
            if not required_tickers.issubset(returns.columns):
                missing = required_tickers - set(returns.columns)
                logger.error(f"Missing required benchmark ticker data after download: {missing}")
                return None

            factors_df = pd.DataFrame(index=returns.index)
            
            # 1. Market Factor (MKT)
            factors_df['MKT'] = returns[self.benchmark_tickers['market']]
            
            # 2. Size Factor (SMB): Small minus Big
            factors_df['SMB'] = returns[self.benchmark_tickers['size_small']] - returns[self.benchmark_tickers['size_big']]
            
            # 3. Value Factor (HML): High Book-to-Market minus Low (Value minus Growth)
            factors_df['HML'] = returns[self.benchmark_tickers['value']] - returns[self.benchmark_tickers['growth']]
            
            # 4. Momentum Factor (UMD): Up minus Down
            factors_df['UMD'] = returns[self.benchmark_tickers['momentum_up']] - returns[self.benchmark_tickers['momentum_down']]
            
            # Note: RMW and CMA factors are harder to replicate with simple ETFs and are omitted for robustness.

            factors_df = factors_df.dropna()

            if self.use_cache:
                self.cache[cache_key] = factors_df
                self.last_cache_time = datetime.now()

            logger.info(f"Fama-French factors calculated: {len(factors_df)} observations.")
            return factors_df

        except Exception as e:
            logger.error(f"Error calculating Fama-French factors: {e}", exc_info=True)
            return None

    def _download_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Downloads historical price data for all benchmark tickers."""
        tickers = list(set(self.benchmark_tickers.values()))
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True, group_by='ticker')
            
            # If only one ticker, yf doesn't return a multi-level column header.
            if len(tickers) == 1:
                return data[['Close']].rename(columns={'Close': tickers[0]})
            
            # Extract 'Close' price for each ticker
            close_prices = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in tickers if not data[ticker].empty})
            
            return close_prices.dropna(how='all')
        except Exception as e:
            logger.error(f"yfinance download failed for tickers {tickers}: {e}")
            return pd.DataFrame()

    def analyze_factor_performance(self, factors: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyzes performance and statistical properties of the provided factor series."""
        stats = {}
        for col in factors.columns:
            f_data = factors[col]
            if f_data.empty:
                continue
            stats[col] = {
                'mean': float(f_data.mean()),
                'std': float(f_data.std()),
                'sharpe': float((f_data.mean() / f_data.std()) * np.sqrt(252)) if f_data.std() != 0 else 0,
                'annual_return': float(f_data.mean() * 252),
                'annual_volatility': float(f_data.std() * np.sqrt(252))
            }
        return stats
