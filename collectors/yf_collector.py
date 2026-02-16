# collectors/yf_collector.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone, date, time
from typing import List, Dict, Optional
from collectors.base_collector import BaseCollector
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from utils.technical_features import add_all_technical_features
from config.config import YF_MAX_PERIODS, DATA_INTERVALS, TICKERS

# Configure logger
logger = logging.getLogger(__name__)

class YFCollector(BaseCollector):
    def __init__(
            self,
            tickers: List[str] = None,
            timeframes: Optional[Dict[str, Dict[str, str]]] = None,
            lookback_days: int = 365,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            table_name: str = "yf_prices",
            db_path: str = ":memory:",
            **kwargs
    ):
        super().__init__(db_path=db_path, table_name=table_name, **kwargs)
        self.tickers = [str(t).strip().upper() for t in tickers] if tickers else list(TICKERS.keys())
        self.timeframes = timeframes or {
            tf: {"interval": interval, "period": YF_MAX_PERIODS.get(tf, "1d")} 
            for tf, interval in DATA_INTERVALS.items()
        }
        self.end_date = self._ensure_utc(end_date) if end_date else datetime.now(timezone.utc)
        self.start_date = self._ensure_utc(start_date) if start_date else self.end_date - timedelta(days=lookback_days)
        self.hash_keys = ["datetime", "ticker", "interval"]
        logger.info(
            f"[YFCollector] Initialized: {len(self.tickers)} tickers, "
            f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}, "
            f"Timeframes: {list(self.timeframes.keys())}"
        )

    def _ensure_utc(self, dt: Optional[datetime]) -> datetime:
        if isinstance(dt, date) and not isinstance(dt, datetime):
            dt = datetime.combine(dt, time.min)
        if dt.tzinfo is None:
            return dt.tz_localize('UTC')
        return dt.astimezone(timezone.utc)

    def _validate_batch_data(self, df: pd.DataFrame, tickers: List[str]) -> bool:
        """Validates the structure of the dataframe downloaded from yfinance."""
        if df.empty:
            logger.warning("Batch download returned an empty dataframe.")
            return False
        if not isinstance(df.columns, pd.MultiIndex):
            logger.warning(f"Batch download returned a single-level index, which is unexpected. Columns: {df.columns}")
            return False
        
        # Check if all tickers are present in the top level of the columns
        downloaded_tickers = df.columns.get_level_values(0).unique().tolist()
        if not all(ticker in downloaded_tickers for ticker in tickers):
            logger.warning(f"Batch download is missing tickers. Requested: {tickers}, Got: {downloaded_tickers}")
            return False
            
        logger.info("Batch data validation successful.")
        return True

    def _process_batch_df(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Processes the batch-downloaded dataframe to the required format."""
        df_list = []
        # The dataframe has a multi-index: ('TICKER', 'PriceMetric')
        for ticker in self.tickers:
            if ticker in df.columns:
                ticker_df = df[ticker].copy()
                ticker_df.reset_index(inplace=True)
                ticker_df.columns = [str(col).lower().replace(' ', '_') for col in ticker_df.columns]
                
                date_col_name = next((col for col in ticker_df.columns if 'date' in col), None)
                if date_col_name:
                    ticker_df.rename(columns={date_col_name: 'datetime'}, inplace=True)
                
                if 'datetime' in ticker_df.columns:
                    ticker_df['datetime'] = pd.to_datetime(ticker_df['datetime'], utc=True)
                    ticker_df['ticker'] = ticker
                    ticker_df['interval'] = tf
                    df_list.append(ticker_df)
        
        if not df_list:
            return pd.DataFrame()
            
        return pd.concat(df_list, ignore_index=True)


    def _fetch_single_ticker_tf(self, ticker: str, tf: str) -> Optional[pd.DataFrame]:
        """Fetches a single ticker and timeframe, used as a fallback."""
        try:
            yf_interval = self.timeframes[tf]["interval"]
            start_date_for_fetch = self.start_date
            if yf_interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
                limit_date = datetime.now(timezone.utc) - timedelta(days=59)
                if start_date_for_fetch < limit_date:
                    start_date_for_fetch = limit_date

            logger.info(f"Fallback: Fetching {ticker} ({tf}) from {start_date_for_fetch:%Y-%m-%d} to {self.end_date:%Y-%m-%d}")
            
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_date_for_fetch, end=self.end_date, interval=yf_interval,
                auto_adjust=False, progress=False
            )

            if df.empty:
                return None

            df.reset_index(inplace=True)
            df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
            
            date_col_name = next((col for col in df.columns if 'date' in col), 'datetime')
            df.rename(columns={date_col_name: 'datetime'}, inplace=True)
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df['ticker'] = ticker
            df['interval'] = tf
            return df
        except Exception as e:
            logger.error(f"Error in fallback fetch for {ticker} ({tf}): {e}")
            return None

    def fetch(self) -> pd.DataFrame:
        all_dfs = []
        for tf in self.timeframes.keys():
            yf_interval = self.timeframes[tf]["interval"]
            start_date_for_fetch = self.start_date
            if yf_interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
                limit_date = datetime.now(timezone.utc) - timedelta(days=59)
                if start_date_for_fetch < limit_date:
                    start_date_for_fetch = limit_date

            # --- Fast Path: Batch Download ---
            logger.info(f"Attempting fast batch download for timeframe: {tf}")
            try:
                batch_df = yf.download(
                    tickers=self.tickers,
                    start=start_date_for_fetch,
                    end=self.end_date,
                    interval=yf_interval,
                    progress=False,
                    auto_adjust=False,
                    group_by='ticker'
                )
                
                # Check if some tickers failed, which results in a single-level index
                if not isinstance(batch_df.columns, pd.MultiIndex) and not batch_df.empty:
                     # This indicates that download for some tickers might have failed.
                     logger.warning(f"Batch download for {tf} resulted in a single-level index. Switching to reliable fallback.")
                     raise ValueError("Partial failure in batch download")

                processed_df = self._process_batch_df(batch_df, tf)
                all_dfs.append(processed_df)
                logger.info(f"Fast path successful for timeframe: {tf}")
                continue # Move to next timeframe

            except Exception as e:
                logger.warning(f"Fast path failed for timeframe {tf} with error: {e}. Switching to reliable fallback.")

            # --- Reliable Fallback Path ---
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(self._fetch_single_ticker_tf, ticker, tf): ticker for ticker in self.tickers}
                for future in futures:
                    try:
                        result_df = future.result()
                        if result_df is not None:
                            all_dfs.append(result_df)
                    except Exception as exc:
                        logger.error(f'{futures[future]} generated an exception in fallback: {exc}')

        if not all_dfs:
            logger.error("Data collection failed for all tickers. No data to process.")
            return pd.DataFrame()

        df_all = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=['datetime', 'ticker', 'interval'])
        logger.info(f"Combined data shape: {df_all.shape}")

        df_all.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        
        logger.info("Applying technical analysis features...")
        df_all[['open', 'high', 'low', 'close', 'volume']] = df_all[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
        df_with_ta = add_all_technical_features(df_all)

        logger.info("Imputing any remaining NaNs in indicator columns...")
        indicator_cols = [c for c in df_with_ta.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'ticker', 'interval', 'datetime', 'adj_close', 'dividends', 'stock_splits']]
        if indicator_cols:
             df_with_ta[indicator_cols] = df_with_ta.groupby(['ticker', 'interval'])[indicator_cols].transform(lambda x: x.ffill().bfill())
        df_with_ta[indicator_cols] = df_with_ta[indicator_cols].fillna(0)
        
        logger.info("Finalizing data and saving to database...")
        self._save_batch(df_with_ta)
        return df_with_ta

    def _save_batch(self, df: pd.DataFrame):
        df_to_save = df.copy()
        df_to_save['datetime'] = df_to_save['datetime'].dt.tz_localize(None)
        records = df_to_save.to_dict(orient="records")
        self.save(records, strict=False)

    def collect(self) -> pd.DataFrame:
        return self.fetch()
