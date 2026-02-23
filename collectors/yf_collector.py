# collectors/yf_collector.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone, date, time
from typing import List, Dict, Optional
from collectors.base_collector import BaseCollector
import logging
import traceback
from utils.technical_features import add_all_technical_features
from config.config import YF_MAX_PERIODS, DATA_INTERVALS, TICKERS

logger = logging.getLogger("trading_project.yf_collector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False


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

        if isinstance(tickers, dict):
            self.tickers = list(tickers.keys())
        elif isinstance(tickers, (list, set, tuple)):
            self.tickers = [str(t).strip().upper() for t in tickers]
        elif isinstance(tickers, str):
            self.tickers = [tickers.strip().upper()]
        else:
            self.tickers = list(TICKERS.keys())

        if not timeframes:
            timeframes = {}
            for tf in DATA_INTERVALS.keys():
                timeframes[tf] = {"interval": DATA_INTERVALS[tf], "period": YF_MAX_PERIODS.get(tf, "1d")}
        self.timeframes = timeframes

        if isinstance(end_date, datetime):
            current_end_date = end_date
        elif isinstance(end_date, date):
            current_end_date = datetime.combine(end_date, time.max)
        else:
            current_end_date = datetime.now()

        self.end_date = current_end_date.astimezone(timezone.utc).replace(tzinfo=None)
        self.start_date = start_date or (self.end_date - timedelta(days=lookback_days))
        if self.start_date.tzinfo is not None:
            self.start_date = self.start_date.astimezone(timezone.utc).replace(tzinfo=None)

        self.hash_keys = ["datetime", "ticker", "interval"]

        logger.info(
            f"[YFCollector] Ініціалізовано: {len(self.tickers)} тікерів, "
            f"{self.start_date.strftime('%Y-%m-%d')} - {self.end_date.strftime('%Y-%m-%d')}, "
            f"таймфрейми: {list(self.timeframes.keys())}"
        )

    def fetch_prices(self,
        ticker: str,
        tf: str,
        period: str,
        start_date: datetime = None,
        end_date: datetime = None) -> pd.DataFrame:
        """Отримує історичні ціни для одного тікера та таймфрейму."""
        try:
            yf_interval = DATA_INTERVALS.get(tf, tf)

            actual_start_date = start_date or self.start_date
            actual_end_date = end_date or self.end_date

            if actual_start_date.tzinfo is not None:
                actual_start_date = actual_start_date.astimezone(timezone.utc).replace(tzinfo=None)
            if actual_end_date.tzinfo is not None:
                actual_end_date = actual_end_date.astimezone(timezone.utc).replace(tzinfo=None)

            if yf_interval in ['1m', '5m', '15m', '30m', '60m', '90m']:
                max_start_date = datetime.now() - timedelta(days=59)
                if actual_start_date < max_start_date:
                    actual_start_date = max_start_date

            self.logger.info(
                f"[YFCollector] Завантаження {ticker} ({tf}) "
                f"з {actual_start_date.strftime('%Y-%m-%d')} по {actual_end_date.strftime('%Y-%m-%d')}"
            )
            df = yf.download(
                ticker,
                start=actual_start_date.strftime('%Y-%m-%d'),
                end=actual_end_date.strftime('%Y-%m-%d'),
                interval=yf_interval,
                progress=False,
                auto_adjust=False,
                group_by='column'  # Helps ensure single-level columns
            )

            if df.empty:
                self.logger.warning(f"[YFCollector] ⚠️ Немає даних для {ticker} ({tf})")
                return pd.DataFrame()

            # Flatten MultiIndex columns if they exist, which can happen unexpectedly
            if isinstance(df.columns, pd.MultiIndex):
                self.logger.warning(f"[YFCollector] Виявлено несподіваний MultiIndex для {ticker} ({tf}). Вирівнювання колонок.")
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            # Standardize all column names
            df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]

            # Robustly find and rename the date column
            date_col_name = None
            if 'datetime' in df.columns:
                date_col_name = 'datetime'
            elif 'date' in df.columns:
                date_col_name = 'date'
            
            if date_col_name and date_col_name != 'datetime':
                df.rename(columns={date_col_name: 'datetime'}, inplace=True)

            if 'datetime' not in df.columns:
                self.logger.error(f"[YFCollector] 🚨 ФАТАЛЬНА ПОМИЛКА: колонка 'datetime' НЕ ЗНАЙДЕНА для {ticker} ({tf}) після всіх перетворень. Колонки: {df.columns.tolist()}")
                return pd.DataFrame()

            if hasattr(df["datetime"].dt, "tz") and df["datetime"].dt.tz is not None:
                df["datetime"] = df["datetime"].dt.tz_localize(None)

            df["ticker"] = ticker
            df["interval"] = tf

            self.logger.info(f"[YFCollector] ✅ {ticker} {tf} завантажено: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"[YFCollector] Помилка завантаження {ticker} ({tf}): {e}\n{traceback.format_exc()}")
            return pd.DataFrame()

    def fetch(self) -> pd.DataFrame:
        all_dfs = []
        for ticker in self.tickers:
            for tf, params in self.timeframes.items():
                period = params.get("period", YF_MAX_PERIODS.get(tf, "1y"))
                start_date = params.get("start_date", self.start_date)
                end_date = params.get("end_date", self.end_date)
                
                df = self.fetch_prices(ticker, tf, period, start_date, end_date)
                
                if not df.empty:
                    all_dfs.append(df)

        if not all_dfs:
            logger.warning("[YFCollector] Жоден тікер не повернув дані")
            return pd.DataFrame()

        df_all = pd.concat(all_dfs, ignore_index=True)

        # Add technical features to the entire concatenated dataframe
        df_all = add_all_technical_features(df_all)

        # Fill NA values for indicator columns
        indicator_cols = [c for c in df_all.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'ticker', 'interval', 'datetime', 'adj_close']]
        df_all[indicator_cols] = df_all[indicator_cols].ffill().bfill().fillna(0)
        
        if 'datetime' not in df_all.columns:
            logger.error("[YFCollector] 致命的なエラー: 'datetime' column is missing after technical feature calculation.")
            return pd.DataFrame()

        df_all["published_at"] = pd.to_datetime(df_all["datetime"])
        if hasattr(df_all["published_at"].dt, "tz") and df_all["published_at"].dt.tz is not None:
            df_all["published_at"] = df_all["published_at"].dt.tz_localize(None)

        self._save_batch(df_all)
        return df_all

    def _save_batch(self, df_all: pd.DataFrame):
        records = df_all.to_dict(orient="records")
        self.save(records, strict=self.strict)

    def collect(self) -> pd.DataFrame:
        return self.fetch()
