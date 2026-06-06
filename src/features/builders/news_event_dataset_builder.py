"""
News Event Dataset Builder (Refactored)
Orchestrates the construction of a news-based event dataset using modular components.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.utils.trading_calendar import TradingCalendar

from .news_event.candle_seeker import NewsCandleSeeker
from .news_event.enricher import NewsGlobalEnricher

# Import modular components
from .news_event.filter import NewsEventDataFilter

logger = ProjectLogger.get_logger(__name__)

class NewsEventDatasetBuilder:
    """
    Facade for building event-centric datasets from news.
    Delegates validation, temporal alignment, and global enrichment to specialized components.
    """

    def __init__(self, calendar: TradingCalendar, runtime_params: dict[str, Any] | None = None):
        self.calendar = calendar
        self.runtime_params = runtime_params or {}

        # 1. Configuration
        test_mode = self.runtime_params.get('test_mode', {})
        self.is_test_mode = test_mode.get('enabled', False)
        self.test_ticker = test_mode.get('test_ticker')
        self.timeframes = ['15m', '60m', '1d']

        # 2. Components
        self.filter = NewsEventDataFilter(is_test_mode=self.is_test_mode, test_ticker=self.test_ticker)
        self.seeker = NewsCandleSeeker(candle_features=[
            'open', 'high', 'low', 'close', 'volume',
            'RSI_14', 'SMA_20', 'EMA_20', 'MACD', 'ATR_14', 'BB_upper',
            'BB_lower', 'Stoch_K', 'Stoch_D'
        ])
        self.enricher = NewsGlobalEnricher()

        logger.info(f"NewsEventDatasetBuilder initialized (Modular). Test mode: {self.is_test_mode}")

    def build_dataset(self, news_df: pd.DataFrame, price_data: dict[str, pd.DataFrame],
                      macro_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """High-level orchestration of dataset construction."""
        if news_df.empty:
            return pd.DataFrame()

        self.filter.stats['total_news'] = len(news_df)
        self.timeframes = list(price_data.keys())

        pub_col = self.filter.find_publication_column(news_df)
        if not pub_col:
            return pd.DataFrame()

        filtered_tickers = self.filter.filter_tickers(tickers)
        records = self._process_all_news(news_df, pub_col, filtered_tickers, price_data, macro_data)

        return self._finalize_dataset(records)

    def _process_all_news(self, news_df: pd.DataFrame, pub_col: str, tickers: list[str],
                         price_data: dict[str, pd.DataFrame], macro_data: pd.DataFrame) -> list[dict]:
        records = []
        for idx, news in news_df.iterrows():
            try:
                published_at = pd.to_datetime(news[pub_col])
                record = self._build_record_for_news(news, published_at, tickers, price_data, macro_data)

                if record:
                    records.append(record)
                    self.filter.stats['valid_records'] += 1

                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(news_df)} news. Valid: {self.filter.stats['valid_records']}")
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.error(f"Error processing news index {idx}: {e}", exc_info=True)
                raise RuntimeError(f"Error processing news index {idx}") from e
        return records

    def _build_record_for_news(self, news: pd.Series, published_at: pd.Timestamp,
                               tickers: list[str], price_data: dict[str, pd.DataFrame],
                               macro_data: pd.DataFrame) -> dict | None:
        """Orchestrates building a single news record."""
        pub_at_norm = self.seeker.normalize_datetime(published_at)

        # Base news data
        record = {
            'news_id': news.get('hash', ''),
            'published_at': pub_at_norm,
            'news_title': news.get('title', ''),
            'news_sentiment': news.get('sentiment', 0.0)
        }

        # 1. Process candles for each ticker/timeframe
        for ticker in tickers:
            for tf in self.timeframes:
                if not self._add_candles_to_record(record, ticker, tf, price_data, pub_at_norm):
                    return None

        # 2. Add ticker identity
        record['ticker'] = tickers[0] if len(tickers) == 1 else None

        # 3. Global enrichment (Macro, long-term MAs, context)
        if not self.enricher.enrich_record(record, macro_data, pub_at_norm, tickers, price_data):
            self.filter.stats['filtered_missing_macro'] += 1
            return None

        return record

    def _add_candles_to_record(self, record: dict, ticker: str, tf: str,
                               price_data: dict[str, pd.DataFrame], pub_at: pd.Timestamp) -> bool:
        """Finds and adds candles for a specific ticker/timeframe."""
        if tf not in price_data:
            return False

        ticker_df = self._get_ticker_prices(price_data[tf], ticker)
        if ticker_df.empty:
            return False

        # Before candle
        candles_before = self.seeker.get_candles_before(ticker_df, pub_at, tf, n=1)
        if not candles_before:
            self.filter.stats['filtered_insufficient_before'] += 1
            return False
        candle_before = candles_before[0]

        # After candles
        candles_after = self.seeker.get_candles_after(ticker_df, pub_at, tf, n=2)
        if len(candles_after) < 2:
            self.filter.stats['filtered_insufficient_after'] += 1
            return False

        # Missing data check
        if self.filter.has_missing_data(candle_before) or any(self.filter.has_missing_data(c) for c in candles_after):
            self.filter.stats['filtered_missing_data'] += 1
            return False

        # Extract features
        record.update(self.seeker.extract_features(ticker, tf, candle_before, suffix=''))
        self._add_targets_from_candle(record, candle_before, ticker)
        record.update(self.seeker.extract_features(ticker, tf, candles_after[0], suffix='_+1'))
        record.update(self.seeker.extract_features(ticker, tf, candles_after[1], suffix='_+2'))

        return True

    def _get_ticker_prices(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if 'ticker' in df.columns:
            return df[df['ticker'] == ticker].copy()
        return df.copy()

    def _add_targets_from_candle(self, record: dict, candle: pd.Series, ticker: str):
        for col in candle.index:
            # audit-ignore: ARCHITECTURAL_USAGE
            if isinstance(col, str) and col.startswith('target_'):
                record[f'{ticker}_{col}'] = candle[col]

    def _finalize_dataset(self, records: list[dict]) -> pd.DataFrame:
        self.filter.log_stats()
        if not records:
            logger.warning("No valid records created!")
            return pd.DataFrame()
        logger.info(f"✅ Generated {len(records)} records")
        return pd.DataFrame(records)
