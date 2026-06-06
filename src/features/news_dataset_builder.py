import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.features.news_impact_classifier import NewsImpactClassifier
from src.utils.trading_calendar import TradingCalendar

from .builders.news_event.candle_seeker import NewsCandleSeeker
from .builders.news_event.enricher import NewsGlobalEnricher

# Import modular components
from .builders.news_event.filter import NewsEventDataFilter

logger = ProjectLogger.get_logger('NewsDatasetBuilder')


class NewsContextDatasetBuilder:
    """
    Будує датасет з структурою:
    1 рядок = 1 новина + контекст на момент + 2 свічки до/після для релевантних тікерів/таймфреймів
    """

    def __init__(self, config_manager: UnifiedConfigManager):
        self.config_manager = config_manager
        self.calendar = TradingCalendar()
        self.impact_classifier = NewsImpactClassifier(config_manager)
        self.tickers = self._get_tickers()
        self.timeframes = ['15m', '60m', '1d']
        self.n_candles_before = 2
        self.n_candles_after = 2

        # Modular components
        self.filter = NewsEventDataFilter()
        self.seeker = NewsCandleSeeker(candle_features=[]) # Empty list means extract all available columns
        self.enricher = NewsGlobalEnricher()

        logger.info(
            f'NewsDatasetBuilder initialized: {len(self.tickers)} tickers, {len(self.timeframes)} timeframes'
            )

    def _get_tickers(self) ->list[str]:
        """Отримати список тікерів з конфігурації"""
        assets_config = self.config_manager.get_config('assets', default={})
        active_preset = assets_config.get('active_preset')
        if active_preset:
            presets = assets_config.get('presets', {})
            preset_config = presets.get(active_preset, {})
            if 'tickers' in preset_config:
                tickers = preset_config['tickers']
                if isinstance(tickers, list):
                    return [str(t) for t in tickers]
        all_tickers = set()
        sectors = assets_config.get('sectors', {})
        for _sector_name, sector_data in sectors.items():
            if isinstance(sector_data, dict) and 'assets' in sector_data:
                all_tickers.update(sector_data['assets'])
        if all_tickers:
            return sorted(all_tickers)
        logger.warning('No tickers found in config, using defaults')
        return ['AAPL', 'NVDA', 'TSLA', 'AMD', 'GOOGL', 'MSFT', 'AMZN', 'META']

    def _filter_news_with_sufficient_candles(self, news_df: pd.DataFrame,
        prices_dict: dict[str, pd.DataFrame]) ->pd.DataFrame:
        """
        ✅ ІНТЕЛЕКТУАЛЬНИЙ ФІЛЬТР: Новина валідна якщо має дані для релевантних тікерів/таймфреймів
        """
        filtered_rows = []
        removed_reasons = {'no_relevant_combinations': 0, 'insufficient_candles': 0, 'datetime_error': 0}

        logger.info(f'🔍 Starting intelligent news filtering for {len(news_df)} articles')

        for idx, news_row in news_df.iterrows():
            try:
                news_text = f"{news_row.get('title', '')} {news_row.get('content', '')}"
                news_type = news_row.get('news_type', 'general')
                news_impact = self.impact_classifier.classify_impact(news_text, news_type)
                relevant_combinations = self.impact_classifier.get_relevant_combinations(news_impact)

                if not relevant_combinations:
                    removed_reasons['no_relevant_combinations'] += 1
                    continue

                has_valid_combination = False
                news_time = self.seeker.normalize_datetime(news_row['published_date'])

                for ticker, timeframe in relevant_combinations:
                    if timeframe not in prices_dict:
                        continue
                    ticker_prices = self._get_ticker_prices(prices_dict[timeframe], ticker)
                    if isinstance(ticker_prices, pd.DataFrame) and ticker_prices.empty:
                        continue

                    # Quick check for at least 1 candle before and after
                    before = self.seeker.get_candles_before(ticker_prices, news_time, timeframe, n=1)
                    after = self.seeker.get_candles_after(ticker_prices, news_time, timeframe, n=1)

                    if before and after:
                        has_valid_combination = True
                        break

                if has_valid_combination:
                    filtered_rows.append(idx)
                else:
                    removed_reasons['insufficient_candles'] += 1

            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.error(f'❌ Error filtering news {idx}: {e}', exc_info=True)
                removed_reasons['datetime_error'] += 1
                raise RuntimeError(f"Error filtering news {idx}") from e

        logger.info(f'✅ Filtering complete: {len(news_df)} → {len(filtered_rows)}')
        return news_df.loc[filtered_rows].reset_index(drop=True)

    def build_dataset(self, news_df: pd.DataFrame, prices_dict: dict[str, pd.DataFrame],
                      macro_df: pd.DataFrame | None = None,
                      market_sentiment_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Builds full dataset with news and context."""
        logger.info(f'Building news context dataset for {len(news_df)} articles')

        news_filtered = self._filter_news_with_sufficient_candles(news_df, prices_dict)
        if news_filtered.empty:
            logger.warning('No news articles with sufficient candles after publication!')
            return pd.DataFrame()

        dataset_rows = []
        for idx, news_row in news_filtered.iterrows():
            try:
                row = self._build_news_row(news_row, prices_dict, macro_df, market_sentiment_df)
                if row:
                    dataset_rows.append(row)
                if (idx + 1) % 100 == 0:
                    logger.info(f'Processed {idx + 1}/{len(news_filtered)} news articles')
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.error(f'Failed to process news {idx}: {e}', exc_info=True)
                raise RuntimeError(f"Failed to process news {idx}") from e

        if not dataset_rows:
            logger.error('No valid rows generated!')
            return pd.DataFrame()

        dataset_df = pd.DataFrame(dataset_rows)
        logger.info(f'✅ Dataset built: {len(dataset_df)} rows, {len(dataset_df.columns)} columns')
        return self._add_metadata(dataset_df)

    def _build_news_row(self, news_row: pd.Series, prices_dict: dict[str, pd.DataFrame],
                        macro_df: pd.DataFrame | None,
                        market_sentiment_df: pd.DataFrame | None) -> dict[str, Any] | None:
        """Orchestrates building a single news row with full context."""
        try:
            # Debugging: check types
            if not isinstance(news_row, pd.Series):
                logger.error(f"Expected pd.Series, got {type(news_row)}")
                return None

            news_time = news_row['published_date']
            pub_at_norm = self.seeker.normalize_datetime(news_time)

            row = {
                'news_id': f"{news_row.get('source', 'unknown')}_{pub_at_norm.strftime('%Y%m%d_%H%M%S')}",
                'news_timestamp': pub_at_norm,
                'news_title': news_row.get('title', ''),
                'news_sentiment': news_row.get('sentiment', 0.0),
                'news_type': news_row.get('news_type', 'general'),
                'news_source': news_row.get('source', 'unknown')
            }

            # 1. Context (Macro, Sentiment, Temporal)
            row.update(self._get_macro_context(pub_at_norm, macro_df))
            row.update(self._get_market_sentiment_context(pub_at_norm, market_sentiment_df))
            row.update(self._get_temporal_features(pub_at_norm))

            # 2. Ticker-specific context (Before and After)
            for ticker in self.tickers:
                for tf in self.timeframes:
                    if tf not in prices_dict:
                        continue
                    ticker_prices = self._get_ticker_prices(prices_dict[tf], ticker)
                    # Use .empty for DataFrame check
                    if not isinstance(ticker_prices, pd.DataFrame) or ticker_prices.empty:
                        continue

                    # Before context
                    candles_before = self.seeker.get_candles_before(ticker_prices, pub_at_norm, tf, n=self.n_candles_before)
                    for i, candle in enumerate(candles_before, start=1):
                        self._add_candle_to_row(row, candle, f"{ticker}_{tf}_before_{i}")

                    # After context
                    candles_after = self.seeker.get_candles_after(ticker_prices, pub_at_norm, tf, n=self.n_candles_after)
                    for i, candle in enumerate(candles_after, start=1):
                        self._add_candle_to_row(row, candle, f"{ticker}_{tf}_after_{i}")

            return row if len(row) > 20 else None
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error in _build_news_row: {e}", exc_info=True)
            raise RuntimeError("Error in _build_news_row") from e

    def _add_candle_to_row(self, row: dict, candle: pd.Series, prefix: str):
        """Extracts all features from a candle into the row."""
        row[f'{prefix}_datetime'] = candle.get('datetime')
        for col in candle.index:
            if col not in ['datetime', 'ticker', 'interval']:
                row[f'{prefix}_{col}'] = candle[col]

    def _get_ticker_prices(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if 'ticker' in df.columns:
            return df[df['ticker'] == ticker].copy()
        return df.copy()

    def _get_macro_context(self, timestamp: datetime, macro_df: pd.DataFrame | None) -> dict[str, Any]:
        if macro_df is None or not isinstance(macro_df, pd.DataFrame) or macro_df.empty:
            return self.enricher.get_macro_features(pd.DataFrame(), timestamp)
        return self.enricher.get_macro_features(macro_df, timestamp)

    def _get_market_sentiment_context(self, timestamp: datetime,
                                      market_sentiment_df: pd.DataFrame | None) -> dict[str, Any]:
        """Отримати ринковий сентимент на момент новини"""
        if market_sentiment_df is None or not isinstance(market_sentiment_df, pd.DataFrame) or market_sentiment_df.empty:
            return {}
        date_col = 'datetime' if 'datetime' in market_sentiment_df.columns else 'date'
        if date_col not in market_sentiment_df.columns:
            return {}
        relevant = market_sentiment_df[market_sentiment_df[date_col] <= timestamp].tail(1)
        if not isinstance(relevant, pd.DataFrame) or relevant.empty:
            return {}

        res = {}
        row = relevant.iloc[0]
        for f in ['vix', 'fear_greed_index', 'put_call_ratio']:
            if f in row:
                res[f'sentiment_{f}'] = row[f]
        return res

    def _get_temporal_features(self, timestamp: datetime) -> dict[str, Any]:
        """Отримати часові фічі"""
        return {
            'temporal_day_of_week': timestamp.weekday(),
            'temporal_hour_of_day': timestamp.hour,
            'temporal_is_trading_hours': self._is_trading_hours(timestamp),
            'temporal_is_first_hour': timestamp.hour == 9 and timestamp.minute < 60,
            'temporal_is_last_hour': timestamp.hour == 15 and timestamp.minute >= 0
        }

    def _is_trading_hours(self, timestamp: datetime) ->bool:
        """Перевірити, чи це торгові години"""
        if not self.calendar.is_trading_day(timestamp):
            return False
        hour = timestamp.hour
        minute = timestamp.minute
        if hour < 9 or hour >= 16:
            return False
        if hour == 9 and minute < 30:
            return False
        return True

    def _add_metadata(self, dataset_df: pd.DataFrame) ->pd.DataFrame:
        """Додати metadata колонку з інформацією про датасет"""
        logger.info('Adding metadata to dataset...')
        dataset_df['dataset_version'] = '1.0'
        dataset_df['generated_at'] = datetime.now().isoformat()
        dataset_df['n_tickers'] = len(self.tickers)
        dataset_df['n_timeframes'] = len(self.timeframes)
        return dataset_df

    def save_dataset(self, dataset_df: pd.DataFrame, output_path: Path):
        """Зберегти датасет"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_df.to_parquet(output_path, compression='snappy', index=False)
        logger.info(f'✅ Dataset saved to {output_path}')
        metadata = {
            'rows': len(dataset_df),
            'columns': len(dataset_df.columns),
            'tickers': self.tickers,
            'timeframes': self.timeframes,
            'n_candles_before': self.n_candles_before,
            'n_candles_after': self.n_candles_after,
            'generated_at': datetime.now().isoformat()
        }
        metadata_path = output_path.parent / f'{output_path.stem}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f'✅ Metadata saved to {metadata_path}')
