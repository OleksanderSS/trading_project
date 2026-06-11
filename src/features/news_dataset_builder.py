"""
News Context Dataset Builder
Створює датасет з новинами + контекст до/після для релевантних тікерів та таймфреймів
"""

from datetime import datetime
from typing import Any

import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.features.news_impact_classifier import NewsImpactClassifier
from src.utils.trading_calendar import TradingCalendar

logger = ProjectLogger.get_logger("NewsDatasetBuilder")


class NewsContextDatasetBuilder:
    """
    Будує датасет з структурою:
    1 рядок = 1 новина + контекст на момент + 2 свічки до/після для релевантних тікерів/таймфреймів
    """

    def __init__(self, config_manager: UnifiedConfigManager):
        self.config_manager = config_manager
        self.calendar = TradingCalendar()

        # Ініціалізувати класифікатор впливу новин
        self.impact_classifier = NewsImpactClassifier(config_manager)

        # Конфігурація
        self.tickers = self._get_tickers()
        self.timeframes = ['15m', '60m', '1d']
        self.n_candles_before = 2
        self.n_candles_after = 2

        # Показники для кожної свічки
        # Використовуємо ВСІ фічі, що генеруються в Stage 3 (125+ фіч)
        # Вони будуть автоматично взяті з prices_df
        self.candle_features = None  # Буде заповнено динамічно з prices_df

        # Мінімальний набір (якщо prices_df не має інших колонок)
        self.base_candle_features = [
            'open', 'high', 'low', 'close', 'volume'
        ]

        logger.info(f"NewsDatasetBuilder initialized: {len(self.tickers)} tickers, {len(self.timeframes)} timeframes")

    @staticmethod
    def _normalize_datetime(dt: Any) -> pd.Timestamp:
        """
        Robustly normalize datetime to naive UTC Timestamp.
        """
        if dt is None or pd.isna(dt):
            return pd.Timestamp.now().tz_localize(None)

        try:
            # Handle if it's a Series/DataFrame (take first element)
            if hasattr(dt, 'iloc'):
                dt = dt.iloc[0] if len(dt) > 0 else pd.Timestamp.now()

            # Convert to Timestamp if not already
            if not isinstance(dt, pd.Timestamp):
                dt = pd.to_datetime(dt)

            # Strict timezone handling
            if dt.tz is not None:
                dt = dt.tz_convert('UTC').tz_localize(None)

            return dt
        except Exception:
            return pd.Timestamp.now().tz_localize(None)

    @staticmethod
    def _normalize_datetime_series(series: Any) -> pd.Series:
        """
        Robustly normalize a Series to naive UTC datetime.
        Handles DataFrames by taking the first matching column.
        """
        # If it's a DataFrame, try to get the first column
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        # Convert to Series if needed
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(series):
            series = pd.to_datetime(series, errors='coerce')

        # Drop timezone
        if hasattr(series, 'dt') and series.dt.tz is not None:
            return series.dt.tz_convert('UTC').dt.tz_localize(None)

        return series

    def _get_news_time(self, news_row: pd.Series) -> pd.Timestamp:
        """
        Find news timestamp using common aliases.
        """
        aliases = ['published_date', 'timestamp', 'date', 'news_timestamp', 'time']
        for alias in aliases:
            if alias in news_row and not pd.isna(news_row[alias]):
                return self._normalize_datetime(news_row[alias])

        # Fallback to current time if no column found
        return pd.Timestamp.now().tz_localize(None)

    def _get_tickers(self) -> list[str]:
        """Отримати список тікерів з конфігурації"""
        assets_config = self.config_manager.get_config('assets', default={})

        # Спробувати отримати з active_preset
        active_preset = assets_config.get('active_preset')
        if active_preset:
            presets = assets_config.get('presets', {})
            preset_config = presets.get(active_preset, {})
            if 'tickers' in preset_config:
                tickers = preset_config['tickers']
                if isinstance(tickers, list):
                    return [str(t) for t in tickers]

        # Спробувати отримати з sectors
        all_tickers = set()
        sectors = assets_config.get('sectors', {})
        for _sector_name, sector_data in sectors.items():
            if isinstance(sector_data, dict) and 'assets' in sector_data:
                all_tickers.update(sector_data['assets'])

        if all_tickers:
            return sorted(all_tickers)

        # Fallback - default tickers
        logger.warning("No tickers found in config, using defaults")
        return ['AAPL', 'NVDA', 'TSLA', 'AMD', 'GOOGL', 'MSFT', 'AMZN', 'META']

    def _filter_news_with_sufficient_candles(
        self,
        news_df: pd.DataFrame,
        preprocessed_prices: dict[str, dict[str, pd.DataFrame]]
    ) -> pd.DataFrame:
        """
        ✅ ІНТЕЛЕКТУАЛЬНИЙ ФІЛЬТР: Новина валідна якщо має дані для релевантних тікерів/таймфреймів
        """
        filtered_rows = []
        removed_reasons = {
            'no_relevant_combinations': 0,
            'insufficient_candles': 0,
            'datetime_error': 0
        }

        logger.info(f" Starting intelligent news filtering for {len(news_df)} articles")

        for idx, news_row in news_df.iterrows():
            try:
                # 1. Класифікувати вплив новини
                news_text = f"{news_row.get('title', '')} {news_row.get('content', '')}"
                news_type = news_row.get('news_type', 'general')

                news_impact = self.impact_classifier.classify_impact(news_text, news_type)

                # 2. Отримати релевантні комбінації
                relevant_combinations = self.impact_classifier.get_relevant_combinations(news_impact)

                if not relevant_combinations:
                    removed_reasons['no_relevant_combinations'] += 1
                    continue

                # 3. Перевірити чи є достатньо свічок
                has_valid_combination = False
                for ticker, timeframe in relevant_combinations:
                    if timeframe not in preprocessed_prices: continue
                    ticker_groups = preprocessed_prices[timeframe]
                    if ticker not in ticker_groups: continue
                    ticker_prices = ticker_groups[ticker]

                    # ✅ Optimized: datetimes are already normalized
                    news_time = self._get_news_time(news_row).tz_localize(None)

                    before_mask = ticker_prices['datetime'] < news_time
                    after_mask = ticker_prices['datetime'] > news_time

                    if before_mask.any() and after_mask.any():
                        has_valid_combination = True
                        break

                if has_valid_combination:
                    filtered_rows.append(idx)
                else:
                    removed_reasons['insufficient_candles'] += 1

            except Exception as e:
                logger.warning(f"❌ Error processing news {idx}: {e}")
                removed_reasons['datetime_error'] += 1
                continue

        # Логування результатів фільтрації
        removed_count = len(news_df) - len(filtered_rows)
        logger.info(f"✅ Intelligent news filtering: {len(news_df)} → {len(filtered_rows)} (removed {removed_count})")
        return news_df.loc[filtered_rows].reset_index(drop=True)

    def build_dataset(
        self,
        news_df: pd.DataFrame,
        prices_dict: dict[str, pd.DataFrame],
        macro_df: pd.DataFrame | None = None,
        market_sentiment_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Побудувати повний датасет з новинами та контекстом
        """
        logger.info(f"Building news context dataset for {len(news_df)} news articles")

        # ✅ FIX: Ensure prices_dict DataFrames don't have duplicate columns (causes "cannot assemble with duplicate keys")
        for tf in prices_dict:
            if not prices_dict[tf].empty:
                prices_dict[tf] = prices_dict[tf].loc[:, ~prices_dict[tf].columns.duplicated()].copy()

        # ✅ FIX: Remove duplicate columns if any in news_df
        if not news_df.empty:
            news_df = news_df.loc[:, ~news_df.columns.duplicated()].copy()

        # ✅ FIX: Remove duplicate news entries before processing
        if not news_df.empty and 'news_id' in news_df.columns:
            news_df = news_df.drop_duplicates(subset=['news_id'])
        elif not news_df.empty and 'title' in news_df.columns:
            news_df = news_df.drop_duplicates(subset=['title', 'published_date'])

        # ✅ OPTIMIZATION: Pre-process and pre-normalize datetimes & group by ticker to avoid filtering 167k times!
        preprocessed_prices = {}
        for tf, df in prices_dict.items():
            if df.empty or 'datetime' not in df.columns:
                continue
            df = df.copy()
            df['datetime'] = self._normalize_datetime_series(df['datetime'])

            ticker_groups = {}
            for ticker in self.tickers:
                t_df = df[df['ticker'] == ticker].copy()
                if not t_df.empty:
                    # Sort by datetime to ensure correct tail/head operations
                    t_df = t_df.sort_values('datetime')
                    ticker_groups[ticker] = t_df
            preprocessed_prices[tf] = ticker_groups

        # ✅ КРОК 1: Фільтрація новин
        news_df_filtered = self._filter_news_with_sufficient_candles(news_df, preprocessed_prices)
        logger.info(f"✅ Filtered news: {len(news_df)} → {len(news_df_filtered)} (removed {len(news_df) - len(news_df_filtered)} without 1+ candles after)")

        if news_df_filtered.empty:
            logger.warning("No news articles with sufficient candles after publication!")
            return pd.DataFrame()

        dataset_rows = []

        for idx, news_row in news_df_filtered.iterrows():
            try:
                row = self._build_news_row(news_row, preprocessed_prices, macro_df, market_sentiment_df)
                if row is not None:
                    dataset_rows.append(row)
            except Exception as e:
                logger.warning(f"Failed to process news {idx}: {e}")
                continue

        dataset_df = pd.DataFrame(dataset_rows)
        logger.info(f"✅ Dataset built: {len(dataset_df)} rows, {len(dataset_df.columns)} columns")
        return self._add_metadata(dataset_df)

    def _build_news_row(
        self,
        news_row: pd.Series,
        preprocessed_prices: dict[str, dict[str, pd.DataFrame]],
        macro_df: pd.DataFrame | None,
        market_sentiment_df: pd.DataFrame | None
    ) -> dict[str, Any] | None:
        """
        Структура рядка: [НОВИНА] + [МАКРО] + [КОНТЕКСТ ДО] + [РЕАКЦІЯ ПІСЛЯ]
        """
        row = {}
        news_time = self._get_news_time(news_row)
        news_time_normalized = self._normalize_datetime(news_time)

        # ========== БЛОК 1: НОВИНА ==========
        row['news_id'] = f"{news_row.get('source', 'unknown')}_{news_time.strftime('%Y%m%d_%H%M%S')}"
        row['news_timestamp'] = news_time
        row['news_title'] = news_row.get('title', '')
        row['news_sentiment'] = news_row.get('sentiment', 0.0)
        row['news_type'] = news_row.get('news_type', 'general')
        row['news_source'] = news_row.get('source', 'unknown')

        # ========== БЛОК 2: МАКРО КОНТЕКСТ ==========
        row.update(self._get_macro_context(news_time, macro_df))
        row.update(self._get_market_sentiment_context(news_time, market_sentiment_df))
        row.update(self._get_temporal_features(news_time))

        # ========== БЛОК 3: КОНТЕКСТ ДО/ПІСЛЯ ==========
        for ticker in self.tickers:
            for timeframe in self.timeframes:
                if timeframe not in preprocessed_prices: continue
                ticker_groups = preprocessed_prices[timeframe]
                if ticker not in ticker_groups: continue
                ticker_prices = ticker_groups[ticker]

                before_candles = ticker_prices[ticker_prices['datetime'] < news_time_normalized].tail(2)
                after_candles = ticker_prices[ticker_prices['datetime'] > news_time_normalized].head(2)

                for i, (_idx, candle) in enumerate(before_candles.iloc[-self.n_candles_before:].iterrows(), start=1):
                    prefix = f"{ticker}_{timeframe}_before_{i}"
                    row.update({f"{prefix}_{k}": v for k, v in candle.items() if k not in ['ticker', 'interval']})

                for i, (_idx, candle) in enumerate(after_candles.iloc[:self.n_candles_after].iterrows(), start=1):
                    prefix = f"{ticker}_{timeframe}_after_{i}"
                    row.update({f"{prefix}_{k}": v for k, v in candle.items() if k not in ['ticker', 'interval']})

        return row if len(row) > 10 else None

    def _get_macro_context(self, timestamp: datetime, macro_df: pd.DataFrame | None) -> dict[str, Any]:
        if macro_df is None or macro_df.empty: return {}
        date_col = 'datetime' if 'datetime' in macro_df.columns else 'date'
        relevant_macro = macro_df[macro_df[date_col] <= timestamp].tail(1)
        return relevant_macro.iloc[0].to_dict() if not relevant_macro.empty else {}

    def _get_market_sentiment_context(self, timestamp: datetime, market_sentiment_df: pd.DataFrame | None) -> dict[str, Any]:
        if market_sentiment_df is None or market_sentiment_df.empty: return {}
        date_col = 'datetime' if 'datetime' in market_sentiment_df.columns else 'date'
        relevant_sentiment = market_sentiment_df[market_sentiment_df[date_col] <= timestamp].tail(1)
        return relevant_sentiment.iloc[0].to_dict() if not relevant_sentiment.empty else {}

    def _get_temporal_features(self, timestamp: datetime) -> dict[str, Any]:
        return {
            'temporal_day_of_week': timestamp.weekday(),
            'temporal_hour_of_day': timestamp.hour,
            'temporal_is_trading_hours': self._is_trading_hours(timestamp),
        }

    def _is_trading_hours(self, timestamp: datetime) -> bool:
        if not self.calendar.is_trading_day(timestamp): return False
        return 9 <= timestamp.hour < 16

    def _add_metadata(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        dataset_df['dataset_version'] = '1.0'
        return dataset_df
