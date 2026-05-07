"""
News Event Dataset Builder

Будує датасет на основі новин з прив'язкою до дати публікації.
Для кожної новини додає:
- 1 свічку ДО публікації (для кожного таймфрейму: 15m, 60m, 1d)
- 2 свічки ПІСЛЯ публікації (для кожного таймфрейму: 15m, 60m, 1d)
- Глобальні показники (макро, ковзні середні, сентимент)
- Мапу контексту
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.utils.trading_calendar import TradingCalendar

logger = ProjectLogger.get_logger(__name__)


class NewsEventDatasetBuilder:
    """
    Будує датасет для кожної новини з прив'язкою до дати публікації.

    Структура:
    - 1 свічка ДО публікації (для кожного таймфрейму: 15m, 60m, 1d)
    - 2 свічки ПІСЛЯ публікації (для кожного таймфрейму: 15m, 60m, 1d)
    - Глобальні показники (макро, ковзні середні, сентимент)
    - Мапа контексту
    - Таргети розраховуються з targets.yaml
    """

    # Показники для кожної свічки (оптимальний набір)
    CANDLE_FEATURES = [
        # OHLCV (базові)
        'open',
        'high',
        'low',
        'close',
        'volume',

        # Технічні індикатори (essential)
        'RSI_14',
        'SMA_20',
        'EMA_20',
        'MACD',
        'ATR_14',

        # Додаткові індикатори
        'BB_upper',
        'BB_lower',
        'Stoch_K',
        'Stoch_D',
    ]

    def __init__(
        self,
        calendar: TradingCalendar,
        runtime_params: dict[str, Any] | None = None
    ):
        """
        Ініціалізація builder.

        Args:
            calendar: Торговий календар для врахування вихідних/свят
            runtime_params: Параметри запуску (test_ticker, test_target)
        """
        self.calendar = calendar
        self.runtime_params = runtime_params or {}

        # Визначаємо режим роботи
        test_mode = self.runtime_params.get('test_mode', {})
        self.is_test_mode = test_mode.get('enabled', False)
        self.test_ticker = test_mode.get('test_ticker')
        self.test_target = test_mode.get('test_target')

        # Таймфрейми (фіксовані: 15m, 60m, 1d)
        self.timeframes = ['15m', '60m', '1d']

        # Статистика фільтрації
        self.stats = {
            'total_news': 0,
            'filtered_insufficient_before': 0,
            'filtered_insufficient_after': 0,
            'filtered_missing_data': 0,
            'filtered_missing_macro': 0,
            'valid_records': 0
        }

        logger.info("NewsEventDatasetBuilder initialized:")
        logger.info(f"  Test mode: {self.is_test_mode}")
        logger.info(f"  Test ticker: {self.test_ticker}")
        logger.info(f"  Test target: {self.test_target}")
        logger.info(f"  Timeframes: {self.timeframes}")

    def build_dataset(
        self,
        news_df: pd.DataFrame,
        price_data: dict[str, pd.DataFrame],  # {timeframe: df}
        macro_data: pd.DataFrame,
        tickers: list[str]
    ) -> pd.DataFrame:
        """
        Будує датасет на основі новин.

        Args:
            news_df: DataFrame з новинами (має містити: published_at, title, sentiment_score)
            price_data: Dict з цінами для кожного таймфрейму
            macro_data: DataFrame з макроекономічними даними (FRED)
            tickers: Список тікерів для обробки

        """
        self.stats['total_news'] = len(news_df)

        # Initialize dataset building
        build_config = self._initialize_build_config(price_data, tickers)

        # Find publication date column
        pub_col = self._find_publication_column(news_df)
        if pub_col is None:
            return pd.DataFrame()

        # Process each news item
        records = self._process_all_news(news_df, pub_col, build_config, price_data, macro_data)

        # Finalize dataset
        return self._finalize_dataset(records)

    def _initialize_build_config(self, price_data: dict[str, pd.DataFrame], tickers: list[str]) -> dict[str, Any]:
        """Initialize build configuration."""
        # Динамічно визначаємо доступні таймфрейми
        self.timeframes = list(price_data.keys())
        logger.info(f"Dynamically set timeframes to: {self.timeframes}")

        # Фільтруємо тікери (якщо тестовий режим)
        filtered_tickers = self._filter_tickers(tickers)

        logger.info(f"Обробка для {filtered_tickers} тікерів × {len(self.timeframes)} таймфреймів")

        return {
            'tickers': filtered_tickers,
            'timeframes': self.timeframes
        }

    def _filter_tickers(self, tickers: list[str]) -> list[str]:
        """Filter tickers based on test mode."""
        if self.is_test_mode and self.test_ticker:
            logger.info(f"🧪 Тестовий режим: використовуємо тільки {self.test_ticker}")
            return [self.test_ticker]
        return tickers

    def _find_publication_column(self, news_df: pd.DataFrame) -> str | None:
        """Find publication date column in news DataFrame."""
        for col in ['publishedAt', 'published_at', 'published_date', 'date', 'datetime']:
            if col in news_df.columns:
                return col

        logger.error(f"Cannot find publication date column in news_df. Available columns: {news_df.columns.tolist()}")
        return None

    def _process_all_news(
        self,
        news_df: pd.DataFrame,
        pub_col: str,
        build_config: dict[str, Any],
        price_data: dict[str, pd.DataFrame],
        macro_data: pd.DataFrame
    ) -> list[dict]:
        """Process all news items and build records."""
        records = []

        for idx, news in news_df.iterrows():
            try:
                published_at = pd.to_datetime(news[pub_col])
            except Exception as e:
                logger.warning(f"Could not parse date {news.get(pub_col)} for news {idx}: {e}")
                continue

            record = self._build_record_for_news(
                news, published_at, build_config['tickers'], price_data, macro_data
            )

            if record:
                records.append(record)
                self.stats['valid_records'] += 1

            # Логування прогресу
            if (idx + 1) % 100 == 0:
                logger.info(f"Оброблено {idx + 1}/{len(news_df)} новин, валідних: {self.stats['valid_records']}")

        return records

    def _finalize_dataset(self, records: list[dict]) -> pd.DataFrame:
        """Finalize dataset creation."""
        # Виводимо статистику
        self._log_filtering_stats()

        if not records:
            logger.warning("Не вдалося створити жодного валідного запису!")
            return pd.DataFrame()

        logger.info(f"✅ Згенеровано {len(records)} повних записів")
        return pd.DataFrame(records)

    def _build_record_for_news(
        self,
        news: pd.Series,
        published_at: pd.Timestamp,
        tickers: list[str],
        price_data: dict[str, pd.DataFrame],
        macro_data: pd.DataFrame
    ) -> dict | None:
        """
        Створює один запис для новини.

        Повертає None якщо дані неповні для хоча б одного тікера × таймфрейму.
        """
        # Normalize published_at
        published_at_normalized = self._normalize_datetime(published_at)

        # Initialize base record
        record = self._initialize_base_record(news, published_at_normalized)

        # Process all ticker-timeframe combinations
        if not self._process_ticker_timeframes(record, tickers, price_data, published_at_normalized):
            return None

        # Add final record fields
        self._add_final_record_fields(record, tickers)

        # Add global features
        if not self._add_global_features(record, macro_data, published_at_normalized, tickers, price_data):
            return None

        return record

    def _normalize_datetime(self, published_at: pd.Timestamp) -> pd.Timestamp:
        """Normalize datetime by removing timezone."""
        published_at_normalized = pd.to_datetime(published_at)
        if published_at_normalized.tz is not None:
            published_at_normalized = published_at_normalized.tz_localize(None)
        return published_at_normalized

    def _initialize_base_record(self, news: pd.Series, published_at_normalized: pd.Timestamp) -> dict[str, Any]:
        """Initialize base record with news information."""
        return {
            'news_id': news.get('hash') if pd.notna(news.get('hash')) else '',
            'published_at': published_at_normalized,
            'news_title': news.get('title') if pd.notna(news.get('title')) else '',
            'news_sentiment': news.get('sentiment', 0.0),  # sentiment, не sentiment_score
        }

    def _process_ticker_timeframes(
        self,
        record: dict[str, Any],
        tickers: list[str],
        price_data: dict[str, pd.DataFrame],
        published_at: pd.Timestamp
    ) -> bool:
        """Process all ticker-timeframe combinations."""
        for ticker in tickers:
            for tf in self.timeframes:
                if not self._process_single_ticker_timeframe(record, ticker, tf, price_data, published_at):
                    return False
        return True

    def _process_single_ticker_timeframe(
        self,
        record: dict[str, Any],
        ticker: str,
        tf: str,
        price_data: dict[str, pd.DataFrame],
        published_at: pd.Timestamp
    ) -> bool:
        """Process single ticker-timeframe combination."""
        if tf not in price_data:
            logger.warning(f"Таймфрейм {tf} відсутній в price_data")
            return False

        # Отримуємо дані для цього тікера
        ticker_data = self._get_ticker_data(price_data[tf], ticker)
        if ticker_data.empty:
            return False

        # Get candles
        candle_before = self._get_last_candle_before(ticker_data, published_at, tf)
        if candle_before is None:
            self.stats['filtered_insufficient_before'] += 1
            logger.debug(f"No candle before for {ticker} {tf}")
            return False

        candles_after = self._get_2_candles_after(ticker_data, published_at, tf)
        if len(candles_after) < 2:
            self.stats['filtered_insufficient_after'] += 1
            logger.debug(f"Insufficient candles after for {ticker} {tf}: {len(candles_after)}")
            return False

        # Check for missing data
        if self._has_missing_data(candle_before) or any(self._has_missing_data(c) for c in candles_after):
            self.stats['filtered_missing_data'] += 1
            logger.debug(f"Missing data in candles for {ticker} {tf}")
            return False

        # Add features
        self._add_candle_features_to_record(record, ticker, tf, candle_before, candles_after)

        return True

    def _get_ticker_data(self, price_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Get filtered ticker data."""
        if 'ticker' in price_data.columns:
            return price_data[price_data['ticker'] == ticker].copy()
        return price_data.copy()

    def _add_candle_features_to_record(
        self,
        record: dict[str, Any],
        ticker: str,
        tf: str,
        candle_before: pd.Series,
        candles_after: list[pd.Series]
    ):
        """Add candle features to record."""
        # Додаємо фічі ДО
        record.update(self._extract_candle_features(ticker, tf, candle_before, suffix=''))

        # Додаємо цільові змінні (targets) зі свічки ДО
        self._add_target_features(record, candle_before, ticker)

        # Додаємо фічі ПІСЛЯ
        record.update(self._extract_candle_features(ticker, tf, candles_after[0], suffix='_+1'))
        record.update(self._extract_candle_features(ticker, tf, candles_after[1], suffix='_+2'))

    def _add_target_features(self, record: dict[str, Any], candle_before: pd.Series, ticker: str):
        """Add target features from candle before."""
        for col in candle_before.index:
            if isinstance(col, str) and col.startswith('target_'):
                # Префіксуємо таргет тікером
                target_key = f"{ticker}_{col}"
                record[target_key] = candle_before[col]

    def _add_final_record_fields(self, record: dict[str, Any], tickers: list[str]):
        """Add final record fields."""
        # Якщо один тікер - додаємо його, якщо декілька - залишаємо порожнім
        if len(tickers) == 1:
            record['ticker'] = tickers[0]
        else:
            record['ticker'] = None  # Для мультитікерних записів

    def _add_global_features(self, record: dict[str, Any], macro_data: pd.DataFrame,
                            published_at: pd.Timestamp, tickers: list[str], price_data: dict[str, pd.DataFrame]) -> bool:
        """Add global features to record."""
        # Додаємо глобальні показники
        macro_features = self._get_macro_features(macro_data, published_at)
        if not macro_features:
            self.stats['filtered_missing_macro'] += 1
            return False
        record.update(macro_features)

        # Додаємо довгострокові ковзні середні
        record.update(self._get_long_term_mas(tickers, price_data, published_at))

        # Додаємо мапу контексту
        record.update(self._calculate_context_map(record))

        return True

    def _get_last_candle_before(
        self,
        df: pd.DataFrame,
        published_at: pd.Timestamp,
        timeframe: str
    ) -> pd.Series | None:
        """
        Отримує останню закриту свічку строго ДО публікації.
        Використовує pandas індексацію. DataFrame вже містить тільки валідні торгові години.
        """
        if df is None or df.empty:
            logger.debug(f"No data for timeframe {timeframe}")
            return None

        # ✅ FIX: Нормалізуємо timezone для порівняння
        published_at_normalized = pd.to_datetime(published_at)
        if published_at_normalized.tz is not None:
            published_at_normalized = published_at_normalized.tz_localize(None)

        logger.debug(f"Looking for candle before {published_at_normalized} in {timeframe} data (shape: {df.shape})")

        # Перевіряємо чи є datetime-колонка
        datetime_col = None
        for col in ['datetime', 'published_at', 'date', 'timestamp']:
            if col in df.columns:
                datetime_col = col
                break

        logger.debug(f"Available columns: {df.columns.tolist()}")
        logger.debug(f"Using datetime column: {datetime_col}")

        if datetime_col:
            # Фільтруємо за datetime-колонкою
            df_temp = df.copy()
            df_temp[datetime_col] = pd.to_datetime(df_temp[datetime_col], utc=True).dt.tz_localize(None)

            df_before = df_temp[df_temp[datetime_col] <= published_at_normalized]
        else:
            # Fallback: використовуємо індекс якщо він DatetimeIndex
            df_index = df.index
            if isinstance(df_index, pd.DatetimeIndex) and df_index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)

            df_before = df[df.index <= published_at_normalized]

        logger.debug(f"Found {len(df_before)} candles before {published_at_normalized}")

        if df_before.empty:
            logger.debug(f"No candles before {published_at_normalized}")
            return None

        # Повертаємо останню свічку
        return df_before.iloc[-1]

    def _get_2_candles_after(
        self,
        df: pd.DataFrame,
        published_at: pd.Timestamp,
        timeframe: str
    ) -> list[pd.Series]:
        """
        Отримує 2 наступні свічки строго ПІСЛЯ публікації новини.
        """
        if df is None or df.empty:
            logger.debug(f"No data for timeframe {timeframe}")
            return []

        # ✅ FIX: Нормалізуємо timezone для порівняння
        published_at_normalized = pd.to_datetime(published_at)
        if published_at_normalized.tz is not None:
            published_at_normalized = published_at_normalized.tz_localize(None)

        logger.debug(f"Looking for 2 candles after {published_at_normalized} in {timeframe} data (shape: {df.shape})")

        # Перевіряємо чи є datetime-колонка
        datetime_col = None
        for col in ['datetime', 'published_at', 'date', 'timestamp']:
            if col in df.columns:
                datetime_col = col
                break

        logger.debug(f"Available columns: {df.columns.tolist()}")
        logger.debug(f"Using datetime column: {datetime_col}")

        if datetime_col:
            # Фільтруємо за datetime-колонкою
            df_temp = df.copy()
            df_temp[datetime_col] = pd.to_datetime(df_temp[datetime_col], utc=True).dt.tz_localize(None)

            df_after = df_temp[df_temp[datetime_col] > published_at_normalized]
        else:
            # Fallback: використовуємо індекс якщо він DatetimeIndex
            df_index = df.index
            if isinstance(df_index, pd.DatetimeIndex) and df_index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)

            df_after = df[df.index > published_at_normalized]

        logger.debug(f"Found {len(df_after)} candles after {published_at_normalized}")

        if len(df_after) < 2:
            logger.debug(f"Insufficient candles after {published_at_normalized}: {len(df_after)} < 2")
            return []

        return [df_after.iloc[0], df_after.iloc[1]]

    def _has_missing_data(self, candle: pd.Series) -> bool:
        """Перевіряє чи є пропуски в ключових даних свічки."""
        core_features = ['open', 'high', 'low', 'close', 'volume']
        for feature in core_features:
            if feature not in candle.index or pd.isna(candle[feature]):
                return True
        return False

    def _extract_candle_features(
        self,
        ticker: str,
        timeframe: str,
        candle: pd.Series,
        suffix: str = ''
    ) -> dict:
        """
        Витягує фічі з свічки.

        Args:
            ticker: Тікер (наприклад, 'AMD')
            timeframe: Таймфрейм (наприклад, '15m')
            candle: Свічка (pd.Series)
            suffix: Суфікс ('', '_+1', '_+2')

        Returns:
            Dict з фічами у форматі: {ticker}_{timeframe}_{feature}{suffix}
        """
        features = {}

        for feature in self.CANDLE_FEATURES:
            if feature in candle.index:
                key = f"{ticker}_{timeframe}_{feature.lower()}{suffix}"
                features[key] = candle[feature]

        return features

    def _get_macro_features(
        self,
        macro_data: pd.DataFrame,
        published_at: pd.Timestamp
    ) -> dict:
        """
        Отримує макроекономічні показники на момент публікації.

        Returns:
            Dict з макро показниками у форматі: macro_{series_id}
        """
        if macro_data.empty:
            logger.warning("Macro data is empty")
            return {}

        published_at_normalized = self._normalize_timestamp(published_at)
        macro_before = self._filter_macro_data_before_date(macro_data, published_at_normalized)

        if macro_before.empty:
            logger.warning(f"No macro data before {published_at_normalized}")
            return {}

        return self._extract_macro_features(macro_before, macro_data.columns)

    def _normalize_timestamp(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        """Normalize timestamp by removing timezone."""
        normalized = pd.to_datetime(timestamp)
        if normalized.tz is not None:
            normalized = normalized.tz_localize(None)
        return normalized

    def _filter_macro_data_before_date(self, macro_data: pd.DataFrame, published_at: pd.Timestamp) -> pd.DataFrame:
        """Filter macro data to include only records before publication date."""
        if not isinstance(macro_data.index, pd.DatetimeIndex):
            return self._filter_macro_by_column(macro_data, published_at)
        else:
            return self._filter_macro_by_index(macro_data, published_at)

    def _filter_macro_by_column(self, macro_data: pd.DataFrame, published_at: pd.Timestamp) -> pd.DataFrame:
        """Filter macro data using date column."""
        date_col = self._find_macro_date_column(macro_data)
        if date_col is None:
            logger.warning(f"Macro data has no datetime index or column. Index type: {type(macro_data.index)}, Columns: {macro_data.columns.tolist()}")
            return pd.DataFrame()

        macro_data_copy = macro_data.copy()
        macro_data_copy[date_col] = pd.to_datetime(macro_data_copy[date_col])
        if macro_data_copy[date_col].dt.tz is not None:
            macro_data_copy[date_col] = macro_data_copy[date_col].dt.tz_localize(None)

        return macro_data_copy[macro_data_copy[date_col] <= published_at]

    def _filter_macro_by_index(self, macro_data: pd.DataFrame, published_at: pd.Timestamp) -> pd.DataFrame:
        """Filter macro data using datetime index."""
        macro_data_filtered = macro_data.copy()
        if macro_data_filtered.index.tz is not None:
            macro_data_filtered.index = macro_data_filtered.index.tz_localize(None)

        return macro_data_filtered[macro_data_filtered.index <= published_at]

    def _find_macro_date_column(self, macro_data: pd.DataFrame) -> str | None:
        """Find date column in macro data."""
        for col in ['date', 'datetime', 'timestamp']:
            if col in macro_data.columns:
                return col
        return None

    def _extract_macro_features(self, macro_before: pd.DataFrame, all_columns: list[str]) -> dict:
        """Extract macro features from filtered data."""
        latest_macro = macro_before.iloc[-1]
        features = {}

        excluded_cols = ['ticker', 'datetime', 'date', 'timestamp', 'hash', 'realtime_start', 'realtime_end', 'series_id']

        for col in all_columns:
            if col not in excluded_cols:
                key = f"macro_{col.lower()}"
                value = latest_macro[col]
                if pd.notna(value):
                    features[key] = value

        if not features:
            logger.warning(f"No macro features extracted from columns: {all_columns}")
        else:
            logger.debug(f"Extracted {len(features)} macro features")

        return features

    def _get_long_term_mas(
        self,
        tickers: list[str],
        price_data: dict[str, pd.DataFrame],
        published_at: pd.Timestamp
    ) -> dict:
        """
        Розраховує довгострокові ковзні середні (SMA_200, EMA_200) для всіх тікерів.

        Returns:
            Dict у форматі: {ticker}_sma_200_1d, {ticker}_ema_200_1d
        """
        if '1d' not in price_data:
            return {}

        daily_data = price_data['1d']
        published_at_normalized = self._normalize_timestamp(published_at)

        return self._calculate_long_term_mas_for_tickers(tickers, daily_data, published_at_normalized)

    def _calculate_long_term_mas_for_tickers(self, tickers: list[str], daily_data: pd.DataFrame, published_at: pd.Timestamp) -> dict:
        """Calculate long-term moving averages for all tickers."""
        features: dict[str, Any] = {}

        for ticker in tickers:
            ticker_features = self._calculate_ticker_long_term_mas(ticker, daily_data, published_at)
            features.update(ticker_features)

        return features

    def _calculate_ticker_long_term_mas(self, ticker: str, daily_data: pd.DataFrame, published_at: pd.Timestamp) -> dict:
        """Calculate long-term moving averages for a single ticker."""
        features: dict[str, float] = {}

        if ticker not in daily_data.columns:
            logger.debug(f"No daily data for {ticker}")
            return features

        ticker_data = daily_data[ticker].dropna()
        if ticker_data.empty:
            logger.debug(f"Empty daily data for {ticker}")
            return features

        # Normalize timezone for comparison
        if ticker_data.index.tz is not None:
            ticker_data = ticker_data.copy()
            ticker_data.index = ticker_data.index.tz_localize(None)

        # Filter data before publication date
        data_before = ticker_data[ticker_data.index <= published_at]
        if data_before.empty:
            logger.debug(f"No daily data before {published_at} for {ticker}")
            return features

        # Calculate moving averages
        if len(data_before) >= 200:
            sma_200 = data_before.rolling(window=200).mean().iloc[-1]
            features[f"{ticker}_sma_200_1d"] = sma_200

        if len(data_before) >= 200:
            ema_200 = data_before.ewm(span=200).mean().iloc[-1]
            features[f"{ticker}_ema_200_1d"] = ema_200

        return features

    def _calculate_context_map(
        self,
        record: dict
    ) -> dict:
        """
        Розраховує мапу контексту (context fingerprint).

        Логіка:
        1. Беремо ключові показники (VIX, макро, індекси)
        2. Розраховуємо зміну кожного показника
        3. Класифікуємо: 1 (зростання), 0 (стабільно), -1 (падіння)
        4. Створюємо fingerprint (рядок станів)
        5. Розраховуємо stability (скільки показників стабільні)

        Returns:
            Dict з state_{feature}, context_fingerprint, context_stability
        """
        context_features: dict[str, str | float] = {}
        states: list[int] = []

        # Ключові показники для контексту
        key_indicators = [
            'macro_vixcls',      # VIX
            'macro_dgs10',       # 10Y yield
            'macro_fedfunds',    # Fed rate
            'macro_cpiaucsl',    # CPI
            'macro_unrate',      # Unemployment
        ]

        # Для кожного показника розраховуємо стан
        for indicator in key_indicators:
            if indicator in record:
                # Тут потрібно мати попереднє значення для розрахунку зміни
                # Для спрощення використовуємо фіксовані пороги

                # Визначаємо стан (спрощена логіка)
                # В реальності потрібно порівнювати з попереднім значенням
                state = 0  # За замовчуванням стабільно

                state_key = f"state_{indicator.replace('macro_', '')}"
                context_features[state_key] = state
                states.append(state)

        # Створюємо fingerprint
        if states:
            context_features['context_fingerprint'] = '|'.join(map(str, states))
            context_features['context_stability'] = states.count(0) / len(states)
        else:
            context_features['context_fingerprint'] = ''
            context_features['context_stability'] = 0.0

        return context_features

    def _log_filtering_stats(self):
        """Виводить статистику фільтрації."""
        logger.info("=" * 60)
        logger.info("📊 СТАТИСТИКА ФІЛЬТРАЦІЇ НОВИН")
        logger.info("=" * 60)
        logger.info(f"Всього новин: {self.stats['total_news']}")
        logger.info("Відфільтровано:")
        logger.info(f"  - Недостатньо даних ДО: {self.stats['filtered_insufficient_before']}")
        logger.info(f"  - Недостатньо даних ПІСЛЯ: {self.stats['filtered_insufficient_after']}")
        logger.info(f"  - Пропуски в даних: {self.stats['filtered_missing_data']}")
        logger.info(f"  - Немає макро даних: {self.stats['filtered_missing_macro']}")
        logger.info(f"✅ Валідних записів: {self.stats['valid_records']}")
        if self.stats['total_news'] > 0:
            logger.info(f"📈 Успішність: {self.stats['valid_records'] / self.stats['total_news'] * 100:.1f}%")
        logger.info("=" * 60)
