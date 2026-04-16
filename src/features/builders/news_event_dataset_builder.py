"""
News Event Dataset Builder

Будує датасет на основі новин з прив'язкою до дати публікації.
Для кожної новини додає:
- 1 свічку ДО публікації (для кожного таймфрейму: 15m, 60m, 1d)
- 2 свічки ПІСЛЯ публікації (для кожного таймфрейму: 15m, 60m, 1d)
- Глобальні показники (макро, ковзні середні, сентимент)
- Мапу контексту
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

from src.utils.trading_calendar import TradingCalendar
from src.core.logging.logger import ProjectLogger

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
        runtime_params: Optional[Dict[str, Any]] = None
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
        
        logger.info(f"NewsEventDatasetBuilder initialized:")
        logger.info(f"  Test mode: {self.is_test_mode}")
        logger.info(f"  Test ticker: {self.test_ticker}")
        logger.info(f"  Test target: {self.test_target}")
        logger.info(f"  Timeframes: {self.timeframes}")
    
    def build_dataset(
        self,
        news_df: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],  # {timeframe: df}
        macro_data: pd.DataFrame,
        tickers: List[str]
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
        
        # Динамічно визначаємо доступні таймфрейми
        self.timeframes = list(price_data.keys())
        logger.info(f"Dynamically set timeframes to: {self.timeframes}")
        
        # Фільтруємо тікери (якщо тестовий режим)
        if self.is_test_mode and self.test_ticker:
            tickers = [self.test_ticker]
            logger.info(f"🧪 Тестовий режим: використовуємо тільки {self.test_ticker}")
        
        logger.info(f"Обробка {len(news_df)} новин для {len(tickers)} тікерів × {len(self.timeframes)} таймфреймів")
        
        records = []
        
        # Find the accurate publication date column
        pub_col = None
        for col in ['publishedAt', 'published_at', 'published_date', 'date', 'datetime']:
            if col in news_df.columns:
                pub_col = col
                break
        
        if pub_col is None:
            logger.error(f"Cannot find publication date column in news_df. Available columns: {news_df.columns.tolist()}")
            return pd.DataFrame()

        # Для кожної новини
        for idx, news in news_df.iterrows():
            try:
                published_at = pd.to_datetime(news[pub_col])
            except Exception as e:
                logger.warning(f"Could not parse date {news.get(pub_col)} for news {idx}: {e}")
                continue
            
            # Перевіряємо наявність даних для ВСІХ тікерів × ВСІХ таймфреймів
            record = self._build_record_for_news(
                news, published_at, tickers, price_data, macro_data
            )
            
            if record:
                records.append(record)
                self.stats['valid_records'] += 1
            
            # Логування прогресу
            if (idx + 1) % 100 == 0:
                logger.info(f"Оброблено {idx + 1}/{len(news_df)} новин, валідних: {self.stats['valid_records']}")
        
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
        tickers: List[str],
        price_data: Dict[str, pd.DataFrame],
        macro_data: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Створює один запис для новини.
        
        Повертає None якщо дані неповні для хоча б одного тікера × таймфрейму.
        """
        # ✅ FIX: Використовуємо правильні назви колонок
        # ✅ FIX: Видаляємо timezone з published_at одразу
        published_at_normalized = pd.to_datetime(published_at)
        if published_at_normalized.tz is not None:
            published_at_normalized = published_at_normalized.tz_localize(None)
        
        record = {
            'news_id': news.get('hash') if pd.notna(news.get('hash')) else '',
            'published_at': published_at_normalized,
            'news_title': news.get('title') if pd.notna(news.get('title')) else '',
            'news_sentiment': news.get('sentiment', 0.0),  # sentiment, не sentiment_score
        }
        
        # Для кожного тікера × кожен таймфрейм (15m, 60m, 1d)
        for ticker in tickers:
            for tf in self.timeframes:
                if tf not in price_data:
                    logger.warning(f"Таймфрейм {tf} відсутній в price_data")
                    return None
                
                # Отримуємо дані для цього тікера
                ticker_data = price_data[tf]
                if 'ticker' in ticker_data.columns:
                    ticker_data = ticker_data[ticker_data['ticker'] == ticker].copy()
                
                if ticker_data.empty:
                    return None
                
                # 1 свічка ДО публікації
                candle_before = self._get_last_candle_before(ticker_data, published_at, tf)
                if candle_before is None:
                    self.stats['filtered_insufficient_before'] += 1
                    return None
                
                # 2 свічки ПІСЛЯ публікації
                candles_after = self._get_2_candles_after(ticker_data, published_at, tf)
                if len(candles_after) < 2:
                    self.stats['filtered_insufficient_after'] += 1
                    return None
                
                # Перевірка на пропуски
                if self._has_missing_data(candle_before) or any(self._has_missing_data(c) for c in candles_after):
                    self.stats['filtered_missing_data'] += 1
                    return None
                
                # Додаємо фічі ДО
                record.update(self._extract_candle_features(ticker, tf, candle_before, suffix=''))
                
                # Додаємо цільові змінні (targets) зі свічки ДО (як правило, розраховані на 1d або основному таймфреймі)
                for col in candle_before.index:
                    if isinstance(col, str) and col.startswith('target_'):
                        # Якщо у нас декілька тікерів, префіксуємо таргет
                        target_key = col if len(tickers) == 1 else f"{ticker}_{col}"
                        record[target_key] = candle_before[col]
                
                # Додаємо фічі ПІСЛЯ
                record.update(self._extract_candle_features(ticker, tf, candles_after[0], suffix='_+1'))
                record.update(self._extract_candle_features(ticker, tf, candles_after[1], suffix='_+2'))
        
        # ✅ FIX: Додати ticker та datetime ПІСЛЯ циклу (не всередині)
        # Якщо один тікер - додаємо його, якщо декілька - залишаємо порожнім
        if len(tickers) == 1:
            record['ticker'] = tickers[0]
        else:
            record['ticker'] = None  # Для мультитікерних записів
        
        # Додаємо глобальні показники
        macro_features = self._get_macro_features(macro_data, published_at)
        if not macro_features:
            self.stats['filtered_missing_macro'] += 1
            return None
        record.update(macro_features)
        
        # Додаємо довгострокові ковзні середні
        record.update(self._get_long_term_mas(tickers, price_data, published_at))
        
        # Додаємо мапу контексту
        record.update(self._calculate_context_map(record, published_at))
        
        # ✅ FIX: Додати datetime як published_at (дата новини), видалити timezone
        record['datetime'] = pd.to_datetime(published_at).tz_localize(None)
        
        return record
    
    def _get_last_candle_before(
        self,
        df: pd.DataFrame,
        published_at: pd.Timestamp,
        timeframe: str
    ) -> Optional[pd.Series]:
        """
        Отримує останню закриту свічку строго ДО публікації.
        Використовує pandas індексацію. DataFrame вже містить тільки валідні торгові години.
        """
        # ✅ FIX: Нормалізуємо timezone для порівняння
        published_at_normalized = pd.to_datetime(published_at)
        if published_at_normalized.tz is not None:
            published_at_normalized = published_at_normalized.tz_localize(None)
        
        # Нормалізуємо індекс DataFrame якщо потрібно
        df_index = df.index
        if isinstance(df_index, pd.DatetimeIndex) and df_index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        
        # Фільтруємо дані строго до початку новини (або <=)
        df_before = df[df.index <= published_at_normalized]
        
        if df_before.empty:
            return None
        
        # Повертаємо останню свічку
        return df_before.iloc[-1]
    
    def _get_2_candles_after(
        self,
        df: pd.DataFrame,
        published_at: pd.Timestamp,
        timeframe: str
    ) -> List[pd.Series]:
        """
        Отримує 2 наступні свічки строго ПІСЛЯ публікації новини.
        """
        # ✅ FIX: Нормалізуємо timezone для порівняння
        published_at_normalized = pd.to_datetime(published_at)
        if published_at_normalized.tz is not None:
            published_at_normalized = published_at_normalized.tz_localize(None)
        
        # Нормалізуємо індекс DataFrame якщо потрібно
        df_index = df.index
        if isinstance(df_index, pd.DatetimeIndex) and df_index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        
        # Фільтруємо дані після current_time. DataFrame вже містить тільки валідні торгові години.
        df_after = df[df.index > published_at_normalized]
        
        if len(df_after) < 2:
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
    ) -> Dict:
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
    ) -> Dict:
        """
        Отримує макроекономічні показники на момент публікації.
        
        Returns:
            Dict з макро показниками у форматі: macro_{series_id}
        """
        if macro_data.empty:
            logger.warning("Macro data is empty")
            return {}
        
        # ✅ FIX: Нормалізуємо timezone для порівняння
        published_at_normalized = pd.to_datetime(published_at)
        if published_at_normalized.tz is not None:
            published_at_normalized = published_at_normalized.tz_localize(None)
        
        # ✅ FIX: Перевіряємо чи індекс є DatetimeIndex
        if not isinstance(macro_data.index, pd.DatetimeIndex):
            # Якщо індекс не datetime, шукаємо колонку з датою
            date_col = None
            # ✅ FIX: Змінено порядок - 'date' перший, бо це найчастіша назва в FRED
            for col in ['date', 'datetime', 'timestamp']:
                if col in macro_data.columns:
                    date_col = col
                    break
            
            if date_col is None:
                logger.warning(f"Macro data has no datetime index or column. Index type: {type(macro_data.index)}, Columns: {macro_data.columns.tolist()}")
                return {}
            
            # Фільтруємо по колонці
            macro_data_copy = macro_data.copy()
            # ✅ FIX: Нормалізуємо timezone в колонці date
            macro_data_copy[date_col] = pd.to_datetime(macro_data_copy[date_col])
            if macro_data_copy[date_col].dt.tz is not None:
                macro_data_copy[date_col] = macro_data_copy[date_col].dt.tz_localize(None)
            macro_before = macro_data_copy[macro_data_copy[date_col] <= published_at_normalized]
        else:
            # Нормалізуємо індекс якщо потрібно
            if macro_data.index.tz is not None:
                macro_data = macro_data.copy()
                macro_data.index = macro_data.index.tz_localize(None)
            # Знаходимо найближчі дані ДО публікації
            macro_before = macro_data[macro_data.index <= published_at_normalized]
        
        if macro_before.empty:
            logger.warning(f"No macro data before {published_at_normalized}")
            return {}
        
        # Беремо останні значення
        latest_macro = macro_before.iloc[-1]
        
        features = {}
        # ✅ FIX: Обробляємо wide format (після pivot)
        for col in macro_data.columns:
            if col not in ['ticker', 'datetime', 'date', 'timestamp', 'hash', 'realtime_start', 'realtime_end', 'series_id']:
                # Додаємо всі колонки як macro_{column_name}
                key = f"macro_{col.lower()}"
                value = latest_macro[col]
                # Пропускаємо NaN значення
                if pd.notna(value):
                    features[key] = value
        
        if not features:
            logger.warning(f"No macro features extracted from columns: {macro_data.columns.tolist()}")
        else:
            logger.debug(f"Extracted {len(features)} macro features")
        
        return features
    
    def _get_long_term_mas(
        self,
        tickers: List[str],
        price_data: Dict[str, pd.DataFrame],
        published_at: pd.Timestamp
    ) -> Dict:
        """
        Розраховує довгострокові ковзні середні (SMA_200, EMA_200) для всіх тікерів.
        
        Returns:
            Dict у форматі: {ticker}_sma_200_1d, {ticker}_ema_200_1d
        """
        features = {}
        
        # Використовуємо денний таймфрейм
        if '1d' not in price_data:
            return features
        
        daily_data = price_data['1d']
        
        # ✅ FIX: Нормалізуємо timezone для порівняння
        published_at_normalized = pd.to_datetime(published_at)
        if published_at_normalized.tz is not None:
            published_at_normalized = published_at_normalized.tz_localize(None)
        
        for ticker in tickers:
            # Фільтруємо по тікеру
            if 'ticker' in daily_data.columns:
                ticker_data = daily_data[daily_data['ticker'] == ticker].copy()
            else:
                ticker_data = daily_data.copy()
            
            if ticker_data.empty:
                continue
            
            # Нормалізуємо індекс якщо потрібно
            if isinstance(ticker_data.index, pd.DatetimeIndex) and ticker_data.index.tz is not None:
                ticker_data = ticker_data.copy()
                ticker_data.index = ticker_data.index.tz_localize(None)
            
            # Фільтруємо ДО публікації
            ticker_before = ticker_data[ticker_data.index <= published_at_normalized]
            
            if len(ticker_before) < 200:
                continue
            
            # Розраховуємо SMA_200 та EMA_200
            close_prices = ticker_before['close']
            
            sma_200 = close_prices.rolling(window=200).mean().iloc[-1]
            ema_200 = close_prices.ewm(span=200, adjust=False).mean().iloc[-1]
            
            features[f"{ticker}_sma_200_1d"] = sma_200
            features[f"{ticker}_ema_200_1d"] = ema_200
        
        return features
    
    def _calculate_context_map(
        self,
        record: Dict,
        published_at: pd.Timestamp
    ) -> Dict:
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
        context_features = {}
        states = []
        
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
                value = record[indicator]
                
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
        logger.info(f"Відфільтровано:")
        logger.info(f"  - Недостатньо даних ДО: {self.stats['filtered_insufficient_before']}")
        logger.info(f"  - Недостатньо даних ПІСЛЯ: {self.stats['filtered_insufficient_after']}")
        logger.info(f"  - Пропуски в даних: {self.stats['filtered_missing_data']}")
        logger.info(f"  - Немає макро даних: {self.stats['filtered_missing_macro']}")
        logger.info(f"✅ Валідних записів: {self.stats['valid_records']}")
        if self.stats['total_news'] > 0:
            logger.info(f"📈 Успішність: {self.stats['valid_records'] / self.stats['total_news'] * 100:.1f}%")
        logger.info("=" * 60)
