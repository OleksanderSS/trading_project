"""
News Context Dataset Builder
Створює датасет з новинами + контекст до/після для релевантних тікерів та таймфреймів
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

from src.core.logging.logger import ProjectLogger
from src.utils.trading_calendar import TradingCalendar
from src.config.unified_config_manager import UnifiedConfigManager
from src.features.news_impact_classifier import NewsImpactClassifier

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
        logger.info(f"NewsImpactClassifier integrated successfully")
    
    def _get_tickers(self) -> List[str]:
        """Отримати список тікерів з конфігурації"""
        assets_config = self.config_manager.get_config('assets', default={})
        
        # Спробувати отримати з active_preset
        active_preset = assets_config.get('active_preset')
        if active_preset:
            presets = assets_config.get('presets', {})
            preset_config = presets.get(active_preset, {})
            if 'tickers' in preset_config:
                return preset_config['tickers']
        
        # Спробувати отримати з sectors
        all_tickers = set()
        sectors = assets_config.get('sectors', {})
        for sector_name, sector_data in sectors.items():
            if isinstance(sector_data, dict) and 'assets' in sector_data:
                all_tickers.update(sector_data['assets'])
        
        if all_tickers:
            return sorted(list(all_tickers))
        
        # Fallback - default tickers
        logger.warning("No tickers found in config, using defaults")
        return ['AAPL', 'NVDA', 'TSLA', 'AMD', 'GOOGL', 'MSFT', 'AMZN', 'META']
    
    def _filter_news_with_sufficient_candles(
        self,
        news_df: pd.DataFrame,
        prices_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        ✅ ІНТЕЛЕКТУАЛЬНИЙ ФІЛЬТР: Новина валідна якщо має дані для релевантних тікерів/таймфреймів
        
        Логіка:
        1. Класифікувати вплив новини
        2. Отримати релевантні комбінації тікер-таймфрейм
        3. Перевірити чи є достатньо свічок для релевантних комбінацій
        4. Новина валідна якщо ХОЧА Б ОДНА релевантна комбінація валідна
        """
        filtered_rows = []
        removed_reasons = {
            'no_relevant_combinations': 0,
            'insufficient_candles': 0,
            'datetime_error': 0
        }
        
        logger.info(f"� Starting intelligent news filtering for {len(news_df)} articles")
        
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
                    logger.debug(f"❌ News {idx}: No relevant combinations found")
                    continue
                
                # 3. Перевірити чи є достатньо свічок для релевантних комбінацій
                has_valid_combination = False
                valid_combinations = []
                
                for ticker, timeframe in relevant_combinations:
                    if timeframe not in prices_dict:
                        continue
                    
                    prices_df = prices_dict[timeframe]
                    if prices_df.empty:
                        continue
                    
                    # Отримати дані для тікера
                    ticker_prices = prices_df[prices_df['ticker'] == ticker].copy()
                    
                    if ticker_prices.empty:
                        continue
                    
                    # Перевірити datetime колонку
                    if 'datetime' not in ticker_prices.columns:
                        removed_reasons['datetime_error'] += 1
                        continue
                    
                    # Нормалізувати datetime
                    ticker_prices['datetime'] = pd.to_datetime(ticker_prices['datetime'])
                    news_time = pd.to_datetime(news_row['published_date'], utc=True)
                    if news_time.tz is not None:
                        news_time = news_time.tz_localize(None)
                    
                    # Перевірити свічки
                    before_candles = ticker_prices[ticker_prices['datetime'] < news_time]
                    after_candles = ticker_prices[ticker_prices['datetime'] > news_time]
                    
                    if len(before_candles) >= 2 and len(after_candles) >= 2:
                        valid_combinations.append((ticker, timeframe))
                        has_valid_combination = True
                
                # 4. Логування результатів
                if has_valid_combination:
                    filtered_rows.append(idx)
                    logger.debug(f"✅ News {idx}: {len(valid_combinations)} valid combinations -> {valid_combinations[:3]}...")
                else:
                    removed_reasons['insufficient_candles'] += 1
                    logger.debug(f"❌ News {idx}: No valid combinations out of {len(relevant_combinations)}")
                
            except Exception as e:
                logger.warning(f"❌ Error processing news {idx}: {e}")
                removed_reasons['datetime_error'] += 1
                continue
        
        # Логування результатів фільтрації
        removed_count = len(news_df) - len(filtered_rows)
        logger.info(f"✅ Intelligent news filtering: {len(news_df)} → {len(filtered_rows)} (removed {removed_count})")
        
        if removed_count > 0:
            logger.info(f"   Removal reasons:")
            logger.info(f"      No relevant combinations: {removed_reasons['no_relevant_combinations']}")
            logger.info(f"      Insufficient candles: {removed_reasons['insufficient_candles']}")
            logger.info(f"      Datetime errors: {removed_reasons['datetime_error']}")
        
        # Повернути відфільтровані новини
        return news_df.loc[filtered_rows].reset_index(drop=True)
    
    def build_dataset(
        self,
        news_df: pd.DataFrame,
        prices_dict: Dict[str, pd.DataFrame],
        macro_df: Optional[pd.DataFrame] = None,
        market_sentiment_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Побудувати повний датасет з новинами та контекстом
        
        Args:
            news_df: DataFrame з новинами (timestamp, title, sentiment, ticker, news_type)
            prices_dict: Dict[timeframe, DataFrame] з цінами
            macro_df: DataFrame з макроекономічними даними
            market_sentiment_df: DataFrame з ринковим сентиментом (VIX, Fear/Greed)
        
        Returns:
            DataFrame з повним контекстом для кожної новини
        """
        logger.info(f"Building news context dataset for {len(news_df)} news articles")
        
        # ✅ КРОК 1: Фільтрація новин - залишаємо тільки ті, що мають 2+ свічки після
        news_df_filtered = self._filter_news_with_sufficient_candles(news_df, prices_dict)
        logger.info(f"✅ Filtered news: {len(news_df)} → {len(news_df_filtered)} (removed {len(news_df) - len(news_df_filtered)} without 2+ candles after)")
        
        if news_df_filtered.empty:
            logger.warning("No news articles with sufficient candles after publication!")
            return pd.DataFrame()
        
        dataset_rows = []
        
        for idx, news_row in news_df_filtered.iterrows():
            try:
                # Побудувати рядок для цієї новини
                row = self._build_news_row(
                    news_row,
                    prices_dict,
                    macro_df,
                    market_sentiment_df
                )
                
                if row is not None:
                    dataset_rows.append(row)
                    
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(news_df_filtered)} news articles")
                    
            except Exception as e:
                logger.warning(f"Failed to process news {idx}: {e}")
                continue
        
        if not dataset_rows:
            logger.error("No valid rows generated!")
            return pd.DataFrame()
        
        dataset_df = pd.DataFrame(dataset_rows)
        logger.info(f"✅ Dataset built: {len(dataset_df)} rows, {len(dataset_df.columns)} columns")
        
        # Додати metadata
        dataset_df = self._add_metadata(dataset_df)
        
        return dataset_df
    
    def _build_news_row(
        self,
        news_row: pd.Series,
        prices_dict: Dict[str, pd.DataFrame],
        macro_df: Optional[pd.DataFrame],
        market_sentiment_df: Optional[pd.DataFrame]
    ) -> Optional[Dict[str, Any]]:
        """
        ✅ ПРАВИЛЬНА СТРУКТУРА ДЛЯ ML:
        [НОВИНА + МАКРО] → [КОНТЕКСТ ДО: 2 свічки + ВСІ фічі] → [РЕАКЦІЯ ПІСЛЯ: 2 свічки + ВСІ фічі]
        
        Один рядок = одна новина з повним контекстом
        
        Структура:
        1. Новина (6 колонок): id, timestamp, title, sentiment, type, source
        2. Макро контекст (~30 колонок): Fed, yields, VIX, час дня, тощо
        3. Контекст ДО (18 тікерів × 3 таймфрейми × 2 свічки × ~200 фічей):
           - Для кожної свічки: datetime + ВСІ фічі (OHLCV + технічні + сентимент + макро + ...)
        4. Реакція ПІСЛЯ (18 тікерів × 3 таймфрейми × 2 свічки × ~200 фічей):
           - Для кожної свічки: datetime + ВСІ фічі (OHLCV + технічні + сентимент + макро + ...)
        
        Всього: ~43,236 колонок на рядок
        """
        
        row = {}
        news_time = news_row['published_date']
        
        # ========== БЛОК 1: НОВИНА ==========
        row['news_id'] = f"{news_row.get('source', 'unknown')}_{news_time.strftime('%Y%m%d_%H%M%S')}"
        row['news_timestamp'] = news_time
        row['news_title'] = news_row.get('title', '')
        row['news_sentiment'] = news_row.get('sentiment', 0.0)
        row['news_type'] = news_row.get('news_type', 'general')
        row['news_source'] = news_row.get('source', 'unknown')
        
        # ========== БЛОК 2: МАКРО КОНТЕКСТ НА МОМЕНТ НОВИНИ ==========
        row.update(self._get_macro_context(news_time, macro_df))
        row.update(self._get_market_sentiment_context(news_time, market_sentiment_df))
        row.update(self._get_temporal_features(news_time))
        
        # ========== БЛОК 3: КОНТЕКСТ ДО НОВИНИ - ВСІ ТІКЕРИ × ВСІ ТАЙМФРЕЙМИ ==========
        # Для кожного тікера/таймфрейму: 2 свічки ДО + ВСІ їх фічі
        for ticker in self.tickers:
            for timeframe in self.timeframes:
                if timeframe not in prices_dict:
                    continue
                
                prices_df = prices_dict[timeframe]
                ticker_prices = prices_df[prices_df['ticker'] == ticker].copy()
                
                if ticker_prices.empty:
                    continue
                
                # Переконатися що є datetime колонка
                if 'datetime' not in ticker_prices.columns:
                    if isinstance(ticker_prices.index, pd.DatetimeIndex):
                        ticker_prices = ticker_prices.reset_index()
                        if 'index' in ticker_prices.columns:
                            ticker_prices = ticker_prices.rename(columns={'index': 'datetime'})
                    else:
                        continue
                
                # ✅ 2 СВІЧКИ ДО НОВИНИ + ВСІ ФІЧІ
                # Fix datetime comparison
                news_time_pd = pd.to_datetime(news_time)
                before_candles = ticker_prices[ticker_prices['datetime'] < news_time_pd].tail(2)
                for i, (idx, candle) in enumerate(before_candles.iterrows(), start=1):
                    prefix = f"{ticker}_{timeframe}_before_{i}"
                    
                    # Додати datetime
                    row[f"{prefix}_datetime"] = candle.get('datetime')
                    
                    # Додати ВСІ фічі (OHLCV + технічні + сентимент + макро + ...)
                    for col in ticker_prices.columns:
                        if col not in ['datetime', 'ticker', 'interval']:
                            row[f"{prefix}_{col}"] = candle.get(col, np.nan)
        
        # ========== БЛОК 4: РЕАКЦІЯ ПІСЛЯ НОВИНИ - ВСІ ТІКЕРИ × ВСІ ТАЙМФРЕЙМИ ==========
        # Для кожного тікера/таймфрейму: 2 свічки ПІСЛЯ + ВСІ їх фічі
        for ticker in self.tickers:
            for timeframe in self.timeframes:
                if timeframe not in prices_dict:
                    continue
                
                prices_df = prices_dict[timeframe]
                ticker_prices = prices_df[prices_df['ticker'] == ticker].copy()
                
                if ticker_prices.empty:
                    continue
                
                # Переконатися що є datetime колонка
                if 'datetime' not in ticker_prices.columns:
                    if isinstance(ticker_prices.index, pd.DatetimeIndex):
                        ticker_prices = ticker_prices.reset_index()
                        if 'index' in ticker_prices.columns:
                            ticker_prices = ticker_prices.rename(columns={'index': 'datetime'})
                    else:
                        continue
                
                # ✅ 2 СВІЧКИ ПІСЛЯ НОВИНИ + ВСІ ФІЧІ
                # Fix datetime comparison
                news_time_pd = pd.to_datetime(news_time)
                after_candles = ticker_prices[ticker_prices['datetime'] > news_time_pd].head(2)
                for i, (idx, candle) in enumerate(after_candles.iterrows(), start=1):
                    prefix = f"{ticker}_{timeframe}_after_{i}"
                    
                    # Додати datetime
                    row[f"{prefix}_datetime"] = candle.get('datetime')
                    
                    # Додати ВСІ фічі (OHLCV + технічні + сентимент + макро + ...)
                    for col in ticker_prices.columns:
                        if col not in ['datetime', 'ticker', 'interval']:
                            row[f"{prefix}_{col}"] = candle.get(col, np.nan)
        
        return row if len(row) > 10 else None  # Перевірка мінімальної кількості даних
    
    def _get_tickers_for_news(self, news_row: pd.Series) -> List[str]:
        """
        Визначити, для яких тікерів генерувати контекст
        
        ЗАВЖДИ генеруємо для ВСІХ тікерів, незалежно від news_type.
        Це дозволяє SmartFeatureSelector вибирати будь-які релевантні фічі.
        
        Наприклад:
        - Новина "Tesla recalls" (news_type="TSLA") може вплинути на AMD, NVDA
        - Новина "Fed rate hike" (news_type="general") впливає на всіх
        """
        return self.tickers  # Завжди всі тікери
    
    
    def _get_macro_context(self, timestamp: datetime, macro_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Отримати макроекономічний контекст на момент новини"""
        if macro_df is None or macro_df.empty:
            return {}
        
        # Переконатися що є datetime колонка
        date_col = 'datetime' if 'datetime' in macro_df.columns else 'date'
        if date_col not in macro_df.columns:
            return {}
        
        # Знайти найближчі макроданні (ffill логіка)
        macro_df = macro_df.sort_values(date_col)
        relevant_macro = macro_df[macro_df[date_col] <= timestamp].tail(1)
        
        if relevant_macro.empty:
            return {}
        
        result = {}
        macro_row = relevant_macro.iloc[0]
        
        # Додати основні макропоказники
        macro_features = ['fed_funds_rate', 'treasury_10y', 'treasury_2y', 'vix', 'cpi', 'unemployment_rate', 'gdp_growth']
        for feature in macro_features:
            if feature in macro_row:
                result[f"macro_{feature}"] = macro_row[feature]
        
        return result
    
    def _get_market_sentiment_context(
        self,
        timestamp: datetime,
        market_sentiment_df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Отримати ринковий сентимент на момент новини"""
        if market_sentiment_df is None or market_sentiment_df.empty:
            return {}
        
        # Переконатися що є datetime колонка
        date_col = 'datetime' if 'datetime' in market_sentiment_df.columns else 'date'
        if date_col not in market_sentiment_df.columns:
            return {}
        
        # Знайти найближчі дані
        market_sentiment_df = market_sentiment_df.sort_values(date_col)
        relevant_sentiment = market_sentiment_df[market_sentiment_df[date_col] <= timestamp].tail(1)
        
        if relevant_sentiment.empty:
            return {}
        
        result = {}
        sentiment_row = relevant_sentiment.iloc[0]
        
        # Додати показники сентименту
        sentiment_features = ['vix', 'fear_greed_index', 'put_call_ratio']
        for feature in sentiment_features:
            if feature in sentiment_row:
                result[f"sentiment_{feature}"] = sentiment_row[feature]
        
        return result
    
    def _get_temporal_features(self, timestamp: datetime) -> Dict[str, Any]:
        """Отримати часові фічі"""
        return {
            'temporal_day_of_week': timestamp.weekday(),
            'temporal_hour_of_day': timestamp.hour,
            'temporal_is_trading_hours': self._is_trading_hours(timestamp),
            'temporal_is_first_hour': timestamp.hour == 9 and timestamp.minute < 60,
            'temporal_is_last_hour': timestamp.hour == 15 and timestamp.minute >= 0,
        }
    
    def _is_trading_hours(self, timestamp: datetime) -> bool:
        """Перевірити, чи це торгові години"""
        if not self.calendar.is_trading_day(timestamp):
            return False
        
        hour = timestamp.hour
        minute = timestamp.minute
        
        # 9:30 - 16:00 ET
        if hour < 9 or hour >= 16:
            return False
        if hour == 9 and minute < 30:
            return False
        
        return True
    
    def _add_metadata(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        """
        Додати metadata колонку з інформацією про датасет
        """
        logger.info("Adding metadata to dataset...")
        
        # Додати загальну metadata
        dataset_df['dataset_version'] = '1.0'
        dataset_df['generated_at'] = datetime.now().isoformat()
        dataset_df['n_tickers'] = len(self.tickers)
        dataset_df['n_timeframes'] = len(self.timeframes)
        
        return dataset_df
    
    def save_dataset(self, dataset_df: pd.DataFrame, output_path: Path):
        """Зберегти датасет"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Зберегти як parquet (компресія)
        dataset_df.to_parquet(output_path, compression='snappy', index=False)
        logger.info(f"✅ Dataset saved to {output_path}")
        
        # Зберегти metadata окремо
        metadata = {
            'rows': len(dataset_df),
            'columns': len(dataset_df.columns),
            'tickers': self.tickers,
            'timeframes': self.timeframes,
            'n_candles_before': self.n_candles_before,
            'n_candles_after': self.n_candles_after,
            'generated_at': datetime.now().isoformat(),
        }
        
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Metadata saved to {metadata_path}")
