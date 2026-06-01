"""
News Event Data Filter
Handles news items validation and filtering.
"""
import logging
import pandas as pd
from typing import List, Optional, Any
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class NewsEventDataFilter:
    def __init__(self, is_test_mode: bool = False, test_ticker: Optional[str] = None):
        self.is_test_mode = is_test_mode
        self.test_ticker = test_ticker
        self.stats = {
            'total_news': 0,
            'filtered_insufficient_before': 0,
            'filtered_insufficient_after': 0,
            'filtered_missing_data': 0,
            'filtered_missing_macro': 0,
            'valid_records': 0
        }

    def filter_tickers(self, tickers: List[str]) -> List[str]:
        """Filter tickers based on test mode."""
        if self.is_test_mode and self.test_ticker:
            logger.info(f'🧪 Тестовий режим: використовуємо тільки {self.test_ticker}')
            return [self.test_ticker]
        return tickers

    def find_publication_column(self, news_df: pd.DataFrame) -> Optional[str]:
        """Find publication date column in news DataFrame."""
        for col in ['publishedAt', 'published_at', 'published_date', 'date', 'datetime']:
            if col in news_df.columns:
                return col
        logger.error(f'Cannot find publication column. Available: {news_df.columns.tolist()}')
        return None

    def has_missing_data(self, candle: pd.Series) -> bool:
        """Checks if key candle features are missing."""
        core_features = ['open', 'high', 'low', 'close', 'volume']
        for feature in core_features:
            if feature not in candle.index or pd.isna(candle[feature]):
                return True
        return False

    def log_stats(self):
        """Logs filtering statistics."""
        logger.info('=' * 60)
        logger.info('📊 СТАТИСТИКА ФІЛЬТРАЦІЇ НОВИН')
        logger.info('=' * 60)
        logger.info(f"Всього новин: {self.stats['total_news']}")
        logger.info('Відфільтровано:')
        logger.info(f"  - Недостатньо даних ДО: {self.stats['filtered_insufficient_before']}")
        logger.info(f"  - Недостатньо даних ПІСЛЯ: {self.stats['filtered_insufficient_after']}")
        logger.info(f"  - Пропуски в даних: {self.stats['filtered_missing_data']}")
        logger.info(f"  - Немає макро даних: {self.stats['filtered_missing_macro']}")
        logger.info(f"✅ Валідних записів: {self.stats['valid_records']}")
        if self.stats['total_news'] > 0:
            rate = self.stats['valid_records'] / self.stats['total_news'] * 100
            logger.info(f"📈 Успішність: {rate:.1f}%")
        logger.info('=' * 60)
