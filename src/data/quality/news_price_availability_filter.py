"""
Quick News Pre-Filter (Stage 2)
Fast, lightweight filter to remove obvious junk news BEFORE heavy processing.

This is a QUICK check - detailed filtering happens in Stage 3 (NewsContextDatasetBuilder).
"""

import pandas as pd
from typing import Dict
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("QuickNewsPreFilter")


def quick_filter_news_by_data_availability(
    news_df: pd.DataFrame,
    prices_dict: Dict[str, pd.DataFrame],
    news_date_col: str = 'published_date'
) -> pd.DataFrame:
    """
    ШВИДКА попередня фільтрація новин (Stage 2).
    
    Мета: Відсіяти новини, для яких ВЗАГАЛІ немає цінових даних до/після.
    Це легка перевірка - детальна фільтрація буде в Stage 3 (NewsContextDatasetBuilder).
    
    Логіка:
    - Перевіряємо чи є БУДЬ-ЯКІ цінові дані до новини
    - Перевіряємо чи є БУДЬ-ЯКІ цінові дані після новини
    - Якщо немає - відсікаємо (це сміття або дуже стара/свіжа новина)
    
    НЕ перевіряємо:
    - Релевантність тікерів (це в Stage 3)
    - Кількість свічок (це в Stage 3)
    - Класифікацію новин (це в Stage 3)
    
    Args:
        news_df: News DataFrame with date column
        prices_dict: Dictionary of price DataFrames by timeframe
        news_date_col: Name of the date column in news DataFrame
        
    Returns:
        Filtered news DataFrame (removed obvious junk)
    """
    if news_df.empty or not prices_dict:
        logger.warning("Empty news or prices data, skipping quick filter")
        return news_df
    
    if news_date_col not in news_df.columns:
        logger.error(f"News DataFrame missing '{news_date_col}' column")
        return news_df
    
    before_filter = len(news_df)
    
    # Об'єднати всі цінові дані в один DataFrame для швидкої перевірки
    all_prices = []
    for tf, prices_df in prices_dict.items():
        if isinstance(prices_df, pd.DataFrame) and not prices_df.empty:
            if 'datetime' in prices_df.columns:
                all_prices.append(prices_df[['datetime']].copy())
    
    if not all_prices:
        logger.warning("No price data available for quick filtering")
        return news_df
    
    # Об'єднати всі datetime
    combined_prices = pd.concat(all_prices, ignore_index=True)
    combined_prices['datetime'] = pd.to_datetime(combined_prices['datetime'])
    
    # Видалити timezone для порівняння
    if hasattr(combined_prices['datetime'].dt, 'tz') and combined_prices['datetime'].dt.tz is not None:
        combined_prices['datetime'] = combined_prices['datetime'].dt.tz_localize(None)
    
    # Знайти min/max дати цін
    min_price_date = combined_prices['datetime'].min()
    max_price_date = combined_prices['datetime'].max()
    
    logger.info(f"Price data range: {min_price_date} to {max_price_date}")
    
    # Фільтрувати новини
    news_df = news_df.copy()
    news_df[news_date_col] = pd.to_datetime(news_df[news_date_col])
    
    # Видалити timezone
    if hasattr(news_df[news_date_col].dt, 'tz') and news_df[news_date_col].dt.tz is not None:
        news_df[news_date_col] = news_df[news_date_col].dt.tz_localize(None)
    
    # Відсікти новини поза діапазоном цін (з невеликим буфером)
    # Буфер: залишаємо новини якщо вони не на самому краю
    from datetime import timedelta
    buffer = timedelta(hours=1)
    
    mask = (
        (news_df[news_date_col] > min_price_date + buffer) &
        (news_df[news_date_col] < max_price_date - buffer)
    )
    
    news_filtered = news_df[mask].copy()
    
    after_filter = len(news_filtered)
    removed = before_filter - after_filter
    
    logger.info(f"✅ Quick pre-filter: {before_filter} → {after_filter} news articles")
    logger.info(f"   Removed {removed} news outside price data range (too old or too fresh)")
    
    if removed > 0:
        # Статистика
        too_old = len(news_df[news_df[news_date_col] <= min_price_date + buffer])
        too_fresh = len(news_df[news_df[news_date_col] >= max_price_date - buffer])
        
        logger.info(f"   - Too old (before {min_price_date + buffer}): {too_old}")
        logger.info(f"   - Too fresh (after {max_price_date - buffer}): {too_fresh}")
    
    return news_filtered


class NewsPriceAvailabilityFilter:
    """
    DEPRECATED: Use quick_filter_news_by_data_availability() instead.
    
    This class is kept for backward compatibility but should not be used.
    Heavy filtering is done in Stage 3 (NewsContextDatasetBuilder).
    """
    
    def __init__(self, *args, **kwargs):
        logger.warning(
            "NewsPriceAvailabilityFilter is deprecated. "
            "Use quick_filter_news_by_data_availability() for Stage 2, "
            "or NewsContextDatasetBuilder for Stage 3."
        )
    
    def filter_news(self, news_df, prices_dict, **kwargs):
        """Deprecated method - use quick_filter_news_by_data_availability()"""
        return quick_filter_news_by_data_availability(
            news_df,
            prices_dict,
            news_date_col=kwargs.get('news_date_col', 'published_date')
        )


def filter_news_by_price_availability(*args, **kwargs):
    """
    DEPRECATED: Use quick_filter_news_by_data_availability() instead.
    
    This function is kept for backward compatibility.
    """
    logger.warning(
        "filter_news_by_price_availability() is deprecated. "
        "Use quick_filter_news_by_data_availability() instead."
    )
    return quick_filter_news_by_data_availability(*args, **kwargs)
