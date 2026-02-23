"""
Enhanced Technical Features - Універсальна функція для додавання всіх індикаторів
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from utils.universal_linear_technical import UniversalLinearTechnical
from config.technical_config import TECHNICAL_WINDOWS, PRIORITY_INDICATORS, TIMEFRAME_OPTIMIZATION
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("TradingProjectLogger")


def add_all_technical_indicators(
    df: pd.DataFrame, 
    timeframe: str = "1h",
    priority_only: bool = False,
    custom_indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Додає всі технічні індикатори до DataFrame
    
    Args:
        df: DataFrame з OHLCV даними
        timeframe: Таймфрейм для оптимізації параметрів
        priority_only: Чи додавати тільки пріоритетні індикатори
        custom_indicators: Кастомний список індикаторів
        
    Returns:
        pd.DataFrame: DataFrame з доданими індикаторами
    """
    logger.info(f"[enhanced_technical] Adding technical indicators for timeframe {timeframe}")
    
    # Використовуємо універсальний лінійний калькулятор
    linear_calc = UniversalLinearTechnical()
    
    # Отримуємо всі індикатори
    all_indicators = linear_calc.calculate_all_indicators_for_all_tickers(df, timeframe=timeframe)
    
    # Додаємо індикатори до DataFrame
    for ticker, indicators in all_indicators.items():
        for indicator_name, value in indicators.items():
            column_name = f"{ticker}_{indicator_name}" if ticker else indicator_name
            df[column_name] = value
    
    logger.info(f"[enhanced_technical] Added {len(all_indicators)} ticker indicators using linear calculator")
    return df


def add_all_technical_features(
    df: pd.DataFrame, 
    timeframe: str = "1h",
    priority_only: bool = False,
    custom_indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Alias для add_all_technical_indicators для сумісності
    """
    return add_all_technical_indicators(df, timeframe, priority_only, custom_indicators)


def add_all_technical_features_optimized(
    df: pd.DataFrame, 
    timeframe: str = "1h",
    priority_only: bool = False,
    custom_indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Оптимізована версія - використовує лінійний калькулятор
    """
    return add_all_technical_indicators(df, timeframe, priority_only, custom_indicators)
