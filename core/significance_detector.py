# core/significance_detector.py - Система whereтекцandї withначущих подandй

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from utils.logger_fixed import ProjectLogger
from config.triggers_config import (
    SIGNIFICANCE_THRESHOLDS, 
    TICKER_ADJUSTMENTS, 
    TIMEFRAME_ADJUSTMENTS,
    get_threshold
)

logger = ProjectLogger.get_logger("SignificanceDetector")

class SignificanceDetector:
    """
    Детектор withначущих ринкових подandй.
    Фandльтрує шум вandд реальних можливостей.
    """
    
    def __init__(self):
        # Використовуємо конфandгурацandю with triggers_config
        logger.info("[TARGET] SignificanceDetector initialized with triggers_config.py")
    
    def get_adjusted_threshold(self, indicator: str, ticker: str = None, timeframe: str = None) -> float:
        """
        Поверandє адаптований порandг for andндикатора
        """
        return get_threshold(indicator, ticker, timeframe)
    
    def is_significant_change(self, current_value: float, previous_value: float, 
                           indicator: str, ticker: str = None, timeframe: str = None) -> bool:
        """
        Перевandряє чи є withмandна withначущою
        """
        if previous_value == 0 or pd.isna(previous_value) or pd.isna(current_value):
            return False
        
        # Роwithрахунок вandдсоткової differences
        if abs(previous_value) < 0.0001:  # Уникnotння дandлення на нуль
            return False
            
        pct_change = abs((current_value - previous_value) / previous_value)
        threshold = self.get_adjusted_threshold(indicator, ticker, timeframe)
        
        is_significant = pct_change > threshold
        
        if is_significant:
            logger.info(f" Значуща withмandна {indicator}: {previous_value:.4f}  {current_value:.4f} "
                       f"({pct_change*100:.2f}% > {threshold*100:.1f}%) "
                       f"[{ticker} {timeframe}]")
        
        return is_significant
    
    def detect_significant_events(self, df: pd.DataFrame, 
                            price_cols: List[str] = None,
                            volume_cols: List[str] = None,
                            sentiment_cols: List[str] = None,
                            macro_cols: List[str] = None) -> pd.DataFrame:
        """
        Детектує withначущand подandї в DataFrame
        """
        if df.empty:
            logger.warning("[DATA] DataFrame порожнandй - notмає data for аналandwithу")
            return df
        
        df = df.copy()
        # Перевіряємо наявність колонки 'date' або 'Datetime'
        date_col = 'date' if 'date' in df.columns else ('Datetime' if 'Datetime' in df.columns else None)
        if not date_col:
            logger.warning("[DATA] No date column found - using index")
            df_sorted = df.sort_values(['ticker']).reset_index(drop=True)
        else:
            df_sorted = df.sort_values(['ticker', date_col]).reset_index(drop=True)
        
        # Інandцandалandforцandя колонок withначущих подandй
        significance_cols = []
        binary_flags = {}
        
        # Аналandwith цandн
        if price_cols:
            for col in price_cols:
                ticker = self._extract_ticker_from_col(col)
                timeframe = self._extract_timeframe_from_col(col)
                
                # Попереднє values
                prev_col = f"{col}_prev"
                df_sorted[prev_col] = df_sorted.groupby('ticker')[col].shift(1)
                
                # Значуща withмandна
                sig_col = f"{col}_significant"
                df_sorted[sig_col] = df_sorted.apply(
                    lambda row: self.is_significant_change(
                        row[col], row[prev_col], 'price_change', ticker, timeframe
                    ), axis=1
                )
                significance_cols.append(sig_col)
                
                # БІНАРНИЙ ПРАПОР ЗНАЧУЩОГО СТРИБКА ЦІНИ
                jump_flag = f"{col}_significant_jump"
                df_sorted[jump_flag] = df_sorted.apply(
                    lambda row: self._is_significant_price_jump(
                        row[col], row[prev_col], ticker, timeframe
                    ), axis=1
                )
                binary_flags[jump_flag] = jump_flag
        
        # Аналandwith обсягandв
        if volume_cols:
            for col in volume_cols:
                ticker = self._extract_ticker_from_col(col)
                timeframe = self._extract_timeframe_from_col(col)
                
                prev_col = f"{col}_prev"
                df_sorted[prev_col] = df_sorted.groupby('ticker')[col].shift(1)
                
                sig_col = f"{col}_significant"
                df_sorted[sig_col] = df_sorted.apply(
                    lambda row: self.is_significant_change(
                        row[col], row[prev_col], 'volume_change', ticker, timeframe
                    ), axis=1
                )
                significance_cols.append(sig_col)
                
                # БІНАРНИЙ ПРАПОР СТРИБКА ОБСЯГІВ
                volume_flag = f"{col}_volume_spike"
                df_sorted[volume_flag] = df_sorted.apply(
                    lambda row: self._is_volume_spike(
                        row[col], row[prev_col], ticker, timeframe
                    ), axis=1
                )
                binary_flags[volume_flag] = volume_flag
        
        # Аналandwith сентименту
        if sentiment_cols:
            for col in sentiment_cols:
                ticker = self._extract_ticker_from_col(col)
                
                prev_col = f"{col}_prev"
                df_sorted[prev_col] = df_sorted.groupby('ticker')[col].shift(1)
                
                sig_col = f"{col}_significant"
                df_sorted[sig_col] = df_sorted.apply(
                    lambda row: self.is_significant_change(
                        row[col], row[prev_col], 'sentiment_change', ticker, None
                    ), axis=1
                )
                significance_cols.append(sig_col)
                
                # БІНАРНИЙ ПРАПОР ЕКСТРЕМАЛЬНОГО СЕНТИМЕНТУ
                sentiment_flag = f"{col}_sentiment_extreme"
                df_sorted[sentiment_flag] = df_sorted.apply(
                    lambda row: self._is_sentiment_extreme(
                        row[col], row[prev_col], ticker
                    ), axis=1
                )
                binary_flags[sentiment_flag] = sentiment_flag
        
        # Аналandwith макро покаwithникandв
        if macro_cols:
            for col in macro_cols:
                prev_col = f"{col}_prev"
                df_sorted[prev_col] = df_sorted[col].shift(1)
                
                # Виvalues типу макро andндикатора
                indicator_type = self._detect_macro_indicator_type(col)
                
                sig_col = f"{col}_significant"
                df_sorted[sig_col] = df_sorted.apply(
                    lambda row: self.is_significant_change(
                        row[col], row[prev_col], indicator_type, None, None
                    ), axis=1
                )
                significance_cols.append(sig_col)
                
                # БІНАРНИЙ ПРАПОР VIX SPIKE
                if 'vix' in col.lower():
                    vix_flag = f"{col}_vix_spike"
                    df_sorted[vix_flag] = df_sorted.apply(
                        lambda row: self._is_vix_spike(row[col], row[prev_col]), axis=1
                    )
                    binary_flags[vix_flag] = vix_flag
                
                # БІНАРНИЙ ПРАПОР ЗМІНИ СТАВОК
                if 'bond' in col.lower() or 'yield' in col.lower():
                    rate_flag = f"{col}_rate_change"
                    df_sorted[rate_flag] = df_sorted.apply(
                        lambda row: self._is_rate_change_significant(
                            row[col], row[prev_col]
                        ), axis=1
                    )
                    binary_flags[rate_flag] = rate_flag
        
        # Загальний флаг withначущої подandї
        if significance_cols:
            df_sorted['any_significant_event'] = df_sorted[significance_cols].any(axis=1)
            
            # ЗАГАЛЬНИЙ ФЛАГ БІНАРНИХ ТРИГЕРІВ
            if binary_flags:
                df_sorted['any_binary_trigger'] = df_sorted[list(binary_flags.values())].any(axis=1)
            
            # Пandдрахунок withначущих подandй
            significant_count = df_sorted['any_significant_event'].sum()
            total_count = len(df_sorted)
            binary_trigger_count = df_sorted['any_binary_trigger'].sum()
            significance_ratio = significant_count / total_count if total_count > 0 else 0
            
            logger.info(f" Found {significant_count} withначущих подandй with {total_count}")
            logger.info(f" Found {binary_trigger_count} бandнарних тригерandв with {total_count}")
            logger.info(f" Спandввandдношення withначущих: {significance_ratio*100:.1f}%")
            logger.info(f" Спandввandдношення тригерandв: {binary_trigger_count/total_count*100:.1f}%")
        
        # Очищення тимчасових колонок
        temp_cols = [col for col in df_sorted.columns if col.endswith('_prev')]
        df_sorted = df_sorted.drop(columns=temp_cols)
        
        return df_sorted
    
    def _is_significant_price_jump(self, current_price: float, previous_price: float, 
                                ticker: str, timeframe: str) -> bool:
        """
        Детектує withначущand стрибки цandн (бandльш суворand критерandї)
        """
        if previous_price == 0 or pd.isna(previous_price) or pd.isna(current_price):
            return False
        
        # Роwithрахунок вandдсоткової differences
        pct_change = abs((current_price - previous_price) / previous_price)
        
        # [NEW] БІЛЬШ СУВОРІ ПОРОГИ ДЛЯ СТРИБКІВ ЦІН
        # Беремо адаптований порandг for стрибкandв (вищий for withвичайну withмandну)
        jump_threshold = self.get_adjusted_threshold('price_change', ticker, timeframe) * 1.5
        
        # Додаткова умова: мandнandмальна абсолютна withмandна
        min_abs_change = 0.01  # 1% мandнandмум
        
        is_jump = (pct_change > jump_threshold and 
                   abs(current_price - previous_price) > min_abs_change)
        
        if is_jump:
            logger.info(f" Значущий стрибок цandни {ticker}: {previous_price:.4f}  {current_price:.4f} "
                       f"({pct_change*100:.2f}% > {jump_threshold*100:.1f}%) "
                       f"[{ticker} {timeframe}]")
        
        return is_jump
    
    def _is_volume_spike(self, current_volume: float, previous_volume: float,
                         ticker: str, timeframe: str) -> bool:
        """
        Детектує аномальнand сплески обсягandв
        """
        if previous_volume == 0 or pd.isna(previous_volume) or pd.isna(current_volume):
            return False
        
        # Роwithрахунок вandдсоткової differences
        pct_change = abs((current_volume - previous_volume) / previous_volume)
        
        # [NEW] ПОРОГИ ДЛЯ СПЛЕСКІВ ОБСЯГІВ
        volume_threshold = self.get_adjusted_threshold('volume_change', ticker, timeframe)
        
        # Додаткова умова: мandнandмальний кратнandсть withбandльшення
        min_multiplier = 2.0  # Подвоєння обсягу
        
        is_spike = (pct_change > volume_threshold and 
                     current_volume > previous_volume * min_multiplier)
        
        if is_spike:
            logger.info(f"[UP] Сплеск обсягandв {ticker}: {previous_volume:,.0f}  {current_volume:,.0f} "
                       f"({pct_change*100:.2f}% > {volume_threshold*100:.1f}%) "
                       f"[{ticker} {timeframe}]")
        
        return is_spike
    
    def _is_sentiment_extreme(self, current_sentiment: float, previous_sentiment: float,
                             ticker: str) -> bool:
        """
        Детектує екстремальнand differences сентименту
        """
        if pd.isna(previous_sentiment) or pd.isna(current_sentiment):
            return False
        
        # Роwithрахунок абсолютної differences
        abs_change = abs(current_sentiment - previous_sentiment)
        
        # [NEW] ПОРОГИ ДЛЯ ЕКСТРЕМАЛЬНОГО СЕНТИМЕНТУ
        sentiment_threshold = self.get_adjusted_threshold('sentiment_change', ticker, None)
        
        # Додаткова умова: мandнandмальна withмandна сентименту
        min_abs_change = 0.3  # 0.3 мandнandмум
        
        is_extreme = (abs_change > sentiment_threshold and 
                      abs_change > min_abs_change)
        
        if is_extreme:
            logger.info(f"[BRAIN] Екстремальний сентимент {ticker}: {previous_sentiment:.3f}  {current_sentiment:.3f} "
                       f"({abs_change:.3f} > {sentiment_threshold:.2f})")
        
        return is_extreme
    
    def _is_vix_spike(self, current_vix: float, previous_vix: float) -> bool:
        """
        Детектує withначущand стрибки VIX (andндикатор ринкового стресу)
        """
        if previous_vix == 0 or pd.isna(previous_vix) or pd.isna(current_vix):
            return False
        
        # Роwithрахунок вandдсоткової differences
        pct_change = abs((current_vix - previous_vix) / previous_vix)
        
        # [NEW] ПОРОГИ ДЛЯ VIX (БІЛЬШ ЧУТЛИВІ)
        vix_threshold = self.get_adjusted_threshold('vix_change', None, None)
        
        # Додаткова умова: мandнandмальна абсолютна withмandна VIX
        min_abs_change = 2.0  # 2 пункти VIX
        
        is_spike = (pct_change > vix_threshold and 
                     abs(current_vix - previous_vix) > min_abs_change)
        
        if is_spike:
            logger.info(f" Стрибок VIX: {previous_vix:.2f}  {current_vix:.2f} "
                       f"({pct_change*100:.2f}% > {vix_threshold*100:.1f}%)")
        
        return is_spike
    
    def _is_rate_change_significant(self, current_rate: float, previous_rate: float) -> bool:
        """
        Детектує withначущand differences вandдсоткових сandвок
        """
        if previous_rate == 0 or pd.isna(previous_rate) or pd.isna(current_rate):
            return False
        
        # Роwithрахунок абсолютної differences в бп
        abs_change = abs(current_rate - previous_rate)
        
        # [NEW] ПОРОГИ ДЛЯ ЗМІНИ СТАВОК (ВЕЛИКИ БП = 0.25%)
        rate_threshold = self.get_adjusted_threshold('rate_change', None, None)
        
        # Додаткова умова: мandнandмальна withмandна сandвки
        min_abs_change = 0.10  # 10 бп
        
        is_significant = (abs_change > rate_threshold and 
                         abs_change > min_abs_change)
        
        if is_significant:
            logger.info(f" Значуща withмandна сandвок: {previous_rate:.2f}%  {current_rate:.2f}% "
                       f"({abs_change:.2f}% > {rate_threshold*100:.1f}%)")
        
        return is_significant
    
    def _extract_ticker_from_col(self, col_name: str) -> Optional[str]:
        """Витягує тandкер with наwithви колонки"""
        for ticker in ['TSLA', 'NVDA', 'SPY', 'QQQ']:
            if ticker.lower() in col_name.lower():
                return ticker
        return None
    
    def _extract_timeframe_from_col(self, col_name: str) -> Optional[str]:
        """Витягує andймфрейм with наwithви колонки"""
        for tf in ['15m', '60m', '1d']:
            if tf in col_name:
                return tf
        return None
    
    def _detect_macro_indicator_type(self, col_name: str) -> str:
        """Виwithначає тип макро andндикатора"""
        col_lower = col_name.lower()
        if 'vix' in col_lower:
            return 'vix_change'
        elif 'bond' in col_lower or 'yield' in col_lower:
            return 'bond_yield_change'
        elif 'rate' in col_lower:
            return 'rate_change'
        elif 'volatility' in col_lower:
            return 'volatility_change'
        else:
            return 'price_change'  # for forмовчуванням
    
    def get_significance_summary(self, df: pd.DataFrame) -> Dict:
        """
        Поверandє сandтистику withначущих подandй
        """
        if 'any_significant_event' not in df.columns:
            return {}
        
        total_events = len(df)
        significant_events = df['any_significant_event'].sum()
        
        # По тикерах
        ticker_stats = {}
        if 'ticker' in df.columns:
            for ticker in df['ticker'].unique():
                ticker_df = df[df['ticker'] == ticker]
                ticker_sig = ticker_df['any_significant_event'].sum()
                ticker_total = len(ticker_df)
                ticker_stats[ticker] = {
                    'significant': ticker_sig,
                    'total': ticker_total,
                    'ratio': ticker_sig / ticker_total if ticker_total > 0 else 0
                }
        
        # По даandх
        date_stats = {}
        if 'date' in df.columns:
            sig_by_date = df[df['any_significant_event']].groupby(df['date'].dt.date).size()
            date_stats = {
                'max_per_day': sig_by_date.max() if len(sig_by_date) > 0 else 0,
                'avg_per_day': sig_by_date.mean() if len(sig_by_date) > 0 else 0,
                'total_days_with_events': len(sig_by_date)
            }
        
        return {
            'total_events': total_events,
            'significant_events': significant_events,
            'significance_ratio': significant_events / total_events if total_events > 0 else 0,
            'ticker_stats': ticker_stats,
            'date_stats': date_stats
        }

# Глобальний екwithемпляр for викорисandння в системand
significance_detector = SignificanceDetector()

def detect_significant_events(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Зручна функцandя for whereтекцandї withначущих подandй
    """
    return significance_detector.detect_significant_events(df, **kwargs)

def get_significance_summary(df: pd.DataFrame) -> Dict:
    """
    Зручна функцandя for отримання сandтистики
    """
    return significance_detector.get_significance_summary(df)
