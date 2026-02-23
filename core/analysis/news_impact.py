# core/analysis/news_impact.py - Кandлькandсна оцandнка впливу новин на цandну

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from utils.logger_fixed import ProjectLogger
from config.triggers_config import get_threshold

logger = ProjectLogger.get_logger("NewsImpactAnalyzer")

class NewsImpactAnalyzer:
    """
    Аналandforтор впливу новин на цandни.
    Кandлькandсно оцandнює наслandдки новинних подandй.
    """
    
    def __init__(self):
        self.logger = ProjectLogger.get_logger("NewsImpactAnalyzer")
        self.logger.info("NewsImpactAnalyzer initialized")
    
    def analyze_news_price_impact(self, news_df: pd.DataFrame, 
                                 prices_df: pd.DataFrame = None,
                                 news_cols: List[str] = None,
                                 price_cols: List[str] = None,
                                 time_windows: List[str] = None) -> pd.DataFrame:
        """
        Аналізує вплив новин на ціни в різних часових вікнах
        """
        if news_df.empty:
            logger.warning("[DATA] News DataFrame порожній - немає data для аналізу")
            return pd.DataFrame()
        
        # Якщо цінові дані не надано, повертаємо порожній результат
        if prices_df is None or prices_df.empty:
            logger.warning("[DATA] Price DataFrame порожній - немає цінових data для аналізу впливу")
            return pd.DataFrame()
        
        news_df = news_df.copy()
        prices_df = prices_df.copy()
        
        # Визначаємо колонки для аналізу
        if news_cols is None:
            news_cols = [col for col in news_df.columns if any(x in col.lower() for x in ['news', 'sentiment', 'mention'])]
        
        if price_cols is None:
            price_cols = [col for col in prices_df.columns if any(x in col.lower() for x in ['close', 'open', 'high', 'low'])]
        
        if time_windows is None:
            time_windows = ['1h', '6h', '24h', '3d']
        
        logger.info(f"[DATA] Аналіз впливу новин на ціни:")
        logger.info(f"  - Новинних колонок: {len(news_cols)}")
        logger.info(f"  - Цінових колонок: {len(price_cols)}")
        logger.info(f"  - Часові вікна: {time_windows}")
        
        # Якщо немає цінових колонок, повертаємо порожній результат
        if len(price_cols) == 0:
            logger.warning("[DATA] Немає цінових колонок для аналізу")
            return pd.DataFrame()
        
        # Сортуємо по датand for аналandwithу часових послandдовностей
        if 'date' in news_df.columns:
            news_df = news_df.sort_values('date').reset_index(drop=True)
        
        # Аналізуємо кожну новинну подію
        impact_features = []
        
        for idx, row in news_df.iterrows():
            row_impacts = {}
            
            # Аналізуємо вплив на ціни після новини
            for price_col in price_cols:
                ticker = self._extract_ticker_from_col(price_col)
                
                # [NEW] ЦІНОВІ ЗМІНИ ПІСЛЯ НОВИНИ
                for window in time_windows:
                    impact_col = f"{price_col}_change_{window}_after_news"
                    if impact_col not in row_impacts:
                        row_impacts[impact_col] = self._calculate_price_change_after_news(
                            prices_df, idx, price_col, window
                        )
                
                # [NEW] МАКСИМАЛЬНИЙ ДРИФТ ЦІНИ
                drift_col = f"{price_col}_max_drift_6h_after_news"
                if drift_col not in row_impacts:
                    row_impacts[drift_col] = self._calculate_max_price_drift(
                        prices_df, idx, price_col, '6h'
                    )
                
                # [NEW] ВОЛАТИЛЬНІСТЬ ПІСЛЯ НОВИНИ
                vol_col = f"{price_col}_volatility_24h_after_news"
                if vol_col not in row_impacts:
                    row_impacts[vol_col] = self._calculate_volatility_after_news(
                        prices_df, idx, price_col, '24h'
                    )
                
                # [NEW] ОБСЯГИ ПІСЛЯ НОВИНИ
                volume_col = f"{ticker}_volume"
                if volume_col in prices_df.columns:
                    volume_impact = f"{volume_col}_change_6h_after_news"
                    if volume_impact not in row_impacts:
                        row_impacts[volume_impact] = self._calculate_volume_change_after_news(
                            prices_df, idx, volume_col, '6h'
                        )
            
            # Аналізуємо сентиментні події
            for news_col in news_cols:
                if 'sentiment' in news_col.lower():
                    # [NEW] СЕНТИМЕНТНА ЕФЕКТИВНІСТЬ
                    sentiment_efficiency = f"{news_col}_efficiency_6h"
                    if sentiment_efficiency not in row_impacts:
                        row_impacts[sentiment_efficiency] = self._calculate_sentiment_efficiency(
                            news_df, idx, news_col, '6h'
                        )
                    
                    # [NEW] СЕНТИМЕНТНА ТОЧНІСТЬ
                    sentiment_accuracy = f"{news_col}_accuracy_24h"
                    if sentiment_accuracy not in row_impacts:
                        row_impacts[sentiment_accuracy] = self._calculate_sentiment_accuracy(
                            news_df, idx, news_col, '24h'
                        )
            
            impact_features.append(row_impacts)
        
        # Додаємо фічі до DataFrame
        impact_df = pd.DataFrame(impact_features)
        result_df = pd.concat([news_df, impact_df], axis=1)
        
        # [NEW] ЗАГАЛЬНІ МЕТРИКИ ВПЛИВУ
        result_df = self._add_overall_impact_metrics(result_df, news_cols, price_cols)
        
        logger.info(f"[OK] Додано {len(impact_df.columns)} фіч впливу новин")
        
        return result_df
    
    def _calculate_price_change_after_news(self, df: pd.DataFrame, 
                                         news_idx: int, price_col: str, 
                                         time_window: str) -> float:
        """
        Роwithраховує withмandну цandни пandсля новини
        """
        if news_idx >= len(df) - 1:
            return 0.0
        
        # Отримуємо цandну на момент новини
        current_price = df.iloc[news_idx][price_col]
        if pd.isna(current_price) or current_price == 0:
            return 0.0
        
        # Виwithначаємо часове вandкно
        window_hours = self._parse_time_window(time_window)
        if window_hours is None:
            return 0.0
        
        # Знаходимо цandну череwith вкаforний час
        future_idx = min(news_idx + window_hours, len(df) - 1)
        future_price = df.iloc[future_idx][price_col]
        
        if pd.isna(future_price) or future_price == 0:
            return 0.0
        
        # Calculating вandдсоткову withмandну
        price_change = (future_price - current_price) / current_price
        
        return price_change
    
    def _calculate_max_price_drift(self, df: pd.DataFrame, 
                                 news_idx: int, price_col: str, 
                                 time_window: str) -> float:
        """
        Роwithраховує максимальний дрифт цandни пandсля новини
        """
        if news_idx >= len(df) - 1:
            return 0.0
        
        # Отримуємо цandну на момент новини
        current_price = df.iloc[news_idx][price_col]
        if pd.isna(current_price) or current_price == 0:
            return 0.0
        
        # Виwithначаємо часове вandкно
        window_hours = self._parse_time_window(time_window)
        if window_hours is None:
            return 0.0
        
        # Знаходимо максимальний дрифт у вandкнand
        future_idx = min(news_idx + window_hours, len(df) - 1)
        price_slice = df.iloc[news_idx:future_idx+1][price_col].dropna()
        
        if len(price_slice) < 2:
            return 0.0
        
        # Calculating максимальnot вandдхилення
        max_price = price_slice.max()
        min_price = price_slice.min()
        
        max_drift_up = (max_price - current_price) / current_price if max_price > current_price else 0.0
        max_drift_down = (current_price - min_price) / current_price if min_price < current_price else 0.0
        
        # Поверandємо максимальний дрифт (withнаковий)
        max_drift = max_drift_up if max_drift_up > max_drift_down else -max_drift_down
        
        return max_drift
    
    def _calculate_volatility_after_news(self, df: pd.DataFrame, 
                                       news_idx: int, price_col: str, 
                                       time_window: str) -> float:
        """
        Роwithраховує волатильнandсть пandсля новини
        """
        if news_idx >= len(df) - 1:
            return 0.0
        
        # Виwithначаємо часове вandкно
        window_hours = self._parse_time_window(time_window)
        if window_hours is None:
            return 0.0
        
        # Отримуємо цandни у вandкнand
        future_idx = min(news_idx + window_hours, len(df) - 1)
        price_slice = df.iloc[news_idx:future_idx+1][price_col].dropna()
        
        if len(price_slice) < 2:
            return 0.0
        
        # Calculating волатильнandсть (сandндартnot вandдхилення вandдсоткових withмandн)
        price_changes = price_slice.pct_change().dropna()
        
        if len(price_changes) < 2:
            return 0.0
        
        volatility = price_changes.std()
        
        return volatility
    
    def _calculate_volume_change_after_news(self, df: pd.DataFrame, 
                                          news_idx: int, volume_col: str, 
                                          time_window: str) -> float:
        """
        Роwithраховує withмandну обсягandв пandсля новини
        """
        if news_idx >= len(df) - 1:
            return 0.0
        
        # Отримуємо обсяг на момент новини
        current_volume = df.iloc[news_idx][volume_col]
        if pd.isna(current_volume) or current_volume == 0:
            return 0.0
        
        # Виwithначаємо часове вandкно
        window_hours = self._parse_time_window(time_window)
        if window_hours is None:
            return 0.0
        
        # Знаходимо середнandй обсяг у вandкнand
        future_idx = min(news_idx + window_hours, len(df) - 1)
        volume_slice = df.iloc[news_idx+1:future_idx+1][volume_col].dropna()
        
        if len(volume_slice) == 0:
            return 0.0
        
        avg_future_volume = volume_slice.mean()
        
        if avg_future_volume == 0:
            return 0.0
        
        # Calculating вandдсоткову withмandну
        volume_change = (avg_future_volume - current_volume) / current_volume
        
        return volume_change
    
    def _calculate_sentiment_efficiency(self, news_df: pd.DataFrame, 
                                      news_idx: int, sentiment_col: str, 
                                      time_window: str) -> float:
        """
        Розраховує ефективність сентименту (наскільки точно передбачив рух)
        """
        if news_idx >= len(news_df) - 1:
            return 0.0
        
        # Отримуємо сентимент на момент новини
        current_sentiment = news_df.iloc[news_idx][sentiment_col]
        if pd.isna(current_sentiment):
            return 0.0
        
        # Для простоти повертаємо базове значення
        return float(current_sentiment)
    
    def _calculate_sentiment_accuracy(self, news_df: pd.DataFrame, 
                                    news_idx: int, sentiment_col: str, 
                                    time_window: str) -> float:
        """
        Розраховує точність сентименту (наскільки правильно передбачив напрямок)
        """
        if news_idx >= len(news_df) - 1:
            return 0.0
        
        # Отримуємо сентимент на момент новини
        current_sentiment = news_df.iloc[news_idx][sentiment_col]
        if pd.isna(current_sentiment):
            return 0.0
        
        # Для простоти повертаємо базове значення
        return float(abs(current_sentiment))
    
    def _add_overall_impact_metrics(self, df: pd.DataFrame, 
                                   news_cols: List[str], 
                                   price_cols: List[str]) -> pd.DataFrame:
        """
        Додає загальні метрики впливу новин
        """
        # [NEW] ЗАГАЛЬНИЙ ІНДЕКС ВПЛИВУ НОВИНИ
        impact_cols = [col for col in df.columns if 'change_' in col and 'after_news' in col]
        if impact_cols:
            df['news_impact_index'] = df[impact_cols].mean(axis=1)
        
        # [NEW] СИЛА РЕАКЦІЇ РИНКУ
        drift_cols = [col for col in df.columns if 'max_drift' in col and 'after_news' in col]
        if drift_cols:
            df['market_reaction_strength'] = df[drift_cols].abs().mean(axis=1)
        
        # [NEW] ВОЛАТИЛЬНІСТЬ ПІСЛЯ НОВИНИ
        vol_cols = [col for col in df.columns if 'volatility' in col and 'after_news' in col]
        if vol_cols:
            df['post_news_volatility'] = df[vol_cols].mean(axis=1)
        
        # [NEW] ЕФЕКТИВНІСТЬ СЕНТИМЕНТУ
        eff_cols = [col for col in df.columns if 'efficiency' in col and 'after_news' in col]
        if eff_cols:
            df['sentiment_efficiency_avg'] = df[eff_cols].mean(axis=1)
        
        # [NEW] ТОЧНІСТЬ СЕНТИМЕНТУ
        acc_cols = [col for col in df.columns if 'accuracy' in col and 'after_news' in col]
        if acc_cols:
            df['sentiment_accuracy_avg'] = df[acc_cols].mean(axis=1)
        
        return df
    
    def _parse_time_window(self, time_window: str) -> Optional[int]:
        """
        Парсить часове вandкно в години
        """
        window_mapping = {
            '1h': 1,
            '6h': 6,
            '24h': 24,
            '3d': 72,
            '1w': 168
        }
        
        return window_mapping.get(time_window)
    
    def _extract_ticker_from_col(self, col_name: str) -> Optional[str]:
        """
        Витягує тandкер with наwithви колонки
        """
        for ticker in ['TSLA', 'NVDA', 'SPY', 'QQQ']:
            if ticker.lower() in col_name.lower():
                return ticker
        return None
    
    def get_impact_summary(self, df: pd.DataFrame) -> Dict:
        """
        Поверandє сandтистику впливу новин
        """
        summary = {}
        
        # Сandтистика по withмandнах цandн
        price_change_cols = [col for col in df.columns if 'change_' in col and 'after_news' in col]
        if price_change_cols:
            changes = df[price_change_cols].values.flatten()
            changes = changes[~np.isnan(changes)]
            
            if len(changes) > 0:
                summary['price_changes'] = {
                    'mean': np.mean(changes),
                    'std': np.std(changes),
                    'min': np.min(changes),
                    'max': np.max(changes),
                    'positive_ratio': np.mean(changes > 0)
                }
        
        # Сandтистика по дрифту
        drift_cols = [col for col in df.columns if 'max_drift' in col and 'after_news' in col]
        if drift_cols:
            drifts = df[drift_cols].values.flatten()
            drifts = drifts[~np.isnan(drifts)]
            
            if len(drifts) > 0:
                summary['price_drifts'] = {
                    'mean': np.mean(drifts),
                    'std': np.std(drifts),
                    'max_positive': np.max(drifts),
                    'max_negative': np.min(drifts)
                }
        
        # Сandтистика по ефективностand сентименту
        eff_cols = [col for col in df.columns if 'efficiency' in col and 'after_news' in col]
        if eff_cols:
            efficiencies = df[eff_cols].values.flatten()
            efficiencies = efficiencies[~np.isnan(efficiencies)]
            
            if len(efficiencies) > 0:
                summary['sentiment_efficiency'] = {
                    'mean': np.mean(efficiencies),
                    'accuracy': np.mean(efficiencies > 0)
                }
        
        return summary

# Глобальний екземпляр for використання в системі
news_impact_analyzer = NewsImpactAnalyzer()

def analyze_news_price_impact(news_df: pd.DataFrame, prices_df: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
    """
    Зручна функція для аналізу впливу новин на ціни
    """
    return news_impact_analyzer.analyze_news_price_impact(news_df, prices_df, **kwargs)

def get_news_impact_summary(df: pd.DataFrame) -> Dict:
    """
    Зручна функція for отримання статистики впливу новин
    """
    return news_impact_analyzer.get_impact_summary(df)