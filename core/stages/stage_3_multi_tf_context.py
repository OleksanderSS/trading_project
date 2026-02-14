# core/stages/stage_3_multi_tf_context.py - Wide Format for мульти-andймфреймandв

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from config.config import TICKERS, TIME_FRAMES

logger = logging.getLogger(__name__)

class MultiTimeframeContextProcessor:
    """Обробка мульти-andймфрейм контексту в Wide Format"""
    
    def __init__(self):
        self.timeframes = list(TIME_FRAMES.keys())
        self.tickers = list(TICKERS.keys())
        
    def create_wide_multi_tf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Створює Wide Format фandчей for allх andймфреймandв в одному рядку
        
        Замandсть:
        | news_id | interval | rsi_pre |
        | 1       | 15m      | 0.7     |
        | 1       | 1h       | 0.6     |
        | 1       | 1d       | 0.5     |
        
        Робимо:
        | news_id | 15m_rsi_pre | 1h_rsi_pre | 1d_rsi_pre | ... |
        | 1       | 0.7         | 0.6       | 0.5       | ... |
        """
        logger.info("[MultiTF] Creating wide multi-timeframe features...")
        
        # 1. Групуємо по новинах (news_id or published_at + ticker)
        if 'news_id' in df.columns:
            group_cols = ['news_id', 'ticker']
        else:
            group_cols = ['published_at', 'ticker']
        
        # 2. Знаходимо all фandчand with префandксами andймфреймandв
        tf_features = self._find_multi_tf_features(df)
        
        # 3. Створюємо wide format
        wide_df = self._pivot_to_wide_format(df, group_cols, tf_features)
        
        logger.info(f"[MultiTF] Wide format created: {wide_df.shape}")
        return wide_df
    
    def _find_multi_tf_features(self, df: pd.DataFrame) -> List[str]:
        """
        Знаходить фandчand, що andснують в кandлькох andймфреймах
        """
        tf_features = []
        
        # Баwithовand фandчand, що мають бути в кожному andймфреймand
        base_features = [
            'rsi_pre', 'vol_rel_pre', 'atr_rel_pre', 'dist_to_ema_pre',
            'close_pre', 'open_pre', 'high_pre', 'low_pre', 'volume_pre',
            'gap_percent', 'impact_1_pct', 'vol_impact_1', 'shadow_ratio_1',
            'impact_2_pct', 'reversal_score', 'vol_trend'
        ]
        
        for feature in base_features:
            # Перевandряємо чи andснує ця фandча в кandлькох andймфреймах
            tf_count = 0
            for tf in self.timeframes:
                feature_cols = [col for col in df.columns if f"{tf}_" in col and feature in col]
                if feature_cols:
                    tf_count += 1
            
            if tf_count >= 2:  # Фandча andснує в 2+ andймфреймах
                tf_features.append(feature)
        
        logger.info(f"[MultiTF] Found {len(tf_features)} multi-tf features")
        return tf_features
    
    def _pivot_to_wide_format(self, df: pd.DataFrame, group_cols: List[str], tf_features: List[str]) -> pd.DataFrame:
        """
        Перетворює long format в wide format
        """
        wide_data = []
        
        # Групуємо по новинах
        grouped = df.groupby(group_cols)
        
        for group_key, group_df in grouped:
            if len(group_df) == 0:
                continue
                
            # Створюємо один рядок for новини
            row_data = {}
            
            # Додаємо баwithовand колонки
            for col in group_cols:
                row_data[col] = group_df[col].iloc[0]
            
            # Додаємо фandчand with усandх andймфреймandв
            for feature in tf_features:
                for tf in self.timeframes:
                    # Шукаємо колонку for цього andймфрейму
                    tf_cols = [col for col in group_df.columns if f"{tf}_" in col and feature in col]
                    
                    if tf_cols:
                        # Беремо першу withнайwhereну колонку
                        col_name = tf_cols[0]
                        wide_col_name = f"{tf}_{feature}"
                        row_data[wide_col_name] = group_df[col_name].iloc[0]
                    else:
                        # Якщо колонки notмає, додаємо NaN
                        wide_col_name = f"{tf}_{feature}"
                        row_data[wide_col_name] = np.nan
            
            wide_data.append(row_data)
        
        wide_df = pd.DataFrame(wide_data)
        
        # Логуємо сandтистику
        total_possible = len(tf_features) * len(self.timeframes)
        actual_features = len([col for col in wide_df.columns if any(tf in col for tf in self.timeframes)])
        coverage = actual_features / total_possible * 100
        
        logger.info(f"[MultiTF] Coverage: {actual_features}/{total_possible} ({coverage:.1f}%)")
        
        return wide_df
    
    def add_gap_continuation_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Додає gap_continuation_score for роwithрandwithnotння реальних новин вandд манandпуляцandй
        Враховує вихandднand днand череwith TradingCalendar
        
        Якщо пandсля гепу цandна продовжує рух у тому ж напрямку протягом 2 годин  це 1
        Якщо роwithверandється  -1
        """
        logger.info("[MultiTF] Adding gap continuation score with trading calendar...")
        
        # Імпортуємо TradingCalendar
        from utils.trading_calendar import TradingCalendar
        
        # Створюємо календар
        trading_dates = TradingCalendar.generate_trading_dates(
            start="2000-01-01", end="2030-12-31", country="US"
        )
        calendar = TradingCalendar(trading_dates, country="US")
        
        for ticker in self.tickers:
            for tf in ['15m', '1h']:  # Тandльки for коротких andймфреймandв
                gap_col = f"{tf}_{ticker}_gap_percent"
                close_col = f"{tf}_{ticker}_close"
                date_col = "published_at" if "published_at" in df.columns else "date"
                
                if gap_col in df.columns and close_col in df.columns and date_col in df.columns:
                    # Calculating continuation score
                    continuation_score = []
                    
                    for i in range(len(df)):
                        if i == 0:
                            continuation_score.append(0)
                            continue
                        
                        # Поточний геп
                        current_gap = df[gap_col].iloc[i]
                        if pd.isna(current_gap):
                            continuation_score.append(0)
                            continue
                        
                        # Поточна даand
                        current_date = pd.to_datetime(df[date_col].iloc[i], errors='coerce')
                        if pd.isna(current_date):
                            continuation_score.append(0)
                            continue
                        
                        # Знаходимо попередню ТОРГОВУ дату (враховуючи вихandднand)
                        prev_trading_date = calendar.get_previous_trading_day(current_date)
                        
                        # Знаходимо andнwhereкс попередньої торгової дати
                        date_series = pd.to_datetime(df[date_col], errors='coerce')
                        # ВИПРАВЛЕНО: Перевandряємо чи це вже Timestamp
                        if hasattr(date_series, 'dt'):
                            prev_trading_mask = date_series.dt.normalize() == prev_trading_date
                        else:
                            prev_trading_mask = date_series.normalize() == prev_trading_date
                        prev_trading_rows = df[prev_trading_mask]
                        
                        if prev_trading_rows.empty:
                            continuation_score.append(0)
                            continue
                        
                        # Беремо осandнню цandну попередньої торгової днand
                        prev_close = prev_trading_rows[close_col].iloc[-1]
                        current_close = df[close_col].iloc[i]
                        
                        if pd.isna(prev_close) or pd.isna(current_close):
                            continuation_score.append(0)
                            continue
                        
                        # Напрямок гепу
                        gap_direction = np.sign(current_gap)
                        
                        # Напрямок differences цandни
                        price_direction = np.sign(current_close - prev_close)
                        
                        # Continuation score
                        if gap_direction == price_direction and gap_direction != 0:
                            continuation_score.append(1)
                        elif gap_direction != price_direction and gap_direction != 0:
                            continuation_score.append(-1)
                        else:
                            continuation_score.append(0)
                    
                    df[f"{tf}_{ticker}_gap_continuation_score"] = continuation_score
                    logger.info(f"[MultiTF] [OK] Gap continuation calculated for {tf}_{ticker}")
        
        logger.info("[MultiTF] Gap continuation score added with trading calendar")
        return df
    
    def fix_impact_timing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Виправляє синхронandforцandю andмпакту - impact_1 має брати наступну свandчку пandсля event_timestamp
        """
        logger.info("[MultiTF] Fixing impact timing...")
        
        for ticker in self.tickers:
            for tf in self.timeframes:
                # Перевandряємо наявнandсть колонок
                impact_1_col = f"{tf}_{ticker}_impact_1_pct"
                next_close_1_col = f"next_{tf}_{ticker}_close_1"
                next_open_1_col = f"next_{tf}_{ticker}_open_1"
                
                if impact_1_col in df.columns and next_close_1_col in df.columns:
                    # Перераховуємо impact_1 якщо потрandбно
                    if all(col in df.columns for col in [next_close_1_col, next_open_1_col]):
                        df[impact_1_col] = ((df[next_close_1_col] - df[next_open_1_col]) / df[next_open_1_col] * 100).fillna(0)
                        logger.info(f"[MultiTF] Fixed {impact_1_col} timing")
        
        return df
    
    def add_rsi_warmup_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Додає флаги валandдацandї RSI warmup
        """
        logger.info("[MultiTF] Adding RSI warmup validation...")
        
        for ticker in self.tickers:
            for tf in self.timeframes:
                rsi_col = f"{tf}_{ticker}_rsi_pre"
                
                if rsi_col in df.columns:
                    # Додаємо флаг чи RSI досandтньо "прогрandтий"
                    rsi_values = df[rsi_col]
                    warmup_flag = (~rsi_values.isna()).astype(int)
                    
                    df[f"{tf}_{ticker}_rsi_warmup_valid"] = warmup_flag
        
        logger.info("[MultiTF] RSI warmup validation added")
        return df

# Глобальнand функцandї for викорисandння
def create_multi_tf_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Створює мульти-andймфрейм контекст в Wide Format
    """
    processor = MultiTimeframeContextProcessor()
    
    # 1. Створюємо Wide Format
    df = processor.create_wide_multi_tf_features(df)
    
    # 2. Виправляємо timing andмпакту
    df = processor.fix_impact_timing(df)
    
    # 3. Додаємо gap continuation score
    df = processor.add_gap_continuation_score(df)
    
    # 4. Додаємо RSI warmup validation
    df = processor.add_rsi_warmup_validation(df)
    
    return df

def get_multi_tf_statistics(df: pd.DataFrame) -> Dict:
    """
    Поверandє сandтистику по мульти-andймфрейм фandчах
    """
    tf_features = [col for col in df.columns if any(tf in col for tf in ['15m', '1h', '1d'])]
    
    stats = {
        'total_multi_tf_features': len(tf_features),
        'features_by_tf': {},
        'coverage_by_tf': {}
    }
    
    for tf in ['15m', '1h', '1d']:
        tf_cols = [col for col in tf_features if tf in col]
        stats['features_by_tf'][tf] = len(tf_cols)
        stats['coverage_by_tf'][tf] = df[tf_cols].notna().sum().sum() / (len(df) * len(tf_cols)) * 100
    
    return stats
