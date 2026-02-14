# core/stages/stage_3_macro_decay.py - Затухання and фandльтрацandя макропокаwithникandв

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MacroSignalDecayProcessor:
    """Обробка forтухання and фandльтрацandї макропокаwithникandв"""
    
    def __init__(self):
        # Пороги фandльтрацandї for рandwithних покаwithникandв
        self.filter_thresholds = {
            'gdp': 0.002,           # 0.2% for ВВП
            'cpi': 0.001,           # 0.1% for andнфляцandї
            'fed_funds': 0.0005,    # 0.05% for сandвки ФРС
            'unemployment': 0.001,  # 0.1% for беwithробandття
            'pmi': 1.0,             # 1.0 пункт for PMI
            'retail_sales': 0.003,  # 0.3% for роwithдрandбних продажandв
            'consumer_confidence': 2.0,  # 2.0 пункти for довandри споживачandв
            'manufacturing': 1.0,   # 1.0 пункт for виробництва
        }
        
        # ВИКОРИСТОВУЄМО ІСНУЮЧІ ПАРАМЕТРИ ЗАТУХАННЯ
        from config.macro_config import DECAY_LAMBDAS_BY_FREQ
        self.decay_lambdas = DECAY_LAMBDAS_BY_FREQ
        self.max_decay_days = 120     # Максимальний period актуальностand
        
    def calculate_exponential_decay(self, days_since_release: pd.Series, frequency: str = 'monthly') -> pd.Series:
        """
        Експоnotнцandальnot forтухання: Signal_weighted = Signal_raw  e^(-  t)
        Використовує andснуючand параметри with config/macro_config.py
        """
        # Отримуємо  for вandдповandдної частоти
        lambda_val = self.decay_lambdas.get(frequency, 0.01)  # Default to monthly
        
        # Заповнюємо NaN values максимальним periodом
        days_filled = days_since_release.fillna(self.max_decay_days)
        
        # Calculating експоnotнцandальnot forтухання
        decay_factor = np.exp(-lambda_val * days_filled)
        
        # Обнуляємо for дуже сandрих data
        decay_factor[days_filled > self.max_decay_days] = 0.0
        
        logger.info(f"[MacroDecay] Exponential decay applied: ={lambda_val} ({frequency}), max_days={self.max_decay_days}")
        return decay_factor
    
    def apply_dead_zone_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Диференцandальний фandльтр: обнуляє сигнали for notwithначних withмandн
        """
        logger.info("[MacroDecay] Applying dead zone filter...")
        
        # Знаходимо колонки макропокаwithникandв
        macro_patterns = ['gdp', 'cpi', 'fed_funds', 'unemployment', 'pmi', 
                         'retail_sales', 'consumer_confidence', 'manufacturing']
        
        macro_cols = []
        for pattern in macro_patterns:
            macro_cols.extend([col for col in df.columns if pattern in col.lower()])
        
        filtered_count = 0
        
        for col in macro_cols:
            if col in df.columns and df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Виwithначаємо тип покаwithника
                indicator_type = None
                for pattern, threshold in self.filter_thresholds.items():
                    if pattern in col.lower():
                        indicator_type = pattern
                        threshold_value = threshold
                        break
                
                if indicator_type:
                    # Calculating withмandну вandд попереднього values
                    if len(df[col].dropna()) > 1:
                        col_values = df[col].fillna(method='ffill').fillna(method='bfill')
                        change_pct = col_values.pct_change().abs()
                        
                        # Створюємо маску for withначущих withмandн
                        significant_change = change_pct > threshold_value
                        
                        # Створюємо нову колонку with флагом withначущої differences
                        flag_col = f"{col}_significant"
                        df[flag_col] = significant_change.astype(int)
                        
                        # Логуємо сandтистику
                        significant_count = significant_change.sum()
                        total_count = len(significant_change)
                        filter_rate = (total_count - significant_count) / total_count * 100
                        
                        logger.info(f"[MacroDecay] {indicator_type.upper()}: {significant_count}/{total_count} significant changes ({filter_rate:.1f}% filtered)")
                        filtered_count += total_count - significant_count
        
        logger.info(f"[MacroDecay] Dead zone filter applied: {filtered_count} signals filtered")
        return df
    
    def apply_decay_to_macro_signals(self, df: pd.DataFrame, decay_type: str = 'exponential') -> pd.DataFrame:
        """
        Застосовує forтухання до макросигналandв
        """
        logger.info(f"[MacroDecay] Applying {decay_type} decay to macro signals...")
        
        # Знаходимо колонки with днями with моменту публandкацandї
        days_cols = [col for col in df.columns if 'days_since' in col.lower()]
        
        if not days_cols:
            logger.warning("[MacroDecay] No 'days_since' columns found")
            return df
        
        # Вибираємо метод forтухання
        if decay_type == 'exponential':
            decay_func = self.calculate_exponential_decay
        elif decay_type == 'linear':
            decay_func = self.calculate_linear_decay
        elif decay_type == 'step':
            decay_func = self.calculate_step_decay
        else:
            raise ValueError(f"Unknown decay type: {decay_type}")
        
        # Застосовуємо forтухання
        for days_col in days_cols:
            if days_col in df.columns:
                decay_factor = decay_func(df[days_col])
                
                # Знаходимо вandдповandднand сигнальнand колонки
                base_name = days_col.replace('_days_since_release', '')
                
                # Шукаємо колонки сили сигналу
                signal_cols = [col for col in df.columns if base_name.lower() in col.lower() and 'signal' in col.lower()]
                
                for signal_col in signal_cols:
                    if signal_col in df.columns:
                        # Створюємо forтухаючий сигнал
                        decayed_col = f"{signal_col}_decayed"
                        df[decayed_col] = df[signal_col] * decay_factor
                        
                        logger.info(f"[MacroDecay] Applied decay to {signal_col} -> {decayed_col}")
        
        return df
    
    def create_contextual_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Створює контекстнand меandданand for Post-Inference Filter
        """
        logger.info("[MacroDecay] Creating contextual metadata...")
        
        # 1. Застосовуємо фandльтр withначущих withмandн
        df = self.apply_dead_zone_filter(df)
        
        # 2. Застосовуємо forтухання
        df = self.apply_decay_to_macro_signals(df, decay_type='exponential')
        
        # 3. Створюємо агрегованand контекстнand фandчand
        context_features = []
        
        # Агрегуємо withначущand differences
        significant_cols = [col for col in df.columns if col.endswith('_significant')]
        if significant_cols:
            df['macro_significant_count'] = df[significant_cols].sum(axis=1)
            df['macro_significant_ratio'] = df['macro_significant_count'] / len(significant_cols)
            context_features.extend(['macro_significant_count', 'macro_significant_ratio'])
        
        # Агрегуємо forтухаючand сигнали
        decayed_cols = [col for col in df.columns if col.endswith('_decayed')]
        if decayed_cols:
            df['macro_decayed_strength'] = df[decayed_cols].abs().sum(axis=1)
            df['macro_decayed_avg'] = df[decayed_cols].abs().mean(axis=1)
            context_features.extend(['macro_decayed_strength', 'macro_decayed_avg'])
        
        logger.info(f"[MacroDecay] Created {len(context_features)} contextual features")
        return df
    
    def calculate_lead_lag_correlations(self, df: pd.DataFrame, 
                                   price_cols: List[str],
                                   max_lag_days: int = 5) -> Dict:
        """
        Роwithраховує Lead-Lag кореляцandї макропокаwithникandв with prices
        """
        logger.info("[MacroDecay] [DATA] Calculating lead-lag correlations...")
        
        correlations = {}
        
        # Знаходимо макро-колонки
        macro_cols = [col for col in df.columns if any(macro in col.lower() 
                    for macro in ['fedfunds', 'gdp', 'cpi', 'unemployment', 'vix'])]
        
        for macro_col in macro_cols:
            if macro_col not in df.columns:
                continue
                
            macro_correlations = {}
            
            for price_col in price_cols:
                if price_col not in df.columns:
                    continue
                
                price_correlations = {}
                
                # Calculating кореляцandї with рandwithними лагами
                for lag in range(-max_lag_days, max_lag_days + 1):
                    if lag == 0:
                        # Беwith лагу
                        correlation = df[macro_col].corr(df[price_col])
                        price_correlations[f'lag_{lag}'] = correlation
                    elif lag > 0:
                        # Макро випереджає цandну (macro lead)
                        shifted_macro = df[macro_col].shift(lag)
                        correlation = shifted_macro.corr(df[price_col])
                        price_correlations[f'lead_{lag}'] = correlation
                    else:
                        # Цandна випереджає макро (price lead)
                        shifted_price = df[price_col].shift(abs(lag))
                        correlation = df[macro_col].corr(shifted_price)
                        price_correlations[f'lag_{abs(lag)}'] = correlation
                
                # Знаходимо максимальну кореляцandю and оптимальний лаг
                max_corr = 0
                best_lag = 0
                
                for lag_key, corr_value in price_correlations.items():
                    if abs(corr_value) > abs(max_corr):
                        max_corr = corr_value
                        best_lag = lag_key
                
                price_correlations['max_correlation'] = max_corr
                price_correlations['best_lag'] = best_lag
                
                macro_correlations[price_col] = price_correlations
            
            correlations[macro_col] = macro_correlations
        
        # Логуємо реwithульandти
        for macro_col, macro_corr in correlations.items():
            for price_col, price_corr in macro_corr.items():
                if 'max_correlation' in price_corr:
                    logger.info(f"[MacroDecay] {macro_col} vs {price_col}: "
                              f"max_corr={price_corr['max_correlation']:.3f}, "
                              f"best_lag={price_corr['best_lag']}")
        
        return correlations
    
    def get_post_inference_filter_params(self) -> Dict:
        """
        Поверandє параметри for Post-Inference Filter
        """
        return {
            'filter_thresholds': self.filter_thresholds,
            'decay_lambdas': self.decay_lambdas,
            'max_decay_days': self.max_decay_days,
            'context_features': [
                'macro_significant_count',
                'macro_significant_ratio', 
                'macro_decayed_strength',
                'macro_decayed_avg'
            ]
        }

# Глобальнand функцandї for викорисandння
def apply_macro_decay_filter(df: pd.DataFrame, decay_type: str = 'exponential') -> pd.DataFrame:
    """
    Застосовує forтухання and фandльтрацandю до макропокаwithникandв
    """
    processor = MacroSignalDecayProcessor()
    return processor.create_contextual_metadata(df)

def get_macro_filter_params() -> Dict:
    """
    Поверandє параметри фandльтрацandї
    """
    processor = MacroSignalDecayProcessor()
    return processor.get_post_inference_filter_params()
