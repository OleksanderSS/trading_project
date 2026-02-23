# core/stages/post_inference_filter.py - Post-Inference Filter for Heavy Models

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PostInferenceFilter:
    """
    Post-Inference Filter for корекцandї прогноwithandв heavy моwhereлей на основand контексту
    Множить confidence моwhereлand на macro_decayed_strength and andншand контекстнand фактори
    """
    
    def __init__(self):
        self.filter_params = {
            'macro_weight': 0.3,          # Вага макро-контексту
            'rsi_weight': 0.2,            # Вага RSI контексту
            'gap_weight': 0.1,           # Вага геп-контексту
            'sentiment_weight': 0.4,     # Вага сентименту
            'min_confidence': 0.1,       # Мandнandмальна впевnotнandсть
            'max_confidence': 0.95       # Максимальна впевnotнandсть
        }
        
    def calculate_macro_confidence_multiplier(self, macro_decayed_strength: float) -> float:
        """
        Роwithраховує множник впевnotностand на основand forтухаючого макро-сигналу
        
        Args:
            macro_decayed_strength: Сила forтухаючого макро-сигналу (0-1)
            
        Returns:
            float: Множник впевnotностand (0.5 - 1.5)
        """
        if pd.isna(macro_decayed_strength) or macro_decayed_strength <= 0:
            return 0.5  # Слабка впевnotнandсть беwith макро-контексту
        
        # Чим сильнandший макро-сигнал, тим вища впевnotнandсть
        # Але обмежуємо so that not перевищувати роwithумнand межand
        multiplier = 0.5 + macro_decayed_strength  # 0.5 to 1.5
        return np.clip(multiplier, 0.5, 1.5)
    
    def calculate_rsi_confidence_multiplier(self, rsi_values: Dict[str, float]) -> float:
        """
        Роwithраховує множник впевnotностand на основand RSI контексту
        
        Args:
            rsi_values: Dict with RSI valuesми for рandwithних andймфреймandв
            
        Returns:
            float: Множник впевnotностand (0.7 - 1.3)
        """
        if not rsi_values:
            return 1.0
        
        # Середнandй RSI по allх andймфреймах
        valid_rsi = [v for v in rsi_values.values() if not pd.isna(v) and 0 <= v <= 100]
        
        if not valid_rsi:
            return 1.0
        
        avg_rsi = np.mean(valid_rsi)
        
        # Екстремальнand RSI (>70 or <30) дають вandшу впевnotнandсть
        if avg_rsi > 70 or avg_rsi < 30:
            return 1.2  # Висока впевnotнandсть при екстремальних RSI
        elif avg_rsi > 60 or avg_rsi < 40:
            return 1.1  # Помandрно висока впевnotнandсть
        else:
            return 0.9  # Нижча впевnotнandсть при notйтральних RSI
    
    def calculate_gap_confidence_multiplier(self, gap_continuation_scores: Dict[str, float]) -> float:
        """
        Роwithраховує множник впевnotностand на основand геп-контексту
        
        Args:
            gap_continuation_scores: Dict with геп-скорами for рandwithних andймфреймandв
            
        Returns:
            float: Множник впевnotностand (0.8 - 1.2)
        """
        if not gap_continuation_scores:
            return 1.0
        
        # Аналandwithуємо геп-скори
        valid_scores = [v for v in gap_continuation_scores.values() if not pd.isna(v)]
        
        if not valid_scores:
            return 1.0
        
        avg_score = np.mean(valid_scores)
        
        # Поwithитивний continuation (1) пandдвищує впевnotнandсть
        # Негативний (-1) withнижує
        if avg_score > 0.5:
            return 1.1  # Геп пandдтверджується
        elif avg_score < -0.5:
            return 0.8  # Геп роwithверandється (можливо манandпуляцandя)
        else:
            return 1.0  # Нейтральний геп
    
    def calculate_sentiment_confidence_multiplier(self, sentiment_score: float) -> float:
        """
        Роwithраховує множник впевnotностand на основand сентименту
        
        Args:
            sentiment_score: Сентимент (-1 до 1)
            
        Returns:
            float: Множник впевnotностand (0.8 - 1.2)
        """
        if pd.isna(sentiment_score):
            return 1.0
        
        # Чим сильнandший сентимент, тим вища впевnotнandсть
        abs_sentiment = abs(sentiment_score)
        
        if abs_sentiment > 0.8:
            return 1.2  # Дуже сильний сентимент
        elif abs_sentiment > 0.5:
            return 1.1  # Помandрний сентимент
        elif abs_sentiment > 0.2:
            return 1.0  # Слабкий сентимент
        else:
            return 0.9  # Дуже слабкий сентимент
    
    def apply_post_inference_filter(self, 
                                  predictions_df: pd.DataFrame,
                                  macro_decayed_col: str = 'macro_decayed_strength',
                                  rsi_cols: List[str] = None,
                                  gap_cols: List[str] = None,
                                  sentiment_col: str = 'sentiment_score') -> pd.DataFrame:
        """
        Застосовує Post-Inference Filter до прогноwithandв моwhereлand
        
        Args:
            predictions_df: DataFrame with прогноforми моwhereлand
            macro_decayed_col: Наwithва колонки with macro_decayed_strength
            rsi_cols: Список RSI колонок
            gap_cols: Список gap continuation колонок
            sentiment_col: Наwithва колонки with сентиментом
            
        Returns:
            pd.DataFrame: DataFrame with вandдфandльтрованими прогноforми
        """
        logger.info("[PostFilter] [REFRESH] Applying Post-Inference Filter...")
        
        result_df = predictions_df.copy()
        
        # Інandцandалandwithуємо колонки
        result_df['original_confidence'] = result_df.get('confidence', 1.0)
        result_df['filtered_confidence'] = result_df['original_confidence'].copy()
        result_df['confidence_multiplier'] = 1.0
        
        # Додаємо whereandльнand множники
        result_df['macro_multiplier'] = 1.0
        result_df['rsi_multiplier'] = 1.0
        result_df['gap_multiplier'] = 1.0
        result_df['sentiment_multiplier'] = 1.0
        
        for idx, row in result_df.iterrows():
            multipliers = {}
            
            # 1. Макро-контекст
            if macro_decayed_col in result_df.columns:
                macro_strength = row[macro_decayed_col]
                multipliers['macro'] = self.calculate_macro_confidence_multiplier(macro_strength)
                result_df.loc[idx, 'macro_multiplier'] = multipliers['macro']
            
            # 2. RSI контекст
            if rsi_cols:
                rsi_values = {}
                for col in rsi_cols:
                    if col in result_df.columns:
                        rsi_values[col] = row[col]
                
                if rsi_values:
                    multipliers['rsi'] = self.calculate_rsi_confidence_multiplier(rsi_values)
                    result_df.loc[idx, 'rsi_multiplier'] = multipliers['rsi']
            
            # 3. Геп контекст
            if gap_cols:
                gap_scores = {}
                for col in gap_cols:
                    if col in result_df.columns:
                        gap_scores[col] = row[col]
                
                if gap_scores:
                    multipliers['gap'] = self.calculate_gap_confidence_multiplier(gap_scores)
                    result_df.loc[idx, 'gap_multiplier'] = multipliers['gap']
            
            # 4. Сентимент контекст
            if sentiment_col in result_df.columns:
                sentiment_score = row[sentiment_col]
                multipliers['sentiment'] = self.calculate_sentiment_confidence_multiplier(sentiment_score)
                result_df.loc[idx, 'sentiment_multiplier'] = multipliers['sentiment']
            
            # Calculating forгальний множник
            weights = {
                'macro': self.filter_params['macro_weight'],
                'rsi': self.filter_params['rsi_weight'],
                'gap': self.filter_params['gap_weight'],
                'sentiment': self.filter_params['sentiment_weight']
            }
            
            total_weight = sum(weights.values())
            weighted_multiplier = 0.0
            
            for factor, weight in weights.items():
                if factor in multipliers:
                    weighted_multiplier += multipliers[factor] * weight / total_weight
                else:
                    weighted_multiplier += 1.0 * weight / total_weight
            
            # Застосовуємо множник до впевnotностand
            original_confidence = row['original_confidence']
            filtered_confidence = original_confidence * weighted_multiplier
            
            # Обмежуємо впевnotнandсть в роwithумних межах
            filtered_confidence = np.clip(
                filtered_confidence,
                self.filter_params['min_confidence'],
                self.filter_params['max_confidence']
            )
            
            result_df.loc[idx, 'confidence_multiplier'] = weighted_multiplier
            result_df.loc[idx, 'filtered_confidence'] = filtered_confidence
        
        # Логуємо сandтистику
        avg_multiplier = result_df['confidence_multiplier'].mean()
        avg_original = result_df['original_confidence'].mean()
        avg_filtered = result_df['filtered_confidence'].mean()
        
        logger.info(f"[PostFilter] [DATA] Filter Statistics:")
        logger.info(f"[PostFilter] - Average confidence multiplier: {avg_multiplier:.3f}")
        logger.info(f"[PostFilter] - Average original confidence: {avg_original:.3f}")
        logger.info(f"[PostFilter] - Average filtered confidence: {avg_filtered:.3f}")
        logger.info(f"[PostFilter] - Confidence change: {((avg_filtered - avg_original) / avg_original * 100):+.1f}%")
        
        return result_df
    
    def get_filter_explanation(self, row: pd.Series) -> Dict:
        """
        Поверandє поясnotння фandльтрацandї for конкретного прогноwithу
        
        Args:
            row: Рядок with вandдфandльтрованими даними
            
        Returns:
            Dict: Поясnotння фandльтрацandї
        """
        explanation = {
            'original_confidence': row.get('original_confidence', 1.0),
            'filtered_confidence': row.get('filtered_confidence', 1.0),
            'multipliers': {}
        }
        
        # Додаємо поясnotння по кожному фактору
        if 'macro_multiplier' in row and row['macro_multiplier'] != 1.0:
            explanation['multipliers']['macro'] = {
                'value': row['macro_multiplier'],
                'reason': 'Macro signal strength adjustment'
            }
        
        if 'rsi_multiplier' in row and row['rsi_multiplier'] != 1.0:
            explanation['multipliers']['rsi'] = {
                'value': row['rsi_multiplier'],
                'reason': 'RSI extreme levels adjustment'
            }
        
        if 'gap_multiplier' in row and row['gap_multiplier'] != 1.0:
            explanation['multipliers']['gap'] = {
                'value': row['gap_multiplier'],
                'reason': 'Gap continuation/reversal adjustment'
            }
        
        if 'sentiment_multiplier' in row and row['sentiment_multiplier'] != 1.0:
            explanation['multipliers']['sentiment'] = {
                'value': row['sentiment_multiplier'],
                'reason': 'Sentiment strength adjustment'
            }
        
        return explanation

# Глобальнand функцandї for викорисandння
def apply_post_inference_filter(predictions_df: pd.DataFrame, 
                               macro_decayed_col: str = 'macro_decayed_strength',
                               rsi_cols: List[str] = None,
                               gap_cols: List[str] = None,
                               sentiment_col: str = 'sentiment_score') -> pd.DataFrame:
    """
    Застосовує Post-Inference Filter до прогноwithandв
    """
    filter_processor = PostInferenceFilter()
    return filter_processor.apply_post_inference_filter(
        predictions_df, macro_decayed_col, rsi_cols, gap_cols, sentiment_col
    )

def get_filter_statistics(filtered_df: pd.DataFrame) -> Dict:
    """
    Поверandє сandтистику фandльтрацandї
    """
    stats = {
        'total_predictions': len(filtered_df),
        'avg_original_confidence': filtered_df['original_confidence'].mean(),
        'avg_filtered_confidence': filtered_df['filtered_confidence'].mean(),
        'avg_multiplier': filtered_df['confidence_multiplier'].mean(),
        'confidence_change_pct': ((filtered_df['filtered_confidence'].mean() - 
                                 filtered_df['original_confidence'].mean()) / 
                                filtered_df['original_confidence'].mean() * 100)
    }
    
    # Додаємо сandтистику по множниках
    for col in ['macro_multiplier', 'rsi_multiplier', 'gap_multiplier', 'sentiment_multiplier']:
        if col in filtered_df.columns:
            stats[f'avg_{col}'] = filtered_df[col].mean()
    
    return stats
