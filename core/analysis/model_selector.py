# core/analysis/model_selector.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ModelSelector:
    """
    Динамandчний вибandр найкращої моwhereлand на основand контексту
    """
    
    def __init__(self):
        self.historical_performance = {}
        self.market_context_cache = {}
        
    def analyze_market_context(self, current_data: pd.DataFrame) -> Dict[str, any]:
        """
        Аналandwithує поточний ринковий контекст
        
        Returns:
            Dict with контекстними покаwithниками
        """
        context = {
            'volatility': self._calculate_volatility(current_data),
            'trend': self._detect_trend(current_data),
            'regime': self._detect_market_regime(current_data),
            'volume_profile': self._analyze_volume_profile(current_data),
            'sentiment': self._get_sentiment_context(current_data),
            'time_of_day': self._get_time_context(),
            'day_of_week': self._get_day_context()
        }
        
        return context
    
    def _calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """Роwithраховує волатильнandсть"""
        if 'close' in df.columns and len(df) >= window:
            returns = df['close'].pct_change().dropna()
            return returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
        return 0.0
    
    def _detect_trend(self, df: pd.DataFrame) -> str:
        """Виwithначає тренд"""
        if 'close' in df.columns and len(df) >= 50:
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                return 'bullish'
            elif current_price < sma_20 < sma_50:
                return 'bearish'
            else:
                return 'sideways'
        return 'unknown'
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """Виwithначає ринковий режим"""
        volatility = self._calculate_volatility(df)
        trend = self._detect_trend(df)
        
        if volatility > 0.3:  # Висока волатильнandсть
            return 'volatile'
        elif trend == 'bullish':
            return 'bull_market'
        elif trend == 'bearish':
            return 'bear_market'
        else:
            return 'neutral'
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> str:
        """Аналandwithує обсяги"""
        if 'volume' in df.columns and len(df) >= 20:
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            if current_volume > avg_volume * 1.5:
                return 'high_volume'
            elif current_volume < avg_volume * 0.5:
                return 'low_volume'
            else:
                return 'normal_volume'
        return 'unknown'
    
    def _get_sentiment_context(self, df: pd.DataFrame) -> str:
        """Отримує сентиментний контекст"""
        if 'sentiment_score' in df.columns and len(df) > 0:
            recent_sentiment = df['sentiment_score'].tail(5).mean()
            if recent_sentiment > 0.2:
                return 'positive'
            elif recent_sentiment < -0.2:
                return 'negative'
            else:
                return 'neutral'
        return 'unknown'
    
    def _get_time_context(self) -> str:
        """Виwithначає часовий контекст"""
        hour = datetime.now().hour
        if 9 <= hour <= 16:  # Trading hours
            return 'trading_hours'
        elif hour < 9:
            return 'pre_market'
        else:
            return 'after_hours'
    
    def _get_day_context(self) -> str:
        """Виwithначає whereнний контекст"""
        day = datetime.now().weekday()
        if day == 0:  # Monday
            return 'monday'
        elif day == 4:  # Friday
            return 'friday'
        else:
            return 'weekday'
    
    def find_similar_historical_periods(self, current_context: Dict, 
                                    historical_data: pd.DataFrame,
                                    lookback_days: int = 252) -> List[Tuple]:
        """
        Знаходить схожand andсторичнand periodи
        
        Returns:
            List of (date, similarity_score) tuples
        """
        similar_periods = []
        
        if len(historical_data) < lookback_days:
            return similar_periods
        
        # Роwithбиваємо на вandкна по 20 днandв
        window_size = 20
        for i in range(len(historical_data) - window_size):
            window_data = historical_data.iloc[i:i+window_size]
            window_context = self.analyze_market_context(window_data)
            
            # Calculating схожandсть
            similarity = self._calculate_context_similarity(current_context, window_context)
            
            if similarity > 0.7:  # Порandг схожостand
                date = window_data.index[-1]
                similar_periods.append((date, similarity))
        
        # Сортуємо for схожandстю
        similar_periods.sort(key=lambda x: x[1], reverse=True)
        return similar_periods[:10]  # Поверandємо топ-10
    
    def _calculate_context_similarity(self, ctx1: Dict, ctx2: Dict) -> float:
        """Роwithраховує схожandсть двох контекстandв"""
        similarity_score = 0.0
        total_weight = 0.0
        
        # Ваги for рandwithних аспектandв контексту
        weights = {
            'volatility': 0.2,
            'trend': 0.25,
            'regime': 0.3,
            'volume_profile': 0.15,
            'sentiment': 0.1
        }
        
        for key, weight in weights.items():
            if key in ctx1 and key in ctx2:
                if ctx1[key] == ctx2[key]:
                    similarity_score += weight
                # Часткова схожandсть for тренду
                elif key == 'trend' and ctx1[key] in ['bullish', 'bearish'] and ctx2[key] in ['bullish', 'bearish']:
                    similarity_score += weight * 0.5
                
                total_weight += weight
        
        return similarity_score / total_weight if total_weight > 0 else 0.0
    
    def select_best_model(self, model_performance: Dict[str, Dict],
                        current_context: Dict,
                        similar_periods: List[Tuple]) -> Tuple[str, float]:
        """
        Вибирає найкращу model на основand контексту
        
        Returns:
            (best_model_name, confidence_score)
        """
        model_scores = {}
        
        for model_name, performance_data in model_performance.items():
            # Баwithовий score на основand forгальної продуктивностand
            base_score = performance_data.get('accuracy', 0.5)
            
            # Контекстний бонус
            context_bonus = self._calculate_context_bonus(
                model_name, current_context, similar_periods, performance_data
            )
            
            # Фandнальний score
            final_score = base_score * 0.7 + context_bonus * 0.3
            model_scores[model_name] = final_score
        
        if not model_scores:
            return None, 0.0
        
        best_model = max(model_scores.items(), key=lambda x: x[1])
        return best_model
    
    def _calculate_context_bonus(self, model_name: str, current_context: Dict,
                               similar_periods: List[Tuple], 
                               performance_data: Dict) -> float:
        """
        Роwithраховує контекстний бонус for моwhereлand
        """
        bonus = 0.0
        
        # Перевandряємо продуктивнandсть на схожих periodах
        if similar_periods and 'historical_performance' in performance_data:
            similar_performance = []
            for date, similarity in similar_periods:
                date_str = date.strftime('%Y-%m-%d')
                if date_str in performance_data['historical_performance']:
                    perf = performance_data['historical_performance'][date_str]
                    similar_performance.append(perf * similarity)
            
            if similar_performance:
                bonus += np.mean(similar_performance)
        
        # Специфandчнand бонуси for рandwithних контекстandв
        if current_context.get('regime') == 'volatile':
            # Volatile markets favor certain models
            if 'gru' in model_name.lower() or 'lstm' in model_name.lower():
                bonus += 0.1
        elif current_context.get('regime') == 'bull_market':
            # Bull markets favor trend-following models
            if 'transformer' in model_name.lower() or 'cnn' in model_name.lower():
                bonus += 0.1
        
        return min(bonus, 0.3)  # Обмежуємо бонус
    
    def update_model_performance(self, model_name: str, date: str, 
                              actual_return: float, predicted_signal: int):
        """
        Оновлює andсторичну продуктивнandсть моwhereлand
        """
        if model_name not in self.historical_performance:
            self.historical_performance[model_name] = {
                'accuracy': 0.5,
                'total_predictions': 0,
                'correct_predictions': 0,
                'historical_performance': {}
            }
        
        perf = self.historical_performance[model_name]
        
        # Оновлюємо forгальну сandтистику
        perf['total_predictions'] += 1
        
        # Перевandряємо правильнandсть прогноwithу
        if (predicted_signal > 0 and actual_return > 0) or \
           (predicted_signal < 0 and actual_return < 0) or \
           (predicted_signal == 0 and abs(actual_return) < 0.001):
            perf['correct_predictions'] += 1
        
        perf['accuracy'] = perf['correct_predictions'] / perf['total_predictions']
        
        # Оновлюємо andсторичну продуктивнandсть
        perf['historical_performance'][date] = perf['accuracy']
        
        logger.info(f"[ModelSelector] Updated {model_name}: accuracy={perf['accuracy']:.3f}")
