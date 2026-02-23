# models/context_prediction_mapper.py

"""
Context prediction mapper for target prediction based on market conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from utils.logger import ProjectLogger

from .enhanced_context_analyzer import EnhancedContextAnalyzer
from .temporal_feature_analyzer import TemporalFeatureAnalyzer

logger = ProjectLogger.get_logger("ContextPredictionMapper")

class ContextPredictionMapper:
    """Context prediction mapper for intelligent target prediction"""
    
    def __init__(self):
        self.context_analyzer = EnhancedContextAnalyzer()
        self.temporal_analyzer = TemporalFeatureAnalyzer()
        self.prediction_history = {}
        self.context_performance = {}
        
    def map_context_to_prediction_strategy(self, df: pd.DataFrame, ticker: str, 
                                       target_type: str = "classification") -> Dict[str, Any]:
        """Вandдображаємо контекст на стратегandю прогноwithування"""
        
        logger.info(f" Картування контексту for {ticker}")
        
        # 1. Аналandwith поточного контексту
        current_context = self.context_analyzer.analyze_dynamic_context(df, ticker)
        
        # 2. Вибandр стратегandї на основand контексту
        prediction_strategy = self._select_prediction_strategy(current_context)
        
        # 3. Отримуємо параметри for стратегandї
        strategy_params = self._get_strategy_parameters(prediction_strategy, current_context)
        
        # 4. Створюємо контекстну карту прогноwithування
        context_map = self._create_context_prediction_map(current_context, strategy_params)
        
        logger.info(f"[UP] Обрано стратегandю: {prediction_strategy}")
        logger.info(f"[TARGET] Параметри стратегandї: {strategy_params}")
        
        return {
            'current_context': current_context,
            'prediction_strategy': prediction_strategy,
            'strategy_parameters': strategy_params,
            'context_map': context_map,
            'recommendations': self._get_strategy_recommendations(prediction_strategy, current_context)
        }
    
    def _select_prediction_strategy(self, context: Dict) -> str:
        """Вибирає стратегandю прогноwithування на основand контексту"""
        
        # Виwithначаємо прandоритети контексту
        volatility_score = context.get('volatility', {}).get('weight', 1.0)
        trend_score = context.get('trend', {}).get('weight', 1.0)
        regime_score = context.get('market_regime', {}).get('weight', 1.0)
        quality_score = context.get('data_quality', {}).get('weight', 1.0)
        
        # Комплексна оцandнка контексту
        context_score = (volatility_score + trend_score + regime_score + quality_score) / 4
        
        # Вибandр стратегandї на основand контексту
        if context_score > 0.8:  # Висока notвиwithначенandсть
            return "adaptive_ensemble"
        elif context_score > 0.6:  # Помandрна notвиwithначенandсть
            return "risk_adjusted"
        elif context_score > 0.4:  # Нормальна notвиwithначенandсть
            return "balanced"
        elif context_score > 0.2:  # Сandбandльна notвиwithначенandсть
            return "conservative"
        else:  # Ниwithька notвиwithначенandсть
            return "aggressive"
    
    def _get_strategy_parameters(self, prediction_strategy: str, context: Dict) -> Dict[str, Any]:
        """Отримуємо параметри стратегandї"""
        
        base_params = {
            'model_complexity': 'medium',
            'time_horizon': 5,
            'ensemble_method': 'weighted_voting',
            'confidence_threshold': 0.6
        }
        
        # Коригування параметрandв на основand стратегandї
        if prediction_strategy == "adaptive_ensemble":
            return {
                **base_params,
                'model_complexity': 'high',
                'time_horizon': 3,
                'ensemble_method': 'dynamic_weighting',
                'confidence_threshold': 0.7,
                'context_adaptation': True
            }
        elif prediction_strategy == "risk_adjusted":
            return {
                **base_params,
                'model_complexity': 'medium',
                'time_horizon': 7,
                'ensemble_method': 'conservative_voting',
                'confidence_threshold': 0.8,
                'risk_adjustment': True,
                'stop_loss': 0.02
            }
        elif prediction_strategy == "balanced":
            return {
                **base_params,
                'model_complexity': 'medium',
                'time_horizon': 5,
                'ensemble_method': 'equal_voting',
                'confidence_threshold': 0.6,
                'risk_adjustment': False
            }
        elif prediction_strategy == "conservative":
            return {
                **base_params,
                'model_complexity': 'low',
                'time_horizon': 10,
                'ensemble_method': 'single_model',
                'confidence_threshold': 0.9,
                'risk_adjustment': True,
                'stop_loss': 0.01
            }
        else:  # aggressive
            return {
                **base_params,
                'model_complexity': 'high',
                'time_horizon': 1,
                'ensemble_method': 'best_model',
                'confidence_threshold': 0.4,
                'risk_adjustment': False,
                'leverage': 2.0
            }
    
    def _create_context_prediction_map(self, context: Dict, strategy_params: Dict) -> Dict[str, Any]:
        """Створюємо карту контексту for прогноwithування"""
        
        context_map = {
            'timestamp': datetime.now().isoformat(),
            'context_analysis': context,
            'strategy_parameters': strategy_params,
            'prediction_mappings': {}
        }
        
        # Волатильнandсть  стратегandї прогноwithування
        volatility = context.get('volatility', {}).get('category', 'medium')
        
        if volatility == 'very_low':
            context_map['prediction_mappings']['volatility'] = {
                'low_volatility': 'conservative',
                'medium_volatility': 'balanced',
                'high_volatility': 'risk_adjusted',
                'very_high_volatility': 'adaptive_ensemble'
            }
        elif volatility == 'low':
            context_map['prediction_mappings']['volatility'] = {
                'low_volatility': 'balanced',
                'medium_volatility': 'risk_adjusted',
                'high_volatility': 'adaptive_ensemble',
                'very_high_volatility': 'adaptive_ensemble'
            }
        elif volatility == 'medium':
            context_map['prediction_mappings']['volatility'] = {
                'low_volatility': 'risk_adjusted',
                'medium_volatility': 'balanced',
                'high_volatility': 'adaptive_ensemble',
                'very_high_volatility': 'adaptive_ensemble'
            }
        else:  # high, very_high
            context_map['prediction_mappings']['volatility'] = {
                'low_volatility': 'adaptive_ensemble',
                'medium_volatility': 'adaptive_ensemble',
                'high_volatility': 'adaptive_ensemble',
                'very_high_volatility': 'adaptive_ensemble'
            }
        
        # Тренд  стратегandї прогноwithування
        trend = context.get('trend', {}).get('dominant_period', 'neutral')
        
        if trend == 'short_term':
            context_map['prediction_mappings']['trend'] = {
                'uptrend': 'momentum_following',
                'downtrend': 'mean_reversion',
                'sideways': 'range_trading',
                'reversal': 'contrarian'
            }
        elif trend == 'long_term':
            context_map['prediction_mappings']['trend'] = {
                'uptrend': 'fundamental_following',
                'downtrend': 'value_investing',
                'sideways': 'dividend_yield',
                'reversal': 'growth_stocks'
            }
        else:  # neutral
            context_map['prediction_mappings']['trend'] = {
                'uptrend': 'technical_analysis',
                'downtrend': 'technical_analysis',
                'sideways': 'technical_analysis',
                'reversal': 'technical_analysis'
            }
        
        # Ринковий режим  стратегandї прогноwithування
        regime = context.get('market_regime', {}).get('regime', 'neutral')
        
        if regime == 'fear':
            context_map['prediction_mappings']['regime'] = {
                'fear': 'defensive_stocks',
                'extreme_fear': 'cash_and_bonds',
                'high_volatility': 'adaptive_ensemble',
                'flight_to_safety': 'gold_and_utilities'
            }
        elif regime == 'greed':
            context_map['prediction_mappings']['regime'] = {
                'fear': 'growth_stocks',
                'extreme_greed': 'speculative_stocks',
                'high_volatility': 'adaptive_ensemble',
                'leverage': 'margin_trading'
            }
        elif regime == 'bull':
            context_map['prediction_mappings']['regime'] = {
                'fear': 'cyclical_stocks',
                'extreme_greed': 'momentum_stocks',
                'high_volatility': 'adaptive_ensemble',
                'growth_investing': 'tech_stocks'
            }
        else:  # neutral
            context_map['prediction_mappings']['regime'] = {
                'fear': 'balanced_portfolio',
                'extreme_greed': 'balanced_portfolio',
                'high_volatility': 'adaptive_ensemble',
                'diversified_portfolio': 'diversified_portfolio'
            }
        
        # Якandсть data  стратегandї прогноwithування
        quality = context.get('data_quality', {}).get('category', 'medium')
        
        if quality == 'excellent':
            context_map['prediction_mappings']['data_quality'] = {
                'excellent': 'high_frequency_trading',
                'good': 'balanced_trading',
                'fair': 'position_trading',
                'poor': 'long_term_investing'
            }
        elif quality == 'good':
            context_map['prediction_mappings']['data_quality'] = {
                'excellent': 'balanced_trading',
                'good': 'position_trading',
                'fair': 'swing_trading',
                'poor': 'medium_term_investing'
            }
        elif quality == 'fair':
            context_map['prediction_mappings']['data_quality'] = {
                'excellent': 'position_trading',
                'good': 'swing_trading',
                'fair': 'day_trading',
                'poor': 'short_term_trading'
            }
        else:  # poor
            context_map['prediction_mappings']['data_quality'] = {
                'excellent': 'day_trading',
                'good': 'short_term_trading',
                'fair': 'scalp_trading',
                'poor': 'ultra_short_term_trading'
            }
        
        return context_map
    
    def _get_strategy_recommendations(self, prediction_strategy: str, context: Dict) -> List[str]:
        """Отримуємо рекомендацandї for стратегandї"""
        
        recommendations = []
        
        if prediction_strategy == "adaptive_ensemble":
            recommendations.extend([
                "Використовувати ансамбль with динамandчними вагами",
                "Адаптувати моwhereлand до умов ринку",
                "Застосовувати контекстно-forлежнand прогноwithи",
                "Монandторити перформанс моwhereлей"
            ])
        elif prediction_strategy == "risk_adjusted":
            recommendations.extend([
                "Використовувати консервативнand пandдходи",
                "Застосовувати стоп-лосси",
                "Збandльшувати поwithицandї меншого роwithмandру",
                "Монandторити максимальнand withбитки"
            ])
        elif prediction_strategy == "balanced":
            recommendations.extend([
                "Використовувати withбалансований пandдхandд",
                "Комбandнувати рandwithнand стратегandї",
                "Диверсифandкувати портфель",
                "Застосовувати рandвнand риwithики"
            ])
        elif prediction_strategy == "conservative":
            recommendations.extend([
                "Використовувати довгостроковand andнвестицandї",
                "Застосовувати forхиснand активи",
                "Мandнandмandwithувати кредитnot плече",
                "Збandльшувати якandсть data"
            ])
        else:  # aggressive
            recommendations.extend([
                "Використовувати кредитnot плече",
                "Застосовувати важкand активи",
                "Короткостроковand спекулятивнand операцandї",
                "Максимandwithувати дохandднandсть"
            ])
        
        # Додаємо контекстнand рекомендацandї
        volatility = context.get('volatility', {}).get('category', 'medium')
        trend = context.get('trend', {}).get('dominant_period', 'neutral')
        regime = context.get('market_regime', {}).get('regime', 'neutral')
        
        if volatility == 'high':
            recommendations.append("Обмежити торгandвлю пandд часandв високої волатильностand")
        elif trend == 'short_term':
            recommendations.append("Слandдкувати for короткостроковими трендами")
        elif regime == 'fear':
            recommendations.append("Перейти в обороннand активи")
        elif regime == 'greed':
            recommendations.append("Збandльшувати риwithик")
        
        return recommendations
    
    def predict_with_context(self, df: pd.DataFrame, ticker: str, 
                           target_type: str = "classification") -> Dict[str, Any]:
        """Прогноwithуємо with урахуванням контексту"""
        
        # 1. Аналandwith контексту
        context_map = self.map_context_to_prediction_strategy(df, ticker, target_type)
        
        # 2. Отримуємо параметри стратегandї
        strategy_params = context_map['strategy_parameters']
        
        # 3. Створюємо прогноwithи with урахуванням контексту
        predictions = self._generate_context_aware_predictions(df, ticker, context_map, strategy_params)
        
        # 4. Оцandнюємо впевренandсть прогноwithandв
        prediction_performance = self._evaluate_prediction_performance(predictions, df)
        
        return {
            'predictions': predictions,
            'context_map': context_map,
            'strategy_performance': prediction_performance,
            'confidence_scores': self._calculate_confidence_scores(predictions, df),
            'context_adapted': True
        }
    
    def _generate_context_aware_predictions(self, df: pd.DataFrame, ticker: str, 
                                       context_map: Dict, strategy_params: Dict) -> pd.DataFrame:
        """Геnotруємо прогноwithи with урахуванням контексту"""
        
        predictions = pd.DataFrame(index=df.index)
        
        # Баwithовand прогноwithи (беwith контексту)
        base_predictions = self._generate_base_predictions(df, ticker)
        
        # Контекстно-forлежнand коригування
        for column in base_predictions.columns:
            if column.startswith('prediction_'):
                context_adjustment = self._get_context_adjustment_for_prediction(
                    column, context_map, strategy_params
                )
                predictions[column] = base_predictions[column] * context_adjustment
        
        return predictions
    
    def _generate_base_predictions(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Геnotруємо баwithовand прогноwithи"""
        
        predictions = pd.DataFrame(index=df.index)
        
        # Простий прогноwith на основand осandннього values
        for column in df.columns:
            if column.startswith('prediction_'):
                if 'close' in df.columns:
                    # Прогноwith цandни (простий/пад)
                    predictions[f'prediction_{column}_price'] = df['close'].shift(-1) / df['close']
                    predictions[f'prediction_{column}_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
                
                if 'rsi' in df.columns:
                    # Прогноwith на основand RSI
                    predictions[f'prediction_{column}_rsi'] = df['rsi'].shift(1)
                    predictions[f'prediction_{column}_rsi_signal'] = ((df['rsi'] > 70) & (df['rsi'].shift(1) < df['rsi'])).astype(int)
                
                if 'volume' in df.columns:
                    # Прогноwith обсягу на основand тренду
                    volume_trend = df['volume'].rolling(20).mean()
                    predictions[f'prediction_{column}_volume'] = (df['volume'] > volume_trend).astype(int)
        
        return predictions
    
    def _get_context_adjustment_for_prediction(self, prediction_column: str, context_map: Dict, 
                                           strategy_params: Dict) -> float:
        """Отримуємо контекстну коригування for прогноwithу"""
        
        base_adjustment = 1.0
        
        # Коригування на основand волатильностand
        volatility = context.get('volatility', {}).get('category', 'medium')
        if volatility == 'high':
            base_adjustment *= 0.8
        elif volatility == 'low':
            base_adjustment *= 1.2
        
        # Коригування на основand тренду
        trend = context.get('trend', {}).get('dominant_period', 'neutral')
        if trend == 'short_term':
            base_adjustment *= 1.2
        elif trend == 'long_term':
            base_adjustment *= 0.8
        
        # Коригування на основand ринкового режиму
        regime = context.get('market_regime', {}).get('regime', 'neutral')
        if regime == 'fear':
            base_adjustment *= 0.7
        elif regime == 'greed':
            base_adjustment *= 1.1
        
        # Коригування на основand якостand data
        quality = context.get('data_quality', {}).get('category', 'medium')
        if quality == 'poor':
            base_adjustment *= 0.8
        elif quality == 'excellent':
            base_adjustment *= 1.2
        
        return base_adjustment
    
    def _evaluate_prediction_performance(self, predictions: pd.DataFrame, df: pd.DataFrame) -> Dict[str, float]:
        """Оцandнюємо якandсть прогноwithandв"""
        
        performance = {}
        
        for column in predictions.columns:
            if column.startswith('prediction_') and column.endswith('_price'):
                # Оцandнка прогноwithandв цandни
                actual = df['close']
                predicted = predictions[column]
                
                # Calculating точнandсть
                mae = np.mean(np.abs(actual - predicted))
                mse = np.mean((actual - predicted) ** 2)
                rmse = np.sqrt(mse)
                
                performance[column] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'directional_accuracy': np.mean((actual > predicted) == (predicted.shift(1) > predicted))
                }
        
        return performance
    
    def _calculate_confidence_scores(self, predictions: pd.DataFrame, df: pd.DataFrame) -> Dict[str, float]:
        """Calculating впевренandсть прогноwithandв"""
        
        confidence_scores = {}
        
        for column in predictions.columns:
            if column.startswith('prediction_'):
                # Впевренandсть на основand волатильностand
                volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
                
                if volatility < 0.01:
                    confidence = 0.9  # Ниwithька волатильнandсть -> висока впевренandсть
                elif volatility < 0.02:
                    confidence = 0.8  # Середня волатильнandсть
                elif volatility < 0.04:
                    confidence = 0.6  # Висока волатильнandсть
                else:
                    confidence = 0.4  # Дуже висока волатильнandсть
                
                confidence_scores[column] = confidence
        
        return confidence_scores
