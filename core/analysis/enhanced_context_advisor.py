# Enhanced Context Advisor - Покращений контекстний радник

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger("EnhancedContextAdvisor")


class EnhancedContextAdvisor:
    """
    Покращений контекстний радник з правильною логікою рекомендацій
    """
    
    def __init__(self):
        self.logger = logging.getLogger("EnhancedContextAdvisor")
        
        # Пороги для визначення режимів ринку
        self.market_regime_thresholds = {
            'bull_market': 0.02,      # > 2% зростання
            'bear_market': -0.02,     # < -2% падіння
            'sideways': 0.005          # < 0.5% коливання
        }
        
        # Пороги волатильності
        self.volatility_thresholds = {
            'high': 0.03,      # > 3%
            'medium': 0.015,    # 1.5-3%
            'low': 0.01         # < 1%
        }
        
        # Рекомендації по моделях
        self.model_recommendations = {
            'bull_high_vol': ['LSTM', 'GRU', 'Transformer'],
            'bull_low_vol': ['RandomForest', 'XGBoost', 'Linear'],
            'bear_high_vol': ['LSTM', 'GRU', 'Ensemble'],
            'bear_low_vol': ['RandomForest', 'XGBoost'],
            'sideways': ['Linear', 'Ridge', 'ElasticNet']
        }
        
        # Рекомендації по фічах
        self.feature_priorities = {
            'bull_market': ['momentum', 'trend', 'volume'],
            'bear_market': ['volatility', 'support_resistance', 'sentiment'],
            'sideways': ['mean_reversion', 'bollinger', 'rsi']
        }
    
    def detect_market_regime(self, price_data: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """
        Визначає поточний режим ринку
        """
        if price_data.empty or len(price_data) < window:
            return {'regime': 'unknown', 'confidence': 0}
        
        # Розраховуємо зміни цін
        price_data_copy = price_data.copy()
        price_data_copy['returns'] = price_data_copy['close'].pct_change()
        price_data_copy['volatility'] = price_data_copy['returns'].rolling(window).std()
        
        # Останні значення
        recent_return = price_data_copy['returns'].iloc[-1]
        recent_vol = price_data_copy['volatility'].iloc[-1]
        avg_return = price_data_copy['returns'].iloc[-window:].mean()
        
        # Визначаємо режим
        if avg_return > self.market_regime_thresholds['bull_market']:
            regime = 'bull_market'
        elif avg_return < self.market_regime_thresholds['bear_market']:
            regime = 'bear_market'
        else:
            regime = 'sideways'
        
        # Визначаємо волатильність
        if recent_vol > self.volatility_thresholds['high']:
            volatility = 'high'
        elif recent_vol > self.volatility_thresholds['medium']:
            volatility = 'medium'
        else:
            volatility = 'low'
        
        # Розраховуємо впевненість
        confidence = min(abs(avg_return) / 0.01, 1.0)  # Макс 1.0
        
        return {
            'regime': regime,
            'volatility': volatility,
            'confidence': confidence,
            'recent_return': recent_return,
            'recent_volatility': recent_vol,
            'avg_return': avg_return
        }
    
    def analyze_model_performance(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Аналізує продуктивність моделей
        """
        if not model_results:
            return {'best_model': None, 'performance_ranking': []}
        
        performance_scores = []
        
        for model_name, results in model_results.items():
            accuracy = results.get('accuracy', 0)
            precision = results.get('precision', 0)
            recall = results.get('recall', 0)
            f1_score = results.get('f1_score', 0)
            
            # Комбінований score
            combined_score = (accuracy * 0.4 + precision * 0.2 + 
                           recall * 0.2 + f1_score * 0.2)
            
            performance_scores.append({
                'model': model_name,
                'combined_score': combined_score,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            })
        
        # Сортуємо за продуктивністю
        performance_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return {
            'best_model': performance_scores[0]['model'] if performance_scores else None,
            'performance_ranking': performance_scores,
            'top_3_models': [p['model'] for p in performance_scores[:3]]
        }
    
    def analyze_news_sentiment_context(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Аналізує контекст новин
        """
        if news_data.empty:
            return {'overall_sentiment': 0, 'sentiment_trend': 'neutral'}
        
        # Переконуємось що sentiment_score існує
        if 'sentiment_score' not in news_data.columns:
            return {'overall_sentiment': 0, 'sentiment_trend': 'neutral'}
        
        # Перетворюємо published_at в datetime
        if 'published_at' in news_data.columns:
            news_data['published_at'] = pd.to_datetime(news_data['published_at'])
        
        # Аналізуємо sentiment за часом
        recent_news = news_data[
            news_data['published_at'] > (datetime.now() - timedelta(hours=24))
        ]
        
        if recent_news.empty:
            return {'overall_sentiment': 0, 'sentiment_trend': 'neutral'}
        
        # Загальний sentiment
        overall_sentiment = recent_news['sentiment_score'].mean()
        
        # Тренд sentiment
        if len(recent_news) >= 2:
            recent_news_sorted = recent_news.sort_values('published_at')
            first_half = recent_news_sorted.iloc[:len(recent_news)//2]
            second_half = recent_news_sorted.iloc[len(recent_news)//2:]
            
            first_sentiment = first_half['sentiment_score'].mean()
            second_sentiment = second_half['sentiment_score'].mean()
            
            if second_sentiment > first_sentiment + 0.1:
                sentiment_trend = 'improving'
            elif second_sentiment < first_sentiment - 0.1:
                sentiment_trend = 'deteriorating'
            else:
                sentiment_trend = 'stable'
        else:
            sentiment_trend = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_trend': sentiment_trend,
            'news_volume': len(recent_news),
            'sentiment_std': recent_news['sentiment_score'].std()
        }
    
    def generate_recommendations(self, market_context: Dict[str, Any],
                             model_performance: Dict[str, Any],
                             news_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерує рекомендації на основі всього контексту
        """
        regime = market_context.get('regime', 'unknown')
        volatility = market_context.get('volatility', 'medium')
        
        # Визначаємо контекстну комбінацію
        context_key = f"{regime}_{volatility}"
        if regime == 'sideways':
            context_key = 'sideways'
        
        # Рекомендовані моделі
        recommended_models = self.model_recommendations.get(context_key, ['RandomForest'])
        
        # Враховуємо фактичну продуктивність
        best_models = model_performance.get('top_3_models', recommended_models)
        final_model_recommendations = []
        
        for model in recommended_models:
            if model in best_models:
                final_model_recommendations.append({
                    'model': model,
                    'reason': f"Recommended for {context_key} and proven performance",
                    'priority': 'high'
                })
            else:
                final_model_recommendations.append({
                    'model': model,
                    'reason': f"Recommended for {context_key}",
                    'priority': 'medium'
                })
        
        # Пріоритетні фічі
        priority_features = self.feature_priorities.get(regime, ['momentum', 'volatility'])
        
        # Рекомендації по налаштуваннях
        risk_adjustments = {}
        if volatility == 'high':
            risk_adjustments = {
                'position_size': 0.5,      # Зменшити розмір позиції
                'stop_loss': 0.02,          # Тісні стопи
                'take_profit': 0.04,         # Ширші тейки
                'confidence_threshold': 0.7    # Вищий поріг впевненості
            }
        elif volatility == 'low':
            risk_adjustments = {
                'position_size': 1.0,      # Повний розмір
                'stop_loss': 0.05,          # Ширші стопи
                'take_profit': 0.03,         # Тісніші тейки
                'confidence_threshold': 0.5    # Нижчий поріг
            }
        else:
            risk_adjustments = {
                'position_size': 0.75,
                'stop_loss': 0.03,
                'take_profit': 0.045,
                'confidence_threshold': 0.6
            }
        
        # Чи потрібно змінювати стратегію
        strategy_change_needed = False
        strategy_change_reason = ""
        
        if news_context.get('sentiment_trend') == 'deteriorating' and regime == 'bull_market':
            strategy_change_needed = True
            strategy_change_reason = "News sentiment deteriorating despite bull market"
        elif news_context.get('sentiment_trend') == 'improving' and regime == 'bear_market':
            strategy_change_needed = True
            strategy_change_reason = "News sentiment improving despite bear market"
        elif volatility == 'high' and market_context.get('confidence', 0) < 0.5:
            strategy_change_needed = True
            strategy_change_reason = "High volatility with low confidence in regime detection"
        
        return {
            'market_context': market_context,
            'model_recommendations': final_model_recommendations,
            'feature_priorities': priority_features,
            'risk_adjustments': risk_adjustments,
            'strategy_change': {
                'needed': strategy_change_needed,
                'reason': strategy_change_reason,
                'urgency': 'high' if strategy_change_needed else 'low'
            },
            'news_context_integration': {
                'overall_sentiment': news_context.get('overall_sentiment', 0),
                'sentiment_trend': news_context.get('sentiment_trend', 'neutral'),
                'news_volume': news_context.get('news_volume', 0)
            },
            'confidence_score': min(
                market_context.get('confidence', 0) * 0.4 + 
                model_performance.get('best_model_score', 0) * 0.4 + 
                (1 - abs(news_context.get('overall_sentiment', 0))) * 0.2,
                1.0
            )
        }
    
    def analyze_and_advise(self, market_data: pd.DataFrame,
                        model_results: Dict[str, Dict] = None,
                        news_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Основна функція аналізу та рекомендацій
        """
        self.logger.info("Starting comprehensive context analysis and advisory")
        
        # 1. Аналіз ринку
        market_context = self.detect_market_regime(market_data)
        self.logger.info(f"Market regime: {market_context['regime']}, volatility: {market_context['volatility']}")
        
        # 2. Аналіз продуктивності моделей
        model_performance = self.analyze_model_performance(model_results or {})
        if model_performance['best_model']:
            self.logger.info(f"Best performing model: {model_performance['best_model']}")
        
        # 3. Аналіз новин
        news_context = self.analyze_news_sentiment_context(news_data or pd.DataFrame())
        self.logger.info(f"News sentiment: {news_context['overall_sentiment']:.3f}, trend: {news_context['sentiment_trend']}")
        
        # 4. Генерація рекомендацій
        recommendations = self.generate_recommendations(
            market_context, model_performance, news_context
        )
        
        # 5. Додаємо метадані
        recommendations['analysis_timestamp'] = datetime.now().isoformat()
        recommendations['data_quality'] = {
            'market_data_points': len(market_data),
            'models_evaluated': len(model_results or {}),
            'news_analyzed': len(news_data or pd.DataFrame())
        }
        
        self.logger.info(f"Context analysis completed. Confidence: {recommendations['confidence_score']:.3f}")
        
        return recommendations


# Створюємо екземпляр для використання
enhanced_context_advisor = EnhancedContextAdvisor()
