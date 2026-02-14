# core/analysis/unified_context_system.py - Об'єднана контекстна система

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import json

logger = logging.getLogger(__name__)

class UnifiedContextSystem:
    """
    Об'єднана контекстна система for вибору моwhereлей and аналandwithу
    """
    
    def __init__(self, trained_models: Optional[Dict] = None):
        self.trained_models = trained_models or {}
        self.model_performance_db = {}
        self.context_features = []
        self.target_model_mapping = {}
        self.context_clusters = {}
        self.advice_history = []
        
        logger.info(f"[UnifiedContextSystem] Initialized with {len(trained_models)} models")
    
    def define_context_features(self) -> List[str]:
        """Виwithначити фandчand контексту for аналandwithу ринку"""
        context_features = [
            # Волатильнandсть
            'volatility_5d',
            'volatility_20d',
            'volatility_ratio',
            
            # Тренд
            'trend_5d',
            'trend_20d',
            'trend_alignment',
            
            # RSI
            'rsi_current',
            'rsi_5d_avg',
            'rsi_divergence',
            
            # MACD
            'macd_current',
            'macd_signal',
            'macd_divergence',
            
            # Обсяги
            'volume_ratio',
            'volume_trend',
            
            # Цandновand рandвнand
            'price_to_ma20',
            'price_to_ma50',
            'price_position',
            
            # Ринок
            'market_phase',
            'vix_level',
            'bond_yield_trend',
            
            # Час
            'hour_of_day',
            'day_of_week',
            'is_trading_hours',
            
            # Макро
            'macro_bias',
            'macro_volatility',
            'sentiment_score'
        ]
        
        self.context_features = context_features
        return context_features
    
    def analyze_current_context(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Аналandwith поточного контексту"""
        try:
            context = {}
            
            # Волатильнandсть
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                context['volatility_5d'] = returns.tail(5).std()
                context['volatility_20d'] = returns.tail(20).std()
                context['volatility_ratio'] = context['volatility_5d'] / (context['volatility_20d'] + 1e-6)
            
            # Тренд
            if len(market_data) >= 20:
                prices = market_data['close'].values
                x_5 = np.arange(len(prices[-5:]))
                x_20 = np.arange(len(prices[-20:]))
                
                trend_5 = np.polyfit(x_5, prices[-5:], 1)[0]
                trend_20 = np.polyfit(x_20, prices[-20:], 1)[0]
                
                context['trend_5d'] = trend_5
                context['trend_20d'] = trend_20
                context['trend_alignment'] = np.sign(trend_5 * trend_20)
            
            # RSI
            if 'RSI_14' in market_data.columns:
                rsi_values = market_data['RSI_14'].dropna()
                if len(rsi_values) > 0:
                    context['rsi_current'] = rsi_values.iloc[-1]
                    context['rsi_5d_avg'] = rsi_values.tail(5).mean()
                    if len(rsi_values) > 14:
                        rsi_5d_ago = rsi_values.iloc[-6]
                        context['rsi_divergence'] = context['rsi_current'] - rsi_5d_ago
            
            # MACD
            if 'MACD_26_12_9' in market_data.columns and 'MACD_signal_26_12_9' in market_data.columns:
                context['macd_current'] = market_data['MACD_26_12_9'].iloc[-1]
                context['macd_signal'] = market_data['MACD_signal_26_12_9'].iloc[-1]
                context['macd_divergence'] = context['macd_current'] - context['macd_signal']
            
            # Обсяги
            if 'volume' in market_data.columns:
                volume_values = market_data['volume'].dropna()
                if len(volume_values) > 20:
                    context['volume_ratio'] = volume_values.iloc[-1] / volume_values.tail(20).mean()
                    context['volume_trend'] = np.sign(volume_values.tail(5).mean() - volume_values.tail(20).mean())
            
            # Цandновand рandвнand
            if 'close' in market_data.columns and len(market_data) >= 50:
                current_price = market_data['close'].iloc[-1]
                ma20 = market_data['close'].tail(20).mean()
                ma50 = market_data['close'].tail(50).mean()
                
                context['price_to_ma20'] = current_price / ma20
                context['price_to_ma50'] = current_price / ma50
                
                # Поwithицandя цandни в дandапаwithонand 20-днandв
                price_20d_range = market_data['close'].tail(20).max() - market_data['close'].tail(20).min()
                context['price_position'] = (current_price - market_data['close'].tail(20).min()) / (price_20d_range + 1e-6)
            
            # Часовand фandчand
            current_time = datetime.now()
            context['hour_of_day'] = current_time.hour
            context['day_of_week'] = current_time.weekday()
            context['is_trading_hours'] = 1 if 9 <= current_time.hour <= 16 else 0
            
            # Ринкова фаfor
            if context.get('trend_alignment', 0) > 0 and context.get('volatility_ratio', 0) < 1.5:
                context['market_phase'] = 'bull_calm'
            elif context.get('trend_alignment', 0) > 0 and context.get('volatility_ratio', 0) >= 1.5:
                context['market_phase'] = 'bull_volatile'
            elif context.get('trend_alignment', 0) < 0:
                context['market_phase'] = 'bear'
            else:
                context['market_phase'] = 'sideways'
            
            logger.debug(f"[UnifiedContextSystem] Analyzed context with {len(context)} features")
            return context
            
        except Exception as e:
            logger.error(f"[UnifiedContextSystem] Error analyzing context: {e}")
            return {}
    
    def select_best_model(self, context: Dict[str, Any], available_models: List[str]) -> Tuple[str, float]:
        """Вибрати найкращу model for поточного контексту"""
        try:
            if not available_models:
                return "random_forest", 0.5
            
            # Якщо notмає andсторичних data, використовуємо евристику
            if not self.model_performance_db:
                return self._heuristic_model_selection(context, available_models)
            
            # Знаходимо схожand контексти в andсторandї
            similar_contexts = self._find_similar_contexts(context)
            
            if similar_contexts:
                # Вибираємо model на основand схожих контекстandв
                return self._context_based_selection(similar_contexts, available_models)
            else:
                # Використовуємо евристику
                return self._heuristic_model_selection(context, available_models)
                
        except Exception as e:
            logger.error(f"[UnifiedContextSystem] Error selecting model: {e}")
            return available_models[0] if available_models else "random_forest", 0.5
    
    def _heuristic_model_selection(self, context: Dict[str, Any], available_models: List[str]) -> Tuple[str, float]:
        """Евристичний вибandр моwhereлand"""
        try:
            # Баwithовand ваги for моwhereлей
            model_weights = {
                'random_forest': 0.8,
                'xgboost': 0.8,
                'lightgbm': 0.8,
                'catboost': 0.7,
                'lstm': 0.6,
                'transformer': 0.6,
                'linear': 0.5,
                'svm': 0.5,
                'knn': 0.4
            }
            
            # Коригуємо ваги на основand контексту
            volatility = context.get('volatility_ratio', 1.0)
            market_phase = context.get('market_phase', 'sideways')
            
            # При високandй волатильностand вandддаємо перевагу ансамблям
            if volatility > 1.5:
                model_weights['random_forest'] *= 1.2
                model_weights['xgboost'] *= 1.2
                model_weights['lightgbm'] *= 1.2
            
            # При трендових ринках вandддаємо перевагу складним моwhereлям
            if market_phase in ['bull_calm', 'bear_calm']:
                model_weights['lstm'] *= 1.1
                model_weights['transformer'] *= 1.1
            
            # Вибираємо найкращу доступну model
            best_model = None
            best_score = 0.0
            
            for model in available_models:
                if model in model_weights:
                    score = model_weights[model]
                    if score > best_score:
                        best_score = score
                        best_model = model
            
            return best_model or available_models[0], best_score
            
        except Exception as e:
            logger.error(f"[UnifiedContextSystem] Error in heuristic selection: {e}")
            return available_models[0] if available_models else "random_forest", 0.5
    
    def _find_similar_contexts(self, current_context: Dict[str, Any]) -> List[Dict]:
        """Find схожand контексти в andсторandї"""
        similar_contexts = []
        
        for context_key, context_data in self.model_performance_db.items():
            similarity = self._calculate_context_similarity(current_context, context_data.get('context', {}))
            if similarity > 0.7:  # Порandг схожостand
                similar_contexts.append({
                    'context': context_data.get('context', {}),
                    'performance': context_data.get('performance', {}),
                    'similarity': similarity
                })
        
        # Сортуємо for схожandстю
        similar_contexts.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_contexts[:5]  # Поверandємо топ-5
    
    def _calculate_context_similarity(self, ctx1: Dict[str, Any], ctx2: Dict[str, Any]) -> float:
        """Роwithрахувати схожandсть контекстandв"""
        try:
            common_features = set(ctx1.keys()) & set(ctx2.keys())
            
            if not common_features:
                return 0.0
            
            similarity = 0.0
            for feature in common_features:
                val1 = ctx1.get(feature, 0)
                val2 = ctx2.get(feature, 0)
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Нормалandwithована рandwithниця
                    if abs(val1) + abs(val2) > 0:
                        diff = abs(val1 - val2) / (abs(val1) + abs(val2))
                        similarity += 1.0 - diff
                elif val1 == val2:
                    similarity += 1.0
            
            return similarity / len(common_features)
            
        except Exception as e:
            logger.error(f"[UnifiedContextSystem] Error calculating similarity: {e}")
            return 0.0
    
    def _context_based_selection(self, similar_contexts: List[Dict], available_models: List[str]) -> Tuple[str, float]:
        """Вибandр моwhereлand на основand схожих контекстandв"""
        try:
            model_scores = {}
            
            for context_data in similar_contexts:
                performance = context_data.get('performance', {})
                similarity = context_data.get('similarity', 0.0)
                
                for model, perf in performance.items():
                    if model in available_models:
                        if model not in model_scores:
                            model_scores[model] = []
                        model_scores[model].append(perf * similarity)
            
            # Calculating середнand withваженand оцandнки
            avg_scores = {}
            for model, scores in model_scores.items():
                if scores:
                    avg_scores[model] = np.mean(scores)
            
            if avg_scores:
                best_model = max(avg_scores.items(), key=lambda x: x[1])
                return best_model[0], best_model[1]
            else:
                # Якщо notмає data, використовуємо евристику
                return self._heuristic_model_selection(similar_contexts[0]['context'], available_models)
                
        except Exception as e:
            logger.error(f"[UnifiedContextSystem] Error in context-based selection: {e}")
            return available_models[0] if available_models else "random_forest", 0.5
    
    def update_model_performance(self, model_name: str, context: Dict[str, Any], performance: Dict[str, float]):
        """Оновити продуктивнandсть моwhereлand"""
        context_key = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        self.model_performance_db[context_key] = {
            'model': model_name,
            'context': context,
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.debug(f"[UnifiedContextSystem] Updated performance for {model_name}")
    
    def analyze_and_advise(self, current_results: Dict[str, float], full_context: Dict) -> Dict:
        """Аналandwithує реwithульandти and контекст, пропонує оптимальну комбandнацandю"""
        try:
            advice = {
                'timestamp': datetime.now().isoformat(),
                'current_context': full_context,
                'current_results': current_results,
                'recommendations': {},
                'confidence': 0.0,
                'reasoning': []
            }
            
            # Аналandwithуємо поточнand реwithульandти
            if current_results:
                best_current = max(current_results.items(), key=lambda x: x[1])
                advice['recommendations']['current_best'] = best_current[0]
                advice['recommendations']['current_score'] = best_current[1]
            
            # Вибираємо оптимальну model for контексту
            available_models = list(current_results.keys()) if current_results else ['random_forest']
            optimal_model, confidence = self.select_best_model(full_context, available_models)
            
            advice['recommendations']['optimal_model'] = optimal_model
            advice['confidence'] = confidence
            
            # Геnotруємо поясnotння
            reasoning = []
            
            volatility = full_context.get('volatility_ratio', 1.0)
            if volatility > 1.5:
                reasoning.append("High volatility detected - preferring ensemble models")
            elif volatility < 0.5:
                reasoning.append("Low volatility - simpler models may suffice")
            
            market_phase = full_context.get('market_phase', 'unknown')
            if market_phase == 'bull_calm':
                reasoning.append("Bull market with low volatility - complex models preferred")
            elif market_phase == 'bear':
                reasoning.append("Bear market - conservative approach recommended")
            
            advice['reasoning'] = reasoning
            
            # Зберandгаємо в andсторandю
            self.advice_history.append(advice)
            
            logger.info(f"[UnifiedContextSystem] Generated advice: {optimal_model} (confidence: {confidence:.2f})")
            return advice
            
        except Exception as e:
            logger.error(f"[UnifiedContextSystem] Error generating advice: {e}")
            return {'error': str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Отримати пandдсумок продуктивностand"""
        try:
            summary = {
                'total_contexts_analyzed': len(self.model_performance_db),
                'models_tracked': set(),
                'avg_performance_by_model': {},
                'context_patterns': {},
                'recent_advice': self.advice_history[-5:] if self.advice_history else []
            }
            
            # Збираємо моwhereлand and продуктивнandсть
            model_performances = {}
            
            for context_data in self.model_performance_db.values():
                model = context_data.get('model')
                performance = context_data.get('performance', {})
                
                if model:
                    summary['models_tracked'].add(model)
                    
                    for metric, value in performance.items():
                        if model not in model_performances:
                            model_performances[model] = []
                        model_performances[model].append(value)
            
            # Calculating середню продуктивнandсть
            for model, performances in model_performances.items():
                if performances:
                    summary['avg_performance_by_model'][model] = np.mean(performances)
            
            summary['models_tracked'] = list(summary['models_tracked'])
            
            return summary
            
        except Exception as e:
            logger.error(f"[UnifiedContextSystem] Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def export_context_data(self, filepath: str):
        """Експортувати контекстнand данand"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'model_performance_db': self.model_performance_db,
                'advice_history': self.advice_history,
                'context_features': self.context_features,
                'performance_summary': self.get_performance_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"[UnifiedContextSystem] Context data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"[UnifiedContextSystem] Error exporting data: {e}")

# Функцandї for сумandсностand withand сandрим codeом
def create_context_advisor(trained_models: Dict) -> UnifiedContextSystem:
    """Create контекстний радник (сумandснandсть)"""
    return UnifiedContextSystem(trained_models)

def create_model_selector() -> UnifiedContextSystem:
    """Create селектор моwhereлей (сумandснandсть)"""
    return UnifiedContextSystem()

if __name__ == "__main__":
    # Тестування
    system = UnifiedContextSystem()
    
    # Тестовий контекст
    test_context = {
        'volatility_ratio': 1.2,
        'trend_alignment': 1,
        'market_phase': 'bull_calm',
        'rsi_current': 65
    }
    
    # Тестовий вибandр моwhereлand
    available_models = ['random_forest', 'xgboost', 'lstm']
    best_model, confidence = system.select_best_model(test_context, available_models)
    
    print(f"Best model: {best_model} (confidence: {confidence:.2f})")
