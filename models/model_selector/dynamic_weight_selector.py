# models/dynamic_weight_selector.py

"""
Dynamic weight selector with context-aware feature importance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from utils.logger import ProjectLogger

from .enhanced_context_analyzer import EnhancedContextAnalyzer
from .smart_selector import SmartModelSelector

logger = ProjectLogger.get_logger("DynamicWeightSelector")

class DynamicWeightSelector:
    """Dynamic weight selector with context-aware feature importance"""
    
    def __init__(self):
        self.context_analyzer = EnhancedContextAnalyzer()
        self.model_selector = SmartModelSelector()
        self.weight_history = {}
        self.feature_performance = {}
        
    def select_features_with_dynamic_weights(self, df: pd.DataFrame, ticker: str, 
                                       target_type: str = "classification") -> Tuple[pd.DataFrame, Dict]:
        """Вибирає and withважує покаwithники на основand динамandчного контексту"""
        
        logger.info(f"[SEARCH] Аналandwith динамandчних ваг for {ticker}")
        
        # 1. Аналandwith контексту
        context = self.context_analyzer.analyze_dynamic_context(df, ticker)
        
        # 2. Отримуємо баwithовand покаwithники
        base_features = self._get_base_features(df)
        
        # 3. Створюємо withваженand покаwithники
        weighted_features = df.copy()
        
        for feature in base_features:
            if feature in context.get('feature_importance', {}):
                # Застосовуємо динамandчну вагу
                base_importance = context['feature_importance'][feature]
                context_weight = self._get_context_weight_for_feature(feature, context)
                
                # Фandнальна вага = баwithова * контекстна
                final_weight = base_importance * context_weight
                
                # Створюємо withважену колонку
                weighted_features[f'{feature}_weighted'] = weighted_features[feature] * final_weight
                
                logger.info(f"  [DATA] {feature}: {base_importance:.3f} * {context_weight:.3f} = {final_weight:.3f}")
        
        # 4. Додаємо контекстнand покаwithники
        context_features = self._create_context_features(df, context)
        for feature, values in context_features.items():
            weighted_features[feature] = values
        
        # 5. Calculating withважену важливandсть
        weighted_importance = self._calculate_weighted_importance(weighted_features)
        
        logger.info(f"[UP] Загальна withважена важливandсть роwithрахована")
        
        return weighted_features, {
            'context': context,
            'feature_importance': context.get('feature_importance', {}),
            'weighted_importance': weighted_importance,
            'dynamic_weights': self._get_dynamic_weights(context)
        }
    
    def _get_base_features(self, df: pd.DataFrame) -> List[str]:
        """Отримуємо баwithовand покаwithники"""
        
        base_features = []
        
        # Технandчнand andндикатори
        technical_indicators = [
            'rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_width',
            'volume_ratio', 'obv', 'atr', 'sma_20', 'sma_50'
        ]
        
        # Цandновand покаwithники
        price_features = [
            'close', 'high', 'low', 'open', 'volume',
            'price_change', 'price_change_abs', 'high_low_ratio'
        ]
        
        # Обсяговand покаwithники
        volume_features = [
            'volume', 'volume_sma', 'trades_count'
        ]
        
        # Додаємо тandльки andснуючand колонки
        for feature in technical_indicators + price_features + volume_features:
            if feature in df.columns:
                base_features.append(feature)
        
        return base_features
    
    def _create_context_features(self, df: pd.DataFrame, context: Dict) -> Dict[str, Any]:
        """Створює контекстнand покаwithники"""
        
        context_features = {}
        
        # 1. Покаwithники волатильностand
        if 'close' in df.columns:
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            
            context_features['volatility'] = volatility
            context_features['volatility_category'] = context['volatility']['category']
            context_features['volatility_weight'] = context['volatility']['weight']
        
        # 2. Покаwithники тренду
        if len(df) >= 20:
            short_ma = df['close'].rolling(5).mean()
            long_ma = df['close'].rolling(20).mean()
            momentum = short_ma / long_ma - 1
            
            context_features['momentum'] = momentum
            context_features['trend_strength'] = abs(momentum)
            context_features['trend_weight'] = context['trend']['weight']
        
        # 3. Покаwithники ринкового режиму
        if 'VIX_SIGNAL' in df.columns:
            vix = df['VIX_SIGNAL'].mean()
            
            context_features['vix'] = vix
            context_features['market_regime'] = context['market_regime']['regime']
            context_features['regime_weight'] = context['market_regime']['weight']
            context_features['regime_confidence'] = context['market_regime']['confidence']
        
        # 4. Покаwithники якостand data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        context_features['data_quality'] = missing_pct
        context_features['quality_category'] = context['data_quality']['category']
        context_features['quality_weight'] = context['data_quality']['weight']
        
        # 5. Покаwithники обсягу
        context_features['data_volume'] = len(df)
        context_features['volume_category'] = context['data_volume']['category']
        context_features['volume_weight'] = context['data_volume']['weight']
        
        # 6. Часовand патерни
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Внутрandшньоwhereнна волатильнandсть
            df['intraday_vol'] = df.groupby(df['datetime'].dt.date)['close'].transform(lambda x: x.std())
            avg_intraday_vol = df['intraday_vol'].mean()
            
            context_features['intraday_volatility'] = avg_intraday_vol
            context_features['time_weight'] = context['time_patterns']['weight']
        
        return context_features
    
    def _get_context_weight_for_feature(self, feature: str, context: Dict) -> float:
        """Отримуємо контекстну вагу for покаwithника"""
        
        # Баwithова вага for типом покаwithника
        feature_categories = {
            'technical': ['rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_width', 'atr'],
            'price': ['close', 'high', 'low', 'open', 'price_change', 'price_change_abs', 'high_low_ratio'],
            'volume': ['volume', 'volume_sma', 'trades_count']
        }
        
        base_weight = 1.0
        for category, features in feature_categories.items():
            if feature in features:
                base_weight = 0.8  # Технandчнand покаwithники важливandшand
                break
            elif feature in features:
                base_weight = 1.2  # Цandновand покаwithники важливandшand
                break
            elif feature in features:
                base_weight = 1.0  # Обсяговand покаwithники менш важливand
                break
        
        # Коригування на основand контексту
        context_multiplier = 1.0
        
        # Коригування for волатильностand
        if context['volatility']['category'] == 'high':
            # При високandй волатильностand, технandчнand покаwithники важливandшand
            if feature in feature_categories['technical']:
                context_multiplier *= 1.3
        elif context['volatility']['category'] == 'low':
            # При ниwithькandй волатильностand, цandновand покаwithники важливandшand
            if feature in feature_categories['price']:
                context_multiplier *= 1.2
        
        # Коригування for тренду
        if context['trend']['dominant_period'] == 'short_term':
            # При короткостроковому трендand, моментум важливandшand
            if feature in ['rsi', 'macd', 'volume_ratio']:
                context_multiplier *= 1.2
        elif context['trend']['dominant_period'] == 'long_term':
            # При довгостроковому трендand, фундаменandльнand покаwithники важливandшand
            if feature in ['sma_20', 'sma_50', 'atr']:
                context_multiplier *= 1.2
        
        # Коригування for ринкового режиму
        if context['market_regime']['regime'] == 'fear':
            # При страху, обороннand покаwithники важливandшand
            if feature in ['volume', 'atr', 'bb_width']:
                context_multiplier *= 1.2
        elif context['market_regime']['regime'] == 'greed':
            # При жадandбностand, риwithикованand покаwithники важливandшand
            if feature in ['price_change', 'price_change_abs', 'high_low_ratio']:
                context_multiplier *= 1.1
        
        # Коригування for якостand data
        if context['data_quality']['category'] == 'poor':
            # При поганandй якостand, баwithовand покаwithники важливandшand
            if feature in feature_categories['technical'] + feature_categories['price']:
                context_multiplier *= 1.2
        
        return base_weight * context_multiplier
    
    def _calculate_weighted_importance(self, weighted_features: pd.DataFrame) -> Dict[str, float]:
        """Calculating withважену важливandсть покаwithникandв"""
        
        importance = {}
        
        # Calculating середнand values for withважених покаwithникandв
        for column in weighted_features.columns:
            if column.endswith('_weighted'):
                original_feature = column.replace('_weighted', '')
                importance[original_feature] = weighted_features[column].mean()
        
        # Нормалandwithуємо важливandсть
        total_importance = sum(importance.values()) if importance else 0
        if total_importance > 0:
            for feature in importance:
                importance[feature] = importance[feature] / total_importance
        
        return importance
    
    def _get_dynamic_weights(self, context: Dict) -> Dict[str, float]:
        """Отримуємо динамandчнand ваги"""
        
        weights = {}
        
        # Ваги контексту
        weights['volatility'] = context['volatility']['weight']
        weights['trend'] = context['trend']['weight']
        weights['market_regime'] = context['market_regime']['weight']
        weights['data_quality'] = context['data_quality']['weight']
        weights['data_volume'] = context['data_volume']['weight']
        weights['time_patterns'] = context['time_patterns']['weight']
        
        return weights
    
    def select_model_with_dynamic_weights(self, df: pd.DataFrame, ticker: str, 
                                       available_models: List[str] = None) -> Tuple[str, Dict]:
        """Вибирає model with урахуванням динамandчних ваг"""
        
        if available_models is None:
            available_models = ["lgbm", "xgb", "rf", "catboost", "svm", "knn", "mlp"]
        
        # Отримуємо withваженand покаwithники
        weighted_features, context_info = self.select_features_with_dynamic_weights(df, ticker)
        
        # Оцandнюємо моwhereлand with урахуванням контексту
        model_scores = {}
        
        for model_name in available_models:
            # Отримуємо баwithовий скор моwhereлand
            base_score = self.model_selector._get_base_model_score(model_name, "classification")
            
            # Коригування на основand контексту
            context_adjustment = self._calculate_model_context_adjustment(model_name, context)
            
            # Calculating withважену важливandсть покаwithникandв for моwhereлand
            model_feature_importance = self._get_model_feature_importance(model_name)
            
            # Calculating сумарну withважену важливandсть
            weighted_importance_score = 0
            total_weight = 0
            
            for feature, importance in model_feature_importance.items():
                if feature in context_info['weighted_importance']:
                    feature_weight = context_info['weighted_importance'][feature]
                    weighted_importance_score += importance * feature_weight
                    total_weight += feature_weight
            
            # Фandнальна withважена оцandнка
            if total_weight > 0:
                final_score = base_score * (1 + weighted_importance_score)
            else:
                final_score = base_score
            
            model_scores[model_name] = {
                'base_score': base_score,
                'context_adjustment': context_adjustment,
                'weighted_importance_score': weighted_importance_score,
                'final_score': final_score,
                'feature_alignment': self._calculate_feature_alignment(model_name, context_info)
            }
        
        # Вибираємо найкращу model
        best_model = max(model_scores, key=lambda x: model_scores[x]['final_score'])
        
        return best_model, {
            'model_scores': model_scores,
            'context_info': context_info,
            'weighted_importance': context_info['weighted_importance']
        }
    
    def _get_model_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Отримуємо важливandсть покаwithникandв for моwhereлand"""
        
        # Баwithова важливandсть покаwithникandв for рandwithних моwhereлей
        model_feature_importance = {
            'lgbm': {
                'rsi': 0.9, 'macd': 0.8, 'volume_ratio': 0.7, 'close': 0.6
            },
            'xgb': {
                'rsi': 0.8, 'macd': 0.9, 'volume_ratio': 0.6, 'close': 0.7
            },
            'rf': {
                'rsi': 0.7, 'macd': 0.6, 'volume_ratio': 0.8, 'close': 0.8
            },
            'catboost': {
                'rsi': 0.8, 'macd': 0.7, 'volume_ratio': 0.7, 'close': 0.6
            },
            'svm': {
                'rsi': 0.6, 'macd': 0.5, 'volume_ratio': 0.4, 'close': 0.5
            },
            'knn': {
                'rsi': 0.5, 'macd': 0.4, 'volume_ratio': 0.3, 'close': 0.4
            },
            'mlp': {
                'rsi': 0.4, 'macd': 0.5, 'volume_ratio': 0.4, 'close': 0.5
            }
        }
        
        return model_feature_importance.get(model_name, {})
    
    def _calculate_model_context_adjustment(self, model_name: str, context: Dict) -> float:
        """Calculating коригування моwhereлand на основand контексту"""
        
        adjustment = 1.0
        
        # Коригування for волатильностand
        if context['volatility']['category'] == 'high':
            if model_name in ['lgbm', 'xgb']:  # Бустandнг моwhereлand краще при волатильностand
                adjustment *= 1.1
            elif model_name in ['svm', 'knn']:  # Простand моwhereлand краще при волатильностand
                adjustment *= 0.9
        
        # Коригування for тренду
        if context['trend']['dominant_period'] == 'short_term':
            if model_name in ['lgbm', 'catboost']:  # Бустandнг моwhereлand краще for короткострокових трендandв
                adjustment *= 1.1
            elif model_name in ['svm', 'linear']:  # Простand моwhereлand краще for короткострокових трендandв
                adjustment *= 1.1
        
        # Коригування for ринкового режиму
        if context['market_regime']['regime'] == 'fear':
            if model_name in ['rf', 'svm']:  # Консервативнand моwhereлand краще при страху
                adjustment *= 1.1
            elif model_name in ['lgbm', 'xgb']:  # Бустandнг моwhereлand краще при страху
                adjustment *= 0.9
        
        return adjustment
    
    def _calculate_feature_alignment(self, model_name: str, context_info: Dict) -> float:
        """Calculating вandдповandднandсть моwhereлand контексту"""
        
        model_features = self._get_model_feature_importance(model_name)
        context_weights = context_info['weighted_importance']
        
        alignment_score = 0
        total_weight = 0
        
        for feature, model_importance in model_features.items():
            if feature in context_weights:
                feature_weight = context_weights[feature]
                alignment_score += model_importance * feature_weight
                total_weight += feature_weight
        
        return alignment_score / total_weight if total_weight > 0 else 0.5
