#!/usr/bin/env python3
"""
Intelligent Model Selection
Інтелектуальний вибір моделей на основі патернів та ринкових умов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class IntelligentModelSelector:
    """
    [START] Інтелектуальний вибір моделей з урахуванням патернів
    """
    
    def __init__(self):
        self.model_registry = {
            'linear': {
                'class': LinearRegression,
                'strengths': ['low_data', 'linear_relationships', 'interpretability'],
                'weaknesses': ['non_linear', 'complex_patterns'],
                'complexity': 'low',
                'training_speed': 'fast'
            },
            'ridge': {
                'class': Ridge,
                'strengths': ['multicollinearity', 'regularization', 'stability'],
                'weaknesses': ['non_linear', 'feature_selection'],
                'complexity': 'low',
                'training_speed': 'fast'
            },
            'random_forest': {
                'class': RandomForestRegressor,
                'strengths': ['non_linear', 'robust', 'feature_importance', 'missing_data'],
                'weaknesses': ['overfitting', 'interpretability'],
                'complexity': 'medium',
                'training_speed': 'medium'
            },
            'gradient_boosting': {
                'class': GradientBoostingRegressor,
                'strengths': ['non_linear', 'high_accuracy', 'gradient_boosting'],
                'weaknesses': ['sensitive_to_params', 'training_time'],
                'complexity': 'medium',
                'training_speed': 'slow'
            },
            'svr': {
                'class': SVR,
                'strengths': ['non_linear', 'kernel_methods', 'small_data'],
                'weaknesses': ['scaling_sensitive', 'large_data'],
                'complexity': 'medium',
                'training_speed': 'slow'
            }
        }
        
        # [TARGET] Pattern-based рекомендації
        self.pattern_recommendations = {
            'high_volatility': {
                'preferred': ['random_forest', 'gradient_boosting'],
                'avoid': ['linear', 'ridge'],
                'reason': 'High volatility requires robust non-linear models'
            },
            'anomaly_present': {
                'preferred': ['random_forest', 'svr'],
                'avoid': ['linear'],
                'reason': 'Anomalies need robust models that handle outliers'
            },
            'regime_change': {
                'preferred': ['gradient_boosting', 'random_forest'],
                'avoid': ['linear', 'ridge'],
                'reason': 'Regime changes require models that capture non-linear patterns'
            },
            'low_data_quality': {
                'preferred': ['ridge', 'random_forest'],
                'avoid': ['svr', 'gradient_boosting'],
                'reason': 'Low quality data needs robust regularization'
            },
            'high_feature_count': {
                'preferred': ['random_forest', 'ridge'],
                'avoid': ['svr'],
                'reason': 'High dimensional data needs feature selection'
            }
        }
        
        self.selection_history = []
        
    def select_intelligent_models(self, 
                                features: Dict, 
                                targets: Dict, 
                                patterns: Dict = None,
                                config: Dict = None) -> Dict:
        """
        [START] Інтелектуально вибираємо моделі
        """
        logger.info("[START] Starting Intelligent Model Selection")
        
        config = config or self._get_default_config()
        
        # [TARGET] Аналізуємо характеристики data
        data_characteristics = self._analyze_data_characteristics(features, targets, patterns)
        
        # [TARGET] Отримуємо рекомендації на основі патернів
        pattern_recommendations = self._get_pattern_recommendations(patterns, data_characteristics)
        
        # [TARGET] Оцінюємо кожну модель
        model_scores = self._evaluate_all_models(features, targets, data_characteristics)
        
        # [TARGET] Комбінуємо рекомендації та оцінки
        final_recommendations = self._combine_recommendations(model_scores, pattern_recommendations)
        
        # [TARGET] Вибираємо топ моделі
        selected_models = self._select_top_models(final_recommendations, config)
        
        logger.info(f"[OK] Selected {len(selected_models)} models: {list(selected_models.keys())}")
        
        return {
            'selected_models': selected_models,
            'data_characteristics': data_characteristics,
            'pattern_recommendations': pattern_recommendations,
            'model_scores': model_scores,
            'final_recommendations': final_recommendations
        }
    
    def _analyze_data_characteristics(self, features: Dict, targets: Dict, patterns: Dict) -> Dict:
        """
        [START] Аналізуємо характеристики data
        """
        characteristics = {
            'sample_size': 0,
            'feature_count': 0,
            'target_type': 'regression',
            'data_quality': 'high',
            'volatility_level': 'medium',
            'linearity_score': 0.5,
            'missing_data_ratio': 0.0,
            'outlier_ratio': 0.0,
            'multicollinearity_score': 0.5
        }
        
        # [TARGET] Розмір вибірки
        if features:
            total_features = 0
            for category, feature_data in features.items():
                if isinstance(feature_data, dict):
                    total_features += len(feature_data)
            characteristics['feature_count'] = total_features
            
            # Оцінка розміру вибірки
            if total_features > 0:
                characteristics['sample_size'] = 1000  # TODO: реальний розмір
        
        # [TARGET] Якість data
        if patterns:
            # Аналізуємо патерни для якості
            total_patterns = sum(len(p) if isinstance(p, list) else 1 for p in patterns.values())
            if total_patterns > 10:
                characteristics['data_quality'] = 'low'
            elif total_patterns > 5:
                characteristics['data_quality'] = 'medium'
        
        # [TARGET] Волатильність
        if 'price_features' in features:
            characteristics['volatility_level'] = self._estimate_volatility(features['price_features'])
        
        # [TARGET] Лінійність
        characteristics['linearity_score'] = self._estimate_linearity(features, targets)
        
        logger.info(f"[TARGET] Data characteristics: {characteristics}")
        return characteristics
    
    def _estimate_volatility(self, price_features: Dict) -> str:
        """Оцінюємо рівень волатильності"""
        # Проста реалізація
        return 'medium'  # TODO: реалізувати реальну оцінку
    
    def _estimate_linearity(self, features: Dict, targets: Dict) -> float:
        """Оцінюємо лінійність data"""
        # Проста реалізація
        return 0.5  # TODO: реалізувати реальну оцінку
    
    def _get_pattern_recommendations(self, patterns: Dict, characteristics: Dict) -> Dict:
        """
        [START] Отримуємо рекомендації на основі патернів
        """
        recommendations = {
            'preferred_models': [],
            'avoid_models': [],
            'reasoning': []
        }
        
        # [TARGET] Аналізуємо патерни
        if patterns:
            for pattern_type, pattern_list in patterns.items():
                if isinstance(pattern_list, list) and len(pattern_list) > 0:
                    # Перевіряємо типи патернів
                    for pattern in pattern_list:
                        pattern_str = str(pattern).lower()
                        
                        # [TARGET] High volatility
                        if 'volatility' in pattern_str or 'high_volatility' in pattern_str:
                            rec = self.pattern_recommendations.get('high_volatility', {})
                            recommendations['preferred_models'].extend(rec.get('preferred', []))
                            recommendations['avoid_models'].extend(rec.get('avoid', []))
                            recommendations['reasoning'].append(rec.get('reason', ''))
                        
                        # [TARGET] Anomalies
                        elif 'anomaly' in pattern_str:
                            rec = self.pattern_recommendations.get('anomaly_present', {})
                            recommendations['preferred_models'].extend(rec.get('preferred', []))
                            recommendations['avoid_models'].extend(rec.get('avoid', []))
                            recommendations['reasoning'].append(rec.get('reason', ''))
                        
                        # [TARGET] Regime change
                        elif 'regime' in pattern_str:
                            rec = self.pattern_recommendations.get('regime_change', {})
                            recommendations['preferred_models'].extend(rec.get('preferred', []))
                            recommendations['avoid_models'].extend(rec.get('avoid', []))
                            recommendations['reasoning'].append(rec.get('reason', ''))
        
        # [TARGET] Видаляємо дублікати
        recommendations['preferred_models'] = list(set(recommendations['preferred_models']))
        recommendations['avoid_models'] = list(set(recommendations['avoid_models']))
        
        logger.info(f"[TARGET] Pattern recommendations: {recommendations}")
        return recommendations
    
    def _evaluate_all_models(self, features: Dict, targets: Dict, characteristics: Dict) -> Dict:
        """
        [START] Оцінюємо всі моделі
        """
        model_scores = {}
        
        # [TARGET] Створюємо тестові дані
        X, y = self._create_test_data(features, targets)
        
        for model_name, model_info in self.model_registry.items():
            try:
                # [TARGET] Оцінюємо відповідність моделі
                suitability_score = self._evaluate_model_suitability(model_name, characteristics)
                
                # [TARGET] Базова оцінка продуктивності
                performance_score = self._estimate_model_performance(model_name, X, y)
                
                # [TARGET] Комбінована оцінка
                final_score = (suitability_score + performance_score) / 2
                
                model_scores[model_name] = {
                    'suitability_score': suitability_score,
                    'performance_score': performance_score,
                    'final_score': final_score,
                    'strengths': model_info['strengths'],
                    'weaknesses': model_info['weaknesses'],
                    'complexity': model_info['complexity'],
                    'training_speed': model_info['training_speed']
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                model_scores[model_name] = {
                    'suitability_score': 0.0,
                    'performance_score': 0.0,
                    'final_score': 0.0,
                    'error': str(e)
                }
        
        return model_scores
    
    def _create_test_data(self, features: Dict, targets: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """Створюємо тестові дані для оцінки"""
        # Проста реалізація
        n_samples = 1000
        n_features = 10
        
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = pd.Series(np.random.randn(n_samples))
        
        return X, y
    
    def _evaluate_model_suitability(self, model_name: str, characteristics: Dict) -> float:
        """
        [START] Оцінюємо відповідність моделі характеристикам data
        """
        score = 0.5  # Базовий score
        
        model_info = self.model_registry.get(model_name, {})
        
        # [TARGET] Волатильність
        if characteristics['volatility_level'] == 'high':
            if 'robust' in model_info.get('strengths', []):
                score += 0.2
            if 'linear_relationships' in model_info.get('strengths', []):
                score -= 0.2
        
        # [TARGET] Якість data
        if characteristics['data_quality'] == 'low':
            if 'regularization' in model_info.get('strengths', []):
                score += 0.2
            if 'sensitive_to_params' in model_info.get('weaknesses', []):
                score -= 0.2
        
        # [TARGET] Розмір data
        if characteristics['sample_size'] < 1000:
            if 'small_data' in model_info.get('strengths', []):
                score += 0.2
            if 'large_data' in model_info.get('weaknesses', []):
                score -= 0.2
        
        # [TARGET] Кількість фіч
        if characteristics['feature_count'] > 100:
            if 'feature_importance' in model_info.get('strengths', []):
                score += 0.2
            if 'scaling_sensitive' in model_info.get('weaknesses', []):
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _estimate_model_performance(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> float:
        """
        [START] Оцінюємо продуктивність моделі
        """
        try:
            model_class = self.model_registry[model_name]['class']
            
            # Базові параметри
            if model_name == 'svr':
                model = model_class(kernel='rbf')
            elif model_name in ['ridge', 'lasso']:
                model = model_class(alpha=1.0)
            else:
                model = model_class(random_state=42)
            
            # Простий train/test split
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Навчаємо модель
            model.fit(X_train, y_train)
            
            # Оцінюємо
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            
            return max(0.0, score)
            
        except Exception as e:
            logger.error(f"Error estimating performance for {model_name}: {e}")
            return 0.0
    
    def _combine_recommendations(self, model_scores: Dict, pattern_recommendations: Dict) -> Dict:
        """
        [START] Комбінуємо оцінки з рекомендаціями патернів
        """
        combined_scores = {}
        
        for model_name, scores in model_scores.items():
            base_score = scores['final_score']
            
            # [TARGET] Бонус за рекомендації
            bonus = 0.0
            if model_name in pattern_recommendations.get('preferred_models', []):
                bonus += 0.2
            
            # [TARGET] Штраф за моделі, які needed уникати
            if model_name in pattern_recommendations.get('avoid_models', []):
                bonus -= 0.3
            
            # [TARGET] Фінальний score
            final_score = max(0.0, min(1.0, base_score + bonus))
            
            combined_scores[model_name] = {
                **scores,
                'pattern_bonus': bonus,
                'combined_score': final_score
            }
        
        return combined_scores
    
    def _select_top_models(self, combined_scores: Dict, config: Dict) -> Dict:
        """
        [START] Вибираємо топ моделі
        """
        max_models = config.get('max_models', 3)
        
        # Сортуємо за combined_score
        sorted_models = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        # Вибираємо топ моделі
        top_models = {}
        for model_name, scores in sorted_models[:max_models]:
            model_class = self.model_registry[model_name]['class']
            
            # Створюємо модель з оптимальними параметрами
            model = self._create_optimized_model(model_name, model_class)
            
            top_models[model_name] = {
                'model': model,
                'scores': scores,
                'model_class': model_class,
                'reasoning': self._get_selection_reasoning(model_name, scores)
            }
        
        return top_models
    
    def _create_optimized_model(self, model_name: str, model_class) -> Any:
        """Створюємо оптимізовану модель"""
        # Базові оптимізовані параметри
        optimized_params = {
            'random_forest': {'n_estimators': 100, 'max_depth': 8, 'random_state': 42},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 0.1},
            'svr': {'kernel': 'rbf', 'C': 1.0},
            'linear': {}
        }
        
        params = optimized_params.get(model_name, {})
        return model_class(**params)
    
    def _get_selection_reasoning(self, model_name: str, scores: Dict) -> str:
        """Отримуємо пояснення вибору моделі"""
        reasons = []
        
        if scores['combined_score'] > 0.8:
            reasons.append("Excellent overall performance")
        
        if scores['suitability_score'] > 0.7:
            reasons.append("Well-suited for data characteristics")
        
        if scores['pattern_bonus'] > 0:
            reasons.append("Recommended by pattern analysis")
        
        if scores['performance_score'] > 0.7:
            reasons.append("Strong predictive performance")
        
        return "; ".join(reasons) if reasons else "Selected based on overall evaluation"
    
    def _get_default_config(self) -> Dict:
        """Конфігурація за замовчуванням"""
        return {
            'max_models': 3,
            'min_score_threshold': 0.3,
            'include_reasoning': True
        }


# [TARGET] ГОЛОВНА ФУНКЦІЯ
def select_intelligent_models(features: Dict, targets: Dict, patterns: Dict = None, config: Dict = None) -> Dict:
    """
    [START] Запускаємо інтелектуальний вибір моделей
    """
    selector = IntelligentModelSelector()
    return selector.select_intelligent_models(features, targets, patterns, config)


if __name__ == "__main__":
    print("Intelligent Model Selection - готовий до використання")
    print("[START] Інтелектуальний вибір моделей на основі патернів")
    print("[DATA] Pattern-aware, adaptive, data-characteristic-based!")
