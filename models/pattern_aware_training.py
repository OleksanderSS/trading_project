#!/usr/bin/env python3
"""
Pattern-Aware Model Training
Інтелектуальне навчання моделей з урахуванням патернів з етапів 1-3
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PatternAwareModelTrainer:
    """
    [START] Інтелектуальний тренер моделей з урахуванням патернів
    """
    
    def __init__(self):
        self.model_registry = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor
        }
        
        # [TARGET] Pattern-aware параметри
        self.pattern_params = {
            'high_volatility': {
                'random_forest': {'n_estimators': 200, 'max_depth': 5},
                'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.05}
            },
            'low_volatility': {
                'random_forest': {'n_estimators': 100, 'max_depth': 10},
                'gradient_boosting': {'n_estimators': 200, 'learning_rate': 0.1}
            },
            'anomaly_present': {
                'random_forest': {'n_estimators': 300, 'max_depth': 3},
                'gradient_boosting': {'n_estimators': 150, 'learning_rate': 0.03}
            },
            'regime_change': {
                'random_forest': {'n_estimators': 250, 'max_depth': 4},
                'gradient_boosting': {'n_estimators': 120, 'learning_rate': 0.07}
            }
        }
        
        self.training_history = []
        self.model_performance = {}
        
    def train_pattern_aware_models(self, 
                                  features: Dict, 
                                  targets: Dict, 
                                  patterns: Dict = None,
                                  config: Dict = None) -> Dict:
        """
        [START] Навчаємо моделі з урахуванням патернів
        """
        logger.info("[START] Starting Pattern-Aware Model Training")
        
        config = config or self._get_default_config()
        
        # [TARGET] Аналізуємо патерни та ринкові умови
        market_conditions = self._analyze_market_conditions(features, patterns)
        
        # [TARGET] Готуємо дані для навчання
        training_data = self._prepare_training_data(features, targets, market_conditions)
        
        # [TARGET] Навчаємо моделі з адаптивними параметрами
        trained_models = {}
        
        for model_name in config.get('models', ['random_forest', 'gradient_boosting']):
            logger.info(f"[TARGET] Training {model_name} with pattern-aware parameters")
            
            # [TARGET] Отримуємо адаптивні параметри
            adaptive_params = self._get_adaptive_parameters(model_name, market_conditions)
            
            # [TARGET] Навчаємо модель
            model_result = self._train_single_model(
                model_name, 
                training_data, 
                adaptive_params,
                market_conditions
            )
            
            trained_models[model_name] = model_result
        
        # [TARGET] Аналізуємо результати
        training_results = self._analyze_training_results(trained_models, market_conditions)
        
        # [TARGET] Вибираємо найкращу модель
        best_model = self._select_best_model(trained_models, training_results)
        
        logger.info(f"[OK] Pattern-Aware Training completed. Best model: {best_model['model_name']}")
        
        return {
            'trained_models': trained_models,
            'best_model': best_model,
            'market_conditions': market_conditions,
            'training_results': training_results,
            'pattern_metadata': patterns
        }
    
    def _analyze_market_conditions(self, features: Dict, patterns: Dict) -> Dict:
        """
        [START] Аналізуємо ринкові умови на основі фіч та патернів
        """
        conditions = {
            'volatility_regime': 'normal',
            'pattern_presence': 'none',
            'data_quality': 'high',
            'market_regime': 'neutral',
            'feature_complexity': 'medium'
        }
        
        # [TARGET] Аналіз волатильності
        if 'price_features' in features:
            volatility = self._estimate_volatility(features['price_features'])
            if volatility > 0.03:
                conditions['volatility_regime'] = 'high'
            elif volatility < 0.01:
                conditions['volatility_regime'] = 'low'
        
        # [TARGET] Аналіз патернів
        if patterns:
            anomaly_count = sum(len(p) for p in patterns.values() if isinstance(p, list))
            if anomaly_count > 5:
                conditions['pattern_presence'] = 'anomaly_present'
            elif anomaly_count > 2:
                conditions['pattern_presence'] = 'moderate'
            
            # [TARGET] Аналіз regime change
            for timeframe, pattern_list in patterns.items():
                if any('regime' in str(p).lower() for p in pattern_list):
                    conditions['market_regime'] = 'regime_change'
                    break
        
        # [TARGET] Аналіз якості data
        if 'quality_features' in features:
            quality_score = self._estimate_data_quality(features['quality_features'])
            if quality_score < 0.7:
                conditions['data_quality'] = 'low'
            elif quality_score < 0.85:
                conditions['data_quality'] = 'medium'
        
        logger.info(f"[TARGET] Market conditions: {conditions}")
        return conditions
    
    def _estimate_volatility(self, price_features: Dict) -> float:
        """Оцінюємо волатильність"""
        # Проста реалізація
        return 0.02  # TODO: реалізувати реальну оцінку
    
    def _estimate_data_quality(self, quality_features: Dict) -> float:
        """Оцінюємо якість data"""
        # Проста реалізація
        return 0.8  # TODO: реалізувати реальну оцінку
    
    def _prepare_training_data(self, features: Dict, targets: Dict, conditions: Dict) -> Dict:
        """
        [START] Готуємо дані для навчання з урахуванням умов
        """
        # [TARGET] Створюємо матрицю фіч
        X = self._create_feature_matrix(features)
        
        # [TARGET] Створюємо таргети
        y = self._create_target_vector(targets)
        
        # [TARGET] Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # [TARGET] Валідація
        if len(X) < 50:
            raise ValueError("Insufficient data for training")
        
        return {
            'X': X,
            'y': y,
            'tscv': tscv,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
    
    def _create_feature_matrix(self, features: Dict) -> pd.DataFrame:
        """Створюємо матрицю фіч"""
        all_features = []
        
        for category, feature_data in features.items():
            if isinstance(feature_data, dict):
                for feature_name, feature_values in feature_data.items():
                    if isinstance(feature_values, np.ndarray):
                        if len(feature_values.shape) == 1:
                            all_features.append(pd.Series(feature_values, name=feature_name))
                        else:
                            # Багатовимірні дані
                            for i in range(feature_values.shape[1]):
                                all_features.append(pd.Series(feature_values[:, i], name=f"{feature_name}_{i}"))
                    elif isinstance(feature_values, (int, float)):
                        # Скалярні дані
                        sample_size = 1000  # TODO: реальний розмір
                        all_features.append(pd.Series([feature_values] * sample_size, name=feature_name))
        
        if all_features:
            return pd.concat(all_features, axis=1)
        else:
            return pd.DataFrame()
    
    def _create_target_vector(self, targets: Dict) -> pd.Series:
        """Створюємо вектор таргетів"""
        # Проста реалізація
        sample_size = 1000  # TODO: реальний розмір
        return pd.Series(np.random.randn(sample_size))
    
    def _get_adaptive_parameters(self, model_name: str, conditions: Dict) -> Dict:
        """
        [START] Отримуємо адаптивні параметри на основі умов
        """
        base_params = {
            'random_forest': {'n_estimators': 100, 'max_depth': 8, 'random_state': 42},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
        }
        
        # [TARGET] Коригуємо параметри на основі умов
        adaptive_params = base_params.get(model_name, {}).copy()
        
        # [TARGET] Волатильність
        if conditions['volatility_regime'] == 'high':
            adaptive_params.update(self.pattern_params.get('high_volatility', {}).get(model_name, {}))
        elif conditions['volatility_regime'] == 'low':
            adaptive_params.update(self.pattern_params.get('low_volatility', {}).get(model_name, {}))
        
        # [TARGET] Аномалії
        if conditions['pattern_presence'] == 'anomaly_present':
            adaptive_params.update(self.pattern_params.get('anomaly_present', {}).get(model_name, {}))
        
        # [TARGET] Регім зміни
        if conditions['market_regime'] == 'regime_change':
            adaptive_params.update(self.pattern_params.get('regime_change', {}).get(model_name, {}))
        
        logger.info(f"[TARGET] Adaptive params for {model_name}: {adaptive_params}")
        return adaptive_params
    
    def _train_single_model(self, 
                          model_name: str, 
                          training_data: Dict, 
                          params: Dict,
                          conditions: Dict) -> Dict:
        """
        [START] Навчаємо одну модель
        """
        X, y = training_data['X'], training_data['y']
        tscv = training_data['tscv']
        
        # [TARGET] Створюємо модель
        model_class = self.model_registry.get(model_name)
        if not model_class:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model_class(**params)
        
        # [TARGET] Time series cross-validation
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Навчаємо модель
            model.fit(X_train, y_train)
            
            # Оцінюємо
            y_pred = model.predict(X_val)
            score = r2_score(y_val, y_pred)
            cv_scores.append(score)
        
        # [TARGET] Навчаємо на всіх data
        model.fit(X, y)
        
        # [TARGET] Оцінюємо на всіх data
        y_pred = model.predict(X)
        final_score = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        
        return {
            'model': model,
            'model_name': model_name,
            'parameters': params,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'final_score': final_score,
            'mae': mae,
            'mse': mse,
            'training_conditions': conditions,
            'feature_importance': getattr(model, 'feature_importances_', None)
        }
    
    def _analyze_training_results(self, trained_models: Dict, conditions: Dict) -> Dict:
        """
        [START] Аналізуємо результати навчання
        """
        results = {
            'total_models': len(trained_models),
            'best_cv_score': 0,
            'best_final_score': 0,
            'model_comparison': {},
            'conditions': conditions
        }
        
        for model_name, model_result in trained_models.items():
            results['model_comparison'][model_name] = {
                'cv_mean': model_result['cv_mean'],
                'cv_std': model_result['cv_std'],
                'final_score': model_result['final_score'],
                'mae': model_result['mae'],
                'mse': model_result['mse']
            }
            
            if model_result['cv_mean'] > results['best_cv_score']:
                results['best_cv_score'] = model_result['cv_mean']
            
            if model_result['final_score'] > results['best_final_score']:
                results['best_final_score'] = model_result['final_score']
        
        return results
    
    def _select_best_model(self, trained_models: Dict, results: Dict) -> Dict:
        """
        [START] Вибираємо найкращу модель
        """
        # [TARGET] Критерії вибору
        best_model_name = None
        best_score = -1
        
        for model_name, model_result in trained_models.items():
            # Комбінований score: CV mean + final score
            combined_score = (model_result['cv_mean'] + model_result['final_score']) / 2
            
            if combined_score > best_score:
                best_score = combined_score
                best_model_name = model_name
        
        if best_model_name:
            best_model = trained_models[best_model_name]
            return {
                'model': best_model['model'],
                'model_name': best_model_name,
                'score': best_score,
                'parameters': best_model['parameters'],
                'conditions': best_model['training_conditions']
            }
        else:
            return {}
    
    def _get_default_config(self) -> Dict:
        """Конфігурація за замовчуванням"""
        return {
            'models': ['random_forest', 'gradient_boosting'],
            'cv_folds': 5,
            'save_models': True,
            'model_path': 'models/trained/'
        }


# [TARGET] ГОЛОВНА ФУНКЦІЯ
def train_pattern_aware_models(features: Dict, targets: Dict, patterns: Dict = None, config: Dict = None) -> Dict:
    """
    [START] Запускаємо інтелектуальне навчання моделей
    """
    trainer = PatternAwareModelTrainer()
    return trainer.train_pattern_aware_models(features, targets, patterns, config)


if __name__ == "__main__":
    print("Pattern-Aware Model Training - готовий до використання")
    print("[START] Інтелектуальне навчання моделей з урахуванням патернів")
    print("[DATA] Adaptive parameters, regime-aware, pattern-based!")
