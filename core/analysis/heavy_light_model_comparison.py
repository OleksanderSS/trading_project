# core/analysis/heavy_light_model_comparison.py - Порandвняння heavy vs light моwhereлей

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import json

logger = logging.getLogger(__name__)

class HeavyLightModelComparison:
    """
    Порandвняння heavy vs light моwhereлей with векторним аналandwithом
    """
    
    def __init__(self):
        self.heavy_models = {}
        self.light_models = {}
        self.comparison_results = {}
        self.model_categories = {
            'heavy': ['gru', 'lstm', 'transformer', 'attention', 'bert', 'gpt'],
            'light': ['linear', 'ridge', 'lasso', 'elastic', 'random_forest', 'xgboost', 'lightgbm', 'catboost']
        }
        
        logger.info("[HeavyLightModelComparison] Initialized")
    
    def categorize_models(self, models: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Категориwithувати моwhereлand на heavy and light
        
        Args:
            models: Словник моwhereлей
            
        Returns:
            Tuple[heavy_models, light_models]
        """
        heavy_models = {}
        light_models = {}
        
        for model_name, model_data in models.items():
            model_name_lower = model_name.lower()
            
            # Виwithначаємо категорandю
            is_heavy = any(keyword in model_name_lower for keyword in self.model_categories['heavy'])
            is_light = any(keyword in model_name_lower for keyword in self.model_categories['light'])
            
            if is_heavy:
                heavy_models[model_name] = model_data
                logger.debug(f"[HeavyLightModelComparison] Categorized {model_name} as HEAVY")
            elif is_light:
                light_models[model_name] = model_data
                logger.debug(f"[HeavyLightModelComparison] Categorized {model_name} as LIGHT")
            else:
                # За forмовчуванням вважаємо light
                light_models[model_name] = model_data
                logger.debug(f"[HeavyLightModelComparison] Categorized {model_name} as LIGHT (default)")
        
        self.heavy_models = heavy_models
        self.light_models = light_models
        
        logger.info(f"[HeavyLightModelComparison] Categorized {len(heavy_models)} heavy, {len(light_models)} light models")
        return heavy_models, light_models
    
    def compare_models_by_type(self, models: Dict[str, Any], 
                              X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray,
                              target_name: str) -> Dict[str, Any]:
        """
        Порandвняти моwhereлand одного типу на кожному andргетand
        
        Args:
            models: Словник моwhereлей одного типу
            X_train, X_test, y_train, y_test: Данand for тренування/тестування
            target_name: Наwithва andргету
            
        Returns:
            Dict with реwithульandandми порandвняння
        """
        results = {
            'target_name': target_name,
            'model_type': 'unknown',
            'model_results': {},
            'best_model': None,
            'worst_model': None,
            'ranking': [],
            'statistical_tests': {}
        }
        
        if not models:
            return results
        
        # Виwithначаємо тип моwhereлей
        model_names = list(models.keys())
        if model_names:
            first_model_name = model_names[0].lower()
            if any(keyword in first_model_name for keyword in self.model_categories['heavy']):
                results['model_type'] = 'heavy'
            else:
                results['model_type'] = 'light'
        
        # Оцandнюємо кожну model
        model_predictions = {}
        model_metrics = {}
        
        for model_name, model_data in models.items():
            try:
                # Отримуємо прогноwithи
                if hasattr(model_data, 'predict'):
                    train_preds = model_data.predict(X_train)
                    test_preds = model_data.predict(X_test)
                elif isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                    train_preds = model.predict(X_train)
                    test_preds = model.predict(X_test)
                else:
                    logger.warning(f"[HeavyLightModelComparison] Cannot get predictions for {model_name}")
                    continue
                
                # Calculating метрики
                train_metrics = self._calculate_metrics(y_train, train_preds)
                test_metrics = self._calculate_metrics(y_test, test_preds)
                
                model_predictions[model_name] = {
                    'train_predictions': train_preds,
                    'test_predictions': test_preds
                }
                
                model_metrics[model_name] = {
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics
                }
                
                logger.debug(f"[HeavyLightModelComparison] Evaluated {model_name}: R2={test_metrics['r2']:.3f}")
                
            except Exception as e:
                logger.error(f"[HeavyLightModelComparison] Error evaluating {model_name}: {e}")
                continue
        
        # Ранжуємо моwhereлand
        model_ranking = []
        for model_name, metrics in model_metrics.items():
            test_metrics = metrics['test_metrics']
            score = test_metrics.get('r2', 0.0)
            
            model_ranking.append({
                'model_name': model_name,
                'score': score,
                'metrics': test_metrics
            })
        
        # Сортуємо for R2 (кращand першand)
        model_ranking.sort(key=lambda x: x['score'], reverse=True)
        
        # Знаходимо найкращу and найгandршу model
        if model_ranking:
            results['best_model'] = model_ranking[0]
            results['worst_model'] = model_ranking[-1]
        
        results['model_results'] = model_metrics
        results['ranking'] = model_ranking
        
        # Сandтистичнand тести
        results['statistical_tests'] = self._perform_statistical_tests(model_predictions, y_test)
        
        return results
    
    def compare_heavy_vs_light(self, heavy_results: Dict[str, Any], 
                              light_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Порandвняти найкращand heavy and light моwhereлand векторно
        
        Args:
            heavy_results: Реwithульandти heavy моwhereлей
            light_results: Реwithульandти light моwhereлей
            
        Returns:
            Dict with реwithульandandми порandвняння
        """
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'heavy_best': heavy_results.get('best_model'),
            'light_best': light_results.get('best_model'),
            'vector_comparison': {},
            'recommendations': {},
            'ensemble_strategy': {}
        }
        
        # Отримуємо найкращand моwhereлand
        heavy_best = heavy_results.get('best_model')
        light_best = light_results.get('best_model')
        
        if not heavy_best or not light_best:
            comparison['error'] = "Missing best models for comparison"
            return comparison
        
        # Векторnot порandвняння
        vector_comparison = self._vector_model_comparison(heavy_best, light_best)
        comparison['vector_comparison'] = vector_comparison
        
        # Рекомендацandї
        comparison['recommendations'] = self._generate_recommendations(heavy_best, light_best, vector_comparison)
        
        # Ансамбльова стратегandя
        comparison['ensemble_strategy'] = self._generate_ensemble_strategy(heavy_best, light_best)
        
        return comparison
    
    def _vector_model_comparison(self, heavy_model: Dict[str, Any], 
                             light_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Векторnot порandвняння моwhereлей
        
        Args:
            heavy_model: Найкраща heavy model
            light_model: Найкраща light model
            
        Returns:
            Dict with реwithульandandми векторного порandвняння
        """
        # Отримуємо прогноwithи (припускаємо, що вони є в метриках)
        heavy_metrics = heavy_model.get('metrics', {})
        light_metrics = light_model.get('metrics', {})
        
        # Баwithовand метрики
        heavy_r2 = heavy_metrics.get('r2', 0.0)
        light_r2 = light_metrics.get('r2', 0.0)
        
        heavy_mae = heavy_metrics.get('mae', float('inf'))
        light_mae = light_metrics.get('mae', float('inf'))
        
        # Вектор характеристик
        heavy_vector = np.array([heavy_r2, -heavy_mae])  # R2 (бandльше краще), MAE (менше краще)
        light_vector = np.array([light_r2, -light_mae])
        
        # Calculating метрики
        euclidean_distance = np.linalg.norm(heavy_vector - light_vector)
        cosine_similarity = np.dot(heavy_vector, light_vector) / (np.linalg.norm(heavy_vector) * np.linalg.norm(light_vector))
        
        # Напрямок покращення
        direction_alignment = np.sign(heavy_vector - light_vector)
        
        return {
            'euclidean_distance': euclidean_distance,
            'cosine_similarity': cosine_similarity,
            'direction_alignment': direction_alignment.tolist(),
            'heavy_vector': heavy_vector.tolist(),
            'light_vector': light_vector.tolist(),
            'heavy_dominance': heavy_r2 > light_r2 and heavy_mae < light_mae,
            'light_dominance': light_r2 > heavy_r2 and light_mae < heavy_mae,
            'trade_off': abs(heavy_r2 - light_r2) < 0.1 and abs(heavy_mae - light_mae) < 0.1
        }
    
    def _generate_recommendations(self, heavy_model: Dict[str, Any], 
                             light_model: Dict[str, Any],
                             vector_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Геnotрувати рекомендацandї на основand порandвняння"""
        recommendations = {}
        
        heavy_r2 = heavy_model.get('metrics', {}).get('r2', 0.0)
        light_r2 = light_model.get('metrics', {}).get('r2', 0.0)
        
        heavy_mae = heavy_model.get('metrics', {}).get('mae', float('inf'))
        light_mae = light_model.get('metrics', {}).get('mae', float('inf'))
        
        # Аналandwith переваг
        if vector_comparison.get('heavy_dominance', False):
            recommendations['primary_choice'] = 'heavy'
            recommendations['reason'] = 'Heavy model outperforms light model significantly'
            recommendations['confidence'] = min(0.9, abs(heavy_r2 - light_r2) * 10)
        elif vector_comparison.get('light_dominance', False):
            recommendations['primary_choice'] = 'light'
            recommendations['reason'] = 'Light model outperforms heavy model significantly'
            recommendations['confidence'] = min(0.9, abs(light_r2 - heavy_r2) * 10)
        elif vector_comparison.get('trade_off', False):
            recommendations['primary_choice'] = 'ensemble'
            recommendations['reason'] = 'Models have similar performance - ensemble recommended'
            recommendations['confidence'] = 0.7
        else:
            # Виwithначаємо на основand R2
            if heavy_r2 > light_r2:
                recommendations['primary_choice'] = 'heavy'
                recommendations['reason'] = 'Heavy model has better R2 but higher MAE'
                recommendations['confidence'] = 0.6
            else:
                recommendations['primary_choice'] = 'light'
                recommendations['reason'] = 'Light model has better R2 but higher MAE'
                recommendations['confidence'] = 0.6
        
        # Контекстнand рекомендацandї
        recommendations['contextual'] = {
            'high_volatility': 'heavy' if heavy_mae < light_mae else 'light',
            'low_volatility': 'light' if light_r2 > heavy_r2 else 'heavy',
            'fast_moving': 'light' if light_mae < heavy_mae else 'heavy',
            'trending_market': 'heavy' if heavy_r2 > light_r2 else 'light'
        }
        
        return recommendations
    
    def _generate_ensemble_strategy(self, heavy_model: Dict[str, Any], 
                                light_model: Dict[str, Any]) -> Dict[str, Any]:
        """Геnotрувати ансамбльову стратегandю"""
        heavy_r2 = heavy_model.get('metrics', {}).get('r2', 0.0)
        light_r2 = light_model.get('metrics', {}).get('r2', 0.0)
        
        heavy_mae = heavy_model.get('metrics', {}).get('mae', float('inf'))
        light_mae = light_model.get('metrics', {}).get('mae', float('inf'))
        
        # Ваги на основand продуктивностand
        total_r2 = heavy_r2 + light_r2
        if total_r2 > 0:
            heavy_weight = heavy_r2 / total_r2
            light_weight = light_r2 / total_r2
        else:
            heavy_weight = 0.5
            light_weight = 0.5
        
        # Коригуємо ваги на основand MAE
        total_mae = heavy_mae + light_mae
        if total_mae < float('inf'):
            mae_heavy_weight = 1.0 - (heavy_mae / total_mae)
            mae_light_weight = 1.0 - (light_mae / total_mae)
            
            # Середня вага
            heavy_weight = (heavy_weight + mae_heavy_weight) / 2
            light_weight = (light_weight + mae_light_weight) / 2
        
        return {
            'weights': {
                'heavy': heavy_weight,
                'light': light_weight
            },
            'method': 'performance_weighted',
            'expected_improvement': (heavy_r2 * heavy_weight + light_r2 * light_weight) - max(heavy_r2, light_r2),
            'risk_reduction': abs(heavy_weight - light_weight) * 0.1
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Роwithрахувати метрики"""
        try:
            return {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100,
                'direction_accuracy': self._calculate_direction_accuracy(y_true, y_pred)
            }
        except Exception as e:
            logger.error(f"[HeavyLightModelComparison] Error calculating metrics: {e}")
            return {}
    
    def _calculate_direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Роwithрахувати точнandсть напрямку"""
        try:
            true_directions = np.sign(y_true)
            pred_directions = np.sign(y_pred)
            return np.mean(true_directions == pred_directions)
        except:
            return 0.0
    
    def _perform_statistical_tests(self, model_predictions: Dict[str, np.ndarray], 
                                y_true: np.ndarray) -> Dict[str, Any]:
        """Виконати сandтистичнand тести"""
        tests = {}
        
        model_names = list(model_predictions.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                try:
                    preds1 = model_predictions[model1].get('test_predictions', [])
                    preds2 = model_predictions[model2].get('test_predictions', [])
                    
                    if len(preds1) > 0 and len(preds2) > 0:
                        # t-test
                        t_stat, p_value = stats.ttest_rel(preds1, preds2)
                        
                        tests[f'{model1}_vs_{model2}'] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant_difference': p_value < 0.05
                        }
                except Exception as e:
                    logger.error(f"[HeavyLightModelComparison] Error in statistical test {model1} vs {model2}: {e}")
        
        return tests
    
    def run_complete_comparison(self, models: Dict[str, Any],
                             X_train: np.ndarray, X_test: np.ndarray,
                             y_train: np.ndarray, y_test: np.ndarray,
                             target_name: str) -> Dict[str, Any]:
        """
        Запустити повnot порandвняння моwhereлей
        
        Args:
            models: Всand моwhereлand
            X_train, X_test, y_train, y_test: Данand
            target_name: Наwithва andргету
            
        Returns:
            Dict with повними реwithульandandми порandвняння
        """
        logger.info(f"[HeavyLightModelComparison] Starting complete comparison for {target_name}")
        
        # 1. Категориwithуємо моwhereлand
        heavy_models, light_models = self.categorize_models(models)
        
        # 2. Порandвнюємо моwhereлand одного типу
        heavy_results = self.compare_models_by_type(heavy_models, X_train, X_test, y_train, y_test, target_name)
        light_results = self.compare_models_by_type(light_models, X_train, X_test, y_train, y_test, target_name)
        
        # 3. Порandвнюємо найкращand heavy vs light
        heavy_vs_light = self.compare_heavy_vs_light(heavy_results, light_results)
        
        # 4. Формуємо фandнальнand реwithульandти
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'target_name': target_name,
            'heavy_models_results': heavy_results,
            'light_models_results': light_results,
            'heavy_vs_light_comparison': heavy_vs_light,
            'summary': {
                'total_models': len(models),
                'heavy_models_count': len(heavy_models),
                'light_models_count': len(light_models),
                'best_heavy_model': heavy_results.get('best_model', {}).get('model_name'),
                'best_light_model': light_results.get('best_model', {}).get('model_name'),
                'recommended_choice': heavy_vs_light.get('recommendations', {}).get('primary_choice'),
                'ensemble_recommended': heavy_vs_light.get('recommendations', {}).get('primary_choice') == 'ensemble'
            }
        }
        
        # Зберandгаємо реwithульandти
        self.comparison_results[target_name] = final_results
        
        logger.info(f"[HeavyLightModelComparison] Complete comparison finished for {target_name}")
        return final_results

# Глобальна функцandя for сумandсностand
def compare_heavy_light_models(models: Dict[str, Any],
                             X_train: np.ndarray, X_test: np.ndarray,
                             y_train: np.ndarray, y_test: np.ndarray,
                             target_name: str) -> Dict[str, Any]:
    """
    Порandвняти heavy vs light моwhereлand (сумandснandсть)
    
    Args:
        models: Всand моwhereлand
        X_train, X_test, y_train, y_test: Данand
        target_name: Наwithва andргету
        
    Returns:
        Dict with реwithульandandми порandвняння
    """
    comparator = HeavyLightModelComparison()
    return comparator.run_complete_comparison(models, X_train, X_test, y_train, y_test, target_name)

if __name__ == "__main__":
    # Тестування
    # Створюємо тестовand данand
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    X_test = np.random.randn(30, 10)
    y_train = np.random.randn(100)
    y_test = np.random.randn(30)
    
    # Створюємо тестовand моwhereлand
    models = {
        'lstm_model': type('TestModel', (), {'predict': lambda x: np.random.randn(len(x))}),
        'random_forest_model': type('TestModel', (), {'predict': lambda x: np.random.randn(len(x))}),
        'linear_model': type('TestModel', (), {'predict': lambda x: np.random.randn(len(x))})
    }
    
    # Запускаємо порandвняння
    results = compare_heavy_light_models(models, X_train, X_test, y_train, y_test, 'test_target')
    
    print("Heavy vs Light Model Comparison Results:")
    print(json.dumps(results, indent=2, default=str))
